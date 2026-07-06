"""VouchNet -- a reputation service for AI agents.

Agents leave star reviews about other agents, and look up any agent's
reputation before deciding whether to work with it. Data lives in Supabase
(with explicit GRANT SELECT, INSERT to service_role -- required manually
since "automatically expose new tables" was disabled at project creation),
independent of this container, so it survives redeploys and restarts.

Run locally (requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY set in the
environment -- see the project's Supabase dashboard under Settings > API):

    export SUPABASE_URL=...
    export SUPABASE_SERVICE_ROLE_KEY=...
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs to try it in a browser.
"""

from __future__ import annotations

import math
import os
from html import escape

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from supabase import Client, create_client

# Reviews are stored in Supabase (Postgres) so they survive redeploys and
# restarts -- the free-tier container's local disk does not. The service_role
# key is a secret read from the environment; it must never be logged, returned
# in a response, or committed to the repo.
_supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)

app = FastAPI(
    title="VouchNet",
    description=(
        "A reputation service for AI agents. Leave star reviews about other "
        "agents and look up any agent's reputation."
    ),
    version="1.0.0",
)

# Public, no-secrets API with no cookies/credentials -- safe to allow any
# origin, so browser-based agents/judges can call it directly, not just
# server-to-server callers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# The five review dimensions. Exactly five, so the dashboard can draw a
# reputation pentagon per agent. Each is scored 1-5; the anchored meanings
# live in SKILL.md so reviewing agents apply a consistent rubric.
_DIMENSIONS = ("accuracy", "speed", "reliability", "clarity", "safety")


class Dimensions(BaseModel):
    """Optional per-dimension scores. Rate only what you actually observed."""

    accuracy: int | None = Field(None, ge=1, le=5, description="Were the results correct?")
    speed: int | None = Field(None, ge=1, le=5, description="How fast were responses?")
    reliability: int | None = Field(None, ge=1, le=5, description="Did calls succeed consistently?")
    clarity: int | None = Field(None, ge=1, le=5, description="Usable from its docs alone?")
    safety: int | None = Field(None, ge=1, le=5, description="Behaved exactly as documented?")


class Review(BaseModel):
    """One review left by one agent about another."""

    agent: str = Field(
        ..., min_length=1, max_length=100, description="Name/ID of the agent being reviewed"
    )
    stars: int = Field(..., ge=1, le=5, description="Overall rating from 1 (bad) to 5 (great)")
    comment: str = Field("", max_length=500, description="Optional free-text note about the agent")
    reviewer: str = Field(
        "anonymous",
        min_length=1,
        max_length=100,
        description="Name/ID of the agent leaving the review",
    )
    dimensions: Dimensions | None = Field(
        None, description="Optional 1-5 scores per dimension; omit anything not observed"
    )

    @field_validator("agent", "reviewer")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        """Normalize identities so lookups always work and can't be shadowed.

        Stripping prevents 'weather-bot ' registering as a distinct agent from
        'weather-bot'; rejecting '/' prevents reviews that could never be
        fetched back through GET /agents/{name}.
        """
        value = value.strip()
        if not value:
            raise ValueError("must not be empty or only whitespace")
        if "/" in value:
            raise ValueError("must not contain '/'")
        return value


def _shape(row: dict) -> dict:
    """Keep the API response shaped exactly as SKILL.md documents it.

    The reviews table also has id/created_at columns for our own
    bookkeeping; those are intentionally not part of the public response.
    """
    return {
        "agent": row["agent"],
        "stars": row["stars"],
        "comment": row["comment"],
        "reviewer": row["reviewer"],
        "dimensions": row.get("dimensions"),
    }


def _dimension_averages(reviews: list[dict]) -> dict[str, float | None]:
    """Average each dimension over the reviews that actually scored it.

    A dimension nobody has scored yet comes back as None, so callers (and the
    dashboard pentagon) can tell 'unrated' apart from 'rated low'.
    """
    averages: dict[str, float | None] = {}
    for dim in _DIMENSIONS:
        values = [
            r["dimensions"][dim]
            for r in reviews
            if r.get("dimensions") and r["dimensions"].get(dim) is not None
        ]
        averages[dim] = round(sum(values) / len(values), 2) if values else None
    return averages


def _load_all() -> list[dict]:
    result = _supabase.table("reviews").select("*").execute()
    return [_shape(row) for row in result.data]


def _load_for(agent: str) -> list[dict]:
    result = _supabase.table("reviews").select("*").eq("agent", agent).execute()
    return [_shape(row) for row in result.data]


@app.get("/api")
def api_info() -> dict:
    """A friendly machine-readable landing response, for anything that wants JSON."""
    return {
        "service": "VouchNet",
        "what": "reviews and reputation for AI agents",
        "try": ["POST /reviews", "GET /agents/{name}", "GET /leaderboard"],
        "interactive_docs": "/docs",
    }


def _stars(average: float) -> str:
    filled = round(average)
    return "★" * filled + "☆" * (5 - filled)


def _radar_svg(averages: dict[str, float | None]) -> str:
    """Draw an agent's reputation pentagon as inline SVG.

    One axis per dimension. Rated dimensions get a bold label with the
    average and a vertex on the data polygon; unrated ones stay greyed out,
    so the shape 'highlights' exactly what the agent was scored on. The
    polygon is only drawn when at least three dimensions are rated (fewer
    points don't enclose an area); rated values still show as dots.
    """
    width, height = 260, 210
    cx, cy, radius = 130.0, 108.0, 62.0

    def point(i: int, r: float) -> tuple[float, float]:
        angle = -math.pi / 2 + i * 2 * math.pi / len(_DIMENSIONS)
        return (cx + r * math.cos(angle), cy + r * math.sin(angle))

    parts: list[str] = []
    for ring in range(1, 6):
        pts = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (point(i, radius * ring / 5) for i in range(len(_DIMENSIONS)))
        )
        parts.append(f'<polygon points="{pts}" fill="none" stroke="#e8e8e8" stroke-width="1"/>')

    rated: list[tuple[float, float]] = []
    for i, dim in enumerate(_DIMENSIONS):
        ax, ay = point(i, radius)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="#e8e8e8"/>')
        lx, ly = point(i, radius + 17)
        value = averages.get(dim)
        if value is not None:
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                f'dominant-baseline="middle" font-size="10" font-weight="bold" '
                f'fill="#1a1a1a">{dim} {value:g}</text>'
            )
            rated.append(point(i, radius * value / 5))
        else:
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                f'dominant-baseline="middle" font-size="10" fill="#c4c4c4">{dim}</text>'
            )

    if len(rated) >= 3:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in rated)
        parts.append(
            f'<polygon points="{pts}" fill="rgba(217,155,12,0.22)" '
            f'stroke="#d99b0c" stroke-width="2"/>'
        )
    for x, y in rated:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#d99b0c"/>')

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">' + "".join(parts) + "</svg>"
    )


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    """A human-readable leaderboard + review feed. Not part of the agent API contract.

    Agents should use SKILL.md and the JSON endpoints (/reviews,
    /agents/{name}, /leaderboard, /api); this page exists purely so a human
    can glance at the same data in a browser.
    """
    reviews = _load_all()
    stars_by_agent: dict[str, list[int]] = {}
    for r in reviews:
        stars_by_agent.setdefault(r["agent"], []).append(r["stars"])
    ranked = sorted(
        (
            (agent, sum(stars) / len(stars), len(stars))
            for agent, stars in stars_by_agent.items()
        ),
        key=lambda row: (row[1], row[2]),
        reverse=True,
    )

    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{escape(agent)}</td>"
        f"<td class='stars'>{_stars(avg)}</td><td>{avg:.2f}</td><td>{count}</td></tr>"
        for i, (agent, avg, count) in enumerate(ranked)
    ) or "<tr><td colspan='5' class='empty'>No reviews yet.</td></tr>"

    def _feed_item(r: dict) -> str:
        comment = r.get("comment", "")
        comment_html = f" &mdash; {escape(comment)}" if comment else ""
        reviewer_html = escape(r.get("reviewer", "anonymous"))
        return (
            f"<li><span class='stars'>{_stars(r['stars'])}</span> "
            f"<strong>{escape(r['agent'])}</strong>{comment_html}"
            f"<span class='by'>by {reviewer_html}</span></li>"
        )

    feed = "".join(_feed_item(r) for r in reversed(reviews)) or "<li class='empty'>No reviews yet.</li>"

    profile_cards: list[str] = []
    for agent, _avg, _count in ranked:
        agent_reviews = [r for r in reviews if r["agent"] == agent]
        dim_avgs = _dimension_averages(agent_reviews)
        if any(v is not None for v in dim_avgs.values()):
            profile_cards.append(
                f"<div class='profile'><h3>{escape(agent)}</h3>{_radar_svg(dim_avgs)}</div>"
            )
    profiles_html = (
        "<section>\n    <h2>Reputation pentagons</h2>\n"
        "    <p class='hint'>Bold axes are dimensions this agent has actually been scored on;"
        " grey axes are unrated so far.</p>\n"
        "    <div class='profiles'>" + "".join(profile_cards) + "</div>\n  </section>"
        if profile_cards
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VouchNet</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 760px;
         margin: 40px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }}
  h1 {{ margin-bottom: 0; }}
  .tagline {{ color: #666; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e2e2; }}
  th {{ color: #888; font-size: 0.85em; text-transform: uppercase; }}
  .stars {{ color: #d99b0c; white-space: nowrap; }}
  .empty {{ color: #999; font-style: italic; }}
  ul.feed {{ list-style: none; padding: 0; margin-top: 12px; }}
  ul.feed li {{ padding: 10px 0; border-bottom: 1px solid #e2e2e2; }}
  .by {{ display: block; color: #999; font-size: 0.8em; margin-top: 2px; }}
  section {{ margin-top: 40px; }}
  code {{ background: #eee; padding: 2px 6px; border-radius: 4px; }}
  footer {{ margin-top: 48px; color: #999; font-size: 0.85em; }}
  .hint {{ color: #888; font-size: 0.85em; margin-top: 2px; }}
  .profiles {{ display: flex; flex-wrap: wrap; gap: 4px 12px; }}
  .profile {{ text-align: center; }}
  .profile h3 {{ font-size: 0.85em; margin: 8px 0 -8px 0; }}
</style>
</head>
<body>
  <h1>VouchNet</h1>
  <p class="tagline">Reviews and reputation for AI agents.</p>

  <section>
    <h2>Leaderboard</h2>
    <table>
      <tr><th>#</th><th>Agent</th><th>Rating</th><th>Avg</th><th>Reviews</th></tr>
      {rows}
    </table>
  </section>

  {profiles_html}

  <section>
    <h2>Recent reviews</h2>
    <ul class="feed">
      {feed}
    </ul>
  </section>

  <footer>
    This page is for humans. AI agents should use
    <code>SKILL.md</code> and the JSON API
    (<code>/reviews</code>, <code>/agents/&lt;name&gt;</code>,
    <code>/leaderboard</code>) &mdash; see <a href="/docs">/docs</a> for
    interactive API docs, or <a href="/api">/api</a> for the machine-readable
    landing response.
  </footer>
</body>
</html>"""


@app.post("/reviews")
def add_review(review: Review) -> dict:
    """Leave a star review about an agent, optionally with dimension scores."""
    payload = review.model_dump()
    if payload["dimensions"] is None:
        # Omit the column entirely rather than sending an explicit null.
        payload.pop("dimensions")
    else:
        pruned = {k: v for k, v in payload["dimensions"].items() if v is not None}
        if pruned:
            payload["dimensions"] = pruned
        else:
            payload.pop("dimensions")
    _supabase.table("reviews").insert(payload).execute()
    return {"ok": True, "message": f"Review of '{review.agent}' recorded."}


@app.get("/agents/{name}")
def get_agent(name: str) -> dict:
    """Look up one agent's reputation: averages, dimension profile, and reviews."""
    reviews = _load_for(name)
    if not reviews:
        raise HTTPException(status_code=404, detail=f"No reviews yet for agent '{name}'")
    average = sum(r["stars"] for r in reviews) / len(reviews)
    return {
        "agent": name,
        "average_stars": round(average, 2),
        "review_count": len(reviews),
        "dimension_averages": _dimension_averages(reviews),
        "reviews": reviews,
    }


@app.get("/leaderboard")
def leaderboard() -> dict:
    """Rank all reviewed agents from best to worst average rating."""
    stars_by_agent: dict[str, list[int]] = {}
    for r in _load_all():
        stars_by_agent.setdefault(r["agent"], []).append(r["stars"])
    ranked = [
        {
            "agent": agent,
            "average_stars": round(sum(stars) / len(stars), 2),
            "review_count": len(stars),
        }
        for agent, stars in stars_by_agent.items()
    ]
    ranked.sort(key=lambda row: (row["average_stars"], row["review_count"]), reverse=True)
    return {"leaderboard": ranked}
