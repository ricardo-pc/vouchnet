"""VouchNet -- a trust layer for AI agents.

Agents leave reviews about other agents, and check an agent's track record
before deciding to work with it. The scoring is not a star average: reviews are
folded into a Beta posterior (so sparse evidence stays honest) and weighted by
the reviewer's TrustRank (so manufactured reviewers cannot move the number).
The engine lives in the `vouchnet` package; this module is the HTTP surface.

Two data planes, deliberately separated:

- **The ledger** (`/`, `/graph`, `/reviews`, `/agents/{name}`, `/leaderboard`)
  is real: reviews agents actually left, stored in Postgres.
- **The sandbox** (`/sandbox/*`) is synthetic: a seeded simulation used to
  demonstrate attacks and measure defenses. Nothing in it is ever written to
  the ledger. A reputation system that invents its own reviews is worthless,
  even when the invented reviews are "just for the demo".

Run locally (no credentials needed -- falls back to an in-memory store):

    uvicorn main:app --reload

Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY for real persistence, and
VOUCHNET_SEEDS to choose the TrustRank seed set.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from vouchnet import detect, env, simulate, store, trust

# Before build_store() reads the environment: a local .env supplies credentials
# in development, and is a no-op wherever real env vars are already set.
env.load_env()

_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="VouchNet",
    description=(
        "A trust layer for AI agents. Leave reviews about other agents and "
        "look up any agent's reputation -- scored with Bayesian shrinkage and "
        "TrustRank rather than a star average."
    ),
    version="2.0.0",
)

# Public, no-secrets API with no cookies or credentials, so any origin is safe;
# this is what lets browser-based agents call it directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_store = store.build_store()
_DIMENSIONS = trust.DIMENSIONS


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


class Attack(BaseModel):
    """One manipulation to stage in the sandbox."""

    kind: str = Field(..., description="collusion_ring | sybil_boost | review_bomb")
    target: str | None = Field(None, max_length=100)
    size: int = Field(8, ge=1, le=30, description="Number of attacker accounts")

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in ("collusion_ring", "sybil_boost", "review_bomb"):
            raise ValueError("unknown attack kind")
        return value


class Scenario(BaseModel):
    """A sandbox request. Stateless: the same body always yields the same world.

    The client owns the attack list and replays it, so the server keeps no
    session state and concurrent visitors cannot corrupt each other's demo.
    """

    seed: int = Field(7, ge=0, le=10_000)
    attacks: list[Attack] = Field(default_factory=list, max_length=6)


def _dimension_averages(scores: trust.AgentScore | None) -> dict[str, float | None]:
    """Dimension averages in the shape SKILL.md documents (null when unrated)."""
    if scores is None:
        return {dimension: None for dimension in _DIMENSIONS}
    return {
        dimension: (
            round(scores.dimensions[dimension].naive, 2)
            if scores.dimensions[dimension].naive is not None
            else None
        )
        for dimension in _DIMENSIONS
    }


def _build_graph(
    reviews: list[dict],
    seeds: list[str],
    mode: str,
    world: simulate.World | None = None,
) -> dict:
    """Shape the review graph plus every score into one payload for the UI.

    One endpoint rather than several because the front end draws a single
    picture: positions depend on edges, colour depends on trust, and radius
    depends on review counts. Fetching those separately would let them arrive
    out of sync and make the graph flicker between inconsistent states.
    """
    scores = trust.score_all(reviews, seeds=seeds)
    ranks = trust.trustrank(reviews, seeds=seeds)
    flags = detect.flag_agents(reviews, ranks)
    weights = trust.reviewer_weights(ranks)
    prior = trust.fit_prior(reviews)
    kappa = trust.estimate_dispersion(reviews, weights)

    everyone = set(ranks) | set(scores)
    truth = world.truth() if world else {}
    malicious = world.malicious if world else set()

    nodes = []
    for name in sorted(everyone):
        score = scores.get(name)
        flag = flags.get(name)
        nodes.append(
            {
                "id": name,
                "reviews": score.review_count if score else 0,
                "naive": round(score.naive_stars, 3) if score else None,
                "bayes": round(score.bayes_stars, 3) if score else None,
                "trust_stars": round(score.trust_stars, 3) if score else None,
                "interval": [round(score.interval[0], 3), round(score.interval[1], 3)]
                if score
                else None,
                "evidence": round(score.evidence, 2) if score else 0,
                "trust": round(ranks.get(name, 0.0), 6),
                "weight": round(weights.get(name, trust.WEIGHT_FLOOR), 3),
                "seed": name in seeds,
                "risk": round(flag.risk, 2) if flag else 0.0,
                "reasons": flag.reasons if flag else [],
                "ring": flag.ring if flag else [],
                # Sandbox only: the answer key, so the UI can show how close
                # each model gets to the quality the agent was built with.
                "truth": round(truth[name], 3) if name in truth else None,
                "malicious": name in malicious,
                "dimensions": {
                    dimension: {
                        "naive": round(value.naive, 2) if value.naive is not None else None,
                        "trusted": round(value.trusted, 2) if value.trusted is not None else None,
                        "count": value.count,
                    }
                    for dimension, value in (score.dimensions.items() if score else [])
                },
            }
        )

    links = [
        {
            "source": review.get("reviewer") or "anonymous",
            "target": review["agent"],
            "stars": review["stars"],
            "comment": review.get("comment", ""),
            "attack": review.get("attack"),
        }
        for review in reviews
        if (review.get("reviewer") or "anonymous") != review["agent"]
    ]

    return {
        "mode": mode,
        "nodes": nodes,
        "links": links,
        "seeds": seeds,
        "attacks": world.attacks if world else [],
        "prior": {
            "mean_stars": round(prior.mean_stars, 3),
            "strength": round(prior.strength, 2),
            "kappa": round(kappa, 2),
        },
        "stats": {
            "agents": len(everyone),
            "reviews": len(reviews),
            "reviewers": len({review.get("reviewer") or "anonymous" for review in reviews}),
            "flagged": sum(1 for flag in flags.values() if flag.flagged),
        },
    }


# --------------------------------------------------------------------------
# The agent-facing API. This contract is documented in SKILL.md -- fields may
# be added, but nothing here may change shape or disappear.
# --------------------------------------------------------------------------


@app.post("/reviews")
def add_review(review: Review) -> dict:
    """Leave a star review about an agent, optionally with dimension scores."""
    payload = review.model_dump()
    if payload["dimensions"] is None:
        payload.pop("dimensions")  # Omit the column rather than storing a null.
    else:
        pruned = {k: v for k, v in payload["dimensions"].items() if v is not None}
        if pruned:
            payload["dimensions"] = pruned
        else:
            payload.pop("dimensions")
    _store.add(payload)
    return {"ok": True, "message": f"Review of '{review.agent}' recorded."}


@app.api_route("/agents/{name}", methods=["GET", "HEAD"])
def get_agent(name: str) -> dict:
    """Look up one agent's reputation: scores, dimension profile, and reviews."""
    reviews = _store.for_agent(name)
    if not reviews:
        raise HTTPException(status_code=404, detail=f"No reviews yet for agent '{name}'")
    scores = trust.score_all(_store.all(), seeds=store.seed_agents())
    score = scores.get(name)
    average = sum(review["stars"] for review in reviews) / len(reviews)
    return {
        "agent": name,
        # Unchanged from v1: the plain mean, still the documented meaning.
        "average_stars": round(average, 2),
        "review_count": len(reviews),
        "dimension_averages": _dimension_averages(score),
        # Added in v2. Additive, so existing agent integrations keep working.
        "trust_score": round(score.trust_stars, 2) if score else None,
        "credible_interval": (
            [round(score.interval[0], 2), round(score.interval[1], 2)] if score else None
        ),
        "trust_weight": round(score.weight, 3) if score else None,
        "reviews": reviews,
    }


@app.api_route("/leaderboard", methods=["GET", "HEAD"])
def leaderboard() -> dict:
    """Rank every reviewed agent, best first, by trust-weighted score."""
    reviews = _store.all()
    scores = trust.score_all(reviews, seeds=store.seed_agents())
    ranked = [
        {
            "agent": score.agent,
            "average_stars": round(score.naive_stars, 2),
            "trust_score": round(score.trust_stars, 2),
            "credible_interval": [round(score.interval[0], 2), round(score.interval[1], 2)],
            "review_count": score.review_count,
        }
        for score in scores.values()
    ]
    # Ranked by the *lower* edge of the credible interval, not the point
    # estimate -- the trick every mature ratings system converges on. Sorting by
    # the estimate itself rewards being unproven: an agent with one glowing
    # review scores well but could be anything, and it would outrank an agent
    # that has actually demonstrated the same level a hundred times over.
    # Sorting by the lower bound asks "what can this agent defend?", so
    # uncertainty costs rank until evidence retires it.
    ranked.sort(
        key=lambda row: (row["credible_interval"][0], row["review_count"]), reverse=True
    )
    return {"leaderboard": ranked}


@app.api_route("/api", methods=["GET", "HEAD"])
def api_info() -> dict:
    """Machine-readable landing response.

    Accepts HEAD as well as GET: uptime monitors default to HEAD, and a
    GET-only route would 405 those and look down when it isn't.
    """
    return {
        "service": "VouchNet",
        "what": "trust scores and reputation for AI agents",
        "try": ["POST /reviews", "GET /agents/{name}", "GET /leaderboard"],
        "scoring": "Beta-Binomial shrinkage weighted by TrustRank over the review graph",
        "skill": "/skill.md",
        "interactive_docs": "/docs",
    }


@app.api_route("/skill.md", methods=["GET", "HEAD"], response_class=PlainTextResponse)
def skill_md() -> str:
    """Serve SKILL.md from this domain so agents can fetch it without GitHub.

    Read from disk per request, so a redeploy always serves the current file.
    """
    return Path(__file__).parent.joinpath("SKILL.md").read_text()


# --------------------------------------------------------------------------
# The human-facing surface: the graph UI and the data behind it.
# --------------------------------------------------------------------------


@app.api_route("/graph", methods=["GET", "HEAD"])
def graph() -> dict:
    """The real ledger as a scored graph. Powers the home page."""
    return _build_graph(_store.all(), store.seed_agents(), mode="ledger")


@app.post("/sandbox/world")
def sandbox_world(scenario: Scenario) -> dict:
    """Build a synthetic world, optionally under attack, and score it.

    Deterministic in (seed, attacks), and never touches the ledger.
    """
    world = simulate.build_scenario(
        seed=scenario.seed,
        attacks=[attack.model_dump() for attack in scenario.attacks],
    )
    payload = _build_graph(world.reviews, world.seeds, mode="sandbox", world=world)
    payload["seed"] = scenario.seed
    return payload


@app.api_route("/stats", methods=["GET", "HEAD"])
def stats() -> dict:
    """Counters for the hero, plus whether reviews are actually persisted."""
    reviews = _store.all()
    return {
        "reviews": len(reviews),
        "agents": len({review["agent"] for review in reviews}),
        "reviewers": len({review.get("reviewer") or "anonymous" for review in reviews}),
        "persistent": getattr(_store, "persistent", False),
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    return {"ok": True}


@app.api_route("/profile/{name}", methods=["GET", "HEAD"])
def profile(name: str) -> RedirectResponse:
    """v1 profile pages are now a panel in the graph UI. Keep the links alive."""
    return RedirectResponse(url=f"/?agent={name}", status_code=307)


@app.api_route("/", methods=["GET", "HEAD"])
def home() -> FileResponse:
    """The trust graph. Agents should use SKILL.md and the JSON API instead."""
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
