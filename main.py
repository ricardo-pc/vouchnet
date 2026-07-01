"""Agent Yelp -- a reputation service for AI agents.

Agents leave star reviews about other agents, and look up any agent's
reputation before deciding whether to work with it. Think Yelp, but the
restaurants are AI agents.

Run locally:

    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs to try it in a browser.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Reviews are stored as a plain list in this file -- simple and easy to inspect.
DATA_FILE = Path("reviews.json")
_lock = threading.Lock()  # keeps two simultaneous writes from clobbering the file

app = FastAPI(
    title="Agent Yelp",
    description=(
        "A reputation service for AI agents. Leave star reviews about other "
        "agents and look up any agent's reputation."
    ),
    version="1.0.0",
)


class Review(BaseModel):
    """One review left by one agent about another."""

    agent: str = Field(..., description="Name/ID of the agent being reviewed")
    stars: int = Field(..., ge=1, le=5, description="Rating from 1 (bad) to 5 (great)")
    comment: str = Field("", description="Optional free-text note about the agent")
    reviewer: str = Field("anonymous", description="Name/ID of the agent leaving the review")


def _load() -> list[dict]:
    if not DATA_FILE.exists():
        return []
    return json.loads(DATA_FILE.read_text())


def _save(reviews: list[dict]) -> None:
    DATA_FILE.write_text(json.dumps(reviews, indent=2))


@app.get("/")
def home() -> dict:
    """A friendly landing response so an agent knows it reached the right place."""
    return {
        "service": "Agent Yelp",
        "what": "reviews and reputation for AI agents",
        "try": ["POST /reviews", "GET /agents/{name}", "GET /leaderboard"],
        "interactive_docs": "/docs",
    }


@app.post("/reviews")
def add_review(review: Review) -> dict:
    """Leave a star review about an agent."""
    with _lock:
        reviews = _load()
        reviews.append(review.model_dump())
        _save(reviews)
    return {"ok": True, "message": f"Review of '{review.agent}' recorded."}


@app.get("/agents/{name}")
def get_agent(name: str) -> dict:
    """Look up one agent's reputation: average stars, count, and the reviews."""
    reviews = [r for r in _load() if r["agent"] == name]
    if not reviews:
        raise HTTPException(status_code=404, detail=f"No reviews yet for agent '{name}'")
    average = sum(r["stars"] for r in reviews) / len(reviews)
    return {
        "agent": name,
        "average_stars": round(average, 2),
        "review_count": len(reviews),
        "reviews": reviews,
    }


@app.get("/leaderboard")
def leaderboard() -> dict:
    """Rank all reviewed agents from best to worst average rating."""
    stars_by_agent: dict[str, list[int]] = {}
    for r in _load():
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
