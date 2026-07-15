"""Review persistence: Supabase in production, in-memory everywhere else.

The original code called `create_client(os.environ["SUPABASE_URL"], ...)` at
import time, so the app could not start -- or be tested, or be run by anyone
who cloned the repo -- without live credentials. Deciding the backend at
startup instead means `uvicorn main:app` works out of the box, the test suite
needs no network, and production behaviour is unchanged when the env vars are
present.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Protocol

# Illustrative reviews about invented agents (weather-bot, spam-bot, ...), used
# only to seed the in-memory fallback so a fresh clone renders a real graph
# instead of an empty canvas. Deliberately fictional: inventing ratings about
# real services would be publishing fabricated reputation data about them.
_SEED_FILE = Path(__file__).resolve().parent.parent / "sample_reviews.json"

# Agents whose reviews seed TrustRank -- the root of trust. Kept configurable
# because "who do you trust to start with" is a policy decision, not a
# technical one, and it is the single most security-relevant knob here:
# anything reachable from this set inherits credibility.
DEFAULT_SEEDS = ("ricardo", "vouchnet-auditor")


def seed_agents() -> list[str]:
    raw = os.environ.get("VOUCHNET_SEEDS")
    if raw:
        return [name.strip() for name in raw.split(",") if name.strip()]
    return list(DEFAULT_SEEDS)


def _shape(row: dict) -> dict:
    """Project a stored row onto the response shape SKILL.md documents.

    The table also carries id/created_at for our own bookkeeping; those stay
    out of the public contract.
    """
    return {
        "agent": row["agent"],
        "stars": row["stars"],
        "comment": row.get("comment", ""),
        "reviewer": row.get("reviewer", "anonymous"),
        "dimensions": row.get("dimensions"),
    }


class Store(Protocol):
    def all(self) -> list[dict]: ...
    def for_agent(self, agent: str) -> list[dict]: ...
    def add(self, review: dict) -> None: ...


class MemoryStore:
    """Fallback store. Data lives for the life of the process, and says so."""

    persistent = False

    def __init__(self, seed_rows: list[dict] | None = None) -> None:
        self._rows: list[dict] = list(seed_rows or [])
        self._lock = threading.Lock()

    def all(self) -> list[dict]:
        with self._lock:
            return [_shape(row) for row in self._rows]

    def for_agent(self, agent: str) -> list[dict]:
        with self._lock:
            return [_shape(row) for row in self._rows if row["agent"] == agent]

    def add(self, review: dict) -> None:
        with self._lock:
            self._rows.append(dict(review))


class SupabaseStore:
    """Production store: Postgres via Supabase, so reviews outlive the container."""

    persistent = True

    def __init__(self, url: str, key: str) -> None:
        from supabase import create_client

        self._client = create_client(url, key)

    def all(self) -> list[dict]:
        result = self._client.table("reviews").select("*").execute()
        return [_shape(row) for row in result.data]

    def for_agent(self, agent: str) -> list[dict]:
        result = self._client.table("reviews").select("*").eq("agent", agent).execute()
        return [_shape(row) for row in result.data]

    def add(self, review: dict) -> None:
        self._client.table("reviews").insert(review).execute()


def _sample_rows() -> list[dict]:
    try:
        rows = json.loads(_SEED_FILE.read_text())
        return rows if isinstance(rows, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def build_store() -> Store:
    """Pick a backend from the environment, preferring real persistence.

    Only the in-memory fallback gets sample rows. Seeding the real ledger with
    canned reviews would make the reputation data a fiction.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if url and key:
        return SupabaseStore(url, key)
    return MemoryStore(_sample_rows())
