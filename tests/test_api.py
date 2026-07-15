"""API tests, with the contract in SKILL.md as the thing under test.

Agents integrate against the documented shapes, and they cannot read a
changelog. Anything an agent was told it could rely on is pinned here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
from vouchnet import store


@pytest.fixture()
def client(monkeypatch):
    """A client over an empty in-memory store, so tests never hit the network."""
    monkeypatch.setattr(main, "_store", store.MemoryStore())
    return TestClient(main.app)


# ------------------------------------------------- the documented contract


def test_post_then_get_round_trips(client):
    response = client.post(
        "/reviews",
        json={"agent": "weather-bot", "stars": 5, "comment": "fast", "reviewer": "alice-agent"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "message": "Review of 'weather-bot' recorded."}

    body = client.get("/agents/weather-bot").json()
    assert body["agent"] == "weather-bot"
    assert body["average_stars"] == 5.0
    assert body["review_count"] == 1
    assert body["reviews"][0]["comment"] == "fast"


def test_unknown_agent_is_404_with_a_message(client):
    response = client.get("/agents/nobody")
    assert response.status_code == 404
    assert "No reviews yet" in response.json()["detail"]


def test_average_stars_is_still_the_plain_mean(client):
    """SKILL.md documents `average_stars` as the plain average. v2 added trust
    scoring alongside it; it must not have quietly redefined this field."""
    for stars in (5, 2):
        client.post("/reviews", json={"agent": "a", "stars": stars, "reviewer": f"r{stars}"})
    assert client.get("/agents/a").json()["average_stars"] == 3.5


def test_dimension_averages_report_null_when_unrated(client):
    client.post(
        "/reviews",
        json={"agent": "a", "stars": 5, "reviewer": "r", "dimensions": {"speed": 4}},
    )
    dims = client.get("/agents/a").json()["dimension_averages"]
    assert dims["speed"] == 4
    assert dims["accuracy"] is None


def test_leaderboard_shape(client):
    client.post("/reviews", json={"agent": "a", "stars": 5, "reviewer": "r"})
    rows = client.get("/leaderboard").json()["leaderboard"]
    assert rows[0]["agent"] == "a"
    for key in ("average_stars", "review_count", "trust_score", "credible_interval"):
        assert key in rows[0]


def test_leaderboard_ranks_by_trust_not_by_raw_average(client):
    """A newcomer with one 5-star review must not outrank a proven agent."""
    client.post("/reviews", json={"agent": "newcomer", "stars": 5, "reviewer": "pal"})
    for i in range(15):
        client.post("/reviews", json={"agent": "veteran", "stars": 5, "reviewer": f"r{i}"})
    rows = client.get("/leaderboard").json()["leaderboard"]
    assert rows[0]["agent"] == "veteran"
    assert rows[0]["average_stars"] == rows[1]["average_stars"] == 5.0


# ---------------------------------------------------------- input limits


@pytest.mark.parametrize(
    "payload",
    [
        {"agent": "a", "stars": 0},
        {"agent": "a", "stars": 6},
        {"agent": "", "stars": 3},
        {"agent": "a/b", "stars": 3},
        {"agent": "a", "stars": 3, "comment": "x" * 501},
        {"agent": "a", "stars": 3, "dimensions": {"speed": 9}},
        {"agent": "x" * 101, "stars": 3},
    ],
)
def test_invalid_input_is_422(client, payload):
    assert client.post("/reviews", json=payload).status_code == 422


def test_names_are_trimmed_so_identities_cannot_be_shadowed(client):
    client.post("/reviews", json={"agent": "  weather-bot  ", "stars": 4})
    assert client.get("/agents/weather-bot").json()["review_count"] == 1


# -------------------------------------------------------------- surfaces


@pytest.mark.parametrize("path", ["/", "/api", "/graph", "/leaderboard", "/stats", "/health", "/skill.md"])
def test_get_and_head_both_work(client, path):
    """Uptime monitors default to HEAD; a GET-only route would 405 and look down."""
    assert client.get(path).status_code == 200
    assert client.head(path).status_code == 200


def test_profile_redirects_into_the_graph_ui(client):
    """v1 published /profile/{name} links; they must keep resolving."""
    response = client.get("/profile/weather-bot", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/?agent=weather-bot"


def test_graph_payload_shape(client):
    client.post("/reviews", json={"agent": "a", "stars": 5, "reviewer": "b"})
    body = client.get("/graph").json()
    assert body["mode"] == "ledger"
    assert {"nodes", "links", "stats", "prior"} <= set(body)
    assert body["links"][0] == {"source": "b", "target": "a", "stars": 5, "comment": "", "attack": None}


def test_graph_is_fine_with_an_empty_ledger(client):
    body = client.get("/graph").json()
    assert body["nodes"] == [] and body["links"] == []
    assert body["stats"]["agents"] == 0


# --------------------------------------------------------------- sandbox


def test_sandbox_is_deterministic(client):
    first = client.post("/sandbox/world", json={"seed": 7}).json()
    second = client.post("/sandbox/world", json={"seed": 7}).json()
    assert first["nodes"] == second["nodes"]


def test_sandbox_seeds_differ(client):
    first = client.post("/sandbox/world", json={"seed": 1}).json()
    second = client.post("/sandbox/world", json={"seed": 2}).json()
    assert first["nodes"] != second["nodes"]


def test_sandbox_never_writes_to_the_ledger(client):
    """The separation the whole design rests on: a demo must not invent
    reputation data that real agents would then read back as fact."""
    client.post(
        "/sandbox/world",
        json={"seed": 7, "attacks": [{"kind": "collusion_ring", "size": 5}]},
    )
    assert client.get("/graph").json()["nodes"] == []
    assert client.get("/stats").json()["reviews"] == 0


def test_sandbox_attack_flags_the_attackers(client):
    body = client.post(
        "/sandbox/world",
        json={"seed": 7, "attacks": [{"kind": "collusion_ring", "size": 6}]},
    ).json()
    flagged = [n for n in body["nodes"] if n["risk"] >= 0.5]
    assert len(flagged) == 6
    assert all(n["malicious"] for n in flagged)
    assert body["stats"]["flagged"] == 6


@pytest.mark.parametrize(
    "payload",
    [
        {"seed": 7, "attacks": [{"kind": "nuke"}]},
        {"seed": 7, "attacks": [{"kind": "collusion_ring", "size": 99}]},
        {"seed": -1},
        {"seed": 7, "attacks": [{"kind": "collusion_ring"}] * 7},
    ],
)
def test_sandbox_rejects_abusive_input(client, payload):
    """The sandbox runs a simulation per request, so its inputs are bounded."""
    assert client.post("/sandbox/world", json=payload).status_code == 422
