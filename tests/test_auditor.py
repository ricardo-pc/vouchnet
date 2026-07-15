"""Tests for the auditor's measured dimensions and its guardrails.

The LLM half is not tested here (it needs an API key and is non-deterministic).
What *is* tested is everything that decides whether a real service gets a fair
score, plus the limits that stop the agent doing something it shouldn't to
somebody else's server.
"""

from __future__ import annotations

import pytest

from vouchnet.auditor import (
    Call,
    Probe,
    ProbeSession,
    score_reliability,
    score_speed,
)


def _probe(*calls: Call) -> Probe:
    return Probe(base_url="https://example.com", calls=list(calls))


def _call(status=200, latency=100.0, documented=True, timed=True, expected=200) -> Call:
    return Call(
        method="GET",
        path="/x",
        status=status,
        latency_ms=latency,
        ok=status is not None and status < 400,
        timed=timed,
        documented=documented,
        expected_status=expected,
    )


# ------------------------------------------------------------------- speed


def test_speed_uses_the_median_not_the_mean():
    """One slow outlier must not define a service's speed."""
    probe = _probe(_call(latency=50), _call(latency=50), _call(latency=9000))
    score, evidence = score_speed(probe)
    assert score == 5  # median is 50ms
    assert "median 50ms" in evidence


def test_speed_is_unrated_when_nothing_succeeded():
    assert score_speed(_probe(_call(status=500))) is None
    assert score_speed(_probe()) is None


@pytest.mark.parametrize(
    "latency, expected",
    [(100, 5), (500, 4), (1500, 3), (3000, 2), (9000, 1)],
)
def test_speed_rubric_is_monotone(latency, expected):
    score, _ = score_speed(_probe(_call(latency=latency)))
    assert score == expected


def test_warmup_calls_never_count():
    """Free-tier cold starts take 30-60s. Timing them would score every
    free-tier service 1/5 and call it measurement."""
    probe = _probe(_call(latency=45000, timed=False), _call(latency=80))
    score, evidence = score_speed(probe)
    assert score == 5
    assert "1 successful" in evidence


# ------------------------------------------------------------- reliability


def test_reliability_ignores_undocumented_probes():
    """The regression that showed up against a real service.

    Guessing /api and /docs on a site that documents neither produced two 404s
    and a 2/5 reliability score. The service was behaving correctly; the
    auditor's guesses were wrong.
    """
    probe = _probe(
        _call(status=200, documented=True),
        _call(status=200, documented=True),
        _call(status=200, documented=True),
        _call(status=404, documented=False),  # guessed /api
        _call(status=404, documented=False),  # guessed /docs
    )
    score, evidence = score_reliability(probe)
    assert score == 5
    assert "3/3" in evidence


def test_reliability_counts_documented_failures():
    """A documented endpoint that 500s is exactly what this should catch."""
    probe = _probe(
        _call(status=200, documented=True),
        _call(status=500, documented=True),
        _call(status=500, documented=True),
        _call(status=200, documented=True),
    )
    score, _ = score_reliability(probe)
    assert score == 2  # 50% behaved as documented


def test_a_documented_error_is_correct_behaviour_not_a_failure():
    """The regression from the AgentPress audit.

    The agent tested an unauthenticated read and a nonexistent record, got the
    documented 401 and 404, and scored safety 5/5 because they were exactly
    right -- while those same two calls dragged reliability to 3/5. A service
    that correctly refuses is working.
    """
    probe = _probe(
        _call(status=200, expected=200, documented=True),
        _call(status=200, expected=200, documented=True),
        _call(status=401, expected=401, documented=True),  # unauthenticated read
        _call(status=404, expected=404, documented=True),  # nonexistent record
    )
    score, evidence = score_reliability(probe)
    assert score == 5
    assert "4/4" in evidence


def test_any_2xx_satisfies_an_expected_success():
    """201 Created is more correct than 200 OK for a create, and AgentPress
    returns it. Exact-matching scored that as a reliability failure -- punishing
    a service for better REST than the auditor's default guess."""
    probe = _probe(
        _call(status=201, expected=200, documented=True),  # create
        _call(status=204, expected=200, documented=True),  # no content
        _call(status=200, expected=200, documented=True),
    )
    score, evidence = score_reliability(probe)
    assert score == 5
    assert "3/3" in evidence


def test_expected_error_still_requires_an_exact_match():
    """Leniency inside 2xx must not leak into the error codes: 401 and 403 mean
    different things and a service that swaps them broke its contract."""
    probe = _probe(
        _call(status=403, expected=401, documented=True),
        _call(status=200, expected=200, documented=True),
        _call(status=200, expected=200, documented=True),
        _call(status=200, expected=200, documented=True),
    )
    score, _ = score_reliability(probe)
    assert score == 3  # 3/4


def test_wrong_status_is_a_failure_even_when_successful():
    """The inverse: 200 where the docs promise 401 is a broken contract --
    and a security hole. It must not score as a pass just for being 2xx."""
    probe = _probe(
        _call(status=200, expected=401, documented=True),
        _call(status=200, expected=200, documented=True),
        _call(status=200, expected=200, documented=True),
        _call(status=200, expected=200, documented=True),
    )
    score, _ = score_reliability(probe)
    assert score == 3  # 3/4 kept the contract


def test_reliability_unrated_without_enough_documented_calls():
    """Two calls cannot distinguish 'reliable' from 'lucky'."""
    probe = _probe(_call(documented=True), _call(documented=True))
    assert score_reliability(probe) is None


def test_reliability_unrated_when_nothing_was_documented():
    """A service whose docs led nowhere gets no reliability score -- that is a
    statement about its docs (see clarity), not about its uptime."""
    probe = _probe(*[_call(documented=False) for _ in range(5)])
    assert score_reliability(probe) is None


# -------------------------------------------------------------- guardrails


def test_probe_refuses_to_leave_the_target_host():
    """A prompt-injected SKILL.md must not be able to point the auditor
    somewhere else."""
    session = ProbeSession("https://example.com")
    try:
        result = session.request("GET", "https://evil.example.net/steal")
        assert "refusing to leave" in result["error"]
        assert session.probe.calls == []
    finally:
        session.close()


def test_probe_enforces_the_call_budget():
    session = ProbeSession("https://example.com", max_calls=0)
    try:
        result = session.request("GET", "/anything")
        assert "call budget exhausted" in result["error"]
    finally:
        session.close()


@pytest.mark.parametrize("method", ["DELETE", "PUT", "PATCH"])
def test_probe_permits_only_safe_methods(method):
    """The auditor reads and posts reviews. It has no business deleting
    anything on a service it is evaluating."""
    session = ProbeSession("https://example.com")
    try:
        result = session.request(method, "/x")
        assert "not permitted" in result["error"]
        assert session.probe.calls == []
    finally:
        session.close()
