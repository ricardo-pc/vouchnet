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


def _call(
    status=200,
    latency=100.0,
    documented=True,
    timed=True,
    expected=200,
    phase="measure",
    path="/x",
    method="GET",
) -> Call:
    return Call(
        method=method,
        path=path,
        status=status,
        latency_ms=latency,
        ok=status is not None and status < 400,
        timed=timed,
        documented=documented,
        expected_status=expected,
        phase=phase,
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
    _, evidence = score_reliability(probe)
    assert "3/3" in evidence  # the guesses are not in the denominator


def test_reliability_counts_documented_failures():
    """A documented endpoint that 500s is exactly what this should catch.

    Asserts the ordering rather than an exact score: the number depends on the
    prior, and a test that pins it breaks every time the model is tuned without
    ever catching a real defect.
    """
    half_broken = score_reliability(_probe(*([_call()] * 5 + [_call(status=500)] * 5)))
    clean = score_reliability(_probe(*([_call()] * 10)))
    assert half_broken[0] < clean[0]
    assert half_broken[0] <= 3


def test_total_failure_scores_the_floor():
    """Every documented endpoint broken is a 1, not a charitable 2."""
    score, _ = score_reliability(_probe(*[_call(status=500) for _ in range(9)]))
    assert score == 1


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
    _, evidence = score_reliability(probe)
    assert "4/4" in evidence  # correct refusals are not failures


def test_any_2xx_satisfies_an_expected_success():
    """201 Created is more correct than 200 OK for a create, and AgentPress
    returns it. Exact-matching scored that as a reliability failure -- punishing
    a service for better REST than the auditor's default guess."""
    assert _call(status=201, expected=200).met_contract  # create
    assert _call(status=204, expected=200).met_contract  # no content
    assert _call(status=200, expected=200).met_contract


def test_expected_error_still_requires_an_exact_match():
    """Leniency inside 2xx must not leak into the error codes: 401 and 403 mean
    different things and a service that swaps them broke its contract."""
    assert not _call(status=403, expected=401).met_contract
    assert _call(status=401, expected=401).met_contract


def test_wrong_status_is_a_failure_even_when_successful():
    """The inverse: 200 where the docs promise 401 is a broken contract -- and a
    security hole. It must not pass just for being 2xx."""
    assert not _call(status=200, expected=401).met_contract


def test_reliability_unrated_without_enough_documented_calls():
    """Two calls cannot distinguish 'reliable' from 'lucky'."""
    probe = _probe(_call(documented=True), _call(documented=True))
    assert score_reliability(probe) is None


def test_reliability_unrated_when_nothing_was_documented():
    """A service whose docs led nowhere gets no reliability score -- that is a
    statement about its docs (see clarity), not about its uptime."""
    probe = _probe(*[_call(documented=False) for _ in range(5)])
    assert score_reliability(probe) is None


# ------------------------------------------------ discovery vs measurement


def test_only_measured_calls_are_scored():
    """The agent's exploration must not reach the score. Its discovery calls
    can fail all day (it is guessing); only the harness's fixed replay counts."""
    probe = _probe(
        *[_call(status=500, phase="discover") for _ in range(5)],
        *[_call(status=200, phase="measure") for _ in range(6)],
    )
    _, evidence = score_reliability(probe)
    assert "6/6" in evidence  # the agent's five failed guesses are not scored


def test_scoring_is_deterministic_given_the_same_plan():
    """The bug this fixes: AgentHall scored 3 then 5 on identical code, because
    one run explored 9 endpoints and the next explored 7. Same plan in, same
    number out."""
    plan = [_call(status=200) for _ in range(9)] + [_call(status=500)]
    assert score_reliability(_probe(*plan)) == score_reliability(_probe(*plan))


def test_measure_replays_reads_a_fixed_number_of_times(monkeypatch):
    from vouchnet import auditor

    session = ProbeSession("https://example.com")
    seen: list[tuple[str, str, str]] = []

    def fake_request(method, path, body=None, documented=False, expected_status=200, phase="discover"):
        seen.append((method, path, phase))
        return {"status": expected_status}

    monkeypatch.setattr(session, "request", fake_request)
    # The agent poked /a twice and /b once while exploring; /c was a guess.
    session.probe.calls.extend([
        _call(path="/a", phase="discover"),
        _call(path="/a", phase="discover"),
        _call(path="/b", phase="discover"),
        _call(path="/c", phase="discover", documented=False),
    ])
    auditor.measure(session)

    # Each distinct documented read, exactly k times. The undocumented guess is
    # not replayed -- it was never the service's promise.
    assert sorted(seen) == sorted(
        [("GET", "/a", "measure")] * auditor.MEASUREMENT_REPEATS
        + [("GET", "/b", "measure")] * auditor.MEASUREMENT_REPEATS
    )


def test_measure_never_replays_writes():
    """Replaying a POST would manufacture state on someone else's service three
    times over -- the measurement would create the thing it claims to observe."""
    from vouchnet import auditor

    session = ProbeSession("https://example.com")
    sent: list[str] = []
    session.request = lambda *a, **k: sent.append(a[0]) or {"status": 200}  # type: ignore
    session.probe.calls.append(_call(method="POST", path="/reviews", phase="discover"))
    auditor.measure(session)
    assert sent == []


# ------------------------------------------------------ bayesian shrinkage


def test_a_tiny_perfect_sample_cannot_buy_a_five():
    """Three lucky calls are not a track record. Shrinkage keeps a thin sample
    near the prior instead of at an extreme."""
    thin = score_reliability(_probe(*[_call(status=200) for _ in range(3)]))
    thick = score_reliability(_probe(*[_call(status=200) for _ in range(30)]))
    assert thin[0] < thick[0]
    assert thick[0] == 5


def test_shrinkage_removes_the_threshold_cliff():
    """8/9 (89%) scored 3 while 9/10 (90%) scored 4 -- near-identical
    reliability, a whole point apart, because of an arbitrary cutoff."""
    a = score_reliability(_probe(*([_call()] * 8 + [_call(status=500)])))
    b = score_reliability(_probe(*([_call()] * 9 + [_call(status=500)])))
    assert abs(a[0] - b[0]) <= 1


def test_evidence_reports_the_interval():
    """A single number hides how much evidence is behind it."""
    _, evidence = score_reliability(_probe(*[_call() for _ in range(9)]))
    assert "posterior" in evidence and "lower bound" in evidence and "CI" in evidence


def test_one_miss_in_nine_is_not_distinguishable_from_perfect():
    """Not a bug -- the honest consequence of a 1-5 scale at n=9. The lower
    bound says you cannot tell 8/9 from 9/9 at that sample size, so the score
    does not pretend to. Resolution comes from more calls, not sharper rounding.
    """
    assert (
        score_reliability(_probe(*([_call()] * 8 + [_call(status=500)])))[0]
        == score_reliability(_probe(*([_call()] * 9)))[0]
    )
    # With a real sample, the same one-in-nine failure rate is visible.
    assert (
        score_reliability(_probe(*([_call()] * 24 + [_call(status=500)] * 3)))[0]
        < score_reliability(_probe(*([_call()] * 27)))[0]
    )


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


def test_read_only_blocks_writes_in_the_harness_not_the_prompt():
    """A review written to a reputation ledger is a public claim others read as
    fact -- and one written by a seed identity mints real trust for whatever it
    names. A prompt rule can be reasoned around; this cannot."""
    session = ProbeSession("https://example.com", read_only=True)
    try:
        result = session.request("POST", "/reviews", {"agent": "x", "stars": 5})
        assert "read-only audit" in result["error"]
        assert session.probe.calls == []
    finally:
        session.close()


def test_read_only_still_allows_reads():
    session = ProbeSession("https://example.com", read_only=True)
    try:
        # Rejected for the host, not the method -- proving GET got past the gate.
        result = session.request("GET", "https://elsewhere.example/x")
        assert "refusing to leave" in result["error"]
    finally:
        session.close()


def test_writes_are_recorded_as_mutations():
    """Auditing a write API creates state on someone else's service. That should
    never be a surprise, so it has to be visible."""
    probe = _probe(
        Call(method="GET", path="/a", status=200, latency_ms=5, ok=True),
        Call(method="POST", path="/reviews", status=200, latency_ms=5, ok=True),
        Call(method="POST", path="/reviews", status=422, latency_ms=5, ok=False),
    )
    assert [c.path for c in probe.mutations] == ["/reviews"]  # only the one that stuck


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
