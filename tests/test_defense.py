"""The tests that matter: does the thing actually resist attack?

These are regression tests on the product's central claim. If someone tunes the
prior, the damping factor, or the weight floor and these go red, the change
made VouchNet worse at the one job it exists to do.
"""

from __future__ import annotations

import random

import pytest

from vouchnet import detect, simulate, trust

ATTACKS = ["collusion_ring", "sybil_boost", "review_bomb"]


def _shift(kind: str, seed: int = 7, size: int = 10):
    """Score one agent before and after an attack on it, under every model."""
    world = simulate.build_world(seed=seed)
    target = simulate.pick_target(world, random.Random(seed))
    before = trust.score_all(world.reviews, seeds=world.seeds)[target]
    simulate.apply_attack(world, kind=kind, target=target, size=size, seed=seed)
    after = trust.score_all(world.reviews, seeds=world.seeds)[target]
    return {
        "naive": after.naive_stars - before.naive_stars,
        "bayes": after.bayes_stars - before.bayes_stars,
        "trust": after.trust_stars - before.trust_stars,
    }


@pytest.mark.parametrize("kind", ATTACKS)
def test_trust_weighting_beats_the_plain_average(kind):
    """Every attack must move the trust score less than the raw average."""
    shift = _shift(kind)
    assert abs(shift["trust"]) < abs(shift["naive"])


@pytest.mark.parametrize("kind", ATTACKS)
def test_attacks_are_mostly_absorbed(kind):
    """At least 60% of each attack's effect must be absorbed."""
    shift = _shift(kind)
    absorbed = 1 - abs(shift["trust"]) / abs(shift["naive"])
    assert absorbed > 0.6, f"{kind}: only {absorbed:.0%} absorbed"


def test_the_plain_average_is_defenceless():
    """The premise. If this ever fails, the project has no reason to exist."""
    assert abs(_shift("collusion_ring")["naive"]) > 0.8


def test_bayes_alone_does_not_stop_a_review_bomb():
    """Shrinkage answers 'how much evidence', not 'whose evidence'.

    Ten hostile reviews are ten real observations, so a Beta update absorbs
    them almost fully. Only reviewer weighting sees that the ten are worthless.
    This is why the project needs both halves, and is worth stating as a test
    so nobody later 'simplifies' TrustRank away.
    """
    shift = _shift("review_bomb")
    assert abs(shift["bayes"]) > 0.8 * abs(shift["naive"])
    assert abs(shift["trust"]) < 0.5 * abs(shift["bayes"])


@pytest.mark.parametrize("kind", ATTACKS)
def test_attackers_are_flagged(kind):
    world = simulate.build_world(seed=7)
    simulate.apply_attack(world, kind=kind, size=10, seed=1)
    ranks = trust.trustrank(world.reviews, seeds=world.seeds)
    flags = detect.flag_agents(world.reviews, ranks)
    flagged = {name for name, flag in flags.items() if flag.flagged}
    assert flagged == world.malicious


def test_honest_agents_are_not_flagged():
    """No false positives on a clean network -- the precision half."""
    for seed in range(6):
        world = simulate.build_world(seed=seed)
        ranks = trust.trustrank(world.reviews, seeds=world.seeds)
        flags = detect.flag_agents(world.reviews, ranks)
        flagged = {name for name, flag in flags.items() if flag.flagged}
        assert not flagged, f"seed {seed}: falsely flagged {flagged}"


def test_a_review_bombed_agent_is_not_blamed_for_it():
    """Being attacked must never count against the victim."""
    world = simulate.build_world(seed=7)
    target = simulate.pick_target(world, random.Random(0))
    simulate.apply_attack(world, kind="review_bomb", target=target, size=10, seed=1)
    ranks = trust.trustrank(world.reviews, seeds=world.seeds)
    flags = detect.flag_agents(world.reviews, ranks)
    assert not flags[target].flagged
    assert any("targeted by" in reason for reason in flags[target].reasons)


def test_a_collusion_ring_cannot_climb_the_leaderboard():
    """Regression: the ring's own members must not out-rank honest agents.

    Attacks are usually framed as pushing a *target's* score around, but a ring
    that praises itself is also 10 agents each holding 9 five-star reviews. An
    earlier build ranked them 2nd through 11th with 4.84 stars. Two things stop
    it: they earn no dispersion bonus (no standing), and the leaderboard ranks
    on the lower credible bound, which their thin evidence cannot support.
    """
    world = simulate.build_world(seed=7)
    simulate.apply_attack(world, kind="collusion_ring", size=10, seed=1)
    scores = trust.score_all(world.reviews, seeds=world.seeds)

    ranked = sorted(scores.values(), key=lambda s: s.interval[0], reverse=True)
    top_ten = {score.agent for score in ranked[:10]}
    assert not (top_ten & world.malicious), "a collusion ring reached the top ten"

    honest_best = max(
        (s.interval[0] for s in scores.values() if s.agent not in world.malicious)
    )
    ring_best = max((s.interval[0] for s in scores.values() if s.agent in world.malicious))
    assert ring_best < honest_best


def test_ring_members_stay_near_the_prior():
    """With no credible reviews, a ring member should read as 'unknown', which
    means the population average -- not as excellent."""
    world = simulate.build_world(seed=7)
    simulate.apply_attack(world, kind="collusion_ring", size=10, seed=1)
    prior = trust.fit_prior(world.reviews)
    scores = trust.score_all(world.reviews, seeds=world.seeds)
    for name in world.malicious:
        assert abs(scores[name].trust_stars - prior.mean_stars) < 0.6


def test_bigger_attacks_stay_bounded():
    """A 30-account swarm must not simply overwhelm the weighting."""
    shift = _shift("sybil_boost", size=30)
    assert abs(shift["trust"]) < 0.75


def test_trust_score_tracks_truth_better_than_the_average_under_attack():
    """The end-to-end claim, in one assertion."""
    world = simulate.build_world(seed=7)
    target = simulate.pick_target(world, random.Random(7))
    truth = trust.p_to_stars(world.agents[target].quality)
    simulate.apply_attack(world, kind="collusion_ring", target=target, size=10, seed=1)
    after = trust.score_all(world.reviews, seeds=world.seeds)[target]
    assert abs(after.trust_stars - truth) < abs(after.naive_stars - truth)


# ------------------------------------------------------------ SCC internals


def test_scc_finds_a_simple_cycle():
    components = detect.strongly_connected_components({"a": {"b"}, "b": {"c"}, "c": {"a"}})
    assert ["a", "b", "c"] in components


def test_scc_separates_a_chain():
    components = detect.strongly_connected_components({"a": {"b"}, "b": {"c"}})
    assert sorted(components) == [["a"], ["b"], ["c"]]


def test_scc_handles_a_deep_chain_without_recursion_limits():
    """Iterative Tarjan: a 3000-node chain must not blow the Python stack."""
    edges = {f"n{i}": {f"n{i + 1}"} for i in range(3000)}
    assert len(detect.strongly_connected_components(edges)) == 3001
