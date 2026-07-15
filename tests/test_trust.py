"""Tests for the scoring engine.

The interesting tests here are the ones that pin down *properties* rather than
values: shrinkage must be monotone in evidence, TrustRank must sum to one,
attackers must not gain mass. Those hold for any prior and any seed, so they
keep protecting the model after it is tuned -- unlike an assertion that some
agent scores 4.31, which just breaks.
"""

from __future__ import annotations

import math

import pytest

from vouchnet import detect, simulate, trust


# --------------------------------------------------------------- beta math


@pytest.mark.parametrize(
    "a, b, x, expected",
    [
        # Closed forms: Beta(1,1) is uniform, so its CDF is the identity.
        (1.0, 1.0, 0.5, 0.5),
        (1.0, 1.0, 0.25, 0.25),
        # I_x(2,3) = 6x²(1/2 - 2x/3 + x²/4); at x=1/2 this is exactly 11/16.
        (2.0, 3.0, 0.5, 0.6875),
        # I_x(a,b) = 1 - I_{1-x}(b,a), checked on the far side of the mode.
        (5.0, 2.0, 0.9, 1.0 - 0.114265),
    ],
)
def test_betainc_matches_closed_form(a, b, x, expected):
    assert trust.betainc(a, b, x) == pytest.approx(expected, abs=1e-5)


def test_betainc_is_a_cdf():
    assert trust.betainc(2.0, 5.0, 0.0) == 0.0
    assert trust.betainc(2.0, 5.0, 1.0) == 1.0
    values = [trust.betainc(2.0, 5.0, x / 20) for x in range(21)]
    assert all(b >= a for a, b in zip(values, values[1:])), "CDF must be monotone"


def test_beta_ppf_inverts_betainc():
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        x = trust.beta_ppf(q, 3.0, 4.0)
        assert trust.betainc(3.0, 4.0, x) == pytest.approx(q, abs=1e-6)


# -------------------------------------------------------------- shrinkage


def test_posterior_of_nothing_is_the_prior():
    prior = trust.Prior(mean_p=0.6, strength=5.0)
    assert trust.posterior([], prior).mean_p == pytest.approx(0.6)


def test_more_evidence_moves_further_from_the_prior():
    """The core shrinkage property: one 5-star review must not beat many."""
    prior = trust.DEFAULT_PRIOR
    one = trust.posterior([(1.0, 1.0)], prior).mean_stars
    many = trust.posterior([(1.0, 1.0)] * 20, prior).mean_stars
    assert prior.mean_stars < one < many < 5.0


def test_a_single_five_star_loses_to_many_high_reviews():
    """The exact failure a plain average has, stated as a test."""
    newcomer = trust.posterior([(1.0, 1.0)], trust.DEFAULT_PRIOR)
    veteran = trust.posterior([(0.95, 1.0)] * 50, trust.DEFAULT_PRIOR)
    assert veteran.mean_stars > newcomer.mean_stars


def test_weight_scales_influence():
    prior = trust.DEFAULT_PRIOR
    light = trust.posterior([(1.0, 0.15)], prior).mean_stars
    heavy = trust.posterior([(1.0, 4.0)], prior).mean_stars
    assert prior.mean_stars < light < heavy


def test_credible_interval_narrows_with_evidence():
    few = trust.posterior([(0.8, 1.0)] * 2, trust.DEFAULT_PRIOR).interval_stars(0.9)
    many = trust.posterior([(0.8, 1.0)] * 60, trust.DEFAULT_PRIOR).interval_stars(0.9)
    assert (many[1] - many[0]) < (few[1] - few[0])
    assert few[0] <= trust.posterior([(0.8, 1.0)] * 2, trust.DEFAULT_PRIOR).mean_stars <= few[1]


# ------------------------------------------------------------- dispersion


def test_dispersion_defaults_to_one_without_data():
    """No estimate is better than a guessed one."""
    assert trust.estimate_dispersion([]) == 1.0
    assert trust.estimate_dispersion([{"agent": "a", "stars": 5, "reviewer": "r"}]) == 1.0


def test_agreeing_reviewers_carry_more_evidence_than_a_coin_flip():
    """Reviewers who agree are more informative than Bernoulli draws."""
    reviews = [
        {"agent": f"a{i}", "stars": 4, "reviewer": f"r{j}"}
        for i in range(6)
        for j in range(4)
    ]
    assert trust.estimate_dispersion(reviews) > 1.0


def test_wildly_disagreeing_reviewers_carry_less():
    reviews = []
    for i in range(6):
        for j, stars in enumerate((1, 5, 1, 5)):
            reviews.append({"agent": f"a{i}", "stars": stars, "reviewer": f"r{j}"})
    assert trust.estimate_dispersion(reviews) < 1.0


def test_dispersion_is_bounded():
    identical = [
        {"agent": f"a{i}", "stars": 3, "reviewer": f"r{j}"}
        for i in range(6)
        for j in range(4)
    ]
    assert trust.estimate_dispersion(identical) <= trust.KAPPA_MAX


def test_dispersion_ignores_untrusted_reviewers():
    """Otherwise an attacker could file identical reviews to drive kappa up and
    weaken shrinkage for everyone."""
    reviews = [
        {"agent": f"a{i}", "stars": 4 if j % 2 else 2, "reviewer": f"r{j}"}
        for i in range(6)
        for j in range(4)
    ]
    spam = [{"agent": "a0", "stars": 5, "reviewer": f"bot{k}"} for k in range(200)]
    weights = {f"r{j}": 1.0 for j in range(4)}
    weights.update({f"bot{k}": trust.WEIGHT_FLOOR for k in range(200)})
    clean = trust.estimate_dispersion(reviews, weights)
    polluted = trust.estimate_dispersion(reviews + spam, weights)
    assert clean == pytest.approx(polluted)


def test_dispersion_correction_beats_the_plain_mean_on_recovery():
    """The regression that motivated the correction: the Bayesian estimate must
    not be *worse* than the average it replaces on an honest network."""
    errors = {"naive": [], "trust": []}
    for seed in range(8):
        world = simulate.build_world(seed=seed)
        scores = trust.score_all(world.reviews, seeds=world.seeds)
        for name, actual in world.truth().items():
            if name in scores:
                errors["naive"].append(abs(scores[name].naive_stars - actual))
                errors["trust"].append(abs(scores[name].trust_stars - actual))
    naive_mae = sum(errors["naive"]) / len(errors["naive"])
    trust_mae = sum(errors["trust"]) / len(errors["trust"])
    assert trust_mae <= naive_mae


def test_fit_prior_falls_back_when_data_is_thin():
    assert trust.fit_prior([]) == trust.DEFAULT_PRIOR
    assert trust.fit_prior([{"agent": "a", "stars": 5}]) == trust.DEFAULT_PRIOR


def test_fit_prior_recovers_the_population_mean():
    world = simulate.build_world(seed=3)
    prior = trust.fit_prior(world.reviews)
    observed = sum(r["stars"] for r in world.reviews) / len(world.reviews)
    # Empirical Bayes fits the mean of per-agent means, which sits near the
    # grand mean but is not identical to it -- hence the loose tolerance.
    assert prior.mean_stars == pytest.approx(observed, abs=0.5)
    assert 1.0 <= prior.strength <= 50.0


# -------------------------------------------------------------- trustrank


def test_trustrank_is_a_distribution():
    world = simulate.build_world(seed=5)
    ranks = trust.trustrank(world.reviews, seeds=world.seeds)
    assert sum(ranks.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(value >= 0.0 for value in ranks.values())


def test_trustrank_without_seeds_is_uniform_teleport():
    """No seed set means nobody is verified, so nobody starts with standing."""
    reviews = [{"agent": "b", "stars": 5, "reviewer": "a"}]
    ranks = trust.trustrank(reviews, seeds=())
    assert sum(ranks.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(value > 0.0 for value in ranks.values())


def test_unreachable_agents_hold_no_trust():
    """The property the whole defense rests on."""
    reviews = [
        {"agent": "honest", "stars": 5, "reviewer": "seed"},
        # A ring praising itself, with no inbound edge from the seed.
        {"agent": "ring-b", "stars": 5, "reviewer": "ring-a"},
        {"agent": "ring-a", "stars": 5, "reviewer": "ring-b"},
    ]
    ranks = trust.trustrank(reviews, seeds=["seed"])
    assert ranks["ring-a"] == pytest.approx(0.0, abs=1e-9)
    assert ranks["ring-b"] == pytest.approx(0.0, abs=1e-9)
    assert ranks["honest"] > 0.0


def test_one_star_reviews_do_not_lend_credibility():
    """Panning someone must not transfer trust to them."""
    reviews = [{"agent": "target", "stars": 1, "reviewer": "seed"}]
    ranks = trust.trustrank(reviews, seeds=["seed"])
    assert ranks["target"] == pytest.approx(0.0, abs=1e-9)


def test_self_review_earns_nothing():
    # The seed has to be a real node in the graph: name a seed that does not
    # exist and TrustRank correctly falls back to a uniform teleport, which
    # would hand the lone self-reviewer all the mass by default.
    reviews = [
        {"agent": "solo", "stars": 5, "reviewer": "solo"},
        {"agent": "honest", "stars": 5, "reviewer": "seed"},
    ]
    ranks = trust.trustrank(reviews, seeds=["seed"])
    assert ranks["solo"] == pytest.approx(0.0, abs=1e-9)
    assert ranks["honest"] > 0.0


def test_reviewer_weights_are_bounded():
    world = simulate.build_world(seed=5)
    ranks = trust.trustrank(world.reviews, seeds=world.seeds)
    weights = trust.reviewer_weights(ranks)
    assert all(trust.WEIGHT_FLOOR <= w <= trust.WEIGHT_CAP for w in weights.values())


# ------------------------------------------------------------- scoreboard


def test_score_all_reports_every_model():
    world = simulate.build_world(seed=11)
    scores = trust.score_all(world.reviews, seeds=world.seeds)
    assert scores
    for score in scores.values():
        assert 1.0 <= score.naive_stars <= 5.0
        assert 1.0 <= score.bayes_stars <= 5.0
        assert 1.0 <= score.trust_stars <= 5.0
        assert score.interval[0] <= score.trust_stars <= score.interval[1]
        assert score.review_count > 0


def test_unrated_dimensions_stay_none():
    """Unrated is not zero -- the rule the pentagon depends on."""
    reviews = [
        {"agent": "a", "stars": 5, "reviewer": "b", "dimensions": {"speed": 5}},
    ]
    scores = trust.score_all(reviews)
    dims = scores["a"].dimensions
    assert dims["speed"].naive == 5
    assert dims["accuracy"].naive is None
    assert dims["accuracy"].trusted is None


def test_scoring_is_deterministic():
    world = simulate.build_world(seed=9)
    first = trust.score_all(world.reviews, seeds=world.seeds)
    second = trust.score_all(world.reviews, seeds=world.seeds)
    assert {k: v.trust_stars for k, v in first.items()} == {
        k: v.trust_stars for k, v in second.items()
    }


def test_empty_input_is_not_an_error():
    assert trust.score_all([]) == {}
    assert trust.trustrank([]) == {}
