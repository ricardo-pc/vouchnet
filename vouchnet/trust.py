"""Trust scoring: Bayesian shrinkage composed with TrustRank.

A plain average of star ratings has two failure modes that matter once the
reviewers are themselves agents:

1. It ignores how much evidence there is. One 5-star review outranks fifty
   4.8-star reviews, which is exactly backwards.
2. It treats every reviewer as equally credible, so anyone who can spin up
   accounts can move a score at will.

The fix for (1) is Bayesian shrinkage: treat an agent's quality as an unknown
Beta-distributed parameter, start from a prior fitted to the population, and
let each review update it. Sparse evidence stays near the prior; the score
only moves as far as the data earns.

The fix for (2) is TrustRank: reputation flows along the review graph from a
small seed set of verified agents, so a reviewer's weight comes from being
vouched for by agents that are themselves vouched for. A collusion ring can
praise itself all it likes -- with no path from a seed, the ring holds
almost no trust mass, and its reviews get almost no weight.

The two compose: TrustRank supplies the per-review weight `w`, and the Beta
update uses `w * p` instead of `p`. That single line is what makes the
scoreboard resistant to review bombing (see `evaluate.py` for the numbers).

Everything here is pure-Python and deterministic -- no numpy/scipy -- so it
runs on a free-tier container with a cold start measured in milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Mapping, Sequence

# Ratings are 1-5 stars, but the Beta distribution lives on [0, 1], so every
# rating is mapped onto that interval and back for display.
STAR_MIN = 1.0
STAR_MAX = 5.0

DIMENSIONS = ("accuracy", "speed", "reliability", "clarity", "safety")


def stars_to_p(stars: float) -> float:
    """Map a 1-5 star rating onto [0, 1] (1 star -> 0.0, 5 stars -> 1.0)."""
    return (stars - STAR_MIN) / (STAR_MAX - STAR_MIN)


def p_to_stars(p: float) -> float:
    """Inverse of `stars_to_p`."""
    return STAR_MIN + p * (STAR_MAX - STAR_MIN)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------
# The regularized incomplete beta function, for credible intervals.
#
# Reporting a score without saying how sure we are would throw away half the
# point of going Bayesian, and the interval is what the UI draws as the error
# bar. scipy would hand us `beta.ppf`, but scipy is a ~90MB dependency for one
# function, so this is the standard continued-fraction evaluation (Lentz's
# method, as in Numerical Recipes) plus a bisection to invert it.
# --------------------------------------------------------------------------

_TINY = 1e-300


def _betacf(a: float, b: float, x: float, iterations: int = 300, eps: float = 3e-14) -> float:
    """Continued fraction for the incomplete beta function, via Lentz's method."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        # Even step.
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        # Odd step.
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) -- i.e. the Beta(a, b) CDF at x."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
    # The continued fraction converges fast only on one side of the mode, so
    # use the symmetry I_x(a, b) = 1 - I_{1-x}(b, a) to stay on the good side.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float, iterations: int = 80) -> float:
    """Quantile function of Beta(a, b): the x where the CDF equals q.

    Bisection rather than Newton -- the CDF is monotone, 80 halvings pin x to
    far beyond double precision, and it cannot diverge the way Newton can near
    the boundaries.
    """
    low, high = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if betainc(a, b, mid) < q:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


# --------------------------------------------------------------------------
# Bayesian shrinkage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    """A Beta prior expressed the way a human would state it.

    `mean_p` is where an unreviewed agent starts; `strength` is how many
    reviews of evidence it takes to talk us out of it.
    """

    mean_p: float
    strength: float

    @property
    def mean_stars(self) -> float:
        return p_to_stars(self.mean_p)

    def as_alpha_beta(self) -> tuple[float, float]:
        return self.mean_p * self.strength, (1.0 - self.mean_p) * self.strength


# Used until there is enough data to fit one: 3.5 stars, worth 5 reviews.
DEFAULT_PRIOR = Prior(mean_p=stars_to_p(3.5), strength=5.0)


def fit_prior(
    reviews: Sequence[Mapping],
    min_agents: int = 5,
    max_strength: float = 50.0,
) -> Prior:
    """Fit the prior to the population itself -- empirical Bayes.

    Rather than inventing a prior, estimate it from how agents actually score:
    take each agent's mean rating, then solve for the Beta(a, b) whose mean and
    variance match that distribution (method of moments). A population where
    everyone lands near 4 stars yields a strong prior that punishes outliers; a
    population with genuinely spread-out quality yields a weak one that lets
    individual agents separate. The prior tightens on its own as data arrives.

    Falls back to DEFAULT_PRIOR when there is too little data to fit, or when
    the moments imply a degenerate (U-shaped or negative-parameter) Beta.
    """
    by_agent: dict[str, list[float]] = {}
    for review in reviews:
        by_agent.setdefault(review["agent"], []).append(stars_to_p(review["stars"]))
    means = [sum(values) / len(values) for values in by_agent.values()]
    if len(means) < min_agents:
        return DEFAULT_PRIOR

    m = sum(means) / len(means)
    variance = sum((value - m) ** 2 for value in means) / (len(means) - 1)
    if variance <= 0.0 or not 0.0 < m < 1.0:
        return DEFAULT_PRIOR
    # Method of moments: for Beta(a, b) with mean m and variance v,
    #   a + b = m(1-m)/v - 1.
    common = m * (1.0 - m) / variance - 1.0
    if common <= 0.0:
        # Observed spread exceeds what any Beta with this mean allows.
        return DEFAULT_PRIOR
    return Prior(mean_p=m, strength=_clamp(common, 1.0, max_strength))


# How many Bernoulli trials one review may be worth. Bounded because the
# estimate is a ratio of variances: a population that happens to agree almost
# perfectly would otherwise imply an unbounded kappa and switch shrinkage off.
KAPPA_MIN = 0.2
KAPPA_MAX = 20.0

# The weight at which a reviewer counts as having standing: 1.0 is, by
# construction, the median agent (see `reviewer_weights`).
TRUSTED_WEIGHT = 1.0


def effective_weight(weight: float, kappa: float) -> float:
    """How much evidence one review contributes, given its reviewer's weight.

    The dispersion bonus is earned in proportion to standing, reaching the full
    measured kappa at the median agent and fading to nearly nothing for an
    account nobody vouches for.

    This is load-bearing, and getting it wrong was visible on the leaderboard.
    Multiplying *every* review by kappa let a 10-account collusion ring reach
    4.84 stars and rank 2nd through 11th: ten sybils at 0.15 weight times an
    11x bonus swamped the prior. The reason that is wrong: kappa is a
    *measurement* of how precisely reviewers agree, taken over reviewers we can
    actually observe. No such measurement exists for anonymous accounts, and an
    attacker would happily manufacture one -- so they get the conservative
    Bernoulli default instead, one review worth one coin flip.

    Interpolating rather than thresholding at the median matters too: a hard
    step stripped the bonus from every below-median *honest* reviewer as well,
    and MAE regressed from 0.168 to 0.193, losing to the plain average it is
    supposed to beat. Half the network being below median is arithmetic, not
    evidence of anything.
    """
    standing = _clamp(weight / TRUSTED_WEIGHT, 0.0, 1.0)
    return weight * (1.0 + (kappa - 1.0) * standing)


def estimate_dispersion(
    reviews: Sequence[Mapping],
    weights: Mapping[str, float] | None = None,
    min_df: int = 8,
) -> float:
    """How much information one review really carries, versus a coin flip.

    The Beta-Binomial likelihood models each review as a Bernoulli(q) draw, so
    it expects reviews of one agent to scatter with variance q(1-q) -- about
    0.20 at a typical quality level. Real reviewers are nothing like that noisy:
    measured on the simulator, reviews of the same agent scatter with variance
    ~0.019, roughly ten times less. The likelihood is therefore misspecified,
    and it under-counts how much a review tells us. The consequence is not
    theoretical: the model shrank ~53% toward the prior when the variance
    components imply ~11% is optimal, and it lost to a plain average on MAE.

    The standard correction is a dispersion (quasi-likelihood) factor: compare
    the variance the model assumes to the variance actually observed, and scale
    the evidence by the ratio. `kappa` is that ratio -- the number of Bernoulli
    trials one review is worth. Reviewers who agree carry more weight; a noisy
    population automatically reverts to heavier shrinkage.

    Estimated only from reviewers who already hold trust: dispersion controls
    how much *everyone's* evidence counts, so letting anonymous accounts inform
    it would hand an attacker a lever on the global shrinkage.
    """
    trusted: dict[str, list[float]] = {}
    for review in reviews:
        reviewer = review.get("reviewer") or "anonymous"
        if weights is not None and weights.get(reviewer, WEIGHT_FLOOR) < 1.0:
            continue
        trusted.setdefault(review["agent"], []).append(stars_to_p(review["stars"]))

    groups = [values for values in trusted.values() if len(values) >= 2]
    df = sum(len(values) - 1 for values in groups)
    if df < min_df:
        # Too little data to estimate dispersion; assume the model is right
        # rather than invent a correction.
        return 1.0

    residual = 0.0
    implied = 0.0
    for values in groups:
        mean_p = sum(values) / len(values)
        residual += sum((value - mean_p) ** 2 for value in values)
        implied += (len(values) - 1) * mean_p * (1.0 - mean_p)

    within = residual / df
    expected = implied / df
    if within <= 0.0 or expected <= 0.0:
        return KAPPA_MAX
    return _clamp(expected / within, KAPPA_MIN, KAPPA_MAX)


@dataclass(frozen=True)
class Posterior:
    """A Beta posterior over one agent's quality."""

    alpha: float
    beta: float
    evidence: float  # Total reviewer weight folded in (effective review count).

    @property
    def mean_p(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def mean_stars(self) -> float:
        return p_to_stars(self.mean_p)

    def interval_stars(self, credibility: float = 0.9) -> tuple[float, float]:
        """Central credible interval, in stars."""
        tail = (1.0 - credibility) / 2.0
        low = beta_ppf(tail, self.alpha, self.beta)
        high = beta_ppf(1.0 - tail, self.alpha, self.beta)
        return p_to_stars(low), p_to_stars(high)


def posterior(
    observations: Iterable[tuple[float, float]],
    prior: Prior = DEFAULT_PRIOR,
) -> Posterior:
    """Update `prior` with `(p, weight)` observations.

    Weight is where TrustRank enters: a review from a reviewer with weight 0.15
    moves the posterior a sixth as far as one from a reviewer with weight 1.0.
    Weight 0 would mean the review is ignored entirely.
    """
    alpha, beta = prior.as_alpha_beta()
    evidence = 0.0
    for p, weight in observations:
        alpha += weight * p
        beta += weight * (1.0 - p)
        evidence += weight
    return Posterior(alpha=alpha, beta=beta, evidence=evidence)


# --------------------------------------------------------------------------
# TrustRank
# --------------------------------------------------------------------------

# How far a reviewer's weight can stray from the median agent's. The floor
# keeps a brand-new reviewer from being silenced outright (every agent starts
# unknown, and a system that ignores unknowns can never bootstrap); the cap
# stops a single highly-trusted agent from becoming an oracle.
WEIGHT_FLOOR = 0.15
WEIGHT_CAP = 4.0


def trustrank(
    reviews: Sequence[Mapping],
    seeds: Sequence[str] = (),
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Personalized PageRank over the review graph, seeded on verified agents.

    An edge runs reviewer -> reviewed, weighted by how strong an endorsement
    the review is (`stars_to_p`, so a 1-star review carries no endorsement at
    all -- you do not lend someone credibility by panning them). Trust is
    injected only at `seeds` and flows outward, which is the whole defense:
    trust must be *received* from somewhere legitimate, and no amount of
    self-dealing manufactures it.

    With no seeds this degrades to ordinary PageRank with a uniform teleport
    vector. That is the honest cold-start behavior -- everyone is equally
    unverified -- but it is also why the seed set matters: an unseeded graph
    can be captured by whoever creates the most accounts.

    Returns a trust mass per agent, summing to 1.
    """
    nodes: set[str] = set()
    for review in reviews:
        nodes.add(review["agent"])
        nodes.add(review.get("reviewer") or "anonymous")
    if not nodes:
        return {}

    # Endorsement edges, summed when one agent reviews another more than once.
    out_edges: dict[str, dict[str, float]] = {}
    for review in reviews:
        reviewer = review.get("reviewer") or "anonymous"
        target = review["agent"]
        if reviewer == target:
            continue  # Self-review earns nothing.
        weight = stars_to_p(review["stars"])
        if weight <= 0.0:
            continue
        out_edges.setdefault(reviewer, {}).setdefault(target, 0.0)
        out_edges[reviewer][target] += weight

    # Teleport vector: all mass on the seeds, or uniform if there are none.
    live_seeds = [seed for seed in seeds if seed in nodes]
    if live_seeds:
        teleport = {node: 0.0 for node in nodes}
        for seed in live_seeds:
            teleport[seed] = 1.0 / len(live_seeds)
    else:
        teleport = {node: 1.0 / len(nodes) for node in nodes}

    rank = dict(teleport)
    out_totals = {node: sum(edges.values()) for node, edges in out_edges.items()}

    for _ in range(iterations):
        nxt = {node: (1.0 - damping) * teleport[node] for node in nodes}
        # Agents who never review anyone (or only ever gave 1 star) are
        # dangling: their mass would evaporate, so send it back to the seeds
        # rather than letting the vector lose norm.
        dangling = sum(rank[node] for node in nodes if out_totals.get(node, 0.0) <= 0.0)
        for node, edges in out_edges.items():
            total = out_totals[node]
            if total <= 0.0:
                continue
            share = damping * rank[node] / total
            for target, weight in edges.items():
                nxt[target] += share * weight
        for node in nodes:
            nxt[node] += damping * dangling * teleport[node]

        delta = sum(abs(nxt[node] - rank[node]) for node in nodes)
        rank = nxt
        if delta < tolerance:
            break
    return rank


def reviewer_weights(
    trust: Mapping[str, float],
    floor: float = WEIGHT_FLOOR,
    cap: float = WEIGHT_CAP,
) -> dict[str, float]:
    """Turn TrustRank mass into a review weight, relative to the median agent.

    TrustRank mass is a probability, so its absolute scale depends on how many
    agents exist -- useless as a multiplier on its own. Dividing by the median
    positive mass fixes a typical agent at weight ~1.0, which keeps the prior's
    strength (measured in reviews) meaningful no matter how the network grows.
    """
    positive = [value for value in trust.values() if value > 0.0]
    reference = median(positive) if positive else 1.0
    if reference <= 0.0:
        reference = 1.0
    return {
        node: _clamp(value / reference, floor, cap)
        for node, value in trust.items()
    }


# --------------------------------------------------------------------------
# The scoreboard: all three models, side by side
# --------------------------------------------------------------------------


@dataclass
class DimensionScore:
    naive: float | None = None
    trusted: float | None = None
    count: int = 0


@dataclass
class AgentScore:
    """One agent's reputation under each model, for comparison in the UI."""

    agent: str
    review_count: int
    naive_stars: float  # Plain mean -- what VouchNet shipped at the hackathon.
    bayes_stars: float  # Shrunk toward the prior, every reviewer equal.
    trust_stars: float  # Shrunk *and* trust-weighted. The headline number.
    interval: tuple[float, float]  # 90% credible interval on trust_stars.
    trust: float  # TrustRank mass.
    weight: float  # This agent's weight when it reviews others.
    evidence: float  # Effective review count after weighting.
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)

    @property
    def uncertainty(self) -> float:
        """Width of the credible interval, in stars. Wide = don't trust it yet."""
        return self.interval[1] - self.interval[0]


def score_all(
    reviews: Sequence[Mapping],
    seeds: Sequence[str] = (),
    prior: Prior | None = None,
) -> dict[str, AgentScore]:
    """Score every reviewed agent under all three models.

    Keeping the naive and Bayes-only numbers alongside the trust-weighted one
    is deliberate: the difference between them is the story (and the demo).
    """
    if prior is None:
        prior = fit_prior(reviews)
    trust = trustrank(reviews, seeds=seeds)
    weights = reviewer_weights(trust)
    # Every review's evidence is scaled by kappa, so shrinkage tracks how much
    # reviewers actually agree instead of assuming they are coin flips. Note
    # this cannot weaken the attack defense: kappa scales honest and hostile
    # evidence alike, and the defense comes from their *ratio* (TrustRank), not
    # from the prior's absolute strength.
    kappa = estimate_dispersion(reviews, weights)

    by_agent: dict[str, list[Mapping]] = {}
    for review in reviews:
        by_agent.setdefault(review["agent"], []).append(review)

    scores: dict[str, AgentScore] = {}
    for agent, agent_reviews in by_agent.items():
        stars = [float(review["stars"]) for review in agent_reviews]
        observations = [
            (
                stars_to_p(review["stars"]),
                effective_weight(
                    weights.get(review.get("reviewer") or "anonymous", WEIGHT_FLOOR),
                    kappa,
                ),
            )
            for review in agent_reviews
        ]
        trusted_post = posterior(observations, prior)
        # The Bayes-only comparison gets the same dispersion correction, so the
        # difference between the two columns isolates trust weighting alone.
        bayes_post = posterior([(p, kappa) for p, _ in observations], prior)

        dimensions: dict[str, DimensionScore] = {}
        for dimension in DIMENSIONS:
            rated = [
                (review, review["dimensions"][dimension])
                for review in agent_reviews
                if review.get("dimensions")
                and review["dimensions"].get(dimension) is not None
            ]
            if not rated:
                # Unrated stays None. A dimension nobody scored is unknown, not
                # zero -- the distinction the whole pentagon rests on.
                dimensions[dimension] = DimensionScore()
                continue
            dim_observations = [
                (
                    stars_to_p(value),
                    effective_weight(
                        weights.get(review.get("reviewer") or "anonymous", WEIGHT_FLOOR),
                        kappa,
                    ),
                )
                for review, value in rated
            ]
            dimensions[dimension] = DimensionScore(
                naive=sum(value for _, value in rated) / len(rated),
                trusted=posterior(dim_observations, prior).mean_stars,
                count=len(rated),
            )

        scores[agent] = AgentScore(
            agent=agent,
            review_count=len(agent_reviews),
            naive_stars=sum(stars) / len(stars),
            bayes_stars=bayes_post.mean_stars,
            trust_stars=trusted_post.mean_stars,
            interval=trusted_post.interval_stars(0.9),
            trust=trust.get(agent, 0.0),
            weight=weights.get(agent, WEIGHT_FLOOR),
            evidence=trusted_post.evidence,
            dimensions=dimensions,
        )
    return scores
