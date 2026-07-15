"""A deterministic synthetic agent ecosystem, with fraud planted on purpose.

Why this exists: you cannot measure a trust model on real review data, because
real review data has no answer key. Nobody labels which agents were actually
good, or which reviews were actually bought. So this module generates a world
where the answer key is known by construction -- every agent has a latent
quality it was built with, and every malicious agent is tagged -- and
`evaluate.py` scores the models against it.

The same generator drives the site's sandbox, which is why the whole thing is
seeded: `build_world(seed=7)` is byte-identical on every call, on any machine,
so the demo a recruiter clicks is the demo the eval harness measured.

The sandbox is kept strictly separate from the real ledger. Synthetic agents
are never written to Supabase -- a reputation system that seeds itself with
invented reviews is worth nothing, even if the invented reviews are only
"for the demo".
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Literal

from .trust import DIMENSIONS, STAR_MAX, STAR_MIN, p_to_stars, stars_to_p

AttackKind = Literal["collusion_ring", "review_bomb", "sybil_boost"]

# Flavour vocabulary, so the graph reads like a town of services rather than
# agent-01..agent-40. Combined deterministically from the seeded RNG.
_PREFIXES = (
    "atlas", "beacon", "cobalt", "delta", "ember", "flint", "gale", "harbor",
    "indigo", "juniper", "kestrel", "lumen", "meridian", "nimbus", "onyx",
    "quartz", "ridge", "sable", "tern", "umbra", "vesper", "willow",
)
_SUFFIXES = (
    "router", "index", "oracle", "scribe", "courier", "ledger", "sentry",
    "forge", "atlas", "probe", "relay", "cache", "warden", "broker",
)


@dataclass
class SimAgent:
    """One agent in the synthetic world, with its ground truth attached."""

    name: str
    persona: str
    quality: float  # Latent overall quality on [0, 1]. The answer key.
    dimension_quality: dict[str, float] = field(default_factory=dict)
    malicious: bool = False
    seed_agent: bool = False  # A verified auditor: where TrustRank injects trust.


@dataclass
class World:
    agents: dict[str, SimAgent]
    reviews: list[dict]
    seeds: list[str]
    attacks: list[dict] = field(default_factory=list)

    @property
    def malicious(self) -> set[str]:
        """Ground-truth culprits, for scoring the detector."""
        return {name for name, agent in self.agents.items() if agent.malicious}

    def truth(self) -> dict[str, float]:
        """Ground-truth quality in stars, for scoring the rating models."""
        return {
            name: p_to_stars(agent.quality)
            for name, agent in self.agents.items()
            if not agent.malicious
        }


def _observe(rng: random.Random, quality: float, bias: float, noise: float) -> int:
    """Turn latent quality into an integer star rating, the way a reviewer would.

    Reviewers do not see quality directly: they see one noisy interaction,
    filtered through their own disposition (`bias`), and then have to round it
    onto a five-point scale. All three of those -- noise, bias, rounding -- are
    why an average of a handful of reviews is a bad estimator, and why the
    shrinkage in `trust.py` earns its keep.
    """
    observed = quality + bias + rng.gauss(0.0, noise)
    stars = round(p_to_stars(max(0.0, min(1.0, observed))))
    return int(max(STAR_MIN, min(STAR_MAX, stars)))


def build_world(
    seed: int = 7,
    honest_agents: int = 26,
    auditors: int = 2,
    density: float = 0.22,
) -> World:
    """Generate an honest baseline world: no fraud, just noisy opinions.

    `density` is the chance any ordered pair of agents has worked together and
    left a review, which is what makes the review graph connected enough for
    trust to propagate from the auditors.
    """
    rng = random.Random(seed)
    agents: dict[str, SimAgent] = {}

    names: set[str] = set()
    while len(names) < honest_agents:
        names.add(f"{rng.choice(_PREFIXES)}-{rng.choice(_SUFFIXES)}")
    ordered_names = sorted(names)

    # Latent quality is Beta-ish: most agents are decent, a few are poor. Using
    # a skewed distribution (not uniform) matters -- it is what makes the
    # empirical-Bayes prior in trust.py fit something non-trivial.
    for name in ordered_names:
        quality = rng.betavariate(5.0, 2.0)
        agents[name] = SimAgent(
            name=name,
            persona=rng.choice(["specialist", "generalist", "workhorse", "flaky"]),
            quality=quality,
            # Per-dimension quality wobbles around the agent's overall level:
            # a fast agent can still be inaccurate.
            dimension_quality={
                dimension: max(0.0, min(1.0, quality + rng.gauss(0.0, 0.12)))
                for dimension in DIMENSIONS
            },
        )

    # Auditors are the seed set: verified, honest, low-noise, and they review
    # widely. They are the root of trust -- everything else is derived.
    seeds: list[str] = []
    for index in range(auditors):
        name = f"vouchnet-auditor-{index + 1}"
        seeds.append(name)
        agents[name] = SimAgent(
            name=name,
            persona="auditor",
            quality=rng.betavariate(9.0, 2.0),
            dimension_quality={dimension: 0.9 for dimension in DIMENSIONS},
            seed_agent=True,
        )

    # Reviewer dispositions: some reviewers are generous, some are harsh, some
    # are just noisy. Bias is a property of the reviewer, not the reviewed --
    # which is precisely the confound a plain average cannot see through.
    disposition = {
        name: (rng.gauss(0.0, 0.06), rng.uniform(0.05, 0.18))
        for name in agents
    }
    for name in seeds:
        disposition[name] = (0.0, 0.04)  # Auditors are calibrated.

    reviews: list[dict] = []

    def leave_review(reviewer: str, target: str) -> None:
        bias, noise = disposition[reviewer]
        agent = agents[target]
        stars = _observe(rng, agent.quality, bias, noise)
        # Reviewers only score the dimensions they actually observed -- the
        # "unrated is not zero" rule the pentagon depends on.
        observed = [d for d in DIMENSIONS if rng.random() < 0.55]
        dimensions = {
            dimension: _observe(rng, agent.dimension_quality[dimension], bias, noise)
            for dimension in observed
        }
        reviews.append(
            {
                "agent": target,
                "stars": stars,
                "comment": "",
                "reviewer": reviewer,
                "dimensions": dimensions or None,
                "t": len(reviews),
                "synthetic": True,
            }
        )

    # Auditors review broadly; that is what makes them useful as seeds.
    for auditor in seeds:
        for name in ordered_names:
            if rng.random() < 0.55:
                leave_review(auditor, name)

    # Honest agents review each other after working together. This mutual
    # reviewing is what lets trust flow outward from the auditors -- and its
    # absence is exactly what will strand the attackers.
    for reviewer in ordered_names:
        for target in ordered_names:
            if reviewer != target and rng.random() < density:
                leave_review(reviewer, target)

    return World(agents=agents, reviews=reviews, seeds=seeds)


def pick_target(world: World, rng: random.Random) -> str:
    """Choose a plausible victim: a mid-quality agent with something to gain."""
    candidates = sorted(
        (name for name, agent in world.agents.items() if not agent.malicious and not agent.seed_agent),
        key=lambda name: abs(world.agents[name].quality - 0.55),
    )
    return candidates[0] if candidates else rng.choice(sorted(world.agents))


def apply_attack(
    world: World,
    kind: AttackKind,
    target: str | None = None,
    size: int = 8,
    seed: int | None = None,
) -> World:
    """Mutate `world` in place with one attack, and record what it did.

    Every attacker is tagged `malicious=True`, which is the label the detector
    is scored against in `evaluate.py`. The attacker never gets reviewed by an
    honest agent -- that is not us being kind to our own model, it is the
    actual structure of the problem: an attacker can manufacture any number of
    accounts and any number of reviews, but it cannot manufacture *being
    vouched for by someone who matters*.
    """
    rng = random.Random(seed if seed is not None else (len(world.reviews) * 131 + size))
    target = target or pick_target(world, rng)
    tag = f"{kind[:4]}{len(world.attacks) + 1}"
    members: list[str] = []

    def enlist(label: str) -> str:
        name = f"{label}-{tag}-{len(members) + 1:02d}"
        world.agents[name] = SimAgent(
            name=name,
            persona=kind,
            quality=rng.betavariate(2.0, 5.0),  # Attackers are junk agents.
            dimension_quality={dimension: 0.2 for dimension in DIMENSIONS},
            malicious=True,
        )
        members.append(name)
        return name

    def file(reviewer: str, agent: str, stars: int) -> None:
        world.reviews.append(
            {
                "agent": agent,
                "stars": stars,
                "comment": "",
                "reviewer": reviewer,
                "dimensions": {dimension: stars for dimension in DIMENSIONS},
                "t": len(world.reviews),
                "synthetic": True,
                "attack": kind,
            }
        )

    if kind == "collusion_ring":
        # Classic reciprocal boosting: a clique that praises itself to
        # manufacture standing, then spends that standing on the target.
        for _ in range(size):
            enlist("ring")
        for member in members:
            for other in members:
                if member != other:
                    file(member, other, 5)
            file(member, target, 5)

    elif kind == "sybil_boost":
        # No pretence of a ring: throwaway accounts, all praising the target.
        for _ in range(size):
            enlist("sybil")
        for member in members:
            file(member, target, 5)

    elif kind == "review_bomb":
        # The same shape, aimed the other way -- burying a competitor.
        for _ in range(size):
            enlist("bomber")
        for member in members:
            file(member, target, 1)

    else:
        raise ValueError(f"unknown attack kind: {kind!r}")

    world.attacks.append(
        {"kind": kind, "target": target, "size": size, "members": members}
    )
    return world


def build_scenario(seed: int = 7, attacks: list[dict] | None = None) -> World:
    """Build a world and apply a list of attacks -- the sandbox entry point.

    Stateless and deterministic: the same (seed, attacks) always yields the
    same world, so the browser can hold the attack list and the server never
    has to remember a session.
    """
    world = build_world(seed=seed)
    for index, attack in enumerate(attacks or []):
        apply_attack(
            world,
            kind=attack["kind"],
            target=attack.get("target"),
            size=int(attack.get("size", 8)),
            seed=seed * 1000 + index,
        )
    return world


def world_to_dict(world: World) -> dict:
    return {
        "agents": {name: asdict(agent) for name, agent in world.agents.items()},
        "reviews": world.reviews,
        "seeds": world.seeds,
        "attacks": world.attacks,
    }
