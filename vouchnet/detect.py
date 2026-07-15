"""Graph signals that flag manipulation.

TrustRank already *defends* the score -- a ring with no seed path barely moves
the numbers whatever it does. This module is the complementary job: naming the
attack out loud, so the UI can draw the ring in red and an operator can act on
it. Defense and detection are different problems, and it is worth being clear
about which one is carrying the weight (per `evaluate.py`, the defense is).

Both signals are deliberately structural rather than learned:

- A ring is mutual praise with no outside validation -- a strongly connected
  component in the praise graph whose members nobody credible vouches for.
- A brigade is a pile of same-rating reviews landing on one target from
  reviewers with no history of their own.

A supervised model would fit this dataset better, but it would only learn the
fraud patterns `simulate.py` already knows how to plant -- circular, and it
would break the moment a real attacker did something new. Structural rules
carry an argument about *why* they generalize, and they are explainable to the
agent that gets flagged, which matters when the flag is public.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Mapping, Sequence

# A review at or above this endorsement level counts as "praise" for the
# purpose of ring-finding: 4 stars maps to p = 0.75.
PRAISE_THRESHOLD = 0.75
MIN_RING_SIZE = 3
MIN_BRIGADE_SIZE = 4

# What counts as "no standing": trust below this fraction of the median agent's.
#
# This threshold is the whole ballgame for precision. The obvious choice --
# below-median trust -- is useless, because half of all honest agents are below
# median by definition, and a well-liked agent collecting consistent 5s from
# ordinary peers looks exactly like a brigade under that rule. (Measured: it
# flagged 23 agents to catch 10 attackers.)
#
# The real asymmetry is not that attackers have *little* trust, it is that they
# have *none*: no path from a seed reaches them, so their TrustRank is
# identically zero, while any agent a verified auditor has touched is strictly
# positive. A fraction of the median separates those two populations by a
# margin instead of splitting one population down the middle.
LOW_TRUST_FRACTION = 0.1


@dataclass
class Flag:
    """Why one agent looks suspicious, and how strongly."""

    agent: str
    risk: float = 0.0  # 0 = clean, 1 = almost certainly manipulating.
    reasons: list[str] = field(default_factory=list)
    ring: list[str] = field(default_factory=list)  # Co-members, if in a ring.

    @property
    def flagged(self) -> bool:
        return self.risk >= 0.5


def _praise_edges(reviews: Sequence[Mapping]) -> dict[str, set[str]]:
    """reviewer -> {agents it praised}, ignoring self-reviews."""
    edges: dict[str, set[str]] = {}
    for review in reviews:
        reviewer = review.get("reviewer") or "anonymous"
        target = review["agent"]
        if reviewer == target:
            continue
        if (review["stars"] - 1.0) / 4.0 >= PRAISE_THRESHOLD:
            edges.setdefault(reviewer, set()).add(target)
    return edges


def strongly_connected_components(edges: Mapping[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCC algorithm, iterative so deep graphs cannot blow the stack.

    A cycle in the praise graph means "I praise you, you praise me" (possibly
    through intermediaries). That is not proof of fraud -- genuinely good
    agents that work together will praise each other too -- which is why the
    caller intersects this with "and nobody credible vouches for any of you".
    """
    index_of: dict[str, int] = {}
    low_link: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    result: list[list[str]] = []
    counter = 0

    nodes = set(edges) | {target for targets in edges.values() for target in targets}

    for root in sorted(nodes):
        if root in index_of:
            continue
        # Each frame is [node, iterator over its successors].
        work: list[tuple[str, list[str]]] = [(root, sorted(edges.get(root, ())))]
        index_of[root] = low_link[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True

        while work:
            node, successors = work[-1]
            progressed = False
            while successors:
                successor = successors.pop(0)
                if successor not in index_of:
                    index_of[successor] = low_link[successor] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack[successor] = True
                    work.append((successor, sorted(edges.get(successor, ()))))
                    progressed = True
                    break
                if on_stack.get(successor):
                    low_link[node] = min(low_link[node], index_of[successor])
            if progressed:
                continue

            work.pop()
            if work:
                parent = work[-1][0]
                low_link[parent] = min(low_link[parent], low_link[node])
            if low_link[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                result.append(sorted(component))
    return result


def low_trust_cutoff(
    trust: Mapping[str, float],
    fraction: float = LOW_TRUST_FRACTION,
) -> float:
    """The trust level below which an agent has no meaningful standing.

    Relative to the median agent rather than absolute, because TrustRank mass
    is a probability: its scale shrinks as the network grows, so any fixed
    constant would silently stop working at a different network size.
    """
    positive = [value for value in trust.values() if value > 0.0]
    return fraction * median(positive) if positive else 0.0


def find_rings(
    reviews: Sequence[Mapping],
    trust: Mapping[str, float],
    min_size: int = MIN_RING_SIZE,
) -> list[list[str]]:
    """Mutual-praise cliques that no trusted agent vouches for.

    The trust test is what separates a collusion ring from a legitimate team:
    both praise each other, but only one of them is also praised from outside.
    """
    cutoff = low_trust_cutoff(trust)
    components = strongly_connected_components(_praise_edges(reviews))
    rings = []
    for component in components:
        if len(component) < min_size:
            continue
        if all(trust.get(member, 0.0) < cutoff for member in component):
            rings.append(component)
    return rings


def find_brigades(
    reviews: Sequence[Mapping],
    trust: Mapping[str, float],
    min_size: int = MIN_BRIGADE_SIZE,
) -> dict[str, list[str]]:
    """Targets hit by a bloc of near-identical ratings from unknown reviewers.

    Catches both directions -- a review bomb and a sybil boost are the same
    shape, just aimed differently. Returns target -> the reviewers involved.

    Suspects are grouped by the rating they gave, and each group is tested on
    its own. Testing the suspect set as a whole looks reasonable and fails
    badly: an agent that is simultaneously boosted and bombed gets a suspect
    set containing both 5s and 1s, whose spread is large, so the bloc test
    passes and *both* attacks go unflagged. (Measured: recall collapsed to
    0.33.) Attackers do not coordinate with each other, so the unit of
    detection has to be one bloc pushing one number, not "everyone unknown".
    """
    cutoff = low_trust_cutoff(trust)

    by_target: dict[str, list[Mapping]] = {}
    for review in reviews:
        by_target.setdefault(review["agent"], []).append(review)

    brigades: dict[str, list[str]] = {}
    for target, target_reviews in by_target.items():
        blocs: dict[int, set[str]] = {}
        for review in target_reviews:
            reviewer = review.get("reviewer") or "anonymous"
            if trust.get(reviewer, 0.0) >= cutoff:
                continue
            blocs.setdefault(int(review["stars"]), set()).add(reviewer)

        # Several accounts with no standing at all, pushing the same agent to
        # the same number, is the signature. Honest newcomers arrive one at a
        # time and disagree; a bloc arrives together and does not.
        involved = {
            reviewer
            for members in blocs.values()
            if len(members) >= min_size
            for reviewer in members
        }
        if involved:
            brigades[target] = sorted(involved)
    return brigades


def flag_agents(
    reviews: Sequence[Mapping],
    trust: Mapping[str, float],
) -> dict[str, Flag]:
    """Run every signal and merge them into one risk score per agent.

    Risk is a bounded sum of independent evidence rather than a learned
    combination -- with the signals this sparse, a fitted weighting would be
    overfitting to `simulate.py`.
    """
    flags: dict[str, Flag] = {}

    def flag_for(agent: str) -> Flag:
        return flags.setdefault(agent, Flag(agent=agent))

    for ring in find_rings(reviews, trust):
        for member in ring:
            flag = flag_for(member)
            flag.risk += 0.6
            flag.ring = [other for other in ring if other != member]
            flag.reasons.append(
                f"mutual-praise ring of {len(ring)} with no outside validation"
            )

    for target, reviewers in find_brigades(reviews, trust).items():
        for reviewer in reviewers:
            flag = flag_for(reviewer)
            flag.risk += 0.5
            flag.reasons.append(f"part of a {len(reviewers)}-account bloc rating {target}")
        # The target is a victim, not a culprit -- record it without adding risk
        # so a review-bombed agent is never punished for being attacked.
        victim = flag_for(target)
        if not victim.reasons:
            victim.risk += 0.0
        victim.reasons.append(f"targeted by a {len(reviewers)}-account bloc")

    for flag in flags.values():
        flag.risk = min(1.0, flag.risk)
    return flags
