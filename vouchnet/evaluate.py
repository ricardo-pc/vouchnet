"""The eval harness: does any of this actually work?

Three questions, three experiments, all against `simulate.py`'s ground truth:

1. **Recovery** -- how close does each model get to an agent's true quality?
   Measured as MAE (in stars) and Spearman rank correlation, because the
   ranking is what a leaderboard actually sells.
2. **Attack resistance** -- when a ring or a sybil swarm attacks one agent,
   how far does its score move? This is the headline: naive averaging has no
   defense at all, and the delta between models is the entire argument for
   TrustRank.
3. **Detection** -- precision/recall/F1 of the structural flags against the
   planted attackers.

Everything is averaged over many seeds and reported with a standard deviation,
because a single seed can say anything. Run it:

    python -m vouchnet.evaluate            # markdown, for the README
    python -m vouchnet.evaluate --json     # machine-readable
    python -m vouchnet.evaluate --trials 50
"""

from __future__ import annotations

import argparse
import json
import random
from statistics import mean, pstdev
from typing import Callable, Sequence

from . import detect, simulate, trust

# The models under test, each mapping an AgentScore to its headline number.
MODELS: dict[str, Callable[[trust.AgentScore], float]] = {
    "naive mean": lambda score: score.naive_stars,
    "bayesian": lambda score: score.bayes_stars,
    "trust-weighted": lambda score: score.trust_stars,
}

ATTACKS: tuple[str, ...] = ("collusion_ring", "sybil_boost", "review_bomb")


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman's rho: Pearson correlation of the ranks, with ties averaged.

    Rank correlation rather than Pearson on the raw values, because the models
    live on different scales -- a shrunk score is deliberately compressed
    toward the prior, and penalising it for that would be measuring the wrong
    thing. What matters is whether it puts the agents in the right order.
    """
    if len(xs) < 2:
        return float("nan")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                result[order[k]] = average_rank
            i = j + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return numerator / denominator if denominator else float("nan")


def _score(world: simulate.World) -> dict[str, trust.AgentScore]:
    return trust.score_all(world.reviews, seeds=world.seeds)


def evaluate_recovery(trials: int = 20) -> dict[str, dict[str, float]]:
    """How well does each model recover latent quality on an honest network?"""
    collected: dict[str, dict[str, list[float]]] = {
        name: {"mae": [], "rho": []} for name in MODELS
    }
    for trial in range(trials):
        world = simulate.build_world(seed=100 + trial)
        scores = _score(world)
        truth = world.truth()
        names = sorted(set(truth) & set(scores))
        actual = [truth[name] for name in names]
        for label, extract in MODELS.items():
            predicted = [extract(scores[name]) for name in names]
            collected[label]["mae"].append(
                mean(abs(p - a) for p, a in zip(predicted, actual))
            )
            collected[label]["rho"].append(spearman(predicted, actual))
    return {
        label: {
            "mae": mean(values["mae"]),
            "mae_sd": pstdev(values["mae"]),
            "rho": mean(values["rho"]),
            "rho_sd": pstdev(values["rho"]),
        }
        for label, values in collected.items()
    }


def evaluate_attacks(trials: int = 20, size: int = 10) -> dict[str, dict[str, dict[str, float]]]:
    """How far does each attack move its target's score, under each model?

    The comparison is before/after on the *same* world with the *same* seed, so
    the only difference is the attack itself.
    """
    collected: dict[str, dict[str, list[float]]] = {
        kind: {label: [] for label in MODELS} for kind in ATTACKS
    }
    for kind in ATTACKS:
        for trial in range(trials):
            world = simulate.build_world(seed=200 + trial)
            before = _score(world)
            target = simulate.pick_target(world, random.Random(trial))
            if target not in before:
                continue
            simulate.apply_attack(world, kind=kind, target=target, size=size, seed=trial)
            after = _score(world)
            for label, extract in MODELS.items():
                collected[kind][label].append(
                    extract(after[target]) - extract(before[target])
                )
    return {
        kind: {
            label: {"shift": mean(values), "shift_sd": pstdev(values)}
            for label, values in per_model.items()
            if values
        }
        for kind, per_model in collected.items()
    }


def evaluate_detection(trials: int = 20, size: int = 10) -> dict[str, float]:
    """Precision/recall/F1 of the structural flags against planted attackers."""
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for trial in range(trials):
        world = simulate.build_world(seed=300 + trial)
        for index, kind in enumerate(ATTACKS):
            simulate.apply_attack(world, kind=kind, size=size, seed=trial * 10 + index)
        ranks = trust.trustrank(world.reviews, seeds=world.seeds)
        flags = detect.flag_agents(world.reviews, ranks)
        predicted = {name for name, flag in flags.items() if flag.flagged}
        actual = world.malicious

        true_positives = len(predicted & actual)
        precision = true_positives / len(predicted) if predicted else 1.0
        recall = true_positives / len(actual) if actual else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "precision": mean(precisions),
        "precision_sd": pstdev(precisions),
        "recall": mean(recalls),
        "recall_sd": pstdev(recalls),
        "f1": mean(f1s),
        "f1_sd": pstdev(f1s),
    }


def run(trials: int = 20, size: int = 10) -> dict:
    return {
        "trials": trials,
        "attack_size": size,
        "recovery": evaluate_recovery(trials),
        "attacks": evaluate_attacks(trials, size),
        "detection": evaluate_detection(trials, size),
    }


def to_markdown(results: dict) -> str:
    trials = results["trials"]
    size = results["attack_size"]
    lines: list[str] = []

    lines.append(f"### Recovering true quality ({trials} seeds, honest network)")
    lines.append("")
    lines.append("| model | MAE (stars) ↓ | Spearman ρ ↑ |")
    lines.append("| --- | --- | --- |")
    for label, values in results["recovery"].items():
        lines.append(
            f"| {label} | {values['mae']:.3f} ± {values['mae_sd']:.3f} "
            f"| {values['rho']:.3f} ± {values['rho_sd']:.3f} |"
        )

    lines.append("")
    lines.append(f"### Attack resistance ({trials} seeds, {size} attacker accounts each)")
    lines.append("")
    lines.append("Star shift on the targeted agent. Closer to zero is better.")
    lines.append("")
    header = "| attack | " + " | ".join(MODELS) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in MODELS) + " |")
    for kind, per_model in results["attacks"].items():
        cells = [
            f"{per_model[label]['shift']:+.2f}" if label in per_model else "n/a"
            for label in MODELS
        ]
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")

    detection = results["detection"]
    lines.append("")
    lines.append(f"### Flagging the attackers ({trials} seeds, all three attacks at once)")
    lines.append("")
    lines.append("| precision | recall | F1 |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| {detection['precision']:.3f} ± {detection['precision_sd']:.3f} "
        f"| {detection['recall']:.3f} ± {detection['recall_sd']:.3f} "
        f"| {detection['f1']:.3f} ± {detection['f1_sd']:.3f} |"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VouchNet's trust models.")
    parser.add_argument("--trials", type=int, default=20, help="seeds per experiment")
    parser.add_argument("--size", type=int, default=10, help="attacker accounts per attack")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args()

    results = run(trials=args.trials, size=args.size)
    print(json.dumps(results, indent=2) if args.json else to_markdown(results))


if __name__ == "__main__":
    main()
