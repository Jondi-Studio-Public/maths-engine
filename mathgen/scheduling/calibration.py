"""Calibration from learners' attempt data.

Two read-only measurements, both pure functions over plain row dicts an app
supplies (no learner data ships with this package):

* **Item difficulty.** How hard each (skill, difficulty tier) actually is for the
  household, as a Rasch (one-parameter logistic) difficulty in logits: learner
  ability and item difficulty are fitted together, so a tier attempted mostly by
  the strongest learner is not mistaken for an easy one. A tier's difficulty is
  only reported once it has `min_attempts`. Tiers whose measured order disagrees
  with their YAML order (a later tier clearly easier than an earlier one) are
  flagged, since `difficulty_progression` is meant to be real escalation.
* **FSRS calibration.** For every review FSRS scored with a prediction, the
  predicted recall beside what happened, binned, with log loss and the binned
  RMSE FSRS itself reports. This is the check to run before switching a profile
  to FSRS scheduling.

Nothing here writes or feeds back into scheduling. Feeding measured difficulty
into FSRS's starting state is a later step, once there is enough data to show it
helps.
"""

from __future__ import annotations

import math
from collections import defaultdict

# The public names: see docs/api.md. Anything else here may change in any release.
__all__ = [
    "item_difficulty",
    "fsrs_calibration",
]

PRIOR_SD = 2.0  # logits; see `rasch`
RASCH_ITERATIONS = 50
WILSON_Z = 1.96


def wilson_interval(correct: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def rasch(responses: dict[tuple, dict[str, list[int]]]) -> tuple[dict[tuple, float], dict[str, float]]:
    """Rasch fit by maximum a posteriori, alternating damped Newton steps.

    `responses[item][learner] = [correct, n]`. Returns (item difficulty, learner
    ability), both in logits, item difficulties centred on zero. A normal prior
    (sd PRIOR_SD) on every parameter keeps perfect and zero scores finite, which
    a household's small samples hit often."""
    items = list(responses)
    learners = sorted({u for by_user in responses.values() for u in by_user})
    by_learner: dict[str, list[tuple[tuple, int, int]]] = defaultdict(list)
    for i, by_user in responses.items():
        for u, (c, n) in by_user.items():
            by_learner[u].append((i, c, n))
    b = {i: 0.0 for i in items}
    theta = {u: 0.0 for u in learners}
    precision = 1 / PRIOR_SD**2

    def newton(grad: float, info: float) -> float:
        return max(-1.0, min(1.0, grad / (info + precision)))

    for _ in range(RASCH_ITERATIONS):
        for u in learners:
            grad = -theta[u] * precision
            info = 0.0
            for i, c, n in by_learner[u]:
                p = _sigmoid(theta[u] - b[i])
                grad += c - n * p
                info += n * p * (1 - p)
            theta[u] += newton(grad, info)
        for i in items:
            grad = -b[i] * precision
            info = 0.0
            for u, (c, n) in responses[i].items():
                p = _sigmoid(theta[u] - b[i])
                grad += n * p - c
                info += n * p * (1 - p)
            b[i] += newton(grad, info)
    mean_b = sum(b.values()) / len(b) if b else 0.0
    return {i: v - mean_b for i, v in b.items()}, {u: v - mean_b for u, v in theta.items()}


def _tier_order(skill) -> list:
    """Difficulty labels in the order the skill's sources first use them."""
    order = []
    for src in getattr(skill, "sources", []) or []:
        if src.difficulty not in order:
            order.append(src.difficulty)
    return order


def item_difficulty(attempt_rows: list[dict], skills: dict, min_attempts: int = 20) -> dict:
    """Measured difficulty per (skill, tier) and tiers whose order looks inverted.

    `attempt_rows`: dicts with user_id, skill_id, difficulty, correct.
    `skills`: skill_id -> Skill, for each skill's tier order. Returns
    {"skills": [...], "items": [...], "inversions": [...], "learners": int,
    "attempts": int}: `skills` pooled over tiers, `items` per tier."""
    responses: dict[tuple, dict[str, list[int]]] = defaultdict(dict)
    for row in attempt_rows:
        key = (row["skill_id"], row["difficulty"])
        cell = responses[key].setdefault(row["user_id"], [0, 0])
        cell[0] += 1 if row["correct"] else 0
        cell[1] += 1

    counted = {
        key: by_user for key, by_user in responses.items() if sum(n for _, n in by_user.values()) >= min_attempts
    }
    b, theta = rasch(counted) if counted else ({}, {})

    items = []
    for (sid, tier), by_user in sorted(counted.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        correct = sum(c for c, _ in by_user.values())
        n = sum(n for _, n in by_user.values())
        low, high = wilson_interval(correct, n)
        items.append(
            {
                "skill_id": sid,
                "difficulty": tier,
                "attempts": n,
                "correct": correct,
                "p_correct": round(correct / n, 4),
                "p_low": round(low, 4),
                "p_high": round(high, 4),
                "rasch_difficulty": round(b[(sid, tier)], 3),
                "learners": len(by_user),
            }
        )

    by_skill: dict[str, dict] = defaultdict(dict)
    for item in items:
        by_skill[item["skill_id"]][item["difficulty"]] = item
    inversions = []
    for sid, tiers in sorted(by_skill.items()):
        order = [t for t in _tier_order(skills.get(sid)) if t in tiers]
        for earlier_idx, earlier in enumerate(order):
            for later in order[earlier_idx + 1 :]:
                e, lt = tiers[earlier], tiers[later]
                if lt["p_low"] > e["p_high"]:  # later tier answered right clearly more often
                    inversions.append(
                        {
                            "skill_id": sid,
                            "earlier": earlier,
                            "later": later,
                            "earlier_p": e["p_correct"],
                            "later_p": lt["p_correct"],
                        }
                    )

    # Tiers rarely reach min_attempts at household volume, so the same fit is
    # also run pooled per skill, which is the figure to trust first.
    pooled: dict[tuple, dict[str, list[int]]] = defaultdict(dict)
    for (sid, _tier), by_user in responses.items():
        for u, (c, n) in by_user.items():
            cell = pooled[(sid,)].setdefault(u, [0, 0])
            cell[0] += c
            cell[1] += n
    pooled = {key: by_user for key, by_user in pooled.items() if sum(n for _, n in by_user.values()) >= min_attempts}
    skill_b, _ = rasch(pooled) if pooled else ({}, {})
    skills_out = []
    for (sid,), by_user in sorted(pooled.items()):
        correct = sum(c for c, _ in by_user.values())
        n = sum(n for _, n in by_user.values())
        skills_out.append(
            {
                "skill_id": sid,
                "attempts": n,
                "correct": correct,
                "p_correct": round(correct / n, 4),
                "rasch_difficulty": round(skill_b[(sid,)], 3),
                "learners": len(by_user),
            }
        )

    return {
        "learners": len(theta),
        "attempts": sum(i["attempts"] for i in items),
        "skills": skills_out,
        "items": items,
        "inversions": inversions,
    }


def fsrs_calibration(log_rows: list[dict], bins: int = 10) -> dict:
    """Predicted recall against observed recall for FSRS-scored reviews.

    `log_rows`: dicts with retrievability (None for first reviews, skipped)
    and rating (1 = forgot). Returns counts, log loss, binned RMSE and the bins."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError(f"bins must be a whole number, 1 or more, got {bins!r}")
    pairs = [(row["retrievability"], 0 if row["rating"] == 1 else 1) for row in log_rows]
    pairs = [(r, y) for r, y in pairs if r is not None]
    for r, _ in pairs:
        if not 0 <= r <= 1:
            raise ValueError(f"retrievability must be from 0 to 1 (or None), got {r!r}")
    if not pairs:
        return {"reviews": 0, "log_loss": None, "rmse_bins": None, "mean_predicted": None, "observed": None, "bins": []}
    eps = 1e-6
    log_loss = -sum(y * math.log(max(eps, r)) + (1 - y) * math.log(max(eps, 1 - r)) for r, y in pairs) / len(pairs)
    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for r, y in pairs:
        grouped[min(bins - 1, int(r * bins))].append((r, y))
    out_bins = []
    sq = 0.0
    for idx in sorted(grouped):
        cell = grouped[idx]
        predicted = sum(r for r, _ in cell) / len(cell)
        observed = sum(y for _, y in cell) / len(cell)
        sq += len(cell) * (predicted - observed) ** 2
        out_bins.append(
            {
                "range": [idx / bins, (idx + 1) / bins],
                "reviews": len(cell),
                "predicted": round(predicted, 4),
                "observed": round(observed, 4),
            }
        )
    return {
        "reviews": len(pairs),
        "log_loss": round(log_loss, 4),
        "rmse_bins": round(math.sqrt(sq / len(pairs)), 4),
        "mean_predicted": round(sum(r for r, _ in pairs) / len(pairs), 4),
        "observed": round(sum(y for _, y in pairs) / len(pairs), 4),
        "bins": out_bins,
    }
