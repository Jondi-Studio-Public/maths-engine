"""Exam tier for "order integers": order places by a temperature that has to be worked out first.

The practice tiers hand over the integers to sort. Here each town's early reading and its overnight change are
given in words ("fell by", "rose by"), so the learner must first find every final temperature, with negative
starting values and falls that cross zero, and only then order the towns from coldest to warmest. The items
ordered are the town names, so the answer cannot be read off the numbers in the prompt.

Built backwards from the final temperatures: draw distinct finals, then a start and a signed change for each.
Distractors are the orderings a learner reaches by a named slip: warmest first, ordering by the start,
reading every fall as a rise, or ordering by distance from zero.
"""

from __future__ import annotations

import random

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, pick_distractors, reversed_and_adjacent_swap

_TOWNS = [
    "Thredbo", "Cooma", "Hobart", "Canberra", "Ballarat", "Orange", "Armidale", "Katoomba", "Bathurst",
    "Goulburn", "Launceston", "Perisher", "Oberon", "Bright",
]  # fmt: skip
_OPENINGS = [
    "At 6 pm the temperature in each town was recorded, and by 6 am it had changed.",
    "Overnight the temperatures in several towns changed from their 9 pm readings.",
]
_ASKS = [
    "Order the towns from coldest to warmest by their final temperature.",
    "List the towns in order from the coldest to the warmest final temperature.",
]


@generator("exd_x_pr7_signed_temperatures_order", level=7, presets={"exam": {"tier": "exam"}})
def exd_x_pr7_signed_temperatures_order(rng: random.Random, tier: str = "exam") -> dict:
    """Find each final temperature from a start and a change, then order the towns. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    count = rng.choice([4, 5])
    for _ in attempts(2000, "towns with distinct starts and finals, a fall crossing zero and a rise"):
        names = rng.sample(_TOWNS, count)
        starts = [rng.randint(-6, 12) for _ in names]
        # Overnight a town cools by a good few degrees but warms by only a little (e.g. a front arriving).
        changes = [rng.randint(2, 12) * -1 if rng.random() < 0.7 else rng.randint(1, 4) for _ in names]
        finals = [s + c for s, c in zip(starts, changes)]
        if len(set(finals)) < count or len(set(starts)) < count:
            continue
        if not any(s > 0 > f for s, f in zip(starts, finals)) or all(c < 0 for c in changes):
            continue
        if not all(-15 <= f <= 14 for f in finals):
            continue
        if len({abs(f) for f in finals}) < count:
            continue
        break

    by_final = sorted(zip(finals, names, starts, changes))
    ordered = [n for _, n, _, _ in by_final]
    # Second route: a bubble sort on the finals recomputed from the sentence-level facts.
    check(ordered == [n for _f, n in sorted(zip((s + c for s, c in zip(starts, changes)), names))], "order disagrees")

    lines = []
    for n, s, c in zip(names, starts, changes):
        verb = "rose" if c > 0 else "fell"
        lines.append(f"{n}: {s} °C, then {verb} by {abs(c)} °C.")
    opening = rng.choice(_OPENINGS)
    text = f"{opening} {' '.join(lines)} {rng.choice(_ASKS)}"

    shown = list(zip(names, starts, changes))
    key = ", ".join(ordered)

    def order_by(keyfn):
        return [n for n, s, c in sorted(shown, key=keyfn)]

    rev, swap = reversed_and_adjacent_swap(ordered)
    by_start = order_by(lambda t: (t[1], t[0]))
    falls_as_rises = order_by(lambda t: (t[1] + abs(t[2]), t[0]))
    by_size = order_by(lambda t: (abs(t[1] + t[2]), t[0]))
    candidates = [
        Distractor(", ".join(rev), "reversed-order", rev),
        Distractor(", ".join(by_start), "ordered-by-starting-value", by_start),
        Distractor(", ".join(falls_as_rises), "fall-treated-as-rise", falls_as_rises),
        Distractor(", ".join(by_size), "magnitude-ignores-sign", by_size),
        Distractor(", ".join(swap), "adjacent-swap", swap),
    ]
    distractors = pick_distractors(key, candidates, 4)
    check(len(distractors) >= 2, "too few distractors")

    return dict(
        text=text,
        answer=ordered,
        answer_kind="ordered_list",
        answer_text=key,
        answer_latex=f"plain:{key}",
        working=[
            *(f"{n}: {s} {'+' if c > 0 else '-'} {abs(c)} = {s + c} °C." for n, s, c in shown),
            f"From coldest to warmest: {key}.",
        ],
        distractors=distractors,
    )
