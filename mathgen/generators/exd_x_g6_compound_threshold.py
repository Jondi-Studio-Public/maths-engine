"""Exam-tier compounding question: how long until a threshold is crossed?

Earlier tiers of the population-compounding skill give the number of periods and ask for the final (or the
original) quantity.  Here the number of periods is the UNKNOWN: a population grows (or a colony shrinks) by a
fixed percentage each period and the learner must find the first whole period at which it passes a stated
threshold.  That is the inverse of the exponent, solved by repeated multiplication or by logarithms, and it
needs a decision about rounding (the first period AT which the threshold is passed, not the last before).

The period count ``n`` is chosen first, then the threshold is placed strictly between the values after
``n - 1`` and ``n`` periods, so the key is exact.  All arithmetic is in ``Fraction`` (no rounding), and the
prompt tells the learner not to round the population between periods.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.core import check, check_distinct, generator, latex_answer, pick_distractors

_RATES = (10, 15, 20, 25, 30)

# (subject, verb phrase for growth, unit)
_GROWTH = (
    ("A town has a population of {p0}.", "grows by {g}% each year", "year"),
    ("A fish stock numbers {p0}.", "increases by {g}% each year", "year"),
    ("A bacteria culture contains {p0} cells.", "grows by {g}% each hour", "hour"),
)
_DECAY = (
    ("A bacteria culture contains {p0} cells.", "decreases by {g}% each hour", "hour"),
    ("A village has a population of {p0}.", "falls by {g}% each year", "year"),
    ("A colony numbers {p0} animals.", "declines by {g}% each year", "year"),
)


def _values(p0: int, g: int, up: bool, n_max: int) -> list[Fraction]:
    mult = Fraction(100 + g if up else 100 - g, 100)
    out = [Fraction(p0)]
    for _ in range(n_max):
        out.append(out[-1] * mult)
    return out


@generator("exd_x_g6_compound_threshold", level=12, presets={"exam": dict(tier="exam")})
def compound_threshold(rng: random.Random, tier: str = "exam") -> dict:
    """Smallest whole number of periods for compound growth or decay to pass a threshold."""
    for _ in range(2000):
        up = rng.random() < 0.5
        g = rng.choice(_RATES)
        n = rng.randint(3, 9)
        p0 = rng.randrange(200, 5001, 50)
        vals = _values(p0, g, up, n + 1)
        lo, hi = (vals[n - 1], vals[n]) if up else (vals[n], vals[n - 1])
        # threshold: a round integer strictly inside the interval, clear of both ends
        step = 10 if hi - lo > 60 else 1
        low_t = math.floor(lo / step) * step + step
        cands = [t for t in range(low_t, int(hi) + 1, step) if lo + 1 < t < hi - 1]
        if not cands:
            continue
        thr = rng.choice(cands)
        # Independent route: logarithms.
        ratio = (100 + g if up else 100 - g) / 100
        n_log = math.floor(math.log(thr / p0) / math.log(ratio)) + 1
        check(n_log == n, "logarithm route disagrees with the construction")
        # Simple (linear) change per period, the misconception.
        per = Fraction(p0 * g, 100)
        gap = Fraction(thr - p0) if up else Fraction(p0 - thr)
        linear = math.ceil(gap / per) if gap > 0 else 1
        if linear < 1:
            continue
        if n - 1 < 1 or linear == n:
            continue
        break
    else:
        raise RuntimeError("exd_x_g6_compound_threshold: no valid draw")
    # Exact check by direct stepping.
    first = next(i for i, v in enumerate(vals) if (v > thr if up else v < thr))
    check(first == n, "stepping route disagrees with the construction")
    subject, verb, unit = rng.choice(_GROWTH if up else _DECAY)
    goal = f"first exceed {thr}" if up else f"first fall below {thr}"
    text = (
        f"{subject.format(p0=p0)} Each {unit} it {verb.split(' each ')[0].format(g=g)} of the previous "
        f"{unit}'s value. After how many whole {unit}s will it {goal}? "
        f"Do not round the value at the end of each {unit}."
    )
    answer_text = str(n)
    check_distinct(n, n - 1, n + 1, linear)
    distractors = pick_distractors(answer_text, [str(n - 1), str(linear), str(n + 1)])
    check(len(distractors) >= 2, "need at least two distractors")
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=n,
        answer_text=answer_text,
        answer_latex=latex_answer(n),
        working=[
            f"After {n - 1} {unit}s the value is {float(vals[n - 1]):.1f}; after {n} it is {float(vals[n]):.1f}; "
            f"the threshold {thr} is first passed after {n} {unit}s."
        ],
        distractors=distractors,
    )
