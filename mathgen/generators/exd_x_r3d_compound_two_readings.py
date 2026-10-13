"""Exam source of `senior-consider-similar-problems-involving-compounding-for-example-population`.

The practice tiers give the starting value and the rate, and ask for a later or an earlier value. An exam question
gives neither: it states the population after 1 period and after 3 periods, and the learner works through

(a) the percentage change per period (a square root of the ratio of the two readings, minus one);
(b) the starting population (the reading after 1 period, undone with the multiplier just found);
(c) the number of whole periods until the population first passes a stated threshold.

Parts (b) and (c) follow through from the earlier answers: (b) is the first reading divided by the multiplier from (a),
and (c) is the least whole number of periods n with start * multiplier^n past the threshold, from a logarithm.

Built backwards from the multiplier (a ratio of small integers) and a starting value divisible by the cube of its
denominator, so both readings are whole numbers. The threshold is a round number strictly between the values after
n - 1 and n periods. An independent check steps the population forward in exact fractions until the threshold is
crossed.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, latex_answer

from .exd_x_r3d_common import tagged
from ._exd_distinct_parts import distinct_part_answers

_RATES = (10, 20, 25, 30, 40, 50)

# (subject, unit, largest sensible percentage change per period)
_GROWTH = (
    ("The population of a town", "year", 30),
    ("The number of fish in a dam", "year", 40),
    ("The number of bacteria in a culture", "hour", 50),
    ("The number of rabbits on an island", "year", 50),
)
_DECAY = (
    ("The population of a country village", "year", 30),
    ("The number of seals in a colony", "year", 25),
    ("The number of bacteria in a treated culture", "hour", 50),
    ("The number of fish in a polluted river", "year", 40),
)
_PAIRS = (
    "{subject} is {p1} after 1 {unit} and {p3} after 3 {unit}s. It changes by the same percentage each {unit}.",
    "{subject} is {p1} after 1 {unit}, and {p3} after 3 {unit}s, changing by a constant percentage each {unit}.",
)


def _round_threshold(lo: Fraction, hi: Fraction) -> list[int]:
    """Round integers strictly inside (lo, hi), clear of both ends."""
    step = 100 if hi - lo > 600 else 50 if hi - lo > 250 else 10 if hi - lo > 40 else 1
    first = math.floor(lo / step) * step + step
    return [t for t in range(first, int(hi) + 1, step) if lo + 1 < t < hi - 1]


@generator("exd_x_r3d_compound_two_readings", level=15, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_compound_two_readings(rng: random.Random, tier: str = "exam") -> dict:
    """The rate, the starting size and the first period past a threshold, from two readings of a population."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(3000, "a population with usable readings and threshold"):
        up = rng.random() < 0.5
        subject, unit, top = rng.choice(_GROWTH if up else _DECAY)
        g = rng.choice([r for r in _RATES if r <= top])
        mult = Fraction(100 + g if up else 100 - g, 100)
        q = mult.denominator
        p0 = q**3 * rng.randint(1, 60)
        if not 200 <= p0 <= 6000:
            continue
        n = rng.randint(4, 9)
        values = [Fraction(p0) * mult**i for i in range(n + 2)]
        p1, p3 = values[1], values[3]
        lo, hi = (values[n - 1], values[n]) if up else (values[n], values[n - 1])
        if p1.denominator != 1 or p3.denominator != 1 or lo < 10:
            continue
        thresholds = _round_threshold(lo, hi)
        if not thresholds:
            continue
        threshold = rng.choice(thresholds)
        first = next(i for i, v in enumerate(values) if (v > threshold if up else v < threshold))
        check(first == n, "stepping route disagrees with the construction")
        check(math.floor(math.log(threshold / p0) / math.log(float(mult))) + 1 == n, "logarithm route disagrees")
        # The rate must come out of the readings: (p3 / p1) = mult^2.
        check(p3 / p1 == mult**2, "readings do not give the multiplier")
        a_cands = [
            (100 * mult, "new-as-percent-of-original"),
            (100 * (p3 / p1 - 1) if up else 100 * (1 - p3 / p1), "two-period-change-as-rate"),
            (50 * (p3 / p1 - 1) if up else 50 * (1 - p3 / p1), "compound-change-divided-by-periods"),
        ]
        reverse = Fraction(100 - g if up else 100 + g, 100)
        b_cands = [
            (p1 * mult, "growth-multiplier-applied-forward"),
            (p1 * reverse, "subtracts-percent" if up else "adds-percent-to-undo-decrease"),
            (p1 / mult**2, "undone-for-too-many-periods"),
            (p1, "reads-off-the-given-value"),
        ]
        per = Fraction(p0 * g, 100)
        gap = threshold - p0 if up else p0 - threshold
        linear = math.ceil(Fraction(gap) / per)
        c_cands = [
            (n - 1, "threshold-period-off-by-one"),
            (n + 1, "threshold-period-off-by-one"),
            (linear, "simple-growth-for-compound-count"),
        ]
        fmt_frac = lambda v: str(int(v))  # noqa: E731 - whole numbers only (filtered below)
        a_d = tagged(str(g), [c for c in a_cands if c and Fraction(c[0]).denominator == 1], fmt_frac)
        b_d = tagged(str(p0), [c for c in b_cands if Fraction(c[0]).denominator == 1], fmt_frac)
        c_d = tagged(str(n), [c for c in c_cands if c[0] >= 1])
        if len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 2:
            break
    verb = "increase" if up else "decrease"
    goal = "exceed" if up else "fall below"
    stem = rng.choice(_PAIRS).format(
        subject=subject,
        p1=f"{int(p1):,}",
        p3=f"{int(p3):,}",
        unit=unit,
    )
    b_ft = f"{int(p1)}/(1 {'+' if up else '-'} part_a/100)"
    c_ft = f"floor(log({threshold}/part_b)/log(1 {'+' if up else '-'} part_a/100)) + 1"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the percentage by which it {verb}s each {unit}.",
                answer_kind="number",
                answer=g,
                answer_text=str(g),
                answer_latex=latex_answer(g),
                marks=2,
                sensible_range=(0, 100),
                working=[
                    f"The multiplier per {unit} m satisfies m² = {int(p3)}/{int(p1)} = {mult**2}, so m = {mult} and the change is {g}%."
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"Hence find the starting value, at 0 {unit}s. Give a whole number.",
                answer_kind="number",
                answer=p0,
                answer_text=str(p0),
                answer_latex=latex_answer(p0),
                marks=1,
                follow_through=b_ft,
                sensible_range=(0, None),
                working=[f"Start = {int(p1)} ÷ {mult} = {p0}."],
                distractors=b_d,
            ),
            dict(
                prompt=f"After how many whole {unit}s from the start will it first {goal} {threshold:,}? "
                f"Do not round the value at the end of each {unit}.",
                answer_kind="number",
                answer=n,
                answer_text=str(n),
                answer_latex=latex_answer(n),
                marks=2,
                follow_through=c_ft,
                sensible_range=(0, None),
                working=[
                    f"{p0}·{mult}^{n - 1} = {float(values[n - 1]):.1f} and {p0}·{mult}^{n} = {float(values[n]):.1f}, so it first passes {threshold} after {n} {unit}s."
                ],
                distractors=c_d,
            ),
        ],
    )
