"""Exam-tier source for `f10-unmarked-partial-measurement`: two readings on one scale, combined and converted.

The practice tiers give one reading and the direction to count in. Here the learner first works out what one
division is worth, then reads two values from the same scale, one counted up from a lower labelled mark and one
counted down from an upper labelled mark, and finally combines them and converts to a larger unit. Parts (b),
(c) and (d) carry follow-through.

Built backwards from the size of one division, so every reading is a whole number of the small unit.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, generator, phrase

from .exd_x_m3_support import finish, numeric_part

# (instrument, small unit, large unit, small units per large unit, decimal places of the large unit)
_FAMILIES = [
    ("measuring jug", "mL", "L", 1000, 3),
    ("kitchen scale", "g", "kg", 1000, 3),
    ("tape measure", "mm", "m", 1000, 3),
    ("builder's tape", "cm", "m", 100, 2),
]
_STEMS = [
    "A {inst} has marks labelled every {step} {u}. Between the {lo} {u} mark and the {hi} {u} mark there are {n} equal "
    "unmarked divisions. Reading P is {d1} above the {lo} {u} mark. Reading Q is {d2} below the "
    "{hi} {u} mark.",
    "On a {inst}, the labelled marks are {step} {u} apart. There are {n} equal unmarked divisions between the "
    "{lo} {u} and {hi} {u} marks. Reading P sits {d1} past the {lo} {u} mark and reading Q sits {d2} "
    "short of the {hi} {u} mark.",
]


def _divisions(count: int) -> str:
    return f"{count} division" if count == 1 else f"{count} divisions"


def _whole(pairs):
    """Only whole-number wrong values for a whole-number key: a decimal would give the key away."""
    return [(v, tag) for v, tag in pairs if Fraction(v).denominator == 1]


@generator("exd_x_m3_scale_reading_chain", level=9, presets={"exam": dict(tier="exam")})
def exd_x_m3_scale_reading_chain(rng: random.Random, tier: str = "exam") -> dict:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a scale with distinct readings"):
        inst, u, big, factor, dp = rng.choice(_FAMILIES)
        n = rng.choice([4, 5, 8, 10])
        q = rng.choice([2, 5, 10, 20, 25, 50])
        step = q * n
        m = rng.randint(1, 5)
        lo, hi = m * step, (m + 1) * step
        d1, d2 = rng.randint(1, n - 1), rng.randint(1, n - 1)
        if d1 == d2:
            continue
        p_read = lo + d1 * q
        q_read = hi - d2 * q
        total = (p_read + q_read) / factor
        if len({q, p_read, q_read, total}) < 4:  # no two parts may share an answer (P = Q repeated one reply)
            continue
        stem = phrase(rng, _STEMS, inst=inst, step=step, u=u, lo=lo, hi=hi, n=n, d1=_divisions(d1), d2=_divisions(d2))
        parts = [
            numeric_part(
                f"Find the size of one division, in {u}.",
                q,
                0,
                1,
                [f"{hi} - {lo} = {step} {u} is shared between {n} divisions, so each is {step} ÷ {n} = {q} {u}."],
                _whole(
                    [
                        (step, "major-interval-as-division"),
                        (Fraction(step, n + 1), "division-count-off-by-one"),
                        (Fraction(step, n - 1), "division-count-off-by-one"),
                        (n, None),  # the number of divisions taken as the size of one
                    ]
                ),
            ),
            numeric_part(
                f"Find reading P, in {u}.",
                p_read,
                0,
                1,
                [f"P = {lo} + {d1} × {q} = {p_read} {u}."],
                [
                    (hi - d1 * q, "counts-from-wrong-mark"),
                    (lo + d1 * step, "major-interval-as-division"),
                    (lo + d1, None),
                ],
                follow_through=f"{lo} + {d1}*part_a",
            ),
            numeric_part(
                f"Find reading Q, in {u}.",
                q_read,
                0,
                1,
                [f"Q = {hi} - {d2} × {q} = {q_read} {u}."],
                [
                    (lo + d2 * q, "counts-from-wrong-mark"),
                    (hi - d2 * step, "major-interval-as-division"),
                    (hi - d2, None),
                ],
                follow_through=f"{hi} - {d2}*part_a",
            ),
            numeric_part(
                f"Find the total of readings P and Q, in {big}. Give your answer to {dp} decimal places.",
                total,
                dp,
                2,
                [
                    f"P + Q = {p_read} + {q_read} = {p_read + q_read} {u}.",
                    f"Divide by {factor} to change {u} to {big}.",
                ],
                [
                    (p_read + q_read, "unit-not-converted"),
                    ((p_read + q_read) / (factor // 10), "unit-factor-wrong-power"),
                    ((p_read + q_read) * factor, "unit-conversion-wrong-way"),
                    (abs(p_read - q_read) / factor, None),
                ],
                follow_through=f"(part_b + part_c)/{factor}",
            ),
        ]
        out = finish(stem, parts)
        if out is not None:
            return out
