"""Exam-tier source for `interpolation_extrapolation_classify`: the prediction point is not handed over as an
x-value inside the data's own units. The learner must first turn the request into an x-value, then classify it
against the stated data range.

Two framings, chosen per seed:

``calendar``  The line is written in a shifted variable ``t`` (years since a base year) and the prediction is
              requested at a value of ``t``, but the data range is quoted as calendar years, so the range must be
              converted before the comparison.
``solve``     The line ``y = a + b*x`` is fitted to data with ``x`` in ``[xmin, xmax]`` and the request is "when does
              the model reach ``Y0``?". The learner solves ``a + b*x = Y0`` for ``x`` and only then classifies it.

Both are built backwards from the chosen prediction point, so the key is exact. The prediction lands strictly
inside, exactly on a boundary (interpolation, boundary inclusive) or outside the range. The answer options are the
skill's four fixed texts; the exam source offers the opposite category and both padding options as distractors.
"""

from __future__ import annotations

import random

from ..core import check, generator, latex_answer
from .p5d_causality_interpolation_build import (
    OPTION_BOTH_B,
    OPTION_EXTRAPOLATION,
    OPTION_INTERPOLATION,
    OPTION_NEITHER_B,
)

_CONTEXTS = (
    ("the price of a share", "dollars"),
    ("the population of a town", "hundreds of people"),
    ("the yearly rainfall at a station", "millimetres"),
    ("the annual sales of a shop", "thousands of dollars"),
)


def _line_text(a: int, b: int, var: str) -> str:
    return f"y = {a} {'+' if b > 0 else '-'} {abs(b)}{var}"


def _position(rng: random.Random, lo: int, hi: int) -> int:
    """A prediction point: inside, on a boundary or outside [lo, hi] (never below 1)."""
    kind = rng.choice(["inside", "inside", "boundary", "outside_hi", "outside_lo"])
    if kind == "inside":
        return rng.randint(lo + 1, hi - 1)
    if kind == "boundary":
        return rng.choice([lo, hi])
    if kind == "outside_hi":
        return rng.randint(hi + 1, hi + 6)
    return rng.randint(max(1, lo - 6), lo - 1)


def _calendar(rng: random.Random) -> tuple[str, bool]:
    base = rng.choice([1990, 2000, 2005, 2010])
    t_lo = rng.randint(3, 6)
    t_hi = t_lo + rng.randint(6, 12)
    t0 = _position(rng, t_lo, t_hi)
    check(t0 >= 1, "prediction must fall after the base year")
    a = rng.randint(10, 60)
    b = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
    thing, unit = rng.choice(_CONTEXTS)
    text = (
        f"A line {_line_text(a, b, 't')} is fitted to data on {thing} (in {unit}), where t is the number of "
        f"years after {base}. The data were collected from {base + t_lo} to {base + t_hi} inclusive. "
        f"The line is used to estimate the value when t = {t0}. Is this interpolation or extrapolation?"
    )
    return text, t_lo <= t0 <= t_hi


def _solve(rng: random.Random) -> tuple[str, bool]:
    lo = rng.randint(3, 10)
    hi = lo + rng.randint(8, 16)
    x0 = _position(rng, lo, hi)
    b = rng.choice([-6, -4, -3, -2, 2, 3, 4, 6])
    a = rng.randint(-20, 40)
    y0 = a + b * x0
    if y0 <= 0:
        a += -y0 + rng.randint(5, 20)  # keep the target value positive
        y0 = a + b * x0
    check(y0 > 0 and a + b * x0 == y0, "target value must be positive and consistent with the line")
    thing, unit = rng.choice(_CONTEXTS)
    text = (
        f"A line {_line_text(a, b, 'x')} is fitted to data on {thing} (in {unit}), where x is the number of "
        f"years since records began. The data cover x from {lo} to {hi}. "
        f"The line is used to find when the value is predicted to reach {y0}. "
        f"Is this prediction interpolation or extrapolation?"
    )
    return text, lo <= x0 <= hi


@generator(
    "exd_x_s3_interp",
    level=11,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s3_interp(rng, tier="exam"):
    """Classify a prediction as interpolation or extrapolation after converting units or solving for x."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    text, inside = _calendar(rng) if rng.random() < 0.5 else _solve(rng)
    answer = OPTION_INTERPOLATION if inside else OPTION_EXTRAPOLATION
    opposite = OPTION_EXTRAPOLATION if inside else OPTION_INTERPOLATION
    check(answer != opposite, "key and opposite category must differ")
    working = [
        "Convert the request to a value of the model's own variable, then compare it with the range of the data "
        "(the boundary counts as inside)."
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=[opposite, OPTION_BOTH_B, OPTION_NEITHER_B],
    )
