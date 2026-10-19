"""Exam tier for "examine slope (direction) fields of a first-order differential equation".

The practice tiers read one slope, or its sign, off the equation. Here the slope field is used as it is on an
exam: to walk a solution curve forward in two Euler steps, so the gradient at each new point must be found from
the previous estimate. A solution curve of dy/dx = a x + b y + c passes through (x0, y0); the step size is h.

(a) the gradient at (x0, y0);
(b) the estimate of y at x0 + h (Euler's step, using the gradient from (a));
(c) the gradient at (x0 + h, that estimate);
(d) the estimate of y at x0 + 2h (a second step, using the gradient from (c)).

The step sizes 0.1, 0.2 and 0.5 keep every exact value on a grid of 0.01, so (b), (c) and (d) are stated to
2 decimal places with no rounding at all. Each follow-through is the one-line formula the part uses, written with
the question's own coefficients.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, join_terms, paren_negative
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3g_kit import clashes, number_part

_STEPS = (Fraction(1, 10), Fraction(1, 5), Fraction(1, 2))
_STEMS = (
    "Consider the differential equation dy/dx = {f}, whose solution curve passes through the point ({x0}, {y0}).",
    "The slope field of dy/dx = {f} is used to trace the solution curve that starts at ({x0}, {y0}).",
    "A quantity y depends on x according to dy/dx = {f}, and its graph goes through the point ({x0}, {y0}).",
)


def _text(value: Fraction) -> str:
    """A step size or abscissa as a short decimal (0.5, 1.2, -0.3)."""
    return f"{float(value):g}"


@generator("exd_x_r3g_slope_euler", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3g_slope_euler(rng: random.Random, tier: str = "exam") -> dict:
    """Two Euler steps along the solution curve of dy/dx = a x + b y + c, with the gradient found at each stage."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "an Euler example with moderate gradients"):
        a = rng.choice((-3, -2, -1, 1, 2, 3))
        b = rng.choice((-3, -2, -1, 1, 2, 3))
        c = rng.choice((0, 0, -4, -3, -2, -1, 1, 2, 3, 4))
        x0, y0 = rng.randint(-3, 3), rng.randint(-4, 4)
        h = rng.choice(_STEPS)
        s0 = a * x0 + b * y0 + c
        y1 = y0 + h * s0
        x1 = x0 + h
        s1 = a * x1 + b * y1 + c
        y2 = y1 + h * s1
        if 0 in (s0, s1, y1, y2) or abs(s0) > 14 or abs(s1) > 16 or s1 == s0 or abs(y2) > 30:
            continue
        f = join_terms([(a, "x"), (b, "y"), (c, "")])
        stem = rng.choice(_STEMS).format(f=f, x0=x0, y0=y0)
        step = _text(h)
        pb = (
            f"Euler's method with a step size of {step} is used to estimate the value of y at x = {_text(x1)}. "
            "Find this estimate, correct to 2 decimal places."
        )
        pc = (
            f"Using your estimate from part (b), find the gradient given by the differential equation at "
            f"x = {_text(x1)}, correct to 2 decimal places."
        )
        pd = f"Hence use a second Euler step of size {step} to estimate y at x = {_text(x0 + 2 * h)}, correct to 2 decimal places."
        if clashes((s0, f"{float(y1):.2f}", f"{float(s1):.2f}", f"{float(y2):.2f}"), stem, pb, pc, pd):
            continue
        break

    check(y1 - y0 == h * s0 and y2 - y1 == h * s1, "Euler steps are inconsistent")
    hs = _text(h)
    parts = [
        number_part(
            "Find the gradient of the solution curve at this point.",
            s0,
            0,
            1,
            [f"dy/dx = {f} at ({x0}, {y0}) = {s0}."],
            [
                (a * y0 + b * x0 + c, "ode-coordinates-swapped"),
                (-s0, "sign-of-result-reversed"),
                (a * x0 + b * y0, "dropped-constant-term") if c else None,
                (a * x0 + c, "ode-slope-from-one-variable-only"),
                (b * y0 + c, "ode-slope-from-one-variable-only"),
                (a * x0 - b * y0 + c, "sign-of-result-reversed"),
            ],
        ),
        number_part(
            pb,
            y1,
            2,
            2,
            [f"y at x = {_text(x1)}: {y0} + {hs} x {paren_negative(s0)} = {float(y1):.2f}."],
            [
                (y0 + s0, "euler-step-size-not-applied"),
                (h * s0, "euler-increment-reported-as-estimate"),
                (y0 + h, "euler-step-size-not-applied"),
                (y0 - h * s0, "subtracted-instead-of-added"),
            ],
            follow_through=f"{y0} + ({h})*part_a",
        ),
        number_part(
            pc,
            s1,
            2,
            1,
            [f"dy/dx = {join_terms([(a, f'({_text(x1)})'), (b, f'({float(y1):.2f})'), (c, '')])} = {float(s1):.2f}."],
            [
                (a * x1 + b * y0 + c, "euler-gradient-uses-old-point"),
                (a * x0 + b * y1 + c, "euler-gradient-uses-old-point"),
                (s0, "euler-gradient-uses-old-point"),
                (a * y1 + b * x1 + c, "ode-coordinates-swapped"),
            ],
            follow_through=f"({a})*({x1}) + ({b})*part_b + ({c})",
        ),
        number_part(
            pd,
            y2,
            2,
            2,
            [f"y at x = {_text(x0 + 2 * h)}: {float(y1):.2f} + {hs} x {paren_negative(float(s1))} = {float(y2):.2f}."],
            [
                (y1 + h * s0, "euler-gradient-uses-old-point"),
                (y1 + s1, "euler-step-size-not-applied"),
                (y0 + h * s1, "euler-restarts-from-initial-value"),
                (h * s1, "euler-increment-reported-as-estimate"),
            ],
            follow_through=f"part_b + ({h})*part_c",
        ),
    ]
    return dict(text=stem, parts=parts)
