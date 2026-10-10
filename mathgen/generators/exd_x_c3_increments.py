"""Exam tier for the increments formula `dy ~ (dy/dx) dx`: a relative change and a composite function.

The hard tier estimates the new value of a quadratic from a small step. Here the learner either converts a
PERCENTAGE change in x into a percentage change in y (the step is `x0 * p / 100` and the result is relative to
`f(x0)`), or estimates the new value of a function of a function, `k (x^2 + c)^n`, where the derivative needs the
chain rule. Both answers are exact fractions.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import diff, symbols

from ..core import check, generator, rational_text
from .exd_x_c3_support import Redraw, draw

x = symbols("x")


def _fraction_answer(value: Fraction) -> dict:
    return {"p": value.numerator, "q": value.denominator}


def _percentage_change(rng: random.Random) -> dict:
    a = rng.randint(1, 4)
    b = rng.randint(-6, 6)
    c = rng.randint(1, 9)
    x0 = rng.randint(2, 6)
    pct = rng.choice([1, 2, 3, 5])
    f = a * x**2 + b * x + c
    y0 = int(f.subs(x, x0))
    slope = int(diff(f, x).subs(x, x0))
    if y0 <= 0 or slope <= 0:
        raise Redraw
    dx = Fraction(x0 * pct, 100)
    answer = slope * dx * 100 / y0  # percentage change in y
    # Independent route: the same estimate through the exact new value of x.
    new_x = Fraction(x0) + dx
    linear_estimate = Fraction(y0) + slope * (new_x - x0)
    check(100 * (linear_estimate - y0) / y0 == answer, "percentage estimate disagrees with the linear approximation")
    cands = [
        Fraction(slope * pct),  # percentage of x times the derivative, ignoring f(x0)
        slope * dx,  # the absolute change, not a percentage
        slope * dx / y0,  # a proportion, not a percentage
        Fraction(pct),  # the percentage change in y equals the percentage change in x
    ]
    f_text = "".join(
        [
            f"{a}x^2" if a != 1 else "x^2",
            f" {'+' if b > 0 else '-'} {abs(b) if abs(b) != 1 else ''}x" if b else "",
            f" + {c}",
        ]
    )
    text = (
        f"A quantity is given by y = f(x) = {f_text}. Starting from x0 = {x0}, the value of x increases by {pct}%. "
        "Using the increments formula δy ≅ dy/dx × δx, estimate the percentage change in y. "
        "Give the estimate as a whole number or a fraction, in percent, without the % sign."
    )
    return _pack(text, answer, cands)


def _composite_new_value(rng: random.Random) -> dict:
    n = rng.choice([3, 4])
    k = rng.choice([1, 2, 3])
    c = rng.randint(1, 5)
    x0 = rng.choice([-3, -2, -1, 1, 2, 3])
    dx = Fraction(rng.choice([-1, 1]), rng.choice([10, 20, 25, 50]))
    f = k * (x**2 + c) ** n
    y0 = Fraction(int(f.subs(x, x0)))
    slope = Fraction(int(diff(f, x).subs(x, x0)))
    answer = y0 + slope * dx
    u0 = x0**2 + c
    cands = [
        y0 + Fraction(n * k * u0 ** (n - 1)) * dx,  # forgets the inner derivative 2x
        y0 + Fraction(n * k * u0**n * 2 * x0) * dx,  # does not lower the power
        slope * dx,  # gives the change, not the new value
        y0 - slope * dx,  # subtracts the change
    ]
    text = (
        f"y = f(x) = {k if k != 1 else ''}(x^2 + {c})^{n}. Estimate the new value of y when x changes from x0 = {x0} "
        f"to x0 + δx, using the increments formula δy ≅ dy/dx × δx, given δx = {rational_text(dx)}."
    )
    return _pack(text, answer, cands)


def _pack(text: str, answer: Fraction, cands) -> dict:
    answer_text = rational_text(answer)
    texts = [rational_text(c) for c in cands]
    distractors = []
    for t in texts:
        if t != answer_text and t not in distractors:
            distractors.append(t)
    if len(distractors) < 3:
        raise Redraw
    return dict(
        text=text,
        latex="",
        answer=_fraction_answer(answer),
        answer_text=answer_text,
        distractors=distractors[:3],
    )


_KINDS = (_percentage_change, _composite_new_value)


@generator("exd_x_c3_increments", level=13, presets={"exam": dict(tier="exam")})
def exd_x_c3_increments(rng: random.Random, tier: str = "exam") -> dict:
    """Estimate a percentage change, or a new value of a composite function, with the increments formula."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(rng.choice(_KINDS), rng)
