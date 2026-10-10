"""Exam tier for tangent lines: the learner must decide what to find before writing the line.

The hard tier gives a point and asks for the tangent there. Here the point is not given. Either the tangent is
required to be parallel to a stated line (so the learner finds where the gradient equals that slope first), or the
question asks for the NORMAL to a cubic at a given x (the perpendicular slope is the negative reciprocal of the
gradient).
"""

from __future__ import annotations

import random

from sympy import Rational, diff, symbols

from ..core import check_equivalent, generator, join_terms
from .exd_x_c3_support import Redraw, draw, expr_distractors, prose, typed

x = symbols("x")


def _parallel_tangent(rng: random.Random) -> dict:
    a = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    b = rng.randint(-8, 8)
    c = rng.randint(-9, 9)
    x0 = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    f = a * x**2 + b * x + c
    m = 2 * a * x0 + b
    if m == 0:
        raise Redraw
    k = rng.randint(-9, 9)
    intercept = f.subs(x, x0) - m * x0
    if k == intercept:
        raise Redraw
    answer = m * x + intercept
    wrong_x = Rational(m - b, a)  # forgets the 2 when solving f'(x) = m
    other_x = Rational(m + b, 2 * a)  # sign error when solving f'(x) = m
    cands = [
        m * x + k,  # gives the line the tangent is parallel to
        m * x + f.subs(x, x0),  # uses f(x0) as the intercept
        m * x + f.subs(x, wrong_x) - m * wrong_x,  # tangent at the wrongly solved point
        m * x + f.subs(x, other_x) - m * other_x,
    ]
    text = (
        f"Find the equation of the tangent to the curve y = {prose(f)} that is parallel to the line y = {_line(m, k)}."
    )
    return _pack(f, answer, cands, text, check_slope=(m, x0))


def _line(m: int, k: int) -> str:
    return join_terms([(m, "x"), (k, "")])


def _normal(rng: random.Random) -> dict:
    a = rng.choice([1, 2, -1, -2])
    b = rng.choice([-3, -2, -1, 1, 2, 3])
    c = rng.randint(-5, 5)
    d = rng.randint(-9, 9)
    p = rng.choice([-2, -1, 1, 2])
    f = a * x**3 + b * x**2 + c * x + d
    s = diff(f, x).subs(x, p)
    if s == 0:
        raise Redraw
    y0 = f.subs(x, p)
    answer = y0 - (x - p) / s
    cands = [
        y0 + s * (x - p),  # gives the tangent
        y0 + (x - p) / s,  # reciprocal slope without the sign change
        y0 - s * (x - p),  # negates the slope but does not take the reciprocal
        y0 - x / s,  # normal slope through (0, f(p)) instead of through the point
    ]
    text = f"Find the equation of the normal to the curve y = {prose(f)} at the point where x = {p}."
    return _pack(f, answer, cands, text, check_slope=None)


def _pack(f, answer, cands, text, check_slope) -> dict:
    answer = answer.expand()
    # Independent route: the line through the tangent point with the derivative's slope (or its negative reciprocal).
    if check_slope is not None:
        m, x0 = check_slope
        check_equivalent(diff(f, x).subs(x, x0), m, "slope disagrees with the derivative at the tangent point")
        check_equivalent(answer.subs(x, x0), f.subs(x, x0), "line does not pass through the curve at the tangent point")
    answer_text = "y = " + typed(answer)
    return dict(
        text=text,
        latex="",
        answer=typed(answer),
        answer_text=answer_text,
        distractors=["y = " + t for t in expr_distractors(answer, [c.expand() for c in cands])],
    )


_KINDS = (_parallel_tangent, _normal)


@generator("exd_x_c3_tangent_normal", level=14, presets={"exam": dict(tier="exam")})
def exd_x_c3_tangent_normal(rng: random.Random, tier: str = "exam") -> dict:
    """A tangent parallel to a given line, or the normal to a cubic at a given x."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(rng.choice(_KINDS), rng)
