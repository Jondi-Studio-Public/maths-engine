"""Generates `exd_x_r2a_quotient_normal` questions: a multi-part exam chain on a quotient and its normal.

Skill: differentiate-quotient-rule (exam tier). The function is a quotient with a non-constant derivative in both
numerator and denominator, ``(ax + b)/(cx^2 + d)`` or ``(ax^2 + b)/(cx + d)``, and the question runs on to a normal:

(a) find f'(x)                                                    expression, 2 marks
(b) hence the gradient of the graph at x = x0                     fraction,   1 mark,  Subs(part_a, x, x0)
(c) hence the equation of the normal at x = x0                    expression, 2 marks, -1/part_b*(x - x0) + f(x0)
(d) hence where the normal meets the y-axis                       fraction,   1 mark,  from the learner's line

## Construction

The parameters are drawn forwards and rejected unless f(x0) and f'(x0) are non-zero rationals with modest
denominators (so the normal exists and is not horizontal or vertical). f'(x) is written by hand as
``(N'D - ND')/D^2`` with the numerator expanded, and re-derived with ``sympy.diff``; the two must agree.
"""

from __future__ import annotations

import random

from sympy import Mul, diff, expand, latex

from ..core import check, generator, join_terms
from .exd_x_r2a_common import (
    YF,
    X,
    Redraw,
    build_with_redraw,
    expr_part,
    prose,
    derivative_agrees,
    same_function,
    tangent_tail,
    x_equals,
)

_PATTERNS = (
    "Let {fx}. The normal to the graph of {yf} is drawn at the point where {xv}.",
    "A function is defined by {fx}. Consider the normal to the curve {yf} at the point where {xv}.",
    "The graph of {yf}, where {fx}, has a normal at the point where {xv}.",
)


def _draw(rng: random.Random):
    family = rng.choice(["lin_over_quad", "quad_over_lin"])
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    b = rng.choice([m for m in range(-5, 6) if m != 0])
    c = rng.choice([1, 1, 2])
    d = rng.choice([m for m in range(-4, 5) if m != 0])
    x0 = rng.choice([-2, -1, 1, 2, 3])
    return family, a, b, c, d, x0


def _build(rng: random.Random):
    family, a, b, c, d, x0 = _draw(rng)
    if family == "lin_over_quad":
        if d < 0:
            d = -d  # a positive constant keeps the denominator away from zero
        num, den = a * X + b, c * X**2 + d
        num_text = join_terms([(a, "x"), (b, "")])
        den_text = join_terms([(c, "x^2"), (d, "")])
    else:
        num, den = a * X**2 + b, c * X + d
        num_text = join_terms([(a, "x^2"), (b, "")])
        den_text = join_terms([(c, "x"), (d, "")])
        if den.subs(X, x0) == 0:
            raise Redraw
    dnum, dden = diff(num, X), diff(den, X)
    top = expand(dnum * den - num * dden)
    fprime = Mul(top, den**-2)
    f = num / den
    check(derivative_agrees(f, fprime), "quotient rule disagrees with the numerical derivative")
    if top == 0:
        raise Redraw
    f0, m = f.subs(X, x0), fprime.subs(X, x0)
    if f0 == 0 or m == 0:
        raise Redraw
    wrong_a = [
        ("quotient-rule-numerator-reversed", Mul(expand(num * dden - dnum * den), den**-2)),
        ("quotient-rule-forgot-square", Mul(top, den**-1)),
        ("quotient-rule-differentiated-separately", dnum / dden),
        ("quotient-rule-plus-instead-of-minus", Mul(expand(dnum * den + num * dden), den**-2)),
    ]
    rng.shuffle(wrong_a)
    part_a = expr_part(
        "Find f'(x).",
        fprime,
        2,
        [
            f"Quotient rule with u = {num_text} and v = {den_text}: f'(x) = (u'v - uv')/v^2.",
            f"Numerator: ({dnum})({den_text}) - ({num_text})({dden}), expanded and collected.",
        ],
        wrong_a,
    )
    parts = [part_a, *tangent_tail(f0, m, x0, variant="normal_y", m_at_neg_x0=fprime.subs(X, -x0))]
    stem, stem_latex = prose(
        rng, _PATTERNS, fx=(f"f(x) = ({num_text})/({den_text})", f"f(x) = {latex(f)}"), yf=YF, xv=x_equals(x0)
    )
    return dict(text=stem, latex=stem_latex, parts=parts)


@generator("exd_x_r2a_quotient_normal", level=16, presets={"exam": {}})
def exd_x_r2a_quotient_normal(rng: random.Random):
    """A quotient: f'(x), the gradient at x0, the normal there, and where the normal meets the y-axis."""
    return build_with_redraw(rng, _build)
