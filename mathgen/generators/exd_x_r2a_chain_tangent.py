"""Generates `exd_x_r2a_chain_tangent` questions: a multi-part exam chain on a function needing the chain rule.

Skill: differentiate-chain-rule (exam tier). The function is a power of a quadratic with a negative or fractional
exponent, ``f(x) = k (ax^2 + bx + c)^e`` with e in {-3, -2, -1, -1/2, 1/2}, written as ``k/(...)^n``, ``k/√(...)``
or ``k√(...)``. The question runs the way a Year 12 paper does:

(a) find f'(x)                                                    expression, 2 marks
(b) hence the gradient of the tangent at x = x0                   fraction,   1 mark,  Subs(part_a, x, x0)
(c) hence the equation of that tangent                            expression, 2 marks, part_b*(x - x0) + f(x0)
(d) hence where the tangent meets the x-axis                      fraction,   1 mark,  from the learner's line

## Construction

The parameters are drawn first and x0 is chosen so that every number is exact: for a square-root form the quadratic
is built to take a perfect-square value s^2 at x0 (c = s^2 - a x0^2 - b x0), so f(x0) and f'(x0) are rational. The
gradient must be non-zero (2a x0 + b != 0) so the tangent meets the x-axis at one point. f'(x) is written by hand
as k e (2ax + b) Q^(e-1) and re-derived with ``sympy.diff``; the two must agree.

## Distractors

(a) the usual chain-rule slips: the inner derivative left out, the exponent raised instead of lowered, only the
leading term of the inner derivative, the power multiplier dropped. (b), (c), (d) come from the shared tangent
tail in `exd_x_r2a_common`.
"""

from __future__ import annotations

import random

from sympy import Rational, diff, factor_terms, latex

from ..core import check, draw_until, generator, join_terms
from .exd_x_r2a_common import (
    YF,
    X,
    build_with_redraw,
    expr_part,
    prose,
    derivative_agrees,
    same_function,
    tangent_tail,
    typed,
    x_equals,
)

_EXPONENTS = (Rational(-3), Rational(-2), Rational(-1), Rational(-1, 2), Rational(1, 2))
_PATTERNS = (
    "Let {fx}. The graph of {yf} has a tangent at the point where {xv}.",
    "A curve has equation {yf}, where {fx}. A tangent is drawn to the curve at the point where {xv}.",
    "The function f is defined by {fx}. Consider the tangent to the graph of {yf} at {xv}.",
)


def _quad_text(a: int, b: int, c: int) -> str:
    return join_terms([(a, "x^2"), (b, "x"), (c, "")])


def _function_text(k: int, e: Rational, quad: str) -> str:
    if e.q == 1:  # a negative whole power
        n = int(-e)
        den = f"({quad})" if n == 1 else f"({quad})^{n}"
        return f"{k}/{den}"
    if e > 0:
        return f"{k}√({quad})"
    return f"{k}/√({quad})"


def _draw(rng: random.Random):
    a = rng.randint(1, 3)
    b = rng.choice([m for m in range(-5, 6) if m != 0])
    x0 = rng.choice([-3, -2, -1, 1, 2, 3])
    e = rng.choice(_EXPONENTS)
    k = rng.choice([2, 3, 4, 5, 6, -2, -3, -4, 8])
    if 2 * a * x0 + b == 0:
        return None
    if e.q == 1:
        c = rng.choice([m for m in range(-6, 7) if m != 0])
        q0 = a * x0 * x0 + b * x0 + c
        if q0 == 0 or abs(q0) ** (1 - e) > 600:  # keeps f'(x0) a modest fraction
            return None
    else:
        s = rng.randint(1, 6)
        c = s * s - a * x0 * x0 - b * x0
        if c == 0 or abs(c) > 14:
            return None
    return a, b, c, x0, e, k


@generator("exd_x_r2a_chain_tangent", level=15, presets={"exam": {}})
def exd_x_r2a_chain_tangent(rng: random.Random):
    """A power of a quadratic: f'(x), the gradient at x0, the tangent, its x-intercept."""
    return build_with_redraw(rng, _build)


def _build(rng: random.Random):
    a, b, c, x0, e, k = draw_until(rng, _draw, lambda d: d is not None, what="a chain-rule function")
    quad = a * X**2 + b * X + c
    inner = 2 * a * X + b
    f = k * quad**e
    fprime = k * e * inner * quad ** (e - 1)
    check(derivative_agrees(f, fprime), "chain rule disagrees with the numerical derivative")
    fprime = factor_terms(fprime)

    q0 = quad.subs(X, x0)
    check(q0 > 0 or e.q == 1, "square root of a non-positive number")
    f0 = f.subs(X, x0)
    m = fprime.subs(X, x0)
    check(f0.is_Rational and m.is_Rational and m != 0, "tangent numbers are not exact rationals")
    m_neg = fprime.subs(X, -x0)

    wrong_a = [
        ("forgot-inner-derivative", k * e * quad ** (e - 1)),
        ("exponent-moved-wrong-way", k * e * inner * quad ** (e + 1)),
        ("inner-derivative-leading-term-only", k * e * (2 * a * X) * quad ** (e - 1)),
        ("forgot-power-multiplier", k * inner * quad ** (e - 1)),
        ("exponent-not-lowered", k * e * inner * quad**e),
    ]
    rng.shuffle(wrong_a)
    part_a = expr_part(
        "Find f'(x).",
        fprime,
        2,
        [
            f"Write f(x) = {k}({_quad_text(a, b, c)})^({e}); the outer power is {e}.",
            f"f'(x) = {k} × {e} × ({_quad_text(0, 2 * a, b)}) × ({_quad_text(a, b, c)})^({e - 1}).",
        ],
        wrong_a,
    )
    parts = [part_a, *tangent_tail(f0, m, x0, variant="tangent_x", m_at_neg_x0=m_neg)]

    stem, stem_latex = prose(
        rng,
        _PATTERNS,
        fx=(f"f(x) = {_function_text(k, e, _quad_text(a, b, c))}", f"f(x) = {latex(f)}"),
        yf=YF,
        xv=x_equals(x0),
    )
    return dict(text=stem, latex=stem_latex, parts=parts)
