"""Generates `exd_x_r2a_stationary_double_root` questions: a multi-part exam chain on a cubic with a repeated root.

Skill: stationary-points (exam tier). The function is given factorised with a repeated factor,
``f(x) = k(x - p)^2 (x - q)``, so the graph touches the x-axis at x = p. The question finds and classifies the
stationary points without expanding:

(a) find f'(x), in any form                                       expression, 2 marks
(b) hence the x-coordinate of the stationary point that is not on the x-axis   fraction, 2 marks
(c) the value of f''(x) at that point (its sign gives the nature)  fraction, 1 mark
(d) hence the stationary value of f at that point                 fraction,   1 mark

## Construction

p and q are small integers with q - p of 2 or more in size, k is a small non-zero integer. Then
``f'(x) = k(x - p)(3x - p - 2q)`` and the second stationary point is x0 = (p + 2q)/3, which is not p. f'(x) is
written by hand and re-derived with ``sympy.diff``.

## Follow-through

(b) is the other root of the learner's quadratic f'(x): the sum of its roots -B/A minus p. (c) is the learner's
f'' = 2A x + B evaluated at the learner's (b). (d) is f at the learner's (b).
"""

from __future__ import annotations

import random

from sympy import Mul, Rational, Symbol, diff, latex

from ..core import check, generator, join_terms
from .exd_x_r2a_common import (
    YF,
    X,
    Redraw,
    build_with_redraw,
    expr_part,
    fraction_part,
    paren,
    part_prompt,
    pretty,
    prose,
    quadratic_coefficients_ft,
    same_function,
)

_PATTERNS = (
    "The function f is defined by {fx}. The graph of {yf} touches the x-axis at one point and has a second "
    "stationary point.",
    "Let {fx}. The graph of {yf} has two stationary points.",
    "Consider {fx}. Its graph has a stationary point on the x-axis and another one away from it.",
)


def _factor(p: int, power: int) -> str:
    base = "x" if p == 0 else f"({join_terms([(1, 'x'), (-p, '')])})"
    return base if power == 1 else f"{base}^{power}"


def _build(rng: random.Random):
    p = rng.randint(-3, 3)
    gap = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
    q = p + gap
    k = rng.choice([1, 1, 2, 3, -1, -2])
    f = k * (X - p) ** 2 * (X - q)
    fprime = Mul(k, X - p, 3 * X - p - 2 * q, evaluate=False) if k != 1 else (X - p) * (3 * X - p - 2 * q)
    check(same_function(diff(f, X), fprime), "f' disagrees with sympy diff")
    f2 = diff(f, X, 2)
    x0 = Rational(p + 2 * q, 3)
    if x0 == p or x0 == q:
        raise Redraw
    f2_x0, f_x0 = f2.subs(X, x0), f.subs(X, x0)
    if f_x0 == 0 or f2_x0 == 0:
        raise Redraw
    coefficient = "" if k == 1 else ("-" if k == -1 else str(k))
    f_text = f"{coefficient}{_factor(p, 2)}{_factor(q, 1)}"

    wrong_a = [
        ("product-rule-multiplied-derivatives", 2 * k * (X - p)),
        ("product-rule-term-dropped", 2 * k * (X - p) * (X - q)),
        ("product-rule-term-dropped", k * (X - p) ** 2),
        ("exponent-not-lowered", 2 * k * (X - p) ** 2 * (X - q)),
    ]
    rng.shuffle(wrong_a)
    part_a = expr_part(
        "Find f'(x).",
        fprime,
        2,
        [
            f"Product rule with u = {_factor(p, 2)} and v = {_factor(q, 1)}.",
            f"f'(x) = {pretty(2 * k * (X - p) * (X - q))} {'+' if k > 0 else '-'} {pretty(abs(k) * (X - p) ** 2)} = {pretty(fprime)}.",
        ],
        wrong_a,
    )

    a_ft, b_ft, c_ft = quadratic_coefficients_ft("a")
    del c_ft
    part_b = fraction_part(
        part_prompt(
            "Find the x-coordinate of the stationary point of the graph of {yf} that is not on the x-axis.", yf=YF
        ),
        x0,
        2,
        [
            f"f'(x) = 0 gives x = {p} or x = ({join_terms([(p, ''), (2 * q, '')])})/3 = {x0}.",
            f"The graph touches the axis at x = {p}, so x = {x0}.",
        ],
        [
            ("solved-f-zero-not-f-prime", q),
            ("midpoint-instead-of-turning-point", Rational(p + q, 2)),
            ("sign-error-moving-term", -x0),
            ("wrong-root-chosen", p),
        ],
        follow_through=f"-{b_ft}/{a_ft} - {paren(p)}",
    )
    f_in_b = str(f.subs(X, Symbol("part_b")))
    part_c = fraction_part(
        part_prompt("Find the value of f''(x) at this stationary point. The sign of this value shows its nature."),
        f2_x0,
        1,
        [
            f"f''(x) = {pretty(f2)}.",
            f"f''({x0}) = {f2_x0}, which is {'negative, so a local maximum' if f2_x0 < 0 else 'positive, so a local minimum'}.",
        ],
        [
            ("sign-reversed-result", -f2_x0),
            ("second-derivative-at-wrong-point", f2.subs(X, p)),
            ("evaluated-wrong-function-order", f_x0),
            ("evaluated-wrong-function-order", fprime.subs(X, x0 + 1)),
        ],
        follow_through=f"2*{a_ft}*part_b + {b_ft}",
    )
    part_d = fraction_part(
        part_prompt("Hence find the stationary value of f at this point."),
        f_x0,
        1,
        [f"f({x0}) = {f_x0}."],
        [
            ("x-given-for-y", x0),
            ("substitution-sign-error", f.subs(X, -x0)),
            ("sign-reversed-result", -f_x0),
            ("evaluated-wrong-function-order", f2_x0),
        ],
        follow_through=f_in_b,
    )
    stem, stem_latex = prose(rng, _PATTERNS, fx=(f"f(x) = {f_text}", f"f(x) = {latex(f, order='none')}"), yf=YF)
    return dict(text=stem, latex=stem_latex, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2a_stationary_double_root", level=16, presets={"exam": {}})
def exd_x_r2a_stationary_double_root(rng: random.Random):
    """A cubic with a repeated root: f'(x), the second stationary point, f'' there, the stationary value."""
    return build_with_redraw(rng, _build)
