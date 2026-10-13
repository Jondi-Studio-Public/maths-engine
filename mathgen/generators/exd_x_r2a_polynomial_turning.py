"""Generates `exd_x_r2a_polynomial_turning` questions: a multi-part exam chain on a cubic given as a product.

Skill: differentiate-polynomial (exam tier). The cubic arrives as ``f(x) = (x - r)(2j x^2 + s x + t)``, a linear factor
times a quadratic, so the learner chooses between expanding first and using the product rule. The chain then uses the
derivative to describe the graph:

(a) find f'(x)                                                    expression, 2 marks
(b) hence the x-coordinate of the local maximum                   fraction,   2 marks, a root of the learner's f'
(c) hence the local maximum value of f                            fraction,   1 mark,  f at the learner's x
(d) hence the x-coordinate of the point of inflection             fraction,   1 mark,  the root of the learner's f''

## Construction: choose the turning points first

The turning points x = p and x = q are integers (q - p at least 2) and ``f'(x) = 6j(x - p)(x - q)`` with j a non-zero
integer, which gives ``f(x) = 2j x^3 - 3j(p + q) x^2 + 6j p q x + k``. The constant k is set so that f has the integer
root r, and the factor ``(x - r)`` is divided out to give the quadratic shown. The local maximum is at the smaller
turning point when j > 0 and at the larger when j < 0; the point of inflection is at (p + q)/2.

## Follow-through

Both (b) and (d) need the learner's own f'. Its coefficients A, B, C are read from its values at x = 1, -1 and 0
(`quadratic_coefficients_ft`), then the local maximum is (-B - sqrt(B^2 - 4AC))/(2A) (this is the maximum whichever
sign A has) and the inflection is -B/(2A).
"""

from __future__ import annotations

import random

from sympy import Rational, Symbol, diff, expand, latex

from ..core import check, generator, join_terms
from .exd_x_r2a_common import (
    X,
    Redraw,
    build_with_redraw,
    expr_part,
    fraction_part,
    part_prompt,
    pretty,
    prose,
    quadratic_coefficients_ft,
    shield,
    same_function,
    typed,
)

_PATTERNS = (
    "The cubic function f is given by {fx}. The graph of {yf} has a local maximum, a local minimum and a point of "
    "inflection.",
    "Let {fx}. The graph of {yf} has two turning points.",
    "Consider the function {fx}. Its graph has a local maximum and a local minimum.",
)
_Y = ("y = f(x)", "y = f(x)")


def _draw(rng: random.Random):
    p = rng.randint(-3, 2)
    q = p + rng.randint(2, 5)
    j = rng.choice([1, 1, 2, -1, -2])
    r = rng.choice([m for m in range(-3, 4) if m not in (0, p, q)])
    return p, q, j, r


def _build(rng: random.Random):
    p, q, j, r = _draw(rng)
    alpha = 2 * j
    beta = -3 * j * (p + q) + r * alpha
    gamma = 6 * j * p * q + r * beta
    k = -r * gamma
    f = expand(2 * j * X**3 - 3 * j * (p + q) * X**2 + 6 * j * p * q * X + k)
    check(same_function(f, (X - r) * (alpha * X**2 + beta * X + gamma)), "the factorised form is not f")
    fprime = expand(6 * j * (X - p) * (X - q))
    check(same_function(diff(f, X), fprime), "f' disagrees with sympy diff")
    x_max, x_min = (p, q) if j > 0 else (q, p)
    inflection = Rational(p + q, 2)
    v_max = f.subs(X, x_max)
    if abs(v_max) > 120 or abs(beta) > 30 or abs(gamma) > 60:
        raise Redraw
    v_min = f.subs(X, x_min)

    quad_text = join_terms([(alpha, "x^2"), (beta, "x"), (gamma, "")])
    lin_text = join_terms([(1, "x"), (-r, "")])
    f_text = f"({lin_text})({quad_text})"
    f_latex = rf"\left({latex(X - r)}\right)\left({latex(alpha * X**2 + beta * X + gamma)}\right)"

    wrong_a = [
        ("product-rule-multiplied-derivatives", expand(2 * alpha * X + beta)),
        ("product-rule-term-dropped", expand(alpha * X**2 + beta * X + gamma)),
        ("product-rule-term-dropped", expand((X - r) * (2 * alpha * X + beta))),
        ("derivative-of-constant-not-zero", fprime + k),
        ("exponent-not-lowered", 3 * alpha * X**3 + 2 * (-3 * j * (p + q)) * X**2 + 6 * j * p * q * X),
    ]
    rng.shuffle(wrong_a)
    part_a = expr_part(
        "Find f'(x).",
        fprime,
        2,
        [
            f"Expand: f(x) = {pretty(f)}.",
            f"Differentiate term by term: f'(x) = {pretty(fprime)}.",
        ],
        wrong_a,
    )

    a_ft, b_ft, c_ft = quadratic_coefficients_ft("a")
    part_b = fraction_part(
        part_prompt("Hence find the x-coordinate of the local maximum point of the graph of {yf}.", yf=_Y),
        x_max,
        2,
        [
            f"f'(x) = {pretty(fprime)} = 0 gives x = {p} or x = {q}.",
            f"f''(x) = {pretty(diff(f, X, 2))}, and f''({x_max}) = {diff(f, X, 2).subs(X, x_max)} < 0, "
            f"so x = {x_max} is the local maximum.",
        ],
        [
            ("second-derivative-sign-reversed", x_min),
            ("midpoint-instead-of-turning-point", inflection),
            ("sign-error-moving-term", -x_max),
            ("solved-f-zero-not-f-prime", r),
        ],
        follow_through=f"(-{b_ft} - sqrt({shield(b_ft + '**2 - 4*' + a_ft + '*' + c_ft)}))/(2*{a_ft})",
    )
    f_in_b = str(f.subs(X, Symbol("part_b")))
    part_c = fraction_part(
        part_prompt("Hence find the local maximum value of f."),
        v_max,
        1,
        [f"f({x_max}) = {v_max}."],
        [
            ("second-derivative-sign-reversed", v_min),
            ("x-given-for-y", x_max),
            ("substitution-sign-error", f.subs(X, -x_max)),
            ("evaluated-wrong-function-order", diff(f, X, 2).subs(X, x_max)),
        ],
        follow_through=f_in_b,
    )
    part_d = fraction_part(
        part_prompt("Find the x-coordinate of the point of inflection of the graph of {yf}.", yf=_Y),
        inflection,
        1,
        [f"f''(x) = {pretty(diff(f, X, 2))} = 0 gives x = {inflection}."],
        [
            ("sign-error-moving-term", -inflection),
            ("divided-by-wrong-coefficient", p + q),
            ("forgot-to-divide-by-coefficient", 6 * j * (p + q)),
            ("midpoint-instead-of-turning-point", x_max),
        ],
        follow_through=f"-{b_ft}/(2*{a_ft})",
    )
    stem, stem_latex = prose(rng, _PATTERNS, fx=(f"f(x) = {f_text}", f"f(x) = {f_latex}"), yf=_Y)
    return dict(text=stem, latex=stem_latex, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2a_polynomial_turning", level=16, presets={"exam": {}})
def exd_x_r2a_polynomial_turning(rng: random.Random):
    """A cubic as a product: f'(x), the local maximum, its value, and the point of inflection."""
    return build_with_redraw(rng, _build)
