"""Generates `exd_x_r2a_rules_tangent` questions: a multi-part exam chain mixing the product, quotient and chain rules.

Skill: senior-apply-the-product-quotient-and-chain-rule-to (exam tier). Each function needs two rules at once:

* ``(x^2 + ax + c) e^(qx)``      product rule with a chain rule in the exponential;
* ``e^(px) / (x^2 + ax + c)``    quotient rule with a chain rule in the numerator;
* ``(sin(bx) + k) / (x + c)``    quotient rule with a chain rule in the sine;
* ``(x + a) e^(qx) cos(bx)``     a three-factor product with two chain rules.

The question then asks for a tangent where every value is exact:

(a) find f'(x)                                                    expression, 2 marks
(b) hence the gradient of the graph at x = 0                      fraction,   1 mark,  Subs(part_a, x, 0)
(c) hence the equation of the tangent at x = 0                    expression, 2 marks, part_b*x + f(0)
(d) hence where that tangent meets the x-axis                     fraction,   1 mark,  from the learner's line

## Construction

At x = 0 the exponential is 1, the cosine is 1 and the sine is 0, so f(0) and f'(0) are rational whatever the
parameters. f'(x) is written by hand (numerators expanded) and re-derived with ``sympy.diff`` at probe points; f(0)
and f'(0) must be non-zero so the tangent meets the x-axis at one point.
"""

from __future__ import annotations

import random

from sympy import Mul, cos, diff, exp, expand, latex, sin

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
    "Let {fx}. The tangent to the graph of {yf} at the point where {xv} meets the x-axis at the point P.",
    "The function f is given by {fx}. A tangent is drawn to the graph of {yf} at the point where {xv}; it meets the "
    "x-axis at P.",
    "Consider the curve {yf}, where {fx}. The tangent at {xv} cuts the x-axis at the point P.",
)


def _e_text(q: int) -> str:
    return "e^x" if q == 1 else "e^(-x)" if q == -1 else f"e^({q}x)"


def _quad(a: int, c: int):
    return X**2 + a * X + c, join_terms([(1, "x^2"), (a, "x"), (c, "")])


def _product_exp(rng):
    a = rng.choice([m for m in range(-4, 5) if m != 0])
    c = rng.choice([m for m in range(-4, 5) if m != 0])
    q = rng.choice([-3, -2, -1, 1, 2, 3])
    quad, quad_text = _quad(a, c)
    f = quad * exp(q * X)
    fprime = Mul(expand(2 * X + a + q * quad), exp(q * X))
    wrong = [
        ("product-rule-multiplied-derivatives", (2 * X + a) * q * exp(q * X)),
        ("forgot-inner-derivative", expand(2 * X + a + quad) * exp(q * X)),
        ("product-rule-term-dropped", (2 * X + a) * exp(q * X)),
        ("product-rule-term-dropped", q * quad * exp(q * X)),
    ]
    return (
        f,
        fprime,
        f"({quad_text}){_e_text(q)}",
        wrong,
        "Product rule with u = the quadratic and v = the exponential.",
    )


def _quotient_exp(rng):
    a = rng.choice([m for m in range(-3, 4) if m != 0])
    c = rng.choice([1, 2, 3, 4])
    p = rng.choice([-3, -2, -1, 1, 2, 3])
    quad, quad_text = _quad(a, c)
    if a * a - 4 * c >= 0:
        raise Redraw  # keeps the denominator positive for every x
    f = exp(p * X) / quad
    fprime = Mul(exp(p * X), expand(p * quad - (2 * X + a)), quad**-2)
    wrong = [
        ("quotient-rule-numerator-reversed", Mul(exp(p * X), expand((2 * X + a) - p * quad), quad**-2)),
        ("forgot-inner-derivative", Mul(exp(p * X), expand(quad - (2 * X + a)), quad**-2)),
        ("quotient-rule-forgot-square", Mul(exp(p * X), expand(p * quad - (2 * X + a)), quad**-1)),
        ("quotient-rule-differentiated-separately", p * exp(p * X) / (2 * X + a)),
    ]
    return (
        f,
        fprime,
        f"{_e_text(p)}/({quad_text})",
        wrong,
        "Quotient rule with u = the exponential and v = the quadratic.",
    )


def _quotient_sin(rng):
    b = rng.choice([2, 3, 4, 5])
    k = rng.choice([m for m in range(-4, 5) if m != 0])
    c = rng.choice([1, 2, 3, 4])
    f = (sin(b * X) + k) / (X + c)
    fprime = Mul(b * (X + c) * cos(b * X) - sin(b * X) - k, (X + c) ** -2)
    wrong = [
        ("quotient-rule-numerator-reversed", Mul(sin(b * X) + k - b * (X + c) * cos(b * X), (X + c) ** -2)),
        ("forgot-inner-derivative", Mul((X + c) * cos(b * X) - sin(b * X) - k, (X + c) ** -2)),
        ("sign-error-trig-derivative", Mul(-b * (X + c) * cos(b * X) - sin(b * X) - k, (X + c) ** -2)),
        ("quotient-rule-forgot-square", Mul(b * (X + c) * cos(b * X) - sin(b * X) - k, (X + c) ** -1)),
    ]
    return (
        f,
        fprime,
        f"(sin({b}x) + {k})/(x + {c})".replace("+ -", "- "),
        wrong,
        "Quotient rule with u = sin(bx) + k and v = x + c.",
    )


def _triple(rng):
    a = rng.choice([m for m in range(-4, 5) if m != 0])
    q = rng.choice([-3, -2, -1, 1, 2, 3])
    b = rng.choice([2, 3, 4])
    f = (X + a) * exp(q * X) * cos(b * X)
    fprime = Mul(exp(q * X), cos(b * X) * (1 + q * (X + a)) - b * (X + a) * sin(b * X))
    wrong = [
        ("forgot-inner-derivative", Mul(exp(q * X), cos(b * X) * (1 + (X + a)) - b * (X + a) * sin(b * X))),
        ("forgot-inner-derivative", Mul(exp(q * X), cos(b * X) * (1 + q * (X + a)) - (X + a) * sin(b * X))),
        ("sign-error-trig-derivative", Mul(exp(q * X), cos(b * X) * (1 + q * (X + a)) + b * (X + a) * sin(b * X))),
        ("product-rule-term-dropped", Mul(exp(q * X), cos(b * X) * (1 + q * (X + a)))),
        ("product-rule-term-dropped", Mul(exp(q * X), -b * (X + a) * sin(b * X))),
    ]
    return (
        f,
        fprime,
        f"(x + {a}){_e_text(q)}{' ' if q == 1 else ''}cos({b}x)".replace("+ -", "- "),
        wrong,
        "Treat f as a product of three factors and apply the product rule, with the chain rule in the last two.",
    )


_FAMILIES = (_product_exp, _quotient_exp, _quotient_sin, _triple)


def _build(rng: random.Random):
    f, fprime, f_text, wrong_a, hint = rng.choice(_FAMILIES)(rng)
    check(derivative_agrees(f, fprime), "the rules disagree with the numerical derivative")
    f0, m = f.subs(X, 0), fprime.subs(X, 0)
    if f0 == 0 or m == 0 or not (f0.is_Rational and m.is_Rational):
        raise Redraw
    rng.shuffle(wrong_a)
    part_a = expr_part(
        "Find f'(x).",
        fprime,
        2,
        [hint, "Differentiate each factor, combine, and collect."],
        wrong_a,
    )
    parts = [part_a, *tangent_tail(f0, m, 0, variant="tangent_x", m_at_neg_x0=None)]
    stem, stem_latex = prose(rng, _PATTERNS, fx=(f"f(x) = {f_text}", f"f(x) = {latex(f)}"), yf=YF, xv=x_equals(0))
    return dict(text=stem, latex=stem_latex, parts=parts)


@generator("exd_x_r2a_rules_tangent", level=16, presets={"exam": {}})
def exd_x_r2a_rules_tangent(rng: random.Random):
    """Two rules at once: f'(x), the gradient at x = 0, the tangent, where it meets the x-axis."""
    return build_with_redraw(rng, _build)
