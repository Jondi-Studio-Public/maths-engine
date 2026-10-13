"""Generates `exd_x_r2a_product_tangent` questions: a multi-part exam chain on a product needing two rules.

Skill: differentiate-product-rule (exam tier). The function is a product of two powers,
``(ax + b)^p (cx^2 + d)^q`` or ``x^n (ax + b)^p``, so the product rule has to be combined with the chain rule in a
factor, and the derivative is best left factorised. The question then uses it the way a paper does:

(a) find f'(x)                                                    expression, 2 marks
(b) hence the gradient of the tangent at x = x0                   fraction,   1 mark,  Subs(part_a, x, x0)
(c) hence the equation of that tangent                            expression, 2 marks, part_b*(x - x0) + f(x0)
(d) hence the area of the triangle the tangent makes with the axes  fraction, 2 marks, from the learner's line

## Construction

The parameters are drawn forwards. f(x0) and f'(x0) are exact integers, and the tangent has a non-zero gradient and
a non-zero intercept, so the triangle exists. f'(x) is written by hand in factorised form,
``u^(p-1) v^(q-1) [p u' v/u^(p-1)... ]``, and re-derived with ``sympy.diff``; the two must agree.
"""

from __future__ import annotations

import random
from math import gcd

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
    "Let {fx}. A tangent is drawn to the graph of {yf} at the point where {xv}. "
    "This tangent and the two coordinate axes enclose a triangle.",
    "The function f is given by {fx}. A tangent is drawn to the graph of {yf} at the point where {xv}. "
    "It cuts both coordinate axes, making a triangle with them.",
    "Consider the curve {yf}, where {fx}. At the point where {xv} the tangent meets the axes at two points.",
)


def _lin(a: int, b: int) -> str:
    return join_terms([(a, "x"), (b, "")])


def _power(text: str, n: int) -> str:
    return f"({text})" if n == 1 else f"({text})^{n}"


def _draw(rng: random.Random):
    family = rng.choice(["quad", "mono"])
    a = rng.choice([1, 1, 2])
    b = rng.choice([m for m in range(-4, 5) if m != 0 and gcd(a, m) == 1])
    p = rng.choice([2, 3])
    x0 = rng.choice([-1, 1, 2])
    if family == "quad":
        c = rng.choice([1, 1, 2])
        d = rng.choice([m for m in range(-4, 5) if m != 0 and gcd(c, m) == 1])
        q = rng.choice([1, 2])
        return family, a, b, p, c, d, q, x0
    n = rng.choice([2, 3])
    return family, a, b, p, n, 0, 0, x0


def _build(rng: random.Random):
    family, a, b, p, c, d, q, x0 = _draw(rng)
    u = (a * X + b) ** p
    du = p * a * (a * X + b) ** (p - 1)
    if family == "quad":
        v = (c * X**2 + d) ** q
        dv = q * 2 * c * X * (c * X**2 + d) ** (q - 1)
        u_text = _power(_lin(a, b), p)
        v_text = _power(join_terms([(c, "x^2"), (d, "")]), q)
        f_text = u_text + v_text
        # f' = (ax+b)^(p-1) (cx^2+d)^(q-1) [p a (cx^2+d) + 2 q c x (ax+b)]
        bracket = expand(p * a * (c * X**2 + d) + 2 * q * c * X * (a * X + b))
        fprime = Mul(
            (a * X + b) ** (p - 1),
            (c * X**2 + d) ** (q - 1),
            bracket,
            evaluate=True,
        )
    else:
        n = c
        v = X**n
        dv = n * X ** (n - 1)
        f_text = f"x^{n}" + _power(_lin(a, b), p)
        # f' = x^(n-1) (ax+b)^(p-1) [n (ax+b) + p a x]
        bracket = expand(n * (a * X + b) + p * a * X)
        fprime = Mul(X ** (n - 1), (a * X + b) ** (p - 1), bracket)
    f = u * v
    check(derivative_agrees(f, fprime), "product rule disagrees with the numerical derivative")

    f0, m = f.subs(X, x0), fprime.subs(X, x0)
    if m == 0 or f0 == 0 or abs(f0) > 150 or abs(m) > 200:
        raise Redraw
    wrong_a = [
        ("product-rule-multiplied-derivatives", du * dv),
        ("product-rule-term-dropped", du * v),
        ("product-rule-term-dropped", u * dv),
        ("forgot-inner-derivative", p * (a * X + b) ** (p - 1) * v + u * dv),
    ]
    rng.shuffle(wrong_a)
    part_a = expr_part(
        "Find f'(x).",
        fprime,
        2,
        [
            f"Product rule: f'(x) = u'v + uv' with u = {_power(_lin(a, b), p)} and v the other factor.",
            "Each derivative needs the chain rule or the power rule; take out the common factors.",
        ],
        wrong_a,
    )
    parts = [part_a, *tangent_tail(f0, m, x0, variant="tangent_area", m_at_neg_x0=fprime.subs(X, -x0))]
    stem, stem_latex = prose(rng, _PATTERNS, fx=(f"f(x) = {f_text}", f"f(x) = {latex(f)}"), yf=YF, xv=x_equals(x0))
    return dict(text=stem, latex=stem_latex, parts=parts)


@generator("exd_x_r2a_product_tangent", level=16, presets={"exam": {}})
def exd_x_r2a_product_tangent(rng: random.Random):
    """A product of powers: f'(x), the gradient at x0, the tangent, the triangle it makes with the axes."""
    return build_with_redraw(rng, _build)
