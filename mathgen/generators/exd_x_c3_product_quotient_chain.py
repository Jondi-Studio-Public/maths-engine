"""Exam tier for differentiating with the product, quotient and chain rules together.

Each question needs two rules at once, and the learner must choose which: a
product whose second factor has a quadratic inner function, a quotient whose
numerator has a quadratic inner function, or a three-factor product with two
chains. The key is the derivative worked out by hand-written rule application
and then checked against sympy's `diff` as an independent route.
"""

from __future__ import annotations

import random

from sympy import Integer, cos, diff, exp, sin, symbols

from ..core import check_equivalent, generator
from .exd_x_c3_support import Redraw, draw, expr_distractors, prose, typed

x = symbols("x")


def _product_chain(rng: random.Random) -> dict:
    """x^m * exp(p x^2 + q x): product rule, with a chain rule inside the second factor."""
    m = rng.choice([2, 3])
    p = rng.choice([-3, -2, -1, 1, 2, 3])
    q = rng.choice([-3, -2, -1, 1, 2, 3])
    inner = p * x**2 + q * x
    f = x**m * exp(inner)
    dinner = 2 * p * x + q
    answer = (m * x ** (m - 1) + x**m * dinner) * exp(inner)
    cands = [
        m * x ** (m - 1) * dinner * exp(inner),  # multiplies the two derivatives
        (m * x ** (m - 1) + x**m) * exp(inner),  # forgets the inner derivative
        m * x ** (m - 1) * exp(inner),  # drops the second product-rule term
        x**m * dinner * exp(inner),  # drops the first product-rule term
    ]
    return _pack(f, answer, cands)


def _quotient_chain(rng: random.Random) -> dict:
    """sin(a x^2) / (x + c): quotient rule, with a chain rule in the numerator."""
    a = rng.choice([1, 2, 3])
    c = rng.choice([1, 2, 3, 4])
    u = sin(a * x**2)
    v = x + c
    f = u / v
    du = 2 * a * x * cos(a * x**2)
    answer = (du * v - u) / v**2
    cands = [
        (u - du * v) / v**2,  # numerator order swapped
        (du * v - u) / v,  # forgets to square the denominator
        (cos(a * x**2) * v - u) / v**2,  # forgets the inner derivative
        du,  # differentiates the numerator and denominator separately
    ]
    return _pack(f, answer, cands)


def _triple_product(rng: random.Random) -> dict:
    """x * exp(a x) * cos(b x): a three-factor product with two chain rules."""
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    b = rng.choice([2, 3, 4])
    f = x * exp(a * x) * cos(b * x)
    answer = exp(a * x) * cos(b * x) + a * x * exp(a * x) * cos(b * x) - b * x * exp(a * x) * sin(b * x)
    cands = [
        exp(a * x) * cos(b * x) + x * exp(a * x) * cos(b * x) - x * exp(a * x) * sin(b * x),  # no chain factors
        a * x * exp(a * x) * cos(b * x) - b * x * exp(a * x) * sin(b * x),  # drops the term from differentiating x
        exp(a * x) * cos(b * x) + a * x * exp(a * x) * cos(b * x) + b * x * exp(a * x) * sin(b * x),  # sign of (cos)'
        exp(a * x) * cos(b * x) * a * cos(b * x) * b,  # multiplies the derivatives
    ]
    return _pack(f, answer, cands)


def _pack(f, answer, cands) -> dict:
    check_equivalent(answer, diff(f, x), "hand-applied rules disagree with sympy diff")
    answer = answer if answer != 0 else Integer(0)
    if answer.is_number:
        raise Redraw
    answer_text = typed(answer)
    return dict(
        text=f"Differentiate {prose(f)} with respect to x.",
        latex="",
        answer=answer_text,
        answer_text=answer_text,
        distractors=expr_distractors(answer, cands),
    )


_KINDS = (_product_chain, _quotient_chain, _triple_product)


@generator("exd_x_c3_product_quotient_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_c3_product_quotient_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Differentiate a function that needs two of the product, quotient and chain rules together."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    kind = rng.choice(_KINDS)
    return draw(kind, rng)
