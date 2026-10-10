"""Exam tier for differentiating inverse trigonometric functions inside a larger expression.

The hard tier differentiates `arcsin`, `arccos` or `arctan` of a linear argument, one rule per question. Here the
inverse function sits inside a product (`x^m arctan(k x)`), takes a quadratic argument (`arcsin(a x^2)`), or has an
exponential argument (`arctan(e^(k x))`), so the learner combines the inverse-trig derivative with the product or
chain rule. The key is the hand-applied rule, checked against sympy's `diff`.
"""

from __future__ import annotations

import random

from sympy import acos, asin, atan, diff, exp, sqrt, symbols

from ..core import check_equivalent, generator
from .exd_x_c3_support import draw, expr_distractors, prose, typed

x = symbols("x")


def _product(rng: random.Random) -> dict:
    """x^m * arctan(k x) or x^m * arcsin(k x): product rule, with a chain rule inside the inverse function."""
    m = rng.choice([1, 2, 3])
    k = rng.choice([2, 3, 4])
    if rng.random() < 0.5:
        f = x**m * atan(k * x)
        rest = x**m * k / (1 + k**2 * x**2)
        other = x**m * k / (1 - k**2 * x**2)
        wrong_square = x**m * k / (1 + k * x**2)
        inner = atan(k * x)
        no_chain = x**m / (1 + k**2 * x**2)
    else:
        f = x**m * asin(k * x)
        rest = x**m * k / sqrt(1 - k**2 * x**2)
        other = x**m * k / sqrt(1 + k**2 * x**2)
        wrong_square = x**m * k / sqrt(1 - k * x**2)
        inner = asin(k * x)
        no_chain = x**m / sqrt(1 - k**2 * x**2)
    answer = m * x ** (m - 1) * inner + rest
    cands = [
        answer - m * x ** (m - 1) * inner,  # drops the term from differentiating x^m
        answer - rest,  # drops the term from differentiating the inverse function
        m * x ** (m - 1) * inner + no_chain,  # forgets the chain-rule factor k
        m * x ** (m - 1) * inner + other,  # wrong sign inside the inverse-trig derivative
        m * x ** (m - 1) * inner + wrong_square,  # does not square k
    ]
    return _pack(f, answer, cands)


def _quadratic_argument(rng: random.Random) -> dict:
    """arcsin(a x^2), arccos(a x^2) or arctan(a x^2): chain rule with an inner function that is not linear."""
    a = rng.choice([1, 2, 3])
    name = rng.choice(["asin", "acos", "atan"])
    u = a * x**2
    du = 2 * a * x
    if name == "asin":
        f, answer = asin(u), du / sqrt(1 - u**2)
        cands = [1 / sqrt(1 - u**2), -du / sqrt(1 - u**2), du / sqrt(1 - u), du / sqrt(1 + u**2)]
    elif name == "acos":
        f, answer = acos(u), -du / sqrt(1 - u**2)
        cands = [-1 / sqrt(1 - u**2), du / sqrt(1 - u**2), -du / sqrt(1 - u), -du / sqrt(1 + u**2)]
    else:
        f, answer = atan(u), du / (1 + u**2)
        cands = [1 / (1 + u**2), -du / (1 + u**2), du / (1 + u), du / (1 - u**2)]
    return _pack(f, answer, cands)


def _exponential_argument(rng: random.Random) -> dict:
    """arctan(e^(k x)): the inner function is exponential."""
    k = rng.choice([2, 3, 4, 5])
    u = exp(k * x)
    f = atan(u)
    answer = k * u / (1 + u**2)
    cands = [1 / (1 + u**2), k * u / (1 + u), k / (1 + u**2), -k * u / (1 + u**2), k * u / (1 - u**2)]
    return _pack(f, answer, cands)


def _pack(f, answer, cands) -> dict:
    check_equivalent(answer, diff(f, x), "hand-applied rules disagree with sympy diff")
    answer_text = typed(answer)
    return dict(
        text=f"Differentiate {prose(f)} with respect to x.",
        latex="",
        answer=str(answer),  # sympy's own spelling, so the grader reads atan/asin/acos natively
        answer_text=answer_text,
        distractors=expr_distractors(answer, cands),
    )


_KINDS = (_product, _quadratic_argument, _exponential_argument)


@generator("exd_x_c3_inverse_trig_derivative", level=10, presets={"exam": dict(tier="exam")})
def exd_x_c3_inverse_trig_derivative(rng: random.Random, tier: str = "exam") -> dict:
    """Differentiate an inverse trigonometric function inside a product or with a non-linear argument."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(rng.choice(_KINDS), rng)
