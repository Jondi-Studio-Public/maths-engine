"""Exam tier for the definition of the derivative: functions whose difference quotient needs a method.

The hard tier differentiates a quadratic, where `(x+h)^2 - x^2` expands and `h` cancels at once. Here the
learner meets a reciprocal `a/(x+b)`, a reciprocal square `a/x^2` or a square root `sqrt(c x + b)`. Each needs
a different move before `h` can cancel: a common denominator, a difference of squares, or the conjugate.

The key is `f'(x)` by the power and chain rules, checked against sympy's limit of the difference quotient at a
sample point (an independent route that does not use the rule that produced the key).
"""

from __future__ import annotations

import random

from sympy import Rational, Symbol, limit, sqrt, symbols

from ..core import check_equivalent, generator
from .exd_x_c3_support import draw, expr_distractors, prose, typed

x = symbols("x")
h = Symbol("h")

_STEM = "Using the definition f'(x) = lim_(h->0) (f(x+h) - f(x)) / h"


def _reciprocal(rng: random.Random) -> dict:
    a = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5, 6, 7])
    b = rng.randint(1, 5)
    f = a / (x + b)
    answer = -a / (x + b) ** 2
    cands = [
        -a / (x + b),  # forgets to square the denominator
        a / (x + b) ** 2,  # loses the sign
        -a / x**2,  # drops the shift b
        -a / (x + b) ** 3,  # exponent goes the wrong way
    ]
    return _pack(f, answer, cands, f"{_STEM}, differentiate f(x) = {prose(f)} from first principles.")


def _reciprocal_square(rng: random.Random) -> dict:
    a = rng.choice([-4, -3, -2, 1, 2, 3, 4, 5])
    f = a / x**2
    answer = -2 * a / x**3
    cands = [
        -a / x**3,  # forgets the factor 2 from the exponent
        2 * a / x**3,  # loses the sign
        -2 * a / x**2,  # does not lower the power
        -a / x,  # treats it as a/x
    ]
    return _pack(
        f, answer, cands, f"Find f'(x) for f(x) = {prose(f)} from first principles, starting from the limit definition."
    )


def _root(rng: random.Random) -> dict:
    c = rng.choice([2, 3, 4, 5, 6])
    b = rng.randint(1, 9)
    f = sqrt(c * x + b)
    answer = c / (2 * sqrt(c * x + b))
    cands = [
        1 / (2 * sqrt(c * x + b)),  # forgets the factor c from the inner function
        c / sqrt(c * x + b),  # forgets the 2 from the conjugate
        c / (2 * sqrt(x)),  # drops the constant b
        c * sqrt(c * x + b) / 2,  # leaves the root in the numerator
    ]
    return _pack(
        f,
        answer,
        cands,
        f"Differentiate f(x) = {prose(f)} from first principles: write the difference quotient and rationalise.",
    )


def _pack(f, answer, cands, text) -> dict:
    # Independent route: the limit of the difference quotient, evaluated at a sample point.
    point = Rational(3, 2)
    quotient = ((f.subs(x, point + h) - f.subs(x, point)) / h).simplify()
    check_equivalent(limit(quotient, h, 0), answer.subs(x, point), "difference-quotient limit disagrees with the key")
    answer_text = typed(answer)
    return dict(
        text=text,
        latex="",
        answer=answer_text,
        answer_text=answer_text,
        distractors=expr_distractors(answer, cands),
    )


_KINDS = (_reciprocal, _reciprocal_square, _root)


@generator("exd_x_c3_first_principles", level=13, presets={"exam": dict(tier="exam")})
def exd_x_c3_first_principles(rng: random.Random, tier: str = "exam") -> dict:
    """Differentiate a reciprocal, reciprocal-square or square-root function from first principles."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(rng.choice(_KINDS), rng)
