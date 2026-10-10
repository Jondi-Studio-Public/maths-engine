"""Exam-tier source for `differentiate-chain-rule`.

The hard tier raises a quadratic ``x^2 + c`` to a positive whole power. Here the outer function is a *reciprocal
power* ``k/(ax^2 + bx + c)^n`` or a *square root* ``k*sqrt(ax^2 + bx + c)``, and the inner function is a full
quadratic, so its derivative ``2ax + b`` is a two-term factor that must be carried through. The learner has to
recognise the negative or fractional outer power before the chain rule can start.

Built forwards from ``(k, n, a, b, c)``. The independent route differentiates the same function with sympy's
own ``diff`` on the unevaluated composite.
"""

from __future__ import annotations

import random
from math import gcd

from sympy import Rational, diff, sqrt

from ..core import check_distinct, check_equivalent, generator, phrase, pick_distractors, value_safe_str
from .exd_x_c1_common import X, quad_text, tex

_PATTERNS = (
    "Find f'(x) given that f(x) = {f}.",
    "If f(x) = {f}, find f'(x).",
    "Find dy/dx when y = {f}.",
)


def _coefficients(rng: random.Random) -> tuple[int, int, int]:
    a = rng.randint(1, 3)
    b = rng.choice([m for m in range(-5, 6) if m != 0])
    c = rng.choice([m for m in range(-6, 7) if m != 0 and gcd(gcd(a, b), m) == 1])
    return a, b, c


def _reciprocal(rng: random.Random) -> dict:
    a, b, c = _coefficients(rng)
    n = rng.choice([2, 3])
    k = rng.choice([1, 2, 3, 4, 5, 6])
    inner = a * X**2 + b * X + c
    d_inner = 2 * a * X + b
    f = k / inner**n
    answer = -k * n * d_inner / inner ** (n + 1)
    check_equivalent(diff(f, X), answer, "chain exam: reciprocal-power derivative disagrees with sympy")

    candidates = [
        -k * n / inner ** (n + 1),  # forgot the inner derivative
        -k * n * d_inner / inner ** (n - 1),  # moved the exponent the wrong way
        -k * n * (2 * a * X) / inner ** (n + 1),  # inner derivative of the leading term only
    ]
    text_inner = quad_text(a, b, c)
    return dict(
        text=phrase(rng, _PATTERNS, f=f"{k}/({text_inner})^{n}"),
        latex=rf"\text{{Find }} f'(x) \text{{ given that }} f(x) = \frac{{{k}}}{{\left({tex(inner)}\right)^{{{n}}}}}",
        answer_expr=answer,
        candidates=candidates,
        working=[
            f"Write f(x) = {k}({text_inner})^(-{n}); the outer power is -{n}.",
            f"Outer: -{n}*{k}({text_inner})^(-{n + 1}); inner derivative: {value_safe_str(d_inner)}",
        ],
    )


def _root(rng: random.Random) -> dict:
    a, b, c = _coefficients(rng)
    k = rng.choice([2, 4, 6, 8])
    inner = a * X**2 + b * X + c
    d_inner = 2 * a * X + b
    f = k * sqrt(inner)
    answer = Rational(k, 2) * d_inner / sqrt(inner)
    check_equivalent(diff(f, X), answer, "chain exam: square-root derivative disagrees with sympy")

    candidates = [
        Rational(k, 2) / sqrt(inner),  # forgot the inner derivative
        k * d_inner / sqrt(inner),  # dropped the factor 1/2 from the outer power
        Rational(k, 2) * d_inner / inner ** Rational(3, 2),  # moved the exponent the wrong way
        Rational(k, 2) * (2 * a * X) / sqrt(inner),  # inner derivative of the leading term only
    ]
    text_inner = quad_text(a, b, c)
    return dict(
        text=phrase(rng, _PATTERNS, f=f"{k}√({text_inner})"),
        latex=rf"\text{{Find }} f'(x) \text{{ given that }} f(x) = {k}\sqrt{{{tex(inner)}}}",
        answer_expr=answer,
        candidates=candidates,
        working=[
            f"Write f(x) = {k}({text_inner})^(1/2).",
            f"Outer: {k}*(1/2)({text_inner})^(-1/2); inner derivative: {value_safe_str(d_inner)}",
        ],
    )


def _build(rng: random.Random) -> dict:
    d = _reciprocal(rng) if rng.random() < 0.5 else _root(rng)
    answer = d.pop("answer_expr")
    candidates = d.pop("candidates")
    check_distinct(answer, *candidates, msg="chain exam: a distractor equals the answer")
    answer_text = value_safe_str(answer)
    d.update(
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=tex(answer),
        distractors=pick_distractors(answer_text, [value_safe_str(c) for c in candidates]),
    )
    return d


@generator("exd_x_c1_chain", level=14, presets={"exam": dict(tier="exam")})
def exd_x_c1_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Chain rule with a reciprocal-power or square-root outer function and a full quadratic inner function."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
