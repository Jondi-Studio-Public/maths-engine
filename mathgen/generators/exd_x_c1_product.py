"""Exam-tier source for `differentiate-product-rule`.

The hard tier multiplies a linear factor by a quadratic. Here both factors are *powers of a function*,
``(ax + b)^p (cx^2 + d)^q``, so the product rule has to be combined with the chain rule inside each of its two
terms, and the result is best left factorised: taking out the common ``(ax + b)^(p-1)(cx^2 + d)^(q-1)`` is the
efficient route a learner has to spot. Expanding first is possible but long.

Built forwards from ``(a, b, c, d, p, q)``. The independent route differentiates the product with sympy's own
``diff``.
"""

from __future__ import annotations

import random
from math import gcd

from sympy import diff

from ..core import check_distinct, check_equivalent, generator, join_terms, phrase, pick_distractors, value_safe_str
from .exd_x_c1_common import X, lin_text, powered, tex

_PATTERNS = (
    "Find f'(x) given that f(x) = {f}.",
    "If f(x) = {f}, find f'(x).",
    "Find dy/dx when y = {f}.",
)


def _build(rng: random.Random) -> dict:
    a = rng.randint(2, 3)  # a >= 2 so dropping the inner derivative a visibly changes the answer
    b = rng.choice([m for m in range(-4, 5) if m != 0 and gcd(a, m) == 1])
    c = rng.randint(1, 2)
    d = rng.choice([m for m in range(-5, 6) if m != 0 and gcd(c, m) == 1])
    p = rng.choice([2, 3])
    q = rng.choice([2, 3])
    lin = a * X + b
    quad = c * X**2 + d

    u, v = lin**p, quad**q
    f = u * v
    # Factorised derivative: pull out the common power of each bracket.
    bracket = p * a * quad + q * 2 * c * X * lin
    answer = lin ** (p - 1) * quad ** (q - 1) * bracket.expand()
    check_equivalent(diff(f, X), answer, "product exam: factorised derivative disagrees with sympy")

    u1, v1 = p * a * lin ** (p - 1), q * 2 * c * X * quad ** (q - 1)
    candidates = [
        u1 * v1,  # multiplied the derivatives
        u1 * v,  # forgot the second product-rule term
        p * lin ** (p - 1) * v + u * v1,  # forgot the inner derivative a of the linear factor
        u1 * v + u * q * quad ** (q - 1),  # forgot the inner derivative 2cx of the quadratic factor
    ]
    check_distinct(answer, *candidates, msg="product exam: a distractor equals the answer")

    text_lin, text_quad = lin_text(a, b), join_terms([(c, "x^2"), (d, "")])
    answer_text = value_safe_str(answer)
    return dict(
        text=phrase(rng, _PATTERNS, f=f"{powered(text_lin, p)}{powered(text_quad, q)}"),
        latex=rf"\text{{Find }} f'(x) \text{{ given that }} f(x) = \left({tex(lin)}\right)^{{{p}}}\left({tex(quad)}\right)^{{{q}}}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=tex(answer),
        working=[
            f"u = ({text_lin})^{p}, v = ({text_quad})^{q}; each needs the chain rule.",
            f"u' = {value_safe_str(u1)}, v' = {value_safe_str(v1)}",
            f"u'v + uv' = {answer_text}",
        ],
        distractors=pick_distractors(answer_text, [value_safe_str(cand) for cand in candidates]),
    )


@generator("exd_x_c1_product", level=14, presets={"exam": dict(tier="exam")})
def exd_x_c1_product(rng: random.Random, tier: str = "exam") -> dict:
    """Product rule on two powers of functions, so the chain rule is needed inside each term."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
