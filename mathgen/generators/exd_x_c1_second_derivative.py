"""Exam-tier source for `second-derivative`.

The hard tier differentiates a sum of powers twice. Here the function is a *product*, ``x^m (ax + b)^n``, so
each of the two differentiations needs the product rule and the chain rule, and the cross term ``2u'v'`` that
appears in ``(uv)'' = u''v + 2u'v' + uv''`` is the step most easily dropped. The answer is given expanded.

Built forwards from ``(m, a, b, n)``. The independent route expands the function first and applies the power
rule twice term by term, a route that never uses the product rule.
"""

from __future__ import annotations

import random
from math import gcd

from sympy import Poly, diff, expand

from ..core import check_distinct, check_equivalent, generator, phrase, pick_distractors, value_safe_str
from .exd_x_c1_common import X, lin_text, powered, tex, x_pow

_PATTERNS = (
    "Find d^2y/dx^2 when y = {f}.",
    "If y = {f}, find the second derivative of y with respect to x.",
    "Find f''(x) given that f(x) = {f}.",
)


def _build(rng: random.Random) -> dict:
    m = rng.choice([1, 2])
    n = 3 if m == 2 else rng.choice([3, 4])
    a = rng.randint(2, 3)
    b = rng.choice([s for s in range(-4, 5) if s != 0 and gcd(a, s) == 1])
    lin = a * X + b
    u, v = X**m, lin**n
    f = u * v
    answer = expand(diff(f, X, 2))

    # Independent route: expand, then the power rule twice term by term.
    manual = 0
    for (power,), coef in Poly(expand(f), X).terms():
        manual += coef * power * (power - 1) * X ** (power - 2) if power >= 2 else 0
    check_equivalent(answer, manual, "second-derivative exam: product-rule result disagrees with the expanded route")

    u1, u2 = diff(u, X), diff(u, X, 2)
    v1, v2 = diff(v, X), diff(v, X, 2)
    chain_free_v1 = n * lin ** (n - 1)
    chain_free_v2 = n * (n - 1) * lin ** (n - 2)
    candidates = [
        expand(diff(f, X)),  # stopped after one derivative
        expand(u2 * v + u * v2),  # dropped the cross term 2u'v'
        expand(u2 * v + 2 * u1 * chain_free_v1 + u * chain_free_v2),  # forgot the inner derivative a each time
    ]
    if m == 2:
        candidates.append(expand(u2 * v2))  # multiplied the second derivatives
    check_distinct(answer, *candidates, msg="second-derivative exam: a distractor equals the answer")

    answer_text = value_safe_str(answer)
    f_text = f"{x_pow(m)}{powered(lin_text(a, b), n)}"
    return dict(
        text=phrase(rng, _PATTERNS, f=f_text),
        latex=rf"\text{{Find }} \frac{{d^2y}}{{dx^2}} \text{{ when }} y = {tex(u)}\left({tex(lin)}\right)^{{{n}}}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=tex(answer),
        working=[
            f"u = {x_pow(m)}, v = ({lin_text(a, b)})^{n}; u'' = {value_safe_str(u2)}, v'' = {value_safe_str(v2)}",
            f"(uv)'' = u''v + 2u'v' + uv'' = {answer_text}",
        ],
        distractors=pick_distractors(answer_text, [value_safe_str(c) for c in candidates]),
    )


@generator("exd_x_c1_second_derivative", level=14, presets={"exam": dict(tier="exam")})
def exd_x_c1_second_derivative(rng: random.Random, tier: str = "exam") -> dict:
    """Second derivative of x^m (ax + b)^n, needing the product and chain rules twice."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
