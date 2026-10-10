"""Exam-tier source for `differentiate-quotient-rule`.

The hard tier divides a linear function by a linear function. Here the numerator is a *power of a linear
function* and the denominator is a quadratic, ``(ax + b)^n / (cx^2 + d)``, so the quotient rule must be combined
with the chain rule in the numerator and the denominator's derivative ``2cx`` is not constant. The tidy answer
factors out ``(ax + b)^(n-1)``.

Built forwards from ``(a, b, c, d, n)``. The independent route differentiates the quotient with sympy's own
``diff``.
"""

from __future__ import annotations

import random
from math import gcd

from sympy import Mul, Pow, diff

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
    c = rng.randint(1, 3)
    # Exclude any d for which ax + b is a factor of cx^2 + d, so the quotient never cancels.
    d = rng.choice([m for m in range(-6, 7) if m != 0 and gcd(c, m) == 1 and c * b * b + m * a * a != 0])
    n = rng.choice([2, 3])
    lin = a * X + b
    quad = c * X**2 + d

    f = Mul(lin**n, Pow(quad, -1, evaluate=False), evaluate=False)
    # (u'v - uv') / v^2 with u = lin^n, v = quad, then the common (ax + b)^(n-1) taken out.
    bracket = (n * a * quad - 2 * c * X * lin).expand()
    answer = lin ** (n - 1) * bracket / quad**2
    check_equivalent(diff(f, X), answer, "quotient exam: factorised derivative disagrees with sympy")

    u1 = n * a * lin ** (n - 1)
    candidates = [
        -answer,  # subtracted in the wrong order
        lin ** (n - 1) * bracket / quad,  # forgot to square the denominator
        (n * lin ** (n - 1) * quad - 2 * c * X * lin**n) / quad**2,  # forgot the inner derivative a
        u1 / (2 * c * X),  # differentiated top and bottom separately
    ]
    check_distinct(answer, *candidates, msg="quotient exam: a distractor equals the answer")

    text_lin, text_quad = lin_text(a, b), join_terms([(c, "x^2"), (d, "")])
    answer_text = value_safe_str(answer)
    return dict(
        text=phrase(rng, _PATTERNS, f=f"{powered(text_lin, n)}/({text_quad})"),
        latex=rf"\text{{Find }} f'(x) \text{{ given that }} f(x) = \frac{{\left({tex(lin)}\right)^{{{n}}}}}{{{tex(quad)}}}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=tex(answer),
        working=[
            f"u = ({text_lin})^{n}, v = {text_quad}; u' = {value_safe_str(u1)}, v' = {value_safe_str(2 * c * X)}",
            f"(u'v - uv')/v^2, then take out the common ({text_lin})^{n - 1}: {answer_text}",
        ],
        distractors=pick_distractors(answer_text, [value_safe_str(cand) for cand in candidates]),
    )


@generator("exd_x_c1_quotient", level=14, presets={"exam": dict(tier="exam")})
def exd_x_c1_quotient(rng: random.Random, tier: str = "exam") -> dict:
    """Quotient rule with a power of a linear function over a quadratic."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
