"""Exam-tier source for `differentiate-polynomial`.

The hard tier differentiates a sum of integer powers that is already written term by term. Here the function
arrives as a *quotient by a power of x* or as a *product with a square root*, so the learner has to choose the
method first: split the fraction (or collect the exponents) into separate powers, which gives negative and
fractional exponents, and only then apply the power rule. Reaching for the quotient rule is legal but slower, and
differentiating numerator and denominator separately is the classic wrong turn this tier catches.

Built forwards from the term list ``[(coefficient, exponent), ...]`` of the expanded form. The prompt is the
grouped form; the answer is the term-by-term power rule. An independent route differentiates the grouped form
directly with sympy.
"""

from __future__ import annotations

import random

from sympy import Mul, Pow, Rational, diff, sqrt

from ..core import check_distinct, check_equivalent, generator, join_terms, phrase, pick_distractors, value_safe_str
from .exd_x_c1_common import X, tex, x_pow

_PATTERNS = (
    "Find f'(x) given that f(x) = {f}.",
    "If f(x) = {f}, find f'(x).",
    "Find dy/dx when y = {f}.",
)


def _nonzero(rng: random.Random, hi: int, step: int = 1) -> int:
    return step * rng.choice([m for m in range(-hi, hi + 1) if m != 0])


def _poly_sum(terms):
    total = 0
    for coef, e in terms:
        total += coef * X**e
    return total


def _build(rng: random.Random) -> dict:
    if rng.random() < 0.5:
        # (a x^n1 + b x^n2 + c) / x^k : exponents n - k, the last two negative.
        k = rng.choice([2, 3])
        n1 = rng.randint(k + 1, k + 3)
        n2 = rng.randint(1, k - 1)
        a, b, c = _nonzero(rng, 6), _nonzero(rng, 6), _nonzero(rng, 9)
        num_terms = [(a, n1), (b, n2), (c, 0)]
        num = a * X**n1 + b * X**n2 + c
        num_text = join_terms([(a, x_pow(n1)), (b, x_pow(n2)), (c, "")])
        f_text = f"({num_text})/{x_pow(k)}"
        f_tex = rf"\frac{{{tex(num)}}}{{{tex(X**k)}}}"
        terms = [(coef, Rational(n - k)) for coef, n in num_terms]
        grouped = Mul(num, Pow(X, -k, evaluate=False), evaluate=False)
        num_deriv = diff(num, X)
        wrong_quotient = num_deriv / X**k
    else:
        # (a x^2 + b x + c) / sqrt(x), even coefficients so every derivative coefficient is an integer.
        a, b, c = _nonzero(rng, 4, 2), _nonzero(rng, 4, 2), _nonzero(rng, 4, 2)
        num = a * X**2 + b * X + c
        num_text = join_terms([(a, "x^2"), (b, "x"), (c, "")])
        f_text = f"({num_text})/√x"
        f_tex = rf"\frac{{{tex(num)}}}{{\sqrt{{x}}}}"
        terms = [(a, Rational(3, 2)), (b, Rational(1, 2)), (c, Rational(-1, 2))]
        grouped = Mul(num, Pow(sqrt(X), -1, evaluate=False), evaluate=False)
        num_deriv = diff(num, X)
        wrong_quotient = num_deriv / sqrt(X)

    answer = _poly_sum([(coef * e, e - 1) for coef, e in terms])
    check_equivalent(diff(grouped, X), answer, "polynomial exam: grouped-form derivative disagrees with the power rule")

    # Wrong turns, each a named slip.
    exponent_up = _poly_sum([(coef * e, e + 1 if e < 0 else e - 1) for coef, e in terms])
    sign_lost = _poly_sum([(coef * abs(e), e - 1) if e < 0 else (coef * e, e - 1) for coef, e in terms])
    candidates = [wrong_quotient, exponent_up, sign_lost]
    check_distinct(answer, *candidates, msg="polynomial exam: a distractor equals the answer")

    answer_text = value_safe_str(answer)
    return dict(
        text=phrase(rng, _PATTERNS, f=f_text),
        latex=rf"\text{{Find }} f'(x) \text{{ given that }} f(x) = {f_tex}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=tex(answer),
        working=[
            f"Split into separate powers of x: f(x) = {value_safe_str(_poly_sum(terms))}",
            f"Power rule on each term: f'(x) = {answer_text}",
        ],
        distractors=pick_distractors(answer_text, [value_safe_str(c) for c in candidates]),
    )


@generator("exd_x_c1_polynomial", level=14, presets={"exam": dict(tier="exam")})
def exd_x_c1_polynomial(rng: random.Random, tier: str = "exam") -> dict:
    """Differentiate a quotient by a power of x (or by sqrt x) after rewriting it as separate powers."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
