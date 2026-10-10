"""Exam-tier source for `polynomial-long-division`: division whose quotient has fractional coefficients.

In the practice tiers the dividend is built from an integer quotient, so every step of the long division
divides evenly. Here the dividend and the non-monic divisor are drawn independently, so the leading
coefficient of the dividend is generally not a multiple of the divisor's, and the quotient and
remainder carry fractional coefficients. The learner has to keep exact fractions through every step.
Dividends may be missing interior powers (a zero coefficient), which a column layout has to hold open.

The key is the exact quotient and remainder from ``sympy.div`` over the rationals; the generator
re-multiplies (divisor x quotient + remainder) to confirm it reproduces the dividend exactly.
"""

from __future__ import annotations

import random

import sympy
from sympy import Poly, QQ, Rational, symbols

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors

x = symbols("x")


def _poly_text(expr) -> str:
    """Learner-facing polynomial text with exact fractional coefficients written as ``3x^2/2``."""
    poly = Poly(expr, x)
    out = ""
    for (power,), coef in poly.terms():
        coef = Rational(coef)
        num, den = abs(coef.p), coef.q
        if power == 0:
            body = str(num) if den == 1 else f"{num}/{den}"
        else:
            var = "x" if power == 1 else f"x^{power}"
            body = ("" if num == 1 else str(num)) + var + ("" if den == 1 else f"/{den}")
        if not out:
            out = ("-" if coef < 0 else "") + body
        else:
            out += f" {'-' if coef < 0 else '+'} {body}"
    return out or "0"


def _draw(rng: random.Random):
    while True:
        divisor_degree = rng.choice([1, 2])
        quotient_degree = rng.choice([2, 3]) if divisor_degree == 1 else rng.choice([1, 2])
        lead = rng.choice([2, 3, 4])
        divisor = lead * x**divisor_degree + sum(rng.randint(-5, 5) * x**k for k in range(divisor_degree))
        degree = divisor_degree + quotient_degree
        coeffs = [rng.randint(-9, 9) for _ in range(degree + 1)]
        coeffs[0] = rng.choice([1, -1]) * rng.randint(1, 9)
        dividend = sum(c * x ** (degree - i) for i, c in enumerate(coeffs))
        quotient, remainder = sympy.div(Poly(dividend, x, domain=QQ), Poly(divisor, x, domain=QQ))
        quotient, remainder = quotient.as_expr(), remainder.as_expr()
        q_coeffs = Poly(quotient, x).all_coeffs()
        if remainder == 0 or all(Rational(c).q == 1 for c in q_coeffs):
            continue
        if max(Rational(c).q for c in q_coeffs + Poly(remainder, x).all_coeffs()) > 16:
            continue
        return dividend, divisor, quotient, remainder


@generator("exd_x_a1_polynomial_long_division", level=11, presets={"exam": dict(tier="exam")})
def exd_x_a1_polynomial_long_division(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: long division by a non-monic divisor with a fractional quotient."""
    check(tier == "exam", f"unknown tier {tier!r}")
    dividend, divisor, quotient, remainder = _draw(rng)
    check_equivalent(sympy.expand(divisor * quotient + remainder), dividend, "divisor*quotient+remainder != dividend")
    check(sympy.degree(remainder, x) < sympy.degree(divisor, x), "remainder degree too large")
    key = quotient + remainder / divisor
    qt, rt, dt = _poly_text(quotient), _poly_text(remainder), _poly_text(divisor)
    answer_text = f"{qt} + ({rt})/({dt})"

    wrong = {
        "sign error on the remainder": (quotient - remainder / divisor, f"{qt} - ({rt})/({dt})"),
        "remainder not divided": (quotient + remainder, f"{qt} + ({rt})"),
        "remainder dropped": (quotient, qt),
    }
    cands = []
    for label, (value, text) in wrong.items():
        check_distinct(key, value, msg=label)
        cands.append(text)
    text = (
        f"Divide f(x) = {_poly_text(dividend)} by g(x) = {dt}. "
        "Give your answer as a single expression in the form quotient + remainder/divisor. "
        "The quotient and remainder may contain fractions."
    )
    return dict(
        text=text,
        latex=(
            rf"f(x)={sympy.latex(dividend)},\quad g(x)={sympy.latex(divisor)}."
            r"\ \text{Give your answer as a single expression in the form quotient + remainder/divisor. "
            r"The quotient and remainder may contain fractions.}"
        ),
        answer=str(key),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Divide f(x) by g(x): quotient {qt}, remainder {rt}.", f"Answer: {answer_text}"],
        distractors=pick_distractors(answer_text, cands),
    )
