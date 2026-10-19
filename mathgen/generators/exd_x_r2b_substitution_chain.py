"""Generates `exd_x_r2b_substitution_chain` questions: a multi-part exam chain on integration by substitution.

Skill: integration-by-substitution (exam tier).

A definite integral `I = integral of m x^p (x^(p+1) + k)^n dx from x = 0 to x = b` is evaluated with the substitution
`u = x^(p+1) + k`, so `du = (p+1) x^p dx` and the integral becomes `(m / (p+1)) integral of u^n du` from `u = k` to
`u = b^(p+1) + k`:

(a) the value of u at the upper limit                             fraction,   1 mark
(b) the integrand in terms of u, g(u) = m u^n / (p+1)             expression, 2 marks
(c) the value of I, to 2 decimal places                           number,     2 marks, from (a) and (b)
(d) the mean value of the original integrand on [0, b], 2 places  number,     1 mark,  I / b

## Construction

`m`, `p`, `k`, `n` and `b` are drawn; `n` is 2, 3 or -2 (never -1, which would need a logarithm) and the numbers are
kept small enough that every value is a modest rational. The value of I is computed twice, once through the
substitution in exact fractions and once from an antiderivative of the original integrand in x (differentiated back
to check it), and the two must agree. The keys are drawn only clear of a rounding tie, and the route through the
rounded I must give the same mean value to 2 places.

The follow-through for (c) reads the learner's coefficient out of (b) as its value at `u = 1`, so a wrong coefficient
(a forgotten `1/(p+1)`) is carried honestly into the value of I.
"""

from __future__ import annotations

from fractions import Fraction

import sympy
from sympy import Rational, symbols

from ..core import check, generator, join_terms, phrase, round_half_up
from .exd_x_r2b_support import (
    Redraw,
    clear_of_boundary,
    draw,
    expression_part,
    fraction_part,
    number_part,
    same_function,
    typed,
)

x, u = symbols("x u")
_STEMS = (
    "Let I be the value of the integral of {integrand} dx between x = 0 and x = {b}. Use the substitution u = {inner}.",
    "The integral I is the integral of {integrand} dx from x = 0 to x = {b}. It is evaluated using the substitution "
    "u = {inner}.",
)


def _integrand_text(m: int, p: int, k: int, n: int) -> str:
    lead = f"{m}x" + (f"^{p}" if p > 1 else "")
    inner = f"(x^{p + 1} + {k})"
    return f"{lead}/{inner}^{-n}" if n < 0 else f"{lead}{inner}^{n}"


def _build(rng):
    p = rng.choice([1, 1, 2])
    n = rng.choice([2, 3, -2])
    k = rng.randint(1, 5)
    b = rng.randint(2, 3 if p == 1 else 2)
    m = rng.choice([2, 3, 4, 5, 6, 8, 9, 10, 12])
    u2 = b ** (p + 1) + k
    coefficient = Fraction(m, p + 1)
    if u2 > 14:
        raise Redraw("an upper limit in u that makes the value unwieldy")

    # ---- the value through the substitution, in exact fractions -------------------------------------------------
    value = coefficient * (Fraction(u2) ** (n + 1) - Fraction(k) ** (n + 1)) / (n + 1)
    # ---- independently: integrate the original integrand over the x-limits -------------------------------------------
    integrand = m * x**p * (x ** (p + 1) + k) ** n
    antiderivative_x = Rational(m, (p + 1) * (n + 1)) * (x ** (p + 1) + k) ** (n + 1)
    check(same_function(sympy.diff(antiderivative_x, x), integrand), "the antiderivative does not differentiate back")
    direct = antiderivative_x.subs(x, b) - antiderivative_x.subs(x, 0)
    check(direct == Rational(value.numerator, value.denominator), "the substitution disagrees with direct evaluation")
    mean = value / b
    if not (clear_of_boundary(value, 2) and clear_of_boundary(mean, 2)):
        raise Redraw("a key too close to a rounding tie")
    if round_half_up(round_half_up(value, 2) / b, 2) != round_half_up(mean, 2):
        raise Redraw("rounding I first changes the last digit of the mean value")

    stem = phrase(
        rng, _STEMS, integrand=_integrand_text(m, p, k, n), b=b, inner=join_terms([(1, f"x^{p + 1}"), (k, "")])
    )

    # (a) the upper limit in u
    a_wrong = [
        (b ** (p + 1), "dropped-constant-term"),  # the +k lost
        (b + k, "limit-not-transformed"),  # x = b put straight in for u
        ((b + k) ** (p + 1), "inner-function-misevaluated"),
    ]
    part_a = fraction_part(
        f"Find the value of u at the upper limit, x = {b}.",
        u2,
        1,
        [f"u = {b}^{p + 1} + {k} = {u2}."],
        a_wrong,
        sensible_range=(0, None),
    )

    # (b) the integrand in terms of u
    g = Rational(coefficient.numerator, coefficient.denominator) * u**n
    antiderivative = g * u / (n + 1)
    b_wrong = [
        (m * u**n, "du-factor-missed"),  # the 1/(p+1) from du = (p+1)x^p dx forgotten
        (m * (p + 1) * u**n, "inverted-ratio"),
        (antiderivative, "antiderivative-given-for-integrand"),
        (
            Rational(coefficient.numerator, coefficient.denominator) * u ** (n + 1) / (n + 1) * (n + 1),
            "power-raised-early",
        ),
    ]
    part_b = expression_part(
        f"Show that I can be written as the integral of g(u) du between u = {k} and u = {u2}, and find g(u).",
        g,
        2,
        [
            f"du = {p + 1}{'x' if p == 1 else 'x^2'} dx, so {m}{'x' if p == 1 else 'x^2'} dx = {coefficient} du.",
            f"g(u) = {typed(g)}.",
        ],
        b_wrong,
    )

    # (c) the exact value of I
    c_wrong = [
        (coefficient * Fraction(u2) ** (n + 1) / (n + 1), "upper-limit-only"),
        (Fraction(m) * (Fraction(u2) ** (n + 1) - Fraction(k) ** (n + 1)) / (n + 1), "du-factor-missed"),
        (coefficient * (Fraction(u2) ** (n + 1) - Fraction(k) ** (n + 1)) / n, "integration-power-not-divided"),
        (coefficient * Fraction(b) ** (n + 1) / (n + 1) if n > 0 else None, "limit-not-transformed"),
    ]
    part_c = number_part(
        "Hence find the value of I.",
        value,
        2,
        2,
        [
            f"I = {f'({coefficient})' if coefficient.denominator != 1 else coefficient} × [u^{n + 1}/{n + 1}] from {k} to {u2} = {value}."
        ],
        c_wrong,
        follow_through=f"Subs(part_b, (u,), (1,))*(part_a**({n + 1}) - {k}**({n + 1}))/({n + 1})",
    )

    # (d) the mean value over [0, b]
    d_wrong = [
        (value, "total-not-divided-by-width"),
        (value / u2, "divided-by-endpoint-not-width"),
        (value * b, "multiplied-instead-of-divided"),
    ]
    part_d = number_part(
        f"Hence find the mean value of {_integrand_text(m, p, k, n)} over the interval 0 ≤ x ≤ {b}.",
        mean,
        2,
        1,
        [f"Mean value = I / {b} = {mean}."],
        d_wrong,
        follow_through=f"part_c/{b}",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2b_substitution_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_substitution_chain(rng, tier: str = "exam"):
    """A definite integral by substitution: the new limit, the integrand in u, the value, the mean value."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
