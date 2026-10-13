"""Exam source for "a single mode of the heat equation": a three-part chain with exact values.

A rod of length L has u_t = alpha u_xx on 0 < x < L, and either both ends held at temperature 0 (Dirichlet:
u = c e^(-lambda t) sin(n pi x / L)) or both ends insulated (Neumann: u = c e^(-lambda t) cos(n pi x / L)). The initial
temperature is the single mode c sin or c cos of (n pi x / L). The learner works through:

(a) the decay rate lambda = alpha n^2 pi^2 / L^2                                   expression, 2 marks
(b) the time T at which the amplitude has halved, ln 2 / lambda                     expression, 2 marks, follows from (a)
(c) the quantity that the boundary condition does not fix, at x = 0 and time T:
    u_x(0, T) = c (n pi / L) e^(-lambda T) for Dirichlet, u(0, T) = c e^(-lambda T) for Neumann    expression, 3 marks,
    follows from (a) and (b)

Because e^(-lambda T) = 1/2 exactly, (c) is c n pi / (2 L) (Dirichlet) or c / 2 (Neumann); a learner who carries a wrong
lambda or T forward is still marked on the consistent value e^(-lambda T) built from their own two answers.
"""

from __future__ import annotations

import random

from sympy import Integer, Rational, exp, log, pi

from ..core import check, generator
from .exd_x_r2f_common import expr_part

_LENGTHS = (6, 8, 10, 12, 15, 18, 20, 24, 30)


def _argument(n: int, length: int) -> str:
    """ "nπx/L" with n/L in lowest terms, as it would be written: πx/12, 3πx/10."""
    ratio = Rational(n, length)
    return f"{'' if ratio.p == 1 else ratio.p}πx/{ratio.q}"


@generator(
    "exd_x_r2f_heat_rod_mode",
    level=17,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_r2f_heat_rod_mode(rng: random.Random, tier: str = "exam") -> dict:
    """Decay rate, half-life of the amplitude, and the boundary quantity at that time."""
    check(tier == "exam", f"unknown tier {tier!r}")
    dirichlet = rng.random() < 0.5
    length, n = rng.choice(_LENGTHS), rng.randint(1, 4)
    alpha = rng.randint(2, 9)
    c = rng.choice((-9, -8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9))

    lam = Rational(alpha * n * n, length * length) * pi**2
    half = log(2) / lam
    end_val = Rational(c) * (n * pi / length if dirichlet else 1) * Rational(1, 2)

    # Independent checks: the decay rate is alpha (n pi / L)^2, the half-life halves the amplitude, and the
    # boundary quantity at that time is c (n pi / L) / 2 or c / 2.
    check(lam == alpha * (n * pi / length) ** 2, "decay rate is not alpha (n pi / L)^2")
    check(exp(-lam * half) == Rational(1, 2), "half-life does not halve the amplitude")
    check(c * exp(-lam * half) * (n * pi / length if dirichlet else 1) == end_val, "boundary value disagrees")

    arg = _argument(n, length)
    if dirichlet:
        condition = f"its ends held at temperature 0, so that u(0, t) = u({length}, t) = 0"
        form = f"u(x, t) = {c} e^(−λt) sin({arg})"
    else:
        condition = f"its ends insulated, so that u_x(0, t) = u_x({length}, t) = 0"
        form = f"u(x, t) = {c} e^(−λt) cos({arg})"
    stem = (
        f"The temperature u(x, t) in a rod of length {length} satisfies u_t = {alpha}u_xx for 0 < x < {length}, with "
        f"{condition}. The solution has the form {form}."
    )

    a_part = expr_part(
        "Find the value of λ.",
        lam,
        2,
        [f"Substituting into u_t = {alpha}u_xx gives λ = {alpha}(nπ/L)² with nπ/L = {n}π/{length}.", f"λ = {lam}."],
        [
            ("drops-the-square", Rational(alpha * n, length * length) * pi**2),
            ("drops-alpha", Rational(n * n, length * length) * pi**2),
            ("forgets-to-square-length", Rational(alpha * n * n, length) * pi**2),
            ("forgets-to-square-pi", Rational(alpha * n * n, length * length) * pi),
        ],
    )
    b_part = expr_part(
        "Find the exact time T at which the amplitude of the solution has fallen to half its initial value.",
        half,
        2,
        ["e^(−λT) = 1/2, so T = ln 2 ÷ λ."],
        [
            ("forgets-ln-2", 1 / lam),
            ("multiplies-instead-of-dividing-by-rate", log(2) * lam),
            ("sign-error-logarithm", -log(2) / lam),
            ("takes-ln-2-as-2", 2 / lam),
        ],
        follow_through="log(2)/part_a",
    )
    if dirichlet:
        c_prompt = "Find the exact value of u_x(0, T), where T is the time found in part (b)."
        c_work = [f"u_x(0, t) = {c}(nπ/L)e^(−λt), so u_x(0, T) = {c}({n}π/{length}) × 1/2 = {end_val}."]
        c_cands = [
            ("drops-exponential-factor", c * n * pi / length),
            ("forgot-inner-derivative", Rational(c, 2)),
            ("sign-error-exponent", 2 * c * n * pi / length),
            ("half-life-applied-twice", c * n * pi / (4 * length)),
        ]
        c_ft = f"{c}*{n}*pi/{length}*exp(-part_a*part_b)"
    else:
        c_prompt = "Find the exact value of u(0, T), where T is the time found in part (b)."
        c_work = [f"u(0, t) = {c} e^(−λt), so u(0, T) = {c} × 1/2 = {end_val}."]
        c_cands = [
            ("drops-exponential-factor", Integer(c)),
            ("sign-error-exponent", Integer(2 * c)),
            ("half-life-applied-twice", Rational(c, 4)),
            ("sign-error-moving-term", Rational(-c, 2)),
        ]
        c_ft = f"{c}*exp(-part_a*part_b)"
    c_part = expr_part(c_prompt, end_val, 3, c_work, c_cands, follow_through=c_ft)
    return dict(text=stem, parts=[a_part, b_part, c_part])
