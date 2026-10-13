"""Exam source for "characteristic roots of a second-order linear ODE": a four-part chain with complex roots.

y'' + b y' + c y = 0 has characteristic roots alpha +/- k i (k > 0), where alpha = -b/2 and k^2 = c - alpha^2, and the
initial conditions y(0) = y0, y'(0) = v0 fix the solution y = e^(alpha x) (C cos kx + D sin kx).

(a) alpha, the real part of the roots                        fraction,   1 mark
(b) k, the imaginary part                                    fraction,   2 marks, sqrt(c - part_a^2)
(c) D, from y(0) and y'(0)                                   fraction,   2 marks, (v0 - part_a y0) / part_b
(d) the exact value of y at x = pi / (2k)                    expression, 2 marks, part_c exp(part_a pi / (2 part_b))

C = y0 comes straight from y(0) and is not asked for. At x = pi/(2k) the cosine term vanishes and sin(kx) = 1, so (d)
is D e^(alpha pi / (2k)) exactly.
"""

from __future__ import annotations

import math
import random

from sympy import Rational, exp, pi

from ..core import attempts, check, generator, join_terms, phrase, rational_text
from ._exd_distinct_parts import distinct_part_answers
from .exd_x_r2f_common import expr_part, frac_part


@generator(
    "exd_x_r2f_damped_oscillation",
    level=17,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
@distinct_part_answers
def exd_x_r2f_damped_oscillation(rng: random.Random, tier: str = "exam") -> dict:
    """Complex characteristic roots, then the solution that fits two initial conditions, evaluated exactly."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "initial conditions with D not zero"):
        alpha = rng.choice((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5))
        k = rng.randint(2, 6)
        y0 = rng.choice((-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6))
        v0 = rng.randint(-9, 9)
        if v0 - alpha * y0 != 0:
            break
    b, c = -2 * alpha, alpha * alpha + k * k
    d_val = Rational(v0 - alpha * y0, k)

    # Independent checks, kept cheap because this runs on every generation. alpha + ki is a root of r^2 + b r + c
    # (so e^(alpha x) cos kx and e^(alpha x) sin kx solve the equation), y = e^(alpha x)(y0 cos kx + D sin kx) has
    # y(0) = y0 and y'(0) = alpha y0 + k D, and its value at pi/(2k) agrees with the key when evaluated numerically.
    root = complex(alpha, k)
    check(root * root + b * root + c == 0, "alpha + ki is not a root of the characteristic equation")
    check(alpha * y0 + k * d_val == v0, "the solution misses y'(0)")
    x_end = math.pi / (2 * k)
    y_end = math.exp(alpha * x_end) * (y0 * math.cos(k * x_end) + float(d_val) * math.sin(k * x_end))
    expected = d_val * exp(Rational(alpha, 2 * k) * pi)
    check(abs(y_end - float(expected)) < 1e-9 * max(1.0, abs(y_end)), "value at pi/(2k) disagrees")

    ode = join_terms([(1, "y''"), (b, "y'"), (c, "y")])
    stem = phrase(
        rng,
        (
            "A function y(x) satisfies {ode} = 0, with y(0) = {y0} and y'(0) = {v0}.",
            "Consider the initial value problem {ode} = 0, with y(0) = {y0} and y'(0) = {v0}.",
            "The displacement y(x) of an oscillating system obeys {ode} = 0, with y(0) = {y0} and y'(0) = {v0}.",
        ),
        ode=ode,
        y0=y0,
        v0=v0,
    )
    a_part = frac_part(
        "The characteristic equation has roots α ± ki with k > 0. Find α.",
        alpha,
        1,
        [f"r² + ({b})r + {c} = 0 has real part α = −b/2 = {alpha}."],
        [("sign-flip-on-b", -alpha), ("forgets-to-halve", b), ("forgets-to-halve", -b)],
    )
    b_part = frac_part(
        "Find k.",
        k,
        2,
        [f"k² = c − α² = {c} − {alpha * alpha} = {k * k}, so k = {k}."],
        [
            ("forgets-to-halve", 2 * k),
            ("forgets-square-root", k * k),
            ("real-part-only", abs(alpha)),
            ("coefficients-as-roots", c),
            ("coefficients-as-roots", abs(b)),
        ],
        follow_through=f"sqrt({c} - part_a^2)",
        sensible_range=(0, None),
    )
    c_part = frac_part(
        f"The solution has the form y = e^(αx)(C cos kx + D sin kx). Find D.",
        d_val,
        2,
        [
            f"y(0) = C = {y0}.",
            f"y'(0) = αC + kD = {v0}, so D = ({v0} − ({alpha})({y0})) ÷ {k} = {rational_text(d_val)}.",
        ],
        [
            ("ignores-real-part-in-derivative", Rational(v0, k)),
            ("sign-error-moving-term", Rational(v0 + alpha * y0, k)),
            ("forgot-to-divide-by-coefficient", Rational(v0 - alpha * y0)),
            ("confuses-c-and-d-constants", Rational(y0)),
        ],
        follow_through=f"({v0} - part_a*({y0}))/part_b",
    )
    d_ans = expected
    d_part = expr_part(
        "Find the exact value of y at x = π/(2k).",
        d_ans,
        2,
        [
            "At x = π/(2k), cos kx = 0 and sin kx = 1.",
            f"y = D e^(απ/(2k)) = ({rational_text(d_val)}) e^({rational_text(Rational(alpha, 2 * k))}π).",
        ],
        [
            ("drops-exponential-factor", d_val),
            ("swaps-sine-and-cosine", y0 * exp(Rational(alpha, 2 * k) * pi)),
            ("sign-error-exponent", d_val * exp(-Rational(alpha, 2 * k) * pi)),
            ("forgets-k-in-exponent", d_val * exp(Rational(alpha, 2) * pi)),
        ],
        follow_through="part_c*exp(part_a*pi/(2*part_b))",
    )
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])
