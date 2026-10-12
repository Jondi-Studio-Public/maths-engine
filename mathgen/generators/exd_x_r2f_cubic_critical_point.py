"""Exam source for "find an unconstrained critical point": a four-part chain on f(x, y) = a x^3 + b y^3 - c x y.

The cubic has two real critical points: the origin, and a second point (x0, y0) found by solving f_x = f_y = 0.

(a) the x-coordinate of the second critical point            solve the system            fraction, 3 marks
(b) its y-coordinate                                          y = 3 a x^2 / c (from f_x = 0)    fraction, 1 mark, follows from (a)
(c) D = f_xx f_yy - f_xy^2 there                              36 a b x y - c^2             fraction, 2 marks, follows from (a), (b)
(d) the value of f there (D > 0, so a local extremum)         a x^3 + b y^3 - c x y        fraction, 2 marks, follows from (a), (b)

## Construction

The point is chosen first: (x0, y0) = (g u, g v) with u and v coprime non-zero integers. Then

    a = t v^3,   b = t u^3,   c = 3 t g u^2 v^2

makes f_x = 3 a x^2 - c y and f_y = 3 b y^2 - c x vanish at (x0, y0): 3 a x0^2 = 3 t g^2 u^2 v^3 = c y0 and
3 b y0^2 = 3 t g^2 u^3 v^2 = c x0. Eliminating y shows x^3 = c^3 / (27 a^2 b) has one real root, so there are no further
real critical points. D at the point is 27 t^2 g^2 u^4 v^4 > 0, so it is always a local extremum.
"""

from __future__ import annotations

import math
import random

from sympy import Rational

from ..core import attempts, check, generator, join_terms, rational_text
from .exd_x_r2f_common import frac_part

_UV = (-3, -2, -1, 1, 2, 3)


@generator(
    "exd_x_r2f_cubic_critical_point",
    level=18,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_r2f_cubic_critical_point(rng: random.Random, tier: str = "exam") -> dict:
    """A cubic with a second critical point: its coordinates, the Hessian determinant there, the value of f."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "coprime u, v with moderate coefficients"):
        u, v = rng.choice(_UV), rng.choice(_UV)
        g, t = rng.choice((1, 2, 3)), rng.choice((1, 2, 3))
        if math.gcd(abs(u), abs(v)) != 1 or 3 * t * g * u * u * abs(v) ** 2 > 300:
            continue
        a, b, c = t * v**3, t * u**3, 3 * t * g * u * u * v * v
        if abs(a) <= 81 and abs(b) <= 81:
            break
    x0, y0 = g * u, g * v

    # Independent check: the gradient vanishes at the point, and eliminating y leaves a single real root x^3 = c^3/(27 a^2 b).
    check(3 * a * x0**2 - c * y0 == 0 and 3 * b * y0**2 - c * x0 == 0, "gradient does not vanish at the point")
    check(Rational(x0) ** 3 == Rational(c**3, 27 * a * a * b), "the cubic for x0 disagrees")
    d_val = 36 * a * b * x0 * y0 - c * c
    check(d_val == 27 * t * t * g * g * (u * v) ** 4 and d_val > 0, "Hessian determinant disagrees")
    f_val = a * x0**3 + b * y0**3 - c * x0 * y0
    check(f_val == -t * g**3 * u**3 * v**3, "value of f disagrees")

    f_text = join_terms([(a, "x³"), (b, "y³"), (-c, "xy")])
    stem = (
        f"The function f(x, y) = {f_text} has exactly two critical points. One of them is the origin; "
        "let (x₀, y₀) be the other."
    )
    a_part = frac_part(
        "Find x₀.",
        x0,
        3,
        [
            f"f_x = {3 * a}x² - {c}y = 0 and f_y = {3 * b}y² - {c}x = 0.",
            f"From f_x = 0, y = {rational_text(Rational(3 * a, c))}x². Substituting: x({rational_text(Rational(27 * a * a * b, c * c))}x³ - {c}) = 0, "
            f"so x₀ = {x0}.",
        ],
        [
            ("swaps-x-and-y", y0),
            ("sign-error-coordinate", -x0),
            ("gives-the-trivial-critical-point", 0),
            ("swaps-x-and-y", Rational(c, 3 * a)),
        ],
    )
    b_part = frac_part(
        "Hence find y₀.",
        y0,
        1,
        [f"y₀ = {rational_text(Rational(3 * a, c))} × {x0}² = {y0}."],
        [
            ("swaps-x-and-y", x0),
            ("sign-error-coordinate", -y0),
            ("gives-the-trivial-critical-point", 0),
            ("sign-error-coordinate", -x0),
        ],
        follow_through=f"3*({a})*part_a^2/({c})",
    )
    c_part = frac_part(
        "Find the value of D = f_xx f_yy − (f_xy)² at (x₀, y₀).",
        d_val,
        2,
        [
            f"f_xx = {6 * a}x, f_yy = {6 * b}y, f_xy = {-c}.",
            f"D = ({6 * a * x0})({6 * b * y0}) − ({-c})² = {d_val}.",
        ],
        [
            ("determinant-sign-error", 36 * a * b * x0 * y0 + c * c),
            ("forgot-off-diagonal-product", 36 * a * b * x0 * y0),
            ("forgot-to-square-mixed-partial", 36 * a * b * x0 * y0 + c),
            ("determinant-sign-error", -d_val),
        ],
        follow_through=f"36*({a})*({b})*part_a*part_b - {c * c}",
    )
    d_part = frac_part(
        "Since D > 0, (x₀, y₀) is a local extremum. Find the value of f there.",
        f_val,
        2,
        [f"f(x₀, y₀) = {join_terms([(a, f'({x0})³'), (b, f'({y0})³'), (-c, f'({x0})({y0})')])} = {f_val}."],
        [
            ("sign-error-cross-term", a * x0**3 + b * y0**3 + c * x0 * y0),
            ("drops-cross-term", a * x0**3 + b * y0**3),
            ("sign-error-coordinate", -f_val),
            ("swaps-x-and-y", d_val),
        ],
        follow_through=f"({a})*part_a^3 + ({b})*part_b^3 - ({c})*part_a*part_b",
    )
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])
