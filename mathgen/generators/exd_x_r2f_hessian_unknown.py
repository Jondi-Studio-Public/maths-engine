"""Exam source for "the Hessian of a function of two variables": a four-part chain with an unknown coefficient.

f(x, y) = a x^3 + b y^3 + c x y + d x^2 + e y^2 + E (x^2 y + x y^2), with E an unknown constant, and the value of the
mixed partial f_xy at a point P(x0, y0) is given. The learner works through:

(a) E, from f_xy = c + 2E(x + y) at P                                 fraction, 2 marks
(b) f_xx at P = 6 a x0 + 2 d + 2 E y0                                  fraction, 1 mark,  follows from (a)
(c) f_yy at P = 6 b y0 + 2 e + 2 E x0                                  fraction, 1 mark,  follows from (a)
(d) the determinant f_xx f_yy - (f_xy)^2 of the Hessian at P           fraction, 2 marks, follows from (b), (c)

E is a whole number, drawn first; the stated value of f_xy at P is built from it. Both second partials depend on E, so a
wrong (a) carries into (b), (c) and (d).
"""

from __future__ import annotations

import random

import sympy
from sympy import Rational, symbols

from ..core import attempts, check, generator, join_terms
from .exd_x_r2f_common import frac_part

x, y = symbols("x y")
_POINT = (-3, -2, -1, 1, 2, 3)


@generator(
    "exd_x_r2f_hessian_unknown",
    level=17,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_r2f_hessian_unknown(rng: random.Random, tier: str = "exam") -> dict:
    """An unknown coefficient fixed by f_xy at a point, then the other second partials and the Hessian determinant."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a function whose Hessian determinant at P is not zero"):
        a, b = rng.choice((-3, -2, -1, 1, 2, 3)), rng.choice((-3, -2, -1, 1, 2, 3))
        c, d, e = (
            rng.choice((-4, -3, -2, -1, 1, 2, 3, 4)),
            rng.choice((-3, -2, -1, 1, 2, 3)),
            rng.choice((-3, -2, -1, 1, 2, 3)),
        )
        big_e = rng.choice((-3, -2, -1, 1, 2, 3))
        x0, y0 = rng.choice(_POINT), rng.choice(_POINT)
        if x0 + y0 == 0:
            continue
        fxy = c + 2 * big_e * (x0 + y0)
        fxx = 6 * a * x0 + 2 * d + 2 * big_e * y0
        fyy = 6 * b * y0 + 2 * e + 2 * big_e * x0
        det = fxx * fyy - fxy * fxy
        if det != 0 and fxy != 0 and 0 not in (fxx, fyy) and len({fxx, fyy, fxy, big_e}) == 4 and abs(det) < 400:
            break

    # Independent check by differentiating the displayed function.
    big = sympy.Symbol("E")
    f = a * x**3 + b * y**3 + c * x * y + d * x**2 + e * y**2 + big * (x**2 * y + x * y**2)
    at_p = {x: x0, y: y0}
    solved = sympy.solve(sympy.diff(f, x, y).subs(at_p) - fxy, big)
    check(solved == [big_e], "E from f_xy disagrees")
    hess = sympy.hessian(f.subs(big, big_e), (x, y)).subs(at_p)
    check((hess[0, 0], hess[1, 1], hess[0, 1], hess.det()) == (fxx, fyy, fxy, det), "Hessian disagrees")

    f_text = join_terms([(a, "x³"), (b, "y³"), (c, "xy"), (d, "x²"), (e, "y²")]) + " + E(x²y + xy²)"
    stem = f"Let f(x, y) = {f_text}, where E is a constant. It is known that f_xy({x0}, {y0}) = {fxy}."
    point = f"({x0}, {y0})"
    first_x = 3 * a * x0**2 + 2 * d * x0 + c * y0 + big_e * (2 * x0 * y0 + y0**2)
    first_y = 3 * b * y0**2 + 2 * e * y0 + c * x0 + big_e * (x0**2 + 2 * x0 * y0)

    a_part = frac_part(
        "Find the value of E.",
        big_e,
        2,
        [f"f_xy = {c} + 2E(x + y). At {point}: {c} + {2 * (x0 + y0)}E = {fxy}.", f"So E = {big_e}."],
        [
            ("forgot-power-multiplier", Rational(fxy - c, x0 + y0)),
            ("sign-error-moving-term", Rational(fxy + c, 2 * (x0 + y0))),
            ("uses-one-coordinate-only", Rational(fxy - c, 2 * x0)),
            ("forgot-to-divide-by-coefficient", Rational(fxy - c)),
        ],
    )
    b_part = frac_part(
        f"Hence find f_xx at {point}.",
        fxx,
        1,
        [f"f_xx = {6 * a}x + {2 * d} + 2Ey = {fxx} at {point}."],
        [
            ("ignores-unknown-term", 6 * a * x0 + 2 * d),
            ("swaps-x-and-y", 6 * a * x0 + 2 * d + 2 * big_e * x0),
            ("forgot-power-multiplier", 6 * a * x0 + 2 * d + big_e * y0),
            ("first-derivative-instead-of-second", first_x),
        ],
        follow_through=f"{6 * a * x0 + 2 * d} + 2*part_a*({y0})",
    )
    c_part = frac_part(
        f"Find f_yy at {point}.",
        fyy,
        1,
        [f"f_yy = {6 * b}y + {2 * e} + 2Ex = {fyy} at {point}."],
        [
            ("ignores-unknown-term", 6 * b * y0 + 2 * e),
            ("swaps-x-and-y", 6 * b * y0 + 2 * e + 2 * big_e * y0),
            ("forgot-power-multiplier", 6 * b * y0 + 2 * e + big_e * x0),
            ("first-derivative-instead-of-second", first_y),
        ],
        follow_through=f"{6 * b * y0 + 2 * e} + 2*part_a*({x0})",
    )
    d_part = frac_part(
        f"Hence find the determinant of the Hessian matrix of f at {point}.",
        det,
        2,
        [f"det H = f_xx f_yy − (f_xy)² = ({fxx})({fyy}) − ({fxy})² = {det}."],
        [
            ("forgot-off-diagonal-product", fxx * fyy),
            ("determinant-sign-error", fxx * fyy + fxy * fxy),
            ("forgot-to-square-mixed-partial", fxx * fyy - fxy),
            ("determinant-sign-error", -det),
        ],
        follow_through=f"part_b*part_c - ({fxy})^2",
    )
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])
