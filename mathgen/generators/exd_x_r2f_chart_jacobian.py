"""Exam source for "compute a local chart Jacobian": a four-part chain through the inverse function theorem.

A chart phi(x, y) = (u, v) with u = p x^2 + q y + e1 and v = r x + s y^2 + e2 is given, and a point P(x0, y0).

(a) the determinant of the Jacobian as an expression in x and y      4ps xy - qr         expression, 2 marks
(b) its value at P                                                    Subs(part_a, ...)   fraction,   1 mark
(c) d x / d u for the local inverse at P: the (1,1) entry of J^-1     2 s y0 / det        fraction,   2 marks
(d) the linear estimate of x at the point phi maps to (u(P)+h, v(P))   x0 + h * part_c      fraction,   2 marks

Parts (c) and (d) are where the exam departs from the practice tiers: the learner must know that the inverse's
derivative is the inverse matrix, and use it as a linear approximation. P is chosen so the determinant at P is not zero.
"""

from __future__ import annotations

import random

import sympy
from sympy import Rational, symbols

from ..core import attempts, check, generator, join_terms, rational_text
from .exd_x_r2f_common import expr_part, frac_part

x, y = symbols("x y")
_COEFFS = (-4, -3, -2, -1, 1, 2, 3, 4)
_SQUARE_COEFFS = (-3, -2, -1, 1, 2, 3)
_POINTS = (-3, -2, -1, 1, 2, 3)
_STEPS = ("1/10", "1/5", "3/10", "1/2")


def _decimal(step: str) -> str:
    return {"1/10": "0.1", "1/5": "0.2", "3/10": "0.3", "1/2": "0.5"}[step]


@generator(
    "exd_x_r2f_chart_jacobian",
    level=18,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_r2f_chart_jacobian(rng: random.Random, tier: str = "exam") -> dict:
    """Jacobian determinant, its value at P, the inverse's derivative, and a linear estimate of the preimage."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a chart that is locally invertible at P"):
        p, s = rng.choice(_SQUARE_COEFFS), rng.choice(_SQUARE_COEFFS)
        q, r = rng.choice(_COEFFS), rng.choice(_COEFFS)
        e1, e2 = rng.randint(-6, 6), rng.randint(-6, 6)
        x0, y0 = rng.choice(_POINTS), rng.choice(_POINTS)
        det_p = 4 * p * s * x0 * y0 - q * r
        if det_p != 0 and abs(det_p) <= 120 and 2 * s * y0 != det_p:
            break
    step = rng.choice(_STEPS)
    h = Rational(step)

    u = p * x**2 + q * y + e1
    v = r * x + s * y**2 + e2
    jac = sympy.Matrix([[sympy.diff(u, x), sympy.diff(u, y)], [sympy.diff(v, x), sympy.diff(v, y)]])
    det_expr = sympy.expand(jac.det())
    check(sympy.expand(det_expr - (4 * p * s * x * y - q * r)) == 0, "determinant disagrees with the closed form")
    det_val = Rational(det_expr.subs({x: x0, y: y0}))
    check(det_val == det_p, "determinant at P disagrees")
    inverse = jac.subs({x: x0, y: y0}).inv()
    dxdu = Rational(inverse[0, 0])
    check(dxdu == Rational(2 * s * y0, det_p), "inverse entry disagrees")
    estimate = x0 + h * dxdu

    u_text = join_terms([(p, "x²"), (q, "y"), (e1, "")])
    v_text = join_terms([(r, "x"), (s, "y²"), (e2, "")])
    stem = (
        f"A chart φ(x, y) = (u, v) is defined by u = {u_text} and v = {v_text}. Let J be its Jacobian matrix "
        f"∂(u, v)/∂(x, y), and let P be the point ({x0}, {y0})."
    )

    d11, d22 = 2 * p * x0, 2 * s * y0  # u_x and v_y at P
    a_part = expr_part(
        "Find det J as an expression in x and y.",
        det_expr,
        2,
        [
            f"J = [[{2 * p}x, {q}], [{r}, {2 * s}y]].",
            f"det J = ({2 * p}x)({2 * s}y) - ({q})({r}) = {sympy.sstr(det_expr)}.",
        ],
        [
            ("forgot-power-multiplier", p * s * x * y - q * r),
            ("determinant-sign-error", 4 * p * s * x * y + q * r),
            ("forgot-off-diagonal-product", 4 * p * s * x * y),
            ("determinant-sign-error", q * r - 4 * p * s * x * y),
        ],
    )
    b_part = frac_part(
        f"Hence find the value of det J at P({x0}, {y0}).",
        det_val,
        1,
        [f"det J = 4({p})({s})({x0})({y0}) - ({q})({r}) = {rational_text(det_val)}."],
        [
            ("origin-substitution", -q * r),
            ("forgot-power-multiplier", p * s * x0 * y0 - q * r),
            ("determinant-sign-error", 4 * p * s * x0 * y0 + q * r),
            ("forgot-off-diagonal-product", 4 * p * s * x0 * y0),
        ],
        follow_through=f"Subs(part_a, (x, y), ({x0}, {y0}))",
    )
    c_part = frac_part(
        f"Since det J ≠ 0 at P, φ has a local inverse near P. Find ∂x/∂u for this local inverse at the point φ(P).",
        dxdu,
        2,
        [
            "The derivative matrix of the inverse is J⁻¹, and ∂x/∂u is its first-row, first-column entry.",
            f"J⁻¹ = (1/det J)[[{d22}, {-q}], [{-r}, {d11}]], so ∂x/∂u = {d22}/{rational_text(det_val)} = {rational_text(dxdu)}.",
        ],
        [
            ("reciprocal-of-partial-derivative", Rational(1, d11)),
            ("inverse-entries-not-swapped", Rational(d11, det_p)),
            ("forgot-to-divide-by-determinant", Rational(d22)),
            ("sign-error-inverse-entry", Rational(-d22, det_p)),
        ],
        follow_through=f"{d22}/part_b",
    )
    d_part = frac_part(
        f"Use the linear approximation to the local inverse to estimate the x-coordinate of the point near P that φ maps "
        f"to (u₀ + {_decimal(step)}, v₀), where (u₀, v₀) = φ(P). Give an exact fraction.",
        estimate,
        2,
        [
            f"Δx ≈ (∂x/∂u)Δu = {rational_text(dxdu)} × {_decimal(step)} = {rational_text(h * dxdu)}.",
            f"x ≈ {x0} + {rational_text(h * dxdu)} = {rational_text(estimate)}.",
        ],
        [
            ("linear-approximation-sign-error", x0 - h * dxdu),
            ("drops-base-point-in-linear-approximation", h * dxdu),
            ("increment-not-scaled-by-derivative", x0 + h),
            ("reciprocal-of-partial-derivative", x0 + h * Rational(1, d11)),
        ],
        follow_through=f"{x0} + ({step})*part_c",
    )
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])
