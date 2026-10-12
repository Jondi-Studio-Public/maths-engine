"""Exam-tier source for `jacobian-two-variable`: use the Jacobian of a map to scale a small area.

A polynomial map ``F(x, y) = (u, v)`` is given with a base point. The learner evaluates the Jacobian determinant,
uses it to estimate the area of the image of a small region, and then uses the Jacobian entries for a linear
estimate of ``u`` at a nearby point:

* (a) the determinant of the Jacobian at the point (2 marks);
* (b) the approximate area of the image of a small region of stated area (1 mark, follow-through from (a),
  using the absolute value);
* (c) the linear estimate of ``u`` at a nearby point (2 marks, independent).

The map's coefficients are drawn small, and a draw is rejected unless the determinant is nonzero with magnitude
at least 2 and the answers of (b) and (c) are short decimals.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    check,
    draw_until,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)

_x, _y = sympy.symbols("x y")
_AREAS = [Fraction(k, 10) for k in (2, 3, 4, 5, 6, 8, 12, 15, 25)]
_STEPS = [Fraction(k, 10) for k in (-3, -2, -1, 1, 2, 3)]


def _poly_text(expr) -> str:
    return str(sympy.expand(expr)).replace("**", "^").replace("*", "")


def _cands(*pairs) -> list[Distractor]:
    out = []
    for value, tag in pairs:
        value = Fraction(value)
        try:
            text = exact_decimal_string(value, 4)
        except ValueError:
            continue
        out.append(Distractor(text, tag, exact_value(value)))
    return out


@generator("exd_x_sc5_jacobian_area", level=18, presets={"exam": dict(tier="exam")})
def exd_x_sc5_jacobian_area(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: Jacobian determinant, area scale factor, then a linear estimate from the Jacobian."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r: random.Random):
        a, b, c, d = (r.randint(-3, 3) for _ in range(4))
        e, f, g, k = (r.randint(-3, 3) for _ in range(4))
        x0, y0 = r.randint(-3, 3), r.randint(-3, 3)
        return a, b, c, d, e, f, g, k, x0, y0

    def acceptable(v) -> bool:
        a, b, c, d, e, f, g, k, x0, y0 = v
        if a == 0 or b == 0 or c == 0 or e == 0 or f == 0 or g == 0:
            return False
        u = a * _x**2 + b * _x * _y + c * _y + d
        w = e * _x + f * _y**2 + g * _x * _y + k
        jac = sympy.Matrix([[sympy.diff(u, _x), sympy.diff(u, _y)], [sympy.diff(w, _x), sympy.diff(w, _y)]])
        det = jac.det().subs({_x: x0, _y: y0})
        origin_det = jac.det().subs({_x: 0, _y: 0})
        u0 = int(u.subs({_x: x0, _y: y0}))
        return (
            2 <= abs(det) <= 60
            and det != origin_det
            and (x0, y0) != (0, 0)
            and abs(u0) >= 2
            and int(jac[0, 0].subs({_x: x0, _y: y0})) != 0
            and int(jac[0, 1].subs({_x: x0, _y: y0})) != 0
        )

    a, b, c, d, e, f, g, k, x0, y0 = draw_until(rng, draw, acceptable, what="a map with a usable Jacobian")
    u = a * _x**2 + b * _x * _y + c * _y + d
    w = e * _x + f * _y**2 + g * _x * _y + k
    jac = sympy.Matrix([[sympy.diff(u, _x), sympy.diff(u, _y)], [sympy.diff(w, _x), sympy.diff(w, _y)]])
    at = {_x: x0, _y: y0}
    j11, j12, j21, j22 = (int(jac[i, j].subs(at)) for i in (0, 1) for j in (0, 1))
    det = j11 * j22 - j12 * j21
    check(det == int(jac.det().subs(at)), "determinant disagrees with sympy")

    area = rng.choice(_AREAS)
    image_area = abs(det) * area
    u0 = int(u.subs(at))

    def draw_steps(r):
        return r.choice(_STEPS), r.choice(_STEPS)

    def usable(steps) -> bool:
        step_x, step_y = steps
        return (
            abs(j11 * step_x) >= Fraction(1, 10)
            and abs(j12 * step_y) >= Fraction(1, 10)
            and abs(u0 + j11 * step_x + j12 * step_y) >= 1
        )

    dx, dy = draw_until(rng, draw_steps, usable, what="increments that keep the estimate distinct from its slips")
    estimate = u0 + j11 * dx + j12 * dy
    area_text = exact_decimal_string(area, 3)
    dx_text, dy_text = exact_decimal_string(dx, 3), exact_decimal_string(dy, 3)
    new_x = exact_decimal_string(x0 + dx, 3)
    new_y = exact_decimal_string(y0 + dy, 3)

    origin_j = {(i, j): int(jac[i, j].subs({_x: 0, _y: 0})) for i in (0, 1) for j in (0, 1)}
    origin_det = origin_j[(0, 0)] * origin_j[(1, 1)] - origin_j[(0, 1)] * origin_j[(1, 0)]
    a_cands = _cands(
        (j11 * j22 + j12 * j21, "determinant-added-cross-terms"),
        (origin_det, "substituted-origin-for-point"),
        (j11 * j22, "determinant-formula-misremembered"),
        (-det, "determinant-formula-misremembered"),
        (j12 * j21, "determinant-formula-misremembered"),
        (j11 + j22, "determinant-formula-misremembered"),
    )
    b_cands = _cands(
        (det * area, "forgot-to-take-absolute-value"),
        (area, "forgot-area-scale-factor"),
        (Fraction(area, abs(det)), "divided-by-scale-factor"),
        (abs(det), "forgot-area-scale-factor"),
        (abs(det) + area, "added-instead-of-multiplied"),
    )
    c_cands = _cands(
        (j11 * dx + j12 * dy, "left-out-the-base-value"),
        (u0 + j11 * dx, "dropped-one-variable-term"),
        (u0 + j21 * dx + j22 * dy, "used-wrong-component-row"),
        (u0 + j11 + j12, "forgot-the-increments"),
    )
    a_text = str(det)
    b_text = stated_precision_string(image_area, 1)
    c_text = stated_precision_string(estimate, 1)
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(c_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors survive")

    stem = (
        f"A map is given by F(x, y) = (u, v), where u = {_poly_text(u)} and v = {_poly_text(w)}. "
        f"Its Jacobian is evaluated at the point ({x0}, {y0})."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the determinant of the Jacobian matrix of F at ({x0}, {y0}).",
                answer_kind="number",
                answer=det,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                working=[
                    f"J = [[{j11}, {j12}], [{j21}, {j22}]] at ({x0}, {y0}).",
                    f"det J = {j11} x {j22} - {j12} x {j21} = {det}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"A small region near ({x0}, {y0}) has an area of {area_text} square units. "
                "Use your determinant to estimate the area of its image under F, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(image_area),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"Abs(part_a)*{sympy.Rational(area.numerator, area.denominator)}",
                working=[f"Image area = |det J| x {area_text} = {abs(det)} x {area_text} = {b_text}."],
                distractors=b_d,
            ),
            dict(
                prompt=f"Use the Jacobian to estimate u at ({new_x}, {new_y}), starting from the value of u at ({x0}, {y0}). Give your answer correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(estimate),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                working=[
                    f"u({x0}, {y0}) = {u0}.",
                    f"u is about {u0} + ({j11})({dx_text}) + ({j12})({dy_text}) = {c_text}.",
                ],
                distractors=c_d,
            ),
        ],
    )
