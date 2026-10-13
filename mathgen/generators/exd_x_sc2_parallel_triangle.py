"""Exam-tier source for `f10-parallel-line-angle-theorems`: algebraic angles on parallel lines, then the third angle.

Two horizontal parallel lines m (above) and n (below). P is on m; Q and R are on n, Q left of R, so PQR is a
triangle. S and T are on m, S left of P and T right of P. Two of the angles at the transversal PQ are given as
algebraic expressions that are related by a parallel-line theorem (alternate angles are equal, or co-interior
angles sum to 180 degrees). Three connected parts:

(a) solve for x;
(b) the size of angle PQR (follow-through from (a));
(c) the size of angle QPR, using a third angle at P that is alternate to angle PRQ and the straight line SPT
    (follow-through from (a), through angle PQR).

Built backwards from the whole-number x. Angle SPQ and angle PQR are alternate (equal); angle TPQ and angle PQR are
co-interior (sum 180 degrees); angle TPR and angle PRQ are alternate.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Eq, Symbol, solve

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, pick_distractors

_X = Symbol("x")


def _expr(coef: int, const: int) -> str:
    """A linear expression such as '3x + 10' or '5x − 20' (coefficients that are never 0 or 1 for x)."""
    sign = "+" if const >= 0 else "−"
    return f"{coef}x {sign} {abs(const)}"


@generator("exd_x_sc2_parallel_triangle", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc2_parallel_triangle(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: solve for x from a parallel-line pair, find angle PQR, then angle QPR."""
    check(tier == "exam", f"unknown tier {tier!r}")
    relationship = rng.choice(("alternate", "co-interior"))
    for _ in attempts(5000, "angles that fit a triangle on parallel lines"):
        x = rng.randint(8, 30)
        a, c = rng.randint(2, 6), rng.randint(2, 6)
        if a == c:
            continue
        b = rng.choice((-1, 1)) * rng.randint(2, 30)
        if relationship == "alternate":
            d = a * x + b - c * x  # a x + b = c x + d
            first_angle = a * x + b  # angle SPQ
        else:
            d = 180 - (a + c) * x - b  # (a x + b) + (c x + d) = 180
            first_angle = a * x + b  # angle TPQ
        alpha = c * x + d  # angle PQR
        e = rng.randint(30, 85)  # angle TPR (alternate to angle PRQ)
        if not (30 <= alpha <= 110 and abs(d) <= 60 and d != 0 and 30 <= first_angle <= 150):
            continue
        if alpha + e >= 165 or alpha == e:
            continue
        break

    # Independent route: solve the stated equation with sympy.
    if relationship == "alternate":
        equation = Eq(a * _X + b, c * _X + d)
    else:
        equation = Eq(a * _X + b + c * _X + d, 180)
    solved = solve(equation, _X)
    check(solved == [x], "equation does not have the intended solution")
    third = 180 - alpha - e
    check(third > 0 and alpha == c * x + d, "triangle angles disagree")
    check(Fraction(alpha + e + third) == 180, "angles on the straight line do not sum to 180")

    if relationship == "alternate":
        pair = f"Angle SPQ = ({_expr(a, b)})° and angle PQR = ({_expr(c, d)})°. These are alternate angles."
        wrong_equation_x = solve(Eq(a * _X + b + c * _X + d, 180), _X)
    else:
        pair = f"Angle TPQ = ({_expr(a, b)})° and angle PQR = ({_expr(c, d)})°. These are co-interior angles."
        wrong_equation_x = solve(Eq(a * _X + b, c * _X + d), _X)
    stem = (
        "Lines m and n are parallel, with m above n. P is a point on m, and Q and R are points on n with Q to the "
        "left of R, so that PQR is a triangle. S and T are points on m with S to the left of P and T to the right of "
        f"P. {pair} Angle TPR = {e}°, and it is alternate to angle PRQ."
    )

    # Distractors for each part, each one named error.
    x_cands = []
    if len(wrong_equation_x) == 1 and wrong_equation_x[0].is_integer:
        wx = int(wrong_equation_x[0])
        tag = "applies-equal-to-co-interior" if relationship == "co-interior" else "applies-supplementary-to-equal-type"
        x_cands.append(Distractor(str(wx), tag, Fraction(wx)))
    x_cands.append(Distractor(str(x + 1), None, Fraction(x + 1)))
    x_cands.append(Distractor(str(x - 1), None, Fraction(x - 1)))
    x_cands.append(Distractor(str(x + 2), None, Fraction(x + 2)))
    alpha_cands = [
        Distractor(str(x), "reports-x-not-angle", Fraction(x)),
        Distractor(str(first_angle), "reports-the-given-partner-angle", Fraction(first_angle)),
        Distractor(str(180 - alpha), "applies-supplementary-to-equal-type", Fraction(180 - alpha)),
        Distractor(str(alpha + 10), None, Fraction(alpha + 10)),
        Distractor(str(alpha - 10), None, Fraction(alpha - 10)),
    ]
    third_cands = [
        Distractor(str(180 - alpha), "ignored-the-given-angle", Fraction(180 - alpha)),
        Distractor(str(180 - e), "ignored-the-found-angle", Fraction(180 - e)),
        Distractor(str(alpha + e), "forgot-to-subtract-from-180", Fraction(alpha + e)),
        Distractor(str(360 - alpha - e), None, Fraction(360 - alpha - e)),
    ]
    x_dis = pick_distractors(str(x), x_cands, 3)
    alpha_dis = pick_distractors(str(alpha), alpha_cands, 3)
    third_dis = pick_distractors(str(third), third_cands, 3)
    check(len(x_dis) >= 2 and len(alpha_dis) >= 2 and len(third_dis) >= 2, "too few distractors")

    rule = (
        f"Alternate angles are equal, so {_expr(a, b)} = {_expr(c, d)}."
        if relationship == "alternate"
        else f"Co-interior angles sum to 180°, so ({_expr(a, b)}) + ({_expr(c, d)}) = 180."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of x.",
                answer_kind="number",
                answer=x,
                answer_text=str(x),
                answer_latex=latex_answer(str(x)),
                marks=2,
                sensible_range=(0, None),
                working=[rule, f"Solving gives x = {x}."],
                distractors=x_dis,
            ),
            dict(
                prompt="Hence find the size of angle PQR, in degrees.",
                answer_kind="number",
                answer=alpha,
                answer_text=str(alpha),
                answer_latex=latex_answer(str(alpha)),
                marks=1,
                sensible_range=(0, 180),
                follow_through=f"{c}*part_a + ({d})",
                working=[f"Angle PQR = {_expr(c, d)} = {c}({x}) + ({d}) = {alpha}°."],
                distractors=alpha_dis,
            ),
            dict(
                prompt="Find the size of angle QPR, in degrees.",
                answer_kind="number",
                answer=third,
                answer_text=str(third),
                answer_latex=latex_answer(str(third)),
                marks=2,
                sensible_range=(0, 180),
                follow_through=f"180 - ({c}*part_a + ({d})) - {e}",
                working=[
                    f"Angle SPQ, on the left of PQ at P, is alternate to angle PQR, so it is {alpha}°. Angle TPR = {e}°.",
                    f"Angles on the straight line SPT sum to 180°: angle QPR = 180° − {alpha}° − {e}° = {third}°.",
                ],
                distractors=third_dis,
            ),
        ],
    )
