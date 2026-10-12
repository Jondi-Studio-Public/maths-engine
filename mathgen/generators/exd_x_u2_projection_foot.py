"""Exam tier for vector projection: the point on a line that is closest to a given point.

The practice tiers project one vector onto another, or onto the direction of a line. Here the line does not
pass through the origin and the learner is asked for the closest point on it to Q, which is a projection of
Q - P onto the direction followed by adding P back on. It is set as the short chain an exam uses: find the
parameter t of the closest point F = P + t d, then each coordinate of F. Both coordinate parts carry
`follow_through`, so a wrong t costs only part (a).

## Construction: draw the foot first

Pick the point P, a direction d and a whole-number parameter t, so the foot is F = P + t d. Q is F moved off
the line along a normal to d: Q = F + m n with n = (-d_y, d_x). Then (Q - P) . d = t |d|^2, so t comes out as
a whole number, and the two other candidate parameters a learner might reach (using Q, and using the normal
direction) are different values, which SymPy-free exact fractions confirm.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    pick_distractors,
    rounded_decimal_string,
)

_COMPONENTS = (-4, -3, -2, -1, 1, 2, 3, 4)
_PARAMETERS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
_OFFSETS = (-3, -2, -1, 1, 2, 3)


def _text(value: Fraction) -> str:
    return rounded_decimal_string(value, 2)


def _tagged(candidates: list[tuple[str, Fraction]], key: int) -> list:
    out = [Distractor(_text(v), ident, v) for ident, v in candidates]
    return pick_distractors(str(key), out, 3, tolerance=GRADER_TOLERANCE)


def _clashes(prompt: str, keys: list[int]) -> bool:
    printed = {int(m) for m in re.findall(r"\d+", prompt)}
    return any(abs(key) in printed for key in keys)


@generator("exd_x_u2_projection_foot", level=12, presets={"exam": dict(tier="exam")})
def exd_x_u2_projection_foot(rng: random.Random, tier: str = "exam") -> dict:
    """Closest point on an offset line, as parameter then coordinates. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a line and point with a clean closest point"):
        px, py = rng.randint(-6, 6), rng.randint(-6, 6)
        dx, dy = rng.choice(_COMPONENTS), rng.choice(_COMPONENTS)
        t = rng.choice(_PARAMETERS)
        m = rng.choice(_OFFSETS)
        fx, fy = px + t * dx, py + t * dy
        qx, qy = fx - m * dy, fy + m * dx
        norm_sq = dx * dx + dy * dy

        # Second route: the parameter from the printed points, by the dot-product formula.
        dot = (qx - px) * dx + (qy - py) * dy
        check(Fraction(dot, norm_sq) == t, "parameter does not follow from the printed points")
        check((qx - fx) * dx + (qy - fy) * dy == 0, "Q - F is not perpendicular to the line")
        if len({t, fx, fy}) < 3:
            continue

        style = rng.choice(["points", "direction", "vector"])
        if style == "points":
            line = f"A line passes through A({px}, {py}) and B({px + dx}, {py + dy})."
            param = "Every point of the line can be written as F = A + t(B - A)."
            named = "A"
        elif style == "direction":
            line = f"A line passes through P({px}, {py}) and has direction vector d = ({dx}, {dy})."
            param = "Every point of the line can be written as F = P + td."
            named = "P"
        else:
            line = f"A line has vector equation r = ({px}, {py}) + t({dx}, {dy})."
            param = "The point of the line with parameter t is F."
            named = "r(0)"
        off = rng.choice(["The point Q({x}, {y}) is not on the line.", "The point Q({x}, {y}) lies off the line."])
        stem = f"{line} {param} {off.format(x=qx, y=qy)}"
        a_prompt = "Find the value of t for which F is the point of the line closest to Q."
        b_prompt = "Using your value of t, find the x-coordinate of this closest point."
        c_prompt = "Using your value of t, find the y-coordinate of this closest point."
        if _clashes(f"{stem} {a_prompt} {b_prompt} {c_prompt}", [t, fx, fy]):
            continue

        origin = Fraction(qx * dx + qy * dy, norm_sq)  # projected as if the line passed through the origin
        a_distractors = _tagged(
            [
                ("projected-through-origin", origin),
                ("projected-onto-normal", Fraction(m)),
                ("forgot-to-divide-by-length-squared", Fraction(dot)),
            ],
            t,
        )
        b_distractors = _tagged(
            [
                ("forgot-to-add-back-point", Fraction(t * dx)),
                ("projected-through-origin", origin * dx),
                ("projected-onto-normal", Fraction(px - m * dy)),
            ],
            fx,
        )
        c_distractors = _tagged(
            [
                ("forgot-to-add-back-point", Fraction(t * dy)),
                ("projected-through-origin", origin * dy),
                ("projected-onto-normal", Fraction(py + m * dx)),
            ],
            fy,
        )
        if min(len(a_distractors), len(b_distractors), len(c_distractors)) < 2:
            continue
        return dict(
            text=stem,
            parts=[
                dict(
                    prompt=a_prompt,
                    answer_kind="number",
                    answer=t,
                    answer_text=str(t),
                    answer_latex=latex_answer(str(t)),
                    marks=3,
                    working=[
                        f"The closest point F satisfies (Q - {named}) · d = 0 after moving along the line: "
                        f"t = ((Q - {named}) · d) / (d · d).",
                        f"t = ({qx - px}×{dx} + {qy - py}×{dy}) / {norm_sq} = {t}.",
                    ],
                    distractors=a_distractors,
                ),
                dict(
                    prompt=b_prompt,
                    answer_kind="number",
                    answer=fx,
                    answer_text=str(fx),
                    answer_latex=latex_answer(str(fx)),
                    marks=1,
                    follow_through=f"{px} + {dx}*part_a",
                    working=[f"x = {px} + {t}×{dx} = {fx}."],
                    distractors=b_distractors,
                ),
                dict(
                    prompt=c_prompt,
                    answer_kind="number",
                    answer=fy,
                    answer_text=str(fy),
                    answer_latex=latex_answer(str(fy)),
                    marks=1,
                    follow_through=f"{py} + {dy}*part_a",
                    working=[f"y = {py} + {t}×{dy} = {fy}."],
                    distractors=c_distractors,
                ),
            ],
        )
