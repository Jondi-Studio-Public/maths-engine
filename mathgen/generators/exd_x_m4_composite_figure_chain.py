"""Exam tier for "find the area of irregular figures by decomposition": a rectangle plus a triangle, worked backwards.

The practice tiers give every dimension and ask for the area. Here the total area is given and a dimension is
missing, so the learner decomposes the figure into a rectangle and a right-angled triangle, forms an equation in the
unknown, and solves it. Part (a) finds the unknown length of the triangle, part (b) the slanted side that follows
from it (Pythagoras), and part (c) the perimeter of the whole figure. Parts (b) and (c) carry `follow_through`.

## Construction: draw the dimensions first

The rectangle's width and height are whole numbers of metres and the triangle's unknown leg is a whole number of
tenths, so the area shown (rectangle plus half of height times leg) is exact to 2 decimal places and the unknown is
recovered exactly. The slanted side is rounded to 1 decimal place and kept clear of a rounding boundary; the
perimeter is defined from that rounded slanted side, as the question says ("using your answer to (b)").
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_STEMS = (
    "The floor plan of a community hall is a rectangle {w} m long and {h} m wide, with a right-angled triangular stage "
    "added along one {h} m end. The stage's two shorter sides are that {h} m end and a side of x m that extends the "
    "hall's bottom wall. The total floor area is {a} m².",
    "A garden bed is made of a rectangle {w} m by {h} m together with a right-angled triangular corner added along one "
    "{h} m side. The triangle's perpendicular sides are that {h} m side and a side of x m that continues the "
    "rectangle's bottom edge. The whole bed has an area of {a} m².",
    "A house block is a rectangle {w} m by {h} m with a right-angled triangular strip attached along one {h} m "
    "boundary. The strip's perpendicular sides are that {h} m boundary and a side of x m lying along the extension of "
    "the rectangle's bottom boundary. The block's total area is {a} m².",
)


def _clear(value: float) -> bool:
    scaled = value * 10
    return abs(scaled - math.floor(scaled) - 0.5) > 0.05


@generator("exd_x_m4_composite_figure_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m4_composite_figure_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Unknown leg from the total area, the slanted side, then the perimeter. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "dimensions with a clear slanted side"):
        w, h = rng.randint(6, 30), rng.randint(4, 20)
        x = Fraction(rng.randint(25, 150), 10)
        if x.denominator == 1:
            continue
        area = w * h + h * x / 2
        hyp_exact = math.hypot(h, float(x))
        if not _clear(hyp_exact):
            continue
        hyp = round_half_up(Fraction(hyp_exact).limit_denominator(10**9), 1)
        perimeter = 2 * w + h + x + hyp
        x_text = exact_decimal_string(x, 1)
        hyp_text = stated_precision_string(hyp, 1)
        p_text = stated_precision_string(perimeter, 1)
        a_text = exact_decimal_string(area, 2)
        stem = rng.choice(_STEMS).format(w=w, h=h, a=a_text)
        if all(t not in stem.replace("m²", "").split() for t in (x_text, hyp_text, p_text)):
            break

    def one(v) -> Distractor | None:
        shown = round_half_up(Fraction(v).limit_denominator(10**9), 1)
        if shown <= 0:
            return None
        return Distractor(stated_precision_string(shown, 1), None, shown)

    def keep(options):
        return [o for o in options if o is not None]

    h_f = Fraction(h)
    a_d = pick_distractors(
        x_text,
        [
            Distractor(exact_decimal_string(round_half_up(v, 2), 2), None, round_half_up(v, 2))
            for v in (x / 2, 2 * area / h, 2 * (area - w * h) / w)
            if v > 0
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    b_d = pick_distractors(
        hyp_text,
        keep([one(h_f + x), one(h_f**2 + x**2), one(math.hypot(w, float(x))), one(abs(h_f - x))]),
        3,
        tolerance=GRADER_TOLERANCE,
    )
    c_d = pick_distractors(
        p_text,
        keep([one(w + h + x + hyp), one(2 * w + 2 * h + x + hyp), one(2 * w + h + x), one(w * h + h_f * x / 2)]),
        3,
        tolerance=GRADER_TOLERANCE,
    )
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "not enough distractors")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of x.",
                answer_kind="number",
                answer=numeric_answer(x),
                answer_text=x_text,
                answer_latex=latex_answer(x_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Area = rectangle + triangle = {w} x {h} + 1/2 x {h} x x = {a_text}.",
                    f"{h}x/2 = {a_text} - {w * h}, so x = {x_text}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Hence find the length of the slanted side, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(hyp),
                answer_text=hyp_text,
                answer_latex=latex_answer(hyp_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"sqrt({h}**2+part_a**2)",
                working=[f"Slanted side = √({h}² + {x_text}²) = {hyp_text} m (1 d.p.)."],
                distractors=b_d,
            ),
            dict(
                prompt="Using your answer to (b), find the perimeter of the whole figure, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(perimeter),
                answer_text=p_text,
                answer_latex=latex_answer(p_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{2 * w + h}+part_a+part_b",
                working=[
                    f"Perimeter = {w} + {w} + {x_text} + {h} + {hyp_text}.",
                    f"= {p_text} m (1 d.p.).",
                ],
                distractors=c_d,
            ),
        ],
    )
