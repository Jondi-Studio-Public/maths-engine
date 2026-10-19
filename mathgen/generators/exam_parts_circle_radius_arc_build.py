"""Generates `exam_parts_circle_radius_arc` questions: a multi-part exam chain on one circle.

The stem gives a circle's area in cm² (2 decimal places). Part (a) finds the radius (a genuine square root,
correct to 1 decimal place), part (b) uses it for the circumference, and part (c) uses it again for the arc
length of a sector with a stated angle. Parts (b) and (c) carry `follow_through`, so a learner whose (a) is
wrong still earns (b) and (c) for using their own radius correctly.

## Construction: draw the radius first

The radius is drawn as a whole number of tenths that is not a whole number (so (a) is a decimal answer). Every
canonical answer is computed from that exact radius with true pi (`Fraction(math.pi)`) and rounded once, at
display. The displayed area is the only rounded input; a learner who works back from it gets the same radius
because the area is rounded to 0.005 and a tenth of radius is worth far more area than that. Each displayed
answer is kept at least `_MARGIN` away from a rounding boundary, so the exact route and the learner's route
(radius rounded to 1 decimal place, then used) always agree.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import (
    GRADER_TOLERANCE,
    VerificationError,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

PI = Fraction(math.pi)
_MARGIN = Fraction(1, 1000)  # keep every displayed value this far from a rounding boundary
_R10_LO, _R10_HI = 25, 125  # radius 2.5 .. 12.5 cm, in tenths
_ANGLES = [a for a in range(30, 340, 5) if a not in (90, 180, 270)]

_STEMS = [
    "A circular garden bed has an area of {area} cm².",
    "A circular trampoline mat has a surface area of {area} cm².",
    "A round pizza base has an area of {area} cm².",
    "The top of a circular table has an area of {area} cm².",
    "A circular rug has an area of {area} cm².",
    "A round clock face has an area of {area} cm².",
    "A circular coaster has an area of {area} cm².",
    "A circular badge has an area of {area} cm².",
]
_SECTORS = [
    "A sector with an angle of {theta}° at the centre is cut from the circle.",
    "A slice with a centre angle of {theta}° is cut from the circle.",
    "A sector of the circle has a centre angle of {theta}°.",
]


def _tenths(value: Fraction) -> Fraction:
    return round_half_up(value, 1)


def _clear_of_boundary(value: Fraction) -> bool:
    """True when `value` is at least `_MARGIN` from any 1-decimal-place rounding boundary."""
    offset = (value * 10) % 1  # boundary is at 0.5
    return abs(offset - Fraction(1, 2)) * Fraction(1, 10) > _MARGIN


def _fmt(value: Fraction) -> str:
    return stated_precision_string(_tenths(value), 1)


@generator("exam_parts_circle_radius_arc", level=11, presets={"exam": {}})
def exam_parts_circle_radius_arc(rng):
    for _ in range(1000):  # bounded rejection loop; draws exactly as `while True` did
        r10 = rng.randint(_R10_LO, _R10_HI)
        if r10 % 10 == 0:
            continue
        r = Fraction(r10, 10)
        theta = rng.choice(_ANGLES)
        circumference = 2 * PI * r
        arc = Fraction(theta, 360) * circumference
        if _clear_of_boundary(circumference) and _clear_of_boundary(arc):
            break
    else:
        raise VerificationError("no radius and angle clear of a rounding boundary in 1000 tries")

    area_true = PI * r * r
    area_display = round_half_up(area_true, 2)
    area_text = stated_precision_string(area_display, 2)

    # Independent route: work back from the DISPLAYED area, as a learner does.
    r_back = math.sqrt(float(area_display) / math.pi)
    check(abs(r_back - float(r)) < 0.05 and round(r_back, 1) == float(r), "radius does not survive the area rounding")

    r_text = stated_precision_string(r, 1)
    c_text = _fmt(circumference)
    arc_text = _fmt(arc)
    c_from_r = _tenths(2 * PI * _tenths(r))  # (b) from the rounded (a), as a learner does
    arc_from_r = _tenths(Fraction(theta, 360) * 2 * PI * _tenths(r))
    check(_fmt(c_from_r) == c_text and _fmt(arc_from_r) == arc_text, "follow-through route disagrees")
    check(abs(float(c_from_r) - 2 * math.pi * float(r)) < 0.05, "circumference check failed")
    check(abs(float(arc_from_r) - theta / 360 * 2 * math.pi * float(r)) < 0.05, "arc check failed")

    stem = f"{rng.choice(_STEMS).format(area=area_text)} {rng.choice(_SECTORS).format(theta=theta)}"

    # (a) radius: square root missing, A taken as 2 pi r, the diameter reported, or the square root of A without pi.
    a_candidates = [
        _fmt(area_true / PI),  # forgot the square root: r^2
        _fmt(area_true / (2 * PI)),  # used A = 2 pi r
        _fmt(2 * r),  # reports the diameter
        _fmt(Fraction(math.sqrt(float(area_true)))),  # square root of the area, no pi
    ]
    # (b) circumference: from the area, pi r, or the diameter.
    b_candidates = [
        _fmt(2 * PI * area_display),  # circumference-direct-from-area
        _fmt(PI * r),  # diameter-formula-on-radius
        _fmt(2 * r),  # the diameter
        _fmt(PI * r * r),  # the area again
    ]
    # (c) arc: the whole circumference, degrees as radians, the sector's area, or the angle fraction inverted.
    c_candidates = [
        c_text,  # arc-uses-full-circumference
        _fmt(Fraction(theta) * r),  # arc-angle-as-radians
        _fmt(Fraction(theta, 360) * PI * r * r),  # sector-area-for-arc
        _fmt(Fraction(360, theta) * circumference),  # arc-fraction-inverted
    ]
    a_distractors = pick_distractors(r_text, a_candidates, 3, tolerance=GRADER_TOLERANCE)
    b_distractors = pick_distractors(c_text, b_candidates, 3, tolerance=GRADER_TOLERANCE)
    c_distractors = pick_distractors(arc_text, c_candidates, 3, tolerance=GRADER_TOLERANCE)
    for text, canon, distractors in (
        (r_text, r, a_distractors),
        (c_text, circumference, b_distractors),
        (arc_text, arc, c_distractors),
    ):
        # Wrong for the canonical value AND for the follow-through value from the canonical (a) (the same here).
        check(
            distractors and all(abs(Fraction(d) - _tenths(canon)) > GRADER_TOLERANCE for d in distractors),
            "a distractor is accepted by the grader",
        )

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the radius of the circle, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(r),
                answer_text=r_text,
                answer_latex=latex_answer(r_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Area = πr², so r² = {area_text} ÷ π.",
                    f"r = √({area_text} ÷ π) = {r_text} cm (1 d.p.).",
                ],
                distractors=a_distractors,
            ),
            dict(
                prompt="Hence find the circumference of the circle, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(_tenths(circumference)),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=1,
                sensible_range=(0, None),
                follow_through="2*pi*part_a",
                working=[f"C = 2πr = 2 × π × {r_text} = {c_text} cm (1 d.p.)."],
                distractors=b_distractors,
            ),
            dict(
                prompt="Using your radius from (a), find the length of the arc of the sector, "
                "correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(_tenths(arc)),
                answer_text=arc_text,
                answer_latex=latex_answer(arc_text),
                marks=2,
                follow_through=f"part_a*2*pi*{theta}/360",
                working=[
                    f"Arc length = {theta}/360 × 2πr.",
                    f"= {theta}/360 × 2 × π × {r_text} = {arc_text} cm (1 d.p.).",
                ],
                distractors=c_distractors,
            ),
        ],
    )
