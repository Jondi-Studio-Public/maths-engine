"""Exam-tier source for `f10-enlargement-invariants`: find the scale factor from two areas, then use it twice.

A rectangle is transformed by an enlargement from a centre O. The stem gives the rectangle's dimensions and the
area of its image. Three connected parts:

(a) the scale factor, the square root of the ratio of the two areas, as a decimal (the factor is below 1 for a
    reduction in about a quarter of the questions);
(b) the perimeter of the image (follow-through from (a));
(c) the distance between a point P on the original and its image P', both on the same ray from O (follow-through
    from (a)), which is |k - 1| times the distance OP.

Built backwards from a scale factor that is a whole number of tenths, so the image area is exact to 2 decimal
places and every key is exact to 1 decimal place.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)

_OBJECTS = (
    "photograph",
    "poster",
    "garden bed plan",
    "school banner",
    "floor tile design",
    "sign",
)


def _tenths(value: Fraction) -> str:
    return stated_precision_string(value, 1)


def _dis(value: Fraction, tag: str | None) -> Distractor:
    """A distractor shown to 1 decimal place; it is built only from values that are exact tenths."""
    return Distractor(_tenths(value), tag, exact_value(value))


@generator("exd_x_sc2_enlargement_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc2_enlargement_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: scale factor from two areas, perimeter of the image, and the gap between a point and its image."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a scale factor in tenths that is not a whole number"):
        if rng.random() < 0.25:
            t = rng.randint(2, 9)  # a reduction, 0.2 .. 0.9
        else:
            t = rng.randint(12, 60)  # an enlargement, 1.2 .. 6.0
            if t % 10 == 0:
                continue
        length, width = rng.randint(3, 14), rng.randint(2, 12)
        if length <= width:
            continue
        dist = rng.randint(3, 20)
        k = Fraction(t, 10)
        break

    area0 = length * width
    area1 = k * k * area0  # exact: hundredths of a square centimetre
    check((area1 * 100).denominator == 1, "image area is not exact to 2 decimal places")
    # Independent route: recover k from the displayed areas.
    ratio = area1 / area0
    rn, rd = math.isqrt(ratio.numerator), math.isqrt(ratio.denominator)
    check(rn * rn == ratio.numerator and rd * rd == ratio.denominator, "area ratio is not a perfect square")
    check(Fraction(rn, rd) == k, "scale factor not recovered from the areas")
    perimeter0 = 2 * (length + width)
    perimeter1 = k * perimeter0
    gap = abs(k - 1) * dist
    for value in (perimeter1, gap):
        check((value * 10).denominator == 1, "a key is not exact to 1 decimal place")

    obj = rng.choice(_OBJECTS)
    area_text = stated_precision_string(area1, 2)
    stem = (
        f"A rectangular {obj} measuring {length} cm by {width} cm is transformed by an enlargement with centre O. "
        f"The image has an area of {area_text} cm². P is a point on the original, {dist} cm from O, and P' is "
        "its image on the same ray from O."
    )
    k_text, per_text, gap_text = _tenths(k), _tenths(perimeter1), _tenths(gap)

    a_cands = [
        Distractor(exact_decimal_string(ratio, 4), "uses-area-ratio-as-scale-factor", exact_value(ratio)),
        Distractor(exact_decimal_string(ratio / 2, 4), "halves-the-area-ratio", exact_value(ratio / 2)),
        _dis(k + Fraction(1, 10), None),
    ]
    b_cands = [
        _dis(k * k * perimeter0, "perimeter-scaled-by-k-squared")
        if (k * k * perimeter0 * 10).denominator == 1
        else None,
        _dis(perimeter0 + k, "adds-instead-of-multiplies"),
        _dis(Fraction(perimeter0), "ignored-the-enlargement"),
        _dis(area1, "finds-area-instead-of-perimeter") if (area1 * 10).denominator == 1 else None,
    ]
    c_cands = [
        _dis(k * dist, "gives-distance-from-centre-not-gap"),
        _dis((k + 1) * dist, "puts-image-on-opposite-side-of-centre"),
        _dis(abs(k - 1) * dist + 1, None),
        _dis(dist + k, "adds-instead-of-multiplies"),
    ]
    a_dis = pick_distractors(k_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_dis = pick_distractors(per_text, [c for c in b_cands if c], 3, tolerance=GRADER_TOLERANCE)
    c_dis = pick_distractors(gap_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(len(a_dis) >= 1 and len(b_dis) >= 2 and len(c_dis) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the scale factor of the enlargement, as a decimal.",
                answer_kind="number",
                answer=numeric_answer(k),
                answer_text=k_text,
                answer_latex=latex_answer(k_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Original area = {length} x {width} = {area0} cm².",
                    f"Areas scale by k²: k² = {area_text} / {area0} = {exact_decimal_string(ratio, 4)}.",
                    f"k = {k_text}.",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt="Hence find the perimeter of the image, in cm, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(perimeter1),
                answer_text=per_text,
                answer_latex=latex_answer(per_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a*{perimeter0}",
                working=[
                    f"Perimeter of the original = 2({length} + {width}) = {perimeter0} cm.",
                    f"Image: {k_text} x {perimeter0} = {per_text} cm.",
                ],
                distractors=b_dis,
            ),
            dict(
                prompt="Find the distance PP', in cm, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(gap),
                answer_text=gap_text,
                answer_latex=latex_answer(gap_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"Abs(part_a - 1)*{dist}",
                working=[
                    f"OP' = {k_text} x {dist} = {_tenths(k * dist)} cm.",
                    f"PP' = |{_tenths(k * dist)} - {dist}| = {gap_text} cm.",
                ],
                distractors=c_dis,
            ),
        ],
    )
