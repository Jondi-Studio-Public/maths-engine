"""Exam tier for "convert between metric units of area and other area units": a land-measurement chain.

A rectangular block of land has one side given in kilometres and the other in metres, so the learner has to bring
the two onto one unit before multiplying. Part (a) finds the area in hectares (the conversion is the whole point:
square kilometres, square metres and hectares are three different sizes). Part (b) uses it to cost treatment at a
price per hectare, and part (c) finds what percentage of a farm whose area is given in square kilometres the block
covers, which needs hectares and square kilometres brought together. Parts (b) and (c) carry `follow_through`.

## Construction: draw the dimensions first

The long side is a whole number of tenths of a kilometre and the short side a multiple of 50 metres, so the area in
hectares is an exact multiple of 0.5 and every canonical answer for (a) and (b) is exact. The percentage in (c) is
rounded once, at display, and kept clear of a rounding boundary.
"""

from __future__ import annotations

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
    "A rectangular block of paddock land measures {l} km by {w} m. It is part of a farm with a total area of {s} km².",
    "A rectangular vineyard block is {l} km long and {w} m wide, on a property of total area {s} km².",
    "A block of land for a solar-panel field is a rectangle {l} km by {w} m, on a station covering {s} km² altogether.",
    "A rectangular block of conservation reserve measures {w} m by {l} km. It lies inside a park of {s} km² in all.",
)
_TREATMENTS = (
    "Fertiliser for the whole block costs ${p} per hectare.",
    "Weed control across the whole block costs ${p} per hectare.",
    "Revegetation work across the whole block costs ${p} per hectare.",
)


def _hundredths(value: Fraction) -> bool:
    return abs((value * 10) % 1 - Fraction(1, 2)) > Fraction(1, 50)


@generator("exd_x_m4_area_units_land_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m4_area_units_land_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Hectares from km by m, a cost per hectare, then a percentage of a farm. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "dimensions with a clear percentage"):
        l10 = rng.randint(6, 45)  # tenths of a kilometre
        w = 50 * rng.randint(4, 18)
        s10 = rng.randint(12, 120)  # tenths of a square kilometre
        price = 10 * rng.randint(4, 19)
        ha = Fraction(l10 * w, 100)
        share = ha / Fraction(s10, 10)  # percent: ha / (100 S) * 100
        if ha >= 100 * Fraction(s10, 10) / 2 or share < 2 or not _hundredths(share):
            continue
        l_text = exact_decimal_string(Fraction(l10, 10), 1)
        s_text = exact_decimal_string(Fraction(s10, 10), 1)
        cost = ha * price
        pc = round_half_up(share, 1)
        ha_text, cost_text = exact_decimal_string(ha, 1), str(int(cost))
        pc_text = stated_precision_string(pc, 1)
        stem = rng.choice(_STEMS).format(l=l_text, w=w, s=s_text) + " " + rng.choice(_TREATMENTS).format(p=price)
        if cost.denominator == 1 and all(t not in stem.split() for t in (ha_text, cost_text, pc_text)):
            break
    area_m2 = Fraction(l10 * 100 * w)
    s = Fraction(s10, 10)
    # Every wrong value is shown at the key's own places: a distractor with a different format gives the key away.
    a_places = len(ha_text.partition(".")[2])
    a_wrong = [area_m2, area_m2 / 1000, ha / 10, area_m2 / 10**6, Fraction(l10, 10) * w / 10000]
    b_wrong = [ha * price * 10, ha * price / 10, ha + price, ha * price / 100]
    c_wrong = [ha / (10 * s), ha / (100 * s), 100 * ha / s, 10000 * s / ha]

    def options(values, places):
        return [
            Distractor(
                stated_precision_string(round_half_up(Fraction(v), places), places),
                None,
                round_half_up(Fraction(v), places),
            )
            for v in values
            if round_half_up(Fraction(v), places) > 0
        ]

    a_d = pick_distractors(ha_text, options(a_wrong, a_places), 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(cost_text, options(b_wrong, 0), 3, tolerance=GRADER_TOLERANCE)
    c_options = [
        Distractor(stated_precision_string(round_half_up(v, 1), 1), None, round_half_up(v, 1)) for v in c_wrong
    ]
    c_d = pick_distractors(pc_text, c_options, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "not enough distractors")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the area of the block in hectares.",
                answer_kind="number",
                answer=numeric_answer(ha),
                answer_text=ha_text,
                answer_latex=latex_answer(ha_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{l_text} km = {l10 * 100} m, so the area is {l10 * 100} × {w} = {area_m2} m².",
                    f"1 ha = 10000 m², so the area is {ha_text} ha.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Hence find the cost of treating the whole block, in dollars.",
                answer_kind="number",
                answer=int(cost),
                answer_text=cost_text,
                answer_latex=latex_answer(cost_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{price}*part_a",
                working=[f"Cost = {ha_text} × {price} = ${cost_text}."],
                distractors=b_d,
            ),
            dict(
                prompt="What percentage of the total area is the block? Give your answer correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(pc),
                answer_text=pc_text,
                answer_latex=latex_answer(pc_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"10*part_a/{s10}",
                working=[
                    f"{s_text} km² = {s_text} × 100 ha = {s * 100} ha.",
                    f"{ha_text} ÷ {s * 100} × 100 = {pc_text}% (1 d.p.).",
                ],
                distractors=c_d,
            ),
        ],
    )
