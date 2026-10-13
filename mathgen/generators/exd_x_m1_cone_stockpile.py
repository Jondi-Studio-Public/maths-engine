"""Exam source of `cone-volume`: a conical stockpile, from its slant height to the number of truck loads.

The practice tiers ask for the volume of one cone from its dimensions. An exam question hides the height: (a) the
vertical height from a base diameter and a slant height (halve the diameter, then Pythagoras), (b) the volume of the
cone, (c) the number of truck loads needed to move it, which is a division followed by rounding up because a partly
filled truck still has to make the trip.

## Construction

The radius, height and slant height come from a Pythagorean triple, so the height is an exact whole number and part
(a) is graded exactly. The volume uses the stated value of pi (3.14159) and is shown to 1 decimal place. A draw is
accepted only when the number of loads is the same whether it comes from the exact volume or from the rounded volume
a learner carries from (b), with the quotient well clear of a whole number.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

PI = Fraction(314159, 100000)  # the value of pi the prompt tells the learner to use
_MARGIN = Fraction(1, 200)
# (radius, height, slant height) with whole-number sides; the legs run 3 to 24 m
_TRIPLES = [
    (a, b, round(math.hypot(a, b)))
    for a in range(3, 25)
    for b in range(3, 25)
    if round(math.hypot(a, b)) ** 2 == a * a + b * b
]
_CAPACITIES = [6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
_PILES = [
    ("A conical stockpile of gravel at a quarry", "gravel"),
    ("A conical heap of sand at a building site", "sand"),
    ("A conical pile of mulch at a garden centre", "mulch"),
    ("A conical stockpile of wheat at a grain depot", "wheat"),
    ("A conical heap of road base at a council depot", "road base"),
    ("A conical pile of woodchips at a sawmill", "woodchips"),
]


def _tenths(value: Fraction) -> Fraction:
    return round_half_up(value, 1)


@generator("exd_x_m1_cone_stockpile", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_m1_cone_stockpile(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a cone whose truck-load count survives the rounding of its volume"):
        r, h, s = rng.choice(_TRIPLES)
        cap = rng.choice(_CAPACITIES)
        pile, material = rng.choice(_PILES)
        volume = PI * r * r * h / 3
        v1 = _tenths(volume)
        loads = math.ceil(volume / cap)
        if loads != math.ceil(v1 / cap) or abs(volume / cap - round(volume / cap)) < Fraction(1, 20):
            continue
        if abs(((volume * 10) % 1) - Fraction(1, 2)) / 10 < _MARGIN:
            continue
        keys = {str(h), stated_precision_string(v1, 1), str(loads)}
        if keys & {str(2 * r), str(s), str(cap), "3.14159", "1"}:
            continue
        break

    d = 2 * r
    v_text, loads_text = stated_precision_string(v1, 1), str(loads)
    stem = f"{pile} has a base diameter of {d} m and a slant height of {s} m. Use π ≈ 3.14159."
    p_a = "Find the vertical height of the cone, in metres."
    p_b = "Hence find the volume of the cone, in cubic metres, correct to 1 decimal place."
    p_c = (
        f"Each truck carries {cap} m³ of {material}. Using your volume from (b), how many truck loads "
        "are needed to move all of it?"
    )

    def vol_text(value: Fraction) -> str:
        return stated_precision_string(_tenths(value), 1)

    # Part (a) is a whole number of metres, so each slip is shown to the nearest metre.
    a_cands: list = [
        Distractor(str(s), "slant-height-for-height", s),
        Distractor(
            str(round_half_up(Fraction(math.sqrt(s * s + r * r)), 0)),
            "hypotenuse-added-not-subtracted",
            int(round_half_up(Fraction(math.sqrt(s * s + r * r)), 0)),
        ),
        str(s - r),
    ]
    if s > d:
        a_cands.append(
            Distractor(
                str(round_half_up(Fraction(math.sqrt(s * s - d * d)), 0)),
                "diameter-for-radius",
                int(round_half_up(Fraction(math.sqrt(s * s - d * d)), 0)),
            )
        )
    b_cands = [
        Distractor(vol_text(PI * r * r * h), "cone-without-one-third", _tenths(PI * r * r * h)),
        Distractor(vol_text(PI * d * d * h / 3), "diameter-for-radius", _tenths(PI * d * d * h / 3)),
        Distractor(vol_text(PI * r * r * s / 3), "slant-height-for-height", _tenths(PI * r * r * s / 3)),
    ]
    c_cands = [
        Distractor(str(math.floor(volume / cap)), "rounded-count-down", math.floor(volume / cap)),
        Distractor(str(math.ceil(volume * 3 / cap)), "cone-without-one-third", math.ceil(volume * 3 / cap)),
        str(math.ceil(volume)),
    ]
    a_d = pick_distractors(str(h), a_cands, 3, tolerance=0)
    b_d = pick_distractors(v_text, b_cands, 3, tolerance=Fraction(51, 1000))
    c_d = pick_distractors(loads_text, c_cands, 3)
    check(len(a_d) >= 2 and len(b_d) >= 3 and len(c_d) >= 2, "too few distractors survive")
    check(all(Fraction(str(x)) != h for x in a_d), "an (a) distractor equals the key")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p_a,
                answer_kind="number",
                answer=h,
                answer_text=str(h),
                answer_latex=latex_answer(str(h)),
                marks=2,
                sensible_range=(0, None),
                working=[f"Radius = {d} ÷ 2 = {r} m.", f"h² = {s}² - {r}² = {s * s - r * r}, so h = {h} m."],
                distractors=a_d,
            ),
            dict(
                prompt=p_b,
                answer_kind="number",
                answer=numeric_answer(v1),
                answer_text=v_text,
                answer_latex=latex_answer(v_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"3.14159*{r}**2*part_a/3",
                working=[f"V = ⅓πr²h = ⅓ × π × {r}² × {h} = {v_text} m³ (1 d.p.)."],
                distractors=b_d,
            ),
            dict(
                prompt=p_c,
                answer_kind="number",
                answer=loads,
                answer_text=loads_text,
                answer_latex=latex_answer(loads_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"ceiling(part_b/{cap})",
                working=[
                    f"{v_text} ÷ {cap} = {float(v1 / cap):.2f}.",
                    f"A part load still needs a truck, so round up to {loads}.",
                ],
                distractors=c_d,
            ),
        ],
    )
