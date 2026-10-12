"""Exam source for `f10-triangular-prism-volume`: a triangular trough described by its slant sides.

The cross-section is an isosceles triangle given by its equal sides and its base, so the height has to be found
first (a Pythagorean triple, so it is exact). Then the volume, then the capacity in litres.

(a) the height of the cross-section, (b) the volume in cubic centimetres, (c) the capacity in litres.
Parts (b) and (c) carry follow-through.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, latex_answer
from mathgen.generators.exd_x_m2_support import display, keys_absent, tagged_distractors

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (6, 8, 10), (9, 12, 15), (12, 16, 20), (15, 20, 25)]
_OBJECTS = [
    "A feeding trough",
    "A planter box",
    "A rain gutter section",
    "A water trough",
    "A cattle feed bin",
    "A compost bay",
]


@generator("exd_x_m2_prism_trough", level=8, presets={"exam": dict(tier="exam")})
def exd_x_m2_prism_trough(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: height from the slant sides, then volume, then litres."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a draw with whole-tenth litres and answers not in the text"):
        half, h, c = rng.choice(_TRIPLES)
        k = 2 if c <= 13 and rng.random() < 0.4 else 1
        half, h, c = half * k, h * k, c * k
        length = rng.randint(30, 150)
        base = 2 * half
        volume = half * h * length
        if volume % 100:
            continue
        litres = Fraction(volume, 1000)
        stem = (
            f"{rng.choice(_OBJECTS)} is a triangular prism lying on one of its rectangular faces. "
            f"Its triangular end is isosceles, with two equal sides of {c} cm and a base of {base} cm. "
            f"The prism is {length} cm long."
        )
        prompts = [
            "Find the perpendicular height of the triangular end, in centimetres.",
            "Hence find the volume of the prism, in cubic centimetres.",
            "Find how many litres it holds when full, correct to 1 decimal place.",
        ]
        a_text, b_text, c_text = str(h), str(volume), display(litres, 1)
        if keys_absent(stem, prompts, [a_text, b_text, c_text]):
            break
    check(half * half + h * h == c * c, "triple is not Pythagorean")
    check(Fraction(half * h * length, 1000) == litres, "litres disagree with the volume")

    a_dist = tagged_distractors(
        a_text, 0, [(c, None), (c - half, None), (base, None), (c + half, None), (h + half, None), (c + base, None)]
    )
    b_dist = tagged_distractors(
        b_text,
        0,
        [
            (base * h * length, None),
            (base * c * length // 2, None),
            (base * c * length, None),
            (half * h * 2 * length / 3, None),
        ],
    )
    c_dist = tagged_distractors(
        c_text,
        1,
        [
            (litres * 10, None),
            (litres / 10, None),
            (Fraction(volume), None),
            (litres * 2, None),
        ],
    )
    for dist in (a_dist, b_dist, c_dist):
        check(len(dist) >= 3, "not enough distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=h,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The height splits the base in half: {half} cm each side.",
                    f"h² = {c}² - {half}² = {c * c - half * half}, so h = {h} cm.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=volume,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{base}*part_a/2*{length}",
                working=[
                    f"Area of end = ½ × {base} × {h} = {half * h} cm².",
                    f"Volume = {half * h} × {length} = {volume} cm³.",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=litres,
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_b/1000",
                working=[f"1 litre = 1000 cm³, so {volume} ÷ 1000 = {c_text} L."],
                distractors=c_dist,
            ),
        ],
    )
