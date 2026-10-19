"""Exam-tier source for `f10-volume-surface-area-prisms-cylinders`: a closed cylindrical tin.

The practice tiers give a cylinder's radius and height and ask for one volume or one surface area. Here the
tin is described by its diameter, so the learner must halve it first, and four connected results are needed:
the volume, the capacity in litres, the curved surface (the label), and the total surface including both
ends. Parts (b) and (d) carry follow-through; (d) reuses the label area from (c) and adds the two ends.

Built forwards from `(diameter, height)` with pi = 3.14 exactly as printed, so the exact value and the
learner's value agree. A draw is rejected if rounding an earlier answer would change a later last digit.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, generator, phrase, round_half_up

from .exd_x_m3_support import clear_of_tie, display, finish, numeric_part

PI = Fraction(314, 100)
_THINGS = ["storage canister", "metal drum", "tin", "sealed container", "plastic bin", "sealed jar"]
_STEMS = [
    "A {thing} is a closed cylinder with a diameter of {D} cm and a height of {H} cm. Use π = 3.14.",
    "A {thing} is a closed cylinder. Its diameter is {D} cm and its height is {H} cm. Use π = 3.14 for this question.",
]


@generator("exd_x_m3_cylinder_tin_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_m3_cylinder_tin_chain(rng: random.Random, tier: str = "exam") -> dict:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a tin whose roundings are unambiguous"):
        r = rng.randint(3, 15)
        D = 2 * r
        H = rng.randint(6, 40)
        if D == H:
            continue
        thing = rng.choice(_THINGS)
        vol = PI * r * r * H
        litres = vol / 1000
        curved = 2 * PI * r * H
        ends = 2 * PI * r * r
        total = curved + ends
        # the learner's route: round each answer as asked, then use it
        litres_via = round(vol, 1) / 1000
        total_via = round(curved, 1) + ends
        if not (
            clear_of_tie(vol, 1) and clear_of_tie(litres, 2) and clear_of_tie(curved, 1) and clear_of_tie(total, 1)
        ):
            continue
        if display(litres_via, 2) != display(litres, 2) or display(total_via, 1) != display(total, 1):
            continue
        stem = phrase(rng, _STEMS, thing=thing, D=D, H=H)
        parts = [
            numeric_part(
                f"Find the volume of the {thing}, in cubic centimetres, correct to 1 decimal place.",
                vol,
                1,
                2,
                [f"Radius = {D} ÷ 2 = {r} cm.", f"V = πr²h = 3.14 × {r}² × {H} = {display(vol, 1)} cm³."],
                [
                    (vol / PI, "pi-left-out"),
                    (PI * D * D * H, "diameter-for-radius"),
                    (vol / 3, "cone-without-one-third"),
                    (PI * r * H * H, None),
                ],
            ),
            numeric_part(
                f"Hence find the capacity of the {thing}, in litres, correct to 2 decimal places.",
                litres,
                2,
                1,
                [
                    "1 L = 1000 cm³, so divide the volume by 1000.",
                    f"{display(vol, 1)} ÷ 1000 = {display(litres, 2)} L.",
                ],
                [
                    (vol / 100, "unit-factor-wrong-power"),
                    (round_half_up(vol, 1), "unit-not-converted"),
                    (round_half_up(vol, 1) * 1000, "unit-conversion-wrong-way"),
                ],
                follow_through="part_a/1000",
            ),
            numeric_part(
                f"A paper label covers the curved surface of the {thing} exactly. Find the area of the label, in "
                "square centimetres, correct to 1 decimal place.",
                curved,
                1,
                1,
                [f"Curved area = 2πrh = 2 × 3.14 × {r} × {H} = {display(curved, 1)} cm²."],
                [
                    (2 * PI * r, None),
                    (curved / 2, None),
                    (2 * PI * D * H, "diameter-for-radius"),
                ],
            ),
            numeric_part(
                f"Hence find the total surface area of the {thing}, including both ends, in square centimetres, "
                "correct to 1 decimal place.",
                total,
                1,
                2,
                [
                    f"Each end has area πr² = 3.14 × {r}² = {display(PI * r * r, 2)} cm², so the two ends are {display(ends, 2)} cm².",
                    f"Total = {display(curved, 1)} + {display(ends, 2)} = {display(total, 1)} cm².",
                ],
                [
                    (curved, "surface-area-omits-ends"),
                    (curved + PI * r * r, None),
                    (ends, None),
                    (curved + 2 * PI * D * D, "diameter-for-radius"),
                ],
                follow_through=f"part_c + {ends}",
            ),
        ]
        out = finish(stem, parts)
        if out is not None:
            return out
