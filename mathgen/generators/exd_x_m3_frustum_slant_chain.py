"""Exam-tier source for `frustum-volume`: a frustum given by its slant height, and the cone cut off it.

The practice tiers give the frustum's vertical height and ask for one volume. Here only the slant height is
given, so the learner must first find the vertical height with Pythagoras (the slant height is the hypotenuse
of a triangle whose other sides are the height and the difference of the radii), then the volume, then use
similar triangles to find the small cone that was cut off, and finally the whole cone. Parts (b), (c) and (d)
carry follow-through.

Built backwards from a Pythagorean triple `(difference of radii, height, slant height)`, so the height is a
whole number. pi = 3.14159 exactly as printed.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase, round_half_up

from .exd_x_m3_support import clear_of_tie, display, finish, numeric_part

PI = Fraction(314159, 100000)
_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 12, 15)]
_THINGS = ["bucket", "lampshade", "plant pot", "paper cup", "flower vase", "concrete pier"]
_STEMS = [
    "A {thing} is a frustum of a cone with a base radius of {R} cm, a top radius of {r} cm and a slant height of "
    "{s} cm. Use π = 3.14159.",
    "The {thing} is shaped like a cone with its top cut off by a plane parallel to the base. The base radius is "
    "{R} cm, the top radius is {r} cm and the slant height is {s} cm. Use π = 3.14159.",
]


def _cone(radius: Fraction, height: Fraction) -> Fraction:
    return PI * radius * radius * height / 3


@generator("exd_x_m3_frustum_slant_chain", level=14, presets={"exam": dict(tier="exam")})
def exd_x_m3_frustum_slant_chain(rng: random.Random, tier: str = "exam") -> dict:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(3000, "a frustum whose roundings are unambiguous"):
        a, b, c = rng.choice(_TRIPLES)
        d, h = (a, b) if rng.random() < 0.5 else (b, a)
        k = rng.choice([1, 1, 2, 3]) if c < 20 else 1
        d, h, s = d * k, h * k, c * k
        r = rng.randint(2, 14)
        R = r + d
        if R > 45:
            continue
        thing = rng.choice(_THINGS)
        frustum = PI * h * (R * R + R * r + r * r) / 3
        small_h = Fraction(r * h, d)
        small = _cone(Fraction(r), small_h)
        whole = _cone(Fraction(R), Fraction(R * h, d))
        check(abs(whole - small - frustum) < Fraction(1, 10**9), "frustum volume disagrees with the two cones")
        frustum_via = frustum  # (b) is built from the whole-number height, so no rounding enters
        small_via = small
        whole_via = round_half_up(frustum, 1) + round_half_up(small, 1)
        if not all(clear_of_tie(v, 1) for v in (frustum, small, whole)):
            continue
        if display(whole_via, 1) != display(whole, 1) or frustum_via != frustum or small_via != small:
            continue
        stem = phrase(rng, _STEMS, thing=thing, R=R, r=r, s=s)
        pi_s = "3.14159"
        parts = [
            numeric_part(
                f"Find the vertical height of the {thing}, in centimetres.",
                h,
                0,
                2,
                [f"Radii differ by {R} - {r} = {d} cm.", f"h² = {s}² - {d}² = {s * s - d * d}, so h = {h} cm."],
                [
                    (s, "slant-height-for-height"),
                    (d, None),
                    (h + d, None),
                ],
            ),
            numeric_part(
                f"Hence find the volume of the {thing}, in cubic centimetres, correct to 1 decimal place.",
                frustum,
                1,
                2,
                [f"V = (1/3)πh(R² + Rr + r²) = (1/3) × π × {h} × ({R * R} + {R * r} + {r * r})."],
                [
                    (_cone(Fraction(R), Fraction(h)), None),
                    (PI * h * (R * R + R * r + r * r), "cone-without-one-third"),
                    (PI * h * (R * R + r * r) / 3, None),
                    (PI * s * (R * R + R * r + r * r) / 3, "slant-height-for-height"),
                ],
                follow_through=f"{pi_s}*part_a*({R * R + R * r + r * r})/3",
            ),
            numeric_part(
                f"The {thing} was made by cutting a small cone off the top of a larger cone. Find the volume of the "
                "small cone that was cut off, in cubic centimetres, correct to 1 decimal place.",
                small,
                1,
                2,
                [
                    f"Similar triangles: small cone height ÷ {r} = {h} ÷ {d}, so its height is {r * h}/{d} cm.",
                    f"V = (1/3)π × {r}² × {float(small_h):.2f}.",
                ],
                [
                    (_cone(Fraction(r), Fraction(h)), "cut-off-cone-uses-frustum-height"),
                    (_cone(Fraction(r), Fraction(h * d, r)), "inverted-ratio"),
                    (_cone(Fraction(r), Fraction(R * h, d)), None),
                ],
                follow_through=f"{pi_s}*{r * r}*part_a*{r}/{d}/3",
            ),
            numeric_part(
                f"Hence find the volume of the whole cone before it was cut, in cubic centimetres, correct to 1 "
                "decimal place.",
                whole,
                1,
                1,
                ["Whole cone = frustum + small cone."],
                [(frustum, None), (frustum - small, None), (_cone(Fraction(R), Fraction(h)), None)],
                follow_through="part_b + part_c",
            ),
        ]
        out = finish(stem, parts)
        if out is not None:
            return out
