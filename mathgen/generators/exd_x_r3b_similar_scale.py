"""Generates `exd_x_r3b_similar_scale`: a multi-part exam question on two similar solids.

The stem gives the surface areas of a small and a large solid of similar shape, the small one's height and its
volume. The chain is the one a general-mathematics or methods paper sets:

(a) the linear scale factor from the small to the large solid   fraction, 2 marks (the square root of the area ratio)
(b) the height of the large solid                                       number,   1 mark,  part_a * h
(c) the volume of the large solid                                  number,   2 marks, part_a^3 * V

The linear factor is drawn first as p/q in lowest terms. The areas are q^2 c and p^2 c square centimetres, the
height a multiple of q centimetres and the volume a multiple of q^3 cubic centimetres, so every answer is a whole number.
Each key is checked against exact arithmetic: sqrt of the printed area ratio, then height and volume scaled by it.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import gcd, isqrt

from ..core import check, generator, phrase
from .exd_x_r3b_common import fraction_part, number_part
from ._exd_r3_distinct import distinct_part_answers

# (plural, singular)
_OBJECTS = (
    ("souvenir statues", "statue"),
    ("decorative candles", "candle"),
    ("concrete garden ornaments", "ornament"),
    ("chocolate Easter bunnies", "bunny"),
    ("trophies", "trophy"),
    ("garden gnomes", "gnome"),
)
# (q, p, height in units of q, volume in units of q^3 cm^3, area in units of q^2 cm^2)
_SHAPES = (
    (2, 3, range(6, 12), range(60, 121), range(100, 200)),
    (2, 5, range(6, 11), range(40, 81), range(100, 160)),
    (3, 4, range(5, 9), range(20, 41), range(50, 90)),
    (3, 5, range(5, 9), range(20, 41), range(50, 90)),
    (3, 7, range(5, 8), range(14, 26), range(50, 90)),
    (4, 5, range(4, 7), range(10, 17), range(30, 50)),
    (4, 7, range(4, 7), range(10, 15), range(30, 50)),
    (5, 6, range(3, 6), range(6, 9), range(20, 30)),
    (5, 7, range(3, 6), range(6, 9), range(20, 30)),
    (5, 8, range(3, 5), range(6, 9), range(20, 30)),
)

_STEMS = (
    "Two solid {plural} are similar in shape. The small {one} has a surface area of {a1} cm², a height of {h1} cm "
    "and a volume of {v1} cm³. The large {one} has a surface area of {a2} cm².",
    "A small {one} and a large {one} are similar solids. The small {one} has height {h1} cm, surface area {a1} cm² "
    "and volume {v1} cm³, and the large {one} has surface area {a2} cm².",
    "Two similar solid {plural} are sold in a shop. The surface area of the small {one} is {a1} cm², its height is "
    "{h1} cm and its volume is {v1} cm³. The surface area of the large {one} is {a2} cm².",
)


def _build(rng: random.Random) -> dict:
    q, p, heights, capacities, areas = rng.choice(_SHAPES)
    check(gcd(p, q) == 1 and p > q, "scale factor is not in lowest terms")
    plural, one = rng.choice(_OBJECTS)
    c = rng.choice(list(areas))
    h1 = q * rng.choice(list(heights))
    v1 = q**3 * rng.choice(list(capacities))
    a1, a2 = q * q * c, p * p * c
    k = Fraction(p, q)

    # Independent route from the printed numbers: square root of the area ratio, then scale.
    ratio = Fraction(a2, a1)
    root = Fraction(isqrt(ratio.numerator), isqrt(ratio.denominator))
    check(root * root == ratio and root == k, "the area ratio is not the square of the scale factor")
    h2, v2 = k * h1, k**3 * v1
    check(h2.denominator == 1 and v2.denominator == 1, "height or volume is not a whole number")
    check(h2 > h1 and v2 > v1 and v2 <= 25000, "the large solid is not a plausible size")

    stem = phrase(rng, _STEMS, plural=plural, one=one, a1=a1, a2=a2, h1=h1, v1=v1)
    h2i, v2i = int(h2), int(v2)
    part_a = fraction_part(
        f"Find the scale factor, as a fraction, from the small {one} to the large {one}.",
        k,
        2,
        [
            f"Area ratio = {a2}/{a1} = {ratio.numerator}/{ratio.denominator}.",
            f"The scale factor is the square root: {p}/{q}.",
        ],
        [
            ("area-ratio-taken-as-scale-factor", ratio),
            ("swapped-p-and-q", 1 / k),
            ("volume-ratio-taken-as-scale-factor", k**3),
        ],
        sensible_range=(1, None),
    )
    part_b = number_part(
        f"Find the height of the large {one}, in centimetres.",
        h2,
        0,
        1,
        [f"Height = {p}/{q} × {h1} = {h2i} cm."],
        [
            ("uses-squared-factor", h1 * k**2),
            ("swapped-p-and-q", h1 / k),
            ("forgets-to-scale", Fraction(h1)),
            ("uses-cubed-factor-for-length", h1 * k**3),
        ],
        follow_through=f"part_a*{h1}",
    )
    part_c = number_part(
        f"Find the volume of the large {one}, in cubic centimetres.",
        v2,
        0,
        2,
        [f"Volume = ({p}/{q})^3 × {v1} = {v2i} cm³."],
        [
            ("volume-scaled-by-linear-factor", v1 * k),
            ("volume-scaled-by-area-factor", v1 * k**2),
            ("swapped-p-and-q", v1 / k**3),
            ("forgets-to-scale", Fraction(v1)),
        ],
        follow_through=f"part_a**3*{v1}",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])


@generator("exd_x_r3b_similar_scale", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_similar_scale(rng: random.Random, tier: str = "exam") -> dict:
    """Two similar solids: the scale factor from their areas, then a height and a volume."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
