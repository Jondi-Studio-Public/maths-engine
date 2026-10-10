"""Generates `exd_x_map_area` exam questions: a real area, in hectares, from an area measured on a map.

A ratio scale 1 : n is a LENGTH scale, so an area on the map is scaled by n squared. The prompt gives either
a map area in square centimetres or the two side lengths of a rectangular paddock measured on the map; the
learner must square the scale and then convert square centimetres to hectares.

Built backwards from a whole number of hectares H: the map area is H x 10^8 / n^2 square centimetres, which
is a number with at most two decimal places for every scale used here. A rectangle is only offered when its
sides (one decimal place each) multiply to exactly that area.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, check_distinct, generator, pick_distractors

_SCALES = [10000, 20000, 25000, 50000]
_FEATURES = ["a lake", "a paddock", "a plantation", "a nature reserve"]
_MAPS = ["road map", "cadastral map", "bushwalking map", "online map"]


def _fixed(value: Fraction, places: int) -> str:
    scaled = value * 10**places
    check(scaled.denominator == 1, "value not exact at the stated places")
    whole, frac = divmod(int(scaled), 10**places)
    return f"{whole}.{frac:0{places}d}" if places else str(whole)


def _rectangle(rng: random.Random, area: Fraction):
    """Sides, in cm to one decimal place, whose product is exactly ``area`` square centimetres (or None)."""
    pairs = []
    for a10 in range(20, 121):
        a = Fraction(a10, 10)
        b = area / a
        if (b * 10).denominator == 1 and 2 <= b <= 12 and a != b:
            pairs.append((a, b))
    return rng.choice(pairs) if pairs else None


def _build(rng: random.Random) -> dict:
    n = rng.choice(_SCALES)
    hectares = rng.randint(6, 90)
    map_area = Fraction(hectares * 10**8, n * n)
    check((map_area * 100).denominator == 1, "map area needs more than two decimal places")

    sides = _rectangle(rng, map_area) if rng.random() < 0.5 else None
    # Second route: real side lengths in metres, multiplied, in square metres.
    if sides:
        real_m2 = (sides[0] * n / 100) * (sides[1] * n / 100)
    else:
        real_m2 = map_area * n * n / 10**4
    check(real_m2 / 10**4 == hectares, "area mismatch")

    candidates = {
        "takes-a-hectare-as-1000-square-metres": str(hectares * 10),
        "converts-square-centimetres-with-100": str(hectares * 100),
        "leaves-the-answer-in-square-metres": str(hectares * 10**4),
        "leaves-the-answer-in-square-centimetres": str(hectares * 10**8),
    }
    check_distinct(hectares, *[Fraction(v) for v in candidates.values()])

    feature = rng.choice(_FEATURES)
    map_name = rng.choice(_MAPS)
    if sides:
        text = (
            f"On a {map_name} with a scale of 1 : {n}, {feature} is a rectangle measuring "
            f"{_fixed(sides[0], 1)} cm by {_fixed(sides[1], 1)} cm. What is its real area, in hectares?"
        )
    else:
        text = (
            f"On a {map_name} with a scale of 1 : {n}, {feature} has an area of {_fixed(map_area, 2)} cm² "
            f"on the map. What is its real area, in hectares?"
        )
    answer_text = str(hectares)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=hectares,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Lengths are scaled by {n}, so areas are scaled by {n}^2.",
            f"{_fixed(map_area, 2)} cm^2 x {n}^2 = {int(real_m2)} m^2 = {hectares} ha (1 ha = 10 000 m^2).",
        ],
        distractors=pick_distractors(answer_text, list(candidates.values())),
    )


@generator(
    "exd_x_map_area",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_map_area(rng: random.Random, tier: str = "exam") -> dict:
    """Real area in hectares from a map area under a ratio scale. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
