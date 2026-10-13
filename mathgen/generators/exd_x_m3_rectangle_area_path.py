"""Exam-tier source for `rectangle-area`: a garden with a path around it.

The practice tiers multiply two sides, or work back to a missing side. Here a rectangle sits inside a larger
rectangle: the learner finds the area of the inner rectangle, works out the outer rectangle (a path runs along
both sides of each dimension, so each dimension grows by twice the path's width), takes the difference to get
the path alone, and prices it. Parts (c) and (d) carry follow-through.

Built forwards from `(length, width, path width, price)`; every answer is a whole number.
"""

from __future__ import annotations

import random

from mathgen.core import attempts, generator, phrase

from .exd_x_m3_support import finish, numeric_part

_THINGS = [
    "vegetable garden",
    "school courtyard",
    "backyard lawn",
    "swimming pool",
    "community herb garden",
    "flower bed",
]
_STEMS = [
    "A rectangular {thing} is {L} m long and {W} m wide. A path {w} m wide is built all the way around the outside of it.",
    "A path {w} m wide runs right around a rectangular {thing} that measures {L} m by {W} m.",
    "A council is paving a path {w} m wide around a rectangular {thing}. The {thing} is {L} m long and {W} m wide.",
]


@generator("exd_x_m3_rectangle_area_path", level=10, presets={"exam": dict(tier="exam")})
def exd_x_m3_rectangle_area_path(rng: random.Random, tier: str = "exam") -> dict:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a rectangle and path with distinct answers"):
        L, W = rng.randint(8, 40), rng.randint(5, 30)
        if L == W:
            continue
        w = rng.randint(1, 3)
        price = rng.choice([18, 22, 25, 30, 35, 40, 45, 55, 60, 75])
        thing = rng.choice(_THINGS)
        inner = L * W
        outer = (L + 2 * w) * (W + 2 * w)
        path = outer - inner
        cost = path * price
        stem = phrase(rng, _STEMS, thing=thing, L=L, W=W, w=w)
        parts = [
            numeric_part(
                f"Find the area of the {thing}, in square metres.",
                inner,
                0,
                1,
                [f"Area = {L} × {W} = {inner} m²."],
                [(2 * (L + W), "perimeter-for-area"), (L + W, "perimeter-for-area"), (L * W * w, None)],
            ),
            numeric_part(
                f"Find the area of the {thing} and the path together, in square metres.",
                outer,
                0,
                2,
                [
                    f"The path is on both sides, so each dimension grows by 2 × {w} = {2 * w} m.",
                    f"Outer rectangle: {L + 2 * w} m by {W + 2 * w} m, area = {outer} m².",
                ],
                [
                    ((L + w) * (W + w), "border-on-one-side"),
                    ((L + 2 * w) * W, "border-on-one-side"),
                    (2 * (L + W + 4 * w), "perimeter-for-area"),
                    (inner + 2 * w * (L + W), None),
                ],
            ),
            numeric_part(
                "Hence find the area of the path alone, in square metres.",
                path,
                0,
                1,
                [f"Path = {outer} - {inner} = {path} m²."],
                [(outer, None), (2 * w * (L + W), None), (w * (L + W), "border-on-one-side")],
                follow_through="part_b - part_a",
            ),
            numeric_part(
                f"Paving costs ${price} per square metre. Find the cost of paving the path, in dollars.",
                cost,
                0,
                1,
                [f"Cost = {path} × {price} = ${cost}."],
                [(outer * price, None), (inner * price, None), (2 * w * (L + W) * price, None)],
                follow_through=f"part_c*{price}",
            ),
        ]
        d = finish(stem, parts)
        if d is not None:
            return d
