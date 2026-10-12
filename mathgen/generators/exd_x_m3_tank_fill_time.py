"""Exam-tier source for `prism-capacity-metric-units`: a tank filled by a pump.

The practice tiers turn three dimensions into a capacity in one stated unit. Here the dimensions arrive in
mixed units (metres and centimetres) and the capacity is only the first step: the learner then works out how
much is still to be filled after the pump has run for a while, and how much longer it must run. Parts (b) and
(c) carry follow-through.

Built backwards: the capacity is `rate * total minutes`, so every answer is a whole number.
"""

from __future__ import annotations

import random
from math import prod

from mathgen.core import attempts, generator, phrase

from .exd_x_m3_support import finish, numeric_part

_THINGS = ["water tank", "fish tank", "rainwater tank", "tub", "pool", "trough"]
_STEMS = [
    "A rectangular {thing} is {L} m long, {W} cm wide and {H} cm deep. It starts empty. A pump fills it at "
    "{r} litres per minute and has been running for {t} minutes.",
    "A {thing} is a rectangular prism, {L} m long, {W} cm wide and {H} cm high. A pump fills it at a steady "
    "{r} litres every minute. The pump has been on for {t} minutes and the {thing} was empty to begin with.",
]


@generator("exd_x_m3_tank_fill_time", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m3_tank_fill_time(rng: random.Random, tier: str = "exam") -> dict:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a tank with a whole-number fill time"):
        tenths = rng.randint(12, 30)  # length in tenths of a metre
        w10, h10 = rng.randint(5, 12), rng.randint(4, 10)
        W, H = 10 * w10, 10 * h10
        capacity = tenths * w10 * h10  # litres: (tenths/10 m = 10*tenths cm) * W * H / 1000
        rates = [r for r in range(6, 61) if capacity % r == 0 and 8 <= capacity // r <= 90]
        if not rates:
            continue
        r = rng.choice(rates)
        total = capacity // r
        t = rng.randint(2, total - 3)
        left = capacity - r * t
        rest = total - t
        thing = rng.choice(_THINGS)
        stem = phrase(rng, _STEMS, thing=thing, L=f"{tenths / 10:.1f}", W=W, H=H, r=r, t=t)
        vol_cm3 = 10 * tenths * W * H
        parts = [
            numeric_part(
                f"Find the capacity of the {thing}, in litres.",
                capacity,
                0,
                2,
                [
                    f"{tenths / 10:.1f} m = {10 * tenths} cm.",
                    f"Volume = {10 * tenths} × {W} × {H} = {vol_cm3} cm³; 1 L = 1000 cm³.",
                ],
                [(vol_cm3, "unit-not-converted"), (capacity * 10, "unit-factor-wrong-power"), (capacity / 10, None)],
            ),
            numeric_part(
                f"How many more litres are needed to fill the {thing}?",
                left,
                0,
                1,
                [f"Pumped so far = {r} × {t} = {r * t} L.", f"Still needed = {capacity} - {r * t} = {left} L."],
                [(r * t, None), (capacity + r * t, None), (capacity, None)],
                follow_through=f"part_a - {r * t}",
            ),
            numeric_part(
                "For how many more minutes must the pump run to fill it?",
                rest,
                0,
                1,
                [f"Time = {left} ÷ {r} = {rest} minutes."],
                [(total, None), (left * r, "unit-conversion-wrong-way"), (t, None)],
                follow_through=f"part_b/{r}",
            ),
        ]
        out = finish(stem, parts)
        if out is not None:
            return out
