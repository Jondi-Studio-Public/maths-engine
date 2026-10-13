"""Exam question: a tank measured in mixed units, its volume, its capacity in litres and the time to fill it.

A three-part chain. Part (a) finds the volume in cm³ from dimensions given partly in metres and partly in
centimetres (every metre figure must be converted before multiplying), part (b) converts to litres
(1 L = 1000 cm³), and part (c) divides by a fill rate. Parts (b) and (c) carry `follow_through`.

Built backwards from whole-centimetre dimensions whose volume is a whole number of litres, and a fill rate that
divides the capacity exactly, so every key is an integer.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase

from .exd_x_m6_support import TooFewDistractors, number_part, shows_no_key, shown, tagged

_STEMS = (
    "A rainwater tank at a farm near Dubbo is a rectangular prism. Its base is {L} by {W} and it is {H} deep.",
    "A rectangular aquarium at a Cairns reef centre has a base measuring {L} by {W}, and its water is {H} deep.",
    "A rectangular concrete trough in a Katherine paddock measures {L} by {W} across the base and is {H} deep.",
)
_FILL = (
    "It is filled from a hose delivering {r} litres per minute.",
    "A pump fills it at a steady {r} litres each minute.",
)


def _dim(cm: int, in_metres: bool) -> str:
    return f"{shown(Fraction(cm, 100))} m" if in_metres else f"{cm} cm"


@generator("exd_x_m6_volume_tank", level=12, presets={"exam": dict(tier="exam")})
def exd_x_m6_volume_tank(rng: random.Random, tier: str = "exam") -> dict:
    """Volume in cm³ from mixed units, capacity in litres, and the time to fill at a given rate."""
    check(tier == "exam", "exd_x_m6_volume_tank: only the exam tier exists")
    for _ in attempts(1000, "a tank with a whole number of litres and a fill rate that divides it"):
        try:
            question = _attempt(rng)
        except TooFewDistractors:
            continue
        if question is not None:
            return question


def _attempt(rng: random.Random) -> dict | None:
    lc, wc, hc = rng.randrange(80, 201, 5), rng.randrange(40, 121, 5), rng.randrange(30, 101, 5)
    metre_dims = rng.choice(((True, False, False), (True, True, False)))
    volume = lc * wc * hc
    if volume % 1000:
        return None
    litres = volume // 1000
    rates = [r for r in range(5, 61) if litres % r == 0 and 4 <= litres // r <= 150]
    if not rates:
        return None
    rate = rng.choice(rates)
    minutes = litres // rate
    dims = [_dim(c, m) for c, m in zip((lc, wc, hc), metre_dims, strict=True)]
    stem = phrase(rng, _STEMS, L=dims[0], W=dims[1], H=dims[2]) + " " + phrase(rng, _FILL, r=rate)
    prompts = (
        "Find the volume of the container in cubic centimetres.",
        "Find the capacity of the container in litres, given that 1 litre is 1000 cm³.",
        "How many minutes does it take to fill the container from empty?",
    )
    if not shows_no_key([stem, *prompts], [volume, litres, minutes]):
        return None

    k = sum(metre_dims)

    def whole(value: Fraction, tag: str):
        """A wrong value offered only when it is a whole number, like the key: a decimal would give the key away."""
        value = Fraction(value)
        return tagged(value, tag) if value.denominator == 1 else None

    a_cands = [
        whole(Fraction(volume, 100**k), "forgot-unit-conversion"),  # multiplied the figures as written
        whole(Fraction(volume, 10**k), "unit-factor-wrong-power"),  # 10 for each metre figure instead of 100
        whole(Fraction(volume * 10**k), "unit-factor-wrong-power"),  # 1000 for each metre figure
        whole(Fraction(volume, 100), "forgot-unit-conversion"),  # converted only one of the metre figures
        whole(Fraction(volume * 100), "unit-factor-wrong-power"),
    ]
    b_cands = [
        whole(Fraction(volume, 100), "unit-factor-wrong-power"),
        whole(Fraction(volume, 10**6), "unit-factor-wrong-power"),
        whole(Fraction(volume * 1000), "unit-conversion-wrong-way"),
        whole(Fraction(volume, 10), "unit-factor-wrong-power"),
    ]
    c_cands = [
        whole(Fraction(litres * rate), "unit-conversion-wrong-way"),
        whole(Fraction(volume, rate), "forgot-unit-conversion"),
        whole(Fraction(minutes * 60), "unit-conversion-wrong-way"),  # minutes turned into seconds
        whole(Fraction(litres - rate), "inverted-ratio"),  # subtracted the rate instead of dividing by it
    ]
    return dict(
        text=stem,
        parts=[
            number_part(
                prompts[0],
                Fraction(volume),
                a_cands,
                marks=2,
                working=[
                    f"In centimetres the dimensions are {lc}, {wc} and {hc}.",
                    f"V = {lc} × {wc} × {hc} = {volume} cm³.",
                ],
            ),
            number_part(
                prompts[1],
                Fraction(litres),
                b_cands,
                marks=1,
                follow_through="part_a/1000",
                working=[f"{volume} ÷ 1000 = {litres} L."],
            ),
            number_part(
                prompts[2],
                Fraction(minutes),
                c_cands,
                marks=1,
                follow_through=f"part_b/{rate}",
                working=[f"{litres} ÷ {rate} = {minutes} minutes."],
            ),
        ],
    )
