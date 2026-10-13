"""Exam tier for "estimate lengths": estimate a boundary by pacing, then plan the materials for it.

The practice tiers choose the best estimate of a familiar object. Here a length is estimated from a measured
pace (paces counted, pace length in centimetres), and the estimate is used in a purchase that needs a judgement
about rounding: a whole number of fence panels, edging strips or pipe lengths must cover the estimated length, so the
count is rounded UP, and the cost follows from that count.

Parts (b) and (c) carry ``follow_through``. Built backwards from the pace, the number of paces and the length of
one unit, keeping the number of units away from a whole number so rounding up is a real step.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, draw_until, generator
from ._exd_x_sc4_kit import num_part, txt

_CONTEXTS = (
    dict(
        setup="A farmer paces out one boundary of a paddock. One pace is about {p} cm and the boundary takes "
        "{n} paces.",
        item="fence panels",
        one="fence panel",
        unit_len=(Fraction(12, 5), Fraction(3)),
        sold="Fence panels are {w} m long and cannot be cut.",
        price=(60, 120),
        ask="How many fence panels are needed to cover the whole boundary?",
    ),
    dict(
        setup="A groundskeeper paces along the edge of a new garden bed. One pace is about {p} cm and the edge "
        "takes {n} paces.",
        item="edging strips",
        one="edging strip",
        unit_len=(Fraction(6, 5), Fraction(3, 2)),
        sold="Garden edging is sold in strips {w} m long and the strips cannot be cut.",
        price=(8, 22),
        ask="How many edging strips are needed to cover the whole edge?",
    ),
    dict(
        setup="A plumber paces out the line of a trench for a water pipe. One pace is about {p} cm and the trench "
        "takes {n} paces.",
        item="lengths of pipe",
        one="length of pipe",
        unit_len=(Fraction(3), Fraction(6)),
        sold="The pipe is sold in {w} m lengths and cannot be cut to save waste.",
        price=(35, 90),
        ask="How many lengths of pipe are needed to cover the whole trench?",
    ),
)


def _w_text(w: Fraction) -> str:
    return str(int(w)) if w.denominator == 1 else txt(w, 1)


@generator("exd_x_sc4_est_length", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc4_est_length(rng: random.Random, tier: str = "exam") -> dict:
    """Estimate a length from paces, round the unit count up, then cost it. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    ctx = rng.choice(_CONTEXTS)

    def draw(r):
        return (
            r.randrange(65, 91, 5),
            r.randint(24, 190),
            r.choice(ctx["unit_len"]),
            r.randint(*ctx["price"]),
        )

    def fine(values):
        p, n, w, _ = values
        length = Fraction(p * n, 100)
        if (length * 10).denominator != 1:
            return False
        units = length / w
        return Fraction(1, 10) <= units % 1 <= Fraction(9, 10) and units > 3

    p, n, w, price = draw_until(rng, draw, fine, what="a pace count whose unit count is not near a whole number")
    length = Fraction(p * n, 100)
    units = length / w
    needed = math.ceil(units)
    total = needed * price
    check(needed - 1 < units < needed, "the unit count must need rounding up")

    stem = f"{ctx['setup'].format(p=p, n=n)} {ctx['sold'].format(w=_w_text(w))}"
    rounded_nearest = round_half_up_int(units)
    parts = [
        num_part(
            "Estimate the length, in metres, correct to 1 decimal place.",
            length,
            1,
            2,
            [f"{n} × {p} cm = {n * p} cm = {txt(length, 1)} m."],
            [
                (Fraction(n * p), "conversion-power-of-ten-slip"),
                (length * 10, "conversion-power-of-ten-slip"),
                (length / 10, "conversion-power-of-ten-slip"),
            ],
        ),
        num_part(
            f"{ctx['ask']} Use your estimate.",
            Fraction(needed),
            0,
            2,
            [f"{txt(length, 1)} ÷ {_w_text(w)} = {float(units):.2f}, so {needed} are needed (round up)."],
            [
                (Fraction(math.floor(units)), "rounded-down-when-covering"),
                (Fraction(rounded_nearest), "rounded-to-nearest-when-covering"),
                (units, "left-count-unrounded"),
                (length * w, "multiplied-instead-of-divided"),
            ],
            follow_through=f"ceiling(part_a/({w.numerator}/{w.denominator}))",
        ),
        num_part(
            f"Each {ctx['one']} costs ${price}. Find the total cost, in dollars.",
            Fraction(total),
            0,
            1,
            [f"{needed} × ${price} = ${total}."],
            [
                (Fraction(needed + price), "added-instead-of-multiplied"),
                (length * price, "priced-per-metre-not-per-unit"),
                (Fraction(price, needed), "divided-instead-of-multiplied"),
            ],
            follow_through=f"part_b*{price}",
        ),
    ]
    parts[0]["marks"], parts[1]["marks"], parts[2]["marks"] = 2, 2, 1
    return dict(text=stem, parts=parts)


def round_half_up_int(value: Fraction) -> int:
    """`value` rounded to the nearest whole number, halves up."""
    return math.floor(value + Fraction(1, 2))
