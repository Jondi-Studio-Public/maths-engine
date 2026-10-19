"""Exam chain `exd_x_r3j_ratio_shares` for `divide-quantity-in-ratio`: three shares, with the difference between two of
them given instead of the total.

A sum of money is shared among three people in the ratio a : b : c. The stem gives how much more the largest share is
than the smallest, never the total. The chain finds the value of one part, the total, the middle share and the
percentage of the total that the smallest share is, each following through from the earlier parts.

The value of a part is rebuilt a second way, from the shares written out one by one.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3j_kit import PEOPLE, dollars, number_part, number_text

_CONTEXTS = [
    ("a lottery prize", "wins"),
    ("the profit from a market stall", "makes"),
    ("an inheritance", "receives"),
    ("the prize money at a quiz night", "wins"),
    ("a club's fundraising total", "raises"),
]


def _parts(n):
    return f"{n} part" if n == 1 else f"{n} parts"


@generator("exd_x_r3j_ratio_shares", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3j_ratio_shares(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a ratio and a part value that give a clean percentage"):
        a, b, c = sorted(rng.sample(range(1, 10), 3))
        if c - a < 2 or (a, b, c) == (1, 2, 3):
            continue
        unit = rng.choice([12, 15, 18, 20, 24, 25, 30, 35, 40, 45, 50, 60, 75])
        total = unit * (a + b + c)
        pct = Fraction(100 * a, a + b + c)
        scaled = pct * 10
        if (scaled - int(scaled)) * 2 not in (1, -1):
            break
    names = rng.sample(PEOPLE, 3)
    thing, verb = rng.choice(_CONTEXTS)
    diff = (c - a) * unit
    shares = [x * unit for x in (a, b, c)]
    check(sum(shares) == total, "shares do not add to the total")
    text = (
        f"{names[0]}, {names[1]} and {names[2]} share {thing} in the ratio {a} : {b} : {c}. {names[2]}'s share is "
        f"{dollars(diff)} more than {names[0]}'s share."
    )
    parts = [
        number_part(
            "Find the value of one part of the ratio, in dollars.",
            Fraction(unit),
            0,
            2,
            [
                f"{names[2]}'s share is {_parts(c)} and {names[0]}'s is {_parts(a)}, so {_parts(c - a)} = {dollars(diff)}.",
                f"One part = {dollars(diff)} / {c - a} = {dollars(unit)}.",
            ],
            [
                (Fraction(diff, c + a), "r3j-ratio-difference-divided-by-sum"),
                (Fraction(diff, c), "r3j-ratio-difference-divided-by-largest-term"),
                (Fraction(diff, a + b + c), "r3j-ratio-difference-divided-by-total-parts"),
                (Fraction(diff, c - a + 1), "r3j-ratio-difference-divided-by-terms-counted"),
            ],
        ),
        number_part(
            "Find the total amount shared, in dollars.",
            Fraction(total),
            0,
            2,
            [
                f"Total parts = {a} + {b} + {c} = {a + b + c}.",
                f"Total = {a + b + c} x {dollars(unit)} = {dollars(total)}.",
            ],
            [
                (Fraction(unit * (c - a)), "r3j-ratio-difference-given-as-total"),
                (Fraction(unit * 3), "r3j-ratio-parts-counted-as-people"),
                (Fraction(unit * (a + b)), "r3j-ratio-two-shares-only"),
                (Fraction(unit * c * 3), "r3j-ratio-largest-share-tripled"),
            ],
            follow_through=f"{a + b + c}*part_a",
        ),
        number_part(
            f"Find {names[1]}'s share, in dollars.",
            Fraction(b * unit),
            0,
            1,
            [f"{names[1]}'s share = {b} x {dollars(unit)} = {dollars(b * unit)}."],
            [
                (Fraction(total, b), "r3j-ratio-total-divided-by-term"),
                (Fraction(total, 3), "r3j-ratio-equal-shares"),
                (Fraction(unit * (b - a)), "r3j-ratio-difference-of-terms-used"),
                (Fraction(unit * (a + c)), "r3j-ratio-other-two-shares-added"),
            ],
            follow_through=f"{b}*part_a",
        ),
        number_part(
            f"What percentage of the total is {names[0]}'s share? Give your answer to 1 decimal place.",
            pct,
            1,
            2,
            [
                f"{names[0]}'s share = {dollars(a * unit)}; {dollars(a * unit)} / {dollars(total)} x 100 = {number_text(pct, 1)}%."
            ],
            [
                (Fraction(100 * a, c), "r3j-ratio-percentage-of-largest"),
                (Fraction(100 * a, b), "r3j-ratio-percentage-of-middle"),
                (Fraction(100, 3), "r3j-ratio-equal-shares"),
                (Fraction(a, a + b + c), "r3j-ratio-fraction-not-percentage"),
            ],
            follow_through=f"100*{a}*part_a/part_b",
        ),
    ]
    return dict(text=text, parts=parts)
