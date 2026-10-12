"""Exam source for money change: a two-item shop, change from a note, and a second purchase from that change.

Part (a) finds the total of two lines of quantity times unit price, part (b) takes it from the note, and part (c)
spends some of the change on goods sold by the kilogram and asks what is left. Parts (b) and (c) carry
`follow_through`. Every price is a multiple of 5 cents, as in Australian shops, so every amount is exact.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator

from ._exd_pr3_support import leaks, part

_SHOPS = [
    ("At a bakery in Hobart", "meat pies", "bottles of milk"),
    ("At a kiosk at Bondi Beach", "iced coffees", "sausage rolls"),
    ("At a canteen in Alice Springs", "toasted sandwiches", "bottles of water"),
    ("At a deli in Fremantle", "lamingtons", "bottles of orange juice"),
]
_BUYERS = ["Mia", "Tom", "Aisha", "Liam", "Priya", "Jack"]
_FRUIT = ["apples", "mandarins", "bananas", "pears"]


def _dollars(cents: int) -> Fraction:
    return Fraction(cents, 100)


@generator("exd_x_pr3_money_change", level=8, presets={"exam": dict(tier="exam")})
def exd_x_pr3_money_change(rng, tier="exam"):
    check(tier == "exam", "only the exam tier exists")
    for _ in attempts(2000, "a purchase whose answers are not in the prompt"):
        shop, item1, item2 = rng.choice(_SHOPS)
        n1, n2 = rng.randint(2, 5), rng.randint(2, 4)
        c1, c2 = 5 * rng.randint(60, 110), 5 * rng.randint(30, 80)  # cents
        kg, rate_c = rng.randint(2, 3), 5 * rng.randint(50, 120)
        total_c = n1 * c1 + n2 * c2
        spend_c = kg * rate_c
        note = rng.choice((20, 50, 100))
        change_c = note * 100 - total_c
        if change_c <= spend_c + 500 or change_c > note * 100 * 3 // 4 and note != 20:
            continue
        who = rng.choice(_BUYERS)
        fruit = rng.choice(_FRUIT)
        stem = (
            f"{shop}, {who} buys {n1} {item1} at ${_dollars(c1):.2f} each and {n2} {item2} at "
            f"${_dollars(c2):.2f} each. {who} pays with a ${note} note. "
            f"{who} then spends part of the change on {kg} kg of {fruit} at ${_dollars(rate_c):.2f} per kilogram."
        )
        keys = [_dollars(total_c), _dollars(change_c), _dollars(change_c - spend_c), _dollars(spend_c)]
        if not leaks(stem, keys):
            break
    total, change, left, spend = (_dollars(v) for v in (total_c, change_c, change_c - spend_c, spend_c))
    p1, p2 = _dollars(c1), _dollars(c2)
    check(total == n1 * p1 + n2 * p2 and change == note - total and left == change - spend, "money chain disagrees")
    unit = "Give your answer in dollars."
    parts = [
        part(
            f"Find the total cost of the {item1} and the {item2}. {unit}",
            total,
            dp=2,
            marks=2,
            working=[f"{n1} × {p1:.2f} + {n2} × {p2:.2f} = {n1 * p1:.2f} + {n2 * p2:.2f} = {total:.2f}."],
            candidates=[
                (p1 + p2, "ignored-quantity"),
                (n1 * p1, "omits-an-item"),
                (n2 * p2, "omits-an-item"),
            ],
        ),
        part(
            f"Using your answer to (a), find the change from the ${note} note. {unit}",
            change,
            dp=2,
            marks=1,
            follow_through=f"{note}-part_a",
            working=[f"{note} − {total:.2f} = {change:.2f}."],
            candidates=[
                (note + total, "adds-price"),
                (total, "reports-cost-not-change"),
                (note - n1 * p1, "omits-an-item"),
            ],
        ),
        part(
            f"How much of the change is left after buying the {fruit}? Use your answer to (b). {unit}",
            left,
            dp=2,
            marks=2,
            follow_through=f"part_b-{spend}",
            working=[f"{kg} × {_dollars(rate_c):.2f} = {spend:.2f}.", f"{change:.2f} − {spend:.2f} = {left:.2f}."],
            candidates=[
                (change + spend, "adds-price"),
                (change - kg, "quantity-not-cost"),
                (change - _dollars(rate_c), "unit-price-not-total"),
                (note - spend, "ignored-first-purchase"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
