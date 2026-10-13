"""Exam-tier chain for `senior-solve-practical-problems-requiring-basic-number-operations`: a group purchase.

A group buys two kinds of item at different prices, gets a percentage discount, has GST (or a service charge) added,
and splits the total. The chain asks for

(a) the cost before the discount (two quantities times two prices),
(b) the cost after the discount,
(c) the cost after the percentage addition, and
(d) each person's share, to the nearest cent.

## Construction

Prices are drawn in whole dollars and ten cents, and the percentages are picked so that the discounted cost and the
cost after the addition are both whole cents: the discount rate is applied to a total divisible by 20 cents, the
addition is 10%. Only (d) can need rounding; it is kept clear of a half-cent tie, and the learner's route from the
stated (c) gives the same cents.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator

from .exd_x_r2i_support import clear_of_boundary, number_part, shown

# (opening, adult items, child items)
_SCENES = [
    ("A group books tickets for a show", "adult tickets", "child tickets"),
    ("A group books entry to a wildlife park", "adult entries", "child entries"),
    ("A group hires bikes for the day", "adult bikes", "child bikes"),
    ("A group books a ferry trip", "adult fares", "child fares"),
    ("A group buys passes for a swimming carnival", "adult passes", "child passes"),
]
_ADDITIONS = [(10, "10% GST is added", "GST"), (10, "a 10% service charge is added", "service charge")]
_DISCOUNTS = [5, 10, 15, 20, 25]
_PEOPLE = ["Priya", "Tom", "Mei", "Jack", "Aisha", "Liam", "Sofia", "Noah"]


def dollars(cents: int | Fraction) -> Fraction:
    return Fraction(cents) / 100


def _money(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


@generator("exd_x_r2i_group_purchase", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_r2i_group_purchase(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: two-price subtotal, percentage discount, percentage addition, then an equal split."""
    check(tier == "exam", f"unknown tier {tier!r}")
    scene, item1, item2 = rng.choice(_SCENES)
    name = rng.choice(_PEOPLE)
    addition_rate, addition_text, addition_label = rng.choice(_ADDITIONS)
    for _ in attempts(5000, "a purchase whose discounted and taxed totals are whole cents"):
        n1, n2 = rng.randint(2, 6), rng.randint(2, 6)
        p1 = rng.randrange(1200, 4200, 50)  # cents
        p2 = rng.randrange(600, p1, 50)
        people = n1  # only the adults pay, so the split is by adults, not by everyone in the group
        discount = rng.choice(_DISCOUNTS)
        subtotal = n1 * p1 + n2 * p2
        after_discount = Fraction(subtotal * (100 - discount), 100)
        if after_discount.denominator != 1 or after_discount % 10 != 0:
            continue
        total = after_discount * (100 + addition_rate) / 100
        share = Fraction(int(total), people) / 100
        if total.denominator != 1 or n1 == n2 or not clear_of_boundary(share, 2):
            continue
        break
    after_discount, total = int(after_discount), int(total)
    check(subtotal == n1 * p1 + n2 * p2 and total * 100 == after_discount * (100 + addition_rate), "totals disagree")
    stem = (
        f"{scene}: {n1} {item1} at ${_money(p1)} each and {n2} {item2} at ${_money(p2)} each. A {discount}% group "
        f"discount is taken off the cost of the items, and then {addition_text}. The final cost is shared equally "
        f"among the {n1} adults, including {name}, and the children pay nothing."
    )

    wrong_discount_dir = Fraction(subtotal * (100 + discount), 100)
    candidates_a = [
        (dollars((n1 + n2) * p1), "applied-one-price-to-all"),
        (dollars(n1 * p1 + p2), "forgot-to-multiply-by-quantity"),
        (dollars(n1 * p2 + n2 * p1), "swapped-between-groups"),
    ]
    candidates_b = [
        (dollars(wrong_discount_dir), "percentage-applied-wrong-way"),
        (dollars(subtotal) - discount, "percent-as-absolute"),
        (dollars(Fraction(subtotal * discount, 100)), "gave-percentage-amount-not-result"),
    ]
    candidates_c = [
        (dollars(Fraction(after_discount * (100 - addition_rate), 100)), "percentage-applied-wrong-way"),
        (dollars(Fraction(after_discount * addition_rate, 100)), "gave-percentage-amount-not-result"),
        (dollars(after_discount) + addition_rate, "percent-as-absolute"),
    ]
    exact_share = Fraction(total, people) / 100
    candidates_d = [
        (dollars(total) / (n1 + n2), "divided-by-wrong-amount"),
        (dollars(total) * people, "multiplies-instead-of-dividing"),
        (dollars(after_discount) / people, "left-out-a-step"),
        (dollars(subtotal) / people, "left-out-a-step"),
    ]
    parts = [
        number_part(
            "Find the cost of the items before any discount, in dollars.",
            dollars(subtotal),
            2,
            2,
            [f"{n1} × {_money(p1)} + {n2} × {_money(p2)} = {_money(subtotal)}."],
            candidates_a,
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the cost after the {discount}% discount, in dollars.",
            dollars(after_discount),
            2,
            1,
            [f"{_money(subtotal)} × {(100 - discount) / 100} = {_money(after_discount)}."],
            candidates_b,
            follow_through=f"part_a*{100 - discount}/100",
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the cost after the {addition_rate}% {addition_label} is added, in dollars.",
            dollars(total),
            2,
            1,
            [f"{_money(after_discount)} × {(100 + addition_rate) / 100} = {_money(total)}."],
            candidates_c,
            follow_through=f"part_b*{100 + addition_rate}/100",
            sensible_range=(0, None),
        ),
        number_part(
            "How much does each adult pay, to the nearest cent?",
            exact_share,
            2,
            2,
            [f"{_money(total)} ÷ {people} = {shown(exact_share, 2)}."],
            candidates_d,
            follow_through=f"part_c/{people}",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
