"""Exam tier for "estimate a percentage of a quantity": estimate a sale price from a rounded price.

The practice tiers estimate one benchmark percentage (10%, 25% or 50%) of a given quantity. Here the percentage is
a sale discount that is NOT a benchmark (15%, 35%, 65%, ...), so the learner has to build it from benchmarks
(10% and 5%, 25% and 10%, 50% and 15%) and, before that, round the price so the benchmarks come out in whole
dollars. Three connected parts:

- (a) round the price to the nearest $20,
- (b) estimate the discount on the rounded price,
- (c) estimate the sale price.

Parts (b) and (c) use the earlier answers, so they carry `follow_through`. A multiple of $20 is used because any
percentage that is a multiple of 5 then gives a whole number of dollars.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors, round_half_up

_ITEMS = [
    ("mountain bike", "Canberra"),
    ("surfboard", "Torquay"),
    ("pair of hiking boots", "Katoomba"),
    ("laptop", "Brisbane"),
    ("camping tent", "Alice Springs"),
    ("cricket bat set", "Hobart"),
    ("classical guitar", "Fremantle"),
    ("fishing rod and reel", "Port Macquarie"),
]
_PERCENTS = [15, 35, 45, 55, 65, 75]
_BENCHMARKS = [50, 25, 10]


def _nearest_20(price: int) -> int:
    return ((price + 10) // 20) * 20


def _partial_benchmark(p: int) -> int:
    """The largest benchmark percentage (50, 25, 10) not above `p`, a learner's 'just the first piece'."""
    return next(b for b in _BENCHMARKS if b <= p)


@generator("exd_x_pr7_sale_price_estimate", level=10, presets={"exam": {"tier": "exam"}})
def exd_x_pr7_sale_price_estimate(rng: random.Random, tier: str = "exam") -> dict:
    """Round a price, estimate a non-benchmark discount, estimate the sale price. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a price that is neither a multiple of 20 nor halfway between two"):
        price = rng.randint(150, 990)
        percent = rng.choice(_PERCENTS)
        rounded = _nearest_20(price)
        # A price ending in 10 (mod 20) sits exactly between two multiples of 20, and the stem does not say which way
        # a half rounds, so those prices are never drawn.
        if price % 20 not in (0, 10) and rounded != price:
            break
    discount = Fraction(rounded * percent, 100)
    check(discount.denominator == 1, "the discount must be a whole number of dollars")
    discount, sale = int(discount), rounded - int(discount)
    # Second route: the discount as sums of benchmark pieces (10% pieces and 5% pieces of the rounded price).
    tens, rest = divmod(percent, 10)
    check(discount == tens * (rounded // 10) + (rounded // 20 if rest == 5 else 0), "benchmark pieces disagree")
    check(round_half_up(Fraction(price, 20)) * 20 == rounded, "rounding disagrees")

    item, town = rng.choice(_ITEMS)
    stem = (
        f"A {item} in {town} is priced at ${price}. During a sale it is advertised with {percent}% off. "
        "To estimate the sale price, round the price to the nearest $20 first."
    )

    def whole(value: int, tag: str | None) -> Distractor | str:
        return Distractor(str(value), tag, value) if tag else str(value)

    unrounded_discount = int(round_half_up(Fraction(price * percent, 100)))
    partial = rounded * _partial_benchmark(percent) // 100
    a_options = pick_distractors(
        str(rounded),
        [
            whole((price // 20) * 20, "rounded-down-not-to-nearest"),
            whole(int(round_half_up(Fraction(price, 10))) * 10, None),
            whole(int(round_half_up(Fraction(price, 100))) * 100, None),
            whole(-(-price // 20) * 20 + 20, None),  # rounded up past the nearest multiple
            whole(int(round_half_up(Fraction(price, 50))) * 50, None),
            whole(rounded + 20, None),  # the other neighbouring multiple of 20
            whole(rounded - 20, None),
        ],
        3,
    )
    b_options = pick_distractors(
        str(discount),
        [
            whole(partial, "partial-benchmark-only"),
            *([whole(unrounded_discount, "used-unrounded-price")] if abs(unrounded_discount - discount) >= 3 else []),
            whole(rounded * percent // 1000, "decimal-place-shift"),
        ],
        3,
    )
    c_options = pick_distractors(
        str(sale),
        [
            whole(discount, "reports-discount-not-sale-price"),
            whole(rounded + discount, "added-discount-to-price"),
            *([whole(price - discount, "used-unrounded-price")] if abs(price - discount - sale) >= 3 else []),
        ],
        3,
    )
    check(
        min(len(a_options), len(b_options), len(c_options)) >= 2,
        f"too few distractors {a_options} {b_options} {c_options}",
    )

    def part(prompt, value, marks, options, working, follow_through=None):
        return dict(
            prompt=prompt,
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            answer_latex=latex_answer(str(value)),
            marks=marks,
            sensible_range=(0, None),
            working=working,
            distractors=options,
            **({"follow_through": follow_through} if follow_through else {}),
        )

    return dict(
        text=stem,
        parts=[
            part(
                "What is the price rounded to the nearest $20, in dollars?",
                rounded,
                1,
                a_options,
                [f"${price} is closest to ${rounded}."],
            ),
            part(
                "Estimate the discount on the rounded price, in dollars.",
                discount,
                2,
                b_options,
                [f"{percent}% of ${rounded} = ${discount}."],
                f"part_a * {percent} / 100",
            ),
            part(
                "Estimate the sale price, in dollars.",
                sale,
                1,
                c_options,
                [f"${rounded} - ${discount} = ${sale}."],
                "part_a - part_b",
            ),
        ],
    )
