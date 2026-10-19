"""Exam source for percentage discount: working back from the sale price to the original price.

The stem gives only the percentage off and the price paid. Part (a) finds the percentage of the original price that
was paid, part (b) uses it to find the original price, and part (c) finds the saving. Parts (b) and (c) carry
`follow_through`. Built backwards from a whole-dollar original price and a percentage that leaves a whole-dollar
sale price.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator

from ._exd_pr3_support import leaks, part

# (item and shop, buyer, lowest and highest original price in units of $5)
_SCENES = [
    ("A pair of running shoes at a sports store in Newcastle", "Jai", 10, 50),
    ("A school backpack at a shop in Toowoomba", "Mei", 8, 40),
    ("A cricket bat at a store in Adelaide", "Sam", 10, 60),
    ("A wetsuit at a surf shop in Torquay", "Ruby", 20, 80),
    ("A bicycle helmet at a shop in Darwin", "Noah", 6, 30),
]
_PERCENTS = (10, 15, 20, 25, 30, 35, 40, 45, 50, 60)


@generator("exd_x_pr3_discount_reverse", level=8, presets={"exam": dict(tier="exam")})
def exd_x_pr3_discount_reverse(rng, tier="exam"):
    check(tier == "exam", "only the exam tier exists")
    for _ in attempts(2000, "an original price whose sale price is a whole number of dollars"):
        p = rng.choice(_PERCENTS)
        scene, who, lo, hi = rng.choice(_SCENES)
        original = 5 * rng.randint(lo, hi)
        if original * (100 - p) % 100:
            continue
        sale = original * (100 - p) // 100
        stem = f"{scene} is on sale at {p}% off. {who} pays ${sale} for it."
        if not leaks(stem, [100 - p, original, original - sale]):
            break
    saving = original - sale
    check(
        Fraction(sale) == Fraction(original) * (100 - p) / 100 and saving == Fraction(original * p, 100),
        "chain disagrees",
    )
    paid_pct = 100 - p
    parts = [
        part(
            f"What percentage of the original price does {who} pay?",
            paid_pct,
            marks=1,
            sensible_range=(0, 100),
            working=[f"100% − {p}% = {paid_pct}%."],
            candidates=[
                (p, "gave-discount-percent"),
                (100 + p, "adds-discount"),
                (sale, "count-given-as-percent"),
            ],
        ),
        part(
            "Use your answer to (a) to find the original price of the item, in dollars, correct to the nearest cent.",
            original,
            dp=2,
            marks=2,
            follow_through=f"{sale}*100/part_a",
            working=[
                f"{paid_pct}% of the original is ${sale}.",
                f"1% is ${sale} ÷ {paid_pct} = ${original / 100:.2f}, so 100% is ${original}.",
            ],
            candidates=[
                (Fraction(sale * (100 + p), 100), "percent-of-sale-price"),
                (sale + p, "percent-as-absolute"),
                (Fraction(sale * paid_pct, 100), "multiplies-sale-by-percent-paid"),
            ],
        ),
        part(
            f"Using your answer to (b), find how much {who} saves, in dollars, correct to the nearest cent.",
            saving,
            dp=2,
            marks=1,
            follow_through=f"part_b-{sale}",
            working=[f"${original} − ${sale} = ${saving}."],
            candidates=[
                (Fraction(sale * p, 100), "percent-of-sale-price"),
                (original + sale, "adds-instead-of-subtracts"),
                (sale, "reports-price-paid-not-saving"),
                (p, "percent-as-absolute"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
