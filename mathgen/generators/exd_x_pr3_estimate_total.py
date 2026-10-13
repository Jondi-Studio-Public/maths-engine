"""Exam source for estimating a financial total: estimate with quantities, find the exact total, compare.

Three lines of quantity times unit price. Part (a) estimates the total by rounding each unit price to the nearest
dollar first, part (b) finds the exact total, and part (c) finds how far the estimate is from the exact total. Part
(c) carries `follow_through`. No unit price ends in 50 cents, so rounding to the nearest dollar is never a tie.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator

from ._exd_pr3_support import leaks, part

# (opening clause, [(item, lowest whole-dollar price, highest whole-dollar price), ...])
_SCENES = [
    (
        "A school in Canberra orders supplies for a camp",
        [("sleeping mats", 12, 29), ("torches", 5, 19), ("first-aid kits", 6, 18)],
    ),
    (
        "A netball club in Geelong buys gear for the season",
        [("netballs", 16, 29), ("bibs", 5, 14), ("water bottles", 4, 12)],
    ),
    (
        "A cafe in Fremantle restocks",
        [("jars of coffee", 9, 22), ("boxes of tea", 4, 12), ("bags of sugar", 2, 6)],
    ),
    (
        "A scout group in Toowoomba shops for a fundraiser",
        [("tins of paint", 18, 45), ("brushes", 4, 14), ("drop sheets", 3, 9)],
    ),
]


@generator("exd_x_pr3_estimate_total", level=8, presets={"exam": dict(tier="exam")})
def exd_x_pr3_estimate_total(rng, tier="exam"):
    check(tier == "exam", "only the exam tier exists")
    for _ in attempts(2000, "prices with a nonzero gap between estimate and exact total"):
        scene, catalogue = rng.choice(_SCENES)
        items = [name for name, _, _ in catalogue]
        qty = [rng.randint(3, 9) for _ in items]
        dollars = [rng.randint(lo, hi) for _, lo, hi in catalogue]
        cents = [rng.choice([c for c in range(5, 100) if c != 50]) for _ in items]
        prices = [Fraction(d * 100 + c, 100) for d, c in zip(dollars, cents)]
        rounded = [d + (1 if c > 50 else 0) for d, c in zip(dollars, cents)]
        estimate = sum(q * r for q, r in zip(qty, rounded))
        exact = sum(q * p for q, p in zip(qty, prices))
        gap = abs(estimate - exact)
        if gap < 1:
            continue
        lines = ", ".join(f"{q} {name} at ${p:.2f} each" for q, name, p in zip(qty, items, prices))
        stem = f"{scene}: {lines}."
        if not leaks(stem, [estimate, exact, gap]):
            break
    check(estimate == sum(q * round(p) for q, p in zip(qty, prices)), "estimate disagrees with rounding each price")
    truncated = sum(q * d for q, d in zip(qty, dollars))
    sum_unit_rounded = sum(rounded)
    parts = [
        part(
            "Estimate the total cost by rounding each price to the nearest dollar first. Give your answer in dollars.",
            estimate,
            marks=2,
            working=[
                "Rounded prices: " + ", ".join(f"${r}" for r in rounded) + ".",
                " + ".join(f"{q} × {r}" for q, r in zip(qty, rounded)) + f" = {estimate}.",
            ],
            candidates=[
                (sum_unit_rounded, "ignored-quantity"),
                (truncated, "truncates-prices"),
                (round(exact), "rounds-total-not-prices"),
            ],
        ),
        part(
            "Find the exact total cost. Give your answer in dollars.",
            exact,
            dp=2,
            marks=2,
            working=[" + ".join(f"{q} × {p:.2f}" for q, p in zip(qty, prices)) + f" = {exact:.2f}."],
            candidates=[
                (sum(prices), "ignored-quantity"),
                (Fraction(estimate), "estimate-as-exact"),
                (sum(q * p for q, p in list(zip(qty, prices))[:2]), "omits-an-item"),
            ],
        ),
        part(
            "By how much does the estimate in (a) differ from the exact total in (b)? Give the size of the "
            "difference in dollars.",
            gap,
            dp=2,
            marks=1,
            follow_through="Abs(part_a-part_b)",
            working=[f"|{estimate} − {exact:.2f}| = {gap:.2f}."],
            candidates=[
                (estimate + exact, "adds-instead-of-subtracts"),
                (-gap, "reversed-subtraction"),
                (abs(truncated - exact), "truncates-prices"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
