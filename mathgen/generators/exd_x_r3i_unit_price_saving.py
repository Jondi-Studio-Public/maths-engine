"""Exam-tier source for `compare-unit-prices`: two packs in different units, priced per 100 g (or 100 mL), then
a saving on a bulk order.

Two packs of the same grocery item are priced in different units (a small pack in grams or millilitres, a large one
in kilograms or litres). The parts carry the comparison through to a dollar amount:

(a) the cost of pack A per 100 g (or 100 mL), in dollars      number, 1 mark
(b) the cost of pack B per 100 g (or 100 mL), in dollars      number, 2 marks
(c) how much cheaper per 100 g the cheaper pack is            number, 1 mark,  |part_a - part_b|
(d) the saving on a bulk order bought as the cheaper pack     number, 2 marks, part_c * kilograms / 10

## Construction

Both unit prices are drawn first, as a whole number of cents per 100 g, and the pack prices are built from them
(the large pack's price is `unit * 10 * kilograms`; the small pack's is `unit * grams / 100`, kept to a whole number
of cents). The two unit prices differ by at least three cents, so the cheaper pack is never in doubt and no part
rests on a rounded comparison. Every answer is an exact number of cents.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_x_r3_distinct import distinct_answers
from .exd_x_r3i_support import number_part

#: (item, family, plural unit label for the 100-unit price, bulk-order unit, price range in cents per 100 units)
_ITEMS = (
    ("basmati rice", "mass", "g", "kg", (25, 80)),
    ("plain flour", "mass", "g", "kg", (12, 40)),
    ("rolled oats", "mass", "g", "kg", (18, 55)),
    ("cheddar cheese", "mass", "g", "kg", (110, 320)),
    ("ground coffee", "mass", "g", "kg", (200, 480)),
    ("frozen peas", "mass", "g", "kg", (20, 60)),
    ("canola oil", "volume", "mL", "L", (30, 90)),
    ("orange juice", "volume", "mL", "L", (12, 36)),
    ("laundry liquid", "volume", "mL", "L", (15, 45)),
    ("dishwashing liquid", "volume", "mL", "L", (14, 40)),
)
_SMALL = (250, 300, 400, 500, 600, 750, 800, 900)
_LARGE = (Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3), Fraction(4), Fraction(5))


def _money(cents: Fraction) -> str:
    cents = Fraction(cents)
    check(cents.denominator == 1, "a price is not a whole number of cents")
    return f"${cents.numerator // 100}.{cents.numerator % 100:02d}"


def _kg_text(value: Fraction) -> str:
    return str(int(value)) if value.denominator == 1 else f"{float(value):.1f}"


@generator("exd_x_r3i_unit_price_saving", level=7, presets={"exam": dict(tier="exam")})
@distinct_answers
def exd_x_r3i_unit_price_saving(rng, tier: str = "exam") -> dict:
    """Unit prices of two packs in different units, their gap, and the saving on a bulk order."""
    check(tier == "exam", f"unknown tier {tier!r}")
    item, family, small_unit, big_unit, (lo, hi) = rng.choice(_ITEMS)
    for _ in attempts(1000, "two unit prices with whole-cent pack prices"):
        u_small, u_large = rng.randint(lo, hi), rng.randint(lo, hi)
        grams = rng.choice(_SMALL)
        big = rng.choice(_LARGE)
        if abs(u_small - u_large) < max(3, lo // 6) or (u_small * grams) % 100 != 0:
            continue
        break
    price_small = Fraction(u_small * grams, 100)
    price_large = Fraction(u_large) * 10 * big
    check(price_small.denominator == 1 and price_large.denominator == 1, "pack prices are not whole cents")
    # Independent route back from the printed prices to the unit prices.
    check(price_small * 100 / grams == u_small, "small pack unit price disagrees")
    check(price_large * 100 / (big * 1000) == u_large, "large pack unit price disagrees")

    small_first = rng.random() < 0.5
    small_label, large_label = ("A", "B") if small_first else ("B", "A")
    small_pack = f"{grams} {small_unit} for {_money(price_small)}"
    large_pack = f"{_kg_text(big)} {big_unit} for {_money(price_large)}"
    packs = {small_label: small_pack, large_label: large_pack}
    stem = f"A supermarket sells {item} in two packs. Pack A: {packs['A']}. Pack B: {packs['B']}."
    u = {small_label: u_small, large_label: u_large}
    p = {small_label: price_small, large_label: price_large}
    q_units = {small_label: Fraction(grams), large_label: big * 1000}
    gap = abs(u_small - u_large)
    cheaper = min(u_small, u_large)
    order = rng.choice([6, 8, 10, 12, 15, 18, 20, 24, 25, 30, 36, 40, 50])
    per = f"100 {small_unit}"

    def unit_part(label, marks):
        key = Fraction(u[label], 100)
        other = "B" if label == "A" else "A"
        return number_part(
            f"Find the cost of Pack {label} per {per}, in dollars, correct to 2 decimal places.",
            key,
            2,
            marks,
            [
                f"Pack {label} is {packs[label]}: "
                f"{_money(p[label])} ÷ ({q_units[label]} {small_unit} ÷ 100) = {float(key):.2f} dollars per {per}."
            ],
            [
                ("wrong-unit-conversion-factor", Fraction(u[label] * 10, 100)),
                ("unit-price-of-the-other-pack", Fraction(u[other], 100)),
            ],
            sensible_range=(0, None),
        )

    part_a = unit_part("A", 1)
    part_b = unit_part("B", 2)
    gap_key = Fraction(gap, 100)
    part_c = number_part(
        f"How much cheaper per {per} is the cheaper pack than the dearer pack? Give the difference in dollars, "
        "correct to 2 decimal places.",
        gap_key,
        2,
        1,
        [f"|{float(Fraction(u['A'], 100)):.2f} - {float(Fraction(u['B'], 100)):.2f}| = {float(gap_key):.2f}."],
        [
            ("added-instead-of-subtracted", Fraction(u_small + u_large, 100)),
            ("compares-pack-totals-not-unit-prices", abs(price_small - price_large) / 100),
            ("wrong-unit-conversion-factor", Fraction(gap, 10)),
            ("sign-of-result-reversed", -gap_key),
        ],
        follow_through="Abs(part_a - part_b)",
        sensible_range=(0, None),
    )
    saving = Fraction(gap * order, 10)
    part_d = number_part(
        f"A school canteen needs {order} {big_unit} of {item}. How much does it save by buying all of it as the "
        "cheaper pack rather than the dearer pack? Give the saving in dollars, correct to 2 decimal places.",
        saving,
        2,
        2,
        [
            f"{order} {big_unit} is {order * 10} lots of {per}, so the saving is {float(gap_key):.2f} × {order * 10} = "
            f"{float(saving):.2f} dollars."
        ],
        [
            ("wrong-unit-conversion-factor", saving * 10),
            ("wrong-unit-conversion-factor", saving / 10),
            ("reports-cost-instead-of-saving", Fraction(cheaper * order, 10)),
            ("compares-pack-totals-not-unit-prices", abs(price_small - price_large) / 100 * order),
        ],
        follow_through=f"part_c*{order * 10}",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
