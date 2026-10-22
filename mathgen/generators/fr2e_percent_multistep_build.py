"""Generates `f10-financial-multistep-percent` questions and its exam chain. `answer: number`.

Scope: Years 5 and 6 money problems that chain percentages (10%, 20%, 25%, 50% discounts, 10% GST) with unit prices.
Every price is exact to the cent by construction, and every answer is typed in dollars and cents. Compound interest,
profit and loss and percentage increase and decrease on their own are out of scope (other skills cover them).

Ladder (reasoning escalation, not size):
  easy:   a sale price after one percentage discount. The stem never says "subtract" or "find the discount".
  medium: a discount and then GST on the discounted price. The two percentages must not be added or applied to
          the same base.
  hard:   a best buy. Two offers of different pack sizes, one of them discounted; the answer is how much less the
          better offer costs per 100 g, so the discount, the pack size and a comparison all have to be handled.
  exam:   a four-part chain on one purchase from two shops: the discounted price at the first (2 marks), the price
          with GST at the second (2), the better shop and the saving (multiple choice, 2), then the cost of several
          items from the better shop (follows through from (a) and (b)).
"""

from __future__ import annotations

from fractions import Fraction

from functools import partial

from ..core import check, generator, phrase
from ._fr2e_kit import Redraw, dollars, draw, exact_number_part, mcq_part, single

_NAMES = ("Priya", "Marcus", "Aiko", "Tom", "Leila", "Sam", "Jess", "Kofi", "Mia", "Dev", "Ella", "Hamish")
_GOODS = (
    ("jacket", "jackets"),
    ("backpack", "backpacks"),
    ("pair of runners", "pairs of runners"),
    ("bike helmet", "bike helmets"),
    ("cricket bat", "cricket bats"),
    ("pair of headphones", "pairs of headphones"),
    ("school bag", "school bags"),
    ("tent", "tents"),
)
_FOODS = (
    ("rice", "500 g box"),
    ("pasta", "packet"),
    ("muesli", "bag"),
    ("cheese", "block"),
    ("flour", "bag"),
    ("cereal", "box"),
    ("mixed nuts", "tub"),
)
_DISCOUNTS = (10, 20, 25, 50)
dollars = partial(dollars, cents=True)
_ASK = "Give your answer in dollars and cents."

E_DISCOUNT_ONLY = "reports-discount-not-sale-price"
E_ADDED = "added-discount-to-price"
E_ABSOLUTE = "percent-as-absolute"
E_GST_ORIGINAL = "fr2e-percent-gst-on-original-price"
E_GST_SKIPPED = "fr2e-percent-gst-left-out"
E_DISCOUNT_SKIPPED = "fr2e-percent-discount-left-out"
E_PERCENTS_ADDED = "fr2e-percent-net-of-two-percents"
E_UNIT_BEFORE = "fr2e-percent-unit-price-before-discount"
E_PACK_PRICES = "fr2e-percent-pack-prices-compared"
E_PER_KG = "fr2e-percent-unit-price-per-wrong-mass"
E_WRONG_SHOP = "fr2e-percent-better-shop-reversed"


def _pct(d: int) -> str:
    return f"{d}%"


def _exact_cents(value: Fraction) -> bool:
    return (value * 100).denominator == 1


def _price(rng, step: int = 4, lo: int = 40, hi: int = 240) -> int:
    return rng.randrange(lo, hi + 1, step)


def _easy(rng):
    name = rng.choice(_NAMES)
    good, _ = rng.choice(_GOODS)
    d = rng.choice(_DISCOUNTS)
    price = _price(rng)
    cut = Fraction(price * d, 100)
    sale = price - cut
    check(sale == Fraction(price) * (100 - d) / 100, "the sale price must agree on both routes")
    stem = phrase(
        rng,
        (
            "{name} sees a {good} marked ${price} with {d} off.",
            "A {good} is marked ${price}. This week every price in the shop is reduced by {d}, and {name} buys it.",
        ),
        name=name,
        good=good,
        price=price,
        d=_pct(d),
    )
    part = exact_number_part(
        f"Find how much {name} pays.",
        sale,
        1,
        [f"{d}% of ${price} = {dollars(cut)}.", f"${price} − {dollars(cut)} = {dollars(sale)}."],
        [
            (cut, E_DISCOUNT_ONLY),
            (price + cut, E_ADDED),
            (price - d, E_ABSOLUTE),
            (price - 2 * cut if price - 2 * cut > 0 else None, "fr2e-percent-discount-applied-twice"),
        ],
        min_places=2,
        suffix=_ASK,
    )
    return single(stem, part)


def _medium(rng):
    name = rng.choice(_NAMES)
    good, _ = rng.choice(_GOODS)
    d = rng.choice(_DISCOUNTS)
    price = _price(rng, 8)
    sale = Fraction(price * (100 - d), 100)
    paid = sale * Fraction(11, 10)
    if not _exact_cents(paid) or paid == sale:
        raise Redraw("the GST must come to a whole number of cents")
    gst = paid - sale
    stem = (
        f"A {good} is marked ${price} and is on sale for {_pct(d)} off. {name} buys it, and 10% GST is added to the "
        "sale price."
    )
    wrong_gst_original = sale + Fraction(price, 10)
    part = exact_number_part(
        f"Find how much {name} pays in total.",
        paid,
        1,
        [
            f"Sale price: ${price} × {100 - d}% = {dollars(sale)}.",
            f"GST: 10% of {dollars(sale)} = {dollars(gst)}.",
            f"Total: {dollars(sale)} + {dollars(gst)} = {dollars(paid)}.",
        ],
        [
            (wrong_gst_original, E_GST_ORIGINAL),
            (sale, E_GST_SKIPPED),
            (Fraction(price) * Fraction(11, 10), E_DISCOUNT_SKIPPED),
            (Fraction(price) * (Fraction(100 - d, 100) + Fraction(1, 10)), E_PERCENTS_ADDED),
        ],
        min_places=2,
        suffix=_ASK,
    )
    return single(stem, part)


def _offers(rng):
    """Two offers of the same food: pack A at a plain price, pack B discounted. Returns the numbers."""
    unit_a = Fraction(rng.randrange(35, 120, 5), 100)  # dollars per 100 g, exact
    mass_a = rng.choice((200, 250, 400, 500))
    unit_b_sale = unit_a + Fraction(rng.choice((-1, 1)) * rng.randrange(5, 31, 5), 100)
    mass_b = rng.choice([m for m in (250, 400, 500, 750, 1000) if m != mass_a])
    d = rng.choice((10, 20, 25))
    price_a = unit_a * mass_a / 100
    sale_b = unit_b_sale * mass_b / 100
    marked_b = sale_b * 100 / (100 - d)
    if not (
        _exact_cents(price_a)
        and _exact_cents(marked_b)
        and _exact_cents(sale_b)
        and unit_b_sale > 0
        and unit_a != unit_b_sale
    ):
        raise Redraw("prices must be exact to the cent")
    return unit_a, mass_a, price_a, unit_b_sale, mass_b, marked_b, sale_b, d


def _hard(rng):
    name = rng.choice(_NAMES)
    food, pack = rng.choice(_FOODS)
    unit_a, mass_a, price_a, unit_b, mass_b, marked_b, sale_b, d = draw(_offers, rng)
    cheaper = abs(unit_a - unit_b)
    unit_b_marked = marked_b * 100 / mass_b
    stem = (
        f"{name} compares two packs of {food}. Pack A is {mass_a} g and costs {dollars(price_a)}. Pack B is "
        f"{mass_b} g and is marked {dollars(marked_b)}, with {_pct(d)} off this week."
    )
    part = exact_number_part(
        "Find how much less per 100 g the cheaper pack costs than the other, in dollars.",
        cheaper,
        1,
        [
            f"Pack A: {dollars(price_a)} ÷ {mass_a // 100 if mass_a % 100 == 0 else mass_a / 100} hundred grams = {dollars(unit_a)} per 100 g.",
            f"Pack B sale price: {dollars(marked_b)} × {100 - d}% = {dollars(sale_b)}, which is {dollars(unit_b)} per 100 g.",
            f"The difference is {dollars(cheaper)} per 100 g.",
        ],
        [
            (abs(unit_a - unit_b_marked), E_UNIT_BEFORE),
            (abs(price_a - sale_b), E_PACK_PRICES),
            (cheaper * 10, E_PER_KG),
            (abs(unit_a * (100 - d) / 100 - unit_b), "fr2e-percent-discount-on-the-wrong-pack"),
        ],
        min_places=2,
        suffix=_ASK,
    )
    return single(stem, part)


def _exam(rng):
    name = rng.choice(_NAMES)
    good, plural = rng.choice(_GOODS)
    d = rng.choice(_DISCOUNTS)
    marked_a = _price(rng, 8, 80, 240)
    sale_a = Fraction(marked_a * (100 - d), 100)
    base_b = _price(rng, 10, 60, 220)
    paid_b = Fraction(base_b) * Fraction(11, 10)
    if not (_exact_cents(sale_a) and _exact_cents(paid_b)) or sale_a == paid_b:
        raise Redraw("prices must be exact to the cent and differ")
    gap = abs(sale_a - paid_b)
    if gap < 2:
        raise Redraw("the shops should differ by a clear amount")
    cheaper_is_a = sale_a < paid_b
    k = rng.choice((2, 3, 4))
    best = min(sale_a, paid_b)
    stem = (
        f"{name} is buying {k} identical {plural}. Shop A marks each {good} at ${marked_a} and takes {_pct(d)} off. "
        f"Shop B marks each at ${base_b} before GST, and 10% GST is added to that price."
    )
    shop = {True: "Shop A", False: "Shop B"}
    other_gap_a = abs(Fraction(marked_a) - paid_b)  # discount forgotten
    other_gap_b = abs(sale_a - Fraction(base_b))  # GST forgotten
    part_a = exact_number_part(
        f"Find the price of one {good} at Shop A.",
        sale_a,
        2,
        [
            f"{d}% of ${marked_a} = {dollars(Fraction(marked_a * d, 100))}.",
            f"${marked_a} − {dollars(Fraction(marked_a * d, 100))} = {dollars(sale_a)}.",
        ],
        [
            (Fraction(marked_a * d, 100), E_DISCOUNT_ONLY),
            (Fraction(marked_a * (100 + d), 100), E_ADDED),
            (marked_a - d, E_ABSOLUTE),
        ],
        min_places=2,
        suffix=_ASK,
        sensible_range=(0, None),
    )
    part_b = exact_number_part(
        f"Find the price of one {good} at Shop B, including GST.",
        paid_b,
        2,
        [
            f"GST: 10% of ${base_b} = {dollars(Fraction(base_b, 10))}.",
            f"${base_b} + {dollars(Fraction(base_b, 10))} = {dollars(paid_b)}.",
        ],
        [
            (Fraction(base_b, 10), "fr2e-percent-gst-reported-not-total"),
            (Fraction(base_b), E_GST_SKIPPED),
            (Fraction(base_b) - Fraction(base_b, 10), "fr2e-percent-gst-subtracted"),
        ],
        min_places=2,
        suffix=_ASK,
        sensible_range=(0, None),
    )
    key = f"{shop[cheaper_is_a]} is cheaper, by {dollars(gap)} for each {good}"
    wrong = [
        (f"{shop[not cheaper_is_a]} is cheaper, by {dollars(gap)} for each {good}", E_WRONG_SHOP),
        (
            f"{shop[cheaper_is_a]} is cheaper, by {dollars(other_gap_a if cheaper_is_a else other_gap_b)} for each {good}",
            E_DISCOUNT_SKIPPED if cheaper_is_a else E_GST_SKIPPED,
        ),
        (
            f"{shop[not cheaper_is_a]} is cheaper, by {dollars(other_gap_b if cheaper_is_a else other_gap_a)} for each {good}",
            E_GST_SKIPPED if cheaper_is_a else E_DISCOUNT_SKIPPED,
        ),
        (
            f"{shop[cheaper_is_a]} is cheaper, by {dollars(abs(Fraction(marked_a * d, 100) - Fraction(base_b, 10)))} for each {good}",
            "fr2e-percent-adjustments-compared",
        ),
    ]
    part_c = mcq_part(
        f"Decide which shop is cheaper for one {good}, and by how much.",
        key,
        wrong,
        2,
        [f"Shop A: {dollars(sale_a)}. Shop B: {dollars(paid_b)}.", f"The difference is {dollars(gap)}."],
        wanted=4,  # all four honest slips, so the part always shows five options
    )
    part_d = exact_number_part(
        f"Find the total cost of the {k} {plural} at the cheaper shop.",
        k * best,
        2,
        [f"{k} × {dollars(best)} = {dollars(k * best)}."],
        [
            (best, "fr2e-percent-one-item-only"),
            (k * max(sale_a, paid_b), E_WRONG_SHOP),
            (k * best + gap, "fr2e-percent-saving-added-to-total"),
            (k * marked_a, E_DISCOUNT_SKIPPED),
        ],
        min_places=2,
        suffix=_ASK,
        follow_through=f"{k} * Min(part_a, part_b)",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator(
    "fr2e_percent_multistep",
    level=6,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2e_percent_multistep(rng, tier: str = "easy"):
    """A shopping problem that chains percentages and unit prices."""
    build = {"easy": _easy, "medium": _medium, "hard": _hard}.get(tier)
    check(build is not None, f"unknown tier {tier!r}")
    return draw(build, rng)


@generator("exd_x_fr2e_percent_multistep", level=6, presets={"exam": dict(tier="exam")})
def exd_x_fr2e_percent_multistep(rng, tier: str = "exam"):
    """A four-part chain: a discounted price, a GST-inclusive price, the better shop, the cost of several items."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return draw(_exam, rng)
