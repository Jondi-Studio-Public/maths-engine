"""Exam tier for unit pricing: comparisons and offers where a unit price has to be built before it is used.

  compare  -- two packs of the same product sold by mass, one labelled in grams and one in kilograms; the
              learner converts both to a price per 100 g and reports how many cents per 100 g the better
              buy saves (a unit conversion, a comparison and a subtraction).
  offer    -- a multi-buy offer ("3 for the price of 2", "second pack half price", a percentage off when
              buying several) applied to a pack of several items; the learner finds the price per item
              when the offer is used.

Each is built backwards from whole-cent unit prices, so the pack prices are exact and the answer is an exact
number of cents (compare) or an exact whole-cent price per item (offer).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, currency_from_cents, generator
from .exd_x_g5_common import number_fields

_PRODUCTS = ("rice", "pasta", "flour", "coffee beans", "oats", "sugar", "dried lentils")
_ITEMS = ("yoghurt pots", "juice boxes", "muesli bars", "tinned tomatoes", "tea bags", "batteries")


def _mass(grams: int) -> str:
    return f"{grams} g" if grams < 1000 or grams % 100 else f"{grams / 1000:g} kg"


def _compare(rng: random.Random) -> dict:
    product = rng.choice(_PRODUCTS)
    cheap = rng.randint(40, 260)
    gap = rng.randint(6, 45)
    dear = cheap + gap
    grams_a = rng.randrange(300, 1001, 100)
    grams_b = rng.randrange(1100, 2501, 100)
    # Which pack is the cheaper buy varies; which is labelled in kg does not (the larger pack is in kg).
    unit_a, unit_b = (cheap, dear) if rng.random() < 0.5 else (dear, cheap)
    price_a = unit_a * (grams_a // 100)
    price_b = unit_b * (grams_b // 100)
    text = (
        f"A supermarket sells {product} in two packs: {_mass(grams_a)} for {currency_from_cents(price_a)} and "
        f"{grams_b / 1000:g} kg for {currency_from_cents(price_b)}. By how many cents per 100 g is the cheaper "
        f"buy cheaper than the other pack?"
    )
    answer = abs(Fraction(price_a * 100, grams_a) - Fraction(price_b * 100, grams_b))
    check(answer == gap, "gap does not reproduce")
    ua, ub = Fraction(price_a * 100, grams_a), Fraction(price_b * 100, grams_b)
    wrong = [
        abs(Fraction(price_a - price_b)),  # the pack prices subtracted
        answer * 10,  # a price-per-kg gap reported as per 100 g
        abs(ua - 10 * ub),  # the kilogram pack read as if it were in units of 100 g
        ub,  # the larger pack's unit price reported instead of the gap
    ]
    working = [
        f"Pack 1: {price_a} cents / {grams_a // 100} (hundreds of grams) = {unit_a} cents per 100 g.",
        f"Pack 2: {price_b} cents / {grams_b // 100} (hundreds of grams) = {unit_b} cents per 100 g.",
        f"Difference = {abs(unit_a - unit_b)} cents per 100 g.",
    ]
    return number_fields(text, answer, 0, wrong, working)


# (offer clause, packs bought, total cost of those packs measured in single-pack prices)
_OFFERS = (
    ("buy 3 packs and pay for only 2 of them", 3, Fraction(2)),
    ("buy 2 packs and the second is half price", 2, Fraction(3, 2)),
    ("buy 4 packs and take 25% off the total", 4, Fraction(3)),
    ("buy 5 packs and get one of them free", 5, Fraction(4)),
)


def _offer(rng: random.Random) -> dict:
    item = rng.choice(_ITEMS)
    clause, packs, cost_packs = rng.choice(_OFFERS)
    count = rng.choice((4, 6, 8, 10, 12, 15, 20))
    # Choose the effective price per item (whole cents), then find a pack price in whole cents that gives it.
    while True:
        unit = rng.randint(30, 260)
        pack_price = Fraction(unit * packs * count) / cost_packs
        if pack_price.denominator == 1:
            break
    price = int(pack_price)
    text = (
        f"A pack of {count} {item} costs {currency_from_cents(price)}. A special offer: {clause}. "
        f"If you buy {packs} packs using the offer, what is the price per item, in cents?"
    )
    answer = Fraction(price) * cost_packs / (packs * count)
    check(answer == unit, "offer does not reproduce")
    p = Fraction(price)
    wrong = [
        p / count,  # the offer ignored
        p * cost_packs / count,  # the discounted total shared over one pack's items only
        p * cost_packs / ((packs - 1) * count),  # one pack too few counted in the sharing
        p * (2 * packs - cost_packs) / (packs * count),  # the saving added to the cost instead of taken off
    ]
    working = [
        f"Cost of {packs} packs = {float(cost_packs):g} x {price} cents = {float(cost_packs * price):g} cents.",
        f"Items bought = {packs} x {count} = {packs * count}.",
        f"Price per item = {float(cost_packs * price):g} / {packs * count} = {unit} cents.",
    ]
    return number_fields(text, Fraction(unit), 0, wrong, working)


_VARIANTS = {"compare": _compare, "offer": _offer}


@generator("exd_x_g5_unit_pricing", level=8, presets={"exam": dict(tier="exam")})
def exd_x_g5_unit_pricing(rng: random.Random, tier: str = "exam") -> dict:
    """Better-buy gap per 100 g across grams and kilograms, or the price per item under a multi-buy offer."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _VARIANTS[rng.choice(tuple(_VARIANTS))](rng)
