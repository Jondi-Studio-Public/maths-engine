"""Generates the `w3_*` depreciation topics: flat-rate, reducing-balance and the comparison of the two, each with
easy, medium and hard presets and an exam chain.

* ``flat-rate-depreciation`` (``w3_flat_depreciation``, exam ``exd_x_w3_flat_depreciation``): the asset loses the same
  percentage of its PURCHASE price every year. Easy: the yearly amount, the value or the total after n years. Medium: the
  percentage from two values, the years to reach a value, the straight-line amount from a scrap value, a rule read off.
  Hard: the price from two readings, the first year the value is below a target (a target met exactly needs one more
  year), the years to zero from one reading, and the percentage from a share of the price remaining.
* ``reducing-balance-depreciation`` (``w3_reducing_depreciation``, exam ``exd_x_w3_reducing_depreciation``): the asset
  loses a percentage of its CURRENT value each year, V = P(1 - r)^n. Easy: the first year, then two years step by step.
  Medium: three to five years (nearest cent), the formula read off, the rate from one year's value. Hard: the first year
  below a target, the rate from two years' value, the price from a later value, the depreciation in one particular year.
* ``compare-flat-and-reducing-depreciation`` (``w3_compare_depreciation``, exam ``exd_x_w3_compare_depreciation``): both
  methods at the same percentage. Easy and medium: the difference in value after two, then three to five years. Hard: the
  first year the reducing-balance value is the higher, and the flat rate that matches a reducing-balance value.

Every answer is built from chosen prices and rates and re-derived a second way (a year-by-year schedule).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._w3_kit import DP1, MONEY, WHOLE, art, clear_of_tie, dollars, pct_text
from ._w6a_kit import enough, part, retry_draw

FLAT_TOPIC = "w3_flat_depreciation"
FLAT_EXAM = "exd_x_w3_flat_depreciation"
RED_TOPIC = "w3_reducing_depreciation"
RED_EXAM = "exd_x_w3_reducing_depreciation"
CMP_TOPIC = "w3_compare_depreciation"
CMP_EXAM = "exd_x_w3_compare_depreciation"

MISCONCEPTIONS = [
    (
        "fdep-rate-not-converted",
        "Multiplies the price by the percentage without dividing by 100.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-value-for-depreciation",
        "Gives the value left after one year where the yearly depreciation was asked.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-decimal-slip",
        "Misplaces the decimal point when finding a percentage of the price.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-divides-by-rate",
        "Divides the price by the percentage instead of finding that percentage of it.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-amount-not-value",
        "Gives the total depreciation where the value after the years was asked.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-one-year-only",
        "Takes off only one year's depreciation however many years have passed.",
        "flat-rate-depreciation",
    ),
    ("fdep-adds-to-price", "Adds the depreciation to the price instead of subtracting it.", "flat-rate-depreciation"),
    (
        "fdep-uses-reducing",
        "Takes the percentage off the new value each year when the flat rate is a percentage of the purchase price.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-year-not-total",
        "Gives one year's depreciation where the total over several years was asked.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-ignores-years",
        "Finds the percentage lost in total and does not divide by the number of years.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-dollars-not-percent",
        "Gives the depreciation per year in dollars where a percentage of the price was asked.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-percent-of-value",
        "Finds the percentage of the later value instead of the purchase price.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-years-from-value",
        "Divides the target value by the yearly depreciation instead of the amount lost.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-years-from-price",
        "Divides the purchase price by the yearly depreciation, ignoring the value to be reached.",
        "flat-rate-depreciation",
    ),
    ("fdep-counts-year-zero", "Adds an extra year, counting the day of purchase as a year.", "flat-rate-depreciation"),
    (
        "fdep-boundary-year",
        "Gives the last year the value is still at or above the target, not the first year it is below it.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-salvage-ignored",
        "Divides the purchase price by the life, forgetting to take off the scrap value first.",
        "flat-rate-depreciation",
    ),
    ("fdep-salvage-added", "Adds the scrap value to the price before dividing by the life.", "flat-rate-depreciation"),
    (
        "fdep-adds-one-year-back",
        "Adds back only one year of depreciation to a later value to find the price.",
        "flat-rate-depreciation",
    ),
    ("fdep-reading-as-price", "Takes a later value as the purchase price.", "flat-rate-depreciation"),
    (
        "fdep-years-remaining",
        "Gives the years still to go, not the total number of years until the value is zero.",
        "flat-rate-depreciation",
    ),
    (
        "fdep-rate-from-price-only",
        "Divides the percentage lost by the years, then forgets the value left is a share of the price.",
        "flat-rate-depreciation",
    ),
    ("fdep-remaining-for-lost", "Uses the percentage remaining as the percentage lost.", "flat-rate-depreciation"),
    (
        "rbd-rate-not-converted",
        "Multiplies by the percentage without dividing by 100.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-depreciation-for-value",
        "Gives the amount depreciated where the value was asked.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-uses-flat",
        "Takes the same amount off each year (flat rate) when the percentage applies to the current value.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-one-year-only",
        "Applies the percentage once however many years have passed.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-multiplies-by-years",
        "Multiplies the rate by the number of years inside the bracket instead of using a power.",
        "reducing-balance-depreciation",
    ),
    ("rbd-extra-year", "Applies the percentage one year too many.", "reducing-balance-depreciation"),
    ("rbd-one-year-short", "Applies the percentage one year too few.", "reducing-balance-depreciation"),
    (
        "rbd-rate-for-multiplier",
        "Multiplies by the rate (for example 0.15) instead of by one less the rate (0.85).",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-growth-multiplier",
        "Uses one plus the rate as the yearly multiplier, which grows the value.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-total-for-value",
        "Gives the value left where the total depreciation was asked.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-first-year-amount",
        "Finds the depreciation for the first year only and calls it the total.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-boundary-year",
        "Gives the last year the value is still at or above the target, not the first year it is below it.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-flat-years",
        "Finds the years to reach the target as if the same amount came off each year.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-rate-from-whole-drop",
        "Takes the whole drop over two years as the yearly rate instead of finding the yearly multiplier.",
        "reducing-balance-depreciation",
    ),
    ("rbd-rate-halved", "Halves the two-year drop to get the yearly rate.", "reducing-balance-depreciation"),
    (
        "rbd-multiplier-for-rate",
        "Gives the yearly multiplier as a percentage instead of the rate of depreciation.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-price-from-flat",
        "Adds back the same amount each year to find the price instead of dividing by the multiplier.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-price-multiplies",
        "Multiplies the later value by the multiplier instead of dividing, so the 'price' is smaller than the value.",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-wrong-year-amount",
        "Gives the depreciation of the wrong year (the year before or after).",
        "reducing-balance-depreciation",
    ),
    (
        "rbd-year-from-first-value",
        "Takes the percentage of the purchase price for the year asked, as if every year lost the same amount.",
        "reducing-balance-depreciation",
    ),
    (
        "cmp-difference-sign",
        "Adds the two values instead of finding their difference.",
        "compare-flat-and-reducing-depreciation",
    ),
    (
        "cmp-reports-one-method",
        "Gives the value under one method where the difference was asked.",
        "compare-flat-and-reducing-depreciation",
    ),
    ("cmp-decimal-slip", "Misplaces the decimal point in the difference.", "compare-flat-and-reducing-depreciation"),
    (
        "cmp-doubles-difference",
        "Doubles the difference, adding the gap for each method instead of comparing them once.",
        "compare-flat-and-reducing-depreciation",
    ),
    (
        "cmp-year-flat-zero",
        "Gives the year the flat-rate value would reach zero instead of the year the methods cross over.",
        "compare-flat-and-reducing-depreciation",
    ),
    (
        "cmp-flat-total-rate",
        "Uses the percentage for each year of the flat method as if it were applied once.",
        "compare-flat-and-reducing-depreciation",
    ),
    (
        "cmp-crossover-early",
        "Names the year before the reducing-balance value becomes the higher.",
        "compare-flat-and-reducing-depreciation",
    ),
    (
        "cmp-crossover-late",
        "Names the year after the reducing-balance value becomes the higher.",
        "compare-flat-and-reducing-depreciation",
    ),
    (
        "cmp-equal-rate",
        "Assumes the matching flat rate equals the reducing-balance rate.",
        "compare-flat-and-reducing-depreciation",
    ),
    (
        "cmp-flat-rate-ignores-years",
        "Gives the total percentage lost, without dividing by the number of years.",
        "compare-flat-and-reducing-depreciation",
    ),
    (
        "cmp-flat-rate-of-value",
        "Divides the dollars lost by the number of years but not by the price, so the rate is in dollars.",
        "compare-flat-and-reducing-depreciation",
    ),
]

_ASSETS = (
    # (noun phrase, lowest price, highest price, step)
    ("laptop", 1500, 4500, 100),
    ("commercial oven", 4000, 16000, 500),
    ("tractor", 30000, 90000, 1000),
    ("delivery van", 25000, 60000, 500),
    ("photocopier", 3000, 12000, 100),
    ("excavator", 80000, 220000, 1000),
    ("ride-on mower", 4000, 12000, 100),
    ("food truck", 40000, 95000, 500),
    ("laser cutter", 6000, 30000, 500),
    ("security camera system", 2000, 9000, 100),
    ("hot-water system", 1500, 4000, 100),
    ("irrigation pump", 3000, 14000, 500),
)

_FLAT_RATES = (4, 5, 8, 10, 12, 15, 20, 25)
_RED_RATES = (5, 8, 10, 12, 15, 18, 20, 25, 30)
_NICE_RATES = (10, 20, 30, 40, 50)  # (100 - r)/100 is a tenth, so a cube of it is a thousandth


def _asset(rng, step_multiple: int = 1):
    name, lo, hi, step = rng.choice(_ASSETS)
    step *= step_multiple
    price = rng.randrange(lo, hi + 1, step) if lo % step == 0 else rng.randrange(lo // step * step + step, hi, step)
    return name, Fraction(price)


def _is_cents(x) -> bool:
    return (Fraction(x) * 100).denominator == 1


def _flat_schedule(price, annual, years):
    """Value at the end of each year, 0..years, by taking the same amount off each time."""
    values = [Fraction(price)]
    for _ in range(years):
        values.append(values[-1] - annual)
    return values


def _red_schedule(price, rate_pct, years):
    """Value at the end of each year, 0..years, by taking the rate off the current value each time."""
    values = [Fraction(price)]
    for _ in range(years):
        values.append(values[-1] - values[-1] * rate_pct / 100)
    return values


def _sentence(rng, name, price, rate, kind):
    how = {
        "flat": (
            "depreciates by a flat rate of {r}% of its purchase price each year",
            "loses {r}% of its purchase price in value every year (flat-rate depreciation)",
        ),
        "red": (
            "depreciates at {r}% per year by the reducing-balance method",
            "loses {r}% of its value at the start of each year (reducing-balance depreciation)",
        ),
    }[kind]
    how = rng.choice(how).format(r=pct_text(rate))
    return f"{art(name).capitalize()} {name} is bought for {dollars(price)}. It {how}."


# -- flat rate: easy --------------------------------------------------------------------------------------------------


def _flat_easy_annual(rng):
    name, price = _asset(rng)
    r = rng.choice(_FLAT_RATES)
    annual = price * r / 100
    check(_flat_schedule(price, annual, 1)[1] == price - annual, "schedule differs")
    d = MONEY.dist(
        annual,
        [
            (price * r, "fdep-rate-not-converted"),
            (price - annual, "fdep-value-for-depreciation"),
            (annual / 10, "fdep-decimal-slip"),
            (price / r, "fdep-divides-by-rate"),
            (annual * 10, "fdep-decimal-slip"),
        ],
    )
    enough(d, 3)
    text = _sentence(rng, name, price, r, "flat") + " How much does it depreciate each year, in dollars?"
    working = [f"Yearly depreciation = {r}% of {dollars(price)} = {r}/100 × {price} = {dollars(annual)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(annual), answer_kind="number")


def _flat_easy_value(rng):
    name, price = _asset(rng)
    r = rng.choice(_FLAT_RATES)
    n = rng.randint(2, min(6, 80 // r))
    annual = price * r / 100
    value = price - annual * n
    check(_flat_schedule(price, annual, n)[n] == value, "schedule differs")
    reducing = price * (1 - Fraction(r, 100)) ** n
    d = MONEY.dist(
        value,
        [
            (annual * n, "fdep-amount-not-value"),
            (price - annual, "fdep-one-year-only"),
            (price + annual * n, "fdep-adds-to-price"),
            (reducing, "fdep-uses-reducing"),
            (price - annual * (n + 1), "fdep-counts-year-zero"),
        ],
    )
    enough(d, 3)
    text = _sentence(rng, name, price, r, "flat") + f" What is its value after {n} years, in dollars?"
    working = [
        f"Yearly depreciation = {dollars(annual)}.",
        f"Value = {dollars(price)} - {n} × {dollars(annual)} = {dollars(value)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(value), answer_kind="number")


def _flat_easy_total(rng):
    name, price = _asset(rng)
    r = rng.choice(_FLAT_RATES)
    n = rng.randint(3, min(7, 80 // r)) if 80 // r >= 3 else 3
    annual = price * r / 100
    total = annual * n
    d = MONEY.dist(
        total,
        [
            (annual, "fdep-year-not-total"),
            (price - total, "fdep-amount-not-value"),
            (price * r * n, "fdep-rate-not-converted"),
            (total / 10, "fdep-decimal-slip"),
        ],
    )
    enough(d, 3)
    text = _sentence(rng, name, price, r, "flat") + f" By how many dollars has it depreciated in total after {n} years?"
    working = [
        f"Yearly depreciation = {dollars(annual)}; after {n} years = {n} × {dollars(annual)} = {dollars(total)}."
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(total), answer_kind="number")


# -- flat rate: medium ------------------------------------------------------------------------------------------------


def _flat_medium_rate(rng):
    name, price = _asset(rng)
    r = rng.choice(_FLAT_RATES)
    n = rng.randint(2, min(6, 80 // r))
    annual = price * r / 100
    value = price - annual * n
    lost = price - value
    d = WHOLE.dist(
        r,
        [
            (lost * 100 / price, "fdep-ignores-years"),
            (lost / n, "fdep-dollars-not-percent"),
            (lost * 100 / (value * n), "fdep-percent-of-value"),
            (r + 1, "fdep-counts-year-zero") if r > 1 else None,
            (r * n, "fdep-ignores-years"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} is bought for {dollars(price)}. It depreciates at a flat rate each year and is worth "
        f"{dollars(value)} after {n} years. What percentage of the purchase price does it depreciate by each year?"
    )
    working = [
        f"Lost in {n} years = {dollars(price)} - {dollars(value)} = {dollars(lost)}; per year = {dollars(annual)}.",
        f"Rate = {dollars(annual)} / {dollars(price)} × 100 = {r}%.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(r), answer_kind="number")


def _flat_medium_years(rng):
    name, price = _asset(rng)
    r = rng.choice(_FLAT_RATES)
    n = rng.randint(3, min(8, 90 // r)) if 90 // r >= 3 else 3
    annual = price * r / 100
    target = price - annual * n
    d = WHOLE.dist(
        n,
        [
            (target / annual, "fdep-years-from-value"),
            (price / annual, "fdep-years-from-price"),
            (n + 1, "fdep-counts-year-zero"),
            (n - 1, "fdep-boundary-year"),
        ],
    )
    enough(d, 3)
    text = _sentence(rng, name, price, r, "flat") + f" After how many years will its value be {dollars(target)}?"
    working = [
        f"Yearly depreciation = {dollars(annual)}.",
        f"Years = ({dollars(price)} - {dollars(target)}) / {dollars(annual)} = {n}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(n), answer_kind="number")


def _flat_medium_salvage(rng):
    name, price = _asset(rng)
    life = rng.randint(4, 12)
    annual = Fraction(rng.randint(2, 12) * 50 if price < 20000 else rng.randint(6, 40) * 100)
    salvage = price - annual * life
    if salvage < price / 20:
        raise_retry()
    d = MONEY.dist(
        annual,
        [
            (price / life, "fdep-salvage-ignored"),
            ((price + salvage) / life, "fdep-salvage-added"),
            (salvage / life, "fdep-value-for-depreciation"),
            ((price - salvage) / (life + 1), "fdep-counts-year-zero"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} is bought for {dollars(price)} and is expected to be worth {dollars(salvage)} after "
        f"{life} years. Using flat-rate (straight-line) depreciation, how much does it depreciate each year, in dollars?"
    )
    working = [f"Annual depreciation = ({dollars(price)} - {dollars(salvage)}) / {life} = {dollars(annual)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(annual), answer_kind="number")


def raise_retry():
    from ._w6a_kit import TooFewDistractors

    raise TooFewDistractors("draw rejected")


def _flat_medium_rule(rng):
    name, price = _asset(rng)
    annual = Fraction(rng.randint(2, 16) * 250)
    m = rng.randint(3, 8)
    value = price - annual * m
    if value < price / 8:
        raise_retry()
    d = MONEY.dist(
        value,
        [
            (annual * m, "fdep-amount-not-value"),
            (price - annual, "fdep-one-year-only"),
            (price + annual * m, "fdep-adds-to-price"),
            (price - annual * (m + 1), "fdep-counts-year-zero"),
        ],
    )
    enough(d, 3)
    text = (
        f"The value, V dollars, of {art(name)} {name} after n years is given by V = {int(price)} - {int(annual)}n. "
        f"What is its value after {m} years, in dollars?"
    )
    working = [f"V = {int(price)} - {int(annual)} × {m} = {dollars(value)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(value), answer_kind="number")


# -- flat rate: hard --------------------------------------------------------------------------------------------------


def _flat_hard_price(rng):
    name, price = _asset(rng)
    r = rng.choice(_FLAT_RATES)
    annual = price * r / 100
    a = rng.randint(1, 3)
    b = a + rng.randint(2, 4)
    if annual * b >= price * Fraction(85, 100):
        raise_retry()
    va, vb = price - annual * a, price - annual * b
    check(Fraction(va - vb, b - a) == annual and va + annual * a == price, "price differs")
    d = MONEY.dist(
        price,
        [
            (va + annual, "fdep-adds-one-year-back"),
            (va, "fdep-reading-as-price"),
            (vb + annual * a, "fdep-adds-one-year-back"),
            (va + (va - vb), "fdep-adds-one-year-back"),
            (va + annual * b, "fdep-counts-year-zero"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} depreciates by the same amount each year. After {a} {'year' if a == 1 else 'years'} it "
        f"is worth {dollars(va)} and after {b} years it is worth {dollars(vb)}. What was its purchase price, in dollars?"
    )
    working = [
        f"Yearly depreciation = ({dollars(va)} - {dollars(vb)}) / ({b} - {a}) = {dollars(annual)}.",
        f"Price = {dollars(va)} + {a} × {dollars(annual)} = {dollars(price)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(price), answer_kind="number")


def _flat_hard_first_below(rng):
    name, price = _asset(rng)
    r = rng.choice(_FLAT_RATES)
    annual = price * r / 100
    n = rng.randint(3, min(9, 92 // r)) if 92 // r >= 3 else 3
    exact = rng.random() < 0.4  # the target is met exactly at the end of year n: the first year BELOW it is n + 1
    if (n + 1) * r > 100:
        exact = False  # year n + 1 would show a negative value for the asset
    value_n = price - annual * n
    if exact:
        target, first = value_n, n + 1
    else:
        target = value_n + annual * Fraction(rng.choice((1, 2, 3)), 4)
        first = n
    check(target > 0, "target")
    sched = _flat_schedule(price, annual, first + 1)
    check(sched[first] < target <= sched[first - 1], "first year below")
    if sched[first] <= 0:
        raise_retry()
    d = WHOLE.dist(
        first,
        [
            (first - 1, "fdep-boundary-year"),
            (first + 1, "fdep-counts-year-zero"),
            (target / annual, "fdep-years-from-value"),
            (price / annual, "fdep-years-from-price"),
        ],
    )
    enough(d, 3)
    text = (
        _sentence(rng, name, price, r, "flat")
        + f" The owner will replace it in the first year its value falls below {dollars(target)}. After how many years will that be?"
    )
    working = [
        f"Yearly depreciation = {dollars(annual)}.",
        f"Value after {first - 1} years = {dollars(sched[first - 1])}; after {first} years = {dollars(sched[first])}.",
        f"{dollars(sched[first])} is the first value below {dollars(target)}, so the answer is {first} years.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(first), answer_kind="number")


def _flat_hard_to_zero(rng):
    name, price = _asset(rng)
    total = rng.choice((8, 10, 12, 16, 20, 25))  # years until the value is zero
    annual = price / total
    if not _is_cents(annual) or annual != int(annual):
        raise_retry()
    a = rng.randint(2, min(6, total - 3))
    value = price - annual * a
    check(price - annual * total == 0, "zero")
    d = WHOLE.dist(
        total,
        [
            (total - a, "fdep-years-remaining"),
            (value / annual + 1, "fdep-years-remaining"),
            (total + 1, "fdep-counts-year-zero"),
            (price / value * a, "fdep-years-from-price"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} bought for {dollars(price)} depreciates by the same amount each year. After {a} years "
        f"it is worth {dollars(value)}. How many years after it was bought will its value be zero?"
    )
    working = [
        f"Yearly depreciation = ({dollars(price)} - {dollars(value)}) / {a} = {dollars(annual)}.",
        f"Years to zero = {dollars(price)} / {dollars(annual)} = {total}.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(total), answer_kind="number")


def _flat_hard_share(rng):
    name, price = _asset(rng)
    n = rng.choice((2, 3, 4, 5, 6, 8))
    r = rng.choice([x for x in (4, 5, 6, 8, 10, 12, 15) if x * n <= 85])
    remaining = 100 - r * n
    d = WHOLE.dist(
        r,
        [
            (100 - remaining, "fdep-ignores-years"),
            (remaining / n, "fdep-remaining-for-lost"),
            (remaining, "fdep-remaining-for-lost"),
            (r + 1, "fdep-counts-year-zero"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} bought for {dollars(price)} depreciates at a flat rate each year. After {n} years it "
        f"is worth {remaining}% of its purchase price. What percentage of the purchase price does it lose each year?"
    )
    working = [f"Lost in {n} years = {100 - remaining}% of the price; per year = {100 - remaining} / {n} = {r}%."]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(r), answer_kind="number")


_FLAT_EASY = (_flat_easy_annual, _flat_easy_value, _flat_easy_total)
_FLAT_MEDIUM = (_flat_medium_rate, _flat_medium_years, _flat_medium_salvage, _flat_medium_rule)
_FLAT_HARD = (_flat_hard_price, _flat_hard_first_below, _flat_hard_to_zero, _flat_hard_share)


@generator(
    FLAT_TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_flat_depreciation(rng, tier="easy"):
    """Flat-rate depreciation; the tiers are described in the module docstring."""
    pool = {"easy": _FLAT_EASY, "medium": _FLAT_MEDIUM, "hard": _FLAT_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(FLAT_EXAM, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w3_flat_depreciation(rng, tier="exam"):
    """Exam chain: yearly amount, value after n years, years to a value, the first year below a target."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        name, price = _asset(rng)
        r = rng.choice(_FLAT_RATES)
        annual = price * r / 100
        n = rng.randint(2, min(5, 70 // r)) if 70 // r >= 2 else 2
        m = rng.randint(n + 1, min(n + 5, 92 // r)) if 92 // r > n else n + 1
        target_years = m
        target = price - annual * target_years
        if target <= 0:
            raise_retry()
        lim_years = (
            rng.randint(max(target_years, 3), min(target_years + 3, 95 // r))
            if 95 // r >= target_years
            else target_years
        )
        sched = _flat_schedule(price, annual, lim_years + 1)
        limit = sched[lim_years] + annual * Fraction(rng.choice((1, 2, 3)), 4)
        first = next(i for i, v in enumerate(sched) if v < limit)
        if sched[first] <= 0:
            raise_retry()
        value_n = price - annual * n
        a_d = MONEY.dist(
            annual,
            [
                (price * r, "fdep-rate-not-converted"),
                (price - annual, "fdep-value-for-depreciation"),
                (annual / 10, "fdep-decimal-slip"),
                (price / r, "fdep-divides-by-rate"),
            ],
        )
        b_d = MONEY.dist(
            value_n,
            [
                (annual * n, "fdep-amount-not-value"),
                (price - annual, "fdep-one-year-only"),
                (price + annual * n, "fdep-adds-to-price"),
                (price * (1 - Fraction(r, 100)) ** n, "fdep-uses-reducing"),
            ],
        )
        c_d = WHOLE.dist(
            target_years,
            [
                (target / annual, "fdep-years-from-value"),
                (price / annual, "fdep-years-from-price"),
                (target_years + 1, "fdep-counts-year-zero"),
                (target_years - 1, "fdep-boundary-year"),
            ],
        )
        d_d = WHOLE.dist(
            first,
            [
                (first - 1, "fdep-boundary-year"),
                (first + 1, "fdep-counts-year-zero"),
                (limit / annual, "fdep-years-from-value"),
            ],
        )
        for x, k in ((a_d, 3), (b_d, 3), (c_d, 3), (d_d, 2)):
            enough(x, k)
        stem = _sentence(rng, name, price, r, "flat")
        return dict(
            text=stem,
            parts=[
                part(
                    "How much does it depreciate each year, in dollars?",
                    "number",
                    MONEY.fields(annual),
                    1,
                    [f"{r}% of {dollars(price)} = {dollars(annual)}."],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"Find its value after {n} years, in dollars.",
                    "number",
                    MONEY.fields(value_n),
                    2,
                    [f"Value = {dollars(price)} - {n} × {dollars(annual)} = {dollars(value_n)}."],
                    b_d,
                    follow_through=f"{price} - part_a*{n}",
                    sensible_range=(0, None),
                ),
                part(
                    f"After how many years will its value be {dollars(target)}?",
                    "number",
                    WHOLE.fields(target_years),
                    2,
                    [f"Years = ({dollars(price)} - {dollars(target)}) / {dollars(annual)} = {target_years}."],
                    c_d,
                    follow_through=f"({price} - {target})/part_a",
                    sensible_range=(0, None),
                ),
                part(
                    f"The owner will replace it in the first year its value falls below {dollars(limit)}. After how many years is that?",
                    "number",
                    WHOLE.fields(first),
                    2,
                    [
                        f"Value after {first - 1} years = {dollars(sched[first - 1])}; after {first} years = {dollars(sched[first])}."
                    ],
                    d_d,
                    sensible_range=(0, None),
                ),
            ],
        )

    return retry_draw(rng, build)


# -- reducing balance -------------------------------------------------------------------------------------------------


def _red_value(price, rate, n):
    return Fraction(price) * (1 - Fraction(rate, 100)) ** n


def _red_easy_first(rng):
    name, price = _asset(rng)
    r = rng.choice(_RED_RATES)
    amount = price * r / 100
    value = price - amount
    check(_red_schedule(price, r, 1)[1] == value, "schedule differs")
    ask_value = rng.random() < 0.5
    if ask_value:
        d = MONEY.dist(
            value,
            [
                (amount, "rbd-depreciation-for-value"),
                (price * (1 + Fraction(r, 100)), "rbd-growth-multiplier"),
                (price * Fraction(r, 100) / 10, "rbd-rate-not-converted"),
                (price * r, "rbd-rate-not-converted"),
            ],
        )
        ask = f" What is its value after 1 year, in dollars?"
        ans = value
        working = [
            f"Depreciation = {r}% of {dollars(price)} = {dollars(amount)}.",
            f"Value = {dollars(price)} - {dollars(amount)} = {dollars(value)}.",
        ]
    else:
        d = MONEY.dist(
            amount,
            [
                (value, "rbd-total-for-value"),
                (price * r, "rbd-rate-not-converted"),
                (amount / 10, "rbd-rate-not-converted"),
                (price * (1 + Fraction(r, 100)), "rbd-growth-multiplier"),
            ],
        )
        ask = " By how many dollars does it depreciate in the first year?"
        ans = amount
        working = [f"Depreciation = {r}% of {dollars(price)} = {dollars(amount)}."]
    enough(d, 3)
    text = _sentence(rng, name, price, r, "red") + ask
    return dict(text=text, working=working, distractors=d, **MONEY.fields(ans), answer_kind="number")


def _red_easy_two(rng):
    name, price = _asset(rng)
    r = rng.choice(_RED_RATES)
    v1 = price * (100 - r) / 100
    v2 = v1 * (100 - r) / 100
    check(_red_schedule(price, r, 2)[2] == v2, "schedule differs")
    flat = price - 2 * price * r / 100
    d = MONEY.dist(
        v2,
        [
            (flat, "rbd-uses-flat"),
            (v1, "rbd-one-year-only"),
            (price * (1 - 2 * Fraction(r, 100) ** 2), "rbd-multiplies-by-years"),
            (v2 * (100 - r) / 100, "rbd-extra-year"),
            (price - v2, "rbd-total-for-value"),
        ],
    )
    enough(d, 3)
    text = _sentence(rng, name, price, r, "red") + " What is its value after 2 years, in dollars?"
    working = [
        f"After 1 year: {dollars(price)} × {pct_text(100 - r)}% = {dollars(v1)}.",
        f"After 2 years: {dollars(v1)} × {pct_text(100 - r)}% = {dollars(v2)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(v2), answer_kind="number")


def _red_medium_value(rng):
    name, price = _asset(rng)
    r = rng.choice(_RED_RATES)
    n = rng.randint(3, 5)
    value = _red_value(price, r, n)
    sched = _red_schedule(price, r, n)
    check(sched[n] == value, "schedule differs")
    if not clear_of_tie(value, 2):
        raise_retry()
    flat = price - n * price * Fraction(r, 100)
    q = Fraction(r, 100)
    d = MONEY.dist(
        value,
        [
            (flat, "rbd-uses-flat"),
            (price * (1 - q * n), "rbd-multiplies-by-years"),
            (sched[n - 1], "rbd-one-year-short"),
            (price * (1 - q) ** (n + 1), "rbd-extra-year"),
            (price * q**n, "rbd-rate-for-multiplier"),
            (price - value, "rbd-total-for-value"),
        ],
    )
    enough(d, 3)
    text = _sentence(rng, name, price, r, "red") + f" What is its value after {n} years, to the nearest cent?"
    working = [
        f"Multiplier = 1 - {r}/100 = {float(1 - q):g}.",
        f"Value = {dollars(price)} × {float(1 - q):g}^{n} = ${float(value):,.2f}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(value), answer_kind="number")


def _red_medium_total(rng):
    name, price = _asset(rng)
    r = rng.choice(_RED_RATES)
    n = rng.randint(3, 4)
    value = _red_value(price, r, n)
    if not clear_of_tie(value, 2):
        raise_retry()
    total = price - round_cents(value)
    q = Fraction(r, 100)
    d = MONEY.dist(
        total,
        [
            (round_cents(value), "rbd-total-for-value"),
            (price * q, "rbd-first-year-amount"),
            (price * q * n, "rbd-uses-flat"),
            (price - round_cents(price * (1 - q) ** (n - 1)), "rbd-one-year-short"),
        ],
    )
    enough(d, 3)
    text = (
        _sentence(rng, name, price, r, "red")
        + f" By how many dollars has it depreciated in total after {n} years, to the nearest cent?"
    )
    working = [
        f"Value after {n} years = {dollars(price)} × {float(1 - q):g}^{n} = ${float(value):,.2f}.",
        f"Total depreciation = {dollars(price)} - ${float(round_cents(value)):,.2f} = ${float(total):,.2f}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(total), answer_kind="number")


def round_cents(x) -> Fraction:
    return MONEY.value(x)


def _red_medium_formula(rng):
    name, price = _asset(rng)
    r = rng.choice(_RED_RATES)
    q = 1 - Fraction(r, 100)
    n = rng.randint(3, 6)
    value = price * q**n
    if not clear_of_tie(value, 2):
        raise_retry()
    d = MONEY.dist(
        value,
        [
            (price * q * n, "rbd-multiplies-by-years"),
            (price * (1 - q) ** n, "rbd-rate-for-multiplier"),
            (price * (1 + Fraction(r, 100)) ** n, "rbd-growth-multiplier"),
            (price * q ** (n - 1), "rbd-one-year-short"),
            (price - price * Fraction(r, 100) * n, "rbd-uses-flat"),
        ],
    )
    enough(d, 3)
    text = (
        f"The value, V dollars, of {art(name)} {name} after n years is given by V = {int(price)} × {float(q):g}^n. "
        f"What is its value after {n} years, to the nearest cent?"
    )
    working = [f"V = {int(price)} × {float(q):g}^{n} = ${float(value):,.2f}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(value), answer_kind="number")


def _red_medium_rate(rng):
    name, price = _asset(rng)
    r = rng.choice(_RED_RATES)
    v1 = price * (100 - r) / 100
    d = WHOLE.dist(
        r,
        [
            (100 - r, "rbd-multiplier-for-rate"),
            (r / 2, "rbd-rate-halved"),
            (r + 1, "rbd-extra-year"),
            (r * 2, "rbd-multiplies-by-years"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} is bought for {dollars(price)}. It depreciates by the reducing-balance method and is worth "
        f"{dollars(v1)} after 1 year. What is the annual rate of depreciation, as a percentage?"
    )
    working = [
        f"Lost in year 1 = {dollars(price)} - {dollars(v1)} = {dollars(price - v1)}; rate = {dollars(price - v1)} / {dollars(price)} × 100 = {r}%."
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(r), answer_kind="number")


def _red_hard_first_below(rng):
    name, price = _asset(rng)
    r = rng.choice((10, 12, 15, 20, 25, 30))
    n = rng.randint(3, 7)
    sched = _red_schedule(price, r, n + 1)
    lo, hi = sched[n], sched[n - 1]
    gap = hi - lo
    target = Fraction(int((lo + gap * Fraction(rng.choice((3, 4, 5)), 8)) // 50 * 50))
    if not (lo < target < hi):
        raise_retry()
    annual = price * r / 100
    flat_years = -(-(price - target) // annual)  # a flat-rate reading
    if flat_years == n:
        raise_retry()
    d = WHOLE.dist(
        n,
        [
            (n - 1, "rbd-boundary-year"),
            (n + 1, "rbd-extra-year"),
            (flat_years, "rbd-flat-years"),
            (target / annual, "rbd-flat-years"),
        ],
    )
    enough(d, 3)
    text = (
        _sentence(rng, name, price, r, "red")
        + f" After how many years will its value first be below {dollars(target)}?"
    )
    working = [
        f"Value after {n - 1} years = ${float(sched[n - 1]):,.2f}, which is not below {dollars(target)}.",
        f"Value after {n} years = ${float(sched[n]):,.2f}, which is below {dollars(target)}. The answer is {n} years.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(n), answer_kind="number")


def _red_hard_rate_two(rng):
    name, price = _asset(rng)
    r = rng.choice(_RED_RATES)
    v2 = price * (100 - r) ** 2 / 10000
    d = WHOLE.dist(
        r,
        [
            (Fraction((price - v2) * 100, price), "rbd-rate-from-whole-drop"),
            (Fraction((price - v2) * 50, price), "rbd-rate-halved"),
            (100 - r, "rbd-multiplier-for-rate"),
            (r + 1, "rbd-extra-year"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} is bought for {dollars(price)}. It depreciates by the reducing-balance method at the same "
        f"percentage rate each year and is worth {dollars(v2)} after 2 years. Find the annual rate of depreciation, as a percentage."
    )
    working = [
        f"({dollars(v2)} / {dollars(price)}) = {float(v2 / price):g} = (1 - r)^2, so 1 - r = {float(Fraction(100 - r, 100)):g}.",
        f"The rate is {r}%.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(r), answer_kind="number")


def _red_hard_price(rng):
    name = rng.choice(_ASSETS)[0]
    r = rng.choice(_NICE_RATES)
    n = rng.choice((2, 3))
    price = Fraction(rng.randrange(2000, 40001, 1000))
    value = price * Fraction(100 - r, 100) ** n
    check(_is_cents(value), "value cents")
    sched = _red_schedule(price, r, n)
    check(sched[n] == value, "schedule differs")
    q = Fraction(100 - r, 100)
    d = MONEY.dist(
        price,
        [
            (value * q**n, "rbd-price-multiplies"),
            (value / q, "rbd-one-year-short"),
            (value + (price - value) / n, "rbd-price-from-flat"),
            (value * (1 + Fraction(r, 100)) ** n, "rbd-growth-multiplier"),
            (value / (1 - q * n) if q * n < 1 else None, "rbd-multiplies-by-years"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} depreciates at {r}% per year by the reducing-balance method. After {n} years it is worth "
        f"{dollars(value)}. What was its purchase price, in dollars?"
    )
    working = [f"Price = {dollars(value)} / {float(q):g}^{n} = {dollars(price)}."]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(price), answer_kind="number")


def _red_hard_year_amount(rng):
    name = rng.choice(_ASSETS)[0]
    r = rng.choice(_NICE_RATES)
    price = Fraction(rng.randrange(3000, 40001, 1000))
    n = rng.choice((2, 3))
    sched = _red_schedule(price, r, n)
    amount = sched[n - 1] - sched[n]
    check(_is_cents(amount) and amount == sched[n - 1] * r / 100, "year amount")
    q = Fraction(r, 100)
    d = MONEY.dist(
        amount,
        [
            (price * q, "rbd-year-from-first-value"),
            (sched[n - 2] * q if n >= 2 else None, "rbd-wrong-year-amount"),
            (sched[n] * q, "rbd-wrong-year-amount"),
            (price - sched[n], "rbd-total-for-value"),
            (sched[n], "rbd-depreciation-for-value"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} is bought for {dollars(price)} and depreciates at {r}% per year by the reducing-balance method. "
        f"By how many dollars does it depreciate during year {n}?"
    )
    working = [
        f"Value at the start of year {n} (after {n - 1} {'year' if n == 2 else 'years'}) = {dollars(sched[n - 1])}.",
        f"Depreciation in year {n} = {r}% of {dollars(sched[n - 1])} = {dollars(amount)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(amount), answer_kind="number")


_RED_EASY = (_red_easy_first, _red_easy_two)
_RED_MEDIUM = (_red_medium_value, _red_medium_total, _red_medium_formula, _red_medium_rate)
_RED_HARD = (_red_hard_first_below, _red_hard_rate_two, _red_hard_price, _red_hard_year_amount)


@generator(
    RED_TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_reducing_depreciation(rng, tier="easy"):
    """Reducing-balance depreciation; the tiers are described in the module docstring."""
    pool = {"easy": _RED_EASY, "medium": _RED_MEDIUM, "hard": _RED_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(RED_EXAM, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w3_reducing_depreciation(rng, tier="exam"):
    """Exam chain: first-year value, value after n years, total depreciation, first year below a target."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        name, price = _asset(rng)
        r = rng.choice(_RED_RATES)
        q = 1 - Fraction(r, 100)
        n = rng.randint(3, 5)
        sched = _red_schedule(price, r, 9)
        v1, vn = sched[1], sched[n]
        if not clear_of_tie(vn, 2):
            raise_retry()
        vn_c = round_cents(vn)
        total = price - vn_c
        k = rng.randint(n + 1, 8)
        target = Fraction(int((sched[k] + (sched[k - 1] - sched[k]) * Fraction(1, 2)) // 50 * 50))
        if not (sched[k] < target < sched[k - 1]):
            raise_retry()
        a_d = MONEY.dist(
            v1,
            [
                (price * Fraction(r, 100), "rbd-depreciation-for-value"),
                (price * (1 + Fraction(r, 100)), "rbd-growth-multiplier"),
                (price * r, "rbd-rate-not-converted"),
                (price * Fraction(r, 100) / 10, "rbd-rate-not-converted"),
            ],
        )
        b_d = MONEY.dist(
            vn,
            [
                (price - n * price * Fraction(r, 100), "rbd-uses-flat"),
                (price * q * n, "rbd-multiplies-by-years"),
                (sched[n - 1], "rbd-one-year-short"),
                (sched[n + 1], "rbd-extra-year"),
            ],
        )
        c_d = MONEY.dist(
            total,
            [
                (vn_c, "rbd-total-for-value"),
                (price * Fraction(r, 100), "rbd-first-year-amount"),
                (price - sched[n - 1], "rbd-one-year-short"),
                (price * Fraction(r, 100) * n, "rbd-uses-flat"),
            ],
        )
        annual = price * Fraction(r, 100)
        flat_years = -(-(price - target) // annual)
        d_d = WHOLE.dist(
            k,
            [
                (k - 1, "rbd-boundary-year"),
                (k + 1, "rbd-extra-year"),
                (flat_years, "rbd-flat-years"),
            ],
        )
        for x, m in ((a_d, 3), (b_d, 3), (c_d, 3), (d_d, 2)):
            enough(x, m)
        stem = _sentence(rng, name, price, r, "red")
        return dict(
            text=stem,
            parts=[
                part(
                    "Find its value after 1 year, in dollars.",
                    "number",
                    MONEY.fields(v1),
                    1,
                    [f"Value = {dollars(price)} × {float(q):g} = {dollars(v1)}."],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"Find its value after {n} years, to the nearest cent.",
                    "number",
                    MONEY.fields(vn),
                    2,
                    [f"Value = {dollars(price)} × {float(q):g}^{n} = ${float(vn):,.2f}."],
                    b_d,
                    follow_through=f"part_a*({q.numerator}/{q.denominator})**{n - 1}",
                    sensible_range=(0, None),
                ),
                part(
                    f"Hence find the total amount it has depreciated by after {n} years, to the nearest cent.",
                    "number",
                    MONEY.fields(total),
                    1,
                    [f"Total = {dollars(price)} - ${float(vn_c):,.2f} = ${float(total):,.2f}."],
                    c_d,
                    follow_through=f"{price} - part_b",
                    sensible_range=(0, None),
                ),
                part(
                    f"After how many years will its value first be below {dollars(target)}?",
                    "number",
                    WHOLE.fields(k),
                    3,
                    [
                        f"After {k - 1} years: ${float(sched[k - 1]):,.2f}. After {k} years: ${float(sched[k]):,.2f}.",
                        f"{k} years is the first year the value is below {dollars(target)}.",
                    ],
                    d_d,
                    sensible_range=(0, None),
                ),
            ],
        )

    return retry_draw(rng, build)


# -- compare flat and reducing balance --------------------------------------------------------------------------------


def _cmp_sentence(name, price, r):
    return (
        f"{art(name).capitalize()} {name} is bought for {dollars(price)}. Its value can be depreciated at {pct_text(r)}% per year "
        f"by the flat-rate method (a percentage of the purchase price) or by the reducing-balance method (a percentage of the current value)."
    )


def _cmp_easy(rng):
    name, price = _asset(rng)
    r = rng.choice((5, 8, 10, 12, 15, 20, 25))
    flat = price * (100 - 2 * r) / 100
    red = price * (100 - r) ** 2 / 10000
    diff = red - flat
    check(diff == price * Fraction(r, 100) ** 2 and _is_cents(diff), "difference")
    d = MONEY.dist(
        diff,
        [
            (price * Fraction(r, 100), "cmp-flat-total-rate"),
            (diff * 10, "cmp-decimal-slip"),
            (diff / 10, "cmp-decimal-slip"),
            (price * Fraction(r, 100) ** 2 * 2, "cmp-doubles-difference"),
            (flat, "cmp-reports-one-method"),
            (flat + red, "cmp-difference-sign"),
        ],
    )
    enough(d, 3)
    text = (
        _cmp_sentence(name, price, r)
        + " After 2 years, by how many dollars is the reducing-balance value greater than the flat-rate value?"
    )
    working = [
        f"Flat-rate value = {dollars(price)} × (1 - 2 × {r}/100) = {dollars(flat)}.",
        f"Reducing-balance value = {dollars(price)} × {float(1 - Fraction(r, 100)):g}^2 = {dollars(red)}.",
        f"Difference = {dollars(red)} - {dollars(flat)} = {dollars(diff)}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(diff), answer_kind="number")


def _cmp_medium(rng):
    name, price = _asset(rng)
    r = rng.choice((5, 8, 10, 12, 15, 20))
    n = rng.randint(3, 5)
    q = Fraction(r, 100)
    flat = price * (1 - q * n)
    red = price * (1 - q) ** n
    if not clear_of_tie(red, 2):
        raise_retry()
    diff = round_cents(red) - flat
    sched = _red_schedule(price, r, n)
    check(round_cents(sched[n]) - flat == diff, "difference")
    d = MONEY.dist(
        diff,
        [
            (diff * 10, "cmp-decimal-slip"),
            (diff / 10, "cmp-decimal-slip"),
            (round_cents(red) - (price - price * q * n), "cmp-flat-total-rate")
            if n == 2
            else (round_cents(red) - price * (1 - q), "cmp-flat-total-rate"),
            (price * q * q, "cmp-flat-total-rate"),
            (flat, "cmp-reports-one-method"),
            (flat + red, "cmp-difference-sign"),
        ],
    )
    enough(d, 3)
    text = (
        _cmp_sentence(name, price, r)
        + f" After {n} years, by how many dollars is the reducing-balance value greater than the flat-rate value? Give the answer to the nearest cent."
    )
    working = [
        f"Flat-rate value = {dollars(price)} × (1 - {n} × {r}/100) = {dollars(flat)}.",
        f"Reducing-balance value = {dollars(price)} × {float(1 - q):g}^{n} = ${float(red):,.2f}.",
        f"Difference = ${float(red):,.2f} - {dollars(flat)} = ${float(diff):,.2f}.",
    ]
    return dict(text=text, working=working, distractors=d, **MONEY.fields(diff), answer_kind="number")


def _crossover(r_flat, r_red, limit=15):
    """First year n in which the reducing-balance value is above the flat-rate value, with the flat value still positive."""
    for n in range(1, limit):
        flat = 1 - Fraction(r_flat * n, 100)
        if flat <= 0:
            return None
        if (1 - Fraction(r_red, 100)) ** n > flat:
            return n
    return None


def _cmp_hard_crossover(rng):
    name, price = _asset(rng)
    for _ in attempts(200, "a crossover draw"):
        r_flat = rng.choice((8, 10, 12, 15, 20))
        r_red = rng.choice((15, 20, 25, 30, 35, 40))
        n = _crossover(r_flat, r_red)
        if r_red > r_flat + 3 and n is not None and 3 <= n <= 8:
            break
    else:
        raise_retry()
    flat_n = price * (1 - Fraction(r_flat * n, 100))
    red_n = price * (1 - Fraction(r_red, 100)) ** n
    check(
        red_n > flat_n
        and price * (1 - Fraction(r_red, 100)) ** (n - 1) <= price * (1 - Fraction(r_flat * (n - 1), 100)),
        "crossover",
    )
    d = WHOLE.dist(
        n,
        [
            (n - 1, "cmp-crossover-early"),
            (n + 1, "cmp-crossover-late"),
            (Fraction(100, r_flat), "cmp-year-flat-zero"),
            (Fraction(100, r_red), "cmp-year-flat-zero"),
            (n + 2, "cmp-crossover-late"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} is bought for {dollars(price)}. Under the flat-rate method it depreciates by {r_flat}% of its "
        f"purchase price each year; under the reducing-balance method it depreciates by {r_red}% of its current value each year. "
        f"In which year (the number of years since purchase) is the reducing-balance value first greater than the flat-rate value?"
    )
    working = [
        f"After {n - 1} years: flat ${float(price * (1 - Fraction(r_flat * (n - 1), 100))):,.2f}, reducing ${float(price * (1 - Fraction(r_red, 100)) ** (n - 1)):,.2f}.",
        f"After {n} years: flat ${float(flat_n):,.2f}, reducing ${float(red_n):,.2f}, so the reducing-balance value is first greater after {n} years.",
    ]
    return dict(text=text, working=working, distractors=d, **WHOLE.fields(n), answer_kind="number")


def _cmp_hard_equivalent(rng):
    name, price = _asset(rng)
    r = rng.choice((10, 20, 25, 30, 40, 50))
    n = rng.choice((2, 3, 4))
    drop = 1 - (1 - Fraction(r, 100)) ** n
    flat_rate = drop * 100 / n
    if not clear_of_tie(flat_rate, 1):
        raise_retry()
    d = DP1.dist(
        flat_rate,
        [
            (r, "cmp-equal-rate"),
            (drop * 100, "cmp-flat-rate-ignores-years"),
            (drop * price / n, "cmp-flat-rate-of-value"),
            (Fraction(r * n, 1), "cmp-flat-total-rate"),
            (flat_rate + 1, "cmp-crossover-late"),
        ],
    )
    enough(d, 3)
    text = (
        f"{art(name).capitalize()} {name} is bought for {dollars(price)} and depreciated at {r}% per year by the reducing-balance method. "
        f"What flat annual rate (as a percentage of the purchase price) would give the same value after {n} years? Give the answer to 1 decimal place."
    )
    working = [
        f"Value after {n} years = {dollars(price)} × {float(1 - Fraction(r, 100)):g}^{n}, so the value lost is {float(drop * 100):.2f}% of the price.",
        f"Flat rate = {float(drop * 100):.2f} / {n} = {float(flat_rate):.1f}% per year.",
    ]
    return dict(text=text, working=working, distractors=d, **DP1.fields(flat_rate), answer_kind="number")


_CMP_EASY = (_cmp_easy,)
_CMP_MEDIUM = (_cmp_medium,)
_CMP_HARD = (_cmp_hard_crossover, _cmp_hard_equivalent)


@generator(
    CMP_TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w3_compare_depreciation(rng, tier="easy"):
    """Flat-rate against reducing-balance depreciation; the tiers are described in the module docstring."""
    pool = {"easy": _CMP_EASY, "medium": _CMP_MEDIUM, "hard": _CMP_HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


@generator(CMP_EXAM, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w3_compare_depreciation(rng, tier="exam"):
    """Exam chain: the value under each method, the difference, and the year the reducing-balance value is higher."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        name, price = _asset(rng)
        for _ in attempts(300, "a comparison chain"):
            r_flat = rng.choice((8, 10, 12, 15, 20))
            r_red = rng.choice((15, 20, 25, 30, 35, 40))
            cross = _crossover(r_flat, r_red)
            if r_red > r_flat + 3 and cross is not None and 3 <= cross <= 8:
                break
        else:
            raise_retry()
        n = rng.randint(2, min(4, cross - 1))
        flat_n = price * (1 - Fraction(r_flat * n, 100))
        red_n = price * (1 - Fraction(r_red, 100)) ** n
        if not clear_of_tie(red_n, 2) or flat_n <= 0:
            raise_retry()
        red_c = round_cents(red_n)
        diff = flat_n - red_c
        if diff <= 0:
            raise_retry()
        a_d = MONEY.dist(
            flat_n,
            [
                (price * (1 - Fraction(r_flat, 100)), "fdep-one-year-only"),
                (price * r_flat * n / 100, "fdep-amount-not-value"),
                (price * (1 - Fraction(r_flat, 100)) ** n, "fdep-uses-reducing"),
                (price * (1 + Fraction(r_flat * n, 100)), "fdep-adds-to-price"),
            ],
        )
        b_d = MONEY.dist(
            red_n,
            [
                (flat_n if flat_n != red_n else None, "rbd-uses-flat"),
                (price * (1 - Fraction(r_red * n, 100)) if r_red * n < 100 else None, "rbd-multiplies-by-years"),
                (price * (1 - Fraction(r_red, 100)) ** (n - 1), "rbd-one-year-short"),
                (price * (1 - Fraction(r_red, 100)) ** (n + 1), "rbd-extra-year"),
            ],
        )
        c_d = MONEY.dist(
            diff,
            [
                (diff * 10, "cmp-decimal-slip"),
                (diff / 10, "cmp-decimal-slip"),
                (flat_n, "cmp-reports-one-method"),
                (red_c, "cmp-reports-one-method"),
                (flat_n + red_c, "cmp-difference-sign"),
            ],
        )
        d_d = WHOLE.dist(
            cross,
            [
                (cross - 1, "cmp-crossover-early"),
                (cross + 1, "cmp-crossover-late"),
                (Fraction(100, r_flat), "cmp-year-flat-zero"),
                (cross + 2, "cmp-crossover-late"),
            ],
        )
        for x, m in ((a_d, 3), (b_d, 3), (c_d, 3), (d_d, 2)):
            enough(x, m)
        stem = (
            f"{art(name).capitalize()} {name} is bought for {dollars(price)}. Method A depreciates it by {r_flat}% of the purchase price each year "
            f"(flat rate). Method B depreciates it by {r_red}% of its current value each year (reducing balance)."
        )
        return dict(
            text=stem,
            parts=[
                part(
                    f"Find the value after {n} years under Method A, in dollars.",
                    "number",
                    MONEY.fields(flat_n),
                    1,
                    [f"Value = {dollars(price)} × (1 - {n} × {r_flat}/100) = {dollars(flat_n)}."],
                    a_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"Find the value after {n} years under Method B, to the nearest cent.",
                    "number",
                    MONEY.fields(red_n),
                    2,
                    [f"Value = {dollars(price)} × {float(1 - Fraction(r_red, 100)):g}^{n} = ${float(red_n):,.2f}."],
                    b_d,
                    sensible_range=(0, None),
                ),
                part(
                    f"By how many dollars is the Method A value greater than the Method B value after {n} years, to the nearest cent?",
                    "number",
                    MONEY.fields(diff),
                    1,
                    [f"Difference = {dollars(flat_n)} - ${float(red_c):,.2f} = ${float(diff):,.2f}."],
                    c_d,
                    follow_through="part_a - part_b",
                    sensible_range=(0, None),
                ),
                part(
                    "In which year (the number of years since purchase) is the Method B value first greater than the Method A value?",
                    "number",
                    WHOLE.fields(cross),
                    2,
                    [f"The Method B value is first greater after {cross} years."],
                    d_d,
                    sensible_range=(0, None),
                ),
            ],
        )

    return retry_draw(rng, build)
