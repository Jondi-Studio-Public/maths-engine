"""CPI, inflation and real values: `r3e_cpi` (easy, medium, hard) and the exam chain `exd_x_r3e_cpi`.

``cpi-and-real-value`` (``financial-discrete``, answer ``number``). Every question prints the index numbers it uses
in a small table; no index number is a fact about a real year.

* easy: the inflation rate between two index numbers, to 1 decimal place.
* medium: what an item that cost a stated amount in the earlier year costs in the later year, if its price follows
  the index.
* hard: the fall in the real value of a pay rise that did not keep up with prices; or the price after three years of
  stated yearly inflation rates, compounded.
* exam (``exd_x_r3e_cpi``): the inflation rate, the later price of an item, the real value of a later salary in
  earlier dollars, and the percentage change in real pay.

Each real value is checked a second way, by comparing both amounts in the later year's dollars.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._exd_distinct_parts import distinct_part_answers
from ._r3e_kit import PEOPLE, dollars, number_part, number_text, single_number

_ITEMS = [
    ("a weekly grocery shop", 90, 260),
    ("a return train fare", 40, 120),
    ("a pair of school shoes", 60, 160),
    ("a quarter's worth of electricity", 250, 600),
    ("a full tank of petrol", 70, 150),
    ("a restaurant meal for two", 60, 140),
]


def _item_price(rng):
    item, lo, hi = rng.choice(_ITEMS)
    return item, Fraction(5 * rng.randint(-(-lo // 5), hi // 5))


def _cpi(rng):
    """(year A, CPI A, year B, CPI B): index numbers to 1 decimal place, B later and higher."""
    a = rng.randint(2012, 2021)
    gap = rng.randint(2, 4)
    ca = Fraction(rng.randint(950, 1250), 10)
    cb = ca + Fraction(rng.randint(40, 190), 10)
    return a, ca, a + gap, cb


def _cpi_table(a, ca, b, cb):
    return f"The CPI was {number_text(ca, 1)} in {a} and {number_text(cb, 1)} in {b}."


def _clear_of_tie(value, places):
    return (Fraction(value) * 10**places * 2).denominator != 1 or Fraction(value) * 10**places * 2 % 2 != 1


def _easy(rng):
    for _ in attempts(500, "an inflation rate clear of a rounding tie"):
        a, ca, b, cb = _cpi(rng)
        rate = (cb - ca) * 100 / ca
        if (rate * 10).denominator != 2:
            break
    name, _job = rng.choice(PEOPLE)
    text = phrase(
        rng,
        [
            "The consumer price index (CPI) is used to measure inflation. {table} What was the rate of inflation "
            "from {a} to {b}? Give your answer as a percentage to 1 decimal place.",
            "{table} Use these figures to find the rate of inflation from {a} to {b}, as a percentage correct to 1 "
            "decimal place.",
            "Inflation is measured by the change in the consumer price index (CPI). {table} Calculate the percentage "
            "rate of inflation between {a} and {b}, correct to 1 decimal place.",
        ],
        table=_cpi_table(a, ca, b, cb),
        a=a,
        b=b,
    )
    slips = [
        (cb * 100 / ca, "r3e-index-ratio-given-as-rate"),
        (cb - ca, "r3e-index-points-given-as-percentage"),
        ((cb - ca) * 100 / cb, "r3e-change-over-later-value"),
        ((cb - ca) / ca, "r3e-ratio-not-percentage"),
    ]
    working = [
        f"Change in CPI = {number_text(cb, 1)} - {number_text(ca, 1)} = {number_text(cb - ca, 1)}.",
        f"Inflation = {number_text(cb - ca, 1)} / {number_text(ca, 1)} x 100 = {number_text(rate, 1)}%.",
    ]
    return single_number(text, rate, 1, working, slips)


def _medium(rng):
    a, ca, b, cb = _cpi(rng)
    item, price = _item_price(rng)
    later = price * cb / ca
    text = (
        f"{_cpi_table(a, ca, b, cb)} In {a}, {item} cost {dollars(price)}. "
        f"If its price follows the CPI, what would it cost in {b}? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (price * ca / cb, "r3e-index-ratio-inverted"),
        (price + (cb - ca), "r3e-index-points-added-to-price"),
        (price * (1 + (cb - ca) / 100), "r3e-index-points-taken-as-percentage"),
        (price * cb / 100, "r3e-base-index-taken-as-100"),
    ]
    working = [f"Price in {b} = {dollars(price)} x {number_text(cb, 1)} / {number_text(ca, 1)} = {dollars(later, 2)}."]
    return single_number(text, later, 2, working, slips)


def _hard_real_fall(rng):
    for _ in attempts(500, "a rise that does not keep up with prices"):
        a, ca, b, cb = _cpi(rng)
        s_a = 1000 * rng.randint(48, 110)
        rise = Fraction(rng.randint(5, 40), 10)
        s_b = s_a * (1000 + 10 * rise) / 1000
        real_b = s_b * ca / cb
        if real_b < s_a:
            break
    # compare both in the later year's dollars as the second check
    check((s_a * cb / ca > s_b) == (real_b < s_a), "the two comparisons disagree")
    fall = s_a - real_b
    name, job = rng.choice(PEOPLE)
    text = (
        f"{_cpi_table(a, ca, b, cb)} {name} is {job}. In {a}, {name}'s salary "
        f"was {dollars(s_a)}. In {b} it is {dollars(s_b)}, a rise of {number_text(rise, 1)}%. In {a} dollars, by how "
        f"much has the real value of {name}'s salary fallen? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (s_a * cb / ca - s_b, "r3e-fall-in-later-dollars"),
        (s_a * (cb - ca) / 100 - (s_b - s_a), "r3e-index-points-taken-as-percentage"),
        (s_b - s_a, "r3e-nominal-rise-given-as-real-fall"),
        (s_a - s_b * cb / ca, "r3e-index-ratio-inverted"),
        (s_a * (cb - ca) / ca, "r3e-price-rise-given-as-real-fall"),
    ]
    working = [
        f"Salary in {a} dollars = {dollars(s_b, 2)} x {number_text(ca, 1)} / {number_text(cb, 1)} = {dollars(real_b, 2)}.",
        f"Fall = {dollars(s_a)} - {dollars(real_b, 2)} = {dollars(fall, 2)}.",
    ]
    return single_number(text, fall, 2, working, slips)


def _hard_compound(rng):
    item, price = _item_price(rng)
    rates = [Fraction(rng.randint(10, 70), 10) for _ in range(3)]
    value = price
    for r in rates:
        value *= 1 + r / 100
    mult = (1 + rates[0] / 100) * (1 + rates[1] / 100) * (1 + rates[2] / 100)
    check(value == price * mult, "compounded price disagrees")
    text = (
        f"{item.capitalize()} cost {dollars(price)} at the start of 2022. Inflation was {number_text(rates[0], 1)}% in "
        f"2022, {number_text(rates[1], 1)}% in 2023 and {number_text(rates[2], 1)}% in 2024, and the price rose with "
        "inflation each year. What did it cost at the end of 2024? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (price * (1 + sum(rates) / 100), "r3e-yearly-rates-added"),
        (price * (1 + rates[0] / 100) ** 3, "r3e-first-rate-repeated"),
        (price * (1 + rates[1] / 100) * (1 + rates[2] / 100), "r3e-year-left-out"),
        (price * (1 + (sum(rates) / 3) / 100) ** 2, "r3e-wrong-number-of-years"),
    ]
    v1 = price * (1 + rates[0] / 100)
    working = [
        f"After 2022: {dollars(price)} x {1 + rates[0] / 100:.3f} = {dollars(v1, 2)}.",
        f"Multiplying by the 2023 and 2024 factors in turn gives {dollars(value, 2)}.",
    ]
    return single_number(text, value, 2, working, slips)


@generator(
    "r3e_cpi",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3e_cpi(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_real_fall, _hard_compound])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r3e_cpi", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_cpi(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a chain clear of rounding ties"):
        a, ca, b, cb = _cpi(rng)
        rate = (cb - ca) * 100 / ca
        s_a = 1000 * rng.randint(48, 110)
        s_b = s_a + 100 * rng.randint(5, 30)
        real = s_b * ca / cb
        change = (real - s_a) * 100 / s_a
        if (rate * 10).denominator != 2 and (change * 10).denominator != 2 and change != 0:
            break
    item, price = _item_price(rng)
    later = price * cb / ca
    name, job = rng.choice(PEOPLE)
    text = (
        f"{_cpi_table(a, ca, b, cb)} {name} is {job}. In {a}, {name}'s salary was "
        f"{dollars(s_a)}, and {item} cost {dollars(price)}. In {b}, {name}'s salary is {dollars(s_b)}."
    )
    parts = [
        number_part(
            f"Find the rate of inflation from {a} to {b}, as a percentage to 1 decimal place.",
            rate,
            1,
            1,
            [f"({number_text(cb, 1)} - {number_text(ca, 1)}) / {number_text(ca, 1)} x 100 = {number_text(rate, 1)}%."],
            [
                (cb * 100 / ca, "r3e-index-ratio-given-as-rate"),
                ((cb - ca) * 100 / cb, "r3e-change-over-later-value"),
                (cb - ca, "r3e-index-points-given-as-percentage"),
            ],
            sensible_range=(0, 100),
        ),
        number_part(
            f"If its price follows the CPI, find what {item} costs in {b}, in dollars.",
            later,
            2,
            2,
            [f"{dollars(price)} x {number_text(cb, 1)} / {number_text(ca, 1)} = {dollars(later, 2)}."],
            [
                (price * ca / cb, "r3e-index-ratio-inverted"),
                (price + (cb - ca), "r3e-index-points-added-to-price"),
                (price * cb / 100, "r3e-base-index-taken-as-100"),
            ],
        ),
        number_part(
            f"Find the real value of {name}'s {b} salary, in {a} dollars.",
            real,
            2,
            2,
            [f"{dollars(s_b)} x {number_text(ca, 1)} / {number_text(cb, 1)} = {dollars(real, 2)}."],
            [
                (s_b * cb / ca, "r3e-index-ratio-inverted"),
                (s_b - (cb - ca), "r3e-index-points-subtracted-from-salary"),
                (s_b * (1 - (cb - ca) / 100), "r3e-index-points-taken-as-percentage"),
            ],
        ),
        number_part(
            f"By what percentage did {name}'s real salary change from {a} to {b}? Give your answer to 1 decimal place "
            "(a fall is negative).",
            change,
            1,
            2,
            [f"({dollars(real, 2)} - {dollars(s_a)}) / {dollars(s_a)} x 100 = {number_text(change, 1)}%."],
            [
                ((s_b - s_a) * 100 / s_a, "r3e-nominal-change-given-as-real"),
                ((real - s_a) * 100 / real, "r3e-change-over-later-value"),
                (real - s_a, "r3e-dollars-given-as-percentage"),
                ((real - s_a) / s_a, "r3e-ratio-not-percentage"),
            ],
            follow_through=f"100*(part_c - {s_a})/{s_a}",
            sensible_range=(-100, 100),
        ),
    ]
    return dict(text=text, parts=parts)
