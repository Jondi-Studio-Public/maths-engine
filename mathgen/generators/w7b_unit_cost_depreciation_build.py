"""Generates `w7b_unit_cost_depreciation` questions (easy, medium, hard) and the exam chain
`exd_x_w7b_unit_cost_depreciation`.

``unit-cost-depreciation`` (``financial-discrete``, answer ``number``): an asset loses value for each unit of use
(kilometre driven, hour run), so its value depends on how much it has been used and not on how old it is.

* easy: the depreciation after a number of units at a stated rate (the rate is sometimes given in cents); the value
  after that use; or the rate per unit from a total depreciation and the units used.
* medium: the rate per unit from the purchase price, the salvage value and the expected life in units; the value after
  some use when that rate has to be found first; or the units used to reach a stated value.
* hard: a rate that changes after a stated number of units (the value after some use, or the units needed to reach a
  value), the value at the end of a year when each year's use differs, or the value after more use when two readings of
  the value are given and the rate is not.
* exam (``exd_x_w7b_unit_cost_depreciation``): rate per unit, value after use, units to reach a target value, and the
  difference from a flat-rate (per year) valuation.

Every amount is built backwards from a rate in whole cents and whole numbers of units, so each answer is an exact
number of cents (given to two decimal places) or a whole number of units, and every answer is re-derived a second way
from the depreciation schedule (the value read back unit by unit across the stages).
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from ._w6a_kit import enough, num_dist, num_fields, part, retry_draw

TOPIC = "w7b_unit_cost_depreciation"
EXAM_TOPIC = "exd_x_w7b_unit_cost_depreciation"

AREA = "unit-cost-depreciation"
MISCONCEPTIONS = [
    (
        "ucd-cents-not-converted",
        "Uses a rate given in cents as if it were dollars, so the amount is 100 times too large.",
        AREA,
    ),
    (
        "ucd-decimal-slip",
        "Misplaces the decimal point in the product of the rate and the units, so the amount is ten times too large or too small.",
        AREA,
    ),
    ("ucd-rate-inverted", "Divides the units by the amount depreciated instead of the amount by the units.", AREA),
    ("ucd-adds-rate-to-units", "Adds the rate to the number of units instead of multiplying them.", AREA),
    ("ucd-amount-not-value", "Gives the amount depreciated where the value after use was asked.", AREA),
    ("ucd-adds-to-price", "Adds the depreciation to the purchase price instead of subtracting it.", AREA),
    ("ucd-one-unit-only", "Subtracts the depreciation for a single unit instead of for all the units used.", AREA),
    (
        "ucd-ignores-salvage",
        "Divides the purchase price by the expected life in units, forgetting to subtract the salvage value first.",
        AREA,
    ),
    (
        "ucd-adds-salvage",
        "Adds the salvage value to the purchase price before dividing, instead of subtracting it.",
        AREA,
    ),
    (
        "ucd-salvage-only",
        "Divides the salvage value, not the amount to be depreciated, by the expected life in units.",
        AREA,
    ),
    (
        "ucd-units-from-value",
        "Divides the target value by the rate instead of the depreciation needed to reach it.",
        AREA,
    ),
    ("ucd-units-from-price", "Divides the purchase price by the rate, ignoring the value the asset is to reach.", AREA),
    ("ucd-multiplies-for-units", "Multiplies the depreciation needed by the rate instead of dividing by it.", AREA),
    (
        "ucd-single-rate",
        "Uses the first rate for every unit of use, ignoring that the rate changes after a stated number of units.",
        AREA,
    ),
    (
        "ucd-second-rate-everywhere",
        "Uses the later rate for every unit of use, ignoring that the first units depreciated at the earlier rate.",
        AREA,
    ),
    (
        "ucd-forgets-first-stage",
        "Depreciates only the units in the second stage and forgets the depreciation already taken in the first.",
        AREA,
    ),
    (
        "ucd-second-stage-only",
        "Gives the units used in the second stage only, where the total units used were asked.",
        AREA,
    ),
    ("ucd-first-year-usage", "Uses the first year's use for every year instead of each year's own use.", AREA),
    ("ucd-last-year-only", "Depreciates only the last year's use instead of all the years' use so far.", AREA),
    (
        "ucd-divides-by-total-units",
        "Divides the fall in value by the larger number of units, not by the difference in units between the two readings.",
        AREA,
    ),
    (
        "ucd-reading-as-price",
        "Takes the earlier reading of the value as the purchase price when finding a later value.",
        AREA,
    ),
    (
        "ucd-years-miscounted",
        "Uses one year too many or one year too few when working out the flat-rate depreciation.",
        AREA,
    ),
    (
        "ucd-compares-depreciation-only",
        "Gives the depreciation under one method where the difference between the two values was asked.",
        AREA,
    ),
]

_ASSETS = (
    # (description, unit, units, rate range in cents, life range in units, step of life)
    ("delivery van", "kilometre", "kilometres", (12, 40), (90, 240), 1000, 60_000),
    ("taxi", "kilometre", "kilometres", (10, 30), (150, 400), 1000, 70_000),
    ("hire truck", "kilometre", "kilometres", (25, 70), (100, 300), 1000, 120_000),
    ("earthmoving machine", "hour", "hours", (800, 2400), (6, 20), 1000, 280_000),
    ("commercial oven", "hour", "hours", (150, 650), (8, 30), 1000, 120_000),
    ("industrial press", "hour", "hours", (400, 1500), (5, 18), 1000, 150_000),
)


# -- money ------------------------------------------------------------------------------------------------------------


def _cents(value) -> int:
    """`value` (dollars as an int or Fraction) as a whole number of cents; refuses anything that is not exact."""
    c = Fraction(value) * 100
    check(c.denominator == 1, f"{value} is not a whole number of cents")
    return int(c)


def _money_text(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"


def _money_fields(cents: int) -> dict:
    text = _money_text(cents)
    return dict(answer=float(text), answer_text=text, answer_latex=text)


def _in_text(cents: int) -> str:
    """A dollar amount as the question prints it: $12,345 or $12,345.50."""
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}" + ("" if cents % 100 == 0 else f".{cents % 100:02d}")


def _rate_text(cents: int, as_cents: bool) -> str:
    return f"{cents} cents" if as_cents else _in_text(cents)


def _money_dist(correct_cents: int, candidates, n: int = 4) -> list[Distractor]:
    """Tagged money distractors ``[(cents or Fraction, id), ...]``: rounded to the cent, kept at least 2 cents from the
    answer and from each other, so the 2 d.p. grader (half a cent either way) cannot accept any of them."""
    seen = [correct_cents]
    out: list[Distractor] = []
    for item in candidates:
        if item is None or item[0] is None:
            continue
        value, tag = item
        cents = int(Fraction(value).__round__())
        if cents < 0 or any(abs(cents - s) < 2 for s in seen):
            continue
        seen.append(cents)
        text = _money_text(cents)
        out.append(Distractor(text, tag, float(text)))
        if len(out) == n:
            break
    return out


def _units_dist(correct: int, candidates, n: int = 4) -> list[Distractor]:
    """Tagged whole-unit distractors: each candidate is rounded to a whole number of units."""
    rounded = []
    for value, tag in candidates:
        if value is None:
            continue
        v = int(Fraction(value).__round__())
        if v > 0:
            rounded.append((v, tag))
    out = num_dist(correct, rounded, n)
    return out


# -- schedule: the second way ----------------------------------------------------------------------------------------


def _value_by_schedule(price_cents: int, stages, units: int) -> int:
    """Value in cents after `units` of use, read stage by stage from a ``[(units in stage or None, cents per unit)]`` list."""
    left, value = units, price_cents
    for size, rate in stages:
        used = left if size is None else min(left, size)
        value -= used * rate
        left -= used
    check(left == 0, "stages do not cover the use")
    return value


# -- asset draw ------------------------------------------------------------------------------------------------------


def _asset(rng):
    desc, one, many, (rlo, rhi), (nlo, nhi), step, cap = rng.choice(_ASSETS)
    for _ in attempts(200, "an asset whose total depreciation is plausible"):
        rate = rng.randint(rlo, rhi)  # cents per unit
        if rate >= 100:
            rate = rate // 5 * 5
        life = rng.randint(nlo, nhi) * step
        if rate * life <= cap * 100:
            break
    salvage = rng.randint(4, 40) * 500
    price_cents = salvage * 100 + rate * life
    return dict(desc=desc, one=one, many=many, rate=rate, life=life, salvage=salvage * 100, price=price_cents)


def _art(desc: str) -> str:
    return "an" if desc[0] in "aeiou" else "a"


def _rate_say(a, as_cents: bool) -> str:
    return f"{_rate_text(a['rate'], as_cents)} per {a['one']}"


def _n(n: int) -> str:
    return f"{n:,}"


def _use(rng, a, lo=0.15, hi=0.7, step=100) -> int:
    return rng.randint(int(a["life"] * lo) // step, int(a["life"] * hi) // step) * step


# -- easy -------------------------------------------------------------------------------------------------------------


def _easy_amount(rng):
    a = _asset(rng)
    n = _use(rng, a, 0.05, 0.4, 100)
    as_cents = a["rate"] < 100 and rng.random() < 0.5
    total = a["rate"] * n
    check(_value_by_schedule(a["price"], [(None, a["rate"])], n) == a["price"] - total, "amount differs")
    cands = [
        (total * 10, "ucd-decimal-slip"),
        (total // 10, "ucd-decimal-slip"),
        ((n + a["rate"]) * 100, "ucd-adds-rate-to-units"),
        (Fraction(n * 100 * 100, a["rate"]), "ucd-rate-inverted"),
    ]
    if as_cents:
        cands = [(total * 100, "ucd-cents-not-converted")] + cands
    dist = _money_dist(total, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "{Art} {d} depreciates at {r}. It has been used for {n} {u}. By how many dollars has it depreciated?",
            "The unit-cost method depreciates {art} {d} by {r}. After {n} {u} of use, how much has it depreciated, in dollars?",
        ),
        d=a["desc"],
        art=_art(a["desc"]),
        Art=_art(a["desc"]).capitalize(),
        r=_rate_say(a, as_cents),
        n=_n(n),
        u=a["many"],
    )
    working = [
        f"Depreciation = rate per {a['one']} x {a['many']} used = {_rate_text(a['rate'], as_cents)} x {_n(n)}.",
        f"= {_in_text(total)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_money_fields(total), answer_kind="number")


def _easy_value(rng):
    a = _asset(rng)
    n = _use(rng, a, 0.05, 0.5, 100)
    total = a["rate"] * n
    value = a["price"] - total
    check(_value_by_schedule(a["price"], [(None, a["rate"])], n) == value, "value differs")
    cands = [
        (total, "ucd-amount-not-value"),
        (a["price"] + total, "ucd-adds-to-price"),
        (a["price"] - a["rate"], "ucd-one-unit-only"),
        (a["price"] - total * 10, "ucd-decimal-slip"),
        (a["price"] - total // 10, "ucd-decimal-slip"),
    ]
    dist = _money_dist(value, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "{Art} {d} was bought for {p} and depreciates at {r}. What is its value, in dollars, after {n} {u} of use?",
            "A business buys {art} {d} for {p}. Using unit-cost depreciation at {r}, find its value in dollars after {n} {u}.",
        ),
        d=a["desc"],
        art=_art(a["desc"]),
        Art=_art(a["desc"]).capitalize(),
        p=_in_text(a["price"]),
        r=_rate_say(a, False),
        n=_n(n),
        u=a["many"],
    )
    working = [
        f"Depreciation = {_in_text(a['rate'])} x {_n(n)} = {_in_text(total)}.",
        f"Value = {_in_text(a['price'])} - {_in_text(total)} = {_in_text(value)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_money_fields(value), answer_kind="number")


def _easy_rate(rng):
    a = _asset(rng)
    n = _use(rng, a, 0.05, 0.4, 100)
    total = a["rate"] * n
    check(Fraction(total, n) == a["rate"], "rate differs")
    cands = [
        (Fraction(n * 100, total), "ucd-rate-inverted"),
        (a["rate"] * 10, "ucd-decimal-slip"),
        (a["rate"] // 10, "ucd-decimal-slip"),
        (a["rate"] * 100, "ucd-cents-not-converted"),
    ]
    dist = _money_dist(a["rate"], cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "{Art} {d} has depreciated by {t} after {n} {u} of use. What is the depreciation per {o}, in dollars?",
            "After {n} {u} of use {art} {d} has lost {t} of its value through unit-cost depreciation. Find the depreciation per {o}, in dollars.",
        ),
        d=a["desc"],
        art=_art(a["desc"]),
        Art=_art(a["desc"]).capitalize(),
        t=_in_text(total),
        n=_n(n),
        u=a["many"],
        o=a["one"],
    )
    working = [
        f"Rate = depreciation / {a['many']} = {_in_text(total)} / {_n(n)}.",
        f"= {_in_text(a['rate'])} per {a['one']}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_money_fields(a["rate"]), answer_kind="number")


# -- medium -----------------------------------------------------------------------------------------------------------


def _medium_rate(rng):
    a = _asset(rng)
    cands = [
        (Fraction(a["price"], a["life"]), "ucd-ignores-salvage"),
        (Fraction(a["price"] + a["salvage"], a["life"]), "ucd-adds-salvage"),
        (Fraction(a["salvage"], a["life"]), "ucd-salvage-only"),
        (a["rate"] * 10, "ucd-decimal-slip"),
    ]
    check(Fraction(a["price"] - a["salvage"], a["life"]) == a["rate"], "rate differs")
    dist = _money_dist(a["rate"], cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "{Art} {d} costs {p} and is expected to be worth {s} after {n} {u} of use. Using unit-cost depreciation, what is the "
            "depreciation per {o}, in dollars?",
            "{Art} {d} is bought for {p}. Its salvage value will be {s} once it has run for {n} {u}. Find the depreciation per {o}, in dollars.",
        ),
        d=a["desc"],
        art=_art(a["desc"]),
        Art=_art(a["desc"]).capitalize(),
        p=_in_text(a["price"]),
        s=_in_text(a["salvage"]),
        n=_n(a["life"]),
        u=a["many"],
        o=a["one"],
    )
    working = [
        f"Total depreciation = {_in_text(a['price'])} - {_in_text(a['salvage'])} = {_in_text(a['price'] - a['salvage'])}.",
        f"Rate = {_in_text(a['price'] - a['salvage'])} / {_n(a['life'])} = {_in_text(a['rate'])} per {a['one']}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_money_fields(a["rate"]), answer_kind="number")


def _medium_value_after(rng):
    a = _asset(rng)
    n = _use(rng, a, 0.15, 0.7, 100)
    value = a["price"] - a["rate"] * n
    check(_value_by_schedule(a["price"], [(None, a["rate"])], n) == value, "value differs")
    dep = a["rate"] * n
    cands = [
        (dep, "ucd-amount-not-value"),
        (a["price"] - Fraction(a["price"], a["life"]) * n, "ucd-ignores-salvage"),
        (a["price"] - Fraction(a["salvage"], a["life"]) * n, "ucd-salvage-only"),
        (a["price"] + dep, "ucd-adds-to-price"),
        (a["price"] - Fraction(a["price"] + a["salvage"], a["life"]) * n, "ucd-adds-salvage"),
    ]
    dist = _money_dist(value, cands)
    enough(dist)
    stem = (
        f"{_art(a['desc']).capitalize()} {a['desc']} costs {_in_text(a['price'])} and is expected to be worth {_in_text(a['salvage'])} after {_n(a['life'])} {a['many']} of use. "
        f"Using unit-cost depreciation, what is its value, in dollars, after {_n(n)} {a['many']}?"
    )
    working = [
        f"Rate = ({_in_text(a['price'])} - {_in_text(a['salvage'])}) / {_n(a['life'])} = {_in_text(a['rate'])} per {a['one']}.",
        f"Value = {_in_text(a['price'])} - {_in_text(a['rate'])} x {_n(n)} = {_in_text(value)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_money_fields(value), answer_kind="number")


def _units_to_value(rng, a):
    """(units, target value in cents): the target is reached after a whole number of hundreds of units."""
    n = _use(rng, a, 0.2, 0.85, 100)
    return n, a["price"] - a["rate"] * n


def _medium_units(rng):
    a = _asset(rng)
    n, target = _units_to_value(rng, a)
    check(Fraction(a["price"] - target, a["rate"]) == n, "units differ")
    need = a["price"] - target
    cands = [
        (Fraction(target, a["rate"]), "ucd-units-from-value"),
        (Fraction(a["price"], a["rate"]), "ucd-units-from-price"),
        (Fraction(need * a["rate"], 100 * 100), "ucd-multiplies-for-units"),
    ]
    dist = _units_dist(n, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "{Art} {d} was bought for {p} and depreciates at {r}. After how many {u} of use will its value be {v}?",
            "Using unit-cost depreciation at {r}, {art} {d} bought for {p} loses value as it is used. How many {u} must it be used for before it is worth {v}?",
        ),
        d=a["desc"],
        art=_art(a["desc"]),
        Art=_art(a["desc"]).capitalize(),
        p=_in_text(a["price"]),
        r=_rate_say(a, False),
        v=_in_text(target),
        u=a["many"],
    )
    working = [
        f"Depreciation needed = {_in_text(a['price'])} - {_in_text(target)} = {_in_text(need)}.",
        f"Units = {_in_text(need)} / {_in_text(a['rate'])} = {_n(n)} {a['many']}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(n), answer_kind="number")


# -- hard -------------------------------------------------------------------------------------------------------------


def _two_stage(rng):
    a = _asset(rng)
    n1 = _use(rng, a, 0.15, 0.4, 1000)
    r1 = a["rate"]
    r2 = (
        max(5, int(r1 * rng.choice((0.5, 0.6, 0.75, 1.25, 1.5)) / 5) * 5)
        if r1 >= 100
        else max(2, int(r1 * rng.choice((0.5, 0.6, 0.75, 1.25, 1.5))))
    )
    if r2 == r1:
        r2 = r1 + 5
    a["r2"] = r2
    a["n1"] = n1
    return a


def _hard_two_stage_value(rng):
    for _ in attempts(200, "a two-stage draw above salvage"):
        a = _two_stage(rng)
        n = a["n1"] + rng.randint(5, 25) * 1000
        r1, r2, n1 = a["rate"], a["r2"], a["n1"]
        value = a["price"] - r1 * n1 - r2 * (n - n1)
        if value >= a["salvage"]:
            break
    check(_value_by_schedule(a["price"], [(n1, r1), (None, r2)], n) == value, "value differs")
    cands = [
        (a["price"] - r1 * n, "ucd-single-rate"),
        (a["price"] - r2 * n, "ucd-second-rate-everywhere"),
        (a["price"] - r2 * (n - n1), "ucd-forgets-first-stage"),
        (a["price"] - r1 * n1 - r2 * n, "ucd-second-rate-everywhere"),
    ]
    dist = _money_dist(value, cands)
    enough(dist)
    stem = (
        f"{_art(a['desc']).capitalize()} {a['desc']} is bought for {_in_text(a['price'])}. It depreciates by {_in_text(r1)} per {a['one']} for the first {_n(n1)} {a['many']} "
        f"and by {_in_text(r2)} per {a['one']} after that. What is its value, in dollars, after {_n(n)} {a['many']} of use?"
    )
    working = [
        f"First {_n(n1)} {a['many']}: {_in_text(r1)} x {_n(n1)} = {_in_text(r1 * n1)}.",
        f"Next {_n(n - n1)} {a['many']}: {_in_text(r2)} x {_n(n - n1)} = {_in_text(r2 * (n - n1))}.",
        f"Value = {_in_text(a['price'])} - {_in_text(r1 * n1)} - {_in_text(r2 * (n - n1))} = {_in_text(value)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_money_fields(value), answer_kind="number")


def _hard_two_stage_units(rng):
    for _ in attempts(200, "a two-stage draw above salvage"):
        a = _two_stage(rng)
        r1, r2, n1 = a["rate"], a["r2"], a["n1"]
        n2 = rng.randint(5, 25) * 1000
        target = a["price"] - r1 * n1 - r2 * n2
        if target >= a["salvage"]:
            break
    n = n1 + n2
    check(_value_by_schedule(a["price"], [(n1, r1), (None, r2)], n) == target, "target differs")
    cands = [
        (Fraction(a["price"] - target, r1), "ucd-single-rate"),
        (Fraction(a["price"] - target, r2), "ucd-second-rate-everywhere"),
        (n2, "ucd-second-stage-only"),
        (n1 + Fraction(a["price"] - target, r2), "ucd-forgets-first-stage"),
    ]
    dist = _units_dist(n, cands)
    enough(dist)
    stem = (
        f"{_art(a['desc']).capitalize()} {a['desc']} is bought for {_in_text(a['price'])}. It depreciates by {_in_text(r1)} per {a['one']} for the first {_n(n1)} {a['many']} "
        f"and by {_in_text(r2)} per {a['one']} after that. After how many {a['many']} in total is its value {_in_text(target)}?"
    )
    working = [
        f"First {_n(n1)} {a['many']} take off {_in_text(r1 * n1)}, leaving {_in_text(a['price'] - r1 * n1)}.",
        f"The remaining fall of {_in_text(a['price'] - r1 * n1 - target)} at {_in_text(r2)} per {a['one']} takes {_n(n2)} {a['many']}.",
        f"Total = {_n(n1)} + {_n(n2)} = {_n(n)} {a['many']}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(n), answer_kind="number")


def _hard_yearly(rng):
    for _ in attempts(200, "a yearly-use draw above salvage"):
        a = _asset(rng)
        years = rng.randint(3, 5)
        usage = [rng.randint(8, 40) * 500 for _ in range(years)]
        if len(set(usage)) < 2:
            usage[-1] += 500
        k = rng.randint(2, years)
        rate = a["rate"]
        total_cents = rate * sum(usage[:k])
        price = a["price"]
        value = price - total_cents
        if value >= a["salvage"] and sum(usage) < a["life"]:
            break
    check(_value_by_schedule(price, [(None, rate)], sum(usage[:k])) == value, "yearly value differs")
    cands = [
        (price - rate * usage[0] * k, "ucd-first-year-usage"),
        (price - rate * usage[k - 1], "ucd-last-year-only"),
        (total_cents, "ucd-amount-not-value"),
        (price - rate * sum(usage), "ucd-first-year-usage") if k != years else None,
    ]
    dist = _money_dist(value, cands)
    enough(dist)
    listing = ", ".join(_n(u) for u in usage[:k])
    stem = (
        f"{_art(a['desc']).capitalize()} {a['desc']} is bought for {_in_text(price)} and depreciated by {_in_text(rate)} per {a['one']}. In its first {k} years it is used for "
        f"{listing} {a['many']} (in that order). What is its value, in dollars, at the end of year {k}?"
    )
    working = [
        f"Use so far = {' + '.join(_n(u) for u in usage[:k])} = {_n(sum(usage[:k]))} {a['many']}.",
        f"Depreciation = {_in_text(rate)} x {_n(sum(usage[:k]))} = {_in_text(total_cents)}.",
        f"Value = {_in_text(price)} - {_in_text(total_cents)} = {_in_text(value)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_money_fields(value), answer_kind="number")


def _hard_readings(rng):
    for _ in attempts(200, "a two-reading draw above salvage"):
        a = _asset(rng)
        u1 = rng.randint(5, 20) * 1000
        u2 = u1 + rng.randint(5, 25) * 1000
        u3 = u2 + rng.randint(5, 20) * 1000
        rate = a["rate"]
        price = a["price"]
        v1, v2 = price - rate * u1, price - rate * u2
        v3 = price - rate * u3
        if v3 >= a["salvage"]:
            break
    # second route: the rate recovered from the two readings, then the value from either reading.
    check(Fraction(v1 - v2, u2 - u1) == rate, "rate differs")
    check(v2 - rate * (u3 - u2) == v3, "value differs")
    wrong_rate = Fraction(v1 - v2, u2)
    cands = [
        (v2 - wrong_rate * (u3 - u2), "ucd-divides-by-total-units"),
        (v2 - rate * u3, "ucd-reading-as-price"),
        (v1 - rate * (u3 - u2), "ucd-reading-as-price"),
    ]
    dist = _money_dist(v3, cands)
    enough(dist)
    stem = (
        f"{_art(a['desc']).capitalize()} {a['desc']} is depreciated by the unit-cost method. After {_n(u1)} {a['many']} of use its value is {_in_text(v1)}, and after "
        f"{_n(u2)} {a['many']} its value is {_in_text(v2)}. What will its value be, in dollars, after {_n(u3)} {a['many']} of use?"
    )
    working = [
        f"Rate = ({_in_text(v1)} - {_in_text(v2)}) / ({_n(u2)} - {_n(u1)}) = {_in_text(rate)} per {a['one']}.",
        f"Value after {_n(u3)} = {_in_text(v2)} - {_in_text(rate)} x ({_n(u3)} - {_n(u2)}) = {_in_text(v3)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_money_fields(v3), answer_kind="number")


_EASY = (_easy_amount, _easy_value, _easy_rate)
_MEDIUM = (_medium_rate, _medium_value_after, _medium_units)
_HARD = (_hard_two_stage_value, _hard_two_stage_units, _hard_yearly, _hard_readings)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w7b_unit_cost_depreciation(rng, tier="easy"):
    """Unit-cost depreciation; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if not (pool is not None):
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w7b_unit_cost_depreciation(rng, tier="exam"):
    """Exam chain: rate per unit, value after use, units to reach a value, and the difference from a flat-rate valuation."""
    if not (tier == "exam"):
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a unit-cost depreciation chain"):
        a = _asset(rng)
        use = _use(rng, a, 0.15, 0.4, 100)
        reach = use + rng.randint(2, 6) * 1000
        if reach >= a["life"]:
            continue
        years = rng.randint(2, 5)
        r, price = a["rate"], a["price"]
        unit_value = price - r * use
        # a flat annual amount (whole dollars, multiple of 100) that depreciates MORE than the unit-cost method over `years`
        flat = (r * use // 100 // years // 100 + rng.randint(1, 4)) * 100
        flat_value = price - flat * 100 * years
        if flat_value <= a["salvage"] or flat_value >= unit_value:
            continue
        break
    else:
        raise AssertionError  # pragma: no cover
    target = price - r * reach
    diff = unit_value - flat_value
    # second routes
    check(_value_by_schedule(price, [(None, r)], reach) == target, "target differs")
    check(Fraction(price - target, r) == reach, "units differ")
    check(diff == flat * 100 * years - r * use, "difference differs")

    a_dist = _money_dist(
        r,
        [
            (Fraction(price, a["life"]), "ucd-ignores-salvage"),
            (Fraction(price + a["salvage"], a["life"]), "ucd-adds-salvage"),
            (Fraction(a["salvage"], a["life"]), "ucd-salvage-only"),
            (r * 10, "ucd-decimal-slip"),
        ],
    )
    b_dist = _money_dist(
        unit_value,
        [
            (r * use, "ucd-amount-not-value"),
            (price + r * use, "ucd-adds-to-price"),
            (price - Fraction(price, a["life"]) * use, "ucd-ignores-salvage"),
            (price - r, "ucd-one-unit-only"),
        ],
    )
    c_dist = _units_dist(
        reach,
        [
            (Fraction(target, r), "ucd-units-from-value"),
            (Fraction(price, r), "ucd-units-from-price"),
            (Fraction((price - target) * r, 100 * 100), "ucd-multiplies-for-units"),
            (reach - use, "ucd-second-stage-only"),
        ],
    )
    d_dist = _money_dist(
        diff,
        [
            (flat * 100 * (years + 1) - r * use, "ucd-years-miscounted"),
            (flat * 100 * (years - 1) - r * use, "ucd-years-miscounted"),
            (flat * 100 * years, "ucd-compares-depreciation-only"),
            (r * use, "ucd-compares-depreciation-only"),
            (unit_value, "ucd-amount-not-value"),
        ],
    )
    for d in (a_dist, b_dist, c_dist, d_dist):
        enough(d)
    stem = (
        f"{_art(a['desc']).capitalize()} {a['desc']} is bought for {_in_text(price)}. Its salvage value is expected to be {_in_text(a['salvage'])} after {_n(a['life'])} "
        f"{a['many']} of use, and it is depreciated by the unit-cost method."
    )
    return dict(
        text=stem,
        parts=[
            part(
                f"Find the depreciation per {a['one']}, in dollars.",
                "number",
                _money_fields(r),
                1,
                [f"Rate = ({_in_text(price)} - {_in_text(a['salvage'])}) / {_n(a['life'])} = {_in_text(r)}."],
                a_dist,
                sensible_range=(0, None),
            ),
            part(
                f"What is its value, in dollars, after {_n(use)} {a['many']} of use?",
                "number",
                _money_fields(unit_value),
                2,
                [f"Value = {_in_text(price)} - {_in_text(r)} x {_n(use)} = {_in_text(unit_value)}."],
                b_dist,
                follow_through=f"{price / 100} - part_a*{use}",
                sensible_range=(0, None),
            ),
            part(
                f"After how many {a['many']} of use will its value be {_in_text(target)}?",
                "number",
                num_fields(reach),
                2,
                [f"Units = ({_in_text(price)} - {_in_text(target)}) / {_in_text(r)} = {_n(reach)}."],
                c_dist,
                follow_through=f"{(price - target) / 100}/part_a",
                sensible_range=(0, None),
            ),
            part(
                f"Instead, flat-rate depreciation of {_in_text(flat * 100)} per year is used. After {years} years, by how many dollars is the value from the "
                f"unit-cost method (with {_n(use)} {a['many']} used) higher than the flat-rate value?",
                "number",
                _money_fields(diff),
                2,
                [
                    f"Flat-rate value = {_in_text(price)} - {years} x {_in_text(flat * 100)} = {_in_text(flat_value)}.",
                    f"Difference = {_in_text(unit_value)} - {_in_text(flat_value)} = {_in_text(diff)}.",
                ],
                d_dist,
                follow_through=f"part_b - {(price - flat * 100 * years) / 100}",
                sensible_range=(0, None),
            ),
        ],
    )
