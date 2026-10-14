"""Tiered energy and water tariffs: `r3e_utility_bill` (easy, medium, hard) and the exam chain
`exd_x_r3e_utility_bill`.

``utility-bill-tiered-tariff`` (``financial-discrete``, answer ``number``). Each question prints the tariff: a daily
supply charge, a usage rate that changes once use passes a stated amount (a piecewise-linear charge), the GST rule and
any solar feed-in rate. The charge for use is the piecewise-linear model: the first block at one rate and each
further block at the next.

* easy: the usage charge for a quarter, where the use passes the first block.
* medium: the whole bill with supply charge, tiered use and GST.
* hard: the bill after a solar feed-in credit; the use that produced a stated usage charge (the piecewise function run
  backwards); or a three-block water charge.
* exam (``exd_x_r3e_utility_bill``): supply charge, usage charge, GST, and the amount payable after the credit.

Each tiered charge is rebuilt a second way, as a rate on all use less the rate difference on the use over the threshold.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3e_kit import PEOPLE, dec, dollars, listing, number_part, single_number


def _tariff(rng):
    """(supply cents/day, block threshold kWh, rate1 c/kWh, rate2 c/kWh, feed-in c/kWh, days)."""
    return dict(
        supply=rng.choice([95, 98, 102, 105, 110, 118, 125]),
        block=rng.choice([300, 400, 500, 600]),
        r1=rng.choice([28, 30, 31, 32, 34]),
        r2=rng.choice([36, 38, 40, 42, 45]),
        feed=rng.choice([5, 6, 7, 8]),
        days=rng.choice([89, 90, 91, 92]),
    )


def _usage_charge(t, kwh):
    kwh = Fraction(kwh)
    charge = Fraction(t["r1"]) * min(kwh, t["block"]) / 100 + Fraction(t["r2"]) * max(kwh - t["block"], 0) / 100
    alt = Fraction(t["r1"]) * kwh / 100 + Fraction(t["r2"] - t["r1"]) * max(kwh - t["block"], 0) / 100
    check(charge == alt, "usage charge disagrees between the two methods")
    return charge


def _supply(t):
    return Fraction(t["supply"]) * t["days"] / 100


def _tariff_text(t, solar=False):
    rows = [
        ("Supply charge", f"{t['supply']} cents per day"),
        (f"Usage, first {t['block']} kWh in the quarter", f"{t['r1']} cents per kWh"),
        (f"Usage, over {t['block']} kWh in the quarter", f"{t['r2']} cents per kWh"),
    ]
    if solar:
        rows.append(("Solar feed-in credit", f"{t['feed']} cents per kWh exported"))
    return "The electricity tariff is: " + listing(rows)


def _easy(rng):
    t = _tariff(rng)
    kwh = t["block"] + 10 * rng.randint(8, 80)
    charge = _usage_charge(t, kwh)
    name, _job = rng.choice(PEOPLE)
    text = (
        f"{name}'s household used {kwh} kWh of electricity in a quarter. "
        f"{_tariff_text(t)} What is the usage charge for the quarter, not counting the supply charge? Give your "
        "answer in dollars, to the nearest cent."
    )
    slips = [
        (Fraction(t["r2"]) * kwh / 100, "r3e-higher-rate-on-all-use"),
        (Fraction(t["r1"]) * kwh / 100, "r3e-lower-rate-on-all-use"),
        (Fraction(t["r2"]) * t["block"] / 100 + Fraction(t["r1"]) * (kwh - t["block"]) / 100, "r3e-rates-swapped"),
        (Fraction(t["r1"]) * t["block"] / 100 + Fraction(t["r2"]) * kwh / 100, "r3e-higher-rate-on-all-use"),
        (Fraction(t["r2"]) * (kwh - t["block"]) / 100, "r3e-first-block-left-out"),
    ]
    working = [
        f"First {t['block']} kWh: {t['block']} x {dec(Fraction(t['r1'], 100))} = {dollars(Fraction(t['r1']) * t['block'] / 100, 2)}.",
        f"Remaining {kwh - t['block']} kWh: {kwh - t['block']} x {dec(Fraction(t['r2'], 100))} = {dollars(Fraction(t['r2']) * (kwh - t['block']) / 100, 2)}.",
        f"Usage charge = {dollars(charge, 2)}.",
    ]
    return single_number(text, charge, 2, working, slips)


def _medium(rng):
    t = _tariff(rng)
    kwh = t["block"] + 10 * rng.randint(8, 80)
    supply = _supply(t)
    usage = _usage_charge(t, kwh)
    gst = (supply + usage) / 10
    total = supply + usage + gst
    name, _job = rng.choice(PEOPLE)
    text = (
        f"{name}'s household used {kwh} kWh of electricity in a {t['days']}-day quarter. {_tariff_text(t)} "
        "GST of 10% is added to the sum of the supply and usage charges. What is the total bill? Give your answer in "
        "dollars, to the nearest cent."
    )
    slips = [
        (supply + usage, "r3e-gst-left-out"),
        (usage * Fraction(11, 10), "r3e-supply-charge-left-out"),
        (supply + usage + usage / 10, "r3e-gst-on-usage-only"),
        (
            supply + Fraction(t["r2"]) * kwh / 100 + (supply + Fraction(t["r2"]) * kwh / 100) / 10,
            "r3e-higher-rate-on-all-use",
        ),
        (
            Fraction(t["supply"]) * 91 / 100 + usage + (Fraction(t["supply"]) * 91 / 100 + usage) / 10,
            "r3e-quarter-taken-as-91-days",
        ),
        (supply + usage + gst + Fraction(t["supply"]) / 100, "r3e-extra-day-of-supply"),
    ]
    working = [
        f"Supply = {t['days']} x {dec(Fraction(t['supply'], 100))} = {dollars(supply, 2)}.",
        f"Usage = {dollars(usage, 2)}.",
        f"GST = 10% of {dollars(supply + usage, 2)} = {dollars(gst, 2)}.",
        f"Total = {dollars(total, 2)}.",
    ]
    return single_number(text, total, 2, working, slips)


def _hard_solar(rng):
    t = _tariff(rng)
    kwh = t["block"] + 10 * rng.randint(4, 60)
    export = 10 * rng.randint(30, 120)
    supply = _supply(t)
    usage = _usage_charge(t, kwh)
    gst = (supply + usage) / 10
    credit = Fraction(t["feed"]) * export / 100
    bill = supply + usage + gst - credit
    name, _job = rng.choice(PEOPLE)
    text = (
        f"{name}'s household used {kwh} kWh of electricity from the grid in a {t['days']}-day quarter and exported "
        f"{export} kWh from its solar panels. {_tariff_text(t, solar=True)} GST of 10% is added to the sum of the supply "
        "and usage charges, and then the feed-in credit is taken off the bill. The credit has no GST. What is the "
        "amount payable? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (supply + usage + gst + credit, "r3e-credit-added"),
        (supply + usage + gst, "r3e-credit-left-out"),
        ((supply + usage - credit) * Fraction(11, 10), "r3e-credit-taken-off-before-gst"),
        (supply + usage - credit, "r3e-gst-left-out"),
        (supply + usage + gst - Fraction(t["feed"]) * kwh / 100, "r3e-credit-on-use-not-export"),
    ]
    working = [
        f"Supply = {dollars(supply, 2)}; usage = {dollars(usage, 2)}; GST = {dollars(gst, 2)}.",
        f"Feed-in credit = {export} x {dec(Fraction(t['feed'], 100))} = {dollars(credit, 2)}.",
        f"Amount payable = {dollars(supply + usage + gst, 2)} - {dollars(credit, 2)} = {dollars(bill, 2)}.",
    ]
    return single_number(text, bill, 2, working, slips)


def _hard_reverse(rng):
    t = _tariff(rng)
    kwh = t["block"] + 10 * rng.randint(8, 90)
    usage = _usage_charge(t, kwh)
    name, _job = rng.choice(PEOPLE)
    text = f"{name}'s electricity bill shows a usage charge of {dollars(usage, 2)} for the quarter. {_tariff_text(t)} How many kWh did the household use? Give your answer as a whole number."
    first = Fraction(t["r1"]) * t["block"] / 100
    slips = [
        (usage * 100 / t["r1"], "r3e-lower-rate-on-all-use"),
        (usage * 100 / t["r2"], "r3e-higher-rate-on-all-use"),
        ((usage - first) * 100 / t["r1"] + t["block"], "r3e-wrong-rate-for-excess"),
        ((usage - first) * 100 / t["r2"], "r3e-first-block-left-out"),
    ]
    working = [
        f"The first {t['block']} kWh cost {dollars(first, 2)}, so the usage is past the first block.",
        f"The rest of the charge, {dollars(usage - first, 2)}, is at {t['r2']} cents per kWh: {dollars(usage - first, 2)} / {dec(Fraction(t['r2'], 100))} = {kwh - t['block']} kWh.",
        f"Total use = {t['block']} + {kwh - t['block']} = {kwh} kWh.",
    ]
    return single_number(text, kwh, 0, working, slips)


def _hard_water(rng):
    b1 = rng.choice([100, 120, 150])
    b2 = b1 + rng.choice([100, 150, 200])
    r1, r2, r3 = rng.choice([(2, 3, 4), (2, 3, 5), (3, 4, 6), (2, 4, 6)])
    fixed = rng.choice([55, 60, 66, 72])
    kl = b2 + 10 * rng.randint(2, 30)
    usage = r1 * Fraction(b1) + r2 * Fraction(b2 - b1) + r3 * Fraction(kl - b2)
    # r/kL in dollars with an extra check from the "rate on everything, plus extras" form
    check(usage == r1 * kl + (r2 - r1) * (kl - b1) + (r3 - r2) * (kl - b2), "water charge disagrees")
    bill = usage + fixed
    name, _job = rng.choice(PEOPLE)
    rows = [
        (f"First {b1} kL", f"${r1} per kL"),
        (f"Next {b2 - b1} kL (up to {b2} kL)", f"${r2} per kL"),
        (f"Over {b2} kL", f"${r3} per kL"),
    ]
    text = (
        f"{name}'s household used {kl} kL of water in a year. There is also a fixed "
        f"service charge of {dollars(fixed)} for the year. The water charges are: {listing(rows)} "
        "What is the total water bill for the year? Give your answer in dollars."
    )
    slips = [
        (r3 * kl + fixed, "r3e-highest-rate-on-all-use"),
        (r1 * kl + fixed, "r3e-lowest-rate-on-all-use"),
        (usage, "r3e-service-charge-left-out"),
        (r1 * b1 + r2 * (kl - b1) + fixed, "r3e-middle-rate-on-all-excess"),
        (r1 * b1 + r2 * (b2 - b1) + r3 * (kl - b1) + fixed, "r3e-top-rate-on-middle-block"),
    ]
    working = [
        f"First block: {b1} x {r1} = {r1 * b1}. Middle block: {b2 - b1} x {r2} = {r2 * (b2 - b1)}. Top block: {kl - b2} x {r3} = {r3 * (kl - b2)}.",
        f"Usage = {dollars(usage)}; with the service charge, the bill is {dollars(bill)}.",
    ]
    return single_number(text, bill, 0, working, slips)


@generator(
    "r3e_utility_bill",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3e_utility_bill(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_solar, _hard_reverse, _hard_water, _hard_solar])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r3e_utility_bill", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_utility_bill(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    t = _tariff(rng)
    kwh = t["block"] + 10 * rng.randint(8, 70)
    export = 10 * rng.randint(30, 120)
    supply = _supply(t)
    usage = _usage_charge(t, kwh)
    gst = (supply + usage) / 10
    credit = Fraction(t["feed"]) * export / 100
    bill = supply + usage + gst - credit
    name, _job = rng.choice(PEOPLE)
    text = (
        f"{name}'s household has solar panels. In a {t['days']}-day quarter it used {kwh} kWh of electricity from "
        f"the grid and exported {export} kWh. {_tariff_text(t, solar=True)} GST of 10% is added to the sum of the "
        "supply and usage charges. The feed-in credit is then taken off the bill, and it has no GST."
    )
    parts = [
        number_part(
            "Find the supply charge for the quarter, in dollars.",
            supply,
            2,
            1,
            [f"{t['days']} x {dec(Fraction(t['supply'], 100))} = {dollars(supply, 2)}."],
            [
                (Fraction(t["supply"]) * 91 / 100, "r3e-quarter-taken-as-91-days"),
                (Fraction(t["supply"]) * t["days"], "r3e-cents-given-as-dollars"),
                (Fraction(t["supply"]) / 100, "r3e-one-day-only"),
            ],
        ),
        number_part(
            "Find the usage charge for the quarter, in dollars.",
            usage,
            2,
            3,
            [
                f"First {t['block']} kWh: {dollars(Fraction(t['r1']) * t['block'] / 100, 2)}.",
                f"Remaining {kwh - t['block']} kWh: {dollars(Fraction(t['r2']) * (kwh - t['block']) / 100, 2)}.",
            ],
            [
                (Fraction(t["r2"]) * kwh / 100, "r3e-higher-rate-on-all-use"),
                (Fraction(t["r1"]) * kwh / 100, "r3e-lower-rate-on-all-use"),
                (
                    Fraction(t["r2"]) * t["block"] / 100 + Fraction(t["r1"]) * (kwh - t["block"]) / 100,
                    "r3e-rates-swapped",
                ),
                (Fraction(t["r2"]) * (kwh - t["block"]) / 100, "r3e-first-block-left-out"),
            ],
        ),
        number_part(
            "Find the GST on the supply and usage charges, in dollars.",
            gst,
            2,
            1,
            [f"10% of ({dollars(supply, 2)} + {dollars(usage, 2)}) = {dollars(gst, 2)}."],
            [
                (usage / 10, "r3e-gst-on-usage-only"),
                ((supply + usage) / 11, "r3e-gst-treated-as-inclusive"),
                ((supply + usage) / 100, "r3e-percent-decimal-slip"),
            ],
            follow_through="(part_a + part_b)/10",
        ),
        number_part(
            "Find the amount payable after the feed-in credit, in dollars.",
            bill,
            2,
            2,
            [
                f"Credit = {export} x {dec(Fraction(t['feed'], 100))} = {dollars(credit, 2)}.",
                f"Payable = {dollars(supply + usage + gst, 2)} - {dollars(credit, 2)} = {dollars(bill, 2)}.",
            ],
            [
                (supply + usage + gst + credit, "r3e-credit-added"),
                (supply + usage + gst, "r3e-credit-left-out"),
                (supply + usage + gst - Fraction(t["feed"]) * kwh / 100, "r3e-credit-on-use-not-export"),
                (supply + usage - credit, "r3e-gst-left-out"),
            ],
            follow_through=f"part_a + part_b + part_c - {credit.numerator}/{credit.denominator}",
        ),
    ]
    return dict(text=text, parts=parts)
