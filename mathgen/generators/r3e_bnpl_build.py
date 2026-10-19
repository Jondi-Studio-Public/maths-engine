"""Buy-now-pay-later plans: `r3e_bnpl` (easy, medium, hard) and the exam chain `exd_x_r3e_bnpl`.

``bnpl-fees-and-instalments`` (``financial-discrete``, answer ``number``). A plan splits a purchase into four equal
instalments, the first at checkout and the rest a fortnight apart. Every question states the fees: an optional
set-up fee added to the amount split, a fee for each late instalment, and a cap on the total of the late fees as a
percentage of the purchase price.

* easy: the size of each instalment when a set-up fee is split along with the price.
* medium: the total paid when some instalments are late and the late fees run into the cap.
* hard: the extra cost of the plan as a percentage of the price (to 1 decimal place); or how much more the plan costs
  than paying upfront at a stated cash discount.
* exam (``exd_x_r3e_bnpl``): the instalment, the late fees under the cap, the total paid and the extra cost as a
  percentage of the price.

Totals are rebuilt a second way, instalment by instalment, rather than from the price plus the fees.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3e_kit import PEOPLE, dollars, number_part, number_text, single_number

_ITEMS = [
    "a pair of running shoes",
    "a school laptop bag",
    "a robot vacuum cleaner",
    "a set of bike lights and a helmet",
    "a pair of noise-cancelling headphones",
    "a winter jacket",
    "a coffee machine",
    "a games console controller bundle",
]


def _plan(rng, bind_cap=None):
    """(price, set-up fee, late fee, late count, cap percent, cap amount)."""
    for _ in attempts(500, "a plan whose cap behaves as asked"):
        price = 20 * rng.randint(6, 80)
        setup = rng.choice([0, 4, 5, 6, 7, 8, 9, 10, 12])
        late_fee = rng.choice([10, 15, 20])
        late = rng.choice([1, 2, 3])
        cap_pct = rng.choice([20, 25])
        cap = Fraction(price) * cap_pct / 100
        binds = late * late_fee > cap
        if bind_cap is None or binds == bind_cap:
            return price, setup, late_fee, late, cap_pct, cap
    raise AssertionError  # pragma: no cover


def _fees(late_fee, late, cap):
    return min(Fraction(late_fee * late), cap)


def _easy(rng):
    price, setup, *_ = _plan(rng)
    setup = setup or 8
    each = Fraction(price + setup, 4)
    name, _job = rng.choice(PEOPLE)
    item = rng.choice(_ITEMS)
    text = (
        f"{name} buys {item} for {dollars(price)} using a buy-now-pay-later plan. The plan adds a set-up fee of "
        f"{dollars(setup)}, and the price and the fee are split into 4 equal instalments, the first paid at checkout. "
        "How much is each instalment? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (Fraction(price, 4), "r3e-setup-fee-left-out"),
        (Fraction(price, 4) + setup, "r3e-setup-fee-added-to-each-instalment"),
        (Fraction(price + setup, 3), "r3e-wrong-number-of-instalments"),
        (Fraction(price + setup, 2), "r3e-wrong-number-of-instalments"),
        (Fraction(price - setup, 4), "r3e-setup-fee-subtracted"),
    ]
    working = [
        f"Amount split = {dollars(price)} + {dollars(setup)} = {dollars(price + setup)}.",
        f"{dollars(price + setup)} / 4 = {dollars(each, 2)}.",
    ]
    return single_number(text, each, 2, working, slips)


def _medium(rng):
    price, setup, late_fee, late, cap_pct, cap = _plan(rng, bind_cap=True)
    fees = _fees(late_fee, late, cap)
    total = price + setup + fees
    name, _job = rng.choice(PEOPLE)
    item = rng.choice(_ITEMS)
    set_text = f" and a set-up fee of {dollars(setup)}" if setup else ""
    text = (
        f"{name} buys {item} for {dollars(price)} on a buy-now-pay-later plan with 4 instalments{set_text}. {name} "
        f"pays {late} of the instalments late. Each late instalment costs a {dollars(late_fee)} late fee, but the "
        f"late fees for one purchase cannot add up to more than {cap_pct}% of the price. What is the total {name} "
        "pays for the item, including all fees? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (price + setup + late_fee * late, "r3e-cap-ignored"),
        (price + setup + cap * late, "r3e-cap-applied-to-each-fee"),
        (price + setup, "r3e-late-fees-left-out"),
        (price + late_fee * late, "r3e-setup-fee-left-out"),
        (price + setup + fees - cap, "r3e-cap-subtracted"),
    ]
    check(
        total == 4 * Fraction(price + setup, 4) + min(sum([late_fee] * late), cap),
        "total disagrees when built instalment by instalment",
    )
    working = [
        f"Late fees without the cap = {late} x {dollars(late_fee)} = {dollars(late_fee * late)}.",
        f"The cap is {cap_pct}% of {dollars(price)} = {dollars(cap, 2)}, which is smaller, so the late fees are {dollars(fees, 2)}.",
        f"Total = {dollars(price)} + {dollars(setup)} + {dollars(fees, 2)} = {dollars(total, 2)}.",
    ]
    return single_number(text, total, 2, working, slips)


def _hard_percent(rng):
    for _ in attempts(500, "a percentage clear of a rounding tie"):
        price, setup, late_fee, late, cap_pct, cap = _plan(rng)
        setup = setup or 8
        fees = _fees(late_fee, late, cap)
        extra = Fraction(setup) + fees
        pct = extra * 100 / price
        if (pct * 10).denominator != 2:
            break
    name, _job = rng.choice(PEOPLE)
    item = rng.choice(_ITEMS)
    text = (
        f"{name} buys {item} for {dollars(price)} on a buy-now-pay-later plan with a set-up fee of {dollars(setup)}. "
        f"{name} pays {late} instalment{'s' if late != 1 else ''} late. Each late instalment costs a "
        f"{dollars(late_fee)} late fee, and the late fees for one purchase cannot add up to more than {cap_pct}% of "
        "the price. What percentage of the price do the fees add to the cost? Give your answer to 1 decimal place."
    )
    slips = [
        ((Fraction(setup) + late_fee * late) * 100 / price, "r3e-cap-ignored"),
        (fees * 100 / price, "r3e-setup-fee-left-out"),
        (extra * 100 / (price + extra), "r3e-percentage-of-total-paid"),
        (Fraction(setup) + fees, "r3e-dollars-given-as-percentage"),
        (extra / price, "r3e-ratio-not-percentage"),
    ]
    working = [
        f"Late fees = {dollars(fees, 2)} (cap {dollars(cap, 2)}).",
        f"Extra cost = {dollars(setup)} + {dollars(fees, 2)} = {dollars(extra, 2)}.",
        f"{dollars(extra, 2)} / {dollars(price)} x 100 = {number_text(pct, 1)}%.",
    ]
    return single_number(text, pct, 1, working, slips)


def _hard_vs_upfront(rng):
    price, setup, late_fee, late, cap_pct, cap = _plan(rng)
    disc = rng.choice([5, 10, 15])
    fees = _fees(late_fee, late, cap)
    bnpl = price + setup + fees
    upfront = Fraction(price) * (100 - disc) / 100
    diff = bnpl - upfront
    name, _job = rng.choice(PEOPLE)
    item = rng.choice(_ITEMS)
    set_text = f" and a set-up fee of {dollars(setup)}" if setup else ""
    text = (
        f"{name} can buy {item} for {dollars(price)} on a buy-now-pay-later plan with 4 instalments{set_text}, or pay "
        f"upfront for {disc}% off. If {name} chooses the plan and pays {late} instalment{'s' if late != 1 else ''} "
        f"late, each late instalment costs {dollars(late_fee)}, but the late fees cannot add up to more than {cap_pct}% "
        f"of the price. How much more does the plan cost {name} than paying upfront? Give your answer in dollars, to "
        "the nearest cent."
    )
    slips = [
        (bnpl - price, "r3e-discount-ignored"),
        (price + setup + late_fee * late - upfront, "r3e-cap-ignored"),
        (bnpl - Fraction(price) * disc / 100, "r3e-discount-given-as-upfront-price"),
        (upfront - bnpl, "r3e-difference-wrong-way"),
        (price + setup - upfront, "r3e-late-fees-left-out"),
    ]
    working = [
        f"Late fees = {dollars(fees, 2)}, so the plan costs {dollars(price)} + {dollars(setup)} + {dollars(fees, 2)} = {dollars(bnpl, 2)}.",
        f"Paying upfront costs {100 - disc}% of {dollars(price)} = {dollars(upfront, 2)}.",
        f"Difference = {dollars(bnpl, 2)} - {dollars(upfront, 2)} = {dollars(diff, 2)}.",
    ]
    return single_number(text, diff, 2, working, slips)


@generator(
    "r3e_bnpl",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3e_bnpl(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_percent, _hard_vs_upfront])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r3e_bnpl", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_bnpl(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a plan with a percentage clear of a rounding tie"):
        price, setup, late_fee, late, cap_pct, cap = _plan(rng, bind_cap=True)
        setup = setup or 8
        fees = _fees(late_fee, late, cap)
        total = price + setup + fees
        pct = (total - price) * 100 / price
        if (pct * 10).denominator != 2:
            break
    each = Fraction(price + setup, 4)
    check(4 * each + fees == total, "total disagrees when built from the instalments")
    name, _job = rng.choice(PEOPLE)
    item = rng.choice(_ITEMS)
    text = (
        f"{name} buys {item} for {dollars(price)} on a buy-now-pay-later plan. The plan adds a set-up fee of "
        f"{dollars(setup)}, and the price and the fee are split into 4 equal instalments. {name} pays {late} "
        f"instalment{'s' if late != 1 else ''} late. Each late instalment costs a {dollars(late_fee)} late fee, but "
        f"the late fees for one purchase cannot add up to more than {cap_pct}% of the price."
    )
    parts = [
        number_part(
            "Find the amount of each instalment, in dollars.",
            each,
            2,
            1,
            [f"({dollars(price)} + {dollars(setup)}) / 4 = {dollars(each, 2)}."],
            [
                (Fraction(price, 4), "r3e-setup-fee-left-out"),
                (Fraction(price, 4) + setup, "r3e-setup-fee-added-to-each-instalment"),
                (Fraction(price + setup, 3), "r3e-wrong-number-of-instalments"),
            ],
        ),
        number_part(
            "Find the total of the late fees charged, in dollars.",
            fees,
            2,
            2,
            [
                f"{late} x {dollars(late_fee)} = {dollars(late_fee * late)}; the cap is {cap_pct}% of {dollars(price)} = {dollars(cap, 2)}.",
                f"The late fees are the smaller of the two: {dollars(fees, 2)}.",
            ],
            [
                (Fraction(late_fee * late), "r3e-cap-ignored"),
                (cap * late, "r3e-cap-applied-to-each-fee"),
                (Fraction(late_fee), "r3e-one-late-fee-only"),
                (Fraction(price) * cap_pct / 1000, "r3e-percent-decimal-slip"),
            ],
        ),
        number_part(
            "Find the total {name} pays for the item, including all fees, in dollars.".replace("{name}", name),
            total,
            2,
            2,
            [f"4 x {dollars(each, 2)} + {dollars(fees, 2)} = {dollars(total, 2)}."],
            [
                (4 * each, "r3e-late-fees-left-out"),
                (Fraction(price) + fees, "r3e-setup-fee-left-out"),
                (4 * each + late_fee * late, "r3e-cap-ignored"),
                (4 * each - fees, "r3e-fees-subtracted"),
            ],
            follow_through="4*part_a + part_b",
        ),
        number_part(
            "By what percentage of the price did the plan add to the cost? Give your answer to 1 decimal place.",
            pct,
            1,
            2,
            [f"({dollars(total, 2)} - {dollars(price)}) / {dollars(price)} x 100 = {number_text(pct, 1)}%."],
            [
                ((total - price) * 100 / total, "r3e-percentage-of-total-paid"),
                (fees * 100 / price, "r3e-setup-fee-left-out"),
                (total - price, "r3e-dollars-given-as-percentage"),
                ((total - price) / price, "r3e-ratio-not-percentage"),
            ],
            follow_through=f"100*(part_c - {price})/{price}",
        ),
    ]
    return dict(text=text, parts=parts)
