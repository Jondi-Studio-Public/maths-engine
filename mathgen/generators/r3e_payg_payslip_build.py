"""Reading a payslip: `r3e_payg_payslip` (easy, medium, hard) and its exam chain `exd_x_r3e_payg_payslip`.

``payg-payslip-reading`` (``financial-discrete``, answer ``number``). A payslip is printed as a table of this-pay
and year-to-date figures. The employer's super is shown on its own line and is paid on top of the wages: it is not
deducted from the pay. The super rate, the PAYG percentage and every fee are printed in the question.

* easy: the net pay deposited, from a payslip that also carries an employer-super line (the line is not a deduction).
* medium: the employer super for the period when it is paid on ordinary-time earnings only (the overtime is not
  counted), or the pay received in earlier periods from a year-to-date column.
* hard: gross pay recovered from the net pay, a PAYG percentage and a fixed fee; or net pay where salary sacrifice
  comes off before the PAYG percentage is applied.
* exam (``exd_x_r3e_payg_payslip``): gross pay, PAYG withheld, net pay and employer super for a fortnight, each
  following through from the one before.

The net pay and the super are each rebuilt a second way: net as gross less the sum of the deductions, super from the
ordinary-time earnings built up line by line.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3e_kit import PEOPLE, dollars, listing, number_part, percent, single_number

_SUPER = [Fraction(23, 2), Fraction(12), Fraction(11)]
_PAYG = [14, 15, 16, 17, 18, 19, 20, 22]


def _slip(rows):
    return listing(rows)


def _pay_lines(rng):
    """Ordinary pay, overtime, allowance, and the lines that describe them."""
    hours = rng.choice([72, 76, 76, 80])
    rate = Fraction(rng.randint(30, 52)) + rng.choice([Fraction(0), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)])
    ordinary = hours * rate
    overtime = Fraction(rng.randint(3, 14) * 10 + rng.choice([0, 5]))
    allowance = Fraction(rng.choice([0, 40, 55, 65, 80, 95]))
    return hours, rate, ordinary, overtime, allowance


def _lines(hours, rate, ordinary, overtime, allowance):
    rows = [(f"Ordinary hours ({hours} h at {dollars(rate, 2)} per hour)", dollars(ordinary, 2))]
    if overtime:
        rows.append(("Overtime", dollars(overtime, 2)))
    if allowance:
        rows.append(("Tool allowance", dollars(allowance, 2)))
    return rows


def _easy(rng):
    name, job = rng.choice(PEOPLE)
    hours, rate, ordinary, overtime, allowance = _pay_lines(rng)
    gross = ordinary + overtime + allowance
    payg_pct = rng.choice(_PAYG)
    payg = gross * payg_pct / 100
    payg = payg.__floor__() if payg.denominator != 1 else payg
    union = Fraction(rng.choice([12, 15, 18, 22, 25]))
    sup_rate = rng.choice(_SUPER)
    sup = (ordinary + allowance) * sup_rate / 100
    net = gross - payg - union
    check(net == gross - (payg + union), "net disagrees")
    rows = _lines(hours, rate, ordinary, overtime, allowance) + [
        ("Gross pay", dollars(gross, 2)),
        ("PAYG tax withheld", dollars(payg, 2)),
        ("Union fees", dollars(union, 2)),
        (f"Employer super ({percent(sup_rate)}%, paid by the employer)", dollars(sup, 2)),
    ]
    text = (
        f"{name} is {job}. Here is part of {name}'s payslip for one fortnight: {_slip(rows)} "
        f"How much pay is deposited into {name}'s bank account? Give your answer in dollars."
    )
    slips = [
        (gross - payg - union - sup, "r3e-employer-super-deducted"),
        (gross - payg, "r3e-fee-left-out"),
        (gross - union, "r3e-payg-left-out"),
        (gross - payg - union + sup, "r3e-employer-super-added-to-pay"),
        (gross, "r3e-gross-given-as-net"),
    ]
    working = [
        f"Net pay = gross pay - PAYG - union fees = {dollars(gross, 2)} - {dollars(payg, 2)} - {dollars(union, 2)} = {dollars(net, 2)}.",
        "The employer super is paid on top of the wages, so it is not taken off.",
    ]
    return single_number(text, net, 2, working, slips)


def _medium_super(rng):
    name, job = rng.choice(PEOPLE)
    hours, rate, ordinary, overtime, allowance = _pay_lines(rng)
    overtime = overtime or Fraction(120)
    allowance = allowance or Fraction(65)
    gross = ordinary + overtime + allowance
    sup_rate = rng.choice(_SUPER)
    ote = ordinary + allowance
    sup = ote * sup_rate / 100
    check(sup == (gross - overtime) * sup_rate / 100, "super disagrees")
    rows = _lines(hours, rate, ordinary, overtime, allowance) + [("Gross pay", dollars(gross, 2))]
    text = (
        f"{name} is {job}. This is {name}'s pay summary for the period: {_slip(rows)} "
        f"The employer pays super of {percent(sup_rate)}% of ordinary-time earnings. Ordinary-time earnings include "
        f"the ordinary hours and the allowance but not the overtime. How much super does the employer pay for this "
        "period? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (gross * sup_rate / 100, "r3e-super-on-overtime-too"),
        (ordinary * sup_rate / 100, "r3e-allowance-left-out-of-super-base"),
        (overtime * sup_rate / 100, "r3e-super-on-overtime-only"),
        (ote * sup_rate / 1000, "r3e-percent-decimal-slip"),
    ]
    working = [
        f"Ordinary-time earnings = {dollars(ordinary, 2)} + {dollars(allowance, 2)} = {dollars(ote, 2)}.",
        f"{percent(sup_rate)}% of {dollars(ote, 2)} = {dollars(sup, 2)}.",
    ]
    return single_number(text, sup, 2, working, slips)


def _medium_ytd(rng):
    name, job = rng.choice(PEOPLE)
    n = rng.randint(6, 20)
    gross = Fraction(rng.randint(180, 260) * 10)
    average = Fraction(rng.randint(1900, 2300))
    prev = average * (n - 1)
    ytd = prev + gross
    payg = (gross * rng.choice(_PAYG) / 100).__floor__()
    rows = [
        ("Pay number", str(n)),
        ("Gross pay", f"{dollars(gross, 2)} this pay and {dollars(ytd, 2)} year to date"),
        ("PAYG tax withheld", f"{dollars(payg, 2)} this pay and {dollars(payg * n, 2)} year to date"),
    ]
    text = (
        f"{name} is {job}. Here is part of {name}'s fortnightly payslip, where year to date includes this pay: "
        f"{listing(rows)} "
        f"What was {name}'s mean gross pay in the pay periods before this one? Give your answer in dollars, to the "
        "nearest cent."
    )
    slips = [
        (ytd / n, "r3e-ytd-mean-over-all-periods"),
        (ytd / (n - 1), "r3e-ytd-includes-this-pay"),
        (prev / n, "r3e-mean-over-wrong-count"),
        (gross, "r3e-current-given-as-previous"),
    ]
    working = [
        f"Gross pay before this pay = {dollars(ytd, 2)} - {dollars(gross, 2)} = {dollars(prev, 2)}.",
        f"There were {n - 1} earlier pays, so the mean is {dollars(prev, 2)} / {n - 1} = {dollars(average, 2)}.",
    ]
    return single_number(text, average, 2, working, slips)


def _hard_gross(rng):
    name, job = rng.choice(PEOPLE)
    p = rng.choice(_PAYG)
    gross = Fraction(rng.randint(150, 320) * 10)
    payg = gross * p / 100
    fee = Fraction(rng.choice([12, 18, 24, 30]))
    net = gross - payg - fee
    text = (
        f"{name} is {job}. A payslip shows that PAYG tax of {p}% of gross pay was withheld and that a union fee of "
        f"{dollars(fee, 2)} was deducted. {name}'s net pay was {dollars(net, 2)}. What was the gross pay? Give your "
        "answer in dollars."
    )
    slips = [
        (net + fee + net * p / 100, "r3e-percentage-taken-of-net"),
        (net / (1 - Fraction(p, 100)), "r3e-fee-left-out"),
        ((net + fee) * (1 + Fraction(p, 100)), "r3e-percentage-added-to-net"),
        (net + fee + payg - fee, "r3e-fee-left-out"),
        (net + payg, "r3e-fee-left-out"),
    ]
    working = [
        f"Before the fee, the pay after PAYG was {dollars(net, 2)} + {dollars(fee, 2)} = {dollars(net + fee, 2)}.",
        f"That is {100 - p}% of the gross pay, so gross pay = {dollars(net + fee, 2)} / {Fraction(100 - p, 100)} = {dollars(gross, 2)}.",
    ]
    return single_number(text, gross, 2, working, slips)


def _hard_sacrifice(rng):
    name, job = rng.choice(PEOPLE)
    gross = Fraction(rng.randint(180, 330) * 10)
    sac = Fraction(rng.randint(8, 30) * 10)
    p = rng.choice([15, 16, 18, 20, 22])
    taxed = gross - sac
    payg = taxed * p / 100
    check(payg == (gross - sac) * p / 100, "payg disagrees")
    net = gross - sac - payg
    text = (
        f"{name} is {job} on a gross pay of {dollars(gross, 2)} a fortnight and sacrifices {dollars(sac, 2)} of it "
        f"into super before tax. PAYG tax of {p}% is withheld from the pay that is left after the sacrifice. How much "
        f"is deposited into {name}'s bank account each fortnight? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (gross - gross * p / 100, "r3e-sacrifice-ignored"),
        (gross - sac - gross * p / 100, "r3e-payg-on-gross-after-sacrifice"),
        (gross - payg, "r3e-sacrifice-not-deducted-from-pay"),
        (taxed * (1 - Fraction(p, 100)) - sac, "r3e-sacrifice-deducted-twice"),
    ]
    working = [
        f"Pay after the sacrifice = {dollars(gross, 2)} - {dollars(sac, 2)} = {dollars(taxed, 2)}.",
        f"PAYG = {p}% of {dollars(taxed, 2)} = {dollars(payg, 2)}.",
        f"Net pay = {dollars(taxed, 2)} - {dollars(payg, 2)} = {dollars(net, 2)}.",
    ]
    return single_number(text, net, 2, working, slips)


@generator(
    "r3e_payg_payslip",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3e_payg_payslip(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return rng.choice([_medium_super, _medium_ytd, _medium_super])(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_gross, _hard_sacrifice])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r3e_payg_payslip", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_payg_payslip(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    name, job = rng.choice(PEOPLE)
    hours, rate, ordinary, overtime, allowance = _pay_lines(rng)
    overtime = overtime or Fraction(150)
    allowance = allowance or Fraction(55)
    gross = ordinary + overtime + allowance
    p = rng.choice(_PAYG)
    payg = gross * p / 100
    union = Fraction(rng.choice([15, 18, 22, 25]))
    net = gross - payg - union
    sup_rate = rng.choice(_SUPER)
    sup = (gross - overtime) * sup_rate / 100
    check(sup == (ordinary + allowance) * sup_rate / 100, "super disagrees")
    rows = _lines(hours, rate, ordinary, overtime, allowance)
    text = (
        f"{name} is {job}. For one fortnight, {name}'s pay is made up as follows: {_slip(rows)} "
        f"PAYG tax of {p}% of the gross pay is withheld, and a union fee of {dollars(union, 2)} is deducted. The "
        f"employer also pays super of {percent(sup_rate)}% of ordinary-time earnings (the gross pay without the "
        "overtime), on top of the wages."
    )
    parts = [
        number_part(
            "Find the gross pay for the fortnight, in dollars.",
            gross,
            2,
            1,
            [f"{dollars(ordinary, 2)} + {dollars(overtime, 2)} + {dollars(allowance, 2)} = {dollars(gross, 2)}."],
            [
                (ordinary + overtime, "r3e-allowance-left-out-of-gross"),
                (ordinary + allowance, "r3e-overtime-left-out-of-gross"),
                (ordinary + overtime + allowance + hours, "r3e-hours-added-to-gross"),
                (hours + rate + overtime + allowance, "r3e-hours-added-to-gross"),
            ],
        ),
        number_part(
            "Find the PAYG tax withheld, in dollars.",
            payg,
            2,
            1,
            [f"{p}% of {dollars(gross, 2)} = {dollars(payg, 2)}."],
            [
                (ordinary * p / 100, "r3e-payg-on-ordinary-pay-only"),
                ((gross - overtime) * p / 100, "r3e-payg-on-ordinary-pay-only"),
                (gross * p / 1000, "r3e-percent-decimal-slip"),
                (gross - gross * p / 100, "r3e-payg-taken-as-remainder"),
            ],
            follow_through=f"{p}/100*part_a",
        ),
        number_part(
            "Find the net pay deposited into the bank account, in dollars.",
            net,
            2,
            2,
            [f"{dollars(gross, 2)} - {dollars(payg, 2)} - {dollars(union, 2)} = {dollars(net, 2)}."],
            [
                (gross - payg, "r3e-fee-left-out"),
                (gross - union, "r3e-payg-left-out"),
                (gross - payg - union - sup, "r3e-employer-super-deducted"),
                (gross - payg + union, "r3e-fee-added-to-pay"),
            ],
            follow_through=f"part_a - part_b - {int(union)}",
        ),
        number_part(
            "Find the employer's super contribution for the fortnight, in dollars.",
            sup,
            2,
            2,
            [
                f"Ordinary-time earnings = {dollars(gross, 2)} - {dollars(overtime, 2)} = {dollars(gross - overtime, 2)}.",
                f"{percent(sup_rate)}% of {dollars(gross - overtime, 2)} = {dollars(sup, 2)}.",
            ],
            [
                (gross * sup_rate / 100, "r3e-super-on-overtime-too"),
                (net * sup_rate / 100, "r3e-super-on-net-pay"),
                (overtime * sup_rate / 100, "r3e-super-on-overtime-only"),
                ((gross - overtime) * sup_rate / 1000, "r3e-percent-decimal-slip"),
            ],
            follow_through=f"{sup_rate.numerator}/{sup_rate.denominator * 100}*(part_a - {int(overtime)})",
        ),
    ]
    return dict(text=text, parts=parts)
