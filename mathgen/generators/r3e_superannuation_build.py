"""Super contributions and balances: `r3e_superannuation` (easy, medium, hard) and the exam chain
`exd_x_r3e_superannuation`.

``superannuation-contributions`` (``financial-discrete``, answer ``number``). The employer rate, the rate of tax on
contributions, the growth rate and any cap are printed in every question.

* easy: the employer's contribution for a quarter, from an annual salary and a stated rate.
* medium: what reaches the fund once contributions tax has been taken from the employer contribution and a salary
  sacrifice together.
* hard: the balance after three years when the same net contribution is added to the end of each year (a first-order
  recurrence); the room left under a contributions cap; or the extra amount in the fund, net of the pay given up,
  from sacrificing a sum at a stated marginal rate.
* exam (``exd_x_r3e_superannuation``): the year's employer contribution, the total concessional contribution, the
  contributions tax and the closing balance after growth, each following through from the one before.

The balance is checked a second way: the recurrence is run year by year and compared with the closed form
B0 g^n + C (g^n - 1) / (g - 1).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3e_kit import PEOPLE, dec, dollars, number_part, percent, single_number

_SG = [Fraction(10), Fraction(23, 2), Fraction(12), Fraction(11)]
_TAX = Fraction(15)
_GROWTH = [4, 5, 6, 7, 8]


def _recurrence(b0, growth, contribution, years):
    """The balance after `years` years: grow, then add the contribution at the year's end."""
    g = 1 + Fraction(growth, 100)
    balance = Fraction(b0)
    for _ in range(years):
        balance = balance * g + contribution
    closed = Fraction(b0) * g**years + contribution * (g**years - 1) / (g - 1)
    check(balance == closed, "the recurrence and the closed form disagree")
    return balance


def _salary(rng):
    return 1000 * rng.randint(58, 140)


def _easy(rng):
    name, job = rng.choice(PEOPLE)
    salary = _salary(rng)
    sg = rng.choice(_SG)
    annual = Fraction(salary) * sg / 100
    quarter = annual / 4
    text = (
        f"{name} is {job} on a salary of {dollars(salary)} a year. {name}'s employer pays super contributions of "
        f"{percent(sg)}% of salary in four equal payments a year. How much does the employer pay into {name}'s "
        "super fund each quarter? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (annual, "r3e-annual-given-for-quarter"),
        (Fraction(salary) / 4, "r3e-rate-ignored"),
        (annual / 12, "r3e-wrong-number-of-payments"),
        (quarter / 10, "r3e-percent-decimal-slip"),
    ]
    working = [
        f"Annual contribution = {percent(sg)}% of {dollars(salary)} = {dollars(annual, 2)}.",
        f"{dollars(annual, 2)} / 4 = {dollars(quarter, 2)}.",
    ]
    return single_number(text, quarter, 2, working, slips)


def _medium(rng):
    name, job = rng.choice(PEOPLE)
    salary = _salary(rng)
    sg = rng.choice(_SG)
    sac = 500 * rng.randint(4, 20)
    employer = Fraction(salary) * sg / 100
    total = employer + sac
    tax = total * _TAX / 100
    added = total - tax
    text = (
        f"{name} is {job} on a salary of {dollars(salary)} a year. {name}'s employer pays super of {percent(sg)}% of "
        f"salary, and {name} also salary sacrifices {dollars(sac)} a year into super. Both amounts are taxed at "
        f"{percent(_TAX)}% when they go into the fund. How much is added to {name}'s super account in the year after "
        "that tax is taken? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (total, "r3e-contributions-tax-left-out"),
        (employer * (100 - _TAX) / 100 + sac, "r3e-tax-on-employer-only"),
        (employer + sac * (100 - _TAX) / 100, "r3e-tax-on-sacrifice-only"),
        (total * _TAX / 100, "r3e-tax-given-for-balance"),
        (employer - tax, "r3e-sacrifice-left-out"),
    ]
    working = [
        f"Employer contribution = {percent(sg)}% of {dollars(salary)} = {dollars(employer, 2)}.",
        f"Total contributions = {dollars(employer, 2)} + {dollars(sac)} = {dollars(total, 2)}.",
        f"Contributions tax = {percent(_TAX)}% of {dollars(total, 2)} = {dollars(tax, 2)}.",
        f"Added to the account = {dollars(total, 2)} - {dollars(tax, 2)} = {dollars(added, 2)}.",
    ]
    return single_number(text, added, 2, working, slips)


def _hard_balance(rng):
    name, job = rng.choice(PEOPLE)
    salary = _salary(rng)
    sg = rng.choice(_SG)
    sac = 500 * rng.randint(0, 12)
    b0 = 1000 * rng.randint(20, 90)
    growth = rng.choice(_GROWTH)
    employer = Fraction(salary) * sg / 100
    net = (employer + sac) * (100 - _TAX) / 100
    balance = _recurrence(b0, growth, net, 3)
    sac_text = f" {name} also salary sacrifices {dollars(sac)} a year." if sac else ""
    text = (
        f"{name} is {job} on a salary of {dollars(salary)} a year. {name}'s employer pays super of {percent(sg)}% of "
        f"salary.{sac_text} Contributions are taxed at {percent(_TAX)}% on entering the fund. The account holds "
        f"{dollars(b0)} now and grows by {growth}% each year, worked out on the balance at the start of the year. "
        "Each year's contributions (after tax) are added at the end of that year, after the growth. Do not round "
        "until the final answer. What is the balance after 3 years? Give your answer in dollars, to the nearest cent."
    )
    g = 1 + Fraction(growth, 100)
    gross_in = employer + sac
    slips = [
        (b0 * g**3 + net * 3, "r3e-contributions-not-grown"),
        (_recurrence(b0, growth, gross_in, 3), "r3e-contributions-tax-left-out"),
        (b0 * g**3, "r3e-contributions-left-out"),
        (b0 * (1 + Fraction(growth * 3, 100)) + net * 3, "r3e-simple-growth-used"),
        (_recurrence(b0, growth, net, 2), "r3e-wrong-number-of-years"),
    ]
    year1 = b0 * g + net
    working = [
        f"Net contribution each year = {dollars(gross_in, 2)} x {dec(Fraction(100 - _TAX, 100))} = {dollars(net, 2)}.",
        f"End of year 1: {dollars(b0)} x {dec(g)} + {dollars(net, 2)} = {dollars(year1, 2)}.",
        f"Repeating for years 2 and 3 gives {dollars(balance, 2)}.",
    ]
    return single_number(text, balance, 2, working, slips)


def _hard_cap(rng):
    name, job = rng.choice(PEOPLE)
    salary = _salary(rng)
    sg = rng.choice(_SG)
    cap = rng.choice([25000, 27500, 30000])
    employer = Fraction(salary) * sg / 100
    sac = 500 * rng.randint(2, 8)
    room = cap - employer - sac
    check(room > 0, "no room under the cap")
    text = (
        f"{name} is {job} on a salary of {dollars(salary)} a year. {name}'s employer pays super of {percent(sg)}% of "
        f"salary, and {name} salary sacrifices {dollars(sac)} a year. Employer and sacrificed contributions together "
        f"count towards a cap of {dollars(cap)} a year. How much more could {name} sacrifice this year before "
        "reaching the cap? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (cap - employer, "r3e-sacrifice-left-out-of-cap"),
        (cap - sac, "r3e-employer-left-out-of-cap"),
        (cap - employer - sac * (100 - _TAX) / 100, "r3e-cap-applied-after-tax"),
        (employer + sac - cap, "r3e-cap-subtracted-wrong-way"),
        (cap - Fraction(salary) * sg / 1000 - sac, "r3e-percent-decimal-slip"),
    ]
    working = [
        f"Employer contribution = {percent(sg)}% of {dollars(salary)} = {dollars(employer, 2)}.",
        f"Contributions so far = {dollars(employer, 2)} + {dollars(sac)} = {dollars(employer + sac, 2)}.",
        f"Room left = {dollars(cap)} - {dollars(employer + sac, 2)} = {dollars(room, 2)}.",
    ]
    return single_number(text, room, 2, working, slips)


def _hard_benefit(rng):
    name, job = rng.choice(PEOPLE)
    sac = 500 * rng.randint(4, 16)
    marginal = rng.choice([30, 32, 35, 37, 39, 45])
    given_up = Fraction(sac) * (100 - marginal) / 100
    added = Fraction(sac) * (100 - _TAX) / 100
    benefit = added - given_up
    text = (
        f"{name} is {job} and is in the {marginal}% tax bracket. {name} salary sacrifices {dollars(sac)} a year into "
        f"super. The sacrifice comes out of pay before tax, so it saves tax at {marginal}%, and the fund taxes it at "
        f"{percent(_TAX)}% when it arrives. How much more is added to {name}'s super than {name}'s take-home pay "
        "falls by? Give your answer in dollars, to the nearest cent."
    )
    slips = [
        (added, "r3e-added-to-super-given-as-benefit"),
        (given_up, "r3e-take-home-fall-given-as-benefit"),
        (Fraction(sac) * (marginal + _TAX) / 100, "r3e-rates-added"),
        (Fraction(sac) - given_up - Fraction(sac) * _TAX / 100 * 2, "r3e-tax-taken-twice"),
        (Fraction(sac) * marginal / 100, "r3e-tax-saved-given-as-benefit"),
    ]
    working = [
        f"Take-home pay falls by {dollars(sac)} less the tax saved: {dollars(sac)} x {dec(Fraction(100 - marginal, 100))} = {dollars(given_up, 2)}.",
        f"Added to super = {dollars(sac)} x {dec(Fraction(100 - _TAX, 100))} = {dollars(added, 2)}.",
        f"Difference = {dollars(added, 2)} - {dollars(given_up, 2)} = {dollars(benefit, 2)}.",
    ]
    return single_number(text, benefit, 2, working, slips)


@generator(
    "r3e_superannuation",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3e_superannuation(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_balance, _hard_balance, _hard_cap, _hard_benefit])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r3e_superannuation", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_superannuation(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    name, job = rng.choice(PEOPLE)
    salary = _salary(rng)
    sg = rng.choice(_SG)
    sac = 500 * rng.randint(2, 14)
    b0 = 1000 * rng.randint(20, 90)
    growth = rng.choice(_GROWTH)
    employer = Fraction(salary) * sg / 100
    total = employer + sac
    tax = total * _TAX / 100
    g = 1 + Fraction(growth, 100)
    closing = b0 * g + (total - tax)
    text = (
        f"{name} is {job} on a salary of {dollars(salary)} a year. {name}'s employer pays super of {percent(sg)}% of "
        f"salary, and {name} also salary sacrifices {dollars(sac)} a year into super. The fund takes "
        f"{percent(_TAX)}% contributions tax from the contributions when they arrive. {name}'s super account held "
        f"{dollars(b0)} at the start of the year and earns {growth}% for the year; the year's contributions are "
        "added at the end of the year, after the earnings."
    )
    parts = [
        number_part(
            "Find the employer's super contribution for the year, in dollars.",
            employer,
            2,
            1,
            [f"{percent(sg)}% of {dollars(salary)} = {dollars(employer, 2)}."],
            [
                (Fraction(salary) * sg / 1000, "r3e-percent-decimal-slip"),
                (Fraction(salary) / 12 * sg / 100, "r3e-monthly-for-annual"),
                (Fraction(salary) * (100 - sg) / 100, "r3e-complement-of-rate-taken"),
            ],
        ),
        number_part(
            "Find the total of the employer's contribution and the salary sacrifice, in dollars.",
            total,
            2,
            1,
            [f"{dollars(employer, 2)} + {dollars(sac)} = {dollars(total, 2)}."],
            [
                (employer - sac, "r3e-sacrifice-subtracted"),
                (employer, "r3e-sacrifice-left-out"),
                (Fraction(sac), "r3e-employer-left-out-of-total"),
            ],
            follow_through=f"part_a + {sac}",
        ),
        number_part(
            f"The fund takes {percent(_TAX)}% of that total as contributions tax. Find the tax, in dollars.",
            tax,
            2,
            1,
            [f"{percent(_TAX)}% of {dollars(total, 2)} = {dollars(tax, 2)}."],
            [
                (employer * _TAX / 100, "r3e-tax-on-employer-only"),
                (total * _TAX / 1000, "r3e-percent-decimal-slip"),
                (total * (100 - _TAX) / 100, "r3e-tax-given-for-balance"),
            ],
            follow_through="15/100*part_b",
        ),
        number_part(
            "Find the balance of the account at the end of the year, in dollars.",
            closing,
            2,
            3,
            [
                f"Earnings: {dollars(b0)} x {dec(g)} = {dollars(b0 * g, 2)}.",
                f"Contributions after tax: {dollars(total, 2)} - {dollars(tax, 2)} = {dollars(total - tax, 2)}.",
                f"Closing balance = {dollars(b0 * g, 2)} + {dollars(total - tax, 2)} = {dollars(closing, 2)}.",
            ],
            [
                ((b0 + total - tax) * g, "r3e-contributions-grown-too"),
                (b0 * g + total, "r3e-contributions-tax-left-out"),
                (b0 + total - tax, "r3e-growth-left-out"),
                (b0 * g, "r3e-contributions-left-out"),
            ],
            follow_through=f"{b0}*{g.numerator}/{g.denominator} + part_b - part_c",
        ),
    ]
    return dict(text=text, parts=parts)
