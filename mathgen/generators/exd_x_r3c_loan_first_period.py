"""Exam source for `senior-with-the-aid-of-a-financial-calculator-or`: the first two rows of a loan schedule.

The stem gives a loan amount, an annual interest rate charged monthly on the balance owing, and a term in years.

(a) the monthly repayment, to the nearest cent;
(b) the interest charged in the first month (the amount times the monthly rate), to the nearest cent;
(c) the balance owing after the first repayment: amount + interest - repayment, from (a) and (b);
(d) the interest charged in the second month: that balance times the monthly rate, from (c).

Parts (c) and (d) carry `follow_through`. Each later part is worked out from the earlier parts as STATED (to the
cent), as a bank statement does, so a learner who carries the displayed cents forward gets the same cents.

The repayment formula is checked by running the loan month by month (Decimal arithmetic) and requiring that
the balance after the last repayment is within the rounding slack of zero (the repayment is rounded to the cent).
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction

from mathgen.core import attempts, check, currency_from_cents, generator, round_half_up

from .exd_x_r3c_parts import number_part
from ._exd_distinct_parts import distinct_part_answers

_CENT = Fraction(1, 100)
_LOANS = (
    # (kind, amounts, terms in years, intro)
    ("home loan", range(250_000, 700_001, 10_000), (15, 20, 25, 30), "A home loan of ${p:,}"),
    ("car loan", range(15_000, 60_001, 500), (3, 4, 5, 7), "A car loan of ${p:,}"),
    ("personal loan", range(8_000, 40_001, 500), (3, 4, 5), "A personal loan of ${p:,}"),
)


def _money(amount) -> str:
    """A dollar amount that is already a whole number of cents, written "$274.51"."""
    return currency_from_cents(int(Fraction(amount) * 100))


def _far_from_half_cent(value):
    """True when `value` (dollars) is at least 0.02 of a cent from the point where rounding to the cent flips."""
    return abs((value * 100) % 1 - Fraction(1, 2)) > Fraction(2, 100)


def _run_loan(principal, i, repayment, months):
    """Balance after `months` repayments, by running the schedule in Decimal arithmetic."""
    getcontext().prec = 50
    balance, rate, pay = (
        Decimal(principal),
        Decimal(i.numerator) / Decimal(i.denominator),
        Decimal(repayment.numerator) / Decimal(repayment.denominator),
    )
    for _ in range(months):
        balance = balance * (1 + rate) - pay
    return balance


@generator("exd_x_r3c_loan_first_period", level=15, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_loan_first_period(rng, tier="exam"):
    """A reducing-balance loan: the repayment, then the first two months' interest and the balance between."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a loan whose cents are clear of a rounding boundary"):
        _kind, amounts, terms, intro = rng.choice(_LOANS)
        principal = rng.choice(amounts)
        years = rng.choice(terms)
        tenths = rng.randint(40, 95)  # annual rate in tenths of a percent: 4.0% to 9.5%
        rate = Fraction(tenths, 1000)  # as a fraction
        i = rate / 12
        n = 12 * years
        exact_repayment = principal * i / (1 - (1 + i) ** -n)
        interest1_exact = principal * i
        if not (_far_from_half_cent(exact_repayment) and _far_from_half_cent(interest1_exact)):
            continue
        repayment = round_half_up(exact_repayment, 2)
        interest1 = round_half_up(interest1_exact, 2)
        balance1 = principal + interest1 - repayment
        interest2_exact = balance1 * i
        if _far_from_half_cent(interest2_exact):
            break
    interest2 = round_half_up(interest2_exact, 2)
    # Independent route: run the schedule month by month; the last balance must be tiny.
    # Rounding the repayment to the cent leaves at most half a cent per month, grown by the schedule's own interest.
    slack = float(Fraction(51, 10000) * ((1 + i) ** n - 1) / i) + 0.01
    check(abs(float(_run_loan(principal, i, repayment, n))) < slack, "the repayment does not clear the loan")
    check(
        abs(float(_run_loan(principal, i, repayment, 1)) - float(balance1)) < 0.02, "balance after one month disagrees"
    )

    pct = f"{tenths / 10:.1f}"
    stem = (
        f"{intro.format(p=principal)} has an interest rate of {pct}% per annum, charged monthly on the balance owing. "
        f"It is repaid in equal monthly repayments over {years} years."
    )
    flat = (principal + principal * rate * years) / n
    part_a = number_part(
        "Find the monthly repayment, to the nearest cent.",
        repayment,
        dp=2,
        marks=2,
        sensible_range=(0, None),
        working=[
            f"Monthly rate = {pct}% ÷ 12; number of repayments = {n}.",
            f"Repayment = P × i ÷ (1 - (1 + i)^(-{n})) = {_money(repayment)}.",
        ],
        wrongs=[
            (flat, "flat-not-reducing-interest"),
            (principal * rate / (1 - (1 + rate) ** -n), "annual-rate-as-periodic"),
            (principal * i / (1 - (1 + i) ** -years), "years-not-months"),
        ],
    )
    part_b = number_part(
        "Find the interest charged in the first month, to the nearest cent.",
        interest1,
        dp=2,
        sensible_range=(0, None),
        working=[f"Interest = {_money(principal)} × {pct}% ÷ 12 = {_money(interest1)}."],
        wrongs=[
            (principal * rate, "annual-rate-as-periodic"),
            (repayment * i, "interest-on-repayment"),
        ],
    )
    part_c = number_part(
        "Using your answers to (a) and (b), find the balance owing immediately after the first repayment, to the nearest cent.",
        balance1,
        dp=2,
        sensible_range=(0, None),
        working=[f"{_money(principal)} + {_money(interest1)} - {_money(repayment)} = {_money(balance1)}."],
        wrongs=[
            (principal - repayment, "interest-not-added-to-balance"),
            (principal + interest1 + repayment, "added-repayment-to-balance"),
            (principal - interest1, "interest-not-added-to-balance"),
        ],
        follow_through=f"{principal} + part_b - part_a",
    )
    part_d = number_part(
        "Using your answer to (c), find the interest charged in the second month, to the nearest cent.",
        interest2,
        dp=2,
        sensible_range=(0, None),
        working=[f"Interest = {_money(balance1)} × {pct}% ÷ 12 = {_money(interest2)}."],
        wrongs=[
            (interest1, "interest-not-recalculated"),
            ((principal - repayment) * i, "interest-on-balance-before-interest"),
            (2 * interest1, "interest-not-recalculated"),
        ],
        follow_through=f"part_c*{i.numerator}/{i.denominator}",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
