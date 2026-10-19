"""Exam tier for "investigate the effect of the interest rate and repayment amount on the time to repay a loan".

The practice tiers solve one linear equation for the term of a flat-interest loan. Here the learner works one
loan through three connected steps, as a financial-maths exam would set it:

(a) the number of months of repayments, found by balancing the repayments against the principal plus the flat
    interest (an unknown term appears on both sides);
(b) the total interest paid over that term, which uses the term from (a);
(c) the saving in interest if a larger monthly repayment is made instead, which uses (b).

Built backwards from whole numbers. The monthly flat interest is I = P * r / 1200 dollars (an integer, because the
principal is a multiple of 1200). A term of n months at repayment M clears the loan when n * M = P + n * I, that is
n = P / (M - I); so a divisor n of P gives the repayment M = I + P / n exactly. The key of every part is then an
integer.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_x_r3g_kit import clashes, number_part

_PURPOSES = ("car", "caravan", "boat", "kitchen renovation", "solar panel", "home extension")
_NAMES = ("Priya", "Liam", "Mei", "Tom", "Aisha", "Callum", "Sophie", "Dean")
_TERMS = (18, 20, 24, 30, 36, 40, 48, 50, 60, 72, 75, 80, 84, 90, 96)
_STEMS = (
    "{name} borrows ${P} for a {purpose}. The loan charges flat interest of {r}% per annum on the original amount "
    "for the whole term. {name} makes repayments of ${M1} at the end of each month until the loan and all of the "
    "interest have been repaid.",
    "A {purpose} loan of ${P} carries flat interest at {r}% per annum, calculated on the amount borrowed for the "
    "entire term. {name} repays ${M1} each month, and keeps paying until the principal and all of the interest "
    "charged are cleared.",
    "To pay for a {purpose}, {name} takes a ${P} loan at a flat {r}% per annum (interest is always worked out on "
    "${P}, whatever has been repaid). The monthly repayment is ${M1}, and it continues until nothing is owing.",
)


def _draw(rng: random.Random):
    P = 1200 * rng.randint(10, 50)
    r = rng.choice((4, 5, 6, 7, 8, 9, 10))
    I = P // 1200 * r  # monthly flat interest, in whole dollars
    divisors = [n for n in _TERMS if P % n == 0]
    if len(divisors) < 2:
        return None
    n1, n2 = sorted(rng.sample(divisors, 2), reverse=True)  # n1 months at M1, fewer months at the larger M2
    if n2 > 0.85 * n1:
        return None
    return P, r, I, n1, n2, I + P // n1, I + P // n2


@generator("exd_x_r3g_loan_term", level=11, presets={"exam": dict(tier="exam")})
def exd_x_r3g_loan_term(rng: random.Random, tier: str = "exam") -> dict:
    """A flat-interest loan: term in months, total interest, then the interest saved by a larger repayment."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a loan whose terms are whole months"):
        drawn = _draw(rng)
        if drawn is None:
            continue
        P, r, I, n1, n2, M1, M2 = drawn
        interest1, interest2 = I * n1, I * n2
        saving = interest1 - interest2
        name, purpose = rng.choice(_NAMES), rng.choice(_PURPOSES)
        stem = rng.choice(_STEMS).format(name=name, purpose=purpose, P=P, r=r, M1=M1)
        prompt_c = f"{name} instead repays ${M2} each month. How many dollars less interest is paid than in part (b)?"
        if clashes((n1, interest1, saving), stem, prompt_c):
            continue
        break

    # independent check of the construction: the loan is cleared exactly in n months at each repayment
    for n, M in ((n1, M1), (n2, M2)):
        check(
            Fraction(n * M) == Fraction(P) * (1 + Fraction(r, 100) * Fraction(n, 12)),
            "repayments do not clear the loan",
        )

    a_prompt = "For how many months must repayments be made?"
    b_prompt = "Find the total interest paid over the whole term, in dollars."
    parts = [
        number_part(
            a_prompt,
            n1,
            0,
            3,
            [
                f"Interest is {r}% of ${P} per year, so ${I} each month.",
                f"After n months: {M1}n = {P} + {I}n, so n = {P} / ({M1} - {I}).",
                f"n = {n1} months.",
            ],
            [
                (Fraction(P, M1), "loan-term-ignores-interest"),
                (Fraction(P, M1 + I), "added-instead-of-subtracted"),
                (Fraction(P, M1 - 12 * I) if M1 > 12 * I else None, "loan-term-annual-rate-as-monthly"),
                (round(Fraction(n1, 12)), "months-and-years-confused"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            b_prompt,
            interest1,
            0,
            1,
            [f"Interest = ${I} x {n1} months = ${interest1}."],
            [
                (M1 * n1, "loan-total-repaid-for-interest"),
                (12 * I, "loan-interest-for-one-year-only"),
                (Fraction(P * r, 100) * n1, "months-and-years-confused"),
            ],
            follow_through=f"{I}*part_a",
            sensible_range=(0, None),
        ),
        number_part(
            prompt_c,
            saving,
            0,
            2,
            [
                f"Term at ${M2} a month: n = {P} / ({M2} - {I}) = {n2} months.",
                f"Interest = ${I} x {n2} = ${interest2}.",
                f"Saving = ${interest1} - ${interest2} = ${saving}.",
            ],
            [
                (n1 - n2, "loan-months-saved-for-dollars-saved"),
                (interest2, "loan-interest-of-one-plan-only"),
                (I * (n1 + n2), "added-instead-of-subtracted"),
            ],
            follow_through=f"part_b - {interest2}",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
