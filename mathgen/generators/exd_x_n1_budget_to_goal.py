"""Exam source: a weekly budget built from bills paid at different intervals, followed through to a savings goal.

One scenario: weekly income, a weekly rent, and three bills paid monthly, quarterly and yearly, plus a discretionary
allowance that is a percentage of weekly income.

(a) the total weekly cost of the fixed commitments (each bill converted to a weekly amount: 12 months, 13 weeks a
    quarter, 52 weeks a year);
(b) the weekly amount left for savings, from (a);
(c) the number of whole weeks needed to reach a savings goal, from (b): the learner must round UP.

Cents are exact `Fraction` arithmetic. The chain is evaluated through the rounded answers a learner writes down; a
draw is kept only when the exact route gives the same displayed (b) and the same whole number of weeks.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import attempts, generator, phrase, round_half_up
from mathgen.generators.exd_x_n1_support import money, number_part, show

_STEM = (
    "A person earns ${income} a week. Each week they pay ${rent} in rent. They also pay ${phone} a month for a "
    "phone plan, ${ins} a quarter for car insurance and ${reg} a year for car registration. They set aside {pct}% "
    "of their weekly income for discretionary spending and save the rest. Use 12 months or 52 weeks in a year, "
    "and 13 weeks in a quarter. They want to save ${goal} for a holiday.",
    "A worker's income is ${income} per week. Weekly rent is ${rent}. Other bills are a ${phone} monthly phone "
    "plan, ${ins} each quarter for car insurance and ${reg} each year for registration. {pct}% of weekly income is "
    "kept for discretionary spending and the remainder is saved. Take a year as 12 months or 52 weeks, and a "
    "quarter as 13 weeks. The worker is saving towards a ${goal} holiday.",
)


@generator("exd_x_n1_budget_to_goal", level=11, presets={"exam": dict(tier="exam")})
def exd_x_n1_budget_to_goal(rng, tier="exam"):
    for _ in attempts(1000, "a budget draw with a stable chain"):
        income = Fraction(rng.randrange(700, 1700, 10))
        rent = Fraction(rng.randrange(180, 400, 5))
        phone = Fraction(rng.randint(45, 95))
        ins = Fraction(rng.randrange(120, 430, 10))
        reg = Fraction(rng.randrange(600, 1250, 10))
        pct = rng.randint(8, 20)
        goal = Fraction(rng.randrange(3000, 15100, 100))
        fixed = rent + phone * 12 / 52 + ins / 13 + reg / 52
        discretionary = income * pct / 100
        savings = income - fixed - discretionary
        if savings < 100:
            continue
        fixed_shown = round_half_up(fixed, 2)
        savings_chain = income - fixed_shown - discretionary
        savings_shown = round_half_up(savings_chain, 2)
        if show(savings, 2) != show(savings_chain, 2):
            continue
        weeks = math.ceil(goal / savings_shown)
        if math.ceil(goal / savings) != weeks:
            continue
        ratio = goal / savings_shown
        if ratio.denominator == 1 or abs(ratio - round(ratio)) < Fraction(1, 200):
            continue
        break

    stem = phrase(
        rng,
        _STEM,
        income=money(income, 0),
        rent=money(rent, 0),
        phone=money(phone, 0),
        ins=money(ins, 0),
        reg=money(reg, 0),
        pct=pct,
        goal=money(goal, 0),
    )
    f = Fraction
    part_a = number_part(
        "Find the total weekly cost of the rent and the three bills, correct to the nearest cent.",
        fixed,
        2,
        3,
        [
            f"Phone: {money(phone, 0)} × 12 ÷ 52 = ${show(phone * 12 / 52, 2)}; insurance: {money(ins, 0)} ÷ 13 = "
            f"${show(ins / 13, 2)}; registration: {money(reg, 0)} ÷ 52 = ${show(reg / 52, 2)}.",
            f"Total = {money(rent, 0)} + {show(phone * 12 / 52, 2)} + {show(ins / 13, 2)} + {show(reg / 52, 2)} = "
            f"${show(fixed, 2)}.",
        ],
        [
            (rent + phone / 4 + ins / 13 + reg / 52, None),
            (rent + phone * 12 / 52 + ins / 12 + reg / 12, None),
            (rent + phone + ins + reg, None),
            (rent + phone * 12 / 52 + ins / 13, None),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        f"Using your answer to (a), find how much is saved each week after the bills and the {pct}% discretionary "
        "spending, correct to the nearest cent.",
        savings_chain,
        2,
        2,
        [
            f"Discretionary = {pct}% × {money(income, 0)} = ${show(discretionary, 2)}.",
            f"Savings = {money(income, 0)} - {show(fixed_shown, 2)} - {show(discretionary, 2)} = ${show(savings_chain, 2)}.",
        ],
        [
            (income - fixed_shown, None),
            (income - fixed_shown + discretionary, None),
            ((income - fixed_shown) * (100 - pct) / 100, None),
            (income - fixed_shown - pct, "percent-as-absolute"),
        ],
        follow_through=f"{int(income)} - part_a - {int(income)}*{pct}/100",
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Using your answer to (b), find the smallest whole number of weeks in which the savings goal can be reached.",
        f(weeks),
        0,
        2,
        [
            f"{money(goal, 0)} ÷ {show(savings_shown, 2)} = {float(ratio):.2f} weeks.",
            f"Part of a week is not enough, so round up to {weeks} weeks.",
        ],
        [
            (f(math.floor(ratio)), None),
            (f(weeks + 1), None),
            (f(math.ceil(goal / (income - fixed_shown))), None),
            (f(math.ceil(goal / (income - fixed_shown - pct))), None),
        ],
        follow_through=f"ceiling({int(goal)}/part_b)",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])
