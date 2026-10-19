"""Salary, wage and piecework conversions: `r3j_salary` (easy, medium, hard) and the exam chain
`exd_x_r3j_salary_vs_hourly`.

``salary-conversion-pay`` (``financial-discrete``, answer ``number``). A year is taken as 52 weeks, 26 fortnights or
12 months, and the hours of a week are always stated.

* easy: an annual salary written as a pay for one week, fortnight or month; or the reverse.
* medium: the hourly rate equal to a salary, a weekly allowance added to a salary paid fortnightly, or the monthly
  pay after a percentage rise.
* hard: piecework paid at two rates with a guaranteed minimum (the larger of the two is paid), or the yearly
  difference between a salary and an hourly job worked for fewer than 52 weeks.
* exam (``exd_x_r3j_salary_vs_hourly``): a salary and an hourly job with overtime; the salary as a weekly pay and an
  hourly rate, the hourly job's week and the yearly gap between the two.

Each key is rebuilt a second way, by converting through the week (or by summing the work done), not through the
formula the slips use.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3j_kit import PEOPLE, dollars, number_part, number_text, plain, single_number

_JOBS = [
    "a primary school teacher",
    "an electrician",
    "a paramedic",
    "a bus driver",
    "a physiotherapist",
    "a surveyor",
    "a pharmacist",
    "a diesel mechanic",
    "an accountant",
    "a civil engineer",
    "a dental assistant",
    "a warehouse supervisor",
]
_PERIOD = {"week": 52, "fortnight": 26, "month": 12}
_HOURS = [35, 36, 37.5, 38, 40]


def _who(rng):
    name = rng.choice(PEOPLE)
    return name, rng.choice(_JOBS)


def _salary(rng, lo=42, hi=140):
    return 1000 * rng.randint(lo, hi) + rng.choice([0, 0, 500, 250, 750])


def _easy(rng):
    name, job = _who(rng)
    period = rng.choice(list(_PERIOD))
    n = _PERIOD[period]
    if rng.random() < 0.65:
        salary = _salary(rng)
        key = Fraction(salary, n)
        text = (
            f"{name}, {job}, earns a salary of {dollars(salary)} a year. How much is {name}'s pay for one {period}, "
            "to the nearest cent? Count a year as 52 weeks, 26 fortnights or 12 months."
        )
        others = [p for p in _PERIOD if p != period]
        slips = [(Fraction(salary, _PERIOD[o]), "r3j-salary-wrong-period-divisor") for o in others]
        slips += [
            (Fraction(salary, n + 2), "r3j-salary-divided-by-near-count"),
            (
                Fraction(salary, 365) * (7 if period == "week" else 14 if period == "fortnight" else 30),
                "r3j-salary-days-not-weeks",
            ),
        ]
        working = [f"One {period} is 1/{n} of a year: {dollars(salary)} / {n} = {dollars(key, 2)}."]
        check(key * n == salary, "converting back does not give the salary")
        return single_number(text, key, 2, working, slips)
    low, high = {"week": (900, 2600), "fortnight": (1800, 5200), "month": (3600, 11000)}[period]
    pay = Fraction(rng.randint(low, high)) + Fraction(rng.choice([0, 50]), 100)
    key = pay * n
    text = f"{name}, {job}, is paid {dollars(pay, 2)} each {period}. What is {name}'s annual salary, in dollars? Count a year as 52 weeks, 26 fortnights or 12 months."
    others = [p for p in _PERIOD if p != period]
    slips = [(pay * _PERIOD[o], "r3j-salary-wrong-period-multiplier") for o in others]
    slips.append((pay * (n + 2), "r3j-salary-multiplied-by-near-count"))
    working = [f"A year has {n} {period}s: {dollars(pay, 2)} x {n} = {dollars(key, 2)}."]
    return single_number(text, key, 0, working, slips)


def _medium_hourly(rng):
    name, job = _who(rng)
    hours = rng.choice(_HOURS)
    salary = _salary(rng, 46, 130)
    rate = Fraction(salary) / 52 / Fraction(hours).limit_denominator(10)
    text = (
        f"{name}, {job}, earns a salary of {dollars(salary)} a year for a {plain(hours)}-hour week, 52 weeks a year. "
        "What hourly rate is that equal to, to the nearest cent?"
    )
    h = Fraction(hours).limit_denominator(10)
    check(rate * h * 52 == salary, "hourly rate does not rebuild the salary")
    slips = [
        (Fraction(salary, 52), "r3j-salary-hours-not-divided"),
        (Fraction(salary, 12) / h, "r3j-salary-month-used-as-week"),
        (Fraction(salary, 26) / h, "r3j-salary-fortnight-used-as-week"),
        (Fraction(salary) / (48 * h), "r3j-salary-leave-weeks-removed"),
        (Fraction(salary) / h, "r3j-salary-year-over-weekly-hours"),
    ]
    working = [
        f"Weekly pay = {dollars(salary)} / 52 = {dollars(Fraction(salary, 52), 2)}.",
        f"Hourly rate = {dollars(Fraction(salary, 52), 2)} / {plain(hours)} = {dollars(rate, 2)}.",
    ]
    return single_number(text, rate, 2, working, slips)


def _medium_allowance(rng):
    name, job = _who(rng)
    salary = _salary(rng, 48, 110)
    allowance = rng.choice([25, 30, 35, 40, 45, 55, 60, 75])
    key = Fraction(salary, 26) + 2 * allowance
    text = (
        f"{name}, {job}, earns a salary of {dollars(salary)} a year and also receives an allowance of "
        f"{dollars(allowance)} each week. How much does {name} receive each fortnight in total, to the nearest "
        "cent? Count a year as 52 weeks or 26 fortnights."
    )
    check(key == Fraction(salary + 52 * allowance, 26), "fortnightly total disagrees when built from the annual total")
    slips = [
        (Fraction(salary, 26) + allowance, "r3j-salary-allowance-counted-once"),
        (Fraction(salary + allowance, 26), "r3j-salary-weekly-allowance-added-to-year"),
        (Fraction(salary, 52) + 2 * allowance, "r3j-salary-weekly-divisor-for-fortnight"),
        (Fraction(salary, 26) + 52 * allowance, "r3j-salary-allowance-annualised-in-fortnight"),
        (Fraction(salary, 26), "r3j-salary-allowance-left-out"),
    ]
    working = [
        f"Salary per fortnight = {dollars(salary)} / 26 = {dollars(Fraction(salary, 26), 2)}.",
        f"Allowance per fortnight = 2 x {dollars(allowance)} = {dollars(2 * allowance)}.",
        f"Total = {dollars(key, 2)}.",
    ]
    return single_number(text, key, 2, working, slips)


def _medium_rise(rng):
    name, job = _who(rng)
    salary = _salary(rng, 50, 120)
    pct = rng.choice([2, 3, 3.5, 4, 4.5, 5, 6])
    p = Fraction(str(pct))
    key = Fraction(salary) * (100 + p) / 100 / 12
    text = (
        f"{name}, {job}, earns a salary of {dollars(salary)} a year. After a pay rise of {plain(p)}%, "
        f"how much will {name} earn each month, to the nearest cent? Count a year as 12 months."
    )
    slips = [
        (Fraction(salary) * p / 100 / 12, "r3j-salary-rise-only"),
        (Fraction(salary, 12) + p, "r3j-salary-percent-added-as-dollars"),
        (Fraction(salary) * (100 + p) / 100 / 52, "r3j-salary-wrong-period-divisor"),
        (Fraction(salary, 12), "r3j-salary-rise-left-out"),
        (Fraction(salary) * (100 - p) / 100 / 12, "r3j-salary-rise-subtracted"),
    ]
    working = [
        f"New salary = {dollars(salary)} x {plain(1 + p / 100)} = {dollars(Fraction(salary) * (100 + p) / 100, 2)}.",
        f"Monthly pay = {dollars(Fraction(salary) * (100 + p) / 100, 2)} / 12 = {dollars(key, 2)}.",
    ]
    return single_number(text, key, 2, working, slips)


def _medium(rng):
    return rng.choice([_medium_hourly, _medium_allowance, _medium_rise])(rng)


_PIECES = [
    ("a fruit picker", "kilograms of fruit", "kilogram", 1.8, 2.6, 120, 90),
    ("a furniture assembler", "chairs assembled", "chair", 14, 19, 25, 280),
    ("a garment machinist", "shirts sewn", "shirt", 6, 8.5, 60, 300),
    ("a newspaper deliverer", "papers delivered", "paper", 0.9, 1.3, 300, 160),
    ("a mail sorter", "bags of parcels sorted", "bag", 4.5, 6.5, 80, 250),
]


def _hard_piecework(rng):
    name = rng.choice(PEOPLE)
    job, plural, singular, r1, r2, tier_at, minimum = rng.choice(_PIECES)
    low = Fraction(str(r1))
    high = Fraction(str(r2))
    for _ in attempts(500, "a week whose piecework differs from the minimum"):
        units = rng.randint(int(tier_at * 0.6), int(tier_at * 2.2))
        piece = low * min(units, tier_at) + high * max(units - tier_at, 0)
        if abs(piece - minimum) > 15:
            break
    key = max(piece, Fraction(minimum))
    text = (
        f"{name}, {job}, is paid {dollars(low, 2)} for each of the first {tier_at} {singular}s "
        f"in a week and {dollars(high, 2)} for each one after that, but is guaranteed at least {dollars(minimum)} a week. "
        f"In one week {name} completes {units} {singular}s. How much is {name} paid for the week, to the nearest cent?"
    )
    slips = [
        (high * units, "r3j-salary-top-rate-on-all"),
        (low * units, "r3j-salary-first-rate-on-all"),
        (min(piece, Fraction(minimum)), "r3j-salary-minimum-taken-as-cap"),
        (piece + minimum, "r3j-salary-minimum-added"),
    ]
    working = [
        f"Piecework = {dollars(low, 2)} x {min(units, tier_at)} + {dollars(high, 2)} x {max(units - tier_at, 0)} = {dollars(piece, 2)}.",
        f"The guarantee is {dollars(minimum)}, so {name} is paid the larger: {dollars(key, 2)}.",
    ]
    return single_number(text, key, 2, working, slips)


def _hard_year(rng):
    name = rng.choice(PEOPLE)
    hours = rng.choice([35, 36, 38, 40])
    salary = _salary(rng, 55, 95)
    weeks = rng.choice([44, 46, 48, 50])
    for _ in attempts(500, "an hourly job that pays more than the salary"):
        rate = Fraction(rng.randint(36, 62)) + Fraction(rng.choice([0, 1, 2, 3]), 4)
        hourly_year = rate * hours * weeks
        if hourly_year > salary + 300:
            break
    key = hourly_year - salary
    text = (
        f"{name} is offered two jobs. Job A pays a salary of {dollars(salary)} a year. Job B pays {dollars(rate, 2)} "
        f"an hour for {hours} hours a week, but only for {weeks} weeks of the year. How much more does Job B pay in a "
        "year than Job A, to the nearest dollar?"
    )
    slips = [
        (rate * hours * 52 - salary, "r3j-salary-fifty-two-weeks-assumed"),
        (salary - hourly_year, "r3j-salary-difference-wrong-way"),
        (rate * weeks - salary, "r3j-salary-hours-left-out"),
        (rate * hours * weeks / 52 - Fraction(salary, 52), "r3j-salary-weekly-gap-given"),
        (hourly_year, "r3j-salary-one-job-only"),
        (hourly_year - Fraction(salary * weeks, 52), "r3j-salary-salary-prorated-to-weeks"),
        (rate * hours * (weeks + 2) - salary, "r3j-salary-weeks-off-by-two"),
    ]
    working = [
        f"Job B in a year = {dollars(rate, 2)} x {hours} x {weeks} = {dollars(hourly_year, 2)}.",
        f"Difference = {dollars(hourly_year, 2)} - {dollars(salary)} = {dollars(key, 2)}.",
    ]
    return single_number(text, key, 0, working, slips)


@generator(
    "r3j_salary",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3j_salary(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_piecework, _hard_year])(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_x_r3j_salary_vs_hourly", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3j_salary_vs_hourly(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    name = rng.choice(PEOPLE)
    for _ in attempts(500, "two jobs where the hourly job pays more"):
        rate = Fraction(rng.randint(30, 52)) + Fraction(rng.choice([0, 1, 2, 3]), 4)
        extra = rng.choice([2, 3, 4, 5, 6])
        weekly_b = 38 * rate + Fraction(3, 2) * rate * extra
        salary = 1000 * round(float(weekly_b) * 52 * rng.choice([0.84, 0.88, 0.92, 0.95]) / 1000)
        weekly_a = Fraction(salary, 52)
        if weekly_b - weekly_a > 40:
            break
    hourly_a = weekly_a / 38
    gap_year = 52 * (weekly_b - weekly_a)
    check(gap_year == 52 * weekly_b - salary, "yearly gap disagrees when built from the yearly totals")
    text = (
        f"{name} is choosing between two jobs, each with a 38-hour ordinary week and 52 weeks of pay a year. Job A pays "
        f"a salary of {dollars(salary)} a year. Job B pays {dollars(rate, 2)} an hour, with overtime for the hours "
        f"beyond 38 paid at time-and-a-half; {name} would expect to work {extra} hours of overtime each week."
    )
    parts = [
        number_part(
            "Find the weekly pay for Job A, to the nearest cent.",
            weekly_a,
            2,
            1,
            [f"{dollars(salary)} / 52 = {dollars(weekly_a, 2)}."],
            [
                (Fraction(salary, 26), "r3j-salary-wrong-period-divisor"),
                (Fraction(salary, 12), "r3j-salary-wrong-period-divisor"),
                (Fraction(salary, 38), "r3j-salary-year-over-weekly-hours"),
            ],
        ),
        number_part(
            "Find the hourly rate that Job A's salary is equal to for an ordinary week, to the nearest cent.",
            hourly_a,
            2,
            1,
            [f"{dollars(weekly_a, 2)} / 38 = {dollars(hourly_a, 2)}."],
            [
                (weekly_a, "r3j-salary-hours-not-divided"),
                (Fraction(salary, 52 * 40), "r3j-salary-wrong-hours"),
                (weekly_a / 48, "r3j-salary-leave-weeks-removed"),
            ],
            follow_through="part_a/38",
        ),
        number_part(
            "Find the weekly pay for Job B including the overtime, to the nearest cent.",
            weekly_b,
            2,
            2,
            [
                f"Ordinary pay = 38 x {dollars(rate, 2)} = {dollars(38 * rate, 2)}.",
                f"Overtime = {extra} x 1.5 x {dollars(rate, 2)} = {dollars(Fraction(3, 2) * rate * extra, 2)}.",
                f"Total = {dollars(weekly_b, 2)}.",
            ],
            [
                ((38 + extra) * rate, "r3j-salary-overtime-at-ordinary-rate"),
                (38 * rate + 2 * rate * extra, "r3j-salary-overtime-at-double"),
                (Fraction(3, 2) * (38 + extra) * rate, "r3j-salary-multiplier-on-all"),
                (38 * rate, "r3j-salary-overtime-left-out"),
            ],
        ),
        number_part(
            "How much more does Job B pay than Job A in a year, to the nearest dollar?",
            gap_year,
            0,
            2,
            [
                f"Weekly gap = {dollars(weekly_b, 2)} - {dollars(weekly_a, 2)} = {dollars(weekly_b - weekly_a, 2)}.",
                f"Yearly gap = 52 x that = {dollars(gap_year, 2)}.",
            ],
            [
                (weekly_b - weekly_a, "r3j-salary-weekly-gap-given"),
                (26 * (weekly_b - weekly_a), "r3j-salary-wrong-period-multiplier"),
                (52 * weekly_b, "r3j-salary-one-job-only"),
                (salary - 52 * weekly_b, "r3j-salary-difference-wrong-way"),
            ],
            follow_through="52*(part_c - part_a)",
        ),
    ]
    return dict(text=text, parts=parts)
