"""Exam source of `calculate-overtime`: a week's pay with two overtime tiers, built up in four steps.

The practice tier applies one overtime multiplier to the hours beyond a threshold. A pay question in an exam sets a
week with tiered overtime (the first overtime hours at one loading, the rest at a higher one) and asks for the
pieces in order:

(a) the number of overtime hours worked;
(b) the overtime pay, in two tiers;
(c) the gross pay for the week, from the ordinary pay and (b);
(d) the average amount earned per hour worked, from (c).

Parts (b) to (d) follow through from the earlier ones, so a slip in the hours still earns the later parts.

## Construction

The ordinary rate is a multiple of 20 cents and the loadings are 1.25, 1.5, 2, so every part is a whole number of
cents, except the average in (d), which is rounded half up to the cent and kept clear of the rounding boundary.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)
from ._exd_distinct_parts import distinct_part_answers

_PEOPLE = [
    ("Mia", "a nurse"),
    ("Jack", "an electrician"),
    ("Priya", "a warehouse supervisor"),
    ("Liam", "a bus driver"),
    ("Chloe", "a hospital cleaner"),
    ("Tom", "a mechanic"),
    ("Aisha", "a pharmacy assistant"),
    ("Ben", "a plumber"),
    ("Sophie", "a chef"),
    ("Daniel", "a forklift operator"),
]
_LOADINGS = [
    (Fraction(3, 2), "time and a half", Fraction(2), "double time"),
    (Fraction(5, 4), "time and a quarter", Fraction(3, 2), "time and a half"),
    (Fraction(3, 2), "time and a half", Fraction(5, 2), "double time and a half"),
]


def _money(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 2), 2)


def _cents(value: Fraction) -> Fraction:
    return round_half_up(value, 2)


@generator("exd_x_r3a_overtime_week", level=9, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_overtime_week(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a week whose average pay is clear of a rounding boundary"):
        rate = Fraction(rng.randrange(2400, 5900, 20), 100)
        ordinary = rng.choice([36, 38, 38, 40])
        first_hours = rng.choice([2, 3])
        m1, m1_name, m2, m2_name = rng.choice(_LOADINGS)
        extra = rng.randint(first_hours + 2, first_hours + 9)
        worked = ordinary + extra
        first = min(extra, first_hours)
        rest = extra - first
        overtime_pay = rate * (m1 * first + m2 * rest)
        gross = rate * ordinary + overtime_pay
        average = gross / worked
        if abs((average * 100) % 1 - Fraction(1, 2)) < Fraction(1, 20):
            continue
        if _money(average) in {_money(rate), _money(gross)} or extra in (first_hours, ordinary, worked):
            continue
        shown = {str(ordinary), str(first_hours), str(worked), _money(rate)}
        if {str(extra), _money(overtime_pay), _money(gross), _money(_cents(average))} & shown:
            continue
        break

    avg = _cents(average)

    # Independent route: pay each hour of the week one at a time.
    total = Fraction(0)
    for hour in range(1, worked + 1):
        if hour <= ordinary:
            total += rate
        elif hour <= ordinary + first_hours:
            total += rate * m1
        else:
            total += rate * m2
    check(total == gross, "hour-by-hour pay disagrees with the key")

    def tag(value, mid):
        return Distractor(_money(_cents(value)), mid, _cents(value))

    a_d = pick_distractors(
        str(extra),
        [
            Distractor(str(worked), "total-hours-for-overtime-hours", worked),
            Distractor(str(extra - first_hours), "overtime-tier-hours-confused", extra - first_hours),
            Distractor(str(ordinary), "ordinary-hours-for-overtime-hours", ordinary),
            Distractor(str(extra + 1), "overtime-hours-off-by-one", extra + 1),
        ],
        3,
    )
    b_d = pick_distractors(
        _money(overtime_pay),
        [
            tag(rate * m1 * extra, "single-loading-for-all-overtime"),
            tag(rate * m2 * extra, "single-loading-for-all-overtime"),
            tag(rate * extra, "overtime-loading-omitted"),
            tag(rate * (m2 * first + m1 * rest), "overtime-tiers-swapped"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    c_d = pick_distractors(
        _money(gross),
        [
            tag(rate * m1 * worked, "overtime-multiplier-on-all-hours"),
            tag(rate * ordinary, "overtime-pay-omitted"),
            tag(rate * worked, "overtime-loading-omitted"),
            tag(rate * ordinary + rate * m1 * extra, "single-loading-for-all-overtime"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    d_d = pick_distractors(
        _money(avg),
        [
            tag(gross / ordinary, "average-over-ordinary-hours"),
            tag(gross / extra, "average-over-overtime-hours"),
            tag(rate, "ordinary-rate-for-average-pay"),
            tag(overtime_pay / worked, "overtime-pay-only-averaged"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    for label, ds in (("a", a_d), ("b", b_d), ("c", c_d), ("d", d_d)):
        check(len(ds) >= 2, f"too few ({label}) distractors: {ds}")
    check(all(abs(Fraction(str(d)) - avg) > GRADER_TOLERANCE for d in d_d), "an (d) distractor is accepted")

    name, job = rng.choice(_PEOPLE)
    pronoun_cap = "Last week"
    stem = (
        f"{name} is {job} paid ${_money(rate)} per hour for the first {ordinary} hours of a week. "
        f"Hours beyond {ordinary} are overtime: the first {first_hours} overtime hours are paid at {m1_name} "
        f"and any further overtime hours at {m2_name}. {pronoun_cap}, {name} worked {worked} hours."
    )

    def money_part(prompt, value, marks, ds, working, ft=None):
        part = dict(
            prompt=prompt,
            answer_kind="number",
            answer=numeric_answer(value),
            answer_text=_money(value),
            answer_latex=latex_answer(_money(value)),
            marks=marks,
            sensible_range=(0, None),
            working=working,
            distractors=ds,
        )
        if ft:
            part["follow_through"] = ft
        return part

    m1_t, m2_t = f"{float(m1):g}", f"{float(m2):g}"
    ft_b = (
        f"{rate.numerator}/{rate.denominator}*({m1.numerator}/{m1.denominator}*Min(part_a, {first_hours})"
        f" + {m2.numerator}/{m2.denominator}*Max(part_a - {first_hours}, 0))"
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="How many hours of overtime did this week include?",
                answer_kind="number",
                answer=extra,
                answer_text=str(extra),
                answer_latex=latex_answer(str(extra)),
                marks=1,
                sensible_range=(0, None),
                working=[f"{worked} - {ordinary} = {extra} hours of overtime."],
                distractors=a_d,
            ),
            money_part(
                "Using your answer to (a), find the overtime pay for the week, in dollars.",
                overtime_pay,
                3,
                b_d,
                [
                    f"First {first_hours} hours: {first_hours} × {m1_t} × {_money(rate)} = {_money(rate * m1 * first)}.",
                    f"Remaining {rest} hours: {rest} × {m2_t} × {_money(rate)} = {_money(rate * m2 * rest)}.",
                    f"Overtime pay = ${_money(overtime_pay)}.",
                ],
                ft_b,
            ),
            money_part(
                "Using your answer to (b), find the gross pay for the week, in dollars.",
                gross,
                2,
                c_d,
                [f"{ordinary} × {_money(rate)} + {_money(overtime_pay)} = ${_money(gross)}."],
                f"{ordinary}*{rate.numerator}/{rate.denominator} + part_b",
            ),
            money_part(
                "Using your answer to (c), find the average amount earned per hour worked, "
                "in dollars correct to the nearest cent.",
                avg,
                2,
                d_d,
                [f"${_money(gross)} ÷ {worked} hours = ${_money(avg)} per hour (nearest cent)."],
                f"part_c/{worked}",
            ),
        ],
    )
