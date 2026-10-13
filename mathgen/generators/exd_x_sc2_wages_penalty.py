"""Exam-tier source for `calculate-wages`: a week's pay with overtime loading, then tax, then an average rate.

A worker is paid an hourly rate for the first block of hours in a week and a loaded rate (time and a half, or
double time) for every hour after that. Three connected parts:

(a) the gross pay for the week (ordinary pay plus loaded pay);
(b) the tax withheld, a stated percentage of the gross pay (follow-through from (a));
(c) the gross pay per hour worked, averaged over the whole week, to the nearest cent (follow-through from (a)).

Built backwards from whole cents: the hourly rate is an even number of cents so the loaded rate is a whole
number of cents too, and the tax percentage is drawn so the tax is a whole number of cents. The averaging in (c)
is kept clear of a half-cent boundary, so rounding the gross pay first or last gives the same key.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    attempts,
    check,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_TOL = stated_precision_tolerance(2)

_JOBS = (
    ("forklift driver", "a warehouse in Dandenong"),
    ("barista", "a cafe in Fremantle"),
    ("apprentice plumber", "a building firm in Newcastle"),
    ("checkout operator", "a supermarket in Toowoomba"),
    ("aged care assistant", "a nursing home in Bendigo"),
    ("kitchen hand", "a restaurant in Hobart"),
    ("delivery driver", "a bakery in Launceston"),
    ("groundskeeper", "a golf club in Albury"),
)
_LOADINGS = (
    (Fraction(3, 2), "time and a half"),
    (Fraction(2), "double time"),
)
_TAX_PERCENTS = (8, 10, 12, 15, 16, 18, 20, 22, 24, 25, 30)


def _money(value: Fraction) -> str:
    """A dollar amount on the cent grid, at exactly two decimal places."""
    return stated_precision_string(value, 2)


def _dis(value: Fraction, tag: str | None) -> Distractor:
    """A distractor showing `value` rounded to the cent (the value is the displayed one)."""
    shown = round_half_up(value, 2)
    return Distractor(_money(shown), tag, exact_value(shown))


@generator("exd_x_sc2_wages_penalty", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc2_wages_penalty(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: gross pay with a loaded rate, tax withheld, and the average gross pay per hour."""
    check(tier == "exam", f"unknown tier {tier!r}")
    job, place = rng.choice(_JOBS)
    loading, loading_text = rng.choice(_LOADINGS)
    for _ in attempts(2000, "a week whose tax is whole cents and whose average is clear of a half cent"):
        rate_cents = 2 * rng.randint(1150, 2350)  # $23.00 .. $47.00, an even number of cents
        ordinary = rng.randint(20, 38)
        extra = rng.randint(2, 9)
        percent = rng.choice(_TAX_PERCENTS)
        rate = Fraction(rate_cents, 100)
        gross = rate * ordinary + rate * loading * extra
        if gross.denominator != 1 and (gross * 100).denominator != 1:
            continue
        if (gross * percent / 100 * 100).denominator != 1:
            continue
        total_hours = ordinary + extra
        average = gross / total_hours
        offset = (average * 100) % 1
        if abs(offset - Fraction(1, 2)) < Fraction(1, 20) or offset == 0:
            continue  # keep the average away from a rounding boundary and not an exact cent
        break

    tax = gross * percent / 100
    average_cents = round_half_up(average, 2)
    check(gross == rate * ordinary + loading * rate * extra, "gross pay disagrees")
    check(tax * 100 == (gross * 100 * percent) / 100, "tax is not whole cents")
    # The learner's route: gross first (to the cent), then divide.
    check(round_half_up(gross / total_hours, 2) == average_cents, "average depends on rounding order")

    stem = (
        f"{'An' if job[0] in 'aeiou' else 'A'} {job} at {place} is paid ${_money(rate)} per hour for the first {ordinary} hours worked in a week and "
        f"{loading_text} for every hour after that. Last week the {job} worked {total_hours} hours."
    )

    gross_text, tax_text, avg_text = _money(gross), _money(tax), _money(average_cents)
    loading_extra = loading - 1
    a_cands = [
        _dis(rate * total_hours, "overtime-at-ordinary-rate"),
        _dis(rate * loading * total_hours, "loading-applied-to-all-hours"),
        _dis(rate * ordinary + rate * loading_extra * extra, "loading-only-for-extra-hours"),
        _dis(rate * ordinary, "omitted-overtime-hours"),
    ]
    b_cands = [
        _dis(gross - tax, "gave-remainder-not-percentage-amount"),
        _dis(gross * percent, "percent-not-divided-by-100"),
        _dis(rate * ordinary * percent / 100, "tax-on-ordinary-pay-only"),
        _dis(gross * (100 - percent) / 100, "found-complement-not-event"),
    ]
    c_cands = [
        _dis(gross / ordinary, "averaged-over-ordinary-hours-only"),
        _dis((gross - tax) / total_hours, "averaged-net-not-gross"),
        _dis(rate, "gave-ordinary-rate-as-average"),
        _dis(rate * (1 + loading) / 2, None),
    ]
    a_dis = pick_distractors(gross_text, a_cands, 3, tolerance=_TOL)
    b_dis = pick_distractors(tax_text, b_cands, 3, tolerance=_TOL)
    c_dis = pick_distractors(avg_text, c_cands, 3, tolerance=_TOL)
    check(len(a_dis) >= 2 and len(b_dis) >= 2 and len(c_dis) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the gross pay for the week, in dollars.",
                answer_kind="number",
                answer=float(gross),
                answer_text=gross_text,
                answer_latex=latex_answer(gross_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Ordinary pay: {ordinary} x ${_money(rate)} = ${_money(rate * ordinary)}.",
                    f"Loaded rate: {loading_text} = {loading} x ${_money(rate)} = ${_money(rate * loading)} per hour.",
                    f"Loaded pay: {extra} x ${_money(rate * loading)} = ${_money(rate * loading * extra)}.",
                    f"Gross pay = ${_money(rate * ordinary)} + ${_money(rate * loading * extra)} = ${gross_text}.",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt=f"Tax of {percent}% of the gross pay is withheld. Find the tax withheld, in dollars.",
                answer_kind="number",
                answer=float(tax),
                answer_text=tax_text,
                answer_latex=latex_answer(tax_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{percent}*part_a/100",
                working=[f"Tax = {percent}% of ${gross_text} = {percent}/100 x {gross_text} = ${tax_text}."],
                distractors=b_dis,
            ),
            dict(
                prompt="Find the gross pay per hour worked, averaged over the whole week, to the nearest cent.",
                answer_kind="number",
                answer=float(average_cents),
                answer_text=avg_text,
                answer_latex=latex_answer(avg_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a/{total_hours}",
                working=[
                    f"Average = gross pay / hours worked = {gross_text} / {total_hours} = {avg_text} "
                    "dollars per hour (nearest cent)."
                ],
                distractors=c_dis,
            ),
        ],
    )
