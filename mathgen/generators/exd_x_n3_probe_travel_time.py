"""Exam-tier source for `f10-scientific-notation-measurements`: a journey time from two scientific-notation values.

The practice tiers divide one scientific-notation distance by one speed. An exam question carries the answer on:
part (a) divides a distance by a speed, both in scientific notation with different units of time implied by the
speed (kilometres per hour), to get a time in hours; part (b) converts that time to days, correct to one decimal
place, following through from the learner's own (a). The distance is shown to three significant figures, so the
coefficient division does not come out as a single digit.

Built backwards: the time (a whole number of hours with two significant figures) and the speed are drawn first,
and the distance is their product written in scientific notation. The answer is re-derived by dividing the
printed values as plain numbers.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import (
    check,
    draw_until,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_PROBES = ["a space probe", "a research probe", "an unmanned spacecraft", "a survey craft"]
_TARGETS = ["an asteroid", "a comet", "a distant moon", "a dwarf planet"]


def _sci(coefficient: Fraction, exponent: int) -> str:
    return f"{exact_decimal_string(coefficient, 3)} × 10^{exponent}"


def _days(hours: Fraction) -> str:
    """Hours as days, correct to 1 decimal place."""
    return stated_precision_string(round_half_up(hours / 24, 1), 1)


@generator("exd_x_n3_probe_travel_time", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n3_probe_travel_time(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: time = distance / speed in scientific notation, then convert hours to days."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r):
        return (
            r.randint(2, 9),
            r.choice([3, 4, 5]),
            r.choice([h for h in range(11, 100) if h % 10]),
            r.choice([1, 2, 3]),
        )

    def accept(t):
        s, a, hh, m = t
        hours = hh * 10**m
        frac = (Fraction(hours * 10, 24)) % 1
        return abs(frac - Fraction(1, 2)) > Fraction(1, 20) and s * hh >= 22

    s, a, hh, m = draw_until(rng, draw, accept, what="a speed and a time")
    hours = hh * 10**m
    product = s * hh
    digits = len(str(product))
    coefficient = Fraction(product, 10 ** (digits - 1))
    exponent = a + m + digits - 1
    check(coefficient * 10**exponent == s * 10**a * hours, "distance disagrees with speed times time")
    # Independent route: expand both printed values to plain numbers and divide.
    check(Fraction(coefficient) * 10**exponent / (s * 10**a) == hours, "plain division disagrees")

    a_text = str(hours)
    days_exact = Fraction(hours, 24)
    b_text = _days(Fraction(hours))
    a_cands = [
        hh * 10 ** (m + 1),  # one order of magnitude too large
        hh * 10 ** (m - 1) if m > 1 else None,  # one order of magnitude too small
        s * hours,  # does not divide the coefficients
    ]
    a_dist = pick_distractors(a_text, [str(v) for v in a_cands if v], 3)
    tol = stated_precision_tolerance(1)
    b_cands = [
        stated_precision_string(round_half_up(Fraction(hours) * 24, 1), 1),  # multiplies by 24 instead of dividing
        stated_precision_string(round_half_up(Fraction(hours, 60) / 24, 1), 1),
        stated_precision_string(round_half_up(Fraction(hours, 12), 1), 1),  # divides by 12
        stated_precision_string(round_half_up(Fraction(hours, 7), 1), 1),  # reads 24 as 7
    ]
    b_dist = pick_distractors(b_text, b_cands, 3, tolerance=tol)
    check(len(a_dist) >= 2 and len(b_dist) >= 2, "too few distractors")

    stem = (
        f"{rng.choice(_PROBES).capitalize()} travels at a steady speed of {s} × 10^{a} km/h towards "
        f"{rng.choice(_TARGETS)}. The distance to travel is {_sci(coefficient, exponent)} km."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="How many hours will the journey take?",
                answer_kind="number",
                answer=hours,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Time = distance ÷ speed = ({_sci(coefficient, exponent)}) ÷ ({s} × 10^{a}).",
                    f"= ({exact_decimal_string(coefficient / s, 6)}) × 10^{exponent - a} = {a_text} hours.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Write this time in days, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(round_half_up(days_exact, 1)),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through="part_a/24",
                working=[f"{a_text} ÷ 24 = {b_text} days (1 d.p.)."],
                distractors=b_dist,
            ),
        ],
    )
