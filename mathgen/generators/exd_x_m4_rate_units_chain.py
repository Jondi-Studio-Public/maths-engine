"""Exam tier for "convert between units for rates": a three-part speed chain.

A distance in kilometres and a time in minutes are given, so the rate has to be built before it can be converted.
Part (a) finds the average speed in km/h (distance over time, with the minutes turned into hours). Part (b) converts
that speed to m/s, and part (c) uses the m/s speed to find how long a further distance in metres takes, to the
nearest second. Parts (b) and (c) carry `follow_through`.

## Construction: the displayed answers are kept consistent

The speed is drawn first, within a plausible range for the context, with a time in whole minutes; the distance is that
speed over that time, rounded to 2 decimal places. Each canonical answer is worked out from the exact values. The
learner's route (each answer rounded as asked and then used in the next part) must give the same displayed answers,
so a draw is rejected unless both routes agree and every rounding is clear of a boundary.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

# (opening of the stem, plural noun for the further trip, speed range in km/h)
_CONTEXTS = (
    ("A cyclist rides {d} km in {t} minutes at a steady pace.", "ride", (16, 42)),
    ("A runner covers {d} km in {t} minutes.", "run", (8, 18)),
    ("A bus travels {d} km between two stops in {t} minutes.", "travel", (30, 70)),
    ("A regional train covers {d} km in {t} minutes.", "travel", (80, 150)),
    ("A car on the highway covers {d} km in {t} minutes.", "travel", (70, 110)),
)
_MARGIN = Fraction(3, 100)


def _clear(value: Fraction, places: int) -> bool:
    scaled = value * 10**places
    return abs(scaled % 1 - Fraction(1, 2)) > _MARGIN


@generator("exd_x_m4_rate_units_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m4_rate_units_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Speed from distance and time, then m/s, then a time for a further distance. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a trip with consistent rounding"):
        opening, verb, (lo, hi) = rng.choice(_CONTEXTS)
        t = rng.randint(12, 95)
        speed_target = Fraction(rng.randint(lo * 10, hi * 10), 10)
        d = round_half_up(speed_target * t / 60, 2)
        m = 50 * rng.randint(4, 40)  # metres of the further trip
        v = 60 * d / t  # km/h, exact
        a = round_half_up(v, 1)
        b_exact = v / Fraction(18, 5)
        b = round_half_up(b_exact, 1)
        if b != round_half_up(a / Fraction(18, 5), 1) or not (_clear(v, 1) and _clear(b_exact, 1)):
            continue
        sec_exact, sec = m / b_exact, m / b
        if round_half_up(sec_exact) != round_half_up(sec) or not _clear(sec_exact / 10, 0) or sec < 20:
            continue
        stem = opening.format(d=exact_decimal_string(d, 2), t=t)
        a_text, b_text = stated_precision_string(a, 1), stated_precision_string(b, 1)
        s_text = str(int(round_half_up(sec)))
        tail = f"At this speed, how long would it take to {verb} a further {m} m?"
        if all(x not in (stem + " " + tail).split() for x in (a_text, b_text, s_text)):
            break
    d_f = d
    sec_int = int(round_half_up(sec))

    def one_dp(value: Fraction) -> Distractor:
        return Distractor(stated_precision_string(round_half_up(value, 1), 1), None, round_half_up(value, 1))

    a_options = [d_f / t, 60 * t / d_f, d_f * t / 60, d_f * 60]
    b_options = [
        Distractor(
            stated_precision_string(round_half_up(a * Fraction(18, 5), 1), 1),
            "unit-conversion-wrong-way",
            round_half_up(a * Fraction(18, 5), 1),
        ),
        one_dp(a / 3),  # 3 used for the factor 3.6
        one_dp(a / 60),
        one_dp(a * 1000 / 60),
    ]
    c_values = [m / a, m * b, m / (b * Fraction(18, 5)), m / b / 60]
    c_options = [Distractor(str(int(round_half_up(x))), None, round_half_up(x)) for x in c_values if x >= 5]
    a_d = pick_distractors(a_text, [one_dp(x) for x in a_options], 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_options, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(s_text, c_options, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "not enough distractors")
    return dict(
        text=f"{stem}",
        parts=[
            dict(
                prompt="Find the average speed in kilometres per hour, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(a),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{t} minutes = {t}/60 hours.",
                    f"Speed = {exact_decimal_string(d, 2)} ÷ ({t}/60) = {a_text} km/h.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Convert this speed to metres per second, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(b),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_a/3.6",
                working=[f"1 km/h = 1000 m ÷ 3600 s, so {a_text} ÷ 3.6 = {b_text} m/s."],
                distractors=b_d,
            ),
            dict(
                prompt=tail.replace("At this speed, h", "Using your speed in m/s, h").replace(
                    "?", ", to the nearest second?"
                ),
                answer_kind="number",
                answer=sec_int,
                answer_text=s_text,
                answer_latex=latex_answer(s_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{m}/part_b",
                working=[f"Time = distance ÷ speed = {m} ÷ {b_text} = {s_text} s (nearest second)."],
                distractors=c_d,
            ),
        ],
    )
