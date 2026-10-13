"""Exam question: hours worked from clock times and an unpaid break, in minutes, decimal hours and pay.

A three-part chain. Part (a) finds the minutes worked between two clock times less a break, part (b) writes
that as a decimal number of hours, and part (c) multiplies by an hourly rate. Parts (b) and (c) carry
`follow_through`.

Built backwards from clock times on the quarter hour and a break that is a multiple of three minutes, so the
minutes worked are a multiple of three and the decimal hours are exact to 2 places; the pay is then exact to
the cent.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase

from .exd_x_m6_support import TooFewDistractors, number_part, shows_no_key, tagged

_STEMS = (
    "A bricklayer in Geelong starts work at {start} and finishes at {end}. An unpaid break of {brk} minutes is "
    "taken during the day. The bricklayer is paid ${rate} for each hour worked.",
    "A barista in Hobart works from {start} until {end}, including an unpaid break of {brk} minutes. Pay is "
    "${rate} for each hour actually worked.",
    "A nursery worker in Townsville clocks on at {start} and off at {end}, and does not get paid for a "
    "{brk}-minute break. The hourly rate is ${rate}.",
)


def _clock(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    suffix = "am" if h < 12 else "pm"
    return f"{(h - 1) % 12 + 1}:{m:02d} {suffix}"


@generator("exd_x_m6_time_pay", level=12, presets={"exam": dict(tier="exam")})
def exd_x_m6_time_pay(rng: random.Random, tier: str = "exam") -> dict:
    """Minutes worked, the same as decimal hours, and the pay for them."""
    check(tier == "exam", "exd_x_m6_time_pay: only the exam tier exists")
    for _ in attempts(1000, "a shift whose figures do not give the answers away"):
        try:
            question = _attempt(rng)
        except TooFewDistractors:
            continue
        if question is not None:
            return question


def _attempt(rng: random.Random) -> dict | None:
    start = rng.randrange(6 * 60, 10 * 60 + 1, 15)
    end = rng.randrange(13 * 60, 18 * 60 + 1, 15)
    brk = rng.randrange(9, 46, 3)
    rate = rng.randint(22, 48)
    worked = end - start - brk
    hours = Fraction(worked, 60)
    pay = hours * rate
    check(worked % 3 == 0 and (hours * 100).denominator == 1 and (pay * 100).denominator == 1, "time_pay: inexact")
    stem = phrase(rng, _STEMS, start=_clock(start), end=_clock(end), brk=brk, rate=rate)
    prompts = (
        "How many minutes of paid work is that?",
        "Write the paid time as a number of hours, as a decimal to 2 decimal places.",
        "Find the pay for the shift, in dollars and cents.",
    )
    if not shows_no_key([stem, *prompts], [worked, hours, pay]):
        return None

    whole, rest = divmod(worked, 60)
    clock_reading = Fraction(whole * 100 + rest, 100)  # 5 h 45 min read as 5.45
    decimal_clock_span = Fraction(end // 60 * 100 + end % 60 - (start // 60 * 100 + start % 60), 100)
    a_cands = [
        tagged(Fraction(end - start), "left-out-the-break"),
        tagged(Fraction(end - start + brk), "left-out-the-break"),
        tagged(decimal_clock_span * 60 - brk, "clock-time-read-as-decimal"),
    ]
    b_cands = [
        tagged(Fraction(worked, 100), "minutes-over-100", 2),
        tagged(clock_reading, "clock-time-read-as-decimal", 2),
        tagged(Fraction(worked, 10), "unit-factor-wrong-power", 2),
        tagged(Fraction(worked), "forgot-unit-conversion", 2),
    ]
    c_cands = [
        tagged(Fraction(worked, 100) * rate, "minutes-over-100", 2),
        tagged(clock_reading * rate, "clock-time-read-as-decimal", 2),
        tagged(Fraction(end - start, 60) * rate, "left-out-the-break", 2),
        tagged(Fraction(worked) * rate, "forgot-unit-conversion", 2),
    ]
    return dict(
        text=stem,
        parts=[
            number_part(
                prompts[0],
                Fraction(worked),
                a_cands,
                marks=2,
                working=[
                    f"From {_clock(start)} to {_clock(end)} is {end - start} minutes.",
                    f"{end - start} - {brk} = {worked} minutes.",
                ],
            ),
            number_part(
                prompts[1],
                hours,
                b_cands,
                marks=1,
                dp=2,
                follow_through="part_a/60",
                working=[f"{worked} ÷ 60 = {float(hours):.2f} hours."],
            ),
            number_part(
                prompts[2],
                pay,
                c_cands,
                marks=1,
                dp=2,
                follow_through=f"part_b*{rate}",
                working=[f"{float(hours):.2f} × {rate} = ${float(pay):.2f}."],
            ),
        ],
    )
