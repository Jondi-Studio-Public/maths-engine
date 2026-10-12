"""Generates `exd_x_r3j_tide_chart_rate` questions: reading a tide chart and working out a rate from it.

Skill: `senior-interpret-complex-timetables-such-as-tide-charts-sunrise` (exam tier). The practice tiers pick one
entry from a table that meets a condition; an exam attaches a calculation to the table. For a day's tide chart:

(a) the time from the first low tide to the first high tide, in minutes     number, 1 mark
(b) the rise in height over that time, in centimetres                       number, 1 mark
(c) the average rate of rise, in cm per hour                                number, 2 marks, 60 part_b / part_a
(d) minutes after the low tide until a boat ramp's depth is reached,
    taking the water to rise steadily                                       number, 2 marks, 60 (L - low) / part_c

## Construction

The rate r (cm/h) and the duration a (minutes) are drawn so that the rise r a / 60 is a whole number of
centimetres; the low-tide height is then a two-decimal height and the high-tide height is that plus the rise. The
depth the ramp needs is the low height plus r t / 60 cm for a whole number of minutes t, so (d) is exact. The
chart also lists a second low and high tide, a day's worth, which serve as decoys for the wrong-pair slips.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._exd_x_r3_distinct import distinct_answers
from ._exd_x_r3j_kit import number_part

_PLACES = [
    "Mooloolaba",
    "Fremantle",
    "Port Macquarie",
    "Glenelg",
    "Eden",
    "Bunbury",
    "Hobart",
    "Gladstone",
]
_RAMP_STEMS = [
    "A boat ramp at {place} can be used once the water is at least {level} m deep.",
    "Boats can be launched from the ramp at {place} once the tide has reached {level} m.",
    "The channel at {place} is safe for a small boat once the water height reaches {level} m.",
]


def _clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _metres(centimetres: int) -> str:
    return f"{centimetres // 100}.{centimetres % 100:02d}"


@generator("exd_x_r3j_tide_chart_rate", level=11, presets={"exam": {}})
@distinct_answers
def exd_x_r3j_tide_chart_rate(rng):
    place = rng.choice(_PLACES)
    for _ in attempts(5000, "a duration and rate with whole-centimetre rise and whole-minute ramp time"):
        duration = rng.randrange(335, 401, 5)  # minutes from low tide to high tide: 5 h 35 min to 6 h 40 min
        rate = rng.randint(14, 44)
        if (rate * duration) % 60 != 0:
            continue
        rise = rate * duration // 60
        if not 70 <= rise <= 240:
            continue
        ramp_minutes = rng.randrange(60, duration - 59, 5)
        if (rate * ramp_minutes) % 60 == 0:
            break
    ramp_rise = rate * ramp_minutes // 60
    low_height = rng.randrange(15, 66, 5)  # cm above chart datum, two-decimal heights in the table
    high_height = low_height + rise
    ramp_level = low_height + ramp_rise
    check(ramp_rise * 60 == rate * ramp_minutes and rise * 60 == rate * duration, "rate does not divide exactly")

    start = rng.randrange(30, 200, 5)  # first low tide, minutes after midnight (00:30 to 03:20)
    high_time = start + duration
    low2 = start + 745 + rng.randrange(-20, 21, 5)
    duration2 = duration + rng.randrange(-15, 16, 5)
    high2 = low2 + duration2
    low2_height = low_height + rng.randrange(-10, 11, 5)
    high2_height = high_height + rng.randrange(-20, 1, 5)
    check(high2 < 24 * 60, "second high tide runs past midnight")
    check(high2_height - low2_height > 0, "second tide does not rise")

    chart = (
        f"A tide chart for {place} shows the following for one day. "
        f"Low tide at {_clock(start)}, height {_metres(low_height)} m. "
        f"High tide at {_clock(high_time)}, height {_metres(high_height)} m. "
        f"Low tide at {_clock(low2)}, height {_metres(low2_height)} m. "
        f"High tide at {_clock(high2)}, height {_metres(high2_height)} m."
    )
    stem = f"{chart} {rng.choice(_RAMP_STEMS).format(place=place, level=_metres(ramp_level))}"

    h_hi, m_hi = divmod(high_time, 60)
    h_lo, m_lo = divmod(start, 60)
    decimal_clock = round(((h_hi + m_hi / 100) - (h_lo + m_lo / 100)) * 60)
    no_minutes = (h_hi - h_lo) * 60
    borrow_slip = abs(h_hi - h_lo) * 60 + abs(m_hi - m_lo)
    hours_only = max(h_hi - h_lo, 1)

    parts = [
        number_part(
            "How many minutes are there from the first low tide to the first high tide?",
            duration,
            0,
            1,
            [f"{_clock(high_time)} − {_clock(start)} = {duration // 60} h {duration % 60} min = {duration} minutes."],
            [
                (decimal_clock, "treated-minutes-as-decimals"),
                (no_minutes, "ignored-the-minutes"),
                (borrow_slip, "subtracted-minutes-without-borrowing"),
                (high2 - low2, "read-wrong-tide"),
                ((h_hi - h_lo) * 60 + m_hi + m_lo, "added-instead-of-subtracted"),
                (duration + 60, "borrowed-hour-wrongly"),
                (duration - 60, "borrowed-hour-wrongly"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "By how many centimetres does the water rise between these two tides?",
            rise,
            0,
            1,
            [f"{_metres(high_height)} m − {_metres(low_height)} m = {_metres(rise)} m = {rise} cm."],
            [
                (high_height - low2_height, "read-wrong-tide"),
                (high_height + low_height, "added-instead-of-subtracted"),
                (high2_height - low2_height, "read-wrong-tide"),
                (high_height, "gave-height-not-rise"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find the average rate at which the water rises over this time, in centimetres per hour.",
            rate,
            0,
            2,
            [f"Rate = {rise} cm ÷ ({duration} ÷ 60) h = {rate} cm per hour."],
            [
                (value, tag)
                for value, tag in (
                    (round(rise / hours_only), "ignored-the-minutes"),
                    (round(rise * duration / 60), "multiplied-instead-of-divided"),
                    (round(rise * 60 / decimal_clock) if decimal_clock else rise, "treated-minutes-as-decimals"),
                    (round(rise / duration), "wrong-rate-time-unit"),
                    (round(rise / (duration / 100)), "treated-minutes-as-decimals"),
                    (round(rise / (-(-duration // 60))), "rounded-the-duration-up-to-whole-hours"),
                )
                if value >= 1
            ],
            follow_through="60*part_b/part_a",
            sensible_range=(0, None),
        ),
        number_part(
            f"Taking the water to rise at this steady rate, how many minutes after the low tide does it reach "
            f"{_metres(ramp_level)} m, to the nearest minute?",
            ramp_minutes,
            0,
            2,
            [
                f"The water must rise {_metres(ramp_level)} m − {_metres(low_height)} m = {ramp_rise} cm.",
                f"Time = {ramp_rise} ÷ {rate} h = {ramp_minutes} minutes.",
            ],
            [
                (duration - ramp_minutes, "timed-from-the-high-tide"),
                (round(ramp_level * 60 / rate), "forgot-to-subtract-the-low-height"),
                (round(ramp_rise * 60 / max(round(rise / hours_only), 1)), "ignored-the-minutes"),
                (
                    round(ramp_rise * 60 / max(round(rise * 60 / decimal_clock), 1)) if decimal_clock else 1,
                    "treated-minutes-as-decimals",
                ),
                (round(ramp_rise / rise * 60), "fraction-of-an-hour-for-fraction-of-the-duration"),
            ],
            follow_through=f"{ramp_rise}*60/part_c",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
