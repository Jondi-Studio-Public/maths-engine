"""Speed, distance, time and average speed over a journey: `r3j_speed` (easy, medium, hard) and the exam chain
`exd_x_r3j_speed_journey`.

``speed-distance-time-average-speed`` (``measurement``, answer ``number``). Times are given in hours and minutes;
the formula is never printed.

* easy: a speed, a time or a distance from the other two, with the minutes converted to hours.
* medium: the average speed over a journey in two legs (each leg given as a distance and a speed), which is the total
  distance over the total time and not the mean of the two speeds.
* hard: the speed needed on the last leg to reach a stated average, or the average speed over three legs and a
  rest stop.
* exam (``exd_x_r3j_speed_journey`` ): the time of each of two legs, the total time with a rest stop, and the
  average speed for the whole journey including the stop.

Each key is rebuilt a second way, from the legs' times in minutes against the distance in kilometres per minute.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from functools import partial

from ._r3j_kit import PEOPLE, number_part, number_text, plain
from ._r3j_kit import single_number as _single_number

single_number = partial(_single_number, spread=100)

# (traveller phrase, vehicle, speed choices in km/h)
_MODES = [
    ("drives", "a car", [50, 60, 70, 80, 90, 100, 110]),
    ("rides", "a bicycle", [15, 18, 20, 24, 25, 30]),
    ("travels by train", "a train", [80, 90, 100, 120, 140, 160]),
    ("runs", "a run", [8, 9, 10, 12, 14]),
    ("walks", "a walk", [4, 5, 6]),
    ("travels by ferry", "a ferry", [20, 24, 30, 36]),
]


def _hm(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    if h and m:
        return f"{h} hour{'s' if h != 1 else ''} {m} minutes"
    if h:
        return f"{h} hour{'s' if h != 1 else ''}"
    return f"{m} minutes"


def _is_tie(value, places) -> bool:
    scaled = Fraction(value) * 10**places
    return (scaled - int(scaled)) * 2 in (1, -1)


def _easy_speed(rng):
    verb, _veh, speeds = rng.choice(_MODES)
    name = rng.choice(PEOPLE)
    for _ in attempts(500, "a speed clear of a rounding tie"):
        speed = rng.choice(speeds)
        minutes = rng.choice([75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225])
        distance = Fraction(int(speed * minutes / 60) + rng.choice([-3, -2, -1, 1, 2, 3, 4]))
        key = distance * 60 / minutes
        if distance > 5 and not _is_tie(key, 1):
            break
    h, m = divmod(minutes, 60)
    text = (
        f"{name} {verb} {plain(distance)} km in {_hm(minutes)}. What is {name}'s average speed in kilometres per hour, "
        "to 1 decimal place?"
    )
    check(key * Fraction(minutes, 60) == distance, "speed does not rebuild the distance")
    slips = [
        (distance / (h + Fraction(m, 100)), "r3j-speed-minutes-as-hundredths"),
        (distance / h if h else distance / 2, "r3j-speed-minutes-ignored"),
        (distance / minutes, "r3j-speed-per-minute-given"),
        (distance * 60 / (h + m), "r3j-speed-hours-and-minutes-added"),
        (distance * minutes / 60, "r3j-speed-multiplied-for-divided"),
    ]
    working = [
        f"{_hm(minutes)} = {number_text(Fraction(minutes, 60), 4)} hours.",
        f"Speed = {plain(distance)} / {number_text(Fraction(minutes, 60), 4)} = {number_text(key, 1)} km/h.",
    ]
    return single_number(text, key, 1, working, slips)


def _easy_time(rng):
    verb, _veh, speeds = rng.choice(_MODES)
    name = rng.choice(PEOPLE)
    for _ in attempts(500, "a whole number of minutes"):
        speed = rng.choice(speeds)
        minutes = rng.choice([20, 25, 30, 36, 40, 45, 48, 50, 54, 72, 80, 84, 90, 96, 105, 108, 120, 150])
        distance = Fraction(speed) * minutes / 60
        if distance.denominator == 1 and distance >= 5:
            break
    text = f"{name} {verb} {plain(distance)} km at an average speed of {speed} km/h. How many minutes does it take?"
    slips = [
        (Fraction(distance, speed), "r3j-speed-hours-given-as-minutes"),
        (Fraction(distance, speed) * 100, "r3j-speed-hours-times-hundred"),
        (distance * speed, "r3j-speed-multiplied-for-divided"),
        (Fraction(speed, 1) / distance * 60, "r3j-speed-divided-the-wrong-way"),
    ]
    working = [
        f"Time = {plain(distance)} / {speed} = {number_text(Fraction(distance, speed), 4)} hours.",
        f"In minutes: {number_text(Fraction(distance, speed), 4)} x 60 = {minutes}.",
    ]
    return single_number(text, Fraction(minutes), 0, working, slips)


def _easy_distance(rng):
    verb, _veh, speeds = rng.choice(_MODES)
    name = rng.choice(PEOPLE)
    for _ in attempts(500, "a speed and a time that give a clean distance"):
        speed = rng.choice(speeds)
        minutes = rng.choice([15, 20, 30, 40, 45, 50, 75, 80, 100, 110, 130])
        distance = Fraction(speed) * minutes / 60
        if (distance * 10).denominator == 1 and distance >= 2:
            break
    text = (
        f"{name} {verb} at an average speed of {speed} km/h for {_hm(minutes)}. How far does {name} go, in kilometres?"
    )
    slips = [
        (Fraction(speed * minutes), "r3j-speed-minutes-not-converted"),
        (Fraction(speed) * Fraction(minutes, 100), "r3j-speed-minutes-as-hundredths"),
        (Fraction(speed, minutes), "r3j-speed-divided-for-multiplied"),
        (Fraction(speed) * (minutes // 60 + minutes % 60), "r3j-speed-hours-and-minutes-added"),
    ]
    working = [
        f"{_hm(minutes)} = {number_text(Fraction(minutes, 60), 4)} hours.",
        f"Distance = {speed} x {number_text(Fraction(minutes, 60), 4)} = {number_text(distance, 1)} km.",
    ]
    return single_number(text, distance, 1, working, slips)


def _easy(rng):
    return rng.choice([_easy_speed, _easy_time, _easy_distance])(rng)


def _legs(rng, equal=False):
    """(verb, d1, v1, d2, v2) with each leg's time a multiple of 1/12 hour."""
    verb, _veh, speeds = rng.choice(_MODES[:3])
    for _ in attempts(1000, "two legs with sensible times"):
        v1, v2 = rng.sample(speeds, 2)
        t1 = Fraction(rng.choice([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18]), 6)
        t2 = t1 if equal else Fraction(rng.choice([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18]), 6)
        d1, d2 = v1 * t1, v2 * t2
        if equal:
            d2 = d1
            t2 = d2 / v2
        if d1.denominator == 1 and d2.denominator == 1 and 0.5 <= t1 <= 4 and 0.5 <= t2 <= 4 and abs(v1 - v2) >= 10:
            return verb, int(d1), v1, int(d2), v2
    raise AssertionError  # pragma: no cover


def _average(d1, v1, d2, v2):
    return Fraction(d1 + d2) / (Fraction(d1, v1) + Fraction(d2, v2))


def _medium(rng):
    name = rng.choice(PEOPLE)
    for _ in attempts(500, "an average speed clear of a rounding tie"):
        equal = rng.random() < 0.4
        verb, d1, v1, d2, v2 = _legs(rng, equal)
        key = _average(d1, v1, d2, v2)
        mean = Fraction(v1 + v2, 2)
        if not _is_tie(key, 1) and abs(key - mean) > Fraction(1, 2):
            break
    t1, t2 = Fraction(d1, v1), Fraction(d2, v2)
    check(key * (t1 + t2) == d1 + d2, "average speed does not rebuild the distance")
    check(key * (t1 * 60 + t2 * 60) == (d1 + d2) * 60, "minute route disagrees")
    if equal:
        text = (
            f"{name} {verb} {d1} km to a town at {v1} km/h and returns along the same road, {d2} km, at {v2} km/h. "
            "What is the average speed for the whole trip, in kilometres per hour, to 1 decimal place?"
        )
    else:
        text = (
            f"{name} {verb} {d1} km at {v1} km/h and then {d2} km at {v2} km/h. What is the average speed for the "
            "whole trip, in kilometres per hour, to 1 decimal place?"
        )
    slips = [
        (mean, "r3j-speed-mean-of-speeds"),
        (Fraction(d1 * v1 + d2 * v2, d1 + d2), "r3j-speed-distance-weighted-mean"),
        (Fraction(d1 + d2) / t1, "r3j-speed-total-distance-over-first-time"),
        (t1 + t2, "r3j-speed-total-time-given"),
        (Fraction(d1 + d2), "r3j-speed-total-distance-given"),
    ]
    working = [
        f"Leg 1 takes {d1} / {v1} = {number_text(t1, 3)} hours; leg 2 takes {d2} / {v2} = {number_text(t2, 3)} hours.",
        f"Average speed = {d1 + d2} / {number_text(t1 + t2, 3)} = {number_text(key, 1)} km/h.",
    ]
    return single_number(text, key, 1, working, slips)


def _hard_target(rng):
    name = rng.choice(PEOPLE)
    for _ in attempts(5000, "a trip whose last-leg speed is sensible and whose average is a clean number"):
        verb, _veh, speeds = rng.choice(_MODES[:3])
        v1, v2 = rng.sample(speeds, 2)
        d1 = rng.choice([60, 80, 90, 100, 120, 150, 180, 200])
        d2 = rng.choice([40, 60, 80, 90, 100, 120, 150])
        t = Fraction(d1, v1) + Fraction(d2, v2)
        target = Fraction(d1 + d2) / t
        if (target * 10).denominator == 1 and v1 != target and v2 != target and abs(v2 - v1) >= 10 and t >= 1.5:
            break
    text = (
        f"{name} {verb} the first {d1} km of a {d1 + d2} km trip at {v1} km/h. To average {plain(target)} km/h for the "
        f"whole trip, at what constant speed must {name} travel the last {d2} km? Give your answer in kilometres per hour."
    )
    key = Fraction(v2)
    total_time = Fraction(d1 + d2) / target
    left = total_time - Fraction(d1, v1)
    check(Fraction(d2) / left == key, "required speed does not rebuild the leg")
    slips = [
        (2 * target - v1, "r3j-speed-mean-of-speeds-reversed"),
        (target, "r3j-speed-target-given"),
        (Fraction(d2) / (total_time), "r3j-speed-last-leg-over-whole-time"),
        (Fraction(d2) / Fraction(d1, v1), "r3j-speed-first-leg-time-used"),
        (v1 + (target - v1) * 2, "r3j-speed-linear-adjustment"),
    ]
    working = [
        f"Total time allowed = {d1 + d2} / {plain(target)} = {number_text(total_time, 3)} hours.",
        f"Time used on the first leg = {d1} / {v1} = {number_text(Fraction(d1, v1), 3)} hours, leaving {number_text(left, 3)} hours.",
        f"Speed needed = {d2} / {number_text(left, 3)} = {plain(key)} km/h.",
    ]
    return single_number(text, key, 0, working, slips)


def _hard_stop(rng):
    name = rng.choice(PEOPLE)
    for _ in attempts(2000, "a trip with a stop and a clean-ish average"):
        verb, _veh, speeds = rng.choice(_MODES[:3])
        v1, v2, v3 = rng.choice(speeds), rng.choice(speeds), rng.choice(speeds)
        t1, t2 = (Fraction(rng.choice([3, 4, 6, 8, 9, 12]), 6) for _ in range(2))
        d1, d2 = v1 * t1, v2 * t2
        stop = rng.choice([15, 20, 30, 40, 45])
        d3 = rng.choice([40, 60, 80, 90, 120])
        if d1.denominator != 1 or d2.denominator != 1 or len({v1, v2, v3}) < 2:
            continue
        total_t = t1 + t2 + Fraction(d3, v3) + Fraction(stop, 60)
        key = Fraction(int(d1) + int(d2) + d3) / total_t
        moving = Fraction(int(d1) + int(d2) + d3) / (total_t - Fraction(stop, 60))
        if not _is_tie(key, 1) and abs(key - moving) >= Fraction(3, 10):
            break
    d1, d2 = int(d1), int(d2)
    text = (
        f"{name} {verb} {d1} km at {v1} km/h, then {d2} km at {v2} km/h. {name} then rests for {stop} minutes and "
        f"finishes with a further {d3} km at {v3} km/h. What is the average speed for the whole journey, including the "
        "rest, in kilometres per hour, to 1 decimal place?"
    )
    slips = [
        (moving, "r3j-speed-stop-left-out"),
        (Fraction(v1 + v2 + v3, 3), "r3j-speed-mean-of-speeds"),
        (Fraction(d1 + d2 + d3) / (total_t - Fraction(stop, 60) + stop), "r3j-speed-stop-minutes-as-hours"),
        (
            Fraction(d1 + d2 + d3) / (Fraction(d1, v1) + Fraction(d2, v2) + Fraction(stop, 60)),
            "r3j-speed-last-leg-left-out",
        ),
    ]
    working = [
        f"Driving time = {number_text(total_t - Fraction(stop, 60), 3)} hours; with the rest, {number_text(total_t, 3)} hours.",
        f"Average speed = {d1 + d2 + d3} / {number_text(total_t, 3)} = {number_text(key, 1)} km/h.",
    ]
    return single_number(text, key, 1, working, slips)


@generator(
    "r3j_speed",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3j_speed(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_target, _hard_stop])(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_x_r3j_speed_journey", level=13, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3j_speed_journey(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    name = rng.choice(PEOPLE)
    for _ in attempts(2000, "a journey with clean times and an average clear of ties"):
        verb, d1, v1, d2, v2 = _legs(rng)
        stop = rng.choice([15, 30, 45])
        t1, t2 = Fraction(d1, v1), Fraction(d2, v2)
        total = t1 + t2 + Fraction(stop, 60)
        key = Fraction(d1 + d2) / total
        if (
            not _is_tie(key, 1)
            and (t1 * 100).denominator == 1
            and (t2 * 100).denominator == 1
            and len({t1, t2, total}) == 3
        ):
            break
    text = (
        f"{name} {verb} {d1} km at {v1} km/h, rests for {stop} minutes, and then goes a further {d2} km at {v2} km/h."
    )
    parts = [
        number_part(
            "Find the time taken for the first part of the trip, in hours.",
            t1,
            2,
            1,
            [f"{d1} / {v1} = {number_text(t1, 2)} hours."],
            [
                (Fraction(v1, d1), "r3j-speed-divided-the-wrong-way"),
                (Fraction(d1 * v1), "r3j-speed-multiplied-for-divided"),
                (Fraction(d1, v2), "r3j-speed-other-leg-speed-used"),
                (t1 * 60, "r3j-speed-minutes-given-as-hours"),
            ],
            spread=100,
        ),
        number_part(
            "Find the time taken for the second part of the trip, in hours.",
            t2,
            2,
            1,
            [f"{d2} / {v2} = {number_text(t2, 2)} hours."],
            [
                (Fraction(v2, d2), "r3j-speed-divided-the-wrong-way"),
                (Fraction(d2 * v2), "r3j-speed-multiplied-for-divided"),
                (Fraction(d2, v1), "r3j-speed-other-leg-speed-used"),
                (t2 * 60, "r3j-speed-minutes-given-as-hours"),
            ],
            spread=100,
        ),
        number_part(
            "Find the total time for the whole journey, rest included, in hours.",
            total,
            2,
            2,
            [f"{number_text(t1, 2)} + {number_text(t2, 2)} + {stop}/60 = {number_text(total, 2)} hours."],
            [
                (t1 + t2, "r3j-speed-stop-left-out"),
                (t1 + t2 + stop, "r3j-speed-stop-minutes-as-hours"),
                (t1 + t2 + Fraction(stop, 100), "r3j-speed-minutes-as-hundredths"),
            ],
            follow_through=f"part_a + part_b + {stop}/60",
            spread=100,
        ),
        number_part(
            "Find the average speed for the whole journey, rest included, in kilometres per hour, to 1 decimal place.",
            key,
            1,
            2,
            [f"{d1 + d2} / {number_text(total, 2)} = {number_text(key, 1)} km/h."],
            [
                (Fraction(v1 + v2, 2), "r3j-speed-mean-of-speeds"),
                (Fraction(d1 + d2) / (t1 + t2), "r3j-speed-stop-left-out"),
                (Fraction(d1 + d2) / (t1 + t2 + Fraction(stop, 100)), "r3j-speed-minutes-as-hundredths"),
            ],
            follow_through=f"{d1 + d2}/part_c",
            spread=100,
        ),
    ]
    return dict(text=text, parts=parts)
