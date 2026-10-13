"""Exam-tier chain for `senior-compare-the-time-taken-to-travel-a-specific`: two ways of making the same trip.

One option is a direct run at a steady speed. The other is a faster service that needs a fixed time to reach
and wait for it, and whose speed may be given in m/s. The chain asks for

(a) the direct trip's time in minutes,
(b) the faster service's door-to-door time in minutes (ride plus the fixed extra),
(c) by how many minutes the quicker option wins, and
(d) the door-to-door average speed of the second option, which the fixed extra drags well below its running speed.

Built backwards from whole-minute times: the distance is drawn so both ride times are whole numbers of minutes,
so every answer except (d) is an integer and (d) is the one rounded value, taken from (b) exactly.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator

from .exd_x_r2i_support import clear_of_boundary, number_part

# (direct option, speeds in km/h)
_DIRECT = [
    ("car", [60, 72, 80, 90, 100]),
    ("bike", [15, 18, 20, 24]),
    ("ferry", [30, 36, 45, 54]),
]
# (service, speeds in km/h, fixed-extra wording)
_SERVICE = [
    ("express train", [72, 90, 108, 120, 144], "reach the station and wait for the train"),
    ("express bus", [60, 72, 80, 90], "walk to the stop and wait for the bus"),
    ("tram", [36, 45, 54, 60], "walk to the stop and wait for the tram"),
]
_PEOPLE = ["Priya", "Tom", "Mei", "Jack", "Aisha", "Liam", "Sofia", "Noah"]
_PLACES = ["a regional town", "the coast", "a sports centre", "a hospital appointment", "a university campus"]


@generator("exd_x_r2i_trip_compare", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_r2i_trip_compare(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: compare a direct trip with a faster service that has a fixed extra time, then find its average speed."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a distance giving whole-minute ride times"):
        gerund, direct_speeds = rng.choice(_DIRECT)
        service, service_speeds, extra_text = rng.choice(_SERVICE)
        v1 = rng.choice(direct_speeds)
        v2 = rng.choice(service_speeds)
        in_ms = v2 % 18 == 0 and rng.random() < 0.5  # speeds divisible by 18 km/h are whole numbers of m/s
        distance = rng.randrange(10, 121, 5)
        wait = rng.choice([10, 15, 20, 25, 30, 35, 40])
        t_direct = Fraction(60 * distance, v1)
        ride = Fraction(60 * distance, v2)
        if t_direct.denominator != 1 or ride.denominator != 1 or not 20 <= t_direct <= 240:
            continue
        total = ride + wait
        if abs(t_direct - total) < 4:
            continue
        avg = Fraction(60 * distance) / total
        if clear_of_boundary(avg, 1):
            break
    t_direct, ride, total = int(t_direct), int(ride), int(total)
    gap = abs(t_direct - total)
    quicker = service if total < t_direct else f"{gerund}"
    name = rng.choice(_PEOPLE)

    speed_service = f"{v2 * 5 // 18} m/s" if in_ms else f"{v2} km/h"
    stem = (
        f"{name} can make a {distance} km trip to {rng.choice(_PLACES)} in two ways. By {gerund}, the "
        f"average speed is {v1} km/h. By {service}, the average speed is {speed_service}, but it takes "
        f"{wait} minutes to {extra_text}."
    )

    hours = Fraction(distance, v1)
    candidates_a = [
        (hours, "unconverted-time-unit", 2),
        (Fraction(100 * distance, v1), "decimal-time-base-100"),
        (Fraction(60 * v1, distance), "inverted-ratio"),
        (Fraction(distance, v1) * 6, "unconverted-time-unit", 1),
    ]
    raw_v2 = v2 * 5 // 18 if in_ms else v2
    candidates_b = [
        (ride, "left-out-fixed-extra"),
        (Fraction(distance, v2) + wait, "mixed-time-units", 1),
        (Fraction(100 * distance, v2) + wait, "decimal-time-base-100"),
    ]
    if in_ms:
        candidates_b.insert(0, (Fraction(60 * distance, raw_v2) + wait, "unconverted-speed-unit"))
    candidates_c = [
        (-gap, "reversed-difference"),
        (t_direct + total, "added-instead-of-subtracted"),
        (abs(t_direct - ride), "left-out-fixed-extra"),
        (t_direct, "gave-one-measure-not-difference"),
    ]
    candidates_d = [
        (Fraction(distance * 60, ride), "left-out-fixed-extra"),
        (Fraction(distance * 100, total), "decimal-time-base-100"),
        (Fraction(distance * 60, t_direct), "used-time-of-other-route"),
        (Fraction(distance, total), "unconverted-time-unit", 2),
    ]
    ride_work = (
        f"Speed = {v2 * 5 // 18} m/s × 3.6 = {v2} km/h. " if in_ms else ""
    ) + f"Ride = {distance} ÷ {v2} × 60 = {ride} minutes."

    parts = [
        number_part(
            f"Find the time for the trip by {gerund}, in minutes.",
            Fraction(t_direct),
            0,
            1,
            [f"Time = {distance} ÷ {v1} × 60 = {t_direct} minutes."],
            candidates_a,
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the total time for the trip by {service}, in minutes, including the {wait} minutes.",
            Fraction(total),
            0,
            2,
            [ride_work, f"Total = {ride} + {wait} = {total} minutes."],
            candidates_b,
            sensible_range=(0, None),
        ),
        number_part(
            "By how many minutes is the quicker option quicker than the other?",
            Fraction(gap),
            0,
            1,
            [f"The {quicker} option is quicker: |{t_direct} − {total}| = {gap} minutes."],
            candidates_c,
            follow_through="Abs(part_a - part_b)",
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the average speed for the whole trip by {service}, from the start of the {wait} minutes to "
            "arrival, in km/h. Give your answer to 1 decimal place.",
            avg,
            1,
            2,
            [f"Average speed = {distance} ÷ ({total} ÷ 60) = {distance * 60 / total:.2f} km/h."],
            candidates_d,
            follow_through=f"{60 * distance}/part_b",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
