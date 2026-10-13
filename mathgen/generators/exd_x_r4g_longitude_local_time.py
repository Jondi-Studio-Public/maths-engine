"""Exam chain on two ports: longitude difference, time difference, local clock time and a distance along a meridian.

(a) the difference in longitude between Port A and Port B, in degrees           number, 1 mark
(b) the time difference between them, in hours (follow-through (a))             number, 1 mark
(c) the local time at Port B when it is a stated time at Port A (follow-through (b))   number, 2 marks
(d) the distance along the meridian from Port B to Port C, in kilometres         number, 2 marks

The clock times keep the whole day inside 1000 to 2359, so adding or taking away whole hours changes only the hour
digits of the four-digit time. The follow-through for (c) is therefore `time +/- 100 * part_b`.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4g_kit import clock_text, hhmm, number_part, number_text, retry
from .r4g_longitude_local_time_build import RULE, _fits, draw_clock, draw_places, local_time, lon_text, wrap

KM_PER_DEGREE = Fraction(1112, 10)  # stated in the question: 111.2 km along a meridian


def lat_text(deg: int) -> str:
    return f"{abs(deg)}°{'N' if deg >= 0 else 'S'}"


@distinct_part_answers
def _build(rng):
    for _ in attempts(3000, "a clock time that stays inside the day"):
        a, b, la, lb = draw_places(rng, opposite=rng.random() < 0.6)
        t = draw_clock(rng, 1000, 2000)
        ans = local_time(t, la, lb)
        if _fits(ans) and ans != t and abs(lb - la) >= 45:
            break
    c = rng.choice([p for p in ("Dunmoor", "Eskwith", "Farrow", "Garrick") if p not in (a, b)])
    for _ in attempts(500, "latitudes in opposite hemispheres"):
        lat_b, lat_c = rng.randint(5, 40), -rng.randint(5, 40)
        if rng.random() < 0.5:
            lat_b, lat_c = -lat_b, -lat_c
        if abs(lat_b - lat_c) >= 20 and abs(lat_b) != abs(lat_c):
            break
    dlat = abs(lat_b - lat_c)
    dist = dlat * KM_PER_DEGREE
    sep = abs(lb - la)
    hours = sep // 15
    sign = 1 if lb > la else -1
    check(hhmm(ans) == hhmm(t) + sign * 100 * hours, "hour digits do not move by the time difference")
    text = (
        f"Port {a} is at longitude {lon_text(la)}. Port {b} is at longitude {lon_text(lb)} and latitude {lat_text(lat_b)}. "
        f"Port {c} is on the same meridian as Port {b}, at latitude {lat_text(lat_c)}. {RULE} "
        f"Along a meridian, 1° of latitude is 111.2 km. The local time at Port {a} is {clock_text(t)}."
    )
    follow = f"{hhmm(t)} + 100*part_b" if sign > 0 else f"{hhmm(t)} - 100*part_b"
    other_sign = -sign
    parts = [
        number_part(
            f"Find the difference in longitude between Port {a} and Port {b}, in degrees.",
            sep,
            0,
            1,
            [
                f"{'Opposite sides of the prime meridian: add' if la * lb < 0 else 'Same side: subtract'} the longitudes: {sep}°."
            ],
            [
                (abs(la) + abs(lb) if la * lb > 0 else abs(abs(la) - abs(lb)), "r4g-longitude-rule-wrong-side"),
                (sep // 15, "r4g-degrees-left-as-hours"),
                (abs(lat_b - lat_c), "r4g-latitude-for-longitude"),
                (abs(la), "r4g-one-longitude-not-the-difference"),
            ],
            sensible_range=(0, 360),
        ),
        number_part(
            f"Use your answer to (a) to find the time difference between the two ports, in hours.",
            hours,
            0,
            1,
            [f"{sep}° / 15 = {hours} hours."],
            [
                (sep // 30, "r4g-half-the-hours"),
                (sep * 15, "r4g-multiplied-by-15"),
                (sep // 10, "r4g-ten-degrees-per-hour"),
            ],
            follow_through="part_a/15",
            sensible_range=(0, 24),
        ),
        number_part(
            f"Use your answer to (b) to find the local time at Port {b} at that moment. "
            "Give the time in 24-hour form as a four-digit number, such as 1342.",
            hhmm(ans),
            0,
            2,
            [
                f"Port {b} is {'east' if sign > 0 else 'west'} of Port {a}, so its clock is {hours} hours {'ahead' if sign > 0 else 'behind'}: {clock_text(ans)}."
            ],
            [
                (wrap(t + other_sign * hours * 60), "r4g-east-west-direction-reversed"),
                (wrap(t), "r4g-no-time-shift"),
                (wrap(t + sign * (hours + 1) * 60), "r4g-time-zone-off-by-one-hour"),
                (wrap(t + sign * (hours - 1) * 60), "r4g-time-zone-off-by-one-hour"),
            ],
            follow_through=follow,
            sensible_range=(1000, 2359),
        ),
        number_part(
            f"Find the distance along the meridian from Port {b} to Port {c}, in kilometres.",
            dist,
            1,
            2,
            [
                f"The latitudes are in opposite hemispheres, so they differ by {abs(lat_b)} + {abs(lat_c)} = {dlat}°; {dlat} x 111.2 = {number_text(dist, 1)} km."
            ],
            [
                (abs(abs(lat_b) - abs(lat_c)) * KM_PER_DEGREE, "wrong-hemisphere-rule"),
                (Fraction(dlat), "r4g-degrees-given-as-km"),
                (dlat * 60, "r4g-nautical-miles-given-as-km"),
                (sep * KM_PER_DEGREE, "r4g-longitude-for-latitude"),
            ],
        ),
    ]
    return dict(text=text, parts=parts)


@generator("exd_x_r4g_longitude_local_time", level=11, presets={"exam": {}})
def exd_x_r4g_longitude_local_time(rng):
    return retry(_build, rng)
