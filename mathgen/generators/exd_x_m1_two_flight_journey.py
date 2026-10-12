"""Exam source of `f10-circle-and-prism-duration-rates`: a two-flight journey across three time zones.

The practice tiers handle one flight and a layover in two zones. An exam question gives the clock times of both
flights, each in the local time of the city it is read in, and the differences between the three clocks, then asks
for the flying time of each flight, the length of the stopover and the total journey time. The learner has to
allow for a different clock difference on each leg, notice that the stopover needs no adjustment (both times are on
one clock), and then add up the elapsed times.

## Construction

The cities have fixed clock offsets from UTC, with no daylight saving, and the prompt states every difference it
uses. The journey is drawn forwards in UTC (departure instant, two flying times, a stopover) and the local clock
times are then printed from it, so every answer is exact and whole-minute.
"""

from __future__ import annotations

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors

_OFFSET = {  # hours from UTC, none of which observes daylight saving
    "Brisbane": 10,
    "Perth": 8,
    "Singapore": 8,
    "Tokyo": 9,
    "Dubai": 4,
    "Manila": 8,
    "Hong Kong": 8,
    "Jakarta": 7,
    "Honolulu": -10,
}
_LEG = {  # (from, to): shortest and longest flying time, in minutes
    ("Brisbane", "Singapore"): (450, 500),
    ("Singapore", "Dubai"): (420, 450),
    ("Perth", "Singapore"): (300, 330),
    ("Singapore", "Tokyo"): (390, 450),
    ("Brisbane", "Manila"): (360, 390),
    ("Manila", "Tokyo"): (210, 240),
    ("Brisbane", "Hong Kong"): (480, 510),
    ("Hong Kong", "Tokyo"): (230, 260),
    ("Tokyo", "Honolulu"): (390, 420),
    ("Singapore", "Jakarta"): (100, 120),
    ("Jakarta", "Perth"): (270, 300),
    ("Singapore", "Hong Kong"): (240, 270),
}
_ROUTES = [
    (a, b, c)
    for (a, b) in _LEG
    for (b2, c) in _LEG
    if b == b2 and a != c and len({_OFFSET[a], _OFFSET[b], _OFFSET[c]}) == 3
]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _clock(utc_minutes: int, city: str) -> tuple[int, str]:
    """Day number (from the start of the draw) and "HH:MM" of a UTC instant on a city's clock."""
    local = utc_minutes + 60 * _OFFSET[city]
    return local // 1440, f"{local % 1440 // 60:02d}:{local % 60:02d}"


def _difference(city: str, other: str) -> str:
    """ "Singapore time is 2 hours behind Brisbane time" -- how `city` reads against `other`."""
    hours = _OFFSET[city] - _OFFSET[other]
    word = "hour" if abs(hours) == 1 else "hours"
    return f"{city} time is {abs(hours)} {word} {'ahead of' if hours > 0 else 'behind'} {other} time"


@generator("exd_x_m1_two_flight_journey", level=8, presets={"exam": {"tier": "exam"}})
def exd_x_m1_two_flight_journey(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    origin, stop, dest = rng.choice(_ROUTES)
    lo1, hi1 = _LEG[(origin, stop)]
    lo2, hi2 = _LEG[(stop, dest)]
    for _ in attempts(1000, "a journey whose answers are all distinct from the wrong ones"):
        d1 = rng.randrange(lo1, hi1 + 1, 5)
        d2 = rng.randrange(lo2, hi2 + 1, 5)
        layover = rng.randrange(60, 260, 5)
        depart_local = rng.randrange(0, 1440, 5)
        start = _DAYS.index("Monday") * 1440 + depart_local - 60 * _OFFSET[origin]  # UTC minutes
        arrive1 = start + d1
        depart2 = arrive1 + layover
        arrive2 = depart2 + d2
        total = d1 + layover + d2
        shift1 = 60 * (_OFFSET[stop] - _OFFSET[origin])
        shift2 = 60 * (_OFFSET[dest] - _OFFSET[stop])
        break
    weekday = rng.randrange(7)

    def when(utc: int, city: str) -> str:
        day, clock = _clock(utc, city)
        return f"{clock} on {_DAYS[(weekday + day - _DAYS.index('Monday')) % 7]}"

    t_dep1, t_arr1 = when(start, origin), when(arrive1, stop)
    t_dep2, t_arr2 = when(depart2, stop), when(arrive2, dest)
    stem = (
        f"A traveller flies from {origin} to {dest} with a stopover in {stop}. "
        f"Flight 1 leaves {origin} at {t_dep1} and lands in {stop} at {t_arr1}. "
        f"Flight 2 leaves {stop} at {t_dep2} and lands in {dest} at {t_arr2}. "
        f"All times are local. {_difference(stop, origin)}, and {_difference(dest, stop)}."
    )
    p_a = "Find the flying time of Flight 1, in minutes."
    p_b = f"Find the length of the stopover in {stop}, in minutes."
    p_c = "Find the flying time of Flight 2, in minutes."
    p_d = f"Using your answers to (a), (b) and (c), find the total time from leaving {origin} to landing in {dest}, in minutes."

    def positive(values):
        return [v for v in values if v[0] >= 20]  # a wrong answer a learner could believe is not a few minutes

    a_cands = [
        Distractor(str(v), tag, v)
        for v, tag in positive(
            [
                ((d1 + shift1) % 1440, "ignored-time-zone-change"),
                ((d1 + 2 * shift1) % 1440, "time-zone-applied-backwards"),
            ]
        )
    ]
    b_cands = [
        Distractor(str(v), "zone-shift-within-one-clock", v)
        for v, _ in positive(
            [(layover + shift1, ""), (layover - shift1, ""), (layover + shift2, ""), (layover - shift2, "")]
        )
    ]
    c_cands = [
        Distractor(str(v), tag, v)
        for v, tag in positive(
            [
                ((d2 + shift2) % 1440, "ignored-time-zone-change"),
                ((d2 + 2 * shift2) % 1440, "time-zone-applied-backwards"),
            ]
        )
    ]
    d_cands = [Distractor(str(d1 + d2), "left-out-stopover", d1 + d2)]
    d_cands += [
        Distractor(str(v), tag, v)
        for v, tag in positive(
            [
                ((total + shift1 + shift2) % 1440, "ignored-time-zone-change"),
                ((total + 2 * (shift1 + shift2)) % 1440, "time-zone-applied-backwards"),
            ]
        )
    ]
    a_d = pick_distractors(str(d1), a_cands, 3)
    b_d = pick_distractors(str(layover), b_cands, 3)
    c_d = pick_distractors(str(d2), c_cands, 3)
    d_d = pick_distractors(str(total), d_cands, 3)
    check(min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 1, f"a part has no distractor {a_d} {b_d} {c_d} {d_d}")

    def hm(m: int) -> str:
        return f"{m // 60} h {m % 60} min"

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p_a,
                answer_kind="number",
                answer=d1,
                answer_text=str(d1),
                answer_latex=latex_answer(str(d1)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Read both times on {origin}'s clock: {stop} is {abs(shift1) // 60} h "
                    f"{'ahead of' if shift1 > 0 else 'behind'} {origin}.",
                    f"Flying time = {hm(d1)} = {d1} minutes.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=p_b,
                answer_kind="number",
                answer=layover,
                answer_text=str(layover),
                answer_latex=latex_answer(str(layover)),
                marks=1,
                sensible_range=(0, None),
                working=[
                    f"Both times are on {stop}'s clock, so no adjustment: {t_arr1} to {t_dep2} is {layover} minutes."
                ],
                distractors=b_d,
            ),
            dict(
                prompt=p_c,
                answer_kind="number",
                answer=d2,
                answer_text=str(d2),
                answer_latex=latex_answer(str(d2)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Read both times on {stop}'s clock: {dest} is {abs(shift2) // 60} h "
                    f"{'ahead of' if shift2 > 0 else 'behind'} {stop}.",
                    f"Flying time = {hm(d2)} = {d2} minutes.",
                ],
                distractors=c_d,
            ),
            dict(
                prompt=p_d,
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=latex_answer(str(total)),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_a + part_b + part_c",
                working=[f"{d1} + {layover} + {d2} = {total} minutes."],
                distractors=d_d,
            ),
        ],
    )
