"""Exam-tier chain for `senior-interpret-timetables-such-as-bus-train-and-ferry`: plan a journey from a timetable.

A traveller reaches the first stop at a stated time and must be at the second stop by a deadline. Six services run
the same route. The chain asks for

(a) the wait for the latest service that still arrives by the deadline,
(b) that service's journey time,
(c) the minutes to spare at the deadline, from (a) and (b), and
(d) the average speed over the whole trip, wait included, from (a) and (b).

## Construction

The chosen service is placed first: it leaves after the traveller arrives and gets in with spare time. Then the
decoys are added: two that leave later but arrive too late, two that arrive in time but leave earlier, and one that
has already left. The answer is then re-derived by the definition (latest departure at or after the arrival time
among services arriving by the deadline), and the draw is rejected unless that is the chosen service alone.
Everything is whole minutes, so only (d) is rounded.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator

from .exd_x_r2i_support import clear_of_boundary, number_part

# (first stop, second stop)
_ROUTES = [
    ("Riverside", "Central"),
    ("Northgate", "Harbour"),
    ("Southbank", "Airport"),
    ("Lakeside", "Town Hall"),
    ("Westfield", "Marina"),
    ("Parkview", "City Wharf"),
]
# (mode, lowest and highest average speed in km/h)
_MODES = [("train", 62, 95), ("bus", 38, 55), ("ferry", 28, 42)]
_PEOPLE = ["Priya", "Tom", "Mei", "Jack", "Aisha", "Liam", "Sofia", "Noah"]


def _clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@generator("exd_x_r2i_timetable_chain", level=20, presets={"exam": {"tier": "exam"}})
def exd_x_r2i_timetable_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: wait, journey time, time to spare and whole-trip speed from a six-service timetable."""
    check(tier == "exam", f"unknown tier {tier!r}")
    start, end = rng.choice(_ROUTES)
    for _ in attempts(2000, "a timetable with exactly one latest service that arrives in time"):
        distance = rng.randrange(24, 73, 2)
        arrive_stop = rng.randrange(6 * 60 + 15, 17 * 60, 5)  # the traveller reaches the first stop
        deadline = arrive_stop + rng.randrange(75, 140, 5)

        def run_time(mode):
            _, lo, hi = next(m for m in _MODES if m[0] == mode)
            return round(60 * distance / rng.randint(lo, hi))

        chosen_mode = rng.choice([m[0] for m in _MODES])
        wait = rng.randint(4, 30)
        journey = run_time(chosen_mode)
        depart = arrive_stop + wait
        arrive = depart + journey
        spare = deadline - arrive
        if not 3 <= spare <= 25:
            continue
        services = [(chosen_mode, depart, arrive)]
        for _ in range(2):  # later, but too late to get in on time
            mode = rng.choice([m[0] for m in _MODES])
            d = depart + rng.randint(6, 35)
            services.append((mode, d, d + run_time(mode)))
        for _ in range(2):  # earlier and still in time
            mode = rng.choice([m[0] for m in _MODES])
            d = arrive_stop + rng.randint(0, max(0, wait - 3))
            services.append((mode, d, d + run_time(mode)))
        mode = rng.choice([m[0] for m in _MODES])
        d = arrive_stop - rng.randint(4, 25)  # already left
        services.append((mode, d, d + run_time(mode)))

        times = [s[1] for s in services]
        if len(set(times)) != len(times):
            continue
        suitable = [s for s in services if s[1] >= arrive_stop and s[2] <= deadline]
        late = [s for s in services[1:3] if s[2] <= deadline]
        if late or max(suitable, key=lambda s: s[1]) != services[0] or len(suitable) < 2:
            continue
        total = wait + journey
        speed = Fraction(60 * distance, total)
        if clear_of_boundary(speed, 1):
            break

    services.sort(key=lambda s: s[1])
    listing = "; ".join(f"the {_clock(d)} {mode} arrives at {_clock(a)}" for mode, d, a in services)
    name = rng.choice(_PEOPLE)
    stem = (
        f"{name} reaches {start} at {_clock(arrive_stop)} and must be at {end} by {_clock(deadline)}. The route is "
        f"{distance} km. Services leave {start} for {end} (24-hour times) as follows: {listing}. {name} can only "
        "catch a service that has not yet left."
    )

    # Re-derive the answers from the listing alone, as a learner reads it.
    ok = [s for s in services if s[1] >= arrive_stop and s[2] <= deadline]
    pick = max(ok, key=lambda s: s[1])
    check(pick[1] - arrive_stop == wait and pick[2] - pick[1] == journey, "wait or journey disagrees with the listing")
    check(deadline - pick[2] == spare, "time to spare disagrees with the listing")
    latest_overall = max(services, key=lambda s: s[1])
    earliest_ok = min(ok, key=lambda s: s[1])

    def wait_of(s):
        return s[1] - arrive_stop

    def journey_of(s):
        return s[2] - s[1]

    k = deadline - arrive_stop
    parts = [
        number_part(
            f"How many minutes does {name} wait at {start} for the latest service that still reaches {end} by "
            f"{_clock(deadline)}?",
            Fraction(wait),
            0,
            2,
            [
                f"Services arriving by {_clock(deadline)}: the latest to leave is the {_clock(depart)} {chosen_mode}.",
                f"Wait = {_clock(depart)} − {_clock(arrive_stop)} = {wait} minutes.",
            ],
            [
                (wait_of(latest_overall), "ignored-the-deadline"),
                (wait_of(earliest_ok), "took-first-service-not-latest"),
                (journey, "swapped-wait-and-journey"),
                *[
                    (wait_of(s), "ignored-the-deadline" if s[1] > depart else "took-first-service-not-latest")
                    for s in services
                    if s[1] >= arrive_stop
                ],
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "How many minutes does the journey on that service take?",
            Fraction(journey),
            0,
            1,
            [f"{_clock(arrive)} − {_clock(depart)} = {journey} minutes."],
            [
                (journey_of(latest_overall), "ignored-the-deadline"),
                (journey_of(earliest_ok), "took-first-service-not-latest"),
                (wait + journey, "included-the-wait-in-the-journey"),
                # The journey times printed for the other services, which a learner reads off the wrong row.
                *[
                    (journey_of(s), "ignored-the-deadline" if s[1] > depart else "took-first-service-not-latest")
                    for s in services
                ],
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"By how many minutes does {name}'s arrival at {end} beat the deadline?",
            Fraction(spare),
            0,
            2,
            [f"From reaching {start} to the deadline is {k} minutes.", f"{k} − {wait} − {journey} = {spare} minutes."],
            [
                (k - journey, "left-out-fixed-extra"),
                (deadline - depart, "used-departure-not-arrival"),
                (-spare, "reversed-difference"),
            ],
            follow_through=f"{k} - part_a - part_b",
            sensible_range=(0, None),
        ),
        number_part(
            f"Find {name}'s average speed from reaching {start} until reaching {end}, in km/h, to 1 decimal place.",
            speed,
            1,
            2,
            [
                f"Total time = {wait} + {journey} = {total} minutes.",
                f"Speed = {distance} ÷ {total} × 60 = {float(speed):.1f} km/h.",
            ],
            [
                (Fraction(60 * distance, journey), "left-out-fixed-extra"),
                (Fraction(100 * distance, total), "decimal-time-base-100"),
                (Fraction(distance, total), "unconverted-time-unit", 2),
            ],
            follow_through=f"{60 * distance}/(part_a + part_b)",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
