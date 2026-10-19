"""Exam chain for converting between 12-hour and 24-hour time (``convert-time-systems``): a late-evening departure,
the time left before midnight, the arrival after midnight, and the arrival in 12-hour time with its day.

Times are written the way the skill's practice tier writes them: ``21:35`` in 24-hour time and ``9:35 pm`` in
12-hour time, as ``mcq`` options. Set the way a school exam sets it:

(a) the departure in 24-hour time                                  mcq,    1 mark
(b) the minutes from the departure to midnight                     number, 1 mark
(c) the minutes after midnight at which the trip ends              number, 2 marks, follow-through (b)
(d) the arrival in 12-hour time, with the day                      mcq,    2 marks

## Construction

The departure is between 6 pm and 11:55 pm, never on the hour, and the trip is long enough to cross midnight, so
the arrival is on the next day. The answers are checked a second way with ``datetime``: the departure plus the
duration gives the arrival's clock time and weekday.

## Distractors, each one error of its own part

* (a): the pm hour left unconverted, the hour before, the hour after.
* (b): minutes counted as hundredths of an hour, the minutes past the hour not allowed for, the minutes past the
  hour used as the minutes left.
* (c): the time to midnight added to the trip instead of taken from it, the trip length used whole, counted back
  from midnight.
* (d): am and pm swapped, the day not advanced, the hour one out.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_w12a_kit import mcq_part, number_part

TOPIC = "exd_x_w12a_time_crossing_midnight_chain"

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

_TRIPS = (
    "An overnight coach leaves the city terminal at {dep} on {day}.",
    "A night train leaves the central station at {dep} on {day}.",
    "A freight truck leaves the depot at {dep} on {day}.",
)
_DUR = "The trip takes {dur}."
_ASK_A = (
    "Write the departure time in 24-hour time.",
    "What is the departure time in 24-hour time?",
)
_ASK_B = (
    "How many minutes are there from the departure time until midnight?",
    "How many minutes after departure does midnight come?",
)
_ASK_C = (
    "How many minutes after midnight does the trip end?",
    "The trip ends after midnight. How many minutes after midnight is that?",
)
_ASK_D = (
    "Write the arrival time in 12-hour time, and give the day.",
    "What is the arrival time in 12-hour time, and on which day?",
)


def clock12(total_minutes: int) -> str:
    """``9:35 pm`` for a time given as minutes since midnight (wrapping around the day)."""
    total_minutes %= 1440
    hour = total_minutes // 60
    return f"{hour % 12 or 12}:{total_minutes % 60:02d} {'am' if hour < 12 else 'pm'}"


def clock24(total_minutes: int) -> str:
    total_minutes %= 1440
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def _duration_text(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    hours = f"{h} hour{'s' if h != 1 else ''}"
    return hours if m == 0 else f"{hours} {m} minutes"


@generator(TOPIC, level=6, presets={"exam": {"tier": "exam"}})
def exd_x_w12a_time_crossing_midnight_chain(rng, tier: str = "exam"):
    """A late departure converted to 24-hour time, a trip across midnight, the arrival in 12-hour time and its day."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a trip across midnight with three distractors per part"):
        hour = rng.randint(18, 23)
        minute = rng.choice((5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55))
        dep = hour * 60 + minute
        to_midnight = 1440 - dep
        # A trip that crosses midnight and takes at most seventeen hours.
        duration = rng.randrange(to_midnight + 15, 17 * 60 + 1, 5)
        after = duration - to_midnight
        day = rng.randrange(7)
        arrive_day = _DAYS[(day + 1) % 7]

        # A second way: add the trip to a real date and read off the weekday and clock time.
        start = datetime(2026, 3, 2 + day, hour, minute)  # 2 March 2026 is a Monday
        end = start + timedelta(minutes=duration)
        check(end.strftime("%A") == arrive_day, "the arrival day disagrees with datetime")
        check(end.hour * 60 + end.minute == after, "the arrival clock time disagrees with datetime")

        part_a = mcq_part(
            phrase(rng, _ASK_A),
            clock24(dep),
            1,
            [f"{clock12(dep)} is a pm time, so add 12 to the hour: {clock24(dep)}."],
            [
                (clock24(dep - 12 * 60), "time-pm-not-converted"),
                (clock24(dep - 60), "time-hour-off-by-one"),
                (clock24(dep + 60 if hour < 23 else dep - 120), "time-hour-off-by-one"),
            ],
        )
        part_b = number_part(
            phrase(rng, _ASK_B),
            to_midnight,
            1,
            [
                f"From {clock24(dep)} to 24:00 is {23 - hour} hour{'s' if 23 - hour != 1 else ''} "
                f"and {60 - minute} minute{'s' if 60 - minute != 1 else ''}.",
                f"{23 - hour} × 60 + {60 - minute} = {to_midnight} minutes.",
            ],
            [
                ((24 - hour) * 60 + (60 - minute), "time-no-minutes-carry"),
                ((23 - hour) * 60 + (100 - minute), "time-base-ten-clock"),
                ((23 - hour) * 60 + minute, "time-minutes-not-complemented"),
                ((24 - hour) * 60, "time-no-minutes-carry"),
            ],
        )
        part_c = number_part(
            phrase(rng, _ASK_C),
            after,
            2,
            [
                f"The trip takes {duration} minutes, and {to_midnight} of them are before midnight.",
                f"{duration} - {to_midnight} = {after} minutes after midnight.",
            ],
            [
                (duration + to_midnight, "time-gap-added-to-duration"),
                (duration, "time-midnight-ignored"),
                (1440 - after, "time-counted-back-from-midnight"),
                (to_midnight - after if to_midnight > after else None, "time-gap-added-to-duration"),
            ],
            follow=f"{duration}-part_b",
        )
        key = f"{clock12(after)} on {arrive_day}"
        flipped = clock12(after + 720)
        near = clock12(after + 60 if after >= 60 else after + 120)
        part_d = mcq_part(
            phrase(rng, _ASK_D),
            key,
            2,
            [
                f"{after} minutes after midnight is {clock24(after)}, which is {clock12(after)}.",
                f"The trip crosses midnight, so the arrival is on {arrive_day}.",
            ],
            [
                (f"{flipped} on {arrive_day}", "time-am-pm-swapped"),
                (f"{clock12(after)} on {_DAYS[day]}", "time-day-not-advanced"),
                (f"{near} on {arrive_day}", "time-clock-hour-off-by-one"),
            ],
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        stem = " ".join(
            [
                phrase(rng, _TRIPS, dep=clock12(dep), day=_DAYS[day]),
                _DUR.format(dur=_duration_text(duration)),
            ]
        )
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
