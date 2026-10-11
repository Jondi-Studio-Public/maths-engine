"""Generates `exd_x_meridian_voyage` exam questions: the duration of a voyage along a line of longitude.

The ship sails between two latitudes (degrees and minutes) at a steady speed in knots. The learner combines
two ideas: the latitude separation (added across the equator, subtracted within a hemisphere) and the fact
that one minute of latitude is one nautical mile, then divides by the speed.

Built backwards from the speed and a whole number of hours: the separation in minutes of arc is their
product, and the second latitude is placed that far from the first, across the equator on about half the
draws.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, check_distinct, generator, pick_distractors

_SPEEDS = [8, 10, 12, 14, 15, 16, 18, 20]
_LIMIT = 80 * 60  # latitudes stay within 80 degrees of the equator
_VESSELS = ["A survey ship", "A research vessel", "A cargo ship", "A yacht"]


def _lat_text(minutes: int) -> str:
    degrees, rest = divmod(abs(minutes), 60)
    return f"{degrees}°{rest:02d}′ {'N' if minutes > 0 else 'S'}"


def _hundredths(minutes: int) -> Fraction:
    """The latitude as a learner who reads 12 deg 40 min as 12.40 degrees would take it."""
    degrees, rest = divmod(abs(minutes), 60)
    return (1 if minutes > 0 else -1) * (degrees + Fraction(rest, 100))


def _build(rng: random.Random) -> dict:
    for _ in range(2000):
        speed = rng.choice(_SPEEDS)
        hours = rng.randint(4, 40)
        sep = speed * hours
        a = rng.choice([-1, 1]) * rng.randint(20, _LIMIT)
        b = a + rng.choice([-1, 1]) * sep
        if abs(b) < 20 or abs(b) > _LIMIT or abs(a) < 20:
            continue
        crosses = (a > 0) != (b > 0)
        wrong_hemisphere = abs(abs(a) - abs(b)) if crosses else abs(a) + abs(b)
        misread = abs(_hundredths(a) - _hundredths(b)) * 60
        candidates = {
            "wrong-hemisphere-rule": round(Fraction(wrong_hemisphere, speed)),
            "reads-minutes-as-hundredths-of-a-degree": round(misread / speed),
            "halves-the-arc-constant": round(Fraction(sep, 2 * speed)),
            "converts-nautical-miles-to-kilometres": round(Fraction(sep * 1852, 1000 * speed)),
        }
        if len({hours, *candidates.values()}) == 5:
            break
    else:
        raise AssertionError("no voyage found")

    check(abs(a - b) == sep, "separation mismatch")
    # Second route: arc length from the two latitudes as decimal degrees of arc, at 1 nautical mile per minute.
    check(Fraction(abs(Fraction(a, 60) - Fraction(b, 60)) * 60, speed) == hours, "voyage time mismatch")

    check_distinct(hours, *candidates.values())

    direction = "north" if b > a else "south"
    vessel = rng.choice(_VESSELS)
    text = (
        f"{vessel} sails due {direction} along a line of longitude from latitude {_lat_text(a)} to latitude "
        f"{_lat_text(b)} at a steady speed of {speed} knots. One knot is one nautical mile per hour, and one "
        f"minute of latitude is one nautical mile. How many hours does the voyage take?"
    )
    answer_text = str(hours)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=hours,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Separation: {'add' if crosses else 'subtract'} the latitudes: {sep} minutes of arc = {sep} nautical miles.",
            f"Time = {sep} / {speed} = {hours} hours.",
        ],
        distractors=pick_distractors(answer_text, [str(v) for v in candidates.values()]),
    )


@generator(
    "exd_x_meridian_voyage",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_meridian_voyage(rng: random.Random, tier: str = "exam") -> dict:
    """Voyage time along a meridian from two latitudes and a speed in knots. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
