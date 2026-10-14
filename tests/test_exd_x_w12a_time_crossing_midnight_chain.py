"""Tests the exam chain for 12-hour and 24-hour time: a late departure, a trip across midnight, the arrival and its day."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pytest

from mathgen import make
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_w12a_time_crossing_midnight_chain"
SKILL_ID = "convert-time-systems"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def oracle(q):
    """The stem's departure and trip length, run through ``datetime``: (24-hour departure, minutes to midnight,
    minutes after midnight at arrival, 12-hour arrival with its day)."""
    h, m, ampm, day = re.search(r"at (\d+):(\d\d) (am|pm) on (\w+)\.", q.text).groups()
    hours = int(re.search(r"takes (\d+) hours?", q.text).group(1))
    extra = re.search(r"hours? (\d+) minutes", q.text)
    trip = timedelta(hours=hours, minutes=int(extra.group(1)) if extra else 0)
    start = datetime(2026, 3, 2 + DAYS.index(day), int(h) % 12 + (12 if ampm == "pm" else 0), int(m))
    end = start + trip
    midnight = datetime(start.year, start.month, start.day) + timedelta(days=1)
    arrival = end.strftime("%-I:%M %p").lower() + " on " + end.strftime("%A")
    return (
        start.strftime("%H:%M"),
        int((midnight - start).total_seconds() // 60),
        int((end - midnight).total_seconds() // 60),
        arrival,
    )


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_prompt_variety_and_new_shapes_over_the_practice_tier():
    check_shapes(TOPIC, "time_system_conversion")


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert q.figure is None
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["mcq", "number", "number", "mcq"]
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, False]
        assert all(len(p.distractors) == 3 for p in q.parts)


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        dep, to_midnight, after, arrival = oracle(q)
        a, b, c, d = canonical(q).values()
        if (a, int(b), int(c), d) != (dep, to_midnight, after, arrival):
            bad.append((q.text, canonical(q)))
    assert bad == [], bad[:2]


def test_every_trip_crosses_midnight_and_covers_the_edge_hours(questions):
    arrivals = set()
    for q in questions[:300]:
        _, to_midnight, after, arrival = oracle(q)
        assert 0 < to_midnight < 360 and 15 <= after <= 17 * 60
        arrivals.add(arrival.split()[1])
        assert re.fullmatch(r"\d{1,2}:\d\d (am|pm) on \w+", arrival)
    assert arrivals == {"am", "pm"}


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_minutes_to_midnight():
    """A minutes-to-midnight reply that is too big takes the same off the minutes after midnight."""

    def carried(q):
        _, to_midnight, after, _ = oracle(q)
        # A reply that merely repeats the earlier one earns no follow-through, so keep the two apart.
        shift = next(d for d in range(5, 60, 5) if after - d > 0 and after - d != to_midnight + d)
        return "b", str(to_midnight + shift), {"c": str(after - shift)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")
