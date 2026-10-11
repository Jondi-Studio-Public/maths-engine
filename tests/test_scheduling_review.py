"""A finished review scored with FSRS (mathgen/scheduling/review.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mathgen.scheduling import fsrs, review

NOW = datetime(2026, 10, 1, 9, 0, 0, tzinfo=timezone.utc)
DELAYS = [1, 3, 7, 14, 30, 60, 120]


def test_outcome_maps_to_rating():
    assert review.OUTCOME_RATING == {
        "reset": fsrs.AGAIN,
        "hold": fsrs.HARD,
        "advance": fsrs.GOOD,
        "maintenance_pass": fsrs.GOOD,
    }


def test_memory_from_history_replays_the_log():
    log = [
        {"outcome": "advance", "created_at": NOW},
        {"outcome": "advance", "created_at": NOW + timedelta(days=3)},
        {"outcome": "hold", "created_at": NOW + timedelta(days=11)},
        {"outcome": "not-a-review", "created_at": NOW + timedelta(days=12)},
    ]
    memory, last = review.memory_from_history(log, stage=2, stage_delays=DELAYS)
    assert last == NOW + timedelta(days=11)
    # Good, Good after 3 days, Hard after 8: the third row of the golden sequence
    # in tests/test_scheduling_fsrs.py.
    assert memory.stability == pytest.approx(14.868805, abs=1e-5)
    assert memory.difficulty == pytest.approx(6.027056, abs=1e-5)


def test_memory_from_history_seeds_from_the_stage_delay_without_a_log():
    memory, last = review.memory_from_history([], stage=4, stage_delays=DELAYS)
    assert last is None
    assert memory.stability == 14.0
    assert review.memory_from_history([], stage=7, stage_delays=DELAYS)[0].stability == 120.0
    assert review.memory_from_history([], stage=0, stage_delays=DELAYS) == (None, None)


def test_good_review_is_due_at_the_desired_retention():
    memory = fsrs.Memory(stability=10.0, difficulty=5.0)
    scored = review.score("advance", memory, NOW - timedelta(days=10), NOW)
    assert scored["rating"] == fsrs.GOOD
    assert scored["retrievability"] == pytest.approx(0.9)
    assert scored["interval_days"] == round(scored["memory"].stability)
    assert scored["due_at"] == NOW + timedelta(days=scored["interval_days"])


def test_interval_is_clamped_to_one_day_and_the_maximum():
    big = fsrs.Memory(stability=900.0, difficulty=2.0)
    assert (
        review.score("advance", big, NOW - timedelta(days=900), NOW, maximum_interval_days=200)["interval_days"] == 200
    )
    tiny = fsrs.Memory(stability=0.05, difficulty=9.0)
    assert review.score("hold", tiny, NOW - timedelta(days=1), NOW)["interval_days"] == 1


def test_higher_retention_means_sooner():
    memory = fsrs.Memory(stability=30.0, difficulty=5.0)
    loose = review.score("advance", memory, NOW - timedelta(days=30), NOW, desired_retention=0.8)
    tight = review.score("advance", memory, NOW - timedelta(days=30), NOW, desired_retention=0.95)
    assert tight["interval_days"] < loose["interval_days"]


def test_reset_is_due_now():
    memory = fsrs.Memory(stability=10.0, difficulty=5.0)
    scored = review.score("reset", memory, NOW - timedelta(days=10), NOW)
    assert scored["rating"] == fsrs.AGAIN
    assert scored["due_at"] == NOW
    assert scored["interval_days"] == 0


def test_first_review_has_no_prediction():
    scored = review.score("advance", None, None, NOW)
    assert scored["retrievability"] is None
    assert scored["elapsed_days"] == 0.0
