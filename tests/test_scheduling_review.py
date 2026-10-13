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


def test_memory_from_history_replays_out_of_order_rows_oldest_first():
    log = [
        {"outcome": "hold", "created_at": NOW + timedelta(days=11)},
        {"outcome": "advance", "created_at": NOW},
        {"outcome": "advance", "created_at": NOW + timedelta(days=3)},
    ]
    memory, last = review.memory_from_history(log, stage=2, stage_delays=DELAYS)
    assert last == NOW + timedelta(days=11)
    assert memory.stability == pytest.approx(14.868805, abs=1e-5)


def test_a_zero_stage_delay_seeds_nothing():
    assert review.memory_from_history([], stage=1, stage_delays=[0, 3]) == (None, None)


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


def test_reset_is_due_a_day_later_by_default():
    memory = fsrs.Memory(stability=10.0, difficulty=5.0)
    scored = review.score("reset", memory, NOW - timedelta(days=10), NOW)
    assert scored["rating"] == fsrs.AGAIN
    assert scored["due_at"] == NOW + timedelta(days=1)
    assert scored["interval_days"] == 1


def test_a_zero_reset_gap_brings_a_reset_straight_back():
    memory = fsrs.Memory(stability=10.0, difficulty=5.0)
    scored = review.score("reset", memory, NOW - timedelta(days=10), NOW, reset_gap_days=0)
    assert scored["due_at"] == NOW
    assert scored["interval_days"] == 0


def test_negative_reset_gap_is_rejected():
    with pytest.raises(ValueError):
        review.score("reset", None, None, NOW, reset_gap_days=-1)


def test_a_hold_never_lengthens_the_gap():
    # A well-learned skill held after a 10-day gap: FSRS alone would push it past 10 days.
    memory = fsrs.Memory(stability=40.0, difficulty=3.0)
    last = NOW - timedelta(days=10, hours=6)
    unclamped = fsrs.interval_days(fsrs.step(memory, fsrs.HARD, 10)[0].stability, 0.9)
    assert round(unclamped) > 10
    scored = review.score("hold", memory, last, NOW)
    assert scored["interval_days"] == 10
    assert scored["due_at"] == NOW + timedelta(days=10)
    # A clean review on the same history still lengthens it.
    assert review.score("advance", memory, last, NOW)["interval_days"] > 10


def test_a_hold_with_a_short_fsrs_interval_keeps_it():
    memory = fsrs.Memory(stability=2.0, difficulty=8.0)
    scored = review.score("hold", memory, NOW - timedelta(days=20), NOW)
    raw = fsrs.interval_days(scored["memory"].stability, 0.9)
    assert scored["interval_days"] == max(1, round(raw)) < 20


@pytest.mark.parametrize(
    ("results", "outcome"),
    [
        ([(1, 1), (1, 1), (1, 1)], "advance"),
        ([(0, 1), (1, 1), (1, 1)], "hold"),
        ([(0, 1), (0, 1), (1, 1)], "reset"),
        ([(3, 4), (1, 1)], "hold"),  # three quarters: half a miss
        ([(2, 4), (2, 4)], "hold"),  # two half-right answers add up to one miss
        ([(2, 4), (2, 4), (2, 4), (2, 4)], "reset"),  # four of them add up to two
        ([(1, 4), (1, 1)], "hold"),  # under half: a whole miss
        ([(1, 4), (2, 4)], "hold"),  # one and a half misses
        ([(1, 4), (1, 4)], "reset"),
    ],
)
def test_outcome_from_marks(results, outcome):
    assert review.outcome_from_marks(results) == outcome


def test_outcome_from_marks_in_maintenance():
    assert review.outcome_from_marks([(1, 1), (2, 2)], maintenance=True) == "maintenance_pass"
    assert review.outcome_from_marks([(0, 1)], maintenance=True) == "hold"


def test_outcome_from_marks_reads_grade_results():
    from fractions import Fraction

    from mathgen.grading.typed import GradeResult

    right = GradeResult(correct=True, canonical="3")
    wrong = GradeResult(correct=False, canonical="3")
    part_marked = GradeResult(correct=False, canonical="", marks=Fraction(3), max_marks=5)
    assert review.outcome_from_marks([right, right]) == "advance"
    assert review.outcome_from_marks([right, wrong]) == "hold"
    assert review.outcome_from_marks([part_marked, part_marked]) == "hold"
    assert review.outcome_from_marks([wrong, wrong]) == "reset"


def test_outcome_from_marks_with_a_higher_reset_threshold():
    assert review.outcome_from_marks([(0, 1), (0, 1)], reset_misses=3) == "hold"
    assert review.outcome_from_marks([(0, 1), (0, 1), (0, 1)], reset_misses=3) == "reset"


@pytest.mark.parametrize("bad", [[], [(2, 1)], [(-1, 1)], [(0, 0)]])
def test_outcome_from_marks_rejects_bad_marks(bad):
    with pytest.raises(ValueError):
        review.outcome_from_marks(bad)


def test_first_review_has_no_prediction():
    scored = review.score("advance", None, None, NOW)
    assert scored["retrievability"] is None
    assert scored["elapsed_days"] == 0.0
