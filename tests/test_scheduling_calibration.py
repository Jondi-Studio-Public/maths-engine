"""Calibration from attempt data (mathgen/scheduling/calibration.py). Pure: rows and
skills are built inline."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from mathgen.scheduling import calibration


def _rows(user, skill, tier, correct, wrong):
    return [{"user_id": user, "skill_id": skill, "difficulty": tier, "correct": 1}] * correct + [
        {"user_id": user, "skill_id": skill, "difficulty": tier, "correct": 0}
    ] * wrong


def _skill(*tiers):
    return SimpleNamespace(sources=[SimpleNamespace(difficulty=t) for t in tiers])


def test_wilson_interval_brackets_the_rate():
    low, high = calibration.wilson_interval(18, 20)
    assert low < 0.9 < high
    assert calibration.wilson_interval(0, 0) == (0.0, 1.0)


def test_rasch_separates_item_difficulty_from_who_attempted_it():
    # Both tiers are 70% right overall, but "hard" was only tried by the strong
    # learner and "easy" only by the weaker one, who scores 50% on a shared tier
    # the strong learner gets 90% on. Rasch should rank "hard" harder.
    rows = (
        _rows("strong", "s", "shared", 18, 2)
        + _rows("weak", "s", "shared", 10, 10)
        + _rows("strong", "s", "hard", 14, 6)
        + _rows("weak", "s", "easy", 14, 6)
    )
    report = calibration.item_difficulty(rows, {"s": _skill("easy", "shared", "hard")}, min_attempts=10)
    b = {i["difficulty"]: i["rasch_difficulty"] for i in report["items"]}
    assert b["hard"] > b["easy"]
    assert sum(b.values()) == pytest.approx(0, abs=2e-3)  # centred, before rounding to 3 places
    assert report["learners"] == 2 and report["attempts"] == 80


def test_skills_are_also_fitted_pooled_over_tiers():
    rows = _rows("a", "easy-skill", "t1", 6, 1) + _rows("a", "easy-skill", "t2", 7, 1)
    rows += _rows("a", "hard-skill", "t1", 3, 5) + _rows("a", "hard-skill", "t2", 2, 6)
    report = calibration.item_difficulty(rows, {}, min_attempts=10)
    assert report["items"] == []  # no single tier has 10 attempts
    b = {s["skill_id"]: s["rasch_difficulty"] for s in report["skills"]}
    assert b["hard-skill"] > b["easy-skill"]


def test_perfect_scores_stay_finite():
    rows = _rows("a", "s", "easy", 25, 0) + _rows("a", "s", "hard", 0, 25)
    report = calibration.item_difficulty(rows, {"s": _skill("easy", "hard")}, min_attempts=1)
    assert all(math.isfinite(i["rasch_difficulty"]) for i in report["items"])


def test_tiers_below_min_attempts_are_left_out():
    rows = _rows("a", "s", "easy", 5, 0)
    assert calibration.item_difficulty(rows, {"s": _skill("easy")}, min_attempts=20)["items"] == []


def test_inverted_tier_order_is_flagged():
    rows = _rows("a", "s", "easy", 10, 30) + _rows("a", "s", "hard", 38, 2)
    report = calibration.item_difficulty(rows, {"s": _skill("easy", "hard")}, min_attempts=10)
    assert report["inversions"] == [
        {"skill_id": "s", "earlier": "easy", "later": "hard", "earlier_p": 0.25, "later_p": 0.95}
    ]


def test_overlapping_tiers_are_not_flagged():
    rows = _rows("a", "s", "easy", 15, 5) + _rows("a", "s", "hard", 16, 4)
    report = calibration.item_difficulty(rows, {"s": _skill("easy", "hard")}, min_attempts=10)
    assert report["inversions"] == []


def test_fsrs_calibration_bins_and_scores():
    rows = [{"retrievability": None, "rating": 3}]  # first review: skipped
    rows += [{"retrievability": 0.9, "rating": 3}] * 9 + [{"retrievability": 0.9, "rating": 1}]
    rows += [{"retrievability": 0.55, "rating": 2}, {"retrievability": 0.55, "rating": 1}]
    report = calibration.fsrs_calibration(rows)
    assert report["reviews"] == 12
    assert [b["reviews"] for b in report["bins"]] == [2, 10]
    assert report["bins"][1]["observed"] == 0.9
    assert report["bins"][0]["observed"] == 0.5
    expected_loss = -(9 * math.log(0.9) + math.log(0.1) + math.log(0.55) + math.log(0.45)) / 12
    assert report["log_loss"] == pytest.approx(expected_loss, abs=1e-4)
    assert report["rmse_bins"] == pytest.approx(math.sqrt(2 * 0.05**2 / 12), abs=1e-4)


def test_fsrs_calibration_with_no_predictions():
    assert calibration.fsrs_calibration([{"retrievability": None, "rating": 3}])["reviews"] == 0
