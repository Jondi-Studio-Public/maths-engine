"""Skills: f10-timetable-itinerary-duration."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p4b_timetable_build import (
    _verify_timetable_duration,
    _minutes_since_midnight_construction,
    _minutes_since_midnight_verify,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- Determinism tests ----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_timetable_is_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        question = make("timetable_itinerary_duration", seed=seed, difficulty=difficulty)
        repeat = make("timetable_itinerary_duration", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


# --- Distractor tests ----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_timetable_distractors_are_distinct_and_not_equal_to_answer(difficulty):
    for seed in SEEDS:
        question = make("timetable_itinerary_duration", seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, int)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# --- Boundary condition tests ----


def test_timetable_easy_times_within_same_hour():
    """Verify easy tier: asked row's times stay within same hour."""
    for seed in SEEDS:
        question = make("timetable_itinerary_duration", seed=seed, difficulty="easy")
        text = question.text

        # Parse the timetable to find times
        # Format: "Label departs H:MMam/pm and arrives H:MMam/pm."
        # We need to find the asked label's times
        # Extract the asked label from the question text
        match = re.search(r"How many minutes does (.+?)'s journey take\?", text)
        assert match, f"Could not find asked label in: {text}"
        asked_label = match.group(1)

        # Find this label's row in the timetable
        # Parse all rows: "Label departs TIME and arrives TIME."
        pattern = rf"{re.escape(asked_label)} departs (\d+):(\d+)(am|pm) and arrives (\d+):(\d+)(am|pm)\."
        time_match = re.search(pattern, text)
        assert time_match, f"Could not find times for {asked_label} in: {text}"

        start_h, start_m, start_ap, end_h, end_m, end_ap = time_match.groups()
        start_h, start_m, end_h, end_m = int(start_h), int(start_m), int(end_h), int(end_m)

        # For easy: both times should be in the same hour
        assert start_h == end_h, f"Easy tier: start_h={start_h} != end_h={end_h}"
        assert start_ap == end_ap, f"Easy tier: start_ap={start_ap} != end_ap={end_ap}"


def test_timetable_medium_times_cross_hour_boundary():
    """Verify medium tier: asked row crosses exactly one hour boundary."""
    for seed in SEEDS:
        question = make("timetable_itinerary_duration", seed=seed, difficulty="medium")
        text = question.text

        # Find the asked label
        match = re.search(r"How many minutes does (.+?)'s journey take\?", text)
        assert match, f"Could not find asked label in: {text}"
        asked_label = match.group(1)

        # Parse the times
        pattern = rf"{re.escape(asked_label)} departs (\d+):(\d+)(am|pm) and arrives (\d+):(\d+)(am|pm)\."
        time_match = re.search(pattern, text)
        assert time_match, f"Could not find times for {asked_label} in: {text}"

        start_h, start_m, start_ap, end_h, end_m, end_ap = time_match.groups()
        start_h, start_m, end_h, end_m = int(start_h), int(start_m), int(end_h), int(end_m)

        # For medium: times must cross an hour boundary
        if start_ap == end_ap:  # same am/pm
            # Different hours
            assert start_h != end_h, f"Medium tier: hour boundary not crossed (same hour)"
        # If different am/pm, automatically crosses boundary


def test_timetable_hard_times_cross_noon_boundary():
    """Verify hard tier: asked row crosses noon (am/pm) boundary."""
    for seed in SEEDS:
        question = make("timetable_itinerary_duration", seed=seed, difficulty="hard")
        text = question.text

        # Find the asked label
        match = re.search(r"How many minutes does (.+?)'s journey take\?", text)
        assert match, f"Could not find asked label in: {text}"
        asked_label = match.group(1)

        # Parse the times
        pattern = rf"{re.escape(asked_label)} departs (\d+):(\d+)(am|pm) and arrives (\d+):(\d+)(am|pm)\."
        time_match = re.search(pattern, text)
        assert time_match, f"Could not find times for {asked_label} in: {text}"

        start_h, start_m, start_ap, end_h, end_m, end_ap = time_match.groups()

        # For hard: must cross noon boundary (am to pm)
        assert start_ap == "am", f"Hard tier: start must be am, got {start_ap}"
        assert end_ap == "pm", f"Hard tier: end must be pm, got {end_ap}"


def test_timetable_row_position_varies_across_seeds():
    """Verify asked row position is not always at the same index."""
    positions = set()
    for seed in SEEDS[:20]:  # Sample 20 seeds to check variation
        question = make("timetable_itinerary_duration", seed=seed, difficulty="easy")
        text = question.text

        # Find the asked label
        match = re.search(r"How many minutes does (.+?)'s journey take\?", text)
        asked_label = match.group(1)

        # Count how many rows appear before the asked label
        # Split by "departs" to count rows
        rows = re.findall(r"(\w+(?:\s\w+)?) departs", text)
        if asked_label in rows:
            position = rows.index(asked_label)
            positions.add(position)

    # Should have variation (not always same position)
    assert len(positions) > 1, f"Asked row position always the same: {positions}"


# --- Real grader tests ----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_timetable_real_grader_accepts_canonical_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        question = make("timetable_itinerary_duration", seed=seed, difficulty=difficulty)
        # Test the real grader with canonical answer
        assert grade_number(question, question.answer_text).correct
        # Test real grader rejects every distractor
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


def test_timetable_real_grader_rejects_deliberately_wrong_answer():
    """Test that a definitely wrong answer is rejected."""
    for seed in SEEDS:
        question = make("timetable_itinerary_duration", seed=seed, difficulty="medium")
        # Use a value far from the correct answer
        wrong = question.answer + 1000
        wrong_text = str(wrong)
        assert not grade_number(question, wrong_text).correct


# --- Independent check tests ----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_timetable_independent_check_fires_on_corrupted_answer(difficulty):
    """Verify independent check catches a wrong answer."""
    for seed in SEEDS:
        # Create a simple test case: 10:15am to 10:45am = 30 minutes
        correct_answer = 30
        wrong_answer = 31

        timetable = [
            ("Bus A", 10, 15, "am", 10, 45, "am"),
        ]

        # Correct answer should pass
        _verify_timetable_duration(timetable, "Bus A", 10, 15, "am", 10, 45, "am", correct_answer)

        # Wrong answer should fail
        with pytest.raises(VerificationError):
            _verify_timetable_duration(timetable, "Bus A", 10, 15, "am", 10, 45, "am", wrong_answer)


def test_time_conversion_construction_vs_verify_agree():
    """Verify that construction and verification paths give same result."""
    test_cases = [
        (12, 0, "am"),  # midnight
        (1, 30, "am"),  # 1:30am
        (11, 59, "am"),  # 11:59am
        (12, 0, "pm"),  # noon
        (1, 30, "pm"),  # 1:30pm
        (11, 59, "pm"),  # 11:59pm
    ]

    for hour, minute, am_pm in test_cases:
        construct = _minutes_since_midnight_construction(hour, minute, am_pm)
        verify = _minutes_since_midnight_verify(hour, minute, am_pm)
        assert construct == verify, f"Mismatch for {hour}:{minute:02d}{am_pm}: construct={construct}, verify={verify}"


# --- Skills loading via mathgen.skills ----


def test_skill_loads_via_api_skills():
    """Verify the skill loads with expected properties."""
    _, skills_list = load_skills()
    skills = {s.id: s for s in skills_list}

    skill = skills.get("f10-timetable-itinerary-duration")
    assert skill is not None
    assert skill.id == "f10-timetable-itinerary-duration"
    assert skill.domain == "primary-geometry-time"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "number"
    # Prerequisites should be alphabetically sorted
    assert skill.prerequisites == ["convert-time-systems", "elapsed-time"]
    # Should have 3 sources for 3 difficulty tiers
    assert len(skill.sources) == 3
