"""Travelling east and west, time-zone difference (senior-solve-problems-relating-to-travelling-east-and-west,
ACMEM167)."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p9b_travel_east_west_build import (
    ROWS,
    EASY_ROWS,
    MEDIUM_ROWS,
    HARD_ROWS,
    _verify_offset_via_inverse,
    _compute,
)

TOPIC = "travel_east_west_zone_difference"
SEEDS = range(40)
DIFFICULTIES = ["easy", "medium", "hard"]


# --- determinism / distractor hygiene ---------------------------------------


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        wrong = q.answer + 5.5  # never a real table offset value we use
        assert not grade_number(q, str(wrong)).correct


# --- answer kind / formatting -------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_never_has_trailing_zero(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert not q.answer_text.endswith(".0"), q.answer_text


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_formatted_alike(difficulty):
    """Every rendered number is either a bare integer or exactly 1 d.p."""
    num_re = re.compile(r"^-?\d+(\.\d)?$")
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        for text in [q.answer_text, *q.distractors]:
            assert num_re.match(text), text


def test_answer_is_only_the_adelaide_perth_row_non_whole():
    """Plausibility: exactly one row's offset is not a whole number."""
    non_whole = [row for row in ROWS if row["offset"].denominator != 1]
    assert len(non_whole) == 1
    assert (non_whole[0]["a"], non_whole[0]["b"]) == ("Adelaide", "Perth")


# --- day-rollover behaviour, per tier -----------------------------------------


def test_easy_and_medium_never_roll_over():
    for difficulty, rows in (("easy", EASY_ROWS), ("medium", MEDIUM_ROWS)):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "on the same day they departed" in q.text


def test_hard_always_rolls_over_and_split_is_reported():
    """Hard tier: day_offset_B is never 0, over 60 seeds."""
    before = 0
    after = 0
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        has_before = "on the day before they departed" in q.text
        has_after = "on the day after they departed" in q.text
        assert has_before or has_after, q.text
        assert not (has_before and has_after)
        if has_before:
            before += 1
        else:
            after += 1
    assert before + after == 60
    # Structural claim from the construction: only the Sydney/Los Angeles row
    # (offset 17h > every realistic duration drawn) can ever produce -1; every
    # other row can only ever produce +1. So `before` should equal exactly the
    # count of Los Angeles seeds among the 60 (never split across other rows).
    assert before > 0, "no -1 seed observed in 60 -- Los Angeles row may not be reachable"
    assert after > 0


# plausibility test -----------------------------


# Lower bound for a "real, experienced" time-zone difference. Half an hour is the smallest
# granularity any real time zone actually uses, so it is the honest lower bound.
MIN_PLAUSIBLE_OFFSET = Fraction(1, 2)


def test_plausibility_offsets_are_real_and_bounded():
    for row in ROWS:
        assert MIN_PLAUSIBLE_OFFSET <= row["offset"] <= 17, row
        assert row["offset"] > 0 and row["offset"] <= 17  # literal wording


def test_plausibility_red_first_then_green():
    """Prove the plausibility assertion is not vacuous: a corrupted table
    (an implausible near-zero 0.02h offset) must fail it; the real table
    must pass."""
    corrupted = [dict(row, offset=Fraction(1, 50)) for row in ROWS]  # 0.02h
    with pytest.raises(AssertionError):
        for row in corrupted:
            assert MIN_PLAUSIBLE_OFFSET <= row["offset"] <= 17, row
    for row in ROWS:
        assert MIN_PLAUSIBLE_OFFSET <= row["offset"] <= 17, row


# --- independent check (inverse derivation) ----------------------------------


def test_inverse_check_fires_on_corrupted_offset():
    # Sydney -> Perth, west, real offset 2h: dep 09:00 (540), duration 120min,
    # arrival 10:00 (600) same day -- signed offset = 600 - 660 = -60?? use
    # a hand-built consistent example instead:
    # dep_A=600 (10:00), duration=120 (12:00 Sydney-equivalent), offset=2h west
    # => arr_B_clock = 600+120-120 = 600 (10:00), same day.
    _verify_offset_via_inverse(600, 120, 600, "the same day", "west", Fraction(2, 1))
    with pytest.raises(VerificationError):
        _verify_offset_via_inverse(600, 120, 600, "the same day", "west", Fraction(3, 1))
    with pytest.raises(VerificationError):
        _verify_offset_via_inverse(600, 120, 600, "the same day", "east", Fraction(2, 1))


def test_compute_day_offset_handles_negative_floor_correctly():
    """Hand-verify Python's floor `%`/`//` on a negative total rather than
    just asserting it works."""
    row = dict(a="Sydney", b="Los Angeles", direction="west", offset=Fraction(17, 1), note="x", duration_range=(1, 1))
    # dep_A=0, duration=60 (1h), offset=17h=1020min (west) -> arr_B_total = 60-1020 = -960
    signed_offset, local_equiv, arr_B_total, arr_B_clock, day_offset_B = _compute(row, 0, 60)
    assert arr_B_total == -960
    assert day_offset_B == -1  # floor(-960/1440) == -1, not 0
    assert arr_B_clock == 480  # -960 % 1440 == 480 (08:00), Python floor mod
    assert -960 == day_offset_B * 1440 + arr_B_clock


# --- skill loading -------------------------------------------------------------


def test_skill_loads_with_expected_fields():
    from mathgen.skills import load_skills

    _, skills = load_skills()
    skills_by_id = {s.id: s for s in skills}
    skill = skills_by_id.get("senior-solve-problems-relating-to-travelling-east-and-west")
    assert skill is not None
    assert skill.prerequisites == []
    assert skill.answer == "number"
    assert skill.placement_eligible is False


def test_at_least_20_distinct_questions_per_tier():
    for difficulty in DIFFICULTIES:
        seen = set()
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            seen.add(q.text)
        assert len(seen) >= 20, f"{difficulty}: only {len(seen)} distinct questions"
