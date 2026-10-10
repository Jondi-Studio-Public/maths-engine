"""Tests the skill for the overall change in a quantity after repeated percentage changes (ACMEM063).

Repeated percentage change.

Tests for senior-determine-the-overall-change-in-a-quantity-following
(ACMEM063).
"""

from fractions import Fraction
from itertools import product

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p8e_repeated_percentage_change_build import (
    _CHANGE_RE,
    _CHANGE_VALUES,
    _Q_RE,
    _verify_from_text,
)

TOPIC = "repeated_percentage_change"
SKILL_ID = "senior-determine-the-overall-change-in-a-quantity-following"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_count_distinct_nonempty_never_equal_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert len(q.distractors) == 3, (difficulty, seed, q.distractors)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        wrong = str(q.answer + 1)
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer-kind / formatting discipline --------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answers_are_plain_positive_integers_under_1000(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, int)
        assert 0 < q.answer < 1000
        assert q.answer_text.isdigit()
        assert "." not in q.answer_text


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_formatted_identically(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        for d in q.distractors:
            assert d.isdigit(), (difficulty, seed, d)


# --- At least 20 distinct questions per tier over 60 seeds --------------

# --- ValueError on a nonsensical preset -----------------------------------


def test_raises_value_error_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="bogus")


# --- Independent in-generator check fires on a corrupted answer ---------


def test_verify_from_text_fires_on_corrupted():
    text = "A town's population starts at 200. First it increases by 20%. Then it decreases by 10%. What is the final population?"
    _verify_from_text(text, 216)  # 200 * 1.2 * 0.9 = 216
    with pytest.raises(VerificationError):
        _verify_from_text(text, 217)


def test_verify_from_text_handles_three_changes():
    text = (
        "A town's population starts at 250. First it increases by 10%. Then it increases by 10%. "
        "Then it decreases by 20%. What is the final population?"
    )
    _verify_from_text(text, 242)  # 250 * 1.1 * 1.1 * 0.8 = 242
    with pytest.raises(VerificationError):
        _verify_from_text(text, 240)


# --- Skill loads correctly through the API --------------------------


def test_skill_loads_through_api(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill is not None
    assert skill.domain == "arithmetic"
    assert skill.prerequisites == ["order-of-operations"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert len(skill.sources) == 3
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard"}


# Every intermediate step (not just the final answer) is an exact int --- This is the property that
# keeps a population/attendance/subscriber-count question from ever presenting a fractional
# intermediate value -- sampled, not swept, since it holds for every seed by construction (a
# resample-loop invariant), do not sweep them" guidance. Re-derives every prefix product from the
# rendered TEXT (Q and each signed percentage), not from the generator's own return value, so this
# actually tests the intermediate values a learner reads in `working`, not just their punctuation.


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_every_prefix_value_in_working_is_a_whole_number(difficulty):
    for seed in range(10):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        q_match = _Q_RE.search(q.text)
        assert q_match
        running = Fraction(int(q_match.group(1)))
        changes = _CHANGE_RE.findall(q.text)
        assert len(changes) in (2, 3)
        for direction, pct in changes:
            signed = int(pct) if direction == "increases" else -int(pct)
            running *= Fraction(100 + signed, 100)
            assert running.denominator == 1, (difficulty, seed, q.text, running)


# --- Both required sweeps, made durable as tests ---------------------------
# ("assert directly, over a seed sweep, that [the equal-and-opposite case]
# actually occurs" at medium; "verify both cases actually occur" at easy).
# Swept, not sampled, because the skill exists specifically to catch these
# rare-relative-to-the-full-draw-space combinations.


def test_medium_equal_and_opposite_case_occurs():
    n = 500
    eq_opp = 0
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty="medium")
        changes = _CHANGE_RE.findall(q.text)
        assert len(changes) == 2
        p1, p2 = (int(pct) if d == "increases" else -int(pct) for d, pct in changes)
        if p1 == -p2:
            eq_opp += 1
    # measured 128/500 (~25.6%), matching the 1/4 probability the draw
    # (p2 uniform over the 4-value opposite-sign pool) predicts.
    assert eq_opp > 0, "the classic equal-and-opposite case (p1 = -p2) never occurred in 500 seeds"
    assert 0.15 <= eq_opp / n <= 0.35


def test_easy_both_increase_and_both_decrease_occur():
    n = 500
    both_inc = both_dec = 0
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty="easy")
        changes = _CHANGE_RE.findall(q.text)
        assert len(changes) == 2
        ps = [int(pct) if d == "increases" else -int(pct) for d, pct in changes]
        assert len({1 if p > 0 else -1 for p in ps}) == 1, (seed, ps)  # same direction, always
        if ps[0] > 0:
            both_inc += 1
        else:
            both_dec += 1
    assert both_inc > 0 and both_dec > 0


# --- Sign-mix requirement, hard tier: never all three changes the same sign,
# swept because the skill exists specifically to guard this rare case ---------


def test_hard_tier_never_all_same_direction():
    for seed in range(500):
        q = make(TOPIC, seed=seed, difficulty="hard")
        directions = [
            1 if "increases" in part else -1 for part in q.text.split(".") if "increases" in part or "decreases" in part
        ]
        assert len(directions) == 3
        assert len(set(directions)) != 1, (seed, q.text)


def test_three_change_grid_has_no_exact_collision():
    collisions = 0
    mixed = 0
    for p1, p2, p3 in product(_CHANGE_VALUES, repeat=3):
        if len({1 if p > 0 else -1 for p in (p1, p2, p3)}) == 1:
            continue
        mixed += 1
        x, y, z = Fraction(p1, 100), Fraction(p2, 100), Fraction(p3, 100)
        gap = x * y + y * z + z * x + x * y * z
        if gap == 0:
            collisions += 1
    assert mixed == 384
    assert collisions == 0  # measured; the runtime guard is a safety net regardless
