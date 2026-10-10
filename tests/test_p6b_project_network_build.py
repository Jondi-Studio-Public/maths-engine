"""Project network EST/LST and critical path.

Tests for:
- senior-use-forward-and-backward-scanning-to-determine-the (ACMGM105)
- senior-use-the-critical-path-to-determine-the-minimum (ACMGM107)
"""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p6b_project_network_build import (
    _verify_easy_est_chain,
    _verify_medium_est_merge,
    _verify_hard_lst_nonchain,
    _verify_completion_time_via_sinks,
    _verify_easy_completion_chain,
    _verify_medium_completion_merge,
    _verify_hard_completion_twopath,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

EST_LST_TOPIC = "use_forward_and_backward_scanning_to_determine_the"
COMPLETION_TOPIC = "use_the_critical_path_to_determine_the_minimum"


# --- Determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [EST_LST_TOPIC, COMPLETION_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    """Same (topic, seed, difficulty) should produce identical questions."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [EST_LST_TOPIC, COMPLETION_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    """Distractors must be distinct, non-empty, and never equal the answer."""
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    # After measuring: easy and medium should have >=2 distractors most of the time
    assert seeds_with_at_least_two / len(SEEDS) >= 0.90


@pytest.mark.parametrize("topic", [EST_LST_TOPIC, COMPLETION_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Bare make(topic, seed=s) with no difficulty should have >=2 distractors on >=80% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.80


# --- Real grader accept/reject, both topics --------------------------------


@pytest.mark.parametrize("topic", [EST_LST_TOPIC, COMPLETION_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    """Real grader must accept canonical answer and reject every distractor."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [EST_LST_TOPIC, COMPLETION_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        if isinstance(question.answer, int):
            wrong = question.answer + 1
        else:
            wrong = question.answer + 1.0
        assert not grade_number(question, str(wrong)).correct


# --- EST/LST (ACMGM105) specific checks --------------------------------


def test_easy_est_always_integer():
    """Easy-tier EST answers are always integers (days)."""
    for seed in SEEDS:
        question = make(EST_LST_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_medium_est_always_integer():
    """Medium-tier EST answers are always integers (days)."""
    for seed in SEEDS:
        question = make(EST_LST_TOPIC, seed=seed, difficulty="medium")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_hard_lst_always_integer():
    """Hard-tier LST answers are always integers (days)."""
    for seed in SEEDS:
        question = make(EST_LST_TOPIC, seed=seed, difficulty="hard")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_hard_est_lst_never_mentions_float_or_slack():
    """Hard tier (LST question) must never mention 'float' or 'slack' (scope boundary)."""
    for seed in range(100):
        question = make(EST_LST_TOPIC, seed=seed, difficulty="hard")
        text_lower = question.text.lower()
        assert "float" not in text_lower, f"Found 'float' in hard tier text (seed {seed})"
        assert "slack" not in text_lower, f"Found 'slack' in hard tier text (seed {seed})"
        # Also check distractors don't mention them
        for distractor in question.distractors:
            assert "float" not in distractor.lower()
            assert "slack" not in distractor.lower()


def test_easy_est_independent_check_fires_on_corrupted():
    """Independent check for easy EST should fire on mismatch."""
    # Chain: A(0-2), B(2-6), C(6-10), D(10-15)
    es_ef_correct = {"A": (0, 2), "B": (2, 6), "C": (6, 10), "D": (10, 15)}
    _verify_easy_est_chain(es_ef_correct, 10, "D")
    with pytest.raises(VerificationError):
        _verify_easy_est_chain(es_ef_correct, 11, "D")


def test_medium_est_independent_check_fires_on_corrupted():
    """Independent check for medium EST should fire on mismatch."""
    es_ef_correct = {"E": (10, 15), "C": (3, 8), "D": (6, 10), "B": (1, 6), "A": (0, 1)}
    _verify_medium_est_merge(es_ef_correct, 10, "E")
    with pytest.raises(VerificationError):
        _verify_medium_est_merge(es_ef_correct, 11, "E")


def test_hard_lst_independent_check_fires_on_corrupted():
    """Independent check for hard LST should fire on mismatch."""
    # Topology: A:[], B:[A], C:[A], D:[B], E:[B,C], F:[D,E]
    # Forward: A(0-1), B(1-3), C(1-3), D(3-5), E(3-5), F(5-7)
    es_ef = {"A": (0, 1), "B": (1, 3), "C": (1, 3), "D": (3, 5), "E": (3, 5), "F": (5, 7)}
    # Backward: F(5,7), E(3,5), D(3,5), C(1,3), B(1,3), A(0,1)
    ls_lf = {"A": (0, 1), "B": (1, 3), "C": (1, 3), "D": (3, 5), "E": (3, 5), "F": (5, 7)}
    activities = {
        "A": {"duration": 1, "predecessors": []},
        "B": {"duration": 2, "predecessors": ["A"]},
        "C": {"duration": 2, "predecessors": ["A"]},
        "D": {"duration": 2, "predecessors": ["B"]},
        "E": {"duration": 2, "predecessors": ["B", "C"]},
        "F": {"duration": 2, "predecessors": ["D", "E"]},
    }
    _verify_hard_lst_nonchain(ls_lf, es_ef, activities, 1, "B")
    with pytest.raises(VerificationError):
        _verify_hard_lst_nonchain(ls_lf, es_ef, activities, 2, "B")


# --- Critical Path (ACMGM107) specific checks ----


def test_easy_completion_always_integer():
    """Easy-tier completion time answers are always integers (days)."""
    for seed in SEEDS:
        question = make(COMPLETION_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_medium_completion_always_integer():
    """Medium-tier completion time answers are always integers (days)."""
    for seed in SEEDS:
        question = make(COMPLETION_TOPIC, seed=seed, difficulty="medium")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_hard_completion_always_integer():
    """Hard-tier completion time answers are always integers (days)."""
    for seed in SEEDS:
        question = make(COMPLETION_TOPIC, seed=seed, difficulty="hard")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_easy_completion_independent_check_fires_on_corrupted():
    """Independent check for easy completion should fire on mismatch."""
    es_ef = {"D": (10, 15), "C": (5, 8), "B": (2, 6), "A": (0, 2)}
    _verify_easy_completion_chain(es_ef, 15, 15)
    with pytest.raises(VerificationError):
        _verify_easy_completion_chain(es_ef, 15, 14)


def test_medium_completion_independent_check_fires_on_corrupted():
    """Independent check for medium completion should fire on mismatch."""
    es_ef = {"E": (10, 15), "C": (3, 8), "D": (6, 10), "B": (1, 6), "A": (0, 1)}
    _verify_medium_completion_merge(es_ef, 15, 15)
    with pytest.raises(VerificationError):
        _verify_medium_completion_merge(es_ef, 15, 14)


def test_hard_completion_independent_check_fires_on_corrupted():
    """Independent check for hard completion should fire on mismatch."""
    es_ef = {"F": (15, 25), "E": (10, 15), "D": (8, 12), "C": (3, 8), "B": (1, 6), "A": (0, 1)}
    _verify_hard_completion_twopath(es_ef, 25, 25)
    with pytest.raises(VerificationError):
        _verify_hard_completion_twopath(es_ef, 25, 24)


def test_completion_time_sink_check():
    """Completion time must equal max EF of sink activities."""
    activities = {
        "A": {"duration": 2, "predecessors": []},
        "B": {"duration": 3, "predecessors": ["A"]},
    }
    es_ef = {"A": (0, 2), "B": (2, 5)}
    _verify_completion_time_via_sinks(es_ef, activities, 5)
    with pytest.raises(VerificationError):
        _verify_completion_time_via_sinks(es_ef, activities, 4)


# --- Both skills load correctly through the API -----


def test_both_skills_load_through_api():
    """Both skills should load correctly with correct prerequisites and answer type."""
    domains, skills = load_skills()
    skills_dict = {skill.id: skill for skill in skills}

    est_lst_skill = skills_dict.get("senior-use-forward-and-backward-scanning-to-determine-the")
    assert est_lst_skill is not None
    assert "simple-interest" in est_lst_skill.prerequisites
    assert est_lst_skill.answer == "number"
    assert est_lst_skill.placement_eligible is False
    assert len(est_lst_skill.sources) == 3  # easy, medium, hard

    completion_skill = skills_dict.get("senior-use-the-critical-path-to-determine-the-minimum")
    assert completion_skill is not None
    assert "simple-interest" in completion_skill.prerequisites
    assert completion_skill.answer == "number"
    assert completion_skill.placement_eligible is False
    assert len(completion_skill.sources) == 3  # easy, medium, hard


def test_est_lst_raises_on_unknown_tier():
    """Unknown tier should raise ValueError."""
    with pytest.raises(ValueError):
        make(EST_LST_TOPIC, seed=1, tier="unknown")


def test_completion_raises_on_unknown_tier():
    """Unknown tier should raise ValueError."""
    with pytest.raises(ValueError):
        make(COMPLETION_TOPIC, seed=1, tier="unknown")
