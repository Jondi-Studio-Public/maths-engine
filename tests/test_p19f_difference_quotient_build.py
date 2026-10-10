"""Tests `p19f_difference_quotient_stated_h`: grading, bare-integer answers and distractors.

The stated-h difference quotient.

senior-interpret-the-difference-quotient-frac-f-left-x (ACMMM077, topic
p19f_difference_quotient_stated_h, answer: number). See the generator
module docstring for the neighbour-boundary tests.
"""

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p19f_difference_quotient_stated_h"
SKILL = "senior-interpret-the-difference-quotient-frac-f-left-x"


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_and_rejects(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed={seed}: distractor {d!r} accepted"
        wrong = str(q.answer + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Answer kind -----------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_bare_integer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int)
        assert "." not in q.answer_text, f"seed={seed}: answer_text {q.answer_text!r} has a decimal point"


# --- Formatted alike --------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_all_bare_integers(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert d.lstrip("-").isdigit(), f"seed={seed}: {d!r} not shaped like the integer answer"


# --- Distinct rendered questions -------------------------------------------


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    from mathgen.core import VerificationError, check_equivalent

    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


# --- Never a limit, never a sequence (the neighbour boundary) --------------


@pytest.mark.parametrize("tier", TIERS)
def test_never_shows_a_sequence_of_h_or_the_word_limit(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "0.1" not in q.text and "0.01" not in q.text, (
            f"seed={seed}: text shows a shrinking-h sequence, the ACMMM081 neighbour's shape"
        )
        assert "approaches" not in q.text.lower()


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["function-notation"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
