"""Tests `p19f_instantaneous_rate`: grading, bare-integer answers and distractors.

The derivative as instantaneous rate.

senior-interpret-the-derivative-as-the-instantaneous-rate-of (ACMMM084,
topic p19f_instantaneous_rate, answer: number). See the generator module
docstring for the neighbour-boundary tests.
"""

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p19f_instantaneous_rate"
SKILL = "senior-interpret-the-derivative-as-the-instantaneous-rate-of"


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


# Contextual tiers never state a negative time --------


def test_hard_and_exam_never_use_a_negative_time_200_seeds():
    import re

    for seed in range(200):
        q_hard = make(TOPIC, seed=seed, difficulty="hard")
        m = re.search(r"t = (-?\d+)\.", q_hard.text) or re.search(r"t = (-?\d+)", q_hard.text)
        assert m is not None, f"hard seed={seed}: could not find 't = <value>' in {q_hard.text!r}"
        assert int(m.group(1)) >= 1, f"hard seed={seed}: t = {m.group(1)} is not >= 1"

        q_exam = make(TOPIC, seed=seed, difficulty="exam")
        m2 = re.search(r"after t = (-?\d+)", q_exam.text)
        assert m2 is not None, f"exam seed={seed}: could not find the base t in {q_exam.text!r}"
        assert int(m2.group(1)) >= 0, f"exam seed={seed}: base t = {m2.group(1)} is not >= 0"


# --- Distinct rendered questions -------------------------------------------


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    from mathgen.core import VerificationError, check_equivalent

    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


# --- Never a bare symbolic expression, never solved for x (neighbour bound) --


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_a_number_never_an_expression(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int)


@pytest.mark.parametrize("tier", ("hard", "exam"))
def test_applied_tiers_wrap_the_cubic_as_q_of_t(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "Q(t)" in q.text, f"seed={seed}/{tier}: no applied-quantity wrapping in {q.text!r}"


def test_exam_tier_states_a_derived_evaluation_point():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert any(f"{unit} after t =" in q.text for unit in ("minutes", "seconds", "hours")), (
            f"seed={seed}: exam tier has no derived point in {q.text!r}"
        )


def test_exam_tier_offset_unit_matches_the_context_unit():
    """The offset must be stated in the SAME unit as the context's own t -- "5 minutes after t = 4" is incoherent
    when t is measured in hours or seconds."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        if "tank" in q.text:
            assert "minutes after t =" in q.text
            assert "seconds after t =" not in q.text and "hours after t =" not in q.text
        elif "drone" in q.text:
            assert "seconds after t =" in q.text
            assert "minutes after t =" not in q.text and "hours after t =" not in q.text
        elif "bacteria" in q.text:
            assert "hours after t =" in q.text
            assert "minutes after t =" not in q.text and "seconds after t =" not in q.text
        else:
            raise AssertionError(f"seed={seed}: unrecognised context in {q.text!r}")


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["differentiate-polynomial"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        if source.difficulty != "exam":  # the exam tier is a multi-part chain on its own topic
            assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
