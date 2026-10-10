"""Tests `volume_of_pyramids_and_spheres` (`senior-use-formulas-to-find-the-volume-of-pyramids`).

Senior volume of pyramids and spheres (senior-use-formulas-to-find-the-volume-of-pyramids)."""

import re

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7e_pyramid_sphere_volume_build import (
    _verify_square_pyramid,
    _verify_rect_pyramid,
    _verify_sphere,
)

SEEDS = range(40)

TOPIC = "volume_of_pyramids_and_spheres"
DIFFICULTIES = ["easy", "medium", "hard"]


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_generator_is_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        repeat = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_expression(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_expression(question, distractor).correct


def test_plain_integer_answer_is_accepted_under_grade_expression():
    """Confirms the grader-kind resolution: a plain integer answer_text
    parses and grades correctly under grade_expression -- not assumed."""
    question = make(TOPIC, seed=0, difficulty="easy")
    assert question.answer_text.isdigit()
    assert grade_expression(question, question.answer_text).correct
    wrong = str(int(question.answer_text) + 1)
    assert not grade_expression(question, wrong).correct


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        if difficulty == "hard":
            match = re.fullmatch(r"(\d*)pi", question.answer_text)
            assert match
            k = int(match.group(1)) if match.group(1) else 1
            wrong = f"{k + 1}pi"
        else:
            wrong = str(int(question.answer_text) + 1)
        assert not grade_expression(question, wrong).correct


# --- independent check fires on corrupted input -----------------------------


def test_square_pyramid_independent_check_fires_on_corrupted():
    _verify_square_pyramid(4, 9, 48)  # (9//3)*4*4 = 48, must not raise
    with pytest.raises(VerificationError):
        _verify_square_pyramid(4, 9, 47)


def test_rect_pyramid_independent_check_fires_on_corrupted():
    _verify_rect_pyramid(5, 6, 9, 90)  # 5*6*9/3 = 90, must not raise
    with pytest.raises(VerificationError):
        _verify_rect_pyramid(5, 6, 9, 89)


def test_sphere_independent_check_fires_on_corrupted():
    _verify_sphere(18, 972)  # r=9, third=3, 36*27=972, must not raise
    with pytest.raises(VerificationError):
        _verify_sphere(18, 971)


# --- skill loading ------------------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skills_by_id = {skill.id: skill for skill in skills}

    skill = skills_by_id.get("senior-use-formulas-to-find-the-volume-of-pyramids")
    assert skill is not None, "skill not found in loaded skills"
    assert skill.domain == "measurement"
    assert skill.prerequisites == ["convert-metric-length"]
    assert skill.answer == "expression"
    assert skill.placement_eligible == False


# --- answer satisfies its declared kind, formatted alike per tier -----------


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_pyramid_answer_is_plain_integer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.answer_text.isdigit()
        for distractor in question.distractors:
            assert distractor.isdigit()


def test_sphere_answer_is_single_pi_term_never_decimal():
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="hard")
        assert re.fullmatch(r"\d*pi", question.answer_text), (
            f"seed {seed}: answer_text={question.answer_text!r} is not a single k*pi term"
        )
        assert "." not in question.answer_text
        assert "+" not in question.answer_text
        assert "*" not in question.answer_text
        for distractor in question.distractors:
            assert re.fullmatch(r"\d*pi", distractor)
            assert "*" not in distractor


# --- variety: at least 20 distinct questions per tier over 60 seeds --------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(difficulty):
    questions = []
    for seed in range(60):
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        questions.append(question.text)
    distinct = len(set(questions))
    assert distinct >= 20, f"{difficulty}: only {distinct} distinct questions over 60 seeds (need >=20)"


# --- unknown mode raises -----------------------------------------------------


def test_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="unknown")


# --- required traps present on every seed -----------------------------------


def test_easy_omits_third_distractor_present_every_seed():
    """REQUIRED trap: treating the pyramid as a prism (omitting the 1/3
    factor) must be offered as a distractor at every swept seed."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="easy")
        match = re.search(
            r"square base with side length (\d+) cm and a height of (\d+) cm",
            question.text,
        )
        assert match
        s, h = int(match.group(1)), int(match.group(2))
        expected_trap = str(s * s * h)
        assert expected_trap in question.distractors


def test_hard_uses_diameter_as_radius_distractor_present_every_seed():
    """REQUIRED trap: cubing the diameter directly instead of halving it
    first must be offered as a distractor at every swept seed."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="hard")
        match = re.search(r"diameter of (\d+) cm", question.text)
        assert match
        d = int(match.group(1))
        expected_trap_k = 4 * d**3 // 3
        expected_trap = "pi" if expected_trap_k == 1 else f"{expected_trap_k}pi"
        assert expected_trap in question.distractors
