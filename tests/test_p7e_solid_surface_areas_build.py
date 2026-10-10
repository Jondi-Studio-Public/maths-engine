"""Tests the senior surface-area skills, split into curved-surface and flat-faced solids.

Senior surface area, curved vs flat-faced split
(senior-calculate-the-surface-areas-of-standard-three-dimensional,
senior-find-the-surface-area-of-familiar-solids-including)."""

import re

import pytest

from mathgen.grading.typed import grade_expression, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7e_solid_surface_areas_build import (
    _verify_sphere,
    _verify_cone,
    _verify_composite,
    _verify_cube,
    _verify_rect_prism,
    _verify_tri_prism,
)

SEEDS = range(40)

CURVED_TOPIC = "calculate_surface_areas_of_standard_solids"
FLAT_TOPIC = "find_surface_area_of_familiar_solids"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [CURVED_TOPIC, FLAT_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [CURVED_TOPIC, FLAT_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_curved_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make(CURVED_TOPIC, seed=seed, difficulty=difficulty)
        assert grade_expression(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_expression(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_flat_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make(FLAT_TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_curved_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(CURVED_TOPIC, seed=seed, difficulty=difficulty)
        # Deliberately wrong: bump the pi coefficient by 1 (canonical is
        # always "Npi" or "pi"); "(N+1)pi" is never symbolically equal.
        match = re.match(r"^(\d*)pi$", question.answer_text)
        assert match, f"unexpected answer_text shape: {question.answer_text!r}"
        k = int(match.group(1)) if match.group(1) else 1
        wrong = f"{k + 1}pi"
        assert not grade_expression(question, wrong).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_flat_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(FLAT_TOPIC, seed=seed, difficulty=difficulty)
        wrong = str(int(question.answer) + 1)
        assert not grade_number(question, wrong).correct


# --- independent check fires on corrupted input -----------------------------


def test_sphere_independent_check_fires_on_corrupted():
    _verify_sphere(5, 100)  # 4*5*5 = 100, must not raise
    with pytest.raises(VerificationError):
        _verify_sphere(5, 99)


def test_cone_independent_check_fires_on_corrupted():
    _verify_cone(3, 10, 39)  # 3*3 + 3*10 = 39, must not raise
    with pytest.raises(VerificationError):
        _verify_cone(3, 10, 38)


def test_composite_independent_check_fires_on_corrupted():
    _verify_composite(3, 10, 48)  # 3*10 + 2*3*3 = 30 + 18 = 48, must not raise
    with pytest.raises(VerificationError):
        _verify_composite(3, 10, 47)


def test_cube_independent_check_fires_on_corrupted():
    _verify_cube(6, 216)  # 6*6*6 = 216, must not raise
    with pytest.raises(VerificationError):
        _verify_cube(6, 215)


def test_rect_prism_independent_check_fires_on_corrupted():
    _verify_rect_prism(3, 5, 18, 318)  # 2*(15+54+90) = 318, must not raise
    with pytest.raises(VerificationError):
        _verify_rect_prism(3, 5, 18, 317)


def test_tri_prism_independent_check_fires_on_corrupted():
    # a=4,b=3,c=5, length=10 -> 2*(4*3/2) + (4+3+5)*10 = 12 + 120 = 132
    _verify_tri_prism(4, 3, 5, 10, 132)  # must not raise
    with pytest.raises(VerificationError):
        _verify_tri_prism(4, 3, 5, 10, 131)


# --- both skills loading ----------------------------------------------------


def test_skills_load_with_correct_properties():
    _, skills = load_skills()
    skills_by_id = {skill.id: skill for skill in skills}

    curved = skills_by_id.get("senior-calculate-the-surface-areas-of-standard-three-dimensional")
    assert curved is not None, "curved skill not found in loaded skills"
    assert curved.prerequisites == ["convert-metric-length"]
    assert curved.answer == "expression"
    assert curved.placement_eligible == False

    flat = skills_by_id.get("senior-find-the-surface-area-of-familiar-solids-including")
    assert flat is not None, "flat skill not found in loaded skills"
    assert flat.prerequisites == ["convert-metric-length"]
    assert flat.answer == "number"
    assert flat.placement_eligible == False


# --- answer satisfies its declared kind -------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_curved_answer_is_single_pi_term_never_decimal(difficulty):
    """Answer is always exactly "pi" or "<int>pi" -- never a decimal, never
    an unsimplified sum of two pi-terms."""
    for seed in SEEDS:
        question = make(CURVED_TOPIC, seed=seed, difficulty=difficulty)
        assert re.fullmatch(r"\d*pi", question.answer_text), (
            f"seed {seed}: answer_text={question.answer_text!r} is not a single k*pi term"
        )
        assert "." not in question.answer_text
        assert "+" not in question.answer_text
        assert "*" not in question.answer_text
        for distractor in question.distractors:
            assert re.fullmatch(r"\d*pi", distractor)
            assert "*" not in distractor


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_flat_answer_is_plain_integer(difficulty):
    for seed in SEEDS:
        question = make(FLAT_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, int)
        assert question.answer_text.isdigit()
        for distractor in question.distractors:
            assert distractor.isdigit()


# --- variety: at least 20 distinct questions per tier over 60 seeds --------


@pytest.mark.parametrize("topic", [CURVED_TOPIC, FLAT_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_at_least_20_distinct_questions_per_tier(topic, difficulty):
    questions = []
    for seed in range(60):
        question = make(topic, seed=seed, difficulty=difficulty)
        questions.append(question.text)
    distinct = len(set(questions))
    assert distinct >= 20, f"{topic} {difficulty}: only {distinct} distinct questions over 60 seeds (need >=20)"


# --- unknown mode raises -----------------------------------------------------


def test_curved_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(CURVED_TOPIC, seed=1, mode="unknown")


def test_flat_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(FLAT_TOPIC, seed=1, mode="unknown")


# --- hard-tier required trap present -----------------------------------------


def test_curved_hard_includes_interface_distractor_on_every_seed():
    """THE hard-tier trap for the curved skill: including the shared
    interface circle must be offered as a distractor at every swept seed."""
    for seed in SEEDS:
        question = make(CURVED_TOPIC, seed=seed, difficulty="hard")
        match = re.match(r"^(\d*)pi$", question.answer_text)
        k = int(match.group(1)) if match.group(1) else 1
        match_r = re.search(r"base radius (\d+) cm", question.text)
        assert match_r
        r = int(match_r.group(1))
        expected_trap = f"{k + r * r}pi"
        assert expected_trap in question.distractors, (
            f"seed {seed}: interface-circle trap {expected_trap!r} not in {question.distractors}"
        )


def test_flat_hard_forgets_half_distractor_on_every_seed():
    """REQUIRED trap for the flat-faced skill: forgetting the 1/2 factor on
    the triangular ends must be offered as a distractor at every swept seed."""
    for seed in SEEDS:
        question = make(FLAT_TOPIC, seed=seed, difficulty="hard")
        match = re.search(
            r"legs (\d+) cm and (\d+) cm and hypotenuse (\d+) cm, and a "
            r"length of (\d+) cm",
            question.text,
        )
        assert match
        a, b, c, length = (int(match.group(i)) for i in range(1, 5))
        perimeter = a + b + c
        expected_trap = str(2 * a * b + perimeter * length)
        assert expected_trap in question.distractors, (
            f"seed {seed}: forgets-half trap {expected_trap!r} not in {question.distractors}"
        )
