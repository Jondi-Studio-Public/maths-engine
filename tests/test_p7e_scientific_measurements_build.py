"""Tests `f10-scientific-notation-measurements` (topic `scientific_notation_measurements`, AC9M9M02)."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7e_scientific_measurements_build import (
    _verify_multiply,
    _verify_divide,
    _verify_compare,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
TOPIC = "scientific_notation_measurements"
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


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_plain_int_never_negative(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, int)
        assert question.answer > 0


# --- real grader accept/reject ------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        wrong = question.answer + 1
        assert not grade_number(question, str(wrong)).correct


# --- independent in-generator check fires on corrupted input -----------------


def test_multiply_independent_check_fires_on_corrupted():
    from fractions import Fraction

    sc, se, tc, te = Fraction(25, 10), 2, 4, 1
    # 2.5 x 10^2 x 4 x 10^1 = 250 x 40 = 10000
    _verify_multiply(sc, se, tc, te, Fraction(10000))  # must not raise
    with pytest.raises(VerificationError):
        _verify_multiply(sc, se, tc, te, Fraction(9999))


def test_divide_independent_check_fires_on_corrupted():
    from fractions import Fraction

    # 6 x 10^7 km / (3 x 10^2 km/h) = 2 x 10^5 = 200000
    _verify_divide(6, 7, 3, 2, Fraction(200000))  # must not raise
    with pytest.raises(VerificationError):
        _verify_divide(6, 7, 3, 2, Fraction(199999))


def test_compare_independent_check_fires_on_corrupted():
    from fractions import Fraction

    # 2 x 10^5 / (4 x 10^3) = 200000 / 4000 = 50
    _verify_compare(2, 5, 4, 3, Fraction(50))  # must not raise
    with pytest.raises(VerificationError):
        _verify_compare(2, 5, 4, 3, Fraction(49))


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, variant="unknown")


# --- variety: at least 20 distinct questions per tier over 60 seeds ---------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(difficulty):
    questions = [make(TOPIC, seed=s, difficulty=difficulty).text for s in range(60)]
    distinct = len(set(questions))
    assert distinct >= 20, f"{difficulty}: only {distinct} distinct questions over 60 seeds"


# --- answer and distractors formatted alike (all plain positive integers) ----


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_formatted_alike(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.answer_text.isdigit(), (
            f"{difficulty} seed {seed}: answer_text={question.answer_text!r} is not all digits"
        )
        for distractor in question.distractors:
            assert distractor.isdigit(), (
                f"{difficulty} seed {seed}: distractor={distractor!r} is not all digits "
                f"(answer_text={question.answer_text!r})"
            )


# --- hard-tier magnitude judgement: heavier object is not always named first ---


def test_hard_tier_heavier_object_label_varies():
    """The physically heavier object must not always be 'Object P' (or always
    'Object Q') -- otherwise the position alone, not the magnitude judgement,
    would answer the question."""
    import re

    p_is_heavier = 0
    q_is_heavier = 0
    for seed in range(60):
        question = make(TOPIC, seed=seed, difficulty="hard")
        matches = re.findall(r"Object (P|Q) has a mass of (\d+) × 10\^(-?\d+) kg", question.text)
        assert len(matches) == 2, f"seed {seed}: could not parse both objects from {question.text!r}"
        (label_a, coef_a, exp_a), (label_b, coef_b, exp_b) = matches
        value_a = int(coef_a) * (10 ** int(exp_a))
        value_b = int(coef_b) * (10 ** int(exp_b))
        heavier_label = label_a if value_a > value_b else label_b
        if heavier_label == "P":
            p_is_heavier += 1
        else:
            q_is_heavier += 1
    assert p_is_heavier > 0 and q_is_heavier > 0, (
        f"heavier object label never varies: P={p_is_heavier}, Q={q_is_heavier}"
    )


# --- skill loads correctly ----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skills_by_id = {skill.id: skill for skill in skills}

    skill = skills_by_id.get("f10-scientific-notation-measurements")
    assert skill is not None, "f10-scientific-notation-measurements not found in loaded skills"
    assert skill.prerequisites == ["f10-powers-ten-multiply-divide", "scientific-notation"]
    assert skill.answer == "number"
    assert skill.placement_eligible == False
