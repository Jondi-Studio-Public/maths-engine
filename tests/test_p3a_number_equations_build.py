"""Focused contract tests for: f10-multiplication-division-inverse-unknown and f10-unit-fractions-combine-whole."""

from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_fraction, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p3a_number_equations_build import (
    _verify_multiplication_division_inverse,
    _verify_fraction_multiple,
    _verify_fraction_complete_whole,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
SWEEP_SEEDS = range(500)


# --- multiplication_division_inverse_unknown: determinism & distractor hygiene ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_multiplication_inverse_generators_are_deterministic_across_seeds_and_have_unique_distractors(difficulty):
    """Determinism: same seed produces identical question. Distractor hygiene:
    distinct from answer, all unique, at least 2 present."""
    for seed in SEEDS:
        question = make("multiplication_division_inverse_unknown", seed=seed, difficulty=difficulty)
        repeat = make("multiplication_division_inverse_unknown", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert isinstance(question.answer, int)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_multiplication_inverse_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    """Real grader must accept canonical answer_text and reject every distractor."""
    for seed in SEEDS:
        question = make("multiplication_division_inverse_unknown", seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct, (
            f"seed {seed} difficulty {difficulty}: canonical {question.answer_text} not accepted"
        )
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct, (
                f"seed {seed} difficulty {difficulty}: distractor {distractor} wrongly accepted"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_multiplication_inverse_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make("multiplication_division_inverse_unknown", seed=seed, difficulty=difficulty)
        wrong = question.answer + 7  # arbitrary offset
        assert not grade_number(question, str(wrong)).correct, f"seed {seed}: deliberate wrong {wrong} not rejected"


def test_multiplication_inverse_independent_check_fires_on_corrupted_answer():
    """The independent check must fire when an answer is corrupted."""
    with pytest.raises(VerificationError):
        _verify_multiplication_division_inverse("a_times_missing_equals_c", a=3, b=5, c=15, missing=7)
    with pytest.raises(VerificationError):
        _verify_multiplication_division_inverse("c_div_a_equals_missing", a=3, b=5, c=15, missing=4)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_multiplication_inverse_text_contains_inverse_relationship_phrase(difficulty):
    """Question text must explicitly mention 'inverse multiplication-division relationship'."""
    for seed in SEEDS:
        question = make("multiplication_division_inverse_unknown", seed=seed, difficulty=difficulty)
        assert "inverse multiplication-division relationship" in question.text, (
            f"seed {seed}: question text missing method phrase: {question.text}"
        )


def test_multiplication_inverse_easy_has_only_multiplication_shapes():
    """Easy tier should only have multiplication shapes."""
    for seed in SEEDS:
        question = make("multiplication_division_inverse_unknown", seed=seed, difficulty="easy")
        # Easy should have multiplication shapes only (look for × or x in text)
        assert "×" in question.text or " x " in question.text or "\\times" in question.text, (
            f"seed {seed} easy: expected multiplication shape, got: {question.text}"
        )


def test_multiplication_inverse_hard_has_all_four_shapes_over_sweep():
    """Over 500 seeds, hard should produce all four shapes."""
    shapes_seen = set()
    for seed in SWEEP_SEEDS:
        question = make("multiplication_division_inverse_unknown", seed=seed, difficulty="hard")
        if " ÷ " in question.text or "//" in question.text or "÷" in question.text:
            shapes_seen.add("division")
        else:
            shapes_seen.add("multiplication")
    assert "division" in shapes_seen, "hard difficulty never produced division shapes in 500 seeds"


# --- unit_fractions_combine_whole: determinism & distractor hygiene ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_fraction_combine_generators_are_deterministic_and_have_unique_distractors(difficulty):
    """Determinism and distractor hygiene for fraction questions."""
    for seed in SEEDS:
        question = make("unit_fractions_combine_whole", seed=seed, difficulty=difficulty)
        repeat = make("unit_fractions_combine_whole", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert isinstance(question.answer, dict)
        assert "p" in question.answer and "q" in question.answer
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_fraction_combine_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    """Real grader accepts canonical and rejects all distractors."""
    for seed in SEEDS:
        question = make("unit_fractions_combine_whole", seed=seed, difficulty=difficulty)
        assert grade_fraction(question, question.answer_text).correct, (
            f"seed {seed} difficulty {difficulty}: canonical {question.answer_text} not accepted"
        )
        for distractor in question.distractors:
            assert not grade_fraction(question, distractor).correct, (
                f"seed {seed} difficulty {difficulty}: distractor {distractor} wrongly accepted"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_fraction_combine_real_grader_rejects_deliberately_wrong_answer(difficulty):
    """Real grader must reject a deliberately wrong fraction."""
    for seed in SEEDS:
        question = make("unit_fractions_combine_whole", seed=seed, difficulty=difficulty)
        correct_frac = Fraction(question.answer["p"], question.answer["q"])
        wrong = correct_frac + Fraction(1, 7)  # arbitrary offset
        assert not grade_fraction(question, str(wrong)).correct, f"seed {seed}: deliberate wrong {wrong} not rejected"


def test_fraction_combine_independent_check_fires_on_corrupted_answer():
    """Independent check must fire on corrupted values."""
    with pytest.raises(VerificationError):
        _verify_fraction_multiple(k=3, d=5, answer=Fraction(4, 5))
    with pytest.raises(VerificationError):
        _verify_fraction_complete_whole(d=5, n_total=2, answer=Fraction(2, 5))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_fraction_combine_answer_is_valid_fraction(difficulty):
    """Answer must be a valid fraction in reduced form."""
    for seed in SEEDS:
        question = make("unit_fractions_combine_whole", seed=seed, difficulty=difficulty)
        p, q = question.answer["p"], question.answer["q"]
        frac = Fraction(p, q)
        # Auto-reduced
        assert p == frac.numerator and q == frac.denominator, (
            f"seed {seed}: answer not reduced: {p}/{q} != {frac.numerator}/{frac.denominator}"
        )
        assert q > 0 and p >= 0, f"seed {seed}: invalid fraction {p}/{q}"


def test_fraction_combine_easy_uses_multiple_shape():
    """Easy should use the 'multiple' shape."""
    for seed in SEEDS:
        question = make("unit_fractions_combine_whole", seed=seed, difficulty="easy")
        # Multiple shape should ask "what fraction is k copies of 1/d"
        assert "copies" in question.text or "1/" in question.text, (
            f"seed {seed}: unexpected easy question: {question.text}"
        )


def test_fraction_combine_medium_uses_complete_whole():
    """Medium should use the 'complete_the_whole' shape."""
    for seed in SEEDS:
        question = make("unit_fractions_combine_whole", seed=seed, difficulty="medium")
        assert "complete the whole" in question.text or "still needed" in question.text, (
            f"seed {seed}: unexpected medium question: {question.text}"
        )


def test_fraction_combine_hard_uses_two_parts():
    """Hard should mention combining two parts."""
    for seed in SEEDS:
        question = make("unit_fractions_combine_whole", seed=seed, difficulty="hard")
        # Hard mentions n1 and n2 parts
        assert "used, then" in question.text or "more parts" in question.text, (
            f"seed {seed}: unexpected hard question: {question.text}"
        )


# --- Skills loading verification ---


def test_both_skills_load_through_api_with_correct_properties():
    """Both skills must load through the API with correct properties."""
    domains, skills = load_skills()

    # Check f10-multiplication-division-inverse-unknown
    mult_div_skill = next((s for s in skills if s.id == "f10-multiplication-division-inverse-unknown"), None)
    assert mult_div_skill is not None, "f10-multiplication-division-inverse-unknown not found"
    assert mult_div_skill.domain == "equations"
    assert mult_div_skill.status == "active"
    assert mult_div_skill.placement_eligible is False
    assert mult_div_skill.answer == "number"
    assert set(mult_div_skill.prerequisites) == {"multiplication-facts", "whole-number-division-facts"}

    # Check f10-unit-fractions-combine-whole
    frac_skill = next((s for s in skills if s.id == "f10-unit-fractions-combine-whole"), None)
    assert frac_skill is not None, "f10-unit-fractions-combine-whole not found"
    assert frac_skill.domain == "primary-number"
    assert frac_skill.status == "active"
    assert frac_skill.placement_eligible is False
    assert frac_skill.answer == "fraction"
    assert set(frac_skill.prerequisites) == {"f10-fractions-half-quarter-eighth", "fraction-arithmetic"}
