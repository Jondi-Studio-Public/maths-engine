"""Contract tests for vector calculus (ACMSM113): differentiate and integrate
a vector function with respect to time."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5e_vector_calculus_build import (
    _verify_easy,
    _verify_medium,
    _verify_hard,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


class TestDeterminismAndDisctractors:
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_generators_are_deterministic_across_seeds(self, difficulty):
        """Same seed/difficulty always produces the same question."""
        for seed in SEEDS:
            question = make(
                "differentiate_and_integrate_a_vector_function_with_respect", seed=seed, difficulty=difficulty
            )
            repeat = make(
                "differentiate_and_integrate_a_vector_function_with_respect", seed=seed, difficulty=difficulty
            )
            assert question.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_distractors_are_unique_and_distinct_from_answer(self, difficulty):
        """Distractors are unique strings and none equals the canonical answer."""
        for seed in SEEDS:
            question = make(
                "differentiate_and_integrate_a_vector_function_with_respect", seed=seed, difficulty=difficulty
            )
            assert question.answer_text not in question.distractors
            assert len(question.distractors) == len(set(question.distractors))
            assert len(question.distractors) >= 2

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_no_candidate_pair_collides_within_tolerance(self, difficulty):
        """No distractor pair has both components within 0.051 of the answer.

        This ensures every candidate will grade as wrong (grade_pair requires
        both components within tolerance to mark correct).
        """
        for seed in range(200):
            question = make(
                "differentiate_and_integrate_a_vector_function_with_respect", seed=seed, difficulty=difficulty
            )
            ax, ay = question.answer["x"], question.answer["y"]
            for distractor_str in question.distractors:
                # Parse distractor string "x = ..., y = ..."
                parts = distractor_str.split(", ")
                assert len(parts) == 2
                x_str = parts[0].split(" = ")[1]
                y_str = parts[1].split(" = ")[1]
                cx, cy = float(x_str), float(y_str)

                # At least one component must differ by more than tolerance
                x_within_tol = abs(cx - ax) <= 0.051
                y_within_tol = abs(cy - ay) <= 0.051
                assert not (x_within_tol and y_within_tol), (
                    f"Distractor '{distractor_str}' collides with answer ({ax}, {ay}) at seed {seed}"
                )


class TestGrading:
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_correct_answer_grades_correct(self, difficulty):
        """The canonical answer_text always grades as correct."""
        for seed in SEEDS:
            question = make(
                "differentiate_and_integrate_a_vector_function_with_respect", seed=seed, difficulty=difficulty
            )
            result = grade_pair(question, question.answer_text)
            assert result.correct, f"Failed to grade canonical answer at seed {seed}: {question.answer_text}"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_distractors_grade_incorrect(self, difficulty):
        """Every distractor grades as incorrect."""
        for seed in SEEDS:
            question = make(
                "differentiate_and_integrate_a_vector_function_with_respect", seed=seed, difficulty=difficulty
            )
            for distractor in question.distractors:
                result = grade_pair(question, distractor)
                assert not result.correct, (
                    f"Distractor incorrectly graded as correct: {distractor} "
                    f"(expected {question.answer_text}) at seed {seed}"
                )

    def test_deliberately_wrong_answer_grades_incorrect_easy(self):
        """A deliberately-wrong answer grades as incorrect (easy tier)."""
        question = make("differentiate_and_integrate_a_vector_function_with_respect", seed=1, difficulty="easy")
        wrong_answer = f"x = {question.answer['x'] + 10}, y = {question.answer['y'] + 10}"
        result = grade_pair(question, wrong_answer)
        assert not result.correct

    def test_deliberately_wrong_answer_grades_incorrect_medium(self):
        """A deliberately-wrong answer grades as incorrect (medium tier)."""
        question = make("differentiate_and_integrate_a_vector_function_with_respect", seed=1, difficulty="medium")
        wrong_answer = f"x = {question.answer['x'] + 5}, y = {question.answer['y'] + 5}"
        result = grade_pair(question, wrong_answer)
        assert not result.correct

    def test_deliberately_wrong_answer_grades_incorrect_hard(self):
        """A deliberately-wrong answer grades as incorrect (hard tier)."""
        question = make("differentiate_and_integrate_a_vector_function_with_respect", seed=1, difficulty="hard")
        wrong_answer = f"x = {question.answer['x'] + 10}, y = {question.answer['y'] + 10}"
        result = grade_pair(question, wrong_answer)
        assert not result.correct


class TestTextQuality:
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_no_bare_multiplication_signs_in_text(self, difficulty):
        """No bare * appears in the prompt text (only in ** exponents are allowed, but we avoid those too)."""
        for seed in range(50):
            question = make(
                "differentiate_and_integrate_a_vector_function_with_respect", seed=seed, difficulty=difficulty
            )
            # Check for bare * (not **)
            import re

            # Match * that is not part of **
            bare_star = re.search(r"(?<!\*)\*(?!\*)", question.text)
            assert not bare_star, f"Found bare * in text at seed {seed}: {question.text}"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_no_python_exponent_notation_in_text(self, difficulty):
        """No ** appears in the prompt text."""
        for seed in range(50):
            question = make(
                "differentiate_and_integrate_a_vector_function_with_respect", seed=seed, difficulty=difficulty
            )
            assert "**" not in question.text, f"Found ** in text at seed {seed}: {question.text}"
            assert "**" not in question.answer_text
            for distractor in question.distractors:
                assert "**" not in distractor


class TestIndependentChecks:
    def test_easy_independent_check_fires_on_corrupted_answer(self):
        """The easy-tier finite-difference check detects a wrong answer."""
        with pytest.raises(VerificationError):
            _verify_easy(2, 3, 1, -4, 2, 999, 10)  # Wrong x_ans
        # A correct answer must pass
        _verify_easy(2, 3, 1, -4, 2, 2 * 2 * 2 + 3, 2 * 1 * 2 + (-4))

    def test_medium_independent_check_fires_on_corrupted_answer(self):
        """The medium-tier sympy differentiation check detects a wrong answer."""
        with pytest.raises(VerificationError):
            _verify_medium(3, 2, 5, 3, 999, 15)  # Wrong x_ans
        # A correct answer must pass
        _verify_medium(3, 2, 5, 3, 3 * 2, 5 * 3)

    def test_hard_independent_check_verifies_ftc(self):
        """The hard-tier FTC check verifies the fundamental theorem of calculus.

        The check differentiates the antiderivative d1*t^2 + e1*t and confirms
        it matches the velocity 2*d1*t + e1. This is always true algebraically.
        """
        # Any coefficients should pass this check, since the FTC is always true
        _verify_hard(2, 3, 1, -4)
        _verify_hard(1, 1, 1, 1)
        _verify_hard(4, -6, 3, 5)


class TestSkillLoading:
    def test_skill_loads_with_correct_fields(self):
        """The skill loads through the API with expected structure."""
        _, skills = load_skills(ROOT / "skills")
        skill = next(
            (s for s in skills if s.id == "senior-differentiate-and-integrate-a-vector-function-with-respect"), None
        )
        assert skill is not None
        assert skill.answer == "pair"
        assert skill.prerequisites == ["differentiate-polynomial"]
        assert skill.placement_eligible is False
        # The skill may carry further exam-tier sources from other generators; this generator owns the first three.
        own = [s for s in skill.sources if s.topic == "differentiate_and_integrate_a_vector_function_with_respect"]
        assert [s.difficulty for s in own] == ["easy", "medium", "hard"]

    def test_skill_misconceptions_are_listed(self):
        """The skill has the expected misconceptions."""
        _, skills = load_skills(ROOT / "skills")
        skill = next(
            (s for s in skills if s.id == "senior-differentiate-and-integrate-a-vector-function-with-respect"), None
        )
        assert skill is not None
        misconception_ids = {m.id for m in skill.misconceptions}
        expected = {
            "evaluated-position-not-velocity",
            "forgot-power-rule-coefficient",
            "sign-error-both-components",
            "forgot-chain-rule-both-components",
            "sign-confused-trig-derivative",
            "sign-error-exponential-component",
            "used-velocity-instead-of-integrating",
            "botched-power-rule-antiderivative",
        }
        assert misconception_ids == expected
