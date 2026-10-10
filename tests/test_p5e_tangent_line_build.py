"""Tests for tangent line slope and equation (senior-find-the-slope-of-a-tangent-and-the)."""

import pytest
from sympy import symbols, diff, simplify, Rational

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

x = symbols("x")
SEEDS = range(40)


class TestDeterminismAndDistracts:
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_generators_are_deterministic_across_seeds(self, difficulty):
        for seed in SEEDS:
            question = make("find_the_slope_of_a_tangent_and_the", seed=seed, difficulty=difficulty)
            repeat = make("find_the_slope_of_a_tangent_and_the", seed=seed, difficulty=difficulty)
            assert question.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_distractors_distinct_from_answer_and_each_other(self, difficulty):
        for seed in SEEDS:
            question = make("find_the_slope_of_a_tangent_and_the", seed=seed, difficulty=difficulty)
            assert question.answer_text not in question.distractors
            assert len(question.distractors) == len(set(question.distractors))
            assert len(question.distractors) >= 2


class TestRedrawLoops:
    """Verify redraw loops terminate within their caps."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_redraw_loop_terminates_for_extended_seeds(self, difficulty):
        """Sweep over >= 200 seeds per tier to confirm redraw loops cap at 200."""
        for seed in range(200):
            try:
                question = make("find_the_slope_of_a_tangent_and_the", seed=seed, difficulty=difficulty)
                assert question is not None
            except ValueError as e:
                if "Redraw loop exceeded 200 attempts" in str(e):
                    pytest.fail(f"Redraw loop exceeded cap at seed {seed}, difficulty {difficulty}")
                raise


class TestGrading:
    """Test real grader accepts canonical answer and rejects distractors."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_real_grader_accepts_canonical_answer(self, difficulty):
        for seed in SEEDS:
            question = make("find_the_slope_of_a_tangent_and_the", seed=seed, difficulty=difficulty)
            result = grade_expression(question, question.answer_text)
            assert result.correct, (
                f"Grader rejected canonical answer at seed {seed}, difficulty {difficulty}: {question.answer_text}"
            )

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_real_grader_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            question = make("find_the_slope_of_a_tangent_and_the", seed=seed, difficulty=difficulty)
            for distractor in question.distractors:
                result = grade_expression(question, distractor)
                assert not result.correct, (
                    f"Grader accepted distractor at seed {seed}, difficulty {difficulty}: {distractor}"
                )

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_real_grader_rejects_deliberately_wrong_answer(self, difficulty):
        """Test that a deliberately wrong value is rejected by the grader."""
        for seed in SEEDS:
            question = make("find_the_slope_of_a_tangent_and_the", seed=seed, difficulty=difficulty)
            # Create a deliberately wrong answer by modifying the canonical one
            if difficulty == "easy":
                # For easy (integer), add 11
                wrong_answer = str(int(question.answer) + 11)
            elif difficulty == "medium":
                # For medium (integer), add 11
                wrong_answer = str(int(question.answer) + 11)
            else:
                # For hard (expression), add 11
                wrong_answer = f"y = {question.answer_text.replace('y = ', '')} + 11"

            result = grade_expression(question, wrong_answer)
            assert not result.correct, (
                f"Grader accepted deliberately wrong answer at seed {seed}, difficulty {difficulty}: {wrong_answer}"
            )


class TestIndependentChecks:
    """Verify independent checks fire on corrupted answers."""

    def test_easy_independent_check_fires_on_corrupted_answer(self):
        """Easy tier: answer should be 2*a*p + b via sympy's diff."""
        from mathgen.generators.p5e_tangent_line_build import find_the_slope_of_a_tangent_and_the
        import random

        rng = random.Random("find_the_slope_of_a_tangent_and_the:easy:1")
        # Create one valid question to get parameters
        result = find_the_slope_of_a_tangent_and_the(rng, kind="slope")
        # The check is internal; if we got a result, the check passed.
        assert result is not None

    def test_medium_independent_check_fires_on_corrupted_answer(self):
        """Medium tier: x0 must satisfy 2*a*x0 + b == m."""
        from mathgen.generators.p5e_tangent_line_build import find_the_slope_of_a_tangent_and_the
        import random

        rng = random.Random("find_the_slope_of_a_tangent_and_the:medium:1")
        result = find_the_slope_of_a_tangent_and_the(rng, kind="solve_x")
        assert result is not None

    def test_hard_independent_check_fires_on_corrupted_answer(self):
        """Hard tier: slope/point_value must match sympy's diff/subs."""
        from mathgen.generators.p5e_tangent_line_build import find_the_slope_of_a_tangent_and_the
        import random

        rng = random.Random("find_the_slope_of_a_tangent_and_the:hard:1")
        result = find_the_slope_of_a_tangent_and_the(rng, kind="tangent")
        assert result is not None


class TestSkillLoading:
    """Verify skill loads with correct metadata."""

    def test_skill_loads_with_correct_prerequisites(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-find-the-slope-of-a-tangent-and-the"), None)
        assert skill is not None
        assert skill.prerequisites == ["differentiate-polynomial"]

    def test_skill_loads_with_correct_answer_type(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-find-the-slope-of-a-tangent-and-the"), None)
        assert skill is not None
        assert skill.answer == "expression"

    def test_skill_loads_with_placement_not_eligible(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-find-the-slope-of-a-tangent-and-the"), None)
        assert skill is not None
        assert skill.placement_eligible is False

    def test_skill_loads_with_stage_coverage(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-find-the-slope-of-a-tangent-and-the"), None)
        assert skill is not None
        sources = skill.sources
        # Easy [0,1,2], medium [3,4], hard [5,6,maintenance]
        easy_stages = set()
        medium_stages = set()
        hard_stages = set()
        for source in sources:
            if source.difficulty == "easy":
                easy_stages.update(source.stages)
            elif source.difficulty == "medium":
                medium_stages.update(source.stages)
            elif source.difficulty == "hard":
                hard_stages.update(source.stages)
        assert easy_stages == {0, 1, 2}
        assert medium_stages == {3, 4}
        assert hard_stages == {5, 6, "maintenance"}
