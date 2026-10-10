"""Tests for complex number plane representation and polar conversion skills."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_pair
from mathgen import make

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


class TestComplexNumberAsPoint:
    """Tests for complex_number_as_point generator."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_determinism(self, difficulty, seed):
        """Every seed produces consistent output."""
        q1 = make("complex_number_as_point", seed=seed, difficulty=difficulty)
        q2 = make("complex_number_as_point", seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_answer_is_dict_pair(self, difficulty, seed):
        """Answer is a dict with exactly 2 numeric values."""
        q = make("complex_number_as_point", seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict)
        assert len(q.answer) == 2
        assert all(isinstance(v, (int, float)) for v in q.answer.values())

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_distractors_distinct_from_answer(self, difficulty, seed):
        """Every distractor differs from the answer by more than 0.051 in at least one component."""
        q = make("complex_number_as_point", seed=seed, difficulty=difficulty)
        answer_vals = list(q.answer.values())
        for distractor in q.distractors:
            # Parse distractor text (e.g., "x = 3, y = 5")
            parts = distractor.split(", ")
            dist_vals = []
            for part in parts:
                val = float(part.split(" = ")[1])
                dist_vals.append(val)

            # Check that at least one component differs by more than 0.051
            assert any(abs(d - a) > 0.051 for d, a in zip(dist_vals, answer_vals)), (
                f"Distractor {distractor} not sufficiently distinct from answer"
            )

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_real_grader_accepts_canonical(self, difficulty, seed):
        """The real grade_pair grader accepts the canonical answer text."""
        q = make("complex_number_as_point", seed=seed, difficulty=difficulty)
        result = grade_pair(q, q.answer_text)
        assert result.correct, f"Grader rejected canonical: {q.answer_text}"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_real_grader_rejects_distractors(self, difficulty, seed):
        """The real grade_pair grader rejects all distractors."""
        q = make("complex_number_as_point", seed=seed, difficulty=difficulty)
        for distractor in q.distractors:
            result = grade_pair(q, distractor)
            assert not result.correct, f"Grader accepted distractor: {distractor}"

    def test_degenerate_has_both_shapes(self):
        """Over 500 seeds, medium (degenerate) variant produces both purely-real and purely-imaginary."""
        real_count = 0
        imag_count = 0

        for seed in range(500):
            q = make("complex_number_as_point", seed=seed, difficulty="medium")
            answer_vals = list(q.answer.values())

            if answer_vals[0] != 0:  # x != 0, so purely real
                real_count += 1
            else:  # x == 0, so purely imaginary
                imag_count += 1

        assert real_count > 0, "No purely-real cases found"
        assert imag_count > 0, "No purely-imaginary cases found"

    def test_sum_has_both_operations(self):
        """Over 500 seeds, hard (sum) variant produces both addition and subtraction."""
        add_count = 0
        sub_count = 0

        for seed in range(500):
            q = make("complex_number_as_point", seed=seed, difficulty="hard")
            # Check if the question text contains + or -
            text = q.text
            if " + " in text:
                add_count += 1
            if "−" in text or " - " in text:
                sub_count += 1

        # At least some of each should be present
        assert add_count > 50, f"Too few addition cases: {add_count}"
        assert sub_count > 50, f"Too few subtraction cases: {sub_count}"

    def test_bare_call_sum_has_enough_distractors(self):
        """Bare call (no difficulty) complex_number_as_point produces >= 2 distractors in >= 85% of 500 seeds."""
        count_with_2_or_more = 0

        for seed in range(500):
            q = make("complex_number_as_point", seed=seed)  # Bare call, uses default variant
            if len(q.distractors) >= 2:
                count_with_2_or_more += 1

        percentage = (count_with_2_or_more / 500) * 100
        assert percentage >= 85, (
            f"Only {percentage:.1f}% of seeds have >= 2 distractors (need >= 85%); count: {count_with_2_or_more}/500"
        )


class TestCartesianPolarConvert:
    """Tests for cartesian_polar_convert generator."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_determinism(self, difficulty, seed):
        """Every seed produces consistent output."""
        q1 = make("cartesian_polar_convert", seed=seed, difficulty=difficulty)
        q2 = make("cartesian_polar_convert", seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_answer_is_dict_pair(self, difficulty, seed):
        """Answer is a dict with exactly 2 numeric values."""
        q = make("cartesian_polar_convert", seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict)
        assert len(q.answer) == 2
        assert all(isinstance(v, (int, float)) for v in q.answer.values())

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_distractors_distinct_from_answer(self, difficulty, seed):
        """Every distractor differs from the answer by more than 0.051 in at least one component."""
        q = make("cartesian_polar_convert", seed=seed, difficulty=difficulty)
        answer_vals = list(q.answer.values())
        for distractor in q.distractors:
            # Parse distractor text
            parts = distractor.split(", ")
            dist_vals = []
            for part in parts:
                val = float(part.split(" = ")[1])
                dist_vals.append(val)

            # Check that at least one component differs by more than 0.051
            assert any(abs(d - a) > 0.051 for d, a in zip(dist_vals, answer_vals)), (
                f"Distractor {distractor} not sufficiently distinct from answer"
            )

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_real_grader_accepts_canonical(self, difficulty, seed):
        """The real grade_pair grader accepts the canonical answer text."""
        q = make("cartesian_polar_convert", seed=seed, difficulty=difficulty)
        result = grade_pair(q, q.answer_text)
        assert result.correct, f"Grader rejected canonical: {q.answer_text}"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_real_grader_rejects_distractors(self, difficulty, seed):
        """The real grade_pair grader rejects all distractors."""
        q = make("cartesian_polar_convert", seed=seed, difficulty=difficulty)
        for distractor in q.distractors:
            result = grade_pair(q, distractor)
            assert not result.correct, f"Grader accepted distractor: {distractor}"

    def test_easy_axis_theta_in_quadrants(self):
        """easy (axis) variant: theta is exactly 0, 90, 180, or 270 degrees."""
        thetas = set()
        for seed in range(100):
            q = make("cartesian_polar_convert", seed=seed, difficulty="easy")
            answer_vals = list(q.answer.values())
            theta = answer_vals[1]
            thetas.add(int(theta))

        assert thetas.issubset({0, 90, 180, 270}), f"Unexpected theta values: {thetas}"
        assert len(thetas) > 1, "Not all axis quadrants represented"

    def test_no_irrational_r_in_easy_and_medium(self):
        """easy (axis) and medium (triple) variants: r is always an exact integer."""
        for difficulty in ["easy", "medium"]:
            for seed in range(100):
                q = make("cartesian_polar_convert", seed=seed, difficulty=difficulty)
                answer_vals = list(q.answer.values())
                r = answer_vals[0]

                # Check that r is an integer
                assert r == int(r), f"{difficulty}: r={r} is not an integer"

    def test_hard_reverse_theta_excludes_axis_and_bands(self):
        """hard (reverse) variant: theta excludes 0, 90, 180, 270 and ±3 degrees around 45, 135, 225, 315."""
        for seed in range(100):
            q = make("cartesian_polar_convert", seed=seed, difficulty="hard")
            # Extract theta from the text
            text = q.text
            import re

            match = re.search(r"θ = (\d+)°", text)
            if match:
                theta = int(match.group(1))

                # Check excluded axis points
                assert theta not in [0, 90, 180, 270], f"theta={theta} is an axis point"

                # Check excluded bands
                excluded_bands = [(45 - 3, 45 + 3), (135 - 3, 135 + 3), (225 - 3, 225 + 3), (315 - 3, 315 + 3)]
                for band in excluded_bands:
                    assert not (band[0] <= theta <= band[1]), (
                        f"theta={theta} is in excluded band around {(band[0] + band[1]) // 2}"
                    )

    def test_no_bad_sign_patterns_in_polar_axis(self):
        """_polar_axis working lines never contain '+ -' or '- -' substrings over >= 200 seeds."""
        bad_plus_minus = re.compile(r"\+ -")
        bad_minus_minus = re.compile(r"- -")

        for seed in range(200):
            q = make("cartesian_polar_convert", seed=seed, difficulty="easy")
            for working_line in q.working:
                assert not bad_plus_minus.search(working_line), f"Seed {seed}: Found '+ -' in: {working_line}"
                assert not bad_minus_minus.search(working_line), f"Seed {seed}: Found '- -' in: {working_line}"

    def test_no_bad_sign_patterns_in_polar_triple(self):
        """_polar_triple working lines never contain '+ -' or '- -' substrings over >= 200 seeds."""
        bad_plus_minus = re.compile(r"\+ -")
        bad_minus_minus = re.compile(r"- -")

        for seed in range(200):
            q = make("cartesian_polar_convert", seed=seed, difficulty="medium")
            for working_line in q.working:
                assert not bad_plus_minus.search(working_line), f"Seed {seed}: Found '+ -' in: {working_line}"
                assert not bad_minus_minus.search(working_line), f"Seed {seed}: Found '- -' in: {working_line}"

    def test_bare_call_triple_has_enough_distractors(self):
        """Bare call (no difficulty) cartesian_polar_convert produces >= 2 distractors in >= 85% of 500 seeds."""
        count_with_2_or_more = 0

        for seed in range(500):
            q = make("cartesian_polar_convert", seed=seed)  # Bare call, uses default variant
            if len(q.distractors) >= 2:
                count_with_2_or_more += 1

        percentage = (count_with_2_or_more / 500) * 100
        assert percentage >= 85, (
            f"Only {percentage:.1f}% of seeds have >= 2 distractors (need >= 85%); count: {count_with_2_or_more}/500"
        )

    def test_skills_load_through_api(self):
        """Both skills load through mathgen.skills.load_skills with correct properties."""
        from mathgen.skills import load_skills

        domains, skills = load_skills()

        # Check skill 1
        skill1 = next((s for s in skills if s.id == "senior-consider-complex-numbers-as-points-in-a-plane"), None)
        assert skill1 is not None, "Skill 1 not found in skills library"
        assert skill1.answer == "pair", f"Skill 1 answer type: {skill1.answer}"
        assert skill1.prerequisites == [], f"Skill 1 prerequisites: {skill1.prerequisites}"
        assert skill1.placement_eligible is False

        # Check skill 2
        skill2 = next((s for s in skills if s.id == "senior-convert-between-cartesian-and-polar-form"), None)
        assert skill2 is not None, "Skill 2 not found in skills library"
        assert skill2.answer == "pair", f"Skill 2 answer type: {skill2.answer}"
        assert skill2.prerequisites == [], f"Skill 2 prerequisites: {skill2.prerequisites}"
        assert skill2.placement_eligible is False
