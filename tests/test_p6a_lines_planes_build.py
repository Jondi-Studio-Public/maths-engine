"""Tests the vector equation of a line and the Cartesian equation of a plane generators.

Lines planes build."""

import re
import pytest
from fractions import Fraction

from mathgen.grading.typed import grade_number, grade_pair
from mathgen.skills import load_skills
from mathgen import make


class TestDetermineVectorEquationOfStraightLine:
    """Tests for determine_a_vector_equation_of_a_straight_line."""

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_determinism_40_seeds(self, tier):
        """Verify determinism across 40 seeds at each difficulty."""
        results = []
        for seed in range(40):
            q = make(topic="determine_a_vector_equation_of_a_straight_line", difficulty=tier, seed=seed)
            results.append((q.text, q.answer_text))

        # All results should be unique across seeds (high diversity)
        assert len(set(results)) >= 20, f"{tier}: fewer than 20 distinct questions over 40 seeds"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_distractors_non_empty_distinct(self, tier):
        """Verify distractors are non-empty, distinct, and never equal answer."""
        for seed in range(20):
            q = make(topic="determine_a_vector_equation_of_a_straight_line", difficulty=tier, seed=seed)
            assert q.distractors, f"{tier} seed {seed}: no distractors"
            assert len(q.distractors) > 0, f"{tier} seed {seed}: empty distractors"
            assert len(set(q.distractors)) == len(q.distractors), f"{tier} seed {seed}: duplicate distractors"
            assert q.answer_text not in q.distractors, f"{tier} seed {seed}: answer is in distractors"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_real_grader_accepts_answer_rejects_distractors(self, tier):
        """Test real grader (mathgen.grading.grade_pair) accepts answer, rejects all distractors."""
        for seed in range(10):
            q = make(topic="determine_a_vector_equation_of_a_straight_line", difficulty=tier, seed=seed)
            # Grade the answer
            result_answer = grade_pair(q, q.answer_text)
            assert result_answer.correct, f"{tier} seed {seed}: grader rejected correct answer '{q.answer_text}'"

            # Grade each distractor
            for distractor in q.distractors:
                result_dist = grade_pair(q, distractor)
                assert not result_dist.correct, f"{tier} seed {seed}: grader accepted distractor '{distractor}'"

    def test_skill_loads_with_correct_metadata(self):
        """Test skill loads through API with correct metadata."""
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-determine-a-vector-equation-of-a-straight-line"), None)
        assert skill is not None, "Skill not found in loaded skills"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert "vector-addition" in skill.prerequisites
        # Check all three tiers are present
        topics_by_name = {s.topic for s in skill.sources}
        assert "determine_a_vector_equation_of_a_straight_line" in topics_by_name

    def test_answer_is_pair_of_integers(self):
        """Verify answer is a pair of floats (representing integer components)."""
        for seed in range(10):
            for tier in ["easy", "medium", "hard"]:
                q = make(topic="determine_a_vector_equation_of_a_straight_line", difficulty=tier, seed=seed)
                assert isinstance(q.answer, dict), f"{tier} seed {seed}: answer not dict"
                assert "x" in q.answer and "y" in q.answer, f"{tier} seed {seed}: answer missing x or y"
                # Should be floats (representing integers)
                assert isinstance(q.answer["x"], float), f"{tier} seed {seed}: x not float"
                assert isinstance(q.answer["y"], float), f"{tier} seed {seed}: y not float"
                # Check they represent integers
                assert q.answer["x"] == int(q.answer["x"]), f"{tier} seed {seed}: x not integer-valued"
                assert q.answer["y"] == int(q.answer["y"]), f"{tier} seed {seed}: y not integer-valued"

    def test_hard_tier_direction_pinning_phrase(self):
        """Verify hard-tier line questions contain explicit direction-pinning phrase."""
        for seed in range(100):
            q = make(topic="determine_a_vector_equation_of_a_straight_line", difficulty="hard", seed=seed)
            # Check for direction-pinning phrases: "minus A" or "from A to"
            has_direction_phrase = "minus A" in q.text or "from A to" in q.text
            assert has_direction_phrase, f"hard seed {seed}: no direction-pinning phrase found in: {q.text}"

    def test_working_steps_and_credited_instinct_note(self):
        """Test that working steps are present and credited-instinct notes are correct.

        For each tier and seed:
        - Assert working list exists and has at least 3 steps
        - Parse answer from answer_text (format: "x = ..., y = ...")
        - For each distractor, check if it's a positive scalar multiple of answer
        - If yes: assert last line of working contains "Note:"
        - If no: assert no line of working contains "Note:"
        - Must test both branches (with and without scalar multiple distractors)
        """
        found_with_note = False
        found_without_note = False

        for tier in ["easy", "medium", "hard"]:
            for seed in range(100):  # Wide seed range to catch both cases
                q = make(topic="determine_a_vector_equation_of_a_straight_line", difficulty=tier, seed=seed)

                # Check working list exists and is non-empty
                assert hasattr(q, "working"), f"{tier} seed {seed}: no working attribute"
                assert q.working is not None, f"{tier} seed {seed}: working is None"
                assert isinstance(q.working, list), f"{tier} seed {seed}: working not a list"
                assert len(q.working) >= 3, f"{tier} seed {seed}: fewer than 3 working steps"

                # Parse answer: format "x = ..., y = ..."
                answer_parts = q.answer_text.replace("x = ", "").replace("y = ", "").split(", ")
                assert len(answer_parts) == 2, f"{tier} seed {seed}: cannot parse answer_text: {q.answer_text}"
                try:
                    ans_x = int(answer_parts[0])
                    ans_y = int(answer_parts[1])
                except ValueError:
                    pytest.fail(f"{tier} seed {seed}: answer_text has non-integer components: {q.answer_text}")

                # Check each distractor for positive scalar multiple relationship
                has_scalar_multiple = False
                for distractor_str in q.distractors:
                    dist_parts = distractor_str.replace("x = ", "").replace("y = ", "").split(", ")
                    if len(dist_parts) == 2:
                        try:
                            dist_x = int(dist_parts[0])
                            dist_y = int(dist_parts[1])
                            # Check colinearity and same direction
                            if ans_x * dist_y - ans_y * dist_x == 0 and ans_x * dist_x + ans_y * dist_y > 0:
                                has_scalar_multiple = True
                                break
                        except ValueError:
                            pass

                # Check working list has/lacks "Note:" as appropriate
                has_note = any("Note:" in step for step in q.working)

                if has_scalar_multiple:
                    assert has_note, (
                        f"{tier} seed {seed}: distractor is scalar multiple but no Note in working: {q.working}"
                    )
                    # Also verify last line contains the note
                    assert "Note:" in q.working[-1], (
                        f"{tier} seed {seed}: Note found but not in last line of working: {q.working}"
                    )
                    found_with_note = True
                else:
                    assert not has_note, (
                        f"{tier} seed {seed}: no scalar multiple distractor but has Note in working: {q.working}"
                    )
                    found_without_note = True

        # Verify we tested both branches
        assert found_with_note, "Did not find any seed/tier with a scalar multiple distractor"
        assert found_without_note, "Did not find any seed/tier without a scalar multiple distractor"


class TestDetermineCartesianEquationOfPlane:
    """Tests for determine_vector_and_cartesian_equations_of_a_plane."""

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_determinism_40_seeds(self, tier):
        """Verify determinism across 40 seeds at each difficulty."""
        results = []
        for seed in range(40):
            q = make(topic="determine_vector_and_cartesian_equations_of_a_plane", difficulty=tier, seed=seed)
            results.append((q.text, q.answer_text))

        # All results should be unique across seeds (high diversity)
        assert len(set(results)) >= 20, f"{tier}: fewer than 20 distinct questions over 40 seeds"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_distractors_non_empty_distinct(self, tier):
        """Verify distractors are non-empty, distinct, and never equal answer."""
        for seed in range(20):
            q = make(topic="determine_vector_and_cartesian_equations_of_a_plane", difficulty=tier, seed=seed)
            assert q.distractors, f"{tier} seed {seed}: no distractors"
            assert len(q.distractors) > 0, f"{tier} seed {seed}: empty distractors"
            assert len(set(q.distractors)) == len(q.distractors), f"{tier} seed {seed}: duplicate distractors"
            assert q.answer_text not in q.distractors, f"{tier} seed {seed}: answer is in distractors"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_real_grader_accepts_answer_rejects_distractors(self, tier):
        """Test real grader (mathgen.grading.grade_number) accepts answer, rejects all distractors."""
        for seed in range(10):
            q = make(topic="determine_vector_and_cartesian_equations_of_a_plane", difficulty=tier, seed=seed)
            # Grade the answer
            result_answer = grade_number(q, q.answer_text)
            assert result_answer.correct, f"{tier} seed {seed}: grader rejected correct answer '{q.answer_text}'"

            # Grade each distractor
            for distractor in q.distractors:
                result_dist = grade_number(q, distractor)
                assert not result_dist.correct, f"{tier} seed {seed}: grader accepted distractor '{distractor}'"

    def test_skill_loads_with_correct_metadata(self):
        """Test skill loads through API with correct metadata."""
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-determine-vector-and-cartesian-equations-of-a-plane"), None)
        assert skill is not None, "Skill not found in loaded skills"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert "vector-addition" in skill.prerequisites
        # Check all three tiers are present
        topics_by_name = {s.topic for s in skill.sources}
        assert "determine_vector_and_cartesian_equations_of_a_plane" in topics_by_name

    def test_answer_is_integer(self):
        """Verify answer is an integer (as a float with no decimal part)."""
        for seed in range(10):
            for tier in ["easy", "medium", "hard"]:
                q = make(topic="determine_vector_and_cartesian_equations_of_a_plane", difficulty=tier, seed=seed)
                # Answer should be a float
                assert isinstance(q.answer, float), f"{tier} seed {seed}: answer not float"
                # Should represent an integer
                assert q.answer == int(q.answer), f"{tier} seed {seed}: answer not integer-valued"
                # answer_text should be plain integer with no decimal point
                assert "." not in q.answer_text, (
                    f"{tier} seed {seed}: answer_text contains decimal point: {q.answer_text}"
                )

    def test_no_invalid_sign_formatting(self):
        """Verify text never contains '+ -' or '- -' patterns."""
        for seed in range(20):
            for tier in ["easy", "medium", "hard"]:
                q = make(topic="determine_vector_and_cartesian_equations_of_a_plane", difficulty=tier, seed=seed)
                assert "+ -" not in q.text, f"{tier} seed {seed}: text contains '+ -': {q.text}"
                assert "- -" not in q.text, f"{tier} seed {seed}: text contains '- -': {q.text}"

    def test_hard_tier_no_bare_coefficient_one(self):
        """Verify hard tier never renders bare coefficient 1 or -1 as '1x', '-1x', '1y', '-1y'."""
        for seed in range(150):
            q = make(topic="determine_vector_and_cartesian_equations_of_a_plane", difficulty="hard", seed=seed)
            # Check for bare coefficient-1 display patterns
            # Pattern matches: "1x", "-1x", "1y", "-1y", "+ 1y", "- 1y", etc.
            # (anywhere not preceded by a digit)
            match = re.search(r"(?<!\d)-?1[xy]", q.text)
            assert match is None, f"hard seed {seed}: text contains bare coefficient 1 display: {q.text}"

    def test_medium_tier_no_zero_normal_vector(self):
        """Verify medium-tier plane questions never have zero normal vector (0, 0, 0)."""
        for seed in range(3000):
            q = make(topic="determine_vector_and_cartesian_equations_of_a_plane", difficulty="medium", seed=seed)
            assert "(0, 0, 0)" not in q.text, f"medium seed {seed}: zero normal vector found in: {q.text}"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_working_steps_non_empty_and_value_matches_answer(self, tier):
        """Verify working steps exist, are non-empty, and final numeric value matches answer.

        For each tier and seed (>= 60 seeds per tier):
        - Assert working attribute exists and is a non-empty list
        - Parse the final numeric value from the last line (after '=')
        - For easy/medium: expect "d = <value>" format
        - For hard: expect "c = <value>" format (possibly with division shown)
        - Verify the parsed value equals question.answer_text
        """
        for seed in range(60):
            q = make(topic="determine_vector_and_cartesian_equations_of_a_plane", difficulty=tier, seed=seed)
            # Assert working exists and is non-empty list
            assert hasattr(q, "working"), f"{tier} seed {seed}: no working attribute"
            assert q.working is not None, f"{tier} seed {seed}: working is None"
            assert isinstance(q.working, list), f"{tier} seed {seed}: working not a list"
            assert len(q.working) > 0, f"{tier} seed {seed}: working is empty"

            # Parse final numeric value from last line
            last_line = q.working[-1]

            # Extract the final number after the last '='
            if "=" in last_line:
                # Get everything after the last '='
                parts = last_line.rsplit("=", 1)
                if len(parts) == 2:
                    final_part = parts[1].strip()
                    # For hard tier, might be like "3" or "18 / -4 = 3", extract the very last number
                    # Split by whitespace and take the last token
                    tokens = final_part.split()
                    if tokens:
                        final_value_str = tokens[-1]
                        try:
                            final_value = int(final_value_str)
                            expected_value = int(q.answer_text)
                            assert final_value == expected_value, (
                                f"{tier} seed {seed}: final value {final_value} != answer {expected_value}; last line: {last_line}"
                            )
                        except (ValueError, IndexError):
                            pytest.fail(f"{tier} seed {seed}: cannot parse final value from last line: {last_line}")
                    else:
                        pytest.fail(f"{tier} seed {seed}: last line has no content after '=': {last_line}")
                else:
                    pytest.fail(f"{tier} seed {seed}: cannot split last line by '=': {last_line}")
            else:
                pytest.fail(f"{tier} seed {seed}: last line has no '=': {last_line}")
