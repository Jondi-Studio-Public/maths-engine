"""Focused contract tests for f10-additive-multiplicative-patterns and
f10-number-algorithms-decisions."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_ordered_list, grade_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5c_patterns_algorithms_build import (
    _verify_additive_pattern,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# ============================================================================
# SKILL 1: f10-additive-multiplicative-patterns
# ============================================================================


class TestAdditivePatterns:
    """Tests for additive/multiplicative patterns (one missing term)."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_deterministic_across_seeds(self, difficulty):
        """Each seed produces the same question on replay."""
        for seed in SEEDS:
            q1 = make("additive_multiplicative_patterns", seed=seed, difficulty=difficulty)
            q2 = make("additive_multiplicative_patterns", seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_distractors_distinct_and_non_empty(self, difficulty):
        """Distractors are distinct, non-empty, never equal to answer."""
        for seed in SEEDS:
            q = make("additive_multiplicative_patterns", seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, list)
            assert len(q.answer) == 1, f"Expected single-item list, got {q.answer}"
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_correct_answer_grades_correct(self, difficulty):
        """Real grader accepts canonical answer."""
        for seed in SEEDS:
            q = make("additive_multiplicative_patterns", seed=seed, difficulty=difficulty)
            result = grade_ordered_list(q, q.answer_text)
            assert result.correct, f"Seed {seed}: {q.answer_text} should grade correct"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_every_distractor_grades_incorrect(self, difficulty):
        """Real grader rejects every distractor."""
        for seed in SEEDS:
            q = make("additive_multiplicative_patterns", seed=seed, difficulty=difficulty)
            for i, distractor in enumerate(q.distractors):
                result = grade_ordered_list(q, distractor)
                assert not result.correct, f"Seed {seed}: distractor {i} ({distractor}) should grade incorrect"

    def test_easy_always_ends_at_position(self):
        """Easy: shows 3 terms, asks for 1 (at position 3)."""
        for seed in SEEDS:
            q = make("additive_multiplicative_patterns", seed=seed, difficulty="easy")
            # Question should say "next number" (singular)
            assert "next number" in q.text.lower()
            # Answer is a single integer
            assert len(q.answer) == 1

    def test_medium_always_ends_at_position(self):
        """Medium: shows 3 decreasing terms, asks for 1 (at position 3)."""
        for seed in SEEDS:
            q = make("additive_multiplicative_patterns", seed=seed, difficulty="medium")
            assert "next number" in q.text.lower()
            assert len(q.answer) == 1

    def test_hard_has_interior_gap(self):
        """Hard: shows 4-5 terms with interior blank, asks for 1."""
        for seed in SEEDS:
            q = make("additive_multiplicative_patterns", seed=seed, difficulty="hard")
            # Question should say "missing number"
            assert "missing number" in q.text.lower()
            # Count underscores: should be exactly 1
            assert q.text.count("__") == 1
            assert len(q.answer) == 1

    def test_independent_check_fires_on_corrupted(self):
        """Independent check recomputes and fires on corrupted answer."""
        first, last, idx_first, idx_last = 10, 22, 0, 2
        missing_idx, missing_val = 3, 28
        _verify_additive_pattern(first, last, idx_first, idx_last, missing_val, missing_idx)

        # Corrupt the value
        corrupted_val = missing_val + 1
        with pytest.raises(VerificationError):
            _verify_additive_pattern(first, last, idx_first, idx_last, corrupted_val, missing_idx)

    def test_skill_loads_correctly(self):
        """Skill loads with correct answer type and prerequisites."""
        _, all_skills = load_skills(ROOT / "skills")
        by_id = {s.id: s for s in all_skills}
        skill = by_id.get("f10-additive-multiplicative-patterns")
        assert skill is not None
        assert skill.answer == "ordered_list"
        assert skill.prerequisites == ["common-difference", "generate-sequence-from-rule"]
        assert skill.placement_eligible is False


# ============================================================================
# SKILL 2: f10-number-algorithms-decisions
# ============================================================================


class TestAlgorithmDecisions:
    """Tests for algorithm tracing with decisions."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_deterministic_across_seeds(self, difficulty):
        """Each seed produces the same question on replay."""
        for seed in SEEDS:
            q1 = make("number_algorithms_decisions", seed=seed, difficulty=difficulty)
            q2 = make("number_algorithms_decisions", seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_distractors_distinct_and_non_empty(self, difficulty):
        """Distractors are distinct, non-empty, never equal to answer."""
        for seed in SEEDS:
            q = make("number_algorithms_decisions", seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, list)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1  # At least 1 distractor

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_correct_answer_grades_correct(self, difficulty):
        """Real grader (grade_list, order-insensitive) accepts canonical answer."""
        for seed in SEEDS:
            q = make("number_algorithms_decisions", seed=seed, difficulty=difficulty)
            result = grade_list(q, q.answer_text)
            assert result.correct, f"Seed {seed}: {q.answer_text} should grade correct"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_every_distractor_grades_incorrect(self, difficulty):
        """Real grader rejects every distractor."""
        for seed in SEEDS:
            q = make("number_algorithms_decisions", seed=seed, difficulty=difficulty)
            for i, distractor in enumerate(q.distractors):
                result = grade_list(q, distractor)
                assert not result.correct, f"Seed {seed}: distractor {i} ({distractor}) should grade incorrect"

    def test_easy_has_2_outputs(self):
        """Easy: 2 inputs, 2 outputs."""
        for seed in SEEDS:
            q = make("number_algorithms_decisions", seed=seed, difficulty="easy")
            assert len(q.answer) == 2, f"Expected 2 outputs, got {len(q.answer)}: {q.answer}"

    def test_medium_has_3_outputs(self):
        """Medium: 3 inputs, 3 outputs."""
        for seed in SEEDS:
            q = make("number_algorithms_decisions", seed=seed, difficulty="medium")
            assert len(q.answer) == 3, f"Expected 3 outputs, got {len(q.answer)}: {q.answer}"

    def test_hard_has_3_outputs(self):
        """Hard: 3 inputs, 3 outputs."""
        for seed in SEEDS:
            q = make("number_algorithms_decisions", seed=seed, difficulty="hard")
            assert len(q.answer) == 3, f"Expected 3 outputs, got {len(q.answer)}: {q.answer}"

    def test_easy_covers_both_branches(self):
        """Easy: at least one even and one odd input (forced)."""
        for seed in SEEDS:
            q = make("number_algorithms_decisions", seed=seed, difficulty="easy")
            # Extract inputs from question text (rough parse)
            import re

            match = re.search(r"inputs?\s+(\d+),?\s+(\d+)", q.text)
            if match:
                inputs = [int(match.group(1)), int(match.group(2))]
                has_even = any(x % 2 == 0 for x in inputs)
                has_odd = any(x % 2 != 0 for x in inputs)
                assert has_even and has_odd, f"Both branches not forced: {inputs}"

    def test_medium_covers_both_branches(self):
        """Medium: 3 inputs force both branches."""
        for seed in SEEDS:
            q = make("number_algorithms_decisions", seed=seed, difficulty="medium")
            # At least one even and one odd (forced)
            import re

            match = re.search(r"inputs?\s+(\d+),?\s+(\d+),?\s+(\d+)", q.text)
            if match:
                inputs = [int(match.group(i)) for i in range(1, 4)]
                has_even = any(x % 2 == 0 for x in inputs)
                has_odd = any(x % 2 != 0 for x in inputs)
                assert has_even and has_odd, f"Both branches not forced: {inputs}"

    def test_skill_loads_correctly(self):
        """Skill loads with correct answer type and prerequisites."""
        _, all_skills = load_skills(ROOT / "skills")
        by_id = {s.id: s for s in all_skills}
        skill = by_id.get("f10-number-algorithms-decisions")
        assert skill is not None
        assert skill.answer == "list"
        assert skill.prerequisites == ["order-of-operations"]
        assert skill.placement_eligible is False

    def test_output_multiset_not_ordered(self):
        """List grader is order-insensitive (multiset)."""
        for seed in SEEDS:
            q = make("number_algorithms_decisions", seed=seed, difficulty="easy")
            # If the answer is e.g. "3, 5", test that "5, 3" also grades correct
            # (since grade_list is order-insensitive)
            outputs = q.answer
            if len(outputs) >= 2 and outputs[0] != outputs[1]:
                # Swap the first two and see if it still passes
                reversed_text = ", ".join(str(o) for o in reversed(outputs))
                result = grade_list(q, reversed_text)
                # grade_list should accept this since it's multiset comparison
                assert result.correct, f"Seed {seed}: reversed outputs {reversed_text} should grade correct (multiset)"

    def test_integer_outputs_and_trace_order_wide_seed_range(self):
        """Regression test: all outputs must be integers, and answer_text must
        match the trace order of inputs stated in the question text.

        This test ensures that:
        1. Every element of question.answer is a Python int (no float),
           so integer division never silently produces a non-exact value
        2. The answer_text string, split on ", ", matches the outputs computed
           by tracing the stated rule on inputs in the order the question states them
        3. Both choice branches are observed across 300+ seeds at each tier
        """
        import re

        def reference_algorithm(rule_name: str, choice_val: int, tier_val: str, inp: int) -> int:
            """Hand-written reference implementation for each rule."""
            if tier_val == "easy":
                if choice_val == 0:
                    return inp // 2 if inp % 2 == 0 else inp + 5
                elif choice_val == 1:
                    return inp - 3 if inp % 2 == 0 else inp * 2
                else:  # choice == 2
                    return inp + 4 if inp % 2 == 0 else inp - 1
            elif tier_val == "medium":
                if choice_val == 0:
                    return inp // 2 if inp % 2 == 0 else inp + 5
                else:  # choice == 1
                    return inp - 3 if inp % 2 == 0 else inp * 2
            else:  # hard
                if choice_val == 0:
                    if inp % 2 == 0:
                        return inp // 2 + 3
                    else:
                        return inp * 2 - 1
                else:  # choice == 1
                    if inp % 2 == 0:
                        return (inp - 2) * 3
                    else:
                        return (inp + 5) // 2
            raise ValueError(f"Invalid tier or choice: {tier_val}, {choice_val}")

        # Track observed choices across wide seed range
        observed_choices = {"easy": set(), "medium": set(), "hard": set()}
        seen_seeds = 0
        max_seeds_to_check = 350  # At least 300+ to ensure both choices observed

        for seed in range(max_seeds_to_check):
            for tier in ["easy", "medium", "hard"]:
                q = make("number_algorithms_decisions", seed=seed, difficulty=tier)

                # 1. All elements of answer must be integers
                for i, output in enumerate(q.answer):
                    assert isinstance(output, int), (
                        f"Seed {seed} {tier}: answer[{i}]={output} is not int (type={type(output).__name__})"
                    )

                # 2. Extract inputs from question text and recompute expected outputs
                # Pattern: "For inputs X, Y, ..." or "For inputs X, Y, Z, ..."
                inputs_match = re.search(r"For inputs?\s+([\d,\s]+?),\s+what are the outputs\?", q.text)
                assert inputs_match, f"Seed {seed} {tier}: could not parse inputs from question text: {q.text}"

                inputs_str = inputs_match.group(1)
                inputs = [int(x.strip()) for x in inputs_str.split(",")]
                assert len(inputs) == len(q.answer), (
                    f"Seed {seed} {tier}: input count {len(inputs)} != answer count {len(q.answer)}"
                )

                # Extract the choice value from the question text or by inference
                # We infer it by checking which rule matches the observed output
                choice_val = None
                for choice_candidate in range(2 if tier != "easy" else 3):
                    expected_outputs = [reference_algorithm("algorithm", choice_candidate, tier, inp) for inp in inputs]
                    if expected_outputs == q.answer:
                        choice_val = choice_candidate
                        break

                assert choice_val is not None, (
                    f"Seed {seed} {tier}: could not match observed answer {q.answer} to any choice"
                )

                observed_choices[tier].add(choice_val)

                # 3. Verify answer_text order matches input/trace order
                answer_text_values = [int(x.strip()) for x in q.answer_text.split(",")]
                assert answer_text_values == q.answer, (
                    f"Seed {seed} {tier}: answer_text order {answer_text_values} does not match trace order {q.answer}"
                )

                seen_seeds += 1

        # Ensure both choice branches were observed at each tier
        assert observed_choices["easy"] == {0, 1, 2}, f"Easy tier: not all choices observed: {observed_choices['easy']}"
        assert observed_choices["medium"] == {0, 1}, (
            f"Medium tier: not all choices observed: {observed_choices['medium']}"
        )
        assert observed_choices["hard"] == {0, 1}, f"Hard tier: not all choices observed: {observed_choices['hard']}"

        assert seen_seeds >= max_seeds_to_check, (
            f"Only checked {seen_seeds} seeds, need at least 300+ to ensure both branches observed"
        )
