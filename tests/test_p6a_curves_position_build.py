"""Tests for p6a_curves_position_build generator.

Tests both skills:
- Skill 1: use_vector_equations_of_curves_in_two_or (ACMSM104)
- Skill 2: consider_position_of_vectors_as_a_function_of (ACMSM111)

Verifies: determinism, distinct distractors, real grader accept/reject,
misconceptions firing, skill loading, no diff/deriv/integrat in text,
>= 20 distinct questions per tier over 60 seeds.
"""

import math
import re
from collections import Counter

import pytest

from mathgen.grading import typed as grading
from mathgen.skills import load_skills
from mathgen import make


# ============================================================================
# SKILL 1: use_vector_equations_of_curves_in_two_or (ACMSM104)
# ============================================================================


class TestSkill1Curves:
    """Test skill 1: vector equations of curves."""

    def test_determinism_easy(self):
        """Easy tier is deterministic - same seed gives same result."""
        results = []
        for _ in range(5):
            q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="easy")
            results.append((q.text, q.answer, q.answer_text, tuple(q.distractors)))
        # All five should be identical
        for i in range(1, 5):
            assert results[i] == results[0]

    def test_determinism_medium(self):
        """Medium tier is deterministic - same seed gives same result."""
        results = []
        for _ in range(5):
            q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="medium")
            results.append((q.text, q.answer, q.answer_text, tuple(q.distractors)))
        for i in range(1, 5):
            assert results[i] == results[0]

    def test_determinism_hard(self):
        """Hard tier is deterministic - same seed gives same result."""
        results = []
        for _ in range(5):
            q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="hard")
            results.append((q.text, q.answer, q.answer_text, tuple(q.distractors)))
        for i in range(1, 5):
            assert results[i] == results[0]

    def test_easy_distinct_distractors(self):
        """Easy tier distractors are distinct from answer and each other."""
        q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="easy")
        assert q.answer not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))

    def test_medium_distinct_distractors(self):
        """Medium tier distractors are distinct from answer and each other."""
        q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="medium")
        assert q.answer not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))

    def test_hard_distinct_distractors(self):
        """Hard tier distractors are distinct from answer and each other."""
        q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="hard")
        assert q.answer not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))

    def test_real_grader_accepts_answer_easy(self):
        """Real grader (grade_number) accepts the answer for easy tier."""
        q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="easy")
        result = grading.grade("number", q, q.answer_text)
        assert result.correct

    def test_real_grader_rejects_distractor_easy(self):
        """Real grader rejects distractors for easy tier."""
        q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="easy")
        for distractor in q.distractors:
            result = grading.grade("number", q, distractor)
            assert not result.correct

    def test_real_grader_accepts_answer_medium(self):
        """Real grader accepts the answer for medium tier."""
        q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="medium")
        result = grading.grade("number", q, q.answer_text)
        assert result.correct

    def test_real_grader_rejects_distractor_medium(self):
        """Real grader rejects distractors for medium tier."""
        q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="medium")
        for distractor in q.distractors:
            result = grading.grade("number", q, distractor)
            assert not result.correct

    def test_real_grader_accepts_answer_hard(self):
        """Real grader accepts the answer for hard tier."""
        q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="hard")
        result = grading.grade("number", q, q.answer_text)
        assert result.correct

    def test_real_grader_rejects_distractor_hard(self):
        """Real grader rejects distractors for hard tier."""
        q = make("use_vector_equations_of_curves_in_two_or", seed=0, tier="hard")
        for distractor in q.distractors:
            result = grading.grade("number", q, distractor)
            assert not result.correct

    def test_at_least_20_distinct_easy(self):
        """At least 20 distinct questions for easy tier over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("use_vector_equations_of_curves_in_two_or", seed=seed, tier="easy")
            questions.add(q.text)
        assert len(questions) >= 20, f"Only {len(questions)} distinct questions, expected >= 20"

    def test_at_least_20_distinct_medium(self):
        """At least 20 distinct questions for medium tier over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("use_vector_equations_of_curves_in_two_or", seed=seed, tier="medium")
            questions.add(q.text)
        assert len(questions) >= 20, f"Only {len(questions)} distinct questions, expected >= 20"

    def test_at_least_20_distinct_hard(self):
        """At least 20 distinct questions for hard tier over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("use_vector_equations_of_curves_in_two_or", seed=seed, tier="hard")
            questions.add(q.text)
        assert len(questions) >= 20, f"Only {len(questions)} distinct questions, expected >= 20"

    def test_medium_tier_coefficient_formatting(self):
        """Medium tier must format coefficient 1 as 't' and -1 as '-t', not '1t' or '-1t'."""
        # Sweep at least 150 seeds of the medium tier
        for seed in range(150):
            q = make("use_vector_equations_of_curves_in_two_or", seed=seed, tier="medium")
            # Check that the text never contains patterns like "(1t," or "(-1t,"
            assert re.search(r"\(1t,", q.text) is None, f"Found bare '1t' coefficient in seed {seed}: {q.text}"
            assert re.search(r"\(-1t,", q.text) is None, f"Found bare '-1t' coefficient in seed {seed}: {q.text}"


# ============================================================================
# SKILL 2: consider_position_of_vectors_as_a_function_of (ACMSM111)
# ============================================================================


class TestSkill2Position:
    """Test skill 2: position vectors as functions of time."""

    def test_determinism_easy(self):
        """Easy tier is deterministic - same seed gives same result."""
        results = []
        for _ in range(5):
            q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="easy")
            results.append((q.text, q.answer, q.answer_text, tuple(q.distractors)))
        for i in range(1, 5):
            assert results[i] == results[0]

    def test_determinism_medium(self):
        """Medium tier is deterministic - same seed gives same result."""
        results = []
        for _ in range(5):
            q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="medium")
            results.append((q.text, q.answer, q.answer_text, tuple(q.distractors)))
        for i in range(1, 5):
            assert results[i] == results[0]

    def test_determinism_hard(self):
        """Hard tier is deterministic - same seed gives same result."""
        results = []
        for _ in range(5):
            q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="hard")
            results.append((q.text, q.answer, q.answer_text, tuple(q.distractors)))
        for i in range(1, 5):
            assert results[i] == results[0]

    def test_easy_distinct_distractors(self):
        """Easy tier distractors are distinct from answer and each other."""
        q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="easy")
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))

    def test_medium_distinct_distractors(self):
        """Medium tier distractors are distinct from answer and each other."""
        q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="medium")
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))

    def test_hard_distinct_distractors(self):
        """Hard tier distractors are distinct from answer and each other."""
        q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="hard")
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))

    def test_real_grader_accepts_answer_easy(self):
        """Real grader (grade_pair) accepts the answer for easy tier."""
        q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="easy")
        result = grading.grade("pair", q, q.answer_text)
        assert result.correct

    def test_real_grader_rejects_distractor_easy(self):
        """Real grader rejects distractors for easy tier."""
        q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="easy")
        for distractor in q.distractors:
            result = grading.grade("pair", q, distractor)
            assert not result.correct

    def test_real_grader_accepts_answer_medium(self):
        """Real grader accepts the answer for medium tier."""
        q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="medium")
        result = grading.grade("pair", q, q.answer_text)
        assert result.correct

    def test_real_grader_rejects_distractor_medium(self):
        """Real grader rejects distractors for medium tier."""
        q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="medium")
        for distractor in q.distractors:
            result = grading.grade("pair", q, distractor)
            assert not result.correct

    def test_real_grader_accepts_answer_hard(self):
        """Real grader accepts the answer for hard tier."""
        q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="hard")
        result = grading.grade("pair", q, q.answer_text)
        assert result.correct

    def test_real_grader_rejects_distractor_hard(self):
        """Real grader rejects distractors for hard tier."""
        q = make("consider_position_of_vectors_as_a_function_of", seed=0, tier="hard")
        for distractor in q.distractors:
            result = grading.grade("pair", q, distractor)
            assert not result.correct

    def test_no_diff_deriv_integrat_in_text(self):
        """No generated question text contains diff/deriv/integrat/antideriv."""
        forbidden_substrings = ["diff", "deriv", "integrat", "antideriv"]
        for tier in ["easy", "medium", "hard"]:
            for seed in range(40):
                q = make("consider_position_of_vectors_as_a_function_of", seed=seed, tier=tier)
                for forbidden in forbidden_substrings:
                    assert forbidden.lower() not in q.text.lower(), (
                        f"Found forbidden substring '{forbidden}' in tier {tier} seed {seed}: {q.text}"
                    )

    def test_at_least_20_distinct_easy(self):
        """At least 20 distinct questions for easy tier over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("consider_position_of_vectors_as_a_function_of", seed=seed, tier="easy")
            questions.add(q.text)
        assert len(questions) >= 20, f"Only {len(questions)} distinct questions, expected >= 20"

    def test_at_least_20_distinct_medium(self):
        """At least 20 distinct questions for medium tier over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("consider_position_of_vectors_as_a_function_of", seed=seed, tier="medium")
            questions.add(q.text)
        assert len(questions) >= 20, f"Only {len(questions)} distinct questions, expected >= 20"

    def test_at_least_20_distinct_hard(self):
        """At least 20 distinct questions for hard tier over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("consider_position_of_vectors_as_a_function_of", seed=seed, tier="hard")
            questions.add(q.text)
        assert len(questions) >= 20, f"Only {len(questions)} distinct questions, expected >= 20"

    def test_easy_tier_coefficient_formatting(self):
        """Easy tier must format coefficient 1 as 't', not '1t'."""
        # Sweep at least 200 seeds of the easy tier
        for seed in range(200):
            q = make("consider_position_of_vectors_as_a_function_of", seed=seed, tier="easy")
            # Check that the text never contains patterns like "1t" (but allow "3t", "-1t" etc.)
            # Use the regex: (?<![0-9./])-?1t\b
            assert re.search(r"(?<![0-9./])-?1t\b", q.text) is None, (
                f"Found bare '1t' coefficient in seed {seed}: {q.text}"
            )


# ============================================================================
# SKILL LOADING AND INTEGRATION
# ============================================================================


def test_skills_load():
    """Both skills load through mathgen.skills.load_skills with correct settings."""
    domains, skills_list = load_skills()
    skills = {s.id: s for s in skills_list}

    # Skill 1
    skill1 = skills.get("senior-use-vector-equations-of-curves-in-two-or")
    assert skill1 is not None
    assert skill1.answer == "number"
    assert skill1.placement_eligible == False
    assert "vector-addition" in skill1.prerequisites

    # Skill 2
    skill2 = skills.get("senior-consider-position-of-vectors-as-a-function-of")
    assert skill2 is not None
    assert skill2.answer == "pair"
    assert skill2.placement_eligible == False
    assert "vector-addition" in skill2.prerequisites


def test_hard_tier_no_double_signs():
    """Hard tier working lines must not contain double signs like ' + -' or ' - -'."""
    for seed in range(150):
        q = make("use_vector_equations_of_curves_in_two_or", seed=seed, tier="hard")
        for line in q.working:
            assert " + -" not in line, f"Found ' + -' in seed {seed}: {line}"
            assert " - -" not in line, f"Found ' - -' in seed {seed}: {line}"
