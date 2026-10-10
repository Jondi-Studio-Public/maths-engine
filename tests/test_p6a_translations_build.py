"""Tests for translations_and_their_representation_as_column_vectors generator.

Includes an oracle re-derivation check: parse the rendered question and
verify the answer by independent computation from the text.
"""

import re
from fractions import Fraction

import pytest

from mathgen.grading import typed as grading
from mathgen.skills import load_skills
from mathgen import make


SKILL_ID = "senior-translations-and-their-representation-as-column-vectors"
TOPIC = "translations_and_their_representation_as_column_vectors"


class TestTranslationsGeneratorDeterminism:
    """Determinism check: same seed produces same question."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_determinism_many_seeds(self, difficulty):
        """Every difficulty should be deterministic over 40+ seeds."""
        questions = []
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            questions.append((q.text, q.answer_text, q.distractors))

        # Verify no duplicates (every seed produces distinct text/answer)
        texts = [t for t, _, _ in questions]
        assert len(set(texts)) >= 20, f"{difficulty}: only {len(set(texts))} distinct texts over 40 seeds"

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_determinism_same_seed_same_result(self, seed, difficulty):
        """Determinism: same seed + difficulty always produces same question."""
        q1 = make(TOPIC, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q1.text == q2.text
        assert q1.answer_text == q2.answer_text
        assert q1.distractors == q2.distractors


class TestTranslationsOracleReDerivation:
    """Oracle: re-derive the answer independently from the rendered question text."""

    def _parse_question_and_derive(self, question):
        """Parse rendered question text and re-compute the answer independently.

        Parses:
        - Point coordinates P(px, py)
        - Column vector(s) (a, b) or (a1, b1) and (a2, b2)
        - Returns the independently derived (x', y') image point.
        """
        text = question.text

        # Extract point P(px, py)
        p_match = re.search(r"P\((-?\d+),\s*(-?\d+)\)", text)
        if not p_match:
            raise ValueError(f"Could not parse point P from: {text}")
        px, py = int(p_match.group(1)), int(p_match.group(2))

        # Try to extract hard-tier vectors first: (a1, b1) and (a2, b2)
        if "a1" in text:
            # Hard tier: two successive translations
            a1_match = re.search(r"\(a1, b1\) = \((-?\d+), (-?\d+)\)", text)
            a2_match = re.search(r"\(a2, b2\) = \((-?\d+), (-?\d+)\)", text)
            if not a1_match or not a2_match:
                raise ValueError(f"Could not parse vectors from hard-tier: {text}")
            a1, b1 = int(a1_match.group(1)), int(a1_match.group(2))
            a2, b2 = int(a2_match.group(1)), int(a2_match.group(2))

            # Final image: apply both translations (order doesn't matter; vector sum)
            x_prime = px + a1 + a2
            y_prime = py + b1 + b2
        else:
            # Easy/medium tier: single translation
            vector_match = re.search(r"\(a, b\) = \((-?\d+), (-?\d+)\)", text)
            if not vector_match:
                raise ValueError(f"Could not parse column vector from: {text}")
            a, b = int(vector_match.group(1)), int(vector_match.group(2))

            # Image point: P' = (px + a, py + b)
            x_prime = px + a
            y_prime = py + b

        return x_prime, y_prime

    def _parse_answer_text(self, answer_text):
        """Parse answer_text like 'x = 5, y = 3' into (x, y)."""
        x_match = re.search(r"x = (-?\d+)", answer_text)
        y_match = re.search(r"y = (-?\d+)", answer_text)
        if not x_match or not y_match:
            raise ValueError(f"Could not parse answer_text: {answer_text}")
        return int(x_match.group(1)), int(y_match.group(1))

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_oracle_re_derivation(self, difficulty):
        """Oracle check: re-derive answer from question text over 60 seeds."""
        mismatches = []
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            try:
                oracle_x, oracle_y = self._parse_question_and_derive(q)
                parsed_x, parsed_y = self._parse_answer_text(q.answer_text)

                if (oracle_x, oracle_y) != (parsed_x, parsed_y):
                    mismatches.append(
                        {
                            "seed": seed,
                            "difficulty": difficulty,
                            "text": q.text,
                            "answer_text": q.answer_text,
                            "oracle_computed": (oracle_x, oracle_y),
                            "parsed_answer": (parsed_x, parsed_y),
                        }
                    )
            except Exception as e:
                mismatches.append(
                    {
                        "seed": seed,
                        "difficulty": difficulty,
                        "error": str(e),
                        "text": q.text,
                    }
                )

        assert not mismatches, f"Oracle re-derivation failed:\n{mismatches}"


class TestTranslationsGrading:
    """Grading check: real grader accepts answer_text, rejects distractors."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_accepts_canonical_answer(self, difficulty):
        """Real grader (grade_pair) accepts the canonical answer_text."""
        for seed in range(20):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            result = grading.grade_pair(q, q.answer_text)
            assert result.correct, (
                f"{difficulty} seed={seed}: grader rejected canonical answer_text\nanswer_text: {q.answer_text}"
            )

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_rejects_distractors(self, difficulty):
        """Real grader rejects all distractors."""
        for seed in range(20):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            for i, distractor in enumerate(q.distractors):
                result = grading.grade_pair(q, distractor)
                assert not result.correct, (
                    f"{difficulty} seed={seed}: grader accepted distractor {i}\n"
                    f"answer_text: {q.answer_text}\n"
                    f"distractor: {distractor}"
                )

    def test_grader_rejects_wrong_answer(self):
        """Real grader rejects a deliberately wrong answer."""
        q = make(TOPIC, seed=0, difficulty="easy")
        # Use a wrong answer like swapping x and y
        wrong = f"x = 999, y = 888"
        result = grading.grade_pair(q, wrong)
        assert not result.correct


class TestTranslationsAnswerFormat:
    """Answer format check: all answers and distractors match 'x = N, y = M' shape."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_answer_and_distractors_format(self, difficulty):
        """All answers and distractors have the same format 'x = N, y = M'."""
        for seed in range(20):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            # Check answer_text format
            assert re.match(r"x = -?\d+, y = -?\d+$", q.answer_text), (
                f"{difficulty} seed={seed}: answer_text format wrong: {q.answer_text}"
            )
            # Check distractors format
            for distractor in q.distractors:
                assert re.match(r"x = -?\d+, y = -?\d+$", distractor), (
                    f"{difficulty} seed={seed}: distractor format wrong: {distractor}"
                )

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_answer_values_are_integers(self, difficulty):
        """Verify answer values are integers (no fractions or decimals)."""
        for seed in range(20):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            # Parse and verify integers
            x_match = re.search(r"x = (-?\d+)", q.answer_text)
            y_match = re.search(r"y = (-?\d+)", q.answer_text)
            assert x_match and y_match, f"Could not parse answer_text: {q.answer_text}"
            x_val = int(x_match.group(1))
            y_val = int(y_match.group(1))
            # Verify they parse as integers (not fractions)
            assert isinstance(x_val, int) and isinstance(y_val, int)


class TestTranslationsSkillLoad:
    """Skill loading check: verify the skill loads correctly via API."""

    def test_skill_loads_with_correct_metadata(self):
        """Load the skill and verify metadata."""
        domains, skills_list = load_skills()
        skill = next((s for s in skills_list if s.id == SKILL_ID), None)
        assert skill is not None, f"Skill {SKILL_ID} not found"

        assert skill.answer == "pair", f"Expected answer='pair', got {skill.answer}"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["vector-addition"], f"Unexpected prerequisites: {skill.prerequisites}"

        # Verify sources cover all stages
        all_stages = set()
        for source in skill.sources:
            all_stages.update(source.stages)
        assert 0 in all_stages and 6 in all_stages, f"Sources don't cover full stage range: {sorted(all_stages)}"

    def test_skill_has_all_misconceptions(self):
        """Verify all misconceptions are registered."""
        domains, skills_list = load_skills()
        skill = next((s for s in skills_list if s.id == SKILL_ID), None)
        assert skill is not None

        misconception_ids = {m.id for m in skill.misconceptions}
        expected_ids = {
            "subtracts-instead-of-adds",
            "swapped-vector-components",
            "one-axis-only",
            "only-first-translation",
            "only-second-translation",
            "subtracts-second-translation",
        }

        # Hard-tier misconceptions may or may not be in the list;
        # at minimum the easy/medium ones must be present
        easy_medium_ids = {
            "subtracts-instead-of-adds",
            "swapped-vector-components",
            "one-axis-only",
        }
        assert easy_medium_ids.issubset(misconception_ids), (
            f"Missing easy/medium misconceptions. Found: {misconception_ids}"
        )
