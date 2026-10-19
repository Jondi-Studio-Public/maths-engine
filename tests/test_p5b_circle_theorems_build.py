"""Tests the `angle_in_semicircle` and `angles_same_arc` circle-theorem generators and that their skills load.

Circle theorems build."""

import re
import pytest
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

ANGLE_TOPIC = "angle_in_semicircle"
ARC_TOPIC = "angles_same_arc"
SEEDS = range(40)

# Bad sign patterns: a doubled or mixed sign in the rendered text
_BAD_SIGN_PATTERNS = [
    re.compile(r"\+ -"),
    re.compile(r"- -"),
]


class TestAngleInSemicircle:
    """Tests for angle_in_semicircle (Thales' theorem)."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_working_text_nonempty(self, difficulty):
        """Test that working text is non-empty (length >= 2) for all variants."""
        for seed in range(40):
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list), f"seed {seed}: working is not a list"
            assert len(q.working) >= 2, f"seed {seed}: working has fewer than 2 steps"
            for step in q.working:
                assert isinstance(step, str), f"seed {seed}: working step is not a string"
                assert len(step) > 0, f"seed {seed}: working step is empty"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_determinism_across_seeds(self, difficulty):
        """Test determinism of all variants across 40 seeds."""
        for seed in SEEDS:
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            repeat = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        """Test that all distractors are distinct from answer and each other."""
        for seed in SEEDS:
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_accepts_answer(self, difficulty):
        """Test that the real grader accepts the canonical answer."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text}"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_rejects_distractors(self, difficulty):
        """Test that the real grader rejects distractors."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_rejects_wrong_answer(self, difficulty):
        """Test that grader rejects a deliberately wrong answer."""
        grader = GRADERS["number"]
        for seed in range(5):
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            answer_val = int(q.answer_text)
            wrong_answer = str(answer_val + 7)
            result = grader(q, wrong_answer)
            assert not result.correct, f"seed {seed}: grader accepted wrong answer {wrong_answer}"

    def test_no_collision_easy_direct_variant(self):
        """Test that all distractors differ from answer in easy tier."""
        for seed in range(100):
            q = make(ANGLE_TOPIC, seed=seed, difficulty="easy")
            answer = q.answer_text
            for dist in q.distractors:
                assert dist != answer, f"seed {seed}: distractor collides with answer"

    def test_no_collision_medium_difference_variant(self):
        """Test that distractors don't collide in medium tier."""
        for seed in range(500):
            q = make(ANGLE_TOPIC, seed=seed, difficulty="medium")
            answer = q.answer_text
            for dist in q.distractors:
                assert dist != answer, f"seed {seed}: distractor collides with answer"

    def test_no_collision_hard_ratio_variant(self):
        """Test that all distractors differ from answer in hard tier."""
        for seed in range(100):
            q = make(ANGLE_TOPIC, seed=seed, difficulty="hard")
            answer = q.answer_text
            for dist in q.distractors:
                assert dist != answer, f"seed {seed}: distractor collides with answer"


class TestAnglesSameArc:
    """Tests for angles_same_arc (same-segment theorem)."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_working_text_nonempty(self, difficulty):
        """Test that working text is non-empty (length >= 2) for all variants."""
        for seed in range(40):
            q = make(ARC_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list), f"seed {seed}: working is not a list"
            assert len(q.working) >= 2, f"seed {seed}: working has fewer than 2 steps"
            for step in q.working:
                assert isinstance(step, str), f"seed {seed}: working step is not a string"
                assert len(step) > 0, f"seed {seed}: working step is empty"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_determinism_across_seeds(self, difficulty):
        """Test determinism of all variants across 40 seeds."""
        for seed in SEEDS:
            q = make(ARC_TOPIC, seed=seed, difficulty=difficulty)
            repeat = make(ARC_TOPIC, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        """Test that all distractors are distinct from answer and each other."""
        for seed in SEEDS:
            q = make(ARC_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_accepts_answer(self, difficulty):
        """Test that the real grader accepts the canonical answer."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(ARC_TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text}"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_rejects_distractors(self, difficulty):
        """Test that the real grader rejects distractors."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(ARC_TOPIC, seed=seed, difficulty=difficulty)
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_rejects_wrong_answer(self, difficulty):
        """Test that grader rejects a deliberately wrong answer."""
        grader = GRADERS["number"]
        for seed in range(5):
            q = make(ARC_TOPIC, seed=seed, difficulty=difficulty)
            answer_val = int(q.answer_text)
            wrong_answer = str(answer_val + 7)
            result = grader(q, wrong_answer)
            assert not result.correct, f"seed {seed}: grader accepted wrong answer {wrong_answer}"

    def test_no_collision_medium_triangle_sum_variant(self):
        """Test that collision guards work in medium tier."""
        for seed in range(500):
            q = make(ARC_TOPIC, seed=seed, difficulty="medium")
            answer = q.answer_text
            for dist in q.distractors:
                assert dist != answer, f"seed {seed}: distractor collides with answer"

    def test_no_collision_hard_algebraic_variant(self):
        """Test that collision guards work in hard tier."""
        for seed in range(500):
            q = make(ARC_TOPIC, seed=seed, difficulty="hard")
            answer = q.answer_text
            for dist in q.distractors:
                assert dist != answer, f"seed {seed}: distractor collides with answer"

    def test_no_bad_sign_patterns_algebraic_hard_variant(self):
        """Test that algebraic variant has no bad sign patterns over 500 seeds."""
        for seed in range(500):
            q = make(ARC_TOPIC, seed=seed, difficulty="hard")
            fields = [q.text, q.latex, *q.working]
            for field in fields:
                if not isinstance(field, str):
                    continue
                for pat in _BAD_SIGN_PATTERNS:
                    assert not pat.search(field), f"seed {seed}: bad sign pattern {pat.pattern!r} in {field!r}"


class TestSkillsLoad:
    """Test that both skills load via mathgen.skills.load_skills."""

    def test_angle_in_semicircle_loads(self):
        """Test that senior-an-angle-in-a-semicircle-is-a-right skill loads."""
        domains, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-an-angle-in-a-semicircle-is-a-right"), None)
        assert skill is not None, "Skill not found in loaded skills"
        assert skill.id == "senior-an-angle-in-a-semicircle-is-a-right"
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "number"
        assert skill.prerequisites == ["senior-the-angle-at-the-centre-subtended-by-an"]
        assert skill.placement_eligible == False
        # Check that all three tiers are represented in sources
        sources_topics = [s.topic for s in skill.sources]
        assert len(sources_topics) == 4, f"Expected 4 sources, got {len(sources_topics)}"  # plus the exam source

    def test_angles_same_arc_loads(self):
        """Test that senior-angles-at-the-circumference-of-a-circle-subtended skill loads."""
        domains, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-angles-at-the-circumference-of-a-circle-subtended"), None)
        assert skill is not None, "Skill not found in loaded skills"
        assert skill.id == "senior-angles-at-the-circumference-of-a-circle-subtended"
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "number"
        assert skill.prerequisites == ["senior-the-angle-at-the-centre-subtended-by-an"]
        assert skill.placement_eligible == False
        # Check that all three tiers are represented in sources
        sources_topics = [s.topic for s in skill.sources]
        # 3 tiers + the multi-part exam chain + the single-part exam source re-added as key exam-single (bf2fca7)
        assert len(sources_topics) == 5, f"Expected 5 sources, got {len(sources_topics)}"
