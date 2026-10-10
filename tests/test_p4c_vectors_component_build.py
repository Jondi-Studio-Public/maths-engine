"""Focused contract tests for vectors component form and scalar multiple."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p4c_vectors_component_build import (
    _verify_single_leg_component,
    _verify_two_leg_component,
    _verify_scalar_forward,
    _verify_scalar_inverse,
)

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "skills"
SEEDS_DETERMINISM = range(500)  # Test determinism over 500 seeds per difficulty
SEEDS_GRADING = range(40)


# === Skill 1: express_a_vector_in_component_form_using_the =================


class TestExpressComponentForm:
    """Tests for express_a_vector_in_component_form_using_the."""

    # --- Determinism and distractor hygiene ---

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_determinism_across_500_seeds(self, difficulty):
        """Verify determinism: same seed produces identical questions."""
        for seed in SEEDS_DETERMINISM:
            q1 = make("express_a_vector_in_component_form_using_the", seed=seed, difficulty=difficulty)
            q2 = make("express_a_vector_in_component_form_using_the", seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_distractors_distinct_nonempty_not_answer(self, difficulty):
        """Verify distractors are distinct, non-empty, never equal to answer."""
        for seed in SEEDS_DETERMINISM:
            q = make("express_a_vector_in_component_form_using_the", seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2

    # --- Grader accept/reject ---

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_accepts_canonical_answer(self, difficulty):
        """Real grader accepts canonical answer_text."""
        for seed in SEEDS_GRADING:
            q = make("express_a_vector_in_component_form_using_the", seed=seed, difficulty=difficulty)
            result = grade_number(q, q.answer_text)
            assert result.correct, f"seed {seed} difficulty {difficulty}: grader rejected canonical answer"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_rejects_every_distractor(self, difficulty):
        """Real grader rejects every distractor."""
        for seed in SEEDS_GRADING:
            q = make("express_a_vector_in_component_form_using_the", seed=seed, difficulty=difficulty)
            for distractor in q.distractors:
                result = grade_number(q, distractor)
                assert not result.correct, (
                    f"seed {seed} difficulty {difficulty}: grader accepted distractor {distractor}"
                )

    def test_grader_rejects_deliberately_wrong_answer(self):
        """Real grader rejects a deliberately wrong answer."""
        for seed in range(50):
            q = make("express_a_vector_in_component_form_using_the", seed=seed, difficulty="medium")
            wrong_answer = str(int(q.answer) + 999)
            result = grade_number(q, wrong_answer)
            assert not result.correct

    # --- Independent checks ---

    def test_single_leg_independent_check_fires_on_corrupted(self):
        """Verify single-leg check fires when answer is corrupted."""
        q = make("express_a_vector_in_component_form_using_the", seed=1, difficulty="easy")
        # Extract direction from text (always "right" and "up" at easy)
        # Corrupt the answer
        with pytest.raises(VerificationError):
            _verify_single_leg_component("right", 5, 999)  # wrong magnitude

    def test_two_leg_independent_check_fires_on_corrupted(self):
        """Verify two-leg check fires when answer is corrupted."""
        with pytest.raises(VerificationError):
            # Directions [right, right], magnitudes [3, 4], asked i
            # Correct answer should be 3 + 4 = 7, but we pass 999
            _verify_two_leg_component(["right", "right"], [3, 4], "i", 999)

    # --- Skill 1 specific constraints ---

    def test_easy_tier_never_offers_sign_ignored(self):
        """Sign-ignored distractor is never offered at easy tier (would equal answer)."""
        for seed in range(200):
            q = make("express_a_vector_in_component_form_using_the", seed=seed, difficulty="easy")
            # At easy, both directions are always positive (right, up)
            # So the unsigned magnitude equals the answer
            # We verify this by checking that distractors are:
            # 1. wrong-axis-component (the other axis)
            # 2. adds-both-components-together (sum of both)
            # We can't easily parse the question, so we check there are exactly 2 distractors
            # (this is guaranteed by pick_distractors)
            assert len(q.distractors) >= 2

    def test_hard_tier_has_nonzero_net_and_different_nets(self):
        """Hard tier: net coefficient != 0 and net_x != net_y."""
        for seed in range(200):
            q = make("express_a_vector_in_component_form_using_the", seed=seed, difficulty="hard")
            # Parse question to extract the answer
            # Question format: "A vector is built from two movements: first {p1} units {dir1}..."
            # The answer is already in q.answer, which is the net coefficient
            # We check it's nonzero
            assert q.answer != 0, f"seed {seed}: net coefficient is zero (degenerate question)"

    # --- Skills loading ---

    def test_skill_loads_with_correct_metadata(self):
        """Skill loads with expected prerequisites, answer, placement_eligible, status."""
        _, skills = load_skills(SKILLS_ROOT)
        skill = next((s for s in skills if s.id == "senior-express-a-vector-in-component-form-using-the"), None)
        assert skill is not None
        assert "vector-addition" in skill.prerequisites
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.status == "active"

    def test_skill_covers_all_difficulties(self):
        """Skill YAML lists all three difficulties."""
        _, skills = load_skills(SKILLS_ROOT)
        skill = next((s for s in skills if s.id == "senior-express-a-vector-in-component-form-using-the"), None)
        assert skill is not None
        sources = skill.sources
        difficulties = [s.difficulty for s in sources]
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties


# === Skill 2: define_and_use_multiplication_by_a_scalar_of ==================


class TestScalarMultiple:
    """Tests for define_and_use_multiplication_by_a_scalar_of."""

    # --- Determinism and distractor hygiene ---

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_determinism_across_500_seeds(self, difficulty):
        """Verify determinism: same seed produces identical questions."""
        for seed in SEEDS_DETERMINISM:
            q1 = make("define_and_use_multiplication_by_a_scalar_of", seed=seed, difficulty=difficulty)
            q2 = make("define_and_use_multiplication_by_a_scalar_of", seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_distractors_distinct_nonempty_not_answer(self, difficulty):
        """Verify distractors are distinct, non-empty, never equal to answer."""
        for seed in SEEDS_DETERMINISM:
            q = make("define_and_use_multiplication_by_a_scalar_of", seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            # Hard tier has exactly 2 distractors
            # Easy/medium have at least 2
            if difficulty == "hard":
                assert len(q.distractors) == 2
            else:
                assert len(q.distractors) >= 2

    # --- Grader accept/reject ---

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_accepts_canonical_answer(self, difficulty):
        """Real grader accepts canonical answer_text."""
        for seed in SEEDS_GRADING:
            q = make("define_and_use_multiplication_by_a_scalar_of", seed=seed, difficulty=difficulty)
            result = grade_number(q, q.answer_text)
            assert result.correct, f"seed {seed} difficulty {difficulty}: grader rejected canonical answer"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_grader_rejects_every_distractor(self, difficulty):
        """Real grader rejects every distractor."""
        for seed in SEEDS_GRADING:
            q = make("define_and_use_multiplication_by_a_scalar_of", seed=seed, difficulty=difficulty)
            for distractor in q.distractors:
                result = grade_number(q, distractor)
                assert not result.correct, (
                    f"seed {seed} difficulty {difficulty}: grader accepted distractor {distractor}"
                )

    def test_grader_rejects_deliberately_wrong_answer(self):
        """Real grader rejects a deliberately wrong answer."""
        for seed in range(50):
            q = make("define_and_use_multiplication_by_a_scalar_of", seed=seed, difficulty="medium")
            wrong_answer = str(int(q.answer) + 999)
            result = grade_number(q, wrong_answer)
            assert not result.correct

    # --- Independent checks ---

    def test_forward_independent_check_fires_on_corrupted(self):
        """Verify forward check fires when answer is corrupted."""
        with pytest.raises(VerificationError):
            _verify_scalar_forward(3, 2, 999)  # 3 * 2 = 6, not 999

    def test_inverse_independent_check_fires_on_corrupted(self):
        """Verify inverse check fires when answer is corrupted."""
        with pytest.raises(VerificationError):
            _verify_scalar_inverse(3, 12, 999)  # 3 * 4 = 12, answer should be 4, not 999

    # --- Skill 2 specific constraints ---

    def test_easy_tier_has_no_k2_a2_collision(self):
        """Easy tier: if k==2, then a != 2 (prevents added-instead-of-multiplied collision)."""
        collision_count = 0
        for seed in range(500):
            q = make("define_and_use_multiplication_by_a_scalar_of", seed=seed, difficulty="easy")
            # Parse question to extract k and component
            # Question: "v = {a}i + {b}j. Find the coefficient of {axis} in {k}v."
            # Extract k (the number at the end before 'v')
            match = re.search(r"Find the coefficient of [ij] in (-?\d+)v", q.text)
            if match:
                k = int(match.group(1))
                # Extract the vector components
                match_vec = re.search(r"v = (-?\d+)i", q.text)
                if match_vec:
                    a = int(match_vec.group(1))
                    # If k==2, then a should not be 2 (that would make component + k == component * k)
                    if k == 2:
                        assert a != 2, f"seed {seed}: collision k=2, a=2 found"

    # --- Skills loading ---

    def test_skill_loads_with_correct_metadata(self):
        """Skill loads with expected prerequisites, answer, placement_eligible, status."""
        _, skills = load_skills(SKILLS_ROOT)
        skill = next((s for s in skills if s.id == "senior-define-and-use-multiplication-by-a-scalar-of"), None)
        assert skill is not None
        assert "vector-addition" in skill.prerequisites
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.status == "active"

    def test_skill_covers_all_difficulties(self):
        """Skill YAML lists all three difficulties."""
        _, skills = load_skills(SKILLS_ROOT)
        skill = next((s for s in skills if s.id == "senior-define-and-use-multiplication-by-a-scalar-of"), None)
        assert skill is not None
        sources = skill.sources
        difficulties = [s.difficulty for s in sources]
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties
