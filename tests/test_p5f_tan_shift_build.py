"""Focused contract tests for tan(x+c) and co-function identities."""

from pathlib import Path
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p5f_tan_shift_build import THETA_POOL, COS_STR, SIN_STR, NEIGHBOR, _negate_str


ROOT = Path(__file__).resolve().parents[1]


class TestTanShiftComplementaryIdentityDeterminism:
    """Test determinism across all pool values and variants."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("pool_index", range(len(THETA_POOL)))
    def test_deterministic_across_all_pool_values(self, difficulty, pool_index):
        """Every pool value should deterministically produce the same question."""
        # Use seed based on pool_index to cycle through different angles
        seed = pool_index * 100
        question = make("tan_shift_complementary_identity", seed=seed, difficulty=difficulty)
        repeat = make("tan_shift_complementary_identity", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_deterministic_across_40_seeds(self, difficulty):
        """Determinism across 40 random seeds for each difficulty."""
        for seed in range(40):
            question = make("tan_shift_complementary_identity", seed=seed, difficulty=difficulty)
            repeat = make("tan_shift_complementary_identity", seed=seed, difficulty=difficulty)
            assert question.to_json() == repeat.to_json()


class TestTanShiftComplementaryIdentityDistractors:
    """Test distractor properties."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", range(40))
    def test_distractors_non_empty_and_distinct(self, difficulty, seed):
        """Distractors must be non-empty, unique, and distinct from answer."""
        question = make("tan_shift_complementary_identity", seed=seed, difficulty=difficulty)
        assert len(question.distractors) >= 1, f"No distractors for {difficulty} seed {seed}"
        assert len(question.distractors) == len(set(question.distractors)), (
            f"Duplicate distractors in {difficulty} seed {seed}"
        )
        assert question.answer_text not in question.distractors, f"Answer in distractors for {difficulty} seed {seed}"


class TestTanShiftComplementaryIdentityNoAsterisk:
    """Test that no bare '*' appears in any field."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", range(40))
    def test_no_asterisk_in_text_or_answers(self, difficulty, seed):
        """No bare '*' should appear in text, answer_text, or distractors."""
        question = make("tan_shift_complementary_identity", seed=seed, difficulty=difficulty)
        assert "*" not in question.text, f"Bare * in text: {question.text}"
        assert "*" not in question.answer_text, f"Bare * in answer: {question.answer_text}"
        for distractor in question.distractors:
            assert "*" not in distractor, f"Bare * in distractor: {distractor}"


class TestTanShiftComplementaryIdentityGrading:
    """Test real grader acceptance/rejection."""

    def test_easy_grading(self):
        """easy variant (sin_shift) uses expression grading."""
        for seed in range(20):
            question = make("tan_shift_complementary_identity", seed=seed, difficulty="easy")
            # Canonical answer must be accepted
            result = grade_expression(question, question.answer_text)
            assert result.correct, f"easy seed {seed}: canonical '{question.answer_text}' rejected"
            # Every distractor must be rejected
            for distractor in question.distractors:
                result = grade_expression(question, distractor)
                assert not result.correct, f"easy seed {seed}: distractor '{distractor}' accepted"

    def test_medium_grading(self):
        """medium variant (cos_shift) uses expression grading."""
        for seed in range(20):
            question = make("tan_shift_complementary_identity", seed=seed, difficulty="medium")
            # Canonical answer must be accepted
            result = grade_expression(question, question.answer_text)
            assert result.correct, f"medium seed {seed}: canonical '{question.answer_text}' rejected"
            # Every distractor must be rejected
            for distractor in question.distractors:
                result = grade_expression(question, distractor)
                assert not result.correct, f"medium seed {seed}: distractor '{distractor}' accepted"

    def test_hard_grading(self):
        """hard variant (tan_asymptote) uses expression grading."""
        for seed in range(20):
            question = make("tan_shift_complementary_identity", seed=seed, difficulty="hard")
            # Canonical answer must be accepted
            result = grade_expression(question, question.answer_text)
            assert result.correct, f"hard seed {seed}: canonical '{question.answer_text}' rejected"
            # Every distractor must be rejected
            for distractor in question.distractors:
                result = grade_expression(question, distractor)
                assert not result.correct, f"hard seed {seed}: distractor '{distractor}' accepted"

    def test_deliberately_wrong_answer_rejected(self):
        """A deliberately wrong answer should be rejected."""
        question = make("tan_shift_complementary_identity", seed=0, difficulty="easy")
        result = grade_expression(question, "0")
        assert not result.correct, "Deliberately wrong answer '0' was accepted"


class TestTanShiftComplementaryIdentityCollisionCases:
    """Verify properties of disclosed collision cases and guaranteed non-collisions."""

    def test_sin_str_cos_str_match_at_pi_4(self):
        """Verify sin(pi/4) == cos(pi/4) == sqrt(2)/2 (collision case)."""
        assert SIN_STR[(1, 4)] == "sqrt(2)/2"
        assert COS_STR[(1, 4)] == "sqrt(2)/2"

    def test_sin_str_has_no_zero_values(self):
        """cos_shift: SIN_STR has no '0' values, so -sin never collides on that basis."""
        for theta, sin_val in SIN_STR.items():
            assert sin_val != "0", f"SIN_STR[{theta}] is '0', breaking the guarantee"

    def test_no_theta_pool_member_is_integer(self):
        """tan_asymptote: no THETA_POOL member is an integer (prevents pi/2 collision)."""
        for theta in THETA_POOL:
            c_coeff = Fraction(*theta)
            assert c_coeff.denominator != 1, f"THETA_POOL member {theta} is integer, breaks no-collision guarantee"

    def test_distractor_pool_exhaustion_on_collision_seed(self):
        """When two distractors collide, the remaining one survives.
        This test documents the known collision cases without relying on seed prediction."""
        # For sin_shift at theta=(1,4): both distractors collide
        # For sin_shift at theta=(1,2): sign-error distractor collides
        # For cos_shift at theta=(1,4): first distractor collides
        # For tan_asymptote at c=(1,2): second distractor collides
        # All other cases: at least one distractor survives

        # Just verify that across 40 seeds, distractors are never empty
        for difficulty in ["easy", "medium", "hard"]:
            for seed in range(40):
                question = make("tan_shift_complementary_identity", seed=seed, difficulty=difficulty)
                assert len(question.distractors) >= 1, f"{difficulty} seed {seed}: distractors is empty"

    def test_neighbor_mapping_complete(self):
        """Verify NEIGHBOR mapping covers all THETA_POOL values."""
        for theta in THETA_POOL:
            assert theta in NEIGHBOR, f"NEIGHBOR missing key {theta}"
            assert NEIGHBOR[theta] in THETA_POOL, f"NEIGHBOR[{theta}] = {NEIGHBOR[theta]} not in THETA_POOL"
        # Verify it's a cycle
        visited = set()
        current = THETA_POOL[0]
        for _ in range(len(THETA_POOL)):
            assert current not in visited, f"NEIGHBOR cycle broken at {current}"
            visited.add(current)
            current = NEIGHBOR[current]
        assert current == THETA_POOL[0], "NEIGHBOR mapping is not a complete cycle"


class TestTanShiftComplementaryIdentityEnoughDistracters:
    """Verify that with the neighbor candidates fix, all instances have >=2 distractors."""

    def test_sin_shift_always_has_enough_distractors(self):
        """For sin_shift (easy), all 200+ seeds should have >=2 distractors."""
        for seed in range(200):
            question = make("tan_shift_complementary_identity", seed=seed, difficulty="easy")
            assert len(question.distractors) >= 2, (
                f"easy seed {seed}: only {len(question.distractors)} distractor(s), need >=2"
            )

    def test_cos_shift_always_has_enough_distractors(self):
        """For cos_shift (medium), all 200+ seeds should have >=2 distractors."""
        for seed in range(200):
            question = make("tan_shift_complementary_identity", seed=seed, difficulty="medium")
            assert len(question.distractors) >= 2, (
                f"medium seed {seed}: only {len(question.distractors)} distractor(s), need >=2"
            )


class TestTanShiftComplementaryIdentitySkillLoading:
    """Test that the skill loads correctly with expected fields."""

    def test_skill_loads_with_correct_fields(self):
        """Skill should load with placement_eligible=False and correct prerequisites."""
        _, skills = load_skills(ROOT / "skills")
        by_id = {skill.id: skill for skill in skills}
        skill = by_id["senior-y-tan-x-c-and-the-relationships-sin"]

        assert skill.placement_eligible is False
        assert skill.prerequisites == ["right-triangle-angle", "senior-recognise-the-exact-values-of-cos-theta-sin"]
        assert skill.answer == "expression"
        assert len(skill.sources) == 4
        assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
        assert [source.topic for source in skill.sources] == ["tan_shift_complementary_identity"] * 3 + [
            "exd_x_f5_tanasym"
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
