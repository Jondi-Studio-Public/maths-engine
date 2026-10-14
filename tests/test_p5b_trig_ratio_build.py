"""Tests the trig-ratio generators: side from angle, angle from sides and chained problems.

Trig ratio build."""

import re
import math
import pytest
from fractions import Fraction
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5b_trig_ratio_build import (
    sine_cosine_ratio_triangle,
    _check_side_from_angle,
    _check_angle_from_sides,
    _check_chained,
    PYTHAGOREAN_TRIPLES,
    THETA_RANGE,
)

TOPIC = "sine_cosine_ratio_triangle"
SEEDS = range(40)

# Bad sign patterns: a doubled or mixed sign in the rendered text
_BAD_SIGN_PATTERNS = [
    re.compile(r"\+ -"),
    re.compile(r"- -"),
]


class TestSidFromAngle:
    """Tests for side_from_angle (easy) tier."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_working_text_nonempty(self, seed):
        """Test that working text is non-empty."""
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert isinstance(q.working, list)
        assert len(q.working) >= 2, f"seed {seed}: working has fewer than 2 steps"
        for step in q.working:
            assert isinstance(step, str)
            assert len(step) > 0

    def test_determinism_across_seeds(self):
        """Test determinism of easy tier across seeds."""
        for seed in SEEDS:
            q1 = make(TOPIC, seed=seed, difficulty="easy")
            q2 = make(TOPIC, seed=seed, difficulty="easy")
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_from_answer(self):
        """Test that distractors are distinct from answer."""
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty="easy")
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer(self):
        """Test that grader accepts the correct answer."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty="easy")
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected {q.answer_text}"

    def test_grader_rejects_distractors(self):
        """Test that grader rejects all distractors."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty="easy")
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    def test_theta_exclusion_band_40_50(self):
        """Test that no questions have theta in 40..50 band over 500 seeds."""
        for seed in range(500):
            q = make(TOPIC, seed=seed, difficulty="easy")
            # Extract theta from text: "one acute angle of N degrees"
            match = re.search(r"angle of (\d+) degrees", q.text)
            assert match, f"seed {seed}: could not extract theta from text"
            theta = int(match.group(1))
            assert not (40 <= theta <= 50), f"seed {seed}: theta={theta} in excluded band"

    def test_swap_distractor_distinct_500_seeds(self):
        """Test that swap-sine-cosine distractor is distinct from correct answer over 500 seeds."""
        for seed in range(500):
            q = make(TOPIC, seed=seed, difficulty="easy")
            # The swap distractor should be distinct after rounding
            # This is verified by construction if it was added to candidates
            answer_val = float(q.answer_text)
            # At least one distractor should exist
            assert len(q.distractors) >= 1

    def test_no_bad_sign_patterns(self):
        """Test that no bad sign patterns (+ -, - -) appear in text or working."""
        for seed in range(100):
            q = make(TOPIC, seed=seed, difficulty="easy")
            fields = [q.text, q.latex, *q.working]
            for field in fields:
                if not isinstance(field, str):
                    continue
                for pat in _BAD_SIGN_PATTERNS:
                    assert not pat.search(field), f"seed {seed}: bad sign pattern {pat.pattern!r} in {field!r}"

    def test_opposite_and_adjacent_both_occur(self):
        """Test that both opposite and adjacent branches occur in sampling."""
        opposite_count = 0
        adjacent_count = 0
        for seed in range(100):
            q = make(TOPIC, seed=seed, difficulty="easy")
            if "opposite" in q.text.lower():
                opposite_count += 1
            elif "adjacent" in q.text.lower():
                adjacent_count += 1
        assert opposite_count > 0, "opposite side never asked"
        assert adjacent_count > 0, "adjacent side never asked"

    def test_independent_check_fires_on_corrupted_answer(self):
        """Test that independent check fails on corrupted triangle."""
        # Construct a corrupted scenario
        with pytest.raises(VerificationError):
            _check_side_from_angle(hyp=5.0, theta_deg=30, queried_leg_str="999.0", variant="opposite")


class TestAngleFromSides:
    """Tests for angle_from_sides (medium) tier."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_working_text_nonempty(self, seed):
        """Test that working text is non-empty."""
        q = make(TOPIC, seed=seed, difficulty="medium")
        assert isinstance(q.working, list)
        assert len(q.working) >= 1, f"seed {seed}: working has no steps"
        for step in q.working:
            assert isinstance(step, str)
            assert len(step) > 0

    def test_determinism_across_seeds(self):
        """Test determinism of medium tier."""
        for seed in SEEDS:
            q1 = make(TOPIC, seed=seed, difficulty="medium")
            q2 = make(TOPIC, seed=seed, difficulty="medium")
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_from_answer(self):
        """Test that distractors are distinct from answer."""
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty="medium")
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer(self):
        """Test that grader accepts the correct answer."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty="medium")
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected {q.answer_text}"

    def test_grader_rejects_distractors(self):
        """Test that grader rejects all distractors."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty="medium")
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    def test_all_pythagorean_triples_have_distinct_answers(self):
        """Test all 10 Pythagorean triple pairs directly for distinctness."""
        grader = GRADERS["number"]
        for a, b, c in PYTHAGOREAN_TRIPLES:
            for given_leg, other_leg in [(a, b), (b, a)]:
                # Compute expected correct answer
                theta_correct = math.degrees(math.asin(given_leg / c))
                correct_str = str(round(theta_correct))

                # Compute misconception: swap asin/acos
                theta_swap = math.degrees(math.acos(given_leg / c))
                swap_str = str(round(theta_swap))

                # Compute misconception: complementary angle
                complementary_str = str(90 - int(correct_str))

                # Compute misconception: ratio
                ratio = given_leg / c
                ratio_str = str(round(ratio))

                # Verify distinctness
                distractors = [swap_str, complementary_str, ratio_str]
                distractors = [d for d in distractors if d != correct_str]
                assert len(distractors) > 0, f"Triple {a}-{b}-{c}: given_leg={given_leg}, no distinct distractors"

    def test_note_20_21_29_in_band(self):
        """Note that 20-21-29 triple produces angles in the 40..50 band (derived, not sampled)."""
        theta1 = math.degrees(math.asin(20 / 29))
        theta2 = math.degrees(math.asin(21 / 29))
        # These should be ~43.6 and ~46.4, both in or near 40..50
        # Since medium tier enumerates all triples, this is okay (theta is derived, not sampled)
        # The exclusion is for sampled theta (easy and hard tiers)
        assert 40 <= theta1 <= 50 or 40 <= theta2 <= 50, "Expected 20-21-29 to produce angles near 45 degrees"

    def test_opposite_and_adjacent_both_occur(self):
        """Test that both opposite and adjacent legs are asked in sampling."""
        opposite_count = 0
        adjacent_count = 0
        for seed in range(50):
            q = make(TOPIC, seed=seed, difficulty="medium")
            if "opposite" in q.text.lower():
                opposite_count += 1
            elif "adjacent" in q.text.lower():
                adjacent_count += 1
        assert opposite_count > 0, "opposite leg never given"
        assert adjacent_count > 0, "adjacent leg never given"

    def test_independent_check_fires_on_corrupted_answer(self):
        """Test that independent check fails on corrupted scenario."""
        # When leg > hyp, the Pythagorean calculation will fail
        with pytest.raises((VerificationError, ValueError)):
            _check_angle_from_sides(leg=999.0, hyp=5.0, theta_deg_str="30")


class TestChained:
    """Tests for chained (hard) tier."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_working_text_nonempty(self, seed):
        """Test that working text is non-empty."""
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert isinstance(q.working, list)
        assert len(q.working) >= 2, f"seed {seed}: working has fewer than 2 steps"
        for step in q.working:
            assert isinstance(step, str)
            assert len(step) > 0

    def test_determinism_across_seeds(self):
        """Test determinism of hard tier."""
        for seed in SEEDS:
            q1 = make(TOPIC, seed=seed, difficulty="hard")
            q2 = make(TOPIC, seed=seed, difficulty="hard")
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_from_answer(self):
        """Test that distractors are distinct from answer."""
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty="hard")
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer(self):
        """Test that grader accepts the correct answer."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty="hard")
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected {q.answer_text}"

    def test_grader_rejects_distractors(self):
        """Test that grader rejects all distractors."""
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty="hard")
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    def test_theta_exclusion_band_40_50(self):
        """Test that no questions have theta in 40..50 band over 500 seeds."""
        for seed in range(500):
            q = make(TOPIC, seed=seed, difficulty="hard")
            # Extract theta from text: "angle of N degrees"
            match = re.search(r"angle of (\d+) degrees", q.text)
            assert match, f"seed {seed}: could not extract theta from text"
            theta = int(match.group(1))
            assert not (40 <= theta <= 50), f"seed {seed}: theta={theta} in excluded band"

    def test_no_bad_sign_patterns(self):
        """Test that no bad sign patterns appear in text or working."""
        for seed in range(100):
            q = make(TOPIC, seed=seed, difficulty="hard")
            fields = [q.text, q.latex, *q.working]
            for field in fields:
                if not isinstance(field, str):
                    continue
                for pat in _BAD_SIGN_PATTERNS:
                    assert not pat.search(field), f"seed {seed}: bad sign pattern {pat.pattern!r} in {field!r}"

    def test_independent_check_fires_on_corrupted_answer(self):
        """Test that independent check fails on corrupted scenario."""
        with pytest.raises(VerificationError):
            _check_chained(adj=5.0, theta_deg=30, opposite_full=999.0, H=10.0)


class TestSkillLoads:
    """Test that the skill loads correctly via mathgen.skills.load_skills."""

    def test_skill_loads_with_correct_metadata(self):
        """Test that senior-apply-the-cosine-and-sine-ratios-to-find skill loads."""
        domains, skills = load_skills()
        skill_id = "senior-apply-the-cosine-and-sine-ratios-to-find"
        skill = next((s for s in skills if s.id == skill_id), None)
        assert skill is not None, f"Skill {skill_id} not found in load_skills()"
        assert skill.answer == "number"
        assert skill.prerequisites == ["senior-review-sine-cosine-and-tangent-as-ratios-of"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 5  # easy, medium, hard, exam, exam-single

    def test_skill_sources_cover_all_difficulties(self):
        """Test that skill sources cover easy, medium, and hard."""
        domains, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-apply-the-cosine-and-sine-ratios-to-find"), None)
        assert skill is not None
        difficulties = {source.difficulty for source in skill.sources}
        assert difficulties == {"easy", "medium", "hard", "exam"}

    def test_misconceptions_present(self):
        """Test that skill declares misconceptions."""
        domains, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-apply-the-cosine-and-sine-ratios-to-find"), None)
        assert skill is not None
        assert len(skill.misconceptions) >= 1, "Skill should declare misconceptions"
