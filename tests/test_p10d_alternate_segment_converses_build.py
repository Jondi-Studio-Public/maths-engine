"""The alternate segment theorem + suitable converses of some of the above circle theorems."""

from __future__ import annotations

import re

import pytest

from mathgen.grading.typed import GRADERS, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p10d_alternate_segment_converses_build import (
    ALL_CONVERSE_OPTIONS,
    OPTION_CONCYCLIC,
    OPTION_CYCLIC_QUAD,
    OPTION_DIAMETER,
    OPTION_NONE,
    OPTION_TANGENT,
    _classify_family_a,
    _classify_family_b,
    _classify_family_c,
    _classify_family_d,
    _verify_alternate_segment,
    _verify_classification,
    _verify_family_d,
    _verify_linear_intersection,
)

ALT_SEG_TOPIC = "alternate_segment_theorem"
CONVERSE_TOPIC = "circle_theorem_converses_classify"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]
LOW, HIGH = 15, 145


# ============================================================================
# Skill 1: alternate_segment_theorem
# ============================================================================


class TestAlternateSegmentTheorem:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = make(ALT_SEG_TOPIC, seed=seed, difficulty=difficulty)
            q2 = make(ALT_SEG_TOPIC, seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(ALT_SEG_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(ALT_SEG_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors_and_wrong(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(ALT_SEG_TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed} {difficulty}: rejected answer {q.answer_text}"
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed} {difficulty}: accepted distractor {dist}"
            wrong = str(int(q.answer_text) + 7)
            if wrong in q.distractors or wrong == q.answer_text:
                wrong = str(int(q.answer_text) + 11)
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed} {difficulty}: accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_is_plain_positive_integer_string(self, difficulty):
        for seed in SEEDS:
            q = make(ALT_SEG_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert q.answer > 0
            assert q.answer_text == str(q.answer)
            for dist in q.distractors:
                assert re.fullmatch(r"-?\d+", dist), f"distractor {dist!r} not formatted like the answer"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_plausibility_every_stated_and_derived_angle_in_bounds(self, difficulty):
        """The shared plausibility requirement: every angle drawn or derived must land in [15, 145] degrees."""
        for seed in SEEDS:
            q = make(ALT_SEG_TOPIC, seed=seed, difficulty=difficulty)
            answer_angle = int(q.answer_text)
            assert LOW <= answer_angle <= HIGH, f"seed {seed} {difficulty}: answer {answer_angle} out of bounds"
            if difficulty == "easy":
                stated = int(re.search(r"measured at T, is (\d+) degrees", q.text).group(1))
                assert LOW <= stated <= HIGH
            elif difficulty == "composed" or difficulty == "medium":
                m = re.search(r"angle QTP is (\d+) degrees and angle TPQ is (\d+) degrees", q.text)
                a, b = int(m.group(1)), int(m.group(2))
                assert LOW <= a <= HIGH and LOW <= b <= HIGH

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(ALT_SEG_TOPIC, seed=seed, difficulty=difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    def test_verify_alternate_segment_fires_on_corrupted_answer(self):
        with pytest.raises(VerificationError):
            _verify_alternate_segment(30, q_arc_fractions=(-0.1,))  # invalid fraction
        _verify_alternate_segment(30)  # does not raise
        with pytest.raises(VerificationError):
            _verify_alternate_segment(200)  # out of domain

    def test_verify_linear_intersection_fires_on_corrupted_answer(self):
        # 2x + 10 = 3x - 15 -> x = 25, angle = 60
        with pytest.raises(VerificationError):
            _verify_linear_intersection(2, 10, 3, -15, 25, 61)  # wrong expected angle
        assert _verify_linear_intersection(2, 10, 3, -15, 25, 60) == 60

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-the-alternate-segment-theorem"), None)
        assert skill is not None
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "number"
        assert skill.prerequisites == []
        assert skill.placement_eligible is False
        assert len(skill.sources) == 4  # three practice tiers plus the exam source


# ============================================================================
# Skill 2: circle_theorem_converses_classify
# ============================================================================


class TestCircleTheoremConversesClassify:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = make(CONVERSE_TOPIC, seed=seed, difficulty=difficulty)
            q2 = make(CONVERSE_TOPIC, seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_is_one_of_the_fixed_options(self, difficulty):
        for seed in SEEDS:
            q = make(CONVERSE_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text in ALL_CONVERSE_OPTIONS
            options = mcq_options(q)
            assert q.answer_text in options
            assert set(q.distractors) == set(ALL_CONVERSE_OPTIONS) - {q.answer_text}

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = make(CONVERSE_TOPIC, seed=seed, difficulty=difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, str(correct_index)).correct
            for dist in q.distractors:
                dist_index = options.index(dist)
                assert not grade_mcq(q, str(dist_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_plausibility_every_stated_angle_in_bounds(self, difficulty):
        for seed in SEEDS:
            q = make(CONVERSE_TOPIC, seed=seed, difficulty=difficulty)
            stated = [int(n) for n in re.findall(r"(\d+) degrees", q.text)]
            assert stated, f"seed {seed} {difficulty}: no stated angle found in {q.text!r}"
            for value in stated:
                assert LOW <= value <= HIGH, f"seed {seed} {difficulty}: {value} out of bounds in {q.text!r}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(CONVERSE_TOPIC, seed=seed, difficulty=difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_satisfied_not_satisfied_balance_at_400_seeds(self, difficulty):
        """Construct-to-target is decided by rng first; report the balance at
        400 seeds rather than judging fairness from a short run."""
        satisfied = 0
        not_satisfied = 0
        for seed in range(400):
            q = make(CONVERSE_TOPIC, seed=seed, difficulty=difficulty)
            if q.answer_text == OPTION_NONE:
                not_satisfied += 1
            else:
                satisfied += 1
        total = satisfied + not_satisfied
        assert total == 400
        share = satisfied / total
        assert 0.35 <= share <= 0.65, f"{difficulty}: satisfied share {share:.3f} looks structurally biased"

    def test_medium_family_split_at_400_seeds(self):
        family_b = 0
        family_c = 0
        for seed in range(400):
            q = make(CONVERSE_TOPIC, seed=seed, difficulty="medium")
            if "same side of line AB" in q.text:
                family_b += 1
            elif "opposite angles of the quadrilateral" in q.text:
                family_c += 1
        total = family_b + family_c
        assert total == 400
        share = family_b / total
        assert 0.35 <= share <= 0.65, f"medium family-B share {share:.3f} looks structurally biased"

    def test_verify_classification_fires_on_corrupted_answer(self):
        with pytest.raises(VerificationError):
            _verify_classification(True, OPTION_NONE)
        with pytest.raises(VerificationError):
            _verify_classification(False, OPTION_TANGENT)
        _verify_classification(True, OPTION_TANGENT)
        _verify_classification(False, OPTION_NONE)

    def test_classify_family_a(self):
        assert _classify_family_a(90) == OPTION_DIAMETER
        assert _classify_family_a(89) == OPTION_NONE

    def test_classify_family_b(self):
        assert _classify_family_b(40, 40) == OPTION_CONCYCLIC
        assert _classify_family_b(40, 45) == OPTION_NONE

    def test_classify_family_c(self):
        assert _classify_family_c(70, 110) == OPTION_CYCLIC_QUAD
        assert _classify_family_c(70, 100) == OPTION_NONE

    def test_classify_family_d(self):
        assert _classify_family_d(60, 60) == OPTION_TANGENT
        assert _classify_family_d(60, 68) == OPTION_NONE

    def test_verify_family_d_fires_on_corrupted_answer(self):
        # n solved from (x_value - q1) / p1; y = p2 * n + q2
        with pytest.raises(VerificationError):
            _verify_family_d(n=13, p1=4, q1=-25, x_value=27, p2=2, q2=1, y_value=99)
        _verify_family_d(n=13, p1=4, q1=-25, x_value=27, p2=2, q2=1, y_value=27)

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-suitable-converses-of-some-of-the-above-results"), None)
        assert skill is not None
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "mcq"
        assert skill.prerequisites == []
        assert skill.placement_eligible is False
        assert len(skill.sources) == 3
