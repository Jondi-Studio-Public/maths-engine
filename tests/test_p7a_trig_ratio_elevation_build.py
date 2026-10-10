"""Tests the trig-ratio review and angle of elevation and depression skills.

Senior trig-ratio review (with tangent) and angle of elevation/depression."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7a_trig_ratio_elevation_build import (
    _check_depression,
    _check_elevation,
    _check_ratio,
    _check_two_observation,
    _hard_tier_two_observation,
    TRIPLES,
)

TOPIC_A = "trig_ratio_review_tan"
TOPIC_B = "elevation_depression_problem"
SEEDS = range(40)
TIERS = ("easy", "medium", "hard")


def _sources_for(skill_id: str) -> dict:
    """difficulty -> (topic, params) from the skill's own `sources`."""
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: (src.topic, src.params or {}) for src in skill.sources}


SOURCES_A = _sources_for("senior-review-sine-cosine-and-tangent-as-ratios-of")
SOURCES_B = _sources_for("senior-work-with-the-concepts-of-angle-of-elevation")


def _make_a(seed, tier):
    topic, params = SOURCES_A[tier]
    return make(topic, seed=seed, difficulty=tier, **params)


def _make_b(seed, tier):
    topic, params = SOURCES_B[tier]
    return make(topic, seed=seed, difficulty=tier, **params)


# --------------------------------------------------------------------------
# Skill A: trig_ratio_review_tan
# --------------------------------------------------------------------------


class TestTrigRatioReviewTan:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism(self, tier):
        for seed in SEEDS:
            q1 = _make_a(seed, tier)
            q2 = _make_a(seed, tier)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty(self, tier):
        for seed in SEEDS:
            q = _make_a(seed, tier)
            assert len(q.distractors) >= 1
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            # exact-value distinctness (grade_fraction compares by value)
            answer_val = Fraction(q.answer_text)
            for d in q.distractors:
                assert Fraction(d) != answer_val, f"seed {seed}/{tier}: distractor {d} == answer {q.answer_text}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, tier):
        grader = GRADERS["fraction"]
        for seed in range(10):
            q = _make_a(seed, tier)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}/{tier}: grader rejected {q.answer_text}"
            for d in q.distractors:
                result = grader(q, d)
                assert not result.correct, f"seed {seed}/{tier}: grader accepted distractor {d}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_grader_rejects_deliberately_wrong_answer(self, tier):
        grader = GRADERS["fraction"]
        for seed in range(10):
            q = _make_a(seed, tier)
            correct = Fraction(q.answer_text)
            wrong = correct + 1  # always a different value
            result = grader(q, f"{wrong.numerator}/{wrong.denominator}")
            assert not result.correct, f"seed {seed}/{tier}: grader accepted a deliberately wrong answer"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_is_reduced_fraction(self, tier):
        for seed in SEEDS:
            q = _make_a(seed, tier)
            frac = Fraction(q.answer_text)
            assert q.answer == {"p": frac.numerator, "q": frac.denominator}
            expected_text = str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
            assert q.answer_text == expected_text, f"seed {seed}/{tier}: {q.answer_text!r} not in lowest terms"

    @pytest.mark.parametrize("tier", TIERS)
    def test_distinct_questions_pool(self, tier):
        texts = {_make_a(seed, tier).text for seed in range(60)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions over 60 seeds"

    @pytest.mark.parametrize("tier", TIERS)
    def test_uniform_option_formatting(self, tier):
        """Every option (answer + distractors) is a plain fraction/int
        string, so no option is identifiable by shape alone."""
        frac_re = re.compile(r"^-?\d+(/\d+)?$")
        for seed in range(20):
            q = _make_a(seed, tier)
            for opt in [q.answer_text, *q.distractors]:
                assert frac_re.match(opt), f"seed {seed}/{tier}: option {opt!r} not plain fraction shape"

    def test_tan_coverage_hard_100_percent(self):
        n = 60
        count = sum(1 for seed in range(n) if re.search(r"tan\(theta\)", _make_a(seed, "hard").text))
        assert count == n, f"hard tier: tan present in only {count}/{n} questions (need 100%)"

    def test_tan_coverage_easy_at_least_one_third(self):
        n = 60
        count = sum(1 for seed in range(n) if "tan(theta)" in _make_a(seed, "easy").text)
        assert count >= n // 3, f"easy tier: tan present in only {count}/{n} questions (need >= 1/3)"

    def test_tan_coverage_medium_at_least_one_third(self):
        n = 60
        count = sum(1 for seed in range(n) if "tan(theta)" in _make_a(seed, "medium").text)
        assert count >= n // 3, f"medium tier: tan present in only {count}/{n} questions (need >= 1/3)"

    def test_sin_and_cos_also_occur_easy_and_medium(self):
        """Tan must not crowd out sin/cos entirely -- both must still occur
        at easy and medium (the skill's own name is sine, cosine AND tangent)."""
        for tier in ("easy", "medium"):
            sin_count = sum(1 for seed in range(60) if "sin(theta)" in _make_a(seed, tier).text)
            cos_count = sum(1 for seed in range(60) if "cos(theta)" in _make_a(seed, tier).text)
            assert sin_count > 0, f"{tier}: sin never asked over 60 seeds"
            assert cos_count > 0, f"{tier}: cos never asked over 60 seeds"

    def test_hard_tier_given_and_asked_differ(self):
        for seed in range(60):
            q = _make_a(seed, "hard")
            m = re.match(
                r"For an acute angle theta in a right-angled triangle, (sin|cos|tan)\(theta\) = "
                r"(\d+/\d+|\d+)\. Find (sin|cos|tan)\(theta\)",
                q.text,
            )
            assert m, f"seed {seed}: could not parse hard-tier text: {q.text!r}"
            given_name, _, asked_name = m.groups()
            assert given_name != asked_name

    def test_working_nonempty(self):
        for tier in TIERS:
            for seed in range(20):
                q = _make_a(seed, tier)
                assert isinstance(q.working, list) and len(q.working) >= 2
                for line in q.working:
                    assert isinstance(line, str) and line

    def test_independent_check_fires_on_corrupted_answer(self):
        with pytest.raises(VerificationError):
            _check_ratio("tan", Fraction(1, 2), 9, 40, 41)
        with pytest.raises(VerificationError):
            _check_ratio("sin", Fraction(9, 41), 9, 41, 40)  # not even a right triangle

    def test_independent_triples_are_valid_right_triangles(self):
        for a, b, c in TRIPLES:
            assert a * a + b * b == c * c


# --------------------------------------------------------------------------
# Skill B: elevation_depression_problem
# --------------------------------------------------------------------------


class TestElevationDepression:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism(self, tier):
        for seed in SEEDS:
            q1 = _make_b(seed, tier)
            q2 = _make_b(seed, tier)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty(self, tier):
        for seed in SEEDS:
            q = _make_b(seed, tier)
            assert len(q.distractors) >= 1
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            answer_val = float(q.answer_text)
            for d in q.distractors:
                assert abs(float(d) - answer_val) > 0.051, (
                    f"seed {seed}/{tier}: distractor {d} within 0.051 of answer {q.answer_text}"
                )

    @pytest.mark.parametrize("tier", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, tier):
        grader = GRADERS["number"]
        for seed in range(10):
            q = _make_b(seed, tier)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}/{tier}: grader rejected {q.answer_text}"
            for d in q.distractors:
                result = grader(q, d)
                assert not result.correct, f"seed {seed}/{tier}: grader accepted distractor {d}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_grader_rejects_deliberately_wrong_answer(self, tier):
        grader = GRADERS["number"]
        for seed in range(10):
            q = _make_b(seed, tier)
            wrong = f"{float(q.answer_text) + 5.0:.1f}"
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed}/{tier}: grader accepted a deliberately wrong answer"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_stated_to_1dp_never_trailing_zero(self, tier):
        for seed in SEEDS:
            q = _make_b(seed, tier)
            assert re.match(r"^-?\d+\.\d$", q.answer_text), f"seed {seed}/{tier}: {q.answer_text!r} not 1dp"
            assert not q.answer_text.endswith(".0"), f"seed {seed}/{tier}: trailing-zero answer {q.answer_text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_distinct_questions_pool(self, tier):
        texts = {_make_b(seed, tier).text for seed in range(60)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions over 60 seeds"

    def test_easy_excludes_45_degrees(self):
        for seed in range(200):
            q = _make_b(seed, "easy")
            theta = int(re.search(r"is (\d+) degrees", q.text).group(1))
            assert theta != 45

    def test_medium_uses_depression_and_alternate_angle_wording(self):
        for seed in range(10):
            q = _make_b(seed, "medium")
            assert "depression" in q.text.lower()
            assert "alternate angles" in q.working[0].lower()

    def test_medium_distance_never_contrived(self):
        """The resample loop now floors the horizontal distance at 5 m."""
        for seed in range(60):
            q = _make_b(seed, "medium")
            assert float(q.answer_text) >= 5, f"seed {seed}: contrived distance {q.answer_text} m"

    def test_hard_states_collinear_same_side(self):
        for seed in range(10):
            q = _make_b(seed, "hard")
            assert "collinear" in q.text.lower()
            assert "same side" in q.text.lower()

    def test_hard_back_substitution_reproduces_d(self):
        """Independent check exercised directly: correct construction never
        raises."""
        import random

        for seed in range(30):
            rng = random.Random(f"elevation_depression_problem:hard:{seed}")
            _hard_tier_two_observation(rng)  # must not raise

    def test_independent_check_fires_on_corrupted_answer(self):
        """Each tier's independent check function, called directly with a
        corrupted answer, must raise VerificationError."""
        with pytest.raises(VerificationError):
            _check_elevation(d=10.0, theta_deg=30.0, h=999.0)
        with pytest.raises(VerificationError):
            _check_depression(H=10.0, theta_deg=30.0, dist=999.0)
        with pytest.raises(VerificationError):
            _check_two_observation(theta1_deg=60.0, theta2_deg=30.0, d=20.0, h=999.0)

    def test_independent_check_accepts_correct_values(self):
        """The same checks must NOT raise on a genuinely correct triple."""
        import math

        d, theta = 10.0, 30.0
        h = d * math.tan(math.radians(theta))
        _check_elevation(d, theta, h)  # must not raise

        H = 10.0
        dist = H / math.tan(math.radians(theta))
        _check_depression(H, theta, dist)  # must not raise

        theta1, theta2, dsep = 60.0, 30.0, 20.0
        cot1 = 1 / math.tan(math.radians(theta1))
        cot2 = 1 / math.tan(math.radians(theta2))
        h2 = dsep / (cot2 - cot1)
        _check_two_observation(theta1, theta2, dsep, h2)  # must not raise

    def test_working_nonempty(self):
        for tier in TIERS:
            for seed in range(20):
                q = _make_b(seed, tier)
                assert isinstance(q.working, list) and len(q.working) >= 2
                for line in q.working:
                    assert isinstance(line, str) and line


# --------------------------------------------------------------------------
# Skill loading
# --------------------------------------------------------------------------


class TestSkillsLoad:
    def test_skill_a_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-review-sine-cosine-and-tangent-as-ratios-of"), None)
        assert skill is not None
        assert skill.answer == "fraction"
        assert skill.prerequisites == []
        assert skill.placement_eligible is False
        assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard"}
        assert len(skill.misconceptions) >= 1

    def test_skill_b_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-work-with-the-concepts-of-angle-of-elevation"), None)
        assert skill is not None
        assert skill.answer == "number"
        assert skill.prerequisites == []
        assert skill.placement_eligible is False
        assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
        assert len(skill.misconceptions) >= 1
