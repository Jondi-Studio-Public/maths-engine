"""Tests the angle-at-the-centre and intersecting-chords circle-theorem generators and that their skills load.

Circle theorems build."""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p8d_circle_theorems_build import (
    _verify_angle_at_centre,
    _verify_chord_products,
    _verify_quadratic_positive_root,
)

ANGLE_TOPIC = "the_angle_at_the_centre_subtended_by_an_arc"
CHORD_TOPIC = "when_two_chords_of_a_circle_intersect_the_product"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

_BAD_SIGN_PATTERNS = [re.compile(r"\+ -"), re.compile(r"- -")]


class TestAngleAtCentre:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            q2 = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors_and_wrong(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed} {difficulty}: rejected answer {q.answer_text}"
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed} {difficulty}: accepted distractor {dist}"
            wrong = str(int(q.answer_text) + 7)
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed} {difficulty}: accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_is_integer(self, difficulty):
        for seed in SEEDS:
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert q.answer_text == str(q.answer)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_no_bad_sign_patterns(self, difficulty):
        for seed in SEEDS:
            q = make(ANGLE_TOPIC, seed=seed, difficulty=difficulty)
            for field in [q.text, q.latex, *q.working]:
                for pat in _BAD_SIGN_PATTERNS:
                    assert not pat.search(field)

    @pytest.mark.parametrize("difficulty,min_distinct", [("easy", 20), ("medium", 20), ("hard", 20)])
    def test_at_least_20_distinct_questions(self, difficulty, min_distinct):
        texts = {make(ANGLE_TOPIC, seed=seed, difficulty=difficulty).text for seed in SEEDS}
        assert len(texts) >= min_distinct, f"{difficulty}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_easy_never_produces_diameter_case(self):
        # The easy tier must never emit a 180-degree central angle (that is
        # angle_in_semicircle's territory, not this skill's).
        for seed in SEEDS:
            q = make(ANGLE_TOPIC, seed=seed, difficulty="easy")
            m = re.search(r"is (\d+) degrees", q.text)
            assert m and int(m.group(1)) != 180

    def test_verify_angle_at_centre_fires_on_corrupted_answer(self):
        # The independent coordinate check must reject a wrong expected value.
        with pytest.raises(VerificationError):
            _verify_angle_at_centre(48, True, 48)  # should be 24, not 48
        _verify_angle_at_centre(48, True, 24)  # correct value must pass

    def test_verify_angle_at_centre_reflex_case(self):
        with pytest.raises(VerificationError):
            _verify_angle_at_centre(160, False, 99)  # should be 100
        _verify_angle_at_centre(160, False, 100)


class TestWhenTwoChordsIntersect:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(CHORD_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = make(CHORD_TOPIC, seed=seed, difficulty=difficulty)
            q2 = make(CHORD_TOPIC, seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(CHORD_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors_and_wrong(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(CHORD_TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed} {difficulty}: rejected answer {q.answer_text}"
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed} {difficulty}: accepted distractor {dist}"
            wrong = str(int(q.answer_text) + 1000)
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed} {difficulty}: accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_is_positive_integer(self, difficulty):
        for seed in SEEDS:
            q = make(CHORD_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert q.answer > 0
            assert q.answer_text == str(q.answer)

    def test_hard_tier_never_offers_negative_distractor(self):
        for seed in SEEDS:
            q = make(CHORD_TOPIC, seed=seed, difficulty="hard")
            for dist in q.distractors:
                assert int(dist) > 0, f"seed {seed}: negative-length distractor {dist!r} in {q.text!r}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions(self, difficulty):
        texts = {make(CHORD_TOPIC, seed=seed, difficulty=difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_medium_tier_states_mismatched_unit(self):
        for seed in SEEDS:
            q = make(CHORD_TOPIC, seed=seed, difficulty="medium")
            assert " mm" in q.text and " cm" in q.text

    def test_verify_chord_products_fires_on_corrupted_answer(self):
        with pytest.raises(VerificationError):
            _verify_chord_products(6, 10, 5, 13)  # should be 12
        _verify_chord_products(6, 10, 5, 12)

    def test_verify_quadratic_root_fires_on_corrupted_answer(self):
        # x(x + 6) = 112 -> x = 8
        with pytest.raises(VerificationError):
            _verify_quadratic_positive_root(6, 112, 7)
        _verify_quadratic_positive_root(6, 112, 8)


class TestSkillsLoad:
    def test_angle_at_centre_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-the-angle-at-the-centre-subtended-by-an"), None)
        assert skill is not None
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "number"
        assert skill.prerequisites == ["f10-angle-relationships"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 4  # three practice tiers plus the exam source

    def test_when_two_chords_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-when-two-chords-of-a-circle-intersect-the"), None)
        assert skill is not None
        assert skill.domain == "measurement"
        assert skill.answer == "number"
        assert skill.prerequisites == ["convert-metric-length"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 3
