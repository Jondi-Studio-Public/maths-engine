"""Tests `f10_trig_ratios_similarity`: determinism, tiers, distractors, grading, skill loading."""

import re
from collections import Counter

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.f10_trig_ratios_similarity_build import _check_easy, _check_hard_text, _check_ratio_text
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills

TOPIC = "f10_trig_ratios_similarity"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)
DP = {"easy": 0, "medium": 1, "hard": 2}


def _dp(text):
    return len(text.split(".", 1)[1]) if "." in text else 0


class TestShape:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            assert (
                make(TOPIC, seed=seed, difficulty=difficulty).to_json()
                == make(TOPIC, seed=seed, difficulty=difficulty).to_json()
            )

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_and_distractors(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert len(q.working) >= 3 and all(isinstance(w, str) and w for w in q.working)
            assert q.answer_text not in q.distractors
            assert len(set(q.distractors)) == len(q.distractors) >= 3
            assert set(q.option_notes) == set(q.distractors)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_stated_precision(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert _dp(q.answer_text) == DP[difficulty]
            assert all(_dp(d) == DP[difficulty] for d in q.distractors)
            if difficulty != "easy":
                word = "1 decimal place" if difficulty == "medium" else "2 decimal places"
                assert f"correct to {word}" in q.text

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_30_distinct_questions(self, difficulty):
        assert len({make(TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}) >= 30

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_pool_is_varied(self, difficulty):
        counts = Counter(make(TOPIC, seed=s, difficulty=difficulty).answer_text for s in range(200))
        assert max(counts.values()) <= 0.2 * 200

    def test_never_asks_for_an_angle(self):
        for difficulty in TIERS:
            for seed in range(60):
                text = make(TOPIC, seed=seed, difficulty=difficulty).text.lower()
                assert "find the size" not in text and "angle is unknown" not in text


class TestGrading:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_accepted_distractors_rejected(self, difficulty):
        grade = GRADERS["number"]
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert grade(q, q.answer_text).correct
            for d in q.distractors:
                assert not grade(q, d).correct, f"seed {seed}: accepted distractor {d}"

    @pytest.mark.parametrize("difficulty", ["medium", "hard"])
    def test_unrounded_and_off_by_a_unit_rejected(self, difficulty):
        grade = GRADERS["number"]
        unit = 10 ** -DP[difficulty]
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            v = float(q.answer_text)
            assert not grade(q, f"{v + 2 * unit:.{DP[difficulty]}f}").correct
            assert not grade(q, f"{v - 2 * unit:.{DP[difficulty]}f}").correct
            assert not grade(q, f"{v + unit / 3:.6f}").correct  # more places than asked, not a rounding

    def test_easy_rejects_a_near_integer(self):
        grade = GRADERS["number"]
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty="easy")
            assert not grade(q, str(int(q.answer_text) + 1)).correct
            assert not grade(q, f"{int(q.answer_text) + 0.3}").correct

    @pytest.mark.parametrize("difficulty", ["medium", "hard"])
    def test_printed_ratio_route_and_calculator_route_agree(self, difficulty):
        """Rounding the answer from the printed 4-d.p. ratios gives the stated answer (so a learner using either is graded alike)."""
        for seed in range(80):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if difficulty == "medium":
                _check_ratio_text(q.text, q.answer_text, DP[difficulty])
            else:
                _check_hard_text(q.text, q.answer_text, DP[difficulty])


class TestTierShapes:
    def test_easy_is_similarity_with_no_trig_values(self):
        for seed in range(40):
            text = make(TOPIC, seed=seed, difficulty="easy").text
            assert "sin" not in text and "degrees" not in text and "same angle" in text

    def test_medium_prints_all_three_ratios_for_the_angle(self):
        for seed in range(40):
            text = make(TOPIC, seed=seed, difficulty="medium").text
            assert len(re.findall(r"(?:sin|cos|tan) \d+ degrees = \d\.\d{4}", text)) >= 3

    def test_hard_names_vertices_and_prints_both_acute_angles(self):
        for seed in range(40):
            text = make(TOPIC, seed=seed, difficulty="hard").text
            assert "right angle at" in text
            assert len(re.findall(r"(?:sin|cos|tan) \d+ degrees", text)) == 6

    @pytest.mark.parametrize("difficulty", ["medium", "hard"])
    def test_every_role_pair_is_used(self, difficulty):
        seen = set()
        for seed in range(150):
            seen.add(
                make(TOPIC, seed=seed, difficulty=difficulty)
                .working[-3 if difficulty == "hard" else -3]
                .split(" so use ")[1]
                .split(" ")[0]
            )
        assert seen == {"sin", "cos", "tan"}


class TestIndependentCheck:
    def test_easy_check_fires_on_corrupted_answer(self):
        q = make(TOPIC, seed=3, difficulty="easy")
        with pytest.raises(VerificationError):
            _check_easy(q.text, str(int(q.answer_text) + 1))

    def test_medium_check_fires_on_corrupted_answer(self):
        q = make(TOPIC, seed=3, difficulty="medium")
        with pytest.raises(VerificationError):
            _check_ratio_text(q.text, f"{float(q.answer_text) + 0.1:.1f}", 1)

    def test_hard_check_fires_on_corrupted_answer(self):
        q = make(TOPIC, seed=3, difficulty="hard")
        with pytest.raises(VerificationError):
            _check_hard_text(q.text, f"{float(q.answer_text) + 0.01:.2f}", 2)


class TestSkillLoads:
    def test_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "f10-trig-ratios-similarity"), None)
        assert skill is not None
        assert skill.domain == "trigonometry" and skill.answer == "number" and skill.status == "active"
        assert [s.difficulty for s in skill.sources] == TIERS
        assert skill.placement_eligible is False
        assert "ac9m9sp01" in skill.tags
