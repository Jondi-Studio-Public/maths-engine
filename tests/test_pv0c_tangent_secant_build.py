"""Tests for the tangent-secant length relation, `senior-when-a-secant-meeting-the-circle-at-a` (ACMSM036)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_pairwise_distinct
from mathgen.generators.pv0c_tangent_secant_build import _TRIPLE_POOL, _PYTHAG_POOL

TOPIC = "pv0c_tangent_secant"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

SKILLS = load_skills()[1]


class TestTangentSecant:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = make(TOPIC, seed=seed, difficulty=difficulty)
            q2 = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 1
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_two_distractors_on_85_percent_of_draws(self, difficulty):
        n = 200
        at_least_two = sum(
            1 for seed in range(n) if len(make(TOPIC, seed=seed, difficulty=difficulty).distractors) >= 2
        )
        share = at_least_two / n
        assert share >= 0.85, f"{difficulty}: only {share:.2f} of draws had >= 2 distractors"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_no_distractor_echoes_a_printed_number(self, difficulty):
        """No distractor may equal any number printed in the prompt -- an echoed given is identifiable without
        any geometry, not a misconception."""
        n = 200
        echoes = 0
        for seed in range(n):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            printed = set(re.findall(r"\d+", q.text))
            for dist in q.distractors:
                if dist in printed:
                    echoes += 1
        assert echoes == 0, f"{difficulty}: {echoes} distractor(s) echoed a printed number over {n} draws"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractor_count_distribution(self, difficulty):
        """0/1/2/3+ distractor counts over 200 draws."""
        n = 200
        dist = Counter()
        for seed in range(n):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            dist[min(len(q.distractors), 3)] += 1
        assert dist[0] == 0, f"{difficulty}: some draw had zero distractors: {dist}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors_and_wrong(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed} {difficulty}: rejected answer {q.answer_text}"
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed} {difficulty}: accepted distractor {dist}"
            wrong = str(int(q.answer_text) + 10000)
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed} {difficulty}: accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_is_plain_positive_integer_string_under_1000(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            assert 0 < q.answer < 1000
            assert q.answer_text == str(q.answer)
            for dist in q.distractors:
                assert re.fullmatch(r"\d+", dist), f"distractor {dist!r} not formatted like the answer"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_concentration_under_25_percent_at_200_seeds(self, difficulty):
        n = 200
        counts = Counter(make(TOPIC, seed=seed, difficulty=difficulty).answer_text for seed in range(n))
        top_share = counts.most_common(1)[0][1] / n
        assert top_share <= 0.25, f"{difficulty}: top answer share {top_share:.2f} over {n} seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_convention_stated_every_time(self, difficulty):
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "between A and M" in q.text or "secant from M passes through O" in q.text

    def test_easy_relation_and_plausibility(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty="easy")
            am = int(re.search(r"AM (?:is|=) (\d+) cm", q.text).group(1))
            bm = int(re.search(r"BM (?:is|=) (\d+) cm", q.text).group(1))
            tm = int(q.answer_text)
            assert am * bm == tm * tm, f"seed {seed}: {am}*{bm} != {tm}^2"
            assert bm < tm < am
            ratio = Fraction(am, bm)
            assert Fraction(13, 10) <= ratio <= Fraction(9, 1), f"seed {seed}: ratio {ratio} out of plausible bounds"
            small = min(am, bm, tm)
            large = max(am, bm, tm)
            if small == 1:
                assert large <= 50

    def test_triple_pool_plausibility(self):
        for d, p, q, am, bm, tm, ab in _TRIPLE_POOL:
            assert am < 1000 and bm < 1000 and tm < 1000 and ab < 1000
            ratio = Fraction(am, bm)
            assert Fraction(13, 10) <= ratio <= Fraction(9, 1)
            small = min(am, bm, tm, ab)
            large = max(am, bm, tm, ab)
            if small == 1:
                assert large <= 50
            assert am * bm == tm * tm
            assert am == bm + ab

    def test_pythagorean_pool_plausibility(self):
        for r, tm, mo, bm, am in _PYTHAG_POOL:
            assert tm * tm + r * r == mo * mo
            assert bm * am == tm * tm
            ratio = Fraction(am, bm)
            assert Fraction(13, 10) <= ratio <= Fraction(9, 1)

    def test_hard_uses_whole_secant_not_chord(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty="hard")
            bm = int(re.search(r"BM (?:is|=) (\d+) cm", q.text).group(1))
            asks_for_tm = q.text.rstrip(".").endswith("TM, in cm")
            asks_for_ab = "chord AB, in cm" in q.text
            assert asks_for_tm != asks_for_ab, q.text
            if asks_for_tm:
                ab = int(re.search(r"AB (?:is|=) (\d+) cm", q.text).group(1))
                am = ab + bm
                tm = int(q.answer_text)
                assert am * bm == tm * tm
            else:
                tm = int(re.search(r"TM (?:is|=) (\d+) cm", q.text).group(1))
                ab = int(q.answer_text)
                am = ab + bm
                assert am * bm == tm * tm

    def test_exam_quadratic_and_centre_variants(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty="exam")
            if "centre O" in q.text:
                r = int(re.search(r"radius (\d+) cm", q.text).group(1))
                tm = int(re.search(r"TM is (\d+) cm", q.text).group(1))
                mo = int(q.answer_text)
                assert tm * tm + r * r == mo * mo
            else:
                tm = int(re.search(r"TM is (\d+) cm", q.text).group(1))
                ab = int(re.search(r"chord AB is (\d+) cm", q.text).group(1))
                x = int(q.answer_text)
                if "length of BM" in q.text:
                    assert x * (x + ab) == tm * tm
                else:
                    assert x * (x - ab) == tm * tm

    def test_check_pairwise_distinct_used(self):
        check_pairwise_distinct(1, 2, 3)
        with pytest.raises(VerificationError):
            check_pairwise_distinct(1, 2, 2)

    def test_skill_loads(self):
        skill = next((s for s in SKILLS if s.id == "senior-when-a-secant-meeting-the-circle-at-a"), None)
        assert skill is not None
        assert skill.domain == "differentiation"
        assert skill.answer == "number"
        assert skill.prerequisites == ["differentiate-polynomial"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 4
        assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
        exam_source = next(s for s in skill.sources if s.difficulty == "exam")
        assert exam_source.stages == ["exam"]
