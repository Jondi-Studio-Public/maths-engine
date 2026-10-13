"""Tests for: conditional independence (ACMMM060,
``senior-use-relative-frequencies-obtained-from-data-as-point-acmmm060``) and estimated discrete distribution
(ACMMM137, ``senior-use-relative-frequencies-obtained-from-data-to-obtain``)."""

from __future__ import annotations

from fractions import Fraction

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import REGISTRY
from mathgen.text_quality import prompt_shape

SEEDS = range(60)

CI_TOPIC = "p16a_conditional_independence"
CI_GRADER = GRADERS["fraction"]
CI_SKILL = "senior-use-relative-frequencies-obtained-from-data-as-point-acmmm060"

ED_TOPIC = "p16a_estimated_distribution"
ED_GRADER = GRADERS["ordered_list"]
ED_SKILL = "senior-use-relative-frequencies-obtained-from-data-to-obtain"


class _CICommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            result = CI_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected {q.answer_text!r}"
            for dist in q.distractors:
                result = CI_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            answer_value = Fraction(q.answer["p"], q.answer["q"])
            wrong = answer_value + Fraction(11, 13)
            result = CI_GRADER(q, str(wrong))
            assert not result.correct

    def test_answer_within_valid_probability_difference_range(self):
        for seed in SEEDS:
            q = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            value = Fraction(q.answer["p"], q.answer["q"])
            assert -1 <= value <= 1

    def test_mentions_independence_in_prompt(self):
        for seed in SEEDS:
            q = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            assert "independent" in q.text

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(CI_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(CI_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == CI_SKILL)
        assert skill.answer == "fraction"
        assert skill.placement_eligible is False
        assert any(src.topic == CI_TOPIC and src.difficulty == self.TIER for src in skill.sources)

    def test_zero_diff_share_under_25_percent_over_400_seeds(self):
        # Answer-concentration measure: the answer 0 (exact
        # independence) must not dominate the answer space.
        zero_count = 0
        for seed in range(400):
            q = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            if Fraction(q.answer["p"], q.answer["q"]) == 0:
                zero_count += 1
        assert zero_count / 400 < 0.25, f"{self.TIER}: 0 appears in {zero_count}/400 draws"


class TestCIEasy(_CICommon):
    TIER = "easy"

    def test_all_four_counts_given_directly(self):
        for seed in SEEDS:
            q = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            assert "both" in q.text and "neither" in q.text


class TestCIMedium(_CICommon):
    TIER = "medium"

    def test_only_totals_and_one_cell_given(self):
        for seed in SEEDS:
            q = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            assert "also" in q.text
            assert "neither" not in q.text


class TestCIHard(_CICommon):
    TIER = "hard"

    def test_percentages_given(self):
        for seed in SEEDS:
            q = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            assert "%" in q.text


class TestCIExam(_CICommon):
    TIER = "exam"

    def test_three_categories_present(self):
        for seed in SEEDS:
            q = make(CI_TOPIC, seed=seed, difficulty=self.TIER)
            assert "walking, cycling or public transport" in q.text


def test_conditional_independence_topic_registered_once():
    _, fn = REGISTRY[CI_TOPIC]
    assert fn.__module__.endswith("p16a_conditional_dist_build")


# --------------------------------------------------------------------------
# p16a_estimated_distribution
# --------------------------------------------------------------------------


class _EDCommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors_and_reversed(self):
        for seed in SEEDS:
            q = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            result = ED_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected {q.answer_text!r}"
            for dist in q.distractors:
                result = ED_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            reversed_text = ", ".join(reversed(q.answer_text.split(", ")))
            if reversed_text != q.answer_text:
                result = ED_GRADER(q, reversed_text)
                assert not result.correct, "reversing the order must be rejected (order-sensitive grader)"

    def test_answer_sums_to_one(self):
        for seed in SEEDS:
            q = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            total = sum(Fraction(str(x)) for x in q.answer)
            assert total == 1

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(ED_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(ED_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_prompt_states_increasing_order(self):
        for seed in SEEDS:
            q = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            assert "INCREASING ORDER" in q.text

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == ED_SKILL)
        assert skill.answer == "ordered_list"
        assert skill.placement_eligible is False
        assert any(
            (src.topic == ED_TOPIC or self.TIER == "exam") and src.difficulty == self.TIER for src in skill.sources
        )


class TestEDEasy(_EDCommon):
    TIER = "easy"

    def test_three_values(self):
        for seed in SEEDS:
            q = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            assert len(q.answer) == 3


class TestEDMedium(_EDCommon):
    TIER = "medium"

    def test_four_values(self):
        for seed in SEEDS:
            q = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            assert len(q.answer) == 4


class TestEDHard(_EDCommon):
    TIER = "hard"

    def test_one_frequency_hidden(self):
        for seed in SEEDS:
            q = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            assert "?" in q.text


class TestEDExam(_EDCommon):
    TIER = "exam"

    def test_two_frequencies_hidden_and_tied_by_a_ratio(self):
        for seed in SEEDS:
            q = make(ED_TOPIC, seed=seed, difficulty=self.TIER)
            # Two missing table entries ("x=k: ? <plural>"), plus the
            # separate "The two '?' entries are not shown" sentence.
            assert q.text.count(": ?") == 2, q.text
            assert "times the frequency for" in q.text


def test_estimated_distribution_topic_registered_once():
    _, fn = REGISTRY[ED_TOPIC]
    assert fn.__module__.endswith("p16a_conditional_dist_build")


def test_module_name_ends_in_build():
    from mathgen.generators import p16a_conditional_dist_build

    assert p16a_conditional_dist_build.__name__.endswith("_build")
