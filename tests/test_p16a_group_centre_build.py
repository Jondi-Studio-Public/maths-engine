"""Tests for: group-centre difference (ACMGM033,
``senior-implement-the-statistical-investigation-process-to-answer-questions``) and linear trend value
(ACMGM093, ``senior-implement-the-statistical-investigation-process-to-answer-questions-acmgm093``)."""

from __future__ import annotations

import re

from fractions import Fraction

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import REGISTRY
from mathgen.text_quality import prompt_shape

SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

GC_TOPIC = "p16a_group_centre_difference"
GC_GRADER = GRADERS["fraction"]
GC_SKILL = "senior-implement-the-statistical-investigation-process-to-answer-questions"

TV_TOPIC = "p16a_trend_value"
TV_GRADER = GRADERS["number"]
TV_SKILL = "senior-implement-the-statistical-investigation-process-to-answer-questions-acmgm093"


# --------------------------------------------------------------------------
# p16a_group_centre_difference
# --------------------------------------------------------------------------


class _GCCommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_working_nonempty(self):
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            assert isinstance(q.working, list) and len(q.working) >= 1

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            result = GC_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected canonical {q.answer_text!r}"
            for dist in q.distractors:
                result = GC_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            answer_value = Fraction(q.answer["p"], q.answer["q"])
            wrong = answer_value + Fraction(7, 3)
            result = GC_GRADER(q, str(wrong))
            assert not result.correct, f"seed {seed}: accepted deliberately wrong {wrong}"

    def test_answer_is_exact_fraction_in_lowest_terms(self):
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            assert isinstance(q.answer, dict) and {"p", "q"} <= set(q.answer)
            frac = Fraction(q.answer["p"], q.answer["q"])
            assert frac.numerator == q.answer["p"] and frac.denominator == q.answer["q"]

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(GC_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20, f"only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(GC_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == GC_SKILL)
        assert skill.answer == "fraction"
        assert skill.placement_eligible is False
        assert any(src.topic == GC_TOPIC and src.difficulty == self.TIER for src in skill.sources)


class TestGCEasy(_GCCommon):
    TIER = "easy"

    def test_asks_for_the_mean_explicitly(self):
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            assert "mean" in q.text


class TestGCMedium(_GCCommon):
    TIER = "medium"

    def test_asks_for_the_median_explicitly(self):
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            assert "median" in q.text

    def test_groups_have_even_size_six(self):
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            lists = re.findall(r"(?:\d+, )+\d+", q.text)
            assert len(lists) == 2, q.text
            for group_list in lists:
                assert len(group_list.split(", ")) == 6


class TestGCHard(_GCCommon):
    TIER = "hard"

    def test_data_given_as_frequency_tables(self):
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            assert "times)" in q.text
            assert "mean" in q.text


class TestGCExam(_GCCommon):
    TIER = "exam"

    def test_never_says_mean_or_median_literally(self):
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            assert "mean" not in q.text.lower()
            assert "median" not in q.text.lower()

    def test_both_reasoning_flavours_occur_over_60_seeds(self):
        # Neither flavour names its measure; each is identified by its own
        # REASON instead (outlier present -> median is implied; no outlier
        # -> mean is implied) -- the actual requirement, that the
        # exam tier make the learner pick the measure the wording asks for.
        uses_outlier_reason = 0
        uses_no_outlier_reason = 0
        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            if "outlier" in q.text and "not affected" in q.text:
                uses_outlier_reason += 1
            if "None of the data" in q.text:
                uses_no_outlier_reason += 1
        assert uses_outlier_reason > 0 and uses_no_outlier_reason > 0

    def test_mean_and_median_of_the_data_genuinely_differ(self):
        # The whole point of the redesign: since neither measure is named,
        # picking the wrong one must give a different, wrong answer -- so
        # the mean-difference and the median-difference of the stated data
        # are never allowed to coincide.
        import re as _re

        for seed in SEEDS:
            q = make(GC_TOPIC, seed=seed, difficulty=self.TIER)
            lists = _re.findall(r"(?:\d+, )+\d+", q.text)
            assert len(lists) == 2, q.text
            values_a = [int(x) for x in lists[0].split(", ")]
            values_b = [int(x) for x in lists[1].split(", ")]
            from fractions import Fraction
            import statistics as _stats

            mean_diff = Fraction(sum(values_a), len(values_a)) - Fraction(sum(values_b), len(values_b))
            med_diff = Fraction(_stats.median(values_a)).limit_denominator(1000) - Fraction(
                _stats.median(values_b)
            ).limit_denominator(1000)
            assert mean_diff != med_diff, f"seed {seed}: mean and median differences coincide. {q.text}"


def test_group_centre_topic_registered_once():
    _, fn = REGISTRY[GC_TOPIC]
    assert fn.__module__.endswith("p16a_group_centre_build")


# --------------------------------------------------------------------------
# p16a_trend_value
# --------------------------------------------------------------------------


def _num(text: str) -> float:
    return float(text)


class _TVCommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_distractors_never_within_grader_tolerance(self):
        for seed in SEEDS:
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            answer_value = _num(q.answer_text)
            for dist in q.distractors:
                assert abs(_num(dist) - answer_value) > 0.051

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            result = TV_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected {q.answer_text!r}"
            for dist in q.distractors:
                result = TV_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            wrong = _num(q.answer_text) + 11.3
            result = TV_GRADER(q, str(round(wrong, 1)))
            assert not result.correct

    def test_answer_states_exactly_one_decimal_place(self):
        for seed in SEEDS:
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            literal = re.search(r"-?\d+\.\d+", q.answer_text)
            assert literal, q.answer_text
            assert len(literal.group().split(".")[1]) == 1

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(TV_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(TV_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_never_asks_for_a_moving_average(self):
        for seed in SEEDS:
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            assert "moving average" not in q.text.lower()
            assert "window" not in q.text.lower()

    def test_answer_and_every_given_value_are_non_negative_over_200_seeds(self):
        import re as _re

        given_re = _re.compile(r"\bis (-?\d+\.\d+)\b")
        for seed in range(200):
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            assert _num(q.answer_text) >= 0, f"seed {seed}: negative answer. {q.text}"
            for given in given_re.findall(q.text):
                assert _num(given) >= 0, f"seed {seed}: negative given value {given!r}. {q.text}"

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == TV_SKILL)
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert any(src.topic == TV_TOPIC and src.difficulty == self.TIER for src in skill.sources)


class TestTVEasy(_TVCommon):
    TIER = "easy"

    def test_equation_given_directly(self):
        for seed in SEEDS:
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            assert "T(t) =" in q.text


class TestTVMedium(_TVCommon):
    TIER = "medium"

    def test_period_label_present(self):
        for seed in SEEDS:
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            assert re.search(r"Year \d+", q.text)


class TestTVHard(_TVCommon):
    TIER = "hard"

    def test_two_points_given_not_equation(self):
        for seed in SEEDS:
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            assert "T(t) =" not in q.text
            assert "trend value in" in q.text


class TestTVExam(_TVCommon):
    TIER = "exam"

    def test_two_nonconsecutive_period_labels_no_equation(self):
        for seed in SEEDS:
            q = make(TV_TOPIC, seed=seed, difficulty=self.TIER)
            assert "T(t) =" not in q.text
            assert "consecutive" not in q.text
            assert q.text.count("the trend value for") == 3
            assert re.search(r"\d+ \w+s per Year", q.text)


def test_trend_value_topic_registered_once():
    _, fn = REGISTRY[TV_TOPIC]
    assert fn.__module__.endswith("p16a_group_centre_build")


def test_module_name_ends_in_build():
    from mathgen.generators import p16a_group_centre_build

    assert p16a_group_centre_build.__name__.endswith("_build")
