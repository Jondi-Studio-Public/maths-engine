"""Tests for: relative frequency as a point estimate (ACMMM055,
``senior-use-relative-frequencies-obtained-from-data-as-point``) and histogram-based continuous probability
(ACMMM164, ``senior-use-relative-frequencies-and-histograms-obtained-from-data``)."""

from __future__ import annotations

from fractions import Fraction

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import REGISTRY
from mathgen.text_quality import prompt_shape

SEEDS = range(60)

RF_TOPIC = "p16a_relative_frequency_estimate"
RF_GRADER = GRADERS["fraction"]
RF_SKILL = "senior-use-relative-frequencies-obtained-from-data-as-point"

HP_TOPIC = "p16a_histogram_probability"
HP_GRADER = GRADERS["fraction"]
HP_SKILL = "senior-use-relative-frequencies-and-histograms-obtained-from-data"


class _RFCommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            result = RF_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected {q.answer_text!r}"
            for dist in q.distractors:
                result = RF_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            answer_value = Fraction(q.answer["p"], q.answer["q"])
            wrong = answer_value + Fraction(9, 17)
            assert not RF_GRADER(q, str(wrong)).correct

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(RF_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(RF_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == RF_SKILL)
        assert skill.answer == "fraction"
        assert skill.placement_eligible is False
        assert any(src.topic == RF_TOPIC and src.difficulty == self.TIER for src in skill.sources)


class TestRFEasy(_RFCommon):
    TIER = "easy"

    def test_probability_between_zero_and_one(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            value = Fraction(q.answer["p"], q.answer["q"])
            assert 0 < value < 1


class TestRFMedium(_RFCommon):
    TIER = "medium"

    def test_asks_for_or_union(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert " or " in q.text


class TestRFHard(_RFCommon):
    TIER = "hard"

    def test_asks_for_complement(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert "NOT" in q.text


class TestRFExam(_RFCommon):
    TIER = "exam"

    def test_projects_onto_future_trials(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert "next" in q.text and "trials" in q.text

    def test_answer_is_a_whole_number(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer["q"] == 1


def test_relative_frequency_topic_registered_once():
    _, fn = REGISTRY[RF_TOPIC]
    assert fn.__module__.endswith("p16a_relative_freq_build")


# --------------------------------------------------------------------------
# p16a_histogram_probability
# --------------------------------------------------------------------------


class _HPCommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(HP_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(HP_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(HP_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(HP_TOPIC, seed=seed, difficulty=self.TIER)
            result = HP_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected {q.answer_text!r}"
            for dist in q.distractors:
                result = HP_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            answer_value = Fraction(q.answer["p"], q.answer["q"])
            wrong = answer_value + Fraction(9, 19)
            assert not HP_GRADER(q, str(wrong)).correct

    def test_probability_strictly_between_zero_and_one(self):
        for seed in SEEDS:
            q = make(HP_TOPIC, seed=seed, difficulty=self.TIER)
            value = Fraction(q.answer["p"], q.answer["q"])
            assert 0 < value < 1

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(HP_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(HP_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == HP_SKILL)
        assert skill.answer == "fraction"
        assert skill.placement_eligible is False
        assert any(src.topic == HP_TOPIC and src.difficulty == self.TIER for src in skill.sources)


class TestHPEasy(_HPCommon):
    TIER = "easy"

    def test_interval_matches_a_single_class(self):
        for seed in SEEDS:
            q = make(HP_TOPIC, seed=seed, difficulty=self.TIER)
            assert "[" in q.text and ")" in q.text


class TestHPMedium(_HPCommon):
    TIER = "medium"


class TestHPHard(_HPCommon):
    TIER = "hard"

    def test_asks_for_at_least(self):
        for seed in SEEDS:
            q = make(HP_TOPIC, seed=seed, difficulty=self.TIER)
            assert "at least" in q.text


class TestHPExam(_HPCommon):
    TIER = "exam"

    def test_states_uniform_assumption(self):
        for seed in SEEDS:
            q = make(HP_TOPIC, seed=seed, difficulty=self.TIER)
            assert "uniformly" in q.text
            assert "at least" in q.text


def test_histogram_probability_topic_registered_once():
    _, fn = REGISTRY[HP_TOPIC]
    assert fn.__module__.endswith("p16a_relative_freq_build")


def test_module_name_ends_in_build():
    from mathgen.generators import p16a_relative_freq_build

    assert p16a_relative_freq_build.__name__.endswith("_build")
