"""Focused contract tests for: the new ``exam`` source on each of ``calculate-median``, ``calculate-mean``,
``calculate-mode`` (``mathgen/generators/exd_e3_statistics_build.py``)."""

import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.exd_e3_statistics_build import (
    _verify_mean_total,
    _verify_median_middle,
    _verify_unique_mode,
)
from mathgen.text_quality import prompt_shape

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(60)
TOPICS = ("exam_median_summary_statistic", "exam_mean_summary_statistic", "exam_mode_summary_statistic")

_VALUES_RUN_RE = re.compile(r"\d+(?:,\s*\d+)+")


def _extract_values(text: str) -> list[int]:
    """The drawn data set as it appears in `text`, template-independent:
    the longest comma-separated run of integers in the rendered prompt is
    always the values list (every template's other digit content -- a
    count, "1 decimal place" -- never itself has a comma-separated run)."""
    runs = _VALUES_RUN_RE.findall(text)
    longest = max(runs, key=lambda s: s.count(","))
    return [int(tok) for tok in longest.split(",")]


# --- determinism / distractor hygiene, all three topics ---------------------


@pytest.mark.parametrize("topic", TOPICS)
def test_deterministic_across_seeds(topic):
    for seed in SEEDS:
        q1 = make(topic, seed=seed, difficulty="exam")
        q2 = make(topic, seed=seed, difficulty="exam")
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("topic", TOPICS)
def test_distractors_are_non_empty_distinct_and_never_the_answer(topic):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        assert q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("topic", TOPICS)
def test_real_grader_accepts_answer_and_rejects_every_distractor(topic):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        assert grade_number(q, q.answer_text).correct, (seed, q.text, q.answer_text)
        for distractor in q.distractors:
            assert not grade_number(q, distractor).correct, (seed, q.text, distractor)


@pytest.mark.parametrize("topic", TOPICS)
def test_real_grader_rejects_a_deliberately_wrong_answer(topic):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        # A value one whole unit off the true answer, formatted the same way
        # the canonical answer_text is -- never within any stated-precision
        # tolerance (>= 1, always > 0.05).
        wrong = Fraction(q.answer) + 1 if isinstance(q.answer, (int, float)) else None
        assert wrong is not None
        wrong_text = str(int(wrong)) if wrong.denominator == 1 else f"{float(wrong):.1f}"
        assert not grade_number(q, wrong_text).correct, (seed, q.text, wrong_text)


@pytest.mark.parametrize("topic", TOPICS)
def test_prompt_variety_at_least_three_shapes_over_60_seeds(topic):
    """Regression guard for a previously fixed defect."""
    shapes = {prompt_shape(make(topic, seed=seed, difficulty="exam")) for seed in SEEDS}
    assert len(shapes) >= 3, f"{topic} [exam] only {len(shapes)} distinct shapes over 60 seeds"


@pytest.mark.parametrize("topic", TOPICS)
def test_at_least_20_distinct_rendered_questions_over_60_seeds(topic):
    texts = {make(topic, seed=seed, difficulty="exam").text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic} [exam] only {len(texts)} distinct renders over 60 seeds"


@pytest.mark.parametrize("topic", TOPICS)
def test_answer_and_every_distractor_share_the_same_decimal_shape(topic):
    """No option is identifiable by carrying a different number of decimal
    places than the others (the `midpoint_coordinate_scalar` defect)."""

    def places(text: str) -> int:
        return len(text.split(".", 1)[1]) if "." in text else 0

    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        answer_places = places(q.answer_text)
        for distractor in q.distractors:
            assert places(distractor) == answer_places, (
                f"{topic} seed={seed}: answer {q.answer_text!r} has {answer_places} dp "
                f"but distractor {distractor!r} has {places(distractor)} dp"
            )


# --- exam_median_summary_statistic -----------------------------------------------


def test_median_count_is_always_even_and_has_at_least_one_repeat():
    for seed in SEEDS:
        q = make("exam_median_summary_statistic", seed=seed, difficulty="exam")
        # `working[0]` ("Order the values: 3, 5, 7, ....") has a fixed shape
        # regardless of which `phrase()` template rendered `text`.
        ordered = [int(tok) for tok in q.working[0].split(":", 1)[1].rstrip(".").split(",")]
        assert len(ordered) % 2 == 0
        assert len(ordered) in (8, 10, 12)
        assert len(ordered) != len(set(ordered)), f"seed {seed}: no repeat in {ordered}"


def test_median_answer_is_mean_of_two_middle_values_and_precision_matches_statement():
    saw_integer = False
    saw_decimal = False
    for seed in SEEDS:
        q = make("exam_median_summary_statistic", seed=seed, difficulty="exam")
        ordered = [int(tok) for tok in q.working[0].split(":", 1)[1].rstrip(".").split(",")]
        count = len(ordered)
        lower, upper = ordered[count // 2 - 1], ordered[count // 2]
        expected = Fraction(lower + upper, 2)
        if expected.denominator == 1:
            saw_integer = True
            assert "." not in q.answer_text
            assert "decimal place" not in q.text
            assert q.answer == int(expected)
        else:
            saw_decimal = True
            assert "decimal place" in q.text
            assert q.answer_text == f"{float(expected):.1f}"
            assert q.answer == float(expected)
    assert saw_integer, "never observed an integer median answer in 60 seeds"
    assert saw_decimal, "never observed a decimal (.5) median answer in 60 seeds"


def test_median_independent_check_fires_on_corrupted_answer():
    values = [3, 5, 5, 7, 9, 11, 13, 20]  # sorted: middle two are 7, 9
    _verify_median_middle(values, 7, 9)  # must not raise

    with pytest.raises(VerificationError):
        _verify_median_middle(values, 5, 9)  # wrong lower middle value


# --- exam_mean_summary_statistic ---------------------------------------------------


def test_mean_values_are_8_to_12_in_1_60_and_sum_divisible_by_count():
    for seed in SEEDS:
        q = make("exam_mean_summary_statistic", seed=seed, difficulty="exam")
        # `working[0]` ("Total: 5 + 10 + ... = 100.") has a fixed shape
        # regardless of which `phrase()` template rendered `text`.
        total_part = q.working[0].split(":", 1)[1].split("=", 1)[0]
        drawn = [int(tok) for tok in total_part.split("+")]
        assert 8 <= len(drawn) <= 12
        assert all(1 <= v <= 60 for v in drawn)
        assert sum(drawn) % len(drawn) == 0
        assert q.answer == sum(drawn) // len(drawn)
        assert isinstance(q.answer, int)
        assert "." not in q.answer_text


def test_mean_independent_check_fires_on_corrupted_answer():
    values = [10, 20, 30, 15, 25, 12, 28, 20]  # sum=160, count=8, mean=20
    _verify_mean_total(values, 8, 20)  # must not raise

    with pytest.raises(VerificationError):
        _verify_mean_total(values, 8, 21)  # wrong target mean


# --- exam_mode_summary_statistic ----------------------------------------------------


def test_mode_values_are_8_to_12_in_1_60_and_mode_occurs_exactly_three_times():
    for seed in SEEDS:
        q = make("exam_mode_summary_statistic", seed=seed, difficulty="exam")
        drawn = _extract_values(q.text)
        assert 8 <= len(drawn) <= 12
        assert all(1 <= v <= 60 for v in drawn)
        mode_value = q.answer
        assert isinstance(mode_value, int)
        assert drawn.count(mode_value) == 3
        for v in set(drawn):
            if v != mode_value:
                assert drawn.count(v) <= 2, f"seed {seed}: {v} ties or beats the mode"


def test_mode_independent_check_fires_on_corrupted_answer():
    values = [4, 4, 4, 5, 5, 6, 6, 9]  # 4 occurs 3 times: unique mode
    _verify_unique_mode(values, 4)  # must not raise

    tied = [4, 4, 5, 5, 6, 9]  # 4 and 5 tied at 2
    with pytest.raises(VerificationError):
        _verify_unique_mode(tied, 4)


# --- skill contracts -------------------------------------------------------


def test_three_skills_each_gained_exactly_one_new_exam_source():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    median = by_id["calculate-median"]
    mean = by_id["calculate-mean"]
    mode = by_id["calculate-mode"]

    for skill, easy_topic, exam_topic in (
        (median, "summary_statistic", "exam_median_summary_statistic"),
        (mean, "summary_statistic", "exam_mean_summary_statistic"),
        (mode, "summary_statistic", "exam_mode_summary_statistic"),
    ):
        assert skill.answer == "number"
        # None of the three skills set `placement_eligible` in their YAML
        # (adding one source only does not add it), so it
        # keeps the loader's own default of True -- not the `False` most
        # other skills declare explicitly.
        assert skill.placement_eligible is True
        assert skill.status == "active"
        # The median also gained an even-count medium source (w2_median_even) between the easy and exam ones.
        # The mean gained medium and hard tiers (w3_mean_tiers) in the same way.
        middle = [("w2_median_even", "medium")] if skill is median else []
        if skill is mean:
            middle = [("w3_mean_tiers", "medium"), ("w3_mean_tiers", "hard")]
        assert [(s.topic, s.difficulty) for s in skill.sources] == [
            (easy_topic, "easy"),
            *middle,
            (exam_topic, "exam"),
        ]
        assert skill.sources[-1].stages == ["exam"]

    assert median.prerequisites == ["order-numbers-to-10"]
    assert mean.prerequisites == ["whole-number-division-facts"]
    assert mode.prerequisites == ["order-numbers-to-10"]


def test_other_summary_statistic_skills_are_untouched():
    """`grep -rl "topic: summary_statistic" skills/` sanity check: any OTHER
    skill sourced from the shared `summary_statistic` topic keeps exactly
    its original single source (this change never edits
    `curriculum_expansion.py` or any skill besides the three named ones)."""
    _, skills = load_skills(ROOT / "skills")
    for skill in skills:
        if skill.id in ("calculate-median", "calculate-mean", "calculate-mode"):
            continue
        topics_used = {s.topic for s in skill.sources}
        if "summary_statistic" in topics_used:
            # A later exam source (difficulty "exam", its own topic) is allowed; the practice source stays single.
            practice = [s for s in skill.sources if s.difficulty != "exam"]
            assert len(practice) == 1, f"{skill.id}: unexpectedly gained a second source"
