"""Tests for `mathgen/generators/p19b_percentile_rank_build.py`: `p19b_percentile_rank` (skill
`senior-calculate-and-interpret-quartiles-deciles-and-percentiles`, ACMEM053).

This file holds the generator-specific proofs: the independent in-generator check firing
on a corrupted value, the forced-condition sweeps (level-type variety, the
hard/exam representation), the real-grader battery, and the skill load.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p19b_percentile_rank_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19b_percentile_rank_build as gen

TOPIC = "p19b_percentile_rank"
TIERS = ("easy", "medium", "hard", "exam")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# Shared / by-hand prompt-shape and construction checks
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(tier):
    for q in _all(tier, seeds=range(30)):
        defects = find_text_defects(q)
        assert defects == [], f"{tier}: {defects} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes(tier):
    shapes = {prompt_shape(q) for q in _all(tier)}
    assert len(shapes) >= 3, f"{tier}: only {len(shapes)} shapes: {shapes}"


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in (0, 1, 2, 40, 59):
        first = make(TOPIC, seed=seed, difficulty=tier).to_json()
        second = make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert first == second


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(tier):
    texts = {q.text for q in _all(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct"


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for q in _all(tier):
        assert len(q.distractors) >= 1, f"{tier} seed {q.seed}: no distractors"
        for d in q.distractors:
            assert d != q.answer_text, f"{tier}: distractor == answer: {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_two_distractors_on_85pct_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    at_least_two = sum(1 for q in seeds if len(q.distractors) >= 2)
    assert at_least_two / len(seeds) >= 0.85, f"{tier}: only {at_least_two}/200 have >= 2 distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_top_answer_under_15pct_of_200_draws(tier):
    seeds = [make(TOPIC, seed=s, difficulty=tier) for s in range(200)]
    counts = Counter(q.answer_text for q in seeds)
    top_share = counts.most_common(1)[0][1] / len(seeds)
    assert top_share < 0.15, f"{tier}: top answer {counts.most_common(1)} share {top_share}"


# ============================================================================
# Answer-space / method-specific properties
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_plain_integer_no_decimal_point(tier):
    for q in _all(tier, seeds=range(40)):
        assert isinstance(q.answer, int), f"{tier}: {q.answer!r}"
        assert "." not in q.answer_text, f"{tier}: {q.answer_text!r} carries a decimal point"
        for d in q.distractors:
            assert "." not in d, f"{tier}: distractor {d!r} carries a decimal point"


_LEVEL_PHRASE_RE = re.compile(r"(?:at|is) (the [^.?]+)[.?]\s*$")


def _level_type(text: str) -> str:
    m = _LEVEL_PHRASE_RE.search(text)
    assert m is not None, f"no trailing level phrase found in {text!r}"
    phrase = m.group(1)
    if "percentile" in phrase:
        return "percentile"
    if "decile" in phrase:
        return "decile"
    assert "quartile" in phrase, phrase
    return "quartile"


def test_easy_is_always_percentile_type():
    for q in _all("easy", seeds=range(60)):
        assert _level_type(q.text) == "percentile"


def test_medium_hard_exam_level_types_vary():
    for tier in ("medium", "hard", "exam"):
        seen = {_level_type(q.text) for q in _all(tier, seeds=range(60))}
        assert len(seen) >= 2, f"{tier}: only saw level types {seen}"


def test_hard_tier_presents_a_frequency_table():
    for q in _all("hard", seeds=range(30)):
        assert re.search(r"-?\d+: \d+", q.text), f"hard: no frequency table in {q.text!r}"
        assert "cumulative" not in q.text.lower()


def test_exam_tier_presents_a_cumulative_frequency_table():
    for q in _all("exam", seeds=range(30)):
        assert "cumulative" in q.text.lower(), f"exam: no cumulative table claim in {q.text!r}"


def test_exam_cumulative_as_frequency_distractor_present_whenever_it_differs():
    hit = 0
    for q in _all("exam", seeds=range(100)):
        pairs = [(int(v), int(c)) for v, c in re.findall(r"(-?\d+): (\d+)", q.text)]
        k = None
        m = re.search(r"the (\d+)(?:th|st|nd|rd) percentile", q.text)
        if m:
            k = int(m.group(1)) // 5
        else:
            m2 = re.search(r"the (\d+)(?:th|st|nd|rd) decile", q.text)
            if m2:
                k = 2 * int(m2.group(1))
            else:
                m3 = re.search(r"the (1st|2nd|3rd) quartile", q.text)
                ordinal = {"1st": 1, "2nd": 2, "3rd": 3}[m3.group(1)]
                k = 5 * ordinal
        misread = gen._rank_from_freq(pairs, k)
        if misread != q.answer:
            hit += 1
            assert str(misread) in q.distractors, f"exam seed {q.seed}: {misread} missing from {q.distractors}"
    assert hit > 0


# ============================================================================
# Independent in-generator check
# ============================================================================


def test_independent_check_fires_on_corrupted_value():
    pairs = [(1, 1), (5, 1), (9, 1), (12, 1), (20, 1)]
    gen._verify_rank(pairs, 3, 9)  # no exception
    with pytest.raises(VerificationError):
        gen._verify_rank(pairs, 3, 12)


def test_rank_from_freq_matches_expanded_index():
    pairs = [(10, 2), (20, 3), (30, 1)]
    expanded = [10, 10, 20, 20, 20, 30]
    for k in range(1, len(expanded) + 1):
        assert gen._rank_from_freq(pairs, k) == expanded[k - 1]


# ============================================================================
# Real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for q in _all(tier, seeds=range(15)):
        assert grade_number(q, q.answer_text).correct, f"{tier} seed {q.seed}: canonical rejected"
        for d in q.distractors:
            assert not grade_number(q, d).correct, f"{tier} seed {q.seed}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for q in _all(tier, seeds=range(10)):
        wrong = str(q.answer + 1000)
        assert not grade_number(q, wrong).correct, f"{tier} seed {q.seed}: wrong value wrongly accepted"


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "senior-calculate-and-interpret-quartiles-deciles-and-percentiles")
        assert skill.domain == "statistics-probability"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["simple-event-probability"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert "exam" in stages_covered
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
