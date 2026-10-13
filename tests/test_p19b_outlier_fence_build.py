"""Tests for `mathgen/generators/p19b_outlier_fence_build.py`: `p19b_outlier_fence` (skill
`senior-recognise-and-identify-outliers`, ACMEM047).

This file holds the generator-specific proofs: the independent in-generator check firing
on a corrupted value, the forced-condition sweeps (outlier side balance, the
medium/hard/exam decoy properties), the real-grader battery, and the skill
load.

The independent ORACLE (parses `question.text` alone, no import from this
module) lives in `tests/test_p19b_outlier_fence_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19b_outlier_fence_build as gen

TOPIC = "p19b_outlier_fence"
TIERS = ("easy", "medium", "hard", "exam")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


def _sorted_values(q):
    line = q.working[0]
    return sorted(int(t) for t in re.findall(r"-?\d+", line.split(":", 1)[1]))


def _fences(q):
    line = q.working[1]
    lo = int(re.search(r"Lower fence = Q1 - 1\.5 times IQR = (-?\d+)", line).group(1))
    hi = int(re.search(r"Upper fence = Q3 \+ 1\.5 times IQR = (-?\d+)", line).group(1))
    return lo, hi


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
def test_answer_is_single_element_integer_list(tier):
    for q in _all(tier, seeds=range(40)):
        assert isinstance(q.answer, list) and len(q.answer) == 1, f"{tier}: {q.answer!r}"
        assert isinstance(q.answer[0], int)
        assert q.answer_text == str(q.answer[0])


@pytest.mark.parametrize("tier", TIERS)
def test_exactly_one_value_beyond_fences(tier):
    for q in _all(tier, seeds=range(60)):
        vals = _sorted_values(q)
        lo, hi = _fences(q)
        beyond = [v for v in vals if v < lo or v > hi]
        assert beyond == [q.answer[0]], f"{tier} seed {q.seed}: beyond={beyond}, answer={q.answer}"


def test_easy_is_always_low_side():
    for q in _all("easy", seeds=range(60)):
        lo, _hi = _fences(q)
        assert q.answer[0] < lo


def test_medium_hard_exam_outlier_side_40_to_60pct_low():
    for tier in ("medium", "hard", "exam"):
        low_count = 0
        seeds = _all(tier, seeds=range(200))
        for q in seeds:
            lo, _hi = _fences(q)
            if q.answer[0] < lo:
                low_count += 1
        share = low_count / len(seeds)
        assert 0.40 <= share <= 0.60, f"{tier}: low-side share {share}"


def test_medium_decoy_beyond_1x_within_1_5x_iqr():
    """medium: the same-side slot is deliberately placed beyond 1*IQR but
    within 1.5*IQR of its quartile -- confirm directly against the fence and
    the 1*IQR line, not merely trust the generator's own claim."""
    hit = 0
    for q in _all("medium", seeds=range(100)):
        vals = _sorted_values(q)
        lo, hi = _fences(q)
        q1_line = re.search(r"Q1 = (-?\d+)", q.text)
        q3_line = re.search(r"Q3 = (-?\d+)", q.text)
        q1, q3 = int(q1_line.group(1)), int(q3_line.group(1))
        iqr = q3 - q1
        outlier = q.answer[0]
        if outlier < lo:
            one_x_line = q1 - iqr
            candidates = [v for v in vals if lo < v < one_x_line]
        else:
            one_x_line = q3 + iqr
            candidates = [v for v in vals if one_x_line < v < hi]
        if candidates:
            hit += 1
    assert hit >= 90, f"medium: only {hit}/100 draws carried the 1xIQR decoy"


def test_hard_opposite_side_decoy_within_1_unit_of_fence():
    hit = 0
    for q in _all("hard", seeds=range(100)):
        vals = _sorted_values(q)
        lo, hi = _fences(q)
        outlier = q.answer[0]
        if outlier < lo:
            # opposite side is the upper fence
            near = [v for v in vals if 0 < hi - v <= 1]
        else:
            near = [v for v in vals if 0 < v - lo <= 1]
        if near:
            hit += 1
    assert hit == 100, f"hard: only {hit}/100 draws carried the tight opposite-side decoy"


def test_exam_opposite_side_value_exactly_on_fence():
    hit = 0
    for q in _all("exam", seeds=range(100)):
        vals = _sorted_values(q)
        lo, hi = _fences(q)
        outlier = q.answer[0]
        on_fence = (hi in vals) if outlier < lo else (lo in vals)
        if on_fence:
            hit += 1
    assert hit == 100, f"exam: only {hit}/100 draws carried the exact-fence value"


# ============================================================================
# Independent in-generator check
# ============================================================================


def test_independent_check_fires_on_corrupted_value():
    q1, q3 = 30, 40
    iqr = q3 - q1
    k = iqr // 2
    lower_fence, upper_fence = q1 - 3 * k, q3 + 3 * k
    outlier = lower_fence - 20
    vals = sorted([outlier, 28, q1, 32, 34, 36, 38, q3, 42, 44])
    gen._verify(vals, q1, q3, outlier, lower_fence, upper_fence)  # no exception
    with pytest.raises(VerificationError):
        gen._verify(vals, q1 + 1, q3, outlier, lower_fence, upper_fence)
    with pytest.raises(VerificationError):
        gen._verify(vals, q1, q3, outlier, lower_fence + 1, upper_fence)


def test_offset_modes_land_in_expected_bands():
    import random

    rng = random.Random(0)
    k = 6
    for _ in range(30):
        assert 2 <= gen._offset("safe", k, rng) <= k
        assert 2 * k < gen._offset("loose", k, rng) < 3 * k
    assert gen._offset("tight", k, rng) == 3 * k - 1
    assert gen._offset("exact", k, rng) == 3 * k


# ============================================================================
# Real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for q in _all(tier, seeds=range(15)):
        assert grade_list(q, q.answer_text).correct, f"{tier} seed {q.seed}: canonical rejected"
        assert grade_list(q, "{" + q.answer_text + "}").correct
        assert grade_list(q, "[" + q.answer_text + "]").correct
        for d in q.distractors:
            assert not grade_list(q, d).correct, f"{tier} seed {q.seed}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for q in _all(tier, seeds=range(10)):
        wrong = str(q.answer[0] + 1)
        assert not grade_list(q, wrong).correct, f"{tier} seed {q.seed}: wrong value wrongly accepted"


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "senior-recognise-and-identify-outliers")
        assert skill.domain == "statistics-probability"
        assert skill.answer == "list"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["calculate-median", "calculate-range"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert "exam" in stages_covered
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
