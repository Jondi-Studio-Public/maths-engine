"""Tests for `mathgen/generators/p19b_histogram_shape_build.py`: `p19b_histogram_shape` (skill
`senior-compare-the-characteristics-of-the-shape-of-histograms`, ACMEM060).

This file holds the generator-specific proofs: the
classifier firing on a corrupted table, the answer-space contract (all four
options always offered, class balance, key-position balance via the real
mcq grader), and each tier's own forced condition.

The independent ORACLE lives in `tests/test_p19b_histogram_shape_oracle.py`.
"""

from __future__ import annotations

import collections
import re

import pytest

from mathgen.grading.typed import GRADERS, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19b_histogram_shape_build as gen

TOPIC = "p19b_histogram_shape"
TIERS = ("easy", "medium", "hard", "exam")
LABELS = ("symmetric", "skewed left", "skewed right", "bimodal")


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# Text quality / variety
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(tier):
    for q in _all(tier, range(30)):
        defects = find_text_defects(q)
        assert defects == [], f"{tier}: {defects} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes(tier):
    shapes = {prompt_shape(q) for q in _all(tier)}
    assert len(shapes) >= 3, f"{tier}: only {len(shapes)} shapes: {shapes}"


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_prompts(tier):
    texts = {q.text for q in _all(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct prompts"


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in (0, 1, 2, 40, 59):
        first = make(TOPIC, seed=seed, difficulty=tier).to_json()
        second = make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert first == second


# ============================================================================
# Answer-space contract
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_all_four_options_always_offered(tier):
    for q in _all(tier, range(60)):
        options = set([q.answer_text, *q.distractors])
        assert options == set(LABELS), f"{tier}: {options}"
        assert len(q.distractors) == 3


@pytest.mark.parametrize("tier", TIERS)
def test_key_share_under_35_percent_at_200_seeds(tier):
    counts = collections.Counter(make(TOPIC, seed=s, difficulty=tier).answer_text for s in range(200))
    for label, n in counts.items():
        assert n <= 70, f"{tier}: {label} is the key on {n}/200 draws (cap 70 = 35%)"
    assert set(counts) == set(LABELS), f"{tier}: not every class drawn: {counts}"


def test_key_position_balanced_at_400_seeds():
    positions = collections.Counter()
    for s in range(400):
        q = make(TOPIC, seed=s, difficulty="medium")
        options = mcq_options(q)
        positions[options.index(q.answer_text)] += 1
    for pos, n in positions.items():
        assert 60 <= n <= 160, f"position {pos}: {n}/400 (expected roughly uniform over 4)"


# ============================================================================
# The classifier: fires on a corrupted table, disjoint by construction
# ============================================================================


def test_classify_rejects_flat_table():
    assert gen._classify([5, 5, 5, 5, 5, 5, 5]) is None


def test_classify_symmetric():
    assert gen._classify([2, 5, 8, 12, 11, 8, 5, 2]) == "symmetric"


def test_classify_skewed_right():
    assert gen._classify([20, 15, 10, 6, 3, 1]) == "skewed right"


def test_classify_skewed_left():
    assert gen._classify([1, 3, 6, 10, 15, 20]) == "skewed left"


def test_classify_bimodal():
    assert gen._classify([2, 6, 3, 6, 2]) == "bimodal"


def test_classify_corrupted_symmetric_is_rejected():
    # Same table as the symmetric case above, but with one class perturbed
    # so the mirror rule no longer holds within 1 -- must not still classify
    # as symmetric.
    corrupted = [2, 5, 8, 12, 11, 8, 5, 9]
    assert gen._classify(corrupted) != "symmetric"


@pytest.mark.parametrize("tier", ("medium", "hard", "exam"))
def test_construction_check_fires_on_corrupted_builder(tier, monkeypatch):
    """Swap `_classify` for a stub that always disagrees -- the construction
    `check()` calls in `_build_symmetric`/`_build_skewed`/`_build_bimodal`
    must raise. (`easy` builds no table -- it never calls `_classify` -- so
    it is exempt, same as the other tiers' own forced-condition tests.)"""
    monkeypatch.setattr(gen, "_classify", lambda freqs: "not-a-real-label")
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty=tier)


# ============================================================================
# Tier-specific forced conditions
# ============================================================================


def test_easy_never_uses_answer_words_in_description():
    for q in _all("easy", range(60)):
        body = q.text.split("described as follows:", 1)
        body = body[1] if len(body) > 1 else q.text
        for word in ("symmetric", "skewed", "bimodal"):
            assert word not in body.lower() or "shape" in body.lower()


def test_medium_histogram_is_drawn_in_class_order_and_prints_no_table():
    for q in _all("medium", range(30)):
        assert q.figure is not None and q.figure["kind"] == "histogram" and len(q.figure["panels"]) == 1
        edges = q.figure["panels"][0]["edges"]
        assert edges == sorted(edges) and len(set(edges)) == len(edges), q.figure
        assert not re.search(r"\d+-\d+:", q.text), q.text


def test_hard_reflects_the_drawn_shape():
    """Every value is subtracted from a stated number: skew swaps direction, symmetric and bimodal stay."""
    seen = collections.Counter()
    for q in _all("hard", range(60)):
        drawn = gen._classify(q.figure["panels"][0]["frequencies"])
        assert drawn is not None
        assert q.answer_text == gen._REFLECTED[drawn], q.text
        assert re.search(r"subtracted|replaced by|taking each value away", q.text), q.text
        seen[drawn] += 1
    assert set(seen) == set(LABELS), seen


def test_exam_two_histograms_differ_on_stated_criterion():
    for q in _all("exam", range(40)):
        panels = q.figure["panels"]
        assert [p["label"] for p in panels] == ["Group A", "Group B"]
        assert ("WIDER range" in q.text) or ("LARGER total" in q.text)
        a, b = panels
        if "WIDER range" in q.text:
            assert len(a["frequencies"]) != len(b["frequencies"])
        else:
            assert sum(a["frequencies"]) != sum(b["frequencies"])


# ============================================================================
# Real grader: accepts the key, rejects every distractor and a corrupted value
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_key_rejects_distractors(tier):
    grade_mcq = GRADERS["mcq"]
    for q in _all(tier, range(30)):
        options = mcq_options(q)
        key_idx = options.index(q.answer_text)
        result = grade_mcq(q, str(key_idx))
        assert result.correct, f"{tier}: key rejected for {q.text!r}"
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            result = grade_mcq(q, str(i))
            assert not result.correct, f"{tier}: distractor {opt!r} accepted for {q.text!r}"


# ============================================================================
# Skill load
# ============================================================================


def test_skill_loads_with_expected_shape():
    _domains, skills = load_skills()
    by_id = {s.id: s for s in skills}
    skill = by_id["senior-compare-the-characteristics-of-the-shape-of-histograms"]
    assert skill.prerequisites == ["simple-event-probability"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    stages = {s.difficulty: s.stages for s in skill.sources}
    assert stages["easy"] == [6]
    assert stages["medium"] == [6]
    assert stages["hard"] == [6, "maintenance"]
    assert stages["exam"] == ["exam"]
