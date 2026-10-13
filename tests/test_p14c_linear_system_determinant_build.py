"""Tests `p14c_linear_system_determinant` (`ucv1-linear-system-determinant`): Cramer's-rule determinants as a
labelled pair."""

from __future__ import annotations

import re

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p14c_linear_system_determinant"
SKILL = "ucv1-linear-system-determinant"

_LABEL_BY_TIER = {
    "easy": ("D_x", "D"),
    "medium": ("D_y", "D"),
    "hard": ("D_x", "D_y"),
    "exam": ("D_x", "D_y"),
}


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_pair(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted (text={q.answer_text!r})"
        for d in q.distractors:
            result = grade_pair(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"
        # A deliberately wrong value: bump the first value by 1.
        label1, label2 = _LABEL_BY_TIER[tier]
        v1, v2 = q.answer[label1], q.answer[label2]
        wrong_text = f"{label1} = {v1 + 1}, {label2} = {v2}"
        result = grade_pair(q, wrong_text)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong value accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_reordered_labels(tier):
    """The labelled-pair grader is order-independent by label name -- proof
    that the labelled path, not a lucky positional match, is what's grading
    this skill (mathgen/grading/typed.py::_labelled_pair_verdict)."""
    for seed in SEEDS[:10]:
        q = make(TOPIC, seed=seed, difficulty=tier)
        label1, label2 = _LABEL_BY_TIER[tier]
        v1, v2 = q.answer[label1], q.answer[label2]
        reordered = f"{label2} = {v2}, {label1} = {v1}"
        result = grade_pair(q, reordered)
        assert result.correct, f"{tier} seed={seed}: reordered labels {reordered!r} should be accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_labelled_dict_matching_labels(tier):
    label1, label2 = _LABEL_BY_TIER[tier]
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and set(q.answer) == {label1, label2}
        assert q.answer_text == f"{label1} = {q.answer[label1]}, {label2} = {q.answer[label2]}"
        for value in q.answer.values():
            assert isinstance(value, int)


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_determinant_magnitude_at_least_two(tier):
    """|det(A)| >= 2 on every draw -- guards against the skill becoming
    swappable into matrix-solve-system's own graded (x, y) answer."""
    d_re = re.compile(r"D = (-?\d+)")
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        if tier == "hard":
            # D itself is not in the graded pair at hard tier; skip.
            continue
        m = d_re.search(q.answer_text)
        assert m is not None
        assert abs(int(m.group(1))) >= 2, f"{tier} seed={seed}: |D| < 2"


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct(5, 5, msg="should fail")


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["matrix-multiplication"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        if source.difficulty == "exam":
            assert source.topic == "exd_x_r3c_cramer_line"  # the multi-part exam source
        else:
            assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
