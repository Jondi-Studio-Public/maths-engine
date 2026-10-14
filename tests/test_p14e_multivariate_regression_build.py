"""Tests `p14e_multivariate_regression` (`ucv1-multivariate-regression`): multivariate regression coefficients,
answered as an ordered list."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p14e_multivariate_regression"
SKILL = "ucv1-multivariate-regression"

_ANSWER_RE = re.compile(r"^\[-?\d+, -?\d+, -?\d+\]$")


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_ordered_list(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_ordered_list(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"
        b0, b1, b2 = q.answer
        wrong = [b0 + 1, b1, b2]
        result = grade_ordered_list(q, f"[{wrong[0]}, {wrong[1]}, {wrong[2]}]")
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong value accepted"


def test_grader_is_order_sensitive_wrong_order_is_wrong():
    """The whole reason ordered_list was chosen over list: a transposition
    of b1/b2 must be rejected."""
    q = make(TOPIC, seed=0, difficulty="easy")
    b0, b1, b2 = q.answer
    swapped = f"[{b0}, {b2}, {b1}]"
    if b1 != b2:
        result = grade_ordered_list(q, swapped)
        assert not result.correct, "swapped b1/b2 was incorrectly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_ordered_list_of_three_ints_formatted_alike(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, list) and len(q.answer) == 3
        assert all(isinstance(v, int) for v in q.answer), f"seed={seed}: answer not all ints: {q.answer!r}"
        assert _ANSWER_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not the expected shape"
        for d in q.distractors:
            assert _ANSWER_RE.match(d), f"seed={seed}: distractor {d!r} not the expected shape"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_states_order_instruction_exactly_once(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert q.text.count("Give your answer in the order [b0, b1, b2].") == 1
        assert q.latex.count("Give your answer in the order [b0, b1, b2].") == 1


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent((1, 2, 3), (1, 2, 4), "should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct((1, 2, 3), (1, 2, 3), msg="should fail")


@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration_at_200_seeds_per_position(tier):
    """No single value at any of the three list positions may exceed 25% of draws."""
    counters = [Counter(), Counter(), Counter()]
    n = 200
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty=tier)
        for i, v in enumerate(q.answer):
            counters[i][v] += 1
    for i, counter in enumerate(counters):
        top_value, top_count = counter.most_common(1)[0]
        share = top_count / n
        assert share <= 0.25, (
            f"{tier} position b{i}: value {top_value} appears in {top_count}/{n} draws ({share:.1%}), over the 25% cap"
        )


def test_easy_tier_is_three_independent_divisions_medium_is_a_genuine_system():
    """Escalation check: easy's XtX is diagonal, medium's is not."""
    from mathgen.generators.p14e_multivariate_regression_build import _X1_EASY, _X2_EASY, _X1_MED, _X2_MED

    s_easy_offdiag = sum(a * b for a, b in zip(_X1_EASY, _X2_EASY))
    s_med_offdiag = sum(a * b for a, b in zip(_X1_MED, _X2_MED))
    assert s_easy_offdiag == 0
    assert s_med_offdiag != 0


def test_hard_and_exam_shapes_carry_no_raw_data():
    for tier in ("hard", "exam"):
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert "XtX" in q.text
            assert not re.search(r"\(-?\d+, -?\d+, -?\d+\)", q.text), (
                f"{tier} seed={seed} appears to carry raw (x1, x2, y) data points"
            )


def test_exam_tier_block_structure_present():
    for seed in range(10):
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "x1bar" in q.text and "x2bar" in q.text and "ybar" in q.text
        assert "[0, 0]" not in q.text  # sanity: not a degenerate all-zero row


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "ordered_list"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["least-squares-line"]
    assert len(skill.sources) == 4
    for source in skill.sources[:3]:  # the exam source is the r3d chain
        assert source.topic == TOPIC
    assert skill.sources[3].topic == "exd_x_r3d_regression_summary_chain"  # the exam source is the multi-part chain
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
