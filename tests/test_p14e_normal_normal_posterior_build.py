"""Tests `p14e_normal_normal_posterior` (`ucv1-bayes-normal-posterior`): a normal-normal Bayesian posterior,
answered as a pair."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import GRADER_TOLERANCE, VerificationError, check_equivalent, check_pairwise_distinct

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p14e_normal_normal_posterior"
SKILL = "ucv1-bayes-normal-posterior"

_PAIR_RE = re.compile(r"^mean = -?\d+(/\d+)?, variance = -?\d+(/\d+)?$")


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_pair(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_pair(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"
        mean = q.answer["mean"]
        variance = q.answer["variance"]
        wrong = f"mean = {mean + 1}, variance = {variance}"
        result = grade_pair(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong value accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_component_gap_clears_pair_tolerance(tier):
    """Grader-kind rule: the pair grader accepts a candidate whose EVERY component is within 0.051 of the
    canonical value. Every distractor offered must therefore have at least one component whose gap from the
    canonical strictly exceeds that tolerance, asserted directly here per candidate, per seed, per component
    -- not merely inferred from the grader accept/reject test above."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        mean = Fraction(q.answer["mean"])
        variance = Fraction(q.answer["variance"])
        for d in q.distractors:
            m = re.match(r"^mean = (-?\d+(?:/\d+)?), variance = (-?\d+(?:/\d+)?)$", d)
            assert m is not None, f"{tier} seed={seed}: distractor {d!r} not pair-shaped"
            d_mean = Fraction(m.group(1))
            d_variance = Fraction(m.group(2))
            gap_mean = abs(d_mean - mean)
            gap_variance = abs(d_variance - variance)
            assert gap_mean > GRADER_TOLERANCE or gap_variance > GRADER_TOLERANCE, (
                f"{tier} seed={seed}: distractor {d!r} within pair-grader tolerance of "
                f"canonical mean={mean}, variance={variance}"
            )


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_pair_dict_and_formatted_alike(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and set(q.answer) == {"mean", "variance"}
        assert _PAIR_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not pair-shaped"
        for d in q.distractors:
            assert _PAIR_RE.match(d), f"seed={seed}: distractor {d!r} not pair-shaped"


def test_medium_tier_x_bar_is_a_genuine_fraction_every_seed():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        xbar_segment = q.text.split("x-bar = ")[1].split(".")[0]
        assert "/" in xbar_segment, f"seed={seed}: x-bar {xbar_segment!r} is not a genuine fraction"


def test_medium_tier_answer_is_sometimes_a_genuine_fraction():
    found = any(
        make(TOPIC, seed=seed, difficulty="medium").answer["mean"].denominator != 1
        or make(TOPIC, seed=seed, difficulty="medium").answer["variance"].denominator != 1
        for seed in SEEDS
    )
    assert found, "medium tier never produced a fractional posterior value over 60 seeds"


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_easy_medium_tiers_state_prior_as_variance(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "tau0^2" in q.text
        assert "lambda0" not in q.text
        assert "precision" not in q.text.lower()


def test_exam_tier_never_states_prior_as_precision():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "lambda0" not in q.text
        assert "precision" not in q.text.lower()


def test_hard_tier_states_prior_as_precision():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert "lambda0" in q.text
        assert "precision" in q.text.lower()


def test_exam_tier_states_two_sequential_samples():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "n1 = " in q.text and "n2 = " in q.text
        assert "x-bar_1 = " in q.text and "x-bar_2 = " in q.text


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(Fraction(5, 7), Fraction(6, 7), "should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct(Fraction(5, 7), Fraction(5, 7), msg="should fail")


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["bernoulli-mle"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
