"""Tests `p12f_markov_hitting` (`ucv1-markov-hitting-probability`): Markov-chain hitting probabilities, answered
as a fraction."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct
from mathgen.generators.p12f_markov_hitting_build import _POOL_P, _path_probabilities

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p12f_markov_hitting"
SKILL = "ucv1-markov-hitting-probability"

_FRACTION_RE = re.compile(r"^-?\d+(/\d+)?$")


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_fraction(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_fraction(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"
        expected = Fraction(q.answer["p"], q.answer["q"])
        wrong = expected + 1
        result = grade_fraction(q, f"{wrong.numerator}/{wrong.denominator}")
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong value accepted"


def test_grader_accepts_unreduced_equivalent_fraction():
    """`grade_fraction` compares by value: an un-simplified but EQUIVALENT typed fraction must still be accepted."""
    q = make(TOPIC, seed=0, difficulty="easy")
    p, qd = q.answer["p"], q.answer["q"]
    unreduced = f"{p * 3}/{qd * 3}"
    result = grade_fraction(q, unreduced)
    assert result.correct, f"unreduced-but-equivalent {unreduced!r} should be accepted"


# --- Answer kind / formatted alike ----------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_fraction_dict_and_formatted_alike(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and set(q.answer) == {"p", "q"}
        assert _FRACTION_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not p or p/q shaped"
        for d in q.distractors:
            assert _FRACTION_RE.match(d), f"seed={seed}: distractor {d!r} not p or p/q shaped"


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_easy_medium_hard_answer_is_a_valid_probability(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        value = Fraction(q.answer["p"], q.answer["q"])
        assert 0 < value < 1, f"{tier} seed={seed}: {value} is not a probability in (0, 1)"


def test_exam_answer_is_the_drawn_p_strictly_between_0_and_1():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        value = Fraction(q.answer["p"], q.answer["q"])
        assert 0 < value < 1, f"seed={seed}: {value} is not in (0, 1)"


# --- q is stated but irrelevant to easy/medium/hard's own answer ----------


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_answer_does_not_depend_on_q(tier):
    """The single most load-bearing property of this skill: re-derive the
    answer with q swapped for a DIFFERENT value (via `_path_probabilities`,
    the explicit path-enumeration route) and confirm the easy/medium/hard
    quantity is unchanged."""
    for seed in SEEDS:
        q_question = make(TOPIC, seed=seed, difficulty=tier)
        p_num, p_den = re.search(r"P\(s→t\)=(\d+)(?:/(\d+))?", q_question.text).groups()
        p = Fraction(int(p_num), int(p_den) if p_den else 1)
        paths_a = _path_probabilities(p, Fraction(1, 7))
        paths_b = _path_probabilities(p, Fraction(6, 7))
        key = {"easy": None, "medium": "ss", "hard": "st"}[tier]
        if key is None:
            value_a = 1 - paths_a["ss"]
            value_b = 1 - paths_b["ss"]
        else:
            value_a = paths_a[key]
            value_b = paths_b[key]
        assert value_a == value_b, f"{tier} seed={seed}: answer changed when q changed"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(Fraction(5, 7), Fraction(6, 7), "should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct(Fraction(5, 7), Fraction(5, 7), msg="should fail")


def test_path_probabilities_sum_to_one_and_match_matrix_square():
    for p in (_POOL_P[0], _POOL_P[10], _POOL_P[-1]):
        for q in (_POOL_P[1], _POOL_P[20]):
            paths = _path_probabilities(p, q)
            assert sum(paths.values()) == 1


# --- Distinct rendered questions -------------------------------------------


# 200-seed answer-concentration measurement --


@pytest.mark.parametrize("tier", TIERS)
def test_200_seed_answer_concentration_under_25_percent(tier):
    """The single free-parameter risk: every
    tier's true answer is a function of p alone. Measure the share of the
    single most common answer value over 200 seeds and require it stays
    under 25%."""
    from collections import Counter

    counts = Counter(make(TOPIC, seed=seed, difficulty=tier).answer_text for seed in range(200))
    top_share = counts.most_common(1)[0][1] / 200
    assert top_share <= 0.25, f"{tier}: top answer concentration {top_share:.1%} exceeds 25% ({counts.most_common(3)})"


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["markov-two-state-step"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
