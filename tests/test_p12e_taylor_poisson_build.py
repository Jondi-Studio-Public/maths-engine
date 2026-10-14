"""The Taylor linear-approximation + Poisson count-probability pair.

Skill 1: ucv1-taylor-linear-approximation (single-variable-calculus, topic p12e_taylor_linear_approx, answer:
number). Skill 2: ucv1-poisson-count-probability (stochastic-processes, topic p12e_poisson_count, answer:
fraction). See the generator module docstring for the boundary against ucv1-ftc-polynomial and
markov-two-state-step/markov-two-state-stationary.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct
from mathgen.generators.p12e_taylor_poisson_build import (
    _E_TABLE,
    _poisson_probability,
    _quad_derivative_at,
    _quad_value,
    _rational_derivative_at,
    _rational_value,
)

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC_1 = "p12e_taylor_linear_approx"
SKILL_1 = "ucv1-taylor-linear-approximation"
TOPIC_2 = "p12e_poisson_count"
SKILL_2 = "ucv1-poisson-count-probability"

_INT_RE = re.compile(r"^-?\d+$")
_TWO_DP_RE = re.compile(r"^-?\d+\.\d{2}$")
_FRACTION_RE = re.compile(r"^-?\d+(/\d+)?$")


# --- Determinism -------------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 1


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_taylor_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"{TOPIC_1}/{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"{TOPIC_1}/{tier} seed={seed}: distractor {d!r} accepted"
        if tier in ("medium", "hard"):
            wrong = f"{float(q.answer) + 1:.2f}"
        else:
            wrong = str(int(q.answer) + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"{TOPIC_1}/{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_poisson_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        result = grade_fraction(q, q.answer_text)
        assert result.correct, f"{TOPIC_2}/{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_fraction(q, d)
            assert not result.correct, f"{TOPIC_2}/{tier} seed={seed}: distractor {d!r} accepted"
        expected = Fraction(q.answer["p"], q.answer["q"])
        wrong = expected + 1
        result = grade_fraction(q, f"{wrong.numerator}/{wrong.denominator}")
        assert not result.correct, f"{TOPIC_2}/{tier} seed={seed}: deliberately wrong value accepted"


def test_poisson_grader_accepts_unreduced_equivalent_fraction():
    """`grade_fraction` compares by value: an un-simplified but EQUIVALENT typed fraction must still be accepted."""
    q = make(TOPIC_2, seed=0, difficulty="easy")
    p, qd = q.answer["p"], q.answer["q"]
    unreduced = f"{p * 3}/{qd * 3}"
    result = grade_fraction(q, unreduced)
    assert result.correct, f"unreduced-but-equivalent {unreduced!r} should be accepted"


# --- Answer kind / formatted alike ----------------------------------------


@pytest.mark.parametrize("tier", ("easy", "exam"))
def test_taylor_integer_tiers_are_bare_integers(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int)
        assert _INT_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not a bare integer"
        for d in q.distractors:
            assert _INT_RE.match(d), f"seed={seed}: distractor {d!r} not shaped like the integer answer"


@pytest.mark.parametrize("tier", ("medium", "hard"))
def test_taylor_decimal_tiers_are_exactly_two_decimal_places(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        assert isinstance(q.answer, float)
        assert _TWO_DP_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not exactly 2 d.p."
        for d in q.distractors:
            assert _TWO_DP_RE.match(d), f"seed={seed}: distractor {d!r} not exactly 2 d.p."


def test_taylor_medium_tier_is_never_actually_rounded():
    """medium tier is restricted to a degree-2 f so every value (answer and
    every distractor) lands exactly on the quarter grid -- `stated_precision_string`
    never rounds anything away here, unlike the hard tier's genuine rounding."""
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty="medium")
        value = Fraction(q.answer_text)
        assert (value * 4).denominator == 1, f"seed={seed}: {q.answer_text!r} not exact on the quarter grid"


def test_taylor_hard_tier_answer_never_on_a_rounding_boundary():
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty="hard")
        from mathgen.core import rounds_unambiguously

        assert rounds_unambiguously(q.answer, 2), f"seed={seed}: {q.answer_text!r} sits on a 2-d.p. rounding boundary"


@pytest.mark.parametrize("tier", TIERS)
def test_poisson_answer_is_fraction_dict_and_formatted_alike(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and set(q.answer) == {"p", "q"}
        assert _FRACTION_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not p or p/q shaped"
        for d in q.distractors:
            assert _FRACTION_RE.match(d), f"seed={seed}: distractor {d!r} not p or p/q shaped"


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_poisson_easy_medium_answer_is_a_valid_probability(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        value = Fraction(q.answer["p"], q.answer["q"])
        assert 0 < value < 1, f"seed={seed}: {value} is not a probability in (0, 1)"


# --- Any instruction phrase in text also appears in latex (auto-copied) --


@pytest.mark.parametrize("tier", ("medium", "hard"))
def test_taylor_rounding_instruction_appears_in_both_text_and_latex(tier):
    for seed in range(10):
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        assert "decimal place" in q.text
        assert "decimal place" in q.latex


# --- Distinct rendered questions -------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct questions over 60 seeds"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct(5, 5, msg="should fail")


# --- Construction re-derivation (the shared value/derivative helpers, exercised
# directly rather than only trusted via the generator's own internal call) ---


def test_quad_value_and_derivative_match_hand_computation():
    # f(x) = 2x^2 - 3x + 1; f(2) = 8 - 6 + 1 = 3; f'(x) = 4x - 3, f'(2) = 5
    assert _quad_value(2, -3, 1, 2) == 3
    assert _quad_derivative_at(2, -3, 2) == 5
    # half-integer point: f(2.5) = 2*6.25 - 3*2.5 + 1 = 12.5 - 7.5 + 1 = 6
    assert _quad_value(2, -3, 1, Fraction(5, 2)) == 6


def test_rational_value_and_derivative_match_hand_computation():
    # f(x) = 6/x; f(3) = 2; f'(x) = -6/x^2, f'(3) = -6/9 = -2/3
    assert _rational_value(6, 3) == 2
    assert _rational_derivative_at(6, 3) == Fraction(-2, 3)


def test_poisson_probability_matches_hand_computation():
    # e^-3 given as 1/20, lambda*t = 3, k = 2: P = 1/20 * 9/2 = 9/40
    assert _poisson_probability(_E_TABLE[3], 3, 2) == Fraction(9, 40)


# --- k=0 deviation, documented as a real assertion ------------------------


def test_poisson_easy_medium_never_draws_k_equal_zero():
    """Found wrong for the originally suggested `k in 0..4`: at
    k = 0, `omits-factorial` collides with the correct answer (0! = 1) and
    the correct answer is a bare copy of the given exponential value. Easy"""
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = make(TOPIC_2, seed=seed, difficulty=tier)
            assert " k = 0)" not in q.text and "= 0)" not in q.text.split("find P(N(t)")[-1]


# --- hard tier: drops-exponential-factor does not exist (no exponential) ---


def test_poisson_hard_tier_has_no_given_exponential_value():
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty="hard")
        assert "e^(-lambda t)" not in q.text and "e^(-lambda" not in q.text


# --- Skill loading -----------------------------------------------------


def test_skill_1_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_1), None)
    assert skill is not None, f"{SKILL_1} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["differentiate-polynomial"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == ("exd_x_r3i_taylor_tangent" if source.difficulty == "exam" else TOPIC_1)
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"


def test_skill_2_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_2), None)
    assert skill is not None, f"{SKILL_2} not found"
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-event-probability"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        # The exam tier is now a multi-part chain (tests/test_exd_x_r2h_poisson_chain.py).
        assert source.topic == ("exd_x_r2h_poisson_chain" if source.difficulty == "exam" else TOPIC_2)
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
