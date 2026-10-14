"""Tests `p13f_stochastic_euler` (`ucv1-stochastic-euler-step`): one stochastic Euler step, answered as a
fraction."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct
from mathgen.generators.p13f_stochastic_euler_build import (
    _build_additive,
    _build_multiplicative,
    _draw_dw,
    _draw_lam_forcing_y0_h,
)

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p13f_stochastic_euler"
SKILL = "ucv1-stochastic-euler-step"

_FRACTION_RE = re.compile(r"^-?\d+(/\d+)?$")


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
    q = make(TOPIC, seed=0, difficulty="easy")
    p, qd = q.answer["p"], q.answer["q"]
    unreduced = f"{p * 3}/{qd * 3}"
    result = grade_fraction(q, unreduced)
    assert result.correct, f"unreduced-but-equivalent {unreduced!r} should be accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_fraction_dict_and_formatted_alike(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and set(q.answer) == {"p", "q"}
        assert _FRACTION_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not p or p/q shaped"
        for d in q.distractors:
            assert _FRACTION_RE.match(d), f"seed={seed}: distractor {d!r} not p or p/q shaped"


def test_additive_draw_never_lets_diffusion_vanish():
    import random

    rng = random.Random(12345)
    for _ in range(60):
        lam, forcing, y0, h, sigma, dw, answer, candidates = _build_additive(
            rng,
            coefficient_hi=4,
            step_denominator_hi=4,
            value_hi=5,
            dw_hi=3,
            dw_denom_hi=3,
            sigma_hi=3,
        )
        assert sigma * dw != 0
        assert answer != candidates["omits-diffusion"]


def test_multiplicative_draw_never_lets_diffusion_vanish():
    import random

    rng = random.Random(54321)
    for _ in range(60):
        lam, forcing, y0, h, k, dw, answer, candidates = _build_multiplicative(
            rng,
            coefficient_hi=6,
            step_denominator_hi=6,
            value_hi=8,
            dw_hi=4,
            dw_denom_hi=4,
            k_hi=4,
        )
        assert Fraction(k) * y0 * dw != 0
        assert answer != candidates["omits-diffusion"]


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(Fraction(5, 7), Fraction(6, 7), "should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct(Fraction(5, 7), Fraction(5, 7), msg="should fail")


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["explicit-euler-linear-ode"]
    assert len(skill.sources) == 4
    for source in skill.sources[:3]:
        assert source.topic == TOPIC
    assert skill.sources[3].topic == "exd_x_r3a_stochastic_two_step"
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
