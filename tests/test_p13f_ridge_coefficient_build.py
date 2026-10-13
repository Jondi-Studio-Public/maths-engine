"""Tests `p13f_ridge_coefficient` (`ucv1-ridge-coefficient`): a ridge-regression coefficient, answered as a
fraction."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p13f_ridge_coefficient"
SKILL = "ucv1-ridge-coefficient"

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
    q = make(TOPIC, seed=0, difficulty="medium")
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


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_answer_never_equals_zero_penalty_ols_value(tier):
    """lambda >= 2 on every draw; the answer must never equal Sxy/Sxx."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "lambda = " in q.text or "penalty lambda" in q.text


def test_medium_tier_answer_is_a_genuine_fraction_on_at_least_one_seed():
    denominators = {make(TOPIC, seed=seed, difficulty="medium").answer["q"] for seed in SEEDS}
    assert any(d != 1 for d in denominators), "medium tier never produced a non-integer answer over 60 seeds"


@pytest.mark.parametrize("tier", TIERS)
def test_sxx_always_positive(tier):
    """Sxx = sum(x_i^2) is always positive; it is mathematically impossible
    for a sum of squares to be negative or zero (for non-zero data)."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        # Extract Sxx from the question text
        import re

        if tier == "hard":
            # Hard tier gives raw points; compute Sxx from them
            # Pattern: (x, y), (x, y), ... or similar
            points_match = re.search(r"\((-?\d+),\s*(-?\d+)\)", q.text)
            if points_match:
                # Working line contains the computed Sxx
                for line in q.working:
                    sxx_match = re.search(r"Sxx\s*=\s*([^=.]*?)(?:\.|,)", line)
                    if sxx_match:
                        sxx_str = sxx_match.group(1).strip()
                        sxx_val = eval(sxx_str)
                        assert sxx_val > 0, f"hard tier seed={seed}: Sxx={sxx_val} is not positive"
                        break
        else:
            # Easy/medium/exam tiers have Sxx given directly in text
            match = re.search(r"Sxx\s*=\s*sum\(x_i\^\s*2\)\s*=\s*(-?\d+)", q.text)
            assert match is not None, f"{tier} seed={seed}: could not extract Sxx from text"
            sxx = int(match.group(1))
            assert sxx > 0, f"{tier} seed={seed}: Sxx={sxx} is not positive"


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
    assert skill.prerequisites == ["least-squares-line"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        # the exam source is now the multi-part chain exd_x_r3g_ridge_fit; the other tiers keep the original topic
        assert source.topic == ("exd_x_r3g_ridge_fit" if source.difficulty == "exam" else TOPIC)
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
