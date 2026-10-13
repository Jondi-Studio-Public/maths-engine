"""Tests `p13c_riemann_sum_area` (ACMMM124): area estimated by a sum of rectangles."""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p13c_riemann_sum_area"
SKILL = "senior-examine-the-area-problem-and-use-sums-of"

_grade_fraction = GRADERS["fraction"]

_FRACTION_TEXT_RE = re.compile(r"^-?\d+(/\d+)?$")


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = _grade_fraction(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = _grade_fraction(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_value(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        # `grade_fraction` compares by exact value, so any value shifted by 1
        # from the canonical is unambiguously wrong.
        p, qd = q.answer["p"], q.answer["q"]
        wrong = f"{p + qd}/{qd}" if qd != 1 else str(p + 1)
        result = _grade_fraction(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_grade_fraction_accepts_plain_typed_integer_for_integer_canonical(tier):
    """Confirms (not assumes) `grade_fraction` accepts a plain typed integer
    string like "12" against a canonical integer answer -- required before
    relying on `answer: fraction` to cover both this tier's integer sums and
    the hard tier's quarter-fraction sums in one grader kind."""
    found_integer_case = False
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        if q.answer["q"] == 1:
            found_integer_case = True
            result = _grade_fraction(q, str(q.answer["p"]))
            assert result.correct, f"{tier} seed={seed}: plain integer string not accepted"
    assert found_integer_case, f"{tier}: no integer-answer seed found in 60 seeds to exercise this"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check_distinct(1, 1, msg="should fail")


# --- Distinct rendered questions -------------------------------------------


# --- Answer kind: a fraction in lowest terms --------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_a_reduced_fraction(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and {"p", "q"} <= set(q.answer), (
            f"{tier} seed={seed}: answer not a {{p, q}} dict: {q.answer!r}"
        )
        p, qd = q.answer["p"], q.answer["q"]
        assert qd > 0, f"{tier} seed={seed}: denominator not positive: {q.answer!r}"
        import math

        assert math.gcd(abs(p), qd) == 1, f"{tier} seed={seed}: fraction not in lowest terms: {q.answer!r}"
        assert _FRACTION_TEXT_RE.match(q.answer_text), (
            f"{tier} seed={seed}: answer_text not a plain fraction/integer: {q.answer_text!r}"
        )


# --- Formatted alike: every distractor is also a plain fraction/integer ----


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_formatted_like_the_answer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert _FRACTION_TEXT_RE.match(d), f"{tier} seed={seed}: distractor not plain fraction/integer: {d!r}"


# --- Hard tier genuinely produces quarter-fraction (non-integer) answers ---


def test_hard_tier_produces_some_fraction_answers():
    fraction_seeds = [seed for seed in SEEDS if make(TOPIC, seed=seed, difficulty="hard").answer["q"] != 1]
    assert len(fraction_seeds) >= 1, "hard: no non-integer (quarter-fraction) answer found in 60 seeds"


# --- Boundary: never the trapezoidal rule ----------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_never_the_trapezoidal_rule(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        text_lower = q.text.lower()
        assert "trapezoid" not in text_lower, f"{tier} seed={seed}: trapezoidal wording found: {q.text!r}"
        assert "riemann sum" in text_lower, f"{tier} seed={seed}: missing 'Riemann sum' naming: {q.text!r}"


def test_easy_states_left_medium_states_right_hard_states_midpoint():
    for seed in SEEDS:
        assert "LEFT Riemann sum" in make(TOPIC, seed=seed, difficulty="easy").text
        assert "RIGHT Riemann sum" in make(TOPIC, seed=seed, difficulty="medium").text
        assert "MIDPOINT Riemann sum" in make(TOPIC, seed=seed, difficulty="hard").text


def test_exam_tier_states_explicit_breakpoints_and_left_or_right():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "breakpoints x = " in q.text, f"seed={seed}: no explicit breakpoints list: {q.text!r}"
        assert ("LEFT Riemann sum" in q.text) or ("RIGHT Riemann sum" in q.text), (
            f"seed={seed}: no stated LEFT/RIGHT rule: {q.text!r}"
        )
        assert "MIDPOINT" not in q.text, f"seed={seed}: exam tier must never use MIDPOINT: {q.text!r}"
        checked += 1
    assert checked == len(SEEDS)


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["integrate-polynomial"]
    assert len(skill.sources) == 4
    for source in skill.sources[:3]:  # the exam source is the r3d chain
        assert source.topic == TOPIC
    assert skill.sources[3].topic == "exd_x_r3d_riemann_error_chain"  # the exam source is the multi-part chain
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
