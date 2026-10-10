"""Tests `p13c_instantaneous_rate_context` (ACMMM092): instantaneous rates of change in context."""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p13c_instantaneous_rate_context"
SKILL = "senior-find-instantaneous-rates-of-change"

_grade_number = GRADERS["number"]


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = _grade_number(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = _grade_number(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_value(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        # Every answer is an exact integer (no decimal point) -- the typed-
        # number path requires an EXACT match, so +7 is comfortably wrong.
        wrong = str(int(q.answer_text) + 7)
        result = _grade_number(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check_distinct(1, 1, msg="should fail")


# --- Distinct rendered questions -------------------------------------------


# --- Answer kind: always a plain integer, exact-match precision ------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_a_plain_integer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert re.fullmatch(r"-?\d+", q.answer_text), (
            f"{tier} seed={seed}: answer_text not a plain integer: {q.answer_text!r}"
        )
        assert isinstance(q.answer, int), f"{tier} seed={seed}: q.answer not a Python int: {q.answer!r}"


# --- Formatted alike: every distractor is also a plain integer -------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_formatted_like_the_answer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert re.fullmatch(r"-?\d+", d), f"{tier} seed={seed}: distractor not plain integer: {d!r}"


# --- Boundary: never the bare "rate-of-change" (linear-relations) average-
# rate idiom, never a bare y = f(x) tangent-line ask, never a sinusoidal
# model, never an "integrate" ask ------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_never_confused_with_neighbouring_skills(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        text_lower = q.text.lower()
        assert "sin(" not in text_lower and "cos(" not in text_lower, (
            f"{tier} seed={seed}: sinusoidal model text found (ACMMM103 boundary): {q.text!r}"
        )
        assert "equation of the tangent" not in text_lower, (
            f"{tier} seed={seed}: tangent-line phrasing found: {q.text!r}"
        )
        assert "integrat" not in text_lower, f"{tier} seed={seed}: integration phrasing found: {q.text!r}"
        # Every question states a polynomial model Q(t) = ... -- a real
        # formula to differentiate, never a bare two-point/table average.
        assert re.search(r"[a-zA-Z]\(t\) = ", q.text), f"{tier} seed={seed}: no stated Q(t) = ... formula: {q.text!r}"


# --- Boundary: exam tier states an interval and pulls toward the average
# rate, but the correct answer is always the INSTANTANEOUS rate -----------


def test_exam_tier_states_interval_and_asks_for_instantaneous_rate():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "interval" in q.text.lower(), f"seed={seed}: exam text missing interval: {q.text!r}"
        assert "INSTANTANEOUS" in q.text, f"seed={seed}: exam text missing INSTANTANEOUS emphasis"
        checked += 1
    assert checked == len(SEEDS)


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["order-of-operations"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
