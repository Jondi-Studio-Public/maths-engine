"""Tests `p13c_exponential_pdf_numint` (ACMSM127): numerical integration of an exponential density."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import (
    VerificationError,
    check_distinct,
    check_equivalent,
    stated_precision_tolerance,
)

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TIER_DP = {"easy": 2, "medium": 2, "hard": 4, "exam": 4}

TOPIC = "p13c_exponential_pdf_numint"
SKILL = "senior-use-numerical-integration-using-technology-f-left-t"

_grade_number = GRADERS["number"]


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = _grade_number(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical {q.answer_text!r} not accepted"
        for d in q.distractors:
            result = _grade_number(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_value(tier):
    dp = TIER_DP[tier]
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        answer_val = Fraction(q.answer_text)
        # Offset by ten whole units in the stated decimal place -- always far
        # outside stated_precision_tolerance(dp), whatever the canonical value.
        offset = Fraction(10, 10**dp)
        wrong = str(float(answer_val + offset))
        result = _grade_number(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Stated-precision distractor gap, asserted directly at the tightest ----
# precision used (4 d.p.), never assumed safe from the 2 d.p. case.


@pytest.mark.parametrize("tier", TIERS)
def test_distractor_gap_clears_stated_precision_tolerance(tier):
    dp = TIER_DP[tier]
    tolerance = stated_precision_tolerance(dp)
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        answer_val = Fraction(q.answer_text)
        for d in q.distractors:
            d_val = Fraction(d)
            gap = abs(d_val - answer_val)
            assert gap > tolerance, (
                f"{tier} seed={seed}: distractor {d!r} within stated_precision_tolerance({dp})="
                f"{tolerance} of answer {q.answer_text!r} (gap={gap})"
            )
            checked += 1
    assert checked > 0


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check_distinct(1, 1, msg="should fail")


# --- Distinct rendered questions -------------------------------------------


# --- Answer-concentration over 200 seeds ------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_not_dominated_by_a_single_value_over_200_seeds(tier):
    answers = [make(TOPIC, seed=seed, difficulty=tier).answer_text for seed in range(200)]
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    worst = max(counts.values())
    assert worst <= 0.5 * len(answers), f"{tier}: answer {max(counts, key=counts.get)!r} appears {worst}/200 times"


# --- Answer kind: exact stated precision, never on a rounding boundary -----


@pytest.mark.parametrize("tier", TIERS)
def test_answer_text_states_exactly_the_tier_precision(tier):
    dp = TIER_DP[tier]
    pattern = re.compile(rf"^-?\d+\.\d{{{dp}}}$")
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert pattern.match(q.answer_text), (
            f"{tier} seed={seed}: answer_text {q.answer_text!r} does not state exactly {dp} d.p."
        )
        assert "{" not in q.text and "}" not in q.text, f"{tier} seed={seed}: stray template braces: {q.text!r}"
        assert f"correct to {dp} decimal places" in q.text, (
            f"{tier} seed={seed}: prompt does not state the {dp} d.p. precision: {q.text!r}"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_formatted_like_the_answer(tier):
    dp = TIER_DP[tier]
    pattern = re.compile(rf"^-?\d+\.\d{{{dp}}}$")
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert pattern.match(d), f"{tier} seed={seed}: distractor {d!r} not formatted at {dp} d.p."


# --- THIS SKILL IS NARROW: no scenario/dataset framing at any tier ---------


_SCENARIO_WORDS = (
    "lightbulb",
    "component",
    "customer",
    "arrival",
    "lifetime",
    "machine",
    "battery",
    "phone call",
    "waiting time",
    "server",
    "queue",
    "breakdown",
    "data",
    "dataset",
    "survey",
    "sample of",
    "measured",
    "recorded",
)


@pytest.mark.parametrize("tier", TIERS)
def test_no_scenario_or_dataset_framing(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        lowered = q.text.lower()
        for word in _SCENARIO_WORDS:
            assert word not in lowered, f"{tier} seed={seed}: scenario/dataset word {word!r} found in {q.text!r}"
        assert "f(t)" in q.text and "e^(-" in q.text, (
            f"{tier} seed={seed}: prompt is not the bare pdf-integral kernel: {q.text!r}"
        )


# --- hard tier: near-cancellation, not merely more decimal places ---------

_SINGLE_BOUNDS_RE = re.compile(
    r"(?:from t = |between t = )(?P<a>-?\d+(?:\.\d+)?)(?:\s+to\s+t\s*=\s*|\s+and\s+t\s*=\s*)(?P<b>-?\d+(?:\.\d+)?)"
)


def _parse_single_bounds(text: str):
    m = _SINGLE_BOUNDS_RE.search(text)
    assert m is not None, f"could not parse bounds out of {text!r}"
    return float(m.group("a")), float(m.group("b"))


def test_hard_tier_forces_near_cancellation():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        a, b = _parse_single_bounds(q.text)
        assert 0 < (b - a) <= 0.16, f"seed={seed}: hard tier gap b-a={b - a} not tight: {q.text!r}"
        checked += 1
    assert checked == len(SEEDS)


# --- exam tier: disjoint union of two intervals, b1 <= a2 -------------------

_EXAM_BOUNDS_RE = re.compile(
    r"(?:from t = (?P<a1>-?\d+(?:\.\d+)?) to t = (?P<b1>-?\d+(?:\.\d+)?) and "
    r"(?:from t = |the definite integral of f\(t\) from t = )"
    r"(?P<a2>-?\d+(?:\.\d+)?) to t = (?P<b2>-?\d+(?:\.\d+)?))"
    r"|(?:over \[(?P<a1b>-?\d+(?:\.\d+)?), (?P<b1b>-?\d+(?:\.\d+)?)\] together with \[(?P<a2b>-?\d+(?:\.\d+)?), (?P<b2b>-?\d+(?:\.\d+)?)\])"
)


def _parse_exam_bounds(text: str):
    m = _EXAM_BOUNDS_RE.search(text)
    assert m is not None, f"could not parse the two intervals out of {text!r}"
    if m.group("a1") is not None:
        return (float(m.group("a1")), float(m.group("b1")), float(m.group("a2")), float(m.group("b2")))
    return (float(m.group("a1b")), float(m.group("b1b")), float(m.group("a2b")), float(m.group("b2b")))


def test_exam_tier_disjoint_union_of_two_intervals():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        a1, b1, a2, b2 = _parse_exam_bounds(q.text)
        assert a1 < b1 <= a2 < b2, f"seed={seed}: intervals not disjoint/ordered: {(a1, b1, a2, b2)} in {q.text!r}"
        checked += 1
    assert checked == len(SEEDS)


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["integrate-polynomial"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
