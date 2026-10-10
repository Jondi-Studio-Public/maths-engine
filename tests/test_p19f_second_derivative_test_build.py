"""Tests `p19f_second_derivative_test`: grading, distractors and the single-critical-point construction.

The second derivative test.

Skill: senior-understand-and-use-the-second-derivative-test-for
(topic ``p19f_second_derivative_test``, answer: ordered_list).
"""

import re
from collections import Counter

import pytest
from sympy import diff, solve, symbols

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p19f_second_derivative_test"
SKILL = "senior-understand-and-use-the-second-derivative-test-for"

_grade_ordered_list = GRADERS["ordered_list"]

x = symbols("x")


def _parse_fx(text: str):
    from sympy import sympify

    m = re.search(r"f\(x\) = ([-+*^·0-9x() ]+)", text)
    assert m is not None, f"could not parse f(x) from: {text!r}"
    expr_text = m.group(1).strip().replace("^", "**").replace("·", "*")
    return sympify(expr_text, locals={"x": x})


# --- Determinism -----------------------------------------------------------


# --- Distractor hygiene ------------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors ---------------------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = _grade_ordered_list(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted: {q.answer_text!r}"
        for d in q.distractors:
            result = _grade_ordered_list(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        x0, label = q.answer
        wrong = f"{x0 + 1}, {label}"
        result = _grade_ordered_list(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Grader behaviour ------------------------------------------------------


def test_grader_accepts_bare_number_but_not_labelled_x_value():
    class Q:
        answer_text = "3, local maximum"
        answer = [3, "local maximum"]

    q = Q()
    assert _grade_ordered_list(q, "3, local maximum").correct
    assert _grade_ordered_list(q, "3, Local Maximum").correct
    assert _grade_ordered_list(q, "3,local maximum").correct
    # Positional grader: swapped order is wrong.
    assert not _grade_ordered_list(q, "local maximum, 3").correct
    # "x = 3, ..." does NOT grade correctly -- the first item fails to parse
    # as a number (`_parse_numeric_item("x = 3")` is None) -- confirmed by
    # direct code read/run, contradicting an earlier expectation. The
    # prompt therefore asks for a bare number, never "x = <value>".
    assert not _grade_ordered_list(q, "x = 3, local maximum").correct


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_condition():
    with pytest.raises(VerificationError):
        check(False, "should fail")


# --- Distinct rendered questions --------------------------------------------


# --- Answer kind: [int x-value, classification word] -----------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_x_value_then_classification(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, list) and len(q.answer) == 2
        assert isinstance(q.answer[0], int)
        assert q.answer[1] in ("local maximum", "local minimum")
        for d in q.distractors:
            assert re.fullmatch(r"-?\d+, .+", d), f"{tier} seed={seed}: malformed distractor {d!r}"


# --- The exactly-one-real-root / nonzero-second-derivative invariant,
# re-derived independently from the RENDERED TEXT with sympy, >= 200
# seeds/tier, quoting the exact pass count (not "mostly"). -----------------


@pytest.mark.parametrize("tier", TIERS)
def test_exactly_one_real_root_invariant_200_seeds(tier):
    n_ok = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier)
        f = _parse_fx(q.text)
        roots = solve(diff(f, x), x)
        real_roots = [r for r in roots if r.is_real]
        assert len(real_roots) == 1, f"{tier} seed={seed}: expected 1 real root, got {roots}"
        x0 = q.answer[0]
        assert real_roots[0] == x0, f"{tier} seed={seed}: root {real_roots[0]} != x0={x0}"
        second = diff(f, x, 2).subs(x, x0)
        assert second != 0, f"{tier} seed={seed}: f''(x0) == 0"
        is_min = q.answer[1] == "local minimum"
        assert (second > 0) == is_min, f"{tier} seed={seed}: sign mismatch"
        n_ok += 1
    assert n_ok == 200, f"{tier}: only {n_ok}/200 seeds passed the invariant"


# --- Classification balance, 35-65%, >= 200 seeds/tier, exact counts -------


@pytest.mark.parametrize("tier", TIERS)
def test_classification_balance_200_seeds(tier):
    counts = Counter(make(TOPIC, seed=seed, difficulty=tier).answer[1] for seed in range(200))
    for label in ("local maximum", "local minimum"):
        frac = counts.get(label, 0) / 200
        assert 0.35 <= frac <= 0.65, f"{tier}: {label!r} is {counts.get(label, 0)}/200 = {frac:.3f}, outside 35-65%"


def test_hard_tier_x0_is_not_copy_pasteable_200_seeds():
    """The hard-tier prompt must never print a literal (x - p)/(x + p)
    bracket that hands the critical x-value straight to the learner."""
    import re

    n_leaked = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        x0 = q.answer[0]
        leaked = (
            re.search(rf"\(x\s*-\s*{x0}\b", q.text) is not None or re.search(rf"\(x\s*\+\s*{-x0}\b", q.text) is not None
        )
        if leaked:
            n_leaked += 1
    assert n_leaked == 0, f"{n_leaked}/200 hard seeds print the critical x-value in a bare (x +/- p) bracket"


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "ordered_list"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
