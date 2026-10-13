"""Tests `p19f_concavity_inflection`: grading, distractors and the single-inflection construction.

Concavity and points of inflection.

Skill: senior-understand-the-concepts-of-concavity-and-points-of
(topic ``p19f_concavity_inflection``, answer: ordered_list).
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

TOPIC = "p19f_concavity_inflection"
SKILL = "senior-understand-the-concepts-of-concavity-and-points-of"

_grade_ordered_list = GRADERS["ordered_list"]

x = symbols("x")


def _parse_fx(text: str):
    from sympy import sympify

    m = re.search(r"[fRQ]\(x\) = ([-+*^·0-9x() ]+)", text)
    assert m is not None, f"could not parse f(x) from: {text!r}"
    # A trailing unmatched "(" is a follow-on parenthetical in the sentence
    # ("... (not yet simplified)"), not part of the expression -- strip it
    # off along with the punctuation/whitespace immediately before it.
    expr_text = m.group(1).strip()
    while expr_text.count("(") > expr_text.count(")"):
        expr_text = expr_text.rsplit("(", 1)[0].rstrip(", ").strip()
    expr_text = expr_text.replace("^", "**").replace("·", "*")
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
        x0, direction = q.answer
        wrong = f"{x0 + 1}, {direction}"
        result = _grade_ordered_list(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# Grader behaviour: confirm the "bare number, not labelled" lesson (direct GRADERS["ordered_list"]
# call).


def test_grader_accepts_bare_number_but_not_labelled_x_value():
    class Q:
        answer_text = "3, concave up to concave down"
        answer = [3, "concave up to concave down"]

    q = Q()
    assert _grade_ordered_list(q, "3, concave up to concave down").correct
    assert _grade_ordered_list(q, "3, Concave Up To Concave Down").correct
    assert _grade_ordered_list(q, "3,concave up to concave down").correct
    # Positional grader: swapped order is wrong.
    assert not _grade_ordered_list(q, "concave up to concave down, 3").correct
    # "x = 3..." does NOT grade correctly -- the first item fails to parse as a number
    # (`_parse_numeric_item("x = 3")` is None). Confirmed by direct code read/run: the prompt must
    # ask for a bare number, never "x = <value>".
    assert not _grade_ordered_list(q, "x = 3, concave up to concave down").correct


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_condition():
    with pytest.raises(VerificationError):
        check(False, "should fail")


# --- Distinct rendered questions --------------------------------------------


# --- Answer kind: [int x-value, direction phrase] ---------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_x_value_then_direction(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, list) and len(q.answer) == 2
        assert isinstance(q.answer[0], int)
        assert q.answer[1] in ("concave up to concave down", "concave down to concave up")
        for d in q.distractors:
            assert re.fullmatch(r"-?\d+, .+", d), f"{tier} seed={seed}: malformed distractor {d!r}"


# --- The exactly-one-inflection-point invariant, re-derived independently
# from the RENDERED TEXT with sympy, >= 200 seeds/tier, quoting the exact
# pass count. ----------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_exactly_one_inflection_point_invariant_200_seeds(tier):
    n_ok = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier)
        f = _parse_fx(q.text)
        second = diff(f, x, 2)
        roots = solve(second, x)
        real_roots = [r for r in roots if r.is_real]
        assert len(real_roots) == 1, f"{tier} seed={seed}: expected 1 real root, got {roots}"
        x0 = q.answer[0]
        assert real_roots[0] == x0, f"{tier} seed={seed}: root {real_roots[0]} != x0={x0}"
        third = diff(f, x, 3)
        assert third != 0, f"{tier} seed={seed}: f'''(x) == 0"
        is_up_to_down = q.answer[1] == "concave up to concave down"
        assert (third > 0) == (not is_up_to_down), f"{tier} seed={seed}: direction mismatch"
        n_ok += 1
    assert n_ok == 200, f"{tier}: only {n_ok}/200 seeds passed the invariant"


# --- Direction balance, 35-65%, >= 200 seeds/tier, exact counts -------------


@pytest.mark.parametrize("tier", TIERS)
def test_direction_balance_200_seeds(tier):
    counts = Counter(make(TOPIC, seed=seed, difficulty=tier).answer[1] for seed in range(200))
    for label in ("concave up to concave down", "concave down to concave up"):
        frac = counts.get(label, 0) / 200
        assert 0.35 <= frac <= 0.65, f"{tier}: {label!r} is {counts.get(label, 0)}/200 = {frac:.3f}, outside 35-65%"


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "ordered_list"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["second-derivative", "stationary-points"]
    assert len(skill.sources) == 4
    for source in skill.sources[:3]:  # the exam source is the multi-part chain
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"


def test_exam_time_unit_only_on_a_positive_x_200_seeds():
    """The exam reaction context measures x in minutes, so
    a negative inflection x ("-5 minutes") must not be told in it. RED before the fix:
    45 of 101 minute prompts had a negative x."""
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        if "minutes" in q.text:
            assert int(q.answer_text.split(",")[0]) > 0, q.text
