"""Trigonometric derivative applications; implicit differentiation.

Skill 1: senior-use-trigonometric-functions-and-their-derivatives-to-solve
(topic ``p12c_trig_deriv_apply``, answer: number).
Skill 2: senior-use-implicit-differentiation-to-determine-the-gradient-of
(topic ``p12c_implicit_diff_gradient``, answer: expression).
"""

import re

import pytest
from sympy import sympify, symbols

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC_1 = "p12c_trig_deriv_apply"
SKILL_1 = "senior-use-trigonometric-functions-and-their-derivatives-to-solve"
TOPIC_2 = "p12c_implicit_diff_gradient"
SKILL_2 = "senior-use-implicit-differentiation-to-determine-the-gradient-of"

_grade_number = GRADERS["number"]
_grade_expression = GRADERS["expression"]

x, y = symbols("x y")


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
        assert len(q.distractors) >= 1, f"seed={seed}: no distractors"


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_skill1_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        result = _grade_number(q, q.answer_text)
        assert result.correct, f"{TOPIC_1}/{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = _grade_number(q, d)
            assert not result.correct, f"{TOPIC_1}/{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_skill1_real_grader_rejects_a_deliberately_wrong_value(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        # Margin exceeds the tolerance derived from `answer_text`'s own
        # stated precision (int -> exact; 1dp -> 0.05; 2dp -> 0.005).
        wrong = str(float(q.answer_text) + 5)
        result = _grade_number(q, wrong)
        assert not result.correct, f"{TOPIC_1}/{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_skill2_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        result = _grade_expression(q, q.answer_text)
        assert result.correct, f"{TOPIC_2}/{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = _grade_expression(q, d)
            assert not result.correct, f"{TOPIC_2}/{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_skill2_real_grader_rejects_a_deliberately_wrong_value(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        wrong = q.answer_text + " + 1"
        result = _grade_expression(q, wrong)
        assert not result.correct, f"{TOPIC_2}/{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check_distinct(1, 1, msg="should fail")


# --- Distinct rendered questions -------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct questions over 60 seeds"


# --- Answer kind: skill 1 is always a plain number, correctly precise ------


@pytest.mark.parametrize("tier", TIERS)
def test_skill1_answer_is_a_plain_number_correctly_precise(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        # No stray symbols/units in the canonical answer text.
        assert re.fullmatch(r"-?\d+(\.\d+)?", q.answer_text), (
            f"{TOPIC_1}/{tier} seed={seed}: answer_text not a plain number: {q.answer_text!r}"
        )
        if tier in ("hard", "exam") and tier == "hard":
            assert "." in q.answer_text and len(q.answer_text.split(".")[1]) == 2, (
                f"{tier} seed={seed}: hard answer not stated to 2 d.p.: {q.answer_text!r}"
            )


# --- Answer kind: skill 2 is always an expression in x and y ---------------


@pytest.mark.parametrize("tier", TIERS)
def test_skill2_answer_and_distractors_are_expressions_in_x_and_y(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        assert "x" in q.answer_text and "y" in q.answer_text, (
            f"{tier} seed={seed}: answer_text missing x or y: {q.answer_text!r}"
        )
        expr = sympify(q.answer_text.replace("^", "**"), locals={"x": x, "y": y})
        assert expr.free_symbols <= {x, y}, f"{tier} seed={seed}: unexpected symbols in {expr}"


# --- Plausibility: skill 2's denominator is never the zero polynomial ------


@pytest.mark.parametrize("tier", TIERS)
def test_skill2_denominator_never_identically_zero(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        expr = sympify(q.answer_text.replace("^", "**"), locals={"x": x, "y": y})
        # A genuine rational function in x, y: substituting a nonzero point
        # must not raise ZeroDivisionError/ComplexInfinity from an
        # identically-zero denominator.
        val = expr.subs({x: 2, y: 3})
        assert val.is_finite is not False, f"{tier} seed={seed}: denominator vanished at (2, 3): {expr}"


# --- Plausibility: skill 1's hard-tier solution lands inside the stated
# interval every seed --------------------------------------------------


_HARD_RE = re.compile(r"For 0 ≤ t ≤ (?P<bound>-?\d+\.\d+), find the time t at which")


def test_skill1_hard_tier_solution_lands_in_stated_interval():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty="hard")
        m = _HARD_RE.search(q.text)
        assert m is not None, f"seed={seed}: hard tier text missing interval: {q.text!r}"
        bound = float(m.group("bound"))
        t_val = float(q.answer_text)
        assert -0.01 <= t_val <= bound + 0.01, f"seed={seed}: hard answer {t_val} outside stated interval [0, {bound}]"
        checked += 1
    assert checked == len(SEEDS)


# --- Boundary: skill 1 never says bare "differentiate" or asks for a
# tangent-line equation --------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_skill1_never_bare_differentiate_or_tangent_equation(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        assert "differentiate" not in q.text.lower(), f"{tier} seed={seed}: bare 'differentiate' in text: {q.text!r}"
        assert "equation of the tangent" not in q.text.lower(), (
            f"{tier} seed={seed}: tangent-line phrasing in text: {q.text!r}"
        )


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
        # The exam tier is now a multi-part chain (tests/test_exd_x_r2j_trig_rate_chain.py).
        assert source.topic == ("exd_x_r2j_trig_rate_chain" if source.difficulty == "exam" else TOPIC_1)
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"


def test_skill_2_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_2), None)
    assert skill is not None, f"{SKILL_2} not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["solve-ode-exponential"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        # The exam tier is now the multi-part chain (tests/test_exam_parts_implicit_tangent.py).
        assert source.topic == ("exam_parts_implicit_tangent" if source.difficulty == "exam" else TOPIC_2)
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"


# Proved red before medium got its own six-angle pool.
def test_trig_medium_answer_zero_is_not_a_quarter_of_draws():
    from mathgen.skills import load_skills
    from mathgen import make

    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "senior-use-trigonometric-functions-and-their-derivatives-to-solve")
    src = next(s for s in skill.sources if s.difficulty == "medium")
    answers = [
        make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {})).answer_text for seed in range(1000)
    ]
    assert answers.count("0") / len(answers) < 0.22
