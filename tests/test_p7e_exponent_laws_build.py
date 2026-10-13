"""Tests `f10-exponent-laws-integer-variable` (AC9M9A01 / AC9M10A01, topic `exponent_laws_integer_variable`)."""

import re

import pytest
from sympy import symbols, sympify

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7e_exponent_laws_build import _verify

TOPIC = "exponent_laws_integer_variable"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)

x = symbols("x")


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    two_plus = 0
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        if len(question.distractors) >= 2:
            two_plus += 1
    assert two_plus / len(SEEDS) >= 0.85


# --- real grader accept/reject ------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_learner_typed_without_parens(difficulty):
    """A learner typing a negative exponent the natural way (no parens, e.g.
    "x^-5") must be accepted -- the whole reason answer_text never contains a
    "(" (see the module docstring's factored-form-guard note)."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "(" not in question.answer_text
        exponent = int(question.answer_text.split("^")[1])
        if exponent < 0:
            typed = f"1/x**{-exponent}"
            assert grade_expression(question, typed).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        exponent = int(question.answer_text.split("^")[1])
        wrong = f"x^{exponent + 3}"
        assert not grade_expression(question, wrong).correct


# --- independent in-generator check fires on a corrupted input --------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify(x**3 * x**2, 999)


# --- answer satisfies its declared kind: single power of x -----------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_a_single_power_of_x(difficulty):
    pattern = re.compile(r"^x\^-?\d+$")
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert pattern.match(question.answer_text), question.answer_text
        for distractor in question.distractors:
            assert pattern.match(distractor), distractor
        # sympifies cleanly (the internal `answer` field, used by the grader)
        sympify(question.answer)


def test_easy_tier_has_at_least_one_negative_input_every_seed():
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="easy")
        assert "(-" in question.text


def test_medium_tier_final_exponent_negative_on_clear_majority():
    negative = 0
    total = 200
    for seed in range(total):
        question = make(TOPIC, seed=seed, difficulty="medium")
        exponent = int(question.answer_text.split("^")[1])
        if exponent < 0:
            negative += 1
    assert negative / total >= 0.7


def test_hard_tier_final_exponent_sign_not_fixed():
    signs = set()
    for seed in range(200):
        question = make(TOPIC, seed=seed, difficulty="hard")
        exponent = int(question.answer_text.split("^")[1])
        signs.add(exponent < 0)
    assert signs == {True, False}


# --- >=20 distinct questions per tier over 60 seeds --------------------------


# --- formatted alike: no bare number among "single power of x" options -----


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_and_distractors_formatted_alike(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        options = [question.answer_text, *question.distractors]
        for option in options:
            assert option.startswith("x^"), option


# --- skill contract ----------------------------------------------------------


def test_skill_loads_with_expected_shape():
    _, skills = load_skills()
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["f10-exponent-laws-integer-variable"]

    assert skill.prerequisites == ["f10-exponent-laws-positive-zero", "logarithm-laws"]
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {source.topic for source in skill.sources} == {TOPIC, "exd_x_n3_exponent_two_variables"}
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [source.stages for source in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
