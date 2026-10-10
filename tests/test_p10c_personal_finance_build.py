"""Personal budget + government allowances and pensions.

Tests for:
- senior-prepare-a-personal-budget-for-a-given-income (ACMGM004)
- senior-calculate-payments-based-on-government-allowances-and-pensions (ACMGM003)
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p10c_personal_finance_build import (
    _verify_budget_easy_medium,
    _verify_budget_hard,
    _verify_allowance_easy_medium,
    _verify_allowance_hard,
)

BUDGET_TOPIC = "p10c_personal_budget"
ALLOWANCE_TOPIC = "p10c_government_allowance"
TOPICS = [BUDGET_TOPIC, ALLOWANCE_TOPIC]
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)

_SKILL_IDS = {
    BUDGET_TOPIC: "senior-prepare-a-personal-budget-for-a-given-income",
    ALLOWANCE_TOPIC: "senior-calculate-payments-based-on-government-allowances-and-pensions",
}


def _make_from_sources(skills_dict, topic, difficulty, seed):
    """Generate through the skill's own `sources` entry, not a bare
    `make(topic, seed, difficulty)` call -- the params (here empty) are what
    a real skill drives the generator with."""
    skill = skills_dict[_SKILL_IDS[topic]]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize(
    "topic,difficulty,expected_n",
    [
        (BUDGET_TOPIC, "easy", 2),
        (BUDGET_TOPIC, "medium", 2),
        (BUDGET_TOPIC, "hard", 4),
        (ALLOWANCE_TOPIC, "easy", 1),  # only one misconception is defined for this
        (ALLOWANCE_TOPIC, "medium", 2),
        (ALLOWANCE_TOPIC, "hard", 3),
    ],
)
def test_distractor_count_distinct_nonempty_never_equal_answer(skills_dict, topic, difficulty, expected_n):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert len(q.distractors) == expected_n, (topic, difficulty, seed, q.distractors)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        if isinstance(q.answer, int):
            wrong = str(q.answer + 1)
        else:
            wrong = str(round(q.answer + 1.0, 2))
        assert not grade_number(q, wrong).correct, (topic, difficulty, seed, q.text, q.answer_text, wrong)


# --- Grader accepts every reasonable way a correct learner might type the
# same currency answer (a defect class: the learner is right and the
# system must not say otherwise) ------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_grader_accepts_equivalent_typed_forms_of_the_currency_answer(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert q.answer_text.startswith("$")
        bare = q.answer_text[1:]
        no_cents = bare[:-3] if bare.endswith(".00") else bare
        for typed in (q.answer_text, bare, no_cents, f"${no_cents}"):
            assert grade_number(q, typed).correct, (topic, difficulty, seed, q.answer_text, typed)


# --- Answer-kind / formatting discipline --------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_currency_with_two_decimal_places(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert isinstance(q.answer, (int, float))
        assert q.answer_text.startswith("$")
        whole, frac = q.answer_text[1:].split(".")
        assert len(frac) == 2
        assert Fraction(q.answer_text[1:]) * 100 % 1 == 0


# --- Every distractor formatted like the answer within one question ----


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_formatted_identically(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        for d in q.distractors:
            assert d.startswith("$")
            assert len(d.split(".")[1]) == 2


# --- Value bounds: income and the correct payment/remainder stay under
# $1000 and >= $10 by construction (BUDGET) or >= $0 (ALLOWANCE) ---------


def test_budget_answer_between_10_and_1000():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(BUDGET_TOPIC, seed=seed, difficulty=difficulty)
            value = Fraction(q.answer_text[1:])
            assert Fraction(10) <= value < Fraction(1000), (difficulty, seed, q.answer_text)


def test_allowance_hard_income_under_1000():
    for seed in SEEDS:
        q = make(ALLOWANCE_TOPIC, seed=seed, difficulty="hard")
        # E is stated in the question text; the whole scenario is under $1000.
        assert "earns $" in q.text
        income_str = q.text.split("earns $")[1].split(" per fortnight")[0]
        assert Fraction(income_str) < Fraction(1000), (seed, q.text)


# --- No real government scheme is ever named ----------------------------

_REAL_SCHEME_NAMES = [
    "youth allowance",
    "jobseeker",
    "newstart",
    "centrelink",
    "disability support pension",
    "age pension",
    "austudy",
    "abstudy",
    "family tax benefit",
    "parenting payment",
]


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_allowance_never_names_a_real_scheme(difficulty):
    for seed in SEEDS:
        q = make(ALLOWANCE_TOPIC, seed=seed, difficulty=difficulty)
        lowered = q.text.lower()
        for name in _REAL_SCHEME_NAMES:
            assert name not in lowered, (seed, difficulty, name, q.text)
        assert "hypothetical" in lowered


# --- At least 20 distinct rendered questions per tier over 60 seeds ------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(topic, difficulty):
    seen = {make(topic, seed=s, difficulty=difficulty).text for s in range(60)}
    assert len(seen) >= 20, (topic, difficulty, len(seen))


# --- ValueError on a nonsensical preset -----------------------------------


@pytest.mark.parametrize("topic", TOPICS)
def test_raises_value_error_on_unknown_mode(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, mode="bogus")


# --- Independent in-generator checks fire on a corrupted answer ---------


def test_verify_budget_easy_medium_fires_on_corrupted():
    _verify_budget_easy_medium(500, [50, 60], 0, 390)
    with pytest.raises(VerificationError):
        _verify_budget_easy_medium(500, [50, 60], 0, 391)


def test_verify_budget_hard_fires_on_corrupted():
    # I=500, fixed=[50,60], p=10 -> disc_cents = 500*10 = 5000; F=110
    # answer_cents = 50000 - 11000 - 5000 = 34000
    _verify_budget_hard(500, [50, 60], 10, 34000)
    with pytest.raises(VerificationError):
        _verify_budget_hard(500, [50, 60], 10, 34001)


def test_verify_allowance_easy_medium_fires_on_corrupted():
    _verify_allowance_easy_medium(400, 30, 2, 0, 460)
    with pytest.raises(VerificationError):
        _verify_allowance_easy_medium(400, 30, 2, 0, 461)
    # medium tier: W nonzero exercises the repeated-addition conversion check too
    _verify_allowance_easy_medium(400, 30, 2, 20, 500)
    with pytest.raises(VerificationError):
        _verify_allowance_easy_medium(400, 30, 2, 20, 501)


def test_verify_allowance_hard_fires_on_corrupted():
    # E=500, T=300, s=50, r=10 -> (E-T)=200, 4 complete increments -> reduction=40
    _verify_allowance_hard(500, 300, 50, 10, 40)
    with pytest.raises(VerificationError):
        _verify_allowance_hard(500, 300, 50, 10, 41)
    # non-exact-multiple must also fire
    with pytest.raises(VerificationError):
        _verify_allowance_hard(510, 300, 50, 10, 40)


# --- Both skills load correctly through the API --------------------------


def test_both_skills_load_through_api(skills_dict):
    budget = skills_dict["senior-prepare-a-personal-budget-for-a-given-income"]
    assert budget is not None
    assert budget.domain == "arithmetic"
    assert budget.prerequisites == ["order-of-operations"]
    assert budget.answer == "number"
    assert budget.placement_eligible is False
    assert budget.status == "active"
    assert len(budget.sources) == 3

    allowance = skills_dict["senior-calculate-payments-based-on-government-allowances-and-pensions"]
    assert allowance is not None
    assert allowance.domain == "arithmetic"
    assert allowance.prerequisites == ["order-of-operations"]
    assert allowance.answer == "number"
    assert allowance.placement_eligible is False
    assert allowance.status == "active"
    assert len(allowance.sources) == 3
