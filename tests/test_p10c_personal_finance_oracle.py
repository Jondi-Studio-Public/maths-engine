"""Independent oracle for `p10c_personal_budget` and `p10c_government_allowance`, recomputing their answers from
the rendered text."""

import functools

"""Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p10c_personal_finance_build` and recomputes the expected
answer from first principles, then compares it against `answer_text`'s own
displayed numeric value.

Generates through each skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

BUDGET_TOPIC = "p10c_personal_budget"
ALLOWANCE_TOPIC = "p10c_government_allowance"

_BUDGET_ID = "senior-prepare-a-personal-budget-for-a-given-income"
_ALLOWANCE_ID = "senior-calculate-payments-based-on-government-allowances-and-pensions"

SEEDS = range(60)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, skill_id, difficulty, seed):
    skill = skills_dict[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _num(token: str) -> Fraction:
    return Fraction(token.lstrip("$"))


def _answer_value(answer_text: str) -> Fraction:
    return _num(answer_text)


# =============================================================================
# Skill 1: p10c_personal_budget (ACMGM004)
# =============================================================================

_BUDGET_EASY_PATTERN = re.compile(
    r"A person earns \$([\d.]+) per week\. Their fixed weekly expenses are \$([\d.]+) for rent "
    r"and \$([\d.]+) for utility bills\. How much of their weekly income remains after paying "
    r"these fixed expenses\?"
)

_BUDGET_MEDIUM_PATTERN = re.compile(
    r"A person earns \$([\d.]+) per week\. Their fixed weekly expenses are \$([\d.]+) for rent, "
    r"\$([\d.]+) for utility bills and \$([\d.]+) for insurance\. They also budget \$([\d.]+) for "
    r"entertainment and \$([\d.]+) for eating out\. How much of their weekly income remains for "
    r"savings after these fixed and discretionary expenses\?"
)

_BUDGET_HARD_PATTERNS = [
    re.compile(
        r"A person earns \$([\d.]+) per week\. Their fixed weekly expenses are \$([\d.]+) for rent "
        r"and \$([\d.]+) for utility bills\. They budget (\d+)% of their income for discretionary "
        r"spending\. How much of their weekly income remains for savings after these fixed "
        r"expenses and their discretionary budget\?"
    ),
    re.compile(
        r"A person earns \$([\d.]+) per week\. Their fixed weekly expenses are \$([\d.]+) for rent, "
        r"\$([\d.]+) for utility bills and \$([\d.]+) for insurance\. They budget (\d+)% of their "
        r"income for discretionary spending\. How much of their weekly income remains for savings "
        r"after these fixed expenses and their discretionary budget\?"
    ),
]


def _oracle_budget_easy(text: str) -> Fraction:
    m = _BUDGET_EASY_PATTERN.search(text)
    if not m:
        raise AssertionError(f"unparseable easy-tier budget question: {text!r}")
    income, f1, f2 = (_num(g) for g in m.groups())
    return income - f1 - f2


def _oracle_budget_medium(text: str) -> Fraction:
    m = _BUDGET_MEDIUM_PATTERN.search(text)
    if not m:
        raise AssertionError(f"unparseable medium-tier budget question: {text!r}")
    income, f1, f2, f3, d1, d2 = (_num(g) for g in m.groups())
    return income - (f1 + f2 + f3) - (d1 + d2)


def _oracle_budget_hard(text: str) -> Fraction:
    for pat in _BUDGET_HARD_PATTERNS:
        m = pat.search(text)
        if m:
            groups = m.groups()
            *fixed_and_income, pct_str = groups
            income = _num(fixed_and_income[0])
            fixed = [_num(g) for g in fixed_and_income[1:]]
            pct = Fraction(int(pct_str))
            discretionary = income * pct / 100
            return income - sum(fixed) - discretionary
    raise AssertionError(f"unparseable hard-tier budget question: {text!r}")


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_budget_easy),
        ("medium", _oracle_budget_medium),
        ("hard", _oracle_budget_hard),
    ],
)
def test_budget_oracle(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _BUDGET_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = _answer_value(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# =============================================================================
# Skill 2: p10c_government_allowance (ACMGM003)
# =============================================================================

_ALLOWANCE_EASY_PATTERN = re.compile(
    r"pays a base rate of \$([\d.]+) per fortnight, plus \$([\d.]+) for each dependent child\. "
    r"Calculate the fortnightly payment for a person with (\d+) dependent child(?:ren)?\."
)

_ALLOWANCE_MEDIUM_PATTERN = re.compile(
    r"pays a base rate of \$([\d.]+) per fortnight, plus \$([\d.]+) for each dependent child, plus "
    r"a weekly energy supplement of \$([\d.]+), which this person is eligible to receive\. "
    r"Calculate the FORTNIGHTLY payment for a person with (\d+) dependent child(?:ren)?\."
)

_ALLOWANCE_HARD_PATTERN = re.compile(
    r"pays a base rate of \$([\d.]+) per fortnight, plus \$([\d.]+) for each dependent child\. "
    r"The payment is reduced by \$([\d.]+) for every complete \$([\d.]+) this person earns above a "
    r"fortnightly income-free threshold of \$([\d.]+)\. This person has (\d+) dependent child(?:ren)? "
    r"and earns \$([\d.]+) per fortnight from part-time work\."
)


def _oracle_allowance_easy(text: str) -> Fraction:
    m = _ALLOWANCE_EASY_PATTERN.search(text)
    if not m:
        raise AssertionError(f"unparseable easy-tier allowance question: {text!r}")
    b, c, n = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
    return b + c * n


def _oracle_allowance_medium(text: str) -> Fraction:
    m = _ALLOWANCE_MEDIUM_PATTERN.search(text)
    if not m:
        raise AssertionError(f"unparseable medium-tier allowance question: {text!r}")
    b, c, w, n = (_num(m.group(i)) for i in (1, 2, 3, 4))
    return b + c * n + 2 * w


def _oracle_allowance_hard(text: str) -> Fraction:
    m = _ALLOWANCE_HARD_PATTERN.search(text)
    if not m:
        raise AssertionError(f"unparseable hard-tier allowance question: {text!r}")
    b, c, r, s, t, n, e = (_num(m.group(i)) for i in (1, 2, 3, 4, 5, 6, 7))
    base = b + c * n
    excess = e - t
    increments = int(excess // s)
    reduction = increments * r
    return base - reduction


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_allowance_easy),
        ("medium", _oracle_allowance_medium),
        ("hard", _oracle_allowance_hard),
    ],
)
def test_allowance_oracle(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _ALLOWANCE_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = _answer_value(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- Distractors: the oracle's own recomputation must never equal a
# distractor either (a stronger check than "the generator's own check_distinct
# passed") --------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,topic,difficulty,oracle_fn",
    [
        (_BUDGET_ID, BUDGET_TOPIC, "easy", _oracle_budget_easy),
        (_BUDGET_ID, BUDGET_TOPIC, "medium", _oracle_budget_medium),
        (_BUDGET_ID, BUDGET_TOPIC, "hard", _oracle_budget_hard),
        (_ALLOWANCE_ID, ALLOWANCE_TOPIC, "easy", _oracle_allowance_easy),
        (_ALLOWANCE_ID, ALLOWANCE_TOPIC, "medium", _oracle_allowance_medium),
        (_ALLOWANCE_ID, ALLOWANCE_TOPIC, "hard", _oracle_allowance_hard),
    ],
)
def test_oracle_answer_never_matches_a_distractor(skill_id, topic, difficulty, oracle_fn):
    skills_dict = _skills_dict()
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        expected = oracle_fn(q.text)
        for d in q.distractors:
            assert expected != _num(d), (
                f"seed {seed} {difficulty} {topic}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )
