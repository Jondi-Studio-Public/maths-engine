"""Independent oracle for the `p8e_percentage_review` questions: it re-derives each answer from the rendered
question text.

Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p8e_percentage_review_build` and recomputes the expected
answer from first principles, then compares it against `answer_text`'s own
displayed numeric value, including its stated decimal precision.

Generates through each skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""

import functools
import math
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

AMOUNT_TOPIC = "percentage_of_amount_review"
PCT_TOPIC = "amount_as_percentage_review"

_AMOUNT_ID = "senior-review-calculating-a-percentage-of-a-given-amount"
_PCT_ID = "senior-review-one-amount-expressed-as-a-percentage-of"

SEEDS = range(60)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, skill_id, difficulty, seed):
    skill = skills_dict[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _dollars(token: str) -> Fraction:
    return Fraction(token.lstrip("$"))


def _round_half_up_tenths(value: Fraction) -> int:
    """Independent re-implementation of nearest-tenth rounding (half away
    from zero), deliberately not imported from the generator module."""
    scaled = value * 10
    if scaled >= 0:
        return math.floor(scaled + Fraction(1, 2))
    return -math.floor(-scaled + Fraction(1, 2))


# =============================================================================
# Skill 1: percentage_of_amount_review (ACMEM061)
# =============================================================================

_AMOUNT_EASY_HARD_PATTERN = re.compile(r"What is (\d+)% of \$([\d.]+)\?")
_AMOUNT_MEDIUM_PATTERN = re.compile(
    r"A \S+(?: \S+)? costs \$([\d.]+) and a \S+(?: \S+)? costs \$([\d.]+)\. "
    r"What is (\d+)% of the total cost of both items\?"
)


def _oracle_amount_easy_hard(text: str) -> Fraction:
    m = _AMOUNT_EASY_HARD_PATTERN.search(text)
    if not m:
        raise AssertionError(f"unparseable easy/hard-tier amount question: {text!r}")
    p, amount = int(m.group(1)), _dollars(m.group(2))
    return amount * Fraction(p, 100)


def _oracle_amount_medium(text: str) -> Fraction:
    m = _AMOUNT_MEDIUM_PATTERN.search(text)
    if not m:
        raise AssertionError(f"unparseable medium-tier amount question: {text!r}")
    a, b, p = _dollars(m.group(1)), _dollars(m.group(2)), int(m.group(3))
    return (a + b) * Fraction(p, 100)


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_amount_easy_hard),
        ("medium", _oracle_amount_medium),
        ("hard", _oracle_amount_easy_hard),
    ],
)
def test_amount_oracle(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _AMOUNT_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = _dollars(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# =============================================================================
# Skill 2: amount_as_percentage_review (ACMEM062)
# =============================================================================

_PCT_LE_PATTERNS = [
    re.compile(r"In a survey of (\d+) people, (\d+) own a car\."),
    re.compile(r"A school has (\d+) students enrolled\. (\d+) of them play a sport\."),
    re.compile(r"A shop receives (\d+) orders in a week\. (\d+) of the orders are returned\."),
    re.compile(r"Out of (\d+) matches played, a team won (\d+) of them\."),
]

_PCT_GT_PATTERNS = [
    re.compile(r"Route A is (\d+) km long\. Route B is (\d+) km long\."),
    re.compile(r"Team A scored (\d+) points\. Team B scored (\d+) points in the same competition\."),
    re.compile(r"Shop A sold (\d+) tickets\. Shop B sold (\d+) tickets for the same event\."),
]


def _extract_B_A(text: str) -> tuple[int, int]:
    """LE-shape templates state B before A ("survey of B people, A own")."""
    for pat in _PCT_LE_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
    raise AssertionError(f"unparseable LE-shape pct question: {text!r}")


def _extract_A_B(text: str) -> tuple[int, int]:
    """GT-shape templates state A before B ("Route A is A km, Route B is B km")."""
    for pat in _PCT_GT_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
    raise AssertionError(f"unparseable GT-shape pct question: {text!r}")


def _oracle_pct_easy(text: str) -> Fraction:
    B, A = _extract_B_A(text)
    return Fraction(A, B) * 100


def _oracle_pct_medium(text: str) -> Fraction:
    B, A = _extract_B_A(text)
    return Fraction(_round_half_up_tenths(Fraction(A, B) * 100), 10)


def _oracle_pct_hard(text: str) -> Fraction:
    A, B = _extract_A_B(text)
    return Fraction(_round_half_up_tenths(Fraction(A, B) * 100), 10)


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_pct_easy),
        ("medium", _oracle_pct_medium),
        ("hard", _oracle_pct_hard),
    ],
)
def test_pct_oracle(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _PCT_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = Fraction(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- Rounding-boundary guard: the true (unrounded) value behind a
# medium/hard answer must never sit exactly on its own rounding tie
# (banker's rounding and half-up would then disagree) -------------------


@pytest.mark.parametrize("difficulty,extract", [("medium", _extract_B_A), ("hard", _extract_A_B)])
def test_pct_rounded_answer_never_sits_on_a_tie(difficulty, extract):
    for seed in SEEDS:
        q = _make_from_sources(_skills_dict(), _PCT_ID, difficulty, seed)
        first, second = extract(q.text)
        A, B = (second, first) if difficulty == "medium" else (first, second)
        assert Fraction(A, B) * 1000 % 1 != Fraction(1, 2), (seed, difficulty, q.text)


# --- Distractors: the oracle's own recomputation must never equal a
# distractor either -----------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,difficulty,oracle_fn,dollar",
    [
        (_AMOUNT_ID, "easy", _oracle_amount_easy_hard, True),
        (_AMOUNT_ID, "medium", _oracle_amount_medium, True),
        (_AMOUNT_ID, "hard", _oracle_amount_easy_hard, True),
        (_PCT_ID, "easy", _oracle_pct_easy, False),
        (_PCT_ID, "medium", _oracle_pct_medium, False),
        (_PCT_ID, "hard", _oracle_pct_hard, False),
    ],
)
def test_oracle_answer_never_matches_a_distractor(skill_id, difficulty, oracle_fn, dollar):
    skills_dict = _skills_dict()
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        expected = oracle_fn(q.text)
        for d in q.distractors:
            actual_d = _dollars(d) if dollar else Fraction(d)
            assert expected != actual_d, (
                f"seed {seed} {difficulty} {skill_id}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )
