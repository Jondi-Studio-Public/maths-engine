"""Independent oracle for the first-order recurrence steady-state skill (ACMGM077).

Independent oracle for senior-use-first-order-linear-recurrence-relations-to-model
(ACMGM077).

Parses the RENDERED question text only (``question.text``/``question.answer_text``) and imports
NOTHING from ``mathgen.generators.pv0a_steady_state_build``. Every route below re-derives the
long-term value from the stated text by ITERATING the recurrence in exact ``Fraction`` arithmetic
until successive values agree far below the stated precision -- a different route from the
generator's own closed form ``b / (1 - a)``.
"""

from dataclasses import replace

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import round_half_up

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS_1000 = range(1000)
SKILL_ID = "senior-use-first-order-linear-recurrence-relations-to-model"


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    if src.difficulty == "exam":
        # The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
        src = replace(src, topic="pv0a_recurrence_steady_state", params={"variant": "exam"})
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _iterate_to_steady_state(a: Fraction, b: Fraction, periods: int = 400) -> Fraction:
    """Independent route: iterate from an arbitrary start (0) rather than solving the closed
    form b/(1-a) directly."""
    u = Fraction(0)
    for _ in range(periods):
        u = a * u + b
    return u


def _answer_to_fraction(s: str) -> Fraction:
    return Fraction(s)


_A_RE = re.compile(r"u\(n\+1\) = (\d(?:\.\d)?)u\(n\) \+ (\d+)")
_A_SPACED_RE = re.compile(r"u\(n\+1\) = (\d(?:\.\d)?) u\(n\) \+ (\d+)")
_PREC_RE = re.compile(r"to (the nearest whole number|1 decimal place)")


def _precision_from_text(text: str) -> int:
    m = _PREC_RE.search(text)
    assert m, text
    return 0 if m.group(1) == "the nearest whole number" else 1


# =============================================================================
# easy / medium: a and b are directly recoverable from the recurrence's own numbers
# =============================================================================

_PERCENT_RE = re.compile(r"(\d+)% of")


def _oracle_easy(text: str) -> Fraction:
    m = _A_RE.search(text)
    a, b = Fraction(m.group(1)), Fraction(m.group(2))
    return _iterate_to_steady_state(a, b)


def _oracle_medium(text: str) -> Fraction:
    p = Fraction(_PERCENT_RE.search(text).group(1))
    a = (100 - p) / 100
    # The added amount is found via the known per-context phrasing markers, not positional
    # guessing (several numbers appear in a medium-tier prompt: the percentage, the amount, u0).
    add_match = re.search(
        r"(?:a (\d+) mg dose|(\d+) fish are then restocked|(\d+) kg of pollutant|"
        r"topped up with (\d+) g|(\d+) new members then join)",
        text,
    )
    b = Fraction(next(g for g in add_match.groups() if g))
    return _iterate_to_steady_state(a, b)


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_easy_medium_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources(difficulty, seed)
        d = _precision_from_text(q.text)
        l_exact = _oracle_easy(q.text) if difficulty == "easy" else _oracle_medium(q.text)
        expected = round_half_up(l_exact, d)
        assert expected == _answer_to_fraction(q.answer_text), (difficulty, seed, q.text, q.answer_text)
        checked += 1
    assert checked == 1000


# =============================================================================
# hard: the OTHER point of the cycle is asked
# =============================================================================

_HARD_BEFORE_RE = re.compile(r"u\(n\+1\) = (\d(?:\.\d)?)\(u\(n\) \+ (\d+)\)")


def _iterate_pre_event(a: Fraction, b: Fraction, periods: int = 400) -> Fraction:
    """Independent route for the 'dose then eliminate' (pre-event) form u(n+1) = a(u(n) + b):
    iterate u <- a*(u + b) directly, rather than expanding it into a*u + a*b first."""
    u = Fraction(0)
    for _ in range(periods):
        u = a * (u + b)
    return u


def _oracle_hard(text: str) -> Fraction:
    # The recurrence FORM shown determines which point u(n) is (the two forms are not
    # interchangeable relabellings of the same formula -- see the generator module's own
    # docstring on this). u(n+1) = a u(n) + b: u(n) is post-event, asked is pre-event = a*L.
    # u(n+1) = a(u(n) + b): u(n) is pre-event, asked is post-event = pre_L + b.
    m_before = _HARD_BEFORE_RE.search(text)
    if m_before:
        a, b = Fraction(m_before.group(1)), Fraction(m_before.group(2))
        pre_l = _iterate_pre_event(a, b)
        return pre_l + b
    m_after = _A_RE.search(text)
    a, b = Fraction(m_after.group(1)), Fraction(m_after.group(2))
    post_l = _iterate_to_steady_state(a, b)
    return a * post_l


def test_hard_oracle_matches_answer_text():
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources("hard", seed)
        d = _precision_from_text(q.text)
        expected = round_half_up(_oracle_hard(q.text), d)
        assert expected == _answer_to_fraction(q.answer_text), ("hard", seed, q.text, q.answer_text)
        checked += 1
    assert checked == 1000


# =============================================================================
# exam: recover a, b from three consecutive recorded values, then iterate
# =============================================================================

_EXAM_VALUES_RE = re.compile(
    r"u\(0\) = (-?\d+(?:\.\d+)?), u\(1\) = (-?\d+(?:\.\d+)?)(?:,| and) u\(2\) = (-?\d+(?:\.\d+)?)"
)


def _oracle_exam(text: str) -> Fraction:
    m = _EXAM_VALUES_RE.search(text)
    u0, u1, u2 = (Fraction(m.group(i)) for i in (1, 2, 3))
    a = (u2 - u1) / (u1 - u0)
    b = u1 - a * u0
    return _iterate_to_steady_state(a, b)


def test_exam_oracle_matches_answer_text():
    checked = 0
    for seed in SEEDS_1000:
        q = _make_from_sources("exam", seed)
        d = _precision_from_text(q.text)
        expected = round_half_up(_oracle_exam(q.text), d)
        assert expected == _answer_to_fraction(q.answer_text), ("exam", seed, q.text, q.answer_text)
        checked += 1
    assert checked == 1000


# =============================================================================
# Boundary assertion: the exact value is never within 0.05 of a unit
# in the last stated place from a rounding boundary.
# =============================================================================


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_answer_sits_on_a_rounding_boundary(difficulty):
    for seed in range(300):
        q = _make_from_sources(difficulty, seed)
        d = _precision_from_text(q.text)
        if difficulty == "easy":
            l_exact = _oracle_easy(q.text)
        elif difficulty == "medium":
            l_exact = _oracle_medium(q.text)
        elif difficulty == "hard":
            l_exact = _oracle_hard(q.text)
        else:
            l_exact = _oracle_exam(q.text)
        scaled = l_exact * (10**d)
        frac_part = scaled - int(scaled)
        # distance from the nearest .5 boundary, in scaled units
        dist_from_half = abs(frac_part - Fraction(1, 2))
        assert dist_from_half >= Fraction(1, 20), (difficulty, seed, q.text, l_exact, d)
