"""Independent oracle for `p12b_annuity_perpetuity` (ACMGM100).

Independent oracle for ACMGM100.

``senior-with-the-aid-of-a-financial-calculator-or-acmgm100`` (topic
``p12b_annuity_perpetuity``, ``answer: number``). Parses ONLY the rendered
``question.text`` -- never importing anything from
``mathgen.generators.p12b_annuity_loan_build`` -- and re-derives the present
value, or (hard tier) the number of years, independently via its own local
(re-implemented, not imported) exact-`Fraction` annuity/perpetuity formula
and its own local round-half-up, across >= 1000 seeds spread over all four
tiers.

Generates through the skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""

from __future__ import annotations

import functools
import math
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

TOPIC = "p12b_annuity_perpetuity"
SKILL = "senior-with-the-aid-of-a-financial-calculator-or-acmgm100"

SEEDS = range(300)
TIERS = ("easy", "medium", "hard", "exam")


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _round_half_up(value: Fraction, places: int = 2) -> Fraction:
    scale = 10**places
    scaled = value * scale
    sign = -1 if scaled < 0 else 1
    magnitude = math.floor(abs(scaled) + Fraction(1, 2))
    return Fraction(sign * magnitude, scale)


def _annuity_pv(payment, i: Fraction, n: int) -> Fraction:
    growth = (1 + i) ** n
    return Fraction(payment) * (growth - 1) / (i * growth)


def _perpetuity_pv(payment, i: Fraction) -> Fraction:
    return Fraction(payment) / i


def _periodic_from_pv(pv, i: Fraction, n: int) -> Fraction:
    growth = (1 + i) ** n
    return Fraction(pv) * i * growth / (growth - 1)


def _num(token: str) -> Fraction:
    return Fraction(token.replace(",", ""))


# The generator draws one of three differently-worded templates per tier, all preserving the same
# ORDER of numeric anchors ($income, rate% per annum...); these patterns are deliberately loose
# (`.*?`, `re.DOTALL`) rather than anchored to one exact sentence, so they match every wording
# variant by construction order alone, never by a literal connecting phrase that could differ
# between templates.
_EASY_RE = re.compile(r"\$(\d+)[^$]*?([\d.]+)% per annum", re.DOTALL)
_MEDIUM_RE = re.compile(r"\$(\d+)[^$]*?(\d+) years[^$]*?([\d.]+)% per annum", re.DOTALL)
_HARD_RE = re.compile(r"\$([\d.]+)[^$]*?([\d.]+)% per annum[^$]*?\$(\d+)", re.DOTALL)
_EXAM_RE = re.compile(r"\$([\d,]+).*?([\d.]+)% per annum.*?(\d+) years.*?following (\d+) years", re.DOTALL)


def _oracle_easy(text: str) -> Fraction:
    m = _EASY_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable easy question: {text!r}")
    income, rate = m.groups()
    i = _num(rate) / 100 / 12
    return _round_half_up(_perpetuity_pv(_num(income), i), 2)


def _oracle_medium(text: str) -> Fraction:
    m = _MEDIUM_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable medium question: {text!r}")
    income, term_years, rate = m.groups()
    i = _num(rate) / 100 / 12
    n = int(term_years) * 12
    return _round_half_up(_annuity_pv(_num(income), i, n), 2)


def _oracle_hard(text: str) -> int:
    m = _HARD_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable hard question: {text!r}")
    pv, rate, income = m.groups()
    pv = _num(pv)
    i = _num(rate) / 100 / 12
    income = _num(income)
    # 1 - PV*i/A == (1+i)**-n -- solve for the smallest whole number of
    # years n whose PV(n), rounded to cents, equals the stated PV. Bounded
    # search (n_years up to 100) rather than a logarithm, matching the
    # discrete, whole-year nature of the constructed answer.
    for n_years in range(1, 101):
        candidate = _round_half_up(_annuity_pv(income, i, n_years * 12), 2)
        if candidate == pv:
            return n_years
    raise AssertionError(f"oracle could not find n_years matching PV={pv} for {text!r}")


def _oracle_exam(text: str) -> Fraction:
    m = _EXAM_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable exam question: {text!r}")
    principal, rate, m_years, n_years = m.groups()
    principal = _num(principal)
    i = _num(rate) / 100 / 12
    m_periods = int(m_years) * 12
    n_periods = int(n_years) * 12
    fv = Fraction(principal) * (1 + i) ** m_periods
    return _round_half_up(_periodic_from_pv(fv, i, n_periods), 2)


_ORACLES = {
    "easy": _oracle_easy,
    "medium": _oracle_medium,
    "hard": _oracle_hard,
    "exam": _oracle_exam,
}


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_recomputes_answer_from_question_text(tier):
    oracle_fn = _ORACLES[tier]
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(tier, seed)
        expected = oracle_fn(q.text)
        actual = int(q.answer_text) if tier == "hard" else _num(q.answer_text)
        assert expected == actual, (
            f"seed={seed} tier={tier}: oracle={expected} answer_text={q.answer_text!r} question={q.text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_answer_never_matches_a_distractor(tier):
    oracle_fn = _ORACLES[tier]
    for seed in SEEDS:
        q = _make_from_sources(tier, seed)
        expected = oracle_fn(q.text)
        for d in q.distractors:
            got = int(d) if tier == "hard" else _num(d)
            assert expected != got, (
                f"seed={seed} tier={tier}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )
