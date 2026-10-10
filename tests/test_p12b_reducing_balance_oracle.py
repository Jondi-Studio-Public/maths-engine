"""Independent oracle for `p12b_reducing_balance_loan` (ACMGM098).

Independent oracle for ACMGM098.

``senior-with-the-aid-of-a-financial-calculator-or`` (topic
``p12b_reducing_balance_loan``, ``answer: number``). Parses ONLY the
rendered ``question.text`` -- never importing anything from
``mathgen.generators.p12b_annuity_loan_build`` -- and re-derives the
repayment, balance or maximum-principal value independently, via its own
local (re-implemented, not imported) exact-`Fraction` amortisation formula
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

TOPIC = "p12b_reducing_balance_loan"
SKILL = "senior-with-the-aid-of-a-financial-calculator-or"

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


def _repayment(principal, i: Fraction, n: int) -> Fraction:
    growth = (1 + i) ** n
    return Fraction(principal) * i * growth / (growth - 1)


def _balance(principal, i: Fraction, repayment, k: int) -> Fraction:
    growth = (1 + i) ** k
    return Fraction(principal) * growth - Fraction(repayment) * (growth - 1) / i


def _max_principal(repayment, i: Fraction, n: int) -> Fraction:
    growth = (1 + i) ** n
    return Fraction(repayment) * (growth - 1) / (i * growth)


def _num(token: str) -> Fraction:
    return Fraction(token.replace(",", ""))


# The generator draws one of three differently-worded templates per tier, all preserving the same
# ORDER of numeric anchors ($principal, rate% per annum, N years/N-year...); these patterns are
# deliberately loose (`.*?`, `re.DOTALL`) rather than anchored to one exact sentence, so they match
# every wording variant by construction order alone, never by a literal connecting phrase that could
# differ between templates.
_EASY_RE = re.compile(r"\$([\d,]+).*?([\d.]+)% per annum.*?(\d+) years", re.DOTALL)
_MEDIUM_RE = re.compile(
    r"\$([\d,]+).*?([\d.]+)% per annum.*?\$([\d.]+) over (\d+) years.*?after (\d+) year",
    re.DOTALL,
)
_HARD_RE = re.compile(r"\$(\d+).*?([\d.]+)% per annum.*?(\d+)-year", re.DOTALL)
# The exam tier's third wording variant states the rate BEFORE the
# principal ("...at the same nominal annual rate of R% per annum, for a
# $P home loan..."), unlike every other tier/variant -- so this one uses
# three independent field searches rather than one ordered pattern.
_EXAM_PRINCIPAL_RE = re.compile(r"\$([\d,]+)")
_EXAM_RATE_RE = re.compile(r"([\d.]+)% per annum")
_EXAM_TERM_RE = re.compile(r"(\d+) years")


def _oracle_easy(text: str) -> Fraction:
    m = _EASY_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable easy question: {text!r}")
    principal, rate, term_years = m.groups()
    principal = _num(principal)
    i = _num(rate) / 100 / 12
    n = int(term_years) * 12
    return _round_half_up(_repayment(principal, i, n), 2)


def _oracle_medium(text: str) -> Fraction:
    m = _MEDIUM_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable medium question: {text!r}")
    principal, rate, repayment, term_years, k_years = m.groups()
    principal = _num(principal)
    i = _num(rate) / 100 / 12
    repayment = _num(repayment)
    k = int(k_years) * 12
    return _round_half_up(_balance(principal, i, repayment, k), 2)


def _oracle_hard(text: str) -> Fraction:
    m = _HARD_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable hard question: {text!r}")
    repayment, rate, term_years = m.groups()
    repayment = _num(repayment)
    i = _num(rate) / 100 / 12
    n = int(term_years) * 12
    return _round_half_up(_max_principal(repayment, i, n), 2)


def _oracle_exam(text: str) -> Fraction:
    p_m = _EXAM_PRINCIPAL_RE.search(text)
    r_m = _EXAM_RATE_RE.search(text)
    t_m = _EXAM_TERM_RE.search(text)
    if not (p_m and r_m and t_m):
        raise AssertionError(f"unparseable exam question: {text!r}")
    principal = _num(p_m.group(1))
    i_annual_tenths_rate = _num(r_m.group(1))
    term_years = int(t_m.group(1))
    i_m = i_annual_tenths_rate / 100 / 12
    n_m = term_years * 12
    i_f = i_annual_tenths_rate / 100 / 26
    n_f = term_years * 26
    # Repayments are rounded to the nearest cent BEFORE totalling (the
    # question states this explicitly) -- matches the generator's own
    # construction, which uses the same cents-rounded repayment for the
    # totals so a learner reading the stated repayment off a financial
    # calculator and multiplying by the period count reaches this value.
    repay_m = _round_half_up(_repayment(principal, i_m, n_m), 2)
    repay_f = _round_half_up(_repayment(principal, i_f, n_f), 2)
    total_m = repay_m * n_m
    total_f = repay_f * n_f
    return _round_half_up(total_f - total_m, 2)


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
        actual = _num(q.answer_text)
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
            assert expected != _num(d), (
                f"seed={seed} tier={tier}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )
