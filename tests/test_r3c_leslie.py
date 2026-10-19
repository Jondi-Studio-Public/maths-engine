"""Tests for `leslie-population-matrix`: age-class populations, with the rates read back from the wording."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.generators.r3c_leslie_build import _SPECIES
from tests.r3c_support import (
    TIERS,
    check_follow_through,
    check_question_grading,
    check_tagged,
    distinct_texts,
    questions,
    top_share,
)

TOPIC = "r3c_leslie"
CHAIN = "r3c_leslie_chain"


def _read(text):
    """(f, s, P): birth rates, survival shares and the counts, read from the wording."""
    sp = next(s for s in _SPECIES if s["who"] in text)
    names = sp["short"]
    f = []
    for name in names:
        m = re.search(rf"each of the {re.escape(name)} produces ([\d.]+) ", text)
        f.append(Fraction(m.group(1)) if m else Fraction(0))
    s = [Fraction(int(x), 100) for x in re.findall(r"(\d+)% survive", text)]
    pop = None
    m = re.search(r"This year there are (\d+) .*?, (\d+) .*? and (\d+) ", text)
    if m:
        pop = [Fraction(int(g)) for g in m.groups()]
    return sp, f, s, pop


def _next(f, s, P):
    return [sum(f[k] * P[k] for k in range(3)), s[0] * P[0], s[1] * P[1]]


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 40


def test_easy_total_next_year():
    for q in questions(TOPIC, "easy", range(1, 41)):
        assert q.answer_kind == "number"
        _sp, f, s, P = _read(q.text)
        assert int(q.answer_text) == sum(_next(f, s, P))


def test_medium_total_in_two_years():
    for q in questions(TOPIC, "medium", range(1, 41)):
        assert q.answer_kind == "number"
        _sp, f, s, P = _read(q.text)
        P2 = _next(f, s, _next(f, s, P))
        assert int(q.answer_text) == sum(P2)
        assert Fraction(sum(P2)).denominator == 1


def test_hard_pattern_is_kept_and_factor_is_the_eigenvalue():
    seen = set()
    for q in questions(TOPIC, "hard", range(1, 41)):
        _sp, f, s, _P = _read(q.text)
        v = [Fraction(int(x)) for x in re.search(r"in the ratio (\d+) : (\d+) : (\d+)", q.text).groups()]
        nxt = _next(f, s, v)
        factor = nxt[0] / v[0]
        assert nxt == [factor * x for x in v]
        assert Fraction(q.answer_text) == factor
        seen.add(q.answer_text)
    assert len(seen) >= 4


def test_exam_chain_recomputed_and_followed_through():
    honoured = 0
    for q in questions(CHAIN, "exam", range(1, 41)):
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "matrix"]
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        _sp, f, s, P = _read(q.text)
        P1 = _next(f, s, P)
        P2 = _next(f, s, P1)
        assert int(q.parts[0].answer_text) == P1[0]
        assert int(q.parts[1].answer_text) == sum(P1)
        change = (sum(P1) - sum(P)) / sum(P) * 100
        assert abs(Fraction(q.parts[2].answer_text) - abs(change)) <= Fraction(1, 20)
        assert [row[0] for row in q.parts[3].answer] == [int(v) for v in P2]
    assert honoured > 20
