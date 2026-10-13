"""Exam tier of `f10-rational-operation-strategy`: fees, a fractional pay-off and a deposit on an overdrawn account."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_pr7_overdrawn_account"
SKILL_ID = "f10-rational-operation-strategy"
HARD = ("rational_operation_strategy", {})
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _text(x: Fraction) -> str:
    cents = int(x * 100 + Fraction(1, 2)) if x >= 0 else -int(-x * 100 + Fraction(1, 2))
    sign = "-" if cents < 0 else ""
    return f"{sign}{abs(cents) // 100}.{abs(cents) % 100:02d}"


def facts(q):
    opening = Fraction(re.search(r"balance is -\$(\d+\.\d\d)\.", q.stem).group(1))
    fee = Fraction(re.search(r"fee of \$(\d+\.\d\d)", q.stem).group(1))
    months = int(re.search(r"next (\d+) months", q.stem).group(1))
    p, qq = (int(x) for x in re.search(r"pays off (\d+)/(\d+) of", q.stem).groups())
    target = int(re.search(r"balance is \$(\d+)\?", q.text).group(1))
    return opening, fee, months, Fraction(p, qq), target


def oracle(q):
    opening, fee, months, paid, target = facts(q)
    after_fees = -opening - months * fee
    after_pay = after_fees * (1 - paid)
    return {"a": _text(after_fees), "b": _text(after_pay), "c": _text(target - after_pay)}


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in range(300)]


def _canon(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC] and exam[0].stages == ["exam"]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, **HARD[1])) for s in range(60)}
    assert len(exam) >= 3 and exam - hard


def test_key_matches_independent_derivation(questions):
    bad = [(q.seed, oracle(q), _canon(q)) for q in questions if oracle(q) != _canon(q)]
    assert bad == []


def test_balances_are_negative_and_deposit_positive(questions):
    for q in questions:
        a, b, c = (Fraction(p.answer_text) for p in q.parts)
        assert a < b < 0 < c


def test_grader_accepts_key_and_rejects_distractors(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5
        for p in q.parts:
            assert len(p.distractors) >= 2
            for d in p.distractors:
                assert grade(_SKILL, q, {**_canon(q), p.label: d}).parts[p.label].marks == 0, (q.seed, p.label, d)


def test_follow_through_from_wrong_fees(questions):
    for q in questions[:100]:
        _o, _f, _m, paid, target = facts(q)
        wrong = Fraction(q.parts[0].distractors[0])
        if wrong >= 0:  # a positive balance is outside the sensible range, so it carries nothing forward
            result = grade(
                _SKILL,
                q,
                {
                    "a": q.parts[0].distractors[0],
                    "b": _text(wrong * (1 - paid)),
                    "c": _text(target - wrong * (1 - paid)),
                },
            )
            assert result.parts["b"].marks == 0 and result.parts["c"].marks == 0
            continue
        b = wrong * (1 - paid)
        result = grade(_SKILL, q, {"a": q.parts[0].distractors[0], "b": _text(b), "c": _text(target - b)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through


def test_follow_through_from_wrong_payoff(questions):
    for q in questions[:100]:
        _o, _f, _m, _paid, target = facts(q)
        wrong = Fraction(q.parts[1].distractors[0])
        if wrong >= 0:
            continue
        result = grade(_SKILL, q, {**_canon(q), "b": q.parts[1].distractors[0], "c": _text(target - wrong)})
        assert result.parts["b"].marks == 0 and result.parts["c"].marks == 1


def test_answer_space(questions):
    assert len({q.answer_text for q in questions}) >= 100
