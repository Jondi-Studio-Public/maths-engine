"""Tests for the multi-part exam source of `senior-with-the-aid-of-a-financial-calculator-or-acmgm100`
(`exd_x_r2e_annuity_chain`): the balance of a savings annuity, the monthly withdrawal it supports, the total
withdrawn and the interest earned while drawing down."""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2e_annuity_chain"
SKILL_ID = "senior-with-the-aid-of-a-financial-calculator-or-acmgm100"
HARD = ("p12b_annuity_perpetuity", {})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _half_up(x: Fraction) -> int:
    return int(x + Fraction(1, 2)) if x >= 0 else -int(-x + Fraction(1, 2))


def _facts(q):
    deposit = int(re.search(r"pays \$([\d,]+) at the end", q.stem).group(1).replace(",", ""))
    years_in = int(re.search(r"for (\d+) years\. The account", q.stem).group(1))
    rate = Fraction(re.search(r"earns ([\d.]+)% p\.a\.", q.stem).group(1))
    years_out = int(re.search(r"for (\d+) years, so that", q.stem).group(1))
    return deposit, 12 * years_in, rate / 1200, 12 * years_out


def _withdrawal(balance: int, i: Fraction, n_out: int) -> int:
    return _half_up(balance * i / (1 - (1 + i) ** (-n_out)))


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem alone, in exact arithmetic."""
    deposit, n_in, i, n_out = _facts(q)
    a = _half_up(deposit * ((1 + i) ** n_in - 1) / i)
    b = _withdrawal(a, i, n_out)
    c = b * n_out
    return {"a": str(a), "b": str(b), "c": str(c), "d": str(c - a)}


def test_wired_to_the_skill():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.params, s.stages) for s in exam] == [(TOPIC, {"tier": "exam"}, ["exam"])]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 3, 1, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_add_to_the_hard_tier():
    shapes = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty="hard", **HARD[1])) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_distractors_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (p.label, d)
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_a_wrong_balance(questions):
    """A wrong (a) carried through (b), (c) and (d) earns all three of them."""
    for q in questions[:120]:
        _deposit, _n_in, i, n_out = _facts(q)
        a_wrong = int(q.parts[0].answer_text) + 1500
        b_wrong = _withdrawal(a_wrong, i, n_out)
        c_wrong = b_wrong * n_out
        reply = {"a": str(a_wrong), "b": str(b_wrong), "c": str(c_wrong), "d": str(c_wrong - a_wrong)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label)
            assert result.parts[label].follow_through is True
        assert result.marks == 5 and not result.correct


def test_follow_through_from_a_wrong_total(questions):
    for q in questions[:60]:
        canon = _canonical(q)
        c_wrong = int(canon["c"]) + 700
        result = grade(_SKILL, q, {**canon, "c": str(c_wrong), "d": str(c_wrong - int(canon["a"]))})
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 1 and result.parts["d"].follow_through


def test_all_wrong_earns_zero(questions):
    for q in questions[:50]:
        result = grade(_SKILL, q, {"a": "1", "b": "7", "c": "9", "d": "99"})
        assert result.marks == 0 and not result.correct
