"""Exam tier of `f10-rational-four-operations`: a fraction, a percentage and a decimal share of a fundraiser's money."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_pr7_fundraiser_three_forms"
SKILL_ID = "f10-rational-four-operations"
HARD = ("rational_four_operations", {})
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _text(x: Fraction) -> str:
    cents = int(x * 100 + Fraction(1, 2)) if x >= 0 else -int(-x * 100 + Fraction(1, 2))
    sign = "-" if cents < 0 else ""
    return f"{sign}{abs(cents) // 100}.{abs(cents) % 100:02d}"


def facts(q):
    total = int(re.search(r"raised \$(\d+)", q.stem).group(1))
    num, den = (int(x) for x in re.search(r"spent (\d+)/(\d+) of", q.stem).groups())
    pct = int(re.search(r"spent (\d+)% of", q.stem).group(1))
    dec = Fraction(re.search(r"donated (0\.\d+) of", q.stem).group(1))
    return total, Fraction(num, den), Fraction(pct, 100), dec


def oracle(q):
    total, frac, pct, dec = facts(q)
    a = total * frac
    b = (total - a) * pct
    return {"a": _text(a), "b": _text(b), "c": _text(total - a - b - total * dec)}


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


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is None for p in q.parts] == [True, False, False]


def test_key_matches_independent_derivation(questions):
    bad = [(q.seed, oracle(q), _canon(q)) for q in questions if oracle(q) != _canon(q)]
    assert bad == []


def test_grader_accepts_key_and_rejects_distractors(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6
        for p in q.parts:
            assert len(p.distractors) >= 2
            for d in p.distractors:
                assert grade(_SKILL, q, {**_canon(q), p.label: d}).parts[p.label].marks == 0, (q.seed, p.label, d)


def test_follow_through_from_wrong_first_part(questions):
    for q in questions[:100]:
        total, _frac, pct, dec = facts(q)
        wrong = Fraction(q.parts[0].distractors[0])
        if wrong >= total:  # a spend above the total is outside the sensible range and carries nothing forward
            continue
        b = (total - wrong) * pct
        reply = {"a": q.parts[0].distractors[0], "b": _text(b), "c": _text(total - wrong - b - total * dec)}
        if len(set(reply.values())) < 3:  # a reply that only repeats an earlier value earns nothing
            continue
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through


def test_follow_through_from_wrong_second_part(questions):
    for q in questions[:100]:
        total, frac, _pct, dec = facts(q)
        wrong_b = Fraction(q.parts[1].distractors[0])
        c = _text(total - total * frac - wrong_b - total * dec)
        if c in (
            q.parts[0].answer_text,
            q.parts[1].distractors[0],
        ):  # a reply that only repeats an earlier value earns nothing
            continue
        reply = {**_canon(q), "b": q.parts[1].distractors[0], "c": c}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0 and result.parts["c"].marks == 2


def test_distractors_tagged(questions):
    for q in questions[:60]:
        assert all(set(p.distractor_values) == set(p.distractors) or len(p.distractor_values) >= 2 for p in q.parts)


def test_answer_space(questions):
    assert len({q.answer_text for q in questions}) >= 100
