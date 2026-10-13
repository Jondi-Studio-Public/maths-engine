"""Tests the multi-part exam tier of `logarithm-laws`: the product law, the quadratic, and the rejected root."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc5_log_equation"
SKILL_ID = "logarithm-laws"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
MARKS = [2, 3, 1]
FT_LABELS = [False, True, True]
TOP_SHARE = 0.05


def _exam(seed):
    return make(TOPIC, seed=seed)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _frac(text):
    return Fraction(text)


def _setup(q):
    b, p, qq, n = (
        int(v) for v in re.search(r"log_\{(\d+)\}\(x \+ (\d+)\) \+ log_\{\d+\}\(x - (\d+)\) = (\d+)", q.stem).groups()
    )
    return b, p, qq, n


def oracle(q):
    b, p, qq, n = _setup(q)
    x = sympy.Symbol("x")
    roots = sympy.solve(sympy.Eq(sympy.expand((x + p) * (x - qq)), b**n), x)
    valid = [r for r in roots if r + p > 0 and r - qq > 0]
    assert len(valid) == 1
    x0 = valid[0]
    return {"a": str(b**n), "b": str(x0), "c": str(sympy.log(x0 + p, b).simplify())}


def follow_through_cases(q):
    b, p, qq, n = _setup(q)
    x0 = int(q.parts[1].answer_text)
    x1 = x0 + 1  # a root of the quadratic for N = (x1 + p)(x1 - q)
    cases = [({"a": str((x1 + p) * (x1 - qq)), "b": str(x1)}, "b", "a")]
    # a wrong x whose x + p is a power of b one step up still gives an exact logarithm
    u = int(q.parts[2].answer_text)
    x2 = b ** (u + 1) - p
    cases.append(({"a": q.parts[0].answer_text, "b": str(x2), "c": str(u + 1)}, "c", "b"))
    return cases


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_shape(questions):
    for q in questions[:100]:
        assert [p.label for p in q.parts] == list("abcd")[: len(MARKS)]
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_LABELS
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("logarithm_law", seed=s)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt(questions):
    for q in questions:
        got = oracle(q)
        want = _canonical(q)
        assert set(got) == set(want)
        for label in want:
            assert _frac(got[label]) == _frac(want[label]), (q.text, label, got[label], want[label])


def test_composed_answer_full_marks(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.answer_text


def test_distractors_rejected(questions):
    shown = 0
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            assert len(p.distractors) == len(set(p.distractors))
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)
    assert shown >= 600


def test_distractors_are_tagged(questions):
    for q in questions[:100]:
        for p in q.parts:
            assert p.distractor_values, (q.text, p.label)
            assert set(p.option_notes) <= set(p.distractors)


def test_follow_through(questions):
    checked = 0
    for q in questions[:100]:
        for reply, earns, loses in follow_through_cases(q):
            result = grade(_SKILL, q, reply)
            for label in earns:
                part = result.parts[label]
                assert part.marks == part.max_marks, (q.text, reply, label)
                assert part.follow_through is True, (q.text, reply, label)
            for label in loses:
                assert result.parts[label].marks == 0, (q.text, reply, label)
            assert not result.correct
            checked += 1
    assert checked >= 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "987654", "b": "7654321", "c": "-31", "d": "-9"})
        assert result.marks == 0 and not result.correct


def test_answer_space(questions):
    answers = [q.answer_text for q in questions]
    distinct = set(answers)
    assert len(distinct) >= 100, len(distinct)
    top = max(answers.count(a) for a in distinct)
    assert top / len(answers) <= TOP_SHARE, top


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
