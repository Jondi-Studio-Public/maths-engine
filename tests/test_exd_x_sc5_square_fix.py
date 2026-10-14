"""Tests the multi-part exam tier of `f10-square-roots-perfect-squares`: what makes a number a perfect square, the square, and its root."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import math
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc5_square_fix"
SKILL_ID = "f10-square-roots-perfect-squares"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
MARKS = [2, 1, 1]
FT_LABELS = [False, True, True]
TOP_SHARE = 0.05


def _exam(seed):
    return make(TOPIC, seed=seed)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _frac(text):
    return Fraction(text)


def _case(q):
    if "is not a perfect square" in q.stem:
        return "multiply", int(re.search(r"number (\d+) is", q.stem).group(1))
    n = int(re.match(r"(?:A tiler has )?(\d+) ", q.stem).group(1))
    return ("add" if "students" in q.stem else "remove"), n


def oracle(q):
    case, n = _case(q)
    if case == "multiply":
        k = next(k for k in range(1, n + 1) if math.isqrt(n * k) ** 2 == n * k)
        sq = n * k
    elif case == "add":
        m = math.isqrt(n) + (0 if math.isqrt(n) ** 2 == n else 1)
        k, sq = m * m - n, m * m
    else:
        m = math.isqrt(n)
        k, sq = n - m * m, m * m
    return {"a": str(k), "b": str(sq), "c": str(math.isqrt(sq))}


def follow_through_cases(q):
    case, n = _case(q)
    root = int(q.parts[2].answer_text)
    if case == "multiply":
        k = int(q.parts[0].answer_text) * 4  # still makes a square, but not the smallest
        sq = n * k
    elif case == "add":
        k = (root + 1) ** 2 - n  # the next square up
        sq = (root + 1) ** 2
    else:
        k = n - (root - 1) ** 2  # the next square down
        sq = (root - 1) ** 2
    return [({"a": str(k), "b": str(sq), "c": str(math.isqrt(sq))}, "bc", "a")]


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def test_shape(questions):
    for q in questions[:100]:
        assert [p.label for p in q.parts] == list("abcd")[: len(MARKS)]
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_LABELS
        assert all(p.answer_kind == "number" for p in q.parts)


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
