"""Exam source of `senior-angles-at-the-circumference-of-a-circle-subtended`: a four-part crossing-chords chain."""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_chords_chain"
SKILL_ID = "senior-angles-at-the-circumference-of-a-circle-subtended"
SEEDS = range(300)
X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_STEM = re.compile(r"Angle BAC = \((.*?)\)°, angle BDC = \((.*?)\)° and angle ACD = \((.*?)\)°")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _expr(text):
    return parse_expr(text, transformations=_TRANSFORMS)


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the rendered stem alone."""
    bac, bdc, acd = (_expr(t) for t in _STEM.search(q.stem).groups())
    (x,) = sympy.solve(sympy.Eq(bac, bdc), X)  # same arc BC
    abd = acd.subs(X, x)  # same arc AD
    cpd = 180 - bac.subs(X, x) - abd  # triangle ABP, then vertically opposite
    return {"a": str(x), "b": str(bac.subs(X, x)), "c": str(abd), "d": str(cpd)}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert all(len(p.distractors) >= 2 and p.option_notes for p in q.parts)


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_x(questions):
    checked = 0
    for q in questions[:100]:
        bac, _bdc, acd = (_expr(t) for t in _STEM.search(q.stem).groups())
        x_true = int(q.parts[0].answer_text)
        for k in (1, -1, 2):
            x = x_true + k
            b, c = int(bac.subs(X, x)), int(acd.subs(X, x))
            d = 180 - b - c
            # The engine withholds follow-through from a reply that merely repeats an earlier one, so skip those.
            if not (0 < b < 180 and 0 < c < 180 and 0 < d < 180) or x <= 0 or d in (b, c):
                continue
            result = grade(_SKILL, q, {"a": str(x), "b": str(b), "c": str(c), "d": str(d)})
            assert result.parts["a"].marks == 0
            for label in "bcd":
                assert result.parts[label].marks == result.parts[label].max_marks, (q.text, x, label)
                assert result.parts[label].follow_through is True
            assert result.marks == 5 and not result.correct
            checked += 1
            break
    assert checked >= 60


def test_follow_through_into_last_part_only(questions):
    for q in questions[:80]:
        canon = _canonical(q)
        wrong_c = int(canon["c"]) + 3
        d = 180 - int(canon["b"]) - wrong_c
        if d <= 0 or d in (wrong_c, int(canon["b"])):
            continue
        result = grade(_SKILL, q, {**canon, "c": str(wrong_c), "d": str(d)})
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 2, (q.text, wrong_c)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "99", "b": "1", "c": "2", "d": "3"})
        assert result.marks == 0 and not result.correct
