"""Exam source of `ucv1-cr-equations-check`: the two Cauchy-Riemann discrepancies at a point, and their total."""

from __future__ import annotations

import re

from sympy import Symbol, diff
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    split_symbols,
    standard_transformations,
)

SKILL_ID = "ucv1-cr-equations-check"
TOPIC = "exd_x_r3i_cr_defects"
LABELS = ["a", "b", "c"]
X, Y = Symbol("x"), Symbol("y")
_TR = standard_transformations + (split_symbols, implicit_multiplication, convert_xor)

from collections import Counter

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
SEEDS = range(300)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert 3 <= len(q.parts) <= 4
        assert all(p.distractors and len(p.distractors) >= 2 for p in q.parts)
        assert sum(1 for p in q.parts if p.follow_through) >= 1


def test_no_text_defects(questions):
    for q in questions[:100]:
        assert find_text_defects(q, include_working=True) == [], q.text


def test_key_rederived_from_prompt(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: str(d)})
                assert result.parts[p.label].marks == 0, (q.text, p.label, str(d))


def test_distractors_match_key_layout(questions):
    for q in questions:
        for p in q.parts:
            if p.answer_kind != "number":
                continue
            for d in p.distractors:
                assert ("." in d) == ("." in p.answer_text), (p.label, d, p.answer_text)
                if "." in d:
                    assert len(d.split(".")[1]) == len(p.answer_text.split(".")[1]), (p.label, d, p.answer_text)


def test_distractors_are_tagged(questions):
    from mathgen.misconceptions import CATALOGUE

    for q in questions[:100]:
        for p in q.parts:
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (q.text, p.label, d)


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        reply = {p.label: ("999999.5" if p.answer_kind == "number" else "999999/7") for p in q.parts}
        result = grade(_SKILL, q, reply)
        assert result.marks == 0 and not result.correct


def _discrepancies(text):
    u_text, v_text = re.search(r"u\(x, y\) = (.*?) and v\(x, y\) = (.*?)\. ", text).groups()
    x0, y0 = (int(v) for v in re.search(r"\(x, y\) = \((-?\d+), (-?\d+)\)", text).groups())
    u = parse_expr(u_text, transformations=_TR, local_dict={"x": X, "y": Y})
    v = parse_expr(v_text, transformations=_TR, local_dict={"x": X, "y": Y})
    at = {X: x0, Y: y0}
    return int((diff(u, X) - diff(v, Y)).subs(at)), int((diff(u, Y) + diff(v, X)).subs(at))


def oracle(q):
    d1, d2 = _discrepancies(q.text)
    return {"a": str(d1), "b": str(d2), "c": str(abs(d1) + abs(d2))}


def test_all_three_failing_cases_occur(questions):
    cases = {(int(q.parts[0].answer_text) != 0, int(q.parts[1].answer_text) != 0) for q in questions}
    # (False, False) is never drawn: three zero keys would let a reply of 0 throughout earn full marks.
    assert cases == {(True, False), (False, True), (True, True)}


def test_follow_through_from_wrong_discrepancies(questions):
    for q in questions[:150]:
        wa, wb = int(q.parts[0].distractors[0]), int(q.parts[1].distractors[0])
        total = abs(wa) + abs(wb)
        reply = {"a": str(wa), "b": str(wb), "c": str(total)}
        result = grade(_SKILL, q, reply)
        if wa != 0 and wb != 0:
            assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
