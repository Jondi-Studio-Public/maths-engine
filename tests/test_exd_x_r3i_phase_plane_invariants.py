"""Exam source of `ucv1-phase-plane-linear-classification`: trace, determinant, discriminant, largest real part."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from sympy import Symbol, Poly
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    split_symbols,
    standard_transformations,
)

SKILL_ID = "ucv1-phase-plane-linear-classification"
TOPIC = "exd_x_r3i_phase_plane_invariants"
LABELS = ["a", "b", "c", "d"]
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


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert 3 <= len(q.parts) <= 4
        assert all(p.distractors and len(p.distractors) >= 2 for p in q.parts)
        assert sum(1 for p in q.parts if p.follow_through) >= 1


def test_prompt_variety_and_new_shapes():
    hard_src = next(s for s in _SKILL.sources if s.difficulty == "hard")
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(hard_src.topic, seed=s, **(hard_src.params or {}))) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


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


def test_answer_space(questions):
    tuples = Counter(tuple(_canonical(q).values()) for q in questions)
    assert len(tuples) >= 100, len(tuples)
    assert max(tuples.values()) / len(questions) <= 0.05


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        reply = {p.label: ("999999.5" if p.answer_kind == "number" else "999999/7") for p in q.parts}
        result = grade(_SKILL, q, reply)
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()


def _matrix(text):
    m = re.search(r"x' = (.*?), y' = (.*?) can be written", text)
    rows = []
    for side in m.groups():
        e = parse_expr(side, transformations=_TR, local_dict={"x": X, "y": Y})
        rows.append((int(e.coeff(X)), int(e.coeff(Y))))
    return rows[0][0], rows[0][1], rows[1][0], rows[1][1]


def oracle(q):
    a, b, c, d = _matrix(q.text)
    T, D = a + d, a * d - b * c
    disc = T * T - 4 * D
    top = Fraction(T + math.isqrt(disc), 2) if disc >= 0 else Fraction(T, 2)
    return {"a": str(T), "b": str(D), "c": str(disc), "d": f"{float(top):.1f}"}


def test_follow_through_from_wrong_trace(questions):
    for q in questions[:150]:
        a, b, c, d = _matrix(q.text)
        D = a * d - b * c
        wt = int(q.parts[0].distractors[0])
        if wt == 0:
            continue  # no follow-through from a reply of zero
        disc = wt * wt - 4 * D
        if disc == wt:
            continue  # a reply that merely repeats the earlier one earns no follow-through
        reply = {**_canonical(q), "a": str(wt), "c": str(disc)}
        if disc < 0:
            reply["d"] = f"{wt / 2:.1f}"
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        if disc != int(q.parts[2].answer_text) and disc != 0:
            assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
        if disc < 0 and int(q.parts[2].answer_text) < 0 and wt != 0:
            assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)
