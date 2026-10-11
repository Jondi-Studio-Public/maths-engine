"""Tests for the exam-tier source `exd_x_g3_det3` of the skill `matrix-determinant-3x3`.

Exam tier: 3x3 determinants through the properties of det, or an unknown entry from the stated determinant.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen import skills as skills_module
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g3_det3"
SKILL_ID = "matrix-determinant-3x3"


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _skill():
    return next(s for s in skills_module.load_skills()[1] if s.id == SKILL_ID)


def _tuplify(params):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (params or {}).items()}


def _hard_shapes():
    shapes = set()
    for src in _skill().sources:
        if src.difficulty != "hard":
            continue
        for seed in range(60):
            shapes.add(prompt_shape(make(src.topic, seed=seed, difficulty="hard", **_tuplify(src.params))))
    return shapes


def test_exam_source_is_last_and_staged_to_exam_only():
    last = _skill().sources[-1]
    assert (last.topic, last.difficulty, list(last.stages)) == (TOPIC, "exam", ["exam"])
    assert dict(last.params) == {"tier": "exam"}


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam = {prompt_shape(_make(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - _hard_shapes(), "the exam tier adds no prompt shape beyond the hard tier"


def test_no_text_defects_and_deterministic():
    for seed in range(60):
        q = _make(seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == _make(seed).to_json()


def test_key_matches_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        expected = _oracle(q.text)
        got = q.answer if not isinstance(q.answer, dict) else (q.answer["x"], q.answer["y"])
        if expected != got:
            bad.append((seed, q.text, expected, got))
    assert bad == [], bad[:5]


def test_key_accepted_and_every_distractor_rejected_by_the_real_grader():
    grader = GRADERS[_skill().answer]
    for seed in range(120):
        q = _make(seed)
        assert grader(q, q.answer_text).correct, (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, q.text, q.answer_text, d)


def _mats(text):
    from sympy import Matrix, Symbol

    x = Symbol("x")
    out = []
    for body in re.findall(r"\[([^\]]+)\]", text):
        rows = [[(x if v == "x" else int(v)) for v in r.split()] for r in body.split(";")]
        out.append(Matrix(rows))
    return out


def _oracle(text):
    from sympy import Symbol, solve

    x = Symbol("x")
    mats = _mats(text)
    if "Find x" in text:
        D = int(re.search(r"has determinant (-?\d+)", text).group(1))
        sol = solve(mats[0].det() - D, x)
        assert len(sol) == 1
        return int(sol[0])
    k = int(re.search(r"det\((\d+)", text).group(1))
    A = mats[0]
    if "adj(A)" in text:
        val = (A.adjugate() * k).det()
    else:
        val = (A.T * k * mats[1].inv()).det()
    assert val == int(val)
    return int(val)


def test_all_forms_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("Find x" in t for t in texts)
    assert any("adj(A)" in t for t in texts)
    assert any("A^T B^(-1)" in t for t in texts)
