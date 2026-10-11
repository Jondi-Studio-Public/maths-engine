"""Tests for the exam-tier source `exd_x_g3_vec_curves` of the skill `senior-use-vector-equations-of-curves-in-two-or`.

Exam tier: recover unknown constants of r(t) = (a t + p, b/t + c) from points on the curve.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen import skills as skills_module
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g3_vec_curves"
SKILL_ID = "senior-use-vector-equations-of-curves-in-two-or"


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


def _oracle(text):
    from sympy import Symbol, solve

    b, c, t = Symbol("b"), Symbol("c"), Symbol("t")
    m = re.search(r"r\(t\) = \((-?\d+)t(?: ([+-]) (\d+))?, (b/t|b/t [+-] \d+|b/t \+ c)\)", text)
    a = int(m.group(1))
    p = 0 if m.group(2) is None else int(m.group(3)) * (1 if m.group(2) == "+" else -1)
    ytxt = m.group(4)
    pts = [tuple(int(v) for v in q) for q in re.findall(r"\((-?\d+), (-?\d+)\)", text.split("constant")[1])]
    asked = re.search(r"Find ([bc])\.", text).group(1)
    eqs = []
    for x, y in pts:
        tv = Fraction(x - p, a)
        yexpr = b / Fraction(tv.numerator, tv.denominator) + (
            c if ytxt.endswith("c") else int(ytxt.split()[-1]) * (1 if "+" in ytxt else -1)
        )
        eqs.append(yexpr - y)
    sol = solve(eqs, [b, c], dict=True)[0] if len(eqs) == 2 else solve(eqs, [b], dict=True)[0]
    return int(sol[b if asked == "b" else c])


def test_both_forms_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("passes through the point (" in t for t in texts)
    assert any("passes through the points (" in t for t in texts)
