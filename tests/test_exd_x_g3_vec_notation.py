"""Tests for the exam-tier source `exd_x_g3_vec_notation` of the skill `senior-use-ordered-pair-notation-and-column-vector-notation`.

Exam tier: chained vector statements (two vectors, a midpoint, a parallelogram) to a coordinate.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen import skills as skills_module
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g3_vec_notation"
SKILL_ID = "senior-use-ordered-pair-notation-and-column-vector-notation"


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
    # The skill's exam source is now the multi-part chain; this module's generator stays registered on its own.
    assert (last.topic, last.difficulty, list(last.stages)) == ("exd_x_r3e_vector_points", "exam", ["exam"])
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
    axis = 0 if "x-coordinate" in text else 1
    if text.startswith("ABCD"):
        a, b, c = [tuple(int(v) for v in m) for m in re.findall(r"[ABC]\((-?\d+), (-?\d+)\)", text)]
        # AB = DC as vectors, so D = A + C - B
        return a[axis] + c[axis] - b[axis]
    a = tuple(int(v) for v in re.search(r"Point A is \((-?\d+), (-?\d+)\)", text).groups())
    cols = [tuple(int(v) for v in m) for m in re.findall(r"column vector \((-?\d+), (-?\d+)\)", text)]
    if "midpoint" in text:
        # M = A + AM and M = (A + B)/2, so B = 2M - A = A + 2 AM
        return a[axis] + 2 * cols[0][axis]
    ab, cb = cols
    # B = A + AB and B = C + CB, so C = A + AB - CB
    return a[axis] + ab[axis] - cb[axis]


def test_all_three_forms_occur():
    texts = [_make(s).text for s in range(60)]
    assert any(t.startswith("ABCD") for t in texts)
    assert any("midpoint" in t for t in texts)
    assert any("from C to B" in t for t in texts)
