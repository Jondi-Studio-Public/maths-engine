"""Tests for the exam-tier source `exd_x_g3_vec_component` of the skill `senior-express-a-vector-in-component-form-using-the`.

Exam tier: components of a vector given its magnitude and a direction vector.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen import skills as skills_module
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g3_vec_component"
SKILL_ID = "senior-express-a-vector-in-component-form-using-the"


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
    m = re.search(r"(-?\d*)i ([+-]) (\d*)j", text)
    p = int(m.group(1)) if m.group(1) not in ("", "-") else (-1 if m.group(1) == "-" else 1)
    q = int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
    mag = int(re.search(r"magnitude (\d+)", text).group(1))
    sign = -1 if "opposite" in text else 1
    want = re.search(r"coefficient of ([ij])", text).group(1)
    from sympy import Integer, sqrt, nsimplify

    norm = sqrt(p * p + q * q)
    comp = Integer(sign * mag) * (p if want == "i" else q) / norm
    assert comp == int(comp)
    return int(comp)
