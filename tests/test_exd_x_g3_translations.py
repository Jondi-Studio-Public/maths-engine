"""Tests for the exam-tier source `exd_x_g3_translations` of the skill `senior-translations-and-their-representation-as-column-vectors`.

Exam tier: recover a translation from a point and its image, then use it in a composite.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen import skills as skills_module
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g3_translations"
SKILL_ID = "senior-translations-and-their-representation-as-column-vectors"


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


def _points(text):
    return {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"([A-Z]'*)\((-?\d+), (-?\d+)\)", text)}


def _oracle(text):
    pts = _points(text)
    a, a1, b = pts["A"], pts["A'"], pts["B"]
    t = (a1[0] - a[0], a1[1] - a[1])
    if "times in succession" in text:
        n = int(re.search(r"T (\d+) times", text).group(1))
        return (float(b[0] + n * t[0]), float(b[1] + n * t[1]))
    if "inverse of T" in text:
        u = tuple(int(v) for v in re.search(r"column vector \((-?\d+), (-?\d+)\)", text).groups())
        return (float(b[0] - t[0] + u[0]), float(b[1] - t[1] + u[1]))
    a2 = pts["A''"]
    return (float(b[0] + a2[0] - a[0]), float(b[1] + a2[1] - a[1]))


def test_all_three_forms_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("times in succession" in t for t in texts)
    assert any("inverse of T" in t for t in texts)
    assert any("translation S" in t for t in texts)
