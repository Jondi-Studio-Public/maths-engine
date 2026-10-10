"""Exam-tier tests for `exd_x_s2_traffic` (conditional probability about traffic lights).

The oracle parses the rendered prompt and enumerates the light outcomes with exact fractions; it shares nothing
with the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s2_traffic"
_F = r"(\d+(?:/\d+)?)"


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(text):
    if text.startswith("The first of two"):
        p1, p2, p3 = (Fraction(x) for x in re.findall(r"probability " + _F, text))
        joint = {
            (1, 1): p1 * p2,
            (1, 0): p1 * (1 - p2),
            (0, 1): (1 - p1) * p3,
            (0, 0): (1 - p1) * (1 - p3),
        }
        given = [k for k in joint if k[1] == 1]
        want = [(1, 1)]
    else:
        p1, p2 = (Fraction(x) for x in re.findall(r"probability " + _F, text))
        joint = {(i, j): (p1 if i else 1 - p1) * (p2 if j else 1 - p2) for i in (0, 1) for j in (0, 1)}
        if "exactly one" in text:
            given = [k for k in joint if sum(k) == 1]
            want = [(1, 0)]
        else:
            assert "at least one" in text, text
            given = [k for k in joint if sum(k) >= 1]
            want = [(1, 1)]
    return sum(joint[k] for k in want) / sum(joint[k] for k in given)


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == []


def test_shape_variety_and_novelty_over_hard():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("traffic_light_probability", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_deterministic():
    for seed in (0, 1, 5, 123):
        assert _q(seed).to_json() == _q(seed).to_json()


def test_oracle_matches_key_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        got = _oracle(q.text)
        if got != Fraction(q.answer["p"], q.answer["q"]):
            bad.append((seed, q.text, got, q.answer_text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["fraction"]
    for seed in range(150):
        q = _q(seed)
        assert grade(q, q.answer_text).correct, seed
        assert len(q.distractors) >= 2, (seed, q.distractors)
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)
