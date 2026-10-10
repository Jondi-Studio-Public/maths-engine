"""Exam source for `finite-joint-covariance`: covariance of linearly transformed variables."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape


TOPIC = "exd_x_s1_covariance"
_CELL = re.compile(r"X=(-?\d+), Y=(-?\d+): (\d+)")
_FRAC = r"(\d+(?:/\d+)?)"
_EVENTS = re.compile(rf"P\(A\) = {_FRAC}, P\(B\) = {_FRAC} and P\(A and B\) = {_FRAC}\.")
_COV = re.compile(r"Cov\((-?\d+)X ([+-]) (\d+), (-?\d+)Y ([+-]) (\d+)\)")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _cov(cells):
    eu = sum(u * w for u, v, w in cells)
    ev = sum(v * w for u, v, w in cells)
    return sum((u - eu) * (v - ev) * w for u, v, w in cells)


def _oracle(text):
    a, s1, b, c, s2, d = _COV.search(text).groups()
    a, c = int(a), int(c)
    b = int(b) if s1 == "+" else -int(b)
    d = int(d) if s2 == "+" else -int(d)
    cells = _CELL.findall(text)
    if cells:
        total = sum(int(n) for _, _, n in cells)
        assert f"One of {total} " in text
        joint = [(Fraction(int(x)), Fraction(int(y)), Fraction(int(n), total)) for x, y, n in cells]
    else:
        pa, pb, pab = (Fraction(v) for v in _EVENTS.search(text).groups())
        joint = [
            (Fraction(1), Fraction(1), pab),
            (Fraction(1), Fraction(0), pa - pab),
            (Fraction(0), Fraction(1), pb - pab),
            (Fraction(0), Fraction(0), 1 - pa - pb + pab),
        ]
    return _cov([(a * x + b, c * y + d, w) for x, y, w in joint])


def test_shapes_over_60_seeds_and_new_over_hard():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {
        prompt_shape(make("finite_joint_covariance", seed=s, difficulty="hard", kind="missing_weight"))
        for s in range(60)
    }
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for s in range(60):
        assert find_text_defects(_q(s)) == []


def test_both_forms_occur():
    forms = {"One of" in _q(s).text for s in range(40)}
    assert forms == {True, False}


def test_independent_oracle_agrees_over_300_seeds():
    bad = []
    for s in range(300):
        q = _q(s)
        if _oracle(q.text) != Fraction(q.answer["p"], q.answer["q"]):
            bad.append((s, q.text, q.answer_text))
    assert bad == []


def test_grader_accepts_key_and_rejects_every_distractor():
    g = GRADERS["fraction"]
    bad = []
    for s in range(150):
        q = _q(s)
        if not g(q, q.answer_text).correct:
            bad.append(("key", s))
        for d in q.distractors:
            if g(q, d).correct:
                bad.append(("distractor", s, d))
        assert len(q.distractors) >= 2
    assert bad == []
