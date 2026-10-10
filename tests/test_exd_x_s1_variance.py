"""Exam source for `finite-distribution-variance`: variance of a transformed or proportionally specified pmf."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape


TOPIC = "exd_x_s1_variance"
_PMF = re.compile(r"P\(X=(-?\d+)\)=(\d+)/(\d+)")
_LIN = re.compile(r"Y = (-?\d+)X ([+-]) (\d+)\.")
_PROP = re.compile(r"values 1, 2, \.\.\., (\d+), with P\(X=x\) proportional to (x\^2|x)\.")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _var(values, weights):
    total = sum(weights)
    mean = sum(Fraction(v) * w for v, w in zip(values, weights)) / total
    return sum((Fraction(v) - mean) ** 2 * w for v, w in zip(values, weights)) / total


def _oracle(text):
    m = _PROP.search(text)
    if m:
        n = int(m.group(1))
        power = 2 if m.group(2) == "x^2" else 1
        return _var(list(range(1, n + 1)), [x**power for x in range(1, n + 1)])
    pmf = _PMF.findall(text)
    values = [int(v) for v, _, _ in pmf]
    weights = [int(w) for _, w, _ in pmf]
    assert all(int(t) == sum(weights) for _, _, t in pmf), text
    a, sign, b = _LIN.search(text).groups()
    shift = int(b) if sign == "+" else -int(b)
    return _var([int(a) * v + shift for v in values], weights)


def test_shapes_over_60_seeds_and_new_over_hard():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("finite_distribution_variance", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for s in range(60):
        assert find_text_defects(_q(s)) == []


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
