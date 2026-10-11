"""Exam source of `quadratic-vertex`: turning point from the intercepts and one more fact."""

from __future__ import annotations

import re

from sympy import Rational

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_a1_quadratic_vertex"
_NUM = r"(-?\d+)"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _rederive(text):
    if "x-intercepts" in text:
        pts = [tuple(int(v) for v in m) for m in re.findall(rf"\({_NUM}, {_NUM}\)", text)]
        (p, _), (q, _), point = pts
    else:
        p, q = (int(v) for v in re.search(rf"x = {_NUM} and x = {_NUM}", text).groups())
        if "y-axis" in text:
            point = (0, int(re.search(rf"y-axis at y = {_NUM}", text).group(1)))
        else:
            point = tuple(int(v) for v in re.search(rf"point \({_NUM}, {_NUM}\)", text).groups())
    a = Rational(point[1], (point[0] - p) * (point[0] - q))
    h = Rational(p + q, 2)
    return h, a * (h - p) * (h - q)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("quadratic", seed=s, difficulty="hard", kind="vertex")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        h, k = _rederive(q.text)
        assert (q.answer["x"], q.answer["y"]) == (h, k), (seed, q.text, q.answer)


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["pair"]
    shown = 0
    for seed in range(120):
        q = _exam(seed)
        assert grade(q, q.answer_text).correct, (seed, q.answer_text)
        for d in q.distractors:
            shown += 1
            assert not grade(q, d).correct, (seed, q.text, d)
    assert shown >= 300


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
