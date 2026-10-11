"""Exam source for quadratic variation posed as percentage changes."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_a2_quadratic_variation_percent"


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _independent(text):
    """Recompute the answer from the rendered prompt with exact fractions."""
    pct = [int(n) for n in re.findall(r"(\d+)%", text)]
    assert len(pct) == 1, text
    p = Fraction(pct[0], 100)
    body = text.split(". ", 1)[1]
    sign = -1 if "decrease" in body else 1
    if "measured as" in body:
        y0 = int(re.search(r"measured as (\d+)", body).group(1))
        return y0 * (1 + sign * p) ** 2
    if "undergoes a" in body:
        # y changed by p (stated), find the percentage change in x
        factor = 1 + sign * p
        root = Fraction(int(round(float(factor) ** 0.5 * 100)), 100)
        assert root * root == factor
        return abs(root - 1) * 100
    return abs((1 + sign * p) ** 2 - 1) * 100


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("p7d_examine_quadratic_variation", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_key_matches_independent_computation_over_400_seeds():
    bad = []
    kinds = set()
    for s in range(400):
        q = _q(s)
        want = _independent(q.text)
        kinds.add(("measured as" in q.text, "undergoes" in q.text))
        if want.denominator != 1 or int(want) != int(q.answer_text):
            bad.append((s, q.text, q.answer_text, want))
    assert bad == []
    assert len(kinds) == 3


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["number"]
    bad = []
    for s in range(200):
        q = _q(s)
        if not grade(q, q.answer_text).correct:
            bad.append(("key", s))
        for d in q.distractors:
            if grade(q, d).correct:
                bad.append(("distractor", s, d))
        assert len(q.distractors) >= 2
    assert bad == []


def test_deterministic():
    assert _q(3).to_json() == _q(3).to_json()
