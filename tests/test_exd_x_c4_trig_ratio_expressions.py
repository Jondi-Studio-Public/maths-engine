"""Exam source for evaluating an expression of sin and cos given one ratio of an acute angle.

The oracle recovers sin and cos from the given ratio with exact fractions and evaluates the rendered expression.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import Question
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_c4_trig_ratio_expressions"
HARD_TOPIC = "trig_ratio_review_tan"
GRADER = GRADERS["fraction"]
_HARD_KW = {"variant": "cofunction"}


def _make(seed: int) -> Question:
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed: int) -> Question:
    return make(HARD_TOPIC, seed=seed, difficulty="hard", **_HARD_KW)


def test_no_text_defects():
    for seed in range(60):
        q = _make(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_make(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert shapes - hard_shapes, (shapes, hard_shapes)


def test_determinism():
    for seed in (0, 1, 7, 99, 12345):
        assert _make(seed).to_json() == _make(seed).to_json()


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")


from fractions import Fraction
from math import isqrt


def _agrees(q: Question) -> bool:
    m = re.match(
        r"For an acute angle theta in a right-angled triangle, (sin|cos|tan)\(theta\) = (\d+)/(\d+)\. Find the exact value of (.+), as a fraction",
        q.text,
    )
    name, p, d, expr = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    # Scale the given ratio to integer sides and complete the right triangle.
    if name == "sin":
        o, h = p, d
        a = isqrt(h * h - o * o)
    elif name == "cos":
        a, h = p, d
        o = isqrt(h * h - a * a)
    else:
        o, a = p, d
        h = isqrt(o * o + a * a)
    assert o * o + a * a == h * h
    s, c = Fraction(o, h), Fraction(a, h)
    py = expr.replace("sin(theta)", "S").replace("cos(theta)", "C").replace("×", "*").replace("^", "**")
    value = eval(py, {"__builtins__": {}}, {"S": s, "C": c})
    return (q.answer["p"], q.answer["q"]) == (value.numerator, value.denominator) and q.answer_text == (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def test_all_four_expression_kinds_occur():
    kinds = {q.text.split("value of ")[1].split(", as")[0] for q in (_make(s) for s in range(80))}
    assert len(kinds) == 4, kinds


def test_independent_oracle_matches_key_over_400_seeds():
    bad = []
    for seed in range(400):
        q = _make(seed)
        if not _agrees(q):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:5]


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for seed in range(150):
        q = _make(seed)
        assert len(q.distractors) >= 1, (seed, q.text)
        if not GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]
