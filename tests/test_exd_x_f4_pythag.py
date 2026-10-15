"""Exam source for `applied-pythagoras-problems`: a ladder that slips, a wire between poles of different
heights, and the missing diagonal of a rhombus. Keys are re-derived from the rendered prompt."""

from __future__ import annotations

import math
import re

import mathgen
import mathgen.generators.exd_x_f4_pythag  # noqa: F401  (registers the topic)
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f4_pythag"


def _make(seed):
    return mathgen.make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return mathgen.make("applied_pythagoras", seed=seed, difficulty="hard")


def _agrees(q) -> bool:
    text = q.text
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if "ladder" in text:
        h, x1, s = nums
        y1 = math.isqrt(h * h - x1 * x1)
        assert y1 * y1 + x1 * x1 == h * h
        y2 = y1 - s
        x2 = math.isqrt(h * h - y2 * y2)
        assert x2 * x2 + y2 * y2 == h * h
        value = x2 - x1
    elif "rhombus" in text:
        s, g = nums
        half = math.isqrt(s * s - (g // 2) ** 2)
        assert half * half + (g // 2) ** 2 == s * s
        value = 2 * half
    else:
        if text.startswith("A straight cable"):
            h1, h2, d = nums
        else:
            d, h1, h2 = nums
        value = math.isqrt(d * d + (h1 - h2) ** 2)
        assert value * value == d * d + (h1 - h2) ** 2
    return value == q.answer and str(value) == q.answer_text


def test_all_three_settings_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("ladder" in t for t in texts)
    assert any("rhombus" in t for t in texts)
    assert any("pole" in t for t in texts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_make(s)) for s in range(80)}
    hard = {prompt_shape(_hard(s)) for s in range(80)}
    assert len(exam) >= 3
    assert len(exam - hard) >= 3


def test_key_matches_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        if not _agrees(q):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["number"]
    for seed in range(150):
        q = _make(seed)
        assert grade(q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, q.text, d)
