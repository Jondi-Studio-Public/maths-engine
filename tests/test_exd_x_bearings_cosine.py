"""Tests for the `exd_x_bearings_cosine` exam source (straight-line distance for a non-right-angled two-leg journey)."""

from __future__ import annotations

import math
import re

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_bearings_cosine"
HARD_TOPIC = "bearings_problem"
HARD_PARAMS = {"difficulty": "hard"}
_GRADER = GRADERS["number"]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_prompt_shapes_vary_and_are_new_over_hard():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD_TOPIC, seed=s, **HARD_PARAMS)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def test_key_matches_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _exam(seed)
        expected = _oracle(q.text)
        if expected != q.answer:
            bad.append((seed, q.text, expected, q.answer))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for seed in range(150):
        q = _exam(seed)
        assert len(q.distractors) >= 3, (seed, q.distractors)
        if not _GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed))
        for d in q.distractors:
            if _GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d))
    assert bad == [], bad[:3]


_RE = re.compile(
    r"(\d+) km on a bearing of (\d+)\u00b0 from A to B, then (\d+) km on a bearing of (\d+)\u00b0 from B to C"
)


def _oracle(text):
    d1, b1, d2, b2 = (int(x) for x in _RE.search(text).groups())
    x = d1 * math.sin(math.radians(b1)) + d2 * math.sin(math.radians(b2))
    y = d1 * math.cos(math.radians(b1)) + d2 * math.cos(math.radians(b2))
    dist = math.hypot(x, y)
    assert abs(dist - round(dist)) < 1e-9
    return round(dist)


def test_the_turn_is_never_a_right_angle():
    for seed in range(100):
        d1, b1, d2, b2 = (int(x) for x in _RE.search(_exam(seed).text).groups())
        assert (b2 - b1) % 360 in (60, 120, 240, 300)
