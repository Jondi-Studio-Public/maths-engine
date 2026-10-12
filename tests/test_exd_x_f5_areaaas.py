"""Tests for the exam source of `triangle-area-sine` (topic `exd_x_f5_areaaas`): the area from two angles and one
side."""

from __future__ import annotations

import re

import pytest
from mpmath import cos, mp, radians, sin

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape


@pytest.fixture(autouse=True)
def _precision():
    # Set per test: a module-level mp.dps leaks into every later test in the same worker.
    with mp.workdps(40):
        yield


TOPIC = "exd_x_f5_areaaas"
_GRADER = GRADERS["number"]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard_shapes():
    return {
        prompt_shape(make("trigonometry", seed=s, difficulty=d, kinds=(k,)))
        for s in range(60)
        for d, k in (("easy", "triangle_area"), ("medium", "triangle_area"), ("hard", "triangle_area_reverse"))
    }


def test_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - _hard_shapes()


def test_deterministic():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nope")


def _independent(text):
    """Area by placing the triangle in coordinates: A at the origin, B on the x-axis, C from the angle at A."""
    angles = {k: int(v) for k, v in re.findall(r"angle ([ABC]) = (\d+)°", text)}
    assert len(angles) == 2, text
    angles[next(k for k in "ABC" if k not in angles)] = 180 - sum(angles.values())
    letter, side = re.search(r"side ([abc]), opposite angle [ABC], is (\d+) cm", text).groups()
    k = int(side) / sin(radians(angles[letter.upper()]))  # common ratio side / sin(opposite angle)
    b, c = k * sin(radians(angles["B"])), k * sin(radians(angles["C"]))
    cx, cy = b * cos(radians(angles["A"])), b * sin(radians(angles["A"]))
    return abs(c * cy) / 2  # cross product of AB = (c, 0) and AC = (cx, cy)


def test_key_matches_independent_derivation_over_320_seeds():
    bad = []
    for seed in range(320):
        q = _exam(seed)
        area = _independent(q.text)
        if round(float(area), 1) != q.answer or abs(float(area * 10) % 1 - 0.5) < 0.01:
            bad.append((seed, q.text, float(area), q.answer))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor_over_120_seeds():
    bad = []
    for seed in range(120):
        q = _exam(seed)
        if not _GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        assert len(q.distractors) == 3
        for d in q.distractors:
            if _GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:3]
