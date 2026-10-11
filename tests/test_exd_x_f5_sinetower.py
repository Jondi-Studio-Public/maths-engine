"""Tests for the exam source of `sine-rule` (topic `exd_x_f5_sinetower`): the height of a vertical object from two
angles of elevation."""

from __future__ import annotations

import re

import pytest
from mpmath import mp, mpf, radians, tan

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

mp.dps = 40
TOPIC = "exd_x_f5_sinetower"
_GRADER = GRADERS["number"]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("sine_rule_solve", seed=seed)


def test_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    hard_shapes |= {
        prompt_shape(make("trigonometry", seed=s, difficulty="hard", kinds=("sine_rule",))) for s in range(60)
    }
    assert exam_shapes - hard_shapes


def test_deterministic():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nope")


def _independent(text):
    """Height from the rendered prompt by coordinates: the foot is the origin, A and B sit at h / tan(angle), and
    they are d apart."""
    d = int(re.search(r"(\d+) m", text).group(1))
    near, far = (int(v) for v in re.findall(r"(\d+) degrees", text))
    return d / (1 / tan(radians(far)) - 1 / tan(radians(near)))


def test_key_matches_independent_derivation_over_320_seeds():
    bad = []
    for seed in range(320):
        q = _exam(seed)
        height = _independent(q.text)
        if round(float(height), 1) != q.answer or abs(float(height * 10) % 1 - 0.5) < 0.01:
            bad.append((seed, q.text, float(height), q.answer))
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
