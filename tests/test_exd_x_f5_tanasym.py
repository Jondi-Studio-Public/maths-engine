"""Tests for the exam source of `senior-y-tan-x-c-and-the-relationships-sin` (topic `exd_x_f5_tanasym`):
vertical asymptotes of y = tan(ax + c)."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f5_tanasym"
_GRADER = GRADERS["expression"]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("tan_shift_complementary_identity", seed=seed, variant="tan_asymptote")


def test_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert exam_shapes - hard_shapes


def test_deterministic():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nope")


def _pi_multiple(text: str) -> Fraction:
    m = re.fullmatch(r"(-?)(\d*)pi(?:/(\d+))?", text)
    assert m, text
    return (-1 if m.group(1) else 1) * Fraction(int(m.group(2) or 1), int(m.group(3) or 1))


def _independent(text):
    """Smallest positive x (in units of pi) where cos(argument) = 0, from the argument printed in the prompt."""
    m = re.search(r"tan\((\d)x ([+-]) (\S+)\)", text)
    if m:
        a, c = int(m.group(1)), (1 if m.group(2) == "+" else -1) * _pi_multiple(m.group(3))
    else:
        m = re.search(r"tan\((\d)\(x ([+-]) (\S+)\)\)", text)
        assert m, text
        a = int(m.group(1))
        c = a * (1 if m.group(2) == "+" else -1) * _pi_multiple(m.group(3))
    # scan the asymptotes x = (1/2 + k - c)/a for k in a wide window and keep the smallest positive one
    candidates = [(Fraction(1, 2) + k - c) / a for k in range(-40, 41)]
    best = min(x for x in candidates if x > 0)
    assert abs(math.cos(math.pi * float(a * best + c))) < 1e-9
    return best


def test_key_matches_independent_derivation_over_320_seeds():
    bad = []
    for seed in range(320):
        q = _exam(seed)
        if _independent(q.text) != _pi_multiple(q.answer_text):
            bad.append((seed, q.text, q.answer_text))
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
