"""Tests for the exam source of `senior-convert-sums-mathrm-a-cos-mathrm-x-mathrm` (topic `exd_x_f5_auxsum`)."""

from __future__ import annotations

import re

import pytest
from mpmath import atan2, degrees, mp, mpf, radians, sin, sqrt
from mpmath import cos as mcos

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

mp.dps = 40
TOPIC = "exd_x_f5_auxsum"
_GRADER = GRADERS["pair"]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("aux_angle_r_cos_conversion", seed=seed, variant="sin_form_mixed_sign")


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


_TERM = re.compile(r"([+-]?)\s*(\d+) (cos|sin)\(x ([+-]) (\d+)°\)")


def _independent(text):
    """(R, alpha) from the rendered prompt alone, by expanding each term at high precision."""
    expr = text.split("f(x) = ")[1].split(" in the form")[0]
    a = b = mpf(0)
    terms = _TERM.findall(expr)
    assert len(terms) == 2, expr
    for sign, amp, fn, shift_sign, mag in terms:
        amp = (-1 if sign == "-" else 1) * int(amp)
        u = radians((-1 if shift_sign == "-" else 1) * int(mag))
        if fn == "cos":
            a += amp * mcos(u)
            b -= amp * sin(u)
        else:
            a += amp * sin(u)
            b += amp * mcos(u)
    return sqrt(a * a + b * b), degrees(atan2(b, a))


def test_key_matches_independent_derivation_over_320_seeds():
    bad = []
    for seed in range(320):
        q = _exam(seed)
        r, alpha = _independent(q.text)
        got = (round(float(r), 1), round(float(alpha), 1) + 0.0)
        if got != (q.answer["R"], q.answer["alpha"]):
            bad.append((seed, q.text, got, q.answer))
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
