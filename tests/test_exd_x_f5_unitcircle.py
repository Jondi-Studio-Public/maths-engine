"""Tests for the exam source of `senior-understand-the-unit-circle-definition-of-cos-theta-acmmm034`
(topic `exd_x_f5_unitcircle`): tan(theta) from one coordinate plus the quadrant."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import mcq_options
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f5_unitcircle"
_GRADER = GRADERS["mcq"]
_QUADRANT = {"I": (1, 1), "II": (-1, 1), "III": (-1, -1), "IV": (1, -1)}


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("unit_circle_tan_evaluate", seed=seed, variant="mixed_with_quadrantal")


def _missing(value: Fraction) -> Fraction:
    rest = 1 - value * value
    num, den = math.isqrt(rest.numerator), math.isqrt(rest.denominator)
    assert num * num == rest.numerator and den * den == rest.denominator
    return Fraction(num, den)


def _independent(text):
    m = re.search(r"P\((.*?), (.*?)\), and P lies in quadrant (\w+)", text)
    if m:
        sx, sy = _QUADRANT[m.group(3)]
        if m.group(2) == "y":
            cos_v = Fraction(m.group(1))
            sin_v = sy * _missing(cos_v)
        else:
            sin_v = Fraction(m.group(2))
            cos_v = sx * _missing(sin_v)
        return sin_v / cos_v
    m = re.search(r"(cos|sin)\(theta\) = (-?[\d/]+) and (cos|sin)\(theta\) ([<>]) 0", text)
    assert m, text
    given = Fraction(m.group(2))
    sign = 1 if m.group(4) == ">" else -1
    if m.group(1) == "cos":
        return sign * _missing(given) / given
    return given / (sign * _missing(given))


def test_key_matches_independent_derivation_over_320_seeds():
    bad = []
    for seed in range(320):
        q = _exam(seed)
        if _independent(q.text) != Fraction(q.answer_text):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor_over_120_seeds():
    bad = []
    for seed in range(120):
        q = _exam(seed)
        options = mcq_options(q)
        assert len(options) == 4 and len(set(options)) == 4
        for idx, option in enumerate(options):
            ok = _GRADER(q, idx).correct
            if ok != (option == q.answer_text):
                bad.append((seed, idx, option, q.answer_text))
        assert all(Fraction(d) != Fraction(q.answer_text) for d in q.distractors)
    assert bad == [], bad[:3]
