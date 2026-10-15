"""Tests for the ucv1-bayes-finite-counts exam source (exd_x_s0_bayes2).

The key is re-derived from the rendered prompt alone, with exact fractions and no import of the generator.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s0_bayes2"
HARD = ("bayes_finite_counts", {})
_NUM = r"-?\d+(?:/\d+)?"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty="hard", **HARD[1])) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard, "the exam source adds no shape beyond the hard tier"


def _frac(text):
    return Fraction(text)


def _key(q):
    return Fraction(q.answer["p"], q.answer["q"])


def test_independent_oracle_300_seeds():
    bad = [(s, _exam(s).text) for s in range(300) if _oracle(_exam(s).text) != _key(_exam(s))]
    assert bad == []


def test_grader_accepts_key_rejects_distractors_100_seeds():
    grader = GRADERS["fraction"]
    for s in range(100):
        q = _exam(s)
        assert grader(q, q.answer_text).correct, (s, q.answer_text)
        assert q.distractors, s
        for d in q.distractors:
            assert not grader(q, d).correct, (s, d, q.answer_text)
        assert not grader(q, "").correct


def _oracle(text):
    m = re.search(r"(\d+) of every (\d+) items", text) or re.search(r"[Oo]f every (\d+) items, (\d+) come", text)
    if "makes" in text:
        n1, tot = int(m.group(1)), int(m.group(2))
    else:
        tot, n1 = int(m.group(1)), int(m.group(2))
    (a1, b1), (a2, b2) = [(int(x), int(y)) for x, y in re.findall(r"(\d+) in (\d+)", text)]
    r1, r2 = Fraction(a1, b1), Fraction(a2, b2)
    if "both" in text:
        j1, j2 = n1 * r1 * r1, (tot - n1) * r2 * r2
    else:
        j1, j2 = n1 * r1 * (1 - r1), (tot - n1) * r2 * (1 - r2)
    return j1 / (j1 + j2)
