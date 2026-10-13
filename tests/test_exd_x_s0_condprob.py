"""Tests for the conditional-probability-counts exam source (exd_x_s0_condprob).

The key is re-derived from the rendered prompt alone, with exact fractions and no import of the generator.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s0_condprob"
HARD = ("conditional_probability_counts", {"kind": "totals"})
_NUM = r"-?\d+(?:/\d+)?"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty="hard", **HARD[1])) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard, "the exam source adds no shape beyond the hard tier"


def test_no_text_defects():
    for s in range(60):
        assert find_text_defects(_exam(s)) == []


def test_deterministic():
    for s in (0, 1, 7, 99):
        assert _exam(s).to_json() == _exam(s).to_json()


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
    n, a, b, z = (int(x) for x in re.findall(r"\d+", text)[:4])
    both = a + b - (n - z)
    only_a, only_b = a - both, b - both
    if "at least one" in text:
        return Fraction(both, n - z)
    if "exactly one" in text:
        return Fraction(only_a, only_a + only_b)
    assert "not do both" in text
    return Fraction(only_a, n - both)


def test_every_option_is_a_probability_300_seeds():
    for s in range(300):
        q = _exam(s)
        for text in (q.answer_text, *q.distractors):
            assert 0 <= Fraction(text) <= 1, (s, text)


def test_no_plural_verb_phrase_after_a_singular_subject():
    banned = ("who speak", "who take", "who play", "who own", "who subscribe", "who follow")
    bad = [(s, w) for s in range(300) for w in banned if w in _exam(s).text]
    assert bad == []
