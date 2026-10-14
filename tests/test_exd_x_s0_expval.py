"""Tests for the discrete-expected-value exam source (exd_x_s0_expval).

The key is re-derived from the rendered prompt alone, with exact fractions and no import of the generator.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s0_expval"
HARD = ("discrete_expected_value", {})
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
    if "c(k +" in text or "proportional" in text:
        s = int(re.search(r"k \+ (\d+)", text).group(1))
        ks = [int(x) for x in re.search(r"(?:values|for k =) ([\d, ]+?)(?: with|\. )", text).group(1).split(",")]
        w = {k: k + s for k in ks}
        return sum(Fraction(k * k * w[k], sum(w.values())) for k in ks)
    vs = [int(x) for x in re.search(r"(?:values|pays) ([-\d, ]+?)(?: dollars|\.)", text).group(1).split(",")]
    ps = (
        re.findall(r"P\(X = -?\d+\) = (" + _NUM + ")", text)
        or re.search(r"probabilities (" + _NUM + ") and (" + _NUM + ")", text).groups()
    )
    p1, p2 = Fraction(ps[0]), Fraction(ps[1])
    r, t = (int(x) for x in re.search(r"ratio (\d+):(\d+)", text).groups())
    rest = 1 - p1 - p2
    return vs[0] * p1 + vs[1] * p2 + vs[2] * rest * Fraction(r, r + t) + vs[3] * rest * Fraction(t, r + t)


def test_working_has_no_repeated_step():
    from mathgen import make as _make

    for seed in range(100):
        for line in _make("exd_x_s0_expval", seed=seed, difficulty="exam").working:
            sides = [x.strip().rstrip(".") for x in line.split(" = ")]
            assert len(set(sides)) == len(sides), line
