"""Tests for the exam tier of `vector-magnitude`.

|pa + qb| from lengths and an angle, and the length of a vector projection; exact surds.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g1_vector_magnitude"
GRADER = GRADERS["expression"]
_SEEDS = range(320)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("vectors", seed=seed, difficulty="hard", kinds=("magnitude_sum",), hi=15)


_NUM = r"-?\d+"
_TRIPLE = re.compile(r"\((-?\d+), (-?\d+), (-?\d+)\)")
_PAIR = re.compile(r"\((-?\d+), (-?\d+)\)")


def _triples(text):
    return [tuple(int(x) for x in m.groups()) for m in _TRIPLE.finditer(text)]


def _pairs(text):
    return [tuple(int(x) for x in m.groups()) for m in _PAIR.finditer(text)]


def _terms(expr):
    """Coefficients of the named vectors in an expression such as '3p - q'."""
    out = {}
    for sign, digits, name in re.findall(r"([+-]?)\s*(\d*)([a-z])", expr):
        out[name] = (-1 if sign == "-" else 1) * (int(digits) if digits else 1)
    return out


def _kind(text):
    return "projection" if "projection" in text else "angle"


def _expected(text):
    if _kind(text) == "projection":
        u, v = (sympy.Matrix(t) for t in _triples(text))
        return abs(u.dot(v)) / sympy.sqrt(v.dot(v))
    la = int(re.search(r"\|a\| = (\d+)", text).group(1))
    lb = int(re.search(r"\|b\| = (\d+)", text).group(1))
    deg = int(re.search(r"(\d+)°", text).group(1))
    expr = re.search(r"\|([^|]*[ab])\|", text.split("Find")[-1]).group(1)
    t = _terms(expr)
    cos_t = sympy.cos(sympy.pi * deg / 180)
    n = t["a"] ** 2 * la**2 + t["b"] ** 2 * lb**2 + 2 * t["a"] * t["b"] * la * lb * cos_t
    return sympy.sqrt(sympy.nsimplify(n))


def _agrees(q, expected):
    given = sympy.sympify(q.answer_text)
    return sympy.simplify(given - expected) == 0 and abs(float(expected) - q.answer_alt) < 0.006


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - hard_shapes, "the exam tier adds no prompt shape beyond the hard tier"


def test_only_the_exam_preset_exists():
    with pytest.raises(KeyError):
        make(TOPIC, seed=0, difficulty="easy")


def test_key_matches_independent_derivation_from_the_rendered_prompt():
    bad = []
    for seed in _SEEDS:
        q = _exam(seed)
        expected = _expected(q.text)
        if not _agrees(q, expected):
            bad.append((seed, q.text, q.answer_text, expected))
    assert bad == [], bad[:5]


def test_both_question_kinds_occur():
    kinds = {_kind(_exam(s).text) for s in _SEEDS}
    assert len(kinds) >= 2, kinds


def test_grader_accepts_key_and_rejects_every_distractor():
    bad = []
    for seed in range(160):
        q = _exam(seed)
        if not GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        assert len(q.distractors) >= 2, (seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.distractors)
        for d in q.distractors:
            if GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]


def test_grader_rejects_deliberately_wrong_values():
    for seed in range(40):
        q = _exam(seed)
        for wrong in ["0", "", "1000", "sqrt(2)", q.answer_text + " + 1"]:
            if wrong == q.answer_text:
                continue
            assert not GRADER(q, wrong).correct, (seed, wrong, q.answer_text)
