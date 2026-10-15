"""Tests for the exam tier of `vector-dot-product`.

Dot products from lengths and an angle, and the k that makes u + kv perpendicular to w.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g1_vector_dot_product"
GRADER = GRADERS["number"]
_SEEDS = range(320)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("vectors", seed=seed, difficulty="hard", kinds=("dot_sum",), hi=15)


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
    return "angle" if "angle" in text else "perpendicular"


def _expected(text):
    if _kind(text) == "perpendicular":
        u, v, w = (sympy.Matrix(t) for t in _triples(text))
        return Fraction(int(-u.dot(w)), int(v.dot(w)))
    la = int(re.search(r"\|a\| = (\d+)", text).group(1))
    lb = int(re.search(r"\|b\| = (\d+)", text).group(1))
    deg = int(re.search(r"(\d+)°", text).group(1))
    ab = sympy.Integer(la) * lb * sympy.cos(sympy.pi * deg / 180)
    left, right = re.search(r"\(([^()]*b)\) · \(([^()]*b)\)", text).groups()
    lt, rt = _terms(left), _terms(right)
    # (p a + q b) . (r a + s b) with a.a = la^2, b.b = lb^2, a.b = ab
    val = lt["a"] * rt["a"] * la**2 + (lt["a"] * rt["b"] + lt["b"] * rt["a"]) * ab + lt["b"] * rt["b"] * lb**2
    return Fraction(int(sympy.nsimplify(val)))


def _agrees(q, expected):
    return Fraction(q.answer) == expected and expected.denominator == 1


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
        for wrong in ["0", "", "1000", str(q.answer + 1), str(-q.answer)]:
            if wrong == q.answer_text:
                continue
            assert not GRADER(q, wrong).correct, (seed, wrong, q.answer_text)
