"""Tests for the exam-tier source of ``definite-integral`` (exd_x_c2_definite_integral).

The key is re-derived from the rendered prompt alone: the two curves are parsed, the intersection points found
with the quadratic formula over Fractions, and the enclosed area computed by integrating the difference exactly.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_c2_definite_integral"
GRADER = "number"
_X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("integration", seed=seed, difficulty="hard", **{"kinds": ("definite",)})


def _key(q):
    """The answer as an exact Fraction (integers and {p, q} dicts)."""
    if isinstance(q.answer, dict):
        return Fraction(q.answer["p"], q.answer["q"])
    return Fraction(q.answer)


def _parse(text):
    """Parse learner-style maths text with sympy: powers of functions, root sign, implicit products."""
    text = text.replace("√", " sqrt").replace("−", "-")
    text = re.sub(r"(sin|cos)\^(\d+)\(([^()]*)\)", r"(\1(\3))**\2", text)
    return parse_expr(text, transformations=_TRANSFORMS, local_dict={"x": _X})


def test_no_text_defects_over_60_seeds():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - hard_shapes, "the exam tier adds no prompt shape beyond the hard tier"


def test_deterministic():
    for seed in (0, 1, 2, 3, 100, 12345):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_key_accepted_and_every_distractor_rejected_over_120_seeds():
    grade = GRADERS[GRADER]
    for seed in range(120):
        q = _exam(seed)
        assert q.distractors, seed
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors
        assert grade(q, q.answer_text).correct, (seed, q.text, q.answer_text)
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, q.text, q.answer_text, d)


def test_wrong_answer_rejected():
    grade = GRADERS[GRADER]
    for seed in range(40):
        q = _exam(seed)
        assert not grade(q, "12345").correct
        assert not grade(q, "banana").correct


def _curves(text):
    found = re.findall(r"y = (.+?)(?: and the | bound|\.$)", text)
    assert len(found) == 2, text
    parsed = [sympy.Poly(_parse(f), _X) for f in found]
    quad = [p for p in parsed if p.degree() == 2]
    other = [p for p in parsed if p.degree() < 2]
    assert len(quad) == 1 and len(other) == 1, text
    return quad[0], other[0]


def _oracle_area(text):
    quad, line = _curves(text)
    diff = quad - line
    a, b, c = (Fraction(int(v.p), int(v.q)) for v in diff.all_coeffs())
    disc = b * b - 4 * a * c
    assert disc > 0
    root = (
        Fraction(int(sympy.sqrt(sympy.Rational(disc.numerator, disc.denominator)).p), 1)
        if disc.denominator == 1
        else None
    )
    assert root is not None and root * root == disc
    lo, hi = sorted([(-b - root) / (2 * a), (-b + root) / (2 * a)])
    # integral of |a (x - lo)(x - hi)| over [lo, hi]
    antiderivative = lambda t: a * (t**3 / 3 - (lo + hi) * t**2 / 2 + lo * hi * t)  # noqa: E731
    return abs(antiderivative(hi) - antiderivative(lo))


def test_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _exam(seed)
        if _oracle_area(q.text) != _key(q):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad


def test_area_is_positive():
    assert all(_key(_exam(s)) > 0 for s in range(100))
