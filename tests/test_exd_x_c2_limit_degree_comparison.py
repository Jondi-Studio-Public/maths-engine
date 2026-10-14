"""Tests for the exam-tier source of ``limit-degree-comparison`` (exd_x_c2_limit_degree_comparison).

The key is re-derived from the rendered prompt: numerator and denominator are expanded to polynomials, and the
answer follows from their degrees, leading coefficients and the direction of approach.
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

TOPIC = "exd_x_c2_limit_degree_comparison"
GRADER = "expression"
_X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("limits", seed=seed, difficulty="hard", **{"kinds": ("degree_comparison",)})


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


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - hard_shapes, "the exam tier adds no prompt shape beyond the hard tier"


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


_PATTERNS = (
    re.compile(r"^Find the limit as x → (-?∞) of (.+)\.$"),
    re.compile(r"^Evaluate lim x→(-?∞) of (.+)\.$"),
    re.compile(r"^Determine the limit of (.+) as x → (-?∞)\.$"),
)


def _split(text):
    text = text.replace(" Give your answer as a number, ∞ or -∞.", "")
    for i, pattern in enumerate(_PATTERNS):
        m = pattern.match(text)
        if m:
            return (m.group(2), m.group(1)) if i < 2 else (m.group(1), m.group(2))
    raise AssertionError(text)


def _oracle(text):
    expr_text, target = _split(text)
    numerator, denominator = expr_text.split(" / ")
    num = sympy.Poly(sympy.expand(_parse(numerator)), _X)
    den = sympy.Poly(sympy.expand(_parse(denominator)), _X)
    dn, dd = num.degree(), den.degree()
    assert dn != dd
    if dn < dd:
        return "0"
    sign = 1 if num.LC() * den.LC() > 0 else -1
    if target == "-∞" and (dn - dd) % 2:
        sign = -sign
    return "∞" if sign > 0 else "-∞"


def test_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _exam(seed)
        if _oracle(q.text) != q.answer_text:
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad


def test_all_three_answers_and_both_directions_occur():
    answers = {_exam(s).answer_text for s in range(100)}
    assert answers == {"0", "∞", "-∞"}
    assert {_split(_exam(s).text)[1] for s in range(60)} == {"∞", "-∞"}


def test_every_prompt_allows_an_infinite_answer():
    for seed in range(300):
        assert "Give your answer as a number, ∞ or -∞." in _exam(seed).text
