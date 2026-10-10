"""Tests for the exam-tier source of ``integrate-polynomial`` (exd_x_c2_integrate_polynomial).

The key is checked against the rendered prompt alone: the stated second derivative, f(x0) and f'(x0) are parsed
and the answer must satisfy all three, which determines f uniquely.
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

TOPIC = "exd_x_c2_integrate_polynomial"
GRADER = "expression"
_X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("integration", seed=seed, difficulty="hard", **{"kinds": ("rewrite_condition",)})


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


def _facts(text):
    second = re.search(r"f''\(x\) = (.+?)(?:, | when |\. Also)", text).group(1)
    d1 = re.search(r"f'\((\d+)\) = (-?[\d/]+)", text)
    d0 = re.search(r"f\((\d+)\) = (-?[\d/]+)", text)
    assert d1.group(1) == d0.group(1)
    frac = lambda s: Fraction(s)  # noqa: E731
    return _parse(second), int(d1.group(1)), frac(d1.group(2)), frac(d0.group(2))


def test_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _exam(seed)
        second, x0, d1, d0 = _facts(q.text)
        f = sympy.sympify(q.answer_text.replace("^", "**"))
        ok = (
            sympy.simplify(sympy.diff(f, _X, 2) - second) == 0
            and Fraction(str(sympy.nsimplify(sympy.diff(f, _X).subs(_X, x0)))) == d1
            and Fraction(str(sympy.nsimplify(f.subs(_X, x0)))) == d0
        )
        if not ok:
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad


def test_condition_is_at_a_single_point_and_second_derivative_text_parses():
    for seed in range(60):
        second, x0, _, _ = _facts(_exam(seed).text)
        assert x0 in (1, 2)
        assert sympy.diff(second, _X, 3) is not None
