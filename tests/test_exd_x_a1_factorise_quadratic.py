"""Exam source of `factorise-quadratic`: two-variable and bracket-substitution quadratics."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_a1_factorise_quadratic"
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _parse(text):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("factorise_quadratic", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        expr = _parse(re.match(r"Factorise (.*)\.$", q.text).group(1))
        factors = sympy.factor_list(sympy.expand(expr))[1]
        assert len(factors) == 2, (seed, q.text)
        assert all(sympy.Poly(f, *sorted(f.free_symbols, key=str)).total_degree() == 1 for f, _ in factors)
        assert sympy.expand(_parse(q.answer_text) - expr) == 0, (seed, q.text, q.answer_text)
        assert "(" in q.answer_text


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["expression"]
    shown = 0
    for seed in range(120):
        q = _exam(seed)
        assert grade(q, q.answer_text).correct, (seed, q.answer_text)
        for d in q.distractors:
            shown += 1
            assert not grade(q, d).correct, (seed, q.text, d)
    assert shown > 120


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
