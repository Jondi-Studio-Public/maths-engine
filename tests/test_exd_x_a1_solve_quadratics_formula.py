"""Exam source of `solve-quadratics-formula`: rearrange (brackets or a reciprocal), then use the formula."""

from __future__ import annotations

import re

import sympy
from sympy import symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_a1_solve_quadratics_formula"
x = symbols("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _parse(text):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("quadratic", seed=s, difficulty="hard", kind="formula_rearrange")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    kinds = set()
    for seed in range(300):
        q = _exam(seed)
        left, right = re.match(r"Solve (.*?) = (.*?) using the quadratic formula \(2 d\.p\.\)$", q.text).groups()
        kinds.add("/x" in left)
        sols = sorted(float(v) for v in sympy.solve(sympy.Eq(_parse(left), _parse(right)), x))
        given = sorted(float(v) for v in re.findall(r"-?\d+\.\d\d", q.answer_text))
        assert len(sols) == 2 and len(given) == 2
        assert all(abs(s - g) <= 0.005 + 1e-9 for s, g in zip(sols, given)), (seed, q.text, q.answer_text)
    assert kinds == {True, False}


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["list"]
    shown = 0
    for seed in range(120):
        q = _exam(seed)
        assert grade(q, q.answer_text).correct, (seed, q.answer_text)
        for d in q.distractors:
            shown += 1
            assert not grade(q, d).correct, (seed, q.text, d)
    assert shown >= 200


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
