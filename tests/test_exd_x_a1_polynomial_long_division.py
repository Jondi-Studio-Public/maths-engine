"""Exam source of `polynomial-long-division`: non-monic division with a fractional quotient."""

from __future__ import annotations

import re

import sympy
from sympy import QQ, Poly, symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_a1_polynomial_long_division"
x = symbols("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _parse(text):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS)


def _fg(q):
    f, g = re.match(r"Let f\(x\) = (.*) and g\(x\) = (.*?)\. Find the (quotient|remainder)", q.text).groups()[:2]
    return _parse(f), _parse(g)


def _ask(q):
    return re.search(r"Find the (quotient|remainder)", q.text).group(1)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("polynomial_long_division", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    fractional = 0
    for seed in range(300):
        q = _exam(seed)
        f, g = _fg(q)
        quo, rem = sympy.div(Poly(f, x, domain=QQ), Poly(g, x, domain=QQ))
        expected = quo.as_expr() if _ask(q) == "quotient" else rem.as_expr()
        assert sympy.simplify(_parse(q.answer_text) - expected) == 0, (seed, q.text, q.answer_text)
        assert sympy.simplify(_parse(q.answer) - expected) == 0
        assert rem.as_expr() != 0
        fractional += any(c.q != 1 for c in quo.all_coeffs())
    assert fractional == 300


def test_restating_the_question_is_rejected():
    """The dividend, f/g, the quotient-plus-remainder form and the wrong part must all grade wrong."""
    grade = GRADERS["expression"]
    asked = set()
    for seed in range(150):
        q = _exam(seed)
        f, g = _fg(q)
        quo, rem = sympy.div(Poly(f, x, domain=QQ), Poly(g, x, domain=QQ))
        asked.add(_ask(q))
        wrong = [str(f), f"({f})/({g})", f"{quo.as_expr()} + ({rem.as_expr()})/({g})"]
        wrong.append(str(rem.as_expr() if _ask(q) == "quotient" else quo.as_expr()))
        for given in wrong:
            assert not grade(q, given).correct, (seed, q.text, given)
    assert asked == {"quotient", "remainder"}


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["expression"]
    shown = 0
    for seed in range(120):
        q = _exam(seed)
        assert grade(q, q.answer_text).correct, (seed, q.answer_text)
        for d in q.distractors:
            shown += 1
            assert not grade(q, d).correct, (seed, q.text, d)
    assert shown >= 300


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
