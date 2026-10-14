"""Exam chain of `limit-degree-comparison`: growth exponent, growth constant, and the limit at minus infinity."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_limit_growth"
SKILL = S.load_skill("limit-degree-comparison")
_X = sympy.Symbol("x")
_TR = standard_transformations + (implicit_multiplication_application, convert_xor)


def _poly(text: str) -> sympy.Poly:
    return sympy.Poly(sympy.expand(parse_expr(text, transformations=_TR, local_dict={"x": _X})), _X)


def _terms(q):
    expr = re.search(r"f(?:\(x\))? = (.+?) / (.+?)\.$", q.stem) or re.search(r"f\(x\) = (.+?) / (.+?)\.$", q.stem)
    return _poly(expr.group(1)), _poly(expr.group(2))


def oracle(q):
    num, den = _terms(q)
    n = num.degree() - den.degree()
    limit = sympy.Rational(num.LC(), den.LC())
    far = limit * (1 if n % 2 == 0 else -1)
    return {"a": str(n), "b": str(limit), "c": str(far)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_oracle_by_limit_value(questions):
    # Independent of the leading-coefficient reading: evaluate the quotient far out on each side.
    for q in questions[:60]:
        num, den = _terms(q)
        n = int(q.parts[0].answer_text)
        for sign, key in ((1, q.parts[1].answer_text), (-1, q.parts[2].answer_text)):
            x = sympy.Integer(sign * 10**8)
            value = num.eval(x) / den.eval(x) / abs(x) ** n if sign == -1 else num.eval(x) / den.eval(x) / x**n
            assert abs(float(value) - float(key)) < abs(float(key)) * 1e-3, (q.stem, sign)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions, minimum=1)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_exponent(questions):
    def chain(q):
        num, den = _terms(q)
        limit = int(q.parts[1].answer_text)
        wrong_a = next((int(d) for d in q.parts[0].distractors if int(d) % 2), None)
        wrong_b = next((int(d) for d in q.parts[1].distractors if abs(int(d)) > 1), None)
        if wrong_a is None or wrong_b is None:
            return None
        c = wrong_b * (-1) ** wrong_a
        if c in (wrong_a, wrong_b):
            return None  # writing an earlier reply again is not follow-through
        return {"a": str(wrong_a), "b": str(wrong_b), "c": str(c)}, {"c"}

    S.check_follow_through(SKILL, questions, chain, n=120)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "99", "b": "98", "c": "97"})


def test_deterministic():
    S.check_deterministic(TOPIC)
