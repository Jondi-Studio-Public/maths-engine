"""Exam chain of `ucv1-initial-value-constant`: the constant, the value at a second point, the tangent there."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from exd_x_r3f_support import (
    assert_all_wrong_earns_nothing,
    assert_answer_space,
    assert_clean_and_deterministic,
    assert_follow_through,
    assert_full_marks_and_distractors,
    assert_shapes,
    assert_wired,
    exam_question,
)

TOPIC = "exd_x_r3f_initial_value_chain"
SKILL_ID = "ucv1-initial-value-constant"
_T = standard_transformations + (implicit_multiplication_application,)


def read_question(q):
    var = "t" if "particle" in q.stem else "x"
    dep = "s" if var == "t" else "y"
    form = re.search(rf"{dep} = (.*?) \+ C", q.stem).group(1)
    point = re.search(rf"\(({'-?\\d+'}), (-?\d+)\)|When t = (-?\d+), s = (-?\d+)", q.stem)
    x0, y0 = (
        (int(point.group(1)), int(point.group(2))) if point.group(1) else (int(point.group(3)), int(point.group(4)))
    )
    x1 = int(re.search(rf"when {var} = (-?\d+)", q.parts[1].prompt).group(1))
    symbol = sympy.Symbol(var)
    antiderivative = parse_expr(form.replace("^", "**"), transformations=_T, local_dict={var: symbol})
    return symbol, antiderivative, x0, y0, x1


def oracle(q):
    v, anti, x0, y0, x1 = read_question(q)
    c = sympy.Rational(y0) - anti.subs(v, x0)
    y = anti + c
    slope = sympy.diff(anti, v).subs(v, x1)
    tangent = slope * (v - x1) + y.subs(v, x1)
    return c, y.subs(v, x1), sympy.expand(tangent), v


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "expression"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    for q in questions:
        c, value, tangent, v = oracle(q)
        assert Fraction(q.parts[0].answer_text) == Fraction(int(c.p), int(c.q)), q.text
        assert Fraction(q.parts[1].answer_text) == Fraction(int(value.p), int(value.q)), q.text
        text = q.parts[2].answer_text.replace("^", "**")
        got = parse_expr(text, transformations=_T, local_dict={str(v): v})
        assert sympy.expand(got - tangent) == 0, (q.text, text)


def test_both_settings_occur(questions):
    assert {"particle" in q.stem for q in questions} == {True, False}


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC, min_options=2)


def test_answer_space():
    assert_answer_space(TOPIC)


def _wrong_constant_replies(q):
    v, anti, x0, y0, x1 = read_question(q)
    c_wrong = sympy.Rational(q.parts[0].distractors[0])
    value = anti.subs(v, x1) + c_wrong
    slope = sympy.diff(anti, v).subs(v, x1)
    tangent = sympy.expand(slope * (v - x1) + value)
    return {"a": str(c_wrong), "b": str(value), "c": str(tangent)}


def test_follow_through_from_a_wrong_constant():
    def usable(q):
        replies = _wrong_constant_replies(q)
        nonzero = sympy.Rational(replies["a"]) != 0 and sympy.Rational(replies["b"]) != 0
        return replies if nonzero else None

    assert_follow_through(SKILL_ID, TOPIC, usable, ["a"], ["b", "c"])


def _wrong_value_replies(q):
    v, anti, x0, y0, x1 = read_question(q)
    c, value, tangent, _ = oracle(q)
    value_wrong = sympy.Rational(q.parts[1].distractors[0])
    slope = sympy.diff(anti, v).subs(v, x1)
    return {"a": q.parts[0].answer_text, "b": str(value_wrong), "c": str(sympy.expand(slope * (v - x1) + value_wrong))}


def test_follow_through_from_a_wrong_value():
    def usable(q):
        replies = _wrong_value_replies(q)
        return replies if sympy.Rational(replies["b"]) != 0 else None

    assert_follow_through(SKILL_ID, TOPIC, usable, ["b"], ["c"])


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "991/7", "b": "992/7", "c": "993*x + 5"})
