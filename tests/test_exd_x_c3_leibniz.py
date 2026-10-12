"""Exam tier of `senior-use-the-leibniz-notation-for-the-derivative-frac`: chained rates in Leibniz notation."""

import re

import exd_x_c3_support as sup
from sympy import diff, symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

SKILL = "senior-use-the-leibniz-notation-for-the-derivative-frac"
TOPIC = "exd_x_c3_leibniz"
x, u, t = symbols("x u t")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_CHAIN = re.compile(
    r"y = (.+), where u = (.+)\. Using the Leibniz chain dy/dx = dy/du × du/dx, find the value of dy/dx when x = (-?\d+)\."
)
_RATE = re.compile(
    r"y = (.+), and x changes with time t so that dx/dt = (-?\d+) when x = (-?\d+)\. "
    r"Using the Leibniz chain dy/dt = dy/dx × dx/dt, find the value of dy/dt at that moment\."
)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), local_dict={"x": x, "u": u}, transformations=_TRANSFORMS)


def _independent_key(text: str) -> int:
    if m := _CHAIN.fullmatch(text):
        y, ux = _parse(m.group(1)), _parse(m.group(2))
        return int(diff(y.subs(u, ux), x).subs(x, int(m.group(3))))  # substitute first, then differentiate
    m = _RATE.fullmatch(text)
    assert m, text
    y, rate, p = _parse(m.group(1)), int(m.group(2)), int(m.group(3))
    return int(diff(y.subs(x, p + rate * t), t).subs(t, 0))  # x moves through p at the stated rate


def test_registered():
    sup.assert_registered_as_exam_source(SKILL, "exd_x_r2b_leibniz_chain")


def test_text_and_determinism():
    sup.assert_clean_text_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_matches_an_independent_derivative_of_the_rendered_prompt():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        assert int(q.answer_text) == _independent_key(q.text) == q.answer, (seed, q.text, q.answer_text)


def test_both_kinds_occur():
    texts = [sup.exam_question(TOPIC, s).text for s in range(40)]
    assert any("dy/du" in t for t in texts) and any("dy/dt" in t for t in texts)


def test_grader():
    sup.assert_grader(SKILL, TOPIC)
