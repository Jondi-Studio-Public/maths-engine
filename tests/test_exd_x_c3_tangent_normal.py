"""Exam tier of `senior-find-the-slope-of-a-tangent-and-the`: tangent parallel to a line, and normal to a cubic."""

import re

import exd_x_c3_support as sup
from sympy import Poly, Rational, diff, expand, solve, symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

SKILL = "senior-find-the-slope-of-a-tangent-and-the"
TOPIC = "exd_x_c3_tangent_normal"
x = symbols("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_PARALLEL = re.compile(
    r"Find the equation of the tangent to the curve y = (.+) that is parallel to the line y = (.+)\."
)
_NORMAL = re.compile(r"Find the equation of the normal to the curve y = (.+) at the point where x = (-?\d+)\.")


def _parse(text: str):
    return parse_expr(text.replace("·", "*").replace("^", "**"), local_dict={"x": x}, transformations=_TRANSFORMS)


def _independent_line(text: str):
    if m := _PARALLEL.fullmatch(text):
        f, line = _parse(m.group(1)), _parse(m.group(2))
        slope = Poly(line, x).coeff_monomial(x)
        (x0,) = solve(diff(f, x) - slope, x)  # the curve is a quadratic, so there is exactly one solution
        return expand(f.subs(x, x0) + slope * (x - x0))
    m = _NORMAL.fullmatch(text)
    assert m, text
    f, p = _parse(m.group(1)), int(m.group(2))
    slope = diff(f, x).subs(x, p)
    return expand(f.subs(x, p) - Rational(1) / slope * (x - p))


def test_registered():
    sup.assert_registered_as_exam_source(SKILL, TOPIC)


def test_text_and_determinism():
    sup.assert_clean_text_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_matches_an_independent_line_built_from_the_rendered_prompt():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        assert q.answer_text.startswith("y = ")
        got = _parse(q.answer_text.removeprefix("y = "))
        assert expand(got - _independent_line(q.text)) == 0, (seed, q.text, q.answer_text)


def test_both_kinds_occur():
    kinds = {("parallel" if "parallel" in sup.exam_question(TOPIC, s).text else "normal") for s in range(40)}
    assert kinds == {"parallel", "normal"}


def test_grader():
    sup.assert_grader(SKILL, TOPIC)
