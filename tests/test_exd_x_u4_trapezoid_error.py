"""Exam source of `composite-trapezoidal-polynomial`: trapezoidal estimate, exact integral and error of a depth model."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from mathgen.generators._u4_kit import survives_nsimplify
from tests.exd_x_u4_support import (
    check_answer_space,
    check_distractors_rejected,
    check_full_marks,
    check_no_text_defects,
    check_tagged,
    exam,
    frac,
    shapes,
    skill,
)

TOPIC = "exd_x_u4_trapezoid_error"
SKILL_ID = "composite-trapezoidal-polynomial"
_X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _setting(text: str):
    body = re.search(r"d\(x\) = (.+?) for (\d+) ≤ x ≤ (\d+)", text)
    expr = parse_expr(
        body.group(1).replace("²", "**2").replace("³", "**3"), transformations=_TRANSFORMS, local_dict={"x": _X}
    )
    n = int(re.search(r"with (\d+) equal strips", text).group(1))
    return expr, int(body.group(2)), int(body.group(3)), n


def _oracle(text: str) -> dict[str, Fraction]:
    expr, left, right, n = _setting(text)
    h = sympy.Rational(right - left, n)
    nodes = [expr.subs(_X, left + i * h) for i in range(n + 1)]
    estimate = h * (sum(nodes) - (nodes[0] + nodes[-1]) / 2)
    exact = sympy.integrate(expr, (_X, left, right))

    def to_fraction(v):
        return Fraction(int(v.p), int(v.q))

    return {"a": to_fraction(estimate), "b": to_fraction(exact), "c": to_fraction(estimate - exact)}


def test_prompt_variety_and_new_shape():
    exam_shapes = shapes(TOPIC, 60, difficulty="exam")
    hard = shapes("composite_trapezoidal_polynomial", 60, difficulty="hard")
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard


def test_no_text_defects():
    check_no_text_defects(TOPIC)


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(TOPIC, seed)
        assert {p.label: frac(p.answer_text) for p in q.parts} == _oracle(q.text), (seed, q.text)


def test_structure():
    for seed in range(60):
        q = exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors_rejected(SKILL_ID, TOPIC)
    check_tagged(TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_estimate_and_wrong_integral():
    sk = skill(SKILL_ID)
    tested = 0
    for seed in range(100):
        q = exam(TOPIC, seed)
        a = frac(q.parts[0].answer_text) + Fraction(3, 2)
        b = frac(q.parts[1].answer_text) + Fraction(1, 3)
        c = a - b
        if len({a, b, c}) < 3 or c == 0 or not survives_nsimplify(a, b, c):
            continue
        tested += 1
        result = grade(sk, q, {"a": str(a), "b": str(b), "c": str(c)})
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks
        assert result.parts["c"].follow_through is True
        assert not result.correct
    assert tested >= 80


def test_all_wrong_earns_zero():
    sk = skill(SKILL_ID)
    for seed in range(40):
        assert grade(sk, exam(TOPIC, seed), {"a": "999", "b": "998", "c": "997"}).marks == 0


def test_deterministic():
    assert exam(TOPIC, 7).to_json() == exam(TOPIC, 7).to_json()
