"""Exam source of `convex-quadratic-minimum`: vertex, constrained minimiser and least value on an allowed range."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.calculus.util import minimum
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

TOPIC = "exd_x_u4_quadratic_on_interval"
SKILL_ID = "convex-quadratic-minimum"
_X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _oracle(text: str) -> dict[str, Fraction]:
    body = re.search(r"\(x\) = (.+?)\. The ", text).group(1)
    f = parse_expr(body.replace("²", "**2"), transformations=_TRANSFORMS, local_dict={"x": _X})
    lo, hi = (int(v) for v in re.search(r"between (\d+) and (\d+)", text).groups())
    vertex = sympy.solve(sympy.diff(f, _X), _X)[0]
    chosen = vertex if lo <= vertex <= hi else (lo if vertex < lo else hi)
    least = minimum(f, _X, sympy.Interval(lo, hi))
    assert f.subs(_X, chosen) == least

    def to_fraction(v):
        return Fraction(int(v.p), int(v.q))

    return {"a": to_fraction(vertex), "b": to_fraction(sympy.Rational(chosen)), "c": to_fraction(least)}


def test_prompt_variety_and_new_shape():
    exam_shapes = shapes(TOPIC, 60, difficulty="exam")
    hard = shapes("convex_quadratic_minimum", 60, difficulty="hard")
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard


def test_no_text_defects():
    check_no_text_defects(TOPIC)


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(TOPIC, seed)
        assert {p.label: frac(p.answer_text) for p in q.parts} == _oracle(q.text), (seed, q.text)


def test_vertex_sometimes_inside_and_sometimes_outside():
    inside = outside = 0
    for seed in range(100):
        q = exam(TOPIC, seed)
        a, b, _ = (frac(p.answer_text) for p in q.parts)
        if a == b:
            inside += 1
        else:
            outside += 1
    assert inside >= 15 and outside >= 40


def test_structure():
    for seed in range(60):
        q = exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors_rejected(SKILL_ID, TOPIC)
    check_tagged(TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_vertex():
    sk = skill(SKILL_ID)
    tested = 0
    for seed in range(100):
        q = exam(TOPIC, seed)
        text = q.text
        coefficients = sympy.Poly(
            parse_expr(
                re.search(r"\(x\) = (.+?)\. The ", text).group(1).replace("²", "**2"),
                transformations=_TRANSFORMS,
                local_dict={"x": _X},
            ),
            _X,
        ).all_coeffs()
        a_, b_, c_ = (int(v) for v in coefficients)
        lo, hi = (int(v) for v in re.search(r"between (\d+) and (\d+)", text).groups())
        a = frac(q.parts[0].answer_text) + Fraction(1, 2)  # a wrong vertex
        b = min(max(a, lo), hi)
        c = a_ * b * b + b_ * b + c_
        if len({a, b, c}) < 3 or not survives_nsimplify(a, b, c):
            continue
        tested += 1
        result = grade(sk, q, {"a": str(a), "b": str(b), "c": str(c)})
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks
        assert not result.correct
    assert tested >= 40


def test_all_wrong_earns_zero():
    sk = skill(SKILL_ID)
    for seed in range(40):
        assert grade(sk, exam(TOPIC, seed), {"a": "9999", "b": "9998", "c": "9997"}).marks == 0


def test_deterministic():
    assert exam(TOPIC, 7).to_json() == exam(TOPIC, 7).to_json()
