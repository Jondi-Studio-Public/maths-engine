"""Exam tier of `senior-suitable-converses-of-some-of-the-above-results`: find x from a converse, then angles."""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests.exd_x_sc6_support import (
    assert_answer_space,
    assert_follow_through,
    assert_grading,
    assert_new_shapes,
    assert_no_text_defects,
    assert_source_wired,
    canonical,
    exam,
    skill_of,
)

SKILL = skill_of("senior-suitable-converses-of-some-of-the-above-results")
TOPIC = "exd_x_sc6_circle_converses"
x = sympy.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _angles(stem: str) -> dict[str, sympy.Expr]:
    return {
        m.group(1): parse_expr(m.group(2), transformations=_T, local_dict={"x": x})
        for m in re.finditer(r"∠([A-Z]{3}) = \(([^)]*)\)°", stem)
    }


def shape_of(stem: str) -> str:
    return "cyclic" if "cyclic quadrilateral" in stem else "arc" if "same segment" in stem else "semi"


def solve_x(q) -> tuple[int, dict[str, sympy.Expr]]:
    a = _angles(q.stem)
    kind = shape_of(q.stem)
    if kind == "cyclic":
        equation = sympy.Eq(a["DAB"] + a["BCD"], 180)
    elif kind == "arc":
        equation = sympy.Eq(a["ACB"], a["ADB"])
    else:
        equation = sympy.Eq(a["ACB"], 90)
    (root,) = sympy.solve(equation, x)
    return int(root), a


def oracle(q) -> dict[str, str]:
    root, a = solve_x(q)
    kind = shape_of(q.stem)
    if kind == "cyclic":
        b = a["ABC"].subs(x, root)
        return {"a": str(root), "b": str(b), "c": str(180 - b)}
    if kind == "arc":
        b = a["ACB"].subs(x, root)
        return {"a": str(root), "b": str(b), "c": str(2 * b)}
    b = a["CAB"].subs(x, root)
    return {"a": str(root), "b": str(b), "c": str(90 - b)}


def test_all_three_shapes_occur_and_x_is_a_whole_number():
    shapes = set()
    for seed in range(90):
        q = exam(SKILL, seed)
        shapes.add(shape_of(q.stem))
        root, a = solve_x(q)
        assert root > 0
        # Every angle in the prompt is a genuine angle at that x.
        assert all(0 < int(v.subs(x, root)) < 180 for v in a.values()), q.stem
    assert shapes == {"cyclic", "arc", "semi"}


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(SKILL, seed)
        assert oracle(q) == canonical(q), (seed, q.stem)


def test_grader_accepts_key_rejects_distractors():
    assert_grading(SKILL)


def test_follow_through_from_wrong_x_and_wrong_angle():
    checked = 0
    for seed in range(150):
        q = exam(SKILL, seed)
        kind = shape_of(q.stem)
        key = canonical(q)
        a = _angles(q.stem)
        angle_name = {"cyclic": "ABC", "arc": "ACB", "semi": "CAB"}[kind]
        last = {"cyclic": lambda b: 180 - b, "arc": lambda b: 2 * b, "semi": lambda b: 90 - b}[kind]
        wrong_x = q.parts[0].distractors[0]
        xv = sympy.Rational(wrong_x)
        b_wrong_x = a[angle_name].subs(x, xv)
        # A fractional x can give a fractional angle; the grader reads replies to the part's own precision.
        # An impossible angle (outside 0 to 180) carries no follow-through, as in a marked exam.
        if b_wrong_x == int(b_wrong_x) and 0 < b_wrong_x < 180 and 0 < last(b_wrong_x) < 360:
            replies = {"a": wrong_x, "b": str(b_wrong_x), "c": str(last(b_wrong_x))}
            assert_follow_through(SKILL, q, replies, "a")
            checked += 1
        wrong_b = q.parts[1].distractors[0]
        replies = {"a": key["a"], "b": wrong_b, "c": str(last(sympy.Rational(wrong_b)))}
        if 0 < sympy.Rational(wrong_b) < 180 and 0 < last(sympy.Rational(wrong_b)) < 360:
            assert_follow_through(SKILL, q, replies, "b")
            checked += 1
    assert checked >= 120


def test_all_wrong_earns_nothing():
    for seed in range(30):
        result = grade(SKILL, exam(SKILL, seed), {"a": "999", "b": "998", "c": "997"})
        assert result.marks == 0


def test_deterministic():
    assert exam(SKILL, 7).to_dict() == exam(SKILL, 7).to_dict()
