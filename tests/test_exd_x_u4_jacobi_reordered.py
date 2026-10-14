"""Exam source of `jacobi-iteration-2x2`: two Jacobi sweeps for a system listed with its dominant entries off-diagonal."""

from __future__ import annotations

import itertools
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

TOPIC = "exd_x_u4_jacobi_reordered"
SKILL_ID = "jacobi-iteration-2x2"
_X, _Y = sympy.symbols("x y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _system(text: str):
    rows = []
    for number in ("1", "2"):
        body = re.search(rf"\({number}\) (.+?) = (-?\d+)[;.]", text)
        expr = parse_expr(body.group(1), transformations=_TRANSFORMS, local_dict={"x": _X, "y": _Y})
        rows.append((expr.coeff(_X), expr.coeff(_Y), sympy.Integer(body.group(2))))
    x0, y0 = (sympy.Integer(v) for v in re.search(r"\(x₀, y₀\) = \((-?\d+), (-?\d+)\)", text).groups())
    return rows, x0, y0


def _oracle(text: str) -> dict[str, Fraction]:
    rows, x0, y0 = _system(text)
    # choose the row order that makes the coefficient matrix diagonally dominant
    for order in itertools.permutations(range(2)):
        (a, b, r1), (c, d, r2) = (rows[i] for i in order)
        if abs(a) > abs(b) and abs(d) > abs(c):
            break
    else:
        raise AssertionError("no dominant arrangement")
    matrix = sympy.Matrix([[a, b], [c, d]])
    rhs = sympy.Matrix([r1, r2])
    diagonal = sympy.diag(matrix[0, 0], matrix[1, 1])
    off = matrix - diagonal
    state = sympy.Matrix([x0, y0])
    sweeps = []
    for _ in range(2):
        state = diagonal.inv() * (rhs - off * state)
        sweeps.append(state)

    def to_fraction(v):
        return Fraction(int(v.p), int(v.q))

    (x1, y1), (x2, y2) = sweeps
    return {"a": to_fraction(x1), "b": to_fraction(y1), "c": to_fraction(x2), "d": to_fraction(y2)}


def test_prompt_variety_and_new_shape():
    exam_shapes = shapes(TOPIC, 60, difficulty="exam")
    hard = shapes("jacobi_iteration_2x2", 60, difficulty="hard")
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(TOPIC, seed)
        assert {p.label: frac(p.answer_text) for p in q.parts} == _oracle(q.text), (seed, q.text)


def test_listed_order_is_not_dominant():
    for seed in range(60):
        rows, _, _ = _system(exam(TOPIC, seed).text)
        (a, b, _), (c, d, _) = rows
        assert abs(a) < abs(b) and abs(d) < abs(c)


def test_structure():
    for seed in range(60):
        q = exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors_rejected(SKILL_ID, TOPIC)
    check_tagged(TOPIC)


def test_follow_through_from_wrong_first_sweep():
    sk = skill(SKILL_ID)
    tested = 0
    for seed in range(100):
        q = exam(TOPIC, seed)
        rows, _, _ = _system(q.text)
        (p, qq, r1), (s, t, r2) = rows
        a = frac(q.parts[0].answer_text) + Fraction(1, 2)
        b = frac(q.parts[1].answer_text) - Fraction(1, 3)
        c = (int(r2) - int(t) * b) / int(s)
        d = (int(r1) - int(p) * a) / int(qq)
        if len({a, b, c, d}) < 4 or 0 in (a, b) or not survives_nsimplify(a, b, c, d):
            continue
        tested += 1
        result = grade(sk, q, {"a": str(a), "b": str(b), "c": str(c), "d": str(d)})
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        for label in "cd":
            assert result.parts[label].marks == result.parts[label].max_marks
            assert result.parts[label].follow_through is True
        assert not result.correct
    assert tested >= 60


def test_all_wrong_earns_zero():
    sk = skill(SKILL_ID)
    for seed in range(40):
        assert grade(sk, exam(TOPIC, seed), {"a": "999", "b": "998", "c": "997", "d": "996"}).marks == 0


def test_deterministic():
    assert exam(TOPIC, 7).to_json() == exam(TOPIC, 7).to_json()
