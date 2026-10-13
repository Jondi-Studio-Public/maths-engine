"""Tests the multi-part exam tier of `ucv1-iterative-linear-solver-step`: a Gauss-Seidel step, residual, error."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_gauss_seidel"
SKILL_ID = "ucv1-iterative-linear-solver-step"
_SKILL = S.skill(SKILL_ID)
_EQUATION = re.compile(r"(-?\d*)x ([+-]) (\d*)y = (-?\d+)")


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _coefficient(text: str) -> int:
    return {"": 1, "-": -1}.get(text, None) or int(text)


def _system(q):
    rows = []
    for ax, sign, ay, rhs in _EQUATION.findall(q.stem):
        rows.append((_coefficient(ax), (-1 if sign == "-" else 1) * _coefficient(ay), int(rhs)))
    x0, y0 = (int(v) for v in re.search(r"\(x0, y0\) = \((-?\d+), (-?\d+)\)", q.stem).groups())
    return rows, x0, y0


def _frac(text: str) -> Fraction:
    return Fraction(text)


def _text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abcd", [2, 2, 1, 2])


def test_keys_match_independent_iteration_over_300_seeds(questions):
    for q in questions:
        rows, x0, y0 = _system(q)
        assert len(rows) == 2, q.stem
        x_row = next(r for r in rows if abs(r[0]) > abs(r[1]))
        y_row = next(r for r in rows if r is not x_row)
        assert abs(y_row[1]) > abs(y_row[0])
        x1 = Fraction(x_row[2] - x_row[1] * y0, x_row[0])
        y1 = Fraction(y_row[2] - y_row[0] * x1, y_row[1])
        residual = x_row[2] - x_row[0] * x1 - x_row[1] * y1
        det = x_row[0] * y_row[1] - x_row[1] * y_row[0]
        x_exact = Fraction(x_row[2] * y_row[1] - x_row[1] * y_row[2], det)
        assert x_exact.denominator == 1 and f"x = {x_exact}" in q.parts[3].prompt
        assert [_frac(p.answer_text) for p in q.parts] == [x1, y1, residual, abs(x1 - x_exact)], q.stem
        assert y_row[2] - y_row[0] * x1 - y_row[1] * y1 == 0


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_a_wrong_x1(questions):
    checked = 0
    for q in questions[:120]:
        rows, x0, y0 = _system(q)
        x_row = next(r for r in rows if abs(r[0]) > abs(r[1]))
        y_row = next(r for r in rows if r is not x_row)
        det = x_row[0] * y_row[1] - x_row[1] * y_row[0]
        x_exact = Fraction(x_row[2] * y_row[1] - x_row[1] * y_row[2], det)
        wrong_x = _frac(q.parts[0].distractors[0])
        y1 = Fraction(y_row[2] - y_row[0] * wrong_x, y_row[1])
        residual = x_row[2] - x_row[0] * wrong_x - x_row[1] * y1
        error = abs(wrong_x - x_exact)
        reply = {"a": q.parts[0].distractors[0], "b": _text(y1), "c": _text(residual), "d": _text(error)}
        if wrong_x == 0 or y1 == 0 or residual == 0 or error == 0 or len(set(reply.values())) < 4:
            continue
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.stem, label, reply)
        checked += 1
    assert checked >= 40


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "101/7", "b": "-83", "c": "59/11", "d": "97"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
