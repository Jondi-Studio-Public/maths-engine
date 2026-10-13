"""Exam chain of the Riemann-sum area skill: rectangle estimate, exact area, absolute error."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_riemann_error_chain"
SKILL = S.load_skill("senior-examine-the-area-problem-and-use-sums-of")
_X = sympy.Symbol("x")
_TR = standard_transformations + (implicit_multiplication_application, convert_xor)


def _read(q):
    f = (
        re.search(r"f\(x\) = (.+?), lies", q.stem)
        or re.search(r"f\(x\) = (.+?)\. The area", q.stem)
        or re.search(r"y = (.+?), the x-axis", q.stem)
    ).group(1)
    ends = re.search(r"x = (\d+) and x = (\d+)", q.stem) or re.search(r"(\d+) ≤ x ≤ (\d+)", q.stem)
    n = int(re.search(r"(\d+) rectangles", q.stem).group(1))
    rule = re.search(r"(left|right|midpoint)", q.parts[0].prompt).group(1)
    return parse_expr(f, transformations=_TR, local_dict={"x": _X}), int(ends.group(1)), int(ends.group(2)), n, rule


def _values(q):
    f, a, b, n, rule = _read(q)
    width = sympy.Rational(b - a, n)
    shift = {"left": 0, "right": 1, "midpoint": sympy.Rational(1, 2)}[rule]
    estimate = sum(f.subs(_X, a + (i + shift) * width) for i in range(n)) * width
    exact = sympy.integrate(f, (_X, a, b))
    return sympy.Rational(estimate), exact


def _text(v) -> str:
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def oracle(q):
    estimate, exact = _values(q)
    return {"a": _text(estimate), "b": _text(exact), "c": _text(abs(estimate - exact))}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_all_three_rules_and_fractions_occur(questions):
    rules = {_read(q)[4] for q in questions}
    assert rules == {"left", "right", "midpoint"}
    assert any("/" in q.parts[1].answer_text for q in questions)
    assert any("/" not in q.parts[1].answer_text for q in questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_curve_stays_above_axis(questions):
    for q in questions[:80]:
        f, a, b, n, rule = _read(q)
        assert all(f.subs(_X, sympy.Rational(a) + sympy.Rational(k, 2) * (b - a) / n) > 0 for k in range(2 * n + 1))


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_estimate(questions):
    def chain(q):
        estimate = S.frac(q.parts[0].distractors[0])
        exact = S.frac(q.parts[1].answer_text)
        error = abs(estimate - exact)
        if error == 0 or error in (estimate, exact):
            return None
        return {"a": q.parts[0].distractors[0], "b": q.parts[1].answer_text, "c": _text(error)}, {"b", "c"}

    S.check_follow_through(SKILL, questions, chain, n=200)


def test_follow_through_from_wrong_exact_area(questions):
    seen = 0
    for q in questions[:100]:
        estimate = S.frac(q.parts[0].answer_text)
        wrong = S.frac(q.parts[1].distractors[0])
        error = abs(estimate - wrong)
        if error == 0 or wrong <= 0 or error in (estimate, wrong):
            continue
        result = S.grade(SKILL, q, {"a": q.parts[0].answer_text, "b": q.parts[1].distractors[0], "c": _text(error)})
        assert result.parts["a"].marks == q.parts[0].marks and result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks
        seen += 1
    assert seen >= 40


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "9999", "b": "9998", "c": "9997"})


def test_deterministic():
    S.check_deterministic(TOPIC)
