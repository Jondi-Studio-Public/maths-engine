"""Exam source of `ucv1-derivative-definition`: a constant found from a gradient, then a gradient and a tangent."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from tests.exd_x_u5_support import (
    as_fraction,
    canonical_reply,
    check_key_and_distractors,
    check_no_text_defects,
    check_shapes,
    exam_question,
    exam_sources,
    parse_expression,
    rational,
    skill_of,
    survives_retarget,
)

from mathgen.engine import grade

SKILL = "ucv1-derivative-definition"
x, k = sympy.symbols("x k")


def read(q):
    stem, rest = q.text.split(" (a)")
    f_text = re.search(r"(?:y|f\(x\)) = (x³ [^,]+?), where k|(?:f\(x\)) = (x³ .+?) contains", stem)
    f = parse_expression(next(g for g in f_text.groups() if g), x=x, k=k)
    a = int(re.search(r"x = (-?\d+)", stem).group(1))
    m = int(re.search(r"(-?\d+)\.$", stem).group(1))
    b = int(re.search(r"\(b\)", q.text) and re.search(r"gradient of the curve at x = (-?\d+)", q.text).group(1))
    return f, a, m, b


def oracle(q):
    f, a, m, b = read(q)
    (k_value,) = sympy.solve(sympy.diff(f, x).subs(x, a) - m, k)
    g = f.subs(k, k_value)
    slope = sympy.diff(g, x).subs(x, b)
    intercept = g.subs(x, b) - b * slope
    return rational(k_value), rational(slope), rational(intercept)


def test_source_wired():
    (source,) = exam_sources(SKILL)
    assert source.topic == "exd_x_u5_derivative_definition"
    assert source.stages == ["exam"]


def test_shapes_and_text():
    check_shapes(SKILL)
    check_no_text_defects(SKILL)


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam_question(SKILL, seed)
        got = tuple(as_fraction(p.answer_text) for p in q.parts)
        assert got == oracle(q), (seed, q.text)


def test_grader_accepts_key_rejects_distractors():
    check_key_and_distractors(SKILL)


def test_follow_through_from_wrong_constant():
    skill = skill_of(SKILL)
    checked = 0
    for seed in range(60):
        q = exam_question(SKILL, seed)
        f, a, m, b = read(q)
        wrong_k = as_fraction(q.parts[0].distractors[0])
        g = f.subs(k, sympy.Rational(wrong_k.numerator, wrong_k.denominator))
        slope = rational(sympy.diff(g, x).subs(x, b))
        intercept = rational(g.subs(x, b) - b * sympy.diff(g, x).subs(x, b))
        if 0 in (wrong_k, slope, intercept) or len({wrong_k, slope, intercept}) < 3:
            continue
        if not survives_retarget(slope, intercept):
            continue
        checked += 1
        result = grade(skill, q, {"a": q.parts[0].distractors[0], "b": str(slope), "c": str(intercept)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["c"].marks == 2, (seed, wrong_k)
        assert result.marks == 3
    assert checked >= 20


def test_follow_through_from_wrong_gradient():
    skill = skill_of(SKILL)
    for seed in range(40):
        q = exam_question(SKILL, seed)
        f, a, m, b = read(q)
        k_value = as_fraction(q.parts[0].answer_text)
        g = f.subs(k, sympy.Rational(k_value.numerator, k_value.denominator))
        wrong_slope = as_fraction(q.parts[1].distractors[0])
        intercept = rational(g.subs(x, b) - b * sympy.Rational(wrong_slope.numerator, wrong_slope.denominator))
        if (
            0 in (wrong_slope, intercept, k_value)
            or not survives_retarget(intercept)
            or len({k_value, wrong_slope, intercept}) < 3
        ):
            continue
        reply = {**canonical_reply(q), "b": q.parts[1].distractors[0], "c": str(intercept)}
        result = grade(skill, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2, (seed, reply)


def test_all_wrong_earns_nothing():
    q = exam_question(SKILL, 4)
    assert grade(skill_of(SKILL), q, {"a": "999", "b": "998", "c": "997"}).marks == 0
