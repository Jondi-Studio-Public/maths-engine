"""Exam source of `partial-derivative-x` and `partial-derivative-y`: a partial derivative of a product, its value
at a point, and a linear estimate from it."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
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

x, y = sympy.symbols("x y")
CASES = [("partial-derivative-x", "x"), ("partial-derivative-y", "y")]


def read(q):
    """(f, variable, point, step) parsed from the rendered question alone."""
    stem, rest = q.text.split(" (a)")
    f_text = re.search(r"f\(x, y\) = (.+?)(?:, and let P| at the point P|\. The point P)", stem).group(1)
    var = re.search(r"Find ∂f/∂([xy])", rest).group(1)
    p, q_ = re.search(r"at P\((-?\d+), (-?\d+)\)", rest).groups()
    step = Fraction(re.search(r"increases by ([\d.]+)", rest).group(1))
    return parse_expression(f_text, x=x, y=y), var, {x: int(p), y: int(q_)}, step


def oracle(q):
    f, var, point, step = read(q)
    symbol = x if var == "x" else y
    partial = sympy.diff(f, symbol)
    slope = rational(partial.subs(point))
    return partial, slope, slope * step


@pytest.mark.parametrize(("skill_id", "var"), CASES)
def test_source_wired(skill_id, var):
    (source,) = exam_sources(skill_id)
    assert source.topic == "exd_x_u5_partial_chain"
    assert source.params == {"tier": "exam", "var": var}
    assert source.stages == ["exam"]


@pytest.mark.parametrize(("skill_id", "var"), CASES)
def test_shapes_and_text(skill_id, var):
    check_shapes(skill_id)
    check_no_text_defects(skill_id)


@pytest.mark.parametrize(("skill_id", "var"), CASES)
def test_key_rederived_from_prompt(skill_id, var):
    for seed in range(300):
        q = exam_question(skill_id, seed)
        partial, slope, estimate = oracle(q)
        shown = parse_expression(q.parts[0].answer_text, x=x, y=y)
        assert sympy.expand(shown - partial) == 0, (seed, q.text)
        assert as_fraction(q.parts[1].answer_text) == slope, (seed, q.text)
        assert as_fraction(q.parts[2].answer_text) == estimate, (seed, q.text)
        assert f"∂f/∂{var}" in q.text


@pytest.mark.parametrize(("skill_id", "var"), CASES)
def test_grader_accepts_key_rejects_distractors(skill_id, var):
    check_key_and_distractors(skill_id, seeds=60)


@pytest.mark.parametrize(("skill_id", "var"), CASES)
def test_equivalent_forms_of_the_derivative_are_accepted(skill_id, var):
    skill = skill_of(skill_id)
    for seed in range(20):
        q = exam_question(skill_id, seed)
        partial, _, _ = oracle(q)
        expanded = str(sympy.expand(partial)).replace("**", "^")
        result = grade(skill, q, {**canonical_reply(q), "a": expanded})
        assert result.parts["a"].marks == 2, (seed, expanded)


@pytest.mark.parametrize(("skill_id", "var"), CASES)
def test_follow_through_from_wrong_derivative(skill_id, var):
    skill = skill_of(skill_id)
    checked = 0
    for seed in range(40):
        q = exam_question(skill_id, seed)
        f, _, point, step = read(q)
        wrong_text = q.parts[0].distractors[0]
        wrong = parse_expression(wrong_text, x=x, y=y)
        value = rational(wrong.subs(point))
        if value == 0 or not survives_retarget(value, value * step):
            continue
        if value == as_fraction(q.parts[1].answer_text) or value * step == value:
            continue
        checked += 1
        result = grade(skill, q, {"a": wrong_text, "b": str(value), "c": str(value * step)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["c"].marks == 2, (seed, wrong_text)
        assert result.parts["b"].follow_through and result.parts["c"].follow_through
    assert checked >= 15


@pytest.mark.parametrize(("skill_id", "var"), CASES)
def test_all_wrong_earns_nothing(skill_id, var):
    q = exam_question(skill_id, 2)
    assert grade(skill_of(skill_id), q, {"a": "x + 1", "b": "9999", "c": "9998"}).marks == 0
