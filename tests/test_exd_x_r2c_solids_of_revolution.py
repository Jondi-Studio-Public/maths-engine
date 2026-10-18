"""Tests the exam chain for volumes of solids of revolution: intersections, cross-section, exact volume."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade
from tests.exd_x_r2c_support import (
    canonical,
    check_follow_through,
    check_answer_space,
    check_clean_and_deterministic,
    check_grading,
    check_shapes,
    check_wired,
    exam,
    skill,
)

TOPIC = "exd_x_r2c_solids_of_revolution"
SKILL_ID = "senior-determine-volumes-of-solids-of-revolution-about-either"
HARD_TOPIC = "determine_volumes_of_solids_of_revolution_about_either"
x = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": x})


def oracle(q):
    """Re-derive all four answers from the rendered stem alone."""
    found = re.search(r"curve y = (.+?) and the line y = (.+?)(?: is rotated|\.)", q.stem)
    curve, line = _parse(found.group(1)), _parse(found.group(2))
    axis = re.search(r"about (?:the line y = -(\d+)|the x-axis)", q.stem)
    shift = int(axis.group(1) or 0)
    roots = sorted(sympy.solve(sympy.Eq(curve, line), x))
    lo, hi = roots
    area = sympy.pi * ((line + shift) ** 2 - (curve + shift) ** 2)
    volume = sympy.integrate(area, (x, lo, hi))
    return lo, hi, area, sympy.simplify(volume)


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "expression", "expression"]
        assert [p.marks for p in q.parts] == [1, 1, 2, 3]
        assert q.parts[3].follow_through and "part_a" in q.parts[3].follow_through


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        lo, hi, area, volume = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        ok = (
            int(a) == lo
            and int(b) == hi
            and sympy.simplify(_parse(c) - area) == 0
            and sympy.simplify(_parse(d) - volume) == 0
        )
        if not ok:
            bad.append((q.stem, canonical(q), (lo, hi, area, volume)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_wrong_limits():
    def carried(q):
        a_wrong = str(int(q.parts[0].answer_text) + 5)  # a lower limit far too large
        area = _parse(q.parts[2].answer_text)
        antiderivative = sympy.integrate(sympy.expand(area), x)
        value = antiderivative.subs(x, int(q.parts[1].answer_text)) - antiderivative.subs(x, int(a_wrong))
        return "a", a_wrong, {"d": str(value)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_follow_through_from_both_limits_wrong():
    for seed in range(40):
        q = exam(TOPIC, seed)
        a_wrong, b_wrong = int(q.parts[0].answer_text) + 1, int(q.parts[1].answer_text) + 2
        area = _parse(q.parts[2].answer_text)
        antiderivative = sympy.integrate(sympy.expand(area), x)
        reply = str(antiderivative.subs(x, b_wrong) - antiderivative.subs(x, a_wrong))
        result = grade(skill(SKILL_ID), q, {**canonical(q), "a": str(a_wrong), "b": str(b_wrong), "d": reply})
        assert result.parts["d"].marks == 3 and result.parts["d"].follow_through
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        assert result.marks == 5 and not result.correct
