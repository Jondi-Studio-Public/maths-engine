"""Tests the exam chain for comparing linear relationships: rule for A from its graph, gradient of B, gap, meeting x.

The oracle never reads the key: it re-derives every part from the drawn figure (relationship A) and the stem
(relationship B as a table or a worded rule), then compares with the question's own answers.
"""

from __future__ import annotations

import re
from fractions import Fraction as F

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import figures, make
from mathgen.generators.pv1a_line_figure_build import read_line
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_w12c_compare_linear"
SKILL_ID = "compare-linear-representations"
HARD_TOPIC = "fg1_compare_linear"
HARD_KWARGS = {"tier": "hard"}
x = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": x})


def _relationship_b(stem: str) -> tuple[F, F]:
    """(gradient, intercept) of B from a table of pairs or a worded rule."""
    pairs = re.findall(r"\((-?\d+), (-?\d+)\)", stem)
    if pairs:
        pts = [(F(a), F(b)) for a, b in pairs]
        (x1, y1), (x2, y2) = pts[0], pts[1]
        m = (y2 - y1) / (x2 - x1)
        assert all(yy == m * xx + (y1 - m * x1) for xx, yy in pts), "table is not linear"
        return m, y1 - m * x1
    rule = re.search(r"multiply?(?:ing)? x by (-?\d+)(?: and)?,? then (add|adding|subtract|subtracting) (\d+)", stem)
    assert rule, stem
    c = int(rule.group(3)) * (1 if rule.group(2).startswith("add") else -1)
    return F(int(rule.group(1))), F(c)


def oracle(q):
    mA, cA = read_line(q.figure)
    mB, cB = _relationship_b(q.stem)
    far = int(re.search(r"when x = (-?\d+)", q.parts[2].prompt).group(1))
    gap = abs((mA * far + cA) - (mB * far + cB))
    meet = (cB - cA) / (mA - mB)
    return mA, cA, mB, cB, far, gap, meet


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    check_shapes(TOPIC, HARD_TOPIC, HARD_KWARGS)


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "number", "number"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.sensible_range is not None for p in q.parts if p.answer_kind == "number")


def test_figure_is_valid_and_hides_the_rule(questions):
    for q in questions[:100]:
        assert q.figure is not None and figures.validate(q.figure) == []
        figures.assert_no_leak(q.figure, [p.answer_text for p in q.parts])
        assert len(q.figure["lines"]) == 1


def test_both_presentations_of_b_occur(questions):
    tables = sum("(x, y) pairs" in q.stem or "pairs" in q.stem for q in questions)
    assert 0.25 < tables / len(questions) < 0.85


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        mA, cA, mB, cB, far, gap, meet = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        rule = _parse(a_t)
        ok = (
            sympy.simplify(rule - (sympy.Rational(mA.numerator, mA.denominator) * x + sympy.Rational(cA))) == 0
            and F(b_t) == mB
            and F(c_t) == gap
            and gap > 0
            and F(d_t) == meet
            and meet.denominator == 1
            and mA != mB
        )
        if not ok:
            bad.append((q.stem, canonical(q)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_a_wrong_rule():
    """A wrong rule for A carries through the gap at the far x and the meeting point."""

    def carried(q):
        mA, cA, mB, cB, far, gap, meet = oracle(q)
        canon_sign = 1 if (mA * far + cA) > (mB * far + cB) else -1
        for dm in (1, -1, 2, -2, 3, -3):
            m2 = round(mA) + dm
            if m2 == 0 or F(m2) == mB:
                continue
            for t in (int(meet) + 2, int(meet) - 2, int(meet) + 3, int(meet) - 3):
                if abs(t) < 2 or t == int(mB):  # a follow-through answer of 0 or +-1 is not "real work" to the grader
                    continue
                c2 = int(cB - (m2 - mB) * t)
                value = (m2 * far + c2) - (mB * far + cB)
                if (
                    value == 0 or (1 if value > 0 else -1) != canon_sign or abs(value) in (1, abs(int(mB)), abs(t))
                ):  # not a repeat of another reply
                    continue
                wrong = f"{m2}x + ({c2})"
                return "a", wrong, {"c": str(abs(value)), "d": str(t)}
        raise AssertionError("no usable wrong rule")

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")
