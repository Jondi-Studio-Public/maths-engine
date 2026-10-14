"""Tests for the multi-part exam source `exd_x_r3g_slope_euler` (two Euler steps along a slope-field curve)."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_slope_euler"
SKILL = "senior-examine-slope-direction-or-gradient-fields-of-a"
X, Y = sympy.symbols("x y")
_T = standard_transformations + (implicit_multiplication_application,)


def _facts(q):
    text = re.search(r"dy/dx = ([^,]*?)(?:,| passes| is used)", q.text).group(1)
    f = parse_expr(text, transformations=_T, local_dict={"x": X, "y": Y})
    x0, y0 = (int(v) for v in re.search(r"\((-?\d+), (-?\d+)\)", q.text).groups())
    h = Fraction(re.search(r"step size (?:of )?([0-9.]+)", q.text).group(1))

    def slope(x, y):
        value = f.subs({X: sympy.Rational(x.numerator, x.denominator), Y: sympy.Rational(y.numerator, y.denominator)})
        return Fraction(int(value.p), int(value.q))

    return slope, Fraction(x0), Fraction(y0), h


def _two(value: Fraction) -> str:
    return f"{float(value):.2f}"


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(TOPIC, "p13b_slope_field_examine", {"variant": "classify_point"})


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        slope, x0, y0, h = _facts(q)
        s0 = slope(x0, y0)
        y1 = y0 + h * s0
        s1 = slope(x0 + h, y1)
        y2 = y1 + h * s1
        want = {"a": str(s0), "b": _two(y1), "c": _two(s1), "d": _two(y2)}
        if sup.canonical(q) != want:
            bad.append((q.text, sup.canonical(q), want))
    assert bad == [], bad[:2]


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def _chain(slope, x0, y0, h, s0):
    y1 = y0 + h * s0
    s1 = slope(x0 + h, y1)
    return y1, s1, y1 + h * s1


def test_follow_through_from_a_wrong_first_gradient():
    def replies(q):
        slope, x0, y0, h = _facts(q)
        # a wrong gradient whose consequences are all non-zero: markers withhold follow-through from a zero
        s0 = next(
            Fraction(d)
            for d in q.parts[0].distractors
            if all(v != 0 for v in _chain(slope, x0, y0, h, Fraction(d))) and Fraction(d) != 0
        )
        y1, s1, y2 = _chain(slope, x0, y0, h, s0)
        return {"a": str(s0), "b": _two(y1), "c": _two(s1), "d": _two(y2)}, {"b", "c", "d"}, {"a"}

    sup.check_follow_through(SKILL, TOPIC, replies)


def test_follow_through_from_a_wrong_estimate():
    def replies(q):
        slope, x0, y0, h = _facts(q)
        y1 = next(
            Fraction(d)
            for d in q.parts[1].distractors
            if Fraction(d) != 0
            and slope(x0 + h, Fraction(d)) != 0
            and Fraction(d) + h * slope(x0 + h, Fraction(d)) != 0
        )
        s1 = slope(x0 + h, y1)
        y2 = y1 + h * s1
        return {**sup.canonical(q), "b": _two(y1), "c": _two(s1), "d": _two(y2)}, {"c", "d"}, {"b"}

    sup.check_follow_through(SKILL, TOPIC, replies)
