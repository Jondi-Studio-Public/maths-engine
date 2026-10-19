"""Tests the parametric-to-cartesian-conics skill: keys checked against the stem's own equations, grading, the exam chain."""

from __future__ import annotations

import math
import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3i_support as sup

SKILL = "parametric-to-cartesian-conics"
TOPIC = "r3i_parametric_conics"
EXAM = "r3i_exam_parametric_conics"
X, Y, T = sympy.symbols("x y t")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SAMPLES = (0.37, 0.81, 1.13, 1.79, 2.41, 3.3, 4.2, 5.1)


def _read(text):
    text = text.replace("²", "**2").replace("^", "**").replace("π", "pi").replace("θ", "t")
    return parse_expr(text, local_dict={"x": X, "y": Y, "t": T}, transformations=_TRANSFORMS)


def _holds(x_of_t, y_of_t, target, key):
    """key(x(t)) equals target(y(t)) at every sample of t where both sides are finite."""
    checked = 0
    for t in _SAMPLES:
        try:
            xv, yv = float(x_of_t.subs(T, t)), float(y_of_t.subs(T, t))
            lhs, rhs = float(key.subs(X, xv)), float(target.subs(Y, yv))
        except (ZeroDivisionError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (lhs, rhs)):
            continue
        assert math.isclose(lhs, rhs, rel_tol=1e-6, abs_tol=1e-6), (t, lhs, rhs)
        checked += 1
    return checked


def _check_parametric(q):
    found = re.search(r"x = (.+?), y = (.+?)\. Find (.+?) in terms of x\.", q.text)
    x_of_t, y_of_t, target = _read(found.group(1)), _read(found.group(2)), _read(found.group(3))
    assert _holds(x_of_t, y_of_t, target, _read(q.answer_text)) >= 5, (q.text, q.answer_text)


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "expression", ["senior-use-vector-equations-of-curves-in-two-or"])


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_satisfy_the_parametric_equations(tier):
    shapes = set()
    for seed in range(150):
        q = sup.tier_question(TOPIC, tier, seed)
        if "polar" in q.text:
            shapes.add("polar")
            m, p = re.search(r"r\(cos\(θ\) ([+-]) (\d+)sin\(θ\)\) = (-?\d+)", q.text).group(1, 2), None
            sign = 1 if m[0] == "+" else -1
            slope, rhs = sign * int(m[1]), int(re.search(r"\) = (-?\d+),", q.text).group(1))
            key = _read(q.answer_text)
            # every point (x, y) with x + slope*y = rhs lies on the polar line, and the key must put y there
            for xv in (-3.0, 0.5, 2.0, 7.0):
                yv = float(key.subs(X, xv))
                assert math.isclose(xv + slope * yv, rhs, abs_tol=1e-9), (seed, q.text, q.answer_text)
            continue
        found = re.search(r"(sec|cos\(t\)|sin\(t\)|1/t|t²|cos\(2t\))", q.text)
        shapes.add(found.group(1) if found else "line")
        _check_parametric(q)
    assert len(shapes) >= (2 if tier == "easy" else 3), shapes


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=20, top=0.1)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(80)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(80)]
    assert not any("cos" in t or "sec" in t for t in easy)
    assert (
        any("sec(t)" in t for t in medium) and any("polar" in t for t in medium) and any("cos(t)" in t for t in medium)
    )
    assert any("(1 - t²)" in t for t in hard) and any("1/t" in t for t in hard) and any("cos(2t)" in t for t in hard)
    assert any("(y " in t for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["number", "number", "expression", "number"], [1, 1, 3, 2], [False, False, False, True]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=15, top=0.15)


def test_exam_keys_match_the_parametric_equations():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        a, b = (int(v) for v in re.search(r"x = (\d+)cos\(t\), y = (\d+)sin\(t\)", q.text).groups())
        t0 = math.pi / 3
        assert math.isclose(a * math.cos(t0), float(q.parts[0].answer_text), abs_tol=1e-9)
        assert math.isclose((b * math.sin(t0)) ** 2, float(q.parts[1].answer_text), abs_tol=1e-9)
        key = _read(q.parts[2].answer_text)
        assert _holds(_read(f"{a}cos(t)"), _read(f"{b}sin(t)"), Y**2, key) >= 6
        shift = int(re.search(r"x is (\d+) more", q.parts[3].prompt).group(1))
        x_new = float(q.parts[0].answer_text) + shift
        assert math.isclose(float(key.subs(X, x_new)), float(q.parts[3].answer_text), abs_tol=1e-9)
        assert str(shift) in q.parts[3].follow_through


def test_exam_follow_through_from_a_wrong_first_part():
    sk = sup.skill_object(SKILL)
    usable = 0
    for seed in range(30):
        q = sup.exam_question(EXAM, seed)
        a, b = (int(v) for v in re.search(r"x = (\d+)cos\(t\), y = (\d+)sin\(t\)", q.text).groups())
        shift = int(re.search(r"x is (\d+) more", q.parts[3].prompt).group(1))
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong in (1, 3, 5, 7, 9):
            if wrong == int(canon["a"]):
                continue
            carried = sympy.Rational(b * b) * (1 - sympy.Rational(wrong + shift, a) ** 2)
            replies = {**canon, "a": str(wrong), "d": str(carried)}
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0
            if result.parts["d"].marks == q.parts[3].marks:
                usable += 1
                break
    assert usable >= 20
