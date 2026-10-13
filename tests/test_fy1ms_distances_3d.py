"""Tests the skill `ucv1-distances-and-triple-product-3d`: an independent oracle that re-derives every key from the
question TEXT with SymPy matrices, the typed grader on every key and distractor, and the four-part exam chain with
follow-through."""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen.engine import grade
from tests import fy1ms_support as T

SKILL_ID = "ucv1-distances-and-triple-product-3d"
TOPIC = "fy1ms_distances_3d"
EXAM = "exd_x_fy1ms_distances_3d"
SK = T.load(SKILL_ID)
TIERS = ("easy", "medium", "hard")
x, y, z = sympy.symbols("x y z")


def _vec(text: str) -> sympy.Matrix:
    return sympy.Matrix([int(c) for c in text.split(",")])


def _plane(text: str):
    lhs, rhs = text.split(" = ")
    poly = sympy.Poly(T.parse_plain(lhs, x=x, y=y, z=z), x, y, z)
    return sympy.Matrix([poly.coeff_monomial(v) for v in (x, y, z)]), int(rhs)


def oracle(tier, q) -> str:
    text = q.text
    m = re.match(r"Find the shortest distance from the point P\((.*?)\) to the plane (.*?)\. Give", text)
    if m:
        n, d = _plane(m.group(2))
        return T.rnd(abs(n.dot(_vec(m.group(1))) - d) / n.norm(), 2)
    m = re.match(
        r"Find the shortest distance from the point P\((.*?)\) to the line r = \((.*?)\) \+ t\((.*?)\)\.", text
    )
    if m:
        p, a, u = (_vec(m.group(i)) for i in (1, 2, 3))
        return T.rnd((p - a).cross(u).norm() / u.norm(), 2)
    m = re.search(r"edges a = \((.*?)\), b = \((.*?)\) and c = \((.*?)\) from one corner", text)
    if m:
        a, b, c = (_vec(m.group(i)) for i in (1, 2, 3))
        return str(abs(sympy.Matrix.hstack(a, b, c).det()))
    m = re.match(
        r"Line L₁ has equation r = \((.*?)\) \+ s\((.*?)\) and line L₂ has equation r = \((.*?)\) \+ t\((.*?)\)\.", text
    )
    if m:
        a1, u1, a2, u2 = (_vec(m.group(i)) for i in (1, 2, 3, 4))
        n = u1.cross(u2)
        return T.rnd(abs((a2 - a1).dot(n)) / n.norm(), 2)
    m = re.match(r"A tetrahedron has vertices A\((.*?)\), B\((.*?)\), C\((.*?)\) and D\((.*?)\)\.", text)
    if m:
        a, b, c, d = (_vec(m.group(i)) for i in (1, 2, 3, 4))
        return T.rnd(abs(sympy.Matrix.hstack(b - a, c - a, d - a).det()) / 6, 2)
    m = re.match(
        r"The points A\((.*?)\), B\((.*?)\) and C\((.*?)\) lie in a plane\. Find the shortest distance from D\((.*?)\)",
        text,
    )
    a, b, c, d = (_vec(m.group(i)) for i in (1, 2, 3, 4))
    n = (b - a).cross(c - a)
    return T.rnd(abs(n.dot(d - a)) / n.norm(), 2)


def test_wiring():
    T.assert_wiring(SK, TOPIC, [EXAM])


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(150) if oracle(tier, T.single(TOPIC, tier, s)) != T.single(TOPIC, tier, s).answer_text]
    assert bad == []


def test_each_tier_uses_its_shapes():
    def shape(tier, s):
        text = T.single(TOPIC, tier, s).text
        for key, name in (
            ("to the plane", "plane"),
            ("to the line", "line"),
            ("parallelepiped", "triple"),
            ("do not meet", "skew"),
            ("tetrahedron has", "tetra"),
            ("lie in a plane", "height"),
        ):
            if key in text:
                return name

    assert {shape("easy", s) for s in range(60)} == {"plane"}
    assert {shape("medium", s) for s in range(60)} == {"line", "triple"}
    assert {shape("hard", s) for s in range(80)} == {"skew", "tetra", "height"}


def test_easy_normals_have_a_whole_number_length_and_a_non_trivial_distance():
    for s in range(60):
        q = T.single(TOPIC, "easy", s)
        n, _d = _plane(re.search(r"to the plane (.*?)\. Give", q.text).group(1))
        assert sympy.sqrt(n.dot(n)).is_Integer
        assert float(q.answer_text) > 0


@pytest.mark.parametrize("tier", TIERS)
def test_grading_and_diagnosis(tier):
    T.assert_single_grading(SK, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_clean_text_variety_and_determinism(tier):
    T.assert_clean_variety_and_determinism(TOPIC, tier, distinct_answers=15)


# --- the exam chain --------------------------------------------------------------------------------------------


def exam_oracle(q):
    m = re.search(r"edges a = \((.*?)\), b = \((.*?)\) and c = \((.*?)\) from", q.stem)
    a, b, c = (_vec(m.group(i)) for i in (1, 2, 3))
    triple = sympy.Matrix.hstack(a, b, c).det()
    area = a.cross(b).norm()
    return dict(a=str(triple), b=str(abs(triple)), c=T.rnd(area, 2), d=T.rnd(abs(triple) / area, 2))


def test_exam_shape():
    T.assert_exam_shape(EXAM, ["a", "b", "c", "d"], [2, 1, 2, 2], [False, True, False, True])


def test_exam_keys_match_the_independent_derivation():
    for seed in range(150):
        q = T.exam(EXAM, seed)
        assert {p.label: p.answer_text for p in q.parts} == exam_oracle(q), seed


def test_exam_grading_distractors_and_notes():
    T.assert_exam_grading(SK, EXAM)


def test_follow_through_from_a_wrong_triple_product_and_a_wrong_area():
    T.assert_follow_through_credited(SK, EXAM, minimum=100)


def test_the_rounded_area_can_be_used_downstream():
    for seed in range(60):
        q = T.exam(EXAM, seed)
        area = q.parts[2].answer_text
        height = T.rnd(sympy.Rational(q.parts[1].answer_text) / sympy.Rational(area), 2)
        reply = {**T.canonical_reply(q), "d": height}
        assert grade(SK, q, reply).parts["d"].marks == 2, seed


def test_exam_answer_space():
    T.assert_exam_variety(EXAM, distinct=100, repeat=4)
