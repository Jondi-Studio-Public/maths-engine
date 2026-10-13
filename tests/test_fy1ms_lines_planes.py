"""Tests the skill `ucv1-lines-and-planes-3d`: an independent oracle that re-derives every key from the question TEXT
(SymPy: cross products, solving for the parameter, angles from normalised dot products), the typed grader on every
key and distractor, and the four-part exam chain with follow-through."""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen.engine import grade
from tests import fy1ms_support as T

SKILL_ID = "ucv1-lines-and-planes-3d"
TOPIC = "fy1ms_lines_planes"
EXAM = "exd_x_fy1ms_lines_planes"
SK = T.load(SKILL_ID)
TIERS = ("easy", "medium", "hard")
x, y, z, t = sympy.symbols("x y z t")


def _vec(text: str) -> sympy.Matrix:
    return sympy.Matrix([int(c) for c in text.split(",")])


def _plane(text: str):
    """(normal, constant) of 'ax + by + cz = k'."""
    lhs, rhs = text.split(" = ")
    poly = sympy.Poly(T.parse_plain(lhs, x=x, y=y, z=z), x, y, z)
    return sympy.Matrix([poly.coeff_monomial(v) for v in (x, y, z)]), int(rhs)


def _angle_deg(cosine):
    return sympy.N(sympy.acos(cosine) * 180 / sympy.pi, 30)


def oracle(tier, q) -> str:
    text = q.text
    if tier == "easy":
        m = re.match(r"The plane through A\((.*?)\), B\((.*?)\) and C\((.*?)\) has equation", text)
        a, b, c = (_vec(m.group(i)) for i in (1, 2, 3))
        n = (b - a).cross(c - a)
        return str(n.dot(a))
    if tier == "medium":
        m = re.match(
            r"The line L has equation r = \((.*?)\) \+ t\((.*?)\), and a plane has equation (.*?)\. "
            r"Find the (?:value of t at|(x|y|z)-coordinate of) the point where L meets the plane\.",
            text,
        )
        a, u = _vec(m.group(1)), _vec(m.group(2))
        n, k = _plane(m.group(3))
        t0 = sympy.solve(sympy.Eq(n.dot(a + t * u), k), t)[0]
        if m.group(4) is None:
            return str(t0)
        return str((a + t0 * u)["xyz".index(m.group(4))])
    m = re.match(r"The planes (.*?) and (.*?) meet along a line\. Find the acute angle", text)
    if m:
        (n1, _k1), (n2, _k2) = _plane(m.group(1)), _plane(m.group(2))
        return T.rnd(_angle_deg(abs(n1.dot(n2)) / (n1.norm() * n2.norm())), 1)
    m = re.match(r"The line r = \((.*?)\) \+ t\((.*?)\) meets the plane (.*?)\. Find the acute angle", text)
    n, _k = _plane(m.group(3))
    u = _vec(m.group(2))
    return T.rnd(90 - _angle_deg(abs(n.dot(u)) / (n.norm() * u.norm())), 1)


def test_wiring():
    T.assert_wiring(SK, TOPIC, [EXAM])


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(150) if oracle(tier, T.single(TOPIC, tier, s)) != T.single(TOPIC, tier, s).answer_text]
    assert bad == []


def test_medium_asks_for_t_and_for_each_coordinate_and_hard_for_both_angles():
    asked = set()
    for s in range(100):
        text = T.single(TOPIC, "medium", s).text
        asked.add("t" if "value of t" in text else re.search(r"the (.)-coordinate", text).group(1))
    assert asked == {"t", "x", "y", "z"}
    kinds = {"planes" if "The planes" in T.single(TOPIC, "hard", s).text else "line" for s in range(60)}
    assert kinds == {"planes", "line"}


@pytest.mark.parametrize("tier", TIERS)
def test_grading_and_diagnosis(tier):
    T.assert_single_grading(SK, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_clean_text_variety_and_determinism(tier):
    T.assert_clean_variety_and_determinism(TOPIC, tier, distinct_answers=15)


def test_a_fraction_distractor_is_still_rejected():
    seen = 0
    for s in range(60):
        q = T.single(TOPIC, "medium", s)
        for d in q.distractors:
            if "/" in d:
                seen += 1
                assert not grade(SK, q, d).correct
    assert seen >= 40


# --- the exam chain --------------------------------------------------------------------------------------------


def exam_oracle(q):
    m = re.match(
        r"A flat mirror lies in the plane through P\((.*?)\), Q\((.*?)\) and R\((.*?)\)\. A laser beam travels along the "
        r"line r = \((.*?)\) \+ t\((.*?)\)\.",
        q.stem,
    )
    p, qq, r, a, u = (_vec(m.group(i)) for i in range(1, 6))
    n = (qq - p).cross(r - p)
    d = n.dot(p)
    t0 = sympy.solve(sympy.Eq(n.dot(a + t * u), d), t)[0]
    axis = "xyz".index(re.search(r"find the (.)-coordinate", q.parts[3].prompt).group(1))
    return dict(a="(" + ", ".join(str(c) for c in n) + ")", b=str(d), c=str(t0), d=str((a + t0 * u)[axis]))


def test_exam_shape():
    T.assert_exam_shape(EXAM, ["a", "b", "c", "d"], [2, 1, 2, 1], [False, False, True, True])


def test_exam_keys_match_the_independent_derivation():
    for seed in range(150):
        q = T.exam(EXAM, seed)
        assert {p.label: p.answer_text for p in q.parts} == exam_oracle(q), seed


def test_exam_vector_part_is_positional():
    for seed in range(40):
        q = T.exam(EXAM, seed)
        n = q.parts[0].answer
        assert grade(SK, q, {**T.canonical_reply(q), "a": "<%d, %d, %d>" % tuple(n)}).parts["a"].marks == 2
        if len(set(n)) == 3:
            assert (
                grade(SK, q, {**T.canonical_reply(q), "a": "(%d, %d, %d)" % (n[1], n[0], n[2])}).parts["a"].marks == 0
            )


def test_exam_grading_distractors_and_notes():
    T.assert_exam_grading(SK, EXAM)


def test_follow_through_from_a_wrong_constant_and_a_wrong_parameter():
    T.assert_follow_through_credited(SK, EXAM, minimum=60)


def test_exam_answer_space():
    T.assert_exam_variety(EXAM, distinct=100, repeat=4)
