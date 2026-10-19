"""Tests the line-sphere-and-plane-intersection skill: keys re-derived with SymPy from the stem, grading, the exam chain."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3i_support as sup

SKILL = "line-sphere-and-plane-intersection"
TOPIC = "r3i_line_sphere"
EXAM = "r3i_exam_line_sphere"
X, Y, Z, L = sympy.symbols("x y z lam")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_LINE = re.compile(r"r = \(([^)]*)\) \+ λ\(([^)]*)\)")


def _read(text):
    text = text.replace("²", "**2").replace("λ", "lam")
    return parse_expr(text, local_dict={"x": X, "y": Y, "z": Z, "lam": L}, transformations=_TRANSFORMS)


def _equation(text):
    left, right = text.split(" = ")
    return _read(left) - _read(right)


def _vec(text):
    return tuple(sympy.Integer(int(v)) for v in text.split(","))


def _line(stem):
    p, v = _LINE.search(stem).groups()
    return _vec(p), _vec(v)


def _on_line(p, v):
    return {s: p[i] + L * v[i] for i, s in enumerate((X, Y, Z))}


def _answer(q):
    return sympy.Rational(Fraction(q.answer_text).numerator, Fraction(q.answer_text).denominator)


def _plane_and_point(stem):
    plane = re.search(r"plane (.+?)\.", stem).group(1)
    return _equation(plane)


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "number",
        [
            "senior-determine-a-vector-equation-of-a-straight-line",
            "senior-determine-vector-and-cartesian-equations-of-a-plane",
        ],
    )


def test_easy_distance_matches_the_foot_of_the_perpendicular():
    for seed in range(150):
        q = sup.tier_question(TOPIC, "easy", seed)
        pt = _vec(re.search(r"point \(([^)]*)\)", q.text).group(1))
        plane = _plane_and_point(q.text)
        normal = [plane.coeff(s) for s in (X, Y, Z)]
        s = sympy.Symbol("s")
        foot = {sym: pt[i] + s * normal[i] for i, sym in enumerate((X, Y, Z))}
        (root,) = sympy.solve(plane.subs(foot), s)
        expected = sympy.sqrt(sum((root * c) ** 2 for c in normal))
        assert sympy.simplify(expected - _answer(q)) == 0, (seed, q.text, q.answer_text)


def test_medium_keys_match_the_intersection():
    seen = set()
    for seed in range(150):
        q = sup.tier_question(TOPIC, "medium", seed)
        plane = _plane_and_point(q.text)
        if "through A" in q.text:
            a, b = (_vec(m) for m in re.findall(r"[AB]\(([^)]*)\)", q.text))
            p, v = a, tuple(bb - aa for aa, bb in zip(a, b))
            seen.add("two points")
        else:
            p, v = _line(q.text)
            seen.add("lambda" if "value of λ" in q.text else "coordinate")
        (lam,) = sympy.solve(plane.subs(_on_line(p, v)), L)
        if "value of λ" in q.text:
            expected = lam
        else:
            axis = "xyz".index(re.search(r"the ([xyz])-coordinate", q.text).group(1))
            expected = p[axis] + lam * v[axis]
        assert expected == _answer(q), (seed, q.text, q.answer_text)
    assert seen == {"lambda", "coordinate", "two points"}


def test_hard_keys_match_the_intersection_with_the_sphere():
    seen = set()
    for seed in range(150):
        q = sup.tier_question(TOPIC, "hard", seed)
        p, v = _line(q.text)
        sphere = _equation(re.search(r"sphere (.+?)(?: at two points|\. Find)", q.text).group(1))
        quadratic = sympy.expand(sphere.subs(_on_line(p, v)))
        if "closest to the centre" in q.text:
            seen.add("nearest")
            expected = sympy.solve(sympy.diff(quadratic, L), L)[0]
        else:
            lams = sorted(sympy.solve(quadratic, L))
            assert len(lams) == 2
            if "larger" in q.text:
                seen.add("larger")
                expected = lams[1]
            else:
                seen.add("distance")
                pts = [[p[i] + l * v[i] for i in range(3)] for l in lams]
                expected = sympy.sqrt(sum((a - b) ** 2 for a, b in zip(*pts)))
        assert sympy.simplify(expected - _answer(q)) == 0, (seed, q.text, q.answer_text)
    assert seen == {"nearest", "larger", "distance"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=12 if tier != "easy" else 30, top=0.2)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(40)]
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(40)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(40)]
    assert all("distance from the point" in t for t in easy)
    assert all("plane" in t and "sphere" not in t for t in medium)
    assert all("sphere" in t for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number", "number", "number", "number"], [1, 2, 2, 2], [False, False, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.05)


def test_exam_keys_match_sympy():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        p, v = _line(q.text)
        sphere = _equation(re.search(r"sphere (.+?) at two points", q.text).group(1))
        quadratic = sympy.Poly(sympy.expand(sphere.subs(_on_line(p, v))), L)
        big_a, big_b2, big_c = (quadratic.coeff_monomial(L**k) for k in (2, 1, 0))
        assert big_a == int(q.parts[0].answer_text)
        assert big_b2 == 2 * int(q.parts[1].answer_text)
        assert str(big_c) in q.parts[2].prompt
        lams = sorted(sympy.solve(quadratic.as_expr(), L))
        assert lams[0] == int(q.parts[2].answer_text)
        pts = [[p[i] + l * v[i] for i in range(3)] for l in lams]
        assert sympy.sqrt(sum((a - b) ** 2 for a, b in zip(*pts))) == int(q.parts[3].answer_text)


def test_exam_follow_through_from_wrong_early_parts():
    sk = sup.skill_object(SKILL)
    usable = 0
    for seed in range(30):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        big_c = int(re.search(r"C = (-?\d+)", q.parts[2].prompt).group(1))
        a, b = int(canon["a"]), int(canon["b"])
        for da, db in ((0, k) for k in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 10, -10, 12, -12)):
            wa, wb = a + da, b + db
            disc = wb * wb - wa * big_c
            if disc < 0 or int(disc**0.5) ** 2 != disc:
                continue
            c_val = sympy.Rational(-wb - int(disc**0.5), wa)
            d_val = sympy.Rational(2 * int(disc**0.5), 1) / sympy.sqrt(wa)
            if not d_val.is_Rational:
                continue
            replies = {**canon, "b": str(wb), "c": str(c_val), "d": str(d_val)}
            result = grade(sk, q, replies)
            assert result.parts["b"].marks == 0
            if result.parts["c"].marks == q.parts[2].marks:
                usable += 1
                break
    assert usable >= 6
