"""Tests the skill `ucv1-tangent-plane-linear-approximation`: an independent oracle that re-derives every key from the
question TEXT with SymPy, the typed grader on every key and distractor, and the four-part exam chain with
follow-through."""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen.engine import grade
from tests import fy1ms_support as T

SKILL_ID = "ucv1-tangent-plane-linear-approximation"
TOPIC = "fy1ms_tangent_plane"
EXAM = "exd_x_fy1ms_tangent_plane"
SK = T.load(SKILL_ID)
TIERS = ("easy", "medium", "hard")
x, y, z = sympy.symbols("x y z")


def _plane_at(f, px, py):
    sub = {x: px, y: py}
    return f.subs(sub), sympy.diff(f, x).subs(sub), sympy.diff(f, y).subs(sub)


def oracle(tier, q):
    text = q.text
    if tier == "easy":
        m = re.fullmatch(
            r"Let f\(x, y\) = (.*?)\. Use the tangent plane to the surface z = f\(x, y\) at the point \((-?\d+), (-?\d+)\) "
            r"to estimate f\((-?[\d.]+), (-?[\d.]+)\)\. Give your answer correct to (\d) decimal places?\.",
            text,
        )
        f = T.parse_plain(m.group(1), x=x, y=y)
        px, py = int(m.group(2)), int(m.group(3))
        x1, y1 = sympy.Rational(m.group(4)), sympy.Rational(m.group(5))
        f0, fx, fy = _plane_at(f, px, py)
        return T.rnd(f0 + fx * (x1 - px) + fy * (y1 - py), int(m.group(6)))
    if tier == "medium":
        m = re.fullmatch(
            r"The surface z = (.*?) has a tangent plane at the point where x = (-?\d+) and y = (-?\d+)\. "
            r"Write this plane as z = Ax \+ By \+ C and find the value of C\.",
            text,
        )
        f = T.parse_plain(m.group(1), x=x, y=y)
        px, py = int(m.group(2)), int(m.group(3))
        f0, fx, fy = _plane_at(f, px, py)
        plane = sympy.expand(f0 + fx * (x - px) + fy * (y - py))
        return str(plane.subs({x: 0, y: 0}))
    m = re.fullmatch(
        r"Let f\(x, y\) = (\d?)√\(x² \+ y²\)\. Use the linear approximation of f at \((-?\d+), (-?\d+)\) to estimate "
        r"f\((-?[\d.]+), (-?[\d.]+)\)\. Give your answer correct to 3 decimal places\.",
        text,
    )
    if m:
        scale = int(m.group(1) or 1)
        f = scale * sympy.sqrt(x**2 + y**2)
        px, py = int(m.group(2)), int(m.group(3))
        x1, y1 = sympy.Rational(m.group(4)), sympy.Rational(m.group(5))
        f0, fx, fy = _plane_at(f, px, py)
        return T.rnd(f0 + fx * (x1 - px) + fy * (y1 - py), 3)
    m = re.fullmatch(
        r"The point P\((-?\d+), (-?\d+), (-?\d+)\) lies on the surface (.*?) = (\d+)\. The tangent plane to the surface "
        r"at P meets the z-axis at the point \(0, 0, t\)\. Find t, correct to 3 decimal places\.",
        text,
    )
    assert m, text
    px, py, pz = (int(m.group(i)) for i in (1, 2, 3))
    big_f = T.parse_plain(m.group(4), x=x, y=y, z=z) - int(m.group(5))
    assert big_f.subs({x: px, y: py, z: pz}) == 0
    grad = [sympy.diff(big_f, v).subs({x: px, y: py, z: pz}) for v in (x, y, z)]
    plane = sum(g * (v - p) for g, v, p in zip(grad, (x, y, z), (px, py, pz)))
    return T.rnd(sympy.solve(plane.subs({x: 0, y: 0}), z)[0], 3)


def test_wiring():
    T.assert_wiring(SK, TOPIC, [EXAM])


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(150) if oracle(tier, T.single(TOPIC, tier, s)) != T.single(TOPIC, tier, s).answer_text]
    assert bad == []


def test_hard_tier_uses_both_shapes():
    kinds = {"radial" if "√" in T.single(TOPIC, "hard", s).text else "implicit" for s in range(60)}
    assert kinds == {"radial", "implicit"}


def test_easy_steps_are_tenths_or_hundredths_and_the_medium_answer_is_whole():
    places = {T.dp_of(T.single(TOPIC, "easy", s).answer_text) for s in range(60)}
    assert places == {1, 2}
    assert all("." not in T.single(TOPIC, "medium", s).answer_text for s in range(60))


@pytest.mark.parametrize("tier", TIERS)
def test_grading_and_diagnosis(tier):
    T.assert_single_grading(SK, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_clean_text_variety_and_determinism(tier):
    T.assert_clean_variety_and_determinism(TOPIC, tier, distinct_answers=10)


def test_the_estimate_is_not_the_exact_value():
    """Easy distractor 'exact value given' must be a genuinely different number from the tangent-plane estimate."""
    seen = 0
    for seed in range(80):
        q = T.single(TOPIC, "easy", seed)
        if q.option_notes and "linear-approx-exact-value-given" in q.option_notes.values():
            seen += 1
    assert seen >= 25


# --- the exam chain --------------------------------------------------------------------------------------------


def exam_oracle(q):
    m = re.search(r"as z = (.*?)\. The point P\((\d+), (\d+), (\d+)\) is on the hillside\.", q.stem)
    f = T.parse_plain(m.group(1), x=x, y=y)
    px, py, f0 = int(m.group(2)), int(m.group(3)), int(m.group(4))
    v0, fx, fy = _plane_at(f, px, py)
    assert v0 == f0
    m = re.search(r"at the point \((-?[\d.]+), (-?[\d.]+)\)\. Give your answer correct to (\d) decimal", q.text)
    x1, y1 = sympy.Rational(m.group(1)), sympy.Rational(m.group(2))
    plane = sympy.expand(f0 + fx * (x - px) + fy * (y - py))
    return dict(a=str(fx), b=str(fy), c=plane, d=T.rnd(f0 + fx * (x1 - px) + fy * (y1 - py), int(m.group(3))))


def test_exam_shape():
    T.assert_exam_shape(EXAM, ["a", "b", "c", "d"], [1, 1, 2, 2], [False, False, True, True])


def test_exam_keys_match_the_independent_derivation():
    for seed in range(200):
        q = T.exam(EXAM, seed)
        o = exam_oracle(q)
        got = {p.label: p.answer_text for p in q.parts}
        assert (got["a"], got["b"], got["d"]) == (o["a"], o["b"], o["d"]), seed
        rhs = T.parse_plain(got["c"].removeprefix("z = "), x=x, y=y)
        assert sympy.expand(rhs - o["c"]) == 0, seed


def test_exam_grading_distractors_and_notes():
    T.assert_exam_grading(SK, EXAM)


def test_an_equivalent_plane_in_point_form_is_accepted():
    for seed in range(40):
        q = T.exam(EXAM, seed)
        o = exam_oracle(q)
        reply = {**T.canonical_reply(q), "c": "z = " + str(sympy.factor_terms(o["c"]))}
        assert grade(SK, q, reply).parts["c"].marks == 2, seed


def test_follow_through_from_a_wrong_partial():
    T.assert_follow_through_credited(SK, EXAM, minimum=100)


def test_the_height_estimate_from_the_rounded_plane_is_credited():
    """A learner who used the plane they wrote in (c) gets (d) right without recomputing from the partials."""
    for seed in range(60):
        q = T.exam(EXAM, seed)
        assert grade(SK, q, T.canonical_reply(q)).marks == 6


def test_exam_answer_space():
    T.assert_exam_variety(EXAM, distinct=100, repeat=6)
