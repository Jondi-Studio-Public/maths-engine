"""Tests the trig-compound-angle-exact skill: keys re-derived from the stem with SymPy and floats, grading, exam chain."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3g_support as sup

SKILL = "trig-compound-angle-exact"
TOPIC = "r3g_compound_angle"
EXAM = "r3g_exam_compound_angle"
_T = standard_transformations + (implicit_multiplication_application,)
_FN = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
_EASY = re.compile(r"(?:value of|Evaluate) (?:(\d) )?(sin|cos|tan)[ (]?([^ .)]+)\)?")
_MED = re.compile(
    r"sin A = (\S+) and angle B is acute with cos B = (\S+)\. Find the exact value of (sin|cos)\(A ([+-]) B\)"
)
_SUM = re.compile(
    r"in the (\w+) quadrant with sin A = (\S+), and angle B is in the (\w+) quadrant with cos B = (\S+)\. "
    r"Find the exact value of (tan|cos)\(A ([+-]) B\)"
)
_DBL = re.compile(r"Given that (sin|cos) x = (\S+) and (.+?), find the exact value of (sin|cos|tan) 2x")
_QUAD_SIGNS = {"second": (1, -1), "third": (-1, -1), "fourth": (-1, 1)}
_RANGE_QUAD = {"π/2 < x < π": "second", "π < x < 3π/2": "third", "3π/2 < x < 2π": "fourth"}


def _num(text: str) -> float:
    return float(Fraction(text))


def _from_sin(sin_value: float, cos_sign: int) -> float:
    return math.atan2(sin_value, cos_sign * math.sqrt(1 - sin_value**2))


def _from_cos(cos_value: float, sin_sign: int) -> float:
    return math.atan2(sin_sign * math.sqrt(1 - cos_value**2), cos_value)


def _value(text: str) -> float:
    return float(sympy.N(parse_expr(text.replace("^", "**"), transformations=_T)))


def _derive(stem: str) -> float:
    m = _EASY.search(stem)
    if m and "Angle" not in stem and "Given" not in stem:
        mult, fn, angle = m.groups()
        if angle.endswith("°"):
            rad = math.radians(float(angle[:-1]))
        else:
            rad = float(sympy.N(parse_expr(angle.replace("π", "pi"), transformations=_T)))
        return (int(mult) if mult else 1) * _FN[fn](rad)
    m = _MED.search(stem)
    if m:
        sa, cb, fn, op = m.groups()
        a, b = math.asin(_num(sa)), math.acos(_num(cb))
        return _FN[fn](a + b if op == "+" else a - b)
    m = _SUM.search(stem)
    if m:
        qa, sa, qb, cb, fn, op = m.groups()
        a = _from_sin(_num(sa), _QUAD_SIGNS[qa][1])
        b = _from_cos(_num(cb), _QUAD_SIGNS[qb][0])
        return _FN[fn](a + b if op == "+" else a - b)
    m = _DBL.search(stem)
    assert m, stem
    given, v, rng_text, fn = m.groups()
    sin_sign, cos_sign = _QUAD_SIGNS[_RANGE_QUAD[rng_text]]
    x = _from_sin(_num(v), cos_sign) if given == "sin" else _from_cos(_num(v), sin_sign)
    return _FN[fn](2 * x)


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["senior-recognise-the-exact-values-of-cos-theta-sin", "angle-degrees-radians-pi-form"],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_derivation(tier):
    for seed in range(120):
        q = sup.tier_question(TOPIC, tier, seed)
        assert abs(_value(q.answer_text) - _derive(q.text)) < 1e-9, (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10, top=0.25)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(120))
    assert "Angle A" not in easy and "Given" not in easy
    assert "is acute" in medium and "quadrant" not in medium
    assert "quadrant" in hard and "2x" in hard
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=1)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["expression"] * 4, [1, 1, 2, 2], [False, False, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=8, top=0.25)


_EXAM_STEM = re.compile(r"sin A = ([\d/]+)\b.*?cos B = ([\d/]+)\b")


def test_exam_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        sa, cb = _EXAM_STEM.search(q.text).groups()
        a = _from_sin(_num(sa), -1)
        b = _from_cos(_num(cb), -1)
        keys = [math.cos(a), math.sin(b), math.sin(a + b), math.cos(a + b)]
        for p, key in zip(q.parts, keys):
            assert abs(_value(p.answer_text) - key) < 1e-9, (seed, p.label, p.answer_text, key)


def test_exam_sums_follow_through_from_wrong_ratios():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        sa, cb = _EXAM_STEM.search(q.text).groups()
        sin_a, cos_b = Fraction(sa), Fraction(cb)
        wrong_cos_a, wrong_sin_b = Fraction(1, 2), Fraction(1, 3)
        canon = {p.label: p.answer_text for p in q.parts}
        carried = sin_a * cos_b + wrong_cos_a * wrong_sin_b
        replies = {**canon, "a": "1/2", "b": "1/3", "c": str(carried)}
        assert grade(sk, q, replies).parts["c"].marks == q.parts[2].marks, seed
        assert grade(sk, q, {**replies, "c": "7"}).parts["c"].marks == 0, seed
