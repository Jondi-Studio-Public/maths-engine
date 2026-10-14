"""Tests the exact-trig-values-typed skill: keys re-derived with SymPy, grading, and the exam chain."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import sr2b_support as sup

SKILL = "exact-trig-values-typed"
TOPIC = "sr2b_exact_trig_values"
EXAM = "sr2b_exam_exact_trig_values"
_T = standard_transformations + (implicit_multiplication_application,)
_FN = {"sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan}
_CALL = re.compile(r"(\d*)\s*(sin|cos|tan)\(([^()]+)\)")
_GIVEN = re.compile(r"(?:Given that|for which) (sin|cos|tan) θ = (.+?) and (sin|cos|tan) θ ([<>]) 0")


def _value(text: str):
    return parse_expr(text.replace("π", "pi"), local_dict={"pi": sympy.pi}, transformations=_T)


def _same(text: str, expected) -> bool:
    return sympy.simplify(_value(text) - expected) == 0


def _angles():
    return [sympy.Rational(k, 12) * sympy.pi for k in range(24)]


def _derive(stem: str):
    given = _GIVEN.search(stem)
    if given:
        fn, value, cond_fn, op = given.groups()
        hits = [
            t
            for t in _angles()
            if sympy.simplify(_FN[fn](t) - _value(value)) == 0
            and ((_FN[cond_fn](t) > 0) if op == ">" else (_FN[cond_fn](t) < 0))
        ]
        assert len(hits) == 1, (stem, hits)
        return hits[0]
    body = re.split(r"\.|\?", stem.split("exact")[0] + stem.split("exact", 1)[-1])[0]
    product = sympy.Integer(1)
    calls = _CALL.findall(body)
    assert calls, stem
    for coef, fn, angle in calls:
        product *= (int(coef) if coef else 1) * _FN[fn](_value(angle))
    return sympy.simplify(product)


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        [
            "senior-understand-the-unit-circle-definition-of-cos-theta-acmmm034",
            "angle-degrees-radians-pi-form",
        ],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_derivation(tier):
    for seed in range(150):
        q = sup.tier_question(TOPIC, tier, seed)
        assert _same(q.answer_text, _derive(q.text)), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(60))
    assert "Given that" not in easy and "×" not in easy
    assert "Given that" in hard and "×" in hard
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_negative_and_large_angles_appear():
    text = " ".join(sup.tier_question(TOPIC, t, s).text for t in ("easy", "medium") for s in range(80))
    assert re.search(r"\(-\d*π", text), "no negative angle set"
    large = [m for m in re.findall(r"(\d+)π/(\d+)", text) if int(m[0]) > 2 * int(m[1])]
    assert large, "no angle beyond 2π set"


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["expression"] * 4, [1, 1, 1, 2], [False, False, True, False])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=8, top=0.25)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        theta = _value(re.search(r"θ (?:=|is) ([0-9π/]+)", q.text).group(1))
        s, c = sympy.sin(theta), sympy.cos(theta)
        (alpha,) = [t for t in _angles() if sympy.simplify(sympy.sin(t) - s) == 0 and t != theta]
        for p, key in zip(q.parts, [s, c, s / c, alpha]):
            assert _same(p.answer_text, key), (seed, p.label, p.answer_text, key)


def test_exam_tangent_follows_through_from_wrong_sine_and_cosine():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        # sine and cosine given as 1/2 and sqrt(3)/2 (possibly wrong): tangent 1/sqrt(3) follows from them
        replies = {**canon, "a": "1/2", "b": "sqrt(3)/2", "c": "sqrt(3)/3"}
        assert grade(sk, q, replies).parts["c"].marks == q.parts[2].marks, seed
        assert grade(sk, q, {**replies, "c": "5"}).parts["c"].marks == 0, seed
