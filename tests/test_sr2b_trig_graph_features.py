"""Tests the reciprocal-and-trig-graph-features skill: keys re-derived from the function in the stem, grading, exam."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from tests import sr2b_support as sup

SKILL = "reciprocal-and-trig-graph-features"
TOPIC = "sr2b_trig_graph_features"
EXAM = "sr2b_exam_trig_graph_features"
X = sympy.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)
_FUNC = re.compile(r"y = (-?\d*)\s*(tan|sec|cosec)\((.+?)\)(?: ([+-]) (\d+))?(?:[ .,?]|$)")
_FUNC_EXAM = re.compile(r"f\(x\) = (-?\d*)\s*(cosec)\((.+?)\)(?: ([+-]) (\d+))?,")


def _value(text: str):
    return parse_expr(text.replace("π", "pi"), local_dict={"x": X, "pi": sympy.pi}, transformations=_T)


def _function(stem: str, pattern=_FUNC):
    amp, kind, inner, sign, shift = pattern.search(stem).groups()
    a = {"": 1, "-": -1}.get(amp) or int(amp)
    d = 0 if shift is None else (int(shift) if sign == "+" else -int(shift))
    return a, kind, _value(inner), d


def _frac(value):
    value = sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


def _linear(u):
    """The inner expression `k x + c` as exact (k, c / pi)."""
    return _frac(sympy.diff(u, X)), _frac(sympy.simplify(u.subs(X, 0) / sympy.pi))


def _period(kind, u):
    """The period as a multiple of pi."""
    k, _ = _linear(u)
    return Fraction(1 if kind == "tan" else 2) / abs(k)


def _asymptotes(kind, u):
    """Every asymptote position, as a multiple of pi, for n from -40 to 40."""
    k, c = _linear(u)
    u0 = {"tan": Fraction(1, 2), "sec": Fraction(1, 2), "cosec": Fraction(0)}[kind]
    return sorted((u0 + n - c) / k for n in range(-40, 41))


def _same(text: str, expected, pi_multiple=False) -> bool:
    if pi_multiple:
        expected = sympy.Rational(expected.numerator, expected.denominator) * sympy.pi
    return sympy.simplify(_value(text.replace("pi", "π")) - expected) == 0


def _derive(stem: str):
    a, kind, u, d = _function(stem)
    if "period" in stem:
        return _period(kind, u), True
    asy = _asymptotes(kind, u)
    if "how many vertical asymptotes" in stem.lower():
        return sum(1 for v in asy if 0 <= v <= 2), False
    if "smallest positive" in stem:
        return min(v for v in asy if v > 0), True
    assert "larger of these two numbers" in stem, stem
    return d + abs(a), False


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        [
            "senior-examine-period-changes-and-the-graphs-of-y",
            "senior-understand-the-unit-circle-definition-of-cos-theta-acmmm034",
        ],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_derivation(tier):
    for seed in range(150):
        q = sup.tier_question(TOPIC, tier, seed)
        expected, multiple = _derive(q.text)
        assert _same(q.answer_text, expected, multiple), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8)


def test_tiers_differ():
    asked = {t: " ".join(sup.tier_question(TOPIC, t, s).text for s in range(60)) for t in sup.TIERS}
    assert "period" in asked["easy"] and "asymptote" not in asked["easy"]
    assert "How many" in asked["medium"] or "how many" in asked["medium"]
    assert "smallest positive" in asked["hard"] and "larger of these two numbers" in asked["hard"]
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_all_three_functions_are_set():
    kinds = {_function(sup.tier_question(TOPIC, "easy", s).text)[1] for s in range(80)}
    assert kinds == {"tan", "sec", "cosec"}


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["expression", "expression", "number", "number"], [1, 2, 2, 2], [False, True, True, False]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=20, top=0.1)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        a, kind, u, d = _function(q.text, _FUNC_EXAM)
        asy = _asymptotes(kind, u)
        keys = [
            (_period(kind, u), True),
            (min(v for v in asy if v > 0), True),
            (sum(1 for v in asy if 0 <= v <= 2), False),
            (d + abs(a), False),
        ]
        for p, (key, multiple) in zip(q.parts, keys):
            assert _same(p.answer_text, key, multiple), (seed, p.label, p.answer_text, key, q.text)


def test_exam_follow_through_on_a_wrong_period():
    from mathgen.engine import grade

    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        # a wrong period of 2pi: the first asymptote carried from it is pi, and the count carried from that is 3
        replies = {**canon, "a": "2pi", "b": "pi", "c": "3"}
        result = grade(sk, q, replies)
        assert result.parts["b"].marks == q.parts[1].marks, seed
        assert result.parts["c"].marks == q.parts[2].marks, seed
        # the carried count is not credited when it does not follow from the reply given for part b
        assert grade(sk, q, {**replies, "c": "4"}).parts["c"].marks == 0, seed
