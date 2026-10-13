"""Tests `ucv1-sequence-limits-squeeze-monotone` (topics `u2_sequence_limits` and its exam chain).

Oracles read the sequence out of the question TEXT and recompute the limit another way: SymPy's `limit` for explicit
sequences, and iterating the recurrence in 60-digit arithmetic for recursive ones.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mpmath
import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from tests import u2_support as su

TOPIC = "u2_sequence_limits"
EXAM = "exd_x_u2_sequence_limits"
SK = su.skill("ucv1-sequence-limits-squeeze-monotone")
N = sympy.Symbol("n", positive=True)
_TR = standard_transformations + (implicit_multiplication_application,)
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₙ₊", "0123456789n+")


def _expr(text):
    text = text.replace("·", "*").replace("√", "sqrt")
    text = re.sub(
        r"([ⁿ⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "**" + ("n" if m.group(1) == "ⁿ" else m.group(1).translate(_SUP)), text
    )
    text = re.sub(r"sqrt\(([^()]*)\)", r"sqrt(\1)", text)
    return parse_expr(text, local_dict={"n": N}, transformations=_TR)


def _frac(text):
    return Fraction(text) if "/" in text or text.lstrip("-").isdigit() else Fraction(str(sympy.nsimplify(text)))


def _key_value(q):
    return Fraction(q.answer_text)


def test_wiring():
    su.check_wiring(SK, TOPIC, EXAM, "fraction")


@pytest.mark.parametrize("tier", su.TIERS)
def test_key_and_distractors(tier):
    su.check_tier(SK, TOPIC, tier, seeds=60)


def _explicit_limit(q):
    body = re.search(r"of (.*)\.$", q.text).group(1)
    return sympy.limit(_expr(body), N, sympy.oo)


def test_easy_key_is_sympys_limit():
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert sympy.Rational(_key_value(q).numerator, _key_value(q).denominator) == _explicit_limit(q), (seed, q.text)


def test_medium_keys():
    forms = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="medium")
        key = sympy.Rational(_key_value(q).numerator, _key_value(q).denominator)
        if "defined by" in q.text:
            forms.add("recurrence")
            a1, p, r = re.search(r"a₁ = (-?\d+) and aₙ₊₁ = \((-?\d+/\d+|-?\d+)\)aₙ \+ (-?\d+)", q.text).groups()
            a1, p, r = sympy.Rational(a1), sympy.Rational(p), sympy.Rational(r)
            term = a1
            for _ in range(400):
                term = p * term + r
            assert abs(float(term - key)) < 1e-9, (seed, q.text)
        else:
            body = re.search(r"of (.*)\.$", q.text).group(1)
            if "sin" in body or "cos" in body:
                forms.add("squeeze")
                expr = _expr(body)
                value = sympy.N(expr.subs(N, 10**9), 30)
                assert abs(float(value - key)) < 1e-6, (seed, q.text)
            else:
                forms.add("explicit")
                point = 400 if "·" in body else 10**9
                value = sympy.N(_expr(body).subs(N, point), 60)
                assert abs(float(value - key)) < 1e-6, (seed, q.text)
    assert {"recurrence", "squeeze", "explicit"} <= forms


def _iterate(text):
    a1 = mpmath.mpf(re.search(r"a₁ = (-?\d+)", text).group(1))
    with mpmath.workdps(60):
        if "/aₙ)/2" in text:
            m = mpmath.mpf(re.search(r"\(aₙ \+ (\d+)/aₙ\)/2", text).group(1))
            step = lambda a: (a + m / a) / 2
        elif "²" in text:
            c, s = (mpmath.mpf(v) for v in re.search(r"\(aₙ² \+ (\d+)\)/(\d+)", text).groups())
            step = lambda a: (a * a + c) / s
        else:
            k = mpmath.mpf(re.search(r"√\((\d+) \+ aₙ\)", text).group(1))
            step = lambda a: mpmath.sqrt(k + a)
        term = a1
        for _ in range(3000):
            term = step(term)
    return term


def test_hard_key_is_the_iterated_limit():
    forms = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="hard")
        forms.add("newton" if "/aₙ)/2" in q.text else "square" if "²" in q.text else "root")
        assert abs(float(_iterate(q.text) - mpmath.mpf(_key_value(q).numerator) / _key_value(q).denominator)) < 1e-9, (
            seed,
            q.text,
        )
    assert len(forms) >= 2


def test_exam_chain():
    su.check_exam(SK, EXAM, "abcd", (2, 1, 3, 2), (False, False, False, True), seeds=15)
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        a1, c, s = (int(v) for v in re.search(r"a₁ = (-?\d+) and aₙ₊₁ = \(aₙ² \+ (-?\d+)\)/(\d+)", q.stem).groups())
        roots = sorted(sympy.solve(sympy.Symbol("L") ** 2 - s * sympy.Symbol("L") + c, sympy.Symbol("L")))
        assert [float(v) for v in q.parts[0].answer] == [float(v) for v in roots] or sorted(
            float(v) for v in q.parts[0].answer
        ) == [float(v) for v in roots]
        assert Fraction(q.parts[1].answer_text) == Fraction(a1 * a1 + c, s)
        limit = _iterate(f"a₁ = {a1} and (aₙ² + {c})/{s}")
        assert (
            abs(
                float(limit - Fraction(q.parts[2].answer_text).numerator / Fraction(q.parts[2].answer_text).denominator)
            )
            < 1e-9
        )
        assert Fraction(q.parts[3].answer_text) == 2 * Fraction(q.parts[2].answer_text)


def test_exam_follow_through():
    su.check_follow_through(SK, EXAM, "c", "d", lambda q: "7/2", lambda q, wrong: "7", seeds=12)


def test_variety():
    for tier, minimum in (("easy", 20), ("medium", 30), ("hard", 15)):
        su.distinct_stems(TOPIC, tier, seeds=60, minimum=minimum)
