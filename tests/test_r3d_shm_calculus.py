"""Tests `shm-displacement-velocity-calculus`: oracles re-derive every answer from the question text alone."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations
from r3d_support import (
    assert_deterministic_and_varied,
    assert_exam_rules,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    fixed,
    frac,
    question,
    questions,
    skill,
    sources,
)

from mathgen import make
from mathgen.engine import grade

SK = skill("shm-displacement-velocity-calculus")
SEEDS = range(60)
t = sympy.Symbol("t")
_IMPLICIT = standard_transformations + (implicit_multiplication_application,)


def _pi_multiple(text: str) -> Fraction:
    """'7π/60' -> 7/60, 'π/12' -> 1/12, '2π' -> 2."""
    m = re.fullmatch(r"(-?\d*)π(?:/(\d+))?", text)
    num = m.group(1)
    num = 1 if num == "" else (-1 if num == "-" else int(num))
    return Fraction(num, int(m.group(2) or 1))


def _function(text: str):
    """The displacement 'x = 0.8 cos(3t)' as (A, name, n)."""
    m = re.search(r"x = (-?[\d.]*) ?(sin|cos)\(([\d.]*)t\)", text)
    a = m.group(1)
    return (Fraction(1) if a == "" else frac(a)), m.group(2), (Fraction(1) if m.group(3) == "" else frac(m.group(3)))


def _sym(a: Fraction, name: str, n: Fraction):
    f = sympy.cos if name == "cos" else sympy.sin
    return sympy.Rational(a.numerator, a.denominator) * f(sympy.Rational(n.numerator, n.denominator) * t)


def oracle_easy(q):
    a, name, n = _function(q.text)
    tt = re.search(r"when t = (\S+) seconds", q.text).group(1)
    x = _sym(a, name, n)
    at = sympy.pi * sympy.Rational(_pi_multiple(tt).numerator, _pi_multiple(tt).denominator)
    derivative = sympy.diff(x, t) if "velocity of the object when" in q.text else sympy.diff(x, t, 2)
    return sympy.nsimplify(derivative.subs(t, at))


def test_easy_grading_and_places():
    assert_single_answer_rules(SK, "easy", SEEDS)
    for q in questions(SK, "easy", SEEDS):
        assert_number_distractors_match_places(q)


def test_easy_oracle():
    kinds = set()
    for q in questions(SK, "easy", range(200)):
        expected = oracle_easy(q)
        assert Fraction(str(expected)) == frac(q.answer_text), (q.text, q.answer_text, expected)
        kinds.add("velocity" if "velocity of the object when" in q.text else "acceleration")
    assert kinds == {"velocity", "acceleration"}


def oracle_medium(q):
    s = q.text
    if m := re.search(r"x = (.+?) metres at time t seconds\. Find the greatest (speed|acceleration)", s):
        x = parse_expr(
            re.sub(r"(\d)t", r"\1*t", m.group(1).replace("−", "-")), local_dict={"t": t}, transformations=_IMPLICIT
        )
        d = sympy.diff(x, t) if m.group(2) == "speed" else sympy.diff(x, t, 2)
        grid = [2 * math.pi * k / 4000 for k in range(4000)]
        f = sympy.lambdify(t, d, "math")
        return Fraction(max(abs(f(g)) for g in grid)).limit_denominator(1000)
    if m := re.search(r"amplitude of the motion", s):
        v, a = (
            frac(x) for x in re.search(r"speed is ([\d.]+) m/s and its greatest acceleration is ([\d.]+)", s).groups()
        )
        return v * v / a
    if "Find the period" in s and "greatest speed is" in s:
        v, a = (
            frac(x) for x in re.search(r"speed is ([\d.]+) m/s and its greatest acceleration is ([\d.]+)", s).groups()
        )
        return 2 * math.pi * float(v / a)
    if "Find the period" in s:
        amp, big = re.search(r"x = ([\d.]+) cos\(nt\).*acceleration is ([\d.]+) m/s²", s).groups()
        return 2 * math.pi / math.sqrt(float(frac(big) / frac(amp)))
    raise AssertionError(s)


def test_medium_grading_and_oracle():
    assert_single_answer_rules(SK, "medium", SEEDS)
    seen = set()
    for q in questions(SK, "medium", range(200)):
        assert_number_distractors_match_places(q)
        expected = oracle_medium(q)
        places = len(q.answer_text.split(".")[1])
        got = float(frac(q.answer_text))
        assert abs(got - float(expected)) < 0.5 * 10**-places + 1e-9, (q.text, q.answer_text, expected)
        seen.add(
            (
                "amplitude"
                if "amplitude of the motion" in q.text
                else "period-from-speed"
                if "greatest speed is" in q.text
                else "period"
                if "Find the period" in q.text
                else "greatest"
            )
        )
    assert len(seen) >= 4


def oracle_hard(q):
    s = q.text
    if "Find ε in radians" in s:
        n = int(re.search(r"cos\((\d+)t \+ ε\)", s).group(1))
        x0, v0 = (frac(x) for x in re.search(r"x = (-?[\d.]+) and its velocity is (-?[\d.]+) m/s", s).groups())
        # a cos e = x0, -a n sin e = v0
        return math.atan2(float(-v0 / n), float(x0))
    if "greatest acceleration" in s:
        n2 = int(re.search(r"acceleration -(\d+)x", s).group(1))
        x0, v0 = (frac(x) for x in re.search(r"at x = (-?[\d.]+) with velocity (-?[\d.]+) m/s", s).groups())
        a = math.sqrt(float(x0) ** 2 + float(v0) ** 2 / n2)
        return n2 * a
    m = re.search(r"cos\((\d+)t ([+-]) (\S+)\)", s)
    nn, sign, ph = int(m.group(1)), m.group(2), _pi_multiple(m.group(3))
    eps = float(ph) * math.pi * (1 if sign == "+" else -1)
    # smallest t > 0 with nt + eps = k pi
    k = math.ceil(eps / math.pi + 1e-12)
    k = k if (k * math.pi - eps) > 1e-9 else k + 1
    return (k * math.pi - eps) / nn


def test_hard_grading_and_oracle():
    assert_single_answer_rules(SK, "hard", SEEDS)
    kinds = set()
    for q in questions(SK, "hard", range(200)):
        assert_number_distractors_match_places(q)
        expected = oracle_hard(q)
        places = len(q.answer_text.split(".")[1])
        assert abs(float(frac(q.answer_text)) - float(expected)) < 0.5 * 10**-places + 1e-9, (
            q.text,
            q.answer_text,
            expected,
        )
        kinds.add("phase" if "Find ε" in q.text else "greatest" if "greatest acceleration" in q.text else "rest")
    assert kinds == {"phase", "greatest", "rest"}


def test_hard_phase_needs_the_quadrant():
    seen = set()
    for q in questions(SK, "hard", range(200)):
        if "Find ε" in q.text:
            x0 = frac(re.search(r"at x = (-?[\d.]+)", q.text).group(1))
            seen.add(x0 > 0)
    assert seen == {True, False}


def _exam(seed):
    return make(sources(SK)["exam"], seed=seed, difficulty="exam")


def exam_oracle(q):
    m = re.search(r"x = ([\d.]+) cos\((\d+)t ([+-]) (\S+)\)", q.text)
    amp, n, sign, ph = frac(m.group(1)), int(m.group(2)), m.group(3), _pi_multiple(m.group(4))
    eps = sympy.pi * sympy.Rational(ph.numerator, ph.denominator) * (1 if sign == "+" else -1)
    x = sympy.Rational(amp.numerator, amp.denominator) * sympy.cos(n * t + eps)
    k = sympy.simplify(-sympy.diff(x, t, 2) / x)
    v0 = sympy.diff(x, t).subs(t, 0)
    return (
        {
            "a": str(int(k)),
            "b": fixed(float(2 * math.pi / math.sqrt(float(k))), 2),
            "c": None,
            "d": None,
        },
        Fraction(str(sympy.nsimplify(v0))),
        amp * n,
    )


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcd", marks=[1, 2, 2, 2])
    for seed in range(120):
        q = _exam(seed)
        keys, v0, vmax = exam_oracle(q)
        got = {p.label: p.answer_text for p in q.parts}
        assert got["a"] == keys["a"] and got["b"] == keys["b"], q.text
        assert frac(got["c"]) == vmax and frac(got["d"]) == v0, (q.text, got)


def test_exam_follows_a_wrong_k_through():
    checked = 0
    for seed in range(80):
        q = _exam(seed)
        wrong_k = Fraction(q.parts[0].distractors[0])
        period = fixed(2 * math.pi / math.sqrt(float(wrong_k)), 2)
        reply = {"a": str(wrong_k), "b": period, "c": q.parts[2].answer_text, "d": q.parts[3].answer_text}
        result = grade(SK, q, reply)
        assert result.parts["a"].marks == 0
        if period not in q.parts[1].distractors and period != q.parts[1].answer_text:
            assert result.parts["b"].marks == q.parts[1].marks and result.parts["b"].follow_through, (q.text, reply)
            checked += 1
    assert checked >= 30


def test_exam_distractors_keep_the_precision():
    for seed in range(40):
        for p in _exam(seed).parts:
            assert_number_distractors_match_places(p)


@pytest.mark.parametrize(
    "topic,difficulty",
    [
        ("r3d_shm_calculus", "easy"),
        ("r3d_shm_calculus", "medium"),
        ("r3d_shm_calculus", "hard"),
        ("r3d_shm_calculus_exam", "exam"),
    ],
)
def test_deterministic_and_varied(topic, difficulty):
    assert_deterministic_and_varied(topic, difficulty, distinct=25)


def test_tiers_differ_in_shape():
    easy = " ".join(q.text for q in questions(SK, "easy", SEEDS))
    medium = " ".join(q.text for q in questions(SK, "medium", SEEDS))
    hard = " ".join(q.text for q in questions(SK, "hard", SEEDS))
    assert "ε" not in easy and "ε" not in medium and "ε" in hard
    assert "greatest" not in easy and "greatest" in medium
    assert question(SK, "easy", 1).topic != _exam(1).topic
