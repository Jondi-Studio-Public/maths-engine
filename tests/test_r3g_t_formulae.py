"""Tests the t-formulae-half-angle skill: keys re-derived from the stem with floats, grading, and the exam chain."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3g_support as sup

SKILL = "t-formulae-half-angle"
TOPIC = "r3g_t_formulae"
EXAM = "r3g_exam_t_formulae"
_T = standard_transformations + (implicit_multiplication_application,)
_EASY = re.compile(r"t = (\S+), find the value of (sin|cos|tan) x")
_COS = re.compile(r"cos x = (\S+) and (.+?), find the exact value of tan\(x/2\)")
_EXACT = re.compile(r"with tan (\d+)° = .+? to find the exact value of tan")
_EQ = re.compile(r"(?:solve|equation) (.+?)(?: for 0| where|,)")
_FN = {"sin": math.sin, "cos": math.cos, "tan": math.tan}


def _val(text: str) -> float:
    return float(sympy.N(parse_expr(text.replace("^", "**"), transformations=_T)))


def _roots(a: float, b: float, c: float) -> list[float]:
    """Solutions of a sin x + b cos x = c in [0, 2 pi) by sign changes on a fine grid, refined by bisection."""
    f = lambda x: a * math.sin(x) + b * math.cos(x) - c  # noqa: E731
    n, out = 20000, []
    prev_x, prev_v = None, None
    for k in range(n + 1):
        x = 2 * math.pi * k / n
        v = f(x)
        if abs(v) < 1e-12:
            out.append(x)
        elif prev_v is not None and abs(prev_v) >= 1e-12 and prev_v * v < 0:
            lo, hi = prev_x, x
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
            out.append((lo + hi) / 2)
        prev_x, prev_v = x, v
    out = [x for x in out if x < 2 * math.pi - 1e-9]
    return out


def _coefficients(eq: str):
    a = b = 0
    for sign, num, fn in re.findall(r"([+-]?)\s*(\d*)\s*(sin|cos) x", eq.split("=")[0].replace(" ", " ")):
        coef = int(num) if num else 1
        coef = -coef if sign == "-" else coef
        if fn == "sin":
            a = coef
        else:
            b = coef
    return a, b, int(eq.split("=")[1])


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["trig-compound-angle-exact", "senior-recognise-the-exact-values-of-cos-theta-sin"],
    )


def test_easy_and_medium_values_match_an_independent_evaluation():
    seen = set()
    for tier in ("easy", "medium"):
        for seed in range(150):
            q = sup.tier_question(TOPIC, tier, seed)
            key = _val(q.answer_text)
            m = _EASY.search(q.text)
            if m:
                t, fn = m.groups()
                expected = _FN[fn](2 * math.atan(float(Fraction(t))))
                seen.add("easy")
            elif _COS.search(q.text):
                c, rng_text = _COS.search(q.text).groups()
                x = math.acos(float(Fraction(c)))
                x = x if rng_text == "0 < x < π" else 2 * math.pi - x
                expected = math.tan(x / 2)
                seen.add("cos")
            else:
                deg = int(_EXACT.search(q.text).group(1))
                expected = math.tan(math.radians(deg / 2))
                seen.add("exact")
            assert abs(key - expected) < 1e-9, (seed, q.text, q.answer_text)
    assert seen == {"easy", "cos", "exact"}


def test_hard_solutions_match_an_independent_root_scan():
    seen = set()
    for seed in range(60):
        q = sup.tier_question(TOPIC, "hard", seed)
        eq = re.search(r"to solve (.+?) for 0", q.text).group(1)
        expected = sorted(_roots(*_coefficients(eq)))
        assert len(expected) == 2
        index = 1 if "the larger of the two" in q.text else 0
        assert abs(_val(q.answer_text.replace("atan", "atan")) - expected[index]) <= 1e-6, (seed, q.text, q.answer_text)
        seen.add(index)
    assert seen == {0, 1}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10, top=0.25)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(60))
    assert "find the value of" in easy and "tan(x/2)" not in easy.replace("Let t = tan(x/2)", "")
    assert "exact value of tan" in medium and "to solve" not in medium
    assert "to solve" in hard and "exactly" in hard
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=1)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number"] * 4, [2, 1, 1, 1], [False, True, False, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=10, top=0.15)


def test_exam_keys_match_an_independent_root_scan():
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        eq = re.search(r"equation (.+?), where", q.text).group(1)
        a, b, c = _coefficients(eq)
        x1, x2 = sorted(_roots(a, b, c))
        t1, t2 = (math.tan(x1 / 2), math.tan(x2 / 2))
        keys = [t1, x1, t2, x2]
        for p, key in zip(q.parts, keys):
            assert abs(float(p.answer_text) - key) <= 0.0051, (seed, p.label, p.answer_text, key)


def test_exam_solutions_follow_through_from_a_wrong_root():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, "a": "3", "b": f"{2 * math.atan(3):.2f}"}
        assert grade(sk, q, replies).parts["b"].marks == q.parts[1].marks, seed
        assert grade(sk, q, {**replies, "b": "0.5"}).parts["b"].marks == 0, seed
