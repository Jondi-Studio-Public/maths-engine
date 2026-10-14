"""Tests `ucv1-hyperbolic-functions` (topics `fy1ca_hyperbolic` and `fy1ca_hyperbolic_exam`).

The oracle reads the function, the given value or the equation back out of the question TEXT and evaluates it with the
standard library's hyperbolic functions and mpmath, then compares with the key.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import mpmath
import pytest
import sympy
from sympy import Rational, cosh, log, sinh, tanh

from mathgen import make
from scripts.measure_tiers import classify_skill
from tests.fy1ca_support import TIERS, check_exam, check_follow_through, check_single_tier, check_wiring, load_skill

TOPIC = "fy1ca_hyperbolic"
EXAM = "fy1ca_hyperbolic_exam"
SKILL = load_skill("ucv1-hyperbolic-functions")
x = sympy.Symbol("x", real=True)
_NAMES = {"e": sympy.E, "ln": sympy.log, "pi": sympy.pi, "sqrt": sympy.sqrt, "x": x}


def _value(text: str) -> float:
    return float(sympy.N(sympy.sympify(text.replace("^", "**"), locals=_NAMES)))


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def test_skill_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "expression")


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single_tier(SKILL, TOPIC, tier)


def test_easy_keys_are_hyperbolic_values():
    seen = Counter()
    for seed in range(150):
        q = _q("easy", seed)
        m = re.search(r"(sinh|cosh|tanh)\(ln (\d+)\)", q.text)
        if m:
            fn = {"sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh}[m.group(1)]
            expected = fn(math.log(int(m.group(2))))
            seen["log"] += 1
        else:
            c = float(Rational(re.search(r"cosh x = (\S+?) and", q.text).group(1)))
            x_value = math.acosh(c)
            ask = re.search(r"value of (sinh x|tanh x)", q.text).group(1)
            expected = math.sinh(x_value) if ask == "sinh x" else math.tanh(x_value)
            seen[ask] += 1
        assert abs(_value(q.answer_text) - expected) < 1e-9, (seed, q.text, q.answer_text)
    assert min(seen.values()) >= 15 and len(seen) == 3, seen


DERIVATIVES = {
    "ln(cosh x)": log(cosh(x)),
    "sinh x cosh x": sinh(x) * cosh(x),
    "cosh(x²)": cosh(x**2),
    "x sinh x": x * sinh(x),
    "x cosh x": x * cosh(x),
}


def _function(name: str):
    m = re.fullmatch(r"(sinh|cosh|tanh)\((\d)x\)", name)
    if m:
        return {"sinh": sinh, "cosh": cosh, "tanh": tanh}[m.group(1)](int(m.group(2)) * x)
    return DERIVATIVES[name]


def test_medium_keys_are_derivatives():
    seen = set()
    for seed in range(200):
        q = _q("medium", seed)
        name = re.search(r"f\(x\) = (.+?)(?: with respect| *$|\.)", q.text).group(1)
        f = _function(name)
        key = sympy.lambdify(x, sympy.sympify(q.answer_text.replace("^", "**"), locals=_NAMES), "mpmath")
        for point in (0.3, 0.8, 1.4):
            expected = float(mpmath.diff(sympy.lambdify(x, f, "mpmath"), point))
            assert abs(float(key(point)) - expected) < 1e-8, (seed, q.text, q.answer_text)
        seen.add(name)
    assert len(seen) >= 12, seen


def _integrand(name: str):
    return {
        "1/cosh²x": lambda t: 1 / mpmath.cosh(t) ** 2,
        "cosh(2x)": lambda t: mpmath.cosh(2 * t),
        "sinh x cosh x": lambda t: mpmath.sinh(t) * mpmath.cosh(t),
        "cosh²x": lambda t: mpmath.cosh(t) ** 2,
    }[name]


def test_hard_keys_solve_equations_and_integrate():
    seen = Counter()
    for seed in range(240):
        q = _q("hard", seed)
        if "Solve cosh x" in q.text or "Solve sinh x" in q.text or "positive value of x for which" in q.text:
            name, a = re.search(r"(sinh|cosh) x = (\S+?)(?: for|\.| Give)", q.text).groups()
            expected = math.acosh(float(Rational(a))) if name == "cosh" else math.asinh(float(Rational(a)))
            seen["single"] += 1
        elif "larger" in q.text:
            m = re.search(r"(\d*) cosh x ([+-]) (\d*) ?sinh x = (\S+?)(?: has|\.)", q.text)
            a = int(m.group(1) or 1)
            b = int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
            c = float(Rational(m.group(4)))
            # (a + b)u^2 - 2cu + (a - b) = 0 with u = e^x
            disc = math.sqrt(c * c - (a + b) * (a - b))
            expected = math.log(max((c + disc) / (a + b), (c - disc) / (a + b)))
            seen["combination"] += 1
        else:
            name = re.search(r"integral of (.+?) (?:dx|with)", q.text).group(1)
            k = int(re.search(r"ln (\d+)", q.text).group(1))
            expected = float(mpmath.quad(_integrand(name), [0, math.log(k)]))
            seen["integral"] += 1
        assert abs(_value(q.answer_text) - expected) < 1e-8, (seed, q.text, q.answer_text)
    assert min(seen.values()) >= 20 and len(seen) == 3, seen


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 6, (tier, Counter(answers).most_common(4))


def test_exam_shape_and_marks():
    check_exam(SKILL, EXAM, ["a", "b", "c", "d"], [2, 1, 2, 2], ["fraction", "fraction", "number", "fraction"])


def test_exam_keys_match_independent_derivation():
    for seed in range(80):
        q = _exam(seed)
        c = float(Rational(re.search(r"cosh x = (\S+?)\.", q.stem).group(1)))
        x_value = math.acosh(c)
        assert abs(float(sympy.N(Rational(q.parts[0].answer_text))) - math.sinh(x_value)) < 1e-9
        assert abs(float(sympy.N(Rational(q.parts[1].answer_text))) - math.tanh(x_value)) < 1e-9
        assert abs(float(q.parts[2].answer_text) - x_value) < 5.1e-4
        assert abs(float(sympy.N(Rational(q.parts[3].answer_text))) - math.cosh(2 * x_value)) < 1e-9


def test_exam_follow_through_from_a_wrong_earlier_part():
    check_follow_through(SKILL, EXAM)


def test_last_tier_adds_a_shape():
    assert classify_skill(SKILL)[0] == "escalates"
