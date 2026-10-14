"""Tests `ucv1-average-value-mvt` (topics `fy1ca_average_value_mvt` and `fy1ca_average_value_mvt_exam`).

The oracle reads the function and the interval back out of the question TEXT, integrates (or solves the slope
condition) with SymPy and mpmath, and compares with the key to nine decimal places.
"""

from __future__ import annotations

import re
from collections import Counter

import mpmath
import pytest
import sympy
from sympy import Rational, exp, sqrt

from mathgen import make
from scripts.measure_tiers import classify_skill
from tests.fy1ca_support import TIERS, check_exam, check_follow_through, check_single_tier, check_wiring, load_skill

TOPIC = "fy1ca_average_value_mvt"
EXAM = "fy1ca_average_value_mvt_exam"
SKILL = load_skill("ucv1-average-value-mvt")
x = sympy.Symbol("x", positive=True)
_NAMES = {"e": sympy.E, "ln": sympy.log, "pi": sympy.pi, "sqrt": sympy.sqrt}

FUNCTIONS = {
    "x² + x": x**2 + x,
    "x² - 4x + 5": x**2 - 4 * x + 5,
    "3x² + 2x": 3 * x**2 + 2 * x,
    "12x² - 4x": 12 * x**2 - 4 * x,
    "5x⁴ + 3x²": 5 * x**4 + 3 * x**2,
    "6x - x²": 6 * x - x**2,
    "2x + 1": 2 * x + 1,
    "x³ - x": x**3 - x,
    "4x³": 4 * x**3,
    "x³": x**3,
    "x⁴": x**4,
    "x²": x**2,
    "sin 2x": sympy.sin(2 * x),
    "sin x": sympy.sin(x),
    "cos x": sympy.cos(x),
    "e^(2x)": exp(2 * x),
    "e^(-x)": exp(-x),
    "e^x": exp(x),
    "1/x²": 1 / x**2,
    "1/x": 1 / x,
    "√x": sqrt(x),
}


def _value(text: str) -> float:
    return float(sympy.N(sympy.sympify(text.replace("^", "**"), locals=_NAMES)))


def _bound(text: str) -> float:
    text = re.sub(r"ln (\d+)", r"ln(\1)", text)
    return _value(re.sub(r"√(\d+)", r"sqrt(\1)", text.replace("π", "pi")))


def _interval(text: str):
    m = re.search(r"\[([^,\]]+), ([^\]]+)\]", text) or re.search(r"over (.+?) ≤ x ≤ (.+)\.$", text)
    return _bound(m.group(1)), _bound(m.group(2))


def _mean(expr, lo, hi) -> float:
    return float(mpmath.quad(sympy.lambdify(x, expr, "mpmath"), [lo, hi])) / (hi - lo)


def _t_limits(text: str):
    m = re.search(r"t = (\d+) to t = (\d+)", text) or re.search(r"(\d+) ≤ t ≤ (\d+)", text)
    return int(m.group(1)), int(m.group(2))


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def test_skill_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "expression")


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single_tier(SKILL, TOPIC, tier)


def test_easy_keys_are_average_values():
    seen = set()
    for seed in range(150):
        q = _q("easy", seed)
        name = re.search(r"f\(x\) = (.+?)(?: on| over)", q.text).group(1)
        lo, hi = _interval(q.text)
        assert abs(_value(q.answer_text) - _mean(FUNCTIONS[name], lo, hi)) < 1e-9, (seed, q.text, q.answer_text)
        seen.add(name)
    assert len(seen) >= 15, seen


def _solve_slope(f, lo, hi):
    slope = (f.subs(x, hi) - f.subs(x, lo)) / (hi - lo)
    roots = [r for r in sympy.solve(sympy.Eq(sympy.diff(f, x), slope), x) if r.is_real and lo < r < hi]
    assert len(roots) == 1, (f, lo, hi, roots)
    return float(roots[0])


def test_medium_keys_are_the_c_of_the_theorem():
    seen = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        if "Rolle" in q.text:
            p = int(
                re.search(r"f\(0\) = f\((\d+)\)|\[0, (\d+)\]", q.text).group(1)
                or re.search(r"\[0, (\d+)\]", q.text).group(1)
            )
            f = x**2 * (x - p) if "x²(x" in q.text else x * (x - p) ** 2
            roots = [r for r in sympy.solve(sympy.diff(f, x), x) if 0 < r < p]
            assert len(roots) == 1 and abs(_value(q.answer_text) - float(roots[0])) < 1e-9, (seed, q.text)
            seen["rolle"] += 1
        else:
            f = sqrt(x) if "√x" in q.text else 1 / x
            lo, hi = (int(v) for v in re.search(r"\[(\d+), (\d+)\]", q.text).groups())
            assert abs(_value(q.answer_text) - _solve_slope(f, lo, hi)) < 1e-9, (seed, q.text, q.answer_text)
            seen["sqrt" if "√x" in q.text else "reciprocal"] += 1
    assert min(seen.values()) >= 15 and len(seen) == 3, seen


def test_hard_keys_are_averages_and_f_of_c():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        if "x e^(x²)" in q.text:
            hi = int(
                re.search(r"0 ≤ x ≤ (\d+)|\[0, (\d+)\]", q.text).group(1) or re.search(r"\[0, (\d+)\]", q.text).group(1)
            )
            expected = _mean(x * exp(x**2), 0, hi)
            seen["exp"] += 1
        elif "√(x²" in q.text:
            k = int(re.search(r"√\(x² \+ (\d+)\)", q.text).group(1))
            hi = int(
                re.search(r"0 ≤ x ≤ (\d+)|\[0, (\d+)\]", q.text).group(1) or re.search(r"\[0, (\d+)\]", q.text).group(1)
            )
            expected = _mean(x * sqrt(x**2 + k), 0, hi)
            seen["root"] += 1
        elif "1/(1 + x²)" in q.text:
            lo, hi = _interval(q.text)
            expected = _mean(1 / (1 + x**2), lo, hi)
            seen["arctan"] += 1
        else:
            b = int(re.search(r"\[0, (\d+)\]", q.text).group(1))
            average = _mean(sqrt(x), 0, b)
            expected = average**2  # f(c) = sqrt(c) equals the average
            seen["c"] += 1
        assert abs(_value(q.answer_text) - expected) < 1e-9, (seed, q.text, q.answer_text)
    assert min(seen.values()) >= 15 and len(seen) == 4, seen


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 6, (tier, Counter(answers).most_common(4))


def test_exam_shape_and_marks():
    check_exam(SKILL, EXAM, ["a", "b", "c", "d"], [2, 1, 2, 2], ["fraction", "fraction", "fraction", "number"])


def test_exam_keys_match_independent_derivation():
    for seed in range(80):
        q = _exam(seed)
        lo, hi = _t_limits(q.stem)
        total = float(mpmath.quad(lambda t: mpmath.sqrt(t), [lo, hi]))
        average = total / (hi - lo)
        assert abs(float(sympy.N(Rational(q.parts[0].answer_text))) - total) < 1e-9
        assert abs(float(sympy.N(Rational(q.parts[1].answer_text))) - average) < 1e-9
        assert abs(float(sympy.N(Rational(q.parts[2].answer_text))) - average**2) < 1e-9
        assert abs(float(q.parts[3].answer_text) - (hi**0.5 - average) / average * 100) < 0.051


def test_exam_follow_through_from_a_wrong_earlier_part():
    check_follow_through(SKILL, EXAM)


def test_last_tier_adds_a_shape():
    assert classify_skill(SKILL)[0] == "escalates"
