"""Tests `ucv1-arc-length` (topics `fy1ca_arc_length` and `fy1ca_arc_length_exam`).

The oracle reads the curve and its limits back out of the question TEXT and integrates sqrt(1 + y'^2) (or the speed of a
parametric curve) with SymPy, then compares with the key to nine decimal places.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy import Rational, log, sqrt

from mathgen import make
from scripts.measure_tiers import classify_skill
from tests.fy1ca_support import TIERS, check_exam, check_follow_through, check_single_tier, check_wiring, load_skill

TOPIC = "fy1ca_arc_length"
EXAM = "fy1ca_arc_length_exam"
SKILL = load_skill("ucv1-arc-length")
x, t = sympy.symbols("x t", positive=True)

CURVES = {
    "(2/3)x^(3/2)": Rational(2, 3) * x ** Rational(3, 2),
    "(1/3)(x² + 2)^(3/2)": Rational(1, 3) * (x**2 + 2) ** Rational(3, 2),
    "x³/6 + 1/(2x)": x**3 / 6 + 1 / (2 * x),
    "x⁴/8 + 1/(4x²)": x**4 / 8 + 1 / (4 * x**2),
    "x²/4 - (1/2)ln x": x**2 / 4 - log(x) / 2,
}


def _value(text: str) -> float:
    return float(sympy.N(sympy.sympify(text.replace("^", "**").replace("ln", "log"), locals={"log": log})))


def _limits(text: str):
    for pattern in (
        r"x = (\d+) to x = (\d+)",
        r"x = (\d+) to the point where x = (\d+)",
        r"x = (\d+) and x = (\d+)",
        r"(\d+) ≤ x ≤ (\d+)",
    ):
        m = re.search(pattern, text)
        if m:
            return int(m.group(1)), int(m.group(2))
    raise AssertionError(text)


def _length(curve, lo, hi) -> float:
    integrand = sqrt(sympy.factor(sympy.together(1 + sympy.diff(curve, x) ** 2)))
    return float(sympy.N(sympy.integrate(integrand, (x, lo, hi)), 20))


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def test_skill_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "expression")


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single_tier(SKILL, TOPIC, tier)


def _check_function_tier(tier, expected_curves, seeds):
    seen = Counter()
    for seed in seeds:
        q = _q(tier, seed)
        name = next((k for k in CURVES if k in q.text), None)
        if name is None:
            continue
        lo, hi = _limits(q.text)
        assert abs(_value(q.answer_text) - _length(CURVES[name], lo, hi)) < 1e-9, (seed, q.text, q.answer_text)
        seen[name] += 1
    assert set(seen) >= expected_curves and min(seen.values()) >= 10, seen


def test_easy_keys_are_exact_lengths():
    _check_function_tier("easy", {"(2/3)x^(3/2)", "(1/3)(x² + 2)^(3/2)"}, range(120))


def test_medium_keys_are_exact_lengths():
    _check_function_tier("medium", {"x³/6 + 1/(2x)", "x⁴/8 + 1/(4x²)"}, range(120))


def test_hard_keys_are_exact_lengths():
    seen = Counter()
    for seed in range(160):
        q = _q("hard", seed)
        if "x = t²" in q.text:
            lo, hi = (Rational(v) for v in re.search(r"(\d+(?:/\d+)?) ≤ t ≤ (\d+(?:/\d+)?)", q.text).groups())
            speed = 2 * t * sqrt(1 + t**2)
            expected = float(sympy.N(sympy.integrate(speed, (t, lo, hi)), 20))
            seen["parametric"] += 1
        else:
            lo, hi = _limits(q.text)
            expected = _length(CURVES["x²/4 - (1/2)ln x"], lo, hi)
            assert "ln" in q.answer_text
            seen["log"] += 1
        assert abs(_value(q.answer_text) - expected) < 1e-9, (seed, q.text, q.answer_text)
    assert min(seen.values()) >= 30 and len(seen) == 2, seen


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 8, (tier, Counter(answers).most_common(4))


def test_exam_shape_and_marks():
    check_exam(SKILL, EXAM, ["a", "b", "c", "d"], [1, 2, 3, 2], ["fraction", "number", "fraction", "number"])


def test_exam_keys_match_independent_derivation():
    for seed in range(80):
        q = _exam(seed)
        lo, hi = _limits(q.stem)
        curve = CURVES["x³/6 + 1/(2x)"]
        slope = sympy.diff(curve, x).subs(x, hi)
        assert Rational(q.parts[0].answer_text) == slope
        assert abs(float(q.parts[1].answer_text) - float(sqrt(1 + slope**2))) < 5e-4
        length = _length(curve, lo, hi)
        assert abs(float(sympy.N(Rational(q.parts[2].answer_text))) - length) < 1e-9
        chord = float(sympy.N(sqrt((hi - lo) ** 2 + (curve.subs(x, hi) - curve.subs(x, lo)) ** 2)))
        assert abs(float(q.parts[3].answer_text) - (length - chord)) < 0.0051


def test_exam_follow_through_from_a_wrong_earlier_part():
    check_follow_through(SKILL, EXAM)


def test_last_tier_adds_a_shape():
    assert classify_skill(SKILL)[0] == "escalates"


def test_exam_difference_working_survives_rounding_first():
    """Rounding the length and the chord to 2 d.p. before subtracting gives the keyed difference."""
    for seed in range(120):
        q = _exam(seed)
        a, b, c = re.search(r"(\d+\.\d\d) - (\d+\.\d\d) = (\d+\.\d\d) m", " ".join(q.parts[3].working)).groups()
        assert abs(float(a) - float(b) - float(c)) < 1e-9, (seed, a, b, c)
