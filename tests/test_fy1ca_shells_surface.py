"""Tests `ucv1-shells-surface-area` (topics `fy1ca_shells_surface` and `fy1ca_shells_surface_exam`).

The oracle reads the region, the axis and the limits back out of the question TEXT and evaluates the shell integral
2*pi*(radius)(height) or the surface integral 2*pi*y*sqrt(1 + y'^2) numerically with mpmath, then compares with the key.
"""

from __future__ import annotations

import re
from collections import Counter

import mpmath
import pytest
import sympy
from sympy import Rational, sqrt

from mathgen import make
from scripts.measure_tiers import classify_skill
from tests.fy1ca_support import TIERS, check_exam, check_follow_through, check_single_tier, check_wiring, load_skill

TOPIC = "fy1ca_shells_surface"
EXAM = "fy1ca_shells_surface_exam"
SKILL = load_skill("ucv1-shells-surface-area")
x = sympy.Symbol("x", positive=True)

# longest first, because "y = x" is a prefix of "y = x²"
CURVES = {
    "y = 4 - x²": 4 - x**2,
    "y = 9 - x²": 9 - x**2,
    "y = x²": x**2,
    "y = x³": x**3,
    "y = √x": sqrt(x),
    "y = x": x,
}
PAIRS = {
    "y = x and y = x³": (x, x**3),
    "y = √x and y = x²": (sqrt(x), x**2),
    "y = x² and y = x³": (x**2, x**3),
    "y = 2x and y = x²": (2 * x, x**2),
    "y = 3x and y = x²": (3 * x, x**2),
    "y = 4x and y = x²": (4 * x, x**2),
    "y = 5x and y = x²": (5 * x, x**2),
    "y = 6x and y = x²": (6 * x, x**2),
    "y = x and y = x²": (x, x**2),
    "y = 4x and y = x³": (4 * x, x**3),
    "y = 9x and y = x³": (9 * x, x**3),
    "y = 16x and y = x³": (16 * x, x**3),
}


def _value(text: str) -> float:
    return float(sympy.N(sympy.sympify(text.replace("^", "**"), locals={"pi": sympy.pi})))


def _integral(expr, lo, hi) -> float:
    f = sympy.lambdify(x, expr, "mpmath")
    return float(mpmath.quad(f, [lo, hi]))


def _shell(top, bottom, lo, hi, axis=None) -> float:
    radius = x if axis is None else (x - axis if axis < 0 else axis - x)
    return 2 * float(sympy.pi) * _integral(radius * (top - bottom), lo, hi)


def _surface(y, lo, hi) -> float:
    return 2 * float(sympy.pi) * _integral(y * sqrt(1 + sympy.diff(y, x) ** 2), lo, hi)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def test_skill_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "expression")


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single_tier(SKILL, TOPIC, tier)


def test_easy_keys_are_shell_volumes_about_the_y_axis():
    seen = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        curve = next(k for k in CURVES if k in q.text)
        hi = int(re.findall(r"x = (\d+)", q.text)[-1])
        assert abs(_value(q.answer_text) - _shell(CURVES[curve], 0, 0, hi)) < 1e-6, (seed, q.text, q.answer_text)
        seen[curve] += 1
    assert len(seen) >= 5, seen


def test_medium_keys_are_shell_volumes():
    seen = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        pair = next((k for k in PAIRS if k in q.text), None)
        if pair:
            top, bottom = PAIRS[pair]
            hi = {"y = x and y = x³": 1, "y = √x and y = x²": 1, "y = x² and y = x³": 1, "y = x and y = x²": 1}.get(
                pair
            )
            if hi is None:
                hi = {"y = 4x and y = x³": 2, "y = 9x and y = x³": 3, "y = 16x and y = x³": 4}.get(pair) or int(
                    re.match(r"y = (\d+)x", pair).group(1)
                )
            expected = _shell(top, bottom, 0, hi)
            seen["between"] += 1
        else:
            nums = [int(v) for v in re.findall(r"x = (-?\d+)", q.text)]
            hi, axis = nums[-2], nums[-1]
            expected = _shell(x**2, 0, 0, hi, axis)
            seen["shifted"] += 1
        assert abs(_value(q.answer_text) - expected) < 1e-6, (seed, q.text, q.answer_text)
    assert seen["between"] >= 40 and seen["shifted"] >= 30, seen


def test_hard_keys_are_surface_areas():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        if "x² + y² =" in q.text or "√(" in q.text and "- x²)" in q.text:
            r2 = int(re.search(r"(?:x² \+ y² = |√\()(\d+)", q.text).group(1))
            a, b = (
                int(v) for v in re.search(r"(?:between x = |for )(-?\d+)(?: and x = | ≤ x ≤ )(-?\d+)", q.text).groups()
            )
            expected = _surface(sqrt(r2 - x**2), a, b)
            seen["zone"] += 1
        elif "(3/4)x" in q.text or re.search(r"\((\d+)/(\d+)\)x", q.text):
            p, qq = (int(v) for v in re.search(r"\((\d+)/(\d+)\)x", q.text).groups())
            hi = int(re.search(r"(?:to x = |≤ x ≤ )(\d+)", q.text).group(1))
            expected = _surface(Rational(p, qq) * x, 0, hi)
            seen["cone"] += 1
        elif "x³/6" in q.text:
            lo, hi = (
                int(v)
                for v in re.search(r"(?:from x = |between x = )(\d+)(?: to x = | and x = )(\d+)", q.text).groups()
            )
            expected = _surface(x**3 / 6 + 1 / (2 * x), lo, hi)
            seen["curve"] += 1
        else:
            lo, hi = (int(v) for v in re.search(r"(?:from x = |, )(\d+)(?: to x = | ≤ x ≤ )(\d+)", q.text).groups())
            expected = _surface(sqrt(x), lo, hi)
            seen["paraboloid"] += 1
        assert abs(_value(q.answer_text) - expected) < 1e-4 * max(1, expected), (seed, q.text, q.answer_text, expected)
    assert min(seen.values()) >= 15 and len(seen) == 4, seen


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 6, (tier, Counter(answers).most_common(4))


def test_exam_shape_and_marks():
    check_exam(SKILL, EXAM, ["a", "b", "c", "d"], [1, 2, 3, 2], ["fraction", "fraction", "number", "number"])


def test_exam_keys_match_independent_derivation():
    for seed in range(80):
        q = _exam(seed)
        k = int(re.search(r"y = (\d)x", q.stem).group(1))
        rho = float(re.search(r"density ([\d.]+) g/cm³", q.stem).group(1))
        x0 = int(re.search(r"radius x = (\d)", q.parts[1].prompt).group(1))
        assert int(q.parts[0].answer_text) == k
        assert int(q.parts[1].answer_text) == k * x0 - x0**2
        volume = _shell(k * x, x**2, 0, k)
        assert abs(float(q.parts[2].answer_text) - volume) < 0.051
        assert abs(float(q.parts[3].answer_text) - volume * rho / 1000) < 0.0051


def test_exam_follow_through_from_a_wrong_earlier_part():
    check_follow_through(SKILL, EXAM)


def test_last_tier_adds_a_shape():
    assert classify_skill(SKILL)[0] == "escalates"
