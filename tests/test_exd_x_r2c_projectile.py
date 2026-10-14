"""Tests the exam chain for projectile motion in a plane: apex time, greatest height, time of flight, landing."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import pytest
import sympy

from mathgen import make
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_r2c_projectile"
SKILL_ID = "senior-apply-vector-calculus-to-motion-in-a-plane"
HARD_TOPIC = "p10a_projectile_motion_vector"
t = sympy.Symbol("t", positive=True)
_COMPONENT = r"(-?(?:\d+(?:\.\d+)?)?)([ij])"


def _round(value, places: int) -> str:
    return str(Decimal(str(sympy.N(value, 40))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _components(text: str) -> dict[str, sympy.Rational]:
    out = {}
    for sign, number, unit in re.findall(
        r"([+-]?)\s*(\d+(?:\.\d+)?)?([ij])", text.replace(" - ", " -").replace(" + ", " +")
    ):
        value = sympy.Rational(number) if number else sympy.Integer(1)
        out[unit] = -value if sign == "-" else value
    return out


def oracle(q):
    """Re-derive all four answers from r(0) and v(0) as printed in the stem."""
    r0 = _components(re.search(r"r\(0\) = (.+?) (?:metres|m)\b", q.stem).group(1))
    v0 = _components(re.search(r"v\(0\) = (.+?) m/s", q.stem).group(1))
    apex = sympy.Rational(v0["j"], 10)
    height = r0["j"] + v0["j"] * apex - 5 * apex**2
    landing = max(sympy.solve(r0["j"] + v0["j"] * t - 5 * t**2, t))
    x_land = r0["i"] + v0["i"] * landing
    return {"a": _round(apex, 1), "b": _round(height, 1), "c": _round(landing, 2), "d": _round(x_land, 1)}


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        if oracle(q) != canonical(q):
            bad.append((q.stem, canonical(q), oracle(q)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_wrong_apex_time():
    """A wrong (a) used correctly in (b), (c) and (d) earns those parts."""

    def carried(q):
        r0 = _components(re.search(r"r\(0\) = (.+?) (?:metres|m)\b", q.stem).group(1))
        v0 = _components(re.search(r"v\(0\) = (.+?) m/s", q.stem).group(1))
        wrong_apex = sympy.Rational(q.parts[0].answer_text) + sympy.Rational(3, 10)
        height = r0["j"] + v0["j"] * wrong_apex - 5 * wrong_apex**2
        flight = wrong_apex + sympy.sqrt(sympy.Rational(_round(height, 1)) / 5)
        x_land = r0["i"] + v0["i"] * sympy.Rational(_round(flight, 2))
        return "a", _round(wrong_apex, 1), {"b": _round(height, 1), "c": _round(flight, 2), "d": _round(x_land, 1)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
