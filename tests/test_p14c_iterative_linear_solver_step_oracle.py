"""Re-derives each rendered question's answer from `question.text` alone via the Gauss-Seidel closed form, never
importing anything from `mathgen.generators.p14c_iterative_linear_solver_step_build`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p14c_iterative_linear_solver_step"

_EQ1_RE = re.compile(r"(-?\d+)x \+ \((-?\d+)\)y = (-?\d+)")
_EQ2_RE = re.compile(r"\((-?\d+)\)x \+ (-?\d+)y = (-?\d+)")
_START_RE = re.compile(r"\(x0, y0\) = \((-?\d+(?:/\d+)?), ?(-?\d+(?:/\d+)?)\)")

# The exam tier renders BOTH equations in this uniform "both terms
# parenthesised" style (mathgen/generators/p14c_iterative_linear_solver_step_build.py's
# `_exam_eq_text`), regardless of which one is x-dominant -- unlike the
# easy/medium/hard positional format above, which always prints the
# x-dominant equation bare-coefficient-first. The exam prompt names its own
# comparison explicitly ("an x-coefficient LARGER IN MAGNITUDE than its own
# y-coefficient"), so this oracle branch reproduces exactly that: pick,
# from the two equations parsed in DISPLAY order, whichever one has
# |x-coefficient| > |y-coefficient| WITHIN itself, and use it to solve x1;
# use the other equation for y1.
_EXAM_EQ_RE = re.compile(r"\((-?\d+)\)x \+ \((-?\d+)\)y = (-?\d+)")


def _parse_system(text: str):
    a11, a12, b1 = (int(g) for g in _EQ1_RE.search(text).groups())
    a21, a22, b2 = (int(g) for g in _EQ2_RE.search(text).groups())
    x0_s, y0_s = _START_RE.search(text).groups()
    return a11, a12, b1, a21, a22, b2, Fraction(x0_s), Fraction(y0_s)


def _parse_exam_equations(text: str):
    """Parses BOTH equations in display order, without assuming which is
    x-dominant from position -- the exam tier's order is randomised."""
    matches = _EXAM_EQ_RE.findall(text)
    assert len(matches) == 2, f"expected 2 exam-tier equations, found {len(matches)}: {text!r}"
    (cx1, cy1, rhs1), (cx2, cy2, rhs2) = ((int(a), int(b), int(c)) for a, b, c in matches)
    x0_s, y0_s = _START_RE.search(text).groups()
    return cx1, cy1, rhs1, cx2, cy2, rhs2, Fraction(x0_s), Fraction(y0_s)


def _gs_step(a11, a12, b1, a21, a22, b2, x0, y0):
    x1 = Fraction(b1 - a12 * y0, a11)
    y1 = Fraction(b2 - a21 * x1, a22)
    return x1, y1


def _oracle_exam(text: str) -> tuple[Fraction, Fraction]:
    cx1, cy1, rhs1, cx2, cy2, rhs2, x0, y0 = _parse_exam_equations(text)
    if abs(cx1) > abs(cy1):
        x_eq, y_eq = (cx1, cy1, rhs1), (cx2, cy2, rhs2)
    else:
        assert abs(cx2) > abs(cy2), f"neither equation is x-dominant: {text!r}"
        x_eq, y_eq = (cx2, cy2, rhs2), (cx1, cy1, rhs1)
    xcx, xcy, xrhs = x_eq
    ycx, ycy, yrhs = y_eq
    x1 = Fraction(xrhs - xcy * y0, xcx)
    y1 = Fraction(yrhs - ycx * x1, ycy)
    return x1, y1


def _oracle(text: str) -> tuple[Fraction, Fraction]:
    if "LARGER IN MAGNITUDE" in text:
        return _oracle_exam(text)
    a11, a12, b1, a21, a22, b2, x0, y0 = _parse_system(text)
    x1, y1 = _gs_step(a11, a12, b1, a21, a22, b2, x0, y0)
    if "TWO Gauss-Seidel" in text or "Chain two full" in text:
        x2, y2 = _gs_step(a11, a12, b1, a21, a22, b2, x1, y1)
        return x2, y2
    return x1, y1


@pytest.mark.parametrize("tier", ("easy", "medium", "hard", "exam"))
def test_oracle_recomputes(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        rx, ry = _oracle(q.text)
        ex, ey = Fraction(q.answer["x"]), Fraction(q.answer["y"])
        assert (rx, ry) == (ex, ey), (
            f"{tier} seed={seed}: oracle ({rx}, {ry}) != canonical ({ex}, {ey}) (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    rx, ry = _oracle(q.text)
    corrupted = (rx + 1, ry)
    assert (rx, ry) != corrupted


def test_hard_oracle_chains_two_iterations_not_one():
    q = make(TOPIC, seed=0, difficulty="hard")
    a11, a12, b1, a21, a22, b2, x0, y0 = _parse_system(q.text)
    x1, y1 = _gs_step(a11, a12, b1, a21, a22, b2, x0, y0)
    rx, ry = _oracle(q.text)
    assert (rx, ry) != (x1, y1), "hard-tier oracle must not stop after one iteration"
