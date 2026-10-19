"""Independent oracle for the period-of-y=sin(bx) skill (ACMMM038).

Imports nothing from the generator module. Every question is generated through the skill's own
sources. The key is re-derived from the drawn curve's samples alone, by a route that is neither
the generator's construction nor its registered reader (which measures zero-to-first-extremum):
the period is found as the one k, among all multiples of pi/12 up to 16pi, for which
sign * sin(2*pi*x/k) reproduces every drawn sample (a least-squares fit, unique), and on the
tiers that show a whole cycle it is cross-checked by the first return of the sample sequence to
its starting value and direction.
"""

from __future__ import annotations

import functools
import math
import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-examine-period-changes-and-the-graphs-of-y"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]
_CANDIDATES = sorted(
    {Fraction(n, 12) for n in range(1, 193)} | {Fraction(2, n) for n in range(1, 16)}
)  # hard now draws k = 2/n for n up to 15


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _question(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _points(fig):
    assert len(fig["curves"]) == 1
    pts = [tuple(p) for piece in fig["curves"][0]["pieces"] for p in piece]
    assert pts == sorted(pts)
    return pts


def _fit_period(pts):
    """The unique k (in units of pi) whose sine reproduces every drawn sample."""
    assert pts[0][1] == 0
    sign = 1 if pts[1][1] > 0 else -1
    stride = max(1, len(pts) // 16)
    probe = pts[1::stride]
    fits = []
    for k in _CANDIDATES:
        w = 2 * math.pi / float(k)
        if all(abs(y - sign * math.sin(w * x)) < 2e-5 for x, y in probe) and all(
            abs(y - sign * math.sin(w * x)) < 2e-5 for x, y in pts
        ):
            fits.append(k)
    assert len(fits) == 1, fits
    return fits[0], sign


def _first_return(pts):
    """First x after the start where the curve is back at zero and heading the same way."""
    sign = 1 if pts[1][1] > 0 else -1
    for i in range(1, len(pts)):
        if pts[i][1] != 0:
            continue
        # the last sample has no successor: a zero there closes the window exactly one cycle in
        if i == len(pts) - 1 or (pts[i + 1][1] > 0) == (sign > 0):
            return Fraction(pts[i][0]).limit_denominator(30)
    return None


def _text(k: Fraction) -> str:
    num = "" if k.numerator == 1 else str(k.numerator)
    return f"{num}pi" if k.denominator == 1 else f"{num}pi/{k.denominator}"


def _on_grid(x, step) -> bool:
    r = x / step
    return abs(r - round(r)) < 1e-3


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        pts = _points(q.figure)
        k, _ = _fit_period(pts)
        assert q.answer_text == _text(k), (tier, seed, q.text)
        if tier != "exam":  # a whole cycle is drawn
            assert _first_return(pts) == k, (tier, seed)
        assert q.answer_text not in q.distractors
        assert q.answer == re.sub(r"(\d)pi", r"\1*pi", q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    negatives = 0
    for seed in range(SEEDS):
        q = _question(tier, seed)
        fig = q.figure
        pts = _points(fig)
        k, sign = _fit_period(pts)
        grid = fig["x_grid"]
        lo, hi = fig["x_range"]
        zeros = [x for x, y in pts if y == 0]
        extremes = [x for x, y in pts if abs(y) == 1]
        assert lo == 0 and abs(max(x for x, _ in pts) - hi) < 1e-6
        if tier == "easy":
            assert sign == 1
            assert abs((hi - lo) / float(k) - 2) < 1e-5  # exactly two cycles
            assert all(_on_grid(x, grid) for x in zeros + extremes)  # peaks AND zeros on gridlines
        elif tier == "medium":
            assert sign == 1
            assert abs((hi - lo) / float(k) - 2) < 1e-5
            assert all(_on_grid(x, grid) for x in zeros)  # zeros on gridlines ...
            assert not any(_on_grid(x, grid) for x in extremes)  # ... peaks are not
            assert abs(grid - float(k) / 2) < 1e-5  # tick spacing = half a period
        elif tier == "hard":
            assert (lo, hi) == (0, 2)
            assert "0 to 2pi" in q.text.replace("x = ", "")
            cycles = 2 / float(k)
            assert abs(cycles - round(cycles)) < 1e-9 and round(cycles) in (
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                14,
                15,
            )
            assert all(_on_grid(x, grid) for x in zeros)
            negatives += sign < 0
        else:
            assert (lo, hi) == (0, 4)
            assert "0 to 4pi" in q.text.replace("x = ", "")
            assert (hi - lo) / float(k) < 1.5  # less than 1.5 cycles shown
            assert extremes and _on_grid(extremes[0], grid)  # first extreme on a gridline
            assert abs(extremes[0] - float(k) / 4) < 1e-5  # a quarter along
            negatives += sign < 0
    if tier in ("hard", "exam"):
        assert 0.35 * SEEDS <= negatives <= 0.65 * SEEDS
