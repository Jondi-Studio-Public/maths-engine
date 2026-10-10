"""Independent oracle for the tangent-slope skill (ACMMM085).

Imports nothing from the generator module. Every question is generated through the skill's
own sources. The key is re-derived from the figure alone by a route that is neither the
generator's construction nor its registered reader: the gradient of the sampled curve at the
stated x is fitted by least squares over the neighbouring samples (not a two-point
difference), and the drawn line whose through-point gradient matches it is the tangent.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-interpret-the-derivative-as-the-slope-or-gradient"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _question(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _stated_x(q):
    m = re.search(r"x = (-?\d+)", q.text)
    assert m, q.text
    return int(m.group(1))


def _lines(fig):
    """[(gradient, through-points)] with every through-point an integer grid point in the window."""
    out = []
    for ln in fig["lines"]:
        (x1, y1), (x2, y2) = ln["through"]
        for v in (x1, y1, x2, y2):
            assert isinstance(v, int) and -6 <= v <= 6
        assert x1 != x2
        out.append((Fraction(y2 - y1, x2 - x1), ((x1, y1), (x2, y2))))
    return out


def _curve_points(fig):
    return [tuple(p) for c in fig["curves"] for piece in c["pieces"] for p in piece]


def _fitted_slope(fig, a):
    near = [(x, y) for x, y in _curve_points(fig) if abs(x - a) <= 0.25]
    assert len(near) >= 5
    n = len(near)
    mx = sum(x for x, _ in near) / n
    my = sum(y for _, y in near) / n
    return sum((x - mx) * (y - my) for x, y in near) / sum((x - mx) ** 2 for x, _ in near)


def _height_at(fig, a):
    ys = [y for x, y in _curve_points(fig) if x == a]
    assert ys, "no curve sample at the stated x"
    return ys[0]


def _derive(q):
    fig = q.figure
    a = _stated_x(q)
    lines = _lines(fig)
    s = _fitted_slope(fig, a)
    best, pts = min(lines, key=lambda g: abs(float(g[0]) - s))
    assert abs(float(best) - s) < 0.1, (q.text, [g for g, _ in lines], s)
    fa = _height_at(fig, a)
    assert fa == int(fa) and -6 <= fa <= 6  # the touch point is a grid intersection
    (x1, y1), _ = pts
    assert y1 - fa == best * (x1 - a)  # the tangent passes through (a, f(a))
    return best, a, fa, lines, s


def _text(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        best, *_ = _derive(q)
        assert q.answer_text == _text(best), (tier, seed, q.text)
        assert q.answer_text not in q.distractors


def _second_meeting(fig, a, fa, m):
    """True if the curve minus the tangent changes sign (or touches zero) between samples away from a."""
    diffs = [(x, y - (fa + float(m) * (x - a))) for x, y in sorted(_curve_points(fig)) if abs(x - a) > 1.0]
    return any(d1 * d2 <= 0 for (_, d1), (_, d2) in zip(diffs, diffs[1:]))


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    zero = 0
    for seed in range(SEEDS):
        q = _question(tier, seed)
        best, a, fa, lines, s = _derive(q)
        if tier == "easy":
            assert len(lines) == 1 and best.denominator == 1 and best > 0
        elif tier == "medium":
            assert len(lines) == 1
            zero += best == 0
        elif tier == "hard":
            assert len(lines) == 1 and best != 0
            assert _second_meeting(q.figure, a, fa, best), (seed, "no second meeting point")
        else:
            assert len(lines) == 2 and lines[0][0] != lines[1][0]
            other = next(g for g, _ in lines if g != best)
            assert abs(float(other) - s) > 0.1  # the secant is visibly not tangent
            for g, ((x1, y1), _) in lines:  # both lines pass through the touch point
                assert y1 - fa == g * (x1 - a)
    if tier == "medium":
        assert 0 < zero <= 0.20 * SEEDS
