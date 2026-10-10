"""Independent oracle for the 1/f vertical-asymptote skill (ACMSM099).

Imports nothing from the generator module. Every question is generated through the skill's own
sources. The key is re-derived from the figure alone by a route that is neither the generator's
construction nor its registered reader (which looks for samples whose y is exactly 0): a line's
zero comes from its slope and intercept, and a parabola is fitted exactly through three of its
samples (Cramer's rule) and its roots found from the discriminant, a double root when the
discriminant vanishes.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-examine-the-relationship-between-the-graph-of-y"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _question(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _line_zero(fig):
    (ln,) = fig["lines"]
    (x1, y1), (x2, y2) = ln["through"]
    for v in (x1, y1, x2, y2):
        assert isinstance(v, int) and -6 <= v <= 6
    m = Fraction(y2 - y1, x2 - x1)
    c = y1 - m * x1  # y = m x + c
    assert m != 0
    return [-c / m], ("line", m, c)


def _quadratic_zeros(fig):
    (curve,) = fig["curves"]
    pts = [tuple(p) for piece in curve["pieces"] for p in piece]
    # three samples well apart in x
    pts.sort()
    chosen = [pts[0], pts[len(pts) // 2], pts[-1]]
    assert chosen[1][0] - chosen[0][0] > 0.5 and chosen[2][0] - chosen[1][0] > 0.5
    (x1, y1), (x2, y2), (x3, y3) = chosen
    # y = A x^2 + B x + C through the three samples
    d = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d
    B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / d
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / d
    disc = B * B - 4 * A * C
    if abs(disc) < 1e-3:
        roots = [-B / (2 * A)]
    else:
        assert disc > 0
        roots = sorted([(-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A)])
    return roots, ("parabola", A, B, C, disc)


def _derive(q):
    fig = q.figure
    roots, info = _line_zero(fig) if fig["lines"] else _quadratic_zeros(fig)
    ints = []
    for r in roots:
        assert abs(r - round(r)) < 0.02, (q.text, roots)
        ints.append(int(round(r)))
    assert all(-5 <= r <= 5 for r in ints)
    return ints, info


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        zeros, _ = _derive(q)
        assert q.answer_text == ", ".join(str(z) for z in zeros), (tier, seed, q.text)
        assert q.answer == zeros
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    shares = {"line": 0, "touch": 0, "pair": 0}
    for seed in range(SEEDS):
        q = _question(tier, seed)
        zeros, info = _derive(q)
        if info[0] == "line":
            shares["line"] += 1
        else:
            shares["touch" if len(zeros) == 1 else "pair"] += 1
        if tier == "easy":
            assert info[0] == "line" and len(zeros) == 1 and zeros[0] != 0
        elif tier == "medium":
            assert info[0] == "parabola" and len(zeros) == 2 and 0 not in zeros
            vertex = (zeros[0] + zeros[1]) / 2
            assert vertex not in zeros  # the turning point is not a zero
        elif tier == "hard":
            assert info[0] == "parabola" and len(zeros) == 1 and zeros[0] != 0  # touches, never crosses
        else:
            assert 0 in zeros  # the graph passes through the origin
            assert info[0] == "line" or len(zeros) == 2
    if tier == "exam":
        assert 0 < shares["line"] <= 0.25 * SEEDS
