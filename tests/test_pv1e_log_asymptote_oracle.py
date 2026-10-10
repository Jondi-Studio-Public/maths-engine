"""Independent oracle for the logarithm-asymptote skill (ACMMM156).

Imports nothing from the generator module. Every question is generated through the skill's own
sources. The key is re-derived from the figure alone by a route that is neither the generator's
construction nor its registered reader (which rounds the x of the curve's first sample): three
samples are taken from the curve (heights near -2 and -0.5, and its right-hand end), and for each base
the family allows, v = x - a^y must come out the same for all three. Exactly one base fits, and
the common v is the asymptote.
"""

from __future__ import annotations

import functools

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-recognise-the-qualitative-features-of-the-graph-of-acmmm156"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]
BASES = (2, 3)


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _question(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _nearest(points, y):
    return min(points, key=lambda p: abs(p[1] - y))


def _derive(q):
    (curve,) = q.figure["curves"]
    (piece,) = curve["pieces"]
    pts = [tuple(p) for p in piece]
    picks = [_nearest(pts, y) for y in (-2.0, -0.5, max(y for _, y in pts))]
    assert len({p[0] for p in picks}) == 3
    fits = []
    for a in BASES:
        vs = [x - a**y for x, y in picks]
        if max(vs) - min(vs) < 0.002:
            fits.append((a, sum(vs) / 3))
    assert len(fits) == 1, (q.text, fits)
    a, v = fits[0]
    assert abs(v - round(v)) < 0.002, (q.text, v)
    return int(round(v)), a


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        v, _ = _derive(q)
        assert q.answer_text == f"x = {v}", (tier, seed, q.text)
        assert q.answer == {"x": v}
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    bases, zero = set(), 0
    for seed in range(SEEDS):
        q = _question(tier, seed)
        v, a = _derive(q)
        bases.add(a)
        zero += v == 0
        f = q.figure
        x0, x1 = f["x_range"]
        assert x0 < v and v + 3 <= x1  # asymptote in the window, curve visible beyond it
        if tier == "easy":
            assert a == 2 and v != 0 and (x0, x1) == (-6, 6)
        elif tier == "medium":
            assert (x0, x1) == (-6, 6) and f["x_grid"] == 1
        elif tier == "hard":
            assert f["equal_scale"] is False and f["x_grid"] == 2 and f["y_grid"] == 1
            assert v % 2 == 0 and v != 0  # on a labelled gridline of the scaled axis
        else:
            assert not (x0 <= 0 <= x1)  # no y-axis in the picture
            assert "y-axis" in q.text
    if tier == "medium":
        assert bases == {2, 3} and zero > 0
    elif tier == "easy":
        assert bases == {2} and zero == 0
