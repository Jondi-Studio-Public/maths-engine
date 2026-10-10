"""Independent oracle for the hyperbola-asymptote skill (ACMMM013).

Imports nothing from the generator module. Every question is generated through the skill's own
sources. The key is re-derived from the figure alone by an algebraic route that is neither the
generator's construction nor its registered reader (which reads the break in the polyline and
the level ends): two samples of a branch are fitted to y = k + 1/(x - h), a quadratic in
u = x - h, and the fit is checked on both branches independently.
"""

from __future__ import annotations

import functools
import math

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-recognise-features-of-the-graphs-of-y-frac1x"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _question(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _fit_branch(piece, right):
    """(h, k) from two samples of one branch of y = k + 1/(x - h): u(u + d) = d / (y1 - y2)."""
    x1, y1 = piece[0]
    x2, y2 = piece[len(piece) // 3]
    d, dy = x2 - x1, y1 - y2
    assert abs(dy) > 0.05 and abs(d) > 0.05
    # u1 = x1 - h solves u1 (u1 + d) = d / dy  ->  u1^2 + d u1 - d/dy = 0
    disc = d * d + 4 * d / dy
    assert disc > 0
    roots = [(-d + s * math.sqrt(disc)) / 2 for s in (1, -1)]
    # the right-hand branch has u1 > 0 (x1 > h), the left-hand one u1 < 0
    u1 = max(roots) if right else min(roots)
    return x1 - u1, y1 - 1 / u1


def _derive(q):
    pieces = sorted(q.figure["curves"][0]["pieces"], key=lambda p: p[0][0])
    assert len(pieces) == 2
    (hl, kl), (hr, kr) = _fit_branch(pieces[0], False), _fit_branch(pieces[1], True)
    h, k = round(hr), round(kr)
    # the two branches, fitted separately, agree with each other and sit on whole numbers
    assert abs(hl - hr) < 0.05 and abs(kl - kr) < 0.05, (q.text, hl, hr, kl, kr)
    assert abs(hr - h) < 0.05 and abs(kr - k) < 0.05
    return h, k


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        h, k = _derive(q)
        assert q.answer_text == f"x = {h}, y = {k}", (tier, seed, q.text)
        assert q.answer == {"x": h, "y": k}
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        h, k = _derive(q)
        f = q.figure
        (x0, x1), (y0, y1) = f["x_range"], f["y_range"]
        assert x0 + 3 <= h <= x1 - 3 and y0 + 3 <= k <= y1 - 3  # every feature is inside the window
        if tier == "easy":
            assert h != 0 and k != 0 and h != k
            assert f["equal_scale"] is True and f["y_grid"] == 1
        elif tier == "medium":
            assert h == 0 or k == 0  # an axis IS an asymptote
        elif tier == "hard":
            assert f["equal_scale"] is False and f["y_grid"] == 2 and f["x_grid"] == 1
            assert k % 2 == 0 and k != 0 and h != 0  # on a labelled gridline of the scaled axis
            assert "2 units" in q.text or "2s" in q.text
        else:
            assert not (x0 <= 0 <= x1) and not (y0 <= 0 <= y1)  # no axis anywhere in the picture
            assert "axes" in q.text or "edge" in q.text
