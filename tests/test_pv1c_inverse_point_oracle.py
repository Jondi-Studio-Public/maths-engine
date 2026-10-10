"""Independent oracle for the inverse-point skill (ACMSM096).

Imports nothing from the generator module. Every question is generated through the skill's own
sources. The key is re-derived from the figure by a route that is not the registered reader's:
the reader swaps P and checks the other curve; here the curve not through P (or, if P is on
both curves, either) is read BY INTERPOLATION at x = q to give the image's y, and the image
(q, g(q)) is compared with the key, then the swap property is asserted separately from the
curve data (f at x = p is q, g at x = q is p). The prompt text supplies the stated convention.
"""

from __future__ import annotations

import bisect
import functools
import re

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-examine-the-reflection-property-of-the-graph-of"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]
CORRUPT = False  # flipped to True by the red-first check only


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _question(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


class _Curve:
    def __init__(self, curve):
        self.pts = sorted((x, y) for pc in curve["pieces"] for x, y in pc)
        self.xs = [x for x, _ in self.pts]

    def at(self, x):
        if not self.xs or x < self.xs[0] - 1e-9 or x > self.xs[-1] + 1e-9:
            return None
        i = max(1, min(bisect.bisect_left(self.xs, x - 1e-9), len(self.xs) - 1))
        (x0, y0), (x1, y1) = self.pts[i - 1], self.pts[i]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def decreasing(self):
        ys = [y for _, y in self.pts]
        return all(b < a for a, b in zip(ys, ys[1:]))

    def increasing(self):
        ys = [y for _, y in self.pts]
        return all(b > a for a, b in zip(ys, ys[1:]))


def _derive(q):
    fig = q.figure
    assert fig["alt"] == "two curves on a coordinate grid"
    assert fig["x_range"] == [-6, 6] and fig["y_range"] == [-6, 6]
    assert len(fig["curves"]) == 2 and len(fig["points"]) == 1
    assert "on the graph of y = f(x)" in q.text  # the prompt says which curve P is on
    (px, py) = fig["points"][0]["at"]
    assert px == int(px) and py == int(py)
    px, py = int(px), int(py)
    cs = [_Curve(c) for c in fig["curves"]]
    on = [c for c in cs if c.at(px) is not None and abs(c.at(px) - py) < 1e-6]
    assert on, "P is on neither curve"
    f = on[0]
    g = next((c for c in cs if c is not f), f)
    if len(on) == 2:
        g = on[1]
    # the image: read the OTHER curve at x = q (the swapped first coordinate)
    gy = g.at(py)
    assert gy is not None and abs(gy - round(gy)) < 1e-6
    return px, py, (py, round(gy)), f, g, len(on)


def _parse(text):
    m = re.fullmatch(r"\((-?\d+), (-?\d+)\)", text)
    assert m, text
    return int(m.group(1)), int(m.group(2))


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        p, qq, img, f, g, n_on = _derive(q)
        expected = (img[0], img[1] + 1) if CORRUPT else img
        assert _parse(q.answer_text) == expected, (tier, seed, q.text)
        assert (q.answer["x"], q.answer["y"]) == img
        # the reflection property itself, from the curves: f(p) = q and g(q) = p
        assert abs(f.at(p) - qq) < 1e-6 and abs(g.at(qq) - p) < 1e-6
        assert img == (qq, p)
        assert q.answer_text not in q.distractors
        assert len({q.answer_text, *q.distractors}) == 1 + len(q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        p, qq, img, f, g, n_on = _derive(q)
        assert -5 <= p <= 5 and -5 <= qq <= 5
        assert not re.search(r"\bp = |\bq = ", q.text)
        if tier == "easy":
            (ln,) = q.figure["lines"]
            (x0, y0), (x1, y1) = ln["through"]
            assert (x0, y0) == (0, 0) and (y1 - y0) == (x1 - x0) and ln["label"] == "y = x"
            assert p != qq and p > 0 and qq > 0 and n_on == 1
            assert f.increasing() and g.increasing()
            assert "y = x" in q.text
        else:
            assert q.figure["lines"] == [] and "y = x" not in q.text
            assert "mirror" not in q.text and "reflection" not in q.text
        if tier == "medium":
            assert p != qq and n_on == 1 and f.increasing() and g.increasing()
        elif tier == "hard":
            assert p != qq and n_on == 1
            assert f.decreasing() and g.decreasing()
            assert p < 0 or qq < 0
        elif tier == "exam":
            assert p != qq and n_on == 2  # P is on both curves, NOT on y = x
            assert img == (qq, p) and img != (p, qq)
            assert f.decreasing() and g.decreasing()
