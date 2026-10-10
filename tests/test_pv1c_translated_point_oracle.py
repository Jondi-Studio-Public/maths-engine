"""Independent oracle for the translated-point skill (ACMMM025, y = f(x) + a).

Imports nothing from the generator module. Every question is generated through the skill's own
sources. The key is re-derived from the figure by a route that is not the registered reader's:
the curve through P is found by linear interpolation of each polyline at x = p; the vertical
gap is the MEAN of g - f over every sample x of f that g also covers (interpolated); and the
image y is q + that gap, which must equal g interpolated at p. The prompt text supplies the
stated convention (same x-coordinate) and each tier's stated constant.
"""

from __future__ import annotations

import bisect
import functools
import re

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-examine-translations-and-the-graphs-of-y-f"
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
    """A polyline curve prepared for interpolation."""

    def __init__(self, curve):
        self.pieces = curve["pieces"]
        self.xs = [[x for x, _ in pc] for pc in self.pieces]

    def at(self, x):
        for pc, xs in zip(self.pieces, self.xs):
            if xs[0] - 1e-9 <= x <= xs[-1] + 1e-9:
                i = max(1, min(bisect.bisect_left(xs, x - 1e-9), len(xs) - 1))
                (x0, y0), (x1, y1) = pc[i - 1], pc[i]
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return None


def _at(curve, x):
    return curve.at(x)


def _derive(q):
    fig = q.figure
    assert fig["alt"] == "two curves on a coordinate grid"
    assert fig["x_range"] == [-8, 8] and fig["y_range"] == [-8, 8]
    assert len(fig["curves"]) == 2 and len(fig["points"]) == 1
    (px, py) = fig["points"][0]["at"]
    assert px == int(px) and py == int(py)
    cs = [_Curve(c) for c in fig["curves"]]
    on = [c for c in cs if (_at(c, px) is not None and abs(_at(c, px) - py) < 1e-6)]
    assert len(on) == 1, "P must be on exactly one curve"
    f = on[0]
    g = next(c for c in cs if c is not f)
    gaps = []
    for piece in f.pieces:
        for x, y in piece:
            gy = _at(g, x)
            if gy is not None:
                gaps.append(gy - y)
    assert len(gaps) >= 5
    mean = sum(gaps) / len(gaps)
    assert all(abs(d - mean) < 1e-4 for d in gaps), "not a pure vertical translation"
    a = round(mean)
    assert a != 0 and abs(mean - a) < 1e-4
    y_img = round(py + a)
    assert abs(_at(g, px) - y_img) < 1e-6
    return int(px), int(py), a, y_img, f, g


def _parse(text):
    m = re.fullmatch(r"\((-?\d+), (-?\d+)\)", text)
    assert m, text
    return int(m.group(1)), int(m.group(2))


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        p, qq, a, y_img, f, g = _derive(q)
        expected = (p, y_img + 1) if CORRUPT else (p, y_img)
        assert _parse(q.answer_text) == expected, (tier, seed, q.text)
        assert (q.answer["x"], q.answer["y"]) == (p, y_img)
        assert q.answer_text not in q.distractors
        assert len({q.answer_text, *q.distractors}) == 1 + len(q.distractors)


def _is_straight(curve):
    for piece in curve.pieces:
        slopes = [(y1 - y0) / (x1 - x0) for (x0, y0), (x1, y1) in zip(piece, piece[1:])]
        if any(abs(s - slopes[0]) > 1e-4 for s in slopes):
            return False
    return True


def _central_slope(curve, px):
    h = 1 / 12
    y1, y0 = _at(curve, px + h), _at(curve, px - h)
    assert y1 is not None and y0 is not None
    return (y1 - y0) / (2 * h)


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    signs = set()
    for seed in range(SEEDS):
        q = _question(tier, seed)
        p, qq, a, y_img, f, g = _derive(q)
        assert "same x-coordinate" in q.text or "vertical line through" in q.text
        assert "y = f(x) + a" in q.text
        assert not re.search(r"\ba = |\bf\(x\) = ", q.text)
        signs.add(a > 0)
        assert -8 <= p <= 8 and -8 <= qq <= 8 and -8 <= y_img <= 8  # inside the window
        if tier == "easy":
            assert a > 0 and "positive" in q.text
            assert abs(_central_slope(f, p)) <= 1 + 1e-3
        elif tier == "medium":
            assert abs(a) <= 5 and "positive or negative" in q.text
            assert abs(_central_slope(f, p)) <= 1 + 1e-3
        elif tier == "hard":
            assert abs(a) <= 5 and "nonzero whole number" in q.text
            assert round(abs(_central_slope(f, p))) in (2, 3, 4)
            # the other curve crosses the row of P at a grid point, which is NOT the key
            hs = [
                h
                for h in (-4, -3, -2, -1, 1, 2, 3, 4)
                if (_at(g, p + h) is not None) and abs(_at(g, p + h) - qq) < 1e-4
            ]
            assert hs, (seed, q.text)
            assert all((p + h, qq) != (p, y_img) for h in hs)
        else:
            assert "straight line" in q.text
            assert _is_straight(f) and _is_straight(g)
            m = _central_slope(f, p)
            assert abs(m) > 0.3
            b = a / m
            assert abs(b - round(b)) < 1e-3 and round(b) not in (0, a)
            b = round(b)
            # the horizontal reading also lands on the other line, and is not the key
            assert abs(_at(g, p - b) - qq) < 1e-4
            assert (p - b, qq) != (p, y_img)
    assert signs == ({True} if tier == "easy" else {True, False})
