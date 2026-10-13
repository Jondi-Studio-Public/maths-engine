"""`f10-area-perimeter-count-squares`: topic `fr2c_grid_area_perimeter` (easy, medium, hard).

The oracle counts the squares: it fills the polygon drawn by the figure's segments cell by cell (a cell centre inside is a
whole square, a half-square corner is found from the lattice area) and counts the unit sides around the outline.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import figures, make
from tests.fr2c_checks import check_single, skill

SKILL = "f10-area-perimeter-count-squares"
TOPIC = "fr2c_grid_area_perimeter"
TIERS = ("easy", "medium", "hard")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _outline(fig):
    return [tuple(s["from"]) for s in fig["segments"]]


def _inside(point, poly):
    x, y = point
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def _cell_area(poly, cx, cy, n=8):
    """Area of the unit cell at (cx, cy) inside the polygon, by sampling an 8 x 8 lattice of points."""
    count = sum(_inside((cx + (i + 0.5) / n, cy + (j + 0.5) / n), poly) for i in range(n) for j in range(n))
    return count / (n * n)


def _counted_area(fig):
    poly = _outline(fig)
    (x0, x1), (y0, y1) = fig["x_range"], fig["y_range"]
    whole = half = 0
    for cx in range(int(x0), int(x1)):
        for cy in range(int(y0), int(y1)):
            a = _cell_area(poly, cx, cy)
            if a > 0.99:
                whole += 1
            elif a > 0.4:
                half += 1
            else:
                assert a < 0.01
    return whole + half / 2, half


def _side(text):
    m = re.search(r"(\d+) m\b", text)  # the only length in the stem is the side of a square
    return int(m.group(1)) if m else 1


def test_skill_wiring():
    sk = skill(SKILL)
    assert sk.answer == "number" and sk.needs_figure is True and sk.capability_gates == ["figure_stimulus"]
    assert [(s.topic, s.difficulty) for s in sk.sources] == [(TOPIC, t) for t in TIERS]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_valid_and_the_reader_agrees(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert q.figure["kind"] == "cartesian" and not figures.validate(q.figure)
        assert figures.get_reader(TOPIC)(q) == q.answer_text, (tier, seed)


def test_easy_counts_whole_squares():
    for seed in range(80):
        q = _q("easy", seed)
        area, half = _counted_area(q.figure)
        assert half == 0 and area == int(q.answer_text), (seed, q.text)


def test_medium_area_pairs_half_squares_and_perimeter_counts_sides():
    seen = set()
    for seed in range(120):
        q = _q("medium", seed)
        if "area" in q.text:
            seen.add("area")
            area, half = _counted_area(q.figure)
            assert half in (2, 4) and area == int(q.answer_text), (seed, q.text)
        else:
            seen.add("perimeter")
            poly = _outline(q.figure)
            sides = sum(math.dist(p, q2) for p, q2 in zip(poly, poly[1:] + poly[:1]))
            assert sides == int(q.answer_text)
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            assert sides > 2 * (max(xs) - min(xs) + max(ys) - min(ys))  # the notches add length
    assert seen == {"area", "perimeter"}


def test_hard_scales_the_side_of_a_square():
    seen = set()
    for seed in range(160):
        q = _q("hard", seed)
        side = _side(q.text)
        assert side in (2, 3, 5)
        if "area" in q.text:
            seen.add("area")
            area, _half = _counted_area(q.figure)
            assert area * side * side == int(q.answer_text)
        else:
            seen.add("perimeter")
            poly = _outline(q.figure)
            sides = sum(math.dist(p, q2) for p, q2 in zip(poly, poly[1:] + poly[:1]))
            assert sides * side == int(q.answer_text)
    assert seen == {"area", "perimeter"}


def test_prompts_and_figures_vary():
    import json

    for tier in TIERS:
        keys = {(_q(tier, s).text, json.dumps(_q(tier, s).figure, sort_keys=True)) for s in range(60)}
        assert len(keys) >= 40
