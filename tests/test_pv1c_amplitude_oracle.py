"""Independent oracle for the amplitude skill (ACMMM037, y = a sin x).

Imports nothing from the generator module. Every question is generated through the skill's
own sources. The key is re-derived from the figure's sampled curve by least squares of y
against sin(pi x) over ALL samples (not the registered reader's per-sample ratio), and the
sign is cross-checked from the x-axis window: the sign of the curve on the first lobe times
(-1)^s, where s is the window's left end.
"""

from __future__ import annotations

import functools
import math

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-examine-amplitude-changes-and-the-graphs-of-y"
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


def _samples(fig):
    pts = [tuple(p) for c in fig["curves"] for piece in c["pieces"] for p in piece]
    assert len(fig["curves"]) == 1 and len(fig["curves"][0]["pieces"]) == 1
    return pts


def _least_squares_a(pts):
    num = sum(y * math.sin(math.pi * x) for x, y in pts)
    den = sum(math.sin(math.pi * x) ** 2 for x, _ in pts)
    return num / den


def _derive(q):
    fig = q.figure
    pts = _samples(fig)
    assert fig["x_tick_format"] == "pi" and fig["y_range"] == [-7, 7]
    a_fit = _least_squares_a(pts)
    a = round(a_fit)
    assert abs(a_fit - a) < 1e-4, (q.text, a_fit)
    # the whole peak is shown and sits on a y gridline
    peak = max(abs(y) for _, y in pts)
    assert abs(peak - abs(a)) < 1e-4 and abs(a) <= 6
    return a, pts


def _first_lobe_sign_route(fig, pts):
    """Sign of a from the window: the lobe starting at s is above the axis iff a * (-1)^s > 0."""
    s = fig["x_range"][0]
    y_mid = next(y for x, y in pts if x == s + 0.5)
    return 1 if y_mid * (-1) ** s > 0 else -1


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        a, _ = _derive(q)
        expected = a + 1 if CORRUPT else a
        assert q.answer_text == str(expected), (tier, seed, q.text)
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        a, pts = _derive(q)
        fig = q.figure
        lo, hi = fig["x_range"]
        assert "amplitude" not in q.text.lower()
        assert "y = a sin x" in q.text
        sign_from_window = _first_lobe_sign_route(fig, pts)
        assert sign_from_window == (1 if a > 0 else -1), (tier, seed)
        if tier == "easy":
            assert a > 0 and (lo, hi) == (-2, 2)
        elif tier == "medium":
            assert (lo, hi) == (-2, 2) and a != 0
        elif tier == "hard":
            assert hi - lo == 3 and (lo > 0 or hi < 0)  # x = 0 not in the window
            assert fig["x_grid"] == 0.5
        else:
            assert hi - lo == 1 and (lo > 0 or hi < 0)
            assert fig["x_grid"] == 0.25
        # a peak (or trough) sample exists in the window at s + 1/2, at the signed height
        s = lo
        y_peak = next(y for x, y in pts if x == s + 0.5)
        assert abs(y_peak - a * (-1) ** s) < 1e-6
