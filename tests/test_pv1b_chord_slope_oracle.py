"""Independent oracle for senior-interpret-the-ratios-frac-f-left-x-h.

Imports nothing from ``mathgen.generators.pv1b_chord_slope_build``. Questions are
generated through the skill's own sources. The expected slope is re-derived from
the figure's CURVE (heights read by linear interpolation of the sampled
polyline at the prompt's stated x-values), not from the drawn line and not by
the registered reader's through-point lookup; the drawn line is then checked
against it.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from fractions import Fraction

import pytest

import mathgen
from mathgen.grading.typed import grade
from mathgen.skills import load_skills

SKILL_ID = "senior-interpret-the-ratios-frac-f-left-x-h"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]
CORRUPT = False  # flipped by the red-first run only


@functools.lru_cache(maxsize=None)
def _sources():
    _, skills = load_skills()
    return {s.difficulty: s for s in {s.id: s for s in skills}[SKILL_ID].sources}


def _make(tier, seed):
    src = _sources()[tier]
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _stated(q):
    xs = [int(v) for v in re.findall(r"\b[xa] = (-?\d+)", q.text)]
    hs = re.findall(r"\bh = (-?\d+)", q.text)
    return xs[0], (xs[0] + int(hs[0]) if hs else xs[1])


def _height(q, x):
    """Curve height at x by interpolating the sampled polyline; must be a grid integer."""
    for piece in q.figure["curves"][0]["pieces"]:
        for (xa, ya), (xb, yb) in zip(piece, piece[1:]):
            if xa <= x <= xb:
                y = ya + (yb - ya) * (x - xa) / (xb - xa) if xb != xa else ya
                assert abs(y - round(y)) < 1e-3, f"height {y} at x={x} is off the grid"
                return round(y)
    raise AssertionError(f"curve has no point at x = {x}")


def _slope(line):
    (xa, ya), (xb, yb) = line["through"]
    return Fraction(yb - ya, xb - xa)


def _oracle(q):
    x1, x2 = _stated(q)
    assert x1 != x2
    expected = Fraction(_height(q, x2) - _height(q, x1), x2 - x1)
    return expected + 1 if CORRUPT else expected


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_key_matches_from_figure_and_prompt(tier):
    for seed in range(SEEDS):
        q = _make(tier, seed)
        expected = _oracle(q)
        assert Fraction(q.answer_text) == expected, (
            f"{tier} seed {seed}: oracle {expected} vs key {q.answer_text}\n{q.text}"
        )
        assert grade("fraction", q, q.answer_text).correct
        x1, x2 = _stated(q)
        chords = [
            l
            for l in q.figure["lines"]
            if _slope(l) == expected
            and {tuple(p) for p in l["through"]} == {(x1, _height(q, x1)), (x2, _height(q, x2))}
        ]
        assert len(chords) == 1, f"{tier} seed {seed}: no single drawn line is the chord"


@pytest.mark.parametrize("tier", TIERS)
def test_boundary_assertions_from_the_figure(tier):
    for seed in range(SEEDS):
        q = _make(tier, seed)
        fig = q.figure
        x1, x2 = _stated(q)
        (xlo, xhi), (ylo, yhi) = fig["x_range"], fig["y_range"]
        assert fig["grid"] == 1 and fig["x_grid"] == 1 and fig["y_grid"] == 1
        assert len(fig["curves"]) == 1 and fig["curves"][0]["pieces"]
        assert fig["segments"] == []
        for x in (x1, x2):
            y = _height(q, x)
            assert xlo <= x <= xhi and ylo <= y <= yhi, "chord point outside the window"
        # the chord's through-points are integer grid points that lie on the curve at the stated x
        on = {(x1, _height(q, x1)), (x2, _height(q, x2))}
        chord_lines = [l for l in fig["lines"] if {tuple(p) for p in l["through"]} == on]
        assert len(chord_lines) == 1
        for p in chord_lines[0]["through"]:
            assert all(isinstance(c, int) for c in p)
        if tier == "hard":
            assert x2 < x1, "hard: h must be negative (second point left of the first)"
        if tier == "exam":
            assert len(fig["lines"]) == 2
            other = next(l for l in fig["lines"] if l is not chord_lines[0])
            (px, py) = other["through"][0]
            # the other line passes through the first curve point and a SECOND point off the curve
            first = (x1, _height(q, x1))
            assert first in [tuple(p) for p in other["through"]]
            second = next(tuple(p) for p in other["through"] if tuple(p) != first)
            if xlo <= second[0] <= xhi:
                assert abs(second[1] - _height_or_none(q, second[0])) > 1e-6, "tangent re-meets the curve"
            # numerical derivative of the sampled curve at x1 equals the other line's slope
            h = 0.1
            left = _height_float(q, round(x1 - h, 6))
            right = _height_float(q, round(x1 + h, 6))
            deriv = (right - left) / (2 * h)
            assert abs(deriv - float(_slope(other))) < 0.06, (deriv, _slope(other))
            assert _slope(other) != _oracle(q), "tangent slope must differ in value from the key"


def _height_float(q, x):
    for piece in q.figure["curves"][0]["pieces"]:
        for (xa, ya), (xb, yb) in zip(piece, piece[1:]):
            if xa <= x <= xb:
                return ya + (yb - ya) * (x - xa) / (xb - xa) if xb != xa else ya
    raise AssertionError(f"no curve at {x}")


def _height_or_none(q, x):
    try:
        return _height_float(q, x)
    except AssertionError:
        return 1e9


@pytest.mark.parametrize("tier", TIERS)
def test_no_answer_above_25_percent_of_200_draws(tier):
    counts = Counter(_make(tier, seed).answer_text for seed in range(200))
    value, n = counts.most_common(1)[0]
    assert n / 200 <= 0.25, f"{tier}: {value!r} is {n}/200"
    assert counts.get("0", 0) == 0
