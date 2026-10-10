"""Independent oracle for the dilation skill (ACMMM026, y = c f(x)).

Imports nothing from the generator module. Every question is generated through the skill's own
sources. The key is re-derived two ways that are not the registered reader: (1) Qy / Py from
the two labelled points, with the stated x parsed from the PROMPT TEXT; (2) a least-squares
fit of the Q curve against the P curve over every common sample (sum yq*yp / sum yp^2).
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-examine-dilations-and-the-graphs-of-y-cf"
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


def _pts(fig):
    by = {p["label"]: p["at"] for p in fig["points"]}
    assert set(by) == {"P", "Q"} and len(fig["points"]) == 2
    return by["P"], by["Q"]


def _curve(c):
    return {x: y for piece in c["pieces"] for x, y in piece}


def _derive(q):
    fig = q.figure
    (px, py), (qx, qy) = _pts(fig)
    stated = int(re.search(r"x = (-?\d+)", q.text).group(1))
    assert px == qx == stated, (q.text, px, qx)
    # grid intersections inside the fixed window
    for x, y in ((px, py), (qx, qy)):
        assert isinstance(x, int) and isinstance(y, int)
        assert fig["x_range"] == [-6, 6] and fig["y_range"] == [-8, 8]
        assert -6 <= x <= 6 and -8 <= y <= 8
    assert len(fig["curves"]) == 2
    cs = [_curve(c) for c in fig["curves"]]
    owner_p = [i for i, c in enumerate(cs) if abs(c.get(px, 1e9) - py) < 1e-6]
    owner_q = [i for i, c in enumerate(cs) if abs(c.get(qx, 1e9) - qy) < 1e-6]
    assert len(owner_p) == 1 and len(owner_q) == 1 and owner_p != owner_q, q.text
    cp, cq = cs[owner_p[0]], cs[owner_q[0]]
    common = [x for x in cp if x in cq]
    num = sum(cq[x] * cp[x] for x in common)
    den = sum(cp[x] ** 2 for x in common)
    assert den > 0 and len(common) >= 16
    c_fit = num / den
    c_pts = Fraction(qy, py)
    assert abs(c_fit - float(c_pts)) < 1e-4, (q.text, c_fit, c_pts)
    return c_pts, py, qy


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        c, _, _ = _derive(q)
        expected = c + 1 if CORRUPT else c
        assert Fraction(q.answer_text) == expected, (tier, seed, q.text)
        assert q.answer_text not in q.distractors
        assert c not in (0, 1)


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        c, py, qy = _derive(q)
        assert "y = f(x)" in q.text and "y = cf(x)" in q.text
        if tier == "easy":
            assert c.denominator == 1 and 2 <= c <= 6 and py > 0 and qy > 0
        elif tier == "medium":
            assert 0 < c and c != 1 and py > 0 and qy > 0
            assert c.denominator != 1
        elif tier == "hard":
            assert c < 0 and py > 0 > qy
        else:
            assert py < 0 and c != 1 and c != 0
            assert (qy < 0) == (c > 0)  # Q's side of the axis is decided by the sign of c
