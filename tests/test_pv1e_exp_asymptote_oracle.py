"""Independent oracle for the exponential-asymptote skill (ACMMM065).

Imports nothing from the generator module. Every question is generated through the skill's own
sources. The key is re-derived from the figure alone by a route that is neither the generator's
construction nor its registered reader (which rounds the level end of the curve): three samples
half a unit apart on the curve y = a^x + k give the common ratio of their rises, which is
a^(1/2), and then k = y1 - (y2 - y1) / (ratio - 1). The recovered base must be one of the
bases the family allows.
"""

from __future__ import annotations

import functools

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-recognise-the-qualitative-features-of-the-graph-of"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]
BASES = (2.0, 3.0, 0.5, 1 / 3)


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _question(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _derive(q):
    (curve,) = q.figure["curves"]
    (piece,) = curve["pieces"]
    best = None
    for i in range(len(piece) - 16):
        (x1, y1), (x2, y2), (x3, y3) = piece[i], piece[i + 8], piece[i + 16]
        assert abs((x2 - x1) - 0.5) < 1e-6 and abs((x3 - x2) - 0.5) < 1e-6  # samples are 1/16 apart
        gap = min(abs(y2 - y1), abs(y3 - y2))
        if best is None or gap > best[0]:
            best = (gap, (x1, y1), (x2, y2), (x3, y3))
    assert best is not None and best[0] > 0.1, "no well-conditioned triple of samples"
    _, (x1, y1), (x2, y2), (x3, y3) = best
    ratio = (y3 - y2) / (y2 - y1)  # a^(1/2)
    base = ratio**2
    assert any(abs(base - b) < 0.02 * b for b in BASES), (q.text, base)
    k = y1 - (y2 - y1) / (ratio - 1)
    assert abs(k - round(k)) < 0.02, (q.text, k)
    return int(round(k)), min(BASES, key=lambda b: abs(b - base))


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        k, _ = _derive(q)
        assert q.answer_text == f"y = {k}", (tier, seed, q.text)
        assert q.answer == {"y": k}
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    decay = zero = 0
    for seed in range(SEEDS):
        q = _question(tier, seed)
        k, base = _derive(q)
        f = q.figure
        (y0, y1) = f["y_range"]
        assert y0 < k < y1  # the asymptote is inside the window
        decay += base < 1
        zero += k == 0
        if tier == "easy":
            assert base > 1 and k != 0 and (y0, y1) == (-6, 6)
        elif tier == "medium":
            assert (y0, y1) == (-6, 6) and f["y_grid"] == 1
        elif tier == "hard":
            assert f["equal_scale"] is False and f["y_grid"] == 2 and k % 2 == 0 and k != 0
            assert "2 units" in q.text or "2s" in q.text
        else:
            assert not (y0 <= 0 <= y1)  # no x-axis in the picture
            assert "x-axis" in q.text
    if tier == "medium":
        assert decay > 0.25 * SEEDS and zero > 0  # both ends occur; the x-axis is sometimes the asymptote
    elif tier == "easy":
        assert decay == 0 and zero == 0
