"""Independent oracle for the step-graph skill (ACMGM047).

Imports nothing from the generator module. Every question is generated through
the skill's own sources, and the key is re-derived from the figure's segments and
the numbers the prompt states, by a route that differs from the generator's (and
from its registered reader): the segments are sorted and the step that a filled
circle covers is found by ordering, not by testing each segment's ends. The
boundary the skill is built around is asserted from the figure itself.
"""

from __future__ import annotations

import functools
import re

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "senior-interpret-piece-wise-linear-and-step-graphs-used"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _source(tier):
    return next(s for s in _skill().sources if s.difficulty == tier)


def _question(tier, seed):
    src = _source(tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _steps(fig):
    """Sorted, contiguous, horizontal integer steps; returns (steps, closed_right)."""
    segs = sorted(fig["segments"], key=lambda s: s["from"][0])
    assert segs and fig["kind"] == "cartesian" and fig["equal_scale"] is False
    closed_right = segs[0]["open_start"] is True
    assert segs[0]["from"][0] == 0
    prev_h, prev_end = 0, 0
    steps = []
    for s in segs:
        (a, h), (b, h2) = s["from"], s["to"]
        assert h == h2 and all(isinstance(v, int) for v in (a, b, h))
        assert a == prev_end and b > a and h > prev_h  # contiguous, strictly rising charge
        assert (s["open_start"], s["open_end"]) == (closed_right, not closed_right)
        steps.append((a, b, h))
        prev_h, prev_end = h, b
    return steps, closed_right


def _value_at(steps, closed_right, x):
    """The charge at x by ordering: (a, b] -> first step whose right end reaches x;
    [a, b) -> last step whose left end is at or below x."""
    if closed_right:
        assert 0 < x <= steps[-1][1]
        return next(h for (a, b, h) in steps if x <= b)
    assert 0 <= x < steps[-1][1]
    return [h for (a, b, h) in steps if a <= x][-1]


def _derive(q):
    """(expected answer text, inputs) from the figure and the prompt alone."""
    steps, closed_right = _steps(q.figure)
    charge = re.search(r"\$(\d+)", q.text)
    if charge:
        c = int(charge.group(1))
        at_c = [(a, b, h) for (a, b, h) in steps if h == c]
        assert len(at_c) == 1 and closed_right
        return str(at_c[0][1]), [c]
    xs = [int(n) for n in re.findall(r"\d+", q.text)]
    return str(sum(_value_at(steps, closed_right, x) for x in xs)), xs


@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier):
    for seed in range(SEEDS):
        q = _question(tier, seed)
        expected, _ = _derive(q)
        assert q.answer_text == expected, (tier, seed, q.text)
        assert str(q.answer) == expected
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    sides = set()
    for seed in range(SEEDS):
        q = _question(tier, seed)
        steps, closed_right = _steps(q.figure)
        sides.add(closed_right)
        _, inputs = _derive(q)
        ends = {b for (_, b, _) in steps[:-1]}
        if tier == "easy":
            (x,) = inputs
            assert any(a < x < b for (a, b, _) in steps)  # strictly inside a step
            assert x not in ends and x not in {0, steps[-1][1]}
        elif tier == "medium":
            (x,) = inputs
            assert x in ends  # exactly on a shared step end
            # the key is the closed dot's step: lower step for (a, b], upper for [a, b)
            idx = next(i for i, (_, b, _) in enumerate(steps) if b == x)
            want = steps[idx][2] if closed_right else steps[idx + 1][2]
            assert q.answer == want
        elif tier == "hard":
            assert len(inputs) == 2 and inputs[0] != inputs[1]
            for x in inputs:
                assert x in ends or any(a < x < b for (a, b, _) in steps)
        else:  # exam
            (c,) = inputs
            assert closed_right is True  # every step is (a, b]
            seg = next(s for s in q.figure["segments"] if s["from"][1] == c)
            assert seg["open_end"] is False and seg["open_start"] is True
            assert q.answer == seg["to"][0]  # the closed right end, not the open left end
            assert q.answer != seg["from"][0]
    if tier == "exam":
        assert sides == {True}
    else:
        assert sides == {True, False}  # the closed side is not a learnable tell
