"""Independent oracle for senior-interpret-distance-versus-time-graphs.

Imports nothing from ``mathgen.generators.pv1b_distance_time_build``. Questions
are generated through the skill's own sources. The expected speed is re-derived
from the figure spec (segments, axis labels) and the two interval times the
prompt states, by INTEGRATING segment slopes over [0, t] -- neither the
generator's interpolation route nor the registered reader's
find-the-covering-segment route.
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

SKILL_ID = "senior-interpret-distance-versus-time-graphs"
SEEDS = 1000
TIERS = ["easy", "medium", "hard", "exam"]


@functools.lru_cache(maxsize=None)
def _sources():
    _, skills = load_skills()
    skill = {s.id: s for s in skills}[SKILL_ID]
    return {s.difficulty: s for s in skill.sources}


def _make(tier, seed):
    src = _sources()[tier]
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _stated_times(text):
    nums = [int(x) for x in re.findall(r"\d+", text)]
    assert len(nums) == 2, f"prompt must state exactly the two interval times: {text!r}"
    return nums


def _height(segments, t):
    """Distance at time t: start height plus the integral of the slopes over [0, t]."""
    total = Fraction(segments[0]["from"][1])
    for s in segments:
        (t0, d0), (t1, d1) = s["from"], s["to"]
        overlap = max(0, min(t, t1) - t0)
        total += Fraction(d1 - d0, t1 - t0) * overlap
    return total


def _oracle(q):
    segments = q.figure["segments"]
    a, b = _stated_times(q.text)
    value = (_height(segments, b) - _height(segments, a)) / (b - a)
    if "minutes" in q.figure["x_label"]:
        assert "per hour" in q.text, "minute axis must be asked in a stated hourly unit"
        value *= 60
    return value


def _typed(q):
    given = q.answer_text
    return Fraction(given)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_key_matches_from_figure_and_prompt(tier):
    for seed in range(SEEDS):
        q = _make(tier, seed)
        expected = _oracle(q)
        assert _typed(q) == expected, (
            f"{tier} seed {seed}: oracle {expected} vs key {q.answer_text}\n{q.text}\n{q.figure['segments']}"
        )
        assert grade("fraction", q, str(q.answer_text)).correct


@pytest.mark.parametrize("tier", TIERS)
def test_boundary_assertions_from_the_figure(tier):
    for seed in range(SEEDS):
        q = _make(tier, seed)
        fig = q.figure
        segs = fig["segments"]
        assert segs[0]["from"][0] == 0, "graph starts at t = 0"
        for s, nxt in zip(segs, segs[1:]):
            assert s["to"] == nxt["from"], "polyline must be connected"
        for s in segs:
            for t, d in (s["from"], s["to"]):
                assert isinstance(t, int) and isinstance(d, int)
                assert t % fig["x_grid"] == 0 and d % fig["y_grid"] == 0, "vertex off the grid"
                assert 0 <= d and t >= 0
            assert s["to"][0] > s["from"][0], "time must advance"
        assert fig["equal_scale"] is False and fig["points"] == [] and fig["lines"] == []
        a, b = _stated_times(q.text)
        covering = [s for s in segs if s["from"][0] <= a and b <= s["to"][0]]
        assert len(covering) == 1, f"{tier} seed {seed}: interval inside {len(covering)} segments"
        seg = covering[0]
        if tier == "easy":
            assert (a, b) == (seg["from"][0], seg["to"][0])
            assert all(s["to"][1] > s["from"][1] for s in segs), "easy moves away from home throughout"
        else:
            assert seg["from"][0] < a < b < seg["to"][0], "interval ends must not be vertices"
        if tier == "hard":
            assert "positive velocity" in q.text and "away from home" in q.text.lower()
            kinds = {(s["to"][1] > s["from"][1]) - (s["to"][1] < s["from"][1]) for s in segs}
            assert kinds == {1, 0, -1}, "hard graph holds an away, a stationary and a return segment"
        if tier == "exam":
            assert "minutes" in fig["x_label"] and "per hour" in q.text
            assert all(s["to"][1] > s["from"][1] for s in segs)
        if tier != "hard":
            assert "positive velocity" not in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_no_answer_above_25_percent_of_200_draws(tier):
    counts = Counter(_make(tier, seed).answer_text for seed in range(200))
    value, n = counts.most_common(1)[0]
    assert n / 200 <= 0.25, f"{tier}: {value!r} is {n}/200"
    if tier in ("medium", "hard"):
        assert counts.get("0", 0) / 200 < 0.20, f"{tier}: zero answers are {counts.get('0', 0)}/200"


def test_hard_key_takes_all_three_signs():
    signs = Counter()
    for seed in range(300):
        v = Fraction(_make("hard", seed).answer_text)
        signs[(v > 0) - (v < 0)] += 1
    assert set(signs) == {-1, 0, 1}, signs
