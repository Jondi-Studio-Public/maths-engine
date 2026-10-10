"""Independent oracle for the end-behaviour skill (ACMMM014, `pv1e_power_end_behaviour`).

Imports nothing from the generator module. Every question is generated through the skill's own
sources. The key is re-derived from the figure and the end the prompt names by a route that is
neither the generator's construction nor its registered reader (a log-log fit): each candidate n
is tried against every sampled point, y = x^n to the figure's rounding, and exactly one n must
fit. The class then follows from the mathematics of that n at the named end.
"""

from __future__ import annotations

import functools

import pytest

import mathgen
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills

SKILL_ID = "senior-recognise-features-of-the-graphs-of-y-x-acmmm014"
TIERS = ["easy", "medium", "hard", "exam"]
CHUNK = 250
CANDIDATES = [1, 2, 3, 4, 5, -1, 0.5]
INC, DEC, APP = "increases without bound", "decreases without bound", "approaches a fixed value"


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _question(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _fits(pts, n):
    for x, y in pts:
        if x == 0:
            if n < 0 or abs(y) > 1e-9:
                return False
            continue
        try:
            want = x**n if n != 0.5 else (x**0.5 if x > 0 else None)
        except ZeroDivisionError:
            return False
        if want is None or isinstance(want, complex) or abs(want - y) > 2e-6 * max(1.0, abs(want)):
            return False
    return True


def _derive(q):
    pts = [p for c in q.figure["curves"] for pc in c["pieces"] for p in pc]
    fitting = [n for n in CANDIDATES if _fits(pts, n)]
    assert len(fitting) == 1, (q.text, fitting)
    n = fitting[0]
    left = "negative infinity" in q.text
    if n < 0:
        return n, APP
    if not left:
        return n, INC
    assert n != 0.5
    return n, (DEC if n % 2 == 1 else INC)


@pytest.mark.parametrize("chunk", range(4))
@pytest.mark.parametrize("tier", TIERS)
def test_key_equals_the_figure_derivation(tier, chunk):
    for seed in range(chunk * CHUNK, (chunk + 1) * CHUNK):
        q = _question(tier, seed)
        n, want = _derive(q)
        assert q.answer_text == want, (tier, seed, n, q.text)
        assert sorted([q.answer_text, *q.distractors]) == sorted([INC, DEC, APP])
        options = mcq_options(q)
        assert grade_mcq(q, str(options.index(want))).correct
        for i, opt in enumerate(options):
            assert grade_mcq(q, str(i)).correct is (opt == want)


@pytest.mark.parametrize("tier", TIERS)
def test_the_boundary_each_tier_is_built_on(tier):
    for seed in range(1000):
        q = _question(tier, seed)
        n, _ = _derive(q)
        f = q.figure
        (x0, x1), (y0, y1) = f["x_range"], f["y_range"]
        assert x1 - x0 == 12 and y1 - y0 == 12 and f["equal_scale"] is True
        left = "negative infinity" in q.text
        assert y0 <= 0 <= y1  # the x-axis is always in the window
        if tier == "easy":
            assert n in (1, 2, 3, -1) and x0 < 0 < x1
        elif tier == "medium":
            assert x0 < 0 < x1
        elif tier == "hard":
            assert (x1 == 0) if left else (x0 == 0)
        else:
            assert n in (1, 2, 3, -1, 0.5)
            assert (x1 <= -1) if left else (x0 >= 1)  # no y-axis anywhere in the picture
            assert "No y-axis" in q.text or "not in the window" in q.text or "away from the y-axis" in q.text
        if n == 0.5:
            assert not left
        assert "x^" not in q.text.replace("y = x^n", "")  # n is never printed


def test_each_tier_draws_every_class_and_the_wider_tiers_add_exponents():
    seen = {}
    for tier in TIERS:
        ns, classes = set(), set()
        for seed in range(400):
            q = _question(tier, seed)
            ns.add(_derive(q)[0])
            classes.add(q.answer_text)
        assert classes == {INC, DEC, APP}, tier
        seen[tier] = ns
    assert seen["easy"] == {1, 2, 3, -1}
    assert {4, 5, 0.5} <= seen["medium"] and {4, 5, 0.5} <= seen["hard"]
    assert seen["exam"] == {1, 2, 3, -1, 0.5}
