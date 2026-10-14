"""Exam source for volumes of revolution of the region between a parabola and a line.

The oracle parses the two curves from the prompt and uses the other method (shells for the y-axis, washers by midpoint test for the x-axis).
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import Question
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_c4_solids_of_revolution"
HARD_TOPIC = "determine_volumes_of_solids_of_revolution_about_either"
GRADER = GRADERS["expression"]
_HARD_KW = {}


def _make(seed: int) -> Question:
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed: int) -> Question:
    return make(HARD_TOPIC, seed=seed, difficulty="hard", **_HARD_KW)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_make(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert shapes - hard_shapes, (shapes, hard_shapes)


def test_determinism():
    for seed in (0, 1, 7, 99, 12345):
        assert _make(seed).to_json() == _make(seed).to_json()


from sympy import Symbol, integrate, pi, simplify, solve, sympify

_X = Symbol("x", positive=True)


def _agrees(q: Question) -> bool:
    t = q.text
    c = int(re.search(r"y = (\d*)x\^2", t).group(1) or 1)
    slope = int(re.search(r"y = (\d+)x(?!\^)", t).group(1))
    axis = re.search(r"about the ([xy])-axis", t).group(1)
    upper = max(r for r in solve(c * _X**2 - slope * _X, _X) if r != 0)
    parabola, line = c * _X**2, slope * _X
    mid = upper / 2
    top, bottom = (line, parabola) if line.subs(_X, mid) > parabola.subs(_X, mid) else (parabola, line)
    if axis == "x":
        v = pi * integrate(top**2 - bottom**2, (_X, 0, upper))
    else:
        v = 2 * pi * integrate(_X * (top - bottom), (_X, 0, upper))
    return simplify(v - sympify(q.answer_text)) == 0


def test_both_axes_occur():
    axes = {("x-axis" in q.text) for q in (_make(s) for s in range(60))}
    assert axes == {True, False}


def test_independent_oracle_matches_key_over_400_seeds():
    bad = []
    for seed in range(400):
        q = _make(seed)
        if not _agrees(q):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:5]


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for seed in range(150):
        q = _make(seed)
        assert len(q.distractors) >= 1, (seed, q.text)
        if not GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]
