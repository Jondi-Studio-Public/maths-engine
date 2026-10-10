"""Independent oracle for `pv1d_abs_vertex`, re-deriving the vertex from the drawn figure.

Independent oracle for V-shaped absolute value graph skill.

Imports NOTHING from ``mathgen.generators.pv1d_abs_vertex_build``. It generates through
the skill's own sources and re-derives the key from the FIGURE (and, where the prompt
selects a graph, from the letter or the axis property the prompt states) by a route that
is not the reader's. The reader takes the lowest sample; this oracle extends the two END
samples of each drawn V back along their arms (y = -x + c1 on the left, y = x + c2 on the
right) and intersects the lines, then cross-checks that every consecutive pair of samples
has slope exactly +-1 and that the corner is itself a sample.

Set ``PV1D_ABS_ORACLE_CORRUPT=1`` to add one to the derived h: the RED proof that this file fails when its
expectation is wrong.
"""

import functools
import os
import re

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL = "senior-use-and-apply-the-notation-left-x-right"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS_1000 = range(1000)
_CORRUPT = int(os.environ.get("PV1D_ABS_ORACLE_CORRUPT", "0"))

_LETTER_RE = re.compile(r"through the point marked ([A-Z])")
_CROSSES_TWICE_RE = re.compile(r"twice|two points|two different points")
_STAYS_CLEAR_RE = re.compile(r"never meets|stays above|does not touch")


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make(difficulty, seed):
    skill = _skills_dict()[SKILL]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _samples(curve):
    pts = [tuple(p) for piece in curve["pieces"] for p in piece]
    return pts


def _vertex_from_arms(curve):
    """(h, k) from the two end samples; asserts the V is made of slope +-1 segments."""
    assert len(curve["pieces"]) == 1
    pts = _samples(curve)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        assert x1 > x0
        assert abs((y1 - y0) / (x1 - x0)) == 1, (pts[0], pts[-1], (x0, y0), (x1, y1))
    (xl, yl), (xr, yr) = pts[0], pts[-1]
    c1, c2 = xl + yl, yr - xr
    assert (c1 - c2) % 2 == 0 and (c1 + c2) % 2 == 0
    h, k = (c1 - c2) // 2, (c1 + c2) // 2
    assert xl < h < xr, "both arms must be drawn on either side of the corner"
    assert (h, k) in pts, "the corner must be a sample point"
    return h, k


def _lies_on(point, vertex):
    return point[1] == abs(point[0] - vertex[0]) + vertex[1]


def _expected(q):
    """The key re-derived from the figure and the prompt's selector."""
    fig = q.figure
    verts = [_vertex_from_arms(c) for c in fig["curves"]]
    assert fig["x_range"] == [-10, 10] and fig["y_range"] == [-10, 10]
    assert fig["grid"] == 1 and fig["equal_scale"] is True
    if len(verts) == 1:
        assert all(abs(v) <= 5 and v != 0 for v in verts[0])
        chosen = verts[0]
        if fig["points"]:  # easy: P marks the corner
            (pt,) = fig["points"]
            assert tuple(pt["at"]) == chosen and pt["label"] == "P"
    else:
        assert len(verts) == 2
        letters = _LETTER_RE.findall(q.text)
        if letters:  # hard: the graph through the named point
            assert len(letters) == 1
            points = {p["label"]: tuple(p["at"]) for p in fig["points"]}
            assert sorted(points) == ["P", "Q"]
            holders = []
            for letter, pt in points.items():
                on = [v for v in verts if _lies_on(pt, v)]
                assert len(on) == 1, (letter, pt, verts)  # each point on exactly one graph
                assert pt != on[0], "a lettered point sits on an arm, never at a vertex"
                assert not _lies_on(pt, [v for v in verts if v != on[0]][0])
                holders.append(on[0])
            assert holders[0] != holders[1]
            chosen = [v for v in verts if _lies_on(points[letters[0]], v)][0]
        else:  # exam: the graph chosen by the x-axis
            assert not fig["points"]
            clear, twice = _STAYS_CLEAR_RE.search(q.text), _CROSSES_TWICE_RE.search(q.text)
            assert bool(clear) != bool(twice), q.text
            ks = sorted(v[1] for v in verts)
            assert ks[0] < 0 < ks[1], "exactly one graph has k > 0 and neither touches the axis"
            chosen = [v for v in verts if (v[1] > 0) == bool(clear)]
            assert len(chosen) == 1
            chosen = chosen[0]
        # Whatever the selector, the two graphs are told apart by their arms.
        (h1, k1), (h2, k2) = verts
        assert k1 - h1 != k2 - h2 and k1 + h1 != k2 + h2
    return f"({chosen[0] + _CORRUPT}, {chosen[1]})"


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_figure_derivation_equals_the_key_on_1000_seeds(difficulty):
    for seed in SEEDS_1000:
        q = _make(difficulty, seed)
        assert _expected(q) == q.answer_text, (difficulty, seed, q.text)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_are_never_the_key_and_the_other_vertex_is_among_them(difficulty):
    for seed in range(300):
        q = _make(difficulty, seed)
        assert q.answer_text not in q.distractors
        if difficulty in ("hard", "exam"):
            verts = {f"({h}, {k})" for h, k in (_vertex_from_arms(c) for c in q.figure["curves"])}
            others = verts - {q.answer_text}
            assert len(others) == 1 and others <= set(q.distractors), (difficulty, seed)


def test_the_oracle_detects_a_figure_that_disagrees_with_the_key():
    """RED inside the suite: shift the answer graph one square right and the oracle disagrees."""
    for difficulty in DIFFICULTIES:
        q = _make(difficulty, 7)
        fig = q.figure
        target = _expected(q)
        for curve in fig["curves"]:
            if f"({_vertex_from_arms(curve)[0] - _CORRUPT}, {_vertex_from_arms(curve)[1]})" == target:
                curve["pieces"] = [[[x + 1, y] for x, y in piece] for piece in curve["pieces"]]
        try:
            disagrees = _expected(q) != q.answer_text
        except AssertionError:  # a boundary rule (point on a graph, corner marked) tripped
            disagrees = True
        assert disagrees, difficulty


def test_the_oracle_detects_the_wrong_graph_being_named():
    for seed in range(50):
        q = _make("hard", seed)
        letter = _LETTER_RE.findall(q.text)[0]
        other = "Q" if letter == "P" else "P"
        q.text = q.text.replace(f"marked {letter}", f"marked {other}")
        assert _expected(q) != q.answer_text


def test_the_oracle_detects_the_wrong_axis_property():
    for seed in range(50):
        q = _make("exam", seed)
        if _CROSSES_TWICE_RE.search(q.text):
            q.text = "One of them never meets the x-axis."
        else:
            q.text = "One of them cuts the x-axis twice."
        assert _expected(q) != q.answer_text
