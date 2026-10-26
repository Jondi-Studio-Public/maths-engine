"""Cross-sections of cube solids on isometric dot paper: ``grid_cross_sections_build``.

Every tier is made through the skill's own sources. The cut face is re-derived here from the cubes in the figure by
routes the generator does not use: edge lengths from the cubes' extents, and the prism property (the layers of a
prism cut parallel to its end all match) rather than a slice filter.
"""

from __future__ import annotations

import functools
import re

import pytest

from mathgen import figures, make
from mathgen.grading import grade
from mathgen.grading.grid import normalised
from mathgen.skills import load_skills

SKILL = "f10-draw-cross-sections-of-solids"
TOPIC = "grid_solid_cross_sections"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(80)
ORD = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}


@functools.lru_cache(maxsize=1)
def _skill():
    return next(s for s in load_skills()[1] if s.id == SKILL)


@functools.lru_cache(maxsize=None)
def _q(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return make(src.topic, seed=seed, difficulty=tier, **(src.params or {}))


def _cells(q):
    return {tuple(c) for c in q.answer["cells"]}


def _extent(cells):
    return (
        max(x for x, _ in cells) - min(x for x, _ in cells) + 1,
        max(y for _, y in cells) - min(y for _, y in cells) + 1,
    )


def _layer(cubes, axis, k):
    """The two other coordinates of the cubes whose ``axis`` coordinate is k, in (across, up) order."""
    order = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]
    return {(c[order[0]], c[order[1]]) for c in cubes if c[axis] == k}


def test_the_skill_is_active_behind_the_figure_gate_with_three_tiers():
    s = _skill()
    assert s.status == "active" and s.answer == "grid_cells" and s.needs_figure is True
    assert s.capability_gates == ["figure_stimulus"] and "ac9m6sp01" in s.tags and "acara-f10-v9" in s.tags
    assert [x.difficulty for x in s.sources] == TIERS
    assert len(s.misconceptions) >= 5 and s.prerequisites


def test_an_unknown_tier_is_refused():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="impossible")


@pytest.mark.parametrize("tier", TIERS)
def test_figure_reader_agrees_with_the_key_and_the_figure_leaks_nothing(tier):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in SEEDS:
        q = _q(tier, seed)
        assert figures.validate(q.figure) == []
        figures.assert_no_leak(q.figure, [q.answer_text])
        assert not any(ch.isdigit() for ch in q.figure["alt"])
        assert q.figure["canvas"] == {"width": 6, "height": 6} and q.answer["grid"] == [6, 6]
        assert grade("grid_cells", q, reader(q)).correct, (tier, seed, q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_right_and_every_distractor_is_wrong_and_on_the_canvas(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade("grid_cells", q, q.answer_text).correct
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        shapes = {normalised(_cells(q))}
        for d in q.distractors:
            assert not grade("grid_cells", q, d).correct, (tier, seed, d)
            cells = {tuple(map(int, m)) for m in re.findall(r"\((\d+), (\d+)\)", d)}
            assert cells and all(0 <= x < 6 and 0 <= y < 6 for x, y in cells)
            shapes.add(normalised(cells))
        assert len(shapes) == 4, "the wrong drawings are four different shapes, not one shape in different places"


def test_easy_is_the_face_of_a_box_and_the_same_in_every_layer():
    faces = set()
    for seed in SEEDS:
        q = _q("easy", seed)
        cubes = q.figure["cubes"]
        extent = [max(c[i] for c in cubes) + 1 for i in range(3)]
        face, ordinal = re.search(r"parallel to its (\w+) face.*?through the (\w+) layer", q.text).groups()
        axis = {"front": 1, "right": 0, "top": 2}[face]
        faces.add(face)
        across, up = [extent[i] for i in {1: (0, 2), 0: (1, 2), 2: (0, 1)}[axis]]
        assert across != up
        assert _extent(_cells(q)) == (across, up) and len(_cells(q)) == across * up
        assert len(cubes) == extent[0] * extent[1] * extent[2]
        assert ORD[ordinal] < extent[axis]
        assert all(_layer(cubes, axis, k) == _layer(cubes, axis, 0) for k in range(extent[axis]))
    assert faces == {"front", "right", "top"}


def test_medium_is_an_l_and_every_parallel_layer_is_the_same_l():
    kinds = set()
    for seed in SEEDS:
        q = _q("medium", seed)
        cubes = q.figure["cubes"]
        face = re.search(r"parallel to its (\w+) face", q.text).group(1)
        axis = {"front": 1, "top": 2}[face]
        kinds.add(face)
        n = max(c[axis] for c in cubes) + 1
        layers = [_layer(cubes, axis, k) for k in range(n)]
        assert all(layer == layers[0] for layer in layers), "a prism's parallel cuts match"
        shape = _cells(q)
        w, h = _extent(shape)
        assert normalised(shape) == normalised(layers[0])
        corner_missing = w * h - len(shape)
        assert corner_missing >= 1, "an L, not a rectangle"
        assert len(shape) == len(layers[0])
    assert kinds == {"front", "top"}


def test_hard_depends_on_where_the_cut_goes_and_compares_two_cuts():
    kinds = set()
    for seed in SEEDS:
        q = _q("hard", seed)
        cubes = q.figure["cubes"]
        heights = [max(c[2] for c in cubes if c[0] == x) + 1 for x in range(max(c[0] for c in cubes) + 1)]
        depth = max(c[1] for c in cubes) + 1
        assert heights == sorted(heights, reverse=True) and len(set(heights)) >= 3
        shape = _cells(q)
        if "area" in q.text:
            kinds.add("compare")
            o1, o2 = re.search(r"through the (\w+) layer and the (\w+) layer", q.text).groups()
            areas = sorted([depth * heights[ORD[o1]], depth * heights[ORD[o2]]])
            assert areas[0] != areas[1]
            want = areas[1] if "greater area" in q.text else areas[0]
            assert len(shape) == want
        elif "parallel to its right face" in q.text:
            kinds.add("column")
            k = ORD[re.search(r"through the (\w+) layer", q.text).group(1)]
            assert _extent(shape) == (depth, heights[k]) and heights[k] < max(heights)
        else:
            kinds.add("layer")
            z = ORD[re.search(r"through the (\w+) layer", q.text).group(1)]
            columns = sum(1 for v in heights if v >= z + 1)
            assert 0 < columns < len(heights)
            assert _extent(shape) == (columns, depth)
    assert kinds == {"compare", "column", "layer"}


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    seen = set()
    shapes = set()
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.to_json() == make(TOPIC, seed=seed, difficulty=tier).to_json()
        seen.add((q.text, q.answer_text, repr(q.figure)))
        shapes.add(
            re.sub(r"\b(first|second|third|fourth|fifth|front|right|top|left|bottom|greater|smaller)\b", "_", q.text)[
                :60
            ]
        )
    assert len(seen) >= 40
