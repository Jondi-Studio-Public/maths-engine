"""Exam chains for the solids, nets, views and composite-outline skills (the w7f work).

Oracles read the cubes, polygons and grid off the FIGURE and the numbers off the prompts, and count with arithmetic
of their own (neighbour tests for surface area, a shoelace sum for areas); none calls the generator's helpers.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import mathgen.generators.w7f_solid_exams  # noqa: F401  (registers the topics)
from tests import w7b_support as sup

STACK = (
    "senior-interpret-diagrams-of-three-dimensional-objects",
    "grid_cube_stack",
    "exd_x_w7f_cube_stack_chain",
    "number",
)
PROPS = (
    "senior-recognise-the-properties-of-common-two-dimensional-geometric",
    "grid_solid_properties",
    "exd_x_w7f_solid_properties_chain",
    "number",
)
NET = (
    "senior-interpret-different-forms-of-two-dimensional-representations-of",
    "grid_net_views",
    "exd_x_w7f_net_chain",
    "mcq",
)
ELEV = (
    "senior-sketch-elevation-views-of-different-models",
    "grid_elevation_views",
    "exd_x_w7f_elevation_chain",
    "grid_cells",
)
COMP = ("f10-composite-shape-representation", "grid_composite_area", "exd_x_w7f_composite_chain", "number")


def _wiring(spec):
    skill = sup.skill_of(spec[0])
    sup.check_wiring(skill, spec[1], spec[2], spec[3])
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and steps[-1] > steps[0]
    return skill


def _exposed_faces(cubes):
    s = {tuple(c) for c in cubes}
    return sum(
        (x + dx, y + dy, z + dz) not in s
        for x, y, z in s
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    )


def _shoelace(v):
    n = len(v)
    return abs(sum(v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1] for i in range(n))) // 2


def test_stack_chain():
    skill = _wiring(STACK)
    sup.check_exam_chain(skill, STACK[2], [1, 2, 2, 1], ["number"] * 4)
    for seed in range(80):
        q = sup.exam(STACK[2], seed)
        cubes = [tuple(c) for c in q.figure["cubes"]]
        assert len(set(cubes)) == len(cubes)
        for x, y, z in cubes:  # no floating cubes: every cube rests on the ground or on another cube
            assert z == 0 or (x, y, z - 1) in set(cubes)
        w, d, h = (
            int(v)
            for v in re.search(r"(\d+) cubes wide, (\d+) cubes deep and (\d+) cubes high", q.parts[1].prompt).groups()
        )
        side = int(re.search(r"side of (\d+) cm", q.parts[3].prompt).group(1))
        area = _exposed_faces(cubes)
        got = [int(p.answer_text) for p in q.parts]
        assert got == [len(cubes), w * d * h - len(cubes), area, area * side * side], seed
        assert w * d * h > len(cubes)


def test_stack_follow_through():
    def wrong(q, canon):
        side = int(re.search(r"side of (\d+) cm", q.parts[3].prompt).group(1))
        bad = int(canon["c"]) + 4
        return {"c": str(bad), "d": str(bad * side * side)}

    assert sup.check_follow_through(sup.skill_of(STACK[0]), STACK[2], "d", wrong) >= 30


def test_properties_chain():
    skill = _wiring(PROPS)
    sup.check_exam_chain(skill, PROPS[2], [1, 1, 2, 2], ["number"] * 4)
    for seed in range(80):
        q = sup.exam(PROPS[2], seed)
        cubes = [tuple(c) for c in q.figure["cubes"]]
        dims = [1 + max(c[i] for c in cubes) for i in range(3)]
        assert len(cubes) == dims[0] * dims[1] * dims[2]  # a solid cuboid
        k = int(re.search(r"exactly (\d+) cubes? long", q.parts[0].prompt).group(1))
        price = int(re.search(r"\$(\d+) for each", q.parts[3].prompt).group(1))
        area = _exposed_faces(cubes)
        want = [4 * dims.count(k), len(cubes), area, area * price]
        assert [int(p.answer_text) for p in q.parts] == want, seed
        assert area == 2 * (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2])


def test_net_chain():
    skill = _wiring(NET)
    sup.check_exam_chain(skill, NET[2], [1, 2, 2, 2], ["number"] * 4)
    for seed in range(80):
        q = sup.exam(NET[2], seed)
        fig = q.figure
        polys = fig["polygons"]
        assert len(polys) == 6
        areas = sorted(_shoelace(p["vertices"]) for p in polys)
        assert areas[0] == areas[1] and areas[2] == areas[3] and areas[4] == areas[5]
        surface = sum(areas)
        # recover the three edges from the pair areas: ab, ac, bc
        ab, ac, bc = areas[0], areas[2], areas[4]
        edges = [
            e
            for e in range(1, 20)
            if any(ab == e * f and ac == e * g and bc == f * g for f in range(1, 20) for g in range(1, 20))
        ]
        assert edges, seed
        a = max(edges)
        # the edges of the box are a, ab/a, ac/a
        b, c = ab // a, ac // a
        assert a * b == ab and a * c == ac and b * c == bc
        sheet = fig["grid"]["width"] * fig["grid"]["height"]
        # the sheet really is the bounding box of the drawn net
        xs = [v[0] for p in polys for v in p["vertices"]]
        ys = [v[1] for p in polys for v in p["vertices"]]
        assert (max(xs) - min(xs)) * (max(ys) - min(ys)) == sheet and min(xs) == 0 and min(ys) == 0
        want = [max(a, b, c), surface, a * b * c, sheet - surface]
        assert [int(p.answer_text) for p in q.parts] == want, seed
        assert sheet > surface


def test_elevation_chain():
    skill = _wiring(ELEV)
    sup.check_exam_chain(skill, ELEV[2], [2, 2, 1, 1], ["mcq", "mcq", "number", "number"])
    for seed in range(80):
        q = sup.exam(ELEV[2], seed)
        cubes = {tuple(c) for c in q.figure["cubes"]}
        # front view: column x, one square per layer filled by any cube in that column
        front = Counter()
        for x, y, z in {(x, z, 0) for x, y, z in cubes}:
            front[x] += 1
        # (a layer is filled if any cube at that x and height exists)
        layers = {(x, z) for x, y, z in cubes}
        front_heights = [sum(1 for (cx, z) in layers if cx == x) for x in sorted({c[0] for c in cubes})]
        side_layers = {(y, z) for x, y, z in cubes}
        side_heights = [sum(1 for (cy, z) in side_layers if cy == y) for y in sorted({c[1] for c in cubes})]
        text = "Column heights, left to right: "
        assert q.parts[0].answer_text == text + ", ".join(map(str, front_heights)), seed
        assert q.parts[1].answer_text == text + ", ".join(map(str, side_heights)), seed
        assert int(q.parts[2].answer_text) == len(layers)
        assert int(q.parts[3].answer_text) == len({(x, y) for x, y, z in cubes})
        # the four candidates for (a) and (b) are all different views, so the right one is not a guess between twins
        assert len({q.parts[0].answer_text, *q.parts[0].distractors}) == 4


def test_composite_chain():
    skill = _wiring(COMP)
    sup.check_exam_chain(skill, COMP[2], [1, 2, 2, 1], ["number"] * 4)
    shapes = set()
    for seed in range(80):
        q = sup.exam(COMP[2], seed)
        verts = q.figure["polygons"][0]["vertices"]
        n = len(verts)
        area = _shoelace(verts)
        perim = sum(
            abs(verts[(i + 1) % n][0] - verts[i][0]) + abs(verts[(i + 1) % n][1] - verts[i][1]) for i in range(n)
        )
        w, h = (
            int(g) for g in re.search(r"rectangle (\d+) units wide and (\d+) units high", q.parts[2].prompt).groups()
        )
        xs, ys = [v[0] for v in verts], [v[1] for v in verts]
        assert (w, h) == (max(xs) - min(xs), max(ys) - min(ys))
        price = int(re.search(r"\$(\d+) for each", q.parts[3].prompt).group(1))
        assert [int(p.answer_text) for p in q.parts] == [area, perim, w * h - area, area * price], seed
        shapes.add(n)
    assert shapes == {6, 8}


def test_composite_follow_through():
    def wrong(q, canon):
        w, h = (
            int(g) for g in re.search(r"rectangle (\d+) units wide and (\d+) units high", q.parts[2].prompt).groups()
        )
        bad = int(canon["a"]) + 2
        if w * h - bad in (bad, 0):  # writing the same number again is not following through
            return None
        return {"a": str(bad), "c": str(w * h - bad)}

    assert sup.check_follow_through(sup.skill_of(COMP[0]), COMP[2], "c", wrong) >= 30


def test_variety():
    for spec in (STACK, PROPS, NET, ELEV, COMP):
        counts = Counter(sup.exam(spec[2], s).answer_text for s in range(60))
        assert len(counts) >= 30, (spec[2], len(counts))
        assert len({sup.exam(spec[2], s).text for s in range(60)}) >= 1
        assert Fraction(max(counts.values()), 60) <= Fraction(1, 10)
