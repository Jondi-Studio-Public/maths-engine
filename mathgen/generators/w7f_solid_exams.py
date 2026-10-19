"""Exam chains for the solids, nets and views skills (the ``w7f`` figure work).

Five chains, each read off one drawing on a square grid or on isometric dot paper (``grid_drawing``). Every part is a
``number`` unless it says ``mcq``.

* ``exd_x_w7f_cube_stack_chain`` (``senior-interpret-diagrams-of-three-dimensional-objects``): a stack of unit cubes
  on isometric dot paper with every top in view. (a) the cubes in the stack, (b) the cubes still needed to fill a box
  of stated size, (c) the surface area in unit squares (hidden faces and the bottom included), (d) the surface area in
  square centimetres when each cube has a stated side (follows (c)).
* ``exd_x_w7f_solid_properties_chain`` (``senior-recognise-the-properties-of-common-two-dimensional-geometric``): a
  cuboid of unit cubes. (a) how many edges are of a stated length, (b) the volume, (c) the surface area, (d) the cost of
  painting the whole outside at a stated price per unit square (follows (c)).
* ``exd_x_w7f_net_chain`` (``senior-interpret-different-forms-of-two-dimensional-representations-of``): the net of a
  closed box on a square grid, with the sheet that just fits round it. (a) the longest edge of the box, (b) its
  surface area, (c) its volume, (d) the area of the sheet left over (follows (b)).
* ``exd_x_w7f_elevation_chain`` (``senior-sketch-elevation-views-of-different-models``): a stack of unit cubes.
  (a) the column heights of the front view (``mcq``), (b) of the right-side view (``mcq``), (c) the squares in the
  front view, (d) the squares in the view from above. These are the read forms of the elevation skill, which a
  learner answers by drawing; the draw form is not an exam part.
* ``exd_x_w7f_composite_chain`` (``f10-composite-shape-representation``): an L-, T-, U- or step-shaped outline.
  (a) its area, (b) its perimeter, (c) the squares of the rectangle round it that the shape leaves bare (follows
  (a)), (d) the cost of tiling it at a stated price per square (follows (a)).

Each answer is built from the drawn solid or outline first and the prompts derived from it; the reader registered
for each topic re-derives every part from the figure and the prompts alone.
"""

from __future__ import annotations

import math
import re

from mathgen import figures
from mathgen.core import check, generator
from mathgen.figure_kinds import grid

from ._w7f_kit import finish_parts, mcq_part, number_part, pick, retry, split_parts
from .grid_solids_build import (
    _column_heights,
    _composite_easy,
    _cuboid_net,
    _heights_text,
    _polygon_figure,
    _stack_figure,
    _stack_heights,
    _surface_area,
    _twice_area,
    _visible_squares,
)


def _only_exam(tier: str) -> None:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")


# ------------------------------------------------------------------------------------------- 1. cube stack

_BOX_RE = re.compile(r"(\d+) cubes wide, (\d+) cubes deep and (\d+) cubes high")
_SIDE_RE = re.compile(r"[Ee]ach cube has a side of (\d+) cm")


def _stack_chain(rng):
    width, depth = rng.randint(2, 4), rng.randint(2, 3)
    top = rng.choice((3, 3, 4))
    h, cubes = _stack_heights(rng, width, depth, top, 6, 20)
    fig = _stack_figure(width, depth, top, cubes)
    n, seen, area = len(cubes), _visible_squares(cubes), _surface_area(cubes)
    columns = len(grid.plan_view(cubes))
    check(area == 2 * seen, "the surface area is not twice the squares in view")
    bw, bd, bh = width + rng.randint(0, 1), depth + rng.randint(0, 1), top + rng.randint(0, 1)
    full = bw * bd * bh
    check(full > n, "the box must hold the stack")
    side = rng.choice((2, 3))
    stem = pick(
        rng,
        (
            "A stack of unit cubes is drawn on isometric dot paper. There are no gaps in the stack: every cube rests "
            "on the ground or on another cube.",
            "The picture is an isometric drawing of a stack made of unit cubes. Nothing is hidden underneath: each "
            "cube sits on the ground or on another cube.",
            "Unit cubes have been glued into a solid stack and drawn on isometric dot paper. Every cube is held up "
            "by the ground or by a cube below it.",
        ),
    )
    part_a = number_part(
        "How many cubes make up the stack?",
        n,
        1,
        [f"Count the cubes in each column: {' + '.join(str(v) for v in h.values() if v)} = {n}."],
        [
            (columns, "stk-counts-the-columns"),
            (seen, "stk-counts-only-the-squares-in-view"),
            (width * depth * top, "stk-fills-the-bounding-box"),
            (n + 1, "stk-count-off-by-one"),
            (n - 1, "stk-count-off-by-one"),
        ],
    )
    part_b = number_part(
        f"The stack is built inside a box that is {bw} cubes wide, {bd} cubes deep and {bh} cubes high. How many "
        f"more cubes are needed to fill the box completely?",
        full - n,
        2,
        [f"The box holds {bw} x {bd} x {bh} = {full} cubes.", f"{full} - {n} = {full - n} more cubes."],
        [
            (n, "stk-gives-the-cubes-in-the-stack"),
            (full, "stk-gives-the-capacity-of-the-box"),
            (full + n, "scd-adds-instead-of-subtracts"),
            (width * depth * top - n, "stk-fills-the-bounding-box"),
            (full - 2 * n, "stk-takes-the-stack-off-twice"),
        ],
    )
    part_c = number_part(
        "The surface area of the stack is the number of unit squares on its outside, including the bottom and the "
        "faces you cannot see. Find the surface area, in unit squares.",
        area,
        2,
        [
            f"The picture shows {seen} squares (tops, fronts and right sides).",
            f"The back, the left side and the bottom match them, so the surface area is 2 x {seen} = {area}.",
        ],
        [
            (seen, "stk-counts-only-the-squares-in-view"),
            (6 * n, "stk-ignores-faces-that-touch"),
            (seen + columns, "stk-counts-the-bottom-once"),
            (2 * seen - columns, "stk-counts-the-bottom-once"),
            (area - 2, "stk-count-off-by-one"),
        ],
    )
    part_d = number_part(
        f"Each cube has a side of {side} cm. Find the surface area of the stack, in square centimetres.",
        area * side * side,
        1,
        [f"Each unit square is {side} cm by {side} cm = {side * side} square centimetres.", f"{area} x {side * side}."],
        [
            (area * side, "stk-scales-an-area-by-the-side-once"),
            (area, "stk-leaves-the-area-unscaled"),
            (area * side * side * side, "stk-cubes-the-side"),
            (n * side * side, "stk-uses-the-cubes-for-the-squares"),
            (area + side * side, "scd-adds-scale"),
        ],
        follow=f"{side * side}*part_c",
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_cube_stack_chain", level=7, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_cube_stack_chain(rng, tier: str = "exam"):
    """Cubes in a stack, cubes to fill a box, the surface area in unit squares and in square centimetres."""
    _only_exam(tier)
    return retry(rng, _stack_chain, "a cube stack with three distractors per part")


def _surface(cubes) -> int:
    s = {tuple(c) for c in cubes}
    exposed = 0
    for x, y, z in s:
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            exposed += (x + dx, y + dy, z + dz) not in s
    return exposed


def _read_stack_chain(question) -> str:
    cubes = question.figure["cubes"]
    parts = split_parts(question.text)
    bw, bd, bh = (int(g) for g in _BOX_RE.search(parts["b"]).groups())
    side = int(_SIDE_RE.search(parts["d"]).group(1))
    area = _surface(cubes)
    return f"(a) {len(cubes)}; (b) {bw * bd * bh - len(cubes)}; (c) {area}; (d) {area * side * side}"


figures.register_reader("exd_x_w7f_cube_stack_chain", _read_stack_chain)

# ------------------------------------------------------------------------------------------- 2. solid properties

_EDGE_LONG_RE = re.compile(r"exactly (\d+) cubes? long")
_PAINT_RE = re.compile(r"costs \$(\d+) for each unit square")


def _properties_chain(rng):
    dims = [rng.randint(1, 4) for _ in range(3)]
    if len(set(dims)) == 1:
        return None
    a, b, c = dims
    cubes = [[x, y, z] for x in range(a) for y in range(b) for z in range(c)]
    fig = grid.grid_drawing(
        alt="A cuboid built from unit cubes, drawn on isometric dot paper.",
        grid={"type": "isometric", "width": a, "height": b, "layers": c},
        cubes=cubes,
    )
    k = rng.choice(sorted(set(dims)))
    same = sum(1 for d in dims if d == k)
    edges = 4 * same
    volume = a * b * c
    faces = [a * b, a * c, b * c]
    area = 2 * sum(faces)
    price = rng.choice((3, 4, 5, 6, 8, 12))
    unit = "cube" if k == 1 else "cubes"
    stem = pick(
        rng,
        (
            "The cuboid is built from unit cubes and drawn on isometric dot paper. There are no gaps inside it.",
            "An isometric drawing shows a solid cuboid. It is made of unit cubes packed together with no spaces left.",
            "Unit cubes are packed into a cuboid with no gaps, and the cuboid is drawn on isometric dot paper.",
        ),
    )
    part_a = number_part(
        f"How many of its edges are exactly {k} {unit} long?",
        edges,
        1,
        [f"The cuboid is {a} by {b} by {c}. There are 4 edges of each length, and {same} of the 3 lengths is {k}."],
        [
            (same, "cub-counts-lengths-not-edges"),
            (12, "cub-counts-every-edge"),
            (4 if edges != 4 else 8, "cub-counts-one-set-of-edges"),
            (3 * same, "cub-counts-three-edges-for-a-length"),
            (6, "cub-counts-faces-for-edges"),
        ],
    )
    part_b = number_part(
        "How many unit cubes make up the cuboid?",
        volume,
        1,
        [f"{a} x {b} x {c} = {volume}."],
        [
            (a + b + c, "cub-adds-the-lengths"),
            (sum(faces), "cub-adds-the-face-areas"),
            (area, "cub-gives-the-surface-for-the-volume"),
            (a * b + c, "cub-multiplies-two-lengths-only"),
            (volume + 1, "stk-count-off-by-one"),
        ],
    )
    part_c = number_part(
        "Find the surface area of the cuboid, in unit squares, counting all six faces.",
        area,
        2,
        [
            f"The faces come in pairs: {a} x {b}, {a} x {c} and {b} x {c}, which are {faces[0]}, {faces[1]} and {faces[2]}.",
            f"2 x ({faces[0]} + {faces[1]} + {faces[2]}) = {area}.",
        ],
        [
            (sum(faces), "cub-counts-one-of-each-pair-of-faces"),
            (volume, "cub-gives-the-volume-for-the-surface"),
            (4 * (a + b + c), "cub-gives-edge-length-for-surface"),
            (area - 2 * min(faces), "cub-leaves-out-a-pair-of-faces"),
            (2 * (a * b + c), "cub-multiplies-two-lengths-only"),
        ],
    )
    part_d = number_part(
        f"Painting the whole outside, bottom included, costs ${price} for each unit square. Find the cost, in dollars.",
        area * price,
        2,
        [f"{area} x ${price} = ${area * price}."],
        [
            (sum(faces) * price, "cub-counts-one-of-each-pair-of-faces"),
            (volume * price, "cub-prices-the-volume"),
            (area + price, "scd-cost-adds-price"),
            ((area - 2 * a * b) * price, "cub-leaves-out-a-pair-of-faces"),
            (area * price * 10, "scd-metre-centimetre-slip"),
        ],
        follow=f"{price}*part_c",
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_solid_properties_chain", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_solid_properties_chain(rng, tier: str = "exam"):
    """Edges of one length, volume, surface area and the cost of painting a cuboid of unit cubes."""
    _only_exam(tier)
    return retry(rng, _properties_chain, "a cuboid with three distractors per part")


def _read_properties_chain(question) -> str:
    cubes = question.figure["cubes"]
    parts = split_parts(question.text)
    dims = [1 + max(c[axis] for c in cubes) for axis in range(3)]
    k = int(_EDGE_LONG_RE.search(parts["a"]).group(1))
    price = int(_PAINT_RE.search(parts["d"]).group(1))
    check(len(cubes) == dims[0] * dims[1] * dims[2], "the picture is not a solid cuboid")
    area = _surface(cubes)
    return f"(a) {4 * sum(1 for d in dims if d == k)}; (b) {len(cubes)}; (c) {area}; (d) {area * price}"


figures.register_reader("exd_x_w7f_solid_properties_chain", _read_properties_chain)

# ------------------------------------------------------------------------------------------- 3. the net of a box


def _net_chain(rng):
    w, d, h = (rng.randint(1, 5) for _ in range(3))
    if w == d == h:
        return None
    polys, size = _cuboid_net(w, d, h, rng)
    fig = _polygon_figure(polys, size, "The net of a closed box, drawn on a square grid.")
    surface = 2 * (w * d + w * h + d * h)
    volume = w * d * h
    sheet = size[0] * size[1]
    left = sheet - surface
    edges = sorted({w, d, h}, reverse=True)
    longest = edges[0]
    check(left > 0 and sheet == (2 * w + 2 * d) * (h + 2 * d), "the sheet is not the net's bounding rectangle")
    stem = pick(
        rng,
        (
            "The net of a closed box is drawn on a square grid with each square 1 unit by 1 unit. It folds up with no "
            "gaps or overlaps. The grid is the rectangular sheet that just fits round the net.",
            "A box is to be made by folding a net cut from a rectangular sheet. The sheet is a square grid with each "
            "square 1 unit by 1 unit, and the net folds with no gaps or overlaps.",
            "On a sheet of square grid paper, each square 1 unit by 1 unit, the net of a closed box is drawn. The "
            "sheet is exactly as big as the smallest rectangle round the net, and the net folds with no overlaps.",
        ),
    )
    part_a = number_part(
        "What is the length of the longest edge of the box, in units?",
        longest,
        1,
        [f"The box is {w} by {d} by {h}, read from the faces of the net.", f"The longest edge is {longest} units."],
        [
            (edges[-1], "net-gives-the-shortest-edge"),
            (edges[1] if len(edges) > 1 and edges[1] != longest else None, "net-gives-another-edge"),
            (size[0], "net-gives-the-width-of-the-sheet"),
            (size[1], "net-gives-the-height-of-the-sheet"),
            (w + d + h, "net-adds-the-edges"),
            (volume, "net-gives-the-volume-for-an-edge"),
        ],
    )
    part_b = number_part(
        "What is the surface area of the box, in square units?",
        surface,
        2,
        [f"The area of the net is the surface area: 2 x ({w} x {d} + {w} x {h} + {d} x {h}) = {surface}."],
        [
            (2 * h * (w + d), "net-leaves-out-the-top-and-bottom"),
            (surface // 2, "net-counts-one-of-each-pair-of-faces"),
            (volume, "net-gives-the-volume-for-the-surface"),
            (sheet, "net-gives-the-sheet-for-the-surface"),
            (surface - 2 * min(w * d, w * h, d * h), "net-leaves-out-a-pair-of-faces"),
        ],
    )
    part_c = number_part(
        "What is the volume of the box, in cubic units?",
        volume,
        2,
        [f"The box is {w} by {d} by {h}: {w} x {d} x {h} = {volume}."],
        [
            (surface, "net-gives-the-surface-for-the-volume"),
            (w * d + w * h + d * h, "net-adds-the-face-areas"),
            (w + d + h, "net-adds-the-edges"),
            (w * d * h + w * d, "net-adds-a-face-to-the-volume"),
            (sheet, "net-gives-the-sheet-for-the-surface"),
        ],
    )
    part_d = number_part(
        "The net is cut out of the sheet. How many square units of the sheet are left over?",
        left,
        2,
        [f"The sheet is {size[0]} by {size[1]} = {sheet} square units.", f"{sheet} - {surface} = {left} left over."],
        [
            (sheet, "net-gives-the-sheet-for-the-surface"),
            (surface, "net-gives-the-net-not-the-leftover"),
            (sheet + surface, "scd-adds-instead-of-subtracts"),
            (sheet - volume, "net-gives-the-volume-for-the-surface"),
            (size[0] + size[1], "net-adds-the-sheet-sides"),
        ],
        follow=f"{sheet}-part_b",
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_net_chain", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_net_chain(rng, tier: str = "exam"):
    """Longest edge, surface area, volume and sheet left over for a closed box from its net."""
    _only_exam(tier)
    return retry(rng, _net_chain, "a box net with three distractors per part")


def _read_net_chain(question) -> str:
    fig = question.figure
    areas = sorted(_twice_area(p["vertices"]) // 2 for p in fig["polygons"])
    check(len(areas) == 6, "a box net has six faces")
    surface = sum(areas)
    ab, ac, bc = areas[0::2]
    volume = math.isqrt(ab * ac * bc)
    check(volume * volume == ab * ac * bc, "the face areas do not make a box")
    longest = volume // ab  # the edge shared by the two larger faces is the longest
    sheet = fig["grid"]["width"] * fig["grid"]["height"]
    return f"(a) {longest}; (b) {surface}; (c) {volume}; (d) {sheet - surface}"


figures.register_reader("exd_x_w7f_net_chain", _read_net_chain)

# ------------------------------------------------------------------------------------------- 4. elevation views

_FRONT_ASK = (
    "The front of the stack is the face on the left of the picture. Which option gives the column heights of the "
    "view from the front, from left to right?"
)
_RIGHT_ASK = (
    "The right side of the stack is the face on the right of the picture. Seen from the right side, which option "
    "gives the column heights, from left to right (the front of the stack is on your left)?"
)
_FRONT_SQUARES_ASK = "How many squares make up the view from the front?"
_PLAN_SQUARES_ASK = "How many squares make up the view from above?"


def _elevation_chain(rng):
    width, depth = rng.randint(3, 4), rng.randint(2, 3)
    h, cubes = _stack_heights(rng, width, depth, 3, 6, 20)
    front = _column_heights(cubes, "front")
    right = _column_heights(cubes, "right")
    views = {"back": front[::-1], "left": right[::-1]}
    if len({tuple(front), tuple(right), *map(tuple, views.values())}) != 4:
        return None
    fig = _stack_figure(width, depth, 3, cubes)
    n, plan = len(cubes), len(grid.plan_view(cubes))
    stem = pick(
        rng,
        (
            "A stack of unit cubes with no gaps is drawn on isometric dot paper. The views of it are drawn on a square "
            "grid, one square for each face of a cube that you can see.",
            "Look at the isometric drawing of a solid stack of unit cubes. Its front, right and top views are drawn "
            "on a square grid with one square for every cube face you can see.",
            "A solid stack of unit cubes is shown on isometric dot paper, and each view of it is drawn on a square "
            "grid: one square stands for one visible face of a cube.",
        ),
    )
    part_a = mcq_part(
        _FRONT_ASK,
        _heights_text(front),
        2,
        [f"Seen from the front, the column heights from left to right are {', '.join(str(v) for v in front)}."],
        [
            (_heights_text(views["back"]), "elev-gives-the-back-view"),
            (_heights_text(right), "elev-gives-the-side-view"),
            (_heights_text(views["left"]), "elev-gives-the-opposite-side-view"),
        ],
    )
    part_b = mcq_part(
        _RIGHT_ASK,
        _heights_text(right),
        2,
        [f"Seen from the right side, the column heights from left to right are {', '.join(str(v) for v in right)}."],
        [
            (_heights_text(views["left"]), "elev-gives-the-opposite-side-view"),
            (_heights_text(front), "elev-gives-the-front-view"),
            (_heights_text(views["back"]), "elev-gives-the-back-view"),
        ],
    )
    part_c = number_part(
        _FRONT_SQUARES_ASK,
        sum(front),
        1,
        [f"Add the column heights of the front view: {' + '.join(str(v) for v in front)} = {sum(front)}."],
        [
            (n, "elev-counts-every-cube-for-a-view"),
            (len(front), "stk-counts-the-columns"),
            (sum(right), "elev-gives-the-side-view"),
            (plan, "elev-gives-the-plan-for-the-elevation"),
            (sum(front) + 1, "stk-count-off-by-one"),
        ],
    )
    part_d = number_part(
        _PLAN_SQUARES_ASK,
        plan,
        1,
        [f"One square for each column of cubes that has a cube: {plan}."],
        [
            (n, "elev-counts-every-cube-for-a-view"),
            (width * depth, "stk-fills-the-bounding-box"),
            (sum(front), "elev-gives-the-front-view"),
            (max(front) * plan, "elev-multiplies-by-the-height"),
            (plan + 1, "stk-count-off-by-one"),
        ],
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_elevation_chain", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_elevation_chain(rng, tier: str = "exam"):
    """Front and right-side column heights of a stack and the squares in its front and top views."""
    _only_exam(tier)
    return retry(rng, _elevation_chain, "a stack whose views all differ")


def _read_elevation_chain(question) -> str:
    cubes = question.figure["cubes"]
    parts = split_parts(question.text)
    check(parts["a"] == _FRONT_ASK and parts["b"] == _RIGHT_ASK, "unknown prompts")
    front, right = _column_heights(cubes, "front"), _column_heights(cubes, "right")
    return (
        f"(a) {_heights_text(front)}; (b) {_heights_text(right)}; (c) {sum(front)}; "
        f"(d) {len({(x, y) for x, y, _ in cubes})}"
    )


figures.register_reader("exd_x_w7f_elevation_chain", _read_elevation_chain)

# ------------------------------------------------------------------------------------------- 5. composite outline

_TILE_RE = re.compile(r"costs \$(\d+) for each square")


def _outline_chain(rng):
    fig, _text, area, _cands, _working = _composite_easy(rng)
    area = int(area)
    verts = [tuple(v) for v in fig["polygons"][0]["vertices"]]
    edges = list(zip(verts, verts[1:] + verts[:1]))
    perimeter = sum(abs(x2 - x1) + abs(y2 - y1) for (x1, y1), (x2, y2) in edges)
    xs, ys = [v[0] for v in verts], [v[1] for v in verts]
    box_w, box_h = max(xs) - min(xs), max(ys) - min(ys)
    box = box_w * box_h
    bare = box - area
    check(bare > 0 and _twice_area(verts) == 2 * area, "area disagrees")
    price = rng.choice((2, 3, 4, 5, 6))
    stem = pick(
        rng,
        (
            "The shape is drawn on a square grid. Each square is 1 unit long and 1 unit wide.",
            "A shape has been outlined on grid paper. Every grid square measures 1 unit by 1 unit.",
            "Here is the outline of a shape on a square grid, where one square is 1 unit across and 1 unit up.",
        ),
    )
    part_a = number_part(
        "Count the squares inside the shape to find its area in square units.",
        area,
        1,
        [f"The shape covers {area} whole squares."],
        [
            (box, "comp-gives-the-bounding-rectangle"),
            (bare, "comp-counts-the-bare-squares"),
            (perimeter, "comp-gives-the-perimeter-for-the-area"),
            (area + 1, "stk-count-off-by-one"),
            (area - 1, "stk-count-off-by-one"),
        ],
    )
    part_b = number_part(
        "Find the perimeter of the shape, the distance all the way round its edge, in units.",
        perimeter,
        2,
        [f"Add the lengths of all {len(verts)} sides: {perimeter} units."],
        [
            (area, "comp-gives-the-area-for-the-perimeter"),
            (2 * (box_w + box_h), "comp-gives-the-bounding-perimeter"),
            (perimeter - 2, "comp-leaves-out-a-side"),
            (perimeter + 2, "comp-counts-a-side-twice"),
            (perimeter // 2, "comp-gives-half-the-perimeter"),
        ],
    )
    part_c = number_part(
        f"A rectangle {box_w} units wide and {box_h} units high just fits round the shape. How many squares of the "
        f"rectangle are not covered by the shape?",
        bare,
        2,
        [f"The rectangle has {box_w} x {box_h} = {box} squares.", f"{box} - {area} = {bare} squares are bare."],
        [
            (box, "comp-gives-the-bounding-rectangle"),
            (area, "comp-counts-the-covered-squares"),
            (box + area, "scd-adds-instead-of-subtracts"),
            (box_w + box_h, "comp-adds-the-sides"),
            (bare + 1, "stk-count-off-by-one"),
        ],
        follow=f"{box}-part_a",
    )
    part_d = number_part(
        f"Tiling the shape costs ${price} for each square. Find the cost of tiling the shape, in dollars.",
        area * price,
        1,
        [f"{area} x ${price} = ${area * price}."],
        [
            (box * price, "comp-gives-the-bounding-rectangle"),
            (perimeter * price, "comp-gives-the-perimeter-for-the-area"),
            (bare * price, "comp-counts-the-bare-squares"),
            (area + price, "scd-cost-adds-price"),
            (area * price * 10, "scd-metre-centimetre-slip"),
        ],
        follow=f"{price}*part_a",
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_composite_chain", level=5, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_composite_chain(rng, tier: str = "exam"):
    """Area, perimeter, bare squares of the rectangle round it and tiling cost for a rectilinear outline."""
    _only_exam(tier)
    return retry(rng, _outline_chain, "an outline with three distractors per part")


_BOX_SIZE_RE = re.compile(r"A rectangle (\d+) units wide and (\d+) units high")


def _read_outline_chain(question) -> str:
    fig = question.figure
    verts = [tuple(v) for v in fig["polygons"][0]["vertices"]]
    parts = split_parts(question.text)
    area = _twice_area(verts) // 2
    perimeter = sum(abs(x2 - x1) + abs(y2 - y1) for (x1, y1), (x2, y2) in zip(verts, verts[1:] + verts[:1]))
    w, h = (int(g) for g in _BOX_SIZE_RE.search(parts["c"]).groups())
    xs, ys = [v[0] for v in verts], [v[1] for v in verts]
    check((max(xs) - min(xs), max(ys) - min(ys)) == (w, h), "the stated rectangle does not fit the shape")
    price = int(_TILE_RE.search(parts["d"]).group(1))
    return f"(a) {area}; (b) {perimeter}; (c) {w * h - area}; (d) {area * price}"


figures.register_reader("exd_x_w7f_composite_chain", _read_outline_chain)
