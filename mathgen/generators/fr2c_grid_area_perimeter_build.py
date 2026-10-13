"""Generates `fr2c_grid_area_perimeter` questions: area and perimeter of an irregular shape drawn on a square grid.

One skill: ``f10-area-perimeter-count-squares`` (``answer: number``). The stimulus is the outline of a shape on a grid of squares,
drawn with the existing ``cartesian`` figure kind (the grid, and one segment for each side of the outline). The shape is a connected set of
whole squares with no holes. Some corners can be cut across a single square, which leaves a half square, so the area is
found from whole and part squares.

- easy: the area of a shape made only of whole squares, each 1 cm by 1 cm.
- medium: either the area of a shape with two or four half squares (the halves pair up to whole squares), or the
  perimeter of a shape made only of whole squares, which is longer than the perimeter of its bounding rectangle.
- hard: the same two questions on a grid whose squares are 2, 3 or 5 metres on a side, so the area scales by the square
  of the side and the perimeter by the side.

The reader (`figures.register_reader`) recomputes every answer from the drawn polygon alone with the shoelace formula and
the lengths of its sides, and the scale from the question text.
"""

from __future__ import annotations

import math
import re

from mathgen import figures
from mathgen.core import attempts, generator

from . import _fr2c_kit as kit

_ALT = "The outline of a shape drawn on a grid of squares."


# ------------------------------------------------------------------------------------------ shapes


def _grow(rng, cols, rows, n):
    """A connected set of ``n`` cells inside a ``cols`` by ``rows`` grid."""
    cells = {(rng.randrange(cols), rng.randrange(rows))}
    for _ in attempts(400, "a polyomino of the wanted size"):
        if len(cells) == n:
            break
        x, y = rng.choice(sorted(cells))
        dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        c = (x + dx, y + dy)
        if 0 <= c[0] < cols and 0 <= c[1] < rows:
            cells.add(c)
    return cells


def _boundary(cells):
    """The boundary of ``cells`` as a counter-clockwise list of lattice points one step apart, or None when the shape
    has a hole or pinches to a single point."""
    nxt = {}
    for x, y in cells:
        edges = []
        if (x, y - 1) not in cells:
            edges.append(((x, y), (x + 1, y)))
        if (x + 1, y) not in cells:
            edges.append(((x + 1, y), (x + 1, y + 1)))
        if (x, y + 1) not in cells:
            edges.append(((x + 1, y + 1), (x, y + 1)))
        if (x - 1, y) not in cells:
            edges.append(((x, y + 1), (x, y)))
        for a, b in edges:
            if a in nxt:
                return None
            nxt[a] = b
    start = min(nxt)
    loop = [start]
    while True:
        v = nxt[loop[-1]]
        if v == start:
            break
        loop.append(v)
        if len(loop) > len(nxt):
            return None
    return loop if len(loop) == len(nxt) else None


def _convex_corners(loop):
    out = []
    n = len(loop)
    for i in range(n):
        a, v, b = loop[i - 1], loop[i], loop[(i + 1) % n]
        cross = (v[0] - a[0]) * (b[1] - v[1]) - (v[1] - a[1]) * (b[0] - v[0])
        if cross > 0:
            out.append(i)
    return out


def _cut_cell(loop, i):
    """The cell whose corner ``loop[i]`` is (the cell on the inside of the boundary at that point)."""
    n = len(loop)
    a, v, b = loop[i - 1], loop[i], loop[(i + 1) % n]
    xs = [a[0], v[0], b[0]]
    ys = [a[1], v[1], b[1]]
    return (min(xs), min(ys))


def _shape(rng, cols, rows, n, cuts, need_notch):
    """(cells, polygon points, number of cuts). The polygon is the boundary with ``cuts`` convex corners cut across."""
    for _ in attempts(3000, "a shape with the wanted features"):
        cells = _grow(rng, cols, rows, n)
        if len(cells) != n:
            continue
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        w, h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        if w < 3 or h < 3 or w * h == n:
            continue
        loop = _boundary(cells)
        if loop is None:
            continue
        if need_notch and 2 * (w + h) == len(loop):
            continue
        if cuts == 0:
            return cells, loop, 0
        corners = _convex_corners(loop)
        rng.shuffle(corners)
        chosen, used = [], set()
        for i in corners:
            cell = _cut_cell(loop, i)
            m = len(loop)
            if cell in used or any(min((i - j) % m, (j - i) % m) < 3 for j in chosen):
                continue
            chosen.append(i)
            used.add(cell)
            if len(chosen) == cuts:
                break
        if len(chosen) < cuts:
            continue
        poly = [p for k, p in enumerate(loop) if k not in set(chosen)]
        return cells, poly, cuts
    raise AssertionError("unreachable")


def _corners(poly):
    """``poly`` without the points that sit in the middle of a straight side."""
    n = len(poly)
    out = []
    for i in range(n):
        a, v, b = poly[i - 1], poly[i], poly[(i + 1) % n]
        if (v[0] - a[0]) * (b[1] - v[1]) - (v[1] - a[1]) * (b[0] - v[0]) != 0:
            out.append(v)
    return out


def _figure(poly, cols, rows):
    """The outline of the shape on a coordinate grid: one segment per side, in order round the shape."""
    pts = _corners(poly)
    segments = [{"from": list(a), "to": list(b)} for a, b in zip(pts, pts[1:] + pts[:1])]
    return figures.cartesian(
        x_range=(0, cols),
        y_range=(0, rows),
        grid=1,
        alt=_ALT,
        segments=segments,
        x_label="",
        y_label="",
        tick_labels=False,
    )


# ------------------------------------------------------------------------------------------ questions


def _stats(cells, loop_full, poly, cuts):
    w = max(c[0] for c in cells) - min(c[0] for c in cells) + 1
    h = max(c[1] for c in cells) - min(c[1] for c in cells) + 1
    boundary_cells = {
        c for c in cells if any((c[0] + dx, c[1] + dy) not in cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    }
    return dict(
        n=len(cells),
        full=len(cells) - cuts,
        cuts=cuts,
        w=w,
        h=h,
        perimeter=len(loop_full),
        boundary_cells=len(boundary_cells),
    )


def _build(rng, kind, side, with_cuts):
    cols, rows = rng.choice([(6, 5), (7, 5), (6, 6), (7, 6), (5, 5)])
    n = rng.randint(9, 16)
    cuts = rng.choice([2, 4]) if with_cuts else 0
    cells, poly, cuts = _shape(rng, cols, rows, n, cuts, need_notch=(kind == "perimeter"))
    loop_full = _boundary(cells)
    st = _stats(cells, loop_full, poly, cuts)
    fig = _figure(poly, cols, rows)
    unit = f"{side} m" if side > 1 else "1 cm"
    square = (
        "Each square on the grid is 1 cm by 1 cm." if side == 1 else f"The side of each square on the grid is {side} m."
    )
    if kind == "area":
        area_units = st["full"] + cuts // 2
        value = area_units * side * side
        noun = "square centimetres" if side == 1 else "square metres"
        ask = (
            "What is the area of the shape, in {noun}? Some squares are cut across and are half a square."
            if cuts
            else "What is the area of the shape, in {noun}?"
        ).format(noun=noun)
        wrong = [
            ("fr2c-counted-the-bounding-rectangle", st["w"] * st["h"] * side * side),
            ("fr2c-part-squares-counted-whole", st["n"] * side * side)
            if cuts
            else ("fr2c-count-off-by-one", (st["n"] + 1) * side * side),
            ("fr2c-part-squares-ignored", st["full"] * side * side)
            if cuts
            else ("fr2c-count-off-by-one", (st["n"] - 1) * side * side),
            ("perimeter-for-area", st["perimeter"] * side),
            ("linear-factor-on-area", area_units * side)
            if side > 1
            else ("fr2c-count-off-by-one", (area_units + 2) * side * side),
            ("fr2c-scale-ignored", area_units)
            if side > 1
            else ("fr2c-count-off-by-one", (area_units - 2) * side * side),
        ]
        working = [
            f"There are {st['full']} whole squares"
            + (f" and {cuts} half squares, which make {cuts // 2} more." if cuts else "."),
            f"{area_units} squares x {side * side} = {value}." if side > 1 else f"The area is {value} cm2.",
        ]
    else:
        value = st["perimeter"] * side
        noun = "centimetres" if side == 1 else "metres"
        ask = f"What is the perimeter of the shape, in {noun}?"
        wrong = [
            ("fr2c-perimeter-of-bounding-rectangle", 2 * (st["w"] + st["h"]) * side),
            ("area-for-perimeter", st["n"] * side * side),
            ("fr2c-boundary-squares-counted", st["boundary_cells"] * side),
            ("fr2c-perimeter-missed-inner-edges", (st["perimeter"] - 2) * side),
            ("fr2c-scale-ignored", st["perimeter"]) if side > 1 else ("fr2c-count-off-by-one", (st["perimeter"] + 2)),
            ("fr2c-perimeter-scaled-as-area", st["perimeter"] * side * side)
            if side > 1
            else ("fr2c-count-off-by-one", st["perimeter"] - 2),
        ]
        working = [
            f"Counting the sides of the squares around the edge gives {st['perimeter']}.",
            f"{st['perimeter']} x {side} = {value}.",
        ]
    text = f"{square} {ask}"
    return fig, text, value, wrong, working


def _easy(rng):
    return _build(rng, "area", 1, False)


def _medium(rng):
    if rng.random() < 0.5:
        return _build(rng, "area", 1, True)
    return _build(rng, "perimeter", 1, False)


def _hard(rng):
    side = rng.choice([2, 3, 5])
    if rng.random() < 0.5:
        return _build(rng, "area", side, True)
    return _build(rng, "perimeter", side, False)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fr2c_grid_area_perimeter",
    level=4,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2c_grid_area_perimeter(rng, tier="easy"):
    """Area or perimeter of a shaded shape on a square grid. See the module docstring for the tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(60, "a shape question with three different wrong answers"):
        fig, text, value, wrong, working = _TIERS[tier](rng)
        wrong = [(m, w) for m, w in wrong if w > 0]
        try:
            fields = kit.number_fields(value, wrong)
        except kit.Redraw:
            continue
        break
    return dict(text=text, working=working, figure=fig, **fields)


# ------------------------------------------------------------------------------------------ reader


def _read_grid(question) -> str:
    fig = question.figure
    poly = [s["from"] for s in fig["segments"]]
    pairs = list(zip(poly, poly[1:] + poly[:1]))
    area = abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in pairs)) / 2
    perimeter = sum(math.dist(p, q) for p, q in pairs)
    m = re.search(r"side of each square on the grid is (\d+) m", question.text)
    side = int(m.group(1)) if m else 1
    value = area * side * side if "area" in question.text else perimeter * side
    return str(round(value))


figures.register_reader("fr2c_grid_area_perimeter", _read_grid)
