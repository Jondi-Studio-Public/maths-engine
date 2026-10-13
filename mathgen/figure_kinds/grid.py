"""The ``grid_drawing`` figure kind: square and isometric grids, shaded cells, lattice shapes and cube stacks.

One kind serves every picture that is drawn on a grid of unit cells: a scale drawing or a floor plan, a map
with grid references and symbols, a shape made of squares, a tessellation tile, a stack of cubes on
isometric dot paper, and the empty grid a learner shades an answer on (a plan view, an elevation, a net).

Coordinates
    A square grid is ``width`` cells across and ``height`` cells up. Cell ``[x, y]`` is the unit square whose
    lower-left corner is the lattice point ``(x, y)``: ``x`` counts columns from the left from 0, ``y``
    counts rows from the bottom from 0, as on the coordinate plane. Lattice points run from ``(0, 0)`` to
    ``(width, height)``.

    An isometric grid draws cubes. Cube ``[x, y, z]`` is a unit cube: ``x`` runs left to right, ``y`` runs
    from front (0) to back, ``z`` counts layers up from the ground (0). The picture is seen from the front
    right, above. ``width`` and ``height`` bound ``x`` and ``y``; ``layers`` bounds ``z``.

Keys (beyond ``kind``, ``alt`` and an optional ``title``)
    - ``grid``: ``{"type": "square" | "isometric", "width": int, "height": int}``, plus ``"layers": int`` on
      an isometric grid.
    - ``cells`` (square): a list of ``{"cells": [[x, y], ...], "style": "shaded" | "hatched" | "dotted"}``.
      Each style is drawn differently by more than colour.
    - ``polygons`` (square): a list of ``{"vertices": [[x, y], ...], "style": "solid" | "dashed", "label"?}``
      with every vertex a lattice point of the grid.
    - ``lines`` (square): a list of ``{"points": [[x, y], ...], "style": "road" | "river" | "rail" | "wall" |
      "fence" | "mirror"}``; points on the grid, in half steps.
    - ``labels``: a list of ``{"at": [x, y], "text": str}`` written at a point of the grid.
    - ``symbols`` (square): a list of ``{"at": [x, y], "symbol": name}`` from :data:`SYMBOLS`, drawn as
      plain pictograms. ``"legend": true`` also draws a key naming every symbol used.
    - ``path`` (square): ``{"cells": [[x, y], ...]}``, a route drawn as arrows from the centre of each cell to the
      next, with a start marker on the first. Each step moves to a cell that shares an edge with the last.
    - ``column_labels`` / ``row_labels`` (square): one short label per column (left to right) and per row
      (bottom to top), for grid references.
    - ``scale``: ``{"text": "1 cm : 2 m", "bar_cells"?: int}``: the scale as written, and a scale bar that
      many cells long.
    - ``north``: ``true`` draws a north arrow pointing up the grid.
    - ``cubes`` (isometric): a list of ``[x, y, z]``. Every cube stands on the ground or on another cube.
    - ``canvas``: where the learner draws a ``grid_cells`` answer: ``{"width": int, "height": int}`` for a
      blank square grid drawn beside the figure, or ``"grid"`` to shade on the figure's own square grid
      (cells already shown there are given, not part of the reply).

The alt text names only the object ("a stack of cubes on isometric dot paper"); :func:`mathgen.figures.labels`
sees every label, legend entry, axis label and the scale through the ``texts`` hook, so the leak check covers
them.
"""

from __future__ import annotations

from typing import Iterable

from mathgen.figures import _is_num, check, register_kind

__all__ = [
    "CELL_STYLES",
    "LINE_STYLES",
    "SYMBOLS",
    "cube_count",
    "elevation",
    "grid_drawing",
    "plan_view",
    "tops_all_visible",
]

MAX_SQUARE_SIDE = 30
MAX_ISOMETRIC_SIDE = 8
MAX_CANVAS_SIDE = 20
MAX_CUBES = 64
MAX_LABEL_LENGTH = 24

CELL_STYLES = ("shaded", "hatched", "dotted")
POLYGON_STYLES = ("solid", "dashed")
LINE_STYLES = ("road", "river", "rail", "wall", "fence", "mirror")

#: Map and plan symbols a figure may draw, with the name a legend writes for each.
SYMBOLS: dict[str, str] = {
    "house": "house",
    "school": "school",
    "church": "church",
    "hospital": "hospital",
    "tree": "tree",
    "bridge": "bridge",
    "campground": "campground",
    "lookout": "lookout",
    "post_office": "post office",
    "railway_station": "railway station",
    "airport": "airport",
    "lighthouse": "lighthouse",
    "picnic_area": "picnic area",
    "toilets": "toilets",
    "parking": "parking",
    "information": "information",
    "door": "door",
    "window": "window",
}

_SQUARE_ONLY = ("cells", "polygons", "lines", "symbols", "path", "column_labels", "row_labels")
MAX_PATH_CELLS = 60


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _int_pair(p) -> bool:
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(_is_int(c) for c in p)


def _short_text(t) -> bool:
    return isinstance(t, str) and bool(t.strip()) and len(t) <= MAX_LABEL_LENGTH


def grid_drawing(
    *,
    alt: str,
    grid: dict,
    cells: Iterable[dict] = (),
    polygons: Iterable[dict] = (),
    lines: Iterable[dict] = (),
    labels: Iterable[dict] = (),
    symbols: Iterable[dict] = (),
    legend: bool = False,
    path: Iterable[Iterable[int]] = (),
    column_labels: Iterable[str] = (),
    row_labels: Iterable[str] = (),
    scale: dict | None = None,
    north: bool = False,
    cubes: Iterable[Iterable[int]] = (),
    canvas: dict | str | None = None,
    title: str = "",
) -> dict:
    """A ``grid_drawing`` spec, checked. Empty groups are left out of the spec. See the module docstring."""
    spec: dict = {"kind": "grid_drawing", "alt": alt, "grid": dict(grid)}
    groups = {
        "cells": [{**c, "cells": [list(p) for p in c["cells"]]} for c in cells],
        "polygons": [{**p, "vertices": [list(v) for v in p["vertices"]]} for p in polygons],
        "lines": [{**ln, "points": [list(v) for v in ln["points"]]} for ln in lines],
        "labels": [dict(lb) for lb in labels],
        "symbols": [dict(s) for s in symbols],
        "column_labels": list(column_labels),
        "row_labels": list(row_labels),
        "cubes": [list(c) for c in cubes],
    }
    spec.update({key: value for key, value in groups.items() if value})
    if legend:
        spec["legend"] = True
    if path:
        spec["path"] = {"cells": [list(c) for c in path]}
    if scale is not None:
        spec["scale"] = dict(scale)
    if north:
        spec["north"] = True
    if canvas is not None:
        spec["canvas"] = canvas if isinstance(canvas, str) else dict(canvas)
    if title:
        spec["title"] = title
    return check(spec)


# ---------------------------------------------------------------- cube helpers


def _cube_set(cubes) -> set[tuple[int, int, int]]:
    return {(int(x), int(y), int(z)) for x, y, z in cubes}


def cube_count(cubes) -> int:
    """How many cubes are in the stack."""
    return len(_cube_set(cubes))


def plan_view(cubes) -> list[list[int]]:
    """The cells of the view from above: ``[x, y]`` wherever any cube stands, sorted."""
    return sorted([x, y] for x, y in {(x, y) for x, y, _ in _cube_set(cubes)})


def elevation(cubes, side: str, depth: int | None = None) -> list[list[int]]:
    """The cells of the view from ``side`` ("front", "back", "left" or "right"), sorted.

    A cell is ``[column, layer]``, columns counted from the viewer's left. From the front the column is ``x``;
    from the right it is ``y`` (the front of the stack is on the viewer's left); from the back it is
    ``width - 1 - x`` and from the left ``depth - 1 - y``, so those two need the grid's extent: ``depth`` is
    the grid's ``width`` from the back and its ``height`` from the left (the largest coordinate plus one if
    omitted).
    """
    cubes = _cube_set(cubes)
    if depth is None:
        axis = 0 if side == "back" else 1
        depth = max((c[axis] for c in cubes), default=-1) + 1
    columns = {
        "front": lambda x, y: x,
        "right": lambda x, y: y,
        "back": lambda x, y: depth - 1 - x,
        "left": lambda x, y: depth - 1 - y,
    }
    if side not in columns:
        raise ValueError(f"unknown side {side!r}")
    to_column = columns[side]
    return sorted([c, z] for c, z in {(to_column(x, y), z) for x, y, z in cubes})


def _heights(cubes) -> dict[tuple[int, int], int]:
    heights: dict[tuple[int, int], int] = {}
    for x, y, z in cubes:
        heights[(x, y)] = max(heights.get((x, y), 0), z + 1)
    return heights


def tops_all_visible(cubes) -> bool:
    """True when no cube can hide behind another in the picture, so the count is readable.

    The picture is seen from the front right, above, so a column can only be hidden by a taller column in
    front of it or to its right. A stack passes when no column has one: every column is then seen down to
    its top, and because every cube is supported each column is solid to the ground.
    """
    heights = _heights(_cube_set(cubes))
    for (x, y), h in heights.items():
        for (x2, y2), h2 in heights.items():
            if (x2, y2) != (x, y) and x2 >= x and y2 <= y and h2 > h:
                return False
    return True


# ---------------------------------------------------------------- validator


def _check_points(points, w: int, h: int, *, step: float, name: str) -> list[str]:
    if not isinstance(points, list) or not points:
        return [f"{name} needs a non-empty list of points"]
    for p in points:
        if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(_is_num(c) for c in p)):
            return [f"{name} has a point that is not [x, y]"]
        if any((c / step) != int(c / step) for c in p):
            return [f"{name} point {list(p)} is off the lattice"]
        if not (0 <= p[0] <= w and 0 <= p[1] <= h):
            return [f"{name} point {list(p)} is off the grid"]
    return []


def _validate_grid(spec: dict) -> list[str]:
    grid = spec.get("grid")
    if not isinstance(grid, dict) or grid.get("type") not in ("square", "isometric"):
        return ['grid must be {"type": "square" | "isometric", "width", "height"}']
    w, h = grid.get("width"), grid.get("height")
    side = MAX_SQUARE_SIDE if grid["type"] == "square" else MAX_ISOMETRIC_SIDE
    if not (_is_int(w) and _is_int(h) and 1 <= w <= side and 1 <= h <= side):
        return [f"grid width and height must be whole numbers from 1 to {side}"]
    problems: list[str] = []
    if grid["type"] == "isometric":
        layers = grid.get("layers")
        if not (_is_int(layers) and 1 <= layers <= MAX_ISOMETRIC_SIDE):
            problems.append(f"an isometric grid needs layers from 1 to {MAX_ISOMETRIC_SIDE}")
            return problems
        for key in _SQUARE_ONLY + ("canvas",):
            if spec.get(key) and not (key == "canvas" and isinstance(spec[key], dict)):
                problems.append(f"{key} is drawn only on a square grid")
        problems.extend(_validate_cubes(spec.get("cubes", []), w, h, layers))
    elif spec.get("cubes"):
        problems.append("cubes are drawn only on an isometric grid")
    else:
        problems.extend(_validate_square(spec, w, h))
    problems.extend(_validate_common(spec, w, h))
    return problems


def _validate_cubes(cubes, w: int, h: int, layers: int) -> list[str]:
    if not isinstance(cubes, list) or not cubes:
        return ["an isometric grid needs at least one cube"]
    if len(cubes) > MAX_CUBES:
        return [f"at most {MAX_CUBES} cubes"]
    seen: set[tuple[int, int, int]] = set()
    for c in cubes:
        if not (isinstance(c, (list, tuple)) and len(c) == 3 and all(_is_int(v) for v in c)):
            return ["a cube is [x, y, z] in whole numbers"]
        x, y, z = c
        if not (0 <= x < w and 0 <= y < h and 0 <= z < layers):
            return [f"cube {list(c)} is outside the grid"]
        if (x, y, z) in seen:
            return [f"cube {list(c)} is listed twice"]
        seen.add((x, y, z))
    for x, y, z in seen:
        if z > 0 and (x, y, z - 1) not in seen:
            return [f"cube {[x, y, z]} is not supported"]
    return []


def _validate_square(spec: dict, w: int, h: int) -> list[str]:
    problems: list[str] = []
    for group in spec.get("cells", []):
        if not isinstance(group, dict) or group.get("style") not in CELL_STYLES:
            problems.append(f"a cell group needs a style from {CELL_STYLES}")
            continue
        cells = group.get("cells")
        if not isinstance(cells, list) or not cells or not all(_int_pair(c) for c in cells):
            problems.append("a cell group needs a non-empty list of [x, y] cells")
        elif any(not (0 <= x < w and 0 <= y < h) for x, y in cells):
            problems.append("a shaded cell is off the grid")
        elif len({tuple(c) for c in cells}) != len(cells):
            problems.append("a cell is listed twice in one group")
    for poly in spec.get("polygons", []):
        if not isinstance(poly, dict) or poly.get("style", "solid") not in POLYGON_STYLES:
            problems.append(f"a polygon style is one of {POLYGON_STYLES}")
            continue
        vertices = poly.get("vertices")
        problems.extend(_check_points(vertices, w, h, step=1, name="polygon"))
        if isinstance(vertices, list) and len(vertices) < 3:
            problems.append("a polygon has at least three vertices")
        if "label" in poly and not _short_text(poly["label"]):
            problems.append("a polygon label must be short text")
    for line in spec.get("lines", []):
        if not isinstance(line, dict) or line.get("style") not in LINE_STYLES:
            problems.append(f"a line style is one of {LINE_STYLES}")
            continue
        problems.extend(_check_points(line.get("points"), w, h, step=0.5, name="line"))
        if isinstance(line.get("points"), list) and len(line["points"]) < 2:
            problems.append("a line has at least two points")
    for sym in spec.get("symbols", []):
        if not isinstance(sym, dict) or sym.get("symbol") not in SYMBOLS:
            problems.append(f"a symbol is one of {sorted(SYMBOLS)}")
            continue
        problems.extend(_check_points([sym.get("at")], w, h, step=0.5, name="symbol"))
    if "path" in spec:
        path = spec["path"]
        cells = path.get("cells") if isinstance(path, dict) and set(path) == {"cells"} else None
        if not isinstance(cells, list) or not 2 <= len(cells) <= MAX_PATH_CELLS or not all(_int_pair(c) for c in cells):
            problems.append(f'path is {{"cells": [[x, y], ...]}} with 2 to {MAX_PATH_CELLS} cells')
        elif any(not (0 <= x < w and 0 <= y < h) for x, y in cells):
            problems.append("a path cell is off the grid")
        elif any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(cells, cells[1:])):
            problems.append("each step of a path moves to a cell that shares an edge with the last")
    if "legend" in spec and (spec["legend"] is not True or not spec.get("symbols")):
        problems.append("legend is true, and only with symbols")
    for key, count in (("column_labels", w), ("row_labels", h)):
        if key in spec:
            tags = spec[key]
            if not isinstance(tags, list) or len(tags) != count or not all(_short_text(t) for t in tags):
                problems.append(f"{key} needs one short label for each of the {count} {key.split('_')[0]}s")
            elif len(set(tags)) != len(tags):
                problems.append(f"{key} must be different from each other")
    canvas = spec.get("canvas")
    if canvas is not None and canvas != "grid":
        if not isinstance(canvas, dict) or set(canvas) != {"width", "height"}:
            problems.append('canvas is {"width", "height"} or "grid"')
        elif not all(_is_int(canvas[k]) and 1 <= canvas[k] <= MAX_CANVAS_SIDE for k in ("width", "height")):
            problems.append(f"canvas width and height must be whole numbers from 1 to {MAX_CANVAS_SIDE}")
    return problems


def _validate_common(spec: dict, w: int, h: int) -> list[str]:
    problems: list[str] = []
    for label in spec.get("labels", []):
        if not isinstance(label, dict) or not _short_text(label.get("text")):
            problems.append("a label is {at, text} with short text")
            continue
        problems.extend(_check_points([label.get("at")], w, h, step=0.5, name="label"))
    scale = spec.get("scale")
    if scale is not None:
        if not isinstance(scale, dict) or not _short_text(scale.get("text")):
            problems.append('scale is {"text", "bar_cells"?} with short text')
        elif "bar_cells" in scale and not (_is_int(scale["bar_cells"]) and 1 <= scale["bar_cells"] <= w):
            problems.append("a scale bar is a whole number of cells, no longer than the grid is wide")
    if "north" in spec and spec["north"] is not True:
        problems.append("north is true or left out")
    canvas = spec.get("canvas")
    if isinstance(canvas, dict) and spec["grid"]["type"] == "isometric":
        if set(canvas) != {"width", "height"} or not all(
            _is_int(canvas[k]) and 1 <= canvas[k] <= MAX_CANVAS_SIDE for k in ("width", "height")
        ):
            problems.append(f"canvas width and height must be whole numbers from 1 to {MAX_CANVAS_SIDE}")
    return problems


def _grid_texts(spec: dict) -> list[str]:
    out = [label.get("text", "") for label in spec.get("labels", [])]
    out.extend(p.get("label", "") for p in spec.get("polygons", []))
    out.extend(spec.get("column_labels", []))
    out.extend(spec.get("row_labels", []))
    if spec.get("legend"):
        out.extend(SYMBOLS[s["symbol"]] for s in spec.get("symbols", []))
    if spec.get("scale"):
        out.append(spec["scale"].get("text", ""))
    return out


register_kind("grid_drawing", _validate_grid, texts=_grid_texts)
