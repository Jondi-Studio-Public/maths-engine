"""Descriptions for the ``grid_drawing`` figure kind (spec in `mathgen.figure_kinds.grid`).

A square grid is described in reading order: its size, how its squares are named, the shaded, hatched and
dotted squares, the shapes, the lines, the path, the symbols, the labels, the key, the scale, the north arrow and
the blank grid beside it. A stack of cubes is described one column at a time. Nothing is worked out: positions
and heights are stated as drawn, and the number of cubes is never stated.
"""

from __future__ import annotations

from mathgen.describe import count, join, num, point, register_describer, sentence
from mathgen.figure_kinds.grid import SYMBOLS
from mathgen.render.grid import canvas_below

_CELL_WORDS = {
    "shaded": "shaded",
    "hatched": "hatched with diagonal lines",
    "dotted": "filled with dots",
}
_LINE_WORDS = {
    "road": "A road (a thick double line)",
    "river": "A river (a wavy blue line)",
    "rail": "A railway (a line with cross ticks)",
    "wall": "A wall (a thick line)",
    "fence": "A fence (a dashed line with posts)",
    "mirror": "A mirror line (a dash-dot line)",
}
_STEPS = {(1, 0): "right", (-1, 0): "left", (0, 1): "up", (0, -1): "down"}


class _Names:
    """How a square is named: by its column and row labels if the figure has them, else by position."""

    def __init__(self, spec: dict):
        self.cols = spec.get("column_labels")
        self.rows = spec.get("row_labels")

    def col(self, x: int) -> str:
        return self.cols[x] if self.cols else num(x + 1)

    def row(self, y: int) -> str:
        return self.rows[y] if self.rows else num(y + 1)

    def square(self, x: int, y: int) -> str:
        if self.cols and self.rows:
            return f"{self.cols[x]}{self.rows[y]}"
        return f"column {self.col(x)}, row {self.row(y)}"


def _runs(xs: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for x in sorted(xs):
        if out and x == out[-1][1] + 1:
            out[-1] = (out[-1][0], x)
        else:
            out.append((x, x))
    return out


def _cell_group(cells: list, names: _Names) -> str:
    rows: dict[int, list[int]] = {}
    for x, y in cells:
        rows.setdefault(y, []).append(x)
    items = []
    for y in sorted(rows, reverse=True):
        runs = _runs(rows[y])
        if names.cols and names.rows:
            for a, b in runs:
                items.append(names.square(a, y) if a == b else f"{names.square(a, y)} to {names.square(b, y)}")
        else:
            spans = [names.col(a) if a == b else f"{names.col(a)} to {names.col(b)}" for a, b in runs]
            noun = "column" if len(runs) == 1 and runs[0][0] == runs[0][1] else "columns"
            items.append(f"row {names.row(y)}, {noun} {join(spans)}")
    return "; ".join(items)


def _is_centre(at) -> bool:
    return all(float(c) % 1 == 0.5 for c in at)


def _where(at, names: _Names) -> str:
    if _is_centre(at):
        return f"in square {names.square(int(at[0]), int(at[1]))}"
    return f"at the point {point(at)}"


def _describe_square(spec: dict) -> list[str]:
    grid = spec["grid"]
    w, h = grid["width"], grid["height"]
    names = _Names(spec)
    out = [sentence(f"A square grid {count(w, 'square')} wide and {count(h, 'square')} high")]
    if names.cols and names.rows:
        out.append(
            sentence(
                f"The columns are labelled {join(names.cols)} from left to right and the rows are labelled "
                f"{join(names.rows)} from the bottom to the top"
            )
        )
    else:
        if names.cols:
            out.append(sentence(f"The columns are labelled {join(names.cols)} from left to right"))
        if names.rows:
            out.append(sentence(f"The rows are labelled {join(names.rows)} from the bottom to the top"))
        if not names.cols or not names.rows:
            out.append(
                sentence(
                    "Squares are named by column, counted from the left, and row, counted from the bottom, "
                    "starting at 1"
                    if not names.cols and not names.rows
                    else "Squares without a label are counted from 1, from the left or from the bottom"
                )
            )
    if any(spec.get(k) for k in ("polygons", "lines", "labels", "symbols")):
        out.append("Points are written (across, up) from the bottom left corner of the grid.")
    for group in spec.get("cells", []):
        out.append(sentence(f"Squares {_CELL_WORDS[group['style']]}: {_cell_group(group['cells'], names)}"))
    for poly in spec.get("polygons", []):
        corners = join([point(v) for v in poly["vertices"]])
        label = f", labelled {poly['label']}" if poly.get("label") else ""
        out.append(
            sentence(f"A {poly.get('style', 'solid')} shape with corners at {corners}, joined in that order{label}")
        )
    for line in spec.get("lines", []):
        pts = [point(p) for p in line["points"]]
        route = (
            f"from {pts[0]} to {pts[1]}" if len(pts) == 2 else f"from {pts[0]} through {join(pts[1:-1])} to {pts[-1]}"
        )
        verb = "is drawn" if line["style"] == "mirror" else "runs"
        out.append(sentence(f"{_LINE_WORDS[line['style']]} {verb} {route}"))
    if spec.get("path"):
        cells = spec["path"]["cells"]
        moves = []
        for (x0, y0), (x1, y1) in zip(cells, cells[1:]):
            moves.append(f"{_STEPS[(x1 - x0, y1 - y0)]} to {names.square(x1, y1)}")
        out.append(
            sentence(
                f"A path, drawn as arrows from the middle of one square to the next, starts in square "
                f"{names.square(*cells[0])} and moves {', then '.join(moves)}"
            )
        )
    for sym in spec.get("symbols", []):
        out.append(sentence(f"The {SYMBOLS[sym['symbol']]} symbol is {_where(sym['at'], names)}"))
    for lab in spec.get("labels", []):
        out.append(sentence(f"The label {lab['text']} is written {_where(lab['at'], names)}"))
    if spec.get("legend"):
        used = list(dict.fromkeys(SYMBOLS[s["symbol"]] for s in spec["symbols"]))
        out.append(sentence(f"A key below the grid names the symbols: {join(used)}"))
    return out


def _describe_cubes(spec: dict) -> list[str]:
    grid = spec["grid"]
    w, h, layers = grid["width"], grid["height"], grid["layers"]
    out = [
        sentence(
            f"A stack of cubes on isometric dot paper, seen from the front right and above. The dot paper has room "
            f"for {count(w, 'column')} from left to right, {count(h, 'row')} from front to back and "
            f"{count(layers, 'layer')} up"
        ),
        "The cubes stand in columns; each column is described below from the back row to the front row and from "
        "left to right.",
    ]
    heights: dict[tuple[int, int], int] = {}
    for x, y, z in spec["cubes"]:
        heights[(x, y)] = max(heights.get((x, y), 0), z + 1)
    for x, y in sorted(heights, key=lambda c: (-c[1], c[0])):
        out.append(
            sentence(
                f"A column {count(heights[(x, y)], 'cube')} high in row {y + 1} from the front, "
                f"column {x + 1} from the left"
            )
        )
    for lab in spec.get("labels", []):
        out.append(sentence(f"The label {lab['text']} is written at the point {point(lab['at'])} on the ground"))
    return out


def _describe_grid_drawing(spec: dict) -> str:
    out = _describe_cubes(spec) if spec["grid"]["type"] == "isometric" else _describe_square(spec)
    scale = spec.get("scale")
    if scale:
        out.append(sentence(f"Scale: {scale['text']}"))
        if "bar_cells" in scale:
            out.append(sentence(f"A scale bar {count(scale['bar_cells'], 'square')} long is drawn below the figure"))
    if spec.get("north"):
        out.append("A north arrow points up the grid.")
    canvas = spec.get("canvas")
    if canvas == "grid":
        out.append("Your answer is drawn on this same grid; any squares already marked are given.")
    elif canvas:
        side = "below" if canvas_below(spec) else "beside"
        out.append(
            sentence(
                f"A blank grid {count(canvas['width'], 'square')} by {count(canvas['height'], 'square')} "
                f"is {side} the figure for your answer"
            )
        )
    return " ".join(out)


register_describer("grid_drawing", _describe_grid_drawing)
