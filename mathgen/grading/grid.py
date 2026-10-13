"""The ``grid_cells`` answer kind: the cells a learner shades on a square grid.

The learner shades whole cells on the figure's canvas (`mathgen.figure_kinds.grid`), so every cell is exact and
the grader compares sets of cells with no tolerance. A cell is ``[x, y]``: column from the left and row from the
bottom, both from 0.

`Question.answer` is ``{"cells": [[x, y], ...], "grid": [width, height], "match": mode}``:

* ``exact``: the same set of cells (a plan view, an elevation, an image under a stated transformation).
* ``translate``: the same set of cells anywhere on the canvas (a shape the learner may place where they like).
* ``congruent``: the same shape turned or flipped anywhere on the canvas (a scale drawing whose orientation the
  question leaves open).
* ``cube_net``: any six cells that fold into a cube; ``cells`` is one such net, shown as the worked answer.

A reply is canonical JSON from a client, ``{"cells": [[0, 0], [1, 0]]}`` or a bare ``[[0, 0], [1, 0]]``, or the
typed form ``(0, 0) (1, 0)``. The JSON is read with `json.loads` and nothing else: no reply is ever evaluated.
A cell listed twice counts once. Any cell off the canvas, a coordinate that is not a whole number, more than
:data:`MAX_CELLS` cells, or anything else in the text makes the reply wrong.
"""

from __future__ import annotations

import json
import re

from .typed import GradeResult, _multi_part_aware

#: Most cells a reply may hold. The largest canvas is 20 by 20.
MAX_CELLS = 400
#: Longest reply read at all: 400 cells as JSON is under 5,000 characters.
MAX_GRID_REPLY_LENGTH = 8_000

MATCH_MODES = frozenset({"exact", "translate", "congruent", "cube_net"})

Cell = tuple[int, int]

_CELL_RE = re.compile(r"\(\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*\)")
_FILLER_RE = re.compile(r"^(?:[\s,;]|\band\b)*$", re.IGNORECASE)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _cells(values) -> set[Cell] | None:
    if isinstance(values, dict):
        if set(values) != {"cells"}:
            return None
        values = values["cells"]
    if not isinstance(values, list) or len(values) > MAX_CELLS:
        return None
    out: set[Cell] = set()
    for v in values:
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_int(c) for c in v)):
            return None
        out.add((v[0], v[1]))
    return out


def _given_cells(given) -> set[Cell] | None:
    if not isinstance(given, str):
        return None
    text = given.strip()
    if not text or len(text) > MAX_GRID_REPLY_LENGTH:
        return None
    if text[0] in "[{":
        try:
            return _cells(json.loads(text))
        except (ValueError, RecursionError):
            return None
    found = [(int(m.group(1)), int(m.group(2))) for m in _CELL_RE.finditer(text)]
    if not found or len(found) > MAX_CELLS or not _FILLER_RE.fullmatch(_CELL_RE.sub(" ", text)):
        return None
    return set(found)


def cell_text(cells) -> str:
    """Cells as typed, sorted by row then column: ``(0, 0) (1, 0) (0, 1)``."""
    return " ".join(f"({x}, {y})" for x, y in sorted({tuple(c) for c in cells}, key=lambda c: (c[1], c[0])))


def cells_answer(cells, grid, match: str = "exact") -> dict:
    """`Question.answer` for a grid_cells question. Generators call this."""
    if match not in MATCH_MODES:
        raise ValueError(f"unknown match mode {match!r}")
    width, height = grid
    cells = sorted({(int(x), int(y)) for x, y in cells}, key=lambda c: (c[1], c[0]))
    if not cells or any(not (0 <= x < width and 0 <= y < height) for x, y in cells):
        raise ValueError("every answer cell must be on the grid")
    return {"cells": [list(c) for c in cells], "grid": [width, height], "match": match}


# ---------------------------------------------------------------- shape comparison


def normalised(cells) -> frozenset[Cell]:
    """The cells moved so the lowest row and the leftmost column are both 0."""
    cells = list(cells)
    if not cells:
        return frozenset()
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return frozenset((x - min_x, y - min_y) for x, y in cells)


def orientations(cells) -> set[frozenset[Cell]]:
    """The shape in each of its eight turns and flips, each normalised."""
    out = set()
    shape = list(cells)
    for _ in range(4):
        shape = [(y, -x) for x, y in shape]
        out.add(normalised(shape))
        out.add(normalised([(-x, y) for x, y in shape]))
    return out


def congruent(a, b) -> bool:
    """True when the two sets of cells are the same shape, allowing turns, flips and moves."""
    return normalised(b) in orientations(a)


def connected(cells) -> bool:
    """True when every cell can be reached from every other through shared edges."""
    cells = set(cells)
    if not cells:
        return False
    start = next(iter(cells))
    seen, todo = {start}, [start]
    while todo:
        x, y = todo.pop()
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n in cells and n not in seen:
                seen.add(n)
                todo.append(n)
    return len(seen) == len(cells)


# A die rolled one cell east, west, north or south: which face ends up where. A face is named by where it
# points: bottom, top, east, west, north, south.
_ROLLS = {
    (1, 0): {"bottom": "east", "east": "top", "top": "west", "west": "bottom"},
    (-1, 0): {"bottom": "west", "west": "top", "top": "east", "east": "bottom"},
    (0, 1): {"bottom": "north", "north": "top", "top": "south", "south": "bottom"},
    (0, -1): {"bottom": "south", "south": "top", "top": "north", "north": "bottom"},
}


def is_cube_net(cells) -> bool:
    """True when the six cells fold into a cube.

    A cube is rolled across the net from one cell, so each cell is the face that lands on it. The cells fold
    into a cube exactly when they are six, joined by edges, with no two by two block (four squares round a
    corner cannot fold), and the roll lands a different face on each.
    """
    cells = set(cells)
    if len(cells) != 6 or not connected(cells):
        return False
    if any({(x + 1, y), (x, y + 1), (x + 1, y + 1)} <= cells for x, y in cells):
        return False
    start = min(cells)
    # orientation: which original face now points each way
    faces = {d: d for d in ("bottom", "top", "east", "west", "north", "south")}
    landed = {start: faces["bottom"]}
    todo = [(start, faces)]
    while todo:
        (x, y), here = todo.pop()
        for (dx, dy), roll in _ROLLS.items():
            n = (x + dx, y + dy)
            if n in cells and n not in landed:
                moved = {direction: here[roll.get(direction, direction)] for direction in here}
                landed[n] = moved["bottom"]
                todo.append((n, moved))
    return len(set(landed.values())) == 6


# ---------------------------------------------------------------- grader


def _matches(mode: str, expected: set[Cell], given: set[Cell]) -> bool:
    if mode == "exact":
        return given == expected
    if mode == "translate":
        return normalised(given) == normalised(expected)
    if mode == "congruent":
        return len(given) == len(expected) and congruent(expected, given)
    if mode == "cube_net":
        return is_cube_net(given)
    return False


@_multi_part_aware
def grade_grid_cells(question, given: str) -> GradeResult:
    answer = question.answer
    canonical = question.answer_text
    if not isinstance(answer, dict) or answer.get("match", "exact") not in MATCH_MODES:
        return GradeResult(correct=False, canonical=canonical)
    expected = _cells(answer.get("cells"))
    grid = answer.get("grid")
    if not expected or not (isinstance(grid, (list, tuple)) and len(grid) == 2 and all(_is_int(v) for v in grid)):
        return GradeResult(correct=False, canonical=canonical)
    cells = _given_cells(given)
    if cells is None:
        return GradeResult(correct=False, canonical=canonical, given_normalised=None)
    width, height = grid
    on_grid = bool(cells) and all(0 <= x < width and 0 <= y < height for x, y in cells)
    correct = on_grid and _matches(answer.get("match", "exact"), expected, cells)
    return GradeResult(correct=correct, canonical=canonical, given_normalised=cell_text(cells))
