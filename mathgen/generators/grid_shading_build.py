"""Questions where the learner shades cells on a grid, and one enlargement question read off a grid.

Five topics, each built from the ``grid_drawing`` figure kind (`mathgen.figure_kinds.grid`):

``grid_elevation_views`` (``answer: grid_cells``, match ``exact``)
    A stack of unit cubes on isometric dot paper, and a blank canvas beside it exactly as wide and as tall as the
    view being drawn, so where each cell goes is pinned: the bottom-left cell of the canvas is the bottom-left of
    the view. Easy draws the front view, medium the right or left view, hard the back view, or, reversed, is given
    one view on a grid and draws the view from the opposite side. Every stack is *readable*: no column of cubes
    hides behind a taller one, so the picture fixes every cube.

``grid_plan_views`` (``answer: grid_cells``)
    Easy draws the plan (top) view of a cube stack (match ``exact``, front of the stack at the bottom of the canvas).
    Medium is given a plan with the number of cubes in each column written on it and draws the front, right or
    left view (``exact``). Hard draws a net of a cube on a canvas (match ``cube_net``): any six cells that fold
    into a cube are right, and a tight canvas leaves only a few nets that fit.

``grid_scale_plan`` (``answer: grid_cells``)
    A room or garden is described in words with real dimensions, and the scale is shown on the grid. Every cell of
    the grid is 1 cm of the drawing. Easy is a rectangle and medium an L shape, each placed anywhere but not turned
    (match ``translate``); hard converts a scale such as ``1 : 200`` or ``1 cm : 50 cm`` and leaves the orientation
    open (match ``congruent``). The wording says which freedom the learner has.

``grid_tessellation_images`` (``answer: grid_cells``, match ``exact``)
    A tile is drawn on the grid; the learner shades its image after one transformation (easy), after a combination
    of two (medium), or the next two tiles of a tessellation (hard). The squares already drawn are given and are
    not part of the reply, and the wording says so.

``grid_enlargement_image`` (a multi-part question for ``f10-enlargement-invariants``)
    A lattice triangle or rectangle is drawn on a square grid with a labelled centre of enlargement and a scale
    factor of 2, 3 or 1/2. Part (a) is a ``number``: the perimeter of the image or the length of one image side.
    Part (b) is ``points``: the image's vertices in order.

Each grid question registers a figure reader that re-derives the answer from the figure and the wording alone.
"""

from __future__ import annotations

import functools
import math
import re
from fractions import Fraction

from .. import figures
from ..core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, phrase, pick_distractors
from ..figure_kinds.grid import elevation, grid_drawing, plan_view, tops_all_visible
from ..grading.grid import cell_text, cells_answer, congruent, is_cube_net, normalised
from ..grading.typed import points_answer_text
from . import _space_kit as kit

Cell = tuple[int, int]

# ------------------------------------------------------------------------------------------- shared helpers


def _canvas(width: int, height: int) -> dict:
    return {"width": width, "height": height}


def _is_wrong(match: str, expected, given) -> bool:
    """Would a reply of exactly the cells ``given`` be marked wrong against ``expected`` under ``match``?"""
    if match == "exact":
        return set(given) != set(expected)
    if match == "translate":
        return normalised(given) != normalised(expected)
    if match == "congruent":
        return len(given) != len(expected) or not congruent(expected, given)
    if match == "cube_net":
        return not is_cube_net(given)
    raise ValueError(f"unknown match mode {match!r}")


def _on_grid(cells, size) -> bool:
    w, h = size
    return bool(cells) and all(0 <= x < w and 0 <= y < h for x, y in cells)


def _distractor_texts(cells, size, match, candidates) -> list[str]:
    """Cell texts of the candidate drawings that are on the canvas and wrong under ``match``, three of them."""
    expected = {tuple(c) for c in cells}
    texts = []
    for candidate in candidates:
        got = {tuple(c) for c in candidate}
        if _on_grid(got, size) and _is_wrong(match, expected, got):
            texts.append(cell_text(got))
    return pick_distractors(cell_text(expected), texts)


def _cells_question(fig, text, cells, size, match, candidates, working) -> dict | None:
    """The dict a ``grid_cells`` generator returns, or None when fewer than three distractors survive."""
    answer_text = cell_text(cells)
    if figures.validate(fig):
        raise ValueError(f"invalid figure: {figures.validate(fig)}")
    distractors = _distractor_texts(cells, size, match, candidates)
    if len(distractors) < 3 or kit.leaks(fig, answer_text):
        return None
    return dict(
        text=text,
        answer=cells_answer(cells, size, match),
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        answer_kind="grid_cells",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _retry(rng, build, what: str) -> dict:
    """Call ``build(rng)`` until it returns a question dict (it returns None to reject a draw)."""
    for _ in attempts(300, what):
        out = build(rng)
        if out is not None:
            return out
    raise AssertionError("unreachable")


def _plural(n, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _column_heights(cells, width: int) -> list[int]:
    return [sum(1 for c, _ in cells if c == col) for col in range(width)]


def _mirror(cells, width: int) -> set[Cell]:
    return {(width - 1 - c, z) for c, z in cells}


# ------------------------------------------------------------------------------------------- cube stacks

_SIDES = ("front", "right", "left", "back")
_OPPOSITE = {"front": "back", "back": "front", "left": "right", "right": "left"}
_STANDING = {
    "front": "standing in front of the stack",
    "back": "standing behind the stack",
    "right": "standing at the right-hand side of the stack",
    "left": "standing at the left-hand side of the stack",
}


def _stack(rng, w: int, d: int, top: int, *, holes: bool):
    """Column heights ``{(x, y): height}`` and the cubes, for a stack whose picture fixes every cube.

    The picture is seen from the front right, above, so a column hides behind any taller column in front of it
    or to its right. Heights that never rise towards the front or the right (and the empty spots with them) leave
    nothing hidden. The back-left column is the tallest, a column stands at the front-left and another at the
    back-right, so every elevation fills the whole canvas, which is exactly as wide and as tall as the view.
    """
    for _ in attempts(400, "a readable cube stack"):
        h: dict[Cell, int] = {}
        for y in range(d - 1, -1, -1):
            for x in range(w):
                if x == 0 and y == d - 1:
                    h[(x, y)] = top
                    continue
                cap = min(h[(x - 1, y)] if x else top, h[(x, y + 1)] if y < d - 1 else top)
                h[(x, y)] = rng.randint(0 if holes and rng.random() < 0.4 else 1, cap) if cap else 0
        if h[(w - 1, d - 1)] < 1 or h[(0, 0)] < 1:
            continue
        has_hole = any(v == 0 for v in h.values())
        if has_hole != holes:
            continue
        cubes = [[x, y, z] for (x, y), v in sorted(h.items()) for z in range(v)]
        if not 5 <= len(cubes) <= 24 or not tops_all_visible(cubes):
            continue
        return h, cubes
    raise AssertionError("unreachable")


def _view(cubes, side: str, w: int, d: int) -> list[Cell]:
    return [tuple(c) for c in elevation(cubes, side, depth=w if side == "back" else d)]


def _view_size(side: str, w: int, d: int, top: int) -> tuple[int, int]:
    return (w if side in ("front", "back") else d), top


def _view_candidates(rng, cells, size, others=()):
    """Real wrong drawings of a view: mirrored, another side's view, a cell missing, a cell too many."""
    w, top = size
    cells = sorted(set(cells))
    out = [_mirror(cells, w)]
    out.extend(others)
    columns = sorted({c for c, _ in cells})
    col = rng.choice(columns)
    out.append(set(cells) - {(col, max(z for c, z in cells if c == col))})
    room = [c for c in columns if max(z for cc, z in cells if cc == c) + 1 < top]
    if room:
        col = rng.choice(room)
        out.append(set(cells) | {(col, max(z for cc, z in cells if cc == col) + 1)})
    return out


def _pin(rng, w: int, h: int, what: str = "view") -> str:
    return phrase(
        rng,
        (
            "The canvas is {w} cells wide and {h} cells tall, the same size as the {what}, and its bottom-left cell "
            "is the bottom-left of the {what}.",
            "Use the canvas: it is exactly {w} cells wide and {h} cells tall, so its bottom-left cell is the "
            "bottom-left of the {what}.",
            "The canvas matches the {what} in size ({w} cells wide, {h} cells tall), with its bottom-left cell at "
            "the bottom-left of the {what}.",
        ),
        w=w,
        h=h,
        what=what,
    )


_ISO_ALT = "A stack of unit cubes on isometric dot paper, with a blank grid beside it to draw on."


def _iso_figure(w, d, top, cubes, canvas):
    return grid_drawing(
        alt=_ISO_ALT,
        grid={"type": "isometric", "width": w, "height": d, "layers": top},
        cubes=cubes,
        canvas=canvas,
    )


def _stack_dims(rng, tier: str):
    w = rng.randint(3, 5) if tier != "easy" else rng.randint(3, 4)
    d = rng.randint(2, 4)
    top = rng.randint(2, 4)
    return w, d, top


def _elevation_easy(rng):
    w, d, top = _stack_dims(rng, "easy")
    _h, cubes = _stack(rng, w, d, top, holes=rng.random() < 0.5)
    side = "front"
    cells = _view(cubes, side, w, d)
    size = _view_size(side, w, d, top)
    fig = _iso_figure(w, d, top, cubes, _canvas(*size))
    text = phrase(
        rng,
        (
            "The picture shows a stack of unit cubes on isometric dot paper, seen from the front right, above. "
            "Shade the {side} view of the stack on the canvas, one cell for each square face you would see. {pin}",
            "A stack of unit cubes is drawn on isometric dot paper, seen from the front right, above. Draw what you "
            "would see {standing}: shade the {side} view on the canvas. {pin}",
            "Look at this stack of unit cubes, drawn from the front right, above. On the canvas, shade the {side} "
            "view, the faces you see {standing}. {pin}",
        ),
        side=side,
        standing=_STANDING[side],
        pin=_pin(rng, *size),
    )
    others = [_view(cubes, s, w, d) for s in ("right", "left")]
    working = _view_working(cells, side, size)
    return _cells_question(fig, text, cells, size, "exact", _view_candidates(rng, cells, size, others), working)


def _view_working(cells, side: str, size) -> list[str]:
    heights = _column_heights(cells, size[0])
    order = {
        "front": "left to right",
        "back": "left to right as seen from behind",
        "right": "from the front of the stack on your left to the back on your right",
        "left": "from the back of the stack on your left to the front on your right",
    }[side]
    return [
        f"Standing at the {side}, you see one column for each position, taken {order}.",
        f"Each column is as tall as the tallest stack of cubes in line with it. In cubes, the heights are {', '.join(map(str, heights))}.",
        "Shade that many cells in each column of the canvas, starting from the bottom row.",
    ]


def _elevation_medium(rng):
    w, d, top = _stack_dims(rng, "medium")
    _h, cubes = _stack(rng, w, d, top, holes=rng.random() < 0.5)
    side = rng.choice(("right", "left"))
    cells = _view(cubes, side, w, d)
    size = _view_size(side, w, d, top)
    fig = _iso_figure(w, d, top, cubes, _canvas(*size))
    text = phrase(
        rng,
        (
            "The picture shows a stack of unit cubes on isometric dot paper, seen from the front right, above. "
            "Shade the {side} view of the stack on the canvas: imagine {standing} and shade what you see. "
            "Left and right are as the viewer sees them. {pin}",
            "A stack of unit cubes is drawn on isometric dot paper, seen from the front right, above. On the "
            "canvas, shade the {side} view of the stack as someone {standing} would see it, with their own left "
            "on the left of the view. {pin}",
            "This stack of unit cubes is drawn from the front right, above. Draw the {side} view on the canvas, "
            "as seen {standing}, one cell for each square face. {pin}",
        ),
        side=side,
        standing=_STANDING[side],
        pin=_pin(rng, *size),
    )
    other = "left" if side == "right" else "right"
    others = [_view(cubes, other, w, d), _view(cubes, "front", w, d), _view(cubes, "back", w, d)]
    return _cells_question(
        fig, text, cells, size, "exact", _view_candidates(rng, cells, size, others), _view_working(cells, side, size)
    )


def _elevation_back(rng):
    w, d, top = _stack_dims(rng, "hard")
    _h, cubes = _stack(rng, w, d, top, holes=rng.random() < 0.5)
    cells = _view(cubes, "back", w, d)
    size = _view_size("back", w, d, top)
    if set(cells) == _mirror(cells, w):
        return None
    fig = _iso_figure(w, d, top, cubes, _canvas(*size))
    text = phrase(
        rng,
        (
            "The picture shows a stack of unit cubes on isometric dot paper, seen from the front right, above. "
            "Shade the back view of the stack on the canvas: the faces you would see {standing}, with your own "
            "left on the left of the view. {pin}",
            "A stack of unit cubes is drawn on isometric dot paper, seen from the front right, above. Walk round "
            "to the back of the stack and shade the back view on the canvas, one cell for each square face. {pin}",
            "This stack of unit cubes is drawn from the front right, above. On the canvas, shade the back view as "
            "seen {standing}. {pin}",
        ),
        standing=_STANDING["back"],
        pin=_pin(rng, *size),
    )
    others = [_view(cubes, "front", w, d), _view(cubes, "left", w, d), _view(cubes, "right", w, d)]
    return _cells_question(
        fig, text, cells, size, "exact", _view_candidates(rng, cells, size, others), _view_working(cells, "back", size)
    )


_REVERSE_ALT = "A view of a stack of unit cubes drawn as shaded squares on a grid, with a blank canvas beside it."


def _elevation_reverse(rng):
    """A view is given on a grid; the view from the opposite side is its mirror image."""
    width = rng.randint(3, 6)
    top = rng.randint(2, 5)
    heights = [rng.randint(1, top) for _ in range(width)]
    if max(heights) != top or heights == heights[::-1]:
        return None
    given_side = rng.choice(_SIDES)
    asked = _OPPOSITE[given_side]
    given = sorted((c, z) for c, v in enumerate(heights) for z in range(v))
    cells = sorted(_mirror(given, width))
    size = (width, top)
    fig = grid_drawing(
        alt=_REVERSE_ALT,
        grid={"type": "square", "width": width, "height": top},
        cells=[{"cells": [list(c) for c in given], "style": "shaded"}],
        canvas=_canvas(*size),
    )
    text = phrase(
        rng,
        (
            "The grid shows the {given} view of a stack of unit cubes. Shade the view of the same stack from the "
            "opposite side, the {asked} view, on the canvas. {pin}",
            "This is the {given} view of a stack of unit cubes. On the canvas, shade what you would see of the "
            "same stack from the other side: the {asked} view. {pin}",
            "A stack of unit cubes looks like the grid when seen from the {given}. Walk round to the opposite side "
            "and shade the {asked} view on the canvas. {pin}",
        ),
        given=given_side,
        asked=asked,
        pin=_pin(rng, *size),
    )
    candidates = [
        set(given),  # draws the view it was given
        set(cells) - {(max(c for c, _ in cells), 0)},
        {(c, top - 1 - z) for c, z in cells},  # upside down
    ]
    extra = [c for c, v in enumerate(heights[::-1]) if v < top]
    if extra:
        c = rng.choice(extra)
        candidates.append(set(cells) | {(c, heights[::-1][c])})
    working = [
        "Looking at a stack from opposite sides shows the same columns in the opposite order.",
        f"The {given_side} view has column heights {', '.join(map(str, heights))}; from the {asked} the order "
        f"reverses: {', '.join(map(str, heights[::-1]))}.",
    ]
    return _cells_question(fig, text, cells, size, "exact", candidates, working)


@generator(
    "grid_elevation_views",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_elevation_views(rng, tier="easy"):
    """Shade an elevation (front, side or back view) of a cube stack on a canvas the size of the view."""
    if tier == "easy":
        return _retry(rng, _elevation_easy, "an easy elevation")
    if tier == "medium":
        return _retry(rng, _elevation_medium, "a medium elevation")
    if tier == "hard":
        build = _elevation_back if rng.random() < 0.5 else _elevation_reverse
        return _retry(rng, build, "a hard elevation")
    raise ValueError(f"unknown tier {tier!r}")


_VIEW_WORDS = re.compile(r"the (front|back|left|right) view")


def _read_elevation(question) -> str:
    """The view, from the cubes (or from the view given) and the side the wording asks for."""
    fig = question.figure
    side = _VIEW_WORDS.findall(question.text)[-1]
    grid = fig["grid"]
    if grid["type"] == "isometric":
        cubes = fig["cubes"]
        return cell_text(_view(cubes, side, grid["width"], grid["height"]))
    given = [tuple(c) for group in fig["cells"] for c in group["cells"]]
    return cell_text(_mirror(given, grid["width"]))


figures.register_reader("grid_elevation_views", _read_elevation)

# ------------------------------------------------------------------------------------------- plans and nets


def _plan_easy(rng):
    w, d, top = _stack_dims(rng, "easy")
    _h, cubes = _stack(rng, w, d, top, holes=True)
    cells = [tuple(c) for c in plan_view(cubes)]
    size = (w, d)
    fig = _iso_figure(w, d, top, cubes, _canvas(*size))
    text = phrase(
        rng,
        (
            "The picture shows a stack of unit cubes on isometric dot paper, seen from the front right, above. "
            "Shade the plan view of the stack on the canvas: the view from directly above, one cell for each "
            "column of cubes. The front of the stack is the bottom of the canvas, and the canvas is {w} cells wide "
            "and {d} cells tall, so its bottom-left cell is the front left corner of the stack.",
            "A stack of unit cubes is drawn on isometric dot paper, seen from the front right, above. Draw the "
            "plan view on the canvas, the stack as seen from above. The bottom of the canvas is the front of the "
            "stack and the canvas, {w} cells wide and {d} cells tall, has its bottom-left cell at the front left "
            "corner.",
            "Imagine looking straight down on this stack of unit cubes, drawn from the front right, above. Shade "
            "its plan view on the canvas, which is {w} cells wide and {d} cells tall. Put the front of the stack "
            "at the bottom of the canvas, so the bottom-left cell is the front left corner.",
        ),
        w=w,
        d=d,
    )
    candidates = [
        {(x, d - 1 - y) for x, y in cells},  # back and front swapped
        {(w - 1 - x, y) for x, y in cells},  # left and right swapped
        {(x, y) for x in range(w) for y in range(d)},  # the whole footprint, empty spots included
        set(cells) - {rng.choice(sorted(cells))},
        set(_view(cubes, "front", w, d)),  # the front view instead
    ]
    spots = [(x, y) for x in range(w) for y in range(d) if (x, y) not in set(cells)]
    candidates.append(set(cells) | {rng.choice(spots)})
    working = [
        "From above, every column of cubes shows as one square, however tall it is.",
        f"The plan has {len(cells)} squares: a square wherever a column stands, and none where the ground is bare.",
    ]
    return _cells_question(fig, text, cells, size, "exact", candidates, working)


_PLAN_ALT = (
    "A plan of a stack of cubes on a square grid: each shaded square has the number of cubes in its column written "
    "on it, and a blank canvas stands beside it."
)


def _heights_plan(rng):
    w = rng.randint(3, 5)
    d = rng.randint(2, 4)
    h = {(x, y): (0 if rng.random() < 0.25 else rng.randint(1, 4)) for x in range(w) for y in range(d)}
    if any(all(h[(x, y)] == 0 for y in range(d)) for x in range(w)):
        return None
    if any(all(h[(x, y)] == 0 for x in range(w)) for y in range(d)):
        return None
    return w, d, h


def _plan_heights(rng):
    drawn = _heights_plan(rng)
    if drawn is None:
        return None
    w, d, h = drawn
    top = max(h.values())
    cubes = [[x, y, z] for (x, y), v in sorted(h.items()) for z in range(v)]
    side = rng.choice(("front", "right", "left"))
    cells = _view(cubes, side, w, d)
    size = _view_size(side, w, d, top)
    footprint = [[x, y] for (x, y), v in sorted(h.items()) if v]
    fig = grid_drawing(
        alt=_PLAN_ALT,
        grid={"type": "square", "width": w, "height": d},
        cells=[{"cells": footprint, "style": "shaded"}],
        labels=[{"at": [x + 0.5, y + 0.5], "text": str(v)} for (x, y), v in sorted(h.items()) if v],
        canvas=_canvas(*size),
    )
    text = phrase(
        rng,
        (
            "The plan of a stack of unit cubes is shown: the number in each square is how many cubes are stacked "
            "there, and the bottom of the plan is the front of the stack. Shade the {side} view on the canvas. "
            "Left and right are as the viewer sees them. {pin}",
            "From above, a stack of cubes looks like the plan shown. The number in a square is the height of that "
            "column in cubes, and the front of the stack is at the bottom of the plan. On the canvas, shade the "
            "{side} view, with the viewer's own left on the left of the view. {pin}",
            "This plan shows how many cubes are in each column of a stack, seen from above with the front at the "
            "bottom of the plan. Show the {side} view of the stack on the canvas, as seen {standing}. {pin}",
        ),
        side=side,
        standing=_STANDING[side],
        pin=_pin(rng, *size),
    )
    nearest = {"front": lambda x, y: y == 0, "right": lambda x, y: x == w - 1, "left": lambda x, y: x == 0}[side]
    near_cubes = [c for c in cubes if nearest(c[0], c[1])]
    near_cells = {({"front": c[0], "right": c[1], "left": d - 1 - c[1]}[side], c[2]) for c in near_cubes}
    sums = {}
    for x, y, _z in cubes:
        key = x if side == "front" else y
        sums[key] = sums.get(key, 0) + 1
    summed = {(c if side != "left" else d - 1 - c, z) for c, n in sums.items() for z in range(n)}
    across = "right" if side == "front" else "front"  # the view from the other pair of sides
    column = rng.choice(sorted({c for c, _ in cells}))
    candidates = [
        _mirror(cells, size[0]),
        set(_view(cubes, across, w, d)),
        near_cells,  # reads only the row of the plan nearest the viewer
        summed,  # adds the cubes in a line instead of taking the tallest
        set(cells) - {(column, max(z for c, z in cells if c == column))},  # a cell missing
    ]
    heights = _column_heights(cells, size[0])
    working = [
        f"Standing at the {side}, each column of the view is as tall as the tallest stack in line with it.",
        f"The column heights, in order from the viewer's left, are {', '.join(map(str, heights))}.",
    ]
    return _cells_question(fig, text, cells, size, "exact", candidates, working)


_NET_ALT = "A single cube drawn on isometric dot paper, with a blank grid beside it to draw on."
_NET_CANVASES = [(6, 6), (6, 6), (6, 6), (5, 3), (4, 4), (4, 3), (3, 4), (5, 2), (2, 5), (6, 4), (4, 6), (5, 5)]


@functools.lru_cache(maxsize=1)
def _net_shapes() -> tuple[frozenset, ...]:
    """Every shape of six squares (fixed in place) that folds into a cube, normalised to the corner."""
    shapes = {frozenset({(0, 0)})}
    for _ in range(5):
        grown = set()
        for shape in shapes:
            for x, y in shape:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (x + dx, y + dy) not in shape:
                        grown.add(normalised(shape | {(x + dx, y + dy)}))
        shapes = grown
    return tuple(sorted((s for s in shapes if is_cube_net(s)), key=sorted))


def _nets_on(width: int, height: int) -> list[frozenset]:
    """Every placement of a cube net on a ``width`` by ``height`` canvas."""
    out = []
    for shape in _net_shapes():
        sw = max(x for x, _ in shape) + 1
        sh = max(y for _, y in shape) + 1
        for ox in range(width - sw + 1):
            for oy in range(height - sh + 1):
                out.append(frozenset((x + ox, y + oy) for x, y in shape))
    return out


def _plan_net(rng):
    width, height = rng.choice(_NET_CANVASES)
    nets = _nets_on(width, height)
    check(bool(nets), f"no cube net fits a {width} by {height} canvas")
    net = rng.choice(nets)
    fig = grid_drawing(
        alt=_NET_ALT,
        grid={"type": "isometric", "width": 1, "height": 1, "layers": 1},
        cubes=[[0, 0, 0]],
        canvas=_canvas(width, height),
    )
    text = phrase(
        rng,
        (
            "The picture shows a cube. Shade a net of this cube on the canvas, which is {w} cells wide and {h} cells "
            "tall. A net is six squares joined edge to edge that fold up into the cube. It can be turned or flipped "
            "and put anywhere on the canvas.",
            "Draw a net for the cube shown, using one cell of the canvas ({w} cells wide, {h} cells tall) for each "
            "face. The six squares must be joined along whole edges and fold into the cube with no gaps or "
            "overlaps. Any turn, flip or position on the canvas will do.",
            "A cube is shown. On the canvas, {w} cells wide and {h} cells tall, shade six cells that make a net of "
            "the cube. The net must fold up to form the cube. You may place it anywhere on the canvas, turned or "
            "flipped.",
        ),
        w=width,
        h=height,
    )
    size = (width, height)
    occupied = set(net)
    moves = []
    for cell in sorted(net):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            spot = (cell[0] + dx, cell[1] + dy)
            if spot not in occupied and 0 <= spot[0] < width and 0 <= spot[1] < height:
                moves.append((net - {cell}) | {spot})
    rng.shuffle(moves)
    near = [m for m in moves if not is_cube_net(m)]
    five = [net - {rng.choice(sorted(net))}]
    free = [(x, y) for x in range(width) for y in range(height) if (x, y) not in occupied]
    seven = [net | {rng.choice(free)}] if free else []
    candidates = near[:1] + five + near[1:2] + seven + near[2:]
    working = [
        "A net folds into a cube only when no two faces land on the same side of the cube.",
        "Four squares in a row with one square above and one below the row, anywhere along it, always fold into "
        "a cube; a two by two block never does.",
    ]
    return _cells_question(fig, text, sorted(net), size, "cube_net", candidates, working)


def _read_net(question) -> str:
    cv = question.figure["canvas"]
    return cell_text(sorted(_nets_on(cv["width"], cv["height"])[0]))


@generator(
    "grid_plan_views",
    level=6,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_plan_views(rng, tier="easy"):
    """Represent a three-dimensional object on a canvas: its plan, an elevation from a plan, or a net of a cube."""
    if tier == "easy":
        return _retry(rng, _plan_easy, "an easy plan view")
    if tier == "medium":
        return _retry(rng, _plan_heights, "a plan with heights")
    if tier == "hard":
        return _retry(rng, _plan_net, "a net of a cube")
    raise ValueError(f"unknown tier {tier!r}")


def _read_plan(question) -> str:
    fig, text = question.figure, question.text
    grid = fig["grid"]
    if grid["type"] == "isometric":
        if len(fig["cubes"]) == 1:
            return _read_net(question)
        return cell_text(plan_view(fig["cubes"]))
    side = _VIEW_WORDS.findall(text)[-1]
    cubes = []
    for label in fig["labels"]:
        x, y = int(label["at"][0]), int(label["at"][1])
        cubes.extend([x, y, z] for z in range(int(label["text"])))
    return cell_text(_view(cubes, side, grid["width"], grid["height"]))


figures.register_reader("grid_plan_views", _read_plan)

# ------------------------------------------------------------------------------------------- scale drawings

_SPACES = ("garden bed", "courtyard", "deck", "classroom", "workshop", "vegetable patch", "swimming pool", "lawn")
_L_SPACES = ("patio", "garden", "courtyard", "living area", "deck", "lawn")
_CORNERS = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))

# (centimetres of real length for 1 cm of drawing, the scale as written on the grid, whole metres per cell step)
_SCALES_EASY = ((100, "1 cm : 1 m"), (200, "1 cm : 2 m"), (500, "1 cm : 5 m"))
_SCALES_HARD = ((50, "1 cm : 50 cm"), (100, "1 : 100"), (200, "1 : 200"), (500, "1 : 500"))

_SCALE_WORDS = re.compile(r"^1 cm : (\d+) (cm|m)$|^1 : (\d+)$")


def _scale_cm(text: str) -> int:
    """Centimetres of real length for 1 cm of drawing, from the scale as written."""
    m = _SCALE_WORDS.match(text)
    check(m is not None, f"cannot read the scale {text!r}")
    if m.group(3):
        return int(m.group(3))
    return int(m.group(1)) * (100 if m.group(2) == "m" else 1)


def _metres(cells: int, per_cm: int) -> Fraction:
    return Fraction(cells * per_cm, 100)


def _metre_text(value: Fraction) -> str:
    return str(int(value)) if value.denominator == 1 else f"{float(value):g}"


def _shape_cells(kind: str, dims, origin, notch=None, corner=None) -> set[Cell]:
    """A rectangle ``dims`` cells across and up, or an L made by cutting ``notch`` out of a ``corner``."""
    (a, b), (ox, oy) = dims, origin
    cells = {(ox + x, oy + y) for x in range(a) for y in range(b)}
    if kind == "L":
        c, d = notch
        v, h = corner
        xs = range(a - c, a) if h == "right" else range(c)
        ys = range(b - d, b) if v == "top" else range(d)
        cells -= {(ox + x, oy + y) for x in xs for y in ys}
    return cells


def _across_up(a: Fraction, b: Fraction) -> str:
    return f"{_metre_text(a)} m across the page and {_metre_text(b)} m up the page"


def _scale_question(rng, tier: str):
    easy, medium = tier == "easy", tier == "medium"
    per_cm, scale_text = rng.choice(_SCALES_EASY if easy or medium else _SCALES_HARD)
    kind = "rect" if easy else ("L" if medium else rng.choice(("L", "L", "rect")))
    step = Fraction(per_cm, 100)  # metres of real length for each cell
    a, b = rng.randint(3, 8), rng.randint(3, 6)
    if kind == "L":
        a, b = rng.randint(4, 8), rng.randint(4, 6)
        c, d = rng.randint(1, a - 2), rng.randint(1, b - 2)
        corner = rng.choice(_CORNERS)
        notch = (c, d)
    else:
        notch = corner = None
        if tier == "hard" and a == b:
            return None
    gw, gh = rng.randint(11, 14), rng.randint(9, 11)
    origin = (rng.randint(0, gw - a), rng.randint(0, gh - b))
    cells = sorted(_shape_cells(kind, (a, b), origin, notch, corner))
    match = "congruent" if tier == "hard" and (kind == "L" or rng.random() < 0.7) else "translate"
    thing = rng.choice(_SPACES if kind == "rect" else _L_SPACES)
    metres = _across_up(a * step, b * step)
    if kind == "rect":
        shape = f"rectangular {thing}"
        body = f"The {thing} is {metres}."
    else:
        shape = f"L-shaped {thing}"
        cut = _across_up(c * step, d * step)
        body = f"The {thing} is {metres} at its widest and tallest. A rectangle {cut} is missing from the {corner[0]} {corner[1]} corner."
    freedom = (
        "You may place the drawing anywhere on the grid and turn or flip it, so its sides need not run the way "
        "they are described."
        if match == "congruent"
        else "You may place the drawing anywhere on the grid, but keep it the same way round as described."
    )
    article = "an" if shape[0] in "aeiouL" else "a"
    text = phrase(
        rng,
        (
            "Draw a scale plan of {article} {shape} on the grid by shading cells, using the scale shown. {body} Each "
            "cell of the grid is 1 cm on the drawing. {freedom}",
            "Make a scale drawing of {article} {shape}, using the scale shown on the grid. {body} Shade the plan on "
            "the grid, where one cell is 1 cm of the drawing. {freedom}",
            "Use the scale on the grid to shade a plan of {article} {shape}. {body} One cell of the grid is 1 cm on "
            "the drawing. {freedom}",
        ),
        article=article,
        shape=shape,
        body=body,
        freedom=freedom,
    )
    fig = grid_drawing(
        alt="A blank square grid with a scale written beside it, ready for a scale drawing.",
        grid={"type": "square", "width": gw, "height": gh},
        scale={"text": scale_text},
        canvas="grid",
    )
    size = (gw, gh)
    candidates = []

    def put(kind_, dims, notch_=None, corner_=None):
        candidates.append(_shape_cells(kind_, dims, origin, notch_, corner_))

    put(kind, (a * 2, b * 2), None if kind == "rect" else (c * 2, d * 2), corner)  # forgets to divide by the scale
    if a % 2 == 0 and b % 2 == 0 and (kind == "rect" or (c % 2 == 0 and d % 2 == 0)):
        put(kind, (a // 2, b // 2), None if kind == "rect" else (c // 2, d // 2), corner)
    put(kind, (a + 1, b + 1), notch, corner)  # counts the grid lines, not the cells
    if a != b:
        put(kind, (b, a), notch, corner)  # across and up swapped
    if kind == "L":
        put("rect", (a, b))  # forgets the missing corner
        if d < a and c < b:
            put("L", (a, b), (d, c), corner)  # the missing corner turned on its side
        for other in _CORNERS:
            if other != corner:
                put("L", (a, b), notch, other)
    else:
        put("rect", (a - 1, b))
    working = [
        f"The scale is {scale_text}, so 1 cell is {_metre_text(step)} m of the real {thing}.",
        f"{_metre_text(a * step)} m is {a} cells across the page and {_metre_text(b * step)} m is {b} cells up the page."
        + (f" The missing corner is {_plural(c, 'cell')} across by {_plural(d, 'cell')} up." if kind == "L" else ""),
    ]
    return _cells_question(fig, text, cells, size, match, candidates, working)


@generator(
    "grid_scale_plan",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_scale_plan(rng, tier="easy"):
    """Shade a scale plan of a room or garden described in words, on a grid whose cells are 1 cm of the drawing."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    return _retry(rng, lambda r: _scale_question(r, tier), f"a {tier} scale plan")


_ACROSS_UP = re.compile(r"(\d+(?:\.\d+)?) m across the page and (\d+(?:\.\d+)?) m up the page")


def _read_scale(question) -> str:
    """Cells the wording and the scale on the grid give, drawn at the lower left corner of the grid."""
    per_cm = _scale_cm(question.figure["scale"]["text"])
    pairs = [(Fraction(a), Fraction(b)) for a, b in _ACROSS_UP.findall(question.text)]
    cells = lambda m: int(m * 100 / per_cm)  # noqa: E731
    a, b = cells(pairs[0][0]), cells(pairs[0][1])
    if len(pairs) == 1:
        return cell_text(_shape_cells("rect", (a, b), (0, 0)))
    corner = re.search(r"missing from the (top|bottom) (left|right) corner", question.text).groups()
    return cell_text(_shape_cells("L", (a, b), (0, 0), (cells(pairs[1][0]), cells(pairs[1][1])), corner))


figures.register_reader("grid_scale_plan", _read_scale)

# ------------------------------------------------------------------------------------------- tessellations


def _reflect_x(cells, m):
    return {(2 * m - 1 - x, y) for x, y in cells}


def _reflect_y(cells, m):
    return {(x, 2 * m - 1 - y) for x, y in cells}


def _shift(cells, dx, dy):
    return {(x + dx, y + dy) for x, y in cells}


def _apply(op, cells):
    """One transformation of a set of cells. ``rx``/``ry`` reflect in the line x = m / y = m, ``t`` translates,
    and ``sx``/``sy`` slide a tile to where the reflection would land it without turning it over."""
    kind = op[0]
    if kind == "rx":
        return _reflect_x(cells, op[1])
    if kind == "ry":
        return _reflect_y(cells, op[1])
    if kind == "t":
        return _shift(cells, op[1], op[2])
    if kind == "sx":
        image = _reflect_x(cells, op[1])
        return _shift(cells, min(x for x, _ in image) - min(x for x, _ in cells), 0)
    if kind == "sy":
        image = _reflect_y(cells, op[1])
        return _shift(cells, 0, min(y for _, y in image) - min(y for _, y in cells))
    raise ValueError(f"unknown transformation {kind!r}")


def _run(tile, ops):
    cells = set(tile)
    for op in ops:
        cells = _apply(op, cells)
    return cells


def _tile(rng):
    """A tile of three to five squares, joined along edges, that is not its own mirror image either way."""
    for _ in attempts(300, "an asymmetric tile"):
        tw, th = rng.randint(2, 3), rng.randint(2, 3)
        n = rng.randint(max(3, tw + th - 1), min(5, tw * th - 1))
        cells = {(rng.randrange(tw), rng.randrange(th))}
        for _ in attempts(1000, "a tile grown to its size"):
            if len(cells) >= n:
                break
            x, y = rng.choice(sorted(cells))
            dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            if 0 <= x + dx < tw and 0 <= y + dy < th:
                cells.add((x + dx, y + dy))
        if {x for x, _ in cells} != set(range(tw)) or {y for _, y in cells} != set(range(th)):
            continue
        if {(tw - 1 - x, y) for x, y in cells} == cells or {(x, th - 1 - y) for x, y in cells} == cells:
            continue
        return sorted(cells), tw, th
    raise AssertionError("unreachable")


def _extent(cells):
    return (min(x for x, _ in cells), min(y for _, y in cells), max(x for x, _ in cells), max(y for _, y in cells))


def _place(rng, cellsets, lines):
    """Grid size and the shift that puts every cell set and mirror line on a grid with a small margin."""
    xs = [x for cs in cellsets for x, _ in cs]
    ys = [y for cs in cellsets for _, y in cs]
    lo_x, hi_x = min(xs), max(xs) + 1
    lo_y, hi_y = min(ys), max(ys) + 1
    for axis, m in lines:
        if axis == "v":
            lo_x, hi_x = min(lo_x, m), max(hi_x, m)
        else:
            lo_y, hi_y = min(lo_y, m), max(hi_y, m)
    left, right, below, above = (rng.randint(1, 3) for _ in range(4))
    width, height = hi_x - lo_x + left + right, hi_y - lo_y + below + above
    width, height = max(width, 6), max(height, 6)
    if width > 22 or height > 22:
        return None
    return width, height, left - lo_x, below - lo_y


def _line_figure(lines, width, height, sx, sy):
    shifted, tags = [], []
    for i, (axis, m) in enumerate(lines):
        if axis == "v":
            points = [[m + sx, 0], [m + sx, height]]
            at = [m + sx, height]
        else:
            points = [[0, m + sy], [width, m + sy]]
            at = [width, m + sy]
        shifted.append({"points": points, "style": "mirror"})
        if len(lines) > 1:
            tags.append({"at": at, "text": f"Line {i + 1}"})
    return shifted, tags


def _op_text(ops) -> str:
    """The transformations as one sentence: ``Reflect the tile in mirror line 1, then translate the result ...``."""
    reflections = sum(1 for op in ops if op[0] in ("rx", "ry"))
    seen = 0
    pieces = []
    for i, op in enumerate(ops):
        if op[0] in ("rx", "ry"):
            seen += 1
            where = f"mirror line {seen}" if reflections > 1 else "the mirror line"
            pieces.append(f"reflect {'the tile' if i == 0 else 'the result'} in {where}")
            continue
        move = []
        if op[1]:
            move.append(f"{_plural(abs(op[1]), 'square')} to the {'right' if op[1] > 0 else 'left'}")
        if op[2]:
            move.append(f"{_plural(abs(op[2]), 'square')} {'up' if op[2] > 0 else 'down'}")
        pieces.append(f"translate {'the tile' if i == 0 else 'the result'} {' and '.join(move)}")
    sentence = ", then ".join(pieces) + "."
    return sentence[0].upper() + sentence[1:]


_TESS_ALT = "A square grid with a tile drawn on it in hatched squares, ready for its image to be shaded."
_TESS_ALT_LINES = (
    "A square grid with a tile drawn on it in hatched squares and mirror lines, ready for its image to be shaded."
)

_TESS_ENDINGS = (
    "Shade the {which} of the tile. The hatched squares are already drawn and are given, so do not shade them.",
    "Show the {which} by shading it on the grid. The hatched tile is already drawn and is given, so only shade the {which}.",
    "Shade the {which} on the grid. The squares already drawn are given and are not part of your answer.",
)


def _tess_candidates(rng, tile, ops):
    steps = [set(tile)]
    for op in ops:
        steps.append(_apply(op, steps[-1]))
    out = []
    if len(ops) > 1:
        out.append(steps[1])  # stops after the first transformation
        out.append(_apply(ops[-1], set(tile)))  # does only the last one
        out.append(_run(tile, ops[::-1]))  # does them in the other order
    for i, op in enumerate(ops):
        if op[0] in ("rx", "ry"):
            out.append(_run(tile, ops[:i] + [("sx" if op[0] == "rx" else "sy", op[1])] + ops[i + 1 :]))
            out.append(_run(tile, ops[:i] + [("ry" if op[0] == "rx" else "rx", op[1])] + ops[i + 1 :]))
            for off in (-1, 1):
                out.append(_run(tile, ops[:i] + [(op[0], op[1] + off)] + ops[i + 1 :]))
        else:
            _t, dx, dy = op
            for alt in ((-dx, dy), (dx, -dy), (dy, dx), (dx + (1 if dx > 0 else -1 if dx else 0), dy)):
                out.append(_run(tile, ops[:i] + [("t", *alt)] + ops[i + 1 :]))
    rng.shuffle(out)
    return out


def _tess_ops(rng, tile, tw, th, tier):
    """The transformations, in order, for an easy or medium question (the tile sits at the origin)."""
    side = lambda lo, hi: rng.choice((lo - rng.randint(0, 2), hi + 1 + rng.randint(0, 2)))  # noqa: E731
    if tier == "easy":
        kind = rng.choice(("rx", "ry", "t"))
        if kind == "rx":
            return [("rx", side(0, tw - 1))]
        if kind == "ry":
            return [("ry", side(0, th - 1))]
        dx = rng.choice((-1, 1)) * rng.randint(tw + 1, tw + 3)
        return [("t", dx, rng.randint(-2, 2))]
    kind = rng.choice(("rt", "tr", "rxry", "rxrx", "ryry"))
    if kind in ("rt", "tr"):
        reflect = ("rx", side(0, tw - 1)) if rng.random() < 0.5 else ("ry", side(0, th - 1))
        move = ("t", rng.choice((-1, 1)) * rng.randint(1, 4), rng.choice((-1, 1)) * rng.randint(1, 3))
        return [reflect, move] if kind == "rt" else [move, reflect]
    if kind == "rxry":
        return [("rx", side(0, tw - 1)), ("ry", side(0, th - 1))]
    if kind == "rxrx":
        return [("rx", side(0, tw - 1)), ("rx", side(0, tw - 1) + rng.choice((-3, -2, 2, 3)))]
    return [("ry", side(0, th - 1)), ("ry", side(0, th - 1) + rng.choice((-3, -2, 2, 3)))]


def _tess_simple(rng, tier: str):
    tile, tw, th = _tile(rng)
    ops = _tess_ops(rng, tile, tw, th, tier)
    final = _run(tile, ops)
    if final & set(tile) or final == set(tile):
        return None
    steps = [set(tile)]
    for op in ops:
        steps.append(_apply(op, steps[-1]))
    lines = [("v" if op[0] == "rx" else "h", op[1]) for op in ops if op[0] in ("rx", "ry")]
    candidates = _tess_candidates(rng, tile, ops)
    placed = _place(rng, steps, lines)
    if placed is None:
        return None
    width, height, sx, sy = placed
    fit = [c for c in candidates if _on_grid(_shift(c, sx, sy), (width, height))]
    mirror_lines, tags = _line_figure(lines, width, height, sx, sy)
    given = sorted(_shift(tile, sx, sy))
    answer = sorted(_shift(final, sx, sy))
    fig = grid_drawing(
        alt=_TESS_ALT_LINES if lines else _TESS_ALT,
        grid={"type": "square", "width": width, "height": height},
        cells=[{"cells": [list(c) for c in given], "style": "hatched"}],
        lines=mirror_lines,
        labels=tags,
        canvas="grid",
    )
    which = "image" if len(ops) == 1 else "final image"
    ops_sentence = _op_text(ops)
    lead = rng.choice(
        (
            "The hatched squares on the grid are a tile.",
            "A tile is drawn on the grid in hatched squares.",
            "Look at the tile drawn in hatched squares on the grid.",
        )
    )
    text = f"{lead} {ops_sentence} " + rng.choice(_TESS_ENDINGS).format(which=which)
    shifted = [_shift(c, sx, sy) for c in fit]
    how = (
        "A reflected square is as far behind the mirror line as the original is in front of it, measured to the "
        "edge of the square."
        if lines
        else "Move every square of the tile the same way."
    )
    working = [how, f"The {which} has {len(answer)} squares, the same as the tile."]
    return _cells_question(fig, text, answer, (width, height), "exact", shifted, working)


def _tess_next(rng):
    """The next two tiles of a strip of tiles: repeated reflections in equally spaced lines, or repeated slides."""
    tile, tw, th = _tile(rng)
    mirrored = rng.random() < 0.5
    flip, transpose = rng.random() < 0.5, rng.random() < 0.5
    if mirrored:
        spacing = rng.randint(max(tw, 3), 4)
        a = spacing
        t0 = _shift(set(tile), rng.randint(0, spacing - tw), 0)
        t1 = _reflect_x(t0, a)
        t2 = _reflect_x(t1, a + spacing)
        t3 = _reflect_x(t2, a + 2 * spacing)
        lines = [("v", a), ("v", a + spacing)]
        given_sets = [t0, t1]
        asked = t2 | t3
        wrong = [
            t2,
            _shift(t0, 2 * spacing, 0) | _shift(t0, 3 * spacing, 0),
            t2 | _reflect_x(t2, a + 2 * spacing + 1),
            t2 | _reflect_x(t2, a + 2 * spacing - 1),
            _reflect_y(t2, th // 2 + 1) | _reflect_y(t3, th // 2 + 1),
            t3,
        ]
    else:
        vx = rng.randint(tw, tw + 2)
        vy = rng.choice((-2, -1, 0, 1, 2))
        t0 = set(tile)
        t1 = _shift(t0, vx, vy)
        t2, t3 = _shift(t0, 2 * vx, 2 * vy), _shift(t0, 3 * vx, 3 * vy)
        lines = []
        given_sets = [t0, t1]
        asked = t2 | t3
        wrong = [
            t2,
            _shift(t0, 3 * vx, 3 * vy) | _shift(t0, 4 * vx, 4 * vy),
            _shift(t0, 2 * vx, 2 * vy) | _shift(t0, 3 * vx, 3 * vy + (1 if vy <= 0 else -1)),
            _shift(t0, 2 * vx, 2 * vy) | _shift(t0, 3 * vx, 3 * vy - vy * 2),
            _shift(t0, 2 * vx, 0) | _shift(t0, 3 * vx, 0),
        ]

    def transform(cells):
        out = {(-1 - x if flip else x, y) for x, y in cells}
        return {(y, x) for x, y in out} if transpose else out

    def transform_line(line):
        axis, m = line
        m = -m if flip else m
        return ("h" if axis == "v" else "v", m) if transpose else (axis, m)

    given_sets, asked, wrong = (
        [transform(c) for c in given_sets],
        transform(asked),
        [transform(c) for c in wrong],
    )
    lines = [transform_line(ln) for ln in lines]
    placed = _place(rng, [*given_sets, asked], lines)
    if placed is None:
        return None
    width, height, sx, sy = placed
    mirror_lines, tags = _line_figure(lines, width, height, sx, sy)
    fig = grid_drawing(
        alt=(
            "A square grid with two tiles of a pattern and two mirror lines drawn on it, ready for the next tiles."
            if mirrored
            else "A square grid with two tiles of a pattern drawn on it in hatched squares, ready for the next tiles."
        ),
        grid={"type": "square", "width": width, "height": height},
        cells=[{"cells": [list(c) for c in sorted(_shift(g, sx, sy))], "style": "hatched"} for g in given_sets],
        lines=mirror_lines,
        labels=tags,
        canvas="grid",
    )
    if mirrored:
        body = (
            "Each tile in this pattern is the reflection of the tile before it in the next mirror line, and the "
            "mirror lines are the same distance apart. The first two tiles are drawn in hatched squares, with the "
            "first two mirror lines."
        )
    else:
        body = (
            "Each tile in this pattern is the tile before it moved the same way along the grid. The first two tiles "
            "are drawn in hatched squares."
        )
    text = phrase(
        rng,
        (
            "{body} Shade the next two tiles of the pattern. The hatched tiles are already drawn and are given, "
            "so do not shade them.",
            "{body} Continue the pattern by shading the next two tiles. The squares already drawn are given and "
            "are not part of your answer.",
            "{body} Show the next two tiles by shading them on the grid. The hatched tiles are already drawn and are given, so only shade the two new tiles.",
        ),
        body=body,
    )
    answer = sorted(_shift(asked, sx, sy))
    candidates = [_shift(w, sx, sy) for w in wrong]
    working = [
        "Find how the second tile comes from the first, then do the same to the second tile to get the third, and "
        "to the third to get the fourth.",
    ]
    return _cells_question(fig, text, answer, (width, height), "exact", candidates, working)


@generator(
    "grid_tessellation_images",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_tessellation_images(rng, tier="easy"):
    """Shade the image of a tile after one transformation, after two, or the next tiles of a tessellation."""
    if tier in ("easy", "medium"):
        return _retry(rng, lambda r: _tess_simple(r, tier), f"a {tier} tessellation question")
    if tier == "hard":
        return _retry(rng, _tess_next, "a tessellation strip")
    raise ValueError(f"unknown tier {tier!r}")


_OP_WORDS = re.compile(r"\b(reflect|translate)\b", re.IGNORECASE)
_MOVE_X = re.compile(r"(\d+) squares? to the (left|right)")
_MOVE_Y = re.compile(r"(\d+) squares? (up|down)")


def _line_of(line):
    (x0, y0), (x1, y1) = line["points"]
    return ("v", x0) if x0 == x1 else ("h", y0)


def _read_tessellation(question) -> str:
    fig, text = question.figure, question.text
    groups = [{tuple(c) for c in g["cells"]} for g in fig["cells"]]
    lines = [_line_of(ln) for ln in fig.get("lines", [])]
    if "next two tiles" in text:
        t0, t1 = groups
        if lines:
            (axis1, m1), (axis2, m2) = lines
            reflect = _reflect_x if axis1 == "v" else _reflect_y
            t2 = reflect(t1, m2)
            t3 = reflect(t2, 2 * m2 - m1)
            return cell_text(t2 | t3)
        dx = min(x for x, _ in t1) - min(x for x, _ in t0)
        dy = min(y for _, y in t1) - min(y for _, y in t0)
        return cell_text(_shift(t1, dx, dy) | _shift(t1, 2 * dx, 2 * dy))
    cells = groups[0]
    marks = list(_OP_WORDS.finditer(text))
    used = 0
    for i, mark in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        piece = text[mark.end() : stop]
        if mark.group(1).lower() == "reflect":
            numbered = re.search(r"mirror line (\d)", piece)
            axis, m = lines[int(numbered.group(1)) - 1] if numbered else lines[used]
            used += 1
            cells = _reflect_x(cells, m) if axis == "v" else _reflect_y(cells, m)
        else:
            mx, my = _MOVE_X.search(piece), _MOVE_Y.search(piece)
            dx = (int(mx.group(1)) * (1 if mx.group(2) == "right" else -1)) if mx else 0
            dy = (int(my.group(1)) * (1 if my.group(2) == "up" else -1)) if my else 0
            cells = _shift(cells, dx, dy)
    return cell_text(cells)


figures.register_reader("grid_tessellation_images", _read_tessellation)

# ------------------------------------------------------------------------------------------- enlargement

_ENL_ALT = "A shape drawn on a square grid with a point O marked as the centre of an enlargement."
_RIGHT_TRIANGLES = ((3, 4, 5), (6, 8, 10))
_FACTORS = (Fraction(2), Fraction(3), Fraction(1, 2))
_LETTERS = "ABCD"


def _fraction_text(k: Fraction) -> str:
    return str(k.numerator) if k.denominator == 1 else f"{k.numerator}/{k.denominator}"


def _side_lengths(vertices) -> list[int]:
    out = []
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1]):
        sq = (x1 - x0) ** 2 + (y1 - y0) ** 2
        root = math.isqrt(sq)
        check(root * root == sq, "a side of the shape is not a whole number of units")
        out.append(root)
    return out


def _enlargement(rng):
    k = rng.choice(_FACTORS)
    if k < 1:
        shape = rng.choice(("triangle", "rectangle"))
        dims = (6, 8) if shape == "triangle" else (rng.choice((2, 4)), rng.choice((2, 4, 6)))
    else:
        shape = rng.choice(("triangle", "rectangle"))
        dims = (3, 4) if shape == "triangle" else (rng.randint(2, 4), rng.randint(2, 4))
    a, b = dims
    base = [(0, 0), (a, 0), (0, b)] if shape == "triangle" else [(0, 0), (a, 0), (a, b), (0, b)]
    turn = rng.randrange(4)
    mirrored = rng.random() < 0.5
    verts = []
    for x, y in base:
        for _ in range(turn):
            x, y = -y, x
        verts.append((-x if mirrored else x, y))
    px, py = rng.randint(0, 3), rng.randint(0, 3)
    verts = [(x + px, y + py) for x, y in verts]
    first = verts[0]
    # a centre that is a lattice point no vertex sits on; for a reduction its distance from each vertex is even
    step = 2 if k < 1 else 1
    ox = first[0] + step * rng.randint(-3, 3)
    oy = first[1] + step * rng.randint(-3, 3)
    centre = (ox, oy)
    if centre in verts:
        return None
    image = [(ox + k * (x - ox), oy + k * (y - oy)) for x, y in verts]
    check(all(v.denominator == 1 for p in image for v in map(Fraction, p)), "image vertex is off the lattice")
    image = [(int(x), int(y)) for x, y in image]
    allpts = verts + image + [centre]
    lo_x, lo_y = min(x for x, _ in allpts), min(y for _, y in allpts)
    hi_x, hi_y = max(x for x, _ in allpts), max(y for _, y in allpts)
    sx, sy = rng.randint(1, 2) - lo_x, rng.randint(1, 2) - lo_y
    width, height = hi_x + sx + rng.randint(1, 2), hi_y + sy + rng.randint(1, 2)
    if width > 30 or height > 30:
        return None
    verts = [(x + sx, y + sy) for x, y in verts]
    image = [(x + sx, y + sy) for x, y in image]
    centre = (ox + sx, oy + sy)
    names = _LETTERS[: len(verts)]
    fig = grid_drawing(
        alt=_ENL_ALT,
        grid={"type": "square", "width": width, "height": height},
        polygons=[{"vertices": [list(v) for v in verts], "style": "solid"}],
        labels=[*({"at": list(v), "text": n} for v, n in zip(verts, names)), {"at": list(centre), "text": "O"}],
    )
    sides = _side_lengths(verts)
    perimeter = sum(sides)
    ask_side = rng.random() < 0.5
    which = rng.randrange(len(verts))
    k_text = _fraction_text(k)
    stem = phrase(
        rng,
        (
            "{shape} {names} is drawn on the grid. It is enlarged from the centre of enlargement O with a scale "
            "factor of {k}, to give the image {primes}. Each square of the grid is 1 cm by 1 cm, and the "
            "bottom-left corner of the grid is the point (0, 0).",
            "On the grid, {shape} {names} is enlarged about O with a scale factor of {k}. Its image is {primes}. "
            "The sides of the squares are 1 cm and the bottom-left corner of the grid is (0, 0).",
            "The grid shows {shape} {names} and the centre of enlargement O. The {lower} is enlarged by a scale "
            "factor of {k}. Each square is 1 cm by 1 cm and the point (0, 0) is the bottom-left corner of the grid.",
        ),
        shape=shape.capitalize(),
        lower=shape,
        names="".join(names),
        primes="".join(n + "'" for n in names),
        k=k_text,
    )
    if ask_side:
        a_name, b_name = names[which], names[(which + 1) % len(names)]
        want = Fraction(sides[which]) * k
        prompt_a = rng.choice(
            (
                f"Find the length of {a_name}'{b_name}', in cm.",
                f"How long is the side {a_name}'{b_name}' of the image, in cm?",
                f"Find, in cm, the length of the side of the image that matches {a_name}{b_name}.",
            )
        )
        wrong_a = [
            Fraction(sides[which]),  # the original length
            Fraction(sides[which]) * k * k,
            Fraction(sides[which]) + k,
            Fraction(sides[which]) / k,
        ]
        if shape == "triangle" and sides[which] == max(sides):
            wrong_a.append(Fraction(sum(sides) - sides[which]) * k)  # adds the two short sides
        work_a = [f"{a_name}{b_name} is {sides[which]} cm long.", f"The image is {_fraction_text(k)} times as long."]
    else:
        want = Fraction(perimeter) * k
        prompt_a = rng.choice(
            (
                "Find the perimeter of the image, in cm.",
                "What is the perimeter of the image, in cm?",
                "Find, in cm, the distance all the way round the image.",
            )
        )
        wrong_a = [Fraction(perimeter), Fraction(perimeter) * k * k, Fraction(perimeter) + k]
        if shape == "triangle":
            wrong_a.append(Fraction(perimeter - max(sides)) * k)  # leaves out the slanted side
        if k < 1:
            wrong_a.append(Fraction(perimeter) / k)
        work_a = [
            f"The perimeter of the original is {' + '.join(map(str, sides))} = {perimeter} cm.",
            f"Lengths are multiplied by {k_text}.",
        ]
    check(want.denominator == 1, "the image length is not a whole number of centimetres")
    a_text = str(int(want))
    a_candidates = [str(int(v)) for v in wrong_a if v.denominator == 1 and v > 0]
    a_wrong = pick_distractors(a_text, a_candidates, 3, tolerance=GRADER_TOLERANCE)
    prompt_b = rng.choice(
        (
            "Write down the coordinates of the vertices {primes} of the image, in that order.",
            "Give the coordinates of {primes}, in order, as points such as (3, 5).",
            "State the coordinates of the image vertices {primes} in order.",
        )
    ).format(primes=", ".join(n + "'" for n in names))
    b_text = points_answer_text(image)

    def pts(points):
        return points_answer_text([(Fraction(x), Fraction(y)) for x, y in points])

    cx, cy = centre
    o = [Fraction(v) for v in centre]
    vv = [(Fraction(x), Fraction(y)) for x, y in verts]
    candidates_b = [
        [(k * x, k * y) for x, y in vv],  # enlarges from (0, 0) instead of O
        [(k * (x - o[0]), k * (y - o[1])) for x, y in vv],  # forgets to add O back
        [(x + k * (x - o[0]), y + k * (y - o[1])) for x, y in vv],  # one scale factor too many
        [(vv[0][0] + k * (x - vv[0][0]), vv[0][1] + k * (y - vv[0][1])) for x, y in vv],  # enlarges from vertex A
        [(Fraction(x), Fraction(y)) for x, y in image[1::-1] + image[2:]],  # vertices out of order
    ]
    b_candidates = [pts(c) for c in candidates_b]
    b_wrong = pick_distractors(b_text, b_candidates, 3)
    if len(a_wrong) < 3 or len(b_wrong) < 3:
        return None
    labelled = [n for n in names]
    work_b = [
        f"Each image vertex is on the ray from O through its vertex, {k_text} times as far from O: "
        f"for {labelled[0]}, O is at ({cx}, {cy}) and {labelled[0]} is at {verts[0]}.",
        f"The image vertices are {b_text}.",
    ]
    return dict(
        text=stem,
        figure=fig,
        parts=[
            dict(
                prompt=prompt_a,
                answer_kind="number",
                answer=int(want),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                sensible_range=(0, None),
                working=work_a,
                distractors=a_wrong,
            ),
            dict(
                prompt=prompt_b,
                answer_kind="points",
                answer={"points": [[x, y] for x, y in image], "ordered": True},
                answer_text=b_text,
                answer_latex="plain:" + b_text,
                working=work_b,
                distractors=b_wrong,
            ),
        ],
    )


@generator("grid_enlargement_image", level=9)
def grid_enlargement_image(rng):
    """Enlarge a lattice triangle or rectangle: a length of the image, then the coordinates of its vertices."""
    return _retry(rng, _enlargement, "an enlargement on a grid")


_K_WORDS = re.compile(r"scale factor of (\d+(?:/\d+)?)")
_SIDE_WORDS = re.compile(r"([A-D])'([A-D])'|matches ([A-D])([A-D])")


def _read_enlargement(question) -> str:
    fig, text = question.figure, question.text
    verts = [tuple(v) for v in fig["polygons"][0]["vertices"]]
    at = {label["text"]: tuple(label["at"]) for label in fig["labels"]}
    check([at[n] for n in "ABCD"[: len(verts)]] == verts, "the vertex labels are not on the vertices")
    centre = at["O"]
    k = Fraction(_K_WORDS.search(text).group(1))
    image = [(centre[0] + k * (x - centre[0]), centre[1] + k * (y - centre[1])) for x, y in verts]
    lengths = _side_lengths(verts)
    prompt_a = text.split("(a) ", 1)[1].split(" (b) ", 1)[0]
    if "perimeter" in prompt_a or "all the way round" in prompt_a:
        value = sum(lengths) * k
    else:
        m = _SIDE_WORDS.search(prompt_a)
        first = m.group(1) or m.group(3)
        value = lengths["ABCD".index(first)] * k
    return f"(a) {int(value)}; (b) {points_answer_text(image)}"


figures.register_reader("grid_enlargement_image", _read_enlargement)
