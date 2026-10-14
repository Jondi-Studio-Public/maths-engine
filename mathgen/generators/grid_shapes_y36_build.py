"""Models, turns, box nets and latitude and longitude, drawn on a grid (``grid_drawing`` figures).

Four topics, each built from the ``grid_drawing`` figure kind (`mathgen.figure_kinds.grid`), each with a figure reader
that re-derives the answer from the figure and the wording alone.

``grid_model_plans`` (Year 3, ``number``)
    A model built from cubes, shown as a plan from above with the number of cubes in each stack written on its
    square. Easy counts the cubes in a plan with no gaps; medium has bare squares and either counts the cubes or
    finds how many more cubes make every stack as tall as the tallest; hard gives the total and one square marked
    ``?``, and finds the stack that is missing from the count.

``grid_shape_turns`` (Year 5, ``grid_cells``, match ``exact``)
    The learner shades the image of a shape on its own grid. Easy is a half turn about a marked point, medium a
    quarter turn clockwise or anticlockwise about it, and hard shows a shape and its image after one slide or one
    quarter turn and asks for the image after the same move again, so the move has to be read from the picture.

``grid_prism_nets`` (Year 6, ``grid_cells``, match ``translate``)
    The learner shades the net of a box. Easy pins the layout in words (top and bottom both joined to the first
    face of a row of four), medium joins the top and bottom to named faces of the row, and hard gives the row
    and the top and asks for the bottom joined to a named face, so its size comes from the face it is joined to.

``grid_latlong_places`` and ``grid_latlong_exam`` (senior, all ``mcq``)
    A map grid with a line every 10 degrees, latitude up the left side and longitude along the top, and places
    marked with letters. Easy reads the latitude and longitude of a place, medium picks the place at stated
    coordinates across the equator or the prime meridian, hard finds the degrees between two places on
    opposite sides of the equator or the prime meridian, and the exam question finds the distance flown along a meridian.

Every wrong grid answer is checked with the grader's own matching rule (`grid_shading_build._is_wrong`) and every
wrong number against the grader's tolerance, before it is offered.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, phrase, pick_distractors
from ..figure_kinds.grid import grid_drawing
from ..grading.grid import cell_text, normalised, orientations
from . import _space_kit as kit
from . import grid_shading_build as gsb

Cell = tuple[int, int]

# ------------------------------------------------------------------------------------------- shared helpers


def _number_question(fig, text, answer: int, candidates, working) -> dict | None:
    """The dict a ``number`` generator returns, or None when the figure leaks the answer or too few wrong
    answers survive (the caller draws again). A miscount by one or two pads the list of real slips."""
    answer_text = str(answer)
    if figures.validate(fig):
        raise ValueError(f"invalid figure: {figures.validate(fig)}")
    if kit.leaks(fig, answer_text):
        return None
    pool = [str(c) for c in candidates if isinstance(c, int) and c > 0 and c != answer]
    pool += [str(answer + d) for d in (1, -1, 2) if answer + d > 0]
    distractors = pick_distractors(answer_text, pool, 3, tolerance=GRADER_TOLERANCE)
    if len(distractors) < 3:
        return None
    return dict(
        text=text,
        answer=answer,
        answer_kind="number",
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _text_question(fig, text, answer_text: str, candidates, working) -> dict | None:
    """The dict an ``mcq`` generator returns, or None when fewer than three different wrong options exist."""
    if figures.validate(fig):
        raise ValueError(f"invalid figure: {figures.validate(fig)}")
    distractors = pick_distractors(answer_text, [c for c in candidates if c != answer_text])
    if len(distractors) < 3 or kit.leaks(fig, answer_text):
        return None
    return dict(
        text=text,
        answer=answer_text,
        answer_kind="mcq",
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _tier(tiers: dict, tier: str):
    if tier not in tiers:
        raise ValueError(f"unknown tier {tier!r}")
    return tiers[tier]


def _fit(cellsets, size_cap: int = 18, margin: int = 1, rng=None):
    """Grid width, height and the shift that puts every set of cells on a grid with a margin, or None if too big."""
    xs = [x for cs in cellsets for x, _ in cs]
    ys = [y for cs in cellsets for _, y in cs]
    left, right, below, above = (margin + (rng.randint(0, 1) if rng else 0) for _ in range(4))
    width, height = max(xs) - min(xs) + 1 + left + right, max(ys) - min(ys) + 1 + below + above
    if width > size_cap or height > size_cap:
        return None
    return max(width, 5), max(height, 5), left - min(xs), below - min(ys)


# ------------------------------------------------------------------------------------------- Year 3: models

_MODEL_ALT = (
    "A plan of a model built from cubes, seen from above, with the number of cubes in each stack written on its square."
)
_MODEL_ALT_GAP = (
    "A plan of a model built from cubes, seen from above, with the number of cubes in each stack written on its "
    "square and one square marked with a question mark."
)


def _model_heights(rng, w: int, d: int, holes: int, top: int) -> dict[Cell, int] | None:
    spots = [(x, y) for x in range(w) for y in range(d)]
    bare = set(rng.sample(spots, holes)) if holes else set()
    h = {s: (0 if s in bare else rng.randint(1, top)) for s in spots}
    if any(all(h[(x, y)] == 0 for y in range(d)) for x in range(w)):
        return None
    if any(all(h[(x, y)] == 0 for x in range(w)) for y in range(d)):
        return None
    return h


def _model_figure(h: dict[Cell, int], w: int, d: int, unknown: Cell | None = None):
    footprint = [[x, y] for (x, y), v in sorted(h.items()) if v]
    labels = [
        {"at": [x + 0.5, y + 0.5], "text": ("?" if (x, y) == unknown else str(v))}
        for (x, y), v in sorted(h.items())
        if v
    ]
    return grid_drawing(
        alt=_MODEL_ALT_GAP if unknown else _MODEL_ALT,
        grid={"type": "square", "width": w, "height": d},
        cells=[{"cells": footprint, "style": "shaded"}],
        labels=labels,
    )


def _model_easy(rng):
    w, d = rng.choice(((2, 2), (3, 2), (2, 3), (3, 3)))
    h = _model_heights(rng, w, d, 0, 3)
    check(h is not None, "a plan with no gaps")
    total, squares, tallest = sum(h.values()), len(h), max(h.values())
    if not 5 <= total <= 14 or len(set(h.values())) < 2:
        return None
    text = phrase(
        rng,
        (
            "A model is built from cubes. The plan shows how many cubes are in each stack. How many cubes are in "
            "the model?",
            "This plan is a model seen from above. The number on a square is how many cubes are stacked there. "
            "How many cubes make the model?",
            "Look at the plan of a cube model. Each number tells how many cubes are in that stack. How many cubes "
            "does the model have in all?",
        ),
    )
    candidates = [squares, tallest * squares, total - tallest, sum(sorted(h.values())[-2:]) + squares]
    working = [
        "Each square of the plan is one stack, and its number is how many cubes are in the stack.",
        "Add the numbers: " + " + ".join(str(v) for _, v in sorted(h.items())) + f" = {total}.",
    ]
    return _number_question(_model_figure(h, w, d), text, total, candidates, working)


def _model_medium(rng):
    w, d = rng.choice(((3, 2), (3, 3), (4, 2), (4, 3)))
    h = _model_heights(rng, w, d, rng.randint(1, 2), 4)
    if h is None:
        return None
    stacks = {s: v for s, v in h.items() if v}
    total, tallest = sum(stacks.values()), max(stacks.values())
    if len(set(stacks.values())) < 3:
        return None
    if rng.random() < 0.5:
        text = phrase(
            rng,
            (
                "The plan shows a model built from cubes. A bare square has no cubes on it. How many cubes are in "
                "the model?",
                "A cube model is seen from above. The number on a square is the height of its stack in cubes, and "
                "a bare square has no stack. How many cubes make the model?",
                "Here is the plan of a model. Squares with a number have that many cubes, and blank squares have "
                "none. How many cubes are in the model altogether?",
            ),
        )
        candidates = [len(stacks), tallest * len(stacks), tallest * w * d, total - tallest]
        working = [
            "A bare square has no cubes, so it adds nothing.",
            "Add the numbers: " + " + ".join(str(v) for _, v in sorted(stacks.items())) + f" = {total}.",
        ]
        answer = total
    else:
        answer = tallest * len(stacks) - total
        if answer < 2:
            return None
        text = phrase(
            rng,
            (
                "The plan shows a cube model. More cubes are added so that every stack is as tall as the tallest "
                "stack. Bare squares stay bare. How many cubes are added?",
                "A model is built from cubes, as the plan shows. Every stack that has cubes is made as tall as the "
                "tallest stack, and bare squares stay bare. How many more cubes are needed?",
                "Look at the plan of the cube model. How many more cubes make every stack the same height as the "
                "tallest one? Do not build on the bare squares.",
            ),
        )
        candidates = [tallest * len(stacks), total, tallest * w * d - total, tallest * w * d]
        working = [
            f"The tallest stack has {tallest} cubes, and {len(stacks)} squares have stacks.",
            f"They would hold {tallest} x {len(stacks)} = {tallest * len(stacks)} cubes, and there are {total} now.",
            f"{tallest * len(stacks)} - {total} = {answer} more cubes.",
        ]
    return _number_question(_model_figure(h, w, d), text, answer, candidates, working)


def _model_hard(rng):
    w, d = rng.choice(((3, 3), (4, 2), (4, 3), (3, 4)))
    h = _model_heights(rng, w, d, rng.randint(0, 2), 5)
    if h is None:
        return None
    stacks = {s: v for s, v in h.items() if v}
    if len(stacks) < 4:
        return None
    unknown = rng.choice(sorted(stacks))
    answer, total = stacks[unknown], sum(stacks.values())
    others = total - answer
    text = phrase(
        rng,
        (
            "A model is built from {total} cubes. The plan shows the cubes in each stack, but one square has a "
            "question mark. How many cubes are in that stack?",
            "The plan shows a cube model that has {total} cubes in all. One stack is marked with a question mark. "
            "How many cubes are stacked on that square?",
            "{total} cubes make this model. The numbers on the plan tell the size of each stack, except the stack "
            "marked with a question mark. How many cubes are in it?",
        ),
        total=total,
    )
    candidates = [others, total, total - len(stacks) + 1, others - answer]
    working = [
        "Add the stacks that show a number: "
        + " + ".join(str(v) for s, v in sorted(stacks.items()) if s != unknown)
        + f" = {others}.",
        f"The model has {total} cubes in all, so the stack with the question mark has {total} - {others} = {answer}.",
    ]
    out = _number_question(_model_figure(h, w, d, unknown), text, answer, candidates, working)
    if out is not None and str(total) == out["answer_text"]:
        return None
    return out


_MODEL_TIERS = {"easy": _model_easy, "medium": _model_medium, "hard": _model_hard}


@generator(
    "grid_model_plans",
    level=3,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_model_plans(rng, tier="easy"):
    """Count the cubes in a model from its plan, fill it to the tallest stack, or find one missing stack."""
    build = _tier(_MODEL_TIERS, tier)
    return gsb._retry(rng, build, f"a {tier} cube model plan")


_TOTAL = re.compile(r"(\d+) cubes")


def _read_model(question) -> str:
    fig, text = question.figure, question.text
    numbers = {(int(lb["at"][0]), int(lb["at"][1])): lb["text"] for lb in fig["labels"]}
    known = [int(v) for v in numbers.values() if v != "?"]
    if "?" in numbers.values():
        return str(int(_TOTAL.search(text).group(1)) - sum(known))
    if "more cubes" in text or "cubes are added" in text:
        return str(max(known) * len(known) - sum(known))
    return str(sum(known))


figures.register_reader("grid_model_plans", _read_model)

# ------------------------------------------------------------------------------------------- Year 5: turns


def _turn(cells, p: Cell, quarters: int) -> set[Cell]:
    """The cells after ``quarters`` quarter turns anticlockwise about the lattice point ``p`` (3 is clockwise)."""
    px, py = p
    out = set()
    for x, y in cells:
        rx, ry = 2 * x + 1 - 2 * px, 2 * y + 1 - 2 * py  # the cell's centre, doubled, from p
        for _ in range(quarters % 4):
            rx, ry = -ry, rx
        out.add(((2 * px + rx - 1) // 2, (2 * py + ry - 1) // 2))
    return out


def _asymmetric_tile(rng, count: int) -> set[Cell]:
    """A tile of ``count`` squares joined along edges, with all eight turns and flips different."""
    for _ in attempts(500, "an asymmetric tile"):
        cells = {(0, 0)}
        for _ in attempts(100, "a tile grown to its size"):
            if len(cells) == count:
                break
            x, y = rng.choice(sorted(cells))
            dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            cells.add((x + dx, y + dy))
        tile = normalised(cells)
        if len(orientations(tile)) == 8 and max(max(x for x, _ in tile), max(y for _, y in tile)) <= 3:
            return set(tile)
    raise AssertionError("unreachable")


_TURN_WORDS = {1: "a quarter turn anticlockwise", 3: "a quarter turn clockwise", 2: "a half turn"}
_TURN_ALT = "A square grid with a shape drawn in shaded squares and a point marked P, ready for its image to be shaded."
_TURN_ENDINGS = (
    "Shade the shape after the turn. The squares already drawn are given, so do not shade them.",
    "Shade where the shape lands. The shaded squares that are already drawn are given and are not part of your answer.",
    "Show the turned shape by shading it on the grid. Do not shade the squares that are already drawn.",
)


def _turn_candidates(tile, p: Cell, quarters: int, extra=()):
    px, py = p
    out = [*extra]
    for other in sorted({1, 2, 3} - {quarters}):
        out.append(_turn(tile, p, other))
    out.append(gsb._reflect_x(tile, px))
    out.append(gsb._reflect_y(tile, py))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
        out.append(_turn(tile, (px + dx, py + dy), quarters))
    return out


def _turn_question(rng, tier: str):
    tile = _asymmetric_tile(rng, rng.choice((4, 5)))
    corners = sorted({(x + a, y + b) for x, y in tile for a in (0, 1) for b in (0, 1)})
    if tier == "easy":
        quarters, p = 2, rng.choice(corners)
    else:
        quarters = rng.choice((1, 3))
        reach = [(x + dx, y + dy) for x, y in corners for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)]
        p = rng.choice(sorted(set(reach)))
    image = _turn(tile, p, quarters)
    if image & tile or image == tile:
        return None
    candidates = _turn_candidates(tile, p, quarters)
    fitted = _fit([tile, image, {p}], rng=rng)
    if fitted is None:
        return None
    width, height, sx, sy = fitted
    fig = grid_drawing(
        alt=_TURN_ALT,
        grid={"type": "square", "width": width, "height": height},
        cells=[{"cells": [list(c) for c in sorted(gsb._shift(tile, sx, sy))], "style": "shaded"}],
        labels=[{"at": [p[0] + sx, p[1] + sy], "text": "P"}],
        canvas="grid",
    )
    start = rng.choice(("The shaded shape on the grid", "The shape drawn in shaded squares", "The shaded shape"))
    text = f"{start} is turned {_TURN_WORDS[quarters]} about the point P. " + rng.choice(_TURN_ENDINGS)
    shifted = [gsb._shift(c, sx, sy) for c in candidates]
    working = [
        "Turn each square of the shape about P, not about the middle of the shape: the point P stays where it is.",
        "A half turn puts every square on the opposite side of P, the same distance away."
        if quarters == 2
        else "A quarter turn swings every square a quarter of the way round P, "
        + ("anticlockwise" if quarters == 1 else "clockwise")
        + ", the same distance from P.",
    ]
    return gsb._cells_question(fig, text, sorted(gsb._shift(image, sx, sy)), (width, height), "exact", shifted, working)


def _repeat_question(rng):
    """A shape and its image after one slide or one quarter turn: the image after the same move again."""
    tile = _asymmetric_tile(rng, rng.choice((4, 5)))
    if rng.random() < 0.5:
        vx = rng.choice((-1, 1)) * rng.randint(3, 5)
        vy = rng.choice((-2, -1, 0, 1, 2))
        move = ("slide", (vx, vy))
        p = None
        first = gsb._shift(tile, vx, vy)
        answer = gsb._shift(tile, 2 * vx, 2 * vy)
        candidates = [
            gsb._reflect_x(first, max(x for x, _ in first) + 1),  # flips the image instead of sliding it
            gsb._shift(first, -vx, -vy),  # the slide the wrong way: back to the start
            gsb._shift(first, 2 * vx, 2 * vy),  # twice as far
            gsb._shift(first, vx, 0),  # forgets the up or down part
            gsb._shift(first, vy, vx),  # swaps across and up
            _turn(first, (max(x for x, _ in first) + 1, min(y for _, y in first)), 2),
        ]
    else:
        quarters = rng.choice((1, 3))
        corners = sorted({(x + a, y + b) for x, y in tile for a in (0, 1) for b in (0, 1)})
        reach = [(x + dx, y + dy) for x, y in corners for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        p = rng.choice(sorted(set(reach)))
        move = ("turn", quarters)
        first = _turn(tile, p, quarters)
        answer = _turn(first, p, quarters)
        px, py = p
        slid = (
            min(x for x, _ in first) - min(x for x, _ in tile),
            min(y for _, y in first) - min(y for _, y in tile),
        )
        candidates = [
            _turn(first, p, -quarters),  # turns the image back the other way
            gsb._shift(first, *slid),  # slides the image the way the shape first moved
            _turn(first, p, 2),  # a half turn of the image
            _turn(first, (px + 1, py), quarters),
            _turn(first, (px, py + 1), quarters),
            gsb._reflect_x(first, px),
            _turn(
                first, (min(x for x, _ in first), min(y for _, y in first)), quarters
            ),  # about the image's own corner
        ]
    if answer & tile or answer & first or first & tile or first == tile:
        return None
    fitted = _fit([tile, first, answer, {p} if p else set()], rng=rng)
    if fitted is None:
        return None
    width, height, sx, sy = fitted
    groups = [
        {"cells": [list(c) for c in sorted(gsb._shift(tile, sx, sy))], "style": "shaded"},
        {"cells": [list(c) for c in sorted(gsb._shift(first, sx, sy))], "style": "hatched"},
    ]
    fig = grid_drawing(
        alt=(
            "A square grid with a shape in shaded squares, the same shape moved to a new place in hatched squares, "
            "and a point marked P."
            if p
            else "A square grid with a shape in shaded squares and the same shape moved to a new place in hatched squares."
        ),
        grid={"type": "square", "width": width, "height": height},
        cells=groups,
        labels=[{"at": [p[0] + sx, p[1] + sy], "text": "P"}] if p else [],
        canvas="grid",
    )
    ending = "Do not shade the squares that are already drawn, which are given."
    text = phrase(
        rng,
        (
            "The shaded shape moved to the hatched shape with one move. {point}Do the same move again to the "
            "hatched shape and shade where it lands. {ending}",
            "One move takes the shaded shape to the hatched shape. {point}Make the same move once more, starting "
            "from the hatched shape, and shade the result. {ending}",
            "The hatched shape is the shaded shape after one move. {point}Shade the shape you get when the hatched "
            "shape makes the same move. {ending}",
        ),
        point="If the move was a turn, it was about the point P. " if p else "",
        ending=ending,
    )
    shifted = [gsb._shift(c, sx, sy) for c in candidates]
    if move[0] == "slide":
        working = [
            "The move is a slide: every square goes the same number of squares across and up or down.",
            "Slide the hatched shape by that same amount again.",
        ]
    else:
        working = [
            "The shaded shape and the hatched shape face different ways, so the move is a turn about P.",
            "Turn the hatched shape about P by the same amount, in the same direction.",
        ]
    return gsb._cells_question(
        fig, text, sorted(gsb._shift(answer, sx, sy)), (width, height), "exact", shifted, working
    )


@generator(
    "grid_shape_turns",
    level=5,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_shape_turns(rng, tier="easy"):
    """Shade a shape after a half turn, after a quarter turn, or after repeating a move read from a picture."""
    if tier in ("easy", "medium"):
        return gsb._retry(rng, lambda r: _turn_question(r, tier), f"a {tier} turn")
    if tier == "hard":
        return gsb._retry(rng, _repeat_question, "a repeated move")
    raise ValueError(f"unknown tier {tier!r}")


_QUARTERS = {"half turn": 2, "quarter turn anticlockwise": 1, "quarter turn clockwise": 3}


def _read_turns(question) -> str:
    fig, text = question.figure, question.text
    groups = [{tuple(c) for c in g["cells"]} for g in fig["cells"]]
    at = {lb["text"]: (int(lb["at"][0]), int(lb["at"][1])) for lb in fig["labels"]} if fig.get("labels") else {}
    if len(groups) == 1:
        word = next(w for w in _QUARTERS if w in text)
        return cell_text(_turn(groups[0], at["P"], _QUARTERS[word]))
    tile, first = groups
    if normalised(first) == normalised(tile):
        dx = min(x for x, _ in first) - min(x for x, _ in tile)
        dy = min(y for _, y in first) - min(y for _, y in tile)
        return cell_text(gsb._shift(first, dx, dy))
    quarters = next(k for k in (1, 3) if _turn(tile, at["P"], k) == first)
    return cell_text(_turn(first, at["P"], quarters))


figures.register_reader("grid_shape_turns", _read_turns)

# ------------------------------------------------------------------------------------------- Year 6: box nets

_ORDINAL = {0: "first", 1: "second", 2: "third", 3: "fourth"}


def _box_net(a: int, b: int, h: int, top_on: int, bottom_on: int) -> tuple[set[Cell], set[Cell], set[Cell], list[int]]:
    """The strip of four side faces (widths a, b, a, b, each h tall), the top joined above face ``top_on`` and the
    bottom joined below face ``bottom_on``, with the strip's lower left corner at (0, 0)."""
    widths = [a, b, a, b]
    starts = [sum(widths[:i]) for i in range(4)]
    strip = {(x, y) for x in range(2 * (a + b)) for y in range(h)}
    top = {(starts[top_on] + dx, h + dy) for dx in range(widths[top_on]) for dy in range(a + b - widths[top_on])}
    bottom = {
        (starts[bottom_on] + dx, -1 - dy) for dx in range(widths[bottom_on]) for dy in range(a + b - widths[bottom_on])
    }
    return strip, top, bottom, starts


_NET_BLANK_ALT = "A blank square grid to shade the net of a box on."
_NET_GIVEN_ALT = (
    "A square grid with the net of a box drawn in hatched squares, a row of four side faces numbered with a top "
    "joined to one of them, and the bottom face missing."
)


def _net_distractors(rng, a, b, h, top_on, bottom_on, strip, top, bottom):
    """Wrong nets for the stated layout, each a real slip: a face the wrong shape for the face it is joined to, a face
    left out, the side faces in the wrong order, or a face joined to the wrong side face."""
    out = []
    # the top as wide as the other pair of faces, so it does not fit the edge it is joined to
    widths = [a, b, a, b]
    starts = [sum(widths[:i]) for i in range(4)]
    wrong_w = a + b - widths[top_on]
    out.append(
        strip | {(starts[top_on] + dx, h + dy) for dx in range(wrong_w) for dy in range(widths[top_on])} | bottom
    )
    wrong_wb = a + b - widths[bottom_on]
    out.append(
        strip | top | {(starts[bottom_on] + dx, -1 - dy) for dx in range(wrong_wb) for dy in range(widths[bottom_on])}
    )
    out.append(strip | bottom)  # no top
    out.append(strip | top)  # no bottom
    # side faces in the wrong order: a, a, b, b
    order = [a, a, b, b]
    o_strip = {(x, y) for x in range(2 * (a + b)) for y in range(h)}
    o_starts = [sum(order[:i]) for i in range(4)]
    out.append(
        o_strip
        | {(o_starts[0] + dx, h + dy) for dx in range(order[0]) for dy in range(a + b - order[0])}
        | {(o_starts[0] + dx, -1 - dy) for dx in range(order[0]) for dy in range(a + b - order[0])}
    )
    # the top joined to the neighbouring side face instead, moved with its size
    other = (top_on + 1) % 4
    out.append(
        strip
        | {(starts[other] + dx, h + dy) for dx in range(widths[other]) for dy in range(a + b - widths[other])}
        | bottom
    )
    # the box standing the wrong height
    if h > 1:
        short = {(x, y) for x in range(2 * (a + b)) for y in range(h - 1)}
        out.append(short | {(x, y - 1) for x, y in top} | {(x, y + 1) for x, y in bottom})
    return out


def _net_layout_text(a, b, h, top_on, bottom_on) -> str:
    where = (
        f"both joined to the {_ORDINAL[top_on]} face"
        if top_on == bottom_on
        else f"the top joined above the {_ORDINAL[top_on]} face and the bottom joined below the {_ORDINAL[bottom_on]} face"
    )
    return (
        f"The four side faces make a row, starting with a face {gsb._plural(a, 'square')} wide, then "
        f"one {b} wide, then {a} wide, then {b} wide, each {gsb._plural(h, 'square')} tall. The net has {where}."
    )


def _net_blank(rng, tier: str):
    if tier == "easy":
        a, b, h = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 2)
        top_on = bottom_on = 0
    else:
        a, b, h = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 3)
        if a == b:
            return None
        top_on, bottom_on = rng.randrange(4), rng.randrange(4)
        if rng.random() < 0.7 and top_on == bottom_on:
            bottom_on = (bottom_on + rng.choice((1, 2, 3))) % 4
    strip, top, bottom, starts = _box_net(a, b, h, top_on, bottom_on)
    net = strip | top | bottom
    lo_y = min(y for _, y in net)
    below = -lo_y
    full_w = max(x for x, _ in net) + 1
    full_h = max(y for _, y in net) - lo_y + 1
    slack_w, slack_h = rng.randint(1, 2), rng.randint(1, 2)
    cw, ch = full_w + slack_w, full_h + slack_h
    if cw > 20 or ch > 20:
        return None
    ox, oy = rng.randint(0, slack_w), rng.randint(0, slack_h) + below
    placed = gsb._shift(net, ox, oy)
    cands = [gsb._shift(c, ox, oy) for c in _net_distractors(rng, a, b, h, top_on, bottom_on, strip, top, bottom)]
    ctx = (
        f"Shade the net of a box on the grid, which is {cw} squares wide and {gsb._plural(ch, 'square')} tall. "
        f"The box has a base {gsb._plural(a, 'square')} by {gsb._plural(b, 'square')} and is {gsb._plural(h, 'square')} tall. "
    )
    text = (
        ctx
        + _net_layout_text(a, b, h, top_on, bottom_on)
        + " Do not turn or flip the net. It can go anywhere on the grid."
    )
    fig = grid_drawing(
        alt=_NET_BLANK_ALT,
        grid={"type": "square", "width": cw, "height": ch},
        canvas="grid",
    )
    working = [
        "Each face of the box is a rectangle of squares. A side face joined to the top or bottom has the same width as "
        "the edge of the top or bottom it meets.",
        f"The faces in the row are {a}, {b}, {a} and {b} wide, in squares. The top joined to the {_ORDINAL[top_on]} face "
        f"is {[a, b, a, b][top_on]} wide and {a + b - [a, b, a, b][top_on]} tall, in squares.",
    ]
    return gsb._cells_question(fig, text, sorted(placed), (cw, ch), "translate", cands, working)


def _net_partial(rng):
    a, b = rng.sample(range(1, 6), 2)
    h = rng.randint(1, 3)
    top_on, bottom_on = rng.randrange(4), rng.randrange(4)
    widths = [a, b, a, b]
    if widths[top_on] == widths[bottom_on] and rng.random() < 0.7:
        bottom_on = (bottom_on + 1) % 4
    strip, top, bottom, starts = _box_net(a, b, h, top_on, bottom_on)
    room = max(a, b)
    width, height = 2 * (a + b), h + 2 * room
    sx, sy = 0, room
    given = gsb._shift(strip | top, sx, sy)
    answer = gsb._shift(bottom, sx, sy)
    labels = [{"at": [starts[i] + widths[i] / 2 + sx, h / 2 + sy], "text": f"Face {i + 1}"} for i in range(4)]
    fig = grid_drawing(
        alt=_NET_GIVEN_ALT,
        grid={"type": "square", "width": width, "height": height},
        cells=[{"cells": [list(c) for c in sorted(given)], "style": "hatched"}],
        labels=labels,
        canvas="grid",
    )
    text = phrase(
        rng,
        (
            "This net of a box has no bottom. The bottom joins the lower edge of face {n}. Shade the bottom. The "
            "hatched squares are given, so do not shade them.",
            "The hatched squares are the net of a box without its bottom face. Shade the bottom so it is joined "
            "below face {n} and the net folds into a closed box. Do not shade the hatched squares.",
            "A box net is drawn in hatched squares, with the bottom missing. Shade the missing bottom, joined to "
            "the lower edge of face {n}. The hatched squares are given.",
        ),
        n=bottom_on + 1,
    )
    wb = widths[bottom_on]
    other = a + b - wb
    cands = []
    tw = widths[top_on]
    same_as_top = {(starts[bottom_on] + dx, -1 - dy) for dx in range(tw) for dy in range(a + b - tw)}
    cands.append(gsb._shift(same_as_top, sx, sy))  # the top's size, whatever face it joins
    cands.append(gsb._shift({(starts[bottom_on] + dx, -1 - dy) for dx in range(other) for dy in range(wb)}, sx, sy))
    cands.append(gsb._shift({(starts[bottom_on] + dx, -1 - dy) for dx in range(wb) for dy in range(h)}, sx, sy))
    nxt = (bottom_on + 1) % 4
    cands.append(
        gsb._shift(
            {(starts[nxt] + dx, -1 - dy) for dx in range(widths[nxt]) for dy in range(a + b - widths[nxt])}, sx, sy
        )
    )
    cands.append(gsb._shift({(starts[bottom_on] + 1 + dx, -1 - dy) for dx in range(wb) for dy in range(other)}, sx, sy))
    cands.append(gsb._shift({(x, y + 1) for x, y in bottom}, sx, sy))  # a gap between the bottom and the face
    working = [
        f"Face {bottom_on + 1} is {wb} wide, so the bottom must be {wb} wide where it joins (all in squares).",
        f"The top is {tw} wide and {a + b - tw} tall, so the base is {a} by {b} and the bottom is {wb} wide and "
        f"{other} tall, in squares.",
    ]
    return gsb._cells_question(fig, text, sorted(answer), (width, height), "exact", cands, working)


@generator(
    "grid_prism_nets",
    level=6,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_prism_nets(rng, tier="easy"):
    """Shade the net of a box in a stated layout, or the missing bottom of a box net."""
    if tier in ("easy", "medium"):
        return gsb._retry(rng, lambda r: _net_blank(r, tier), f"a {tier} box net")
    if tier == "hard":
        return gsb._retry(rng, _net_partial, "a missing bottom")
    raise ValueError(f"unknown tier {tier!r}")


_WIDE = re.compile(
    r"starting with a face (\d+) squares? wide, then one (\d+) wide, then \d+ wide, then \d+ wide, each (\d+)"
)
_JOIN_BOTH = re.compile(r"both joined to the (first|second|third|fourth) face")
_JOIN_EACH = re.compile(
    r"top joined above the (first|second|third|fourth) face and the bottom joined below the (first|second|third|fourth) face"
)
_FACE_N = re.compile(r"face (\d)")
_FROM_WORD = {v: k for k, v in _ORDINAL.items()}


def _read_nets(question) -> str:
    fig, text = question.figure, question.text
    if "cells" not in fig:
        a, b, h = (int(v) for v in _WIDE.search(text).groups())
        if (m := _JOIN_BOTH.search(text)) is not None:
            top_on = bottom_on = _FROM_WORD[m.group(1)]
        else:
            m = _JOIN_EACH.search(text)
            top_on, bottom_on = _FROM_WORD[m.group(1)], _FROM_WORD[m.group(2)]
        strip, top, bottom, _starts = _box_net(a, b, h, top_on, bottom_on)
        net = strip | top | bottom
        return cell_text(gsb._shift(net, 0, -min(y for _, y in net)))
    given = {tuple(c) for c in fig["cells"][0]["cells"]}
    counts: dict[int, int] = {}
    for _, y in given:
        counts[y] = counts.get(y, 0) + 1
    perimeter = max(counts.values())
    strip_rows = sorted(y for y, n in counts.items() if n == perimeter)
    h = len(strip_rows)
    top_rows = sorted(y for y, n in counts.items() if n < perimeter)
    top_cells = [(x, y) for x, y in given if y in top_rows]
    top_w = len({x for x, _ in top_cells})
    top_x = min(x for x, _ in top_cells)
    half = perimeter // 2
    # which face the top is joined to: the faces start 0, a, a + b, 2a + b; the top starts at one of them
    face_n = int(_FACE_N.search(text).group(1)) - 1
    for first in range(1, half):
        widths = [first, half - first, first, half - first]
        starts = [sum(widths[:i]) for i in range(4)]
        if top_x in starts and widths[starts.index(top_x)] == top_w:
            break
    wb, base = widths[face_n], strip_rows[0]
    tall = half - wb
    return cell_text({(starts[face_n] + dx, base - 1 - dy) for dx in range(wb) for dy in range(tall)})


figures.register_reader("grid_prism_nets", _read_nets)

# ------------------------------------------------------------------------------------------- senior: latitude and longitude


def _lat(v: int) -> str:
    return "0°" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"


def _lon(v: int) -> str:
    return "0°" if v == 0 else f"{abs(v)}°{'E' if v > 0 else 'W'}"


_ALT_MAP = (
    "A map grid with a line every ten degrees, the latitudes written up the left side and the longitudes along "
    "the top, and places marked with letters."
)


def _map_figure(lat0: int, lon0: int, w: int, h: int, places: dict[str, tuple[int, int]], every: int = 1):
    """A grid with its lines labelled and the places, ``{letter: (latitude, longitude)}``, centred on their crossings.

    Every line is labelled for latitude up the left side. Longitude is written along the top for every ``every``-th
    line, counting from the right, so a wide map keeps its labels apart."""
    labels = [{"at": [0, y], "text": _lat(lat0 + 10 * y)} for y in range(h + 1)]
    labels += [{"at": [x, h], "text": _lon(lon0 + 10 * x)} for x in range(w, 0, -every)]
    for letter, (la, lo) in places.items():
        labels.append({"at": [(lo - lon0) // 10, (la - lat0) // 10], "text": letter})
    return grid_drawing(alt=_ALT_MAP, grid={"type": "square", "width": w, "height": h}, labels=labels)


def _spots(rng, lat0, lon0, w, h, n):
    """``n`` places on different latitudes and different longitudes, away from the edge, as ``(lat, lon)``."""
    ys = rng.sample(range(1, h), n)
    xs = rng.sample(range(1, w), n)
    return [(lat0 + 10 * y, lon0 + 10 * x) for x, y in zip(xs, ys)]


def _latlong_easy(rng):
    la_sign, lo_sign = rng.choice(((-1, 1), (1, 1), (1, -1)))
    h, w = 4, 6
    lat0 = -10 * h - 10 * rng.randint(0, 1) if la_sign < 0 else 10 * rng.randint(0, 2)
    lon0 = 100 + 10 * rng.randint(0, 4) if lo_sign > 0 else -(100 + 10 * rng.randint(0, 4)) - 10 * w
    spots = _spots(rng, lat0, lon0, w, h, 3)
    letters = "ABC"
    places = dict(zip(letters, spots))
    ask = rng.choice(letters)
    la, lo = places[ask]
    fig = _map_figure(lat0, lon0, w, h, places)
    text = phrase(
        rng,
        (
            "Each square on the map grid is 10 degrees. What are the latitude and longitude of place {p}, written "
            "with the latitude first?",
            "A map has a line every 10 degrees. Give the position of place {p} as latitude, then longitude.",
            "Find place {p} on the map grid. Which choice gives its latitude, then its longitude?",
        ),
        p=ask,
    )
    answer = f"{_lat(la)}, {_lon(lo)}"
    candidates = [
        f"{_lon(lo)}, {_lat(la)}",  # longitude first
        f"{_lat(-la)}, {_lon(lo)}" if la else f"{_lat(10)}, {_lon(lo)}",  # the wrong hemisphere for the latitude
        f"{_lat(la)}, {_lon(-lo)}" if lo else f"{_lat(la)}, {_lon(10)}",  # the wrong side of the prime meridian
        f"{_lat(la + 10)}, {_lon(lo)}",  # a line too far north
        f"{_lat(la)}, {_lon(lo - 10)}",  # a line too far west
        f"{_lat(la - 10)}, {_lon(lo + 10)}",
    ]
    rng.shuffle(candidates)  # every slip gets its turn as one of the three wrong options
    working = [
        "Latitude is read up the side of the map (north or south of the equator), longitude along the top (east or west of the prime meridian).",
        f"Place {ask} is on the {_lat(la)} line across and the {_lon(lo)} line along, so it is {answer}.",
    ]
    return _text_question(fig, text, answer, candidates, working)


def _latlong_medium(rng):
    la, lo = rng.choice((10, 20)) * rng.choice((-1, 1)), rng.choice((10, 20)) * rng.choice((-1, 1))
    if abs(la) == abs(lo):
        return None
    right = (la, lo)
    wrongs = [(-la, lo), (la, -lo), (lo, la)]  # the wrong hemisphere for either one, and the two swapped
    letters = list("ABCD")
    rng.shuffle(letters)
    places = dict(zip(letters, [right, *wrongs]))
    ask = letters[0]
    fig = _map_figure(-30, -30, 6, 6, places)
    text = phrase(
        rng,
        (
            "Each square on the map grid is 10 degrees. Which place is at {la}, {lo}?",
            "Find the place at latitude {la} and longitude {lo}. Each square is 10 degrees.",
            "Four places are marked on the map grid, which has a line every 10 degrees. Which one is at {la}, {lo}?",
        ),
        la=_lat(la),
        lo=_lon(lo),
    )
    answer = f"Place {ask}"
    candidates = [f"Place {letter}" for letter in letters[1:]]
    working = [
        f"{_lat(la)} is {abs(la) // 10} line{'s' if abs(la) > 10 else ''} {'north' if la > 0 else 'south'} of the equator, "
        f"and {_lon(lo)} is {abs(lo) // 10} line{'s' if abs(lo) > 10 else ''} {'east' if lo > 0 else 'west'} of the prime meridian.",
        f"The place where those two lines cross is Place {ask}.",
    ]
    return _text_question(fig, text, answer, candidates, working)


_HARD_LAT0, _HARD_LON0, _HARD_H, _HARD_W = -50, -60, 10, 12


def _opposite(rng):
    """Two places on opposite sides of both the equator and the prime meridian, at least 70 degrees apart each way."""

    def pair(limit: int):
        sums = [(p, q) for p in range(10, limit + 1, 10) for q in range(10, limit + 1, 10) if p + q >= 70]
        p, q = rng.choice(sums)
        sign = rng.choice((-1, 1))
        return sign * p, -sign * q

    (la_a, la_b), (lo_a, lo_b) = pair(40), pair(50)
    return (la_a, lo_a), (la_b, lo_b)


def _latlong_hard(rng):
    a, b = _opposite(rng)
    fig = _map_figure(_HARD_LAT0, _HARD_LON0, _HARD_W, _HARD_H, {"A": a, "B": b}, every=2)
    on_lon = rng.random() < 0.5
    i = 1 if on_lon else 0
    diff, other = abs(a[i] - b[i]), abs(a[1 - i] - b[1 - i])
    what = "longitude" if on_lon else "latitude"
    text = phrase(
        rng,
        (
            "Each square on the map grid is 10 degrees. How many degrees of {what} are there between place A and "
            "place B?",
            "Place A and place B are marked on the map grid, which has a line every 10 degrees. What is the "
            "difference in {what} between A and B, in degrees?",
            "Find the number of degrees of {what} from A to B. The lines on the map grid are 10 degrees apart.",
        ),
        what=what,
    )
    smaller = abs(abs(a[i]) - abs(b[i]))
    candidates = [f"{v}°" for v in (smaller, diff // 10, diff + 10, other, diff - 10) if v > 0]
    working = [
        f"A is {_lat(a[0]) if not on_lon else _lon(a[1])} and B is {_lat(b[0]) if not on_lon else _lon(b[1])}.",
        f"They are on opposite sides of the {'prime meridian' if on_lon else 'equator'}, so add the two: {abs(a[i])} + {abs(b[i])} = {diff}.",
    ]
    return _text_question(fig, text, f"{diff}°", candidates, working)


def _read_map(fig):
    lat_at, lon_at, places = {}, {}, {}
    for lb in fig["labels"]:
        x, y, t = lb["at"][0], lb["at"][1], lb["text"]
        if len(t) == 1:
            places[t] = (x, y)
        elif x == 0:
            lat_at[y] = _parse_deg(t)
        else:
            lon_at[x] = _parse_deg(t)
    lat0 = lat_at[0]
    x_any, lon_any = next(iter(lon_at.items()))
    lon0 = lon_any - 10 * x_any
    return {t: (lat0 + 10 * y, lon0 + 10 * x) for t, (x, y) in places.items()}


def _parse_deg(t: str) -> int:
    n = int(t.split("°")[0])
    return -n if t.endswith(("S", "W")) else n


def _read_latlong(question) -> str:
    fig, text = question.figure, question.text
    places = _read_map(fig)
    if "latitude first" in text or "latitude, then" in text or "then its longitude" in text:
        letter = re.search(r"place ([ABC])", text).group(1)
        la, lo = places[letter]
        return f"{_lat(la)}, {_lon(lo)}"
    if "Which place is at" in text or "Which one is at" in text or "Find the place at latitude" in text:
        m = re.search(r"(\d+°[NS]|0°)(?:,| and longitude) (?:longitude )?(\d+°[EW]|0°)", text)
        la, lo = _parse_deg(m.group(1)), _parse_deg(m.group(2))
        return "Place " + next(t for t, p in places.items() if p == (la, lo))
    on_lon = "longitude" in text
    (a, b) = places["A"], places["B"]
    return f"{abs(a[1] - b[1]) if on_lon else abs(a[0] - b[0])}°"


_LATLONG_TIERS = {"easy": _latlong_easy, "medium": _latlong_medium, "hard": _latlong_hard}


@generator(
    "grid_latlong_places",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_latlong_places(rng, tier="easy"):
    """Read latitude and longitude off a map grid, find the place at given coordinates, or the degrees between two."""
    return gsb._retry(rng, _tier(_LATLONG_TIERS, tier), f"a {tier} latitude and longitude question")


figures.register_reader("grid_latlong_places", _read_latlong)


@generator("grid_latlong_exam", level=11, presets={"exam": {}})
def grid_latlong_exam(rng):
    """The distance north-south between two places on a map grid, from their latitudes and 111 km a degree."""

    def build(r):
        a, b = _opposite(r)
        fig = _map_figure(_HARD_LAT0, _HARD_LON0, _HARD_W, _HARD_H, {"A": a, "B": b}, every=2)
        d_lon, d_lat = abs(a[1] - b[1]), abs(a[0] - b[0])
        km = 111 * d_lat
        text = phrase(
            r,
            (
                "Each square of the map grid is 10 degrees. A plane flies due {way} along a line of longitude from "
                "the latitude of place A to the latitude of place B. Taking 1 degree of latitude as 111 km, how far "
                "does it fly?",
                "Places A and B are marked on a map grid with a line every 10 degrees. A plane flies from the "
                "latitude of A to the latitude of B, due {way} along one line of longitude. If 1 degree of latitude "
                "is 111 km, how far does it fly?",
                "A map grid has a line every 10 degrees and shows places A and B. Taking 1 degree of latitude as "
                "111 km, find the distance flown by a plane going due {way} along a line of longitude from the "
                "latitude of A to the latitude of B.",
            ),
            way="south" if a[0] > b[0] else "north",
        )
        smaller_lat = abs(abs(a[0]) - abs(b[0]))
        candidates = [f"{111 * v} km" for v in (smaller_lat, d_lon, d_lat // 10, d_lat + 10, d_lat - 10) if v > 0]
        r.shuffle(candidates)  # every slip gets its turn as one of the three wrong options
        working = [
            f"A is {_lat(a[0])} and B is {_lat(b[0])}, on opposite sides of the equator, so add: "
            f"{abs(a[0])} + {abs(b[0])} = {d_lat} degrees of latitude.",
            f"{d_lat} x 111 = {km} km.",
        ]
        return _text_question(fig, text, f"{km} km", candidates, working)

    return gsb._retry(rng, build, "a latitude and longitude exam question")


def _read_latlong_exam(question) -> str:
    places = _read_map(question.figure)
    a, b = places["A"], places["B"]
    return f"{111 * abs(a[0] - b[0])} km"


figures.register_reader("grid_latlong_exam", _read_latlong_exam)
