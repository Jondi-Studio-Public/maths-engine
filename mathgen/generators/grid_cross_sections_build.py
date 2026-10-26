"""Cross-sections of solids built from unit cubes, drawn on isometric dot paper (``grid_drawing`` figures).

One topic, ``grid_solid_cross_sections`` (Year 6, ``grid_cells``, match ``translate``). A solid is drawn on isometric
dot paper and cut by a flat plane parallel to one of its faces, through a stated layer of cubes. The learner shades
the cut face on a blank canvas, as seen straight on, so the answer is a drawn shape and is not a pick from four.

Easy cuts a rectangular prism parallel to a named face: the two edges of that face are read from the picture.
Medium cuts a prism whose end is an L-shape, parallel to that end (a horizontal cut of an L-shaped footprint, or a
cut parallel to an L-shaped front face): every parallel cut is the same L, whichever layer it passes through.
Hard cuts a prism with a stepped end parallel to a face where the cut is not the end shape and depends on where
it passes: a cut parallel to the side through one column, a horizontal cut through one layer, or two cuts
parallel to the side compared, shading the one with the greater or the smaller area.

The figure reader re-derives the cut face from the cubes in the figure and the layer named in the wording, by
filtering the cubes on one coordinate. The generator builds the answer from the solid's measurements instead.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, generator, phrase
from ..figure_kinds.grid import grid_drawing, tops_all_visible
from ..grading.grid import cell_text
from . import grid_shading_build as gsb

Cell = tuple[int, int]

CANVAS = 6
_ORD = ("first", "second", "third", "fourth", "fifth")
_FROM = {"front": "front", "right": "left", "top": "bottom"}
_VIEW = {
    "front": "from the front",
    "right": "from the right, with the front of the solid on your left",
    "top": "from above, with the front of the solid at the bottom of the canvas",
}
_ALT = "A solid made of unit cubes on isometric dot paper, with a blank grid beside it to draw on."
_TIERS = ("easy", "medium", "hard")


def _rect(a: int, b: int) -> set[Cell]:
    return {(i, j) for i in range(a) for j in range(b)}


def _mirror_x(cells: set[Cell]) -> set[Cell]:
    top = max(x for x, _ in cells)
    return {(top - x, y) for x, y in cells}


def _mirror_y(cells: set[Cell]) -> set[Cell]:
    top = max(y for _, y in cells)
    return {(x, top - y) for x, y in cells}


def _pin(rng) -> str:
    return phrase(
        rng,
        (
            "The canvas is {n} cells wide and {n} cells tall. Draw the shape anywhere on it, without turning it.",
            "Use the canvas, which is {n} cells wide and {n} cells tall. The shape can go anywhere on it, but do not turn it.",
            "Draw it on the canvas ({n} cells wide, {n} cells tall), anywhere you like and the right way round.",
        ),
        n=CANVAS,
    )


def _section(cubes, face: str, k: int) -> set[Cell]:
    """The cut face: the cubes of one layer, as the cells seen looking straight at the cut."""
    if face == "front":
        return {(x, z) for x, y, z in cubes if y == k}
    if face == "right":
        return {(y, z) for x, y, z in cubes if x == k}
    if face == "top":
        return {(x, y) for x, y, z in cubes if z == k}
    raise ValueError(f"unknown face {face!r}")


def _place(rng, answer: set[Cell], candidates):
    """Shift the answer and every candidate by one offset that keeps the answer on the canvas."""
    ax = max(x for x, _ in answer) + 1
    ay = max(y for _, y in answer) + 1
    ox, oy = rng.randint(0, CANVAS - ax), rng.randint(0, CANVAS - ay)
    seen = set()
    out = []
    for cand in candidates:
        if not cand:
            continue
        low_x, low_y = min(x for x, _ in cand), min(y for _, y in cand)
        key = frozenset((x - low_x, y - low_y) for x, y in cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(gsb._shift(cand, ox, oy))
    return gsb._shift(answer, ox, oy), out


def _build(rng, cubes, dims, face, k, answer, candidates, cut_text, working):
    w, d, top = dims
    if len(cubes) > 64 or not tops_all_visible(cubes):
        return None
    fig = grid_drawing(
        alt=_ALT,
        grid={"type": "isometric", "width": w, "height": d, "layers": top},
        cubes=cubes,
        canvas={"width": CANVAS, "height": CANVAS},
    )
    cells, cands = _place(rng, answer, candidates)
    text = cut_text + " " + _pin(rng)
    return gsb._cells_question(fig, text, sorted(cells), (CANVAS, CANVAS), "translate", cands, working)


def _cut_text(rng, face: str, k: int) -> str:
    return phrase(
        rng,
        (
            "A solid made of unit cubes is drawn on isometric dot paper. A flat cut parallel to its {face} face goes "
            "through the {ord} layer of cubes, counting from the {frm}. Shade the cut face as seen {view}.",
            "This solid is built from unit cubes. Slice it with a plane parallel to its {face} face, through the "
            "{ord} layer of cubes, counting from the {frm}. On the canvas, shade the cut face as seen {view}.",
            "Cut the solid in the picture parallel to its {face} face, through the {ord} layer of cubes, counting "
            "from the {frm}. Draw the new flat surface on the canvas as seen {view}.",
        ),
        face=face,
        ord=_ORD[k],
        frm=_FROM[face],
        view=_VIEW[face],
    )


def _box_cubes(w, d, h, heights=None):
    if heights is None:
        heights = [h] * w
    return [[x, y, z] for x in range(w) for y in range(d) for z in range(heights[x])]


# ---------------------------------------------------------------------------------------------- easy: box


def _easy(rng):
    w, d, h = (rng.randint(2, 4) for _ in range(3))
    face = rng.choice(("front", "right", "top"))
    dims = {"front": (w, h), "right": (d, h), "top": (w, d)}
    a, b = dims[face]
    if a == b:
        return None
    k = rng.randrange({"front": d, "right": w, "top": h}[face])
    answer = _rect(a, b)
    others = [_rect(*dims[f]) for f in dims if f != face]
    cands = [_rect(b, a), *others, _rect(a + 1, b), _rect(a, b + 1), _rect(a - 1, b), _rect(a, b - 1)]
    working = [
        "A box is a prism, so a cut parallel to a face has the same size and shape as that face.",
        f"The {face} face is {a} cells across and {b} cells up, so the cut face is a rectangle {a} by {b}.",
    ]
    cubes = _box_cubes(w, d, h)
    return _build(rng, cubes, (w, d, h), face, k, answer, cands, _cut_text(rng, face, k), working)


# ---------------------------------------------------------------------------------------------- medium: L prisms


def _medium_footprint(rng):
    """An L-shaped footprint standing H cubes tall, cut by a horizontal plane."""
    w, d = rng.randint(3, 5), rng.randint(3, 4)
    nw, nd = rng.randint(1, w - 1), rng.randint(1, d - 1)
    height = rng.randint(2, 3)
    corner = rng.choice(("fl", "fr", "bl", "br"))

    def notch(c, nw=nw, nd=nd):
        xs = range(nw) if c[1] == "l" else range(w - nw, w)
        ys = range(nd) if c[0] == "f" else range(d - nd, d)
        return {(x, y) for x in xs for y in ys}

    shape = _rect(w, d) - notch(corner)
    k = rng.randrange(height)
    cubes = [[x, y, z] for (x, y) in sorted(shape) for z in range(height)]
    cands = [_mirror_x(shape), _mirror_y(shape), _rect(w, d)]
    cands += [_rect(w, d) - notch(c) for c in ("fl", "fr", "bl", "br") if c != corner]
    cands += [_rect(w, d) - notch(corner, nw=nw + 1) if nw + 1 < w else _rect(w, d) - notch(corner, nd=nd - 1 or 1)]
    working = [
        "The solid is a prism with an L-shaped end standing on the table, so a cut parallel to the base has the "
        "same shape as the base.",
        "Every layer of cubes has the same outline, so it does not matter which layer the cut goes through. "
        "Shade the L as it is seen from above.",
    ]
    return _build(rng, cubes, (w, d, height), "top", k, shape, cands, _cut_text(rng, "top", k), working)


def _medium_end(rng):
    """A prism with an L-shaped front face, cut parallel to that face."""
    w, d = rng.randint(3, 5), rng.randint(2, 4)
    tall = rng.randint(3, 4)
    low = rng.randint(1, tall - 1)
    a = rng.randint(1, w - 1)
    heights = [tall] * a + [low] * (w - a)
    shape = {(x, z) for x in range(w) for z in range(heights[x])}
    k = rng.randrange(d)
    cubes = _box_cubes(w, d, tall, heights)
    cands = [
        _mirror_x(shape),
        _mirror_y(shape),
        _rect(w, tall),
        _rect(d, tall),
        {(x, z) for x in range(w) for z in range(([tall] * (a + 1) + [low] * w)[x])} if a + 1 < w else None,
        {(x, z) for x in range(w) for z in range(([low] * a + [tall] * w)[x])},
    ]
    cands = [c for c in cands if c]
    working = [
        "The solid is a prism with an L-shaped front face, so a cut parallel to the front face has the shape of "
        "the front face.",
        "Every layer from the front to the back is the same L, so shade it as seen from the front.",
    ]
    return _build(rng, cubes, (w, d, tall), "front", k, shape, cands, _cut_text(rng, "front", k), working)


# ---------------------------------------------------------------------------------------------- hard: stepped prism


def _steps(rng):
    for _ in attempts(200, "a stepped end"):
        w, top = rng.randint(3, 5), rng.randint(3, 4)
        heights = [top]
        for _x in range(w - 1):
            heights.append(rng.randint(1, heights[-1]))
        if len(set(heights)) >= 3:
            return w, top, heights
    raise AssertionError("unreachable")


def _hard_column(rng, w, d, top, heights):
    """A cut parallel to the right face, through one column: a rectangle as tall as that column."""
    k = rng.choice([i for i in range(1, w) if heights[i] < top])
    cut = heights[k]
    answer = _rect(d, cut)
    cands = [
        _rect(d, top),
        _rect(d, heights[w - 1 - k]),
        _rect(d, heights[k - 1]),
        _rect(cut, d),
        _rect(w, cut),
        _rect(d, cut + 1),
    ]
    if k + 1 < w:
        cands.append(_rect(d, heights[k + 1]))
    working = [
        f"The solid has steps along its front, so a cut parallel to the right face is not the same everywhere: "
        f"it is as tall as the column it goes through.",
        f"Column {k + 1} from the left is {gsb._plural(cut, 'cube')} tall and {gsb._plural(d, 'cube')} deep, so the cut face is a rectangle "
        f"{gsb._plural(d, 'cell')} across and {gsb._plural(cut, 'cell')} up.",
    ]
    cubes = _box_cubes(w, d, top, heights)
    return _build(rng, cubes, (w, d, top), "right", k, answer, cands, _cut_text(rng, "right", k), working)


def _hard_layer(rng, w, d, top, heights):
    """A horizontal cut through one layer: only the columns at least that tall are cut."""
    options = [z for z in range(1, top) if sum(1 for v in heights if v > z) < w]
    if not options:
        return None
    z = rng.choice(options)
    count = sum(1 for v in heights if v > z)
    answer = _rect(count, d)
    lower = sum(1 for v in heights if v > z - 1)
    cands = [
        _rect(w, d),
        _rect(lower, d),
        _rect(d, count),
        _rect(count + 1, d),
        _rect(count, d - 1),
        _rect(heights[0], d) if heights[0] != count else None,
    ]
    cands = [c for c in cands if c]
    working = [
        "The solid has steps, so a horizontal cut is not the whole base: it only goes through the columns that "
        "reach that layer.",
        f"{count} of the columns reach this layer and each is {gsb._plural(d, 'cube')} deep, so the cut face is a rectangle "
        f"{gsb._plural(count, 'cell')} across and {gsb._plural(d, 'cell')} up.",
    ]
    cubes = _box_cubes(w, d, top, heights)
    return _build(rng, cubes, (w, d, top), "top", z, answer, cands, _cut_text(rng, "top", z), working)


def _hard_compare(rng, w, d, top, heights):
    """Two cuts parallel to the right face: shade the one with the greater or smaller area."""
    pairs = [(i, j) for i in range(w) for j in range(w) if i != j and heights[i] != heights[j]]
    first, second = rng.choice(pairs)
    pick = rng.choice(("greater", "smaller"))
    ha, hb = heights[first], heights[second]
    cut = max(ha, hb) if pick == "greater" else min(ha, hb)
    wrong = min(ha, hb) if pick == "greater" else max(ha, hb)
    answer = _rect(d, cut)
    cands = [_rect(d, wrong), _rect(cut, d), _rect(d, ha + hb), _rect(d, top), _rect(wrong, d), _rect(d, cut + 1)]
    text = phrase(
        rng,
        (
            "A solid is built from unit cubes. Two flat cuts parallel to its right face go through the "
            "{o1} layer and the {o2} layer of cubes, counting from the left. Shade the cut face with the {pick} area, "
            "as seen from the right, with the front of the solid on your left.",
            "This solid has two cuts parallel to its right face, through the {o1} layer and the {o2} layer of cubes, "
            "counting from the left. Draw on the canvas the cut face with the {pick} area, seen from the right with "
            "the front of the solid on your left.",
            "Slice the solid twice, parallel to its right face, through the {o1} layer and the {o2} layer of cubes, "
            "counting from the left. Of the two cut faces, shade the one with the {pick} area, as seen from the "
            "right with the front of the solid on your left.",
        ),
        o1=_ORD[first],
        o2=_ORD[second],
        pick=pick,
    )
    working = [
        "Both cuts are parallel to the right face and go through columns that are the same number of cubes deep, "
        "so the cut face with the taller column has the greater area.",
        f"The columns are {gsb._plural(ha, 'cube')} and {gsb._plural(hb, 'cube')} tall, so the {pick} area is the one {gsb._plural(cut, 'cell')} up and {gsb._plural(d, 'cell')} across.",
    ]
    cubes = _box_cubes(w, d, top, heights)
    fig = grid_drawing(
        alt=_ALT,
        grid={"type": "isometric", "width": w, "height": d, "layers": top},
        cubes=cubes,
        canvas={"width": CANVAS, "height": CANVAS},
    )
    cells, cands = _place(rng, answer, cands)
    return gsb._cells_question(
        fig, text + " " + _pin(rng), sorted(cells), (CANVAS, CANVAS), "translate", cands, working
    )


def _hard(rng):
    w, top, heights = _steps(rng)
    d = rng.randint(2, 4)
    if sum(heights) * d > 64:
        return None
    kind = rng.choice((_hard_column, _hard_layer, _hard_compare))
    return kind(rng, w, d, top, heights)


@generator(
    "grid_solid_cross_sections",
    level=6,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_solid_cross_sections(rng, tier="easy"):
    """Shade the cross-section of a solid of unit cubes cut parallel to one of its faces."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if tier == "easy":
        return gsb._retry(rng, _easy, "an easy cross-section")
    if tier == "medium":
        return gsb._retry(
            rng, lambda r: (_medium_footprint if r.random() < 0.5 else _medium_end)(r), "an L cross-section"
        )
    return gsb._retry(rng, _hard, "a stepped cross-section")


_ONE = re.compile(
    r"parallel to its (front|right|top) face.*?through the (first|second|third|fourth|fifth) layer of cubes"
)
_TWO = re.compile(r"through the (\w+) layer and the (\w+) layer of cubes")
_PICK = re.compile(r"the (greater|smaller) area")


def _read_cross_section(question) -> str:
    """The cut face, from the cubes in the figure and the layers named in the wording."""
    cubes = [tuple(c) for c in question.figure["cubes"]]
    text = question.text
    if (m := _TWO.search(text)) is not None:
        ks = [_ORD.index(m.group(1)), _ORD.index(m.group(2))]
        faces = [_section(cubes, "right", k) for k in ks]
        pick = _PICK.search(text).group(1)
        chosen = max(faces, key=len) if pick == "greater" else min(faces, key=len)
    else:
        m = _ONE.search(text)
        chosen = _section(cubes, m.group(1), _ORD.index(m.group(2)))
    low_x, low_y = min(x for x, _ in chosen), min(y for _, y in chosen)
    return cell_text({(x - low_x, y - low_y) for x, y in chosen})


figures.register_reader("grid_solid_cross_sections", _read_cross_section)
