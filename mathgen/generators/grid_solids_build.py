"""Generates the solids, nets and composite-shape questions drawn on a grid (``grid_drawing`` figures).

Five topics, one per skill. Every figure is a ``grid_drawing`` (``mathgen/figure_kinds/grid.py``); a reader
registered for each topic re-derives the answer from the figure and the sentence alone.

``grid_cube_stack`` (``senior-interpret-diagrams-of-three-dimensional-objects``, number)
    A stack of unit cubes on isometric dot paper, drawn so that no cube hides behind another
    (``tops_all_visible``): easy counts a one or two layer stack; medium counts a three layer stack; hard
    either finds how many more cubes fill a stated box, or finds the surface area in unit squares, hidden
    faces and the bottom included.

``grid_solid_properties`` (``senior-recognise-the-properties-of-common-two-dimensional-geometric``, number)
    Easy reads the three edge lengths of a cuboid of cubes from the picture and counts its edges or faces by
    length, squareness or area. Medium is words only: faces, edges or vertices of a named prism or pyramid, or
    the diagonals of a polygon. Hard uses Euler's rule ``v + f - e = 2`` and the shape of prisms and pyramids to
    work back from one count to another.

``grid_net_views`` (``senior-interpret-different-forms-of-two-dimensional-representations-of``, mcq)
    Easy reads a front view (column heights) from the isometric picture of a stack; medium picks the one drawing
    of four that is NOT a net of a cube; hard reads the net of a closed box off the grid and gives its surface
    area or volume.

``grid_composite_area`` (``f10-composite-shape-representation``, number)
    Easy counts the squares inside a rectilinear outline; medium splits a rectangle-and-triangle outline into
    familiar parts; hard estimates an irregular shape from squares drawn whole, more than half and less than
    half (or whole and part), with the counting rule stated in the sentence so the answer is exact.

``grid_nets_build_solids`` (``f10-nets-build-solids``, mcq)
    Easy names the solid a drawn net folds into; medium picks the one net of a cube from four drawings; hard
    names the face opposite a lettered face when a cube net is folded (folded by rolling a cube over the net).

Every cube net and every non-net in an option is checked with ``is_cube_net``.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

from .. import figures
from ..core import attempts, check, generator, phrase, pick_distractors
from ..figure_kinds import grid
from ..grading.grid import OPPOSITE_FACE, congruent, cube_net_faces, is_cube_net, orientations

# ----------------------------------------------------------------------------------------- shared helpers


def _fmt(value) -> str:
    """A count or an area in halves, as typed: ``27`` or ``27.5``."""
    return str(int(value)) if value == int(value) else f"{value:.1f}"


def _number(value):
    return int(value) if value == int(value) else float(value)


def _finish_number(tier_fn, rng):
    for _ in attempts(80, "a question with three distinct wrong answers"):
        fig, text, answer, candidates, working = tier_fn(rng)
        check(answer > 0, "the answer must be positive")
        # Every option has the answer's shape (a whole number, or a number of halves), so the point alone gives
        # nothing away. A miscount by a square or two pads the misconceptions when too few share that shape.
        halves = answer != int(answer)
        same_shape = [c for c in candidates if (c != int(c)) == halves]
        pool = [_fmt(c) for c in [*same_shape, answer + 1, answer - 1, answer + 2] if c > 0 and c != answer]
        distractors = pick_distractors(_fmt(answer), pool)
        if len(distractors) == 3:
            break
    out = dict(
        text=text,
        answer=_number(answer),
        answer_text=_fmt(answer),
        answer_latex=_fmt(answer),
        distractors=distractors,
        working=working,
    )
    if fig is not None:
        out["figure"] = fig
    return out


def _finish_mcq(tier_fn, rng):
    for _ in attempts(80, "a question with three distinct wrong options"):
        fig, text, answer_text, candidates, working = tier_fn(rng)
        distractors = pick_distractors(answer_text, [c for c in candidates if c != answer_text])
        if len(distractors) == 3:
            break
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=r"\text{" + answer_text + "}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _tier_of(tiers, tier):
    if tier not in tiers:
        raise ValueError(f"unknown tier {tier!r}")
    return tiers[tier]


def _twice_area(vertices) -> int:
    """Twice the area of a polygon with whole-number vertices (the shoelace sum)."""
    total = 0
    for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1]):
        total += x1 * y2 - x2 * y1
    return abs(total)


def _perimeter(vertices) -> int:
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1]))


# ----------------------------------------------------------------------------------------- cube stacks


def _stack_heights(rng, width, depth, top, lo, hi):
    """Column heights ``{(x, y): h}`` of a stack no cube of which hides behind another in the picture.

    The picture is seen from the front right, so heights must not increase to the right or towards the front:
    they are cut down along x and built up along y. Every column and row of the grid holds a cube, the tallest
    column is ``top`` and the stack has between ``lo`` and ``hi`` cubes.
    """
    for _ in attempts(2000, "a stack of cubes with every top visible"):
        g = {(x, y): rng.randint(0, top) for x in range(width) for y in range(depth)}
        low = {(x, y): min(g[(i, y)] for i in range(x + 1)) for (x, y) in g}
        h = {(x, y): max(low[(x, j)] for j in range(y + 1)) for (x, y) in g}
        n = sum(h.values())
        if max(h.values()) != top or not lo <= n <= hi:
            continue
        if any(not any(h[(x, y)] for y in range(depth)) for x in range(width)):
            continue
        if any(not any(h[(x, y)] for x in range(width)) for y in range(depth)):
            continue
        cubes = [[x, y, z] for (x, y), height in sorted(h.items()) for z in range(height)]
        if grid.tops_all_visible(cubes):
            return h, cubes
    raise AssertionError("unreachable")


def _stack_figure(width, depth, top, cubes):
    return grid.grid_drawing(
        alt="A stack of unit cubes drawn on isometric dot paper.",
        grid={"type": "isometric", "width": width, "height": depth, "layers": top},
        cubes=cubes,
    )


def _visible_squares(cubes) -> int:
    """Squares of the stack seen in the picture: every top, front and right face that is not covered."""
    s = {tuple(c) for c in cubes}
    return sum(((x, y, z + 1) not in s) + ((x, y - 1, z) not in s) + ((x + 1, y, z) not in s) for x, y, z in s)


def _surface_area(cubes) -> int:
    s = {tuple(c) for c in cubes}
    touching = sum(((x + 1, y, z) in s) + ((x, y + 1, z) in s) + ((x, y, z + 1) in s) for x, y, z in s)
    return 6 * len(s) - 2 * touching


def _stack_easy(rng):
    top = rng.choice((1, 2, 2))
    for _ in attempts(50, "a grid roomy enough for the stack"):
        width, depth = rng.randint(2, 4), rng.randint(1, 3)
        if width * depth * top >= 6:
            break
    h, cubes = _stack_heights(rng, width, depth, top, 4, 12)
    fig = _stack_figure(width, depth, top, cubes)
    n = len(cubes)
    columns = grid.plan_view(cubes)
    text = phrase(
        rng,
        (
            "The picture shows a stack of unit cubes on isometric dot paper. Every cube rests on the ground or on "
            "another cube, with no gaps. How many cubes make up the stack?",
            "A stack of unit cubes is drawn on isometric dot paper. There are no gaps: each cube sits on the "
            "ground or on another cube. How many cubes are in the stack?",
            "Count the unit cubes in the stack drawn on isometric dot paper. Every cube rests on the ground or "
            "on another cube.",
        ),
    )
    cands = [len(columns), width * depth * top, _visible_squares(cubes), n + 1, n - 1, n + 2]
    working = [f"Count the cubes in each column and add them: {' + '.join(str(v) for v in h.values() if v)} = {n}."]
    return fig, text, n, cands, working


def _stack_medium(rng):
    width, depth = rng.randint(2, 4), rng.randint(2, 3)
    for _ in attempts(200, "a stack three layers high"):
        h, cubes = _stack_heights(rng, width, depth, 3, 7, 18)
        if {1, 2, 3} <= set(h.values()):
            break
    fig = _stack_figure(width, depth, 3, cubes)
    n = len(cubes)
    columns = grid.plan_view(cubes)
    text = phrase(
        rng,
        (
            "The picture shows a stack of unit cubes on isometric dot paper, three layers high at its tallest. "
            "Every cube rests on the ground or on another cube, with no gaps. How many cubes make up the stack?",
            "A stack of unit cubes is drawn on isometric dot paper. It is three layers high at its tallest and "
            "has no gaps: each cube sits on the ground or on another cube. How many cubes are in the stack?",
            "Count the unit cubes in this stack drawn on isometric dot paper. It rises to three layers, and "
            "every cube rests on the ground or on another cube.",
        ),
    )
    cands = [len(columns), width * depth * 3, _visible_squares(cubes), len(columns) * 3, n + 1, n - 1]
    working = [
        "Count the cubes in each column, including the ones hidden below the top: "
        + f"{' + '.join(str(v) for v in h.values() if v)} = {n}."
    ]
    return fig, text, n, cands, working


def _stack_hard(rng):
    width, depth = rng.randint(2, 4), rng.randint(2, 3)
    top = rng.choice((3, 3, 4))
    h, cubes = _stack_heights(rng, width, depth, top, 6, 20)
    fig = _stack_figure(width, depth, top, cubes)
    n = len(cubes)
    if rng.random() < 0.5:
        box_w, box_d, box_h = width + rng.randint(0, 1), depth + rng.randint(0, 1), top + rng.randint(0, 1)
        full = box_w * box_d * box_h
        check(full > n, "the box must hold the stack")
        text = phrase(
            rng,
            (
                "The stack of unit cubes is built inside a box that is {w} cubes wide, {d} cubes deep and {h} cubes "
                "high. There are no gaps in the stack. How many more cubes are needed to fill the box completely?",
                "A box is {w} cubes wide, {d} cubes deep and {h} cubes high. The stack of unit cubes drawn has no "
                "gaps and sits inside the box. How many more unit cubes would fill the box completely?",
            ),
            w=box_w,
            d=box_d,
            h=box_h,
        )
        answer = full - n
        cands = [n, full, full + n, width * depth * top - n, full - n + box_w * box_d, full - 2 * n]
        working = [
            f"The stack has {n} cubes. The box holds {box_w} × {box_d} × {box_h} = {full} cubes.",
            f"{full} - {n} = {answer} more cubes.",
        ]
    else:
        seen = _visible_squares(cubes)
        text = phrase(
            rng,
            (
                "The surface area of the stack is the number of unit squares on its outside, including the bottom "
                "and the faces you cannot see. There are no gaps in the stack. What is the surface area of the "
                "stack of unit cubes, in unit squares?",
                "Find the surface area, in unit squares, of the stack of unit cubes. Count every square on the "
                "outside, including the bottom and the faces hidden from view. There are no gaps in the stack.",
            ),
        )
        answer = _surface_area(cubes)
        cands = [seen, 6 * n, seen + len(grid.plan_view(cubes)), 2 * seen - len(grid.plan_view(cubes)), answer - 2]
        working = [
            f"The picture shows {seen} squares: tops, fronts and right sides.",
            f"The back, the left side and the bottom match the squares you can see, so the total is 2 × {seen} = "
            f"{answer} unit squares.",
        ]
    return fig, text, answer, cands, working


_STACK_TIERS = {"easy": _stack_easy, "medium": _stack_medium, "hard": _stack_hard}


@generator(
    "grid_cube_stack",
    level=7,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_cube_stack(rng, tier="easy"):
    """Count the cubes, or the cubes still missing, or the surface area, of a stack on isometric dot paper."""
    return _finish_number(_tier_of(_STACK_TIERS, tier), rng)


_FILL = re.compile(r"(\d+) cubes wide, (\d+) cubes deep and (\d+) cubes high")


def _read_cube_stack(question) -> str:
    cubes = question.figure["cubes"]
    if (m := _FILL.search(question.text)) is not None:
        return str(int(m.group(1)) * int(m.group(2)) * int(m.group(3)) - len(cubes))
    if "surface area" in question.text:
        return str(_surface_area(cubes))
    return str(len(cubes))


figures.register_reader("grid_cube_stack", _read_cube_stack)


# ----------------------------------------------------------------------------------------- properties


_PRISM_FIRST = {3: "triangular", 4: "square", 5: "pentagonal", 6: "hexagonal", 7: "heptagonal", 8: "octagonal"}
_POLYGON = {5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon", 9: "nonagon", 10: "decagon"}
_SIDES_OF = {v: k for k, v in _PRISM_FIRST.items()}
_SIDES_OF_POLYGON = {v: k for k, v in _POLYGON.items()}


def _counts(family: str, n: int) -> dict[str, int]:
    """Faces, edges and vertices of the prism or pyramid whose base has ``n`` sides."""
    if family == "prism":
        return {"faces": n + 2, "edges": 3 * n, "vertices": 2 * n}
    return {"faces": n + 1, "edges": 2 * n, "vertices": n + 1}


def _properties_easy(rng):
    for _ in attempts(100, "a cuboid"):
        a, b, c = (rng.randint(1, 4) for _ in range(3))
        if (a, b, c) != (1, 1, 1):
            break
    cubes = [[x, y, z] for x in range(a) for y in range(b) for z in range(c)]
    fig = grid.grid_drawing(
        alt="A cuboid built from unit cubes, drawn on isometric dot paper.",
        grid={"type": "isometric", "width": a, "height": b, "layers": c},
        cubes=cubes,
    )
    dims = (a, b, c)
    kind = rng.choice(("edges", "squares", "area"))
    if kind == "squares" and len(set(dims)) == 3:
        kind = rng.choice(("edges", "area"))
    if kind == "edges":
        k = rng.choice(dims)
        same = sum(1 for d in dims if d == k)
        text = phrase(
            rng,
            (
                "The cuboid is built from unit cubes. How many of its edges are exactly {k} {unit} long?",
                "A cuboid is built from unit cubes on isometric dot paper. Count its edges: how many are exactly "
                "{k} {unit} long?",
                "Look at the cuboid of unit cubes. Some of its edges are exactly {k} {unit} long. How many edges "
                "is that?",
            ),
            k=k,
            unit="cube" if k == 1 else "cubes",
        )
        answer = 4 * same
        cands = [same, 12, 8, 4, 6, 3 * same]
        working = [f"The cuboid is {a} by {b} by {c}. Each length has 4 edges, and {same} of the 3 lengths is {k}."]
    elif kind == "squares":
        text = phrase(
            rng,
            (
                "The cuboid is built from unit cubes. How many of its 6 faces are squares?",
                "A cuboid is built from unit cubes on isometric dot paper. How many of its 6 faces are squares?",
            ),
        )
        answer = {1: 6, 2: 2}[len(set(dims))] if len(set(dims)) < 3 else 0
        check(answer > 0, "a cuboid with three different lengths has no square face")
        cands = [4, 6, 1, 3, 2, 8]
        working = [f"The cuboid is {a} by {b} by {c}: the faces that are squares come from the two equal lengths."]
    else:
        areas = [a * b, a * c, b * c]
        area = rng.choice(areas)
        text = phrase(
            rng,
            (
                "The cuboid is built from unit cubes. How many of its faces are rectangles that cover exactly "
                "{area} unit squares?",
                "A cuboid is built from unit cubes on isometric dot paper. How many of its faces cover exactly "
                "{area} unit squares?",
            ),
            area=area,
        )
        pairs = sum(1 for v in areas if v == area)
        answer = 2 * pairs
        cands = [pairs, 6, 4, 2 * pairs + 2, 3, 12]
        working = [f"The cuboid is {a} by {b} by {c}. Its faces come in 3 equal pairs, with areas {areas}."]
    return fig, text, answer, cands, working


def _properties_medium(rng):
    if rng.random() < 0.7:
        family = rng.choice(("prism", "pyramid"))
        n = rng.choice(list(_PRISM_FIRST))
        prop = rng.choice(("faces", "edges", "vertices"))
        c = _counts(family, n)
        name = f"{_PRISM_FIRST[n]} {family}"
        text = phrase(
            rng,
            (
                "How many {prop} does a {name} have?",
                "Think about a {name}. How many {prop} does it have?",
            ),
            prop=prop,
            name=name,
        )
        pool = {
            ("prism", "faces"): [n, n + 1, 2 * n, 3 * n],
            ("prism", "edges"): [2 * n, n + 2, 6 * n, 4 * n],
            ("prism", "vertices"): [n, n + 2, 3 * n, 4 * n],
            ("pyramid", "faces"): [n, n + 2, 2 * n, 2 * n + 2],
            ("pyramid", "edges"): [n, 3 * n, n + 1, 4 * n],
            ("pyramid", "vertices"): [n, 2 * n, n + 2, 3 * n],
        }[(family, prop)]
        working = [
            f"A {name} has a base with {n} sides: {c['faces']} faces, {c['edges']} edges, {c['vertices']} vertices."
        ]
        return None, text, c[prop], pool, working
    n = rng.choice(list(_POLYGON))
    text = phrase(
        rng,
        (
            "A diagonal joins two vertices that are not next to each other. How many diagonals does a {p} have?",
            "How many diagonals can be drawn inside a {p}? A diagonal joins two vertices that are not next to each "
            "other.",
        ),
        p=_POLYGON[n],
    )
    answer = n * (n - 3) // 2
    cands = [n * (n - 3), n - 3, n * (n - 1) // 2, answer + n, n]
    working = [
        f"Each of the {n} vertices joins to {n - 3} others by a diagonal, and each diagonal is counted twice: "
        f"{n} × {n - 3} / 2 = {answer}."
    ]
    return None, text, answer, cands, working


def _properties_hard(rng):
    family = rng.choice(("prism", "pyramid"))
    n = rng.randint(4, 12)
    c = _counts(family, n)
    v, e, f = c["vertices"], c["edges"], c["faces"]
    check(v + f - e == 2, "Euler's rule must hold for the solid")
    kind = rng.choice(("edges", "faces", "prism_edges", "pyramid_vertices", "prism_faces"))
    rule = "Euler's rule says vertices + faces - edges = 2."
    if kind == "edges":
        text = f"A solid with flat faces has {f} faces and {v} vertices. {rule} How many edges does it have?"
        answer, cands = e, [v + f, v + f + 2, v + f - 1, abs(v - f) + 2]
        working = [f"{v} + {f} - edges = 2, so edges = {v} + {f} - 2 = {e}."]
    elif kind == "faces":
        text = f"A solid with flat faces has {v} vertices and {e} edges. {rule} How many faces does it have?"
        answer, cands = f, [e - v, e - v - 2, e + v - 2, e + v]
        working = [f"{v} + faces - {e} = 2, so faces = 2 + {e} - {v} = {f}."]
    elif kind == "prism_edges":
        text = f"A prism has {3 * n} edges. How many faces does it have?"
        answer, cands = n + 2, [n, n + 1, 2 * n, 3 * n + 2]
        working = [
            f"A prism has 3 edges for each side of its base, so the base has {3 * n} / 3 = {n} sides.",
            f"Faces: {n} + 2 = {n + 2}.",
        ]
    elif kind == "pyramid_vertices":
        text = f"A pyramid has {n + 1} vertices. How many edges does it have?"
        answer, cands = 2 * n, [n, 2 * n + 2, n + 1, 3 * n]
        working = [f"One vertex is the apex, so the base has {n} sides.", f"Edges: 2 × {n} = {2 * n}."]
    else:
        text = f"A prism has {n + 2} faces. How many edges does it have?"
        answer, cands = 3 * n, [n, 2 * n, n + 2, 3 * (n + 2)]
        working = [f"Two faces are the bases, so the base has {n + 2} - 2 = {n} sides.", f"Edges: 3 × {n} = {3 * n}."]
    text = rng.choice(("", "Think about solids with flat faces. ", "Here is a puzzle about solids. ")) + text
    return None, text, answer, cands, working


_PROPERTIES_TIERS = {"easy": _properties_easy, "medium": _properties_medium, "hard": _properties_hard}


@generator(
    "grid_solid_properties",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_solid_properties(rng, tier="easy"):
    """Faces, edges and vertices of cuboids, prisms and pyramids, and diagonals of polygons."""
    return _finish_number(_tier_of(_PROPERTIES_TIERS, tier), rng)


_EDGES = re.compile(r"exactly (\d+) cubes? long")
_AREA = re.compile(r"exactly (\d+) unit squares")
_NAMED = re.compile(r"(?:does a|about a) (\w+) (prism|pyramid)\b")
_ASKED = re.compile(r"How many (faces|edges|vertices) does")
_DIAGONALS = re.compile(r"diagonals (?:does a|can be drawn inside a) (\w+)")
_SOLID_FV = re.compile(r"has (\d+) faces and (\d+) vertices")
_SOLID_VE = re.compile(r"has (\d+) vertices and (\d+) edges")
_PRISM_E = re.compile(r"prism has (\d+) edges")
_PYR_V = re.compile(r"pyramid has (\d+) vertices")
_PRISM_F = re.compile(r"prism has (\d+) faces")


def _read_properties(question) -> str:
    text = question.text
    fig = question.figure
    if fig is not None:
        cubes = fig["cubes"]
        dims = [1 + max(c[axis] for c in cubes) for axis in range(3)]
        if (m := _EDGES.search(text)) is not None:
            return str(4 * sum(1 for d in dims if d == int(m.group(1))))
        if "are squares" in text:
            return str(2 * sum(1 for i in range(3) if dims[i] == dims[(i + 1) % 3]) if len(set(dims)) == 2 else 6)
        m = _AREA.search(text)
        check(m is not None, "cannot read the question")
        a, b, c = dims
        return str(2 * sum(1 for v in (a * b, a * c, b * c) if v == int(m.group(1))))
    if (m := _DIAGONALS.search(text)) is not None:
        n = _SIDES_OF_POLYGON[m.group(1)]
        return str(n * (n - 3) // 2)
    if (m := _NAMED.search(text)) is not None:
        n = _SIDES_OF[m.group(1)]
        family = m.group(2)
        prop = _ASKED.search(text)
        return str(_counts(family, n)[prop.group(1)])
    if (m := _SOLID_FV.search(text)) is not None:
        return str(int(m.group(2)) + int(m.group(1)) - 2)
    if (m := _SOLID_VE.search(text)) is not None:
        return str(2 + int(m.group(2)) - int(m.group(1)))
    if (m := _PRISM_E.search(text)) is not None:
        return str(int(m.group(1)) // 3 + 2)
    if (m := _PYR_V.search(text)) is not None:
        return str(2 * (int(m.group(1)) - 1))
    m = _PRISM_F.search(text)
    check(m is not None, "cannot read the question")
    return str(3 * (int(m.group(1)) - 2))


figures.register_reader("grid_solid_properties", _read_properties)


# ----------------------------------------------------------------------------------------- nets on a grid


def _canonical(shape) -> tuple:
    return min(tuple(sorted(o)) for o in orientations(shape))


@lru_cache(maxsize=1)
def _hexominoes() -> tuple[frozenset, ...]:
    """The 35 free hexominoes, each once, as sets of cells, that fit a 5 by 4 room (turned as needed)."""
    level = {_canonical({(0, 0)}): frozenset({(0, 0)})}
    for _ in range(5):
        grown = {}
        for shape in level.values():
            for x, y in shape:
                for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if cell not in shape:
                        bigger = shape | {cell}
                        grown.setdefault(_canonical(bigger), min(orientations(bigger), key=lambda o: sorted(o)))
        level = grown

    def fits(shape):
        w, h = max(x for x, _ in shape) + 1, max(y for _, y in shape) + 1
        return min(w, h) <= 4 and max(w, h) <= 5

    return tuple(shape for _key, shape in sorted(level.items()) if fits(shape))


def cube_nets() -> list[frozenset]:
    """The cube nets among the hexominoes (each once, up to turns and flips): all 11 of them."""
    return [s for s in _hexominoes() if is_cube_net(s)]


def non_nets() -> list[frozenset]:
    """Hexominoes that look like nets and are not (every one checked against ``is_cube_net``)."""
    return [s for s in _hexominoes() if not is_cube_net(s)]


_SLOT_W, _SLOT_H = 6, 5  # a slot: the label's column, then room for a drawing 5 wide and 4 tall
_ORIGINS = {"A": (0, 5), "B": (6, 5), "C": (0, 0), "D": (6, 0)}


def _place(rng, shape, origin):
    """The shape turned or flipped at random, set somewhere in the 5 by 4 room of a slot."""
    fitting = [
        o for o in sorted(orientations(shape), key=sorted) if max(x for x, _ in o) < 5 and max(y for _, y in o) < 4
    ]
    chosen = rng.choice(fitting)
    w = max(x for x, _ in chosen) + 1
    h = max(y for _, y in chosen) + 1
    dx, dy = rng.randint(0, 5 - w), rng.randint(0, 4 - h)
    ox, oy = origin
    return [[ox + 1 + x + dx, oy + y + dy] for x, y in sorted(chosen)]


def _choice_figure(placed):
    """Four drawings A to D, one per slot of a 12 by 9 grid, each labelled in the slot's top left."""
    return grid.grid_drawing(
        alt="Four flat shapes, each made of six squares, drawn on a square grid and labelled with letters.",
        grid={"type": "square", "width": 12, "height": 9},
        cells=[{"cells": cells, "style": "shaded"} for cells in placed.values()],
        labels=[{"at": [_ORIGINS[k][0] + 0.5, _ORIGINS[k][1] + 3.5], "text": k} for k in placed],
    )


def _four_nets(rng, nets: int):
    """``nets`` cube nets and the rest non-nets, all different shapes, in a random order of the letters A to D."""
    for _ in attempts(100, "four different drawings"):
        good = rng.sample(cube_nets(), nets)
        bad = rng.sample(non_nets(), 4 - nets)
        letters = rng.sample("ABCD", 4)
        placed = {k: _place(rng, s, _ORIGINS[k]) for k, s in zip(letters, good + bad)}
        flags = {k: is_cube_net(_cells_of(v)) for k, v in placed.items()}
        check(sum(flags.values()) == nets, "the drawings are not the intended mix of nets and non-nets")
        if all(not congruent(_cells_of(placed[a]), _cells_of(placed[b])) for a in placed for b in placed if a < b):
            return dict(sorted(placed.items())), flags
    raise AssertionError("unreachable")


def _cells_of(cells):
    return {(x, y) for x, y in cells}


def folded_faces(cells) -> dict[tuple[int, int], str]:
    """Which face of a cube lands on each cell of a cube net (`mathgen.grading.grid.cube_net_faces`)."""
    landed = cube_net_faces(cells)
    check(landed is not None, "not a net of a cube")
    return landed


def opposite_cell(cells, cell):
    """The cell of the net whose face is opposite the face on ``cell`` once the net is folded."""
    landed = folded_faces(cells)
    target = OPPOSITE_FACE[landed[cell]]
    return next(c for c, face in landed.items() if face == target)


# ---- polygon nets (cuboid, square pyramid, triangular prism)


def _cuboid_net(w, d, h, rng):
    """The six rectangles of a cuboid net: a strip of four sides with the top and the bottom on one side face."""
    xs = [0, w, w + d, 2 * w + d, 2 * w + 2 * d]
    polys = [[(xs[i], d), (xs[i + 1], d), (xs[i + 1], d + h), (xs[i], d + h)] for i in range(4)]
    i = rng.choice((0, 2))  # a side face of width w carries the top and the bottom
    polys.append([(xs[i], d + h), (xs[i + 1], d + h), (xs[i + 1], 2 * d + h), (xs[i], 2 * d + h)])
    polys.append([(xs[i], 0), (xs[i + 1], 0), (xs[i + 1], d), (xs[i], d)])
    return polys, (2 * w + 2 * d, h + 2 * d)


def _square_pyramid_net(s, t):
    """A square of side ``s`` (even) with an isosceles triangle of height ``t`` on each side."""
    x0 = y0 = t
    half = s // 2
    base = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)]
    tris = [
        [(x0, y0 + s), (x0 + s, y0 + s), (x0 + half, y0 + s + t)],
        [(x0, y0), (x0 + s, y0), (x0 + half, y0 - t)],
        [(x0, y0), (x0, y0 + s), (x0 - t, y0 + half)],
        [(x0 + s, y0), (x0 + s, y0 + s), (x0 + s + t, y0 + half)],
    ]
    return [base] + tris, (s + 2 * t, s + 2 * t)


def _triangular_prism_net(rng):
    """Three rectangles in a row with a triangle above and below the middle one: a 3-4-5 or a 5-5-6 prism."""
    length = rng.randint(2, 4)
    if rng.random() < 0.5:
        # a right triangle with legs 3 and 4 and hypotenuse 5; its leg of 3 sits beside the rectangle of width 3
        order = rng.choice(((3, 4, 5), (5, 4, 3)))
        left, right = order[0], order[0] + 4
        apex_x = left if order[0] == 3 else right
        up, down, shift = 3, -3, 3
    else:
        order = (5, 6, 5)
        left, right = 5, 11
        apex_x = 8
        up, down, shift = 4, -4, 4
    xs = [0, order[0], order[0] + order[1], sum(order)]
    rects = [[(xs[i], 0), (xs[i + 1], 0), (xs[i + 1], length), (xs[i], length)] for i in range(3)]
    tri_up = [(left, length), (right, length), (apex_x, length + up)]
    tri_down = [(left, 0), (right, 0), (apex_x, down)]
    polys = [[(x, y + shift) for x, y in p] for p in rects + [tri_up, tri_down]]
    return polys, (sum(order), length + 2 * shift)


def _polygon_figure(polys, size, alt):
    return grid.grid_drawing(
        alt=alt,
        grid={"type": "square", "width": size[0], "height": size[1]},
        polygons=[{"vertices": [list(v) for v in p], "style": "solid"} for p in polys],
    )


# ---- f10 nets build solids

_SOLIDS = ("a cube", "a cuboid", "a square pyramid", "a triangular prism", "a triangular pyramid")


def _nets_easy(rng):
    solid = rng.choice(_SOLIDS[:4])
    if solid == "a cube":
        shape = rng.choice(cube_nets())
        chosen = rng.choice(sorted(orientations(shape), key=sorted))
        check(is_cube_net(chosen), "a cube's net must be a net")
        polys = [[(x + 1, y + 1), (x + 2, y + 1), (x + 2, y + 2), (x + 1, y + 2)] for x, y in sorted(chosen)]
        size = (max(x for x, _ in chosen) + 3, max(y for _, y in chosen) + 3)
    elif solid == "a cuboid":
        for _ in attempts(50, "a cuboid that is not a cube"):
            w, d, h = (rng.randint(1, 3) for _ in range(3))
            if not w == d == h:
                break
        polys, size = _cuboid_net(w, d, h, rng)
    elif solid == "a square pyramid":
        s = rng.choice((2, 4))
        polys, size = _square_pyramid_net(s, rng.randint(s // 2 + 1, s + 2))
    else:
        polys, size = _triangular_prism_net(rng)
    fig = _polygon_figure(polys, size, "A net drawn on a square grid.")
    text = phrase(
        rng,
        (
            "The net is drawn on a square grid. Which solid does it fold up into?",
            "A net is drawn on a square grid. It is cut out and folded along its edges. Which solid does it make?",
            "Which of these solids can be made by cutting out the net on the grid and folding it up?",
        ),
    )
    # A cube is also a cuboid, so a cube's net must not offer "a cuboid" as a wrong answer.
    others = [s for s in _SOLIDS if s != solid and not (solid == "a cube" and s == "a cuboid")]
    rng.shuffle(others)
    working = {
        "a cube": "Six equal squares fold into a cube.",
        "a cuboid": "Six rectangles, the opposite ones equal and not all squares, fold into a cuboid.",
        "a square pyramid": "One square with four triangles on its sides folds into a square pyramid.",
        "a triangular prism": "Three rectangles in a row with a triangle at each end fold into a triangular prism.",
    }[solid]
    return fig, text, solid, others[:3] + others[3:], [working]


def _nets_medium(rng):
    placed, flags = _four_nets(rng, 1)
    answer = f"Net {next(k for k, v in flags.items() if v)}"
    text = phrase(
        rng,
        (
            "Each of the four drawings is made of six squares. Which one is a net of a cube?",
            "Four drawings, each made of six squares, are shown on the grid. Only one of them can be folded along "
            "its edges to make a cube. Which one is it?",
            "Only one of drawings A to D can be cut out and folded into a cube, and the rest are made of six "
            "squares too. Which drawing is the net of the cube?",
        ),
    )
    cands = [f"Net {k}" for k in placed]
    working = [
        f"{answer} folds into a cube. In each of the others, folding makes two squares overlap or leaves a face missing."
    ]
    return _choice_figure(placed), text, answer, cands, working


def _nets_hard(rng):
    shape = rng.choice(cube_nets())
    cells = sorted(rng.choice(sorted(orientations(shape), key=sorted)))
    name = dict(zip(cells, rng.sample("ABCDEF", 6)))
    marked = rng.choice(cells)
    partner = opposite_cell(cells, marked)
    # the wrong answers a learner reaches from the flat picture: the face at the far end of the marked face's row
    # or column, or a face that touches it
    far = [c for c in cells if c not in (marked, partner) and (c[0] == marked[0] or c[1] == marked[1])]
    wrong = [c for c in cells if c not in (marked, partner)]
    rng.shuffle(wrong)
    wrong.sort(key=lambda c: c not in far)
    fig = grid.grid_drawing(
        alt="A net made of six squares on a square grid, each square labelled with a letter.",
        grid={"type": "square", "width": max(x for x, _ in cells) + 2, "height": max(y for _, y in cells) + 2},
        cells=[{"cells": [[x + 0, y + 0] for x, y in cells], "style": "shaded"}],
        labels=[{"at": [x + 0.5, y + 0.5], "text": name[(x, y)]} for x, y in cells],
    )
    answer = f"Face {name[partner]}"
    cands = [f"Face {name[c]}" for c in wrong]
    text = phrase(
        rng,
        (
            "The net is folded along its edges to make a cube. Which face is opposite face {m}?",
            "The six squares of the net are folded up into a cube. Which face ends up opposite face {m}?",
        ),
        m=name[marked],
    )
    working = [
        f"Fold the net into a cube: face {name[marked]} and face {name[partner]} end up on opposite sides of the cube.",
    ]
    return fig, text, answer, cands, working


_NETS_TIERS = {"easy": _nets_easy, "medium": _nets_medium, "hard": _nets_hard}


@generator(
    "grid_nets_build_solids",
    level=5,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_nets_build_solids(rng, tier="easy"):
    """Which solid a net folds into, which drawing is a net of a cube, which face is opposite which."""
    return _finish_mcq(_tier_of(_NETS_TIERS, tier), rng)


def _slot_of(label_at):
    return int(label_at[0]) // _SLOT_W * _SLOT_W, (int(label_at[1]) - 3) // _SLOT_H * _SLOT_H


def _drawings_by_letter(fig):
    """``{letter: set of cells}`` for a figure of four lettered drawings in slots."""
    out = {}
    for label in fig["labels"]:
        ox, oy = _slot_of(label["at"])
        cells = {
            (x, y)
            for group in fig["cells"]
            for x, y in group["cells"]
            if ox <= x < ox + _SLOT_W and oy <= y < oy + _SLOT_H
        }
        out[label["text"]] = cells
    return out


def _polygon_corners(poly):
    xs = [v[0] for v in poly["vertices"]]
    ys = [v[1] for v in poly["vertices"]]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _read_nets(question) -> str:
    fig, text = question.figure, question.text
    if "polygons" in fig:
        polys = fig["polygons"]
        sides = sorted(len(p["vertices"]) for p in polys)
        if sides == [4] * 6:
            unit = all(_polygon_corners(p)[2:] == (1, 1) for p in polys)
            if unit:
                cells = {_polygon_corners(p)[:2] for p in polys}
                check(is_cube_net(cells), "six unit squares that are not a net of a cube")
                return "a cube"
            return "a cuboid"
        if sides == [3, 3, 4, 4, 4]:
            return "a triangular prism"
        if sides == [3, 3, 3, 3, 4]:
            return "a square pyramid"
        if sides == [3, 3, 3, 3]:
            return "a triangular pyramid"
        raise AssertionError("not a net of a solid in this set")
    if "opposite face" in text or "opposite" in text:
        m = re.search(r"opposite face (\w)", text)
        by_cell = {}
        for label in fig["labels"]:
            by_cell[(int(label["at"][0]), int(label["at"][1]))] = label["text"]
        cells = sorted(by_cell)
        marked = next(c for c in cells if by_cell[c] == m.group(1))
        return f"Face {by_cell[opposite_cell(cells, marked)]}"
    drawings = _drawings_by_letter(fig)
    nets = [k for k, cells in drawings.items() if is_cube_net(cells)]
    if "NOT" in text:
        (odd,) = [k for k in drawings if k not in nets]
        return f"Net {odd}"
    check(len(nets) == 1, "exactly one drawing should be a net of a cube")
    return f"Net {nets[0]}"


figures.register_reader("grid_nets_build_solids", _read_nets)


# ----------------------------------------------------------------------------------------- senior nets and views


def _column_heights(cubes, side):
    cells = grid.elevation(cubes, side)
    columns = sorted({c for c, _ in cells})
    return [sum(1 for c, _ in cells if c == col) for col in columns]


def _heights_text(heights) -> str:
    return "Column heights, left to right: " + ", ".join(str(v) for v in heights)


def _views_easy(rng):
    width, depth = rng.randint(3, 4), rng.randint(2, 3)
    top = 3
    for _ in attempts(200, "a stack whose four views differ"):
        h, cubes = _stack_heights(rng, width, depth, top, 6, 20)
        front = _column_heights(cubes, "front")
        views = {
            "back": front[::-1],
            "right": _column_heights(cubes, "right"),
            "left": _column_heights(cubes, "right")[::-1],
        }
        if len({tuple(front), *map(tuple, views.values())}) == 4:
            break
    fig = _stack_figure(width, depth, top, cubes)
    text = phrase(
        rng,
        (
            "The stack of unit cubes has no gaps. Its front is the face on the left of the picture and its right "
            "side is the face on the right of the picture. Which option gives the column heights of the view from "
            "the front, from left to right?",
            "Look at the stack of unit cubes, which has no gaps. The face on the left of the picture is the front "
            "and the face on the right of the picture is the right side. Seen from the front, what are the column "
            "heights from left to right?",
            "Imagine standing in front of the stack of unit cubes, which has no gaps. The front is the left face "
            "in the picture and the right side is the right face. Which option describes what you see?",
        ),
    )
    answer = _heights_text(front)
    cands = [_heights_text(views["back"]), _heights_text(views["right"]), _heights_text(views["left"])]
    working = [f"Seen from the front, the column heights from left to right are {', '.join(str(v) for v in front)}."]
    return fig, text, answer, cands, working


def _views_medium(rng):
    placed, flags = _four_nets(rng, 3)
    answer = f"Net {next(k for k, v in flags.items() if not v)}"
    text = phrase(
        rng,
        (
            "Each of the four drawings is made of six squares, and three of them can be folded into a cube. "
            "Which drawing is NOT a net of a cube?",
            "Three of these four drawings are nets of a cube. Which drawing is NOT a net of a cube?",
            "Fold each drawing in your head. Three fold up into a cube and one does not. Which drawing is NOT a "
            "net of a cube?",
        ),
    )
    cands = [f"Net {k}" for k in placed]
    working = [f"{answer} cannot be folded into a cube: folding makes two squares overlap and leaves a face missing."]
    return _choice_figure(placed), text, answer, cands, working


def _views_hard(rng):
    for _ in attempts(50, "a box that is not a cube"):
        w, d, h = (rng.randint(1, 4) for _ in range(3))
        if not w == d == h:
            break
    polys, size = _cuboid_net(w, d, h, rng)
    fig = _polygon_figure(polys, size, "The net of a closed box, drawn on a square grid.")
    surface = 2 * (w * d + w * h + d * h)
    volume = w * d * h
    if rng.random() < 0.5:
        text = phrase(
            rng,
            (
                "The net is drawn on a square grid with each square 1 unit by 1 unit. It folds up into a closed box "
                "with no gaps or overlaps. What is the surface area of the box?",
                "A closed box is made by folding the net on the grid, where each square is 1 unit by 1 unit. Nothing "
                "overlaps and there are no gaps. Find the surface area of the box.",
            ),
        )
        answer = f"{surface} square units"
        cands = [
            f"{2 * h * (w + d)} square units",  # leaves out the top and the bottom
            f"{surface // 2} square units",  # one of each pair of faces
            f"{volume} square units",
            f"{surface - 2 * w * d + w * d} square units",
        ]
        working = [
            f"The area of the net is the surface area of the box: 2 × ({w} × {d} + {w} × {h} + {d} × {h}) = {surface}."
        ]
    else:
        text = phrase(
            rng,
            (
                "The net is drawn on a square grid with each square 1 unit by 1 unit. It folds up into a closed box "
                "with no gaps or overlaps. What is the volume of the box in cubic units?",
                "A closed box is made by folding the net on the grid, where each square is 1 unit by 1 unit. Nothing "
                "overlaps and there are no gaps. Find the volume of the box in cubic units.",
            ),
        )
        answer = f"{volume} cubic units"
        cands = [
            f"{surface} cubic units",
            f"{w * d + w * h + d * h} cubic units",
            f"{w + d + h} cubic units",
            f"{w * d * h + w * d} cubic units",
        ]
        working = [f"Read the box off the net: {w} by {d} by {h}. Volume = {w} × {d} × {h} = {volume}."]
    return fig, text, answer, cands, working


_VIEWS_TIERS = {"easy": _views_easy, "medium": _views_medium, "hard": _views_hard}


@generator(
    "grid_net_views",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_net_views(rng, tier="easy"):
    """Read a view of a stack, spot the drawing that is not a cube net, or measure a box from its net."""
    return _finish_mcq(_tier_of(_VIEWS_TIERS, tier), rng)


def _read_views(question) -> str:
    fig, text = question.figure, question.text
    if "cubes" in fig:
        heights = _column_heights(fig["cubes"], "front")
        return _heights_text(heights)
    if "polygons" in fig:
        areas = sorted(_twice_area(p["vertices"]) // 2 for p in fig["polygons"])
        surface = sum(areas)
        if "surface area" in text:
            return f"{surface} square units"
        ab, ac, bc = areas[0::2]
        volume = math.isqrt(ab * ac * bc)
        check(volume * volume == ab * ac * bc, "the face areas do not make a box")
        return f"{volume} cubic units"
    drawings = _drawings_by_letter(fig)
    nets = [k for k, cells in drawings.items() if is_cube_net(cells)]
    (odd,) = [k for k in drawings if k not in nets]
    return f"Net {odd}"


figures.register_reader("grid_net_views", _read_views)


# ----------------------------------------------------------------------------------------- composite shapes


def _flip(vertices, size, rng):
    w, h = size
    fx, fy = rng.random() < 0.5, rng.random() < 0.5
    return [((w - x) if fx else x, (h - y) if fy else y) for x, y in vertices]


def _outline_figure(vertices, size, margin=1):
    shifted = [[x + margin, y + margin] for x, y in vertices]
    return grid.grid_drawing(
        alt="A shape drawn on a square grid.",
        grid={"type": "square", "width": size[0] + 2 * margin, "height": size[1] + 2 * margin},
        polygons=[{"vertices": shifted, "style": "solid"}],
    )


def _composite_easy(rng):
    kind = rng.choice(("L", "T", "U", "steps"))
    if kind == "L":
        a, b = rng.randint(4, 7), rng.randint(4, 6)
        c, d = rng.randint(1, a - 2), rng.randint(1, b - 2)
        verts, size = [(0, 0), (a, 0), (a, d), (c, d), (c, b), (0, b)], (a, b)
    elif kind == "T":
        s = rng.choice((1, 2, 3))
        a = s + 2 * rng.randint(1, 2)
        p, q = rng.randint(1, 2), rng.randint(2, 4)
        lft = (a - s) // 2
        verts = [(lft, 0), (lft + s, 0), (lft + s, q), (a, q), (a, q + p), (0, q + p), (0, q), (lft, q)]
        size = (a, q + p)
    elif kind == "U":
        n = rng.choice((1, 2, 3))
        a = n + 2 * rng.randint(1, 2)
        b, m = rng.randint(3, 5), rng.randint(1, 2)
        lft = (a - n) // 2
        verts = [(0, 0), (a, 0), (a, b), (lft + n, b), (lft + n, b - m), (lft, b - m), (lft, b), (0, b)]
        size = (a, b)
    else:
        s, t = rng.randint(1, 2), rng.randint(1, 2)
        verts = [(0, 0), (3 * s, 0), (3 * s, t), (2 * s, t), (2 * s, 2 * t), (s, 2 * t), (s, 3 * t), (0, 3 * t)]
        size = (3 * s, 3 * t)
    verts = _flip(verts, size, rng)
    twice = _twice_area(verts)
    area = twice // 2
    fig = _outline_figure(verts, size)
    box = size[0] * size[1]
    text = phrase(
        rng,
        (
            "The shape is drawn on a square grid. Each square is 1 unit long and 1 unit wide. Count the squares "
            "inside the shape to find its area in square units.",
            "A shape is drawn on a square grid in which each square is 1 unit by 1 unit. What is the area of the "
            "shape, in square units?",
            "How many square units does the shape on the grid cover? Each square of the grid is 1 unit long and 1 "
            "unit wide.",
        ),
    )
    cands = [box, round(_perimeter(verts)), box - area, area + 1, area - 1, area + size[0]]
    working = [f"The shape covers {area} whole squares, so its area is {area} square units."]
    return fig, text, area, cands, working


def _composite_medium(rng):
    kind = rng.choice(("slope", "house", "arrow", "cut"))
    if kind == "slope":
        w, h, t = rng.randint(3, 7), rng.randint(2, 4), rng.randint(1, 4)
        verts, size = [(0, 0), (w, 0), (w, h), (0, h + t)], (w, h + t)
        cands = [w * (h + t), w * h + w * t, w * h, w * t / 2]
        parts = f"a rectangle {w} by {h} and a triangle with base {w} and height {t}"
    elif kind == "house":
        w, h, t = 2 * rng.randint(2, 3), rng.randint(2, 4), rng.randint(1, 4)
        verts, size = [(0, 0), (w, 0), (w, h), (w // 2, h + t), (0, h)], (w, h + t)
        cands = [w * (h + t), w * h + w * t, w * h, w * t / 2]
        parts = f"a rectangle {w} by {h} and a triangle with base {w} and height {t}"
    elif kind == "arrow":
        w, h, t = rng.randint(3, 6), 2 * rng.randint(1, 3), rng.randint(1, 4)
        verts, size = [(0, 0), (w, 0), (w + t, h // 2), (w, h), (0, h)], (w + t, h)
        cands = [(w + t) * h, w * h + t * h, w * h, t * h / 2]
        parts = f"a rectangle {w} by {h} and a triangle with base {h} and height {t}"
    else:
        w, h = rng.randint(4, 7), rng.randint(3, 5)
        a, b = rng.randint(1, w - 2), rng.randint(1, h - 2)
        verts, size = [(0, 0), (w, 0), (w, h), (a, h), (0, h - b)], (w, h)
        cands = [w * h, w * h - a * b, w * h + a * b / 2, a * b / 2]
        parts = f"a rectangle {w} by {h} with a triangle with legs {a} and {b} cut off one corner"
    verts = _flip(verts, size, rng)
    twice = _twice_area(verts)
    area = twice / 2
    fig = _outline_figure(verts, size)
    text = phrase(
        rng,
        (
            "The outline on the grid combines a rectangle and a triangle: the triangle is either added on or cut "
            "off. Each square is 1 unit long and 1 unit wide. Split the shape into familiar parts to find its area "
            "in square units.",
            "The shape on the grid is a rectangle with a triangle joined to it or cut from it. Each square of the "
            "grid is 1 unit by 1 unit. Find the area of the shape in square units.",
            "What is the area, in square units, of the shape on the grid? It is a rectangle with a triangle added "
            "on or cut off, and each square is 1 unit long and 1 unit wide.",
        ),
    )
    working = [f"Split the shape into {parts}.", f"Add or subtract the areas: {_fmt(area)} square units."]
    return fig, text, area, cands, working


def _coverage(discs, x, y):
    """The share of the unit square at (x, y) inside any of the discs, by 6 by 6 sample points."""
    corners = ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))
    if any(all((px - cx) ** 2 + (py - cy) ** 2 <= r * r for px, py in corners) for cx, cy, r in discs):
        return 1.0
    near = [d for d in discs if (x + 0.5 - d[0]) ** 2 + (y + 0.5 - d[1]) ** 2 <= (d[2] + 0.75) ** 2]
    if not near:
        return 0.0
    inside = 0
    for i in range(6):
        for j in range(6):
            px, py = x + (i + 0.5) / 6, y + (j + 0.5) / 6
            if any((px - cx) ** 2 + (py - cy) ** 2 <= r * r for cx, cy, r in near):
                inside += 1
    return inside / 36


def _composite_hard(rng):
    width, height = rng.randint(10, 13), rng.randint(8, 10)
    rule = rng.choice(("half", "part"))
    for _ in attempts(500, "an irregular shape with clear whole, more than half and less than half squares"):
        discs = [
            (rng.uniform(3, width - 3), rng.uniform(3, height - 3), rng.uniform(2.0, 3.6))
            for _ in range(rng.randint(2, 3))
        ]
        lo_x = max(0, int(min(cx - r for cx, _, r in discs)) - 1)
        hi_x = min(width, int(max(cx + r for cx, _, r in discs)) + 2)
        lo_y = max(0, int(min(cy - r for _, cy, r in discs)) - 1)
        hi_y = min(height, int(max(cy + r for _, cy, r in discs)) + 2)
        shares = {(x, y): _coverage(discs, x, y) for x in range(lo_x, hi_x) for y in range(lo_y, hi_y)}
        whole = [c for c, s in shares.items() if s == 1]
        part = [c for c, s in shares.items() if 0 < s < 1]
        if rule == "half":
            more = [c for c in part if shares[c] > 0.5]
            less = [c for c in part if shares[c] < 0.5]
            if len(whole) >= 6 and len(more) >= 3 and len(less) >= 3:
                break
        elif len(whole) >= 6 and len(part) >= 4 and len(part) % 2 == 0:
            more = less = None
            break
    if rule == "half":
        groups = [
            {"cells": sorted(map(list, whole)), "style": "shaded"},
            {"cells": sorted(map(list, more)), "style": "hatched"},
            {"cells": sorted(map(list, less)), "style": "dotted"},
        ]
        text = phrase(
            rng,
            (
                "An irregular shape is drawn on a square grid. Each square is 1 unit by 1 unit. The shaded squares "
                "are whole squares inside the shape, the hatched squares are more than half inside it and the dotted "
                "squares are less than half inside it. Estimate the area of the shape in square units by counting "
                "every shaded square and every hatched square, and ignoring every dotted square.",
                "Estimate the area of the irregular shape on the grid, in square units, where each square is 1 unit "
                "by 1 unit. Shaded squares are whole squares inside the shape, hatched squares are more than half "
                "inside it and dotted squares are less than half inside it. Count the shaded and hatched squares, "
                "ignoring every dotted square.",
            ),
        )
        answer = len(whole) + len(more)
        cands = [
            len(whole),
            len(whole) + len(more) + len(less),
            len(whole) + (len(more) + len(less)) / 2,
            len(whole) + len(less),
        ]
        working = [f"Whole squares: {len(whole)}. More than half: {len(more)}. {len(whole)} + {len(more)} = {answer}."]
    else:
        groups = [
            {"cells": sorted(map(list, whole)), "style": "shaded"},
            {"cells": sorted(map(list, part)), "style": "hatched"},
        ]
        text = phrase(
            rng,
            (
                "An irregular shape is drawn on a square grid. Each square is 1 unit by 1 unit. The shaded squares "
                "are whole squares inside the shape and the hatched squares are only partly inside it. Estimate the "
                "area of the shape in square units by counting each shaded square as 1 and each hatched square as "
                "half a square.",
                "Estimate the area, in square units, of the irregular shape on the grid, where each square is 1 unit "
                "by 1 unit. Shaded squares are whole squares inside the shape and hatched squares are only partly "
                "inside it. Count each shaded square as 1 and each hatched square as half a square.",
            ),
        )
        answer = len(whole) + len(part) / 2
        cands = [len(whole), len(whole) + len(part), (len(whole) + len(part)) / 2, len(part) / 2]
        working = [
            f"Whole squares: {len(whole)}. Part squares: {len(part)}, worth {len(part) // 2} altogether.",
            f"Total {_fmt(answer)}.",
        ]
    fig = grid.grid_drawing(
        alt="An irregular shape drawn on a square grid, made of shaded, hatched and dotted squares.",
        grid={"type": "square", "width": width, "height": height},
        cells=groups,
    )
    return fig, text, answer, cands, working


_COMPOSITE_TIERS = {"easy": _composite_easy, "medium": _composite_medium, "hard": _composite_hard}


@generator(
    "grid_composite_area",
    level=5,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_composite_area(rng, tier="easy"):
    """Area of a composite shape on a square grid: by counting squares, by splitting, or by estimating."""
    return _finish_number(_tier_of(_COMPOSITE_TIERS, tier), rng)


def _read_composite(question) -> str:
    fig, text = question.figure, question.text
    if "polygons" in fig:
        return _fmt(sum(_twice_area(p["vertices"]) for p in fig["polygons"]) / 2)
    count = {g["style"]: len(g["cells"]) for g in fig["cells"]}
    if "ignoring every dotted" in text:
        return _fmt(count["shaded"] + count["hatched"])
    check("half a square" in text, "cannot read the counting rule")
    return _fmt(count["shaded"] + count["hatched"] / 2)


figures.register_reader("grid_composite_area", _read_composite)

__all__ = [
    "cube_nets",
    "folded_faces",
    "grid_composite_area",
    "grid_cube_stack",
    "grid_net_views",
    "grid_nets_build_solids",
    "grid_solid_properties",
    "non_nets",
    "opposite_cell",
]
