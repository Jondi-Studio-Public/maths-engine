"""``f10_symmetry_lines_rotation``: lines of symmetry and order of rotational symmetry of a named shape.

Skill ``f10-symmetry-construction`` (AC9M4SP03). The learner is told a shape by its name or its features
(nothing is drawn) and states two numbers as a labelled pair, ``lines = 4, order = 4``: how many lines of
symmetry the shape has and its order of rotational symmetry (the number of positions in one full turn, the
starting position included, in which the shape fits exactly onto itself; a shape with no turn symmetry
other than the full turn has order 1). The "create a symmetric pattern" half of the curriculum clause is a
drawn deliverable and is not built here.

Every shape's counts come from a table of vertex coordinates, not from typed-in numbers. ``_symmetries``
finds the lines and rotations of the polygon directly (try each isometry that sends the first vertex onto a
vertex, keep the ones that preserve every edge), and ``SHAPES`` also carries the textbook counts so a wrong
coordinate or a wrong textbook value fails loudly at import time.

Tiers:

- easy:   a regular polygon of 3 to 8 sides, a rectangle or an isosceles triangle, by name.
- medium: any named triangle or quadrilateral (scalene, right-angled, parallelogram, rhombus, kite, trapeziums)
          or a regular polygon of up to 10 sides, by name. Lines and order now differ, so the learner cannot
          assume they are equal.
- hard:   the shape is described only by its features (so it must be identified first), or is a regular
          polygon given by the size of each exterior angle, so the number of sides must be worked out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from ..core import check, generator, latex_answer, pick_distractors

_TOL = 1e-7


# -- independent symmetry finder ------------------------------------------------------------------


def _symmetries(vertices):
    """Return (number of lines of symmetry, order of rotational symmetry) of a polygon.

    `vertices` is the polygon's corners in order around it. An isometry about the centroid is a symmetry
    when it sends corners onto corners and every side onto a side.
    """
    n = len(vertices)
    cx = sum(x for x, _ in vertices) / n
    cy = sum(y for _, y in vertices) / n
    rel = [(x - cx, y - cy) for x, y in vertices]
    edges = {frozenset((i, (i + 1) % n)) for i in range(n)}

    def index_of(point):
        for k, (x, y) in enumerate(rel):
            if math.hypot(x - point[0], y - point[1]) < _TOL:
                return k
        return None

    def preserves(transform):
        images = [index_of(transform(p)) for p in rel]
        if None in images or len(set(images)) != n:
            return False
        return {frozenset((images[a], images[b])) for a, b in map(tuple, edges)} == edges

    r0 = math.hypot(*rel[0])
    a0 = math.atan2(rel[0][1], rel[0][0])
    rotations = set()
    lines = set()
    for j in range(n):
        rj = math.hypot(*rel[j])
        if abs(rj - r0) > _TOL:
            continue
        aj = math.atan2(rel[j][1], rel[j][0])
        turn = aj - a0

        def rotate(p, turn=turn):
            c, s = math.cos(turn), math.sin(turn)
            return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)

        if preserves(rotate):
            rotations.add(round((turn % (2 * math.pi)) / (2 * math.pi), 6) % 1.0)
        axis = (a0 + aj) / 2

        def reflect(p, axis=axis):
            c, s = math.cos(2 * axis), math.sin(2 * axis)
            return (p[0] * c + p[1] * s, p[0] * s - p[1] * c)

        if preserves(reflect):
            lines.add(round((axis % math.pi) / math.pi, 6) % 1.0)
    return len(lines), len(rotations)


def _regular(n, radius=5.0):
    return [(radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n)) for k in range(n)]


def _star(points=5, outer=5.0, inner=2.0):
    out = []
    for k in range(2 * points):
        r = outer if k % 2 == 0 else inner
        a = math.pi * k / points
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


@dataclass(frozen=True)
class Shape:
    key: str
    name: str  # with its article: "a rectangle"
    features: str  # a description that picks out this shape alone: "a quadrilateral with ..."
    sides: int
    vertices: tuple
    lines: int  # textbook value, cross-checked against the coordinates
    order: int


def _poly(key, name, features, sides, vertices, lines, order):
    return Shape(key, name, features, sides, tuple(vertices), lines, order)


_REGULAR_NAMES = {
    3: "an equilateral triangle",
    4: "a square",
    5: "a regular pentagon",
    6: "a regular hexagon",
    7: "a regular heptagon",
    8: "a regular octagon",
    9: "a regular nonagon",
    10: "a regular decagon",
    12: "a regular dodecagon",
}

SHAPES = {
    s.key: s
    for s in [
        *[
            _poly(
                f"regular-{n}",
                name,
                f"a regular polygon with {n} sides",
                n,
                _regular(n),
                n,
                n,
            )
            for n, name in _REGULAR_NAMES.items()
        ],
        _poly(
            "isosceles-triangle",
            "an isosceles triangle with sides 5 cm, 5 cm and 6 cm",
            "a triangle with exactly two equal sides",
            3,
            [(0, 0), (6, 0), (3, 4)],
            1,
            1,
        ),
        _poly(
            "scalene-triangle",
            "a scalene triangle",
            "a triangle whose three sides all have different lengths",
            3,
            [(0, 0), (6, 0), (1, 3)],
            0,
            1,
        ),
        _poly(
            "right-isosceles-triangle",
            "a right-angled isosceles triangle",
            "a triangle with one right angle and two equal sides",
            3,
            [(0, 0), (4, 0), (0, 4)],
            1,
            1,
        ),
        _poly(
            "right-scalene-triangle",
            "a triangle with sides 3 cm, 4 cm and 5 cm",
            "a triangle with one right angle and no two sides equal",
            3,
            [(0, 0), (4, 0), (0, 3)],
            0,
            1,
        ),
        _poly(
            "rectangle",
            "a rectangle that is not a square",
            "a quadrilateral with four right angles and sides of two different lengths",
            4,
            [(0, 0), (6, 0), (6, 3), (0, 3)],
            2,
            2,
        ),
        _poly(
            "parallelogram",
            "a parallelogram with no right angles and adjacent sides of different lengths",
            "a quadrilateral with two pairs of parallel sides, no right angles and adjacent sides of different lengths",
            4,
            [(0, 0), (5, 0), (7, 3), (2, 3)],
            0,
            2,
        ),
        _poly(
            "rhombus",
            "a rhombus that is not a square",
            "a quadrilateral with four equal sides and no right angles",
            4,
            [(0, 2), (3, 0), (6, 2), (3, 4)],
            2,
            2,
        ),
        _poly(
            "kite",
            "a kite that is not a rhombus",
            "a quadrilateral with two pairs of equal adjacent sides, whose four sides are not all equal",
            4,
            [(-2, 0), (0, 4), (2, 0), (0, -2)],
            1,
            1,
        ),
        _poly(
            "isosceles-trapezium",
            "an isosceles trapezium",
            "a quadrilateral with exactly one pair of parallel sides and the other two sides equal in length",
            4,
            [(0, 0), (6, 0), (5, 3), (1, 3)],
            1,
            1,
        ),
        _poly(
            "right-trapezium",
            "a right trapezium",
            "a quadrilateral with exactly one pair of parallel sides and two right angles",
            4,
            [(0, 0), (6, 0), (4, 3), (0, 3)],
            0,
            1,
        ),
        _poly(
            "arrowhead",
            "an arrowhead (a concave kite)",
            "a quadrilateral with one reflex angle and two pairs of equal adjacent sides",
            4,
            [(0, 4), (3, -2), (0, 0), (-3, -2)],
            1,
            1,
        ),
        _poly(
            "s-tetromino",
            "the S-shaped tetromino (four identical squares joined edge to edge: two side by side in the bottom row "
            "and two side by side in the row above, that pair shifted one square to the left)",
            "made of four identical squares joined edge to edge: two side by side in the bottom row and two side by "
            "side in the row above, that pair shifted one square to the left",
            8,
            [(1, 0), (3, 0), (3, 1), (2, 1), (2, 2), (0, 2), (0, 1), (1, 1)],
            0,
            2,
        ),
        _poly(
            "l-tetromino",
            "the L-shaped tetromino (a column of three identical squares with one more square joined to the right of "
            "the bottom one)",
            "made of four identical squares joined edge to edge: a column of three with one more square joined to the "
            "right of the bottom one",
            6,
            [(0, 0), (2, 0), (2, 1), (1, 1), (1, 3), (0, 3)],
            0,
            1,
        ),
        _poly(
            "t-tetromino",
            "the T-shaped tetromino (a row of three identical squares with one more square joined above the middle one)",
            "made of four identical squares joined edge to edge: a row of three with one more square joined above the "
            "middle one",
            8,
            [(0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (1, 2), (1, 1), (0, 1)],
            1,
            1,
        ),
        _poly(
            "plus-sign",
            "a plus sign made from five identical squares",
            "a shape made of a central square with an identical square joined to each of its four sides",
            12,
            [
                (1, 1),
                (1, 3),
                (-1, 3),
                (-1, 1),
                (-3, 1),
                (-3, -1),
                (-1, -1),
                (-1, -3),
                (1, -3),
                (1, -1),
                (3, -1),
                (3, 1),
            ],
            4,
            4,
        ),
        _poly(
            "five-pointed-star",
            "a regular five-pointed star",
            "a star with five identical points, evenly spaced",
            10,
            _star(5),
            5,
            5,
        ),
    ]
}

# The coordinates are the authority: this runs once at import and fails if either side is wrong.
for _shape in SHAPES.values():
    _found = _symmetries(_shape.vertices)
    if _found != (_shape.lines, _shape.order):
        raise AssertionError(f"symmetry table disagrees for {_shape.key}: computed {_found}")

EASY_KEYS = [f"regular-{n}" for n in range(3, 9)] + ["rectangle", "isosceles-triangle"]
MEDIUM_KEYS = [f"regular-{n}" for n in (5, 6, 8, 10)] + [
    "s-tetromino",
    "l-tetromino",
    "t-tetromino",
    "isosceles-triangle",
    "scalene-triangle",
    "right-isosceles-triangle",
    "right-scalene-triangle",
    "rectangle",
    "parallelogram",
    "rhombus",
    "kite",
    "isosceles-trapezium",
    "right-trapezium",
]
HARD_FEATURE_KEYS = [
    "isosceles-triangle",
    "scalene-triangle",
    "right-isosceles-triangle",
    "right-scalene-triangle",
    "rectangle",
    "parallelogram",
    "rhombus",
    "kite",
    "isosceles-trapezium",
    "right-trapezium",
    "arrowhead",
    "s-tetromino",
    "l-tetromino",
    "t-tetromino",
    "plus-sign",
    "five-pointed-star",
]
EXTERIOR_SIDE_COUNTS = [5, 6, 8, 9, 10, 12, 15, 18, 20]

_ORDER_DEFINITION = (
    "The order of rotational symmetry is the number of positions in one full turn, the starting position "
    "included, where the shape fits exactly onto itself (a shape with no turn symmetry has order 1)."
)
_FORMAT = "Give your answer as lines = _, order = _."


@lru_cache(maxsize=None)
def _counts(key):
    return _symmetries(SHAPES[key].vertices)


def _regular_counts(n):
    """Counts for a regular n-gon of any size, from the same finder (n up to 20 are all computed)."""
    return _symmetries(_regular(n))


def _distractors(rng, lines, order, sides, quad):
    """(text, misconception id) pairs; every kept text differs from the answer in at least one value."""
    cand = []

    def add(l, o, note):
        if (l, o) != (lines, order) and l >= 0 and o >= 0:
            cand.append((f"lines = {l}, order = {o}", note))

    add(sides, order, "lines-equals-sides")
    add(order, lines, "lines-order-swapped")
    add(lines, lines, "order-equals-lines")
    if order == 1:
        add(lines, 0, "no-turn-symmetry-is-order-zero")
    if quad:
        add(2, order, "lines-from-diagonals")
    if order > 1:
        add(lines, 1, "turn-symmetry-ignored")
    if lines > 0:
        add(0, order, "line-symmetry-ignored")
    rng.shuffle(cand)
    seen = {}
    for text, note in cand:
        seen.setdefault(text, note)
    return seen


def _build(rng, text, lines, order, sides, quad, working):
    answer_text = f"lines = {lines}, order = {order}"
    notes = _distractors(rng, lines, order, sides, quad)
    picked = pick_distractors(answer_text, list(notes))[:3]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"lines": lines, "order": order},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=picked,
        option_notes={t: notes[t] for t in picked},
    )


_NAMED_FRAMES = [
    "Consider {name}.",
    "A floor tile is shaped like {name}.",
    "A sticker is cut in the shape of {name}.",
    "A designer sketches {name}.",
    "Think about {name}.",
]
_DESCRIBED_FRAMES = [
    "A shape is {features}.",
    "A cut-out is {features}.",
    "A window pane is {features}.",
]


def _named(rng, keys):
    shape = SHAPES[rng.choice(keys)]
    lines, order = _counts(shape.key)
    check((lines, order) == (shape.lines, shape.order), "symmetry counts disagree with the table")
    frame = rng.choice(_NAMED_FRAMES).format(name=shape.name)
    text = (
        f"{frame} How many lines of symmetry does it have, and what is its order of "
        f"rotational symmetry? {_ORDER_DEFINITION} {_FORMAT}"
    )
    working = [
        f"{shape.name.capitalize()} has {lines} line{'s' if lines != 1 else ''} of symmetry.",
        f"It fits onto itself in {order} position{'s' if order != 1 else ''} in one full turn, so its order is {order}.",
    ]
    return _build(rng, text, lines, order, shape.sides, shape.sides == 4, working)


def _described(rng):
    shape = SHAPES[rng.choice(HARD_FEATURE_KEYS)]
    lines, order = _counts(shape.key)
    check((lines, order) == (shape.lines, shape.order), "symmetry counts disagree with the table")
    frame = rng.choice(_DESCRIBED_FRAMES).format(features=shape.features)
    text = (
        f"{frame} How many lines of symmetry does it have, and what is its order of "
        f"rotational symmetry? {_ORDER_DEFINITION} {_FORMAT}"
    )
    working = [
        f"The description fits {shape.name}.",
        f"It has {lines} line{'s' if lines != 1 else ''} of symmetry and order {order}.",
    ]
    return _build(rng, text, lines, order, shape.sides, shape.sides == 4, working)


def _from_exterior_angle(rng):
    n = rng.choice(EXTERIOR_SIDE_COUNTS)
    exterior = 360 // n
    check(exterior * n == 360, "exterior angle does not divide 360")
    lines, order = _regular_counts(n)
    check(lines == n and order == n, "a regular polygon's counts must equal its number of sides")
    text = (
        f"Each exterior angle of a regular polygon is {exterior}°. How many lines of symmetry does the "
        f"polygon have, and what is its order of rotational symmetry? {_ORDER_DEFINITION} {_FORMAT}"
    )
    answer_text = f"lines = {lines}, order = {order}"
    cand = []
    for slip_n, note in [
        (n - 1, "sides-from-angle-slip"),
        (n + 1, "sides-from-angle-slip"),
        (180 // exterior if 180 % exterior == 0 else None, "half-turn-used-for-sides"),
        (exterior, "angle-reported-as-count"),
    ]:
        if slip_n is not None and slip_n >= 3 and slip_n != n:
            cand.append((f"lines = {slip_n}, order = {slip_n}", note))
    cand.append((f"lines = {n}, order = 1", "regular-polygon-turn-order-one"))
    cand.append((f"lines = {n * 2}, order = {n}", "lines-doubled"))
    rng.shuffle(cand)
    notes = {}
    for t, note in cand:
        notes.setdefault(t, note)
    picked = pick_distractors(answer_text, list(notes))[:3]
    working = [
        f"Exterior angles of a regular polygon add to 360°, so n = 360 ÷ {exterior} = {n} sides.",
        f"A regular polygon with {n} sides has {n} lines of symmetry and rotational order {n}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"lines": lines, "order": order},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=picked,
        option_notes={t: notes[t] for t in picked},
    )


@generator(
    "f10_symmetry_lines_rotation",
    level=6,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def f10_symmetry_lines_rotation(rng, tier="easy"):
    """State a shape's number of lines of symmetry and its order of rotational symmetry."""
    if tier == "easy":
        return _named(rng, EASY_KEYS)
    if tier == "medium":
        return _named(rng, MEDIUM_KEYS)
    if tier == "hard":
        return _from_exterior_angle(rng) if rng.random() < 0.4 else _described(rng)
    raise ValueError("tier must be easy, medium or hard")
