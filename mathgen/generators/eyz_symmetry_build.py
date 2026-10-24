"""Recognising line and rotational symmetry in a drawn shape (AC9M4SP03), on the ``diagram`` figure.

A flat shape is drawn as one outline, with no vertex names and no marks. The learner counts its lines of symmetry,
or its order of rotational symmetry (how many times it fits onto itself in one full turn). Creating a symmetric
design is not asked here: it needs a drawn answer.

- ``eyz_symmetry_count`` (skill ``ey-symmetry-from-drawn-shapes``, ``number``): easy asks the lines of symmetry of
  common shapes (square, rectangle, rhombus, triangles, regular pentagon and hexagon); medium asks the order of
  rotational symmetry of those shapes and the parallelogram, kite and a skew shape; hard draws harder outlines (L, T,
  plus, dart, house, stars, a pinwheel with turns but no lines, a stretched hexagon) and asks for either count.
- ``eyz_symmetry_count_exam`` (exam tier): one shape in three parts: lines of symmetry, order of rotational symmetry,
  then the smallest turn that fits the shape onto itself in degrees (360 over the order, following through from
  the second part).

Every shape is built from known vertices and then checked by brute force: the lines are the reflections, and the
order the rotations about the middle, that carry its outline onto itself. The registered reader repeats the brute
force on the drawn points alone.
"""

from __future__ import annotations

import math
import re

from .. import figures
from ..answers import Distractor
from ..core import GRADER_TOLERANCE, check, generator, latex_answer
from ..figure_kinds.space import diagram
from ._spread_kit import pick_distractors, tagged_numbers, uses_rng

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_EPS = 0.02


def _tier(tier: str) -> str:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return tier


# =============================================================================== brute-force symmetry


def _mean(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _edge_set(pts, mapping):
    n = len(pts)
    return {frozenset((mapping[i], mapping[(i + 1) % n])) for i in range(n)}


def _maps_onto_itself(pts, f) -> bool:
    """Does the transformation ``f`` carry the outline onto itself (vertices to vertices, edges to edges)?"""
    scale = max(math.dist(p, _mean(pts)) for p in pts)
    mapping = []
    for p in pts:
        q = f(p)
        near = [i for i, r in enumerate(pts) if math.dist(q, r) <= _EPS * scale]
        if len(near) != 1:
            return False
        mapping.append(near[0])
    if len(set(mapping)) != len(pts):
        return False
    n = len(pts)
    return {frozenset((i, (i + 1) % n)) for i in range(n)} == _edge_set(pts, mapping)


def _reflector(c, theta: float):
    cos2, sin2 = math.cos(2 * theta), math.sin(2 * theta)
    return lambda p: (
        c[0] + (p[0] - c[0]) * cos2 + (p[1] - c[1]) * sin2,
        c[1] + (p[0] - c[0]) * sin2 - (p[1] - c[1]) * cos2,
    )


def _rotator(c, theta: float):
    cos, sin = math.cos(theta), math.sin(theta)
    return lambda p: (
        c[0] + (p[0] - c[0]) * cos - (p[1] - c[1]) * sin,
        c[1] + (p[0] - c[0]) * sin + (p[1] - c[1]) * cos,
    )


def symmetry_counts(pts) -> tuple[int, int]:
    """``(lines of symmetry, order of rotational symmetry)`` of the polygon with these vertices in order.

    A symmetry fixes the mean of the vertices and carries each vertex to one at the same distance from it, so the
    candidates are the reflections and the turns that take some vertex to another at that distance.
    """
    c = _mean(pts)
    n = len(pts)
    axes: set[int] = set()  # candidate axis directions, in hundredths of a degree
    turns: set[int] = set()  # candidate turns, in hundredths of a degree
    for i in range(n):
        vi = (pts[i][0] - c[0], pts[i][1] - c[1])
        ri = math.hypot(*vi)
        if ri < 1e-9:
            continue
        for j in range(n):
            vj = (pts[j][0] - c[0], pts[j][1] - c[1])
            if abs(math.hypot(*vj) - ri) > _EPS * ri:
                continue
            ai, aj = math.atan2(vi[1], vi[0]), math.atan2(vj[1], vj[0])
            turns.add(round(math.degrees(aj - ai) % 360 * 100))
            axes.add(round(math.degrees(ai + aj) / 2 % 180 * 100))
    lines: list[int] = []
    for a in sorted(axes):
        if _maps_onto_itself(pts, _reflector(c, math.radians(a / 100))) and not any(
            min(abs(a - b), 18000 - abs(a - b)) < 50 for b in lines
        ):
            lines.append(a)
    spins: list[int] = []
    for t in sorted(turns):
        if (
            50 <= t <= 36000 - 50
            and _maps_onto_itself(pts, _rotator(c, math.radians(t / 100)))
            and not any(abs(t - b) < 50 for b in spins)
        ):
            spins.append(t)
    return len(lines), 1 + len(spins)


# =============================================================================== the shapes


def _regular(n: int):
    return [(math.cos(math.radians(90 + 360 * k / n)), math.sin(math.radians(90 + 360 * k / n))) for k in range(n)]


def _star(n: int, inner: float = 0.45):
    out = []
    for k in range(n):
        out.append((math.cos(math.radians(90 + 360 * k / n)), math.sin(math.radians(90 + 360 * k / n))))
        a = math.radians(90 + 360 * (k + 0.5) / n)
        out.append((inner * math.cos(a), inner * math.sin(a)))
    return out


def _pinwheel(n: int):
    """Blades with a twist: n-fold turning symmetry and no line of symmetry."""
    out = []
    for k in range(n):
        a = math.radians(90 + 360 * k / n)
        out.append((math.cos(a), math.sin(a)))
        b = math.radians(90 + 360 * k / n + 360 / n * 0.2)
        out.append((0.4 * math.cos(b), 0.4 * math.sin(b)))
    return out


def _stretched_hexagon():
    return [(1.5 * x, y) for x, y in _regular(6)]


# name -> (vertices, wording of the shape for the working). Counts are never stored: they are found by brute force.
def _catalogue(rng) -> dict[str, list[tuple[float, float]]]:
    a, b = rng.choice((1.4, 1.6, 1.8)), 1.0
    s = rng.choice((0.4, 0.6))
    return {
        "square": [(-1, -1), (1, -1), (1, 1), (-1, 1)],
        "rectangle": [(-a, -b), (a, -b), (a, b), (-a, b)],
        "rhombus": [(0, -b), (a, 0), (0, b), (-a, 0)],
        "parallelogram": [(-a - s, -b), (a - s, -b), (a + s, b), (-a + s, b)],
        "kite": [(0, -1.4), (0.9, 0), (0, 1.0), (-0.9, 0)],
        "isosceles trapezium": [(-1.5, -1), (1.5, -1), (0.8, 1), (-0.8, 1)],
        "equilateral triangle": _regular(3),
        "isosceles triangle": [(-0.8, -1), (0.8, -1), (0, 1.4)],
        "scalene triangle": [(-1.2, -0.8), (1.3, -0.6), (0.1, 1.1)],
        "regular pentagon": _regular(5),
        "regular hexagon": _regular(6),
        "regular octagon": _regular(8),
        "regular nonagon": _regular(9),
        "regular decagon": _regular(10),
        "regular dodecagon": _regular(12),
        "L-shape, equal arms": [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)],
        "L-shape, unequal arms": [(0, 0), (3, 0), (3, 1), (1, 1), (1, 4), (0, 4)],
        "T-shape": [(-0.5, -2), (0.5, -2), (0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (-1.5, 1.5), (-1.5, 0.5), (-0.5, 0.5)],
        "plus": [
            (-0.5, -1.5),
            (0.5, -1.5),
            (0.5, -0.5),
            (1.5, -0.5),
            (1.5, 0.5),
            (0.5, 0.5),
            (0.5, 1.5),
            (-0.5, 1.5),
            (-0.5, 0.5),
            (-1.5, 0.5),
            (-1.5, -0.5),
            (-0.5, -0.5),
        ],
        "S-shape": [(0, 0), (2, 0), (2, 1), (3, 1), (3, 2), (1, 2), (1, 1), (0, 1)],
        "dart": [(0, 1.5), (1.2, -1), (0, -0.3), (-1.2, -1)],
        "house": [(-1, -1), (1, -1), (1, 0.4), (0, 1.4), (-1, 0.4)],
        "five-point star": _star(5),
        "six-point star": _star(6),
        "stretched hexagon": _stretched_hexagon(),
        "three-blade pinwheel": _pinwheel(3),
        "four-blade pinwheel": _pinwheel(4),
        "six-blade pinwheel": _pinwheel(6),
    }


_EASY = (
    "square",
    "rectangle",
    "rhombus",
    "equilateral triangle",
    "isosceles triangle",
    "scalene triangle",
    "regular pentagon",
    "regular hexagon",
)
_MEDIUM = (
    "square",
    "rectangle",
    "rhombus",
    "parallelogram",
    "kite",
    "equilateral triangle",
    "regular pentagon",
    "regular hexagon",
    "regular octagon",
    "isosceles triangle",
    "S-shape",
)
_HARD = (
    "L-shape, equal arms",
    "L-shape, unequal arms",
    "T-shape",
    "plus",
    "S-shape",
    "dart",
    "house",
    "five-point star",
    "six-point star",
    "stretched hexagon",
    "three-blade pinwheel",
    "four-blade pinwheel",
    "parallelogram",
    "kite",
    "isosceles trapezium",
)
_EXAM = (
    "square",
    "rectangle",
    "rhombus",
    "kite",
    "isosceles trapezium",
    "parallelogram",
    "equilateral triangle",
    "regular pentagon",
    "regular hexagon",
    "regular octagon",
    "regular nonagon",
    "regular decagon",
    "regular dodecagon",
    "plus",
    "five-point star",
    "three-blade pinwheel",
    "four-blade pinwheel",
    "six-blade pinwheel",
    "S-shape",
    "L-shape, unequal arms",
)

_SIDES_WORD = {"rectangle": 4, "rhombus": 4, "parallelogram": 4, "square": 4, "kite": 4, "isosceles trapezium": 4}


def _place(pts, rng) -> list[list[float]]:
    """Scale the shape to fill the frame, turn it by a multiple of 5 degrees, and put its middle at (50, 50)."""
    c = _mean(pts)
    r = max(math.dist(p, c) for p in pts)
    turn = math.radians(rng.choice(range(0, 180, 5)))
    k = 38.0 / r
    out = []
    for x, y in pts:
        x, y = (x - c[0]) * k, (y - c[1]) * k
        out.append(
            [
                round(50 + x * math.cos(turn) - y * math.sin(turn), 3),
                round(50 + x * math.sin(turn) + y * math.cos(turn), 3),
            ]
        )
    return out


def _shape(rng, name: str):
    verts = _catalogue(rng)[name]
    placed = _place(verts, rng)
    lines, order = symmetry_counts(placed)
    check((lines, order) == symmetry_counts(verts), "turning and scaling changed the symmetry")
    ids = [f"p{i + 1}" for i in range(len(placed))]
    fig = diagram(
        alt="a flat shape drawn as one outline",
        points=[{"id": i, "at": at} for i, at in zip(ids, placed)],
        polygons=[{"points": ids}],
        frame=(100, 100),
    )
    return fig, lines, order, len(placed)


_COUNTS: dict[str, tuple[int, int]] = {}


def _by_count(rng, names, index: int) -> str:
    """A shape from ``names`` whose count (0 lines, 1 order) is drawn over the counts on offer (the larger
    the likelier), so the many shapes with 1 or 2 lines do not crowd out the counts that leave wrong values both sides."""
    if not _COUNTS:
        import random

        for key, verts in _catalogue(random.Random(0)).items():
            _COUNTS[key] = symmetry_counts(verts)
    groups: dict[int, list[str]] = {}
    for name in names:
        groups.setdefault(_COUNTS[name][index], []).append(name)
    counts = sorted(groups)
    # the larger the count, the more wrong values fit below it: weight each count by its size
    chosen = rng.choices(counts, weights=[c + 1 - index for c in counts])[0]
    return rng.choice(groups[chosen])


def _number_question(text, answer, cands, working, fig, *, low=0, high=40) -> dict:
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        answer_kind="number",
        working=working,
        distractors=tagged_numbers(answer, cands, low=low, high=high),
        figure=fig,
    )


def _line_wrongs(name: str, lines: int, order: int, sides: int) -> list[tuple[int, str | None]]:
    out: list[tuple[int, str | None]] = []
    if name in ("rectangle", "parallelogram"):
        out.append((lines + 2, "eyz-counts-diagonals-as-lines"))
    out.append((sides, "eyz-gives-the-number-of-sides"))
    out.append((order, "eyz-gives-the-order-for-the-lines"))
    out.append((2 * lines, "eyz-counts-each-line-twice"))
    out.append((lines + 1, None))
    out.append((lines + 2, None))
    out.append((max(lines - 1, 0), None))
    return out


def _order_wrongs(name: str, lines: int, order: int, sides: int) -> list[tuple[int, str | None]]:
    out: list[tuple[int, str | None]] = [
        (lines, "eyz-gives-the-lines-for-the-order"),
        (sides, "eyz-gives-the-number-of-sides"),
    ]
    out.append(
        (order - 1, "eyz-forgets-the-turn-back-to-the-start") if order > 1 else (0, "eyz-says-no-rotational-symmetry")
    )
    out.append((2 * order, "eyz-counts-twice-round"))
    out.append((order + 1, None))
    out.append((order + 2, None))
    return out


def _lines_working(lines: int) -> str:
    if lines == 0:
        return "No fold makes the two halves match, so there are 0 lines of symmetry."
    return f"Folding along {'the' if lines == 1 else 'each of the'} {lines} line{'' if lines == 1 else 's'} makes the two halves match."


def _lines_question(rng, name: str) -> dict:
    fig, lines, order, sides = _shape(rng, name)
    sides = _SIDES_WORD.get(name, sides)
    text = rng.choice(
        (
            "How many lines of symmetry does this shape have?",
            "Count the lines of symmetry of this shape.",
            "A mirror line makes both halves match. How many does the shape have?",
        )
    )
    working = [_lines_working(lines)]
    return _number_question(text, lines, _line_wrongs(name, lines, order, sides), working, fig)


def _order_question(rng, name: str) -> dict:
    fig, lines, order, sides = _shape(rng, name)
    sides = _SIDES_WORD.get(name, sides)
    text = rng.choice(
        (
            "How many times does this shape fit onto itself in one full turn?",
            "What is the order of rotational symmetry of this shape?",
            "Turn the shape one full turn. How many times does it look the same?",
        )
    )
    working = [
        f"Turn the shape a full turn about its middle. It fits onto itself {order} time{'s' if order != 1 else ''}, counting the start."
    ]
    return _number_question(text, order, _order_wrongs(name, lines, order, sides), working, fig, low=1)


@generator("eyz_symmetry_count", level=4, presets=_PRESETS)
@uses_rng
def eyz_symmetry_count(rng, tier="easy"):
    _tier(tier)
    if tier == "easy":
        return _lines_question(rng, _by_count(rng, _EASY, 0))
    if tier == "medium":
        return _order_question(rng, _by_count(rng, _MEDIUM, 1))
    lines_first = rng.random() < 0.5
    name = _by_count(rng, _HARD, 0 if lines_first else 1)
    return (_lines_question if lines_first else _order_question)(rng, name)


def _read_symmetry(q) -> str:
    fig = q.figure
    pos = {p["id"]: p["at"] for p in fig["points"]}
    pts = [pos[i] for i in fig["polygons"][0]["points"]]
    lines, order = symmetry_counts(pts)
    if q.parts:
        k = int(re.search(r"\d+", q.parts[3]["prompt"] if isinstance(q.parts[3], dict) else q.parts[3].prompt).group())
        return f"(a) {lines}; (b) {order}; (c) {360 // order}; (d) {k * (360 // order)}"
    return str(lines if "line" in q.text else order)


figures.register_reader("eyz_symmetry_count", _read_symmetry)
figures.register_reader("eyz_symmetry_count_exam", _read_symmetry)


# =============================================================================== exam


def _part(
    prompt, value, marks, candidates, working, *, follow_through=None, sensible_range=(-0.5, None), low=0
) -> dict:
    """A number-graded part whose answer is the whole number ``value`` (zero allowed)."""
    tagged = [Distractor(str(v), mid, v) for v, mid in candidates if isinstance(v, int) and v >= 0]
    picked = pick_distractors(str(value), tagged, 3, lo=low, near_misses=True, tolerance=GRADER_TOLERANCE)
    check(len(picked) >= 2, "a part needs at least two distractors")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=latex_answer(str(value)),
        marks=marks,
        sensible_range=sensible_range,
        working=working,
        distractors=picked,
    )
    if follow_through is not None:
        part["follow_through"] = follow_through
    return part


@generator("eyz_symmetry_count_exam", level=4, presets={"exam": dict(tier="exam")})
@uses_rng
def eyz_symmetry_count_exam(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    name = _by_count(rng, _EXAM, rng.randint(0, 1))
    fig, lines, order, sides = _shape(rng, name)
    sides = _SIDES_WORD.get(name, sides)
    turn = 360 // order
    check(turn * order == 360, "the smallest turn is a whole number of degrees")
    k = rng.randint(2, 19)
    stem = rng.choice(
        (
            "Look at this shape.",
            "A shape is drawn here. Think about how it can be flipped and turned onto itself.",
            "Here is a shape. Answer each question about its symmetry.",
        )
    )
    return dict(
        text=stem,
        figure=fig,
        parts=[
            _part(
                "How many lines of symmetry does the shape have?",
                lines,
                1,
                _line_wrongs(name, lines, order, sides),
                [_lines_working(lines)],
            ),
            _part(
                "What is its order of rotational symmetry?",
                order,
                1,
                _order_wrongs(name, lines, order, sides),
                [f"The shape fits onto itself {order} time{'s' if order != 1 else ''} in a full turn."],
                low=1,  # every shape fits onto itself at least once in a full turn
            ),
            _part(
                "Hence, what is the smallest turn that fits it onto itself, in degrees?",
                turn,
                2,
                [
                    (360, "eyz-gives-a-full-turn"),
                    (180, "eyz-always-gives-a-half-turn"),
                    (order, "eyz-gives-the-order-not-the-angle"),
                    (360 - turn, "eyz-gives-the-rest-of-the-turn"),
                    (turn * 2, None),
                    (turn + 30, None),
                    (max(turn // 2, 1), None),
                ],
                [f"360 ÷ {order} = {turn} degrees."],
                follow_through="360/part_b",
                sensible_range=(0, 360.5),
            ),
            _part(
                f"It is turned through {k} of those smallest turns. How many degrees is that?",
                k * turn,
                2,
                [
                    (k * order, "eyz-gives-the-order-not-the-angle"),
                    (turn, "eyz-gives-one-turn-for-several"),
                    (k + turn, "eyz-gives-one-turn-for-several"),
                    (k * turn + turn, None),
                    (k * turn - turn, None),
                    (k * turn + 10, None),
                ],
                [f"{k} × {turn} = {k * turn} degrees."],
                follow_through=f"{k}*part_c",
                sensible_range=(0, None),
            ),
        ],
    )
