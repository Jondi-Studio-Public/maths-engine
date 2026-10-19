"""Shared helpers for the ``fr2a_*`` generators: moves on the coordinate plane, 3D boxes, and drawn grids.

Not a generator module (the leading underscore keeps the package loader from importing it as one). It holds:

* a small vocabulary of plane moves (translation, reflection in an axis or a vertical or horizontal line, half turn
  and quarter turn about a point), each applied two ways (tuple arithmetic and complex numbers), written as a phrase
  for a question stem and read back from that phrase;
* ``plane_figure``: draws a polygon on a ``cartesian`` grid with its corners named;
* ``labelled_pair``: the ``x = 3, y = -2`` answer text and its dict;
* ``box_figure``: a cuboid in oblique view as a ``diagram``, hidden edges dashed.

Every figure uses an existing kind (``cartesian`` or ``diagram``) and nothing else.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import check
from mathgen.figure_kinds import space

# ---------------------------------------------------------------------------------------- answer text


def num_text(v) -> str:
    """A whole number or a finite decimal as the learner would type it (``-3``, ``2.5``)."""
    f = Fraction(v)
    if f.denominator == 1:
        return str(f.numerator)
    check((f * 10).denominator == 1 or (f * 100).denominator == 1, f"{f} is not a short decimal")
    text = f"{float(f):.2f}".rstrip("0").rstrip(".")
    return text


def labelled_pair(**values) -> tuple[str, dict]:
    """``("x = 3, y = -2", {"x": 3, "y": -2})`` for ``labelled_pair(x=3, y=-2)``."""
    text = ", ".join(f"{k} = {num_text(v)}" for k, v in values.items())
    return text, {k: (int(v) if Fraction(v).denominator == 1 else float(v)) for k, v in values.items()}


def pair_distractor(tag: str | None, **values) -> Distractor:
    text, value = labelled_pair(**values)
    return Distractor(text, tag, value)


def units(n: int) -> str:
    """``1 unit``, ``3 units``."""
    return f"{n} unit" if n == 1 else f"{n} units"


def coord_text(x, y) -> str:
    return f"({num_text(x)}, {num_text(y)})"


# ---------------------------------------------------------------------------------------- plane moves
# A step is a tuple: ("T", dx, dy) translation; ("RX",) reflection in the x-axis; ("RY",) in the y-axis;
# ("RV", k) in the line x = k; ("RH", k) in the line y = k; ("H", cx, cy) half turn about (cx, cy);
# ("Q", cx, cy, "clockwise" | "anticlockwise") quarter turn about (cx, cy).


def apply_step(step, p):
    """Image of point ``p`` under one step, by coordinate arithmetic."""
    x, y = p
    kind = step[0]
    if kind == "T":
        return (x + step[1], y + step[2])
    if kind == "RX":
        return (x, -y)
    if kind == "RY":
        return (-x, y)
    if kind == "RV":
        return (2 * step[1] - x, y)
    if kind == "RH":
        return (x, 2 * step[1] - y)
    if kind == "H":
        return (2 * step[1] - x, 2 * step[2] - y)
    if kind == "Q":
        cx, cy = step[1], step[2]
        if step[3] == "clockwise":
            return (cx + (y - cy), cy - (x - cx))
        return (cx - (y - cy), cy + (x - cx))
    raise ValueError(f"unknown step {step!r}")


def apply_step_complex(step, p):
    """The same image by complex-number arithmetic (an independent second route)."""
    z = complex(p[0], p[1])
    kind = step[0]
    if kind == "T":
        w = z + complex(step[1], step[2])
    elif kind == "RX":
        w = z.conjugate()
    elif kind == "RY":
        w = -z.conjugate()
    elif kind == "RV":
        w = 2 * step[1] - z.conjugate()
    elif kind == "RH":
        w = z.conjugate() + 2j * step[1]
    elif kind == "H":
        c = complex(step[1], step[2])
        w = c - (z - c)
    elif kind == "Q":
        c = complex(step[1], step[2])
        w = c + (-1j if step[3] == "clockwise" else 1j) * (z - c)
    else:
        raise ValueError(f"unknown step {step!r}")
    return (round(w.real), round(w.imag))


def image_of(steps, p):
    """Image of ``p`` after ``steps`` in order, checked against the complex-number route."""
    q = p
    z = p
    for s in steps:
        q = apply_step(s, q)
        z = apply_step_complex(s, z)
    check(tuple(q) == tuple(z), "the two routes to the image disagree")
    return q


def _move(n: int, positive: str, negative: str) -> str:
    unit = "unit" if abs(n) == 1 else "units"
    return f"{abs(n)} {unit} {positive if n > 0 else negative}"


def step_phrase(step) -> str:
    """The step as a clause after 'is': ``reflected in the line x = 3``."""
    kind = step[0]
    if kind == "T":
        dx, dy = step[1], step[2]
        parts = []
        if dx:
            parts.append(_move(dx, "to the right", "to the left"))
        if dy:
            parts.append(_move(dy, "up", "down"))
        return "translated " + " and ".join(parts)
    if kind == "RX":
        return "reflected in the x-axis"
    if kind == "RY":
        return "reflected in the y-axis"
    if kind == "RV":
        return f"reflected in the line x = {step[1]}"
    if kind == "RH":
        return f"reflected in the line y = {step[1]}"
    if kind == "H":
        return "rotated 180° about " + _centre(step[1], step[2])
    if kind == "Q":
        return f"rotated 90° {step[3]} about " + _centre(step[1], step[2])
    raise ValueError(f"unknown step {step!r}")


def _centre(cx, cy) -> str:
    return "the origin" if (cx, cy) == (0, 0) else f"the point {coord_text(cx, cy)}"


_NUM = r"(-?\d+)"
_PHRASES = [
    (re.compile(r"reflected in the x-axis"), lambda m: ("RX",)),
    (re.compile(r"reflected in the y-axis"), lambda m: ("RY",)),
    (re.compile(rf"reflected in the line x = {_NUM}"), lambda m: ("RV", int(m[1]))),
    (re.compile(rf"reflected in the line y = {_NUM}"), lambda m: ("RH", int(m[1]))),
    (
        re.compile(rf"rotated 180° about (?:the origin|the point \({_NUM}, {_NUM}\))"),
        lambda m: ("H", int(m[1] or 0), int(m[2] or 0)),
    ),
    (
        re.compile(rf"rotated 90° (clockwise|anticlockwise) about (?:the origin|the point \({_NUM}, {_NUM}\))"),
        lambda m: ("Q", int(m[2] or 0), int(m[3] or 0), m[1]),
    ),
]
_TRANSLATION = re.compile(r"translated (\d+) units? (to the right|to the left|up|down)(?: and (\d+) units? (up|down))?")


def steps_in(text: str) -> list:
    """Every step named in ``text``, in the order the text names them (the figure readers use this)."""
    found = []
    for pattern, build in _PHRASES:
        for m in pattern.finditer(text):
            found.append((m.start(), build(m)))
    for m in _TRANSLATION.finditer(text):
        dx = dy = 0
        for amount, word in ((m[1], m[2]), (m[3], m[4])):
            if amount is None:
                continue
            n = int(amount)
            if word == "to the right":
                dx += n
            elif word == "to the left":
                dx -= n
            elif word == "up":
                dy += n
            else:
                dy -= n
        found.append((m.start(), ("T", dx, dy)))
    return [s for _, s in sorted(found, key=lambda t: t[0])]


# ---------------------------------------------------------------------------------------- the plane figure


def _cross(o, a, b) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def plane_figure(
    shapes,
    *,
    alt: str,
    extra_points=(),
    lines=(),
    hull=(),
):
    """A ``cartesian`` grid with each shape drawn as an outline and its corners labelled.

    ``shapes`` is a list of ``(corners, labels)`` where ``corners`` are (x, y) in order and ``labels`` the
    corner names (``None`` leaves a corner unnamed). ``extra_points`` are ``(x, y, label)``; ``lines`` are
    ``((x1, y1), (x2, y2), label)`` infinite lines; ``hull`` is more points the grid must cover (an image that
    is not drawn). The grid always includes the origin and a one-unit margin.
    """
    xs, ys = [0], [0]
    for corners, _ in shapes:
        xs += [p[0] for p in corners]
        ys += [p[1] for p in corners]
    for x, y, _ in extra_points:
        xs.append(x)
        ys.append(y)
    for x, y in hull:
        xs.append(x)
        ys.append(y)
    x0, x1 = min(xs) - 1, max(xs) + 1
    y0, y1 = min(ys) - 1, max(ys) + 1
    # a square-ish window keeps the grid readable
    span = max(x1 - x0, y1 - y0, 8)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = int(cx - span / 2 - 0.5), int(cx + span / 2 + 0.5)
    y0, y1 = int(cy - span / 2 - 0.5), int(cy + span / 2 + 0.5)
    check(x0 <= min(xs) - 1 and x1 >= max(xs) + 1 and y0 <= min(ys) - 1 and y1 >= max(ys) + 1, "grid window too small")
    segments, points = [], []
    for corners, names in shapes:
        n = len(corners)
        for i in range(n):
            segments.append({"from": list(corners[i]), "to": list(corners[(i + 1) % n])})
        for p, name in zip(corners, names):
            if name:
                points.append({"at": list(p), "label": name})
    for x, y, label in extra_points:
        points.append({"at": [x, y], "label": label})
    return figures.cartesian(
        x_range=(x0, x1),
        y_range=(y0, y1),
        alt=alt,
        grid=1,
        equal_scale=True,
        points=points,
        segments=segments,
        lines=[{"through": [list(a), list(b)], "label": label} for a, b, label in lines],
    )


def leaks(fig: dict, *answer_texts: str) -> bool:
    """Does any answer text appear in the figure's visible text? (Generators redraw when it does.)"""
    try:
        figures.assert_no_leak(fig, answer_texts)
    except AssertionError:
        return True
    return False


# ---------------------------------------------------------------------------------------- the 3D box


BOX_NAMES = {  # (x, y, z) in {0, 1}^3 -> vertex name; O is the origin, x along OP, y along OR, z along OS
    (0, 0, 0): "O",
    (1, 0, 0): "P",
    (1, 1, 0): "Q",
    (0, 1, 0): "R",
    (0, 0, 1): "S",
    (1, 0, 1): "T",
    (1, 1, 1): "U",
    (0, 1, 1): "V",
}
_EDGES = [
    ((0, 0, 0), (1, 0, 0)),
    ((1, 0, 0), (1, 1, 0)),
    ((1, 1, 0), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 0)),
    ((0, 0, 1), (1, 0, 1)),
    ((1, 0, 1), (1, 1, 1)),
    ((1, 1, 1), (0, 1, 1)),
    ((0, 1, 1), (0, 0, 1)),
    ((0, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 0, 1)),
    ((1, 1, 0), (1, 1, 1)),
    ((0, 1, 0), (0, 1, 1)),
]
_HIDDEN = {
    frozenset(((0, 1, 0), (0, 0, 0))),
    frozenset(((0, 1, 0), (1, 1, 0))),
    frozenset(((0, 1, 0), (0, 1, 1))),
}


def box_figure(dims, *, alt: str, labelled=("OP", "OR", "OS"), unit: str = "m", base=None):
    """A cuboid with edges ``dims = (a, b, c)`` along x, y and z, in oblique view, as a ``diagram``.

    Edges named in ``labelled`` (any of ``OP``, ``OR``, ``OS``) carry their length. All eight vertices are named
    (``O`` at the origin corner, then ``P`` to ``V``); ``base`` is ``None`` here and only changes the stem.
    Not drawn to scale. Returns ``(spec, positions)`` where ``positions`` maps a vertex name to its drawn point.
    """
    a, b, c = dims

    def drawn(length):
        # a longer edge is always drawn longer, but the shortest and longest are not wildly different
        return 1.2 * (4 + length)

    la, lb, lc = drawn(a), drawn(b), drawn(c)
    depth = (math.cos(math.radians(35)), math.sin(math.radians(35)))
    margin = 9.0
    width = la + lb * depth[0]
    height = lc + lb * depth[1]

    def at(v):
        return (margin + v[0] * la + v[1] * lb * depth[0], margin + v[2] * lc + v[1] * lb * depth[1])

    points = [{"id": name, "at": list(at(v)), "label": name} for v, name in BOX_NAMES.items()]
    labels = {"OP": f"{num_text(a)} {unit}", "OR": f"{num_text(b)} {unit}", "OS": f"{num_text(c)} {unit}"}
    edge_name = {
        frozenset(((0, 0, 0), (1, 0, 0))): "OP",
        frozenset(((0, 0, 0), (0, 1, 0))): "OR",
        frozenset(((0, 0, 0), (0, 0, 1))): "OS",
    }
    segments = []
    for u, v in _EDGES:
        seg = {"from": BOX_NAMES[u], "to": BOX_NAMES[v]}
        key = frozenset((u, v))
        if key in _HIDDEN:
            seg["dashed"] = True
        name = edge_name.get(key)
        if name in labelled:
            seg["label"] = labels[name]
        segments.append(seg)
    spec = space.diagram(
        alt=alt,
        points=points,
        segments=segments,
        frame=(width + 2 * margin, height + 2 * margin),
        not_to_scale=True,
    )
    return spec, {name: tuple(at(v)) for v, name in BOX_NAMES.items()}


def vertex(name: str) -> tuple[int, int, int]:
    """The (0/1, 0/1, 0/1) corner of the box a vertex name stands for."""
    for v, n in BOX_NAMES.items():
        if n == name:
            return v
    raise KeyError(name)


# ---------------------------------------------------------------------------------------- right-triangle tail

#: (leg, leg, hypotenuse) triples for the exam chains' last two parts; the chains put the legs either way round.
LEG_TRIPLES = [(3, 4, 5), (6, 8, 10), (5, 12, 13)]


def side_and_perimeter_parts(leg1: int, leg2: int, hyp: int, name: str, *, extra_cands=()):
    """The last two parts of a chain on a right-angled triangle named ``name`` (``A'B'C'``): the longest side,
    then the perimeter, which follows through from the longest side. Returns two ``parts`` entries."""
    from mathgen.generators.exd_x_sc3_support import number_part, tagged

    cands = [
        (Fraction(leg1 + leg2), "adds-instead-of-using-pythagoras"),
        (Fraction(leg1 * leg1 + leg2 * leg2), "forgets-the-square-root"),
        (Fraction(max(leg1, leg2)), "fr2a-leg-for-longest-side"),
        *extra_cands,
    ]
    part_c = number_part(
        f"The longest side of triangle {name} is the side opposite the right angle. How long is it, in metres?",
        Fraction(hyp),
        0,
        2,
        [
            "A reflection, a translation and a turn do not change lengths, so the longest side is the same as in "
            f"triangle ABC, where the shorter sides are {leg1} m and {leg2} m.",
            f"√({leg1}² + {leg2}²) = √{leg1 * leg1 + leg2 * leg2} = {hyp} m.",
        ],
        tagged(Fraction(hyp), cands, 0, lo=0),
        sensible_range=(0, None),
    )
    perimeter = leg1 + leg2 + hyp
    per_cands = [
        (Fraction(hyp + max(leg1, leg2)), "fr2a-side-left-out-of-perimeter"),
        (Fraction(leg1 + leg2), "fr2a-hypotenuse-left-out-of-perimeter"),
        (Fraction(leg1 * leg2, 2), "area-for-perimeter"),
        (Fraction(2 * hyp + leg1 + leg2), "fr2a-side-counted-twice"),
    ]
    part_d = number_part(
        f"Find the perimeter of triangle {name}, in metres.",
        Fraction(perimeter),
        0,
        2,
        [f"Perimeter = {leg1} + {leg2} + {hyp} = {perimeter} m."],
        tagged(Fraction(perimeter), far_from(perimeter, per_cands, 1), 0, lo=0),
        follow_through=f"part_c+{leg1}+{leg2}",
        sensible_range=(0, None),
    )
    return part_c, part_d


def far_from(key, cands, gap):
    """The ``(value, tag)`` candidates more than ``gap`` from ``key``. A part that follows through from an earlier
    part accepts a reply within one unit of its last place of the follow-through value, so a wrong answer for it
    must lie further out than that."""
    return [(v, t) for v, t in cands if abs(Fraction(v) - Fraction(key)) > gap]


def require_distractors(parts, minimum: int = 3) -> None:
    """Raise ``VerificationError`` (so the generator draws again) unless every part has enough wrong answers."""
    for part in parts:
        check(len(part["distractors"]) >= minimum, f"part {part.get('prompt', '')[:30]!r} has too few distractors")
