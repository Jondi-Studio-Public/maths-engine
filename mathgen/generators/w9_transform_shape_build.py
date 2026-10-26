"""Generates `w9_transform_shape` questions (easy, medium, hard) and the exam chain `exd_x_w9_transform_shape`.

``transform-a-shape-on-grid`` (``geometry``, answer ``polygon``): the learner places the vertices of the image of a shape
under a translation, reflection or rotation. The shape is drawn on a coordinate grid with its vertices labelled; the
front end sends the image's vertices in label order (``{"vertices": [[3, 1], [6, 1], [3, 4]]}`` for A', B', C'). The
``polygon`` grader (labelled) needs each vertex exactly where it belongs, so swapping two labels is wrong even though the
outline is the same. The typed form is the vertices written out in order, ``(3, 1) (6, 1) (3, 4)``.

* easy: a translation by a stated vector, or a reflection in the x-axis or the y-axis.
* medium: a reflection in a vertical or horizontal line, or in y = x or y = -x; or a rotation of 90 degrees (either way)
  or 180 degrees about the origin.
* hard: a rotation of 90 degrees about a marked centre that is not the origin, or two transformations in a row
  (reflect then translate, rotate then reflect) with the image of the second asked for.
* exam (``exd_x_w9_transform_shape``): reflect, translate, rotate about a centre, and find the area of an image.

Each image is computed with coordinate rules and checked a second way: rotations with complex numbers
(``z' = c + i^n (z - c)``), reflections with vector projection onto the mirror line, and distances and areas with
the isometry checks (every side length and the shoelace area are unchanged). Every wrong image is checked against the
`polygon` grader before it is offered.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, paren_negative, phrase, rational_text
from ._w6a_kit import num_dist, num_fields
from ._w9_graph_kit import HALF, dist, enough, part, plane, point_text, points_text, retry_draw
from ..grading import interactive as ia

TOPIC = "w9_transform_shape"
EXAM_TOPIC = "exd_x_w9_transform_shape"

AREA = "transform-a-shape-on-grid"
MISCONCEPTIONS = [
    (
        "w9-translate-direction",
        "Moves the shape the wrong way along one direction: right for left, or up for down.",
        AREA,
    ),
    ("w9-translate-components-swapped", "Swaps the across and up parts of the translation.", AREA),
    ("w9-translate-one-component", "Applies only the across part or only the up part of the translation.", AREA),
    ("w9-translate-negated", "Moves the shape by the opposite vector, undoing the translation.", AREA),
    ("w9-reflect-wrong-axis", "Reflects in the other axis, or the other mirror line, instead of the one named.", AREA),
    (
        "w9-reflect-distance-not-doubled",
        "Moves each point only as far as the mirror line instead of through it, so the image lands on the line.",
        AREA,
    ),
    (
        "w9-reflect-line-value-not-doubled",
        "Writes k - x instead of 2k - x for the image of a point in the line x = k (or k - y for y = k).",
        AREA,
    ),
    (
        "w9-reflect-adds-instead",
        "Moves each point the distance to the mirror line in the same direction, which translates the shape instead of reflecting it.",
        AREA,
    ),
    (
        "w9-reflect-diagonal-as-swap-sign",
        "Reflects in y = -x by swapping the coordinates but forgetting to change both signs (or the reverse for y = x).",
        AREA,
    ),
    ("w9-rotate-direction", "Turns the shape the wrong way round: clockwise for anticlockwise or the reverse.", AREA),
    ("w9-rotate-angle", "Turns the shape through a half turn when a quarter turn was asked, or the reverse.", AREA),
    ("w9-rotate-about-origin", "Turns the shape about the origin instead of about the marked centre.", AREA),
    (
        "w9-rotate-as-reflection",
        "Swaps the coordinates, which reflects the shape in y = x instead of turning it.",
        AREA,
    ),
    ("w9-rotate-sign-slip", "Swaps the coordinates in a quarter turn but gives both the wrong sign.", AREA),
    ("w9-image-vertices-misordered", "Puts the image vertices in the wrong order, so A' is not where A went.", AREA),
    ("w9-composite-order-reversed", "Does the two transformations in the wrong order.", AREA),
    ("w9-composite-first-only", "Applies only the first of the two transformations.", AREA),
    ("w9-composite-second-only", "Applies only the second of the two transformations to the original shape.", AREA),
    (
        "w9-area-changes-with-image",
        "Thinks the area changes under a reflection, translation or rotation, and recalculates it wrongly or doubles it.",
        AREA,
    ),
    ("w9-area-no-half", "Gives the area of the box round the triangle, forgetting to halve it.", AREA),
    (
        "w9-area-base-times-side",
        "Multiplies two sides of the triangle instead of using base times height over two.",
        AREA,
    ),
]

Pt = tuple[int, int]


# ---------------------------------------------------------------------------------------------- geometry (two ways)


def translate(p: Pt, v: Pt) -> Pt:
    return (p[0] + v[0], p[1] + v[1])


def reflect(p: Pt, mirror: str) -> Pt:
    x, y = p
    if mirror == "x-axis":
        return (x, -y)
    if mirror == "y-axis":
        return (-x, y)
    if mirror == "y=x":
        return (y, x)
    if mirror == "y=-x":
        return (-y, -x)
    if mirror.startswith("x="):
        return (2 * int(mirror[2:]) - x, y)
    if mirror.startswith("y="):
        return (x, 2 * int(mirror[2:]) - y)
    raise ValueError(mirror)


_MIRROR_POINTS = {
    "x-axis": ((0, 0), (1, 0)),
    "y-axis": ((0, 0), (0, 1)),
    "y=x": ((0, 0), (1, 1)),
    "y=-x": ((0, 0), (1, -1)),
}


def _mirror_through(mirror: str):
    if mirror in _MIRROR_POINTS:
        return _MIRROR_POINTS[mirror]
    k = int(mirror[2:])
    return ((k, 0), (k, 1)) if mirror.startswith("x=") else ((0, k), (1, k))


def reflect_by_projection(p: Pt, mirror: str) -> Pt:
    """The second way: p' = 2 * (foot of the perpendicular) - p, with exact fractions."""
    (ax, ay), (bx, by) = _mirror_through(mirror)
    dx, dy = bx - ax, by - ay
    t = Fraction((p[0] - ax) * dx + (p[1] - ay) * dy, dx * dx + dy * dy)
    foot = (ax + t * dx, ay + t * dy)
    out = (2 * foot[0] - p[0], 2 * foot[1] - p[1])
    check(out[0].denominator == 1 and out[1].denominator == 1, "a reflection left the lattice")
    return (int(out[0]), int(out[1]))


def rotate(p: Pt, centre: Pt, quarter_turns: int) -> Pt:
    """Anticlockwise by `quarter_turns` quarter turns (negative: clockwise), by coordinate rules."""
    x, y = p[0] - centre[0], p[1] - centre[1]
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return (centre[0] + x, centre[1] + y)


def rotate_by_complex(p: Pt, centre: Pt, quarter_turns: int) -> Pt:
    z = complex(p[0] - centre[0], p[1] - centre[1]) * (1j ** (quarter_turns % 4))
    out = (centre[0] + round(z.real), centre[1] + round(z.imag))
    check(abs(z.real - round(z.real)) < 1e-9 and abs(z.imag - round(z.imag)) < 1e-9, "a rotation left the lattice")
    return out


def area2(poly) -> int:
    return abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1])))


def _side_lengths_sq(poly):
    return [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for p, q in zip(poly, poly[1:] + poly[:1])]


# ---------------------------------------------------------------------------------------------- shapes and text

_NAMES = "ABCD"


def _in(p) -> bool:
    return all(abs(v) <= HALF for v in p)


def _draw_shape(rng, sides: int, spread: int = 4):
    for _ in attempts(500, "a clearly scalene shape on the grid"):
        pts = [(rng.randint(-spread, spread), rng.randint(-spread, spread)) for _ in range(sides)]
        if len(set(pts)) < sides or area2(pts) < 6:
            continue
        lens = _side_lengths_sq(pts)
        if len(set(lens)) < sides:  # no two sides equal: every vertex is told apart
            continue
        if sides == 4:
            # a simple quadrilateral: both diagonals cross, so the outline does not intersect itself
            if not _convex(pts):
                continue
        # no vertex on an axis or mirror line, so the picture is not degenerate
        if any(0 in p for p in pts):
            continue
        return pts
    raise AssertionError  # pragma: no cover


def _convex(pts) -> bool:
    signs = set()
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross == 0:
            return False
        signs.add(cross > 0)
    return len(signs) == 1


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _mirror_label(mirror: str) -> str:
    return {"x-axis": "the x-axis", "y-axis": "the y-axis", "y=x": "the line y = x", "y=-x": "the line y = -x"}.get(
        mirror, f"the line {mirror.replace('=', ' = ')}"
    )


def _figure(shape, extra_lines=(), extra_points=(), images=1, alt=None):
    n = len(shape)
    pts = [{"at": list(p), "label": _NAMES[i]} for i, p in enumerate(shape)] + [
        {"at": list(p), "label": name} for name, p in extra_points
    ]
    segs = [{"from": list(shape[i]), "to": list(shape[(i + 1) % n])} for i in range(n)]
    lines = [{"through": [list(a), list(b)], "label": label} for label, a, b in extra_lines]
    ticks = "'" * images
    return plane(
        reply="polygon",
        tools=["polygon_points"],
        alt=alt
        or f"A coordinate grid with the {'triangle' if n == 3 else 'quadrilateral'} {_NAMES[:n]} drawn on it and its vertices labelled.",
        snap=1,
        points=pts,
        segments=segs,
        lines=lines,
        vertex_labels=[f"{_NAMES[i]}{ticks}" for i in range(n)],
        max_points=n,
    )


def _answer_fields(image) -> dict:
    text = points_text(image)
    return dict(
        answer=ia.polygon_answer(image, x_range=(-HALF, HALF), y_range=(-HALF, HALF), snap=1, labelled=True),
        answer_text=text,
        answer_latex="plain:" + text,
        answer_kind="polygon",
    )


def _dist(fields, candidates, n=4):
    """`candidates` is ``[(image vertices, tag), ...]``; an image off the grid cannot be placed and is skipped."""
    cands = []
    for img, tag in candidates:
        if img is None or not all(_in(p) for p in img) or len(set(img)) < 3:
            continue
        cands.append(
            (points_text(img), ia.polygon_answer(img, x_range=(-HALF, HALF), y_range=(-HALF, HALF), snap=1), tag)
        )
    return dist("polygon", fields["answer"], fields["answer_text"], cands, n)


def _misordered(image):
    return [image[1], image[0], *image[2:]]


def _checks(shape, image, how: str) -> None:
    """Isometry checks: side lengths and area are unchanged, and the vertex count is."""
    check(len(shape) == len(image), "the image has a different number of vertices")
    check(_side_lengths_sq(shape) == _side_lengths_sq(image), f"a side changed length under the {how}")
    check(area2(shape) == area2(image), f"the area changed under the {how}")


# ---------------------------------------------------------------------------------------------- easy


def _easy_translate(rng):
    for _ in attempts(300, "a translation that keeps the image on the grid"):
        shape = _draw_shape(rng, rng.choice((3, 3, 4)))
        v = (rng.choice([d for d in range(-5, 6) if d]), rng.choice([d for d in range(-5, 6) if d]))
        image = [translate(p, v) for p in shape]
        if all(_in(p) for p in image) and abs(v[0]) != abs(v[1]):
            break
    check(all(q[0] - p[0] == v[0] and q[1] - p[1] == v[1] for p, q in zip(shape, image)), "not a translation")
    _checks(shape, image, "translation")
    name = _NAMES[: len(shape)]
    right = f"{abs(v[0])} unit{'s' if abs(v[0]) != 1 else ''} to the {'right' if v[0] > 0 else 'left'}"
    up = f"{abs(v[1])} unit{'s' if abs(v[1]) != 1 else ''} {'up' if v[1] > 0 else 'down'}"
    text = phrase(
        rng,
        (
            "Translate {shape} {right} and {up}. Place the image {img}.",
            "{Shape} is moved {right} and {up}. Place the vertices of the image {img}.",
        ),
        shape=f"the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name}",
        Shape=_cap(f"the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name}"),
        right=right,
        up=up,
        img=f"{'′'.join(name)}′".replace("′", "'"),
    )
    fields = _answer_fields(image)
    cands = [
        ([translate(p, (-v[0], v[1])) for p in shape], "w9-translate-direction"),
        ([translate(p, (v[0], -v[1])) for p in shape], "w9-translate-direction"),
        ([translate(p, (v[1], v[0])) for p in shape], "w9-translate-components-swapped"),
        ([translate(p, (v[0], 0)) for p in shape], "w9-translate-one-component"),
        ([translate(p, (0, v[1])) for p in shape], "w9-translate-one-component"),
        ([translate(p, (-v[0], -v[1])) for p in shape], "w9-translate-negated"),
        (_misordered(image), "w9-image-vertices-misordered"),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    working = [
        f"Every vertex moves the same way: {abs(v[0])} {'right' if v[0] > 0 else 'left'} (x {'+' if v[0] > 0 else '-'} {abs(v[0])}) and {abs(v[1])} {'up' if v[1] > 0 else 'down'} (y {'+' if v[1] > 0 else '-'} {abs(v[1])}).",
        "Image: " + ", ".join(f"{_NAMES[i]}' = {point_text(p)}" for i, p in enumerate(image)) + ".",
    ]
    return dict(text=text, figure=_figure(shape), working=working, distractors=d, **fields)


def _easy_reflect(rng):
    mirror = rng.choice(("x-axis", "y-axis"))
    for _ in attempts(300, "a reflection that keeps the shape on the grid"):
        shape = _draw_shape(rng, rng.choice((3, 3, 4)))
        image = [reflect(p, mirror) for p in shape]
        if all(_in(p) for p in image):
            break
    check(image == [reflect_by_projection(p, mirror) for p in shape], "the two reflection rules disagree")
    _checks(shape, image, "reflection")
    name = _NAMES[: len(shape)]
    text = phrase(
        rng,
        (
            "Reflect {shape} in {m}. Place the image {img}.",
            "{Shape} is reflected in {m}. Place the vertices of the image {img}.",
        ),
        shape=f"the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name}",
        Shape=_cap(f"the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name}"),
        m=_mirror_label(mirror),
        img=f"{'′'.join(name)}′".replace("′", "'"),
    )
    other = "y-axis" if mirror == "x-axis" else "x-axis"
    fields = _answer_fields(image)
    cands = [
        ([reflect(p, other) for p in shape], "w9-reflect-wrong-axis"),
        ([(-p[0], -p[1]) for p in shape], "w9-reflect-wrong-axis"),
        ([(p[1], p[0]) for p in shape], "w9-rotate-as-reflection"),
        ([(p[0], -p[1]) if mirror == "y-axis" else (-p[0], p[1]) for p in shape], "w9-reflect-wrong-axis"),
        (_misordered(image), "w9-image-vertices-misordered"),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    flip = "y-coordinate" if mirror == "x-axis" else "x-coordinate"
    working = [
        f"Reflecting in {_mirror_label(mirror)} changes the sign of the {flip} and keeps the other coordinate.",
        "Image: " + ", ".join(f"{_NAMES[i]}' = {point_text(p)}" for i, p in enumerate(image)) + ".",
    ]
    fig = _figure(shape)
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


def _easy(rng):
    return _easy_translate(rng) if rng.random() < 0.5 else _easy_reflect(rng)


# ---------------------------------------------------------------------------------------------- medium


def _reflect_wrong(shape, mirror):
    """Wrong images for a reflection of `shape` in `mirror`, as ``(image, tag)``."""
    if mirror == "y=x":
        return [
            ([(-p[1], -p[0]) for p in shape], "w9-reflect-diagonal-as-swap-sign"),
            ([(p[0], -p[1]) for p in shape], "w9-reflect-wrong-axis"),
            ([(-p[0], p[1]) for p in shape], "w9-reflect-wrong-axis"),
            ([(-p[1], p[0]) for p in shape], "w9-rotate-direction"),
        ]
    if mirror == "y=-x":
        return [
            ([(p[1], p[0]) for p in shape], "w9-reflect-diagonal-as-swap-sign"),
            ([(p[0], -p[1]) for p in shape], "w9-reflect-wrong-axis"),
            ([(-p[0], p[1]) for p in shape], "w9-reflect-wrong-axis"),
            ([(p[1], -p[0]) for p in shape], "w9-rotate-direction"),
        ]
    if mirror.startswith("x="):
        k = int(mirror[2:])
        return [
            ([(k, p[1]) for p in shape], "w9-reflect-distance-not-doubled"),
            ([(k - p[0], p[1]) for p in shape], "w9-reflect-line-value-not-doubled"),
            ([(-p[0], p[1]) for p in shape], "w9-reflect-wrong-axis"),
            ([(p[0], 2 * k - p[1]) for p in shape], "w9-reflect-wrong-axis"),
            ([(p[0] + 2 * k, p[1]) for p in shape], "w9-reflect-adds-instead"),
        ]
    if mirror.startswith("y="):
        k = int(mirror[2:])
        return [
            ([(p[0], k) for p in shape], "w9-reflect-distance-not-doubled"),
            ([(p[0], k - p[1]) for p in shape], "w9-reflect-line-value-not-doubled"),
            ([(p[0], -p[1]) for p in shape], "w9-reflect-wrong-axis"),
            ([(2 * k - p[0], p[1]) for p in shape], "w9-reflect-wrong-axis"),
            ([(p[0], p[1] + 2 * k) for p in shape], "w9-reflect-adds-instead"),
        ]
    raise ValueError(mirror)


def _medium_reflect(rng):
    mirror = rng.choice(("x=1", "x=2", "x=-1", "x=-2", "x=3", "y=1", "y=2", "y=-1", "y=-2", "y=3", "y=x", "y=-x"))
    for _ in attempts(500, "a reflection that keeps the shape on the grid"):
        shape = _draw_shape(rng, rng.choice((3, 3, 4)))
        image = [reflect(p, mirror) for p in shape]
        # the mirror line must not pass through a vertex (the picture would be degenerate)
        if all(_in(p) for p in image) and all(p != q for p, q in zip(shape, image)):
            break
    check(image == [reflect_by_projection(p, mirror) for p in shape], "the two reflection rules disagree")
    _checks(shape, image, "reflection")
    name = _NAMES[: len(shape)]
    text = phrase(
        rng,
        (
            "Reflect {shape} in {m}. Place the image {img}.",
            "{Shape} is reflected in {m}. Place the vertices of the image {img}.",
        ),
        shape=f"the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name}",
        Shape=_cap(f"the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name}"),
        m=_mirror_label(mirror),
        img=f"{'′'.join(name)}′".replace("′", "'"),
    )
    a, b = _mirror_through(mirror)
    if mirror == "y=x" or mirror == "y=-x":
        a, b = (-HALF, -HALF if mirror == "y=x" else HALF), (HALF, HALF if mirror == "y=x" else -HALF)
    elif mirror.startswith("x="):
        k = int(mirror[2:])
        a, b = (k, -HALF), (k, HALF)
    else:
        k = int(mirror[2:])
        a, b = (-HALF, k), (HALF, k)
    fields = _answer_fields(image)
    cands = _reflect_wrong(shape, mirror)
    cands.append((_misordered(image), "w9-image-vertices-misordered"))
    d = _dist(fields, cands)
    enough(d, 3)
    working = [
        "Each vertex and its image are the same distance from the mirror line, on opposite sides, and the line between them is at right angles to the mirror.",
        "Image: " + ", ".join(f"{_NAMES[i]}' = {point_text(p)}" for i, p in enumerate(image)) + ".",
    ]
    fig = _figure(shape, extra_lines=[(mirror.replace("=", " = "), a, b)])
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


_TURNS = {
    "a quarter turn anticlockwise": 1,
    "a quarter turn clockwise": -1,
    "a half turn": 2,
}


def _medium_rotate(rng):
    words = rng.choice(list(_TURNS))
    q = _TURNS[words]
    for _ in attempts(500, "a rotation that keeps the shape on the grid"):
        shape = _draw_shape(rng, rng.choice((3, 3, 4)))
        image = [rotate(p, (0, 0), q) for p in shape]
        if all(_in(p) for p in image):
            break
    check(image == [rotate_by_complex(p, (0, 0), q) for p in shape], "the two rotation rules disagree")
    _checks(shape, image, "rotation")
    name = _NAMES[: len(shape)]
    text = phrase(
        rng,
        (
            "Rotate {shape} {w} about the origin. Place the image {img}.",
            "{Shape} is turned through {w} about the origin. Place the vertices of the image {img}.",
        ),
        shape=f"the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name}",
        Shape=_cap(f"the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name}"),
        w=words,
        img=f"{'′'.join(name)}′".replace("′", "'"),
    )
    fields = _answer_fields(image)
    cands = [
        ([rotate(p, (0, 0), -q) for p in shape] if abs(q) == 1 else None, "w9-rotate-direction"),
        ([rotate(p, (0, 0), 2 if abs(q) == 1 else 1) for p in shape], "w9-rotate-angle"),
        ([(p[1], p[0]) for p in shape], "w9-rotate-as-reflection"),
        (
            [(-p[1], -p[0]) for p in shape] if abs(q) == 1 else [(-p[0], p[1]) for p in shape],
            "w9-rotate-sign-slip" if abs(q) == 1 else "w9-reflect-wrong-axis",
        ),
        (_misordered(image), "w9-image-vertices-misordered"),
    ]
    d = _dist(fields, [(c, t) for c, t in cands if c])
    enough(d, 3)
    rule = {1: "(x, y) -> (-y, x)", -1: "(x, y) -> (y, -x)", 2: "(x, y) -> (-x, -y)"}[q]
    working = [
        f"About the origin, {words} sends {rule}.",
        "Image: " + ", ".join(f"{_NAMES[i]}' = {point_text(p)}" for i, p in enumerate(image)) + ".",
    ]
    return dict(text=text, figure=_figure(shape), working=working, distractors=d, **fields)


def _medium(rng):
    return _medium_reflect(rng) if rng.random() < 0.5 else _medium_rotate(rng)


# ---------------------------------------------------------------------------------------------- hard


def _hard_rotate_centre(rng):
    words = rng.choice(list(_TURNS)[:2])
    q = _TURNS[words]
    for _ in attempts(1000, "a rotation about a centre that keeps the shape on the grid"):
        centre = (rng.randint(-3, 3), rng.randint(-3, 3))
        if centre == (0, 0):
            continue
        shape = _draw_shape(rng, rng.choice((3, 3, 4)), spread=4)
        image = [rotate(p, centre, q) for p in shape]
        if all(_in(p) for p in image) and centre not in shape:
            break
    check(image == [rotate_by_complex(p, centre, q) for p in shape], "the two rotation rules disagree")
    _checks(shape, image, "rotation")
    check(
        all(
            (p[0] - centre[0]) ** 2 + (p[1] - centre[1]) ** 2 == (r[0] - centre[0]) ** 2 + (r[1] - centre[1]) ** 2
            for p, r in zip(shape, image)
        ),
        "a vertex is not the same distance from the centre as its image",
    )
    name = _NAMES[: len(shape)]
    text = (
        f"Rotate the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name} {words} about the point R{point_text(centre)}. "
        f"Place the image {''.join(c + chr(39) for c in name)}."
    )
    fields = _answer_fields(image)
    cands = [
        ([rotate(p, centre, -q) for p in shape], "w9-rotate-direction"),
        ([rotate(p, (0, 0), q) for p in shape], "w9-rotate-about-origin"),
        ([rotate(p, centre, 2) for p in shape], "w9-rotate-angle"),
        ([(p[1], p[0]) for p in shape], "w9-rotate-as-reflection"),
        (_misordered(image), "w9-image-vertices-misordered"),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    rule = "(x, y) -> (-y, x)" if q == 1 else "(x, y) -> (y, -x)"
    working = [
        f"Subtract the centre {point_text(centre)} from each vertex, apply {rule}, then add the centre back.",
        "Image: " + ", ".join(f"{_NAMES[i]}' = {point_text(p)}" for i, p in enumerate(image)) + ".",
    ]
    fig = _figure(shape, extra_points=[("R", centre)])
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


def _compose(first, second):
    return lambda p: second(first(p))


def _hard_composite(rng):
    kind = rng.choice(("reflect-translate", "rotate-reflect"))
    for _ in attempts(1000, "a two-step transformation that keeps every image on the grid"):
        shape = _draw_shape(rng, rng.choice((3, 3, 4)), spread=3)
        if kind == "reflect-translate":
            mirror = rng.choice(("x-axis", "y-axis", "y=x", "y=-x"))
            v = (rng.choice([d for d in range(-4, 5) if d]), rng.choice([d for d in range(-4, 5) if d]))
            f1, f2 = (lambda p: reflect(p, mirror)), (lambda p: translate(p, v))
            t1 = f"Reflect {{s}} in {_mirror_label(mirror)}"
            t2 = f"translate the result {abs(v[0])} unit{'s' if abs(v[0]) != 1 else ''} {'right' if v[0] > 0 else 'left'} and {abs(v[1])} unit{'s' if abs(v[1]) != 1 else ''} {'up' if v[1] > 0 else 'down'}"
            tags = ("w9-composite-first-only", "w9-composite-second-only")
        else:
            q = rng.choice((1, -1, 2))
            mirror = rng.choice(("x-axis", "y-axis"))
            f1, f2 = (lambda p: rotate(p, (0, 0), q)), (lambda p: reflect(p, mirror))
            words = {1: "a quarter turn anticlockwise", -1: "a quarter turn clockwise", 2: "a half turn"}[q]
            t1 = f"Rotate {{s}} {words} about the origin"
            t2 = f"reflect the result in {_mirror_label(mirror)}"
            tags = ("w9-composite-first-only", "w9-composite-second-only")
        mid = [f1(p) for p in shape]
        image = [f2(p) for p in mid]
        swapped = [f1(f2(p)) for p in shape]
        if all(_in(p) for p in (*mid, *image, *swapped)) and swapped != image:
            break
    _checks(shape, image, "composite transformation")
    name = _NAMES[: len(shape)]
    s = f"the {'triangle' if len(shape) == 3 else 'quadrilateral'} {name}"
    text = f"{t1.format(s=s)}, then {t2}. Place the final image {''.join(c + chr(39) * 2 for c in name)}."
    fields = _answer_fields(image)
    cands = [
        (swapped, "w9-composite-order-reversed"),
        (mid, tags[0]),
        ([f2(p) for p in shape], tags[1]),
        (_misordered(image), "w9-image-vertices-misordered"),
    ]
    d = _dist(fields, cands)
    enough(d, 3)
    working = [
        "Do the first transformation: " + ", ".join(f"{_NAMES[i]}' = {point_text(p)}" for i, p in enumerate(mid)) + ".",
        "Then do the second transformation to those points: "
        + ", ".join(f"{_NAMES[i]}'' = {point_text(p)}" for i, p in enumerate(image))
        + ".",
    ]
    fig = _figure(shape, images=2)
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


_HARD = (_hard_rotate_centre, _hard_composite)


@generator(
    TOPIC,
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w9_transform_shape(rng, tier="easy"):
    """Transform a shape on a grid; the tiers are described in the module docstring."""
    pool = {"easy": (_easy,), "medium": (_medium,), "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# ---------------------------------------------------------------------------------------------- exam


@generator(EXAM_TOPIC, level=10, presets={"exam": dict(tier="exam")})
def exd_x_w9_transform_shape(rng, tier="exam"):
    """Exam chain: reflect; translate; rotate about a centre; the area of an image."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam)


def _exam(rng):
    q = rng.choice((1, -1))
    mirror = rng.choice(("x=1", "x=-1", "x=2", "y=1", "y=-1", "y=2", "y=x", "y=-x"))
    for _ in attempts(2000, "three images that all fit the grid"):
        shape = _draw_shape(rng, 3, spread=3)
        v = (rng.choice([d for d in range(-4, 5) if d]), rng.choice([d for d in range(-4, 5) if d]))
        centre = (rng.randint(-2, 2), rng.randint(-2, 2))
        img_r = [reflect(p, mirror) for p in shape]
        img_t = [translate(p, v) for p in shape]
        img_o = [rotate(p, centre, q) for p in shape]
        if (
            all(_in(p) for p in (*img_r, *img_t, *img_o))
            and centre not in shape
            and centre != (0, 0)
            and all(p != q for p, q in zip(shape, img_r))
        ):
            break
    check(img_r == [reflect_by_projection(p, mirror) for p in shape], "reflection rules disagree")
    check(img_o == [rotate_by_complex(p, centre, q) for p in shape], "rotation rules disagree")
    for image, how in ((img_r, "reflection"), (img_t, "translation"), (img_o, "rotation")):
        _checks(shape, image, how)
    twice = Fraction(area2(shape), 2)
    check(twice == Fraction(area2(img_t), 2), "area changed")
    fr, ft, fo = _answer_fields(img_r), _answer_fields(img_t), _answer_fields(img_o)
    words = "a quarter turn anticlockwise" if q == 1 else "a quarter turn clockwise"
    d_r = _dist(fr, [*_reflect_wrong(shape, mirror), (_misordered(img_r), "w9-image-vertices-misordered")])
    d_t = _dist(
        ft,
        [
            ([translate(p, (-v[0], v[1])) for p in shape], "w9-translate-direction"),
            ([translate(p, (v[1], v[0])) for p in shape], "w9-translate-components-swapped"),
            ([translate(p, (-v[0], -v[1])) for p in shape], "w9-translate-negated"),
            ([translate(p, (v[0], 0)) for p in shape], "w9-translate-one-component"),
        ],
    )
    d_o = _dist(
        fo,
        [
            ([rotate(p, centre, -q) for p in shape], "w9-rotate-direction"),
            ([rotate(p, (0, 0), q) for p in shape], "w9-rotate-about-origin"),
            ([rotate(p, centre, 2) for p in shape], "w9-rotate-angle"),
            ([(p[1], p[0]) for p in shape], "w9-rotate-as-reflection"),
        ],
    )
    xs, ys = [p[0] for p in img_t], [p[1] for p in img_t]
    box = (max(xs) - min(xs)) * (max(ys) - min(ys))
    base_side = _side_lengths_sq(shape)
    d_area = num_dist(
        twice,
        [
            (box, "w9-area-no-half"),
            (2 * twice, "w9-area-changes-with-image"),
            (Fraction(area2(shape)), "w9-area-no-half"),
            (twice + 1, "w9-area-changes-with-image"),
            (Fraction(abs(shape[1][0] - shape[0][0]) * abs(shape[2][1] - shape[0][1])), "w9-area-base-times-side"),
        ],
    )
    for dl in (d_r, d_t, d_o, d_area):
        enough(dl)
    del base_side
    stem = (
        f"The triangle ABC has vertices A{point_text(shape[0])}, B{point_text(shape[1])} and C{point_text(shape[2])}."
    )
    a, b = _mirror_through(mirror)
    if mirror in ("y=x", "y=-x"):
        a, b = (-HALF, -HALF if mirror == "y=x" else HALF), (HALF, HALF if mirror == "y=x" else -HALF)
    elif mirror.startswith("x="):
        a, b = (int(mirror[2:]), -HALF), (int(mirror[2:]), HALF)
    else:
        a, b = (-HALF, int(mirror[2:])), (HALF, int(mirror[2:]))
    fig = _figure(shape, extra_lines=[(mirror.replace("=", " = "), a, b)], extra_points=[("R", centre)])
    return dict(
        text=stem,
        figure=fig,
        parts=[
            part(
                f"Reflect ABC in {_mirror_label(mirror)}. Place the image A'B'C'.",
                "polygon",
                fr,
                2,
                ["Image: " + ", ".join(f"{_NAMES[i]}' = {point_text(p)}" for i, p in enumerate(img_r)) + "."],
                d_r,
            ),
            part(
                f"Translate ABC {abs(v[0])} unit{'s' if abs(v[0]) != 1 else ''} {'right' if v[0] > 0 else 'left'} and {abs(v[1])} unit{'s' if abs(v[1]) != 1 else ''} {'up' if v[1] > 0 else 'down'}. Place the image A'B'C'.",
                "polygon",
                ft,
                2,
                ["Image: " + ", ".join(f"{_NAMES[i]}' = {point_text(p)}" for i, p in enumerate(img_t)) + "."],
                d_t,
            ),
            part(
                f"Rotate ABC {words} about the point R{point_text(centre)}. Place the image A'B'C'.",
                "polygon",
                fo,
                3,
                ["Image: " + ", ".join(f"{_NAMES[i]}' = {point_text(p)}" for i, p in enumerate(img_o)) + "."],
                d_o,
            ),
            part(
                "What is the area of the image in the translation, in square units?",
                "number",
                num_fields(twice),
                2,
                [
                    "A translation keeps the size and shape, so the area is the area of ABC.",
                    f"Area of ABC = half of |{paren_negative(shape[1][0] - shape[0][0])} × {paren_negative(shape[2][1] - shape[0][1])}"
                    f" - {paren_negative(shape[2][0] - shape[0][0])} × {paren_negative(shape[1][1] - shape[0][1])}|"
                    f" = half of {area2(shape)} = {rational_text(twice)}.",
                ],
                d_area,
                sensible_range=(0, None),
            ),
        ],
    )
