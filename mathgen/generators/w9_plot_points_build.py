"""Generates `w9_plot_points` questions (easy, medium, hard) and the exam chain `exd_x_w9_plot_points`.

``plot-a-point-from-coordinates`` (``linear-relations``, answer ``points``): the learner places points on a coordinate
grid. The reply is the list of points the front end sends (``[[3, 5]]``), compared exactly and in any order; the typed
form is the points written out, ``(3, 5)``. The figure is a blank grid with a ``canvas`` (`mathgen.figure_kinds.canvas`)
that enables the point tool, snapped to whole units (to half units where a coordinate is a half).

* easy: one point in the first quadrant, from its coordinates, from distances east and north of the origin, or from
  the point it is named at.
* medium: points in any quadrant, from coordinates or from distances from each axis.
* hard: a point that has to be found first: the fourth corner of a parallelogram, the midpoint of two plotted points, the
  point on a line at a given x, or a point whose coordinates are half units.
* exam (``exd_x_w9_plot_points``): plot three vertices, plot the midpoint of two, plot the fourth vertex of the
  parallelogram, and find its area.

Every answer is built first as a point and checked a second way (the parallelogram by its diagonals, the area by the
shoelace formula). Every wrong point is checked against the `points` grader before it is offered.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ..answers import Distractor
from ._w9_graph_kit import HALF, dist, enough, frac_text, part, plane, retry_draw
from ._w6a_kit import num_dist, num_fields

TOPIC = "w9_plot_points"
EXAM_TOPIC = "exd_x_w9_plot_points"

AREA = "plot-a-point-from-coordinates"
MISCONCEPTIONS = [
    (
        "w9-swapped-coordinates",
        "Plots the point with its coordinates the wrong way round: goes up first and across second.",
        AREA,
    ),
    (
        "w9-x-sign-ignored",
        "Plots a negative x-coordinate to the right of the y-axis, or a positive one to the left.",
        AREA,
    ),
    ("w9-y-sign-ignored", "Plots a negative y-coordinate above the x-axis, or a positive one below it.", AREA),
    ("w9-both-signs-ignored", "Ignores the signs of both coordinates, so the point lands in the first quadrant.", AREA),
    (
        "w9-counts-from-the-wrong-line",
        "Counts one gridline too far or too few along an axis, so the point is a square out.",
        AREA,
    ),
    (
        "w9-midpoint-not-halved",
        "Adds the coordinates of the two points to find the midpoint but does not halve the sums.",
        AREA,
    ),
    (
        "w9-midpoint-uses-difference",
        "Subtracts the coordinates of the two points instead of adding them to find the midpoint.",
        AREA,
    ),
    (
        "w9-midpoint-on-one-axis",
        "Finds the middle of the x-values but keeps the y-value of one of the points (or the reverse).",
        AREA,
    ),
    (
        "w9-parallelogram-wrong-pairing",
        "Completes the parallelogram on the wrong pair of sides, so the fourth point is not opposite the middle vertex.",
        AREA,
    ),
    ("w9-parallelogram-adds-all-three", "Adds the coordinates of all three vertices to find the fourth.", AREA),
    (
        "w9-parallelogram-area-half",
        "Gives the area of triangle ABC, which is half the area of the parallelogram.",
        AREA,
    ),
    (
        "w9-parallelogram-area-sides",
        "Multiplies the lengths of two adjacent sides, as if the parallelogram were a rectangle.",
        AREA,
    ),
    ("w9-parallelogram-area-bounding-box", "Gives the area of the box that just contains the parallelogram.", AREA),
    (
        "w9-line-point-sign",
        "Subtracts the constant instead of adding it (or the reverse) when finding the point on the line.",
        AREA,
    ),
    (
        "w9-line-point-forgets-gradient",
        "Uses the x-value as the y-value, or adds the gradient instead of multiplying by it.",
        AREA,
    ),
    ("w9-half-unit-rounded", "Rounds a half-unit coordinate to a whole unit.", AREA),
]

_CONTEXTS = (
    ("a drone", "from the launch pad", "east", "north"),
    ("a hiker", "from the car park", "east", "north"),
    ("a robot", "from its charging dock", "east", "north"),
    ("a treasure chest", "from the campfire", "east", "north"),
)


def _in(v: int) -> bool:
    return -HALF <= v <= HALF


def frac_dec(v: Fraction) -> str:
    """A coordinate as a decimal: 2.5, -1."""
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{float(v):g}"


def point_text(p) -> str:
    return f"({frac_dec(p[0])}, {frac_dec(p[1])})"


def points_text(points) -> str:
    return " ".join(point_text(p) for p in points)


def _on_grid(points) -> bool:
    return all(abs(v) <= HALF for p in points for v in p)


def _kept(*candidates):
    """The ``(points, tag)`` candidates whose points a learner could place on the grid."""
    return [c for c in candidates if _on_grid(c[0])]


def _answer_fields(points) -> dict:
    # coordinates are stored as the JSON the grader reads: a whole number, or a/b text for a half unit
    pts = [[_js(x), _js(y)] for x, y in points]
    text = points_text(points)
    return dict(answer=pts, answer_text=text, answer_latex="plain:" + text, answer_kind="points")


def _js(v):
    v = Fraction(v)
    return v.numerator if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _figure(alt, points=(), snap=1, given=()):
    pts = [{"at": [x, y], "label": name} for name, (x, y) in given]
    return plane(
        reply="points",
        tools=["plot_points"],
        alt=alt,
        snap=float(snap) if snap != 1 else 1,
        points=pts,
        max_points=max(len(points), 1) + 2,
    )


def _candidates_for_point(p, others=()):
    """Typed wrong replies for ONE point `p` = (x, y), with the misconception behind each."""
    x, y = p
    out = []
    if x != y:
        out.append(((y, x), "w9-swapped-coordinates"))
    if x != 0:
        out.append(((-x, y), "w9-x-sign-ignored"))
    if y != 0:
        out.append(((x, -y), "w9-y-sign-ignored"))
    if x != 0 and y != 0:
        out.append(((-x, -y), "w9-both-signs-ignored"))
    out.append(((x + 1, y), "w9-counts-from-the-wrong-line"))
    out.append(((x, y - 1), "w9-counts-from-the-wrong-line"))
    return out


def _point_dist(answer_fields: dict, candidate_sets) -> list[Distractor]:
    cands = []
    for pts, tag in candidate_sets:
        if not _on_grid(pts):
            continue  # a learner cannot place a point off the grid, so it is not a reply anyone could send
        cands.append((points_text(pts), [[_js(x), _js(y)] for x, y in pts], tag))
    return dist("points", answer_fields["answer"], answer_fields["answer_text"], cands)


def _question(text, points, fig, candidate_sets, working) -> dict:
    fields = _answer_fields(points)
    d = _point_dist(fields, candidate_sets)
    enough(d, 3)
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


# ---------------------------------------------------------------------------------------------- easy


def _easy(rng):
    x, y = rng.randint(1, HALF), rng.randint(1, HALF)
    if x == y:
        y = y % HALF + 1
    style = rng.choice(("coords", "named", "distance"))
    if style == "coords":
        text = phrase(
            rng,
            (
                "Plot the point ({x}, {y}) on the grid.",
                "Mark the point with coordinates ({x}, {y}) on the grid.",
                "Place a point at ({x}, {y}).",
            ),
            x=x,
            y=y,
        )
    elif style == "named":
        name = rng.choice("PQRSTUVW")
        text = phrase(
            rng,
            ("Plot the point {n}({x}, {y}).", "Mark point {n} at ({x}, {y}) on the grid."),
            n=name,
            x=x,
            y=y,
        )
    else:
        thing, origin, east, north = rng.choice(_CONTEXTS)
        text = (
            f"The origin of the grid is the point {origin}. Plot the position of {thing} that is {_sq(x)} {east} "
            f"and {_sq(y)} {north} of the origin."
        )
    working = [
        f"The first coordinate is across: start at the origin and move {x} to the right.",
        f"The second coordinate is up: move {y} up. The point is ({x}, {y}).",
    ]
    fig = _figure("A blank coordinate grid with x and y axes, for plotting a point.", [(x, y)])
    return _question(text, [(x, y)], fig, [([c], tag) for c, tag in _candidates_for_point((x, y))], working)


def _sq(n: int) -> str:
    return f"{n} square" + ("" if n == 1 else "s")


# ---------------------------------------------------------------------------------------------- medium


def _signed_word(v: int, neg: str, pos: str) -> str:
    return f"{_sq(abs(v))} {neg if v < 0 else pos}"


def _medium_single(rng):
    for _ in attempts(100, "a point off both axes in any quadrant"):
        x, y = rng.randint(-HALF, HALF), rng.randint(-HALF, HALF)
        if x and y and x != y and x != -y:
            break
    if rng.random() < 0.5:
        text = phrase(
            rng,
            (
                "Plot the point ({x}, {y}) on the grid.",
                "Mark the point with coordinates ({x}, {y}).",
            ),
            x=x,
            y=y,
        )
    else:
        h = _signed_word(x, "to the left of", "to the right of")
        v = _signed_word(y, "below", "above")
        text = f"Plot the point that is {h} the y-axis and {v} the x-axis."
    working = [
        f"Move {abs(x)} {'left' if x < 0 else 'right'} from the origin, because the x-coordinate is {x}.",
        f"Then move {abs(y)} {'down' if y < 0 else 'up'}, because the y-coordinate is {y}. The point is ({x}, {y}).",
    ]
    fig = _figure("A blank coordinate grid with x and y axes, for plotting a point.", [(x, y)])
    return _question(text, [(x, y)], fig, [([c], tag) for c, tag in _candidates_for_point((x, y))], working)


def _medium_two(rng):
    for _ in attempts(200, "two points in different quadrants"):
        a = (rng.randint(-HALF, HALF), rng.randint(-HALF, HALF))
        b = (rng.randint(-HALF, HALF), rng.randint(-HALF, HALF))
        if (
            a != b
            and all(c and c != -d for c, d in (a, b))
            and a[0] != a[1]
            and b[0] != b[1]
            and (a[0] > 0) != (b[0] > 0)
        ):
            break
    names = rng.choice((("A", "B"), ("P", "Q"), ("M", "N")))
    text = f"Plot the points {names[0]}({a[0]}, {a[1]}) and {names[1]}({b[0]}, {b[1]}) on the grid."
    cands = []
    for p, q in ((a, b), (b, a)):
        pass
    cands.append(([(a[1], a[0]), b], "w9-swapped-coordinates"))
    cands.append(([a, (b[1], b[0])], "w9-swapped-coordinates"))
    cands.append(([(-a[0], a[1]), b], "w9-x-sign-ignored"))
    cands.append(([a, (b[0], -b[1])], "w9-y-sign-ignored"))
    cands.append(([(abs(a[0]), abs(a[1])), (abs(b[0]), abs(b[1]))], "w9-both-signs-ignored"))
    cands.append(([a, (b[0] + 1, b[1])], "w9-counts-from-the-wrong-line"))
    working = [
        f"{names[0]}: move {abs(a[0])} {'left' if a[0] < 0 else 'right'}, then {abs(a[1])} {'down' if a[1] < 0 else 'up'}.",
        f"{names[1]}: move {abs(b[0])} {'left' if b[0] < 0 else 'right'}, then {abs(b[1])} {'down' if b[1] < 0 else 'up'}.",
    ]
    fig = _figure("A blank coordinate grid with x and y axes, for plotting two points.", [a, b])
    return _question(text, [a, b], fig, cands, working)


def _medium(rng):
    return _medium_single(rng) if rng.random() < 0.55 else _medium_two(rng)


# ---------------------------------------------------------------------------------------------- hard


def _hard_parallelogram(rng):
    for _ in attempts(300, "a parallelogram that fits the grid"):
        a = (rng.randint(-5, 5), rng.randint(-5, 5))
        b = (a[0] + rng.randint(2, 6), a[1] + rng.randint(-3, 3))
        c = (b[0] + rng.randint(-3, 3), b[1] + rng.randint(2, 5))
        d = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
        area = abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if all(_in(v) for p in (a, b, c, d) for v in p) and area >= 6 and a[0] != a[1]:
            break
    check(
        Fraction(a[0] + c[0], 2) == Fraction(b[0] + d[0], 2) and Fraction(a[1] + c[1], 2) == Fraction(b[1] + d[1], 2),
        "the diagonals of ABCD do not bisect each other",
    )
    text = phrase(
        rng,
        (
            "The points A, B and C are plotted. ABCD is a parallelogram with its vertices in that order. Plot D.",
            "A, B and C are three vertices of the parallelogram ABCD, taken in order. Plot the fourth vertex D.",
        ),
    )
    cands = [
        ([(a[0] + b[0] - c[0], a[1] + b[1] - c[1])], "w9-parallelogram-wrong-pairing"),
        ([(b[0] + c[0] - a[0], b[1] + c[1] - a[1])], "w9-parallelogram-wrong-pairing"),
        ([(a[0] + b[0] + c[0], a[1] + b[1] + c[1])], "w9-parallelogram-adds-all-three"),
        ([(b[0], c[1])], "w9-parallelogram-wrong-pairing"),
        ([(d[1], d[0])], "w9-swapped-coordinates"),
    ]
    working = [
        "In a parallelogram the diagonals bisect each other, so the midpoints of AC and BD are the same point.",
        f"Midpoint of AC = ({frac_text(Fraction(a[0] + c[0], 2))}, {frac_text(Fraction(a[1] + c[1], 2))}), so D = A + C - B = {point_text(d)}.",
    ]
    fig = _figure(
        "A coordinate grid with three labelled points A, B and C plotted, for plotting a fourth point.",
        [d],
        given=[("A", a), ("B", b), ("C", c)],
    )
    return _question(text, [d], fig, cands, working)


def _hard_midpoint(rng):
    for _ in attempts(300, "two points with a whole-number midpoint"):
        a = (rng.randint(-HALF, HALF), rng.randint(-HALF, HALF))
        b = (rng.randint(-HALF, HALF), rng.randint(-HALF, HALF))
        if (a[0] + b[0]) % 2 == 0 and (a[1] + b[1]) % 2 == 0 and abs(a[0] - b[0]) >= 4 and abs(a[1] - b[1]) >= 4:
            break
    m = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    check(m[0] - a[0] == b[0] - m[0] and m[1] - a[1] == b[1] - m[1], "M is not halfway from A to B")
    text = phrase(
        rng,
        (
            "The points A and B are plotted. Plot M, the midpoint of the line segment AB.",
            "A and B are plotted on the grid. Plot the point halfway between A and B.",
        ),
    )
    cands = [
        ([(a[0] + b[0], a[1] + b[1])], "w9-midpoint-not-halved"),
        ([(abs(a[0] - b[0]) // 2, abs(a[1] - b[1]) // 2)], "w9-midpoint-uses-difference"),
        ([(m[0], a[1])], "w9-midpoint-on-one-axis"),
        ([(a[0], m[1])], "w9-midpoint-on-one-axis"),
        ([(m[1], m[0])], "w9-swapped-coordinates"),
    ]
    working = [
        f"Add the x-values and halve: ({a[0]} + {b[0]}) / 2 = {m[0]}. Do the same for the y-values: ({a[1]} + {b[1]}) / 2 = {m[1]}.",
        f"The midpoint is {point_text(m)}.",
    ]
    fig = _figure(
        "A coordinate grid with two labelled points A and B plotted, for plotting a third point.",
        [m],
        given=[("A", a), ("B", b)],
    )
    return _question(text, [m], fig, cands, working)


def _hard_on_line(rng):
    m = rng.choice((-3, -2, -1, 2, 3))
    c = rng.randint(-3, 3)
    for _ in attempts(200, "an x-value that keeps the point on the grid"):
        x = rng.randint(-4, 4)
        y = m * x + c
        if x not in (0, 1) and _in(y) and x != y:
            break
    eq = f"y = {m}x" + (f" + {c}" if c > 0 else f" - {-c}" if c < 0 else "")
    text = f"A point on the line {eq} has x-coordinate {x}. Plot the point."
    cands = [
        ([(x, m * x - c)], "w9-line-point-sign"),
        ([(x, m + x + c)], "w9-line-point-forgets-gradient"),
        ([(x, x * c + m)], "w9-line-point-forgets-gradient"),
        ([(y, x)], "w9-swapped-coordinates"),
        ([(x, -y)], "w9-y-sign-ignored"),
    ]
    working = [
        f"Substitute x = {x}: y = {m} x ({x}) {'+' if c >= 0 else '-'} {abs(c)} = {y}.",
        f"The point is ({x}, {y}).",
    ]
    fig = _figure("A blank coordinate grid with x and y axes, for plotting a point.", [(x, y)])
    return _question(text, [(x, y)], fig, cands, working)


def _hard_halves(rng):
    for _ in attempts(100, "a half-unit point"):
        x = Fraction(rng.randint(-14, 14), 2)
        y = Fraction(rng.randint(-14, 14), 2)
        if (x.denominator == 2 or y.denominator == 2) and x != y and x != -y and x and y:
            break
    text = f"Plot the point ({frac_dec(x)}, {frac_dec(y)}). The grid snaps to half squares."
    near = lambda v: Fraction(int(v - Fraction(1, 2)) if v > 0 else int(v + Fraction(1, 2)))  # noqa: E731
    cands = [
        ([(near(x) if x.denominator == 2 else x, near(y) if y.denominator == 2 else y)], "w9-half-unit-rounded"),
        ([(y, x)], "w9-swapped-coordinates"),
        ([(-x, y)], "w9-x-sign-ignored"),
        ([(x, -y)], "w9-y-sign-ignored"),
        ([(x + Fraction(1, 2), y)], "w9-counts-from-the-wrong-line"),
    ]
    working = [
        f"Each grid square is split into halves, so {frac_dec(x)} is {'a whole number of squares' if x.denominator == 1 else 'between two gridlines'}.",
        f"The point is ({frac_dec(x)}, {frac_dec(y)}).",
    ]
    fig = _figure(
        "A blank coordinate grid with x and y axes, for plotting a point at half squares.",
        [(x, y)],
        snap=Fraction(1, 2),
    )
    fields = _answer_fields([(x, y)])
    d = dist(
        "points",
        fields["answer"],
        fields["answer_text"],
        [(points_text(p), [[_js(a), _js(b)] for a, b in p], tag) for p, tag in cands],
    )
    enough(d, 3)
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


_HARD = (_hard_parallelogram, _hard_midpoint, _hard_on_line, _hard_halves)


@generator(
    TOPIC,
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w9_plot_points(rng, tier="easy"):
    """Plot points on a grid; the tiers are described in the module docstring."""
    pool = {"easy": (_easy,), "medium": (_medium,), "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# ---------------------------------------------------------------------------------------------- exam


@generator(EXAM_TOPIC, level=9, presets={"exam": dict(tier="exam")})
def exd_x_w9_plot_points(rng, tier="exam"):
    """Exam chain: plot A, B and C; plot the midpoint of AC; plot D; find the area of the parallelogram."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam)


def _exam(rng):
    for _ in attempts(500, "a parallelogram that fits the grid"):
        a = (rng.randint(-5, 3), rng.randint(-5, 3))
        b = (a[0] + rng.randint(2, 5), a[1] + rng.randint(-2, 3))
        c = (b[0] + rng.randint(-3, 3), b[1] + rng.randint(2, 4))
        d = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
        if (a[0] + c[0]) % 2 or (a[1] + c[1]) % 2:
            continue
        area = abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if all(_in(v) for p in (a, b, c, d) for v in p) and area >= 6 and len({a, b, c, d}) == 4 and a[0] != a[1]:
            break
    m = ((a[0] + c[0]) // 2, (a[1] + c[1]) // 2)
    check(m == ((b[0] + d[0]) // 2, (b[1] + d[1]) // 2), "the diagonals do not share a midpoint")
    shoelace = Fraction(abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip((a, b, c, d), (b, c, d, a)))), 2)
    check(shoelace == area, "the shoelace area differs from the cross-product area")
    box = (max(p[0] for p in (a, b, c, d)) - min(p[0] for p in (a, b, c, d))) * (
        max(p[1] for p in (a, b, c, d)) - min(p[1] for p in (a, b, c, d))
    )
    side_ab = (abs(b[0] - a[0]) or 1) * (abs(c[1] - b[1]) or 1)

    fa = _answer_fields([a, b, c])
    fb = _answer_fields([m])
    fc = _answer_fields([d])
    da = dist(
        "points",
        fa["answer"],
        fa["answer_text"],
        [
            (points_text(p), [[_js(x), _js(y)] for x, y in p], tag)
            for p, tag in _kept(
                ([(a[1], a[0]), b, c], "w9-swapped-coordinates"),
                ([a, (-b[0], b[1]), c], "w9-x-sign-ignored"),
                ([a, b, (c[0], -c[1])], "w9-y-sign-ignored"),
                ([a, b, (c[0] + 1, c[1])], "w9-counts-from-the-wrong-line"),
            )
        ],
    )
    db = dist(
        "points",
        fb["answer"],
        fb["answer_text"],
        [
            (points_text(p), [[_js(x), _js(y)] for x, y in p], tag)
            for p, tag in _kept(
                ([(a[0] + c[0], a[1] + c[1])], "w9-midpoint-not-halved"),
                (
                    [
                        (
                            (a[0] + b[0]) // 2 if (a[0] + b[0]) % 2 == 0 else a[0],
                            (a[1] + b[1]) // 2 if (a[1] + b[1]) % 2 == 0 else a[1],
                        )
                    ],
                    "w9-midpoint-on-one-axis",
                ),
                ([(abs(a[0] - c[0]) // 2, abs(a[1] - c[1]) // 2)], "w9-midpoint-uses-difference"),
                ([(m[1], m[0])], "w9-swapped-coordinates"),
            )
        ],
    )
    dc = dist(
        "points",
        fc["answer"],
        fc["answer_text"],
        [
            (points_text(p), [[_js(x), _js(y)] for x, y in p], tag)
            for p, tag in _kept(
                ([(a[0] + b[0] - c[0], a[1] + b[1] - c[1])], "w9-parallelogram-wrong-pairing"),
                ([(b[0] + c[0] - a[0], b[1] + c[1] - a[1])], "w9-parallelogram-wrong-pairing"),
                ([(a[0] + b[0] + c[0], a[1] + b[1] + c[1])], "w9-parallelogram-adds-all-three"),
                ([(d[1], d[0])], "w9-swapped-coordinates"),
            )
        ],
    )
    dd = num_dist(
        area,
        [
            (Fraction(area, 2), "w9-parallelogram-area-half"),
            (side_ab, "w9-parallelogram-area-sides"),
            (box, "w9-parallelogram-area-bounding-box"),
            (area + 2, "w9-parallelogram-area-sides"),
        ],
    )
    for dl in (da, db, dc, dd):
        enough(dl)
    stem = (
        f"A({a[0]}, {a[1]}), B({b[0]}, {b[1]}) and C({c[0]}, {c[1]}) are three vertices of the parallelogram ABCD, "
        "taken in that order."
    )
    fig = _figure("A blank coordinate grid with x and y axes, for plotting points.", [a, b, c, d])
    return dict(
        text=stem,
        figure=fig,
        parts=[
            part(
                "Plot A, B and C on the grid.",
                "points",
                {k: fa[k] for k in ("answer", "answer_text", "answer_latex")},
                2,
                [f"A is {point_text(a)}, B is {point_text(b)} and C is {point_text(c)}."],
                da,
            ),
            part(
                "Plot M, the midpoint of AC.",
                "points",
                {k: fb[k] for k in ("answer", "answer_text", "answer_latex")},
                1,
                [f"M = (({a[0]} + {c[0]}) / 2, ({a[1]} + {c[1]}) / 2) = {point_text(m)}."],
                db,
            ),
            part(
                "Plot D, the fourth vertex.",
                "points",
                {k: fc[k] for k in ("answer", "answer_text", "answer_latex")},
                2,
                [f"The diagonals bisect each other, so M is also the midpoint of BD: D = 2M - B = {point_text(d)}."],
                dc,
            ),
            part(
                "Find the area of the parallelogram ABCD.",
                "number",
                num_fields(area),
                2,
                [f"Area = |AB x AD| = |({b[0] - a[0]})({d[1] - a[1]}) - ({b[1] - a[1]})({d[0] - a[0]})| = {area}."],
                dd,
                sensible_range=(0, None),
            ),
        ],
    )
