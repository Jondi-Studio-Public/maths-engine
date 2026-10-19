"""Generates `w9_line_draw` questions (easy, medium, hard) and the exam chain `exd_x_w9_line_draw`.

``line-draw-from-gradient-and-point`` (``linear-relations``, answer ``line``): the learner draws a straight line on a
coordinate grid. The front end sends two points the learner placed on the line (``[[0, -1], [1, 1]]``); the existing
``line`` grader (`mathgen.grading.typed.grade_line`) accepts any two different points that lie on the target line, so
every way of drawing it is right. The typed form is the two points written out, ``(0, -1) (1, 1)``.

* easy: a line from its gradient and y-intercept, as ``y = mx + c`` or in words.
* medium: a line through a stated point with a stated gradient (a fraction as often as a whole number), or with a
  stated x-intercept and gradient.
* hard: a line from an equation in standard form, a line through a point parallel to a line given in standard form, or
  perpendicular to a line given as ``y = mx + c``.
* exam (``exd_x_w9_line_draw``): draw ``y = mx + c``, find its x-intercept, draw the parallel line through a point,
  and find that line's y-intercept.

Each line is built from two lattice points on the grid, and the key is checked a second way: both points satisfy the
line's equation, and (parallel and perpendicular) the direction vectors have equal cross product or zero dot product.
Every wrong line is checked against the `line` grader before it is offered.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..core import attempts, check, generator, phrase
from ._w6a_kit import num_dist, num_fields
from ._w9_graph_kit import HALF, dist, enough, frac_text, line_text, part, plane, point_text, retry_draw

TOPIC = "w9_line_draw"
EXAM_TOPIC = "exd_x_w9_line_draw"

AREA = "line-draw-from-gradient-and-point"
MISCONCEPTIONS = [
    (
        "w9-line-gradient-sign",
        "Draws a line as steep as the gradient says but sloping the wrong way: the sign of the gradient is flipped.",
        AREA,
    ),
    ("w9-line-intercept-sign", "Starts the line at the opposite end of the y-axis, at minus the intercept.", AREA),
    (
        "w9-line-gradient-inverted",
        "Moves across by the rise and up by the run, so the line has the reciprocal of the gradient.",
        AREA,
    ),
    (
        "w9-line-axes-swapped",
        "Plots the points of the line with x and y the wrong way round, which reflects the line in y = x.",
        AREA,
    ),
    ("w9-line-starts-off-the-intercept", "Starts the line one square above or below the intercept.", AREA),
    (
        "w9-line-ignores-the-point",
        "Draws a line with the right gradient but through the origin instead of through the given point.",
        AREA,
    ),
    (
        "w9-line-through-swapped-point",
        "Draws the line through the given point with its coordinates the wrong way round.",
        AREA,
    ),
    (
        "w9-line-parallel-reflected",
        "Draws the line through the point with the opposite gradient, instead of the same gradient.",
        AREA,
    ),
    (
        "w9-line-perpendicular-parallel",
        "Draws a parallel line through the point when a perpendicular one was asked for.",
        AREA,
    ),
    (
        "w9-line-perpendicular-no-flip",
        "Takes the reciprocal of the gradient but keeps its sign, so the line is not perpendicular.",
        AREA,
    ),
    (
        "w9-line-perpendicular-negated-only",
        "Negates the gradient but forgets to take the reciprocal, so the line is not perpendicular.",
        AREA,
    ),
    (
        "w9-line-intercepts-swapped",
        "Swaps the x- and y-intercepts of the line, which gives a line with the wrong slope.",
        AREA,
    ),
    ("w9-line-standard-form-sign", "Takes the gradient of ax + by = c as a/b instead of -a/b.", AREA),
    (
        "w9-line-intercept-from-coefficient",
        "Reads the y-intercept off the constant on the right of the equation without dividing by the y-coefficient.",
        AREA,
    ),
    ("w9-line-x-intercept-sign", "Finds the x-intercept as c/m instead of -c/m.", AREA),
    ("w9-line-x-intercept-is-y-intercept", "Gives the y-intercept where the x-intercept was asked.", AREA),
    ("w9-line-x-intercept-inverted", "Finds the x-intercept as -m/c, dividing the wrong way round.", AREA),
    (
        "w9-line-intercept-uses-gradient",
        "Finds the y-intercept of the parallel line by using the gradient of the first line as the intercept.",
        AREA,
    ),
    (
        "w9-line-intercept-point-sign",
        "Subtracts the gradient times x from y but adds it, or the reverse, when finding the intercept through the point.",
        AREA,
    ),
]


def _in(p) -> bool:
    return all(abs(v) <= HALF for v in p)


def _lattice_points(anchor, direction):
    """Two lattice points on the line through `anchor` with direction vector `direction`, both on the grid, or None."""
    dx, dy = direction
    g = gcd(abs(int(dx)), abs(int(dy)))
    dx, dy = int(dx) // g, int(dy) // g
    for sign in (1, -1):
        pts = [anchor, (anchor[0] + sign * dx, anchor[1] + sign * dy)]
        if all(_in(p) for p in pts):
            return pts
    return None


def _dir_of(m: Fraction):
    return (m.denominator, m.numerator)


def _ans(points) -> dict:
    return dict(
        answer=[[int(x), int(y)] for x, y in points],
        answer_text=" ".join(point_text(p) for p in points),
        answer_latex="plain:" + " ".join(point_text(p) for p in points),
        answer_kind="line",
    )


def _line_dist(fields: dict, candidates) -> list:
    """`candidates` is ``[(points, tag), ...]``: wrong lines by two lattice points each. Off-grid ones are skipped."""
    cands = []
    for pts, tag in candidates:
        if pts is None or not all(_in(p) for p in pts) or pts[0] == pts[1]:
            continue
        cands.append((" ".join(point_text(p) for p in pts), [[int(x), int(y)] for x, y in pts], tag))
    return dist("line", fields["answer"], fields["answer_text"], cands)


def _figure(alt, given=()):
    pts = [{"at": list(p), "label": name} for name, p in given]
    return plane(reply="line", tools=["line_two_points"], alt=alt, snap=1, points=pts, max_points=2)


def _swap(points):
    return [(y, x) for x, y in points]


def _on_line(points, m: Fraction, c: Fraction) -> bool:
    return all(Fraction(y) == m * x + c for x, y in points) and points[0] != points[1]


def _slope_intercept_wrong(m: Fraction, c: Fraction):
    """Wrong lines for y = mx + c, as ``(points, tag)``."""
    out = []
    out.append((_lattice_points((0, -c), _dir_of(-m)), "w9-line-gradient-sign"))
    out.append((_lattice_points((0, -c), _dir_of(m)), "w9-line-intercept-sign"))
    if abs(m) != 1:
        out.append((_lattice_points((0, c), _dir_of(1 / m)), "w9-line-gradient-inverted"))
    out.append((_lattice_points((0, c), _dir_of(-m)), "w9-line-gradient-sign"))
    out.append((_lattice_points((0, c + 1), _dir_of(m)), "w9-line-starts-off-the-intercept"))
    out.append((_lattice_points((0, c - 1), _dir_of(m)), "w9-line-starts-off-the-intercept"))
    base = _lattice_points((0, c), _dir_of(m))
    if base:
        out.append((_swap(base), "w9-line-axes-swapped"))
    return [(p, t) for p, t in out if p]


# ---------------------------------------------------------------------------------------------- easy


def _easy(rng):
    for _ in attempts(200, "a line that crosses the grid"):
        m = Fraction(rng.choice((-3, -2, -1, 1, 2, 3)))
        c = Fraction(rng.randint(-4, 4))
        pts = _lattice_points((0, c), _dir_of(m))
        if pts and not (c == 0 and abs(m) == 1):
            break
    check(_on_line(pts, m, c), "the points are not on the line")
    if rng.random() < 0.6:
        text = phrase(
            rng,
            ("Draw the line {eq} on the grid.", "Draw the graph of {eq}."),
            eq=line_text(m, c),
        )
    else:
        text = (
            f"Draw the straight line with gradient {frac_text(m)} and y-intercept {frac_text(c)}."
            if c
            else f"Draw the straight line with gradient {frac_text(m)} that passes through the origin."
        )
    fields = _ans(pts)
    d = _line_dist(fields, _slope_intercept_wrong(m, c))
    enough(d, 3)
    working = [
        f"The y-intercept is {frac_text(c)}, so the line passes through (0, {frac_text(c)}).",
        f"The gradient is {frac_text(m)}: for every 1 across, the line goes {'up' if m > 0 else 'down'} {abs(m)}. "
        f"So it also passes through {point_text(pts[1])}.",
    ]
    return dict(
        text=text,
        figure=_figure("A blank coordinate grid with x and y axes, for drawing a straight line."),
        working=working,
        distractors=d,
        **fields,
    )


# ---------------------------------------------------------------------------------------------- medium


def _medium(rng):
    if rng.random() < 0.7:
        return _medium_point_gradient(rng)
    return _medium_x_intercept(rng)


def _gradient_text(m: Fraction) -> str:
    return frac_text(m)


def _medium_point_gradient(rng):
    for _ in attempts(300, "a point and gradient whose line fits the grid"):
        m = rng.choice(
            (
                Fraction(1, 2),
                Fraction(-1, 2),
                Fraction(2, 3),
                Fraction(-2, 3),
                Fraction(3, 2),
                Fraction(-3, 2),
                Fraction(1, 3),
                Fraction(-1, 3),
                Fraction(3, 4),
                Fraction(-3, 4),
                Fraction(-2),
                Fraction(3),
                Fraction(-3),
                Fraction(2),
                Fraction(4, 3),
                Fraction(-4, 3),
            )
        )
        p = (rng.randint(-5, 5), rng.randint(-5, 5))
        pts = _lattice_points(p, _dir_of(m))
        if pts and p[0] != 0 and p[1] != 0 and abs(p[0]) != abs(p[1]):
            break
    c = p[1] - m * p[0]
    check(_on_line(pts, m, c), "the points are not on the line")
    text = phrase(
        rng,
        (
            "Draw the line that passes through the point P{p} and has gradient {m}.",
            "Draw the straight line through P{p} with a gradient of {m}.",
        ),
        p=point_text(p),
        m=_gradient_text(m),
    )
    fields = _ans(pts)
    cands = [
        (_lattice_points(p, _dir_of(-m)), "w9-line-gradient-sign"),
        (_lattice_points(p, _dir_of(1 / m)), "w9-line-gradient-inverted"),
        (_lattice_points((0, 0), _dir_of(m)), "w9-line-ignores-the-point"),
        (_lattice_points((p[1], p[0]), _dir_of(m)), "w9-line-through-swapped-point"),
        (_lattice_points((p[0], -p[1]), _dir_of(m)), "w9-line-through-swapped-point"),
    ]
    d = _line_dist(fields, cands)
    enough(d, 3)
    run, rise = _dir_of(m)
    working = [
        f"Start at P{point_text(p)}.",
        f"A gradient of {frac_text(m)} means: for every {abs(run)} across to the right, go {'up' if m > 0 else 'down'} {abs(rise)}. "
        f"This gives a second point {point_text((p[0] + run, p[1] + rise))}.",
        "Draw the straight line through the two points.",
    ]
    fig = _figure(
        "A coordinate grid with the point P plotted, for drawing a straight line through it.", given=[("P", p)]
    )
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


def _medium_x_intercept(rng):
    for _ in attempts(300, "an x-intercept and gradient whose line fits the grid"):
        x0 = rng.choice([v for v in range(-6, 7) if v])
        m = rng.choice(
            (
                Fraction(-2),
                Fraction(-1),
                Fraction(1),
                Fraction(2),
                Fraction(3),
                Fraction(-3),
                Fraction(1, 2),
                Fraction(-1, 2),
                Fraction(3, 2),
                Fraction(-3, 2),
            )
        )
        pts = _lattice_points((x0, 0), _dir_of(m))
        c = -m * x0
        if pts and abs(c) <= HALF:
            break
    check(_on_line(pts, m, c) and Fraction(0) == m * x0 + c, "the points are not on the line")
    text = f"Draw the line that crosses the x-axis at ({x0}, 0) and has gradient {frac_text(m)}."
    fields = _ans(pts)
    cands = [
        (_lattice_points((x0, 0), _dir_of(-m)), "w9-line-gradient-sign"),
        (_lattice_points((0, x0), _dir_of(m)), "w9-line-intercepts-swapped"),
        (_lattice_points((-x0, 0), _dir_of(m)), "w9-line-intercept-sign"),
        (_lattice_points((x0, 0), _dir_of(1 / m)), "w9-line-gradient-inverted"),
        (_swap(pts), "w9-line-axes-swapped"),
    ]
    d = _line_dist(fields, cands)
    enough(d, 3)
    run, rise = _dir_of(m)
    working = [
        f"The line passes through ({x0}, 0).",
        f"A gradient of {frac_text(m)} gives a second point {point_text((x0 + run, rise))}: {abs(run)} across, {abs(rise)} {'up' if m > 0 else 'down'}.",
    ]
    return dict(
        text=text,
        figure=_figure("A blank coordinate grid with x and y axes, for drawing a straight line."),
        working=working,
        distractors=d,
        **fields,
    )


# ---------------------------------------------------------------------------------------------- hard


def _std_text(a: int, b: int, c: int) -> str:
    """``3x - 2y = 6`` with a > 0."""
    left = []
    for coef, var in ((a, "x"), (b, "y")):
        if coef == 0:
            continue
        mag = "" if abs(coef) == 1 else str(abs(coef))
        left.append(("- " if coef < 0 else "+ ") + f"{mag}{var}" if left else ("-" if coef < 0 else "") + f"{mag}{var}")
    return f"{' '.join(left)} = {c}"


def _std_line(rng):
    """A line ax + by = c with lattice intercepts on the grid: (a, b, c, x-intercept, y-intercept)."""
    for _ in attempts(300, "a standard-form line with whole intercepts"):
        p = rng.choice([v for v in range(-6, 7) if v])
        q = rng.choice([v for v in range(-6, 7) if v])
        if abs(p) == abs(q):
            continue
        a, b, c = q, p, p * q
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        a, b, c = a // g, b // g, c // g
        if a < 0:
            a, b, c = -a, -b, -c
        if abs(a) <= 9 and abs(b) <= 9 and abs(c) <= 40:
            return a, b, c, p, q
    raise AssertionError  # pragma: no cover


def _hard_standard(rng):
    a, b, c, p, q = _std_line(rng)
    pts = [(p, 0), (0, q)]
    check(all(a * x + b * y == c for x, y in pts), "the intercepts are not on the line")
    m = Fraction(-a, b)
    text = phrase(rng, ("Draw the line {eq}.", "Draw the graph of {eq} on the grid."), eq=_std_text(a, b, c))
    fields = _ans(pts)
    cands = [
        (_lattice_points((0, q), _dir_of(-m)), "w9-line-standard-form-sign"),
        (_swap(pts), "w9-line-axes-swapped"),
        ([(0, c), (c // b if c % b == 0 else c, 0)], "w9-line-intercept-from-coefficient"),
        (_lattice_points((0, q), _dir_of(1 / m)), "w9-line-gradient-inverted"),
        (_lattice_points((0, -q), _dir_of(m)), "w9-line-intercept-sign"),
    ]
    d = _line_dist(fields, cands)
    enough(d, 3)
    working = [
        f"Put x = 0: {b}y = {c}, so y = {frac_text(Fraction(c, b))}. The line crosses the y-axis at (0, {q}).",
        f"Put y = 0: {a}x = {c}, so x = {frac_text(Fraction(c, a))}. The line crosses the x-axis at ({p}, 0).",
        "Draw the line through the two intercepts.",
    ]
    return dict(
        text=text,
        figure=_figure("A blank coordinate grid with x and y axes, for drawing a straight line."),
        working=working,
        distractors=d,
        **fields,
    )


def _hard_parallel(rng):
    for _ in attempts(500, "a parallel line that fits the grid"):
        a, b, c, p, q = _std_line(rng)
        pt = (rng.randint(-5, 5), rng.randint(-5, 5))
        direction = (b, -a)
        pts = _lattice_points(pt, direction)
        if pts and a * pt[0] + b * pt[1] != c and pt[0] and pt[1]:
            break
    check(a * pts[1][0] + b * pts[1][1] == a * pt[0] + b * pt[1], "the second point is not on the parallel line")
    m = Fraction(-a, b)
    text = (
        f"The line {_std_text(a, b, c)} is given. Draw the line parallel to it that passes through P{point_text(pt)}."
    )
    fields = _ans(pts)
    cands = [
        (_lattice_points(pt, _dir_of(-m)), "w9-line-parallel-reflected"),
        (_lattice_points(pt, _dir_of(1 / m)), "w9-line-gradient-inverted"),
        (_lattice_points((0, 0), _dir_of(m)), "w9-line-ignores-the-point"),
        (_lattice_points(pt, (b, a)), "w9-line-standard-form-sign"),
        (_swap(pts), "w9-line-axes-swapped"),
    ]
    d = _line_dist(fields, cands)
    enough(d, 3)
    working = [
        f"Parallel lines have the same gradient. Writing {_std_text(a, b, c)} as y = ..., the gradient is {frac_text(m)}.",
        f"From P{point_text(pt)}, go {abs(b)} across and {abs(a)} {'down' if (b > 0) == (a > 0) else 'up'} to reach {point_text(pts[1])}.",
    ]
    fig = _figure(
        "A coordinate grid with the point P plotted, for drawing a straight line through it.", given=[("P", pt)]
    )
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


def _hard_perpendicular(rng):
    for _ in attempts(500, "a perpendicular line that fits the grid"):
        m = rng.choice(
            (
                Fraction(2),
                Fraction(3),
                Fraction(-2),
                Fraction(-3),
                Fraction(1, 2),
                Fraction(-1, 2),
                Fraction(2, 3),
                Fraction(-2, 3),
                Fraction(3, 2),
                Fraction(-3, 2),
            )
        )
        c0 = rng.randint(-4, 4)
        pt = (rng.randint(-5, 5), rng.randint(-5, 5))
        run, rise = _dir_of(m)
        pts = _lattice_points(pt, (-rise, run))  # perpendicular to (run, rise)
        if pts and pt[1] != m * pt[0] + c0 and pt[0] and pt[1]:
            break
    # perpendicular: the dot product of the two direction vectors is zero
    d1 = (run, rise)
    d2 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    check(d1[0] * d2[0] + d1[1] * d2[1] == 0, "the new line is not perpendicular")
    text = f"Draw the line that is perpendicular to {line_text(m, c0)} and passes through P{point_text(pt)}."
    fields = _ans(pts)
    cands = [
        (_lattice_points(pt, _dir_of(m)), "w9-line-perpendicular-parallel"),
        (_lattice_points(pt, _dir_of(1 / m)), "w9-line-perpendicular-no-flip"),
        (_lattice_points(pt, _dir_of(-m)), "w9-line-perpendicular-negated-only"),
        (_lattice_points((0, 0), _dir_of(-1 / m)), "w9-line-ignores-the-point"),
        (_swap(pts), "w9-line-axes-swapped"),
    ]
    d = _line_dist(fields, cands)
    enough(d, 3)
    working = [
        f"The gradient of the given line is {frac_text(m)}. A perpendicular line has gradient {frac_text(-1 / m)} (the negative reciprocal).",
        f"From P{point_text(pt)}, go {abs(d2[0])} across and {abs(d2[1])} {'up' if d2[1] > 0 else 'down'} (or the reverse) to reach {point_text(pts[1])}.",
    ]
    fig = _figure(
        "A coordinate grid with the point P plotted, for drawing a straight line through it.", given=[("P", pt)]
    )
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


_HARD = (_hard_standard, _hard_parallel, _hard_perpendicular)


@generator(
    TOPIC,
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w9_line_draw(rng, tier="easy"):
    """Draw a straight line on a grid; the tiers are described in the module docstring."""
    pool = {"easy": (_easy,), "medium": (_medium,), "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# ---------------------------------------------------------------------------------------------- exam


@generator(EXAM_TOPIC, level=9, presets={"exam": dict(tier="exam")})
def exd_x_w9_line_draw(rng, tier="exam"):
    """Exam chain: draw y = mx + c; its x-intercept; the parallel line through P; that line's y-intercept."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam)


def _exam(rng):
    for _ in attempts(1000, "a line, an x-intercept and a parallel line that all fit"):
        m = Fraction(rng.choice((-3, -2, -1, 1, 2, 3)))
        x0 = rng.choice([v for v in range(-5, 6) if v])
        c = -m * x0
        pt = (rng.randint(-5, 5), rng.randint(-5, 5))
        c2 = pt[1] - m * pt[0]
        pts1 = _lattice_points((0, c), _dir_of(m))
        pts2 = _lattice_points(pt, _dir_of(m))
        if pts1 and pts2 and abs(c) <= 6 and c2 != c and abs(c2) <= HALF and pt[0] and pt[1] and c2 != 0:
            break
    check(_on_line(pts1, m, c) and m * x0 + c == 0, "line 1 is wrong")
    check(_on_line(pts2, m, c2) and pts2[0] == pt, "line 2 is wrong")
    f1, f2 = _ans(pts1), _ans(pts2)
    d1 = _line_dist(f1, _slope_intercept_wrong(m, c))
    d3 = _line_dist(
        f2,
        [
            (_lattice_points(pt, _dir_of(-m)), "w9-line-parallel-reflected"),
            (_lattice_points((0, 0), _dir_of(m)), "w9-line-ignores-the-point"),
            (_lattice_points(pt, _dir_of(1 / m)), "w9-line-gradient-inverted"),
            (_lattice_points((pt[1], pt[0]), _dir_of(m)), "w9-line-through-swapped-point"),
        ],
    )
    d2 = num_dist(
        x0,
        [
            (c / m if m else None, "w9-line-x-intercept-sign"),
            (c, "w9-line-x-intercept-is-y-intercept"),
            (-m / c if c else None, "w9-line-x-intercept-inverted"),
            (m * c, "w9-line-x-intercept-is-y-intercept"),
        ],
    )
    d4 = num_dist(
        c2,
        [
            (m, "w9-line-intercept-uses-gradient"),
            (pt[1] + m * pt[0], "w9-line-intercept-point-sign"),
            (c, "w9-line-intercept-uses-gradient"),
            (pt[1], "w9-line-intercept-point-sign"),
        ],
    )
    for dl in (d1, d2, d3, d4):
        enough(dl)
    stem = f"The line l has equation {line_text(m, c)}."
    fig = _figure("A blank coordinate grid with x and y axes, for drawing straight lines.", given=[("P", pt)])
    return dict(
        text=stem + f" The point P is {point_text(pt)}.",
        figure=fig,
        parts=[
            part(
                "Draw the line l on the grid.",
                "line",
                {k: f1[k] for k in ("answer", "answer_text", "answer_latex")},
                2,
                [f"l passes through {point_text(pts1[0])} and {point_text(pts1[1])}."],
                d1,
            ),
            part(
                "Find the x-coordinate of the point where l crosses the x-axis.",
                "number",
                num_fields(x0),
                2,
                [f"Put y = 0: 0 = {frac_text(m)}x + ({frac_text(c)}), so x = {x0}."],
                d2,
            ),
            part(
                "Draw the line parallel to l that passes through P.",
                "line",
                {k: f2[k] for k in ("answer", "answer_text", "answer_latex")},
                2,
                [f"A parallel line has gradient {frac_text(m)}. From P go to {point_text(pts2[1])}."],
                d3,
            ),
            part(
                "Find the y-intercept of the line you drew in the previous part.",
                "number",
                num_fields(c2),
                2,
                [
                    f"y = {frac_text(m)}x + c through {point_text(pt)}: c = {pt[1]} - ({frac_text(m)})({pt[0]}) = {frac_text(c2)}."
                ],
                d4,
            ),
        ],
    )
