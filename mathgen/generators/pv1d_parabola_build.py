"""- ``pv1d_quadratic_root_count`` (senior-recognise-features-of-the-graph-of-the-general,
  ACMMM011): how many real roots (x-intercepts) the drawn parabola shows, an mcq over
  {0, 1, 2}. The key class is drawn FIRST and uniformly, then a parabola realising it is
  built. Tiers escalate by the decision made: count the contacts with the x-axis (easy);
  decide touch / cross / miss when the vertex is on or one unit from the axis (medium);
  ignore a distractor line L and count against the x-axis only (hard); pick the right
  one of two curves by a lettered point (exam).

- ``pv1d_parabola_axis_x`` (senior-recognise-features-of-the-graphs-of-y-x, ACMMM007):
  the equation ``x = b`` of the axis of symmetry of a drawn parabola, ``answer: pair``
  with the single-equation key. Easy reads the marked vertex V; medium finds the unmarked
  vertex; hard takes the midpoint of two marked points at equal height (vertex off the
  grid); exam picks the one equal-height pair out of three marked points.

Every curve is ``y = a(x - h)^2 + c`` with integer ``a`` in {+-1, +-2}, integer ``h`` and
``c`` and is sampled on a grid whose step (1/8) makes every integer x an EXACT sample, so
the vertex, every integer root and every marked point is a sample of the polyline the
learner sees. Nothing the key needs is printed (point labels are single capitals, alt text
names the object). Each topic registers a figure reader that works from ``question.figure``
alone (plus the LETTER the prompt names, for the root-count exam tier).
"""

from __future__ import annotations

import re
from math import isqrt

from mathgen import figures
from mathgen.core import check, check_pairwise_distinct, generator, pick_distractors

# --------------------------------------------------------------------------- shared

_LO, _HI = -10, 10  # x and y window, grid 1
_N = 161  # (20 units) * 8 + 1 -> step 0.125, every integer x is a sample
_EPS = 1e-9

_TIERS = ("easy", "medium", "hard", "exam")
_PRESETS = {t: {"tier": t} for t in _TIERS}


def _sample(a: int, h: int, c: int) -> list:
    return figures.sample_curve(lambda x: a * (x - h) ** 2 + c, _LO, _HI, (_LO, _HI), n=_N)


def _plane(*, curves, alt, points=(), lines=()) -> dict:
    return figures.cartesian(
        x_range=(_LO, _HI),
        y_range=(_LO, _HI),
        alt=alt,
        grid=1,
        equal_scale=True,
        curves=[{"pieces": _sample(*cv)} for cv in curves],
        points=points,
        lines=lines,
    )


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


def _letters_in(text: str) -> str | None:
    m = re.search(r"point marked ([A-Z])\b", text)
    return m.group(1) if m else None


# =========================================================================== ROW 1
# root count of the general quadratic (ACMMM011)


def _r1_specs(family: str, k: int) -> list:
    """(a, h, c) triples realising root count ``k`` for the 'easy' or 'medium' family."""
    out = []
    if family == "easy":
        if k == 1:
            out = [(a, h, 0) for a in (-2, -1, 1, 2) for h in range(-7, 8)]
        elif k == 2:
            for amp, d in ((1, 2), (1, 3), (2, 1), (2, 2)):
                for s in (-1, 1):
                    for h in range(-8, 9):
                        if abs(h) + d <= 9:
                            out.append((s * amp, h, -s * amp * d * d))
        else:
            for amp in (1, 2):
                for s in (-1, 1):
                    for c0 in range(2, 9):
                        for h in range(-7, 8):
                            out.append((s * amp, h, s * c0))
    else:  # medium: vertex on the axis, or exactly one unit off it
        if k == 1:
            out = [(a, h, 0) for a in (-2, -1, 1, 2) for h in range(-8, 9)]
        elif k == 2:
            out = [(s, h, -s) for s in (-1, 1) for h in range(-8, 9)]
        else:
            out = [(s * amp, h, s) for amp in (1, 2) for s in (-1, 1) for h in range(-8, 9)]
    return out


def _discriminant_class(a: int, h: int, c: int) -> int:
    """Root count from the EXPANDED coefficients (a, -2ah, ah^2 + c): an independent route."""
    A, B, C = a, -2 * a * h, a * h * h + c
    disc = B * B - 4 * A * C
    return 1 if disc == 0 else (2 if disc > 0 else 0)


def _contacts(pieces) -> int:
    """Count x-axis contacts by walking the polyline: each run of zero samples is one
    contact (a tangent vertex or an exactly-sampled root); each sign change between
    consecutive non-zero samples is one more."""
    n = 0
    for piece in pieces:
        prev = None
        in_zero = False
        for _x, y in piece:
            s = 0 if abs(y) < _EPS else (1 if y > 0 else -1)
            if s == 0:
                if not in_zero:
                    n += 1
                in_zero = True
                continue
            if prev is not None and prev != s and not in_zero:
                n += 1
            prev = s
            in_zero = False
    return n


def _on_curve(curve: dict, pt) -> bool:
    px, py = pt
    for piece in curve["pieces"]:
        for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
            dx, dy = x1 - x0, y1 - y0
            ll = dx * dx + dy * dy
            t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / ll))
            qx, qy = x0 + t * dx, y0 + t * dy
            if (px - qx) ** 2 + (py - qy) ** 2 <= 1e-12:
                return True
    return False


def _count_from_figure(fig: dict, text: str) -> str:
    curves = fig["curves"]
    letter = _letters_in(text)
    if letter is not None:
        pts = [p["at"] for p in fig["points"] if p.get("label") == letter]
        check(len(pts) == 1, "the named letter is not a single marked point")
        chosen = [c for c in curves if _on_curve(c, pts[0])]
        check(len(chosen) == 1, "the named point is not on exactly one curve")
        curve = chosen[0]
    else:
        check(len(curves) == 1, "no letter named but the figure has several curves")
        curve = curves[0]
    return str(_contacts(curve["pieces"]))


def _read_root_count(question) -> str:
    derived = _count_from_figure(question.figure, question.text)
    return derived


figures.register_reader("pv1d_quadratic_root_count", _read_root_count)

_R1_CONV_EASY = (
    "A real root of ax^2 + bx + c = 0 is a point where the graph meets the x-axis.",
    "Real roots of the equation ax^2 + bx + c = 0 are the x-values where the graph meets the x-axis.",
    "Each point where the graph meets the x-axis gives one real root of ax^2 + bx + c = 0.",
)
_R1_POOLS = {
    "easy": (
        (
            "The graph of the quadratic y = ax^2 + bx + c is shown.",
            "The diagram shows the graph of a quadratic y = ax^2 + bx + c.",
            "Below is the graph of y = ax^2 + bx + c, a quadratic.",
            "A quadratic y = ax^2 + bx + c has the graph shown.",
        ),
        _R1_CONV_EASY,
        (
            "How many real roots does the quadratic have?",
            "Count the real roots of ax^2 + bx + c = 0 from the graph.",
            "How many real roots does ax^2 + bx + c = 0 have?",
        ),
    ),
    "medium": (
        (
            "Study this graph of y = ax^2 + bx + c.",
            "Here is the graph of a quadratic function y = ax^2 + bx + c, drawn on a grid.",
            "Use the grid to study the graph of y = ax^2 + bx + c.",
            "This grid shows the graph of the function y = ax^2 + bx + c.",
        ),
        (
            "Remember that a real root of ax^2 + bx + c = 0 is an x-value at which y = 0.",
            "A real root of the equation is any x-value for which the graph has y = 0.",
            "The real roots of ax^2 + bx + c = 0 are the x-values where y = 0 on the graph.",
        ),
        (
            "Decide how many different real roots the equation ax^2 + bx + c = 0 has.",
            "State the number of distinct real roots of ax^2 + bx + c = 0.",
            "How many distinct real x-values satisfy ax^2 + bx + c = 0?",
        ),
    ),
    "hard": (
        (
            "The graph of y = ax^2 + bx + c is drawn together with a second line, L.",
            "A quadratic y = ax^2 + bx + c is graphed, and the line L is drawn on the same grid.",
            "The line L is drawn on the same grid as the graph of y = ax^2 + bx + c.",
            "On one grid are the graph of y = ax^2 + bx + c and a line labelled L.",
        ),
        (
            "A real root of ax^2 + bx + c = 0 is a point where the parabola meets the x-axis, the line y = 0.",
            "Real roots of ax^2 + bx + c = 0 are found where the parabola crosses or meets the x-axis, y = 0.",
            "The real roots of ax^2 + bx + c = 0 are the x-values at which the parabola is on the x-axis.",
        ),
        (
            "How many real roots does ax^2 + bx + c = 0 have?",
            "Find the number of real roots of the equation ax^2 + bx + c = 0.",
            "Give the number of real roots of the quadratic equation ax^2 + bx + c = 0.",
        ),
    ),
    "exam": (
        (
            "Two quadratics are graphed on the same grid.",
            "The graphs of two different quadratics are drawn on one grid.",
            "Two parabolas are drawn on the same set of axes.",
            "One grid shows the graphs of two quadratics.",
        ),
        (
            "A real root of a quadratic is a point where its graph meets the x-axis.",
            "Real roots are the x-values where a graph meets the x-axis.",
            "Each place where a graph meets the x-axis is one real root of its quadratic.",
        ),
        (
            "How many real roots does the quadratic whose graph passes through the point marked {L} have?",
            "The graph of y = ax^2 + bx + c passes through the point marked {L}. How many real roots does ax^2 + bx + c = 0 have?",
            "State the number of real roots of ax^2 + bx + c = 0 for the curve through the point marked {L}.",
        ),
    ),
}

_R1_NOTES = {
    0: "The graph never meets the x-axis, so there are no real roots.",
    1: "The graph meets the x-axis at one point only, so there is one real root.",
    2: "The graph meets the x-axis at two points, so there are two real roots.",
}


def _l_count(a: int, h: int, c: int, m: int):
    """Intersections of y = a(x-h)^2 + c with y = m, or None if the crossing is not an
    integer x at least one unit inside the window (so it would be hard to count)."""
    num = m - c
    if num == 0:
        return 1
    if num * a < 0:
        return 0
    if num % a != 0:
        return None
    t = num // a
    r = isqrt(t)
    if r * r != t or abs(h) + r > 9:
        return None
    return 2


def _r1_exam_points(rng, c1, c2):
    """P on curve ``c1`` and Q on curve ``c2``: grid intersections, off the x-axis, not a
    vertex, at least 2 units from the other curve, and well apart."""

    def cands(own, other):
        a, h, c = own
        out = []
        for x in range(-9, 10):
            y = a * (x - h) ** 2 + c
            if x == h or y == 0 or abs(y) > 9:
                continue
            if abs(y - (other[0] * (x - other[1]) ** 2 + other[2])) < 2:
                continue
            out.append((x, y))
        return out

    p1, p2 = cands(c1, c2), cands(c2, c1)
    pairs = [(p, q) for p in p1 for q in p2 if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= 9]
    return _pick(rng, pairs) if pairs else None


def _verify_root_count(fig: dict, text: str, key: int) -> None:
    check(_count_from_figure(fig, text) == str(key), "figure does not show the keyed root count")


def _build_root_count(rng, tier: str):
    k = rng.randrange(3)  # the answer class FIRST, uniform
    lines, points = [], []
    if tier == "easy" or tier == "medium":
        spec = _pick(rng, _r1_specs(tier, k))
        curves = [spec]
        check(_discriminant_class(*spec) == k, "root-count class disagrees with the discriminant")
        alt = "a parabola on a coordinate grid"
        text = " ".join(_pick(rng, pool) for pool in _R1_POOLS[tier])
    elif tier == "hard":
        pool = _r1_specs("easy", k) + _r1_specs("medium", k)
        for _ in range(60):
            spec = _pick(rng, pool)
            ms = [m for m in range(-9, 10) if m != 0 and _l_count(*spec, m) is not None and _l_count(*spec, m) != k]
            if ms:
                break
        else:
            check(False, "no distractor line found")
        m = _pick(rng, ms)
        check(_discriminant_class(*spec) == k)
        check(_l_count(*spec, m) != k, "line L would give the keyed count")
        curves = [spec]
        lines = [{"through": [[_LO, m], [_HI, m]], "label": "L"}]
        alt = "a parabola and a horizontal line on a coordinate grid"
        text = " ".join(_pick(rng, pool_) for pool_ in _R1_POOLS[tier])
    else:  # exam
        k2 = (k + 1 + rng.randrange(2)) % 3  # a different class, uniform over the other two
        for _ in range(60):
            s1 = _pick(rng, _r1_specs("easy", k) + _r1_specs("medium", k))
            s2 = _pick(rng, _r1_specs("easy", k2) + _r1_specs("medium", k2))
            pq = _r1_exam_points(rng, s1, s2)
            if pq is not None:
                break
        else:
            check(False, "no exam placement found")
        check(_discriminant_class(*s1) == k and _discriminant_class(*s2) == k2)
        asked = "P" if rng.random() < 0.5 else "Q"
        other = "Q" if asked == "P" else "P"
        curves = [s1, s2]
        points = [{"at": list(pq[0]), "label": asked}, {"at": list(pq[1]), "label": other}]
        alt = "two parabolas on a coordinate grid"
        text = " ".join(_pick(rng, pool_) for pool_ in _R1_POOLS[tier]).replace("{L}", asked)
        # keep the asked curve's drawing order random so position never gives it away
        if rng.randrange(2):
            curves = [s2, s1]
            points = [points[1], points[0]]
    fig = _plane(curves=curves, alt=alt, points=points, lines=lines)
    _verify_root_count(fig, text, k)
    return fig, text, k


@generator("pv1d_quadratic_root_count", level=13, presets=_PRESETS)
def pv1d_quadratic_root_count(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, k = _build_root_count(rng, tier)
    answer_text = str(k)
    distractors = pick_distractors(answer_text, [str(j) for j in (0, 1, 2) if j != k], n=2)
    check_pairwise_distinct(k, *[int(d) for d in distractors])
    check(len(distractors) == 2, "the mcq must offer all three counts")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=[_R1_NOTES[k]],
        figure=fig,
    )


# =========================================================================== ROW 2
# axis of symmetry x = b (ACMMM007)


def _axis_from_coefficients(a: int, h: int, c: int) -> int:
    """x = -B/(2A) from the EXPANDED coefficients: an independent route to b."""
    A, B = a, -2 * a * h
    check(B % (2 * A) == 0, "axis is not an integer")
    return -B // (2 * A)


def _x_eq(v) -> str:
    return f"x = {v}"


def _b_values(limit: int) -> list:
    return [b for b in range(-limit, limit + 1) if b != 0]


_R2_POOLS = {
    "easy": (
        (
            "The parabola shown has its turning point marked V.",
            "A parabola is drawn on a grid. Its turning point is the point marked V.",
            "The point V marks the turning point of the parabola in the graph.",
            "On the graph, V is the turning point of the parabola.",
        ),
        (
            "Write the equation of its axis of symmetry in the form x = ... .",
            "What is the equation of the axis of symmetry? Give it as x = ... .",
            "Give the equation of the vertical line that is the parabola's axis of symmetry, in the form x = ... .",
        ),
    ),
    "medium": (
        (
            "A parabola is graphed on a coordinate grid.",
            "The graph of a parabola is drawn below; no points are labelled.",
            "Here is a parabola drawn on a grid with no points labelled.",
            "A parabola has been drawn on the grid shown.",
        ),
        (
            "Find the equation of its axis of symmetry. Write it as x = ... .",
            "Its axis of symmetry is a vertical line. Write its equation as x = ... .",
            "State the equation of the axis of symmetry of this parabola as x = ... .",
        ),
    ),
    "hard": (
        (
            "The points P and Q are marked on the parabola in the graph.",
            "A parabola is graphed, with two points P and Q marked on the curve.",
            "Two points on the parabola, P and Q, are marked on the grid.",
            "P and Q are points on the parabola drawn on the grid.",
        ),
        (
            "Write the equation of the axis of symmetry of the parabola in the form x = ... .",
            "Give the axis of symmetry of the parabola as an equation x = ... .",
            "What is the equation of the parabola's axis of symmetry? Answer in the form x = ... .",
        ),
    ),
    "exam": (
        (
            "Three points P, Q and R are marked on the parabola shown.",
            "The parabola in the graph passes through the marked points P, Q and R.",
            "A parabola is graphed with three points on it marked P, Q and R.",
            "The marked points P, Q and R all lie on the parabola drawn on the grid.",
        ),
        (
            "Find the equation of the axis of symmetry of the parabola, giving it as x = ... .",
            "The parabola has a vertical axis of symmetry. State its equation in the form x = ... .",
            "Work out the equation of the axis of symmetry and write it as x = ... .",
        ),
    ),
}


_R2_GRID = (
    "Each grid square is 1 unit wide.",
    "The grid squares are 1 unit by 1 unit.",
    "Read the values from the grid.",
)


def _read_axis_text(fig: dict) -> str:
    """Re-derive ``x = b`` from the figure: the marked V (easy), the extreme sample
    (medium), or the equal-height pair of marked points confirmed by mirror symmetry of
    the sampled curve (hard, exam)."""
    pts = fig["points"]
    curve = fig["curves"][0]["pieces"]
    if any(p.get("label") == "V" for p in pts):
        x = next(p["at"][0] for p in pts if p.get("label") == "V")
        return _x_eq(int(x))
    if not pts:
        samples = [tuple(s) for piece in curve for s in piece]
        ys = [y for _x, y in samples]
        lo, hi = min(ys), max(ys)
        ends = (samples[0][1] + samples[-1][1]) / 2
        pick = lo if abs(ends - hi) < abs(ends - lo) else hi
        xs = [x for x, y in samples if y == pick]
        check(len(xs) == 1, "the extreme sample is not unique")
        return _x_eq(int(xs[0]))
    marked = [tuple(p["at"]) for p in pts]
    pairs = [(p, q) for i, p in enumerate(marked) for q in marked[i + 1 :] if p[1] == q[1]]
    check(len(pairs) == 1, "not exactly one equal-height pair of marked points")
    mid2 = pairs[0][0][0] + pairs[0][1][0]
    check(mid2 % 2 == 0, "midpoint is not an integer")
    mid = mid2 // 2
    index = {round(x * 8): y for piece in curve for x, y in piece}
    checked = 0
    for piece in curve:
        for x, y in piece:
            mx = round((2 * mid - x) * 8)
            if mx in index:
                check(abs(index[mx] - y) < 1e-6, "curve is not symmetric about the midpoint")
                checked += 1
    check(checked >= 10, "too few samples to confirm the symmetry")
    return _x_eq(mid)


def _read_axis(question) -> str:
    return _read_axis_text(question.figure)


figures.register_reader("pv1d_parabola_axis_x", _read_axis)


def _axis_vertex_form(rng, b: int, family: str):
    """(a, h, c) with vertex on the grid and inside the window, c not in {0, b, -b}."""
    cs = [c for c in range(-8, 9) if c not in (0, b, -b)]
    return (_pick(rng, (-2, -1, 1, 2)), b, _pick(rng, cs))


def _axis_root_form(rng, b: int):
    """y = a(x - r1)(x - r2) with r1, r2 = b -+ d the same parity by construction."""
    opts = [
        (a, d)
        for a in (-2, -1, 1, 2)
        for d in (1, 2, 3)
        if abs(a) * d * d <= 8 and abs(b) + d <= 9 and -a * d * d not in (b, -b, 0)
    ]
    if not opts:
        return None
    a, d = _pick(rng, opts)
    return a, b, -a * d * d


def _off_grid_options(rng, b: int):
    """(a, c, d, e, e_side) for hard/exam: vertex beyond the window, points inside it."""
    out = []
    for a in (-2, -1, 1, 2):
        s = 1 if a > 0 else -1
        for mag in range(12, 17):
            c = s * (-mag)  # a > 0: vertex below the window; a < 0: above
            for d in range(1, 9):
                y = a * d * d + c
                if abs(y) > 8 or abs(b) + d > 9:
                    continue
                out.append((a, c, d))
    return out


def _exam_options(b: int) -> list:
    """(a, c, d, e): P, Q at b -+ d (equal height) and R at b +- e (e != d, same parity as
    d so every pair midpoint is an integer), all marked points inside the window."""
    opts = []
    for a, c, d in _off_grid_options(None, b):
        for e in range(1, 9):
            if e != d and (e - d) % 2 == 0 and abs(b) + e <= 9 and abs(a * e * e + c) <= 8:
                opts.append((a, c, d, e))
    return opts


def _build_axis(rng, tier: str):
    if tier == "exam":
        bs = [b for b in _b_values(7) if _exam_options(b)]
    else:
        bs = _b_values(7)
    b = _pick(rng, bs)  # the key FIRST
    points = []
    if tier == "easy":
        a, h, c = _axis_vertex_form(rng, b, tier)
        curve = (a, h, c)
        points = [{"at": [b, c], "label": "V"}]
        pool = [-b, c, 0]
    elif tier == "medium":
        curve = _axis_root_form(rng, b) if rng.randrange(2) else None
        root_x = None
        if curve is None:
            curve = _axis_vertex_form(rng, b, tier)
            pool = [-b, curve[2], 0]
        else:
            a_, _h, c_ = curve
            d = isqrt(-c_ // a_) if (-c_ // a_) > 0 else 1
            root_x = b + d
            pool = [root_x, -b, c_, 0]
    elif tier == "hard":
        a, c, d = _pick(rng, _off_grid_options(rng, b))
        curve = (a, b, c)
        yp = a * d * d + c
        left = rng.randrange(2)
        pl, pr = ("P", "Q") if left else ("Q", "P")
        points = [{"at": [b - d, yp], "label": pl}, {"at": [b + d, yp], "label": pr}]
        one = _pick(rng, (b - d, b + d))
        rest = [one, 0, yp, (b + d) if one == b - d else (b - d)]
        rng.shuffle(rest)  # which three of the named misconceptions appear varies
        pool = [-b] + rest
    else:  # exam
        a, c, d, e = _pick(rng, _exam_options(b))
        sign = _pick(rng, (-1, 1))
        yp = a * d * d + c
        xr = b + sign * e
        locs = [(b - d, yp), (b + d, yp), (xr, a * e * e + c)]
        letters = list("PQR")
        rng.shuffle(letters)
        points = [{"at": [x, y], "label": ch} for (x, y), ch in zip(locs, letters)]
        curve = (a, b, c)
        pool = [(b - d + xr) // 2, (b + d + xr) // 2, -b, 0]
    a, h, c = curve
    check(_axis_from_coefficients(a, h, c) == b, "axis disagrees with the expanded coefficients")
    for p in points:
        x, y = p["at"]
        check(a * (x - h) ** 2 + c == y, "marked point is not on the curve")
        check(x == int(x) and y == int(y))
    fig = _plane(curves=[curve], alt="a parabola on a coordinate grid", points=points)
    return fig, b, pool


@generator("pv1d_parabola_axis_x", level=13, presets=_PRESETS)
def pv1d_parabola_axis_x(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, b, pool = _build_axis(rng, tier)
    answer_text = _x_eq(b)
    check(_read_axis_text(fig) == answer_text, "figure does not show the keyed axis")
    cands = [_x_eq(v) for v in pool]
    distractors = pick_distractors(answer_text, cands, n=3)
    check(len(distractors) >= 2, "too few distractors")
    check_pairwise_distinct(b, *[int(d.split("=")[1]) for d in distractors])
    figures.assert_no_leak(fig, [answer_text, *distractors])
    open_, ask = _R2_POOLS[tier]
    text = " ".join((_pick(rng, open_), _pick(rng, _R2_GRID), _pick(rng, ask)))
    return dict(
        text=text,
        answer={"x": b},
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[f"The axis of symmetry is the vertical line through the turning point, {answer_text}."],
        figure=fig,
    )
