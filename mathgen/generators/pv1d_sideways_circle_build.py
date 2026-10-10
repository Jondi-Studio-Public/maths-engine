"""- ``pv1d_parabola_axis_y`` (senior-recognise-features-of-the-graph-of-y-2, ACMMM021): a
  SIDEWAYS parabola (y - k)^2 = x (or its mirror image x = -(y - k)^2, opening left) is
  drawn and the learner types the equation of its axis of symmetry, ``y = k`` (``answer:
  pair``, a single-equation key: ``mathgen.grading._equation_pair_matches`` strips the label
  and compares right-hand sides, so every distractor keeps the label ``y =``).
- ``pv1d_circle_centre`` (senior-recognise-features-of-the-graphs-of-x-2, ACMMM020): one or
  two circles are drawn and the learner types the centre of one of them, ``(a, b)``
  (``answer: pair``, an unlabelled positional pair, exactly like ``quadratic-vertex``).

Both draw the ANSWER first (uniformly over a stated space) and then build the figure that
realises it; every value the key needs lies on a grid line or a grid intersection and
nothing the key needs is printed (only single capital letters label points). The curves
are not functions of x, so they are sampled here as parametric polylines and passed to
``figures.cartesian`` as ``curves`` pieces; the vertex, the lettered points, the four
extreme points of a circle and the places a curve meets the window edge are written
exactly, because the figure readers (registered below) find them from the sampled pieces.

Tier decisions (the learner's decision, never the size of a number):

``pv1d_parabola_axis_y``
  easy   -- read the marked vertex V; the axis is the horizontal line through it.
  medium -- no marked point: locate the turning point of the curve (opening right or left).
  hard   -- the vertex is off the grid: use two marked points that are mirror images
            (same x) and take the height midway between them.
  exam   -- three marked points, exactly one pair sharing an x: decide WHICH two are mirror
            images, then take their midpoint (the other two pairs are the distractors).

``pv1d_circle_centre``
  easy   -- the centre is marked C: read its coordinates.
  medium -- an unmarked circle: find the centre as the middle of its left/right and
            top/bottom extremes.
  hard   -- two disjoint circles with a lettered point on each: say which circle the named
            point is on, then find that circle's centre (the other centre is a distractor).
  exam   -- two circles, one of which touches the x-axis: decide which, then find its centre
            (the other centre and the point of contact are distractors).
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

from .. import figures
from ..core import check, check_pairwise_distinct, generator

# --------------------------------------------------------------------------- shared helpers

_TOL = 1e-6
_LETTER_PAIRS = (("P", "Q"), ("A", "B"), ("M", "N"), ("S", "T"), ("C", "D"), ("E", "F"), ("G", "H"), ("J", "K"))
_CIRCLE_LETTER_PAIRS = tuple(p for p in _LETTER_PAIRS if "C" not in p)
_LETTER_TRIPLES = (
    ("P", "Q", "R"),
    ("A", "B", "C"),
    ("L", "M", "N"),
    ("S", "T", "U"),
    ("D", "E", "F"),
    ("G", "H", "J"),
    ("K", "L", "M"),
)


def _clean(v) -> int | float:
    r = round(float(v), 6)
    return int(r) if r == int(r) else r


def _as_int(v: float, what: str) -> int:
    nearest = round(v)
    check(abs(v - nearest) < _TOL, f"{what} is not on a grid line: {v}")
    return int(nearest)


def _pieces_of(fig: dict) -> list[list[list[float]]]:
    return [piece for curve in fig["curves"] for piece in curve["pieces"]]


def _latex_with_math(text: str, snippet: str) -> str:
    """The prompt's LaTeX twin: prose stays in ``\\text{}``, and the one maths snippet
    (``y = k`` or ``(a, b)``) sits outside it so it renders as maths with no English in it."""
    check(text.count(snippet) == 1, f"expected exactly one {snippet!r} in the prompt")
    before, after = text.split(snippet)
    parts = [r"\text{" + before + "}", snippet]
    if after:
        parts.append(r"\text{" + after + "}")
    return "".join(parts)


def _phrase(rng, *banks) -> str:
    """One wording drawn independently from each bank and joined with a space."""
    return " ".join(rng.choice(bank) for bank in banks if bank).strip()


# =========================================================================================
# Row 3: pv1d_parabola_axis_y -- axis of symmetry y = k of a sideways parabola
# =========================================================================================

_K_VALUES = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)


def _parabola_pieces(k: int, sign: int, x_range, y_range) -> list[list[list[float]]]:
    """Pieces of ``x = sign * (y - k)^2`` inside the window.

    Sampled in ``u = y - k`` at steps of 0.1 (so the vertex, and every point whose offset
    from the axis is a whole number, is hit exactly), plus the exact places the curve meets
    the left and right window edges so an arm runs right up to the edge. Anything outside
    the window is dropped, which splits the curve into two arms when the vertex is off the
    grid.
    """
    (xl, xh), (yl, yh) = x_range, y_range
    us = {round(j / 10, 9) for j in range(int((yl - k) * 10), int((yh - k) * 10) + 1)}
    for edge in (xl, xh):
        t = sign * edge
        if t > 0:
            root = math.sqrt(t)
            for s in (-1, 1):
                if yl <= k + s * root <= yh:
                    us.add(round(s * root, 9))
    pieces, cur = [], []
    for u in sorted(us):
        x, y = sign * u * u, k + u
        if xl - 1e-6 <= x <= xh + 1e-6:
            cur.append([_clean(x), _clean(y)])
        else:
            if len(cur) > 1:
                pieces.append(cur)
            cur = []
    if len(cur) > 1:
        pieces.append(cur)
    return pieces


def _parabola_figure(k, sign, x_range, y_range, labelled, alt) -> dict:
    pieces = _parabola_pieces(k, sign, x_range, y_range)
    check(sum(len(p) for p in pieces) <= 200, "parabola sampled with more than 200 points")
    return figures.cartesian(
        x_range=x_range,
        y_range=y_range,
        grid=1,
        equal_scale=True,
        alt=alt,
        points=[{"at": [x, y], "label": label} for label, (x, y) in labelled],
        curves=[{"pieces": pieces}],
    )


# --- the figure reader --------------------------------------------------------------


def _vertex_height_from_curve(fig: dict) -> float:
    """Medium: the curve is a single piece through the vertex. Ordered by height, x is
    extreme (smallest for a right-opening curve, largest for a left-opening one) at the
    vertex, which sits strictly inside the piece, while the other extreme of x is at an
    arm's end."""
    pieces = _pieces_of(fig)
    if len(pieces) != 1:
        raise ValueError(f"expected one unbroken curve through the vertex, found {len(pieces)} pieces")
    pts = sorted(pieces[0], key=lambda p: p[1])
    xs = [p[0] for p in pts]
    interior = {i for i in (xs.index(min(xs)), xs.index(max(xs))) if 0 < i < len(pts) - 1}
    if len(interior) != 1:
        raise ValueError("could not find a single turning point strictly inside the curve")
    return pts[interior.pop()][1]


def _assert_symmetric_about(fig: dict, k: float) -> None:
    """Every sampled point whose mirror image in y = k lies inside the window has that
    mirror image on the curve too."""
    y_lo, y_hi = fig["y_range"]
    samples = {(round(p[0], 6), round(p[1], 6)) for piece in _pieces_of(fig) for p in piece}
    for x, y in samples:
        mirror = round(2 * k - y, 6)
        if y_lo - 1e-9 <= mirror <= y_hi + 1e-9 and (x, mirror) not in samples:
            raise ValueError(f"the curve is not symmetric about y = {k}: ({x}, {y}) has no mirror image")


def read_pv1d_parabola_axis_y(question) -> str:
    fig = question.figure
    labelled = {p["label"]: tuple(p["at"]) for p in fig["points"] if p.get("label")}
    if "V" in labelled:
        k = labelled["V"][1]
    elif not labelled:
        k = _vertex_height_from_curve(fig)
    else:
        by_x: dict[float, list[float]] = {}
        for x, y in labelled.values():
            by_x.setdefault(x, []).append(y)
        pairs = [ys for ys in by_x.values() if len(ys) == 2]
        if len(pairs) != 1:
            raise ValueError(f"expected exactly one pair of marked points sharing an x, found {len(pairs)}")
        k = (pairs[0][0] + pairs[0][1]) / 2
    _assert_symmetric_about(fig, k)
    return f"y = {_as_int(k, 'the axis of symmetry')}"


figures.register_reader("pv1d_parabola_axis_y", read_pv1d_parabola_axis_y)


# --- prompts ---------------------------------------------------------------------------

_P_EASY_OPEN = (
    "The graph shows a parabola that opens sideways, drawn on a coordinate grid. The point V is its vertex.",
    "A sideways parabola is drawn on the grid, with its vertex marked V.",
    "On the grid, V marks the vertex of a parabola that lies on its side.",
    "Look at the parabola on the grid. It opens along the x-axis, and V is its vertex.",
    "The parabola shown has its vertex at the point labelled V, and it opens sideways.",
)
_P_EASY_ASK = (
    "Give the equation of its axis of symmetry in the form y = k.",
    "Write the equation of the axis of symmetry as y = k.",
    "What is the equation of its axis of symmetry? Write it as y = k.",
    "State the equation of the line it is symmetric about, in the form y = k.",
    "Use V to write down its axis of symmetry as an equation of the form y = k.",
)
_P_MED_OPEN = (
    "The graph shows a parabola that opens sideways, drawn on a coordinate grid.",
    "A sideways parabola is drawn on the grid.",
    "The curve on the grid is a parabola lying on its side.",
    "Look at the parabola on the grid. It opens along the x-axis.",
    "This grid shows a parabola that opens to one side.",
)
_P_MED_ASK = (
    "Find the equation of its axis of symmetry, written as y = k.",
    "Locate the turning point, then write the equation of the axis of symmetry in the form y = k.",
    "What is the equation of the line it is symmetric about? Give it as y = k.",
    "State the axis of symmetry as an equation of the form y = k.",
    "Work out where the curve turns and give the equation of its axis of symmetry as y = k.",
)
_P_TAIL = ("", " Read what you need from the grid.")
_P_HARD_OPEN = (
    "The graph shows part of a parabola that opens sideways. Its vertex is off the grid, but the points {a} and {b} are marked on the curve.",
    "Only part of a sideways parabola is drawn on the grid. The points {a} and {b} lie on it.",
    "A sideways parabola passes through the points {a} and {b} marked on the grid, though its vertex is not shown.",
    "On the grid, {a} and {b} are two points on a parabola that opens sideways. The vertex is outside the grid.",
)
_P_HARD_ASK = (
    "Find the equation of the parabola's axis of symmetry, in the form y = k.",
    "Use the points to write the equation of its axis of symmetry as y = k.",
    "Give the equation of the line the whole parabola is symmetric about, in the form y = k.",
    "What is its axis of symmetry? Write the equation as y = k.",
)
_P_EXAM_OPEN = (
    "The graph shows part of a sideways parabola with three marked points, {a}, {b} and {c}. The vertex is off the grid.",
    "Three points {a}, {b} and {c} are marked on a parabola that opens sideways. Only part of the curve is drawn.",
    "A sideways parabola passes through {a}, {b} and {c} on the grid, and its vertex is not shown.",
)
_P_EXAM_MID = (
    "Exactly two of the three points are mirror images of each other in the axis of symmetry.",
    "Two of the points are reflections of each other across the axis of symmetry.",
    "One pair of the marked points are mirror images across the parabola's axis.",
)
_P_EXAM_ASK = (
    "Give the equation of the axis of symmetry in the form y = k.",
    "Write the axis of symmetry as y = k.",
    "State the equation of the axis in the form y = k, using the two mirror-image points.",
)

_P_ALT = {
    "easy": "a sideways parabola on a coordinate grid with its vertex marked",
    "medium": "a sideways parabola on a coordinate grid",
    "hard": "part of a sideways parabola on a coordinate grid with two marked points",
    "exam": "part of a sideways parabola on a coordinate grid with three marked points",
}


# --- distractors ------------------------------------------------------------------------


def _axis_distractors(candidates, k: int, need: int = 3):
    """First ``need`` distinct integer candidates that differ from the key, or None."""
    out: list[int] = []
    for c in candidates:
        if c != k and c not in out:
            out.append(c)
    return out[:need] if len(out) >= need else None


def _off_by_one(rng, k: int) -> list[int]:
    first = rng.choice((1, -1))
    return [k + first, k - first]


def _finish_axis(rng, tier, text, k, distractors, fig, working):
    answer_text = f"y = {k}"
    distractor_texts = [f"y = {v}" for v in distractors]
    check_pairwise_distinct(k, *distractors, msg="axis distractors collide with each other or the key")
    figures.assert_no_leak(fig, [answer_text, *distractor_texts])
    reread = read_pv1d_parabola_axis_y(SimpleNamespace(figure=fig, text=text))
    check(reread == answer_text, f"parabola figure reader read {reread!r}, key is {answer_text!r}")
    return dict(
        text=text,
        latex=_latex_with_math(text, "y = k"),
        answer={"y": k},
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractor_texts,
        working=working,
        figure=fig,
    )


# --- the four tiers ---------------------------------------------------------------------


def _pick_sign(rng, allow_left: bool) -> int:
    return rng.choice((1, -1)) if allow_left else 1


def _window_with_vertex(rng, k, sign, height=9, width=9):
    """Window that contains the vertex, the y-axis and the x-axis."""
    if sign > 0:
        x_lo = -rng.randint(1, 3)
        x_range = (x_lo, x_lo + width)
    else:
        x_hi = rng.randint(1, 3)
        x_range = (x_hi - width, x_hi)
    lo, hi = max(-height, k + 2 - height), min(0, k - 2)
    check(lo <= hi, "no window holds both axes and the vertex")
    y_lo = rng.randint(lo, hi)
    return x_range, (y_lo, y_lo + height)


def _parabola_easy(rng, allow_left: bool):
    for _ in range(50):
        k = rng.choice(_K_VALUES)
        sign = _pick_sign(rng, allow_left)
        x_range, y_range = _window_with_vertex(rng, k, sign)
        dist = _axis_distractors([-k, 0, *_off_by_one(rng, k)], k)
        if dist is None:
            continue
        fig = _parabola_figure(k, sign, x_range, y_range, [("V", (0, k))], _P_ALT["easy"])
        text = _phrase(rng, _P_EASY_OPEN, _P_EASY_ASK) + rng.choice(_P_TAIL)
        return _finish_axis(
            rng,
            "easy",
            text,
            k,
            dist,
            fig,
            [f"V is the vertex, at height y = {k}; the axis of symmetry is the horizontal line through it."],
        )
    check(False, "no easy parabola draw met the distractor rule")


def _parabola_medium(rng):
    for _ in range(50):
        k = rng.choice(_K_VALUES)
        sign = _pick_sign(rng, True)
        x_range, y_range = _window_with_vertex(rng, k, sign)
        dist = _axis_distractors([-k, 0, *_off_by_one(rng, k)], k)
        if dist is None:
            continue
        fig = _parabola_figure(k, sign, x_range, y_range, [], _P_ALT["medium"])
        text = _phrase(rng, _P_MED_OPEN, _P_MED_ASK) + rng.choice(_P_TAIL)
        return _finish_axis(
            rng, "medium", text, k, dist, fig, [f"The curve turns at (0, {k}), so its axis of symmetry is y = {k}."]
        )
    check(False, "no medium parabola draw met the distractor rule")


def _parabola_hard(rng):
    for _ in range(80):
        k = rng.choice(_K_VALUES)
        sign = _pick_sign(rng, True)
        m = rng.choice((2, 3))
        near = rng.randint(1, 2)
        x_range = (near, near + 9) if sign > 0 else (-near - 9, -near)
        height = 11
        y_lo = rng.randint(k + m + 1 - height, k - m - 1)
        y_range = (y_lo, y_lo + height)
        la, lb = rng.choice(_LETTER_PAIRS)
        up_first = rng.random() < 0.5
        up, down = (la, lb) if up_first else (lb, la)
        px = sign * m * m
        labelled = [(up, (px, k + m)), (down, (px, k - m))]
        in_window_zero = [0] if y_range[0] <= 0 <= y_range[1] else []
        pool = [k + m, k - m, -k, *in_window_zero, *_off_by_one(rng, k)]
        rng.shuffle(pool)
        dist = _axis_distractors(pool, k)
        if dist is None:
            continue
        fig = _parabola_figure(k, sign, x_range, y_range, labelled, _P_ALT["hard"])
        text = _phrase(rng, _P_HARD_OPEN).format(a=la, b=lb) + " " + rng.choice(_P_HARD_ASK)
        return _finish_axis(
            rng,
            "hard",
            text,
            k,
            dist,
            fig,
            [
                f"{la} and {lb} have the same x, so they are mirror images; "
                f"the axis is midway between heights {k + m} and {k - m}: y = {k}."
            ],
        )
    check(False, "no hard parabola draw met the distractor rule")


def _parabola_exam(rng):
    for _ in range(120):
        k = rng.choice(_K_VALUES)
        sign = _pick_sign(rng, True)
        m = rng.choice((2, 4))
        n = 6 - m  # the odd point's offset: 4 when m = 2, 2 when m = 4
        tau = rng.choice((1, -1))
        near, far = rng.randint(2, 3), rng.randint(17, 18)
        x_range = (near, far) if sign > 0 else (-far, -near)
        height = 13
        y_lo = rng.randint(k - 8, k - 5)
        y_range = (y_lo, y_lo + height)
        letters = list(rng.choice(_LETTER_TRIPLES))
        shuffled = letters[:]
        rng.shuffle(shuffled)
        up, down, odd = shuffled
        labelled = [(up, (sign * m * m, k + m)), (down, (sign * m * m, k - m)), (odd, (sign * n * n, k + tau * n))]
        wrong_up = k + (m + tau * n) // 2  # midpoint of (up, odd)
        wrong_down = k + (-m + tau * n) // 2  # midpoint of (down, odd)
        check((m + tau * n) % 2 == 0 and (-m + tau * n) % 2 == 0, "wrong-pair midpoints are not whole numbers")
        dist = _axis_distractors([wrong_up, wrong_down, -k], k)
        if dist is None or set(dist) != {wrong_up, wrong_down, -k}:
            continue
        fig = _parabola_figure(k, sign, x_range, y_range, labelled, _P_ALT["exam"])
        text = (
            _phrase(rng, _P_EXAM_OPEN).format(a=letters[0], b=letters[1], c=letters[2])
            + " "
            + rng.choice(_P_EXAM_MID)
            + " "
            + rng.choice(_P_EXAM_ASK)
        )
        return _finish_axis(
            rng,
            "exam",
            text,
            k,
            dist,
            fig,
            [
                f"{up} and {down} share an x, so they are the mirror images; "
                f"the axis is midway between heights {k + m} and {k - m}: y = {k}."
            ],
        )
    check(False, "no exam parabola draw met the distractor rule")


_PARABOLA_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}


@generator("pv1d_parabola_axis_y", level=12, presets=_PARABOLA_PRESETS)
def pv1d_parabola_axis_y(rng, variant="easy"):
    if variant == "easy":
        return _parabola_easy(rng, allow_left=False)
    if variant == "medium":
        return _parabola_medium(rng)
    if variant == "hard":
        return _parabola_hard(rng)
    if variant == "exam":
        return _parabola_exam(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# =========================================================================================
# Row 4: pv1d_circle_centre -- centre (a, b) of a circle on an integer grid
# =========================================================================================

_NONZERO = (-4, -3, -2, -1, 1, 2, 3, 4)
_RADII = (1, 2, 3)
_N_STEPS = 192  # 4 quarters of 48 steps; with the closing point 193 <= 200


def _draw_centre(rng) -> tuple[int, int]:
    """Centre drawn first: 5% the origin (x^2 + y^2 = r^2), 10% on exactly one axis, 85%
    off both axes with a != b (so the swapped pair is always a different pair)."""
    u = rng.random()
    if u < 0.05:
        return 0, 0
    if u < 0.15:
        v = rng.choice(_NONZERO)
        return (0, v) if rng.random() < 0.5 else (v, 0)
    for _ in range(100):
        a, b = rng.choice(_NONZERO), rng.choice(_NONZERO)
        if a != b:
            return a, b
    check(False, "could not draw an off-axis centre with a != b")


def _circle_points(a: int, b: int, r: int) -> list[list[float]]:
    exact = ([a + r, b], [a, b + r], [a - r, b], [a, b - r])
    pts = []
    for i in range(_N_STEPS):
        if i % (_N_STEPS // 4) == 0:
            pts.append(list(exact[i // (_N_STEPS // 4)]))
        else:
            t = 2 * math.pi * i / _N_STEPS
            pts.append([_clean(a + r * math.cos(t)), _clean(b + r * math.sin(t))])
    pts.append(list(pts[0]))
    return pts


def _circle_figure(circles, x_range, y_range, labelled, alt) -> dict:
    curves = [{"pieces": [_circle_points(a, b, r)]} for (a, b, r) in circles]
    return figures.cartesian(
        x_range=x_range,
        y_range=y_range,
        grid=1,
        equal_scale=True,
        alt=alt,
        points=[{"at": list(at), "label": label} for label, at in labelled],
        curves=curves,
    )


# --- the figure reader --------------------------------------------------------------


def _bounding_centre(piece) -> tuple[int, int, float]:
    xs, ys = [p[0] for p in piece], [p[1] for p in piece]
    a = _as_int((min(xs) + max(xs)) / 2, "the centre's x")
    b = _as_int((min(ys) + max(ys)) / 2, "the centre's y")
    return a, b, (max(xs) - min(xs)) / 2


_ASKED_LETTER_RE = re.compile(r"through (?:the point (?:marked )?)?([A-Z])\b")


def read_pv1d_circle_centre(question) -> str:
    fig = question.figure
    pieces = _pieces_of(fig)
    labelled = {p["label"]: tuple(p["at"]) for p in fig["points"] if p.get("label")}
    if len(pieces) == 1:  # easy / medium: the only circle
        a, b, _r = _bounding_centre(pieces[0])
        if "C" in labelled and labelled["C"] != (a, b):
            raise ValueError(f"the marked point C {labelled['C']} is not the middle of the circle ({a}, {b})")
    else:
        letters = _ASKED_LETTER_RE.findall(question.text or "")
        if letters:  # hard: the circle through a named point
            at = labelled[letters[-1]]
            owners = [pc for pc in pieces if any(abs(p[0] - at[0]) < _TOL and abs(p[1] - at[1]) < _TOL for p in pc)]
            if len(owners) != 1:
                raise ValueError(f"point {letters[-1]} lies on {len(owners)} circles, expected one")
            a, b, _r = _bounding_centre(owners[0])
        else:  # exam: the circle whose bottom or top meets y = 0
            touching = [pc for pc in pieces if abs(min(p[1] for p in pc)) < _TOL or abs(max(p[1] for p in pc)) < _TOL]
            if len(touching) != 1:
                raise ValueError(f"{len(touching)} circles touch the x-axis, expected exactly one")
            a, b, _r = _bounding_centre(touching[0])
    return f"({a}, {b})"


figures.register_reader("pv1d_circle_centre", read_pv1d_circle_centre)


# --- prompts ---------------------------------------------------------------------------

_C_EASY_OPEN = (
    "The graph shows a circle on a coordinate grid. The point C marks its centre.",
    "A circle is drawn on the grid, with its centre marked C.",
    "On the grid, C is the centre of the circle shown.",
    "Look at the circle on the grid. Its centre is the point labelled C.",
)
_C_EASY_ASK = (
    "Give the coordinates of the centre as (a, b).",
    "Write down the coordinates of C in the form (a, b).",
    "What are the coordinates of the centre? Answer as (a, b).",
    "State the centre of the circle as a coordinate pair (a, b).",
    "Read the position of C and give it as (a, b).",
)
_C_MED_OPEN = (
    "The graph shows a circle drawn on a coordinate grid.",
    "A circle is drawn on the grid, but its centre is not marked.",
    "Look at the circle on the grid.",
    "This grid shows a single circle.",
    "The circle on the grid has no marked centre.",
)
_C_MED_ASK = (
    "Find the coordinates of its centre and give them as (a, b).",
    "Work out where the centre is and write its coordinates as (a, b).",
    "What are the coordinates of the centre? Give your answer as (a, b).",
    "State the centre of the circle as a coordinate pair (a, b).",
    "Determine the centre of the circle and write it in the form (a, b).",
)
_C_TAIL = ("", " Read what you need from the grid.")
_C_HARD_OPEN = (
    "The graph shows two circles. The point {x} is marked on one circle and the point {y} on the other.",
    "Two separate circles are drawn on the grid, with {x} on one of them and {y} on the other.",
    "On the grid, the points {x} and {y} each lie on a different circle.",
)
_C_HARD_ASK = (
    "Give the centre of the circle through the point marked {L} as (a, b).",
    "Write the coordinates of the centre of the circle that passes through {L}, in the form (a, b).",
    "What are the coordinates of the centre of the circle through {L}? Answer as (a, b).",
    "State the centre of the circle through the point marked {L} as a coordinate pair (a, b).",
)
_C_EXAM_OPEN = (
    "Two circles are drawn on the grid. One of them touches the x-axis and the other does not. A circle touches the axis when it meets it at exactly one point.",
    "The graph shows two separate circles, and exactly one of them touches the x-axis. A circle touches the axis when it meets it at exactly one point.",
    "Two separate circles are shown on the coordinate grid. Only one of them touches the x-axis. A circle touches the axis when it meets it at exactly one point.",
    "Of the two circles drawn on the grid, one just touches the x-axis. A circle touches the axis when it meets it at exactly one point.",
    "Look at the two circles on the grid. Exactly one of them touches the x-axis. A circle touches the axis when it meets it at exactly one point.",
    "Two circles sit on the grid, and only one of them touches the x-axis. A circle touches the axis when it meets it at exactly one point.",
)
_C_EXAM_ASK = (
    "Give the coordinates of the centre of the circle that touches the x-axis, as (a, b).",
    "Write the centre of the circle that touches the x-axis in the form (a, b).",
    "What are the coordinates of the centre of the circle that just touches the x-axis? Answer as (a, b).",
    "State, as a coordinate pair (a, b), the centre of the circle that touches the x-axis.",
)

_C_ALT = {
    "easy": "a circle on a coordinate grid with its centre marked",
    "medium": "a circle on a coordinate grid",
    "hard": "two circles on a coordinate grid, each with a marked point on it",
    "exam": "two circles on a coordinate grid",
}


# --- distractors ------------------------------------------------------------------------


def _pair_text(p) -> str:
    return f"({p[0]}, {p[1]})"


def _centre_distractors(candidates, key, need=3):
    out = []
    for c in candidates:
        if c != key and c not in out:
            out.append(c)
    return out[:need] if len(out) >= need else None


def _finish_circle(text, key, distractors, fig, working):
    answer_text = _pair_text(key)
    distractor_texts = [_pair_text(d) for d in distractors]
    check_pairwise_distinct(answer_text, *distractor_texts, msg="centre distractors collide with each other or the key")
    figures.assert_no_leak(fig, [answer_text, *distractor_texts])
    reread = read_pv1d_circle_centre(SimpleNamespace(figure=fig, text=text))
    check(reread == answer_text, f"circle figure reader read {reread!r}, key is {answer_text!r}")
    return dict(
        text=text,
        latex=_latex_with_math(text, "(a, b)"),
        answer={"x": key[0], "y": key[1]},
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractor_texts,
        working=working,
        figure=fig,
    )


def _single_circle_distractors(rng, a, b, r):
    named = [(b, a), (-a, b), (a, -b), (a + r, b), (a, b + r)]
    extra = [(a - r, b), (a, b - r)]
    rng.shuffle(named)
    return named + extra


def _single_window(rng, a, b, r, size=9):
    x0 = rng.randint(a + r + 1 - size, a - r - 1)
    y0 = rng.randint(b + r + 1 - size, b - r - 1)
    return (x0, x0 + size), (y0, y0 + size)


def _circle_single(rng, marked: bool):
    for _ in range(50):
        a, b = _draw_centre(rng)
        r = rng.choice(_RADII)
        dist = _centre_distractors(_single_circle_distractors(rng, a, b, r), (a, b))
        if dist is None:
            continue
        x_range, y_range = _single_window(rng, a, b, r)
        labelled = [("C", (a, b))] if marked else []
        fig = _circle_figure([(a, b, r)], x_range, y_range, labelled, _C_ALT["easy" if marked else "medium"])
        if marked:
            text = _phrase(rng, _C_EASY_OPEN, _C_EASY_ASK)
        else:
            text = _phrase(rng, _C_MED_OPEN, _C_MED_ASK)
        text += rng.choice(_C_TAIL)
        work = (
            f"C is the centre, at ({a}, {b})."
            if marked
            else f"The circle runs from x = {a - r} to x = {a + r} and from y = {b - r} to y = {b + r}, "
            f"so its centre is ({a}, {b})."
        )
        return _finish_circle(text, (a, b), dist, fig, [work])
    check(False, "no single-circle draw met the distractor rule")


def _bbox(circles, margin_rng):
    x0 = min(a - r for a, b, r in circles) - margin_rng.randint(1, 2)
    x1 = max(a + r for a, b, r in circles) + margin_rng.randint(1, 2)
    y0 = min(b - r for a, b, r in circles) - margin_rng.randint(1, 2)
    y1 = max(b + r for a, b, r in circles) + margin_rng.randint(1, 2)
    return (x0, x1), (y0, y1)


def _disjoint(c1, c2) -> bool:
    (a, b, r), (c, d, s) = c1, c2
    return (a - c) ** 2 + (b - d) ** 2 >= (r + s + 1) ** 2


_EXTREMES = (
    lambda a, b, r: (a + r, b),
    lambda a, b, r: (a, b + r),
    lambda a, b, r: (a - r, b),
    lambda a, b, r: (a, b - r),
)


def _circle_hard(rng):
    for _ in range(300):
        a, b = _draw_centre(rng)
        r = rng.choice(_RADII)
        s = rng.choice(_RADII)
        dx, dy = rng.randint(-8, 8), rng.randint(-6, 6)
        other = (a + dx, b + dy, s)
        mine = (a, b, r)
        if not _disjoint(mine, other):
            continue
        x_range, y_range = _bbox([mine, other], rng)
        if x_range[1] - x_range[0] > 12 or y_range[1] - y_range[0] > 11:
            continue
        p_at = rng.choice(_EXTREMES)(a, b, r)
        q_at = rng.choice(_EXTREMES)(other[0], other[1], s)
        # never "C": the easy tier teaches that C marks a centre, and here a letter marks a
        # point on the circumference
        letters = list(rng.choice(_CIRCLE_LETTER_PAIRS))
        rng.shuffle(letters)
        ask, rest = letters
        pool = [(b, a), (-a, b), (a, -b), p_at, (a + r, b), (a, b + r)]
        rng.shuffle(pool)
        other_centre = (other[0], other[1])
        dist = _centre_distractors([other_centre, *pool], (a, b))
        if dist is None or other_centre not in dist:
            continue
        labelled = [(ask, p_at), (rest, q_at)]
        circles = [mine, other]
        if rng.random() < 0.5:
            circles.reverse()
        fig = _circle_figure(circles, x_range, y_range, labelled, _C_ALT["hard"])
        x_l, y_l = sorted((ask, rest))
        text = rng.choice(_C_HARD_OPEN).format(x=x_l, y=y_l) + " " + rng.choice(_C_HARD_ASK).format(L=ask)
        return _finish_circle(
            text,
            (a, b),
            dist,
            fig,
            [
                f"{ask} lies on the circle with centre ({a}, {b}); the other circle's centre is "
                f"({other[0]}, {other[1]})."
            ],
        )
    check(False, "no hard circle draw met the placement rules")


def _circle_exam(rng):
    for _ in range(400):
        r = rng.choice(_RADII)
        b = r * rng.choice((1, -1))  # tangent to the x-axis at (a, 0)
        # a != b always, so the swapped pair (b, a) is a real, different distractor
        a = 0 if rng.random() < 0.1 else rng.choice([v for v in _NONZERO if v != b])
        s = rng.choice(_RADII)
        if rng.random() < 0.5:  # the other crosses the axis at whole numbers
            d = 0
        else:  # or misses it altogether
            d = rng.choice((1, -1)) * rng.randint(s + 1, s + 3)
        c = a + rng.randint(-9, 9)
        mine, other = (a, b, r), (c, d, s)
        if not _disjoint(mine, other):
            continue
        x_range, y_range = _bbox([mine, other], rng)
        if x_range[1] - x_range[0] > 13 or y_range[1] - y_range[0] > 11:
            continue
        check(y_range[0] < 0 < y_range[1], "the x-axis is not inside the window")
        tangent_point = (a, 0)
        other_centre = (c, d)
        tail = [(b, a), (a, -b), (-a, b)]
        rng.shuffle(tail)
        dist = _centre_distractors([other_centre, tangent_point, *tail], (a, b))
        if dist is None or other_centre not in dist or tangent_point not in dist:
            continue
        circles = [mine, other]
        if rng.random() < 0.5:
            circles.reverse()
        fig = _circle_figure(circles, x_range, y_range, [], _C_ALT["exam"])
        text = rng.choice(_C_EXAM_OPEN) + " " + rng.choice(_C_EXAM_ASK) + rng.choice(_C_TAIL)
        return _finish_circle(
            text,
            (a, b),
            dist,
            fig,
            [
                f"The circle with centre ({a}, {b}) has radius {r} and its centre is {r} "
                f"from the x-axis, so it touches the axis at ({a}, 0)."
            ],
        )
    check(False, "no exam circle draw met the placement rules")


_CIRCLE_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}


@generator("pv1d_circle_centre", level=12, presets=_CIRCLE_PRESETS)
def pv1d_circle_centre(rng, variant="easy"):
    if variant == "easy":
        return _circle_single(rng, marked=True)
    if variant == "medium":
        return _circle_single(rng, marked=False)
    if variant == "hard":
        return _circle_hard(rng)
    if variant == "exam":
        return _circle_exam(rng)
    raise ValueError(f"unknown variant: {variant!r}")
