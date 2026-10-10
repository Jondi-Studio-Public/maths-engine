"""Generates point-of-intersection questions (ACMEM126): read where two drawn lines meet, answered as a pair.

One skill: ``senior-interpret-the-point-of-intersection-and-other-important``
(ACMEM126), ``domain: functions-sequences``, ``answer: pair``, answer text
``(x, y)`` with both parts integers (graded by the real ``pair`` grader). Two
lines labelled A and B (letters only) are drawn on one grid. Each line is drawn
through two grid intersections at least two columns apart, the crossing is a
grid intersection but NEVER one of those four through points, and no equation,
coordinate or rate is printed anywhere.

Four tiers, escalating by what the learner DECIDES (not by reading precision):

- ``easy``: the crossing is in quadrant I with both axes drawn, equal scales;
  the decision is finding where the lines share a point.
- ``medium``: the crossing has a negative or zero coordinate, or one line is
  horizontal; the decision is reading signs (and a constant y) correctly.
- ``hard``: unequal axis scales (2 or 5 real units per grid step on exactly one
  axis); the decision is carrying the axis scale into the coordinates instead
  of counting squares.
- ``exam``: a context (two hire, call-out, phone, saving or printing plans)
  with axis labels and units; the prompt asks where the two plans agree and
  names the order of the pair; the decision is mapping each coordinate to the
  quantity on its own axis.

Distractors are named misconceptions: swapped (y, x); the y-intercept of one
line; the x-intercept of one line; a neighbouring grid point; (extras) a point
a line passes through, and the square count when the scales are not 1.
Every one is an integer pair different from the key, so the exact integer
pair grader cannot accept it.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen import figures
from mathgen.core import (
    check,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
    rational_text,
)
from mathgen.generators.pv1a_line_figure_build import (
    Window,
    lattice_points,
    line_figure,
    pick_two_points,
    read_intersection,
    read_line,
)

_ATTEMPTS = 3000
_ALT = "two straight lines on a coordinate grid"
_MIN_ANGLE = 18.0  # degrees, in grid units; the minimum required is 15

_F = Fraction
_EASY_SLOPES = tuple(_F(s) for s in (-3, -2, -1, 1, 2, 3))
_MED_SLOPES = _EASY_SLOPES + tuple(_F(n, 2) for n in (-3, -1, 1, 3))
_EXAM_SLOPES = (_F(1, 3), _F(1, 2), _F(2, 3), _F(1), _F(3, 2))
_HARD_SCALES = ((1, 2), (1, 5), (2, 1), (5, 1))

# context: story (what the lines are), the quantity asked for, axis labels,
# the real x step options and y step options (dollars per grid line).
_CONTEXTS = (
    dict(
        story=(
            "Two kayak hire shops, A and B, each charge a total cost that depends on the time hired.",
            "Kayak hire shops A and B charge different prices, and each charge depends on the time hired.",
        ),
        graph="The graph shows each shop's total cost, in dollars, against the time hired, in hours.",
        ask=(
            "For how many hours do the two shops charge the same total, and what is that total? "
            "Give your answer as (hours, dollars).",
            "The two shops charge the same amount at one point. Write that point as (hours, dollars).",
        ),
        x_label="Time hired (hours)",
        y_label="Total cost (dollars)",
        xs=(1, 2),
        ys=(5, 10, 20),
    ),
    dict(
        story=(
            "Two gyms, A and B, charge a total cost that depends on the number of classes attended.",
            "Gym A and gym B have different pricing, and the total depends on the classes attended.",
        ),
        graph="The graph shows each gym's total cost, in dollars, against the number of classes attended.",
        ask=(
            "After how many classes do the two gyms cost the same, and what is that cost? "
            "Give your answer as (classes, dollars).",
            "The two gyms cost the same at one point. Write that point as (classes, dollars).",
        ),
        x_label="Classes attended",
        y_label="Total cost (dollars)",
        xs=(1, 2),
        ys=(10, 20, 25),
    ),
    dict(
        story=(
            "Two phone plans, A and B, charge a monthly cost that depends on the data used.",
            "Plan A and plan B are two phone plans whose monthly cost depends on the data used.",
        ),
        graph="The graph shows each plan's monthly cost, in dollars, against the data used, in gigabytes.",
        ask=(
            "For how many gigabytes do the two plans cost the same, and what is that cost? "
            "Give your answer as (gigabytes, dollars).",
            "The two plans cost the same at one point. Write that point as (gigabytes, dollars).",
        ),
        x_label="Data used (gigabytes)",
        y_label="Monthly cost (dollars)",
        xs=(1, 2, 5),
        ys=(10, 20, 25),
    ),
    dict(
        story=(
            "Two savings plans, A and B, start with different balances and grow at different rates.",
            "Savings plan A and savings plan B start from different balances and are topped up each week.",
        ),
        graph="The graph shows each plan's balance, in dollars, against the number of weeks.",
        ask=(
            "After how many weeks are the two balances equal, and what is that balance? "
            "Give your answer as (weeks, dollars).",
            "The two plans have the same balance at one point. Write that point as (weeks, dollars).",
        ),
        x_label="Time (weeks)",
        y_label="Balance (dollars)",
        xs=(1, 2, 5),
        ys=(10, 20, 50),
    ),
    dict(
        story=(
            "Two print shops, A and B, quote a total cost that depends on the number of posters printed.",
            "Print shop A and print shop B each charge a total that depends on the number of posters.",
        ),
        graph="The graph shows each shop's total cost, in dollars, against the number of posters printed.",
        ask=(
            "For how many posters do the two shops quote the same total, and what is that total? "
            "Give your answer as (posters, dollars).",
            "The two shops quote the same total at one point. Write that point as (posters, dollars).",
        ),
        x_label="Posters printed",
        y_label="Total cost (dollars)",
        xs=(10, 20, 50),
        ys=(10, 20, 50),
    ),
    dict(
        story=(
            "Two taxi companies, A and B, charge a fare that depends on the distance travelled.",
            "Taxi company A and taxi company B charge a fare that depends on the distance travelled.",
        ),
        graph="The graph shows each company's fare, in dollars, against the distance travelled, in kilometres.",
        ask=(
            "At what distance do the two companies charge the same fare, and what is that fare? "
            "Give your answer as (kilometres, dollars).",
            "The two companies charge the same fare at one point. Write that point as (kilometres, dollars).",
        ),
        x_label="Distance travelled (kilometres)",
        y_label="Fare (dollars)",
        xs=(1, 2, 5),
        ys=(5, 10, 20),
    ),
)

_OPEN = [
    "The graph shows two straight lines, A and B, drawn on a coordinate grid.",
    "Lines A and B are drawn on the coordinate grid below.",
    "Two straight lines, labelled A and B, are graphed on this grid.",
    "The diagram shows line A and line B on a coordinate grid.",
    "Look at lines A and B graphed on this coordinate grid.",
    "A coordinate grid carries two straight lines, A and B.",
]
_ASK = [
    "State the coordinates of the point where the two lines meet, as (x, y).",
    "Where do lines A and B cross? Give your answer as the ordered pair (x, y).",
    "Find the point of intersection of the two lines and write it as (x, y).",
    "Read off the coordinates (x, y) of the point where line A meets line B.",
    "What are the coordinates (x, y) of the point shared by both lines?",
    "Write down the ordered pair (x, y) at which the lines cross.",
]
_CONV = {
    "easy": ["Both coordinates are whole numbers.", "Give both coordinates as whole numbers."],
    "medium": ["The coordinates may be negative or zero.", "A coordinate can be negative or zero here."],
    "hard": [
        "The two axes use different scales, so give the coordinates in the units shown on the axes.",
        "One grid step is not one unit on both axes; use the numbers on the axes for the coordinates.",
    ],
}


def _angle(m1: Fraction, m2: Fraction) -> float:
    return abs(math.degrees(math.atan(float(m1)) - math.atan(float(m2))))


def _label_anchor(win: Window, pts):
    """Where the renderer puts a line's label: the second end of the line
    clipped to the plane, in real units."""
    (x1, y1), (x2, y2) = [(_F(p[0] * win.xs), _F(p[1] * win.ys)) for p in pts]
    dx, dy = x2 - x1, y2 - y1
    (xmin, xmax), (ymin, ymax) = win.x_range, win.y_range
    t_hi = None
    for p, q in ((dx, xmax - x1), (-dx, x1 - xmin), (dy, ymax - y1), (-dy, y1 - ymin)):
        if p > 0:
            t_hi = q / p if t_hi is None else min(t_hi, q / p)
    return x1 + t_hi * dx, y1 + t_hi * dy


def _labels_readable(win: Window, lines) -> bool:
    """The A and B labels are both visible, apart from each other, and neither
    line lies along an axis."""
    (xmin, xmax), (ymin, ymax) = win.x_range, win.y_range
    xs_, ys_ = _F(xmax - xmin), _F(ymax - ymin)
    anchors = []
    for _, pts in lines:
        (x1, y1), (x2, y2) = [(p[0] * win.xs, p[1] * win.ys) for p in pts]
        if (y1 == y2 == 0) or (x1 == x2 == 0):
            return False
        bx, by = _label_anchor(win, pts)
        if (ymax - by) / ys_ < _F(12, 100) or (bx - xmin) / xs_ < _F(15, 100):
            return False
        anchors.append((bx, by))
    (ax, ay), (bx, by) = anchors
    return abs(ax - bx) / xs_ >= _F(12, 100) or abs(ay - by) / ys_ >= _F(12, 100)


def _window(rng, tier: str, ctx=None):
    nu, nv = rng.randint(8, 10), rng.randint(8, 10)
    xs = ys = 1
    if tier == "easy":
        u0, v0 = -rng.randint(0, 3), -rng.randint(0, 3)
    elif tier == "medium":
        u0, v0 = -rng.randint(2, nu - 3), -rng.randint(2, nv - 3)
    elif tier == "hard":
        u0, v0 = -rng.randint(1, nu - 2), -rng.randint(1, nv - 2)
        xs, ys = rng.choice(_HARD_SCALES)
    else:
        u0 = v0 = 0
        xs, ys = rng.choice(ctx["xs"]), rng.choice(ctx["ys"])
        if xs == ys:
            return None
    return Window(u0, u0 + nu - 1, v0, v0 + nv - 1, xs=xs, ys=ys)


def _draw(rng, tier: str):
    """Sample a window, a crossing and two lines. Returns the pieces."""
    slopes = {"easy": _EASY_SLOPES, "medium": _MED_SLOPES, "hard": _MED_SLOPES, "exam": _EXAM_SLOPES}[tier]
    for _ in range(_ATTEMPTS):
        ctx = rng.choice(_CONTEXTS) if tier == "exam" else None
        win = _window(rng, tier, ctx)
        if win is None:
            continue
        if tier == "easy":
            pu, pv = rng.randint(1, win.u1 - 1), rng.randint(1, win.v1 - 1)
        else:
            pv_hi = (
                win.v0 + max(2, (win.v1 - win.v0) // 2) if tier == "exam" else win.v1 - 1
            )  # exam: low crossing, so both lines end on the right edge
            pu, pv = rng.randint(win.u0 + 1, win.u1 - 1), rng.randint(win.v0 + 1, pv_hi)
        if tier == "medium" and not (pu <= 0 or pv <= 0):
            continue
        if tier == "exam" and not (pu >= 1 and pv >= 1):
            continue
        if tier == "medium" and rng.random() < 0.4:
            m1 = _F(0)
            m2 = rng.choice([s for s in slopes if s != 0])
        else:
            m1, m2 = rng.sample(slopes, 2)
        if _angle(m1, m2) < _MIN_ANGLE:
            continue
        lines = []
        for m in (m1, m2):
            c = pv - m * pu
            if tier == "exam" and c < 0:
                break
            pts = [p for p in lattice_points(win, m, c) if p != (pu, pv)]
            try:
                lines.append(((m, c), pick_two_points(rng, pts, min_du=2)))
            except ValueError:
                break
        else:
            if not _labels_readable(win, lines):
                continue
            if rng.random() < 0.5:
                lines.reverse()
            return win, (pu, pv), lines, ctx
    raise ValueError(f"no plausible draw for tier {tier!r}")


def _pair_text(x, y) -> str:
    return f"({rational_text(x)}, {rational_text(y)})"


def _candidates(rng, win: Window, P, lines, X: int, Y: int):
    """(text, value, note id) per named misconception: the four main ones
    first, in a seed-shuffled order, then the extras."""
    out = []

    def add(x, y, note):
        out.append((_pair_text(x, y), (x, y), note))

    named = []
    if X != Y:
        named.append((Y, X, "swapped-coordinates"))
    (m1, c1), _ = lines[0]
    (m2, c2), _ = lines[1]
    for m, c in rng.sample([(m1, c1), (m2, c2)], 2):
        cy = c * win.ys
        if cy.denominator == 1 and win.shows_y_axis and win.v0 <= c <= win.v1:
            named.append((0, int(cy), "y-intercept-of-a-line"))
            break
    for m, c in rng.sample([(m1, c1), (m2, c2)], 2):
        if m != 0:
            ux = -c / m
            xx = ux * win.xs
            if xx.denominator == 1 and win.shows_x_axis and win.u0 <= ux <= win.u1:
                named.append((int(xx), 0, "x-intercept-of-a-line"))
                break
    pu, pv = P
    shifts = [(pu + 1, pv), (pu - 1, pv), (pu, pv + 1), (pu, pv - 1)]
    shifts = [(a, b) for a, b in shifts if win.u0 <= a <= win.u1 and win.v0 <= b <= win.v1]
    if shifts:
        a, b = rng.choice(shifts)
        named.append((a * win.xs, b * win.ys, "neighbouring-grid-point"))
    rng.shuffle(named)
    extras = []
    thru = [p for _, pts in lines for p in pts]
    a, b = rng.choice(thru)
    extras.append((a * win.xs, b * win.ys, "a-point-one-line-passes-through"))
    if (win.xs, win.ys) != (1, 1):
        extras.append((pu, pv, "grid-squares-not-axis-units"))
    for x, y, note in named + extras:
        add(x, y, note)
    return out


@generator(
    "pv1a_intersection",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def pv1a_intersection(rng, tier="easy", notes=False):
    if tier not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(400):
        win, P, lines, ctx = _draw(rng, tier)
        X, Y = P[0] * win.xs, P[1] * win.ys
        kwargs = {}
        if ctx is not None:
            kwargs = dict(x_label=ctx["x_label"], y_label=ctx["y_label"])
        fig = line_figure(win, [pts for _, pts in lines], _ALT, labels=["A", "B"], **kwargs)

        # Independent route: re-derive from the figure alone and compare.
        rx, ry = read_intersection(fig)
        check((rx, ry) == (X, Y), "the drawn lines do not meet at the key")
        check(all(p != P for _, pts in lines for p in pts), "the crossing is a through point")
        check(fig["equal_scale"] is (win.xs == win.ys), "scale flag disagrees with the window")

        answer_text = _pair_text(X, Y)
        cands = _candidates(rng, win, P, lines, X, Y)
        cands = [c for c in cands if c[1] != (X, Y)]
        texts = pick_distractors(answer_text, [c[0] for c in cands], n=3)
        if len(texts) < 2:
            continue
        values = {c[0]: c[1] for c in cands}
        check_pairwise_distinct(*[values[t] for t in texts], (X, Y))
        for t in texts:
            check(values[t] != (X, Y), "a distractor equals the key")
        try:
            figures.assert_no_leak(fig, [answer_text, *texts])
        except AssertionError:
            continue
        break
    else:
        raise ValueError("could not build a question with two distractors")

    note_of = {c[0]: c[2] for c in cands}
    if tier == "exam":
        text = " ".join([phrase(rng, ctx["story"]), ctx["graph"], phrase(rng, ctx["ask"])])
    else:
        text = " ".join([phrase(rng, _OPEN), phrase(rng, _ASK), phrase(rng, _CONV[tier])])
    (a1, a2), (b1, b2) = lines[0][1], lines[1][1]
    working = [
        f"Line A passes through ({a1[0] * win.xs}, {a1[1] * win.ys}) and ({a2[0] * win.xs}, {a2[1] * win.ys}).",
        f"Line B passes through ({b1[0] * win.xs}, {b1[1] * win.ys}) and ({b2[0] * win.xs}, {b2[1] * win.ys}).",
        f"The lines cross at the grid point {answer_text}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer={"x": X, "y": Y},
        answer_text=answer_text,
        answer_latex=rf"\left({X}, {Y}\right)",
        working=working,
        distractors=texts,
        **({"option_notes": {t: note_of[t] for t in texts}} if notes else {}),
        figure=fig,
    )


def read_pv1a_intersection(question) -> str:
    """Plausibility reader: where the two drawn lines meet, from the figure alone."""
    x, y = read_intersection(question.figure)
    return _pair_text(x, y)


figures.register_reader("pv1a_intersection", read_pv1a_intersection)
