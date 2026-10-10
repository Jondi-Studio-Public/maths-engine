"""Generates figure questions on the slope of a chord of y = f(x) (`senior-interpret-the-ratios-frac-f-left-
x-h`).

Skill ``senior-interpret-the-ratios-frac-f-left-x-h``. The learner sees the
graph of y = f(x) on a fixed integer grid with a chord (a straight line)
through two points of the curve, each at a grid intersection with no
coordinates printed, and types the chord's slope: rise over run between the
two stated x-values. (f(x+h) - f(x)) / h IS that slope.

Construction is answer-first: the slope is drawn, then the two chord points
(integers inside the window), then a curve through both points::

    f(x) = y1 + m (x - x1) + (x - x1)(x - x2)(A + C (x - c))

so the chord meets the curve exactly at the two points whatever A and C are
(exact ``Fraction`` arithmetic; only the chord points need be on the grid).
The key is re-derived a second way (:func:`_verify`, from ``f`` itself).

Tiers escalate by the learner's DECISION, not by bigger numbers:

- easy: two plain x-values, positive integer slope, the curve keeps rising
  across the whole interval -- rise over run of the chord.
- medium: negative or fractional slope and the curve turns between the points
  (it heads the other way at an end), so the learner reads the chord, not the
  curve's direction.
- hard: ratio notation ``x = a``, ``h = -k`` (second point LEFT of the first)
  or the ``delta y / delta x`` form going from the first point to the second:
  the ratio is unchanged when the points are swapped, so a negative h still
  gives the chord's slope.
- exam: a chord AND the tangent at the first point are both drawn; the learner
  picks the line that meets the curve again at x = a + h. The tangent's slope
  is a checked-distinct distractor.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .. import figures
from ..core import (
    VerificationError,
    check,
    check_pairwise_distinct,
    generator,
    paren_negative,
    phrase,
    rational_text,
)

TOPIC = "pv1b_chord_slope"
X_RANGE = (-6, 6)
Y_RANGE = (-10, 10)
N_SAMPLES = 121  # step 0.1: every integer x is sampled exactly
ALT_ONE = "the graph of a curve and a straight line on a coordinate grid"
ALT_TWO = "the graph of a curve and two straight lines on a coordinate grid"

_ANSWER_FORM = " Give your answer as a whole number or a fraction."


# ---------------------------------------------------------------- the curve


def _make_f(x1, x2, y1, m, A, C, c):
    """f through (x1, y1) and (x2, y1 + m (x2 - x1)); exact for Fraction/int
    arguments, float for float arguments."""

    def f(x):
        return y1 + m * (x - x1) + (x - x1) * (x - x2) * (A + C * (x - c))

    return f


def _tangent(x, x1, x2, m, A, C, c) -> Fraction:
    """Exact f'(x): m + g'(x) (A + C (x - c)) + g(x) C with g = (x-x1)(x-x2)."""
    g = (x - x1) * (x - x2)
    dg = 2 * x - x1 - x2
    return m + dg * (A + C * (x - c)) + g * C


def _visible(f, xs) -> bool:
    """Enough of the curve shows, and the neighbourhood of each chord point is in the window."""
    lo, hi = Y_RANGE
    inside = 0
    for i in range(N_SAMPLES):
        x = X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * i / (N_SAMPLES - 1)
        if lo <= f(x) <= hi:
            inside += 1
    if inside < 40:
        return False
    for x in xs:
        for dx in (-1, 1):
            xn = x + dx
            if X_RANGE[0] <= xn <= X_RANGE[1] and not (lo <= f(xn) <= hi):
                return False
    return True


def _curve_points(fig) -> dict:
    """{integer x: y} for every integer-x sample of the curve pieces."""
    out = {}
    for piece in fig["curves"][0]["pieces"]:
        for x, y in piece:
            if isinstance(x, int):
                out[x] = y
    return out


# ---------------------------------------------------------------- checking


def _verify(f, x1, x2, y1, y2, key) -> None:
    """Independent route to the key: evaluate f itself at both stated x-values
    (exactly), confirm the chord's through-points are those curve points, then
    take (f(x2) - f(x1)) / (x2 - x1)."""
    check(x1 != x2, "chord: the two x-values must differ")
    check(f(Fraction(x1)) == y1 and f(Fraction(x2)) == y2, "chord: a through-point is not on the curve")
    check(
        (f(Fraction(x2)) - f(Fraction(x1))) / (x2 - x1) == key, "chord: the difference quotient disagrees with the key"
    )


# ---------------------------------------------------------------- answer text


def _answer_fields(key: Fraction):
    if key.denominator == 1:
        return int(key), rational_text(key), str(int(key))
    sign = "-" if key < 0 else ""
    return (
        {"p": key.numerator, "q": key.denominator},
        rational_text(key),
        f"{sign}\\frac{{{abs(key.numerator)}}}{{{key.denominator}}}",
    )


def _pick(key: Fraction, named, n: int = 3) -> list[str]:
    eligible: list[Fraction] = []
    for _name, value in named:
        if value != key and value not in eligible:
            eligible.append(value)
    kept = eligible[:n]
    # the key must not be the only option of its shape (integer vs fraction)
    same = [v for v in eligible if (v.denominator == 1) == (key.denominator == 1)]
    if kept and same and not any(v in same for v in kept):
        kept[-1] = same[0]
    check_pairwise_distinct(key, *kept)
    return [rational_text(v) for v in kept]


# ---------------------------------------------------------------- drawing


def _signed_slope(rng, *, int_share: float, allow_pos: bool, max_abs: int = 5) -> Fraction:
    """Draw the answer first: an integer with probability ``int_share``, else a
    reduced fraction with denominator 2 or 3; never zero."""
    sgn = 1 if allow_pos and rng.random() < 0.5 else -1
    if rng.random() < int_share:
        return Fraction(sgn * rng.randint(1, max_abs))
    q = rng.choice((2, 3))
    while True:
        p = rng.randint(1, max_abs * q - 1)
        if Fraction(p, q).denominator == q:
            return Fraction(sgn * p, q)


def _run_for(rng, m: Fraction, max_run: int) -> int:
    """A run (a multiple of the slope's denominator) of 1..max_run."""
    q = m.denominator
    return q * rng.randint(1, max(1, max_run // q))


def _points_fit(x1, x2, y1, y2) -> bool:
    return -5 <= x1 <= 5 and -5 <= x2 <= 5 and -8 <= y1 <= 8 and -8 <= y2 <= 8


def _figure(f, marked, lines, alt):
    pieces = figures.sample_curve(lambda x: float(f(x)), X_RANGE[0], X_RANGE[1], Y_RANGE, N_SAMPLES)
    return figures.cartesian(
        x_range=X_RANGE,
        y_range=Y_RANGE,
        alt=alt,
        grid=1,
        points=marked,
        lines=[{"through": [list(p), list(q)]} for p, q in lines],
        curves=[{"pieces": pieces}],
    )


def _draw_m(rng, tier) -> Fraction:
    """The answer, drawn first and uniformly over the tier's pool."""
    if tier == "easy":
        return Fraction(rng.randint(1, 6))
    if tier == "medium":
        while True:  # negative or fractional
            m = _signed_slope(rng, int_share=0.4, allow_pos=True, max_abs=4)
            if not (m > 0 and m.denominator == 1):
                return m
    return _signed_slope(rng, int_share=0.5, allow_pos=True, max_abs=5)


def _draw(rng, tier):
    """Return a dict: f, x1, x2, y1, y2, m, plus tier extras. The slope is drawn
    first and kept; only the geometry around it is retried."""
    for _outer in range(60):
        m = _draw_m(rng, tier)
        for _attempt in range(40):
            out = _try_geometry(rng, tier, m)
            if out is not None:
                return out
    raise RuntimeError(f"chord_slope/{tier}: could not draw a figure")


def _try_geometry(rng, tier, m):
    """One attempt at points and a curve for slope ``m``; None if it does not fit."""
    if tier == "easy":
        run = rng.choice((1, 2, 3))
        xl = rng.randint(-5, 5 - run)
        x1, x2 = xl, xl + run
        rise = int(m * run)
        if rise > 16:
            return None
        y1 = rng.randint(-8, 8 - rise)
        A = rng.choice((1, -1)) * rng.choice((Fraction(1, 4), Fraction(1, 2), Fraction(1)))
        if abs(A) * run >= m:  # the curve must keep rising across [x1, x2]
            return None
        C, c = Fraction(0), 0
    elif tier == "medium":
        run = _run_for(rng, m, 5)
        xl = rng.randint(-5, 5 - run)
        x1, x2 = xl, xl + run
        rise = int(m * run)
        lo, hi = sorted((0, rise))
        if hi - lo > 16:
            return None
        y1 = rng.randint(-8 - lo, 8 - hi)
        A = rng.choice((1, -1)) * rng.choice((Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)))
        if abs(A) * run <= abs(m):  # the curve must turn between the points
            return None
        C, c = Fraction(0), 0
    else:  # hard / exam
        run = _run_for(rng, m, 4)
        h = -run if (tier == "hard" or rng.random() < 0.5) else run  # hard: second point LEFT
        x1 = rng.randint(-5, 5)
        x2 = x1 + h
        rise = int(m * h)
        lo, hi = sorted((0, rise))
        if not (-5 <= x2 <= 5) or hi - lo > 16:
            return None
        y1 = rng.randint(-8 - lo, 8 - hi)
        A = rng.choice((1, -1)) * rng.choice((Fraction(1, 2), Fraction(1), Fraction(2)))
        if rng.random() < 0.3:
            C = rng.choice((1, -1)) * rng.choice((Fraction(1, 4), Fraction(1, 2)))
            c = rng.randint(-3, 3)
        else:
            C, c = Fraction(0), 0
    y2 = y1 + int(m * (x2 - x1))
    if not _points_fit(x1, x2, y1, y2):
        return None
    f = _make_f(x1, x2, y1, m, A, C, c)
    if not _visible(f, (x1, x2)):
        return None
    out = dict(f=f, x1=x1, x2=x2, y1=y1, y2=y2, m=m)
    if tier == "exam":
        t = _tangent(Fraction(x1), x1, x2, m, A, C, c)
        if t == m or t.denominator > 3 or abs(t) > 8:
            return None
        tan_pts = None
        for s in (t.denominator, -t.denominator):
            sx = x1 + s
            sy = y1 + t * s
            if -5 <= sx <= 5 and -8 <= sy <= 8 and sx != x2 and f(Fraction(sx)) != sy:
                tan_pts = ((x1, y1), (sx, int(sy)))
                break
        if tan_pts is None:
            return None
        out.update(t=t, tan_pts=tan_pts)
    return out


# ---------------------------------------------------------------- prompts

_EASY = (
    "The graph of y = f(x) is drawn below with a chord joining the points on the curve where "
    "x = {a} and x = {b}. State the slope of the chord.",
    "A chord is drawn on the graph of y = f(x) between the points where x = {a} and x = {b}. "
    "What is the gradient of the chord?",
    "Use the graph to find the slope of the straight line through the points of the curve at x = {a} and x = {b}.",
)
_MEDIUM = (
    "The graph of y = f(x) is drawn with a chord between the points where x = {a} and x = {b}. "
    "State the gradient of the chord.",
    "Find the slope of the chord of y = f(x) that joins the points at x = {a} and x = {b} on the graph.",
    "A straight line (a chord) cuts the curve y = f(x) at x = {a} and x = {b}. What is the slope of this line?",
)
_HARD = (
    "A chord is drawn on the graph of y = f(x). For x = {a} and h = {h}, state the value of (f(x + h) - f(x)) / h.",
    "The graph of y = f(x) is shown with a chord. Taking x = {a} and h = {h}, work out (f(x + h) - f(x)) / h.",
    "A chord is drawn on the graph of y = f(x). Going from the point where x = {a} to the point "
    "where x = {b}, state δy / δx, the change in y divided by the change in x.",
)
_EXAM = (
    "Two straight lines are drawn through the point on the curve y = f(x) where x = {a}. "
    "For a = {a} and h = {h}, use the graph to find (f(a + h) - f(a)) / h.",
    "The graph of y = f(x) is drawn with two straight lines through the point at x = {a}. "
    "Taking a = {a} and h = {h}, state the value of (f(a + h) - f(a)) / h.",
    "On the graph of y = f(x), two lines pass through the point where x = {a}. "
    "State (f(a + h) - f(a)) / h when a = {a} and h = {h}.",
)
_PATTERNS = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD, "exam": _EXAM}


def _build(rng, tier):
    d = _draw(rng, tier)
    f, x1, x2, y1, y2, m = d["f"], d["x1"], d["x2"], d["y1"], d["y2"], d["m"]
    key = m
    _verify(f, x1, x2, y1, y2, key)
    rise, run = y2 - y1, x2 - x1

    wrong_run = x2 if x2 not in (0, run) else run + (1 if run > 0 else -1)
    named = []
    if tier == "exam":
        named.append(("tangent-slope-at-first-point", d["t"]))
    if tier != "easy":
        named.append(("sign-flipped", -key))
    named.append(("reciprocal", 1 / key))
    named.append(("rise-only", Fraction(rise)))
    named.append(("curve-height-difference-over-wrong-run", Fraction(rise, wrong_run)))
    named.append(("average-of-heights", Fraction(y1 + y2, 2)))
    if tier == "easy":
        named.append(("sign-flipped", -key))
    distractors = _pick(key, named)

    answer, answer_text, answer_latex = _answer_fields(key)
    if tier == "exam":
        chord = ((x1, y1), (x2, y2))
        order = [chord, d["tan_pts"]]
        rng.shuffle(order)
        figure = _figure(f, [{"at": [x1, y1], "label": "A"}], order, ALT_TWO)
    else:
        figure = _figure(
            f, [{"at": [x1, y1], "label": "A"}, {"at": [x2, y2], "label": "B"}], [((x1, y1), (x2, y2))], ALT_ONE
        )
    figures.assert_no_leak(figure, [answer_text, *distractors])

    text = phrase(rng, _PATTERNS[tier], a=x1, b=x2, h=run) + _ANSWER_FORM
    working = [
        f"The chord passes through ({x1}, {y1}) and ({x2}, {y2}).",
        f"Slope = rise / run = ({y2} - {paren_negative(y1)}) / ({x2} - {paren_negative(x1)}) = {rise} / {paren_negative(run)} = {answer_text}.",
    ]
    if tier in ("hard", "exam"):
        working.append(
            "The ratio is the slope of the chord joining x and x + h. Swapping the two "
            "points flips the sign of both the rise and the run, so a negative h "
            "still gives the same slope."
        )
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
        distractors=distractors,
        working=working,
        figure=figure,
    )


@generator(
    TOPIC,
    level=13,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def pv1b_chord_slope(rng, variant="hard"):
    """Slope of a drawn chord of y = f(x). See the module docstring for the four tiers."""
    if variant not in _PATTERNS:
        raise ValueError(f"unknown variant: {variant!r}")
    return _build(rng, variant)


# ---------------------------------------------------------------- reader

_X_RE = re.compile(r"\b[xa] = (-?\d+)")
_H_RE = re.compile(r"\bh = (-?\d+)")


def read_chord_slope(question) -> str:
    """Re-derive the key from the figure alone plus the stated x-values: the
    chord is the drawn line whose two through-points are BOTH curve points at
    the stated x-values (a tangent touches the curve at the first only)."""
    fig = question.figure
    xs = [int(v) for v in _X_RE.findall(question.text)]
    hs = _H_RE.findall(question.text)
    x1 = xs[0]
    x2 = x1 + int(hs[0]) if hs else xs[1]
    on_curve = _curve_points(fig)
    want = {(x1, on_curve[x1]), (x2, on_curve[x2])}
    for ln in fig["lines"]:
        p, q = (tuple(pt) for pt in ln["through"])
        if {p, q} == want:
            return rational_text(Fraction(q[1] - p[1], q[0] - p[0]))
    raise VerificationError("no drawn line joins the curve points at the stated x-values")


figures.register_reader(TOPIC, read_chord_slope)
