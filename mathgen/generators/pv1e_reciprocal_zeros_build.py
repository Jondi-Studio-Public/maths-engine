"""Generates `pv1e_reciprocal_zeros` figure questions: relate the graph of y = f(x) to the graph of its
reciprocal.

Skill ``senior-examine-the-relationship-between-the-graph-of-y``, topic ``pv1e_reciprocal_zeros``.
The learner sees the graph of y = f(x), a straight line or a parabola, and lists the x-values at
which y = 1/f(x) has vertical asymptotes: the zeros of f, because 1/f is undefined exactly where
f(x) = 0. The graph of 1/f is never shown and f is never printed. ``answer: list`` (an
order-insensitive multiset, graded count-sensitively, so the prompt says to list each x-value
once).

Every zero is an integer, in the window, where the drawn graph meets the x-axis. The drawing is
exact: a parabola is ``a(x - p)(x - q)`` (or ``a(x - r)^2`` for a touch) sampled at step 1/16, so
a zero is a sample whose y is exactly 0.

Tiers differ in the learner's DECISION:

- easy   -- a straight line: one zero, where it crosses the x-axis (not its y-intercept, not
            its gradient).
- medium -- a parabola that crosses the x-axis twice: both crossings are asymptotes of 1/f, and
            the turning point (a feature that stands out on the picture) is not one.
- hard   -- a parabola whose turning point sits ON the x-axis: it touches without crossing, and
            that single touch still gives an asymptote, listed once.
- exam   -- a zero at the origin (the graph passes through (0, 0)): the learner lists 0 itself,
            alone for a line through the origin or beside the other zero for a parabola.
"""

from __future__ import annotations

import re
from fractions import Fraction as F
from types import SimpleNamespace

from .. import figures
from ..core import check, generator
from . import pv1e_curve_figure_build as cf

TOPIC = "pv1e_reciprocal_zeros"
_TIERS = ("easy", "medium", "hard", "exam")
LO, HI = -6, 6

_ALT_LINE = "a straight line on a coordinate grid"
_ALT_CURVE = "a parabola on a coordinate grid"

_OPEN = [
    "The graph of y = f(x) is drawn on the grid.",
    "A function f has the graph shown on the grid.",
    "Below is the graph of y = f(x).",
    "The diagram shows the graph of y = f(x).",
]
_NOTE = [
    "Each grid square is 1 unit on both axes.",
    "One grid square is 1 unit across and 1 unit up.",
    "Both axes have 1 unit per grid square.",
]
_ASK = [
    "The graph of y = 1/f(x) has vertical asymptotes. List the x-values at which they occur, "
    "separated by commas, giving each x-value once.",
    "Find the x-values at which the graph of y = 1/f(x) has vertical asymptotes. Give them as a "
    "list separated by commas, with each x-value once.",
    "State the vertical asymptotes of y = 1/f(x) as x-values, in a list separated by commas (each x-value once).",
]


#: the exam tier names the reciprocal function g, so the learner works with g(x) = 1/f(x)
_ASK_EXAM = [
    "Let g(x) = 1/f(x). List the x-values at which the graph of y = g(x) has vertical asymptotes, "
    "separated by commas, giving each x-value once.",
    "A function g is defined by g(x) = 1/f(x). State the x-values of the vertical asymptotes of "
    "y = g(x) as a list separated by commas, each x-value once.",
    "Define g(x) = 1/f(x). Find every x-value at which y = g(x) has a vertical asymptote. Give "
    "them as a list separated by commas, with each x-value once.",
]


def _tex(text: str) -> str:
    text = (
        text.replace("g(x) = 1/f(x)", "\x00C")
        .replace("y = 1/f(x)", "\x00A")
        .replace("y = f(x)", "\x00B")
        .replace("y = g(x)", "\x00D")
    )
    text = re.sub(r"\b([fg])\b", r"\\(\1\\)", text)
    return (
        text.replace("\x00A", r"\(y = \frac{1}{f(x)}\)")
        .replace("\x00B", r"\(y = f(x)\)")
        .replace("\x00C", r"\(g(x) = \frac{1}{f(x)}\)")
        .replace("\x00D", r"\(y = g(x)\)")
    )


# ---------------------------------------------------------------- drawing the answer first

_ROOTS = tuple(r for r in range(-5, 6) if r != 0)
_SLOPES = (F(1), F(-1), F(2), F(-2), F(3), F(-3), F(1, 2), F(-1, 2))
_A_TWO = (F(1), F(-1), F(1, 2), F(-1, 2), F(2), F(-2))
_A_TOUCH = (F(1), F(-1), F(1, 2), F(-1, 2))


def _line_points(rng, m: F, r: int):
    """Two grid intersections on y = m(x - r), neither on the x-axis, inside the window."""
    p, q = m.numerator, m.denominator
    ts = [t for t in range(-12, 13) if t != 0 and LO <= r + q * t <= HI and LO <= p * t <= HI]
    check(len(ts) >= 2, "reciprocal_zeros: no grid points on the line")
    t1, t2 = rng.sample(ts, 2)
    return [[r + q * t1, p * t1], [r + q * t2, p * t2]]


def _parabola_ok(a: F, p: int, q: int) -> bool:
    """The turning point is inside the window and the parabola is not a sliver."""
    return abs(a) * (q - p) ** 2 / 4 <= HI - 0.5


def _draw(rng, tier: str):
    """-> (zeros, shape) with shape a dict describing what to draw and which distractors apply."""
    if tier == "easy":
        r, m = rng.choice(_ROOTS), rng.choice(_SLOPES)
        return [r], {"kind": "line", "m": m, "r": r}
    if tier == "medium":
        while True:
            p, q = sorted(rng.sample(_ROOTS, 2))
            a = rng.choice(_A_TWO)
            if _parabola_ok(a, p, q):
                return [p, q], {"kind": "pair", "a": a, "p": p, "q": q}
    if tier == "hard":
        r, a = rng.choice(_ROOTS), rng.choice(_A_TOUCH)
        return [r], {"kind": "touch", "a": a, "r": r}
    if tier == "exam":
        if rng.random() < 0.12:
            return [0], {"kind": "line", "m": rng.choice(_SLOPES), "r": 0}
        while True:
            q = rng.choice(_ROOTS)
            a = rng.choice(_A_TWO)
            if _parabola_ok(a, min(0, q), max(0, q)):
                return sorted([0, q]), {"kind": "pair", "a": a, "p": 0, "q": q}
    raise ValueError(f"unknown tier {tier!r}")


def _function(shape):
    kind = shape["kind"]
    if kind == "pair":
        a, p, q = float(shape["a"]), shape["p"], shape["q"]
        return lambda x: a * (x - p) * (x - q)
    if kind == "touch":
        a, r = float(shape["a"]), shape["r"]
        return lambda x: a * (x - r) ** 2
    raise ValueError(kind)


def _build_figure(rng, shape) -> dict:
    window = dict(x_range=(LO, HI), y_range=(LO, HI))
    if shape["kind"] == "line":
        return cf.plane(alt=_ALT_LINE, lines=[{"through": _line_points(rng, shape["m"], shape["r"])}], **window)
    return cf.plane(alt=_ALT_CURVE, curves=[{"pieces": cf.pieces(_function(shape), (LO, HI), (LO, HI))}], **window)


# ---------------------------------------------------------------- reading a figure


def _line_zero(ln) -> F | None:
    (x1, y1), (x2, y2) = ln["through"]
    if y1 == y2:
        return None
    return F(x1) - F(y1) * (F(x2) - F(x1)) / (F(y2) - F(y1))


def read_zeros(question) -> str:
    """Re-derive the asymptote x-values from the figure alone: where the graph meets y = 0."""
    fig = question.figure
    zeros = set()
    for ln in fig["lines"]:
        z = _line_zero(ln)
        if z is not None:
            check(z.denominator == 1, "reciprocal_zeros reader: a line crosses between grid lines")
            zeros.add(int(z))
    for piece in cf.curve_pieces(fig):
        for x, y in piece:
            if abs(y) < 1e-9:
                check(abs(x - round(x)) < 1e-9, "reciprocal_zeros reader: a zero off the grid")
                zeros.add(int(round(x)))
    check(bool(zeros), "reciprocal_zeros reader: the graph never meets the x-axis")
    return ", ".join(str(z) for z in sorted(zeros))


figures.register_reader(TOPIC, read_zeros)


def _verify(fig, zeros) -> None:
    got = read_zeros(SimpleNamespace(figure=fig))
    check(got == ", ".join(str(z) for z in zeros), "reciprocal_zeros: figure reader disagrees with the key")
    check(all(LO < z < HI for z in zeros), "reciprocal_zeros: a zero sits on the window edge")


# ---------------------------------------------------------------- distractors


def _lst(values) -> str:
    return ", ".join(str(v) for v in values)


def _is_int(v) -> bool:
    return isinstance(v, int) or (isinstance(v, F) and v.denominator == 1)


def _candidates(tier, zeros, shape):
    """[(misconception, list of ints)] in priority order; may repeat the key (filtered later)."""
    kind = shape["kind"]
    out = []
    if kind == "line" and tier == "easy":
        r, m = shape["r"], shape["m"]
        yint = -m * r
        out.append(("sign-flipped", [-r]))
        if _is_int(yint) and abs(yint) <= HI:
            out.append(("y-intercept", [int(yint)]))
        if _is_int(m):
            out.append(("gradient", [int(m)]))
        out += [("off-by-one", [r + 1]), ("off-by-one", [r - 1])]
    elif kind == "pair" and tier == "medium":
        p, q = shape["p"], shape["q"]
        out.append(("only-one-zero", [p]))
        out.append(("only-one-zero", [q]))
        out.append(("sign-flipped", sorted([-p, -q])))
        if (p + q) % 2 == 0:
            mid = (p + q) // 2
            out.append(("vertex-x", [mid]))
            out.append(("zeros-and-vertex", [p, mid, q]))
        yint = shape["a"] * p * q
        if _is_int(yint) and abs(yint) <= HI:
            out.append(("y-intercept", [int(yint)]))
    elif kind == "touch":
        r, a = shape["r"], shape["a"]
        out.append(("sign-flipped", [-r]))
        yint = a * r * r
        if _is_int(yint) and abs(yint) <= HI:
            out.append(("y-intercept", [int(yint)]))
        out.append(("y-axis", [0]))
        out += [("off-by-one", [r + 1]), ("off-by-one", [r - 1])]
    elif kind == "pair":  # exam: zeros 0 and q
        q = shape["q"]
        out.append(("omits-origin", [q]))
        out.append(("origin-only", [0]))
        out.append(("sign-flipped", sorted([0, -q])))
        if q % 2 == 0:
            out.append(("vertex-x", [q // 2]))
        out.append(("off-by-one", sorted([0, q + (1 if q > 0 else -1)])))
    else:  # exam: a line through the origin
        m = shape["m"]
        if _is_int(m):
            out.append(("gradient", [int(m)]))
        out += [("off-by-one", [1]), ("off-by-one", [-1]), ("off-by-one", [2])]
    return out


def _distractors(tier, zeros, shape):
    out, seen = [], {tuple(zeros)}
    for name, vals in _candidates(tier, zeros, shape):
        key = tuple(sorted(vals))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, list(key)))
    return out[:3]


# ---------------------------------------------------------------- building


@generator(TOPIC, level=13, presets={t: {"tier": t} for t in _TIERS})
def pv1e_reciprocal_zeros(rng, tier: str = "easy") -> dict:
    """x-values of the vertical asymptotes of 1/f from the graph of f. Answer first."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    zeros, shape = _draw(rng, tier)
    fig = _build_figure(rng, shape)
    _verify(fig, zeros)

    answer_text = _lst(zeros)
    dist = _distractors(tier, zeros, shape)
    check(len(dist) >= 2, "reciprocal_zeros: fewer than two distinct distractors")
    texts = [_lst(v) for _, v in dist]
    figures.assert_no_leak(fig, [answer_text, *texts])

    ask = _ASK_EXAM if tier == "exam" else _ASK
    text = " ".join((rng.choice(_OPEN), rng.choice(_NOTE), rng.choice(ask)))
    plural = len(zeros) > 1
    working = [
        "1/f(x) is undefined wherever f(x) = 0, so its vertical asymptotes are at the x-values "
        "where the graph of f meets the x-axis.",
        (
            f"The graph meets the x-axis at x = {zeros[0]}"
            + (
                f" and x = {zeros[1]}"
                if plural
                else (" (it touches the axis there)" if shape["kind"] == "touch" else "")
            )
            + f", so the vertical asymptotes are at {answer_text}."
        ),
    ]
    return {
        "text": text,
        "latex": _tex(text),
        "answer": list(zeros),
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": texts,
        "working": working,
        "figure": fig,
    }
