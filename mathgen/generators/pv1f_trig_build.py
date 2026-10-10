"""Generates `pv1f_trig_graph_id` questions: identify an unlabelled graph as y = sin x, y = cos x or y = tan x.

``pv1f_trig_graph_id`` (senior-recognise-the-graphs-of-y-sin-x-y, ACMMM036): an unlabelled
graph of y = sin x, y = cos x or y = tan x on an x-axis marked in multiples of pi
(``x_tick_format="pi"``), and a three-option mcq naming the function. The key class is drawn
FIRST and uniformly; each tier's window always contains x = 0, so the learner always has a
distinguishing feature (sin: a zero at 0 rising through the origin; cos: a maximum at 0;
tan: asymptote branches running off the top and bottom). Tiers escalate by the decision made:

- easy:   one curve over two periods in all, from -2 pi to 2 pi;
- medium: one curve over a narrow window (a quarter to one and a half periods), so a single
          cue must be told apart from the look-alike (sin and tan both rise through O);
- hard:   two of the three functions are drawn and the key belongs to the one through the
          lettered point P;
- exam:   all three are drawn and the key belongs to the one through P.

Nothing the key needs is printed: alt text names the object, the only label is a single
capital on a point. The topic registers a figure reader that works from ``question.figure``
alone (plus the LETTER the prompt names on the hard and exam tiers).
"""

from __future__ import annotations

import math
import re

from mathgen import figures
from mathgen.core import check, generator, pick_distractors

_NAMES = ("sin", "cos", "tan")
_FUNCS = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
_TIERS = ("easy", "medium", "hard", "exam")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_Y = (-3, 3)
_PER_UNIT = 40  # samples per unit of pi: every multiple of 1/40 is an exact sample
_EASY_WINDOWS = ((-2, 2),)
_MEDIUM_WINDOWS = ((-0.5, 0.5), (-1, 1), (-1.5, 1.5))
_MULTI_WINDOW = (-2, 2)  # hard and exam draw several curves on the widest window
_MIN_SEPARATION = 0.055  # a locator point must sit at least this far (normalised) from other curves

# Points of each function on a grid intersection (x in units of pi, y integer), with the
# grid fine enough (a quarter of pi) for tan to pass through y = +-1.
_LOCATORS = {
    "sin": [(0.5, 1), (-0.5, -1), (1.5, -1), (-1.5, 1)],
    "cos": [(0, 1), (1, -1), (-1, -1), (2, 1), (-2, 1), (0.5, 0), (-0.5, 0), (1.5, 0), (-1.5, 0)],
    "tan": [(0.25, 1), (-0.25, -1), (0.75, -1), (-0.75, 1), (1.25, 1), (-1.25, -1), (1.75, -1), (-1.75, 1)],
}


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


def _option(name: str) -> str:
    return f"y = {name} x"


def _curve(name: str, window) -> dict:
    lo, hi = window
    n = int(round((hi - lo) * _PER_UNIT)) + 1
    f = _FUNCS[name]
    return {"pieces": figures.sample_curve(lambda u: f(math.pi * u), lo, hi, _Y, n=n)}


def _plane(*, window, x_grid, curves, alt, points=()) -> dict:
    return figures.cartesian(
        x_range=window,
        y_range=_Y,
        alt=alt,
        x_grid=x_grid,
        y_grid=1,
        equal_scale=False,
        x_tick_format="pi",
        curves=curves,
        points=points,
    )


# ----------------------------------------------------------------------- the reader


def _letter_in(text: str) -> str | None:
    m = re.search(r"point marked ([A-Z])\b", text)
    return m.group(1) if m else None


def _norm_distance(curve: dict, pt, window) -> float:
    """Smallest distance from ``pt`` to the polyline, in window-normalised units."""
    sx = 1.0 / (window[1] - window[0])
    sy = 1.0 / (_Y[1] - _Y[0])
    px, py = pt[0] * sx, pt[1] * sy
    best = float("inf")
    for piece in curve["pieces"]:
        for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
            ax, ay, bx, by = x0 * sx, y0 * sy, x1 * sx, y1 * sy
            dx, dy = bx - ax, by - ay
            ll = dx * dx + dy * dy
            t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
            best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def _classify(curve: dict) -> str:
    """Which of sin, cos, tan the sampled polyline IS: the one every sample lies on."""
    pts = [(u, y) for piece in curve["pieces"] for u, y in piece]
    check(len(pts) >= 20, "too few samples to identify the curve")
    fits = [name for name in _NAMES if all(abs(y - _FUNCS[name](math.pi * u)) < 2e-5 for u, y in pts)]
    check(len(fits) == 1, f"the curve fits {len(fits)} of sin, cos, tan")
    return fits[0]


def _read_from_figure(fig: dict, text: str) -> str:
    curves = fig["curves"]
    letter = _letter_in(text)
    if letter is None:
        check(len(curves) == 1, "no letter named but the figure has several curves")
        return _option(_classify(curves[0]))
    pts = [p["at"] for p in fig["points"] if p.get("label") == letter]
    check(len(pts) == 1, "the named letter is not a single marked point")
    window = fig["x_range"]
    near = [c for c in curves if _norm_distance(c, pts[0], window) < 1e-9]
    check(len(near) == 1, "the named point is not on exactly one curve")
    return _option(_classify(near[0]))


def _read_trig(question) -> str:
    # Reader contract: return the option TEXT, not its index.
    return _read_from_figure(question.figure, question.text)


figures.register_reader("pv1f_trig_graph_id", _read_trig)


def _shows_distinguishing_feature(name: str, fig_curve: dict, window) -> bool:
    """The window holds the cue: sin passes through the origin rising, cos peaks at x = 0,
    tan has branches that leave the window through the top and the bottom."""
    pts = {round(u, 6): y for piece in fig_curve["pieces"] for u, y in piece}
    if not (window[0] < 0 < window[1]):
        return False
    if name == "sin":
        return abs(pts.get(0.0, 9)) < 1e-9 and pts.get(0.025, 0) > 0.05
    if name == "cos":
        return abs(pts.get(0.0, 0) - 1) < 1e-9
    top = max(y for y in pts.values())
    bottom = min(y for y in pts.values())
    return top > 2.4 and bottom < -2.4


# ----------------------------------------------------------------------- the prompts

_POOLS = {
    "easy": (
        (
            "The graph of one of y = sin x, y = cos x and y = tan x is drawn, with x marked in multiples of π.",
            "One of the functions y = sin x, y = cos x and y = tan x has the graph shown, with x in multiples of π.",
            "A graph of y = sin x, y = cos x or y = tan x is drawn on the grid below; the x-axis is marked in multiples of π.",
            "The curve shown is the graph of y = sin x, y = cos x or y = tan x, with x marked in multiples of π.",
        ),
        (
            "The y-axis is marked in whole numbers.",
            "Each horizontal grid line is one unit apart.",
            "Study the shape of the curve before you choose.",
        ),
        ("Which of the three functions is it?", "Which function is drawn?", "Choose the function whose graph this is."),
    ),
    "medium": (
        (
            "Only a short part of the graph of one of y = sin x, y = cos x and y = tan x is shown here, with x in multiples of π.",
            "A narrow section of the graph of y = sin x, y = cos x or y = tan x is drawn, with the x-axis marked in multiples of π.",
            "This is a short piece of the graph of one of y = sin x, y = cos x and y = tan x; x is marked in multiples of π.",
            "The grid shows just a small part of the graph of y = sin x, y = cos x or y = tan x, with x in multiples of π.",
        ),
        (
            "The y-axis is marked in whole numbers.",
            "Each horizontal grid line is one unit apart.",
            "Study the shape of the curve before you choose.",
        ),
        (
            "Decide which of the three functions it comes from.",
            "Which of these functions does the curve belong to?",
            "Name the function whose graph this piece belongs to.",
        ),
    ),
    "hard": (
        (
            "Two of the graphs y = sin x, y = cos x and y = tan x are drawn together, with x in multiples of π.",
            "The grid shows two different curves chosen from y = sin x, y = cos x and y = tan x; x is marked in multiples of π.",
            "On one set of axes, with x in multiples of π, are the graphs of two of the functions y = sin x, y = cos x and y = tan x.",
            "Two curves, each the graph of one of y = sin x, y = cos x and y = tan x, share this grid; x is in multiples of π.",
        ),
        (
            "The y-axis is marked in whole numbers.",
            "Each horizontal grid line is one unit apart.",
            "Study the shapes of the curves before you choose.",
        ),
        (
            "Which function has the graph through the point marked P?",
            "The point marked P lies on one of the curves. Which function is that curve the graph of?",
            "Name the function whose graph passes through the point marked P.",
        ),
    ),
    "exam": (
        (
            "The graphs of all of y = sin x, y = cos x and y = tan x are drawn on one grid, with x in multiples of π.",
            "Three curves, one for each of y = sin x, y = cos x and y = tan x, are drawn together; x is marked in multiples of π.",
            "One grid, with x in multiples of π, holds all three graphs y = sin x, y = cos x and y = tan x.",
            "All three of the functions y = sin x, y = cos x and y = tan x are graphed on the same axes, with x in multiples of π.",
        ),
        (
            "The y-axis is marked in whole numbers.",
            "Each horizontal grid line is one unit apart.",
            "Study the shapes of the curves before you choose.",
        ),
        (
            "Which function's graph passes through the point marked P?",
            "The point marked P lies on one of the three curves. Which function is its graph?",
            "Pick the function whose curve goes through the point marked P.",
        ),
    ),
}

_WORKING = {
    "sin": "The curve rises through the origin and never leaves the band between -1 and 1, so it is y = sin x.",
    "cos": "The curve has its maximum at x = 0 and stays between -1 and 1, so it is y = cos x.",
    "tan": "The curve runs off the top and bottom of the grid in separate branches, so it is y = tan x.",
}


# ----------------------------------------------------------------------- the draw


def _locator(rng, name: str, others: list, curves_by_name: dict, window):
    pool = []
    for u, y in _LOCATORS[name]:
        if not (window[0] <= u <= window[1]):
            continue
        if any(abs(_FUNCS[o](math.pi * u) - y) < 0.5 for o in others if o != "tan"):
            continue
        if any(_norm_distance(curves_by_name[o], (u, y), window) < _MIN_SEPARATION for o in others):
            continue
        pool.append((u, y))
    return _pick(rng, pool) if pool else None


def _build(rng, tier: str):
    key = _pick(rng, _NAMES)  # the answer class FIRST, uniform
    if tier == "easy":
        window, x_grid, shown = _pick(rng, _EASY_WINDOWS), 0.5, [key]
    elif tier == "medium":
        window = _pick(rng, _MEDIUM_WINDOWS)
        x_grid, shown = (0.25 if window[1] - window[0] <= 2 else 0.5), [key]
    else:
        window, x_grid = _MULTI_WINDOW, 0.25
        others = [n for n in _NAMES if n != key]
        shown = [key] + ([_pick(rng, others)] if tier == "hard" else others)
    curves_by_name = {n: _curve(n, window) for n in shown}
    points = []
    if tier in ("hard", "exam"):
        at = _locator(rng, key, [n for n in shown if n != key], curves_by_name, window)
        check(at is not None, "no locator point for the keyed curve")
        points = [{"at": [at[0], at[1]], "label": "P"}]
    order = list(shown)
    rng.shuffle(order)  # drawing order never gives it away
    alt = {
        1: "a curve on a coordinate grid",
        2: "two curves on a coordinate grid",
        3: "three curves on a coordinate grid",
    }[len(shown)]
    fig = _plane(window=window, x_grid=x_grid, curves=[curves_by_name[n] for n in order], alt=alt, points=points)
    check(
        _shows_distinguishing_feature(key, curves_by_name[key], window),
        "the window hides the distinguishing feature of the keyed curve",
    )
    open_, mid, ask = _POOLS[tier]
    text = " ".join((_pick(rng, open_), _pick(rng, mid), _pick(rng, ask)))
    check(_read_from_figure(fig, text) == _option(key), "figure does not show the keyed function")
    return fig, text, key


@generator("pv1f_trig_graph_id", level=13, presets=_PRESETS)
def pv1f_trig_graph_id(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, key = _build(rng, tier)
    answer_text = _option(key)
    distractors = pick_distractors(answer_text, [_option(n) for n in _NAMES if n != key], n=2)
    check(len(distractors) == 2, "the mcq must offer all three functions")
    check(len({answer_text, *distractors}) == 3, "the three options must be distinct")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"y=\\{key} x",
        distractors=distractors,
        working=[_WORKING[key]],
        figure=fig,
    )
