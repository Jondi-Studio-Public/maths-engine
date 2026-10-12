"""Generates `pv1e_log_asymptote` figure questions (ACMMM156): read the asymptote of a drawn logarithmic curve.

Skill ``senior-recognise-the-qualitative-features-of-the-graph-of-acmmm156``, topic
``pv1e_log_asymptote``. The learner sees a logarithmic curve (a > 1) on a labelled integer grid
and types the equation of its vertical asymptote, ``x = -c`` (``answer: pair``, one labelled
component). The asymptote is not drawn: the curve is sampled finely toward the line x = -c, in
zones that shrink geometrically, so it runs down along that gridline and leaves the bottom of the
window within 0.011 of it (base 2) or 0.0007 (base 3).

Tiers differ in the learner's DECISION:

- easy   -- base 2, the asymptote is a non-zero gridline: the learner names the line the curve
            runs down along, not the x-intercept one square to its right.
- medium -- base 2 or 3, and the asymptote may be x = 0, the y-axis itself, so the learner names
            an axis as the asymptote.
- hard   -- one grid square is 2 units across: the learner reads x = -c in units from the tick
            numbers (counting squares gives half of it).
- exam   -- the window is shifted away from the y-axis, so no y-axis is drawn: the learner reads
            the asymptote against the numbers along the x-axis (the x-axis is drawn, the y-axis is not).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from .. import figures
from ..core import check, generator, latex_from_inline
from . import pv1e_curve_figure_build as cf

TOPIC = "pv1e_log_asymptote"
_TIERS = ("easy", "medium", "hard", "exam")
_ALT = "a curve climbing from the bottom of the grid on a coordinate grid"

_NONZERO = tuple(v for v in range(-5, 4) if v != 0)  # the curve needs 3 squares to its right
_ALL = tuple(range(-5, 4))
_OFF_X0 = tuple(range(1, 9)) + tuple(range(-20, -13))  # a 12-wide window that excludes 0
_Y_RANGE = (-6, 6)
#: distances from the asymptote at which the sampling zones meet (finer toward the line)
_NEAR = (5e-4, 5e-3, 5e-2, 0.5)

_OPEN = [
    (
        "The graph of y = log_a(x + c) is drawn on the grid, where a > 1 and c is an integer.",
        r"The graph of \(y = \log_{a}(x + c)\) is drawn on the grid, where \(a > 1\) and \(c\) is an integer.",
    ),
    (
        "A curve with equation y = log_a(x + c) is shown, with a > 1 and c an integer.",
        r"A curve with equation \(y = \log_{a}(x + c)\) is shown, with \(a > 1\) and \(c\) an integer.",
    ),
    (
        "The graph shows y = log_a(x + c) for a number a greater than 1 and an integer c.",
        r"The graph shows \(y = \log_{a}(x + c)\) for a number \(a\) greater than \(1\) and an integer \(c\).",
    ),
    (
        "A logarithmic curve y = log_a(x + c), with a > 1 and c an integer, is drawn below.",
        r"A logarithmic curve \(y = \log_{a}(x + c)\), with \(a > 1\) and \(c\) an integer, is drawn below.",
    ),
]
_NOTES = {
    "easy": [
        "Each grid square is 1 unit on both axes.",
        "One grid square is 1 unit across and 1 unit up.",
        "Both axes have 1 unit per grid square.",
    ],
    "hard": [
        "Each grid square is 2 units across and 1 unit up.",
        "The horizontal scale is 2 units per grid square and the vertical scale is 1 unit per grid square.",
        "Across, one grid square is 2 units; up, one grid square is 1 unit.",
    ],
    "exam": [
        "The y-axis is not shown, so read the value from the numbers along the x-axis. Each grid square is 1 unit.",
        "No y-axis is drawn: use the numbers marked along the x-axis, where one square is 1 unit each way.",
        "The window is to one side of the y-axis, so use the numbers along the x-axis; one grid "
        "square is 1 unit on both scales.",
    ],
}
_NOTES["medium"] = _NOTES["easy"]
_ASK = [
    "State the equation of its vertical asymptote.",
    "Find the equation of the vertical asymptote.",
    "Give the equation of its vertical asymptote.",
]


def _tex_note(note: str) -> str:
    return note.replace("y-axis", r"\(y\)-axis").replace("x-axis", r"\(x\)-axis")


# ---------------------------------------------------------------- drawing the answer first


def _draw(rng, tier: str):
    """-> (base, v, x_range, x_grid, equal_scale) where v is the asymptote x = v (= -c)."""
    if tier == "easy":
        return 2, rng.choice(_NONZERO), (-6, 6), 1, True
    if tier == "medium":
        return rng.choice((2, 3)), rng.choice(_ALL), (-6, 6), 1, True
    if tier == "hard":
        return (rng.choice((2, 3)), rng.choice((-10, -8, -6, -4, -2, 2, 4, 6, 8)), (-12, 12), 2, False)
    if tier == "exam":
        x0 = rng.choice(_OFF_X0)
        return rng.choice((2, 3)), x0 + rng.randint(1, 6), (x0, x0 + cf.SPAN), 1, True
    raise ValueError(f"unknown tier {tier!r}")


def _curve(base: int, v: int, x_range) -> list:
    """The curve y = log_base(x - v), finest next to the asymptote, ending at the window's edge."""
    f = lambda x: math.log(x - v, base)  # noqa: E731 - sampled, never solved
    zones = [(v + a, v + b) for a, b in zip(_NEAR, _NEAR[1:])]
    out = cf.zoned_pieces(f, zones, _Y_RANGE, n_zone=40)
    tail = figures.sample_curve(f, v + _NEAR[-1], x_range[1], _Y_RANGE, n=80)
    for piece in tail:
        if out and out[-1][-1] == piece[0]:
            out[-1].extend(piece[1:])
        else:
            out.append(list(piece))
    return out


# ---------------------------------------------------------------- reading a figure


def read_log_asymptote(question) -> str:
    """Re-derive ``x = -c`` from the figure alone: the whole-number gridline the curve starts on."""
    pcs = cf.curve_pieces(question.figure)
    check(len(pcs) == 1, "log reader: expected one unbroken curve")
    x_first = pcs[0][0][0]
    v = round(x_first)
    check(0 < x_first - v < 0.02, "log reader: the curve does not start against a whole-number gridline")
    ys = [p[1] for p in pcs[0]]
    check(ys[0] < ys[-1], "log reader: the curve does not climb away from the line")
    return f"x = {v}"


figures.register_reader(TOPIC, read_log_asymptote)


def _verify(fig, v) -> None:
    got = read_log_asymptote(SimpleNamespace(figure=fig))
    check(got == f"x = {v}", "log: figure reader disagrees with the key")
    pcs = cf.curve_pieces(fig)
    pts = pcs[0]
    check(pts[0][1] <= fig["y_range"][0] + 0.5, "log: the curve does not run down to the bottom edge")
    check(pts[-1][0] > v + 2, "log: too little of the curve is to the right of the asymptote")


# ---------------------------------------------------------------- distractors

_PRIORITY = {
    "easy": ("x-intercept", "sign-flipped", "y-axis", "off-by-one", "off-by-two"),
    "medium": ("x-intercept", "sign-flipped", "y-axis", "off-by-one", "off-by-two"),
    "hard": ("squares-not-units", "x-intercept", "sign-flipped", "y-axis", "off-by-one"),
    "exam": ("x-intercept", "off-by-one", "sign-flipped", "off-by-two", "y-axis"),
}


def _distractors(tier: str, v: int):
    cands = {
        "x-intercept": v + 1,
        "sign-flipped": -v,
        "y-axis": 0,
        "off-by-one": v - 1,
        "off-by-two": v + 2,
        "squares-not-units": v // 2,
    }
    out, seen = [], {v}
    for name in _PRIORITY[tier]:
        val = cands[name]
        if val in seen:
            continue
        seen.add(val)
        out.append((name, val))
    return out[:3]


# ---------------------------------------------------------------- building


@generator(TOPIC, level=13, presets={t: {"tier": t} for t in _TIERS})
def pv1e_log_asymptote(rng, tier: str = "easy") -> dict:
    """The vertical asymptote of y = log_a(x + c) from its graph. Answer first, then the window."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    base, v, x_range, x_grid, equal_scale = _draw(rng, tier)
    fig = cf.plane(
        x_range=x_range,
        y_range=_Y_RANGE,
        alt=_ALT,
        x_grid=x_grid,
        equal_scale=equal_scale,
        curves=[{"pieces": _curve(base, v, x_range)}],
    )
    _verify(fig, v)

    answer_text = f"x = {v}"
    dist = _distractors(tier, v)
    check(len(dist) >= 2, "log: fewer than two distinct distractors")
    texts = [f"x = {val}" for _, val in dist]
    check(len({v, *[val for _, val in dist]}) == len(dist) + 1, "log: a distractor repeats the key")
    figures.assert_no_leak(fig, [answer_text, *texts])

    open_text, open_latex = rng.choice(_OPEN)
    note = rng.choice(_NOTES[tier])
    ask = rng.choice(_ASK)
    working = [
        f"The curve runs down along a vertical line at the left without ever reaching it: that line is x = {v}.",
        f"So the vertical asymptote is x = {v}.",
    ]
    return {
        "text": " ".join((open_text, note, ask)),
        "latex": latex_from_inline(" ".join((open_latex, _tex_note(note), ask))),
        "answer": {"x": v},
        "answer_text": answer_text,
        "answer_latex": f"x = {v}",
        "distractors": texts,
        "working": working,
        "figure": fig,
    }
