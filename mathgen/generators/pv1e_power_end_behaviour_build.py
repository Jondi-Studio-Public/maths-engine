"""End behaviour of y = x^n read off its graph (ACMMM014).

Skill ``senior-recognise-features-of-the-graphs-of-y-x-acmmm014``, topic ``pv1e_power_end_behaviour``.
The learner sees the curve y = x^n for one hidden n (a positive whole number, n = -1 or n = 1/2) on a
labelled integer grid and is asked what y does as x tends to infinity OR as x tends to negative
infinity: the QUESTION names the direction. Three options, always:
``increases without bound``, ``decreases without bound``, ``approaches a fixed value``. n = 1/2 has
domain x >= 0, so it is drawn and asked only at +infinity.

Key classes by (n, end):

========  ==========  ==========
n         +infinity   -infinity
========  ==========  ==========
odd       increases   DECREASES
even      increases   increases
-1        approaches  approaches
1/2       increases   (never asked)
========  ==========  ==========

The CLASS is drawn first, uniformly, then one (n, end) of that class, then the window, so each label is
the key about a third of the time. Nothing about n, the class or the direction's answer is printed.

Tiers differ in the learner's DECISION:

- easy   -- n is 1, 2, 3 or -1 in a centred window with both axes drawn, so both ends of the curve are
            in view; the learner reads the end the question names, and a left-end question on an odd
            power is the one that falls to the lower left.
- medium -- adds n = 4, 5 (flat near the origin, then steep) and n = 1/2, which bends like a curve that is
            levelling off yet keeps rising: the learner must not read a bend as a limit.
- hard   -- only the half of the curve on the named side of the y-axis is drawn, so there is no mirror
            half to compare with: the learner has to know the family, and a left-end even power then
            falls toward the origin when read left to right but increases as x -> -infinity.
- exam   -- that half-curve is also shifted 1 or 2 units away from the y-axis, so no y-axis is drawn and
            the curve is a bare fragment read against the numbers along the edges of the grid.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .. import figures
from ..core import check, generator, latex_answer
from . import pv1e_curve_figure_build as cf

TOPIC = "pv1e_power_end_behaviour"
_TIERS = ("easy", "medium", "hard", "exam")

INC, DEC, APP = "increases without bound", "decreases without bound", "approaches a fixed value"
_LABELS = (INC, DEC, APP)
_ALT = "a curve on a coordinate grid"
_HALF = Fraction(1, 2)

_POOL = {
    "easy": (1, 2, 3, -1),
    "medium": (1, 2, 3, 4, 5, -1, _HALF),
    "hard": (1, 2, 3, 4, 5, -1, _HALF),
    "exam": (1, 2, 3, -1, _HALF),  # 4 and 5 leave a one-unit sliver once the window is off the y-axis
}
_CANDIDATES = (1, 2, 3, 4, 5, -1, _HALF)  # every n the skill can draw


def _cls(n, end: int) -> str:
    """The class of y = x^n as x -> end * infinity (end is +1 or -1)."""
    if n < 0:
        return APP
    if end > 0:
        return INC
    check(n != _HALF, "n = 1/2 has no left end")
    return DEC if n % 2 == 1 else INC


def _f(n):
    if n == _HALF:
        return math.sqrt  # raises ValueError for x < 0, so the piece breaks
    if n < 0:
        return lambda x: x**-1  # ZeroDivisionError at x = 0
    return lambda x: x ** int(n)


# ---------------------------------------------------------------- the prompt

_FAMILY = ("y = x^n", r"\(y = x^n\)")
_NSET = {
    "easy": ("n is a positive whole number or n = -1", r"\(n\) is a positive whole number or \(n = -1\)"),
    "wide": (
        "n is a positive whole number, n = -1 or n = 1/2",
        r"\(n\) is a positive whole number, \(n = -1\) or \(n = \tfrac{1}{2}\)",
    ),
}
_END = {
    1: ("x tends to infinity", r"\(x \to \infty\)"),
    -1: ("x tends to negative infinity", r"\(x \to -\infty\)"),
}

_OPEN = [
    "The graph below is the curve {fam} for one value of n, where {nset}.",
    "A graph of {fam} is drawn on the grid, for one value of n with {nset}.",
    "The grid shows the curve {fam} for one value of n, where {nset}.",
    "Here is the graph of {fam} for a single value of n ({nset}).",
]
_ASK = [
    "What happens to y as {end}?",
    "Describe the behaviour of y as {end}.",
    "Which statement describes y as {end}?",
    "Choose the description of the behaviour of y as {end}.",
]
_CONTINUE = [
    "Assume the curve continues beyond the window in the same way.",
    "Take the curve to carry on past the edges of the grid in the same way.",
    "The curve continues beyond the window in the same way.",
]
_NOTES = {
    "easy": [
        "Each grid square is 1 unit on both axes.",
        "One grid square is 1 unit across and 1 unit up.",
        "Both axes have 1 unit per grid square.",
    ],
    "medium": [
        "Each grid square is 1 unit on both axes.",
        "One grid square is 1 unit across and 1 unit up.",
        "Both axes have 1 unit per grid square.",
    ],
    "hard": [
        "Only the part of the curve on the side of the y-axis where x is {side} is drawn. Each grid square is 1 unit.",
        "The window shows just the part of the curve where x is {side}, with 1 unit per grid square.",
        "Only the part of the graph for {side} x is drawn, one grid square being 1 unit each way.",
    ],
    "exam": [
        "Only the part of the curve where x is {side} is drawn, and the y-axis is not in the window, so read "
        "the values from the numbers along the edges of the grid. Each grid square is 1 unit.",
        "The window shows just a piece of the curve where x is {side}. No y-axis is drawn: use the numbers on "
        "the edges of the grid, where one square is 1 unit each way.",
        "The window is away from the y-axis and shows only the curve where x is {side}; use the numbers "
        "along its edges. One grid square is 1 unit on both axes.",
    ],
}
_SIDE = {1: "positive", -1: "negative"}


def _prompt(rng, tier: str, end: int) -> tuple[str, str]:
    """(text, latex): one sentence choice per slot, the mathematics delimited in the LaTeX twin."""
    opening, ask, cont = rng.choice(_OPEN), rng.choice(_ASK), rng.choice(_CONTINUE)
    note = rng.choice(_NOTES[tier])
    nset = _NSET["easy" if tier == "easy" else "wide"]
    text, latex = (
        " ".join(
            (
                opening.format(fam=_FAMILY[i], nset=nset[i]),
                note.format(side=_SIDE[end]),
                cont,
                ask.format(end=_END[end][i]),
            )
        )
        for i in (0, 1)
    )
    return text, latex


# ---------------------------------------------------------------- drawing the answer first


def _pairs(tier: str, cls: str) -> list[tuple]:
    out = []
    for n in _POOL[tier]:
        for end in (1, -1):
            if n == _HALF and end < 0:
                continue
            if _cls(n, end) == cls:
                out.append((n, end))
    return out


def _window(rng, tier: str, n, end: int):
    """(x_range, y_range): twelve squares each way, the y-window always holds the x-axis."""
    y0 = rng.choice((-7, -6, -5, -4))
    yr = (y0, y0 + cf.SPAN)
    if tier in ("easy", "medium"):
        x0 = rng.choice((-7, -6, -5))
        return (x0, x0 + cf.SPAN), yr
    shift = 0 if tier == "hard" else (1 if n in (2, 3) else rng.choice((1, 2)))
    if end > 0:
        return (shift, shift + cf.SPAN), yr
    return (-shift - cf.SPAN, -shift), yr


def _build_figure(n, x_range, y_range) -> dict:
    return cf.plane(x_range=x_range, y_range=y_range, alt=_ALT, curves=[{"pieces": cf.pieces(_f(n), x_range, y_range)}])


# ---------------------------------------------------------------- reading a figure


def _end_points(fig: dict, end: int):
    """Every sampled point on the named side of the y-axis, ordered from the axis outward."""
    pts = [p for pc in cf.curve_pieces(fig) for p in pc if p[0] * end > 0]
    return sorted(pts, key=lambda p: abs(p[0]))


def _exponent(fig: dict) -> object:
    """The n of y = x^n, fitted from the sampled points alone (log-log slope), snapped to a candidate."""
    pts = [p for pc in cf.curve_pieces(fig) for p in pc if abs(p[0]) > 0.05 and abs(p[1]) > 0.02]
    check(len(pts) >= 4, "power reader: too few points to read an exponent")
    (xa, ya), (xb, yb) = min(pts, key=lambda p: abs(p[0])), max(pts, key=lambda p: abs(p[0]))
    check(abs(xb) > 1.3 * abs(xa), "power reader: the two samples are too close to fit an exponent")
    fit = math.log(abs(yb / ya)) / math.log(abs(xb / xa))
    n = min(_CANDIDATES, key=lambda c: abs(float(c) - fit))
    check(abs(float(n) - fit) < 0.03, "power reader: the exponent is not one of the family")
    return n


def read_end_behaviour(question) -> str:
    """Re-derive the option text from the figure (the exponent) and the end the prompt names."""
    n = _exponent(question.figure)
    end = -1 if "negative infinity" in question.text else 1
    return _cls(n, end)


figures.register_reader(TOPIC, read_end_behaviour)


def _verify(fig: dict, n, end: int, cls: str, x_range, y_range) -> None:
    got = _exponent(fig)
    check(got == n, "power: figure reader disagrees with the drawn exponent")
    check(_cls(got, end) == cls, "power: the drawn curve is not of the class asked")
    pts = _end_points(fig, end)
    inner, outer = pts[0], pts[-1]
    if cls == APP:
        # the far end must hug the x-axis closely enough to read as levelling off
        check(abs(outer[1]) <= 0.21 and abs(outer[1]) < abs(inner[1]), "power: the far end is not level")
        check(abs(outer[0]) >= 4.5, "power: the window ends too near the origin for a limit")
    elif cls == INC:
        check(outer[1] > inner[1] + 0.5, "power: the end of the curve does not rise")
        check(n != _HALF or outer[1] > pts[len(pts) // 2][1], "power: the root curve stops rising")
    else:
        check(outer[1] < inner[1] - 0.5, "power: the end of the curve does not fall")
    check(len(cf.curve_pieces(fig)) >= 1, "power: nothing is drawn")


# ---------------------------------------------------------------- distractors

_NOTE_ID = {
    "wrong-end": "reads-the-other-end",
    "left-to-right": "reads-left-to-right-at-the-left-end",
    "bend": "reads-a-bend-as-levelling-off",
    "app-falls": "reads-approach-as-falling",
    "app-rises": "reads-the-branch-beside-the-axis-as-the-end",
}


def _note(cls: str, label: str, end: int) -> str:
    """The misconception behind offering ``label`` when the key is ``cls``."""
    if cls == APP:
        return _NOTE_ID["app-falls" if label == DEC else "app-rises"]
    if label == APP:
        return _NOTE_ID["bend"]
    return _NOTE_ID["left-to-right" if end < 0 else "wrong-end"]


def _n_text(n) -> str:
    return "x^(1/2) = sqrt(x)" if n == _HALF else ("x^(-1) = 1/x" if n == -1 else ("x" if n == 1 else f"x^{n}"))


def _working(n, end: int, cls: str) -> list[str]:
    first = f"The curve is y = {_n_text(n)}."
    short = "x" if n == 1 else (f"x^{n}" if n > 0 else "")
    if cls == APP:
        side = "positive" if end > 0 else "negative"
        return [
            first,
            f"As x becomes large and {side}, 1/x gets closer and closer to 0 without reaching it.",
            "So y approaches a fixed value, 0.",
        ]
    if cls == DEC:
        return [
            first,
            f"For large negative x, {short} is negative (an odd power) and its size keeps growing.",
            "So y decreases without bound.",
        ]
    if end < 0:
        why = f"for large negative x, {short} is positive (an even power) and its size keeps growing"
    elif n == _HALF:
        why = "the square root of x keeps growing as x grows, though more and more slowly"
    else:
        why = f"{short} keeps growing as x grows"
    return [first, f"{why[0].upper()}{why[1:]}.", "So y increases without bound."]


# ---------------------------------------------------------------- building


@generator(TOPIC, level=13, presets={t: {"tier": t} for t in _TIERS})
def pv1e_power_end_behaviour(rng, tier: str = "easy", notes: bool = False) -> dict:
    """End behaviour of y = x^n from its graph. Class first, then (n, end), then the window."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    cls = rng.choice(_LABELS)
    n, end = rng.choice(_pairs(tier, cls))
    x_range, y_range = _window(rng, tier, n, end)
    fig = _build_figure(n, x_range, y_range)
    _verify(fig, n, end, cls, x_range, y_range)

    distractors = [c for c in _LABELS if c != cls]
    check(len(set([cls, *distractors])) == 3, "power: the options must be three distinct labels")
    figures.assert_no_leak(fig, [cls, *distractors])
    rng.shuffle(distractors)

    text, latex = _prompt(rng, tier, end)
    out = {
        "text": text,
        "latex": latex,
        "answer": cls,
        "answer_text": cls,
        "answer_latex": latex_answer(cls, plain=True),
        "distractors": distractors,
        "working": _working(n, end, cls),
        "figure": fig,
    }
    if notes:
        out["option_notes"] = {d: _note(cls, d, end) for d in distractors}
    return out
