"""End behaviour of y = x^n read off its graph (ACMMM014).

Skill ``senior-recognise-features-of-the-graphs-of-y-x-acmmm014``, topic ``pv1e_power_end_behaviour``.
The learner sees the curve y = x^n for one hidden n (a positive whole number, n = -1 or n = 1/2) on a
labelled integer grid and is asked what y does as x tends to infinity OR as x tends to negative
infinity, or (hyperbola only, from the medium tier up) as x tends to 0 from the right OR from the left:
the QUESTION names the direction. Three options, always:
``increases without bound``, ``decreases without bound``, ``approaches a fixed value``. n = 1/2 has
domain x >= 0, so it is drawn and asked only at +infinity.

Key classes by (n, direction asked):

========  ==========  ==========  ==========  ==========
n         +infinity   -infinity   0 from the  0 from the
                                  right       left
========  ==========  ==========  ==========  ==========
odd       increases   DECREASES   (never)     (never)
even      increases   increases   (never)     (never)
-1        approaches  approaches  INCREASES   DECREASES
1/2       increases   (never)     (never)     (never)
========  ==========  ==========  ==========  ==========

So the hyperbola can carry all three answers: the far ends level off, but beside the y-axis it runs
up on the right and down on the left. The two near-zero questions are the only ones where the
asymptote x = 0 is the point of the question.

The CLASS is drawn first, uniformly, then one (n, direction) of that class, then the window, so each
label is the key about a third of the time. Nothing about n, the class or the direction's answer is
printed.

Tiers differ in the learner's DECISION:

- easy   -- n is 1, 2, 3 or -1 in a centred window with both axes drawn, so both ends of the curve are
            in view; the learner reads the end the question names, and a left-end question on an odd
            power is the one that falls to the lower left. Only the two infinity ends are asked.
- medium -- adds n = 4, 5 (flat near the origin, then steep) and n = 1/2, which bends like a curve that is
            levelling off yet keeps rising: the learner must not read a bend as a limit. The
            hyperbola is also asked near x = 0, where the level far end is the wrong place to look and
            the side of the y-axis (right rises, left falls) decides the answer.
- hard   -- only the half of the curve on the named side of the y-axis is drawn, so there is no mirror
            half to compare with: the learner has to know the family, and a left-end even power then
            falls toward the origin when read left to right but increases as x -> -infinity. A
            near-zero question on the hyperbola shows only the one branch, which is then read against
            the y-axis along the edge of the window.
- exam   -- that half-curve is also shifted 1 or 2 units away from the y-axis, so no y-axis is drawn and
            the curve is a bare fragment read against the numbers along the edges of the grid. The
            near-zero question cannot be shifted (the curve has to reach the y-axis to be read there),
            so it keeps the y-axis on the window's edge and sends the learner to the edge numbers.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .. import figures
from ..core import check, generator, latex_answer, latex_from_inline
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

# The direction a question names: x -> +infinity, x -> -infinity, x -> 0 from the right, x -> 0 from the left.
PLUS, MINUS, ZERO_R, ZERO_L = "+inf", "-inf", "0+", "0-"
_ASKS = (PLUS, MINUS, ZERO_R, ZERO_L)
_ZERO_TIERS = ("medium", "hard", "exam")  # easy keeps to the two infinity ends


def _side(ask: str) -> int:
    """+1 when the question is about the right of the y-axis, -1 for the left."""
    return 1 if ask in (PLUS, ZERO_R) else -1


def _cls(n, ask) -> str:
    """The class of y = x^n in the direction ``ask``."""
    if ask in (ZERO_R, ZERO_L):
        check(n == -1, "only the hyperbola is asked near x = 0")
        return INC if ask == ZERO_R else DEC
    end = _side(ask)
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
    PLUS: ("x tends to infinity", r"\(x \to \infty\)"),
    MINUS: ("x tends to negative infinity", r"\(x \to -\infty\)"),
    ZERO_R: ("x tends to 0 from the right", r"\(x \to 0^{+}\)"),
    ZERO_L: ("x tends to 0 from the left", r"\(x \to 0^{-}\)"),
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
_EXAM_EDGE = [
    "Only the part of the curve where x is {side} is drawn, and the y-axis is the edge of the window, so "
    "read the values from the numbers along the edges of the grid. Each grid square is 1 unit.",
    "The window shows just the part of the curve where x is {side}, with the y-axis along its edge. Use the "
    "numbers along the edges of the grid, where one square is 1 unit each way.",
    "Only the curve where x is {side} is drawn, the y-axis being the edge of the window; use the numbers "
    "along the edges. One grid square is 1 unit on both axes.",
]
_SIDE = {1: "positive", -1: "negative"}


def _prompt(rng, tier: str, ask: str) -> tuple[str, str]:
    """(text, latex): one sentence choice per slot, the mathematics delimited in the LaTeX twin."""
    opening, ask_, cont = rng.choice(_OPEN), rng.choice(_ASK), rng.choice(_CONTINUE)
    note = rng.choice(_EXAM_EDGE if tier == "exam" and ask in (ZERO_R, ZERO_L) else _NOTES[tier])
    nset = _NSET["easy" if tier == "easy" else "wide"]
    text, latex = (
        " ".join(
            (
                opening.format(fam=_FAMILY[i], nset=nset[i]),
                note.format(side=_SIDE[_side(ask)]),
                cont,
                ask_.format(end=_END[ask][i]),
            )
        )
        for i in (0, 1)
    )
    return text, latex


# ---------------------------------------------------------------- drawing the answer first


def _pairs(tier: str, cls: str) -> list[tuple]:
    out = []
    for n in _POOL[tier]:
        for ask in _ASKS:
            if n == _HALF and ask == MINUS:
                continue
            if ask in (ZERO_R, ZERO_L) and (n != -1 or tier not in _ZERO_TIERS):
                continue
            if _cls(n, ask) == cls:
                out.append((n, ask))
    return out


_NEAR_ZERO_SHARE = 0.2  # of the increasing / decreasing questions, where the tier allows: the hyperbola near x = 0


def _pick(rng, tier: str, cls: str) -> tuple:
    """One (n, direction) of the class. The near-zero questions are drawn as a group, so the right and the
    left appear equally often instead of in proportion to how many other pairs share their class."""
    pairs = _pairs(tier, cls)
    near = [p for p in pairs if p[1] in (ZERO_R, ZERO_L)]
    rest = [p for p in pairs if p not in near]
    if near and rng.random() < _NEAR_ZERO_SHARE:
        return rng.choice(near)
    return rng.choice(rest)


def _window(rng, tier: str, n, ask: str):
    """(x_range, y_range): twelve squares each way, the y-window always holds the x-axis."""
    y0 = rng.choice((-7, -6, -5, -4))
    yr = (y0, y0 + cf.SPAN)
    if tier in ("easy", "medium"):
        x0 = rng.choice((-7, -6, -5))
        return (x0, x0 + cf.SPAN), yr
    # near x = 0 the curve has to reach the y-axis, so those windows start at it
    shift = 0 if tier == "hard" or ask in (ZERO_R, ZERO_L) else (1 if n in (2, 3) else rng.choice((1, 2)))
    if _side(ask) > 0:
        return (shift, shift + cf.SPAN), yr
    return (-shift - cf.SPAN, -shift), yr


_NEAR = 0.25  # within this of the y-axis the hyperbola is sampled finely, so it is drawn right up the axis


def _far(a, b):
    return a, b, int(round((b - a) * 8)) + 1


def _hyperbola_pieces(x_range, y_range) -> list:
    """y = 1/x, sampled every 1/8 and every 0.005 within 1/4 of x = 0, so a branch visibly runs along the y-axis.

    Every x is a multiple of 0.005 (three decimals), so rounding x in the spec costs nothing: the point
    still satisfies y = 1/x to the spec's own rounding."""
    zones = []
    for lo, hi in ((x_range[0], min(x_range[1], 0)), (max(x_range[0], 0), x_range[1])):
        if lo >= hi:
            continue
        if hi == 0:
            zones += [_far(lo, -_NEAR), (-_NEAR, 0, 51)] if lo < -_NEAR else [(lo, 0, 51)]
        elif lo == 0:
            zones += [(0, _NEAR, 51), _far(_NEAR, hi)] if hi > _NEAR else [(0, hi, 51)]
        else:
            zones.append(_far(lo, hi))
    out: list = []
    for a, b, k in zones:
        for piece in cf.zoned_pieces(_f(-1), [(a, b)], y_range, n_zone=k):
            if out and out[-1][-1] == piece[0]:
                out[-1].extend(piece[1:])
            else:
                out.append(list(piece))
    return out


def _build_figure(n, x_range, y_range) -> dict:
    pcs = _hyperbola_pieces(x_range, y_range) if n == -1 else cf.pieces(_f(n), x_range, y_range)
    return cf.plane(x_range=x_range, y_range=y_range, alt=_ALT, curves=[{"pieces": pcs}])


# ---------------------------------------------------------------- reading a figure


def _end_points(fig: dict, end: int):
    """Every sampled point on the named side of the y-axis (+1 right, -1 left), ordered from the axis outward."""
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


def ask_of(text: str) -> str:
    """The direction a prompt names."""
    if "from the right" in text:
        return ZERO_R
    if "from the left" in text:
        return ZERO_L
    return MINUS if "negative infinity" in text else PLUS


def read_end_behaviour(question) -> str:
    """Re-derive the option text from the figure (the exponent) and the end the prompt names."""
    n = _exponent(question.figure)
    return _cls(n, ask_of(question.text))


figures.register_reader(TOPIC, read_end_behaviour)


def _verify(fig: dict, n, ask: str, cls: str, x_range, y_range) -> None:
    got = _exponent(fig)
    check(got == n, "power: figure reader disagrees with the drawn exponent")
    check(_cls(got, ask) == cls, "power: the drawn curve is not of the class asked")
    pts = _end_points(fig, _side(ask))
    inner, outer = pts[0], pts[-1]
    if ask in (ZERO_R, ZERO_L):
        # the branch must visibly run up (or down) the y-axis to the edge of the window
        edge = y_range[1] if cls == INC else y_range[0]
        check(abs(inner[0]) <= 0.3, "power: the curve stops short of the y-axis")
        short = edge - inner[1] if cls == INC else inner[1] - edge  # how far inside the edge the curve stops
        check(short <= 0.3, "power: the curve does not reach the top or bottom of the window")
        check(abs(inner[1]) >= 4, "power: the near-axis branch is too short to read")
        check(abs(outer[1]) < abs(inner[1]) - 2, "power: the branch does not turn away from the axis")
    elif cls == APP:
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
    "far-end": "reads-the-far-end-instead-of-the-axis",
    "other-side": "reads-the-other-side-of-the-axis",
}


def _note(cls: str, label: str, ask: str) -> str:
    """The misconception behind offering ``label`` when the key is ``cls``."""
    end = _side(ask)
    if ask in (ZERO_R, ZERO_L):
        return _NOTE_ID["far-end"] if label == APP else _NOTE_ID["other-side"]
    if cls == APP:
        return _NOTE_ID["app-falls" if label == DEC else "app-rises"]
    if label == APP:
        return _NOTE_ID["bend"]
    return _NOTE_ID["left-to-right" if end < 0 else "wrong-end"]


def _n_text(n) -> str:
    return "x^(1/2) = sqrt(x)" if n == _HALF else ("x^(-1) = 1/x" if n == -1 else ("x" if n == 1 else f"x^{n}"))


def _working(n, ask: str, cls: str) -> list[str]:
    end = _side(ask)
    first = f"The curve is y = {_n_text(n)}."
    if ask in (ZERO_R, ZERO_L):
        side, sign, verb = ("right", "positive", "increases") if ask == ZERO_R else ("left", "negative", "decreases")
        return [
            first,
            f"As x gets closer and closer to 0 from the {side}, x stays {sign} and small, "
            f"so 1/x becomes large and {sign}.",
            f"So y {verb} without bound.",
        ]
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
    n, ask = _pick(rng, tier, cls)
    x_range, y_range = _window(rng, tier, n, ask)
    fig = _build_figure(n, x_range, y_range)
    _verify(fig, n, ask, cls, x_range, y_range)

    distractors = [c for c in _LABELS if c != cls]
    check(len(set([cls, *distractors])) == 3, "power: the options must be three distinct labels")
    figures.assert_no_leak(fig, [cls, *distractors])
    rng.shuffle(distractors)

    text, latex = _prompt(rng, tier, ask)
    out = {
        "text": text,
        "latex": latex_from_inline(latex),
        "answer": cls,
        "answer_text": cls,
        "answer_latex": latex_answer(cls, plain=True),
        "distractors": distractors,
        "working": _working(n, ask, cls),
        "figure": fig,
    }
    if notes:
        out["option_notes"] = {d: _note(cls, d, ask) for d in distractors}
    return out
