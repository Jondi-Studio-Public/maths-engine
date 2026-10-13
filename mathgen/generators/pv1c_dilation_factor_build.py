"""Generates `pv1c_dilation_factor` questions: find the factor c from the graphs of y = f(x) and y = c f(x)
(ACMMM026).

Skill ``senior-examine-dilations-and-the-graphs-of-y-cf``, topic ``pv1c_dilation_factor``.
The learner sees two curves on one integer grid and two marked points at the same stated x:
P (on y = f(x)) and Q (on y = c f(x)). c is never printed; it is the ratio of the two heights.

Answer first. c is drawn uniformly from the tier's set, then a height m = |f(a)| is chosen so
that f(a) and c f(a) are both grid integers inside the window, then a curve f (parabola,
cubic or V shape with exact ``Fraction`` coefficients) and an integer x = a with that f(a).
Both curves are sampled at step 1/16, so x = a is an exact sample.

Tiers differ in the learner's DECISION:

- easy   -- c a whole number above 1: Q is c times as high as P, so divide the two heights.
- medium -- c a fraction (compression, or a non-integer stretch): the ratio is Q / P, not P / Q.
- hard   -- c negative: P is above the x axis and Q below it, so the sign of c is decided.
- exam   -- P is BELOW the x axis: c is the signed ratio of the heights, so with both points
            below the axis c is positive although the heights are negative, and a Q above the
            axis means c is negative; comparing sizes ("Q is lower, so negative") is wrong.
"""

from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
from types import SimpleNamespace
from typing import NamedTuple

from .. import figures
from ..core import check, check_pairwise_distinct, generator, paren_negative, rational_text, latex_from_inline

TOPIC = "pv1c_dilation_factor"

XLO, XHI = -6, 6
YLO, YHI = -8, 8
N_SAMPLES = 193  # step 12 / 192 = 1/16, so every integer x is an exact sample
QMAX = 7  # |f(a)| and |c f(a)| never exceed this (the window is +-8)

_EASY_C = tuple(F(n) for n in (2, 3, 4, 5, 6))
_MEDIUM_C = (F(1, 4), F(1, 3), F(1, 2), F(2, 3), F(3, 4), F(3, 2))
_HARD_C = (F(-1), F(-2), F(-3), F(-1, 2), F(-1, 3), F(-3, 2), F(-2, 3))
_EXAM_C = tuple(s * v for v in (F(1, 2), F(1, 3), F(2, 3), F(3, 2), F(2), F(3)) for s in (1, -1))

#: sign of f(a) in the exam tier (-1: P below the axis). The build test flips it to prove the
#: sign property red against the unconstrained (positive) draw.
_EXAM_SIGN = -1

_ALT = "two curves on a coordinate grid"

# ---------------------------------------------------------------- exact curves


class Entry(NamedTuple):
    kind: str  # "parab" | "cubic" | "abs"
    par: tuple  # Fractions
    a: int
    fa: int
    ys: tuple  # f(x) at integer x in [XLO, XHI], floats (visibility screen only)


def _ev(kind, par, x):
    """f(x). Works for Fraction x (exact) and float x (sampling)."""
    if kind == "parab":
        s, h, k = par
        return s * (x - h) * (x - h) + k
    if kind == "cubic":
        s, p, q, r = par
        return s * (x - p) * (x - q) * (x - r)
    s, h, k = par
    return s * abs(x - h) + k


_S_PARAB = tuple(F(n, d) * s for n, d in ((1, 4), (1, 3), (1, 2), (1, 1), (2, 1)) for s in (1, -1))
_S_CUBIC = tuple(F(n, d) * s for n, d in ((1, 6), (1, 4), (1, 3), (1, 2), (1, 1)) for s in (1, -1))
_S_ABS = tuple(F(n, d) * s for n, d in ((1, 3), (1, 2), (1, 1), (2, 1), (3, 1)) for s in (1, -1))


def _mk(kind, par, a, fa) -> Entry:
    ys = tuple(float(_ev(kind, par, F(x))) for x in range(XLO, XHI + 1))
    return Entry(kind, par, a, int(fa), ys)


@lru_cache(maxsize=None)
def _pool() -> dict:
    """f(a) -> entries. f(a) is a nonzero integer of magnitude <= QMAX; enough of the curve is
    inside the window (a float screen that only decides which curves are shown)."""
    out: dict[int, list] = {}

    def add(kind, par):
        for a in range(-5, 6):
            fa = _ev(kind, par, F(a))
            if fa.denominator != 1 or fa == 0 or abs(fa) > QMAX:
                continue
            e = _mk(kind, par, a, fa)
            if sum(1 for y in e.ys if abs(y) <= YHI) >= 5:
                out.setdefault(int(fa), []).append(e)

    for s in _S_PARAB:
        for h in range(-4, 5):
            for k in range(-5, 6):
                add("parab", (s, F(h), F(k)))
    for s in _S_CUBIC:
        for p in range(-3, 4):
            for q in range(p, 4):
                for r in range(q, 4):
                    add("cubic", (s, F(p), F(q), F(r)))
    for s in _S_ABS:
        for h in range(-4, 5):
            for k in range(-5, 6):
                add("abs", (s, F(h), F(k)))
    return out


def _shows_both(e: Entry, c: F) -> bool:
    """At least 3 integer x where the dilated curve is inside the window."""
    cf = float(c)
    return sum(1 for y in e.ys if abs(cf * y) <= YHI) >= 3


# ---------------------------------------------------------------- prompts
# {F} y = f(x), {G} y = cf(x), {C} c, {XA} x = a, {P} {Q} the labels.

_OPEN = [
    "The graphs of {F} and {G} are drawn on the same grid, where {C} is a constant.",
    "Two curves are shown: the graph of {F} and the graph of {G}, with {C} a constant.",
    "Below are the graphs of {F} and {G} for one particular value of {C}.",
    "Here are the curves {F} and {G} on one set of axes, where {C} is a fixed number.",
]

_COND = {
    "stretch": "{C} is a positive whole number greater than 1.",
    "compress_fraction": "{C} is positive and is not 1.",
    "negative": "{C} may be positive or negative, but it is not 1.",
    "below_axis": "{C} can be positive or negative, and is not 1.",
}

_MID = [
    "{P} lies on the graph of {F} and {Q} lies on the graph of {G}, both at {XA}.",
    "At {XA}, the point {P} is on the graph of {F} and the point {Q} is on the graph of {G}.",
    "Points {P} and {Q} are marked at {XA}: {P} on {F}, {Q} on {G}.",
]

_SCALE = [
    "Each grid square is 1 unit on both axes.",
    "Both axes have 1 unit per grid square.",
]

_ASK = ["State the value of {C}.", "What is {C}?", "Find {C}."]

_FILL_TEXT = {"F": "y = f(x)", "G": "y = cf(x)", "C": "c", "P": "P", "Q": "Q"}
_FILL_TEX = {"F": r"\(y = f(x)\)", "G": r"\(y = cf(x)\)", "C": r"\(c\)", "P": r"\(P\)", "Q": r"\(Q\)"}


def _pick_prompt(rng, variant: str) -> str:
    """The phrase pattern, chosen before the answer is drawn (the stated x is filled in later)."""
    return " ".join([rng.choice(_OPEN), _COND[variant], rng.choice(_MID), rng.choice(_SCALE), rng.choice(_ASK)])


def _fill_prompt(pattern: str, a: int) -> tuple[str, str]:
    text = pattern.format(XA=f"x = {a}", **_FILL_TEXT)
    latex = pattern.format(XA=rf"\(x = {a}\)", **_FILL_TEX)
    return text, latex


# ---------------------------------------------------------------- reading a figure


def _samples(curve) -> dict:
    return {float(x): float(y) for piece in curve["pieces"] for x, y in piece}


def _point(fig, label):
    hits = [p for p in fig["points"] if p.get("label") == label]
    check(len(hits) == 1, f"dilation reader: need exactly one point labelled {label}")
    x, y = hits[0]["at"]
    return x, y


def _on(curve, x, y) -> bool:
    return abs(_samples(curve).get(float(x), 1e9) - y) < 1e-6


def read_dilation_factor(question) -> str:
    """Re-derive c from the figure alone. P (on y = f(x)) and Q (on y = c f(x)) share their x;
    c = Qy / Py. Cross-check on the curves: the curve through Q is the curve through P times
    c at every common sample where f is not near zero."""
    fig = question.figure
    check(len(fig["curves"]) == 2, "dilation reader: expected two curves")
    (px, py), (qx, qy) = _point(fig, "P"), _point(fig, "Q")
    check(px == qx and py != 0, "dilation reader: P and Q must share x and P not on the axis")
    c = F(qy) / F(py)
    on_p = [cv for cv in fig["curves"] if _on(cv, px, py)]
    on_q = [cv for cv in fig["curves"] if _on(cv, qx, qy)]
    check(
        len(on_p) == 1 and len(on_q) == 1 and on_p[0] is not on_q[0],
        "dilation reader: P and Q must lie on different curves",
    )
    sp, sq = _samples(on_p[0]), _samples(on_q[0])
    ratios = [sq[x] / sp[x] for x in sp if x in sq and abs(sp[x]) >= 0.5]
    check(len(ratios) >= 3, "dilation reader: too few common samples")
    check(all(abs(r - float(c)) < 1e-4 for r in ratios), "dilation reader: curves disagree with Q / P")
    return rational_text(c)


figures.register_reader(TOPIC, read_dilation_factor)

# ---------------------------------------------------------------- verification


def _sample_fn(e: Entry, c: F):
    """The curve y = c f(x) as a float function (c = 1 gives f itself)."""
    cf = float(c)
    return lambda x: cf * float(_ev(e.kind, e.par, x))


def _verify(fig, e: Entry, c: F, qy: int) -> None:
    """Independent checks on a built question (raises VerificationError)."""
    a = e.a
    check(_ev(e.kind, e.par, F(a)) == e.fa and c * e.fa == qy, "dilation: heights do not satisfy Q = c P")
    check(c not in (0, 1) and e.fa != 0, "dilation: degenerate factor")
    check(abs(e.fa) <= QMAX and abs(qy) <= QMAX, "dilation: a point is outside the window")
    check(
        read_dilation_factor(SimpleNamespace(figure=fig)) == rational_text(c),
        "dilation: figure reader disagrees with the key",
    )


# ---------------------------------------------------------------- distractors

_PRIORITY = {
    "stretch": ("reciprocal", "extra-not-total", "difference", "height-of-q"),
    "compress_fraction": ("reciprocal", "extra-not-total", "difference", "height-of-q"),
    "negative": ("sign-flipped", "reciprocal", "difference", "height-of-q"),
    "below_axis": ("sign-flipped", "reciprocal", "difference", "height-of-q"),
}


def _distractors(variant: str, c: F, py: int, qy: int) -> list[tuple[str, F]]:
    cands = {
        "reciprocal": 1 / c,
        "sign-flipped": -c,
        "difference": F(abs(qy - py) if variant in ("stretch", "compress_fraction") else qy - py),
        "height-of-q": F(qy),
        "extra-not-total": (c - 1 if c > 1 else 1 - c) if c != F(1, 2) else c + 1,
    }  # c = 1/2: 1 - c is c itself
    out, seen = [], {c}
    for name in _PRIORITY[variant]:
        v = cands[name]
        if v in seen:
            continue
        seen.add(v)
        out.append((name, v))
    return out[:3]


# ---------------------------------------------------------------- building


def _heights(variant: str, c: F) -> list[int]:
    top = 2 if variant == "stretch" else QMAX
    return [m for m in range(1, top + 1) if (c * m).denominator == 1 and abs(c * m) <= QMAX]


def _tier_c(variant: str):
    return {"stretch": _EASY_C, "compress_fraction": _MEDIUM_C, "negative": _HARD_C, "below_axis": _EXAM_C}[variant]


def _fraction_dict(v: F):
    return int(v) if v.denominator == 1 else {"p": v.numerator, "q": v.denominator}


def _latex_num(v: F) -> str:
    if v.denominator == 1:
        return str(v.numerator)
    return ("-" if v < 0 else "") + rf"\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"


@generator(
    TOPIC,
    level=13,
    presets={
        "easy": {"variant": "stretch"},
        "medium": {"variant": "compress_fraction"},
        "hard": {"variant": "negative"},
        "exam": {"variant": "below_axis"},
    },
)
def pv1c_dilation_factor(rng, variant: str = "stretch") -> dict:
    """c read off the graphs of y = f(x) and y = c f(x). Answer first: draw c, then f and a."""
    if variant not in _COND:
        raise ValueError(f"unknown variant {variant!r}")
    pool = _pool()
    sign = _EXAM_SIGN if variant == "below_axis" else 1
    cs = _tier_c(variant)
    pattern = _pick_prompt(rng, variant)
    c = rng.choice(cs)  # the answer is drawn first and never redrawn (uniform over cs)
    heights = _heights(variant, c)
    rng.shuffle(heights)
    for m in heights:
        fa = sign * m
        entries = [x for x in pool[fa] if _shows_both(x, c)]
        if entries:
            e = rng.choice(entries)
            break
    else:
        raise RuntimeError(f"{TOPIC}/{variant}: could not build a question for c = {c}")
    qy = int(c * fa)
    a = e.a
    fig = figures.cartesian(
        x_range=(XLO, XHI),
        y_range=(YLO, YHI),
        alt=_ALT,
        equal_scale=True,
        points=[{"at": [a, fa], "label": "P"}, {"at": [a, qy], "label": "Q"}],
        curves=[
            {"pieces": figures.sample_curve(_sample_fn(e, F(1)), XLO, XHI, (YLO, YHI), n=N_SAMPLES)},
            {"pieces": figures.sample_curve(_sample_fn(e, c), XLO, XHI, (YLO, YHI), n=N_SAMPLES)},
        ],
    )
    _verify(fig, e, c, qy)

    dist = _distractors(variant, c, fa, qy)
    answer_text = rational_text(c)
    texts = [rational_text(v) for _, v in dist]
    check_pairwise_distinct(c, *[v for _, v in dist])
    figures.assert_no_leak(fig, [answer_text, *texts])
    text, latex = _fill_prompt(pattern, a)
    working = [
        f"P is at height {fa} and Q is at height {qy} at x = {a}.",
        f"Q = c × P, so c = Q ÷ P = {qy} ÷ {paren_negative(fa)} = {answer_text}.",
    ]
    return {
        "text": text,
        "latex": latex_from_inline(latex),
        "answer": _fraction_dict(c),
        "answer_text": answer_text,
        "answer_latex": _latex_num(c),
        "distractors": texts,
        "working": working,
        "figure": fig,
    }
