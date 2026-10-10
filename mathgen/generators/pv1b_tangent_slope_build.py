"""Generates `pv1b_tangent_slope` figure questions: read the slope of the tangent to y = f(x) as the derivative.

Skill ``senior-interpret-the-derivative-as-the-slope-or-gradient``, topic
``pv1b_tangent_slope``. The learner sees the graph of y = f(x) (f is never printed) and a
tangent line drawn at a stated x, and types f'(a): the gradient of the drawn tangent, read
off the grid as rise over run.

Answer first. A slope is drawn from the tier's pool, then a curve is chosen whose tangent at
some integer a has exactly that slope and whose touch point (a, f(a)) is a grid intersection.
Every curve is a polynomial with exact ``Fraction`` coefficients (a parabola
k + c(x-h)^2 or a cubic (x-p)(x-q)(x-r)/s), so the slope is exact, and it is sampled at
193 points (step 1/16) so every integer x is a sample.

Tiers differ in the learner's DECISION:

- easy   -- parabola, tangent where the curve rises: rise over run of the drawn tangent.
- medium -- the slope may be negative or zero (a horizontal tangent), and fractional: the
            sign comes from the tangent's direction and a level line means 0.
- hard   -- a cubic; the tangent touches at the stated x and also cuts the curve again
            elsewhere, and the touch point is not marked: the slope of the drawn tangent is
            the answer, not anything read off the curve at the other crossing.
- exam   -- two lines are drawn through the point (a, f(a)), the tangent and a secant: the
            learner decides which one is the tangent (the one that touches the curve there,
            not the one cutting through it) before reading its gradient.
"""

from __future__ import annotations

import math
from fractions import Fraction as F
from functools import lru_cache
from types import SimpleNamespace
from typing import NamedTuple

from .. import figures
from ..core import check, check_pairwise_distinct, generator, paren_negative, phrase, rational_text

TOPIC = "pv1b_tangent_slope"

# Fixed window, unit gridlines, equal scale.
LO, HI = -6, 6
N_SAMPLES = 193  # step (HI - LO) / 192 = 1/16, so every integer x is a sample

_CS = tuple(F(n, d) * s for n, d in ((1, 6), (1, 4), (1, 2), (1, 1), (2, 1)) for s in (1, -1))

# ---------------------------------------------------------------- prompts

_EASY = [
    "The graph of y = f(x) is drawn below, with the tangent to the graph at x = {a}. "
    "Each grid square is 1 unit on both axes. What is f'({a})?",
    "The tangent to the graph of y = f(x) at x = {a} is drawn. Reading along the grid "
    "(one square is 1 unit on each axis), find the gradient of this tangent, which is f'({a}).",
    "Use the tangent drawn at x = {a} on the graph of y = f(x) to state the value of f'({a}). "
    "One grid square is 1 unit across and 1 unit up.",
    "A graph of y = f(x) and its tangent at x = {a} are shown. Both axes have 1 unit per grid square. State f'({a}).",
]

_HARD = [
    "The graph of y = f(x) is drawn with a tangent line at x = {a}. The line meets the graph "
    "at another point as well. One grid square is 1 unit on both axes. What is f'({a})?",
    "A line is drawn tangent to the graph of y = f(x) at x = {a}. Using 1 unit per grid "
    "square on both axes, state f'({a}).",
    "The graph of y = f(x) and its tangent at x = {a} are shown. The tangent also cuts the "
    "graph at a second point, which is not where it touches. Each grid square is 1 unit on "
    "both axes. Find f'({a}).",
    "Find f'({a}) for the graph of y = f(x) shown, given that the line drawn is the tangent "
    "at x = {a}. Both axes use 1 unit per grid square.",
]

_EXAM = [
    "Two straight lines are drawn through the point on the graph of y = f(x) where x = {a}. "
    "One of them is the tangent to the graph at that point. Each grid square is 1 unit on "
    "both axes. What is f'({a})?",
    "The graph of y = f(x) is shown with two lines that both pass through the point where "
    "x = {a}; only one is the tangent there. With 1 unit per grid square on both axes, "
    "state f'({a}).",
    "A tangent and a secant to the graph of y = f(x) are drawn through the point at "
    "x = {a}, but they are not labelled. Using 1 unit per grid square on both axes, find "
    "f'({a}).",
    "Two lines are drawn through the point on the graph of y = f(x) at x = {a}. Exactly one "
    "of them is the tangent at that point. Both axes have 1 unit per grid square. "
    "State the value of f'({a}).",
]

_PROMPTS = {"rising": _EASY, "sign_and_zero": _EASY, "cubic_crossing": _HARD, "tangent_or_secant": _EXAM}


def _tex(pattern: str) -> str:
    """The LaTeX twin of a prompt pattern: instructions stay prose, only the maths is delimited."""
    for plain in ("y = f(x)", "x = {a}", "f'({a})"):
        pattern = pattern.replace(plain, r"\(" + plain + r"\)")
    return pattern


_ALT_ONE = "the graph of a curve and a straight line on a coordinate grid"
_ALT_TWO = "the graph of a curve and two straight lines on a coordinate grid"

# ---------------------------------------------------------------- exact curves


class Entry(NamedTuple):
    co: tuple  # (c0, c1, c2, c3) as Fractions
    a: int
    fa: int
    m: F
    family: str
    x2: int | None  # where a cubic's tangent meets the curve again


def _ev(co, x):
    r = F(0)
    for c in reversed(co):
        r = r * x + c
    return r


def _visible(co, a) -> bool:
    """Float screen on the drawn curve: enough of it inside the window, and both samples
    next to the touch point (a +- 1/16) drawn. Only decides which curves are shown; never
    a value the key uses."""
    cf = [float(c) for c in co]

    def f(x):
        return ((cf[3] * x + cf[2]) * x + cf[1]) * x + cf[0]

    if sum(1 for x in range(LO, HI + 1) if abs(f(x)) <= HI) < 6:
        return False
    return abs(f(a - 0.0625)) <= HI and abs(f(a + 0.0625)) <= HI


@lru_cache(maxsize=None)
def _parabolas() -> tuple:
    out = []
    for c in _CS:
        for d in range(-6, 7):
            m = 2 * c * d
            if abs(m) > 6:
                continue
            for a in range(-4, 5):
                h = a - d
                for fa in range(-4, 5):
                    k = fa - c * d * d
                    co = (k + c * h * h, -2 * c * h, c, F(0))
                    if _visible(co, a):
                        out.append(Entry(co, a, fa, m, "parabola", None))
    return tuple(out)


@lru_cache(maxsize=None)
def _cubics() -> tuple:
    """(x-p)(x-q)(x-r)/s with f(a) an integer and a tangent that meets the curve again
    inside the window (second crossing x2 = p+q+r-2a, at least 2 away from a)."""
    out = []
    for p in range(-4, 5):
        for q in range(p, 5):
            for r in range(q, 5):
                e1, e2, e3 = p + q + r, p * q + p * r + q * r, p * q * r
                for s in (1, 2, 3, 4, 6, -1, -2, -3, -4, -6):
                    co = (F(-e3, s), F(e2, s), F(-e1, s), F(1, s))
                    for a in range(-4, 5):
                        fa = _ev(co, a)
                        if fa.denominator != 1 or abs(fa) > 4:
                            continue
                        m = co[1] + 2 * co[2] * a + 3 * co[3] * a * a
                        if abs(m) > 6 or m.denominator > 3:
                            continue
                        x2 = e1 - 2 * a
                        if abs(x2) > HI - 1 or abs(x2 - a) < 2 or abs(_ev(co, x2)) > HI:
                            continue
                        if _visible(co, a):
                            out.append(Entry(co, a, int(fa), m, "cubic", x2))
    return tuple(out)


def _group(entries, keep) -> dict:
    by = {}
    for e in entries:
        if keep(e):
            by.setdefault(e.m, []).append(e)
    return by


@lru_cache(maxsize=None)
def _pool(variant: str) -> dict:
    if variant == "rising":
        return _group(_parabolas(), lambda e: e.m.denominator == 1 and 1 <= e.m <= 6)
    if variant == "sign_and_zero":
        return _group(_parabolas(), lambda e: abs(e.m) <= 4 and e.m.denominator <= 3)
    if variant == "cubic_crossing":
        return _group(_cubics(), lambda e: e.m != 0 and abs(e.m) <= 4)
    if variant == "tangent_or_secant":
        return _group(_parabolas() + _cubics(), lambda e: abs(e.m) <= 4 and e.m.denominator <= 3)
    raise ValueError(f"unknown variant {variant!r}")


#: probability that the drawn slope is zero (a horizontal tangent), per variant
_ZERO_WEIGHT = {"sign_and_zero": 0.12, "tangent_or_secant": 0.06}

# ---------------------------------------------------------------- geometry


def _inside(x, y) -> bool:
    return LO <= x <= HI and LO <= y <= HI


def _tangent_points(rng, a, fa, m, far):
    """Two grid intersections on the tangent, inside the window, run 1..3 (``far``: 3..6)."""
    p, q = m.numerator, m.denominator
    allowed = range(3, 7) if far else range(1, 4)
    opts = []
    for t1 in range(-13, 13):
        x1, y1 = a + t1 * q, fa + t1 * p
        if not _inside(x1, y1):
            continue
        for t2 in range(t1 + 1, t1 + 7):
            dx = (t2 - t1) * q
            x2, y2 = a + t2 * q, fa + t2 * p
            if dx in allowed and _inside(x2, y2):
                opts.append(([x1, y1], [x2, y2]))
    if not opts:
        return None
    pts = rng.choice(opts)
    return pts if rng.random() < 0.5 else (pts[1], pts[0])


def _chord_slope(co, a, fa, m):
    """Slope from (a, f(a)) to the nearest other grid point on the curve (not on the tangent)."""
    for k in range(1, 13):
        for dx in (k, -k):
            x = a + dx
            if not LO <= x <= HI:
                continue
            y = _ev(co, x)
            if y.denominator != 1 or abs(y) > HI:
                continue
            s = (y - fa) / dx
            if s != m:
                return s
    return None


def _secants(co, a, fa, m):
    """Grid points b on the curve (b != a) whose chord from (a, f(a)) is visibly not the tangent."""
    out = []
    for b in range(LO, HI + 1):
        if b == a:
            continue
        y = _ev(co, b)
        if y.denominator != 1 or abs(y) > HI:
            continue
        s = (y - fa) / (b - a)
        if abs(math.atan(float(s)) - math.atan(float(m))) >= 0.25:
            out.append((b, int(y), s))
    return out


# ---------------------------------------------------------------- reading a figure


def _line_slope(ln) -> F:
    (x1, y1), (x2, y2) = ln["through"]
    return (F(y2) - F(y1)) / (F(x2) - F(x1))


def _curve_slope_at(fig, a):
    """Central difference of the drawn curve's samples at x = a (a sample by construction)."""
    for curve in fig["curves"]:
        for piece in curve["pieces"]:
            for i in range(1, len(piece) - 1):
                if abs(piece[i][0] - a) < 1e-9:
                    (xl, yl), (xr, yr) = piece[i - 1], piece[i + 1]
                    return (yr - yl) / (xr - xl)
    return None


def read_tangent_slope(question) -> str:
    """Re-derive the answer from the figure alone. One line: its gradient. Two lines: the one
    whose gradient matches the drawn curve's own slope at the marked point is the tangent."""
    fig = question.figure
    slopes = [_line_slope(ln) for ln in fig["lines"]]
    if len(slopes) == 1:
        return rational_text(slopes[0])
    a = fig["points"][0]["at"][0]
    d = _curve_slope_at(fig, a)
    check(d is not None, "tangent_slope reader: no curve sample at the marked point")
    return rational_text(min(slopes, key=lambda s: abs(float(s) - d)))


figures.register_reader(TOPIC, read_tangent_slope)

# ---------------------------------------------------------------- verification


def _verify(fig, co, a, fa, key: F, tangent) -> None:
    """Independent checks on a built question (raises VerificationError)."""
    # 1. central-difference route (exact for a polynomial of degree <= 3): the tangent slope.
    central = (_ev(co, a + 1) - _ev(co, a - 1)) / 2 - co[3]
    check(central == key, "tangent_slope: key is not the curve's slope at a")
    # 2. the drawn tangent: grid points inside the window, on the line through (a, f(a)) with the key slope.
    for x, y in tangent:
        check(
            _inside(x, y) and isinstance(x, int) and isinstance(y, int),
            "tangent_slope: tangent point is not a grid intersection in the window",
        )
        check(F(y) - fa == key * (x - a), "tangent_slope: tangent point is off the tangent line")
    # 3. the registered reader agrees with the key on the figure as built.
    check(
        read_tangent_slope(SimpleNamespace(figure=fig)) == rational_text(key),
        "tangent_slope: figure reader disagrees with the key",
    )
    # 4. the drawn curve passes through the touch point.
    hit = [p for c in fig["curves"] for pc in c["pieces"] for p in pc if p[0] == a]
    check(bool(hit) and hit[0][1] == fa, "tangent_slope: curve does not pass through (a, f(a))")


# ---------------------------------------------------------------- building

_PRIORITY = {
    "rising": ("reciprocal-slope", "chord-slope", "f-of-a", "x-coordinate-a", "sign-flipped"),
    "sign_and_zero": ("sign-flipped", "reciprocal-slope", "chord-slope", "f-of-a", "x-coordinate-a"),
    "cubic_crossing": ("chord-slope", "sign-flipped", "f-of-a", "reciprocal-slope", "x-coordinate-a"),
    "tangent_or_secant": ("chord-slope", "sign-flipped", "reciprocal-slope", "f-of-a", "x-coordinate-a"),
}


def _distractors(variant, e: Entry, secant_slope):
    cands = {
        "reciprocal-slope": (1 / e.m) if e.m != 0 else None,
        "sign-flipped": -e.m,
        "f-of-a": F(e.fa),
        "chord-slope": secant_slope if variant == "tangent_or_secant" else _chord_slope(e.co, e.a, e.fa, e.m),
        "x-coordinate-a": F(e.a),
    }
    out, seen = [], {e.m}
    for name in _PRIORITY[variant]:
        v = cands[name]
        if v is None or v in seen:
            continue
        seen.add(v)
        out.append((name, v))
    return out


def _try(rng, variant, e: Entry):
    far = variant == "cubic_crossing"
    tangent = _tangent_points(rng, e.a, e.fa, e.m, far)
    if tangent is None:
        return None
    secant = None
    if variant == "tangent_or_secant":
        secants = _secants(e.co, e.a, e.fa, e.m)
        if not secants:
            return None
        b, fb, s = rng.choice(secants)
        secant = ([e.a, e.fa], [b, fb])
        if rng.random() < 0.5:
            secant = (secant[1], secant[0])
    dist = _distractors(variant, e, None if secant is None else s)
    if len(dist) < 3:
        return None
    dist = dist[:3]

    def f(x):
        return float(_ev(e.co, F(x)))

    lines = [{"through": [list(tangent[0]), list(tangent[1])]}]
    if secant is not None:
        lines.append({"through": [list(secant[0]), list(secant[1])]})
        if rng.random() < 0.5:
            lines.reverse()
    fig = figures.cartesian(
        x_range=(LO, HI),
        y_range=(LO, HI),
        alt=_ALT_TWO if secant else _ALT_ONE,
        points=[] if far else [{"at": [e.a, e.fa]}],
        lines=lines,
        curves=[{"pieces": figures.sample_curve(f, LO, HI, (LO, HI), n=N_SAMPLES)}],
    )
    _verify(fig, e.co, e.a, e.fa, e.m, tangent)

    key = e.m
    answer_text = rational_text(key)
    texts = [rational_text(v) for _, v in dist]
    check_pairwise_distinct(key, *[v for _, v in dist])
    figures.assert_no_leak(fig, [answer_text, *texts])

    i = rng.randrange(len(_PROMPTS[variant]))
    text = phrase(rng, [_PROMPTS[variant][i]], a=e.a)
    latex = phrase(rng, [_tex(_PROMPTS[variant][i])], a=e.a)
    (x1, y1), (x2, y2) = tangent
    working = [
        f"The tangent passes through ({x1}, {y1}) and ({x2}, {y2}), so its gradient is "
        f"rise/run = ({y2} - {paren_negative(y1)})/({x2} - {paren_negative(x1)}) = {answer_text}.",
        f"The gradient of the tangent at x = {e.a} is f'({e.a}) = {answer_text}.",
    ]
    if secant is not None:
        working.insert(
            0,
            f"The line that touches the curve at x = {e.a} is the tangent; "
            f"the other line cuts the curve again, so it is a secant.",
        )
    return {
        "text": text,
        "latex": latex,
        "answer": int(key) if key.denominator == 1 else {"p": key.numerator, "q": key.denominator},
        "answer_text": answer_text,
        "answer_latex": (
            str(key.numerator)
            if key.denominator == 1
            else ("-" if key < 0 else "") + rf"\frac{{{abs(key.numerator)}}}{{{key.denominator}}}"
        ),
        "distractors": texts,
        "working": working,
        "figure": fig,
    }


@generator(
    TOPIC,
    level=13,
    presets={
        "easy": {"variant": "rising"},
        "medium": {"variant": "sign_and_zero"},
        "hard": {"variant": "cubic_crossing"},
        "exam": {"variant": "tangent_or_secant"},
    },
)
def pv1b_tangent_slope(rng, variant: str = "rising") -> dict:
    """f'(a) from a drawn tangent. Answer first: draw the slope, then a curve and tangent."""
    pool = _pool(variant)
    slopes = sorted(pool)
    zero_w = _ZERO_WEIGHT.get(variant, 0.0)
    nonzero = [s for s in slopes if s != 0]
    for _ in range(60):
        if 0 in pool and rng.random() < zero_w:
            m = F(0)
        else:
            m = rng.choice(nonzero)
        entries = pool[m]
        for _ in range(12):
            built = _try(rng, variant, rng.choice(entries))
            if built is not None:
                return built
    raise RuntimeError(f"{TOPIC}/{variant}: could not build a question")
