"""Generates `pv1f_ftc_derivative_read` questions: read F'(x0) from the graph of f using the fundamental theorem
of calculus.

``pv1f_ftc_derivative_read`` (senior-understand-and-use-the-theorem-f-left-x, ACMMM130): the
graph of y = f(x) is drawn on an integer grid, the prompt defines F(x) as the integral of
f(t) dt from t = a to t = x and names x0, and the learner types F'(x0). By the theorem
F'(x0) = f(x0), so the key is the height of the drawn curve at x = x0: an integer on a grid
intersection. The key is drawn FIRST (its sign by tier, its size uniformly), then a curve
passing through (x0, key) is chosen. Tiers escalate by the decision made:

- easy:   f(x0) is above the axis and the lower limit a lies to the left of x0;
- medium: f(x0) is below the axis, so the answer is a negative height;
- hard:   the lower limit a lies to the RIGHT of x0, so the integral runs backwards; the
          answer is still f(x0) and the limit plays no part in it;
- exam:   two curves are drawn and F is built from the one through the lettered point P.

The distractor ``F(x0)`` (the signed area the integral measures) is always offered and is
always an integer different from the key, as are f(a), -f(x0) and the slope f'(x0). Every
curve is sampled with step 1/8, so each integer x is an EXACT sample of the polyline. The
prompt prints a and x0 (the question's data); it never prints f. The topic registers a figure
reader that works from ``question.figure`` alone plus x0 and the LETTER named in the prompt.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import figures
from mathgen.core import check, check_pairwise_distinct, generator, pick_distractors

_LO, _HI = -10, 10  # the x window
_YLO, _YHI = -12, 12  # the y window: taller, so f(x0) can take ten values of each sign
_N = 161
_TIERS = ("easy", "medium", "hard", "exam")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_MAX_V = 10
_X0S = tuple(range(-6, 7))


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


# ----------------------------------------------------------------------- the curves
# ("line", m, q): m x + q        ("quad", s, h, c): s (x - h)^2 + c      ("vee", s, h, c): s |x - h| + c


def _f(spec, x) -> Fraction:
    x = Fraction(x)
    kind = spec[0]
    if kind == "line":
        return spec[1] * x + spec[2]
    if kind == "quad":
        return spec[1] * (x - spec[2]) ** 2 + spec[3]
    return spec[1] * abs(x - spec[2]) + spec[3]


def _slope(spec, x) -> Fraction:
    x = Fraction(x)
    kind = spec[0]
    if kind == "line":
        return spec[1]
    if kind == "quad":
        return 2 * spec[1] * (x - spec[2])
    if x == spec[2]:
        return Fraction(0)
    return spec[1] * (1 if x > spec[2] else -1)


def _antiderivative(spec, x) -> Fraction:
    """G with G' = f, written out by hand: the key's independent route through the area."""
    x = Fraction(x)
    kind = spec[0]
    if kind == "line":
        return spec[1] * x * x / 2 + spec[2] * x
    if kind == "quad":
        return spec[1] * (x - spec[2]) ** 3 / 3 + spec[3] * x
    return spec[1] * (x - spec[2]) * abs(x - spec[2]) / 2 + spec[3] * x


def _area(spec, a, x0) -> Fraction:
    """F(x0) = integral of f from a to x0, an integer by construction (see _limits)."""
    return _antiderivative(spec, x0) - _antiderivative(spec, a)


def _specs() -> list:
    out = [("line", Fraction(m), Fraction(q)) for m in (-2, -1, 1, 2) for q in range(-7, 8)]
    out += [
        ("quad", Fraction(s), Fraction(h), Fraction(c)) for s in (-1, 1) for h in range(-4, 5) for c in range(-6, 7)
    ]
    out += [("vee", Fraction(s), Fraction(h), Fraction(c)) for s in (-1, 1) for h in range(-4, 5) for c in range(-6, 7)]
    return out


_SPECS = _specs()
_BY_VALUE: dict[int, list] = {}
for _spec in _SPECS:
    for _x0 in _X0S:
        _v = _f(_spec, _x0)
        if _v.denominator == 1 and 0 < abs(_v) <= _MAX_V:
            _BY_VALUE.setdefault(int(_v), []).append((_spec, _x0))


def _visible(spec, lo, hi) -> bool:
    """f stays inside the window on [lo, hi] (a, x0 and everything between them)."""
    return all(abs(_f(spec, Fraction(i, 8))) <= _YHI - 1 for i in range(int(lo * 8), int(hi * 8) + 1))


def _limits(spec, x0: int, v: int, tier: str):
    """Lower limits a with F(x0) an integer, every printed number clear of the key."""
    out = []
    if x0 == v:
        return out
    for a in range(-9, 10):
        if abs(a - x0) < 2 or a in (v, x0):
            continue
        if tier in ("easy", "medium") and a > x0:
            continue
        if tier == "hard" and a < x0:
            continue
        fa, area = _f(spec, a), _area(spec, a, x0)
        if fa.denominator != 1 or area.denominator != 1 or abs(area) > 99:
            continue
        if not _visible(spec, min(a, x0), max(a, x0)):
            continue
        if int(area) in (0, v, -v, int(fa)) or int(fa) in (0, v, -v):
            continue
        out.append(a)
    return out


def _curve(spec) -> dict:
    fn = lambda x: float(_f(spec, Fraction(x).limit_denominator(8)))  # noqa: E731
    return {"pieces": figures.sample_curve(fn, _LO, _HI, (_YLO, _YHI), n=_N)}


# ----------------------------------------------------------------------- the reader


def _letter_in(text: str) -> str | None:
    m = re.search(r"point marked ([A-Z])\b", text)
    return m.group(1) if m else None


def _x0_in(text: str) -> int:
    m = re.search(r"F'\((-?\d+)\)", text)
    check(m is not None, "the prompt does not name x0")
    return int(m.group(1))


def _on_curve(curve: dict, pt) -> bool:
    px, py = pt
    for piece in curve["pieces"]:
        for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
            dx, dy = x1 - x0, y1 - y0
            ll = dx * dx + dy * dy
            t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / ll))
            qx, qy = x0 + t * dx, y0 + t * dy
            if (px - qx) ** 2 + (py - qy) ** 2 <= 1e-12:
                return True
    return False


def _height_at(curve: dict, x: int) -> int:
    hits = set()
    for piece in curve["pieces"]:
        for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
            if x0 <= x <= x1:
                y = y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                hits.add(round(y, 6))
    check(len(hits) == 1, "the curve does not meet x = x0 at exactly one point")
    (y,) = hits
    check(abs(y - round(y)) < 1e-9, "f(x0) is not on a grid line")
    return int(round(y))


def _read_from_figure(fig: dict, text: str) -> str:
    curves = fig["curves"]
    letter = _letter_in(text)
    if letter is not None:
        pts = [p["at"] for p in fig["points"] if p.get("label") == letter]
        check(len(pts) == 1, "the named letter is not a single marked point")
        chosen = [c for c in curves if _on_curve(c, pts[0])]
        check(len(chosen) == 1, "the named point is not on exactly one curve")
        curve = chosen[0]
    else:
        check(len(curves) == 1, "no letter named but the figure has several curves")
        curve = curves[0]
    return str(_height_at(curve, _x0_in(text)))


def _read_ftc(question) -> str:
    return _read_from_figure(question.figure, question.text)


figures.register_reader("pv1f_ftc_derivative_read", _read_ftc)

# ----------------------------------------------------------------------- the prompts

_POOLS = {
    "easy": (
        (
            "The graph of y = f(x) is drawn on the grid.",
            "The diagram shows the graph of y = f(x).",
            "Here is the graph of the function y = f(x).",
            "A function y = f(x) has the graph shown.",
        ),
        (
            "Let F(x) be the integral of f(t) dt from t = {a} to t = x.",
            "Define F(x) as the integral of f(t) dt from t = {a} up to t = x.",
            "F(x) is the integral of f(t) dt with lower limit t = {a} and upper limit t = x.",
        ),
        ("Find the value of F'({x0}).", "What is F'({x0})?", "Use the graph to find F'({x0})."),
    ),
    "medium": (
        (
            "Part of the graph of y = f(x) lies below the x-axis.",
            "The graph of y = f(x) is shown; some of it is under the x-axis.",
            "Study this graph of y = f(x), which dips below the x-axis.",
            "This grid shows y = f(x), with part of the curve under the x-axis.",
        ),
        (
            "F(x) is defined by F(x) = the integral of f(t) dt from t = {a} to t = x.",
            "The function F is given by the integral of f(t) dt from t = {a} to t = x.",
            "Take F(x) to be the integral of f(t) dt, starting at t = {a} and ending at t = x.",
        ),
        ("Determine F'({x0}).", "State the value of F'({x0}).", "Calculate F'({x0})."),
    ),
    "hard": (
        (
            "The curve y = f(x) is graphed below.",
            "Below is the curve y = f(x), drawn on a grid.",
            "A graph of y = f(x) is given.",
            "This grid carries the graph of y = f(x).",
        ),
        (
            "The function F is built from f by integrating from t = {a} (the lower limit) to t = x: F(x) = the integral of f(t) dt.",
            "For each x, F(x) is the integral of f(t) dt with lower limit t = {a}; the upper limit is t = x.",
            "Write F(x) for the integral of f(t) dt where the limits run from t = {a} to t = x.",
        ),
        (
            "What is F'({x0}), the value of the derivative of F at x = {x0}?",
            "Work out F'({x0}), the derivative of F at x = {x0}.",
            "Evaluate F'({x0}).",
        ),
    ),
    "exam": (
        (
            "Two curves, y = f(x) and y = g(x), are drawn on one grid.",
            "The graphs of y = f(x) and y = g(x) share this grid.",
            "On one grid are the graphs of two functions, f and g.",
            "One grid shows the curves of both f and g.",
        ),
        (
            "The point marked P lies on the graph of f. F(x) is the integral of f(t) dt from t = {a} to t = x.",
            "The graph of f is the curve through the point marked P. Let F(x) be the integral of f(t) dt from t = {a} to t = x.",
            "f is the function whose graph passes through the point marked P, and F(x) is the integral of f(t) dt from t = {a} to t = x.",
        ),
        ("Find F'({x0}).", "State the value of F'({x0}).", "What is F'({x0})?"),
    ),
}

_WORKING = (
    "By the fundamental theorem of calculus F'(x) = f(x), so F'({x0}) is the height of the "
    "graph of f at x = {x0}, which is {v}."
)


# ----------------------------------------------------------------------- the draw


def _values_for(tier: str) -> list:
    if tier == "easy":
        return [v for v in _BY_VALUE if v > 0]
    if tier == "medium":
        return [v for v in _BY_VALUE if v < 0]
    return sorted(_BY_VALUE)


def _choose(rng, tier: str):
    """(spec, x0, a, v): the key v first, then a curve through (x0, v), then a limit."""
    v = _pick(rng, sorted(_values_for(tier)))
    for _ in range(400):
        spec, x0 = _pick(rng, _BY_VALUE[v])
        options = _limits(spec, x0, v, tier)
        if options:
            return spec, x0, _pick(rng, options), v
    check(False, "no curve and limit realise the keyed value")


def _build(rng, tier: str):
    spec, x0, a, v = _choose(rng, tier)
    check(_f(spec, x0) == v, "the curve does not take the keyed value at x0")
    # F'(x0) from the area function by a difference quotient over the exact antiderivative
    h = Fraction(1, 64)
    quotient = (_area(spec, a, Fraction(x0) + h) - _area(spec, a, Fraction(x0) - h)) / (2 * h)
    check(
        abs(quotient - v) <= abs(_slope(spec, x0)) * h + Fraction(1, 32),
        "the area function's rate of change at x0 is not f(x0)",
    )
    area = int(_area(spec, a, x0))
    fa = int(_f(spec, a))
    slope = int(_slope(spec, x0))
    pool = [area, fa, -v] + ([slope] if slope else [])
    curves = [_curve(spec)]
    points = []
    if tier == "exam":
        for _ in range(300):
            other = _pick(rng, _SPECS)
            w = _f(other, x0)
            if (
                w.denominator == 1
                and w != 0
                and abs(w - v) >= 2
                and abs(w) <= _MAX_V + 1
                and _visible(other, x0 - 1, x0 + 1)
            ):
                break
        else:
            check(False, "no second curve found")
        at = _px(rng, spec, other, x0)
        curves = [_curve(spec), _curve(other)]
        points = [{"at": list(at), "label": "P"}]
        pool = [area, int(w), fa, -v] + ([slope] if slope else [])
        if rng.randrange(2):
            curves.reverse()
        alt = "two curves on a coordinate grid"
    else:
        alt = "a curve on a coordinate grid"
    fig = figures.cartesian(
        x_range=(_LO, _HI),
        y_range=(_YLO, _YHI),
        alt=alt,
        grid=1,
        equal_scale=True,
        curves=curves,
        points=points,
    )
    open_, define, ask = _POOLS[tier]
    text = " ".join((_pick(rng, open_), _pick(rng, define), _pick(rng, ask)))
    text = text.replace("{a}", str(a)).replace("{x0}", str(x0))
    check(_read_from_figure(fig, text) == str(v), "figure does not show the keyed F'(x0)")
    return fig, text, v, pool


def _px(rng, f_spec, other, x0):
    """P on f at an integer x away from x0, clear of the other curve, off the vertex."""
    cands = []
    for x in range(-9, 10):
        if x == x0:
            continue
        y = _f(f_spec, x)
        if y.denominator != 1 or abs(y) > 11 or y == 0:
            continue
        if abs(_f(other, x) - y) < 2:
            continue
        cands.append((x, int(y)))
    check(bool(cands), "no locator point for the exam curve")
    return _pick(rng, cands)


@generator("pv1f_ftc_derivative_read", level=13, presets=_PRESETS)
def pv1f_ftc_derivative_read(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, v, pool = _build(rng, tier)
    x0 = _x0_in(text)
    answer_text = str(v)
    distractors = pick_distractors(answer_text, [str(p) for p in pool], n=3)
    check(len(distractors) >= 2, "too few distractors")
    check_pairwise_distinct(v, *[int(d) for d in distractors])
    figures.assert_no_leak(fig, [answer_text, *distractors])
    return dict(
        text=text,
        answer=v,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[_WORKING.format(x0=x0, v=v)],
        figure=fig,
    )
