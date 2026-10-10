"""Generates `pv1c_inverse_point` figure questions: use the reflection property of y = f(x) and y = f^-1(x).

Skill ``senior-examine-the-reflection-property-of-the-graph-of``, topic ``pv1c_inverse_point``.
The learner sees the graphs of y = f(x) and y = f^-1(x) on one integer grid, a point P marked on
y = f(x) (coordinates not printed), and types the coordinates of the corresponding point on
y = f^-1(x). By the reflection property that is the swap of P's coordinates.

Answer first. P = (p, q) is drawn, then a strictly monotone f through P (a line, a cubic
(x - h)^3 / s + k with whole k, or 2^x + k), then the inverse curve is sampled from f by
bisection, so the drawn pair really are reflections of each other. Both curves use step 1/12
(145 samples) so every integer x is an exact sample.

Tiers differ in the learner's DECISION:

- easy   -- the mirror line y = x is drawn: the curves are mirror images in it, so swap.
- medium -- no mirror line, f increasing and nonlinear: recognise y = x without being shown it.
- hard   -- f DECREASING and a coordinate negative: the swap keeps each coordinate's sign.
- exam   -- ``crossing_pair``: f DECREASING and P is a point where the two graphs cross OFF y = x.
            P lies on both curves, so "it is on the inverse already" tempts a copy of P, but the
            image under the reflection is (b, a), the OTHER crossing. (``fixed_point``, P on
            y = x, is kept as a variant but no tier uses it: copying P scored the same there.)
"""

from __future__ import annotations

import bisect
from fractions import Fraction as F
from types import SimpleNamespace

from .. import figures
from ..core import check, generator

TOPIC = "pv1c_inverse_point"

LO, HI = -6, 6  # x and y window, unit gridlines
LIM = 5  # P and its image have |coordinate| <= LIM
STEP_DEN = 12
N_SAMPLES = (HI - LO) * STEP_DEN + 1  # 145: every integer x is an exact sample
_ALT = "two curves on a coordinate grid"

# ---------------------------------------------------------------- prompts

_S1 = [
    "The graphs of {F} and its inverse {G} are drawn below. The point {P} is marked on the graph of {F}.",
    "Below are the graph of {F} and the graph of its inverse {G}. The marked point {P} lies on the graph of {F}.",
    "The point {P} lies on the graph of {F}. The graph of the inverse function {G} is also drawn.",
    "A function {F} and its inverse {G} are graphed together, and {P} is a marked point on the graph of {F}.",
]
_NOTE = {
    "mirror_drawn": [
        "The two graphs are reflections of each other in the line {L}, which is also drawn.",
        "The line {L} is drawn as well, and each graph is the mirror image of the other in it.",
    ],
    "no_mirror": ["Both graphs are drawn on the same axes.", "The two graphs share one set of axes."],
    "decreasing_negative": ["Both graphs are drawn on the same axes.", "The two graphs share one set of axes."],
    "fixed_point": ["Both graphs are drawn on the same axes.", "The two graphs share one set of axes."],
    "crossing_pair": ["Both graphs are drawn on the same axes.", "The two graphs share one set of axes."],
}
_SCALE = [
    "Each grid square is 1 unit on both axes.",
    "Both axes have 1 unit per grid square.",
    "Read both axes at one unit per square.",
]
_ASK = [
    "Find the coordinates of the corresponding point on the graph of {G}.",
    "State the coordinates of the point on {G} that corresponds to {P}.",
    "Write down the coordinates of the corresponding point on the inverse graph {G}.",
]
_FILL_TEXT = {"F": "y = f(x)", "G": "y = f⁻¹(x)", "P": "P", "L": "y = x"}
_FILL_TEX = {"F": r"\(y = f(x)\)", "G": r"\(y = f^{-1}(x)\)", "P": r"\(P\)", "L": r"\(y = x\)"}


def _prompt(rng, variant: str) -> tuple[str, str]:
    pattern = " ".join([rng.choice(_S1), rng.choice(_NOTE[variant]), rng.choice(_SCALE), rng.choice(_ASK)])
    return pattern.format(**_FILL_TEXT), pattern.format(**_FILL_TEX)


# ---------------------------------------------------------------- the curves
# A function is (fn, increasing). Closures use floats; P is exact by construction.

_LINE_UP = (F(1, 3), F(1, 2), F(2), F(3))
_LINE_DOWN = (F(-1, 4), F(-1, 3), F(-1, 2), F(-2), F(-3), F(-4))  # never -1: that line is its own inverse
_S_GENTLE = (8, 9, 12, 16)
_S_ANY = (2, 3, 4, 6, 8, 9, 12, 16)


def _line(p, q, m):
    mf, pf, qf = float(m), float(p), float(q)
    return lambda x: qf + mf * (x - pf)


def _cubic_options(p, q, sign, ss):
    """(h, s, k) with whole k and |k| <= LIM so f(x) = sign (x-h)^3 / s + k passes through (p, q)."""
    out = []
    for h in range(-4, 5):
        for s in ss:
            num = sign * (p - h) ** 3
            if p != h and num % s == 0 and abs(q - num // s) <= LIM:
                out.append((h, s, q - num // s))
    return out


def _cubic(h, s, k, sign):
    return lambda x: sign * (x - h) ** 3 / s + k


def _exp(k):
    return lambda x: 2.0**x + k


def _invert(fn, increasing):
    """The inverse of a strictly monotone fn by bisection (60 halvings of [-40, 40]); None
    where the value is not attained."""
    flo, fhi = fn(-40.0), fn(40.0)

    def inv(y):
        if increasing:
            if y < flo or y > fhi:
                return None
        elif y > flo or y < fhi:
            return None
        lo, hi = -40.0, 40.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (fn(mid) < y) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return inv


def _build_f(rng, variant, p, q):
    """(fn, increasing, kind) through (p, q) for the tier, or None if no member fits."""
    if variant == "mirror_drawn":
        opts = _cubic_options(p, q, 1, _S_GENTLE) if rng.random() < 0.3 else []
        if opts:
            h, s, k = rng.choice(opts)
            return _cubic(h, s, k, 1), True, "cubic"
        return _line(p, q, rng.choice(_LINE_UP)), True, "line"
    if variant == "no_mirror":
        if p >= 0 and rng.random() < 0.4:
            k = q - 2**p
            if p <= 3 and abs(k) <= LIM:
                return _exp(k), True, "exp"
        opts = _cubic_options(p, q, 1, _S_ANY)
        if not opts:
            return None
        h, s, k = rng.choice(opts)
        return _cubic(h, s, k, 1), True, "cubic"
    if variant == "decreasing_negative":
        if rng.random() < 0.25:  # a decreasing exponential: 2^-x + k or -2^x + k
            if -3 <= p <= 0 and abs(q - 2**-p) <= LIM:
                k = q - 2**-p
                return (lambda x: 2.0**-x + k), False, "exp"
            if 0 <= p <= 3 and abs(q + 2**p) <= LIM:
                k = q + 2**p
                return (lambda x: -(2.0**x) + k), False, "exp"
        opts = _cubic_options(p, q, -1, _S_ANY) if rng.random() < 0.6 else []
        if opts:
            h, s, k = rng.choice(opts)
            return _cubic(h, s, k, -1), False, "cubic"
        return _line(p, q, rng.choice(_LINE_DOWN)), False, "line"
    if variant == "crossing_pair":
        h, s, k = rng.choice(_PAIRS[(p, q)])
        return _cubic(h, s, k, -1), False, "cubic"
    # fixed_point: P on y = x; f increasing or decreasing
    r = rng.random()
    if r < 0.25 and 0 <= p <= 3 and abs(p - 2**p) <= LIM:
        return _exp(p - 2**p), True, "exp"
    if r < 0.6:
        sign = rng.choice((1, -1))
        opts = _cubic_options(p, q, sign, _S_ANY)
        if opts:
            h, s, k = rng.choice(opts)
            return _cubic(h, s, k, sign), sign == 1, "cubic"
    m = rng.choice(_LINE_UP + _LINE_DOWN)
    return _line(p, q, m), m > 0, "line"


# ---------------------------------------------------------------- reading a figure


def _merge(curve):
    """(xs, ys) of a curve's samples, pieces concatenated in x order."""
    pts = sorted((x, y) for piece in curve["pieces"] for x, y in piece)
    return [x for x, _ in pts], [y for _, y in pts]


def _interp(xs, ys, x):
    """Linear interpolation of a sampled curve at x (None outside its sampled span)."""
    if not xs or x < xs[0] - 1e-9 or x > xs[-1] + 1e-9:
        return None
    i = max(1, min(bisect.bisect_left(xs, x - 1e-9), len(xs) - 1))
    x0, x1 = xs[i - 1], xs[i]
    if x1 == x0:
        return ys[i]
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - x0) / (x1 - x0)


def _mirror_error(a, b) -> float:
    """Largest |x - interp_a(y)| over the samples (x, y) of b whose y lies inside a's x-span
    (margin 0.3 from the span ends, where a chord is least accurate)."""
    axs, ays = a
    bxs, bys = b
    if not axs:
        return 0.0
    worst = 0.0
    for x, y in list(zip(bxs, bys))[::3]:
        if not (axs[0] + 0.3 <= y <= axs[-1] - 0.3):
            continue
        v = _interp(axs, ays, y)
        lo_, hi_ = _interp(axs, ays, y - 1 / 12), _interp(axs, ays, y + 1 / 12)
        if lo_ is None or hi_ is None or abs(hi_ - lo_) * 6 > 3.5:
            continue  # a is steep here: the mirrored region is checked from the other curve
        if v is not None:
            worst = max(worst, abs(v - x))
    return worst


def read_inverse_point(question) -> str:
    """Re-derive the answer from the figure alone. The curve through the marked point P is f
    (if P is on both curves it is its own image). The reflection rule gives (q, p); it must lie
    on the other curve, and the two curves must be mirror images in y = x."""
    fig = question.figure
    check(
        len(fig["curves"]) == 2 and len(fig["points"]) == 1,
        "inverse point reader: need two curves and one marked point",
    )
    px, py = fig["points"][0]["at"]
    check(float(px) == int(px) and float(py) == int(py), "inverse point reader: P is off the grid")
    px, py = int(px), int(py)
    maps = [dict(zip(*_merge(c))) for c in fig["curves"]]
    on_p = [i for i, mp in enumerate(maps) if px in mp and abs(mp[px] - py) < 1e-6]
    check(len(on_p) >= 1, "inverse point reader: P is on neither curve")
    if len(on_p) == 2:
        check(
            all(py in m and abs(m[py] - px) < 1e-6 for m in maps),
            "inverse point reader: P is on both curves but its swap is not",
        )
        g_map = maps[0]
    else:
        g_map = maps[1 - on_p[0]]
    check(
        py in g_map and abs(g_map[py] - px) < 1e-6, "inverse point reader: the swapped point is not on the other curve"
    )
    a, b = (_merge(c) for c in fig["curves"])
    check(
        _mirror_error(a, b) < 0.05 and _mirror_error(b, a) < 0.05,
        "inverse point reader: the curves are not reflections in y = x",
    )
    return f"({py}, {px})"


figures.register_reader(TOPIC, read_inverse_point)

# ---------------------------------------------------------------- distractors

_PRIORITY = {
    "mirror_drawn": ("unswapped", "negated-swap", "reflected-in-axis", "sign-flipped-second"),
    "no_mirror": ("unswapped", "negated-swap", "sign-flipped-second", "sign-flipped-first"),
    "decreasing_negative": ("negated-swap", "sign-flipped-second", "sign-flipped-first", "unswapped"),
    "fixed_point": ("negated-swap", "sign-flipped-second", "sign-flipped-first", "reflected-in-axis"),
    "crossing_pair": ("unswapped", "negated-swap", "reflected-in-axis", "sign-flipped-second"),
}


def _pair_text(x, y) -> str:
    return f"({x}, {y})"


def _distractors(variant, p, q):
    key = (q, p)
    cands = {
        "unswapped": (p, q),
        "negated-swap": (-q, -p),
        "sign-flipped-second": (q, -p),
        "sign-flipped-first": (-q, p),
        "reflected-in-axis": (p, -q),
    }
    out, seen = [], {key}
    order = _PRIORITY[variant] + tuple(n for n in cands if n not in _PRIORITY[variant])
    for name in order:
        v = cands[name]
        if v in seen:
            continue
        seen.add(v)
        out.append((name, v))
    return out[:3]


# ---------------------------------------------------------------- drawing P

_ALL = [(a, b) for a in range(-LIM, LIM + 1) for b in range(-LIM, LIM + 1) if a != b]
_POS = [(a, b) for a, b in _ALL if a > 0 and b > 0]
_HAS_NEG = [(a, b) for a, b in _ALL if a < 0 or b < 0]
_NO_NEG = [(a, b) for a, b in _ALL if a >= 0 and b >= 0]
# crossing_pair: decreasing cubic f(x) = -(x-h)^3/s + k (whole k, s >= 2) with f(p) = q AND f(q) = p,
# p != q, so P = (p, q) and its image (q, p) are both on f and on f^-1.
_PAIRS: dict = {}
for _p in range(-LIM, LIM + 1):
    for _q in range(-LIM, LIM + 1):
        if _p == _q:
            continue
        for _h in range(-4, 5):
            for _s in _S_ANY:
                _n = -((_p - _h) ** 3)
                if _h not in (_p, _q) and _n % _s == 0:  # no point at the inflection: the inverse is vertical there
                    _k = _q - _n // _s
                    if abs(_k) <= LIM and -((_q - _h) ** 3) == _s * (_p - _k):
                        _PAIRS.setdefault((_p, _q), []).append((_h, _s, _k))
_CROSSING = sorted(_PAIRS)
_FIXED = [
    (a, a) for a in range(-LIM, LIM + 1) if a != 0
]  # (0, 0) is left out: all five distractors would equal the key


def _draw_point(rng, variant):
    if variant == "mirror_drawn":
        return rng.choice(_POS)
    if variant == "no_mirror":
        return rng.choice(_HAS_NEG if rng.random() < 0.5 else _NO_NEG)
    if variant == "decreasing_negative":
        return rng.choice(_HAS_NEG)
    if variant == "crossing_pair":
        return rng.choice(_CROSSING)
    return rng.choice(_FIXED)


# ---------------------------------------------------------------- verification


def _slope_sign(curve) -> int:
    xs, ys = _merge(curve)
    ups = sum(y1 > y0 for y0, y1 in zip(ys, ys[1:]))
    downs = sum(y1 < y0 for y0, y1 in zip(ys, ys[1:]))
    check(ups == 0 or downs == 0, "inverse point: a curve is not monotone")
    return 1 if ups else -1


def _verify(fig, p, q, key_text, variant) -> None:
    """Independent checks on a built question (raises VerificationError)."""
    check(abs(p) <= LIM and abs(q) <= LIM, "inverse point: point outside the allowed range")
    check(
        read_inverse_point(SimpleNamespace(figure=fig)) == key_text,
        "inverse point: figure reader disagrees with the key",
    )
    check(
        all(1 <= sum(len(pc) for pc in c["pieces"]) <= 200 for c in fig["curves"]), "inverse point: bad curve sampling"
    )
    signs = [_slope_sign(c) for c in fig["curves"]]
    check(signs[0] == signs[1], "inverse point: f and its inverse must both rise or both fall")
    if variant == "fixed_point":
        check(p == q, "inverse point: the exam point must lie on y = x")
        check(fig["lines"] == [], "inverse point: exam draws no mirror line")
        return
    check(p != q, "inverse point: P must not lie on y = x")
    if variant == "crossing_pair":
        check(signs[0] == -1, "inverse point: the exam crossing is decreasing")
        check(fig["lines"] == [], "inverse point: exam draws no mirror line")
        check(
            all(abs(dict(zip(*_merge(c))).get(x, 1e9) - y) < 1e-6 for c in fig["curves"] for x, y in ((p, q), (q, p))),
            "inverse point: both crossings must lie on both curves",
        )
        return
    if variant == "mirror_drawn":
        check(p > 0 and q > 0 and signs[0] == 1, "inverse point: easy is increasing in quadrant 1")
        check(len(fig["lines"]) == 1, "inverse point: easy draws the mirror line")
    else:
        check(fig["lines"] == [], "inverse point: only easy draws the mirror line")
    if variant == "decreasing_negative":
        check(signs[0] == -1 and (p < 0 or q < 0), "inverse point: hard is decreasing with a negative coordinate")
    if variant == "no_mirror":
        check(signs[0] == 1, "inverse point: medium is increasing")


# ---------------------------------------------------------------- building


def _working(variant, p, q) -> list[str]:
    steps = [
        f"P = ({p}, {q}) is on y = f(x). Reflecting a graph in the line y = x swaps the coordinates "
        f"of every point, and the graph of f⁻¹ is the reflection of the graph of f."
    ]
    if variant == "crossing_pair":
        steps.append(
            f"P = ({p}, {q}) is where the graphs cross, but it is not on y = x, so it is not its own "
            f"reflection. The corresponding point is ({q}, {p}), the other crossing."
        )
    elif variant == "fixed_point":
        steps.append(f"P is on the line y = x, so it is its own reflection: the corresponding point is ({q}, {p}).")
    else:
        steps.append(f"So ({p}, {q}) on f corresponds to ({q}, {p}) on f⁻¹.")
    return steps


@generator(
    TOPIC,
    level=13,
    presets={
        "easy": {"variant": "mirror_drawn"},
        "medium": {"variant": "no_mirror"},
        "hard": {"variant": "decreasing_negative"},
        "exam": {"variant": "crossing_pair"},
    },
)
def pv1c_inverse_point(rng, variant: str = "mirror_drawn") -> dict:
    """The corresponding point on y = f^-1(x). Answer first: draw P, then a monotone f through it."""
    if variant not in _NOTE:
        raise ValueError(f"unknown variant {variant!r}")
    for _ in range(80):
        p, q = _draw_point(rng, variant)
        built = _build_f(rng, variant, p, q)
        if built is None:
            continue
        fn, increasing, _kind = built
        inv = _invert(fn, increasing)
        if variant not in ("fixed_point", "crossing_pair"):
            # P must be on f only: the point (p, q) is on the inverse only when f(q) = p
            back = inv(float(p))
            if back is not None and abs(back - q) < 1e-9:
                continue
        break
    else:
        raise RuntimeError("could not place an inverse-point question")
    curves = [
        {"pieces": figures.sample_curve(fn, LO, HI, (LO, HI), n=N_SAMPLES)},
        {"pieces": figures.sample_curve(lambda x: inv(x), LO, HI, (LO, HI), n=N_SAMPLES)},
    ]
    if rng.random() < 0.5:
        curves.reverse()
    lines = [{"through": [[0, 0], [1, 1]], "label": "y = x"}] if variant == "mirror_drawn" else []
    fig = figures.cartesian(
        x_range=(LO, HI),
        y_range=(LO, HI),
        alt=_ALT,
        grid=1,
        equal_scale=True,
        points=[{"at": [p, q], "label": "P"}],
        curves=curves,
        lines=lines,
    )
    key_text = _pair_text(q, p)
    _verify(fig, p, q, key_text, variant)

    dist = _distractors(variant, p, q)
    texts = [_pair_text(*v) for _, v in dist]
    pts = [(q, p)] + [v for _, v in dist]
    check(len(set(pts)) == len(pts), "inverse point: a distractor collides with the key")
    figures.assert_no_leak(fig, [key_text, *texts])
    text, latex = _prompt(rng, variant)
    return {
        "text": text,
        "latex": latex,
        "answer": {"x": q, "y": p},
        "answer_text": key_text,
        "answer_latex": key_text,
        "distractors": texts,
        "working": _working(variant, p, q),
        "figure": fig,
    }
