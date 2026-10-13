"""Generates `pv1c_translated_point` figure questions: follow a point on y = f(x) to its image on a translated
graph.

Skill ``senior-examine-translations-and-the-graphs-of-y-f``, topic ``pv1c_translated_point``.
The learner sees the graphs of y = f(x) and y = f(x) + a on one integer grid (a is never
printed), a point P marked on y = f(x), and types the coordinates of the point on
y = f(x) + a that has the same x-coordinate as P.

Answer first. a is drawn, then P = (p, q), then f is built THROUGH P as
f(p + t) = q + m t + c t^2 + d t^3 with exact ``Fraction`` coefficients. The answer is
(p, q + a). Both curves are sampled with step 1/12 (193 samples) so every integer x is an
exact sample and every pertinent point is a grid intersection inside the window.

Tiers differ in the learner's DECISION:

- easy   -- gentle f, the other curve is above: add the upward gap to the y of P.
- medium -- the other curve may be above or below P: the sign of the gap is the decision.
- hard   -- f steep at P, and the other curve ALSO passes through a grid point (p + h, q) on
            the row of P: the corresponding point keeps P's x; do not slide along the row.
- exam   -- f a straight line, so the two lines are also a horizontal shift of each other by
            b = a/m (an integer): the prompt defines a vertical shift, so the answer is
            (p, q + a), not (p - b, q), though that point is also on the other line.
"""

from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
from types import SimpleNamespace

from .. import figures
from ..core import check, generator, latex_from_inline

TOPIC = "pv1c_translated_point"

LO, HI = -8, 8  # x and y window, unit gridlines
WIN = 7  # every pertinent point has |coordinate| <= WIN
STEP_DEN = 12
N_SAMPLES = (HI - LO) * STEP_DEN + 1  # 193: every integer x is an exact sample
_ALT = "two curves on a coordinate grid"

# ---------------------------------------------------------------- prompts

_S1 = [
    "The graph of {F} is drawn below, together with the graph of {G}. The point {P} is marked on the graph of {F}.",
    "Two graphs are drawn on the same axes: {F} and {G}. The marked point {P} lies on the graph of {F}.",
    "The point {P} lies on the graph of {F}. The other curve drawn is the graph of {G}.",
    "Below are the graphs of {F} and {G}, and {P} is a marked point on the graph of {F}.",
]
_NOTE = {
    "shift_up": ["Here {A} is a positive constant.", "The constant {A} is positive."],
    "either_direction": ["Here {A} may be positive or negative.", "The constant {A} can be positive or negative."],
    "steep_same_height_trap": [
        "The constant {A} is a nonzero whole number and may be positive or negative.",
        "Here {A} is a nonzero whole number that may be negative.",
    ],
    "line_both_readings": [
        "Both graphs are straight lines, and {A} is a nonzero whole number.",
        "Here {A} is a nonzero whole number, and each graph is a straight line.",
    ],
}
_SCALE = [
    "Each grid square is 1 unit on both axes.",
    "Both axes have 1 unit per grid square.",
    "Read both axes at one unit per square.",
]
_ASK = [
    "Find the coordinates of the point on {G} that has the same x-coordinate as {P}.",
    "State the coordinates of the point on the graph of {G} that has the same x-coordinate as {P}.",
    "Write down the coordinates of the point where the vertical line through {P} meets the graph of {G}.",
]
_FILL_TEXT = {"F": "y = f(x)", "G": "y = f(x) + a", "P": "P", "A": "a"}
_FILL_TEX = {"F": r"\(y = f(x)\)", "G": r"\(y = f(x) + a\)", "P": r"\(P\)", "A": r"\(a\)"}


def _prompt(rng, variant: str) -> tuple[str, str]:
    pattern = " ".join([rng.choice(_S1), rng.choice(_NOTE[variant]), rng.choice(_SCALE), rng.choice(_ASK)])
    return pattern.format(**_FILL_TEXT), pattern.format(**_FILL_TEX)


# ---------------------------------------------------------------- exact curves
# f(p + t) - q = m t + c t^2 + d t^3  (Fractions); the curve passes through P.


def _rel(co, t) -> F:
    m, c, d = co
    return m * t + c * t * t + d * t * t * t


_ABS_UP = (1, 2, 3, 4, 5)
_ABS_ALL = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
_M_GENTLE = (F(-1), F(-1, 2), F(0), F(1, 2), F(1))
_C_GENTLE = (F(0), F(1, 8), F(-1, 8), F(1, 4), F(-1, 4), F(1, 2), F(-1, 2))


@lru_cache(maxsize=None)
def _gentle() -> tuple:
    return tuple((m, c, F(0)) for m in _M_GENTLE for c in _C_GENTLE if not (m == 0 and c == 0))


@lru_cache(maxsize=None)
def _hard_pool() -> tuple:
    """(m, c, d, h, a): slope +-2..4 at P, nonzero c or d, and the other curve passes through
    (p + h, q): f(p + h) - q = -a with a a nonzero whole number, |a| <= 5, 1 <= |h| <= 4."""
    out = []
    cs = [F(k, 2) for k in range(-6, 7)]
    ds = (F(0), F(1, 6), F(-1, 6), F(1, 3), F(-1, 3), F(1, 2), F(-1, 2), F(1), F(-1))
    for m in (-4, -3, -2, 2, 3, 4):
        for c in cs:
            for d in ds:
                if c == 0 and d == 0:
                    continue
                for h in (-4, -3, -2, -1, 1, 2, 3, 4):
                    a = -_rel((F(m), c, d), h)
                    if a.denominator == 1 and 1 <= abs(a) <= 5:
                        out.append((F(m), c, d, h, int(a)))
    return tuple(out)


@lru_cache(maxsize=None)
def _exam_pool() -> tuple:
    """(m, a, b): a line of slope m whose vertical shift a is also the horizontal shift
    b = a/m (whole, nonzero, |b| <= 6, different from both a and -a)."""
    out = []
    for m in (F(-4), F(-3), F(-2), F(2), F(3), F(4), F(1, 2), F(-1, 2), F(1, 3), F(-1, 3)):
        for a in range(-6, 7):
            if a == 0:
                continue
            b = F(a) / m
            if b.denominator == 1 and 1 <= abs(b) <= 6 and b != a and b != -a:
                out.append((m, a, int(b)))
    return tuple(out)


def _draw(rng, variant: str):
    """(co, a, offsets): the curve through P in relative form, the shift, and the relative x
    offsets t (always including 0) whose points must be inside the window."""
    if variant == "shift_up":
        return rng.choice(_gentle()), rng.choice(_ABS_UP), (0,)
    if variant == "either_direction":
        return rng.choice(_gentle()), rng.choice(_ABS_ALL), (0,)
    if variant == "steep_same_height_trap":
        m, c, d, h, a = rng.choice(_hard_pool())
        step = 1 if h > 0 else -1
        return (m, c, d), a, tuple(range(0, h + step, step))
    m, a, b = rng.choice(_exam_pool())
    return (m, F(0), F(0)), a, (0, -b)


def _place(rng, co, a, offsets):
    """Pick the integer P = (p, q) so every pertinent point is inside the window."""
    ys = [_rel(co, t) for t in offsets]
    ys += [y + a for y in ys]
    qlo = max(-WIN - y for y in ys)
    qhi = min(WIN - y for y in ys)
    plo = max(-WIN - t for t in offsets)
    phi = min(WIN - t for t in offsets)
    qlo, qhi = -((-qlo.numerator) // qlo.denominator), qhi.numerator // qhi.denominator
    pl, ph = max(plo, -6), min(phi, 6)
    if qlo > qhi or pl > ph:
        return None
    return rng.randint(pl, ph), rng.randint(qlo, qhi)


# ---------------------------------------------------------------- figure


def _sample_fn(co, p, q, shift):
    m, c, d = float(co[0]), float(co[1]), float(co[2])

    def fn(x):
        t = x - p
        return q + shift + m * t + c * t * t + d * t * t * t

    return fn


def _pair_text(x, y) -> str:
    return f"({x}, {y})"


# ---------------------------------------------------------------- reading a figure


def _curve_map(curve) -> dict:
    return {x: y for piece in curve["pieces"] for x, y in piece}


def read_translated_point(question) -> str:
    """Re-derive the answer from the figure alone. P is the one marked point; the curve that
    passes through P is f; the other curve's sample at x = p is the image's y. Curves are
    matched by x value, never by piece index. The vertical gap must be the same at every
    common sample (a pure vertical translation)."""
    fig = question.figure
    check(
        len(fig["curves"]) == 2 and len(fig["points"]) == 1,
        "translated point reader: need two curves and one marked point",
    )
    px, py = fig["points"][0]["at"]
    check(float(px) == int(px) and float(py) == int(py), "translated point reader: P is off the grid")
    maps = [_curve_map(c) for c in fig["curves"]]
    on_p = [i for i, mp in enumerate(maps) if px in mp and abs(mp[px] - py) < 1e-6]
    check(len(on_p) == 1, "translated point reader: P must lie on exactly one curve")
    f_map, g_map = maps[on_p[0]], maps[1 - on_p[0]]
    common = sorted(set(f_map) & set(g_map))
    check(len(common) >= 5, "translated point reader: too few common samples")
    gaps = [g_map[x] - f_map[x] for x in common]
    check(max(gaps) - min(gaps) < 1e-4, "translated point reader: the gap is not constant")
    gap = round(gaps[0])
    check(abs(gaps[0] - gap) < 1e-4 and gap != 0, "translated point reader: gap is not a whole number")
    check(px in g_map, "translated point reader: the other curve has no sample at x = p")
    y_img = round(g_map[px])
    check(
        abs(g_map[px] - y_img) < 1e-4 and y_img == int(py) + gap,
        "translated point reader: the other curve at x = p is off the grid",
    )
    return _pair_text(int(px), y_img)


figures.register_reader(TOPIC, read_translated_point)

# ---------------------------------------------------------------- distractors

_PRIORITY = {
    "shift_up": ("opposite-direction", "horizontal-by-a", "gap-as-y", "horizontal-slide"),
    "either_direction": ("opposite-direction", "horizontal-by-a", "gap-as-y", "horizontal-slide"),
    "steep_same_height_trap": ("horizontal-slide", "opposite-direction", "horizontal-by-a", "gap-as-y"),
    "line_both_readings": ("horizontal-slide", "opposite-direction", "horizontal-by-a", "gap-as-y"),
}


def _slide(co, a, p) -> int | None:
    """Nearest nonzero whole t where the other curve is on the row of P (f(p+t) + a = q) and
    the point is inside the window; None if there is none."""
    for k in range(1, 7):
        for t in (k, -k):
            if abs(p + t) <= WIN and _rel(co, t) + a == 0:
                return t
    return None


def _distractors(variant, co, a, p, q):
    cands = {"opposite-direction": (p, q - a), "horizontal-by-a": (p + a, q), "gap-as-y": (p, a)}
    t = _slide(co, a, p)
    if t is not None:
        cands["horizontal-slide"] = (p + t, q)
    key = (p, q + a)
    out, seen = [], {key}
    for name in _PRIORITY[variant]:
        v = cands.get(name)
        if v is None or v in seen:
            continue
        seen.add(v)
        out.append((name, v))
    return out[:3]


# ---------------------------------------------------------------- verification


def _verify(fig, co, a, p, q, key_text, variant) -> None:
    """Independent checks on a built question (raises VerificationError)."""
    check(_rel(co, 0) == 0, "translated point: f does not pass through P")
    y_img = q + a
    check(abs(p) <= WIN and abs(q) <= WIN and abs(y_img) <= WIN, "translated point: point outside window")
    check(
        read_translated_point(SimpleNamespace(figure=fig)) == key_text,
        "translated point: figure reader disagrees with the key",
    )
    check(
        all(len(c["pieces"]) >= 1 and sum(len(pc) for pc in c["pieces"]) <= 200 for c in fig["curves"]),
        "translated point: bad curve sampling",
    )
    if variant == "steep_same_height_trap":
        t = _slide(co, a, p)
        check(t is not None and abs(t) <= 4, "translated point: no same-row trap on the other curve")
        check(abs(co[0]) in (2, 3, 4), "translated point: f is not steep at P")
    elif variant == "line_both_readings":
        t = _slide(co, a, p)
        check(
            co[1] == 0 and co[2] == 0 and t is not None and F(a) / co[0] == -t,
            "translated point: the lines are not a whole horizontal shift apart",
        )
    else:
        check(abs(co[0]) <= 1, "translated point: f is not gentle at P")
        check(a > 0 or variant == "either_direction", "translated point: easy shift must be upward")


# ---------------------------------------------------------------- building


def _working(variant, a, p, q, y_img) -> list[str]:
    way = "above" if a > 0 else "below"
    steps = [
        f"P = ({p}, {q}) is on y = f(x). At x = {p} the other curve is at height {y_img}, {abs(a)} {way} P, so a = {a}."
    ]
    steps.append(
        f"A translation by a moves a point straight up or down, so the corresponding point "
        f"keeps x = {p} and has y = {q} + ({a}) = {y_img}: ({p}, {y_img})."
    )
    if variant in ("steep_same_height_trap", "line_both_readings"):
        steps.append(
            "The other curve also crosses the row of P at a different x, but that point "
            "is not the image of P under a vertical shift."
        )
    return steps


@generator(
    TOPIC,
    level=13,
    presets={
        "easy": {"variant": "shift_up"},
        "medium": {"variant": "either_direction"},
        "hard": {"variant": "steep_same_height_trap"},
        "exam": {"variant": "line_both_readings"},
    },
)
def pv1c_translated_point(rng, variant: str = "shift_up") -> dict:
    """The corresponding point on y = f(x) + a. Answer first: draw a, then P, then build f."""
    if variant not in _NOTE:
        raise ValueError(f"unknown variant {variant!r}")
    for _ in range(60):
        co, a, offsets = _draw(rng, variant)
        placed = _place(rng, co, a, offsets)
        if placed is not None:
            p, q = placed
            break
    else:
        raise RuntimeError("could not place a translated-point question")
    y_img = q + a
    curves = [
        {"pieces": figures.sample_curve(_sample_fn(co, p, q, shift), LO, HI, (LO, HI), n=N_SAMPLES)} for shift in (0, a)
    ]
    if rng.random() < 0.5:
        curves.reverse()
    fig = figures.cartesian(
        x_range=(LO, HI),
        y_range=(LO, HI),
        alt=_ALT,
        grid=1,
        equal_scale=True,
        points=[{"at": [p, q], "label": "P"}],
        curves=curves,
    )
    key_text = _pair_text(p, y_img)
    _verify(fig, co, a, p, q, key_text, variant)

    dist = _distractors(variant, co, a, p, q)
    texts = [_pair_text(*v) for _, v in dist]
    pts = [(p, y_img)] + [v for _, v in dist]
    check(len(set(pts)) == len(pts), "translated point: a distractor collides with the key")
    figures.assert_no_leak(fig, [key_text, *texts])
    text, latex = _prompt(rng, variant)
    return {
        "text": text,
        "latex": latex_from_inline(latex),
        "answer": {"x": p, "y": y_img},
        "answer_text": key_text,
        "answer_latex": key_text,
        "distractors": texts,
        "working": _working(variant, a, p, q, y_img),
        "figure": fig,
    }
