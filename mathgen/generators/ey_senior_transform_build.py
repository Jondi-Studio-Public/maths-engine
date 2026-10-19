"""Generates figure questions on transformed graphs (Year 11-12), read from the drawn graphs.

Three skills, each with an easy, medium and hard tier and a connected exam chain (`<topic>_exam`):

* ``ey_graph_translation`` (ACMMM025, ``senior-graph-translation-read-shift``): the graphs of y = f(x) and
  y = g(x) = f(x + b) + a on one integer grid. Easy reads a (a vertical shift), medium reads b (a horizontal
  shift, where "f(x + b)" moves the graph the other way), hard picks the equation that matches both shifts, and the
  exam chain reads b, then a, then follows the marked point P to its image.
* ``ey_graph_dilation`` (ACMMM026, ``senior-graph-dilation-read-factor``): the graphs of y = f(x) and a dilation of
  it, y = c f(x) or y = f(kx). Easy reads c from the y-intercepts, medium reads k from the x-intercepts (the trap:
  f(kx) divides the x-coordinates by k), hard picks the equation, and the exam chain reads k, then c, then follows
  the marked point P to its image under y = c f(kx).
* ``ey_graph_reciprocal_abs`` (ACMSM099): the graph of y = f(x) against y = |f(x)|, y = -f(x), y = f(|x|) and
  y = 1/f(x). Easy and medium pick which equation is the drawn graph g; hard reads a value of the transformed
  function from the graph of f alone; the exam chain finds the vertical asymptotes of 1/f from the zeros of f.

Every value the learner reads sits on a gridline crossing (integer x and integer y), the curves are sampled every 1/12
so each integer is an exact sample, and each reader refits the transformation from the two drawn curves (or reads f)
and never looks at the stored answer.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction as F

from .. import figures
from ..core import attempts, check, generator, paren_negative as pn, pick_distractors
from ._ey_senior_kit import (
    HI,
    LO,
    TooFewDistractors,
    arg,
    by_label,
    curve_map,
    frac_text,
    mcq,
    near,
    num_part,
    retry_short,
    sample,
)

SHAPES = {
    "up": lambda t: t * t,
    "down": lambda t: -t * t,
    "vee": lambda t: abs(t),
    "cap": lambda t: -abs(t),
}
_ALT = "the graphs of f and g on a coordinate grid, one unit per grid square"
_SCALE = ("Each grid square is 1 unit on both axes.", "Both axes have 1 unit per grid square.")
_LABELS_TEXT = "The two graphs are labelled f and g."


def _bt(text: str) -> str:
    """Bracket a rendered number that is negative or a fraction, so it can follow an operator."""
    return f"({text})" if text.startswith("-") or "/" in text else text


def _plane(curves, points=(), alt=_ALT):
    return figures.cartesian(
        x_range=(LO, HI), y_range=(LO, HI), alt=alt, grid=1, equal_scale=True, points=list(points), curves=curves
    )


def _ints(curve) -> dict:
    """``{int x: y}`` at the integer samples only."""
    return {int(x): y for x, y in curve_map(curve).items() if x.denominator == 1}


_TAILS = (
    "Read the values from the grid.",
    "Every value you need sits on a grid crossing.",
    "Work through the parts in order.",
)

# =============================================================================================== translation (ACMMM025)

TRANSLATION = "ey_graph_translation"
TRANSLATION_EXAM = "ey_graph_translation_exam"


def _fit_shift(f_map, g_map) -> tuple[int, int]:
    """The one (b, a) with g(x) = f(x + b) + a at every common integer x (at least 3 of them)."""
    fits = []
    for b in range(-12, 13):
        diffs = [g_map[x] - f_map[x + b] for x in g_map if x + b in f_map]
        if len(diffs) >= 3 and max(diffs) - min(diffs) < 1e-4 and near(diffs[0], round(diffs[0])):
            fits.append((b, round(diffs[0])))
    check(len(fits) == 1, f"the two graphs fit {len(fits)} translations, not one")
    return fits[0]


def _draw_translation(rng, tier: str):
    for _ in attempts(500, "a translation with both vertices well inside the grid"):
        shape = rng.choice(tuple(SHAPES))
        s = 1 if shape in ("up", "vee") else -1
        h, k = rng.randint(-3, 3), rng.choice((-3, -2, -1, 1, 2, 3))
        if tier == "easy":
            a, b = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4)), 0
        elif tier == "medium":
            a, b = 0, rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
            if h == 0:
                continue
        else:
            a, b = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4)), rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
            if h == 0:
                continue
        if not (-4 <= s * k <= 3 and -4 <= s * (k + a) <= 3 and abs(h - b) <= 5):
            continue
        return shape, h, k, a, b
    raise AssertionError("unreachable")


def _translation_figure(shape, h, k, a, b, point=None):
    fn = SHAPES[shape]
    curves = [
        sample(lambda x: fn(x - h) + k, label="f"),
        sample(lambda x: fn(x + b - h) + k + a, label="g"),
    ]
    return _plane(curves, points=[{"at": list(point), "label": "P"}] if point else ())


def _shift_text(n: int) -> str:
    return str(n)


def _eq(b: int, a: int) -> str:
    text = f"g(x) = f({arg('x', b)})"
    return text + (f" {'+' if a > 0 else '-'} {abs(a)}" if a else "")


@generator(
    TRANSLATION,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def ey_graph_translation(rng, tier="easy"):
    """Read a, b or the matching equation from the graphs of f and g = f(x + b) + a."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    shape, h, k, a, b = _draw_translation(rng, tier)
    fig = _translation_figure(shape, h, k, a, b)
    scale = rng.choice(_SCALE)
    if tier == "easy":
        text = rng.choice(
            (
                f"The graph of g is the graph of f moved up or down, so g(x) = f(x) + a. {scale} Find a.",
                f"g is a vertical translation of f, with g(x) = f(x) + a. {scale} What is the value of a?",
            )
        )
        key, wrong = str(a), [-a, k + a, k, -(k + a)]
        working = [
            f"The vertex of f is at ({h}, {k}) and the vertex of g is at ({h}, {k + a}).",
            f"a = {k + a} - {pn(k)} = {a}.",
        ]
    elif tier == "medium":
        text = rng.choice(
            (
                f"The graph of g is the graph of f moved left or right, so g(x) = f(x + b). {scale} Find b.",
                f"g is a horizontal translation of f, with g(x) = f(x + b). {scale} What is the value of b?",
            )
        )
        key, wrong = str(b), [-b, h - b, h, -(h - b)]
        working = [
            f"The vertex of f is at x = {h} and the vertex of g is at x = {h - b}.",
            f"g(x) = f(x + b) moves the graph by -b, so -b = {h - b} - {pn(h)} = {-b} and b = {b}.",
        ]
    else:
        text = rng.choice(
            (
                f"The graph of g is the graph of f moved horizontally and vertically. {scale} Which equation gives g?",
                f"g is a translation of f along both axes. {scale} Which equation gives g(x)?",
            )
        )
        key = _eq(b, a)
        wrong = [_eq(-b, a), _eq(b, -a), _eq(a, b), _eq(-b, -a)]
        working = [
            f"The vertex moves from ({h}, {k}) to ({h - b}, {k + a}): {-b} in x and {a} in y.",
            f"A move of {-b} in x is f({arg('x', b)}) and a move of {a} in y adds {pn(a)}, so {key}.",
        ]
        wrong = [w for w in wrong if w != key]
    distractors = mcq(key, [str(w) for w in wrong])
    check(
        _read_translation(type("Q", (), {"figure": fig, "text": text})) == key,
        "the translation reader disagrees with the key",
    )
    figures.assert_no_leak(fig, [key, *distractors])
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=f"plain:{key}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _read_translation(question) -> str:
    """Refit g(x) = f(x + b) + a from the two drawn curves, then answer what the wording asks."""
    curves = by_label(question.figure)
    b, a = _fit_shift(_ints(curves["f"]), _ints(curves["g"]))
    if "Which equation" in question.text:
        return _eq(b, a)
    return str(a) if "f(x) + a" in question.text else str(b)


figures.register_reader(TRANSLATION, _read_translation)


@generator(TRANSLATION_EXAM, level=13, presets={"exam": dict(tier="exam")})
@retry_short
def ey_graph_translation_exam(rng, tier="exam"):
    """Exam chain: b, then a, then the coordinates of the image of the marked point P."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a translation with a marked point whose image stays on the grid"):
        shape, h, k, a, b = _draw_translation(rng, "hard")
        p = h + rng.choice((-2, -1, 1, 2))
        q = SHAPES[shape](p - h) + k
        if abs(p) <= 6 and abs(q) <= 6 and abs(p - b) <= 6 and abs(q + a) <= 6:
            break
    fig = _translation_figure(shape, h, k, a, b, point=(p, q))
    text = (
        "The graph of g is the graph of f moved horizontally and vertically, so g(x) = f(x + b) + a. "
        "The point P is marked on the graph of f. Each grid square is 1 unit on both axes."
    )
    parts = [
        num_part(
            "Find the value of b.",
            b,
            [-b, h - b, h],
            working=[f"The vertex moves from x = {h} to x = {h - b}, a move of {-b}, so b = {b}."],
        ),
        num_part(
            "Find the value of a.",
            a,
            [-a, k + a, k],
            working=[f"The vertex moves from y = {k} to y = {k + a}, so a = {a}."],
        ),
        num_part(
            "Hence find the x-coordinate of the point on g that P moves to.",
            p - b,
            [p + b, p, p - a],
            follow=f"{p}-part_a",
            working=[f"The x-coordinate moves by -b: {p} - ({b}) = {p - b}."],
        ),
        num_part(
            "Hence find the y-coordinate of the point on g that P moves to.",
            q + a,
            [q - a, q, q + b],
            follow=f"{q}+part_b",
            working=[f"The y-coordinate moves by a: {q} + ({a}) = {q + a}."],
        ),
    ]
    figures.assert_no_leak(fig, [str(b), str(a)])
    return dict(text=f"{text} {rng.choice(_TAILS)}", figure=fig, parts=parts)


def _read_translation_exam(question) -> str:
    fig = question.figure
    curves = by_label(fig)
    b, a = _fit_shift(_ints(curves["f"]), _ints(curves["g"]))
    p, q = fig["points"][0]["at"]
    return f"(a) {b}; (b) {a}; (c) {p - b}; (d) {q + a}"


# (the exam chain of an mcq skill has no registered reader: the sweep cannot hand a labelled reply to the mcq path)

# =============================================================================================== dilation (ACMMM026)

DILATION = "ey_graph_dilation"
DILATION_EXAM = "ey_graph_dilation_exam"

# f(x) = s (x^2 - m^2) or s (|x| - m): the x-intercepts are +-m and the y-intercept is -s m^2 (or -s m).
_BASES = [(kind, m, s) for kind in ("sq", "abs") for m in ((1, 2) if kind == "sq" else (2, 3, 4)) for s in (1, -1)]


def _base(kind, m, s):
    if kind == "sq":
        return lambda x: s * (x * x - m * m)
    return lambda x: s * (abs(x) - m)


def _f0(kind, m, s) -> int:
    return -s * (m * m if kind == "sq" else m)


def _dil_fn(kind, m, s, c, k):
    f = _base(kind, m, s)
    return lambda x: float(c) * f(float(k) * x)


def _fit_dilation(f_map, g_map, forms=("c", "k")):
    """All (c, k) with g(x) = c f(kx), k > 0, over the common samples. ``forms``: which of c / k may differ from 1."""
    cs = [F(v) for v in (-3, -2, -1, F(-1, 2), F(-1, 3), F(1, 3), F(1, 2), 1, 2, 3)] if "c" in forms else [F(1)]
    ks = [F(v) for v in (F(1, 3), F(1, 2), 1, 2, 3)] if "k" in forms else [F(1)]
    fits = []
    for c in cs:
        for k in ks:
            pairs = [(g_map[x], f_map[k * x]) for x in g_map if k * x in f_map]
            if len(pairs) >= 4 and all(near(g, float(c) * fv) for g, fv in pairs):
                fits.append((c, k))
    return fits


def _vertical_pool():
    out = []
    for kind, m, s in _BASES:
        for c in (F(2), F(3), F(1, 2), F(-1), F(-2)):
            g0 = c * _f0(kind, m, s)
            if g0.denominator == 1 and abs(g0) <= 7 and (c > 0 or True):
                out.append((kind, m, s, c))
    return out


def _horizontal_pool():
    out = []
    for kind, m, s in _BASES:
        for k in (F(2), F(3), F(1, 2)):
            root = F(m) / k
            if root.denominator == 1 and root <= 7 and abs(_f0(kind, m, s)) <= 7:
                out.append((kind, m, s, k))
    return out


_VERTICAL, _HORIZONTAL = _vertical_pool(), _horizontal_pool()
_BOTH = [
    (kind, m, s, c, k)
    for kind, m, s in _BASES
    for c in (F(2), F(3), F(1, 2), F(-1), F(-2))
    for k in (F(2), F(3), F(1, 2))
    if (F(m) / k).denominator == 1
    and F(m) / k <= 7
    and (c * _f0(kind, m, s)).denominator == 1
    and abs(c * _f0(kind, m, s)) <= 7
]


def _dil_figure(kind, m, s, c, k, point=None):
    f = _base(kind, m, s)
    curves = [sample(f, label="f"), sample(_dil_fn(kind, m, s, c, k), label="g")]
    return _plane(curves, points=[{"at": list(point), "label": "P"}] if point else ())


def _dil_eq(form: str, v: F) -> str:
    """g(x) = c f(x) ('v') or f(kx) ('h') written the way the options show it."""
    if form == "v":
        if v == 1:
            return "g(x) = f(x)"
        if v == -1:
            return "g(x) = -f(x)"
        if v.denominator == 1:
            return f"g(x) = {v}f(x)"
        if v == F(1, v.denominator):
            return f"g(x) = f(x)/{v.denominator}"
        return f"g(x) = {frac_text(v)}f(x)"
    if v == 1:
        return "g(x) = f(x)"
    if v == -1:
        return "g(x) = f(-x)"
    if v.denominator == 1:
        return f"g(x) = f({v}x)"
    return f"g(x) = f(x/{v.denominator})"


def _dil_wrong(form: str, v: F) -> list[str]:
    """Wrong equations in priority order: the other direction, the inverse, and the sign dropped."""
    other = "h" if form == "v" else "v"
    av = abs(v)
    out = [_dil_eq(other, av), _dil_eq(form, 1 / v if v > 0 else 1 / av), _dil_eq(other, 1 / av)]
    if v < 0:
        out.insert(0, _dil_eq(form, av))
    return out


@generator(
    DILATION,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def ey_graph_dilation(rng, tier="easy"):
    """Read c, k or the matching equation from the graphs of f and a dilation of f."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    scale = rng.choice(_SCALE)
    if tier == "easy":
        kind, m, s, c = rng.choice([r for r in _VERTICAL if r[3] > 0])
        form, v, k = "v", c, F(1)
        text = rng.choice(
            (
                f"The graph of g is the graph of f stretched or squashed up and down, so g(x) = c f(x) with c > 0. "
                f"{scale} Find c.",
                f"g is a vertical dilation of f, with g(x) = c f(x) and c > 0. {scale} What is the value of c?",
            )
        )
        f0 = _f0(kind, m, s)
        key = frac_text(c)
        wrong = [1 / c, c - 1, c * f0, f0 * c - f0]
        working = [
            f"The y-intercept of f is {f0} and the y-intercept of g is {c * f0}.",
            f"c = {c * f0} / {f0} = {frac_text(c)}.",
        ]
    elif tier == "medium":
        kind, m, s, k = rng.choice(_HORIZONTAL)
        form, v, c = "h", k, F(1)
        text = rng.choice(
            (
                f"The graph of g is the graph of f stretched or squashed left and right, so g(x) = f(kx) with k > 0. "
                f"{scale} Find k.",
                f"g is a horizontal dilation of f, with g(x) = f(kx) and k > 0. {scale} What is the value of k?",
            )
        )
        key = frac_text(k)
        root = F(m) / k
        wrong = [1 / k, k * k, root, m - root, F(m) * k]
        working = [
            f"The x-intercepts of f are ±{m} and the x-intercepts of g are ±{frac_text(root)}.",
            f"f(kx) = 0 when kx = {m}, so x = {m}/k: k = {m} / {frac_text(root)} = {frac_text(k)}.",
        ]
    else:
        if rng.random() < 0.5:
            kind, m, s, c = rng.choice(_VERTICAL)
            form, v, k = "v", c, F(1)
        else:
            kind, m, s, k = rng.choice(_HORIZONTAL)
            form, v, c = "h", k, F(1)
        text = rng.choice(
            (
                f"The graph of g is a dilation of the graph of f. {scale} Which equation gives g?",
                f"g is a dilation of f. {scale} Which equation gives g(x)?",
            )
        )
        key = _dil_eq(form, v)
        wrong = _dil_wrong(form, v)
        if form == "v":
            working = [
                "The x-intercepts of f and g are the same, so the dilation is vertical.",
                f"The y-intercept goes from {_f0(kind, m, s)} to {c * _f0(kind, m, s)}: {key}.",
            ]
        else:
            working = [
                "The y-intercepts of f and g are the same, so the dilation is horizontal.",
                f"The x-intercepts go from ±{m} to ±{frac_text(F(m) / k)}, divided by {frac_text(k)}: {key}.",
            ]
    fig = _dil_figure(kind, m, s, c, k)
    distractors = mcq(key, [str(w) if isinstance(w, str) else frac_text(w) for w in wrong])
    check(
        _read_dilation(type("Q", (), {"figure": fig, "text": text})) == key,
        "the dilation reader disagrees with the key",
    )
    figures.assert_no_leak(fig, [key, *distractors])
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=f"plain:{key}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _read_dilation(question) -> str:
    curves = by_label(question.figure)
    f_map, g_map = curve_map(curves["f"]), curve_map(curves["g"])
    text = question.text
    if "vertical dilation" in text or "up and down" in text:
        fits = _fit_dilation(f_map, g_map, forms=("c",))
        check(len(fits) == 1, "vertical dilation: not exactly one factor fits")
        return frac_text(fits[0][0])
    if "horizontal dilation" in text or "left and right" in text:
        fits = _fit_dilation(f_map, g_map, forms=("k",))
        check(len(fits) == 1, "horizontal dilation: not exactly one factor fits")
        return frac_text(fits[0][1])
    fits = _fit_dilation(f_map, g_map)
    check(len(fits) == 1, "dilation: not exactly one equation fits")
    c, k = fits[0]
    return _dil_eq("h" if k != 1 else "v", k if k != 1 else c)


figures.register_reader(DILATION, _read_dilation)


@generator(DILATION_EXAM, level=13, presets={"exam": dict(tier="exam")})
@retry_short
def ey_graph_dilation_exam(rng, tier="exam"):
    """Exam chain: k from the x-intercepts, c from the y-intercept, then the image of the marked point P."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a dilation with a marked point whose image stays on the grid"):
        kind, m, s, c, k = rng.choice(_BOTH)
        p = rng.choice([t for t in range(-6, 7) if t != 0 and (F(t) / k).denominator == 1])
        q = int(_base(kind, m, s)(p))
        if abs(q) <= 7 and abs(F(p) / k) <= 7 and abs(c * q) <= 7 and (c * q).denominator == 1 and q != 0:
            break
    fig = _dil_figure(kind, m, s, c, k, point=(p, q))
    text = (
        "The graph of g is a dilation of the graph of f, so g(x) = c f(kx) with k > 0. "
        "The point P is marked on the graph of f. Each grid square is 1 unit on both axes."
    )
    root, f0 = F(m) / k, _f0(kind, m, s)
    parts = [
        num_part(
            "Find the value of k.",
            k,
            [1 / k, root, F(m) * k],
            places=2,
            working=[
                f"The x-intercepts go from +-{m} to +-{frac_text(root)}, so k = {m} / {frac_text(root)} = {frac_text(k)}."
            ],
        ),
        num_part(
            "Find the value of c.",
            c,
            [1 / c, c * f0, -c],
            places=2,
            working=[f"The y-intercept goes from {f0} to {c * f0}, so c = {frac_text(c)}."],
        ),
        num_part(
            "Hence find the x-coordinate of the point on g that P moves to.",
            F(p) / k,
            [p * k, p, F(p) / c],
            places=2,
            follow=f"{p}/part_a",
            working=[f"The x-coordinate is divided by k: {p} ÷ {_bt(frac_text(k))} = {frac_text(F(p) / k)}."],
        ),
        num_part(
            "Hence find the y-coordinate of the point on g that P moves to.",
            c * q,
            [q / c, q, q * k],
            places=2,
            follow=f"{q}*part_b",
            working=[f"The y-coordinate is multiplied by c: {pn(q)} × {_bt(frac_text(c))} = {frac_text(c * q)}."],
        ),
    ]
    figures.assert_no_leak(fig, [frac_text(k), frac_text(c)])
    return dict(text=f"{text} {rng.choice(_TAILS)}", figure=fig, parts=parts)


def _read_dilation_exam(question) -> str:
    fig = question.figure
    curves = by_label(fig)
    fits = _fit_dilation(curve_map(curves["f"]), curve_map(curves["g"]))
    check(len(fits) == 1, "dilation exam: not exactly one (c, k) fits")
    c, k = fits[0]
    p, q = fig["points"][0]["at"]
    return f"(a) {_dec(k)}; (b) {_dec(c)}; (c) {_dec(F(p) / k)}; (d) {_dec(c * q)}"


def _dec(v: F) -> str:
    """A rational as a two-decimal-place answer shows it: ``2.00``, ``-0.50``."""
    from ..core import round_half_up, stated_precision_string

    return stated_precision_string(round_half_up(F(v), 2), 2)


# (the exam chain of an mcq skill has no registered reader: the sweep cannot hand a labelled reply to the mcq path)

# =============================================================================================== |f|, 1/f, f(|x|) (ACMSM099)

RECIP = "ey_graph_reciprocal_abs"
RECIP_EXAM = "ey_graph_reciprocal_abs_exam"

_G = {
    "|f(x)|": lambda v, av: abs(v),
    "-f(x)": lambda v, av: -v,
    "f(|x|)": lambda v, av: av,
    "1/f(x)": lambda v, av: 1 / v if v else None,
    "-|f(x)|": lambda v, av: -av,
    "f(-x)": lambda v, av: v,  # (read from f at -x; see _which_g)
}
_OPTIONS = tuple(f"g(x) = {name}" for name in _G)


def _f_family(rng, kind: str):
    """A line s(x - r) or a parabola s(x - r1)(x - r2) with integer zeros, returned as (fn, description)."""
    if kind == "line":
        s, r = rng.choice((1, -1, 2, -2)), rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
        return (lambda x: s * (x - r)), ("line", s, r)
    s = rng.choice((1, -1))
    r1 = rng.randint(-5, 0)
    r2 = r1 + rng.choice((2, 4))
    if r1 == 0 or r2 == 0:
        r1, r2 = -2, 2
    return (lambda x: s * (x - r1) * (x - r2)), ("quad", s, r1, r2)


def _g_curve(name: str, fn):
    g = {
        "|f(x)|": lambda x: abs(fn(x)),
        "-f(x)": lambda x: -fn(x),
        "f(|x|)": lambda x: fn(abs(x)),
        "1/f(x)": lambda x: 1 / fn(x),
        "-|f(x)|": lambda x: -abs(fn(x)),
        "f(-x)": lambda x: fn(-x),
    }[name]
    return sample(g, label="g")


def _which_g(f_map, g_map) -> str:
    """The one transformation whose values match g at every common integer x (>= 4 of them)."""
    fits = []
    for name, tf in _G.items():
        pairs = []
        for x, gv in g_map.items():
            if x.denominator != 1 or x not in f_map:
                continue
            fv = f_map[x]
            if name == "f(|x|)":
                want = f_map[abs(x)] if abs(x) in f_map else None
            elif name == "f(-x)":
                want = f_map[-x] if -x in f_map else None
            else:
                want = tf(fv, abs(fv))
            if want is not None:
                pairs.append((gv, want))
        if len(pairs) >= 4 and all(near(g, w) for g, w in pairs):
            fits.append(name)
    check(len(fits) == 1, f"the graph of g fits {len(fits)} transformations: {fits}")
    return fits[0]


def _distinct_forms(fn) -> bool:
    """f is non-trivial for the options: |f| and -f and f(|x|) and 1/f differ at a few integer points."""
    xs = [x for x in range(-6, 7) if abs(fn(x)) <= 7]
    return (
        any(fn(x) < 0 for x in xs)
        and any(fn(x) > 0 for x in xs)
        and any(fn(x) != fn(-x) for x in xs if -x in xs)
        and any(fn(x) not in (-1, 1) for x in xs)
    )


@generator(
    RECIP,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def ey_graph_reciprocal_abs(rng, tier="easy"):
    """Identify g (|f|, -f, f(|x|), 1/f) from the two graphs, or read a value of g from the graph of f."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    scale = rng.choice(_SCALE)
    if tier in ("easy", "medium"):
        name = rng.choice(
            ("|f(x)|", "-f(x)", "-|f(x)|", "f(-x)")
            if tier == "easy"
            else ("f(|x|)", "|f(x)|", "1/f(x)", "f(-x)", "-|f(x)|")
        )
        for _ in attempts(300, "a function f whose transformation is unambiguous"):
            fn, desc = _f_family(rng, rng.choice(("line", "quad")))
            if _distinct_forms(fn):
                fig = _plane([sample(fn, label="f"), _g_curve(name, fn)])
                try:
                    if _which_g(curve_map(by_label(fig)["f"]), curve_map(by_label(fig)["g"])) == name:
                        break
                except Exception:
                    continue
        if tier == "easy":
            text = rng.choice(
                (
                    f"The graph of g is the graph of f reflected in an axis, in part or in whole. {scale} Which equation gives g?",
                    f"g is drawn by reflecting f, wholly or partly. {scale} Which equation gives g(x)?",
                )
            )
        else:
            text = rng.choice(
                (
                    f"The graph of g is a transformation of the graph of f. {scale} {_LABELS_TEXT} Which equation gives g?",
                    f"g is drawn from f by one of the transformations below. {scale} Which equation gives g(x)?",
                )
            )
        key = f"g(x) = {name}"
        distractors = mcq(key, [o for o in _OPTIONS if o != key], count=4)
        working = {
            "|f(x)|": ["Every part of f below the x-axis is reflected above it; the rest is unchanged."],
            "-f(x)": ["Every point of f is reflected in the x-axis: above becomes below and below becomes above."],
            "f(|x|)": [
                "The part of f with x >= 0 is kept and reflected in the y-axis, so g is symmetric about the y-axis."
            ],
            "1/f(x)": ["g has a vertical asymptote at each zero of f, and is positive where f is positive."],
            "-|f(x)|": ["Every part of f above the x-axis is reflected below it, so g is never positive."],
            "f(-x)": [
                "Every point of f is reflected in the y-axis: the graph of g is the mirror image of f left to right."
            ],
        }[name] + [f"So {key}."]
        figures.assert_no_leak(fig, [key])
        return dict(
            text=text,
            answer=key,
            answer_text=key,
            answer_latex=f"plain:{key}",
            distractors=distractors,
            working=working,
            figure=fig,
        )
    # hard: a value of the transformed function, read from the graph of f alone
    mode = rng.choice(("abs", "fabs", "recip"))
    for _ in attempts(500, "a value of the transformed function that reads exactly off the graph"):
        fn, desc = _f_family(rng, rng.choice(("line", "quad")))
        xs = [x for x in range(-6, 7) if abs(fn(x)) <= 7]
        if mode == "abs":
            ts = [t for t in xs if fn(t) < 0 and t != 0]
        elif mode == "fabs":
            ts = [t for t in xs if t > 0 and -t in xs and fn(-t) != fn(t)]
        else:
            ts = [t for t in xs if fn(t) in (-2, -1, 1, 2) and t != 0]
        if ts:
            break
    t = rng.choice(ts)
    if mode == "abs":
        gname, at, ans = "|f(x)|", t, abs(F(fn(t)))
        wrong = [fn(t), abs(t), -abs(fn(t)), fn(-t), abs(fn(t)) + 1, abs(fn(t)) - 1]  # f(-t): read at -t
        working = [f"From the graph, f({t}) = {fn(t)}.", f"|f({t})| = |{fn(t)}| = {abs(fn(t))}."]
    elif mode == "fabs":
        at = -t
        gname, ans, wrong = "f(|x|)", F(fn(t)), [fn(-t), -fn(t), abs(fn(t)), -fn(-t), fn(t) + 1, fn(t) - 1]
        working = [f"|{at}| = {t}, so g({at}) = f({t}).", f"From the graph, f({t}) = {fn(t)}."]
    else:
        at = t
        gname, ans, wrong = (
            "1/f(x)",
            1 / F(fn(t)),
            [-1 / F(fn(t)), F(fn(t)), F(1, t), F(1, fn(t) ** 2), -F(1, t), F(1, 2 * fn(t)), F(2, fn(t))],
        )
        working = [f"From the graph, f({t}) = {fn(t)}.", f"1/f({t}) = 1/{fn(t)} = {frac_text(ans)}."]
    fig = _plane([sample(fn, label="f")], alt="the graph of f on a coordinate grid, one unit per grid square")
    text = rng.choice(
        (
            f"The graph of f is drawn. Let g(x) = {gname}. {scale} Find g({at}).",
            f"Here g(x) = {gname}, and the graph of f is drawn. {scale} What is the value of g({at})?",
        )
    )
    key = frac_text(ans)
    distractors = mcq(key, [frac_text(F(w)) for w in wrong], minimum=3)
    q = type("Q", (), {"figure": fig, "text": text})
    check(_read_recip(q) == key, "the reciprocal reader disagrees with the key")
    figures.assert_no_leak(fig, [key])
    return dict(
        text=text,
        answer=key,
        answer_text=key,
        answer_latex=f"plain:{key}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


_VALUE = re.compile(r"g\(x\) = (\|f\(x\)\||f\(\|x\|\)|1/f\(x\)).*?g\((-?\d+)\)")
_VALUE2 = re.compile(r"Let g\(x\) = (\|f\(x\)\||f\(\|x\|\)|1/f\(x\))\.")


def _read_recip(question) -> str:
    fig, text = question.figure, question.text
    curves = by_label(fig)
    if "Which equation" in text:
        return f"g(x) = {_which_g(curve_map(curves['f']), curve_map(curves['g']))}"
    m = re.search(r"g\(x\) = (\|f\(x\)\||f\(\|x\|\)|1/f\(x\))", text)
    at = int(re.search(r"g\((-?\d+)\)", text.split("g(x) =", 1)[1]).group(1))
    check(m is not None, "cannot read the transformation")
    f = {x: F(y).limit_denominator(1) for x, y in _ints(curves["f"]).items()}
    name = m.group(1)
    if name == "|f(x)|":
        return frac_text(abs(f[at]))
    if name == "f(|x|)":
        return frac_text(f[abs(at)])
    return frac_text(1 / f[at])


figures.register_reader(RECIP, _read_recip)


@functools.cache
def _recip_exam_options():
    """Every parabola a (x - r1)(x - r2): integer zeros of even sum in the window, a turning point of whole height
    at most 7 and a stretch factor a from 1 to 6, 1/2, 1/3, 1/4 (either sign)."""
    out = []
    for r1 in range(-6, 5):
        for r2 in range(r1 + 2, 7, 2):
            half = F(r2 - r1, 2)
            for mag in (F(1), F(2), F(3), F(4), F(5), F(6), F(1, 2), F(1, 3), F(1, 4)):
                if (mag * half * half).denominator == 1 and mag * half * half <= 7:
                    out += [(r1, r2, mag), (r1, r2, -mag)]
    return tuple(out)


@generator(RECIP_EXAM, level=13, presets={"exam": dict(tier="exam")})
@retry_short
def ey_graph_reciprocal_abs_exam(rng, tier="exam"):
    """Exam chain: the asymptotes of 1/f from the zeros of f, their distance apart and the midpoint between them."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    r1, r2, a = rng.choice(_recip_exam_options())
    fn = lambda x: float(a) * (x - r1) * (x - r2)  # noqa: E731
    fig = _plane([sample(fn, label="f")], alt="the graph of f on a coordinate grid, one unit per grid square")
    text = (
        "The graph of f is drawn, a parabola that crosses the x-axis twice. "
        "The graph of y = 1/f(x) has two vertical asymptotes. Each grid square is 1 unit on both axes."
    )
    mid = (r1 + r2) // 2
    top = int(a * (mid - r1) * (mid - r2))
    parts = [
        num_part(
            "Find the x-value of the vertical asymptote on the left.",
            r1,
            [r2, -r1, mid],
            working=[f"1/f is undefined where f(x) = 0, and the left zero of f is x = {r1}."],
        ),
        num_part(
            "Find the x-value of the vertical asymptote on the right.",
            r2,
            [r1, -r2, mid],
            working=[f"The right zero of f is x = {r2}."],
        ),
        num_part(
            "Hence find the distance between the two asymptotes.",
            r2 - r1,
            [r2 + r1, (r2 - r1) // 2, r2 - r1 + 2],
            follow="part_b-part_a",
            working=[f"{r2} - ({r1}) = {r2 - r1}."],
        ),
        num_part(
            "Hence find the x-value halfway between the asymptotes.",
            mid,
            [r2 - r1, (r2 - r1) // 2, r1],
            follow="(part_a+part_b)/2",
            working=[f"({r1} + {pn(r2)}) / 2 = {mid}, the axis of symmetry of f."],
        ),
        num_part(
            "Hence find the y-value of the turning point of the graph of f.",
            top,
            [-top, mid, 2 * top],
            working=[f"The turning point of f is at x = {mid}; the graph shows y = {top} there."],
        ),
    ]
    figures.assert_no_leak(fig, [str(r1), str(r2), str(top)])
    return dict(text=f"{text} {rng.choice(_TAILS)}", figure=fig, parts=parts)


def _read_recip_exam(question) -> str:
    f = {x: y for x, y in _ints(by_label(question.figure)["f"]).items()}
    zeros = sorted(x for x, y in f.items() if near(y, 0))
    check(len(zeros) == 2, "the parabola does not cross the axis twice at integers")
    r1, r2 = zeros
    mid = (r1 + r2) // 2
    return f"(a) {r1}; (b) {r2}; (c) {r2 - r1}; (d) {mid}; (e) {round(f[mid])}"


# (the exam chain of an mcq skill has no registered reader: the sweep cannot hand a labelled reply to the mcq path)

_ = (TooFewDistractors, pick_distractors)
