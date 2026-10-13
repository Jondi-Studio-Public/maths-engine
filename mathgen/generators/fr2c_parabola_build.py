"""Generates the parabola parameter questions: `fr2c_parabola_effects`, `fr2c_parabola_equation` and their exam chains.

Two skills share this module. Both work with ``y = a(x - h)^2 + k`` and the graph of ``y = x^2``.

``f10-quadratic-parameter-effects`` (``answer: mcq``, four options each time) is about what each parameter does:

- easy: graph B is ``y = x^2`` moved sideways or up or down (a drawn pair of curves); which equation is B? The wrong options move
  the graph the other way or along the other axis.
- medium: an equation ``y = a(x - h)^2 + k`` with ``a`` equal to 1 or -1 is given and the options describe the graph in
  words (reflected or not, translated left or right, up or down).
- hard: either B is drawn with ``a`` equal to 2, 1/2, -2 or -1/2 next to ``y = x^2`` and the options describe its width, the way it
  opens and its vertex, or the image of a curve ``y = f(x)`` is drawn and the options are ``f(x - h) + k`` and its look-alikes.

``f10-parabola-equation-from-graph`` (``answer: expression``) writes the equation of a drawn parabola:

- easy: ``a`` is 1 or -1 and the vertex is on the grid, so only ``h``, ``k`` and the way it opens have to be read;
- medium: ``a`` is 2, -2, 3 or -3 and a point P on the curve is marked;
- hard: ``a`` is a fraction such as 1/2 or -1/3, found from the marked point, which is not next to the vertex.

``fr2c_parabola_effects_exam`` and ``fr2c_parabola_equation_exam`` are the chains. Every curve is sampled every 1/8 so each
integer x is an exact sample.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy import Symbol

from mathgen import figures
from mathgen.core import attempts, check, check_equivalent, generator, phrase

from . import _fr2c_kit as kit

X = Symbol("x")
_LO, _HI = -10, 10
_N = 161
_CAP = "^"


# ------------------------------------------------------------------------------------------ text helpers


def _frac(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def vertex_form(a: Fraction, h: int, k: int, *, y: bool = True) -> str:
    """``a(x - h)^2 + k`` as written on the page: no 1 in front, no ``- 0`` and no ``+ -``."""
    a = Fraction(a)
    coef = "" if a == 1 else "-" if a == -1 else _frac(a) if a.denominator == 1 else f"({_frac(a)})"
    if h == 0:
        body = f"{coef}x{_CAP}2"
    else:
        body = f"{coef}(x {'-' if h > 0 else '+'} {abs(h)}){_CAP}2"
    if k:
        body += f" {'+' if k > 0 else '-'} {abs(k)}"
    return ("y = " if y else "") + body


def expr_of(a: Fraction, h: int, k: int):
    a = Fraction(a)
    return sympy.Rational(a.numerator, a.denominator) * (X - h) ** 2 + k


def _curve(f) -> dict:
    return {"pieces": figures.sample_curve(f, _LO, _HI, (_LO, _HI), n=_N)}


def _plane(curves, *, alt, points=()):
    return figures.cartesian(
        x_range=(_LO, _HI), y_range=(_LO, _HI), grid=1, equal_scale=True, alt=alt, curves=curves, points=points
    )


def _parabola(a, h, k, label=None):
    a, h, k = float(a), int(h), int(k)
    cv = _curve(lambda x: a * (x - h) ** 2 + k)
    if label:
        cv["label"] = label
    return cv


def _visible(a, h, k, span: int = 3) -> bool:
    """The vertex and the points ``span`` units either side of it lie inside the window."""
    a = Fraction(a)
    return all(-9 <= a * d * d + k <= 9 for d in (0, span, -span)) and -8 <= h <= 8 and -8 <= k <= 8


def _nonzero(rng, lo, hi):
    return rng.choice([v for v in range(lo, hi + 1) if v])


# ------------------------------------------------------------------------------------------ effects: easy


def _effects_easy(rng):
    kind = rng.choice(["right", "left", "up", "down"])
    n = rng.randint(1, 5) + (1 if kind in ("up", "down") else 0)
    h, k = {"right": (n, 0), "left": (-n, 0), "up": (0, n), "down": (0, -n)}[kind]
    key = vertex_form(1, h, k)
    cand = []
    if k == 0:
        cand = [
            ("fr2c-shift-direction-reversed", vertex_form(1, -h, 0)),
            ("fr2c-shift-on-wrong-axis", vertex_form(1, 0, h)),
            ("fr2c-shift-on-wrong-axis", vertex_form(1, 0, -h)),
        ]
    else:
        cand = [
            ("fr2c-vertical-shift-direction-reversed", vertex_form(1, 0, -k)),
            ("fr2c-shift-on-wrong-axis", vertex_form(1, k, 0)),
            ("fr2c-shift-on-wrong-axis", vertex_form(1, -k, 0)),
        ]
    fig = _plane(
        [_parabola(1, 0, 0, "y = x^2"), _parabola(1, h, k, "B")], alt="The parabola y = x^2 and a second parabola B."
    )
    text = phrase(
        rng,
        (
            "The graph of y = x^2 is drawn, and so is graph B, which is y = x^2 moved without turning or stretching it. "
            "Which equation is graph B?",
            "Graph B is the graph of y = x^2 after it has been moved. Both are drawn. Which equation is graph B?",
        ),
    )
    working = [f"The vertex of B is at ({h}, {k}), so the equation is {key}."]
    return text, kit.mcq_fields(key, cand), working, fig


# ------------------------------------------------------------------------------------------ effects: medium


def _words(refl: bool, h: int, k: int) -> str:
    parts = []
    if h:
        parts.append(f"{abs(h)} unit{'s' if abs(h) != 1 else ''} to the {'right' if h > 0 else 'left'}")
    if k:
        parts.append(f"{abs(k)} unit{'s' if abs(k) != 1 else ''} {'up' if k > 0 else 'down'}")
    move = " and ".join(parts)
    start = "y = x^2 reflected in the x-axis and then translated " if refl else "y = x^2 translated "
    return f"The graph of {start}{move}."


def _effects_medium(rng):
    h, k = _nonzero(rng, -6, 6), _nonzero(rng, -6, 6)
    a = rng.choice([1, -1])
    refl = a == -1
    eq = vertex_form(a, h, k)
    key = _words(refl, h, k)
    cand = [
        ("fr2c-shift-direction-reversed", _words(refl, -h, k)),
        ("fr2c-vertical-shift-direction-reversed", _words(refl, h, -k)),
        ("fr2c-reflection-missed" if refl else "fr2c-reflected-when-not-needed", _words(not refl, h, k)),
    ]
    rng.shuffle(cand)
    text = phrase(
        rng,
        (
            "Which statement describes the graph of {eq} compared with the graph of y = x^2?",
            "How is the graph of {eq} related to the graph of y = x^2?",
        ),
        eq=eq,
    )
    working = [
        f"In y = a(x - h)^2 + k the vertex is at (h, k) = ({h}, {k}), and a = {a}.",
        f"So the graph is y = x^2 {'reflected in the x-axis, ' if refl else ''}moved {abs(h)} {'right' if h > 0 else 'left'} and {abs(k)} {'up' if k > 0 else 'down'}.",
    ]
    return text, kit.mcq_fields(key, cand), working, None


# ------------------------------------------------------------------------------------------ effects: hard

_BASES = {
    "|x|": (lambda x: abs(x), "|x|"),
    "x^2": (lambda x: x * x, "x^2"),
    "x^3/4": (lambda x: x**3 / 4, "x^3/4"),
}


def _f_arg(f: str, shift: str) -> str:
    return f"f({shift})"


def _fshift(h: int, k: int) -> str:
    arg = "x" if h == 0 else f"x {'-' if h > 0 else '+'} {abs(h)}"
    out = f"y = f({arg})"
    if k:
        out += f" {'+' if k > 0 else '-'} {abs(k)}"
    return out


def _effects_hard_width(rng):
    for _ in attempts(500, "a visible parabola"):
        a = rng.choice([Fraction(2), Fraction(1, 2), Fraction(-2), Fraction(-1, 2)])
        h, k = _nonzero(rng, -5, 5), _nonzero(rng, -5, 5)
        if _visible(a, h, k, 2):
            break
    narrow = abs(a) > 1
    up = a > 0

    def stmt(narrow, up, h, k):
        return (
            f"It is {'narrower' if narrow else 'wider'} than y = x^2, opens {'upwards' if up else 'downwards'}, and has its "
            f"vertex at ({h}, {k})."
        )

    key = stmt(narrow, up, h, k)
    attrs = [
        ("fr2c-width-reversed", stmt(not narrow, up, h, k)),
        ("fr2c-reflection-missed" if not up else "fr2c-reflected-when-not-needed", stmt(narrow, not up, h, k)),
        ("fr2c-vertex-sign-slip", stmt(narrow, up, -h, k)),
        ("fr2c-vertex-sign-slip", stmt(narrow, up, h, -k)),
    ]
    rng.shuffle(attrs)
    fig = _plane(
        [_parabola(1, 0, 0, "y = x^2"), _parabola(a, h, k, "B")], alt="The parabola y = x^2 and a second parabola B."
    )
    text = phrase(
        rng,
        (
            "The graphs of y = x^2 and of another parabola, B, are drawn. Which statement about B is correct?",
            "A second parabola, B, is drawn on the same axes as y = x^2. Which statement about B is correct?",
        ),
    )
    working = [
        f"B is {'narrower' if narrow else 'wider'} than y = x^2 and opens {'upwards' if up else 'downwards'}.",
        f"Its turning point is at ({h}, {k}).",
    ]
    return text, kit.mcq_fields(key, attrs), working, fig


def _effects_hard_f(rng):
    name = rng.choice(list(_BASES))
    f, _ = _BASES[name]
    for _ in attempts(200, "a shift that keeps both curves in view"):
        h, k = _nonzero(rng, -4, 4), _nonzero(rng, -4, 4)
        if name == "x^2" and k < -3:
            continue
        if name == "x^3/4" and abs(h) > 2:
            continue
        break
    key = _fshift(h, k)
    cand = [
        ("fr2c-shift-direction-reversed", _fshift(-h, k)),
        ("fr2c-vertical-shift-direction-reversed", _fshift(h, -k)),
        ("fr2c-shift-on-wrong-axis", _fshift(k, h)),
    ]
    rng.shuffle(cand)
    fig = _plane(
        [_curve(lambda x: f(x)) | {"label": "y = f(x)"}, _curve(lambda x: f(x - h) + k) | {"label": "B"}],
        alt="The graph of y = f(x) and a second graph B, the same shape in a new position.",
    )
    text = phrase(
        rng,
        (
            "The graph of y = f(x) is drawn, with graph B, which is the same shape in a new position. Which equation is graph B?",
            "Graph B is the graph of y = f(x) moved without turning it or changing its size. Which equation gives graph B?",
        ),
    )
    working = [
        f"The graph moved {abs(h)} unit{'s' if abs(h) != 1 else ''} {'right' if h > 0 else 'left'} and "
        f"{abs(k)} unit{'s' if abs(k) != 1 else ''} {'up' if k > 0 else 'down'}.",
        f"A move of {abs(h)} {'right' if h > 0 else 'left'} changes x to x {'-' if h > 0 else '+'} {abs(h)}, and a move of {abs(k)} {'up' if k > 0 else 'down'} adds {k}. So B is {key}.",
    ]
    return text, kit.mcq_fields(key, cand), working, fig


def _effects_hard(rng):
    return _effects_hard_width(rng) if rng.random() < 0.5 else _effects_hard_f(rng)


_EFFECT_TIERS = {"easy": _effects_easy, "medium": _effects_medium, "hard": _effects_hard}


@generator(
    "fr2c_parabola_effects",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2c_parabola_effects(rng, tier="easy"):
    """What the parameters of y = a(x - h)^2 + k, or the shift in y = f(x - h) + k, do to a graph."""
    if tier not in _EFFECT_TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(60, "a question with three different wrong options"):
        try:
            text, fields, working, fig = _EFFECT_TIERS[tier](rng)
        except kit.Redraw:
            continue
        break
    out = dict(text=text, working=working, **fields)
    if fig is not None:
        out["figure"] = fig
    return out


# ------------------------------------------------------------------------------------------ figure readers


def _points(curve) -> list:
    return [p for piece in curve["pieces"] for p in piece]


def _vertex(points) -> tuple[int, int]:
    """The turning point of a parabola's samples: the extreme sample nearest the middle of its x-spread."""
    xs = [p[0] for p in points]
    mid = (min(xs) + max(xs)) / 2
    low, high = min(points, key=lambda p: p[1]), max(points, key=lambda p: p[1])
    x, y = min((low, high), key=lambda p: abs(p[0] - mid))
    return round(x), round(y)


def _y_at(points, x) -> float:
    return min(points, key=lambda p: abs(p[0] - x))[1]


def _read_shift(base, image) -> tuple[int, int]:
    """The (h, k) for which every shared sample of ``base`` moved by it lies on ``image``."""
    on_image = {p[0]: p[1] for p in image}
    for h in range(-6, 7):
        for k in range(-6, 7):
            shared = [(x, y) for x, y in ((p[0], p[1]) for p in base) if round(x + h, 3) in on_image]
            if len(shared) >= 20 and all(abs(on_image[round(x + h, 3)] - (y + k)) < 1e-6 for x, y in shared):
                return h, k
    raise AssertionError("graph B is not a shifted copy of the first graph")


def _statement(narrow: bool, up: bool, h: int, k: int) -> str:
    return (
        f"It is {'narrower' if narrow else 'wider'} than y = x^2, opens {'upwards' if up else 'downwards'}, and has its "
        f"vertex at ({h}, {k})."
    )


def _read_effects(question) -> str:
    """The key, recomputed from the drawn curves alone."""
    curves = {c.get("label"): c for c in question.figure["curves"]}
    b = _points(curves["B"])
    if "y = f(x)" in curves:
        return _fshift(*_read_shift(_points(curves["y = f(x)"]), b))
    h, k = _vertex(b)
    if "Which equation is graph B" in question.text:
        return vertex_form(1, h, k)
    rise = _y_at(b, h + 1) - k
    return _statement(abs(rise) > 1, rise > 0, h, k)


figures.register_reader("fr2c_parabola_effects", _read_effects)


# ------------------------------------------------------------------------------------------ equation from a graph


def _marked_point(rng, a, h, k, tier):
    """A point P on the curve, with whole-number coordinates, that is not the vertex."""
    a = Fraction(a)
    options = []
    for dx in (1, -1, 2, -2, 3, -3):
        y = a * dx * dx + k
        if y.denominator == 1 and -9 <= y <= 9 and -9 <= h + dx <= 9:
            options.append((h + dx, int(y)))
    near = [o for o in options if abs(o[0] - h) == 1]
    far = [o for o in options if abs(o[0] - h) >= 2]
    pool = near if tier == "medium" and near else far if tier == "hard" and far else options
    return rng.choice(pool)


def _equation_spec(rng, tier):
    for _ in attempts(2000, "a parabola with a readable vertex and marked point"):
        h, k = _nonzero(rng, -5, 5), _nonzero(rng, -5, 5)
        if tier == "easy":
            a = Fraction(rng.choice([1, -1]))
        elif tier == "medium":
            a = Fraction(rng.choice([2, -2, 3, -3]))
        else:
            a = Fraction(rng.choice([1, -1])) / rng.choice([2, 2, 3])
        if not _visible(a, h, k, 2 if tier != "hard" else 3):
            continue
        if tier == "easy":
            return a, h, k, None
        try:
            p = _marked_point(rng, a, h, k, tier)
        except IndexError:
            continue
        if p[0] != h:
            return a, h, k, p
    raise AssertionError("unreachable")


def _equation_figure(a, h, k, p):
    pts = [{"at": [p[0], p[1]], "label": "P"}] if p else []
    return _plane([_parabola(a, h, k)], alt="A parabola drawn on a grid.", points=pts)


def _wrong(mid, a, h, k, *, y=True):
    """A wrong equation as ``(id, expression, text)``."""
    a = Fraction(a)
    return (mid, expr_of(a, h, k), vertex_form(a, h, k, y=False))


def _equation_wrong(a, h, k, p):
    a = Fraction(a)
    cands = [
        _wrong("fr2c-horizontal-shift-sign-reversed", a, -h, k),
        _wrong("fr2c-vertical-shift-sign-reversed", a, h, -k),
        _wrong("fr2c-reflection-missed", -a, h, k),
    ]
    if p:
        cands.append(_wrong("fr2c-a-taken-as-the-rise", Fraction(p[1] - k), h, k))
    else:
        cands.append(_wrong("fr2c-horizontal-shift-sign-reversed", a, -h, -k))
    if abs(a) != 1:
        cands.append(_wrong("fr2c-width-factor-ignored", Fraction(1) if a > 0 else Fraction(-1), h, k))
    return cands


def _equation_question(rng, tier):
    a, h, k, p = _equation_spec(rng, tier)
    answer = expr_of(a, h, k)
    if p:
        # second way: a from the marked point and the vertex
        check_equivalent(sympy.Rational(p[1] - k, (p[0] - h) ** 2), sympy.Rational(a.numerator, a.denominator))
    fields = kit.expr_fields(
        answer, _equation_wrong(a, h, k, p), var=X, prefix="y = ", text=vertex_form(a, h, k, y=False)
    )
    text = phrase(
        rng,
        (
            "The graph of a parabola is drawn{pt}. Write its equation in the form y = a(x - h)^2 + k.",
            "A parabola is drawn on the grid{pt}. Find its equation in the form y = a(x - h)^2 + k.",
        ),
        pt="" if p is None else ", with the point P marked on it",
    )
    if tier == "hard":
        text += " Write a as a fraction if it is not a whole number."
    working = [f"The vertex is at ({h}, {k}), so h = {h} and k = {k}."]
    if p:
        working.append(
            f"P({p[0]}, {p[1]}) is on the graph, so {p[1]} = a x ({p[0] - h})^2 {'+' if k >= 0 else '-'} {abs(k)}, which gives a = {_frac(a)}."
        )
    else:
        working.append(
            f"The graph opens {'upwards' if a > 0 else 'downwards'} and has the same width as y = x^2, so a = {_frac(a)}."
        )
    working.append(f"The equation is {vertex_form(a, h, k)}.")
    return text, fields, working, _equation_figure(a, h, k, p)


@generator(
    "fr2c_parabola_equation",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2c_parabola_equation(rng, tier="easy"):
    """The equation y = a(x - h)^2 + k of a drawn parabola. See the module docstring for the tiers."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(60, "an equation question with three different wrong equations"):
        try:
            text, fields, working, fig = _equation_question(rng, tier)
        except kit.Redraw:
            continue
        break
    return dict(text=text, working=working, figure=fig, **fields)


def _read_equation(question) -> str:
    fig = question.figure
    pts = _points(fig["curves"][0])
    h, k = _vertex(pts)
    if fig["points"]:
        px, py = fig["points"][0]["at"]
        a = Fraction(py - k).limit_denominator(1000) / Fraction((px - h) ** 2)
    else:
        a = Fraction(1) if _y_at(pts, h + 1) > k else Fraction(-1)
    return vertex_form(a, h, k)


figures.register_reader("fr2c_parabola_equation", _read_equation)


# ------------------------------------------------------------------------------------------ exam chains


def _paren(v) -> str:
    v = Fraction(v)
    return (
        str(v.numerator)
        if v >= 0 and v.denominator == 1
        else f"({v.numerator}/{v.denominator})"
        if v.denominator != 1
        else f"({v.numerator})"
    )


@generator("fr2c_parabola_effects_exam", level=9, presets={"exam": {}})
def fr2c_parabola_effects_exam(rng):
    """An equation y = a(x - h)^2 + k: how its graph compares with y = x^2, its vertex, its turning value, its y-intercept."""
    return kit.build_with_redraw(rng, _build_fr2c_parabola_effects_exam, what="fr2c_parabola_effects_exam")


def _build_fr2c_parabola_effects_exam(rng):
    for _ in attempts(500, "a parabola with a non-zero vertex and a different intercept"):
        a = Fraction(rng.choice([-3, -2, 2, 3])) if rng.random() < 0.65 else Fraction(rng.choice([-1, 1]), 2)
        h, k = _nonzero(rng, -5, 5), _nonzero(rng, -8, 8)
        if a.denominator != 1 and h % 2:
            continue
        if a * h * h + k not in (k, 0):
            break
    value = a * h * h + k
    check(value.denominator == 1, "the y-intercept is not a whole number")
    value = int(value)
    eq = vertex_form(a, h, k)
    narrow, up = abs(a) > 1, a > 0
    word = "greatest" if a < 0 else "least"

    def stmt(n, u):
        return f"It is {'narrower' if n else 'wider'} than y = x^2 and opens {'upwards' if u else 'downwards'}"

    parts = [
        kit.mcq_part(
            "Which statement about the graph is correct?",
            stmt(narrow, up),
            2,
            [
                f"|a| = {_frac(abs(a))}, which is {'more' if narrow else 'less'} than 1, so the graph is {'narrower' if narrow else 'wider'}; a is {'positive' if up else 'negative'}, so it opens {'upwards' if up else 'downwards'}."
            ],
            [
                ("fr2c-width-reversed", stmt(not narrow, up)),
                ("fr2c-reflection-missed" if not up else "fr2c-reflected-when-not-needed", stmt(narrow, not up)),
                ("fr2c-width-reversed", stmt(not narrow, not up)),
            ],
        ),
        kit.num_part(
            "What is the x-coordinate of the vertex?",
            h,
            1,
            [f"In y = a(x - h)^2 + k the vertex is at x = h, so x = {h}."],
            [
                ("fr2c-horizontal-shift-sign-reversed", -h),
                ("fr2c-vertex-coordinates-swapped", k),
                ("fr2c-a-taken-as-the-vertex", int(a) if a.denominator == 1 else 1),
            ],
        ),
        kit.num_part(
            f"What is the {word} value of y on this graph?",
            k,
            1,
            [f"The vertex is the {'highest' if a < 0 else 'lowest'} point, so the {word} value of y is k = {k}."],
            [
                ("fr2c-vertical-shift-sign-reversed", -k),
                ("fr2c-vertex-coordinates-swapped", h),
                ("fr2c-a-taken-as-the-vertex", int(a) if a.denominator == 1 else 1),
            ],
        ),
        kit.num_part(
            "Using your vertex, find the y-intercept of the graph.",
            value,
            2,
            [f"Put x = 0: y = {_frac(a)} x ({-h})^2 {'+' if k >= 0 else '-'} {abs(k)} = {value}."],
            [
                ("fr2c-bracket-not-squared", int(a * (-h) + k)),
                ("fr2c-intercept-is-the-vertex-height", k),
                ("fr2c-square-applied-after-multiplying", int((a * h) ** 2 + k)),
                ("fr2c-constant-dropped", int(a * h * h)),
            ],
            follow_through=f"{_paren(a)}*part_b**2 + part_c",
        ),
    ]
    stem = f"A parabola has the equation {eq}."
    return dict(text=stem, parts=parts)


@generator("fr2c_parabola_equation_exam", level=9, presets={"exam": {}})
def fr2c_parabola_equation_exam(rng):
    """A drawn parabola with a marked point: h, k, a, then the equation."""
    return kit.build_with_redraw(rng, _build_fr2c_parabola_equation_exam, what="fr2c_parabola_equation_exam")


def _build_fr2c_parabola_equation_exam(rng):
    for _ in attempts(2000, "a parabola with a marked point and a readable a"):
        a = Fraction(rng.choice([2, -2, 3, -3, 1, -1])) if rng.random() < 0.6 else Fraction(rng.choice([1, -1]), 2)
        if abs(a) == 1:
            continue
        h, k = _nonzero(rng, -5, 5), _nonzero(rng, -5, 5)
        if not _visible(a, h, k, 2):
            continue
        try:
            p = _marked_point(rng, a, h, k, "medium" if a.denominator == 1 else "hard")
        except IndexError:
            continue
        if p[0] != h:
            break
    px, py = p
    check(Fraction(py - k, (px - h) ** 2) == a, "the marked point does not give a")
    places = 0 if a.denominator == 1 else 1
    answer = expr_of(a, h, k)
    stem = (
        "The graph of a parabola is drawn. Its vertex is on whole-number coordinates and the point P is marked on it."
    )
    parts = [
        kit.num_part(
            "Write the x-coordinate of the vertex.",
            h,
            1,
            [f"The turning point is at x = {h}."],
            [
                ("fr2c-horizontal-shift-sign-reversed", -h),
                ("fr2c-vertex-coordinates-swapped", k),
                ("fr2c-point-read-for-vertex", px),
            ],
        ),
        kit.num_part(
            "Write the y-coordinate of the vertex.",
            k,
            1,
            [f"The turning point is at y = {k}."],
            [
                ("fr2c-vertical-shift-sign-reversed", -k),
                ("fr2c-vertex-coordinates-swapped", h),
                ("fr2c-point-read-for-vertex", py),
            ],
        ),
        kit.num_part(
            "Use the point P and your vertex to find a, in y = a(x - h)^2 + k.",
            a,
            3,
            [
                f"P is at ({px}, {py}), so {py} = a x ({px - h})^2 {'+' if k >= 0 else '-'} {abs(k)} and a = {py - k} / {(px - h) ** 2}."
            ],
            [
                ("fr2c-a-taken-as-the-rise", py - k),
                ("fr2c-distance-not-squared", Fraction(py - k, abs(px - h))),
                ("fr2c-vertex-height-ignored", Fraction(py, (px - h) ** 2)),
                ("fr2c-reflection-missed", -a),
            ],
            places=places,
            follow_through=f"({py} - part_b)/({px} - part_a)**2",
        ),
        kit.expr_part(
            "Write the equation of the parabola in the form y = a(x - h)^2 + k.",
            answer,
            2,
            [f"With a = {_frac(a)}, h = {h} and k = {k}, the equation is {vertex_form(a, h, k)}."],
            [
                _wrong("fr2c-horizontal-shift-sign-reversed", a, -h, k),
                _wrong("fr2c-vertical-shift-sign-reversed", a, h, -k),
                _wrong("fr2c-reflection-missed", -a, h, k),
                _wrong("fr2c-width-factor-ignored", Fraction(1) if a > 0 else Fraction(-1), h, k),
            ],
            var=X,
            prefix="y = ",
            text=vertex_form(a, h, k, y=False),
            follow_through="part_c*(x - part_a)**2 + part_b",
        ),
    ]
    return dict(text=stem, figure=_equation_figure(a, h, k, p), parts=parts)
