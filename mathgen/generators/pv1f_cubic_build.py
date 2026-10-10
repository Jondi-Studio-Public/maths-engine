"""Generates `pv1f_cubic_y_intercept` questions: read the y-intercept of a drawn cubic.

``pv1f_cubic_y_intercept`` (senior-recognise-features-of-the-graphs-of-y-x-acmmm017, ACMMM017):
given a cubic ``y = a(x - b)^3 + c`` or ``y = k(x - p)(x - q)(x - r)`` drawn on an integer
grid, type the value of y where the curve crosses the y-axis (an integer by construction).
The key is drawn FIRST and uniformly, then a cubic realising it is chosen. Tiers escalate by
the decision the learner makes:

- easy:   one S-shaped cubic ``a(x - b)^3 + c``; read the height at the y-axis, not at the
          centre of the S;
- medium: one cubic with three x-intercepts; the y-axis crossing is none of the roots;
- hard:   the cubic is drawn with a horizontal line L that also crosses the y-axis;
          the intercept asked is the cubic's, never L's;
- exam:   two cubics are drawn and the key belongs to the one through the lettered point P.

Every curve is sampled with step 1/8, so x = 0 and every integer x is an EXACT sample of the
polyline the learner sees. Nothing the key needs is printed: alt text names the object, the
only label is a single capital on a point or a line. The topic registers a figure reader that
works from ``question.figure`` alone (plus the LETTER the prompt names on the exam tier).
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import figures
from mathgen.core import check, check_pairwise_distinct, generator, pick_distractors

_LO, _HI = -10, 10  # x and y window, grid 1
_N = 161  # 20 units * 8 + 1 -> step 0.125; x = 0 is sample 80
_TIERS = ("easy", "medium", "hard", "exam")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_MAX_INTERCEPT = 8
_MAX_SLOPE_AT_AXIS = 6  # the curve must cross the y-axis at a readable angle

_A_SHIFT = (Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2))
_K_ROOTS = (Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2))


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


# ----------------------------------------------------------------------- the cubics
# A spec is a tuple whose first item names the family: ("shift", a, b, c) or
# ("roots", k, p, q, r). Fractions throughout; floats only when sampling.


def _value(spec, x) -> Fraction:
    x = Fraction(x)
    if spec[0] == "shift":
        _f, a, b, c = spec
        return a * (x - b) ** 3 + c
    _f, k, p, q, r = spec
    return k * (x - p) * (x - q) * (x - r)


def _y_intercept(spec) -> Fraction:
    """The constant term of the EXPANDED cubic: an independent route to the key."""
    if spec[0] == "shift":
        _f, a, b, c = spec
        return c - a * b**3
    _f, k, p, q, r = spec
    return -k * p * q * r


def _slope_at_axis(spec) -> Fraction:
    if spec[0] == "shift":
        _f, a, b, _c = spec
        return 3 * a * b * b
    _f, k, p, q, r = spec
    return k * (p * q + p * r + q * r)


def _shift_specs(y0: int) -> list:
    out = []
    for a in _A_SHIFT:
        for b in (-1, 1) if abs(a) == 1 else (-2, 2):
            c = y0 + a * b**3
            if c.denominator == 1 and abs(c) <= 7:
                out.append(("shift", a, Fraction(b), c))
    return out


_ROOT_SETS = [(p, q, r) for p in range(-4, 5) for q in range(p + 1, 5) for r in range(q + 1, 5) if 0 not in (p, q, r)]


def _roots_ok(spec) -> bool:
    _f, k, p, q, r = spec
    if abs(_slope_at_axis(spec)) > _MAX_SLOPE_AT_AXIS:
        return False
    y0 = _y_intercept(spec)
    if y0.denominator != 1 or y0 == 0 or abs(y0) > _MAX_INTERCEPT:
        return False
    # the turning points must sit inside the window so the whole S-bend is visible
    xs = [Fraction(p) + Fraction(i, 8) for i in range(int((r - p) * 8) + 1)]
    return max(abs(_value(spec, x)) for x in xs) <= 9


def _root_specs(y0: int) -> list:
    out = []
    for k in _K_ROOTS:
        for p, q, r in _ROOT_SETS:
            spec = ("roots", k, Fraction(p), Fraction(q), Fraction(r))
            if _y_intercept(spec) == y0 and _roots_ok(spec):
                out.append(spec)
    return out


_SPECS_BY_Y0 = {y0: (_shift_specs(y0), _root_specs(y0)) for y0 in range(-_MAX_INTERCEPT, _MAX_INTERCEPT + 1) if y0 != 0}
_Y0S = sorted(_SPECS_BY_Y0)
_Y0S_BY_TIER = {
    "easy": [y for y in _Y0S if _SPECS_BY_Y0[y][0]],
    "medium": [y for y in _Y0S if _SPECS_BY_Y0[y][1]],
}
_Y0S_BY_TIER["hard"] = _Y0S_BY_TIER["exam"] = _Y0S


def _curve(spec) -> dict:
    f = lambda x: float(_value(spec, Fraction(x).limit_denominator(8)))  # noqa: E731
    return {"pieces": figures.sample_curve(f, _LO, _HI, (_LO, _HI), n=_N)}


def _plane(*, curves, alt, points=(), lines=()) -> dict:
    return figures.cartesian(
        x_range=(_LO, _HI),
        y_range=(_LO, _HI),
        alt=alt,
        grid=1,
        equal_scale=True,
        curves=curves,
        points=points,
        lines=lines,
    )


# ----------------------------------------------------------------------- the reader


def _letter_in(text: str) -> str | None:
    m = re.search(r"point marked ([A-Z])\b", text)
    return m.group(1) if m else None


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


def _height_at_axis(curve: dict) -> int:
    """y where the polyline meets x = 0; it must be a single integer point."""
    hits = set()
    for piece in curve["pieces"]:
        for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
            if x0 <= 0 <= x1:
                y = y0 if x1 == x0 else y0 + (y1 - y0) * (0 - x0) / (x1 - x0)
                hits.add(round(y, 6))
    check(len(hits) == 1, "the curve does not meet the y-axis at exactly one point")
    (y,) = hits
    check(abs(y - round(y)) < 1e-9, "the y-axis crossing is not on a grid line")
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
    return str(_height_at_axis(curve))


def _read_cubic(question) -> str:
    return _read_from_figure(question.figure, question.text)


figures.register_reader("pv1f_cubic_y_intercept", _read_cubic)

# ----------------------------------------------------------------------- the prompts

_POOLS = {
    "easy": (
        (
            "The graph of the cubic y = a(x - b)^3 + c is drawn on the grid.",
            "The diagram shows the graph of y = a(x - b)^3 + c, a cubic.",
            "Here is the graph of a cubic y = a(x - b)^3 + c.",
            "A cubic y = a(x - b)^3 + c is graphed on the grid below.",
        ),
        ("Each grid square is 1 unit wide.", "The grid squares are 1 unit by 1 unit.", "Read values from the grid."),
        (
            "State its y-intercept, the value of y where the curve crosses the y-axis.",
            "What is the y-intercept? Give the value of y where the graph meets the y-axis.",
            "Find the y-intercept of the graph, the value of y at which it crosses the y-axis.",
        ),
    ),
    "medium": (
        (
            "The graph of y = k(x - a)(x - b)(x - c) is drawn, showing its three x-intercepts.",
            "A cubic y = k(x - a)(x - b)(x - c) has the graph shown, with three places where it meets the x-axis.",
            "Study this graph of y = k(x - a)(x - b)(x - c), which crosses the x-axis three times.",
            "The diagram shows a cubic of the form y = k(x - a)(x - b)(x - c) cutting the x-axis in three places.",
        ),
        (
            "The grid squares are 1 unit by 1 unit.",
            "Each grid square is 1 unit wide.",
            "Values can be read from the grid.",
        ),
        (
            "Find its y-intercept, where the curve meets the y-axis (not the x-axis).",
            "What value of y does the graph take where it meets the y-axis?",
            "Give the y-intercept of the graph, the value of y where x = 0.",
        ),
    ),
    "hard": (
        (
            "A cubic and a horizontal line L are drawn on the same grid.",
            "The grid shows a cubic curve together with a horizontal line labelled L.",
            "On one grid are a cubic and the horizontal line L.",
            "A horizontal line L is drawn across the graph of a cubic.",
        ),
        (
            "Both the line and the curve cross the y-axis.",
            "The line L and the cubic each meet the y-axis.",
            "Each of the two graphs meets the y-axis.",
        ),
        (
            "State the y-intercept of the cubic, not of L.",
            "What is the y-intercept of the cubic curve? Do not give the value for L.",
            "Find the value of y where the cubic, rather than L, crosses the y-axis.",
        ),
    ),
    "exam": (
        (
            "Two cubic curves are drawn on the same grid.",
            "The graphs of two different cubics share one grid.",
            "One grid shows two cubic graphs.",
            "Two cubics are graphed together on the grid shown.",
        ),
        (
            "The point marked {L} lies on one of them.",
            "One curve passes through the point marked {L}.",
            "Exactly one of the curves goes through the point marked {L}.",
        ),
        (
            "State the y-intercept of the cubic through the point marked {L}.",
            "What is the y-intercept of the curve that passes through the point marked {L}?",
            "Find the value of y where the cubic through the point marked {L} meets the y-axis.",
        ),
    ),
}

_WORKING = "The curve meets the y-axis (x = 0) at y = {y}."


# ----------------------------------------------------------------------- the draw


def _spec_pool(y0: int, tier: str) -> list:
    shift, roots = _SPECS_BY_Y0[y0]
    if tier == "easy":
        return shift
    if tier == "medium":
        return roots
    return shift + roots


def _lures(spec) -> list:
    """Integer wrong answers, most tempting first, as the learner might read the picture."""
    y0 = _y_intercept(spec)
    out = []
    if spec[0] == "shift":
        _f, a, b, c = spec
        out = [c, -y0, c + a * b**3, a * b**3]
    else:
        _f, _k, p, q, r = spec
        out = [p, q, r, -y0, p + q + r]
    out += [_value(spec, 1), _value(spec, -1), _value(spec, 2), _value(spec, -2)]  # squares off the axis
    return [int(v) for v in out if Fraction(v).denominator == 1 and int(v) != y0 and v != 0 and abs(v) <= 9]


def _l_heights(y0: int) -> list:
    return [m for m in range(-9, 10) if m != 0 and abs(m - y0) >= 2]


def _clear_of_each_other_near_axis(s1, s2) -> bool:
    """Two drawn cubics stay at least a unit and a half apart for |x| <= 1, so the two
    y-axis crossings cannot be confused."""
    return all(abs(_value(s1, Fraction(i, 8)) - _value(s2, Fraction(i, 8))) >= Fraction(3, 2) for i in range(-8, 9))


def _exam_points(rng, s1, s2):
    cands = []
    for x in range(-9, 10):
        if abs(x) < 2:  # keep P's label clear of the y-axis crossing being read
            continue
        v = _value(s1, x)
        if v.denominator != 1 or v == 0 or abs(v) > 9:
            continue
        if abs(_value(s2, x) - v) < 2:
            continue
        cands.append((x, int(v)))
    return _pick(rng, cands) if cands else None


def _build(rng, tier: str, y0: int):
    spec = _pick(rng, _spec_pool(y0, tier))
    check(_y_intercept(spec) == y0, "expanded constant term disagrees with the key")
    check(_value(spec, 0) == y0, "the cubic does not take the keyed value at x = 0")
    pool = _lures(spec)
    points, lines = [], []
    curves = [_curve(spec)]
    if tier == "hard":
        m = _pick(rng, _l_heights(y0))
        lines = [{"through": [[_LO, m], [_HI, m]], "label": "L"}]
        pool = [m] + pool
        alt = "a cubic curve and a horizontal line on a coordinate grid"
    elif tier == "exam":
        for _ in range(300):
            spec = _pick(rng, _spec_pool(y0, "hard"))
            y2 = _pick(rng, [v for v in _Y0S if abs(v - y0) >= 2])
            s2 = _pick(rng, _spec_pool(y2, "hard"))
            at = _exam_points(rng, spec, s2) if _clear_of_each_other_near_axis(spec, s2) else None
            if at is not None:
                break
        else:
            check(False, "no exam placement found")
        pool = _lures(spec)  # the lures belong to the curve DRAWN for the key
        check(_value(s2, 0) == y2 and _y_intercept(s2) == y2)
        curves = [_curve(spec), _curve(s2)]
        points = [{"at": list(at), "label": "P"}]
        pool = [y2] + pool
        if rng.randrange(2):  # the drawing order never gives it away
            curves.reverse()
        alt = "two cubic curves on a coordinate grid"
    else:
        alt = "a cubic curve on a coordinate grid"
    fig = _plane(curves=curves, alt=alt, points=points, lines=lines)
    open_, grid_, ask = _POOLS[tier]
    text = " ".join((_pick(rng, open_), _pick(rng, grid_), _pick(rng, ask)))
    text = text.replace("{L}", "P")
    check(_read_from_figure(fig, text) == str(y0), "figure does not show the keyed intercept")
    return fig, text, y0, pool


@generator("pv1f_cubic_y_intercept", level=13, presets=_PRESETS)
def pv1f_cubic_y_intercept(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    y0 = _pick(rng, _Y0S_BY_TIER[tier])  # the key FIRST, uniform
    for _ in range(40):  # redraw the picture, never the key
        fig, text, y0, pool = _build(rng, tier, y0)
        answer_text = str(y0)
        distractors = pick_distractors(answer_text, [str(v) for v in pool], n=3)
        if len(distractors) >= 3:
            break
    check(len(distractors) >= 3, "too few distractors")
    check_pairwise_distinct(y0, *[int(d) for d in distractors])
    figures.assert_no_leak(fig, [answer_text, *distractors])
    return dict(
        text=text,
        answer=y0,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[_WORKING.format(y=y0)],
        figure=fig,
    )
