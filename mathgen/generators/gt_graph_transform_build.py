"""Generates `gt_graph_transform` (easy, medium, hard) and `gt_graph_transform_exam`: state the equation of
``y = f(x)`` after a sequence of translations, dilations and reflections.

Skill ``graph-transformation-equation`` (``functions-sequences``, answer ``expression``). The base graph ``f`` is a
power (``x^2``, ``x^3``), an exponential (``2^x``, ``3^x``, ``e^x``), a trigonometric function (``sin``, ``cos``, ``tan``,
with shifts in multiples of pi) or the reciprocal ``1/x``. A dilation is "by a factor of k from the x-axis" (the
y-values are multiplied) or "from the y-axis" (the x-values are multiplied), the VCE and QCE wording; the
transformations are applied in the order the stem gives them, so the order matters.

* easy: two transformations in words, no graph (one translation at least, the other a translation, reflection or
  dilation from the x-axis). The reply is the typed equation of the new graph.
* medium: three transformations in words, at least one dilation and one translation, in an order that changes the
  answer (translate then dilate gives a different graph from dilate then translate).
* hard: a drawn graph. ``f`` and its image ``g`` are drawn on one grid (trigonometric graphs on an axis marked in
  multiples of pi) with three points of ``g`` marked, and the reply is the typed equation of ``g``. The hidden
  chain is two or three transformations; the reader refits ``g`` from the drawn curve alone.
* exam (``gt_graph_transform_exam``): (a) the equation of the image after three transformations, (b) a feature of
  the image (turning point, point of inflection, asymptote or greatest value, by base), (c) its y-intercept, which
  follows through from the equation in (a) by substitution, and (d) the image of a named point of ``f`` (a
  period for the trigonometric bases).

Every key is built from the parameters of ``y = a F(bx + c) + d``, then checked a second way: points of the base
graph are moved by the transformations geometrically and must lie on the typed equation.
"""

from __future__ import annotations

import functools
import math
import re
from fractions import Fraction as Frac

import sympy

from .. import figures
from ..answers import Distractor
from ..core import VerificationError, attempts, check, generator, latex_answer, phrase
from ._spread_kit import NEAR, pick as spread_pick, uses_rng
from ._gt_kit import (
    FAMILIES,
    TooFewDistractors,
    check_by_points,
    compose,
    equation_text,
    frac_text,
    functions_match,
    move_point,
    normalise,
    ops_sentence,
    same_function,
    shift_of,
    value_at,
    wrong_chains,
)
from ._sr2b_kit import Pair, pair_candidates, pi_latex, pi_shown, pi_text, retry_short

# ------------------------------------------------------------------------------------------------------- drawing

_TX = [Frac(v) for v in (1, 2, 3, 4)]
_TX_TRIG = [Frac(1, 2), Frac(1), Frac(1, 3), Frac(1, 4), Frac(1, 6)]
_TY = [Frac(v) for v in (1, 2, 3, 4)]
_DY = [Frac(2), Frac(3), Frac(4), Frac(1, 2)]
_DX = [Frac(2), Frac(3), Frac(1, 2)]

_EASY_BASES = ("x^2", "x^3", "2^x", "3^x", "e^x", "sin(x)", "cos(x)", "tan(x)", "1/x")
_DRAWN_BASES = ("x^2", "x^3", "2^x", "3^x", "sin(x)", "cos(x)", "1/x")
_EXAM_BASES = ("x^2", "x^3", "2^x", "3^x", "sin(x)", "cos(x)", "1/x")

# The shifts and amounts a trigonometric graph can show on an axis marked in multiples of pi/2.
_DRAWN_TX_TRIG = [Frac(1, 2), Frac(1)]


def _make_op(rng, kind: str, fam, drawn: bool = False):
    sign = rng.choice((-1, 1))
    if kind == "tx":
        pool = (_DRAWN_TX_TRIG if drawn else _TX_TRIG) if fam.trig else _TX[:3] if drawn else _TX
        return ("tx", sign * rng.choice(pool))
    if kind == "ty":
        return ("ty", sign * rng.choice(_TY[:3] if drawn else _TY))
    if kind == "dy":
        return ("dy", rng.choice((Frac(2), Frac(3), Frac(1, 2)) if drawn else _DY))
    if kind == "dx":
        return ("dx", rng.choice((Frac(2), Frac(1, 2)) if drawn else _DX))
    return (kind,)


def _shift_ok(fam, p) -> bool:
    h = shift_of(p) if p[2] != 0 else Frac(0)
    if fam.trig:
        return h.denominator in (1, 2, 3, 4, 6) and abs(h) <= 2
    return h.denominator == 1 and abs(h) <= 6


def _acceptable(fam, ops, tier: str) -> bool:
    kinds = [o[0] for o in ops]
    if fam.even and "ry" in kinds:
        return False
    p = compose(ops)
    a = normalise(fam, p)[0]
    if p[3].denominator != 1 or abs(p[3]) > 6:
        return False
    if a.denominator not in (1, 2) or abs(a.numerator) > 6 or abs(a) > 4:
        return False
    if not _shift_ok(fam, p):
        return False
    if abs(p[1]) not in (Frac(1), Frac(2), Frac(3), Frac(1, 2), Frac(1, 3), Frac(1, 4), Frac(2, 3), Frac(3, 2)):
        return False
    key = equation_text(fam, p)
    if same_function(key, fam.key):
        return False
    if tier == "medium":
        # the order must matter: some adjacent swap gives a different graph
        for i in range(len(ops) - 1):
            swapped = ops[:i] + [ops[i + 1], ops[i]] + ops[i + 2 :]
            q = compose(swapped)
            try:
                other = equation_text(fam, q, strict=False)
            except VerificationError:
                continue
            if not same_function(key, other):
                return True
        return False
    return True


def _draw_chain(rng, fam, tier: str):
    drawn = tier in ("hard", "exam")
    for _ in attempts(3000, "a chain of transformations that gives a clean equation"):
        if tier == "easy":
            kinds = rng.sample(["tx", "ty", "rx", "ry", "dy"], 2)
            if not ({"tx", "ty"} & set(kinds)):
                continue
        elif tier == "medium":
            kinds = rng.sample(["tx", "ty", "rx", "ry", "dy", "dx"], 3)
            if not ({"dx", "dy"} & set(kinds)) or not ({"tx", "ty"} & set(kinds)):
                continue
        elif tier == "hard":
            kinds = rng.sample(["tx", "ty", "rx", "ry", "dy", "dx"], rng.choice((2, 3, 3)))
        else:
            kinds = rng.sample(["tx", "ty", "rx", "ry", "dy", "dx"], 3)
            if not ({"tx", "ty"} & set(kinds)) or not ({"dx", "dy"} & set(kinds)):
                continue
            if fam.trig and "dx" not in kinds:
                continue
        ops = [_make_op(rng, k, fam, drawn) for k in kinds]
        if _acceptable(fam, ops, tier):
            return ops
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------- distractors


def _wrong_equations(fam, ops, key: str) -> list[Distractor]:
    """Wrong equations from the common slips, one per slip first, none equal to the key or each other."""
    seen = [key]
    first, rest = [], []
    used = set()
    chains = wrong_chains(ops)
    for tag, chain in chains:
        try:
            q = compose(chain)
            if not _shift_ok(fam, q):
                continue
            text = equation_text(fam, q)
        except VerificationError:
            continue
        if any(same_function(text, s) for s in seen):
            continue
        seen.append(text)
        (rest if tag in used else first).append(Distractor(text, tag, text))
        used.add(tag)
    if len(first) + len(rest) < 3:
        # too few single slips: a second common slip made on top of the first is still a plausible wrong equation
        for _, chain in chains:
            for tag, chain2 in wrong_chains(chain):
                try:
                    q = compose(chain2)
                    if not _shift_ok(fam, q):
                        continue
                    text = equation_text(fam, q)
                except VerificationError:
                    continue
                if any(same_function(text, s) for s in seen):
                    continue
                seen.append(text)
                rest.append(Distractor(text, tag, text))
    return first + rest


def _near_pairs(rng, ans: Pair) -> list[tuple[str, Frac, Frac]]:
    """Plain near misses of a coordinate pair, in a random order: one coordinate a unit or two out, the other right."""
    x, y = ans.values
    out = [(NEAR, x + dx, y + dy) for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0), (0, 2), (2, 0), (0, -2), (-2, 0))]
    rng.shuffle(out)
    return out


def _need(options: list, minimum: int = 2):
    if len(options) < minimum:
        raise TooFewDistractors("too few wrong equations")
    return options[:3]


def _steps(fam, ops) -> list[str]:
    out = [f"Start from y = {fam.key}."]
    done: list = []
    for op in ops:
        done.append(op)
        from ._gt_kit import op_phrase

        eq = equation_text(fam, compose(done), strict=False)
        out.append(f"After the graph is {op_phrase(op, fam.trig)}: y = {eq}.")
    return out


def _shown(line: str) -> str:
    """A working line as a learner reads it: no asterisk for times (3*cos(x) is 3cos(x), 3*2^x is 3 × 2^x)."""
    line = re.sub(r"(?<=[\d)])\*(?=[a-z(])", "", line)
    return re.sub(r"(?<=\d)\*(?=\d)", " × ", line)


def _tidy(result: dict) -> dict:
    """The question with every working line shown without Python's asterisk."""
    if "working" in result:
        result["working"] = [_shown(w) for w in result["working"]]
    for part in result.get("parts", ()):
        part["working"] = [_shown(w) for w in part["working"]]
    return result


def _latex(text: str) -> str:
    return latex_answer(text, evaluate=False)


_STEM_TAIL = (
    "State the equation of the resulting graph.",
    "Find the equation of the new graph.",
    "What is the equation of the image?",
)


# -------------------------------------------------------------------------------------------------- easy, medium


def _worded(rng, tier: str):
    fam = FAMILIES[rng.choice(_EASY_BASES)]
    ops = _draw_chain(rng, fam, tier)
    key = equation_text(fam, compose(ops))
    check_by_points(fam, ops, key)
    check(functions_match(fam, compose(ops), key, [0.31 * i - 1.7 for i in range(12)]), "key differs from parameters")
    options = _need(_wrong_equations(fam, ops, key))
    stem = phrase(
        rng,
        (
            "The graph of y = {f} is {ops}. {tail}",
            "A graph has equation y = {f}. It is {ops}. {tail}",
            "The curve y = {f} is {ops}, in that order. {tail}",
        ),
        f=fam.key,
        ops=ops_sentence(ops, fam.trig),
        tail=rng.choice(_STEM_TAIL),
    )
    return dict(
        text=stem,
        answer_kind="expression",
        answer=key,
        answer_text=key,
        answer_latex=_latex(key),
        working=_steps(fam, ops) + [f"The equation is y = {key}."],
        distractors=options,
    )


# ------------------------------------------------------------------------------------------------------- drawn

_LO, _HI = -8, 8


def _exact_value(fam, p, x: Frac):
    """a F(b x + c) + d exactly at a rational x, or None where it is not a rational number."""
    a, b, c, d = p
    arg = b * x + c
    key = fam.key
    try:
        if fam.kind == "poly":
            f = arg ** int(key[-1])
        elif fam.kind == "exp":
            if arg.denominator != 1 or abs(arg) > 12:
                return None
            f = Frac(int(key[0])) ** int(arg)
        elif fam.kind == "recip":
            if arg == 0:
                return None
            f = 1 / arg
        else:
            val = (sympy.sin if key.startswith("sin") else sympy.cos)(
                sympy.pi * sympy.Rational(arg.numerator, arg.denominator)
            )
            if not val.is_Rational:
                return None
            f = Frac(int(val.p), int(val.q))
    except (ZeroDivisionError, OverflowError):
        return None
    return a * f + d


def _lattice(fam, p):
    """Points of g with a rational x on the marked grid and a whole-number y inside the window."""
    out = []
    if fam.trig:
        xs = [Frac(k, 2) for k in range(-4, 5)]
        ylim = 5
    else:
        xs = [Frac(k) for k in range(_LO + 1, _HI)]
        ylim = 7
    for x in xs:
        y = _exact_value(fam, p, x)
        if y is not None and y.denominator == 1 and abs(y) <= ylim:
            out.append((x, int(y)))
    return out


def _marked(points):
    """Three marked points spread along g: as many different heights as there are, then the widest spread of x."""
    if len(points) < 3:
        return None
    best = None
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                trio = [points[i], points[j], points[k]]
                score = (len({y for _, y in trio}), float(trio[2][0] - trio[0][0]))
                if best is None or score > best[0]:
                    best = (score, trio)
    return best[1]


def _point_label(fam, pt) -> str:
    x, y = pt
    xs = pi_text(x).replace("pi", "π") if fam.trig else frac_text(x)
    return f"({xs}, {y})"


def _sample(fam, p, lo, hi, ylo, yhi, steps):
    pieces = figures.sample_curve(lambda u: value_at(p, fam, u), lo, hi, (ylo, yhi), n=(hi - lo) * steps + 1)
    return pieces


def _plane(fam, ops, tier_key: str):
    p = compose(ops)
    trig = fam.trig
    lo, hi, ylo, yhi = (-2, 2, -6, 6) if trig else (_LO, _HI, _LO, _HI)
    steps = 48 if trig else 12
    f_pieces = _sample(fam, (Frac(1), Frac(1), Frac(0), Frac(0)), lo, hi, ylo, yhi, steps)
    g_pieces = _sample(fam, p, lo, hi, ylo, yhi, steps)
    npts = sum(len(piece) for piece in g_pieces)
    if npts < 0.4 * (hi - lo) * steps:
        return None
    points = _marked(_lattice(fam, p))
    if points is None:
        return None
    marks = [{"at": [float(x), y], "label": _point_label(fam, (x, y))} for x, y in points]
    curves = [{"pieces": f_pieces, "label": "f"}, {"pieces": g_pieces, "label": "g"}]
    if trig:
        fig = figures.cartesian(
            x_range=(lo, hi),
            y_range=(ylo, yhi),
            alt="the graphs of f and g, with three points of g marked, on a grid whose x-axis is marked in multiples of pi",
            x_grid=0.5,
            y_grid=1,
            equal_scale=False,
            x_tick_format="pi",
            points=marks,
            curves=curves,
        )
    else:
        fig = figures.cartesian(
            x_range=(lo, hi),
            y_range=(ylo, yhi),
            alt="the graphs of f and g, with three points of g marked, on a coordinate grid, one unit per grid square",
            grid=1,
            equal_scale=True,
            points=marks,
            curves=curves,
        )
    return fig


def _hard(rng):
    for _ in attempts(400, "a drawn image with three marked points"):
        fam = FAMILIES[rng.choice(_DRAWN_BASES)]
        ops = _draw_chain(rng, fam, "hard")
        p = compose(ops)
        if abs(p[1]) not in (Frac(1), Frac(2), Frac(1, 2)):
            continue
        fig = _plane(fam, ops, fam.key)
        if fig is not None:
            break
    key = equation_text(fam, p)
    check_by_points(fam, ops, key)
    options = _need(_wrong_equations(fam, ops, key))
    # the marked points lie on the key (a third route, from the figure's own labels)
    for pt in fig["points"]:
        x, y = pt["at"]
        u = math.pi * x if fam.trig else x
        from ._gt_kit import typed_function

        check(abs(typed_function(key)(u) - y) < 1e-7, "a marked point is not on the key")
    figures.assert_no_leak(fig, [key, *[d.text for d in options]])
    stem = phrase(
        rng,
        (
            "The graph of y = f(x), where f(x) = {f}, is drawn with its image g under a sequence of transformations. "
            "Three points of g are marked. Find the equation of g, in the form y = ...",
            "The graph of f(x) = {f} and a transformed graph g are shown. Three points on g are marked. "
            "State the equation of g in the form y = ...",
        ),
        f=fam.key,
    )
    return dict(
        text=stem,
        answer_kind="expression",
        answer=key,
        answer_text=key,
        answer_latex=_latex(key),
        working=[
            f"Compare g with f(x) = {fam.key} using the marked points and the shape of the graph.",
            f"The image is {ops_sentence(ops, fam.trig)}, which gives y = {key}.",
            "Check: each marked point satisfies the equation.",
        ],
        distractors=options,
        figure=fig,
    )


# the reader: refits g from the drawn curve alone ----------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _candidates(trig: bool):
    A = [Frac(n, d) for n in (1, 2, 3, 4, 6) for d in (1, 2)] + [Frac(1, 3)]
    A = sorted({s * v for v in A for s in (1, -1)})
    B = [Frac(v) for v in (1, 2, 3)] + [Frac(1, 2), Frac(1, 3)]
    B = sorted({s * v for v in B for s in (1, -1)})
    H = [Frac(k, 12) for k in range(-24, 25)] if trig else [Frac(k, 2) for k in range(-16, 17)]
    D = [Frac(k) for k in range(-8, 9)]
    return A, B, H, D


def _read(question) -> str:
    fig = question.figure
    base = re.search(r"f\(x\) = ([^\s,]+)", question.text).group(1)
    fam = FAMILIES[base]
    g = next(c for c in fig["curves"] if c.get("label") == "g")
    pts = [(x, y) for piece in g["pieces"] for x, y in piece]
    probe = [pts[int(i * (len(pts) - 1) / 6)] for i in range(7)]
    A, B, H, D = _candidates(fam.trig)
    for b in B:
        for h in H:
            c = -b * h
            for a in A:
                for d in D:
                    p = (a, b, c, d)
                    ok = True
                    for u, y in probe:
                        try:
                            if abs(value_at(p, fam, u) - y) > 1e-4 * (1 + abs(y)):
                                ok = False
                                break
                        except (ZeroDivisionError, OverflowError, ValueError):
                            ok = False
                            break
                    if ok and all(abs(value_at(p, fam, u) - y) <= 1e-4 * (1 + abs(y)) for u, y in pts):
                        return equation_text(fam, p, strict=False)
    raise AssertionError("no transformation of the base fits the drawn graph")


figures.register_reader("gt_graph_transform", _read)


@generator(
    "gt_graph_transform",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
@retry_short
def gt_graph_transform(rng, tier="easy"):
    """The equation of a transformed graph; the tiers are described in the module docstring."""
    if tier in ("easy", "medium"):
        return _tidy(_worded(rng, tier))
    if tier != "hard":
        raise ValueError(f"unknown tier {tier!r}")
    return _tidy(_hard(rng))


# -------------------------------------------------------------------------------------------------------- exam

_POINTS = {
    "x^2": [(-2, 4), (-1, 1), (1, 1), (2, 4), (3, 9)],
    "x^3": [(-2, -8), (-1, -1), (1, 1), (2, 8)],
    "2^x": [(0, 1), (1, 2), (2, 4), (3, 8)],
    "3^x": [(0, 1), (1, 3), (2, 9)],
    "1/x": [(-4, Frac(-1, 4)), (-2, Frac(-1, 2)), (-1, -1), (1, 1), (2, Frac(1, 2)), (4, Frac(1, 4))],
}


def _y_intercept(fam, p):
    """g(0) as an exact rational, or None (trig: where sin or cos of the shifted argument is not rational)."""
    return _exact_value(fam, p, Frac(0))


def _feature(fam, p):
    """(prompt, kind, value(s), label) for the feature asked in (b): a pair for a point, else a number."""
    a, b, c, d = p
    h = shift_of(p) if c != 0 else Frac(0)
    if fam.key == "x^2":
        return "pair", (h, d)
    if fam.key == "x^3":
        return "pair", (h, d)
    if fam.kind == "exp":
        return "number", d
    if fam.kind == "recip":
        return "number", h
    return "number", abs(a) + d  # sin, cos: the greatest value


def _feature_prompt(fam) -> str:
    if fam.key == "x^2":
        return "Find the coordinates of the turning point of the graph of g."
    if fam.key == "x^3":
        return "Find the coordinates of the point of inflection of the graph of g."
    if fam.kind == "exp":
        return "The graph of g has a horizontal asymptote with equation y = k. Find k."
    if fam.kind == "recip":
        return "The graph of g has a vertical asymptote with equation x = k. Find k."
    return "Find the greatest value of g."


def _feature_working(fam, ops, p) -> str:
    h = shift_of(p) if p[2] != 0 else Frac(0)
    if fam.key in ("x^2", "x^3"):
        where = "turning point" if fam.key == "x^2" else "point of inflection"
        return f"The {where} of y = {fam.key} is (0, 0). The transformations move it to ({frac_text(h)}, {frac_text(p[3])})."
    if fam.kind == "exp":
        return (
            f"y = {fam.key} has the horizontal asymptote y = 0. The transformations move it to y = {frac_text(p[3])}."
        )
    if fam.kind == "recip":
        return f"y = 1/x has the vertical asymptote x = 0. The transformations move it to x = {frac_text(h)}."
    shift = p[3]
    total = f"{frac_text(abs(p[0]))} {'+' if shift >= 0 else '-'} {frac_text(abs(shift))}"
    return (
        f"The greatest value of {fam.key} is 1, so the greatest value of g is {total} = {frac_text(abs(p[0]) + shift)}."
    )


def _exam_parts(rng, fam, ops, p, key, y_int):
    # (a) the equation
    a_opts = _need(_wrong_equations(fam, ops, key))
    part_a = dict(
        prompt="Find the equation of g.",
        answer_kind="expression",
        answer=key,
        answer_text=key,
        answer_latex=_latex(key),
        marks=2,
        working=_steps(fam, ops) + [f"So g(x) = {key}."],
        distractors=a_opts,
    )
    # (b) a feature of g
    kind, value = _feature(fam, p)
    wrongs = []
    for tag, chain in wrong_chains(ops):
        try:
            q = compose(chain)
            if not _shift_ok(fam, q):
                continue
            equation_text(fam, q)
        except VerificationError:
            continue
        wrongs.append((tag, _feature(fam, q)[1]))
    if kind == "pair":
        ans = Pair("x", value[0], "y", value[1])
        cands = pair_candidates(ans, [(t, v[0], v[1]) for t, v in wrongs] + _near_pairs(rng, ans))
        fields = ans.fields()
    else:
        from ._sr2b_kit import number_fields
        from ..answers import exact_value

        fields = number_fields(Frac(value), 0)
        cands = []
        for tag, v in wrongs:
            if Frac(v).denominator == 1 and abs(Frac(v) - Frac(value)) >= Frac(1, 20):
                cands.append(Distractor(str(int(v)), tag, int(v)))
    seen: set = set()
    b_opts = []
    for c in cands:
        if c.text not in seen and c.text != fields["answer_text"]:
            seen.add(c.text)
            b_opts.append(c)
    b_opts = (
        _need(b_opts)
        if kind == "pair"
        else spread_pick(
            fields["answer_text"],
            b_opts,
            lo=None,
            hi=None,
            unbounded="an asymptote or intercept of g can sit at any integer",
        )
    )
    part_b = dict(
        prompt=_feature_prompt(fam),
        marks=2 if kind == "pair" else 1,
        working=[_feature_working(fam, ops, p)],
        distractors=b_opts,
        **fields,
    )
    if kind == "number":
        part_b["sensible_range"] = (-100, 100)
    # (c) the y-intercept: follows through from the equation by substituting x = 0
    from ._sr2b_kit import rat_dist
    from ..core import rational_text

    y0 = Frac(y_int)
    c_wrong = []
    for tag, chain in wrong_chains(ops):
        try:
            q = compose(chain)
            if not _shift_ok(fam, q):
                continue
            v = _exact_value(fam, q, Frac(0))
        except VerificationError:
            continue
        if v is not None and v != y0 and v.denominator == 1:
            c_wrong.append(Distractor(rational_text(v), tag, _frac_value(v)))
    seen = set()
    c_opts = []
    for c in c_wrong:
        if c.text not in seen and c.text != rational_text(y0):
            seen.add(c.text)
            c_opts.append(c)
    if y0.denominator != 1:  # a fraction: its neighbours are a whole step of its denominator away
        for k in rng.sample((-2, -1, 1, 2), 4):
            v = y0 + Frac(k, y0.denominator)
            if v != 0 and rational_text(v) not in seen:
                c_opts.append(Distractor(rational_text(v), NEAR, _frac_value(v)))
    c_opts = spread_pick(
        rational_text(y0), c_opts, lo=None, hi=None, unbounded="the y-intercept of g can be any rational number"
    )
    part_c = dict(
        prompt="Find the y-intercept of the graph of g.",
        answer_kind="fraction",
        answer=_frac_value(y0),
        answer_text=rational_text(y0),
        answer_latex=_frac_latex(y0),
        marks=1,
        working=[f"Put x = 0 in g(x) = {key}: y = {rational_text(y0)}."],
        distractors=c_opts,
        follow_through="Subs(part_a, (x,), (0,))",
    )
    # (d) the image of a point of f, or the period of g for a trigonometric base
    if fam.trig:
        period = 2 / abs(p[1])
        d_ans = pi_text(period)
        d_wrong = []
        for tag, chain in wrong_chains(ops):
            q = compose(chain)
            if _shift_ok(fam, q) and 2 / abs(q[1]) != period:
                d_wrong.append(Distractor(pi_text(2 / abs(q[1])), tag, pi_text(2 / abs(q[1]))))
        d_wrong.append(Distractor(pi_text(Frac(2)), "gt-period-unchanged", pi_text(Frac(2))))
        # plain near misses: the period out by a half, a third or a whole more
        for f in rng.sample((Frac(1, 2), Frac(3, 2), Frac(1, 3), Frac(3), Frac(2), Frac(1, 4)), 6):
            d_wrong.append(Distractor(pi_text(period * f), NEAR, pi_text(period * f)))
        seen, d_opts = set(), []
        for c in d_wrong:
            if c.text not in seen and c.text != d_ans:
                seen.add(c.text)
                d_opts.append(c)
        d_opts = _need(d_opts)
        part_d = dict(
            prompt="Find the period of g. Give an exact value in radians.",
            answer_kind="expression",
            answer=d_ans,
            answer_text=d_ans,
            answer_latex=pi_latex(period),
            marks=1,
            working=[
                f"{fam.key} has period 2π. The dilation by a factor of {frac_text(1 / abs(p[1]))} from the y-axis makes the period {pi_shown(period)}."
            ],
            distractors=d_opts,
        )
        return [part_a, part_b, part_c, part_d], None
    pt = rng.choice(_POINTS[fam.key])
    px, py = Frac(pt[0]), Frac(pt[1])
    image = move_point((px, py), ops)
    ans = Pair("x", image[0], "y", image[1])
    wrongs = []
    for tag, chain in wrong_chains(ops):
        q = move_point((px, py), chain)
        wrongs.append((tag, q[0], q[1]))
    d_opts = pair_candidates(ans, wrongs + _near_pairs(rng, ans))
    seen, uniq = set(), []
    for c in d_opts:
        if c.text not in seen and c.text != ans.text:
            seen.add(c.text)
            uniq.append(c)
    part_d = dict(
        prompt=f"The point P({frac_text(px)}, {frac_text(py)}) lies on the graph of f. Find the coordinates of the point on g that P is moved to.",
        marks=2,
        working=[
            f"Apply the transformations to P in order: it moves to ({frac_text(image[0])}, {frac_text(image[1])})."
        ],
        distractors=_need(uniq),
        **ans.fields(),
    )
    return [part_a, part_b, part_c, part_d], pt


def _frac_value(v):
    from ..answers import exact_value

    return exact_value(Frac(v))


def _frac_latex(v):
    from ._sr2b_kit import value_latex

    return value_latex(Frac(v))


@generator("gt_graph_transform_exam", level=14, presets={"exam": dict(tier="exam")})
@uses_rng
@retry_short
def gt_graph_transform_exam(rng, tier="exam"):
    """Exam chain: the equation of the image, a feature of it, its y-intercept (follows through), a point's image."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "an exam chain with a rational y-intercept"):
        fam = FAMILIES[rng.choice(_EXAM_BASES)]
        ops = _draw_chain(rng, fam, "exam")
        p = compose(ops)
        y_int = _y_intercept(fam, p)
        if y_int is None or abs(y_int) > 200:
            continue
        kind, value = _feature(fam, p)
        if kind == "number" and Frac(value).denominator != 1:
            continue
        key = equation_text(fam, p)
        break
    check_by_points(fam, ops, key)
    parts, _pt = _exam_parts(rng, fam, ops, p, key, y_int)
    stem = phrase(
        rng,
        (
            "The graph of y = f(x), where f(x) = {f}, is transformed to the graph of y = g(x) by these transformations, in order: {ops}.",
            "Let f(x) = {f}. The graph of y = f(x) is {ops}, in that order, to give the graph of y = g(x).",
        ),
        f=fam.key,
        ops=ops_sentence(ops, fam.trig),
    )
    return _tidy(dict(text=stem, parts=parts))
