"""Generates `r3h_methods_optimisation` questions (easy, medium, hard) and the exam chain `exd_x_r3h_methods_optimisation`.

``methods-optimisation-context`` (``differentiation``, answers ``expression`` and ``number``): turn a described
situation into a function of one variable, state the allowed values of the variable, find the stationary point and say
why it is the greatest or least value. This is the school-level skill; the first-year skill ``ucv1-applied-optimisation``
uses different situations (fenced paddocks, folded square sheets, tanks, cables), and so do these:

* a closed cylindrical tin of fixed volume (least surface area),
* an open box with a square base and fixed volume (least material),
* a rectangle inscribed under the parabola y = k - x^2 (greatest area),
* an open box folded from a rectangular sheet (greatest volume; the larger stationary point is outside the allowed values),
* a business whose price falls as output rises and whose cost has a cubic term (greatest profit),
* the point of the parabola y = x^2 nearest a point on the y-axis (least distance).

Tiers: easy asks for the function (expression); medium for the value of the variable at the optimum (number, 2 decimal
places); hard for the optimum value itself (number) in the situations where the allowed values or a second stationary
point matter. The exam chain (``exd_x_r3h_methods_optimisation``) is an open box from a rectangular sheet or a profit
function: (a) the function, (b) the optimising value, (c) the second derivative there (follows through from (b)) and
(d) the optimum value (follows through from (b)).

Every optimum is checked by a fine grid search over the allowed values, and every stationary point by substituting it
into the derivative.
"""

from __future__ import annotations

import functools
import math

import numpy
import sympy

from ..core import attempts, check, generator, join_terms, rounds_unambiguously
from ._r3h_kit import (
    Redraw,
    build_with_redraws,
    expr_options,
    expr_part,
    machine,
    number_options,
    number_part,
    plain,
    say_marked,
)

_R = sympy.Rational
_INF = float("inf")

#: Whether `solve` also checks the optimum with a brute-force grid search. Off in the generator (it is the slow part);
#: the tests turn it on and also search the printed stem independently.
SECOND_ROUTE = False


def _real_roots(poly):
    """The real roots of a polynomial with rational coefficients (NumPy, polished by Newton steps)."""
    coeffs = [float(c) for c in poly.all_coeffs()]
    deriv = [c * (len(coeffs) - 1 - i) for i, c in enumerate(coeffs[:-1])]
    out = []
    for r in numpy.roots(coeffs):
        if abs(r.imag) > 1e-9 * max(1.0, abs(r)):
            continue
        v = float(r.real)
        for _ in range(3):
            d = numpy.polyval(deriv, v) if deriv else 0.0
            if d == 0:
                break
            v -= numpy.polyval(coeffs, v) / d
        out.append(complex(v))
    return out


@functools.lru_cache(maxsize=4096)
def _zeros(f, var):
    """Positive-or-not real zeros of the numerator of `f` (cached)."""
    try:
        return tuple(r.real for r in _real_roots(sympy.Poly(sympy.expand(sympy.numer(sympy.together(f))), var)))
    except (sympy.PolynomialError, ValueError, TypeError):
        return ()


@functools.lru_cache(maxsize=4096)
def all_stationary(g, var):
    """Every real stationary point of `g`, as floats, from the numerator of g' (cached: this is the slow step)."""
    d = sympy.together(sympy.diff(g, var))
    num, _den = sympy.fraction(d)
    poly = sympy.Poly(sympy.expand(num), var)
    if poly.degree() < 1:
        return ()
    return tuple(sorted(r.real for r in _real_roots(poly)))


def _say(expr) -> str:
    """`expr` as text for a worked line: products written with a multiplication sign, never ``*``."""
    return plain(expr).replace("*", " × ")


def stationary(g, var, lo, hi):
    """Real stationary points of `g` strictly between lo and hi."""
    return [r for r in all_stationary(g, var) if lo < r < hi]


@functools.lru_cache(maxsize=4096)
def _evaluator(g, var):
    return sympy.lambdify(var, g, "math")


@functools.lru_cache(maxsize=4096)
def _second(g, var):
    return sympy.diff(g, var, 2)


def _value(g, var, x):
    try:
        v = _evaluator(g, var)(x)
        v = complex(v)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return v.real if abs(v.imag) < 1e-9 and math.isfinite(v.real) else None


def _grid_best(g, var, lo, hi, maximise, n=4000):
    """The best value of `g` on a fine grid over (lo, hi): the independent second route."""
    fn = _evaluator(g, var)
    best = None
    for i in range(1, n):
        v = fn(lo + (hi - lo) * i / n)
        if best is None or (v > best if maximise else v < best):
            best = v
    return best


# ---------------------------------------------------------------------------------------------------------------
# The situations


class _Ctx:
    """One situation, drawn: its function, allowed values, kind and the slips a learner makes in setting it up."""

    def __init__(self, key, var, f, lo, hi, maximise, wrongs, easy, medium, hard, unit_x, unit_v, what_x, what_v):
        self.key, self.var, self.f, self.lo, self.hi, self.maximise = key, var, f, lo, hi, maximise
        self.wrongs, self.easy, self.medium, self.hard = wrongs, easy, medium, hard
        self.unit_x, self.unit_v, self.what_x, self.what_v = unit_x, unit_v, what_x, what_v
        self.optimum_x = None
        # the quantity reported at the optimum: the function's value, or its square root for a distance
        self.report = (
            (lambda v: math.sqrt(v) if v is not None and v >= 0 else None) if key == "nearest" else (lambda v: v)
        )

    def solve(self):
        points = stationary(self.f, self.var, self.lo, self.hi)
        check(
            len(points) == 1, f"{self.key}: expected exactly one stationary point in the allowed values, got {points}"
        )
        self.optimum_x = points[0]
        self.optimum_v = _value(self.f, self.var, self.optimum_x)
        second = _value(_second(self.f, self.var), self.var, self.optimum_x)
        check(
            (second < 0) == self.maximise,
            f"{self.key}: the stationary point is not the {'max' if self.maximise else 'min'}",
        )
        if SECOND_ROUTE:
            if math.isfinite(self.hi):
                best = _grid_best(self.f, self.var, self.lo, self.hi, self.maximise)
            else:
                best = _grid_best(self.f, self.var, max(self.lo, 1e-3), self.optimum_x * 6, self.maximise)
            check(abs(best - self.optimum_v) < 1e-3 * max(1, abs(self.optimum_v)), f"{self.key}: grid search disagrees")
        self.second = second
        return self


def _ctx_tin(rng):
    r = sympy.Symbol("r", positive=True)
    V = rng.choice([330, 375, 400, 500, 600, 750, 1000, 1500])
    f = 2 * sympy.pi * r**2 + 2 * V / r
    wrongs = [
        ("r3h-opt-dropped-factor-in-surface", 2 * sympy.pi * r**2 + V / r),
        ("r3h-opt-one-end-only", sympy.pi * r**2 + 2 * V / r),
        ("r3h-opt-height-substitution-slip", 2 * sympy.pi * r**2 + 2 * V / (sympy.pi * r)),
    ]
    return _Ctx(
        "tin",
        r,
        f,
        0,
        _INF,
        False,
        wrongs,
        easy=f"A closed cylindrical tin has a volume of {V} cm³ and a base of radius «r» cm. Write the total surface area «S» cm² of the tin as a function of «r» alone.",
        medium=f"Closed cylindrical tins with a volume of {V} cm³ are made from sheet metal. Find the radius, in centimetres correct to 2 decimal places, that uses the least metal.",
        hard=f"Closed cylindrical tins with a volume of {V} cm³ are made from sheet metal. Find the least possible total surface area of a tin, in square centimetres correct to 1 decimal place.",
        unit_x="cm",
        unit_v="cm²",
        what_x="radius",
        what_v="surface area",
    )


def _ctx_box(rng):
    x = sympy.Symbol("x", positive=True)
    V = rng.choice([500, 800, 1000, 1500, 2000, 2400, 3000])
    f = x**2 + 4 * V / x
    wrongs = [
        ("r3h-opt-dropped-factor-in-surface", x**2 + 2 * V / x),
        ("r3h-opt-extra-face", 2 * x**2 + 4 * V / x),
        ("r3h-opt-height-substitution-slip", x**2 + 4 * V / x**2),
    ]
    return _Ctx(
        "box",
        x,
        f,
        0,
        _INF,
        False,
        wrongs,
        easy=f"An open-topped box has a square base of side «x» cm and a volume of {V} cm³. Write the area «A» cm² of material needed for the base and four sides as a function of «x» alone.",
        medium=f"An open-topped box with a square base is to hold {V} cm³. Find the side length of the base, in centimetres correct to 2 decimal places, that uses the least material.",
        hard=f"An open-topped box with a square base is to hold {V} cm³. Find the least area of material needed, in square centimetres correct to 1 decimal place.",
        unit_x="cm",
        unit_v="cm²",
        what_x="side length",
        what_v="area of material",
    )


def _ctx_rectangle(rng):
    x = sympy.Symbol("x", positive=True)
    k = rng.choice([10, 12, 15, 18, 20, 24, 30, 40])
    f = 2 * x * (k - x**2)
    wrongs = [
        ("r3h-opt-wrong-height", 2 * x * (k - x)),
        ("r3h-opt-half-width-area", x * (k - x**2)),
        ("r3h-opt-curve-not-substituted", 2 * x * (k - x**3)),
    ]
    return _Ctx(
        "rectangle",
        x,
        f,
        0,
        math.sqrt(k),
        True,
        wrongs,
        easy=f"A rectangle has its base on the x-axis and its two upper vertices on the parabola «y = {k} - x^2», symmetrically about the y-axis, with the right-hand vertex at «x». Write the area «A» of the rectangle as a function of «x».",
        medium=f"A rectangle has its base on the x-axis and its two upper vertices on the parabola «y = {k} - x^2», symmetrically about the y-axis. Find the x-coordinate of the right-hand vertex, correct to 2 decimal places, that gives the rectangle of greatest area.",
        hard=f"A rectangle has its base on the x-axis and its two upper vertices on the parabola «y = {k} - x^2», symmetrically about the y-axis. Find the greatest possible area of the rectangle, correct to 2 decimal places.",
        unit_x="",
        unit_v="square units",
        what_x="x-coordinate",
        what_v="area",
    )


def _ctx_sheet(rng):
    x = sympy.Symbol("x", positive=True)
    a, b = rng.choice([(18, 30), (20, 32), (24, 36), (25, 40), (30, 48), (40, 60), (30, 50)])
    f = x * (a - 2 * x) * (b - 2 * x)
    wrongs = [
        ("r3h-opt-cut-one-end", x * (a - x) * (b - x)),
        ("r3h-opt-forgot-height", (a - 2 * x) * (b - 2 * x)),
        ("r3h-opt-fold-allowance-slip", x * (a - 2 * x) * (b - x)),
    ]
    return _Ctx(
        "sheet",
        x,
        f,
        0,
        a / 2,
        True,
        wrongs,
        easy=f"A rectangular sheet of cardboard {a} cm by {b} cm has a square of side «x» cm cut from each corner. The sides are folded up to make an open box. Write the volume «V» cm³ of the box as a function of «x».",
        medium=f"A rectangular sheet of cardboard {a} cm by {b} cm has a square of side «x» cm cut from each corner, and the sides are folded up to make an open box. Find the value of «x», in centimetres correct to 2 decimal places, that gives the box of greatest volume.",
        hard=f"A rectangular sheet of cardboard {a} cm by {b} cm has a square of side «x» cm cut from each corner, and the sides are folded up to make an open box. Find the greatest possible volume of the box, in cubic centimetres correct to 1 decimal place.",
        unit_x="cm",
        unit_v="cm³",
        what_x="side of the cut squares",
        what_v="volume",
    )


def _ctx_profit(rng):
    q = sympy.Symbol("q", positive=True)
    a = rng.choice([100, 120, 150, 180, 200])
    b = rng.choice([2, 3, 4])
    c0 = rng.choice([50, 100, 150, 200])
    c1 = rng.choice([10, 20, 30])
    c2 = rng.choice([1, 2, 3])
    f = (a - b * q) * q - (c0 + c1 * q + c2 * q**3)
    price = join_terms([(a, ""), (-b, "q")])
    cost = join_terms([(c0, ""), (c1, "q"), (c2, "q^3")])
    wrongs = [
        ("r3h-opt-revenue-not-profit", (a - b * q) * q),
        ("r3h-opt-derivative-power-slip", (a - b * q) * q - (c0 + c1 * q + c2 * q**2)),
        ("r3h-opt-cost-not-subtracted", (a - b * q) * q + c0 + c1 * q + c2 * q**3),
    ]
    return _Ctx(
        "profit",
        q,
        f,
        0,
        a / b,
        True,
        wrongs,
        easy=f"A business sells «q» units a day. The price of each unit is «({price})» dollars and the daily cost of production is «({cost})» dollars. Write the daily profit «P» dollars as a function of «q».",
        medium=f"A business sells «q» units a day at a price of «({price})» dollars each. The daily cost of production is «({cost})» dollars. Find the number of units a day, correct to 2 decimal places, that gives the greatest daily profit.",
        hard=f"A business sells «q» units a day at a price of «({price})» dollars each. The daily cost of production is «({cost})» dollars. Find the greatest possible daily profit, in dollars correct to 2 decimal places.",
        unit_x="units",
        unit_v="dollars",
        what_x="output",
        what_v="profit",
    )


def _ctx_nearest(rng):
    x = sympy.Symbol("x", positive=True)
    c = rng.choice([2, 3, 4, 5, 6, 7])
    f = x**2 + (x**2 - c) ** 2
    wrongs = [
        ("r3h-opt-wrong-coordinate", x**2 + (x - c) ** 2),
        ("r3h-opt-difference-not-squared", x**2 + (x**2 - c) ** 2 - 2 * x),
        ("r3h-opt-distance-formula-slip", x**2 + (x**2 + c) ** 2),
    ]
    return _Ctx(
        "nearest",
        x,
        f,
        0,
        _INF,
        False,
        wrongs,
        easy=f"The point «P(x, x^2)» lies on the parabola «y = x^2». Write the square of the distance «D» from «P» to the point «(0, {c})» as a function of «x».",
        medium=f"The point «P(x, x^2)» lies on the parabola «y = x^2», with «x» positive. Find the value of «x», correct to 2 decimal places, for which «P» is closest to the point «(0, {c})».",
        hard=f"Find the least distance from the point «(0, {c})» to the parabola «y = x^2», correct to 2 decimal places.",
        unit_x="",
        unit_v="units",
        what_x="x-coordinate",
        what_v="distance",
    )


_EASY = (_ctx_tin, _ctx_box, _ctx_rectangle, _ctx_sheet, _ctx_profit, _ctx_nearest)
_MEDIUM = (_ctx_tin, _ctx_box, _ctx_rectangle, _ctx_nearest, _ctx_profit)
_HARD = (_ctx_tin, _ctx_sheet, _ctx_profit, _ctx_nearest, _ctx_rectangle)


# ---------------------------------------------------------------------------------------------------------------
# distractors


def _optimum_of(g, ctx):
    """The stationary point of a wrong function inside (the right) allowed values, or None."""
    pts = stationary(g, ctx.var, ctx.lo, ctx.hi if math.isfinite(ctx.hi) else _INF)
    return pts[0] if len(pts) == 1 else (pts[0] if pts else None)


def _x_wrongs(ctx):
    out = []
    for tag, g in ctx.wrongs:
        p = _optimum_of(g, ctx)
        if p is not None:
            out.append((tag, p))
    for p in all_stationary(ctx.f, ctx.var):
        if abs(p - ctx.optimum_x) > 1e-6:
            out.append(("r3h-opt-root-outside-domain", p))
    roots = _zeros(ctx.f, ctx.var)
    for z in roots:
        if z > 1e-6 and abs(z - ctx.optimum_x) > 1e-6:
            out.append(("r3h-opt-set-function-zero", z))
    if math.isfinite(ctx.hi):
        out.append(("r3h-opt-halfway-guess", (ctx.lo + ctx.hi) / 2))
    return out


def _v_wrongs(ctx):
    f, var = ctx.f, ctx.var
    out = [("r3h-opt-reports-variable", ctx.optimum_x)]
    if ctx.key == "nearest":
        out.append(("r3h-opt-distance-not-rooted", ctx.optimum_v))
    for tag, x in _x_wrongs(ctx):
        v = ctx.report(_value(f, var, x))
        if v is not None:
            out.append(("r3h-opt-evaluated-at-wrong-point" if tag != "r3h-opt-root-outside-domain" else tag, v))
    for tag, g in ctx.wrongs:
        p = _optimum_of(g, ctx)
        if p is not None:
            v = ctx.report(_value(g, var, p))
            if v is not None:
                out.append((tag, v))
    if math.isfinite(ctx.hi):
        v = ctx.report(_value(f, var, ctx.hi * 0.999))
        if v is not None:
            out.append(("r3h-opt-endpoint-value", v))
    return out


def _plausible(ctx, pairs, scale):
    return [(t, v) for t, v in pairs if v is not None and math.isfinite(v) and 0.5 < v < 40 * scale]


# ---------------------------------------------------------------------------------------------------------------
# practice tiers


@generator(
    "r3h_methods_optimisation",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3h_methods_optimisation(rng, tier="easy"):
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}[tier]
    for _ in attempts(80, "an optimisation question with a clean rounding and enough distractors"):
        ctx = rng.choice(pool)(rng).solve()
        if tier == "easy":
            wrongs = list(ctx.wrongs)
            options = expr_options(ctx.f, wrongs, 4, ctx.var)
            if len(options) < 2:
                continue
            stem, latex = say_marked(ctx.easy)
            return dict(
                text=stem,
                latex=latex,
                answer=machine(ctx.f),
                answer_text=plain(ctx.f),
                answer_kind="expression",
                distractors=options,
                working=[f"Eliminate the second variable using the constraint: {_say(ctx.f)}."],
            )
        if tier == "medium":
            value, dp, text = ctx.optimum_x, 2, ctx.medium
            wrongs = _plausible(ctx, _x_wrongs(ctx), value)
        else:
            value, dp, text = ctx.report(ctx.optimum_v), (1 if ctx.key in ("tin", "box", "sheet") else 2), ctx.hard
            wrongs = _plausible(ctx, _v_wrongs(ctx), value)
        if not rounds_unambiguously(value, dp, 0.1) or not 0.05 < abs(value) < 1e5:
            continue
        options = number_options(value, dp, wrongs)
        if len(options) < 2:
            continue
        stem, latex = say_marked(text)
        return dict(
            text=stem,
            latex=latex,
            answer=round(value, dp),
            answer_text=f"{value:.{dp}f}",
            answer_kind="number",
            distractors=options,
            working=[
                f"f({ctx.var}) = {_say(ctx.f)}; f'({ctx.var}) = 0 at {ctx.var} = {ctx.optimum_x:.4f}; the second derivative there is "
                f"{'negative' if ctx.maximise else 'positive'}, so this is the {'greatest' if ctx.maximise else 'least'} value, {ctx.optimum_v:.4f}."
            ],
        )
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    ctx = rng.choice([_ctx_sheet, _ctx_profit])(rng).solve()
    f, var = ctx.f, ctx.var
    xs, vs = ctx.optimum_x, ctx.optimum_v
    second = ctx.second
    if not (
        rounds_unambiguously(xs, 2, 0.1) and rounds_unambiguously(vs, 1, 0.1) and rounds_unambiguously(second, 1, 0.1)
    ):
        raise Redraw
    if abs(second) < 1 or not 0.5 < xs < 400:
        raise Redraw
    stem, latex = say_marked(
        ctx.medium.replace(
            "Find the value of «x», in centimetres correct to 2 decimal places, that gives the box of greatest volume.",
            "",
        )
        .replace(
            "Find the number of units a day, correct to 2 decimal places, that gives the greatest daily profit.", ""
        )
        .strip()
    )
    part_sym = sympy.Symbol("part_b")
    f2 = _second(f, var)
    others = [p for p in all_stationary(f, var) if abs(p - xs) > 1e-6]
    fname = "V" if ctx.key == "sheet" else "P"
    sym = var.name
    domain_note = f"0 < x < {ctx.hi:g}" if ctx.key == "sheet" else f"0 < q < {ctx.hi:g}"
    a_prompt = (
        "Write the volume «V» cm³ of the box as a function of «x»."
        if ctx.key == "sheet"
        else "Write the daily profit «P» dollars as a function of «q»."
    )
    b_prompt = (
        "Find the value of «x», correct to 2 decimal places, that gives the greatest volume."
        if ctx.key == "sheet"
        else "Find the value of «q», correct to 2 decimal places, that gives the greatest profit."
    )
    c_prompt = f"Find the value of «{fname}''({sym})» at the value of «{sym}» you found in part (b), correct to 1 decimal place. Its sign shows whether this is a greatest or a least value."
    d_prompt = (
        "Find the greatest volume of the box, in cubic centimetres correct to 1 decimal place."
        if ctx.key == "sheet"
        else "Find the greatest daily profit, in dollars correct to 1 decimal place."
    )
    xs_wrongs = _plausible(ctx, _x_wrongs(ctx), xs)
    parts = [
        expr_part(a_prompt, f, 2, [f"{fname}({sym}) = {_say(f)}, for {domain_note}."], list(ctx.wrongs), var=var),
        number_part(
            b_prompt,
            xs,
            2,
            3,
            [
                f"{fname}'({sym}) = 0 gives {sym} = {xs:.2f} (the other solution, {', '.join(f'{p:.2f}' for p in others)}, is outside {domain_note})."
            ],
            xs_wrongs,
            sensible_range=(0, None),
        ),
        number_part(
            c_prompt,
            second,
            1,
            1,
            [
                f"{fname}''({sym}) = {_say(f2)}; at {sym} = {xs:.2f} it is {second:.1f}, which is negative, so this is a greatest value."
            ],
            [
                ("r3h-opt-nature-sign-reversed", -second),
                ("r3h-opt-second-derivative-at-other-root", _value(f2, var, others[0]) if others else None),
                (
                    "r3h-opt-second-derivative-at-end",
                    _value(f2, var, ctx.hi * 0.999) if math.isfinite(ctx.hi) else None,
                ),
            ],
            follow_through=sympy.sstr(f2.subs(var, part_sym)),
        ),
        number_part(
            d_prompt,
            vs,
            1,
            2,
            [f"{fname}({xs:.2f}) = {vs:.1f}."],
            _plausible(ctx, _v_wrongs(ctx), vs),
            follow_through=sympy.sstr(f.subs(var, part_sym)),
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, latex=latex, parts=parts)


@generator("exd_x_r3h_methods_optimisation", level=15, presets={"exam": dict(tier="exam")})
def exd_x_r3h_methods_optimisation(rng, tier="exam"):
    """Exam tier: the function, the optimising value, the second derivative and the optimum value."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_with_redraws(rng, _exam_once)
