"""Generates `u2_parametric_polar` questions and the exam chain `exd_x_u2_parametric_polar`.

``ucv1-parametric-polar-slope-area`` (``single-variable-calculus``, answer ``expression``): the slope of a curve given
parametrically or in polar form, a second derivative along a parametric curve, and the area enclosed by a polar curve.
Arc length is not asked (it has its own skill).

* easy: dy/dx at a stated parameter value for a parametric curve (polynomial, exponential or an ellipse).
* medium: d2y/dx2 at a parameter value (the derivative of dy/dx with respect to t, divided by dx/dt again), or the slope
  dy/dx of a polar curve r = f(theta) at a stated angle.
* hard: the area enclosed by a polar curve: a cardioid, one petal of a rose, or the part of a cardioid outside a circle
  (the limits come from where the curves meet).
* exam: x = a t^2, y = t^3 - b t. (a) dy/dx as a function of t, (b) the slope at t = t0 (follows through from (a)),
  (c) d2y/dx2 at t = t0, (d) the gradient of the normal at t0 (follows through from (b)).

Every value is checked numerically with mpmath: dy/dx from numerical derivatives of x(t) and y(t), areas by
quadrature of (1/2) r^2.
"""

from __future__ import annotations

import re

import mpmath as mp
import sympy

from ..answers import Distractor
from ..core import attempts, check, generator
from ._u2_kit import keep, redraws

TOPIC = "u2_parametric_polar"
EXAM_TOPIC = "exd_x_u2_parametric_polar"

t, th = sympy.symbols("t theta")
PI = sympy.pi


def _s(expr):
    """Cheap normal form (no SymPy simplify, which is slow): expand, then cancel common factors."""
    return sympy.cancel(sympy.expand(sympy.sympify(expr)))


def nice(expr) -> str:
    """Learner-facing spelling: π, ln, ^, and no multiplication sign before a symbol or bracket."""
    s = sympy.sstr(expr)
    s = s.replace("log(", "ln(").replace("**", "^").replace("pi", "π")
    s = re.sub(r"(\d)\*(?=[A-Za-zπ(])", r"\1", s)
    s = s.replace("*π", "π")
    return s


def ex_d(expr, tag):
    return Distractor(nice(expr), tag, sympy.sstr(expr))


def answer(expr):
    return dict(
        answer_kind="expression",
        answer=sympy.sstr(expr),
        answer_text=nice(expr),
        answer_latex="plain:" + nice(expr),
    )


def _mp(expr, var):
    return sympy.lambdify(var, expr, "mpmath")


def _num(expr, subs=None):
    return mp.mpf(str(sympy.N(expr if subs is None else expr.subs(subs), 30)))


def _check_slope(x_expr, y_expr, var, point, value, what):
    """dy/dx from central numerical derivatives of x(var) and y(var) at `point`."""
    with mp.workdps(40):
        fx, fy = _mp(x_expr, var), _mp(y_expr, var)
        p = _num(point)
        slope = mp.diff(fy, p) / mp.diff(fx, p)
        check(abs(slope - _num(value)) < mp.mpf(10) ** -10, f"{what}: numerical slope {slope} differs from {value}")


def _text(expr) -> str:
    """A rule written for a stem: 2t² + 1, e^(2t), 3 cos t, with θ for the polar angle."""
    s = sympy.sstr(expr)
    s = s.replace("theta", "θ").replace("**2", "²").replace("**3", "³").replace("**", "^")
    s = re.sub(r"exp\(([^()]*)\)", lambda m: f"e^({m.group(1).replace('*', '')})", s)
    s = re.sub(r"(cos|sin|tan)\(([^()]*)\)", lambda m: f"{m.group(1)} {m.group(2)}", s)
    s = re.sub(r"(\d)\*(?=[a-zθe])", r"\1", s)
    s = s.replace("*", "")
    return s.replace("pi", "π")


# -- easy ----------------------------------------------------------------------------------------------------------


def _curve_poly(rng):
    a, b = rng.choice([1, 2, 3, 4]), rng.randint(-3, 5)
    c, d, e = rng.choice([1, 2, 3, -1, -2]), rng.choice([v for v in range(-6, 7) if v]), rng.randint(-4, 4)
    x_expr, y_expr = a * t**2 + b, c * t**3 + d * t + e
    t0 = rng.choice([1, 2, 3, -1, -2])
    return x_expr, y_expr, sympy.Integer(t0)


def _curve_exp(rng):
    k, m = rng.choice([1, 2, 3]), rng.choice([2, 3, 4])
    if k == m:
        m += 1
    x_expr = sympy.exp(k * t)
    y_expr = rng.choice([sympy.exp(m * t) + t, t * sympy.exp(m * t), sympy.exp(m * t) - t**2])
    return x_expr, y_expr, sympy.Integer(0)


def _curve_ellipse(rng):
    a, b = rng.choice([2, 3, 4, 5]), rng.choice([1, 2, 3, 4, 6])
    if a == b:
        b += 1
    t0 = rng.choice([PI / 4, PI / 3, PI / 6])
    return a * sympy.cos(t), b * sympy.sin(t), t0


def _slope_at(x_expr, y_expr, t0):
    dx, dy = sympy.diff(x_expr, t), sympy.diff(y_expr, t)
    return sympy.nsimplify(_s((dy / dx).subs(t, t0))), dx, dy


def _easy(rng):
    for _ in attempts(200, "a parametric curve with a defined slope"):
        x_expr, y_expr, t0 = rng.choice([_curve_poly, _curve_poly, _curve_exp, _curve_ellipse])(rng)
        dx0 = sympy.diff(x_expr, t).subs(t, t0)
        if dx0 == 0:
            continue
        slope = _s(((sympy.diff(y_expr, t) / sympy.diff(x_expr, t)).subs(t, t0)))
        if slope != 0:
            break
    _check_slope(x_expr, y_expr, t, t0, slope, "parametric slope")
    dx, dy = sympy.diff(x_expr, t), sympy.diff(y_expr, t)
    ndx, ndy = dx.subs(t, t0), dy.subs(t, t0)
    cands = [
        ex_d(_s(ndx / ndy), "param-slope-inverted") if ndy != 0 else None,
        ex_d(_s(ndy), "param-slope-numerator-only"),
        ex_d(_s(-ndy / ndx), "calc-sign-slip"),
        ex_d(
            _s(sympy.diff(y_expr, t, 2).subs(t, t0) / sympy.diff(x_expr, t, 2).subs(t, t0)),
            "param-slope-second-derivatives",
        )
        if sympy.diff(x_expr, t, 2).subs(t, t0) != 0
        else None,
    ]
    cands = [c for c in cands if c is not None and sympy.sympify(c.value) != slope]
    at = nice(t0) if t0 != 0 else "0"
    text = f"A curve is given by x = {_text(x_expr)}, y = {_text(y_expr)}. Find dy/dx at t = {at}."
    working = [
        f"dx/dt = {nice(dx)} and dy/dt = {nice(dy)}.",
        f"dy/dx = (dy/dt)/(dx/dt); at t = {at} this is {nice(ndy)}/{nice(ndx)} = {nice(slope)}.",
    ]
    return dict(text=text, working=working, distractors=keep(nice(slope), cands, 4), **answer(slope))


# -- medium --------------------------------------------------------------------------------------------------------


def _medium_second(rng):
    for _ in attempts(500, "a parametric second derivative"):
        a, b = rng.choice([1, 2, 3]), rng.randint(-2, 4)
        c, d = rng.choice([1, 2, -1, -2, 3]), rng.choice([v for v in range(-5, 6) if v])
        x_expr = a * t**2 + b * t
        y_expr = c * t**3 + d * t
        t0 = rng.choice([1, 2, 3, -1, -2])
        dx0 = sympy.diff(x_expr, t).subs(t, t0)
        if dx0 == 0:
            continue
        break
    dx, dy = sympy.diff(x_expr, t), sympy.diff(y_expr, t)
    first = dy / dx
    second = sympy.diff(first, t) / dx
    value = sympy.nsimplify(second.subs(t, t0))
    with mp.workdps(40):  # numerical second derivative of y along x: d/dt(dy/dx) / dx/dt
        fx, fy = _mp(x_expr, t), _mp(y_expr, t)
        p = _num(t0)
        g = lambda s: mp.diff(fy, s) / mp.diff(fx, s)  # noqa: E731
        num = mp.diff(g, p) / mp.diff(fx, p)
        check(abs(num - _num(value)) < mp.mpf(10) ** -8, f"numerical second derivative {num} differs from {value}")
    dy2, dx2 = sympy.diff(y_expr, t, 2), sympy.diff(x_expr, t, 2)
    cands = [
        ex_d(sympy.diff(first, t).subs(t, t0), "param-second-derivative-not-divided"),
        ex_d(
            sympy.nsimplify((dy2 / dx2).subs(t, t0)) if dx2.subs(t, t0) != 0 else sympy.Integer(0),
            "param-slope-second-derivatives",
        ),
        ex_d(sympy.nsimplify((dy2 / dx).subs(t, t0)), "param-second-derivative-wrong-divisor"),
        ex_d(sympy.nsimplify((dy / dx).subs(t, t0)), "param-second-derivative-is-first"),
    ]
    cands = [c for c in cands if sympy.sympify(c.value) != value]
    text = f"A curve is given by x = {_text(x_expr)}, y = {_text(y_expr)}. Find d²y/dx² at t = {t0}."
    working = [
        f"dy/dx = {nice(_s(first))}.",
        f"d²y/dx² = (d/dt of dy/dx)/(dx/dt) = {nice(_s(second))}; at t = {t0} this is {nice(value)}.",
    ]
    return dict(text=text, working=working, distractors=keep(nice(value), cands, 4), **answer(value))


def _polar_slope(r, theta0):
    s, c = sympy.sin(theta0), sympy.cos(theta0)
    r0, dr0 = r.subs(th, theta0), sympy.diff(r, th).subs(th, theta0)
    num, den = dr0 * s + r0 * c, dr0 * c - r0 * s
    return r0, dr0, num, den


def _medium_polar(rng):
    for _ in attempts(500, "a polar curve with a finite slope"):
        kind = rng.choice(["limacon_cos", "limacon_sin", "spiral", "rose"])
        a, b = rng.randint(1, 4), rng.choice([1, 2, 3])
        if kind == "limacon_cos":
            r, theta0 = a + b * sympy.cos(th), PI / 2
        elif kind == "limacon_sin":
            r, theta0 = a + b * sympy.sin(th), PI
        elif kind == "spiral":
            r, theta0 = a * th, rng.choice([PI / 2, PI])
        else:
            r, theta0 = a * sympy.sin(2 * th), PI / 4
            r = a * sympy.sin(2 * th) + b  # keeps the slope informative at pi/4
        r0, dr0, num, den = _polar_slope(r, theta0)
        if _s(den) == 0 or _s(num) == 0:
            continue
        value = _s((num / den))
        break
    # Second route: the curve as (x(theta), y(theta)) = (r cos, r sin), differentiated numerically.
    _check_slope(r * sympy.cos(th), r * sympy.sin(th), th, theta0, value, "polar slope")
    s, c = sympy.sin(theta0), sympy.cos(theta0)
    cands = [
        ex_d(_s((den / num)), "polar-slope-inverted"),
        ex_d(_s((dr0 / r0)) if r0 != 0 else sympy.Integer(1), "polar-slope-is-dr-over-r"),
        ex_d(_s(((dr0 * s - r0 * c) / (dr0 * c + r0 * s))), "polar-slope-sign-slip"),
        ex_d(_s((-value)), "polar-slope-sign-slip"),
        ex_d(_s((-sympy.cot(theta0))), "polar-slope-forgot-dr") if s != 0 else None,
        ex_d(_s(((dr0 + r0) / (dr0 - r0))), "polar-slope-trig-dropped") if dr0 != r0 else None,
        ex_d(_s(((dr0 * s + r0 * c) / (dr0 * c + r0 * s))), "polar-slope-sign-slip"),
        ex_d(_s((sympy.tan(theta0))), "polar-slope-is-tan-theta") if c != 0 else None,
    ]
    cands = [c_ for c_ in cands if c_ is not None and _s(sympy.sympify(c_.value) - value) != 0]
    text = f"Find the gradient dy/dx of the polar curve r = {_text(r)} at θ = {nice(theta0)}."
    working = [
        "With x = r cos θ and y = r sin θ: dy/dx = (r' sin θ + r cos θ)/(r' cos θ - r sin θ).",
        f"At θ = {nice(theta0)}: r = {nice(r0)}, r' = {nice(dr0)}, so dy/dx = ({nice(_s(num))})/({nice(_s(den))}) = {nice(value)}.",
    ]
    return dict(text=text, working=working, distractors=keep(nice(value), cands, 4), **answer(value))


def _medium(rng):
    return rng.choice([_medium_second, _medium_polar])(rng)


# -- hard ----------------------------------------------------------------------------------------------------------


def _quad_area(r, lo, hi):
    with mp.workdps(40):
        f = _mp(r**2 / 2, th)
        return mp.quad(f, [_num(lo), _num(hi)])


def _hard(rng):
    kind = rng.choice(["cardioid", "petal", "outside", "lemniscate", "spiral", "sine_petal"])
    a = rng.choice([1, 2, 3, 4])
    unit = "" if a == 1 else str(a)
    card = "1 + cos θ" if a == 1 else f"{a}(1 + cos θ)"
    if kind == "cardioid":
        r = a * (1 + sympy.cos(th))
        area = sympy.Rational(3, 2) * PI * a**2
        check(abs(_quad_area(r, 0, 2 * PI) - _num(area)) < mp.mpf(10) ** -12, "cardioid area differs")
        cands = [
            ex_d(sympy.Rational(3, 4) * PI * a**2, "polar-area-half-range"),
            ex_d(3 * PI * a**2, "polar-area-forgot-half"),
            ex_d(sympy.Rational(1, 2) * PI * a**2, "polar-area-r-not-squared"),
            ex_d(sympy.Rational(5, 2) * PI * a**2, "polar-area-expansion-slip"),
        ]
        text = f"Find the exact area enclosed by the cardioid r = {card}."
        working = [
            f"A = (1/2)∫₀^(2π) r² dθ = ({a * a}/2)∫₀^(2π) (1 + 2cos θ + cos²θ) dθ.",
            f"The integral of 1 + 2cos θ + (1 + cos 2θ)/2 over a full turn is 3π, so A = {nice(area)}.",
        ]
    elif kind == "petal":
        n = rng.choice([2, 3, 4, 5])
        r = a * sympy.cos(n * th)
        area = PI * a**2 / (4 * n)
        check(
            abs(_quad_area(r, -PI / (2 * n), PI / (2 * n)) - _num(area)) < mp.mpf(10) ** -12,
            "petal area differs",
        )
        cands = [
            ex_d(PI * a**2 / (2 * n), "polar-area-forgot-half"),
            ex_d(PI * a**2 / 4, "polar-area-petal-count-ignored"),
            ex_d(PI * a**2 / (8 * n), "polar-area-half-range"),
            ex_d(PI * a**2 / 2, "polar-area-whole-rose"),
        ]
        text = (
            f"The rose r = {unit}cos {n}θ has a petal along the positive x-axis. Find the exact area of that one petal."
        )
        working = [
            f"The petal runs from θ = -π/{2 * n} to θ = π/{2 * n}, where r = 0.",
            f"A = (1/2)∫ {a * a}cos²({n}θ) dθ over that range = {nice(area)}.",
        ]
    elif kind == "lemniscate":
        r2 = a**2 * sympy.cos(2 * th)
        area = sympy.Integer(a) ** 2
        with mp.workdps(40):
            f = _mp(r2 / 2, th)
            check(abs(4 * mp.quad(f, [0, _num(PI / 4)]) - _num(area)) < mp.mpf(10) ** -12, "lemniscate area differs")
        cands = [
            ex_d(sympy.Integer(a) ** 2 / 2, "polar-area-petal-count-ignored"),
            ex_d(2 * sympy.Integer(a) ** 2, "polar-area-forgot-half"),
            ex_d(PI * a**2 / 2, "polar-area-whole-rose"),
            ex_d(sympy.Integer(a) ** 2 / 4, "polar-area-half-range"),
        ]
        text = f"Find the exact area enclosed by the lemniscate r² = {a * a if a > 1 else ''}cos 2θ."
        working = [
            "The curve has two loops, from θ = -π/4 to π/4 and from 3π/4 to 5π/4. By symmetry, A = 4 × (1/2)∫₀^(π/4) r² dθ.",
            f"A = 2∫₀^(π/4) {a * a}cos 2θ dθ = {a * a}[sin 2θ]₀^(π/4) = {nice(area)}.",
        ]
    elif kind == "spiral":
        r = a * th
        area = a**2 * PI**3 / 6
        check(abs(_quad_area(r, 0, PI) - _num(area)) < mp.mpf(10) ** -12, "spiral area differs")
        cands = [
            ex_d(a**2 * PI**3 / 3, "polar-area-forgot-half"),
            ex_d(a * PI**2 / 4, "polar-area-r-not-squared"),
            ex_d(a**2 * PI**3 / 18, "polar-area-expansion-slip"),
            ex_d(a**2 * PI**2 / 2, "polar-area-expansion-slip"),
        ]
        text = f"Find the exact area swept out by the spiral r = {unit}θ for 0 ≤ θ ≤ π."
        working = [f"A = (1/2)∫₀^π {a * a}θ² dθ = ({a * a}/2)(π³/3) = {nice(area)}."]
    elif kind == "sine_petal":
        n = rng.choice([2, 3, 4])
        r = a * sympy.sin(n * th)
        area = PI * a**2 / (4 * n)
        check(abs(_quad_area(r, 0, PI / n) - _num(area)) < mp.mpf(10) ** -12, "sine petal area differs")
        cands = [
            ex_d(PI * a**2 / (2 * n), "polar-area-forgot-half"),
            ex_d(PI * a**2 / 4, "polar-area-petal-count-ignored"),
            ex_d(PI * a**2 / (8 * n), "polar-area-half-range"),
            ex_d(PI * a**2 / 2, "polar-area-whole-rose"),
        ]
        text = (
            f"The rose r = {unit}sin {n}θ has a petal for 0 ≤ θ ≤ π/{n}, traced as θ increases from 0. "
            "Find the exact area of that petal."
        )
        working = [f"A = (1/2)∫₀^(π/{n}) {a * a}sin²({n}θ) dθ = {nice(area)}."]
    else:
        r = a * (1 + sympy.cos(th))
        area = a**2 * (2 + PI / 4)
        with mp.workdps(40):
            f = _mp((r**2 - a**2) / 2, th)
            check(
                abs(mp.quad(f, [-_num(PI / 2), _num(PI / 2)]) - _num(area)) < mp.mpf(10) ** -12, "outside area differs"
            )
        cands = [
            ex_d(a**2 * 2, "polar-area-circle-subtracted-wrong"),
            ex_d(sympy.Rational(3, 2) * PI * a**2 - PI * a**2, "polar-area-limits-whole-turn"),
            ex_d(a**2 * (4 + PI / 2), "polar-area-forgot-half"),
            ex_d(a**2 * (1 + PI / 4), "polar-area-expansion-slip"),
        ]
        text = f"Find the exact area of the region inside the cardioid r = {card} and outside the circle r = {a}."
        working = [
            f"The curves meet where {a}(1 + cos θ) = {a}, at θ = ±π/2.",
            f"A = (1/2)∫ from -π/2 to π/2 of ({a}²(1 + cos θ)² - {a}²) dθ = {nice(area)}.",
        ]
    cands = [c for c in cands if _s(sympy.sympify(c.value) - area) != 0]
    return dict(text=text, working=working, distractors=keep(nice(area), cands, 4), **answer(area))


@generator(
    TOPIC,
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraws
def u2_parametric_polar(rng, tier="easy"):
    """Slopes of parametric and polar curves and areas in polar form; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=18, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_u2_parametric_polar(rng, tier="exam"):
    """x = a t², y = t³ - b t: dy/dx(t), the slope at t0, d²y/dx² at t0 and the normal's gradient."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a parametric curve whose slope is defined at t0"):
        a, b = rng.choice([1, 2, 3]), rng.choice([1, 2, 3, 4, 5, 6])
        t0 = rng.choice([2, 3, -2, -3])
        x_expr, y_expr = a * t**2, t**3 - b * t
        slope0 = sympy.Rational(3 * t0**2 - b, 2 * a * t0)
        if abs(slope0) != 1 and slope0 != 0:
            break
    dx, dy = sympy.diff(x_expr, t), sympy.diff(y_expr, t)
    first = dy / dx
    second = sympy.together(sympy.diff(dy / dx, t) / dx)
    second0 = second.subs(t, t0)
    _check_slope(x_expr, y_expr, t, sympy.Integer(t0), slope0, "slope")
    check(_s(first.subs(t, t0) - slope0) == 0, "dy/dx(t) differs at t0")
    normal = -1 / slope0

    a_cands = [
        ex_d(dx / dy, "param-slope-inverted"),
        ex_d(dy, "param-slope-numerator-only"),
        ex_d(_s(sympy.diff(y_expr, t, 2) / sympy.diff(x_expr, t, 2)), "param-slope-second-derivatives"),
    ]
    b_cands = [
        ex_d(sympy.Rational(2 * a * t0, 3 * t0**2 - b), "param-slope-inverted"),
        ex_d(sympy.Integer(3 * t0**2 - b), "param-slope-numerator-only"),
        ex_d(-slope0, "calc-sign-slip"),
    ]
    c_cands = [
        ex_d(sympy.diff(first, t).subs(t, t0), "param-second-derivative-not-divided"),
        ex_d(sympy.Rational(6 * t0, 2 * a), "param-slope-second-derivatives"),
        ex_d(sympy.Rational(6 * t0, 2 * a * t0), "param-second-derivative-wrong-divisor"),
    ]
    d_cands = [
        ex_d(1 / slope0, "normal-gradient-not-negated"),
        ex_d(-slope0, "normal-gradient-not-reciprocal"),
        ex_d(slope0, "normal-gradient-is-tangent"),
    ]

    def part(prompt, expr, marks, working, cands, ft=None):
        d = dict(prompt=prompt, marks=marks, working=working, distractors=keep(nice(expr), cands), **answer(expr))
        if ft:
            d["follow_through"] = ft
        return d

    return dict(
        text=f"A curve is given parametrically by x = {_text(x_expr)}, y = {_text(y_expr)}.",
        parts=[
            part(
                "Find dy/dx as a function of t.",
                first,
                3,
                [f"dx/dt = {nice(dx)} and dy/dt = {nice(dy)}, so dy/dx = {nice(first)}."],
                a_cands,
            ),
            part(
                f"Find the gradient of the curve at t = {t0}.",
                slope0,
                2,
                [f"Substitute t = {t0}: dy/dx = {nice(slope0)}."],
                b_cands,
                ft=f"Subs(part_a, (t,), ({t0},))",
            ),
            part(
                f"Find d²y/dx² at t = {t0}.",
                second0,
                3,
                [f"d²y/dx² = (d/dt of dy/dx)/(dx/dt) = {nice(second)}; at t = {t0} this is {nice(second0)}."],
                c_cands,
            ),
            part(
                f"Find the gradient of the normal to the curve at t = {t0}.",
                normal,
                2,
                [f"The normal's gradient is -1/({nice(slope0)}) = {nice(normal)}."],
                d_cands,
                ft="-1/part_b",
            ),
        ],
    )
