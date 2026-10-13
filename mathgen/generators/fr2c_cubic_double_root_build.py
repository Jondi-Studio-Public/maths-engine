"""Generates `fr2c_cubic_equation` and `fr2c_cubic_features_exam` questions: the cubic y = k(x - a)(x - b)^2.

One skill: ``f10-cubic-double-root-graph`` (``answer: expression``). The graph of ``y = k(x - a)(x - b)^2`` crosses the
x-axis at ``x = a`` and only touches it at ``x = b``, where it has a turning point. The tiers write the equation of a
drawn cubic (the existing ``cartesian`` figure kind, a curve sampled every 1/8 so each integer x is an exact sample):

- easy: ``k`` is 1. The two x-intercepts are on the grid; the learner has to see which one is a touch and so is squared.
- medium: ``k`` is 1 or -1, so the end behaviour (rising or falling to the right) decides the sign.
- hard: ``k`` is 2, -2, 3, 1/2 or -1/2; a marked point P gives it.

``fr2c_cubic_features_exam`` is the chain for an equation given in words and symbols: (a) which statement about the
x-intercepts is true (mcq), (b) the y-intercept, (c) the x-coordinate of the turning point that is not on the x-axis,
(d) its y-coordinate (follows through from (c)). The turning point is found from the derivative
``k(x - b)(3x - 2a - b)``; the generator checks it with SymPy.
"""

from __future__ import annotations

from fractions import Fraction

import sympy
from sympy import Symbol

from mathgen import figures
from mathgen.core import attempts, check, check_equivalent, generator, phrase

from . import _fr2c_kit as kit

X = Symbol("x")
_LO, _HI = -10, 10
_N = 161


def _frac(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _factor(root: int, power: int = 1) -> str:
    base = "x" if root == 0 else f"(x {'-' if root > 0 else '+'} {abs(root)})"
    return base if power == 1 else f"{base}^{power}"


def _sub(left, value) -> str:
    """``left - value`` written as ``left + |value|`` when the value is negative."""
    return f"{left} - {value}" if value >= 0 else f"{left} + {-value}"


def cubic_text(k: Fraction, a: int, b: int, *, y: bool = True) -> str:
    """``k(x - a)(x - b)^2`` as written on the page."""
    k = Fraction(k)
    coef = "" if k == 1 else "-" if k == -1 else _frac(k) if k.denominator == 1 else f"({_frac(k)})"
    return ("y = " if y else "") + f"{coef}{_factor(a)}{_factor(b, 2)}"


def cubic_expr(k: Fraction, a: int, b: int):
    k = Fraction(k)
    return sympy.Rational(k.numerator, k.denominator) * (X - a) * (X - b) ** 2


def _value(k, a, b, x):
    return k * (x - a) * (x - b) ** 2


def _wrong(mid, k, a, b, a2=None, b2=None):
    """A wrong equation as ``(id, expression, text)`` where the roots/k are given afresh."""
    return (mid, cubic_expr(k, a, b), cubic_text(k, a, b, y=False))


def _wrong_bundle(k, a, b, *, with_sign=True):
    k = Fraction(k)
    out = [
        _wrong("fr2c-double-root-swapped", k, b, a),
        _wrong("fr2c-root-sign-reversed", k, -a, -b),
        (
            "fr2c-double-root-missed",
            sympy.Rational(k.numerator, k.denominator) * (X - a) * (X - b),
            ("" if k == 1 else "-" if k == -1 else _frac(k) if k.denominator == 1 else f"({_frac(k)})")
            + _factor(a)
            + _factor(b),
        ),
        (
            "fr2c-both-roots-squared",
            sympy.Rational(k.numerator, k.denominator) * (X - a) ** 2 * (X - b) ** 2,
            ("" if k == 1 else "-" if k == -1 else _frac(k) if k.denominator == 1 else f"({_frac(k)})")
            + _factor(a, 2)
            + _factor(b, 2),
        ),
    ]
    if with_sign:
        out.append(_wrong("fr2c-cubic-leading-sign-missed", -k, a, b))
    if abs(k) != 1:
        out.append(_wrong("fr2c-leading-coefficient-ignored", Fraction(1) if k > 0 else Fraction(-1), a, b))
    return out


def _fits(k, a, b) -> bool:
    """Both intercepts on the grid, the y-intercept and the turning point inside the window."""
    k = Fraction(k)
    lo, hi = min(a, b) - 1, max(a, b) + 1
    if not (-9 <= lo and hi <= 9):
        return False
    d = Fraction(abs(b - a))
    if abs(Fraction(4) * k * d**3 / 27) > 8:
        return False
    return abs(_value(k, a, b, 0)) <= 9 and abs(_value(k, a, b, 0)) >= 1


def _curve(k, a, b) -> dict:
    kf = float(k)
    return {"pieces": figures.sample_curve(lambda x: kf * (x - a) * (x - b) ** 2, _LO, _HI, (_LO, _HI), n=_N)}


def _spec(rng, tier):
    for _ in attempts(5000, "a cubic that fits the window"):
        a, b = rng.randint(-6, 6), rng.randint(-6, 6)
        if a == b or abs(a - b) > 6:
            continue
        if tier == "easy":
            k = Fraction(1)
        elif tier == "medium":
            k = Fraction(rng.choice([1, -1]))
        else:
            k = rng.choice([Fraction(2), Fraction(-2), Fraction(3), Fraction(1, 2), Fraction(-1, 2)])
        if not _fits(k, a, b):
            continue
        p = None
        if tier == "hard":
            options = []
            for x in range(min(a, b) - 1, max(a, b) + 2):
                y = _value(k, a, b, x)
                if y.denominator == 1 and x not in (a, b) and 1 <= abs(y) <= 9:
                    options.append((x, int(y)))
            if not options:
                continue
            p = rng.choice(options)
        return k, a, b, p
    raise AssertionError("unreachable")


@generator(
    "fr2c_cubic_equation",
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2c_cubic_equation(rng, tier="easy"):
    """The equation of a drawn cubic that crosses the x-axis once and touches it once. See the module docstring."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(60, "an equation question with three different wrong equations"):
        k, a, b, p = _spec(rng, tier)
        try:
            fields = kit.expr_fields(
                cubic_expr(k, a, b), _wrong_bundle(k, a, b), var=X, prefix="y = ", text=cubic_text(k, a, b, y=False)
            )
        except kit.Redraw:
            continue
        break
    if p:
        check(_value(k, a, b, p[0]) == p[1], "the marked point is not on the curve")
    fig = figures.cartesian(
        x_range=(_LO, _HI),
        y_range=(_LO, _HI),
        grid=1,
        equal_scale=True,
        alt="The graph of a cubic drawn on a grid.",
        curves=[_curve(k, a, b)],
        points=[{"at": [p[0], p[1]], "label": "P"}] if p else [],
    )
    form = "y = k(x - p)(x - q)^2" if tier != "easy" else "y = (x - p)(x - q)^2"
    text = phrase(
        rng,
        (
            "The graph of a cubic is drawn{pt}. Its equation has the form {form}, where{k}. Write the equation.",
            "A cubic graph is drawn on the grid{pt}. Find its equation. It has the form {form}, where{k}.",
            "Write the equation of the cubic shown{pt}, in the form {form}, where{k}.",
        ),
        pt="" if not p else ", with the point P marked on it",
        form=form,
        k=" p and q are whole numbers" if tier == "easy" else " p and q are whole numbers and k is a number",
    )
    working = [
        f"The graph crosses the x-axis at x = {a} and touches it at x = {b}, so the factor for {b} is squared.",
        (
            f"The graph rises to the right, so k is positive."
            if tier == "easy"
            else f"Using P({p[0]}, {p[1]}): {p[1]} = k x ({_sub(p[0], a)})({_sub(p[0], b)})^2, so k = {_frac(k)}."
            if p
            else f"The graph {'rises' if k > 0 else 'falls'} to the right, so k = {int(k)}."
        ),
        f"The equation is {cubic_text(k, a, b)}.",
    ]
    return dict(text=text, working=working, figure=fig, **fields)


def _read_cubic(question) -> str:
    """The equation, recomputed from the drawn curve alone: zeros found in the samples, the touching one is squared."""
    pts = [p for piece in question.figure["curves"][0]["pieces"] for p in piece]
    by_x = {round(x, 4): y for x, y in pts}
    zeros = [x for x, y in by_x.items() if abs(y) < 1e-9 and float(x).is_integer()]
    touch, cross = None, None
    for z in zeros:
        left, right = by_x.get(round(z - 0.125, 4)), by_x.get(round(z + 0.125, 4))
        if left is None or right is None:
            continue
        if left * right > 0:
            touch = int(z)
        else:
            cross = int(z)
    x0, y0 = next((x, y) for x, y in sorted(by_x.items()) if abs(y) > 1e-6 and abs(x - round(x)) < 1e-9)
    k = Fraction(y0 / ((x0 - cross) * (x0 - touch) ** 2)).limit_denominator(10)
    return cubic_text(k, cross, touch)


figures.register_reader("fr2c_cubic_equation", _read_cubic)


# ------------------------------------------------------------------------------------------ exam chain


@generator("fr2c_cubic_features_exam", level=10, presets={"exam": {}})
def fr2c_cubic_features_exam(rng):
    """Intercepts, then the turning point that is not on the x-axis of y = k(x - a)(x - b)^2."""
    return kit.build_with_redraw(rng, _build_features, what="a cubic exam chain")


def _build_features(rng):
    for _ in attempts(3000, "a cubic whose second turning point has whole-number coordinates"):
        d = rng.choice([3, 3, 6, -3, -6])
        a = rng.randint(-5, 4)
        b = a + d
        k = rng.choice([1, -1, 2, -2])
        if abs(b) > 7 or a == 0 or b == 0:
            continue
        xt = Fraction(2 * a + b, 3)
        if xt.denominator != 1:
            continue
        break
    xt = int(xt)
    yt = k * (xt - a) * (xt - b) ** 2
    y0 = k * (0 - a) * (0 - b) ** 2
    expr = cubic_expr(Fraction(k), a, b)
    derivative = sympy.diff(expr, X)
    check_equivalent(derivative.subs(X, xt), 0, "the second turning point is not a stationary point")
    check_equivalent(derivative.subs(X, b), 0, "the double root is not a stationary point")
    check(
        expr.subs(X, xt) == yt and expr.subs(X, 0) == y0, "the turning point or intercept disagrees with the equation"
    )

    def stmt(cross, touch):
        return f"The graph crosses the x-axis at x = {cross} and touches it at x = {touch}"

    parts = [
        kit.mcq_part(
            "Which statement about the x-intercepts of the graph is correct?",
            stmt(a, b),
            2,
            [
                f"The factor ({_sub('x', b)}) is squared, so the graph only touches the x-axis at x = {b}; it crosses at x = {a}."
            ],
            [
                ("fr2c-double-root-swapped", stmt(b, a)),
                ("fr2c-root-sign-reversed", stmt(-a, -b)),
                ("fr2c-both-roots-crossed", f"The graph crosses the x-axis at x = {a} and at x = {b}"),
                ("fr2c-both-roots-squared", f"The graph touches the x-axis at x = {a} and at x = {b}"),
            ],
        ),
        kit.num_part(
            "Find the y-intercept of the graph.",
            y0,
            1,
            [f"Put x = 0: y = {k} x ({-a}) x ({-b})^2 = {y0}."],
            [
                ("fr2c-intercept-sign-slip", -y0),
                ("fr2c-intercept-brackets-not-squared", k * (-a) * (-b)),
                ("fr2c-intercept-at-roots", a),
                ("fr2c-leading-coefficient-ignored", (-a) * b * b),
            ],
        ),
        kit.num_part(
            "The graph has two turning points, one on the x-axis. Find the x-coordinate of the other one.",
            xt,
            2,
            [
                f"y' = {'' if k == 1 else '-' if k == -1 else k}({_sub('x', b)})({_sub('3x', 2 * a + b)}) after taking the derivative by the product rule.",
                f"The turning points are at x = {b} and x = {xt}.",
            ],
            [
                ("fr2c-turning-point-at-midpoint", Fraction(a + b, 2)),
                ("fr2c-turning-point-weights-swapped", Fraction(a + 2 * b, 3)),
                ("fr2c-turning-point-on-axis-reported", b),
                ("fr2c-intercept-at-roots", a),
            ],
        ),
        kit.num_part(
            "Using your x-coordinate, find the y-coordinate of that turning point.",
            yt,
            2,
            [f"y = {'' if k == 1 else '-' if k == -1 else k}({_sub(xt, a)})({_sub(xt, b)})^2 = {yt}."],
            [
                ("fr2c-bracket-not-squared", k * (xt - a) * (xt - b)),
                ("fr2c-intercept-sign-slip", -yt),
                ("fr2c-leading-coefficient-ignored", (xt - a) * (xt - b) ** 2),
                ("fr2c-x-coordinate-given-for-y", xt),
            ],
            follow_through=f"{k}*(part_c - ({a}))*(part_c - ({b}))**2",
        ),
    ]
    stem = f"A graph has the equation {cubic_text(Fraction(k), a, b)}."
    return dict(text=stem, parts=parts)
