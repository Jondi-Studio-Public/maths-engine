"""Generates `w6c_taylor` questions (easy, medium, hard) and the exam chain `exd_x_w6c_taylor`.

``uc2-multivariable-taylor-hessian`` (``multivariable-and-vector-calculus``): the degree-2 Taylor polynomial of a
function of two variables about a point,

    T₂ = f + fₓ(x - a) + f_y(y - b) + ½[fₓₓ(x - a)² + 2fₓ_y(x - a)(y - b) + f_yy(y - b)²],

and the classification of a stationary point with the Hessian. Every function is chosen so that its value and
derivatives at the point are rational: polynomials, √(x² + y²) at a Pythagorean point, g·e^w,
sin w and ln(1 + w) with w vanishing at the point.

* easy: one coefficient of T₂ (of (x - a)², (y - b)² or (x - a)(y - b)): ½fₓₓ, ½f_yy or fₓ_y.
* medium: T₂ evaluated at a nearby point, as an exact fraction.
* hard: the Hessian of f = Ax² + kxy + Cy² + (terms of degree 3 and 4, which do not touch the Hessian at the origin):
  the number of integers k for which the origin is a local maximum or minimum, the smallest positive integer k for
  which it is a saddle, or the positive k for which the second-derivative test is inconclusive.
* exam (``exd_x_w6c_taylor``): a cubic with a stationary point P. (a) fₓₓ(P), (b) D = fₓₓf_yy - fₓ_y² (given f_yy and
  fₓ_y; follows through from (a)), (c) the classification of P (multiple choice), (d) the degree-2 estimate of f
  near P (follows through from (a)).

The derivatives are SymPy's; each T₂ is checked a second way against the function itself: the error |f - T₂| along
three directions at distance 2·10⁻⁴ must be of third order, which a wrong second-order coefficient is not.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

import sympy
import math

from ..core import VerificationError, attempts, check, generator
from ._w6c_kit import Math, Redraw, choice_part, compose, math_of, pieces_of, redrawn, value_fields, value_part

X, Y = sympy.symbols("x y")
H = Fraction(1, 2)


def _shift(var, a):
    """``x``, ``(x − 2)`` or ``(x + 1)``: the variable measured from the point's coordinate a."""
    if a == 0:
        return var
    return f"({var} − {a})" if a > 0 else f"({var} + {-a})"


def _shift_tex(var, a):
    if a == 0:
        return var
    return f"({var}-{a})" if a > 0 else f"({var}+{-a})"


# -- jets: value, gradient and Hessian at a point, in exact fractions (no computer algebra on the hot path) ----------


class D2:
    """``(v, gx, gy, hxx, hxy, hyy)``: a function's value and first and second derivatives at one point."""

    __slots__ = ("v", "gx", "gy", "hxx", "hxy", "hyy")

    def __init__(self, v, gx, gy, hxx, hxy, hyy):
        self.v, self.gx, self.gy = Fraction(v), Fraction(gx), Fraction(gy)
        self.hxx, self.hxy, self.hyy = Fraction(hxx), Fraction(hxy), Fraction(hyy)

    def scaled(self, c):
        return D2(*(c * q for q in (self.v, self.gx, self.gy, self.hxx, self.hxy, self.hyy)))

    def __add__(self, o):
        return D2(self.v + o.v, self.gx + o.gx, self.gy + o.gy, self.hxx + o.hxx, self.hxy + o.hxy, self.hyy + o.hyy)

    def __mul__(self, o):  # the product rule, twice
        return D2(
            self.v * o.v,
            self.v * o.gx + o.v * self.gx,
            self.v * o.gy + o.v * self.gy,
            self.v * o.hxx + o.v * self.hxx + 2 * self.gx * o.gx,
            self.v * o.hxy + o.v * self.hxy + self.gx * o.gy + self.gy * o.gx,
            self.v * o.hyy + o.v * self.hyy + 2 * self.gy * o.gy,
        )


def _monomial(i, j, a, b):
    """x^i y^j at (a, b)."""
    p = lambda base, n: Fraction(base) ** n  # noqa: E731
    return D2(
        p(a, i) * p(b, j),
        i * p(a, i - 1) * p(b, j) if i else 0,
        j * p(a, i) * p(b, j - 1) if j else 0,
        i * (i - 1) * p(a, i - 2) * p(b, j) if i > 1 else 0,
        i * j * p(a, i - 1) * p(b, j - 1) if i and j else 0,
        j * (j - 1) * p(a, i) * p(b, j - 2) if j > 1 else 0,
    )


def _poly_jet(terms, a, b):
    total = D2(0, 0, 0, 0, 0, 0)
    for (i, j), c in terms.items():
        total = total + _monomial(i, j, a, b).scaled(c)
    return total


def _poly_value(terms, x, y):
    return sum(c * x**i * y**j for (i, j), c in terms.items())


def _expr_of(terms):
    return sum((c * X**i * Y**j for (i, j), c in terms.items()), sympy.Integer(0))


class Func:
    """A function of x and y from one of the families, with its SymPy form (for the stem), its jet at a point (exact)
    and a float evaluator (to check the jet against the function itself)."""

    def __init__(self, kind, g=None, w=None, terms=None):
        self.kind, self.g, self.w, self.terms = kind, g or {(0, 0): 1}, w, terms

    def expr(self):
        k = self.kind
        if k == "poly":
            return sympy.expand(_expr_of(self.terms))
        if k == "exp":
            return sympy.expand(_expr_of(self.g)) * sympy.exp(_expr_of(self.w)) if self.g != {(0, 0): 1} else sympy.exp(_expr_of(self.w))
        if k == "log":
            return sympy.log(sympy.expand(1 + _expr_of(self.w)))
        if k == "sin":
            return sympy.expand(_expr_of(self.g)) * sympy.sin(_expr_of(self.w))
        if k == "sqrt-sum":
            return sympy.sqrt(X**2 + Y**2)
        return sympy.sqrt(X * Y)

    def jet(self, a, b):
        k = self.kind
        if k == "poly":
            return _poly_jet(self.terms, a, b)
        if k in ("exp", "log", "sin"):
            jw = _poly_jet(self.w, a, b)
            check(jw.v == 0, "w does not vanish at the point")
            if k == "exp":
                out = D2(1, jw.gx, jw.gy, jw.hxx + jw.gx**2, jw.hxy + jw.gx * jw.gy, jw.hyy + jw.gy**2)
            elif k == "log":
                out = D2(0, jw.gx, jw.gy, jw.hxx - jw.gx**2, jw.hxy - jw.gx * jw.gy, jw.hyy - jw.gy**2)
            else:
                out = D2(0, jw.gx, jw.gy, jw.hxx, jw.hxy, jw.hyy)
            return _poly_jet(self.g, a, b) * out
        a, b = Fraction(a), Fraction(b)
        if k == "sqrt-sum":
            r = Fraction(math.isqrt(int(a * a + b * b)))
            check(r * r == a * a + b * b, "not a Pythagorean point")
            return D2(r, a / r, b / r, b * b / r**3, -a * b / r**3, a * a / r**3)
        s = Fraction(math.isqrt(int(a * b)))
        check(s * s == a * b, "xy is not a square at the point")
        return D2(s, b / (2 * s), a / (2 * s), -b * b / (4 * s**3), 1 / (4 * s), -a * a / (4 * s**3))

    def value(self, x, y):
        k = self.kind
        if k == "poly":
            return float(_poly_value(self.terms, x, y))
        if k == "sqrt-sum":
            return math.sqrt(x * x + y * y)
        if k == "sqrt-prod":
            return math.sqrt(x * y)
        w = float(_poly_value(self.w, x, y))
        if k == "log":
            return math.log(1 + w)
        g = float(_poly_value(self.g, x, y))
        return g * (math.exp(w) if k == "exp" else math.sin(w))


class Jet:
    """The value and the first and second derivatives of a function at ``(a, b)``, as exact fractions."""

    def __init__(self, func, a, b):
        d = func.jet(a, b)
        self.func, self.a, self.b = func, a, b
        self.c00, self.fx, self.fy, self.fxx, self.fxy, self.fyy = d.v, d.gx, d.gy, d.hxx, d.hxy, d.hyy
        self._check_against_the_function()

    @property
    def f(self):
        return self.func.expr()

    def is_good_estimate(self, h, k):
        """Whether T₂ at the offset (h, k) is within 20% of the true value, so "estimate" is honest."""
        try:
            true = self.func.value(self.a + float(h), self.b + float(k))
        except (ValueError, ZeroDivisionError, OverflowError):
            return False
        return abs(true - float(self.estimate(h, k))) <= 0.2 * max(1.0, abs(true))

    def estimate(self, h, k, *, order=2):
        h, k = Fraction(h), Fraction(k)
        total = self.c00 + self.fx * h + self.fy * k
        if order >= 2:
            total += H * (self.fxx * h * h + 2 * self.fxy * h * k + self.fyy * k * k)
        return total

    def _check_against_the_function(self):
        """|f - T₂| is of third order along three directions: a wrong second-order coefficient would show as t²."""
        step = Fraction(1, 5000)
        for h, k in ((1, 0), (0, 1), (1, 1)):
            model = float(self.estimate(h * step, k * step))
            actual = self.func.value(self.a + h * float(step), self.b + k * float(step))
            check(abs(actual - model) < 1e-8, "T₂ does not match f to second order")


# -- families of functions ----------------------------------------------------------------------------------------

_W_MONOMIALS = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]


def _vanishing(rng, a, b):
    """A small polynomial w (as terms) with w(a, b) = 0 that is not constant."""
    terms = {m: rng.choice([-2, -1, 1, 2]) for m in rng.sample(_W_MONOMIALS, rng.randint(2, 3))}
    value = _poly_value(terms, a, b)
    if value:
        terms[(0, 0)] = -value
    return terms


_POLY_MONOMIALS = [(i, j) for i in range(4) for j in range(4) if 1 <= i + j <= 3]


def _poly(rng, a, b):
    for _ in attempts(500, "a polynomial with a non-trivial second-order part"):
        chosen = rng.sample(_POLY_MONOMIALS, rng.randint(3, 4))
        terms = {m: rng.choice([-3, -2, -1, 1, 2, 3]) for m in chosen}
        constant = rng.choice([-2, 0, 1, 3])
        if constant:
            terms[(0, 0)] = constant
        if any(i for (i, _j) in terms) and any(j for (_i, j) in terms) and max(i + j for (i, j) in terms) >= 3:
            return Func("poly", terms=terms)
    raise Redraw("no polynomial")  # pragma: no cover


_EXP_FACTORS = [{(0, 0): 1}, {(1, 0): 1}, {(0, 1): 1}, {(1, 0): 1, (0, 1): 1}, {(1, 1): 1}]
_SIN_FACTORS = [{(0, 0): 1}, {(1, 0): 1}, {(0, 1): 1}, {(0, 0): 2}]


def _exp(rng, a, b):
    return Func("exp", g=rng.choice(_EXP_FACTORS), w=_vanishing(rng, a, b))


def _log(rng, a, b):
    return Func("log", w=_vanishing(rng, a, b))


def _sin(rng, a, b):
    return Func("sin", g=rng.choice(_SIN_FACTORS), w=_vanishing(rng, a, b))


_SQRT_POINTS = [(3, 4), (4, 3), (-3, 4), (3, -4), (-4, 3), (4, -3), (-3, -4), (-4, -3)]
_PRODUCT_POINTS = [(4, 1), (1, 4), (9, 1), (1, 9), (4, 9), (9, 4), (4, 4), (9, 9)]


def _draw_function(rng):
    """``(func, a, b)``: a function of x and y and a point at which its derivatives are rational."""
    kind = rng.choice(["poly", "poly", "poly", "exp", "log", "sin", "sqrt", "sqrt-product"])
    if kind == "sqrt":
        a, b = rng.choice(_SQRT_POINTS)
        return Func("sqrt-sum"), a, b
    if kind == "sqrt-product":
        a, b = rng.choice(_PRODUCT_POINTS)
        return Func("sqrt-prod"), a, b
    a, b = rng.randint(-2, 3), rng.randint(-2, 3)
    if kind == "exp" or kind == "sin":
        a, b = rng.randint(-1, 2), rng.randint(-1, 2)
    build = {"poly": _poly, "exp": _exp, "log": _log, "sin": _sin}[kind]
    return build(rng, a, b), a, b


def _draw_jet(rng):
    for _ in attempts(500, "a function with non-zero second derivatives"):
        func, a, b = _draw_function(rng)
        try:
            jet = Jet(func, a, b)
        except (Redraw, ZeroDivisionError, TypeError, ValueError, VerificationError):
            continue
        values = (jet.c00, jet.fx, jet.fy, jet.fxx, jet.fyy, jet.fxy)
        if jet.fxx and jet.fyy and jet.fxy and all(abs(v) <= 60 and v.denominator <= 64 for v in values):
            return jet
    raise Redraw("no usable function")  # pragma: no cover




# -- easy -----------------------------------------------------------------------------------------------------------

_COEFFICIENT_STEMS = (
    "Find the coefficient of {term} in the Taylor polynomial of degree 2 of f(x, y) = {f} about the point ({a}, {b}).",
    "In the degree-2 Taylor polynomial of f(x, y) = {f} about ({a}, {b}), what is the coefficient of {term}?",
    "Expand f(x, y) = {f} to second order about ({a}, {b}). Find the coefficient of {term}.",
)


@redrawn
def _easy(rng):
    jet = _draw_jet(rng)
    a, b = jet.a, jet.b
    which = rng.choice(["xx", "yy", "xy"])
    u, v = _shift("x", a), _shift("y", b)
    u_tex, v_tex = _shift_tex("x", a), _shift_tex("y", b)
    if which == "xx":
        term = Math(f"{u}²", f"{u_tex}^2")
        value = H * jet.fxx
        wrong = [
            ("taylor-second-order-no-half", jet.fxx),
            ("taylor-wrong-second-partial", H * jet.fyy),
            ("taylor-first-derivative-used", jet.fx),
            ("taylor-wrong-second-partial", jet.fxy),
        ]
    elif which == "yy":
        term = Math(f"{v}²", f"{v_tex}^2")
        value = H * jet.fyy
        wrong = [
            ("taylor-second-order-no-half", jet.fyy),
            ("taylor-wrong-second-partial", H * jet.fxx),
            ("taylor-first-derivative-used", jet.fy),
            ("taylor-wrong-second-partial", jet.fxy),
        ]
    else:
        term = Math(f"{u}{v}", f"{u_tex}{v_tex}")
        value = jet.fxy
        wrong = [
            ("taylor-mixed-term-halved", H * jet.fxy),
            ("taylor-mixed-term-doubled", 2 * jet.fxy),
            ("taylor-wrong-second-partial", H * jet.fxx),
            ("taylor-wrong-second-partial", jet.fxx),
            ("taylor-first-derivative-used", jet.fx),
        ]
    text, latex = compose(*pieces_of(rng.choice(_COEFFICIENT_STEMS), term=term, f=math_of(jet.f), a=a, b=b))
    working = [
        f"At ({a}, {b}): fₓₓ = {_f(jet.fxx)}, f_yy = {_f(jet.fyy)}, fₓ_y = {_f(jet.fxy)}.",
        f"The coefficient of {term} is {_f(value)}.",
    ]
    return dict(text=text, latex=latex, working=working, **value_fields(value, wrong))


def _f(value):
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


# -- medium ---------------------------------------------------------------------------------------------------------

_ESTIMATE_STEMS = (
    "Use the Taylor polynomial of degree 2 of f(x, y) = {f} about ({a}, {b}) to estimate f({x1}, {y1}). "
    "Give the estimate as a fraction in lowest terms.",
    "Approximate f({x1}, {y1}) for f(x, y) = {f} using its second-order Taylor polynomial about ({a}, {b}). "
    "Give your answer as a fraction in lowest terms.",
    "Let T₂ be the degree-2 Taylor polynomial of f(x, y) = {f} about ({a}, {b}). Find T₂({x1}, {y1}) as a fraction in lowest terms.",
)
_OFFSETS = [Fraction(1, 2), Fraction(-1, 2), Fraction(1, 4), Fraction(-1, 4)]


def _decimal(value):
    value = Fraction(value)
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text if "." in text or value.denominator == 1 else f"{float(value)}"


@redrawn
def _medium(rng):
    jet = _draw_jet(rng)
    h, k = rng.choice(_OFFSETS), rng.choice(_OFFSETS)
    check(jet.is_good_estimate(h, k), "the estimate is far from the function")
    value = jet.estimate(h, k)
    check(value.denominator <= 128, "an awkward denominator")
    wrong = [
        ("taylor-second-order-omitted", jet.estimate(h, k, order=1)),
        ("taylor-second-order-no-half", jet.c00 + jet.fx * h + jet.fy * k + jet.fxx * h * h + 2 * jet.fxy * h * k + jet.fyy * k * k),
        ("taylor-mixed-term-halved", jet.c00 + jet.fx * h + jet.fy * k + H * (jet.fxx * h * h + jet.fxy * h * k + jet.fyy * k * k)),
        ("taylor-mixed-term-omitted", jet.c00 + jet.fx * h + jet.fy * k + H * (jet.fxx * h * h + jet.fyy * k * k)),
        ("taylor-offsets-swapped", jet.estimate(k, h)),
        ("taylor-constant-omitted", value - jet.c00),
    ]
    text, latex = compose(
        *pieces_of(
            rng.choice(_ESTIMATE_STEMS),
            f=math_of(jet.f),
            a=jet.a,
            b=jet.b,
            x1=_decimal(jet.a + h),
            y1=_decimal(jet.b + k),
        )
    )
    return dict(
        text=text,
        latex=latex,
        working=[
            f"At ({jet.a}, {jet.b}): f = {_f(jet.c00)}, fₓ = {_f(jet.fx)}, f_y = {_f(jet.fy)}, fₓₓ = {_f(jet.fxx)}, fₓ_y = {_f(jet.fxy)}, f_yy = {_f(jet.fyy)}.",
            f"With the step ({_f(h)}, {_f(k)}), T₂ = {_f(value)}.",
        ],
        **value_fields(value, wrong),
    )


# -- hard -----------------------------------------------------------------------------------------------------------

_HIGHER_TERMS = [X**3, X**2 * Y, X * Y**2, Y**3, X**3 * Y, X * Y**3, X**2 * Y**2, X**4, Y**4]


def _quadratic_form_text(a, c, rng):
    """``f(x, y)`` with A x² + k x y + C y² and higher-order terms: ``(SymPy expression with k, plain, latex)``."""
    k = sympy.symbols("k")
    higher = sum(rng.choice([-2, -1, 1, 2, 3]) * m for m in rng.sample(_HIGHER_TERMS, 2))
    expr = a * X**2 + k * X * Y + c * Y**2 + higher
    return expr, k


_COUNT_STEMS = (
    "The function f(x, y) = {f} has a stationary point at the origin. For how many integer values of k does f have a local {kind} there?",
    "Let f(x, y) = {f}, which has a stationary point at (0, 0). How many integers k make (0, 0) a local {kind} of f?",
)
_SADDLE_STEMS = (
    "The function f(x, y) = {f} has a stationary point at the origin. Find the smallest positive integer k for which the origin is a saddle point.",
    "Let f(x, y) = {f}, with a stationary point at (0, 0). What is the least positive integer k that makes (0, 0) a saddle point?",
)
_INCONCLUSIVE_STEMS = (
    "The function f(x, y) = {f} has a stationary point at the origin. For the positive value of k that makes the second-derivative test inconclusive there, find k.",
    "Let f(x, y) = {f}. At the stationary point (0, 0) the second-derivative test is inconclusive for exactly one positive k. Find it.",
)


def _hard_draw(rng, shape):
    for _ in attempts(500, "coefficients for the Hessian question"):
        if shape == "inconclusive":
            unit, s, t = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)
            a, c = unit * s * s, unit * t * t
        else:
            a, c = rng.randint(1, 6), rng.randint(1, 6)
            if isqrt(4 * a * c) ** 2 == 4 * a * c:
                continue
        sign = rng.choice([1, -1]) if shape == "count" else 1
        return a, c, sign
    raise Redraw("no coefficients")  # pragma: no cover


@redrawn
def _hard(rng):
    shape = rng.choice(["count", "saddle", "inconclusive"])
    a, c, sign = _hard_draw(rng, shape)
    expr, k = _quadratic_form_text(sign * a, sign * c, rng)
    four = 4 * a * c
    root = isqrt(four)
    if shape == "count":
        # A, C of the same sign and k² < 4AC: an extremum, a minimum for A > 0 and a maximum for A < 0.
        value = 2 * isqrt(four - 1) + 1
        check(sum(1 for kk in range(-60, 61) if kk * kk < four) == value, "the integer count differs")
        wrong = [
            ("hessian-one-sided-range", isqrt(four - 1)),
            ("hessian-boundary-included", 2 * root + 1 + (0 if root * root != four else 2)),
            ("hessian-count-ceiling", 2 * (root + 1) + 1),
            ("hessian-count-forgot-zero", 2 * isqrt(four - 1)),
        ]
        stem = rng.choice(_COUNT_STEMS)
        kind = "minimum" if sign > 0 else "maximum"
        working = [
            f"The Hessian at the origin is [[{2 * sign * a}, k], [k, {2 * sign * c}]], with D = {4 * a * c} - k².",
            f"A local {kind} needs D > 0 (and the sign of fₓₓ is already {'positive' if sign > 0 else 'negative'}), so k² < {four}: k = -{isqrt(four - 1)}, ..., {isqrt(four - 1)}, which is {value} integers.",
        ]
    elif shape == "saddle":
        value = root + 1
        wrong = [
            ("hessian-boundary-off-by-one", root),
            ("hessian-unrooted", four),
            ("hessian-second-derivative-halved", isqrt(a * c) + 1),
            ("hessian-boundary-off-by-one", root + 2),
        ]
        stem = rng.choice(_SADDLE_STEMS)
        kind = ""
        working = [
            f"D = fₓₓf_yy - fₓ_y² = {four} - k² at the origin. A saddle needs D < 0, so k² > {four}.",
            f"The smallest positive integer with k² > {four} is {value}.",
        ]
    else:
        value = 2 * isqrt(a * c)
        check(value * value == four, "4AC is not a square")
        wrong = [
            ("hessian-unrooted", four),
            ("hessian-second-derivative-halved", isqrt(a * c)),
            ("hessian-unrooted", a * c),
            ("hessian-boundary-off-by-one", value + 1),
        ]
        stem = rng.choice(_INCONCLUSIVE_STEMS)
        kind = ""
        working = [
            f"D = {four} - k² = 0 for k = ±{value}; the positive value is {value}.",
        ]
    wrong = [(tag, v) for tag, v in wrong if v > 0]
    text, latex = compose(*pieces_of(stem, f=math_of(expr), kind=kind))
    return dict(text=text, latex=latex, working=working, **value_fields(value, wrong))



@generator(
    "w6c_taylor",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6c_taylor(rng, tier="easy"):
    """Degree-2 Taylor polynomials and the Hessian of a function of two variables; tiers in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------

_CLASSES = ("a local minimum", "a local maximum", "a saddle point", "inconclusive")


def _stationary_cubic(rng):
    """``(f, a, b, alpha, beta, gamma, f0)``: a cubic with a stationary point (a, b) at which the Hessian is
    [[2α, β], [β, 2γ]] and the value is f0."""
    for _ in attempts(500, "a cubic with a clear classification"):
        a, b = rng.randint(-2, 3), rng.randint(-2, 3)
        alpha, beta, gamma = rng.randint(-3, 3), rng.randint(-3, 3), rng.randint(-3, 3)
        if alpha == 0 or gamma == 0 or 4 * alpha * gamma - beta * beta == 0:
            continue
        cubic = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(4)]
        f0 = rng.randint(-3, 5)
        u, v = X - a, Y - b
        f = sympy.expand(f0 + alpha * u**2 + beta * u * v + gamma * v**2 + cubic[0] * u**3 + cubic[1] * u**2 * v + cubic[2] * u * v**2 + cubic[3] * v**3)
        if sum(1 for c in cubic if c) < 2:
            continue
        return f, a, b, alpha, beta, gamma, f0
    raise Redraw("no cubic")  # pragma: no cover


@generator("exd_x_w6c_taylor", level=19, presets={"exam": dict(tier="exam")})
@redrawn
def exd_x_w6c_taylor(rng, tier="exam"):
    """Exam chain on a cubic with a stationary point P: fₓₓ(P), D, the classification, the degree-2 estimate."""
    check(tier == "exam", f"unknown tier {tier!r}")
    f, a, b, alpha, beta, gamma, f0 = _stationary_cubic(rng)
    terms = {(i, j): int(c) for (i, j), c in sympy.Poly(f, X, Y).as_dict().items()}
    jet = Jet(Func("poly", terms=terms), a, b)
    check((jet.fx, jet.fy) == (0, 0), "P is not stationary")
    check((jet.fxx, jet.fxy, jet.fyy, jet.c00) == (2 * alpha, beta, 2 * gamma, f0), "the derivatives differ from the construction")
    d = jet.fxx * jet.fyy - jet.fxy * jet.fxy
    if d < 0:
        correct = _CLASSES[2]
    elif jet.fxx > 0:
        correct = _CLASSES[0]
    else:
        correct = _CLASSES[1]
    h, k = rng.choice(_OFFSETS), rng.choice(_OFFSETS)
    check(jet.is_good_estimate(h, k), "the estimate is far from the function")
    estimate = jet.estimate(h, k)

    def tag(wrong_class):
        if wrong_class == _CLASSES[3]:
            return "hessian-test-misapplied"
        if {correct, wrong_class} == {_CLASSES[0], _CLASSES[1]}:
            return "hessian-fxx-sign-ignored"
        return "hessian-determinant-sign-misread"

    cubic_text, cubic_latex = compose("Let f(x, y) = ", math_of(f), f". The point P = ({a}, {b}) is a stationary point of f.")
    fyy2, fxy2 = jet.fyy, jet.fxy * jet.fxy
    return dict(
        text=cubic_text,
        latex=cubic_latex,
        parts=[
            value_part(
                "Find fₓₓ at P.",
                jet.fxx,
                1,
                [f"fₓₓ(P) = {_f(jet.fxx)}."],
                [
                    ("taylor-second-order-no-half", H * jet.fxx),
                    ("taylor-wrong-second-partial", jet.fyy),
                    ("taylor-wrong-second-partial", jet.fxy),
                    ("taylor-mixed-term-doubled", 2 * jet.fxx),
                ],
            ),
            value_part(
                f"Given that f_yy(P) = {_f(jet.fyy)} and fₓ_y(P) = {_f(jet.fxy)}, find D = fₓₓf_yy - (fₓ_y)² at P.",
                d,
                2,
                [f"D = ({_f(jet.fxx)})({_f(jet.fyy)}) - ({_f(jet.fxy)})² = {_f(d)}."],
                [
                    ("hessian-mixed-term-not-squared", jet.fxx * jet.fyy - jet.fxy),
                    ("hessian-determinant-sign-misread", jet.fxx * jet.fyy + jet.fxy * jet.fxy),
                    ("hessian-trace-used", jet.fxx + jet.fyy),
                    ("hessian-diagonal-only", jet.fxx * jet.fyy),
                ],
                follow_through=f"part_a*({_f(fyy2)}) - {_f(fxy2)}",
            ),
            choice_part(
                "What does the second-derivative test say about P?",
                correct,
                2,
                ["D < 0 gives a saddle point; D > 0 gives a minimum when fₓₓ > 0 and a maximum when fₓₓ < 0; D = 0 is inconclusive."],
                [(tag(c), c) for c in _CLASSES if c != correct],
            ),
            value_part(
                f"Use the degree-2 Taylor polynomial of f about P to estimate f({_decimal(a + h)}, {_decimal(b + k)}). "
                "Give the estimate as a fraction in lowest terms.",
                estimate,
                3,
                [f"f(P) = {_f(jet.c00)} and the first derivatives are 0, so T₂ = {_f(jet.c00)} + ½[({_f(jet.fxx)})({_f(h)})² + 2({_f(jet.fxy)})({_f(h)})({_f(k)}) + ({_f(jet.fyy)})({_f(k)})²] = {_f(estimate)}."],
                [
                    ("taylor-constant-omitted", estimate - jet.c00),
                    ("taylor-second-order-no-half", jet.c00 + jet.fxx * h * h + 2 * jet.fxy * h * k + jet.fyy * k * k),
                    ("taylor-mixed-term-halved", jet.c00 + H * (jet.fxx * h * h + jet.fxy * h * k + jet.fyy * k * k)),
                    ("taylor-mixed-term-omitted", jet.c00 + H * (jet.fxx * h * h + jet.fyy * k * k)),
                    ("taylor-offsets-swapped", jet.estimate(k, h)),
                ],
                follow_through=(
                    f"{_f(jet.c00)} + (part_a*({_f(h)})**2 + 2*({_f(jet.fxy)})*({_f(h)})*({_f(k)}) + ({_f(jet.fyy)})*({_f(k)})**2)/2"
                ),
            ),
        ],
    )
