"""Generates `u2_log_differentiation` questions and the exam chain `exd_x_u2_log_differentiation`.

``ucv1-logarithmic-differentiation`` (``differentiation``, answer ``expression``): derivatives found by taking the
logarithm of the function first. The stems never name the method: the learner sees a power with a variable exponent, or
a long product and quotient, and has to see that taking logs is the way in.

* easy: dy/dx, for x > 0, of x^(kx), (px)^x or x^(x^2) (a variable base and a variable exponent).
* medium: the value of dy/dx at a point for a long product and quotient of powers of linear factors, which is a rational
  number (log-derivative times the value of y).
* hard: the value of dy/dx at x = 1 or 2 for u(x)^v(x) where both parts need the chain rule, in exact form with a
  natural logarithm.
* exam: f(x) = (x + a)^x. (a) the derivative of ln f(x), (b) f'(1) (follows through from (a)), (c) the y-intercept of
  the tangent at x = 1 (from (b)), (d) the gradient of the normal at x = 1 (from (b)).

Every derivative is checked against SymPy's own `diff` of the function, evaluated at the point where one is asked for.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, rational_text
from ._u2_kit import keep, redraws

TOPIC = "u2_log_differentiation"
EXAM_TOPIC = "exd_x_u2_log_differentiation"

x = sympy.Symbol("x")


def nice(expr) -> str:
    """The learner-facing spelling of a SymPy expression: ln, ^, and no multiplication sign before a function."""
    s = sympy.sstr(expr)
    s = s.replace("log(", "ln(").replace("**", "^")
    s = re.sub(r"(\d)\*(?=[a-zA-Z(])", r"\1", s)
    s = re.sub(r"\)\*(?=[a-zA-Z(])", ")", s)
    return s.replace("*", " ")


def ex_d(expr, tag):
    """An expression distractor from a SymPy value."""
    return Distractor(nice(expr), tag, sympy.sstr(expr))


def answer(expr):
    return dict(
        answer_kind="expression",
        answer=sympy.sstr(expr),
        answer_text=nice(expr),
        answer_latex="plain:" + nice(expr),
    )


_POINTS = (sympy.Rational(7, 10), sympy.Rational(13, 10), sympy.Rational(21, 10))


def equal(a, b) -> bool:
    """Whether two expressions in x agree (to 1e-9) at three points where every family here is defined."""
    return all(abs(float(sympy.N((a - b).subs(x, p), 20))) < 1e-9 for p in _POINTS)


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy_family(rng):
    """(u, v, text of u^v, domain, dy/dx written as y times a tidy bracket)."""
    log, sin, cos, exp = sympy.log, sympy.sin, sympy.cos, sympy.exp
    items = (
        [("xkx", k) for k in (2, 3, 4, 5, 6, 7, -1, -2, -3)]
        + [("pxx", p) for p in (2, 3, 4, 5, 6, 7, 8, 9)]
        + [("xpl", c) for c in (1, 2, 3, 4, 5)]
        + [("xplk", (c, k)) for c in (1, 2, 3, 4) for k in (2, 3)]
        + [(name, None) for name in ("xx2", "xsin", "sinx", "x1x", "xcos", "xex")]
    )
    form, arg = rng.choice(items)
    if form == "xkx":
        k = arg
        return x, k * x, f"x^({'-' if k == -1 else k}x)", "x > 0", x ** (k * x) * (k * log(x) + k)
    if form == "pxx":
        return arg * x, x, f"({arg}x)^x", "x > 0", (arg * x) ** x * (log(arg * x) + 1)
    if form == "xpl":
        return x + arg, x, f"(x + {arg})^x", "x > 0", (x + arg) ** x * (log(x + arg) + x / (x + arg))
    if form == "xplk":
        c, k = arg
        return (
            x + c,
            k * x,
            f"(x + {c})^({k}x)",
            "x > 0",
            (x + c) ** (k * x) * (k * log(x + c) + k * x / (x + c)),
        )
    if form == "xx2":
        return x, x**2, "x^(x²)", "x > 0", x ** (x**2 + 1) * (2 * log(x) + 1)
    if form == "xsin":
        return x, sin(x), "x^(sin x)", "x > 0", x ** sin(x) * (cos(x) * log(x) + sin(x) / x)
    if form == "sinx":
        return sin(x), x, "(sin x)^x", "0 < x < π", sin(x) ** x * (log(sin(x)) + x * cos(x) / sin(x))
    if form == "xcos":
        return x, cos(x), "x^(cos x)", "x > 0", x ** cos(x) * (-sin(x) * log(x) + cos(x) / x)
    if form == "xex":
        return x, exp(x), "x^(e^x)", "x > 0", x ** exp(x) * (exp(x) * log(x) + exp(x) / x)
    return x, 1 / x, "x^(1/x)", "x > 0", x ** (1 / x - 2) * (1 - log(x))


def _easy(rng):
    u, v, shown, domain, ans = _easy_family(rng)
    y = u**v
    log_y = v * sympy.log(u)
    dlog = sympy.diff(log_y, x)
    d = sympy.diff(y, x)
    check(equal(d, ans), "the derivative differs from SymPy's")
    check(equal(y * dlog, ans), "y times the log-derivative differs")
    slips = [
        (v * u ** (v - 1) * sympy.diff(u, x), "log-diff-power-rule-on-variable-exponent"),
        (y * sympy.diff(v, x) * sympy.log(u), "log-diff-forgot-base-derivative"),
        (dlog, "log-diff-forgot-original-function"),
        (y * (sympy.log(u) + v * sympy.diff(u, x) / u), "log-diff-forgot-exponent-derivative"),
    ]
    cands = [ex_d(e, tag) for e, tag in slips if not equal(e, d)]
    text = f"Find dy/dx, for {domain}, where y = {shown}."
    working = [
        f"Take logarithms: ln y = {nice(log_y)}.",
        f"Differentiate both sides: (1/y)dy/dx = {nice(dlog)}, so dy/dx = y × ({nice(dlog)}) = {nice(ans)}.",
    ]
    return dict(text=text, working=working, distractors=keep(nice(ans), cands, 3), **answer(ans))


def _factor_form(y, d):
    """The derivative as y times the log-derivative, an expression SymPy keeps unexpanded."""
    return y * sympy.simplify(sympy.diff(sympy.expand_log(sympy.log(y), force=True), x))


# -- medium --------------------------------------------------------------------------------------------------------


def _factor_text(a, e):
    base = f"(x {'+' if a > 0 else '-'} {abs(a)})"
    return base if e == 1 else f"{base}^{e}"


def _medium(rng):
    for _ in attempts(500, "a product and quotient with a rational derivative at a point"):
        n = rng.choice([3, 3, 4])
        shifts = rng.sample([-3, -2, -1, 1, 2, 3, 4], n)
        exps = [rng.choice([2, 3, 4, 5]) if i < n - 1 else -rng.choice([1, 2, 3]) for i in range(n)]
        if rng.random() < 0.5:
            exps[1] = -exps[1]  # a second factor in the denominator
        x0 = rng.choice([0, 1, 2, 3])
        if any(x0 + s == 0 for s in shifts) or exps[0] == exps[1]:
            continue
        if sum(Fraction(e, x0 + s) for s, e in zip(shifts, exps)) == 0:
            continue  # a zero gradient is a degenerate answer
        break
    bases = [Fraction(x0 + s) for s in shifts]
    y0 = Fraction(1)
    for base, e in zip(bases, exps):
        y0 *= base**e
    logder = sum(Fraction(e) / base for base, e in zip(bases, exps))
    d0 = y0 * logder
    # Second route: the product rule, one factor differentiated at a time.
    product_rule = Fraction(0)
    for i, (base, e) in enumerate(zip(bases, exps)):
        term = e * base ** (e - 1)
        for j, (other, f) in enumerate(zip(bases, exps)):
            if j != i:
                term *= other**f
        product_rule += term
    check(product_rule == d0, "the product rule differs from the logarithmic derivative")

    def rat(v):
        return sympy.Rational(v.numerator, v.denominator)

    wrong_sign = sum(Fraction(abs(e)) / base for base, e in zip(bases, exps))
    no_exps = sum(Fraction(1 if e > 0 else -1) / base for base, e in zip(bases, exps))
    first_only = y0 * Fraction(exps[0]) / bases[0]
    multiplied = sum(Fraction(e) * base for base, e in zip(bases, exps))
    cands = [
        ex_d(rat(logder), "log-diff-forgot-original-function"),
        ex_d(rat(y0 * wrong_sign), "log-diff-denominator-sign"),
        ex_d(rat(y0 * no_exps), "log-diff-exponent-dropped"),
        ex_d(rat(first_only), "log-diff-product-rule-one-factor"),
        ex_d(rat(y0 * multiplied), "log-diff-multiplied-not-divided"),
    ]
    cands = [c for c in cands if sympy.sympify(c.value) != rat(d0)]
    num = " ".join(_factor_text(s, e) for s, e in zip(shifts, exps) if e > 0)
    den = " ".join(_factor_text(s, -e) for s, e in zip(shifts, exps) if e < 0)
    shown = f"({num})/({den})" if den else num
    text = f"Let y = {shown}. Find the value of dy/dx when x = {x0}."
    terms = " + ".join(f"({e})/(x {'+' if s > 0 else '-'} {abs(s)})" for s, e in zip(shifts, exps))
    working = [
        f"ln|y| is the sum of {', '.join(f'({e}) ln|x {chr(43) if s > 0 else chr(45)} {abs(s)}|' for s, e in zip(shifts, exps))}, so (1/y)dy/dx = {terms}.",
        f"At x = {x0}: y = {rational_text(y0)} and the sum is {rational_text(logder)}; dy/dx = {rational_text(d0)}.",
    ]
    return dict(text=text, working=working, distractors=keep(rational_text(d0), cands, 4), **answer(rat(d0)))


# -- hard ----------------------------------------------------------------------------------------------------------

_BASES = (
    [x**2 + c for c in (1, 2, 3, 4, 5)]
    + [a * x + b for a in (2, 3, 4) for b in (-1, 1, 2, 3)]
    + [x**3 + c for c in (1, 2, 3)]
    + [x**2 + x, x**2 + 2 * x, 2 * x**2 + 1, x**2 - 1]
)
_EXPONENTS = [x, 2 * x, 3 * x, x**2, x + 1, x - 1, 2 * x - 1, 2 * x + 1, x**2 + 1, x + 2, 3 * x - 2]


def _spelled(expr) -> str:
    """A base or exponent as a learner writes it: 3x - 1, x² + 1."""
    return nice(expr).replace("^2", "²").replace("^3", "³")


def _hard_draw(rng):
    """(u, v, text of u^v, x0): the base is an integer above 1 at x0 and so is the exponent."""
    for _ in attempts(500, "a base above 1 and an integer exponent at the point"):
        u, v, x0 = rng.choice(_BASES), rng.choice(_EXPONENTS), rng.choice([1, 2])
        u0, v0 = u.subs(x, x0), v.subs(x, x0)
        if u0 >= 2 and u0 <= 12 and v0.is_integer and v0 != 0:
            break
    expo = _spelled(v)
    return u, v, f"({_spelled(u)})^{expo if v == x else f'({expo})'}", x0


def _hard(rng):
    u, v, shown, x0 = _hard_draw(rng)
    y = u**v
    u0, v0 = u.subs(x, x0), v.subs(x, x0)
    du0, dv0 = sympy.diff(u, x).subs(x, x0), sympy.diff(v, x).subs(x, x0)
    check(u0 > 1, "the base at the point is not above 1")
    y0 = u0**v0
    d0 = sympy.expand(y0 * (dv0 * sympy.log(u0) + v0 * du0 / u0))
    check(abs(float(sympy.N(sympy.diff(y, x).subs(x, x0) - d0, 20))) < 1e-9, "SymPy's derivative differs")
    cands = [
        ex_d(sympy.expand(v0 * u0 ** (v0 - 1) * du0), "log-diff-power-rule-on-variable-exponent"),
        ex_d(sympy.expand(y0 * dv0 * sympy.log(u0)), "log-diff-forgot-base-derivative"),
        ex_d(sympy.expand(dv0 * sympy.log(u0) + v0 * du0 / u0), "log-diff-forgot-original-function"),
        ex_d(sympy.expand(y0 * (sympy.log(u0) + v0 * du0 / u0)), "log-diff-forgot-exponent-derivative"),
    ]
    cands = [c for c in cands if abs(float(sympy.N(sympy.sympify(c.value) - d0, 20))) > 1e-9]
    text = f"Let y = {shown}. Find the exact value of dy/dx when x = {x0}."
    working = [
        f"ln y = ({nice(v)}) ln({nice(u)}), so (1/y)dy/dx = ({nice(dv0)}-type term) ln({nice(u)}) + ({nice(v)})·({nice(sympy.diff(u, x))})/({nice(u)}).".replace(
            f"({nice(dv0)}-type term)", f"{nice(sympy.diff(v, x))}"
        ),
        f"At x = {x0}: y = {y0}, so dy/dx = {y0}·({nice(dv0)} ln {u0} + {nice(v0 * du0 / u0)}) = {nice(d0)}.",
    ]
    return dict(text=text, working=working, distractors=keep(nice(d0), cands, 4), **answer(d0))


@generator(
    TOPIC,
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraws
def u2_log_differentiation(rng, tier="easy"):
    """Derivatives that need logarithmic differentiation; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=18, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_u2_log_differentiation(rng, tier="exam"):
    """f(x) = (x + a)^(kx): ln f differentiated, f'(1), the tangent's y-intercept and the normal's gradient at x = 1."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a, k = rng.choice([1, 2, 3, 4, 5]), rng.choice([1, 1, 2, 3])
    f = (x + a) ** (k * x)
    f1 = sympy.Integer((1 + a) ** k)
    dlog = k * sympy.log(x + a) + k * x / (x + a)
    slope = sympy.expand(f1 * dlog.subs(x, 1))
    check(abs(float(sympy.N(sympy.diff(f, x).subs(x, 1) - slope, 20))) < 1e-9, "SymPy's f'(1) differs")
    check(equal(sympy.diff(k * x * sympy.log(x + a), x), dlog), "d/dx ln f differs")
    intercept = sympy.expand(f1 - slope)
    normal = -1 / slope
    expo = "x" if k == 1 else f"{k}x"
    lnf = f"{expo} ln(x + {a})"

    a_cands = [
        ex_d(k * x / (x + a), "log-diff-forgot-base-derivative-log"),
        ex_d(k * sympy.log(x + a), "log-diff-forgot-exponent-derivative"),
        ex_d(k * sympy.log(x + a) + k / (x + a), "log-diff-product-rule-dropped-x"),
        ex_d(sympy.log(x + a) + x / (x + a), "log-diff-forgot-exponent-derivative") if k > 1 else None,
    ]
    b_cands = [
        ex_d(dlog.subs(x, 1), "log-diff-forgot-original-function"),
        ex_d(sympy.Integer(k), "log-diff-power-rule-on-variable-exponent"),
        ex_d(sympy.expand(f1 * k * sympy.log(1 + a)), "log-diff-forgot-base-derivative"),
    ]
    c_cands = [
        ex_d(sympy.expand(f1 + slope), "tangent-intercept-sign"),
        ex_d(sympy.expand(slope - f1), "tangent-intercept-sign"),
        ex_d(sympy.expand(f1), "tangent-intercept-forgot-slope"),
    ]
    d_cands = [
        ex_d(1 / slope, "normal-gradient-not-negated"),
        ex_d(-slope, "normal-gradient-not-reciprocal"),
        ex_d(slope, "normal-gradient-is-tangent"),
    ]
    a_cands = [c for c in a_cands if c is not None and not equal(sympy.sympify(c.value), dlog)]

    def part(prompt, expr, marks, working, cands, ft=None):
        d = dict(prompt=prompt, marks=marks, working=working, distractors=keep(nice(expr), cands), **answer(expr))
        if ft:
            d["follow_through"] = ft
        return d

    rule = f"(x + {a})^{expo if k == 1 else f'({expo})'}"
    stem = rng.choice(
        [
            f"Let f(x) = {rule} for x > 0.",
            f"A function is defined by f(x) = {rule} for x > 0.",
            f"Consider the function f given by f(x) = {rule}, where x > 0.",
        ]
    )
    return dict(
        text=stem,
        parts=[
            part(
                "Find the derivative of ln f(x) with respect to x.",
                dlog,
                2,
                [f"ln f(x) = {lnf}, so d/dx ln f(x) = {nice(dlog)} by the product rule."],
                a_cands,
            ),
            part(
                "Hence find f'(1).",
                slope,
                3,
                [
                    f"f'(x) = f(x) × (d/dx ln f(x)); f(1) = {f1}, so f'(1) = {f1}({nice(dlog.subs(x, 1))}) = {nice(slope)}."
                ],
                b_cands,
                ft=f"{f1}*Subs(part_a, (x,), (1,))",
            ),
            part(
                "Find the y-intercept of the tangent to the graph of y = f(x) at x = 1.",
                intercept,
                3,
                [f"The tangent is y - {f1} = f'(1)(x - 1), so its y-intercept is {f1} - f'(1) = {nice(intercept)}."],
                c_cands,
                ft=f"{f1} - part_b",
            ),
            part(
                "Find the gradient of the normal to the graph at x = 1.",
                normal,
                2,
                [f"The normal's gradient is -1/f'(1) = {nice(normal)}."],
                d_cands,
                ft="-1/part_b",
            ),
        ],
    )
