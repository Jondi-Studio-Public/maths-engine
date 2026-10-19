"""Generates `s2c1_bernoulli_equation` questions (easy, medium, hard) and the exam chain
`exd_x_s2c1_bernoulli_equation`.

``uc2-bernoulli-equation`` (``ordinary-differential-equations``, answer ``expression``): the Bernoulli equation
``y' + P(x) y = Q(x) y^n`` solved by the substitution ``v = y^(1 - n)`` and the integrating factor of the linear
equation ``v' + (1 - n) P v = (1 - n) Q``, with an initial condition fixing the constant.

* easy: constant coefficients and n = 2, such as ``y' + 2y = 6y^2``.
* medium: constant coefficients with n = 3 or n = -1, or ``P = a/x`` and ``Q = q x^p`` with n = 2.
* hard: ``P = a/x`` with n = 3, -1 or 1/2, or constant P with an exponential ``Q = q e^(rx)`` and n = 2, 3, -1, 1/2.
* exam (``exd_x_s2c1_bernoulli_equation``): (a) the exponent k in v = y^k, (b) the integrating factor of the linear
  equation for v (follows through from (a)), (c) the solution y(x) with the initial condition, (d) y at a stated x to
  3 decimal places (follows through from (c)).

Every equation is built from its linear solution ``v = v_p + C h`` (so Q is derived, never guessed); the solution is
then checked by substituting it back into ``y' + P y = Q y^n`` numerically at three points near the initial point, and
the initial condition is checked directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, pick_distractors, rounds_unambiguously
from ._s2c1_kit import (
    X,
    agree,
    second_route,
    machine,
    neat,
    number_options,
    plain,
    prompt_part,
    same_function,
    say_marked,
    small_fraction,
)

x = X
y = sympy.Symbol("y")
_R = sympy.Rational


@dataclass(frozen=True)
class _Eq:
    family: str  # "const", "power" (P = a/x) or "exp" (Q has e^(rx))
    n: sympy.Rational
    a: int  # P = a or a/x
    q: sympy.Expr  # Q(x)
    x0: sympy.Rational
    y0: sympy.Rational
    C: sympy.Rational
    vp: sympy.Expr
    h: sympy.Expr
    r: int = 0  # exponent in e^(rx), family "exp"
    j: int = 0  # v_p = c x^j, family "power"

    @property
    def s(self):
        return 1 - self.n

    @property
    def P(self):
        return sympy.Integer(self.a) / x if self.family == "power" else sympy.Integer(self.a)

    @property
    def v(self):
        return self.vp + self.C * self.h

    @property
    def y_sol(self):
        return self.v ** (1 / self.s)


def _linear_parts(family, a, qcoef, j, r, alpha, beta):
    """v_p and h for v' + alpha P v = beta Q, Q = qcoef (const), qcoef x^(j-1) or qcoef e^(rx)."""
    if family == "const":
        return _R(beta) * qcoef / (alpha * a), sympy.exp(-alpha * a * x)
    if family == "power":
        return (None if j + alpha * a == 0 else beta * qcoef * x**j / (j + alpha * a)), x ** (-alpha * a)
    return (None if r + alpha * a == 0 else beta * qcoef * sympy.exp(r * x) / (r + alpha * a)), sympy.exp(
        -alpha * a * x
    )


def _draw(rng, family: str, n):
    s = 1 - sympy.Rational(n)
    for _ in attempts(400, "a Bernoulli equation with a clean solution"):
        a = rng.choice([-3, -2, -1, 1, 2, 3])
        j = r = 0
        if family == "const":
            m = rng.choice([-3, -2, -1, 1, 2, 3])
            vp = _R(m)
            q = sympy.Integer(a * m)  # Q = P v_p, since s cancels from s P v_p = s Q
        elif family == "power":
            j = rng.choice([1, 2, 3])
            c = rng.choice([-3, -2, -1, 1, 2, 3])
            if j + s * a == 0:
                continue
            qcoef = c * (j + s * a) / s
            if not small_fraction(qcoef, 2, 18) or qcoef == 0:
                continue
            q = sympy.nsimplify(qcoef) * x ** (j - 1)
            vp = c * x**j
        else:
            r = rng.choice([-2, -1, 1, 2])
            c = rng.choice([-3, -2, -1, 1, 2, 3])
            if r + s * a == 0:
                continue
            qcoef = c * (r + s * a) / s
            if not small_fraction(qcoef, 2, 18) or qcoef == 0:
                continue
            q = sympy.nsimplify(qcoef) * sympy.exp(r * x)
            vp = c * sympy.exp(r * x)
        h = x ** (-s * a) if family == "power" else sympy.exp(-s * a * x)
        x0 = _R(1) if family == "power" else _R(0)
        y0 = rng.choice([1, 2, 3, 4, 9, _R(1, 2), _R(1, 3), _R(1, 4), _R(9, 4)])
        target = sympy.Rational(y0) ** s
        if not target.is_Rational:
            continue
        h0 = sympy.sympify(h).subs(x, x0)
        C = (target - sympy.sympify(vp).subs(x, x0)) / h0
        if C == 0 or not small_fraction(C, 3, 12):
            continue
        eq = _Eq(family, sympy.Rational(n), a, q, x0, sympy.Rational(y0), sympy.nsimplify(C), vp, h, r, j)
        pts = [x0 + _R(k, 10) for k in range(0, 5)]
        if any(float(eq.v.subs(x, p).evalf()) <= 0.05 for p in pts):
            continue
        return eq
    raise AssertionError("unreachable")


def _residual_ok(eq: _Eq) -> bool:
    if not second_route():
        return True
    sol = eq.y_sol
    for k in range(0, 3):
        p = eq.x0 + _R(k + 1, 10)
        lhs = sympy.diff(sol, x) + eq.P * sol
        rhs = eq.q * sol**eq.n
        try:
            a, b = float(lhs.subs(x, p).evalf(30)), float(rhs.subs(x, p).evalf(30))
        except TypeError:
            return False
        if abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
            return False
    return abs(float((sol.subs(x, eq.x0) - eq.y0).evalf(30))) < 1e-12


def _tidy(eq: _Eq):
    """The solution in its neatest spelling, checked against the exact form."""
    sol = neat(eq.y_sol)
    pts = tuple(eq.x0 + _R(k + 1, 10) for k in range(3))
    check(agree(sol, eq.y_sol, x, pts), "the neat form differs from the solution")
    return sol


def _ode_text(eq: _Eq) -> str:
    n = eq.n
    lhs = f"y' {'+' if eq.a > 0 else '-'} {'' if abs(eq.a) == 1 else abs(eq.a)}y" + (
        "/x" if eq.family == "power" else ""
    )
    power = {2: y**2, 3: y**3, -1: 1 / y, _R(1, 2): sympy.sqrt(y)}[n]
    rhs = plain(eq.q * power)
    return f"{lhs} = {rhs}"


def _wrong_solutions(eq: _Eq):
    s = eq.s
    out = [("bernoulli-forgot-final-power", eq.v), ("bernoulli-inverted-power", eq.v ** (-1 / s))]
    # initial condition applied to v instead of y
    C_bad = (eq.y0 - sympy.sympify(eq.vp).subs(x, eq.x0)) / sympy.sympify(eq.h).subs(x, eq.x0)
    out.append(("bernoulli-ic-applied-to-v", (eq.vp + C_bad * eq.h) ** (1 / s)))
    # the factor (1 - n) left out of the linear equation, entirely or on Q only
    for alpha, beta, tag in ((1, 1, "bernoulli-forgot-one-minus-n"), (s, 1, "bernoulli-q-not-scaled")):
        if eq.family == "const":
            vp, h = _linear_parts("const", eq.a, eq.q, 0, 0, alpha, beta)
        elif eq.family == "power":
            vp, h = _linear_parts("power", eq.a, eq.q / x ** (eq.j - 1), eq.j, 0, alpha, beta)
        else:
            vp, h = _linear_parts("exp", eq.a, eq.q / sympy.exp(eq.r * x), 0, eq.r, alpha, beta)
        if vp is None:
            continue
        C2 = (eq.y0**s - sympy.sympify(vp).subs(x, eq.x0)) / sympy.sympify(h).subs(x, eq.x0)
        out.append((tag, (vp + C2 * h) ** (1 / s)))
    return out


def _options(answer, wrongs, n=4):
    kept = []
    pts = tuple(_R(10 + k, 10) + 0 for k in range(3))
    for tag, w in wrongs:
        w = sympy.sympify(w)
        if w.has(sympy.zoo, sympy.nan, sympy.oo, sympy.I):
            continue
        try:
            if same_function(w, answer, x, pts) or any(same_function(w, o, x, pts) for _, o in kept):
                continue
        except Exception:  # noqa: BLE001
            continue
        kept.append((tag, w))
    return pick_distractors(plain(answer), [Distractor(plain(w), tag, machine(w)) for tag, w in kept], n)


_EASY_FAMILY = [("const", 2)]
_MEDIUM_FAMILY = [("const", 3), ("const", -1), ("power", 2)]
_HARD_FAMILY = [
    ("power", 3),
    ("power", -1),
    ("power", _R(1, 2)),
    ("exp", 2),
    ("exp", 3),
    ("exp", -1),
    ("exp", _R(1, 2)),
]


@generator(
    "s2c1_bernoulli_equation",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c1_bernoulli_equation(rng, tier="easy"):
    family, n = rng.choice({"easy": _EASY_FAMILY, "medium": _MEDIUM_FAMILY, "hard": _HARD_FAMILY}[tier])
    eq = _draw(rng, family, n)
    check(_residual_ok(eq), "the solution does not satisfy the equation")
    sol = _tidy(eq)
    domain = " for x > 0" if family == "power" else ""
    stem, stem_latex = say_marked(
        rng.choice(
            [
                "Solve the initial value problem «{ode}, y({x0}) = {y0}»{dom}.",
                "Find «y(x)» given «{ode}» and «y({x0}) = {y0}»{dom}.",
            ]
        ).format(ode=_ode_text(eq), x0=eq.x0, y0=_frac(eq.y0), dom=domain)
    )
    dist = _options(sol, _wrong_solutions(eq))
    check(len(dist) >= 2, "too few Bernoulli distractors")
    return dict(
        text=stem,
        latex=stem_latex,
        answer=machine(sol),
        answer_text=plain(sol),
        answer_kind="expression",
        distractors=dist,
        working=[
            f"Let v = y^({sympy.nsimplify(eq.s)}). The equation becomes linear in v, and v = {plain(eq.v)}.",
            f"The initial condition gives the constant {plain(eq.C)}, so y = {plain(sol)}.",
        ],
    )


def _signed_term(value, symbol: str) -> str:
    """``+ 3v`` or ``- 3v/2`` style term following an operator, with a unit coefficient left out."""
    value = sympy.Rational(value)
    sign = "-" if value < 0 else "+"
    mag = abs(value)
    if mag == 1:
        return f"{sign} {symbol}"
    if mag.q == 1:
        return f"{sign} {mag.p}{symbol}"
    return f"{sign} ({mag.p}/{mag.q}){symbol}"


def _frac(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


@generator("exd_x_s2c1_bernoulli_equation", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c1_bernoulli_equation(rng, tier="exam"):
    """Exam tier: a Bernoulli initial value problem in four connected parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam Bernoulli question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    family, n = rng.choice(_MEDIUM_FAMILY + _HARD_FAMILY)
    eq = _draw(rng, family, n)
    check(_residual_ok(eq), "the solution does not satisfy the equation")
    s, a = eq.s, eq.a
    sol = _tidy(eq)
    x1 = eq.x0 + rng.choice([_R(1, 2), _R(1), _R(3, 2)])
    v1 = float(eq.v.subs(x, x1).evalf(30))
    if v1 <= 0.05:
        raise _Redraw
    value = float(sol.subs(x, x1).evalf(30))
    if not (0.01 <= abs(value) <= 400 and rounds_unambiguously(value, 3, 0.1)):
        raise _Redraw

    mu = x ** (s * a) if eq.family == "power" else sympy.exp(s * a * x)
    # (a) the exponent
    a_wrongs = [
        ("bernoulli-wrong-exponent", eq.n),
        ("bernoulli-wrong-exponent", eq.n - 1),
        ("bernoulli-wrong-exponent", -s),
        ("bernoulli-wrong-exponent", 1 / s),
    ]
    a_text = _frac(s)
    seen = {sympy.Rational(s)}
    a_dist = []
    for tag, w in a_wrongs:
        w = sympy.Rational(w)
        if w in seen:
            continue
        seen.add(w)
        a_dist.append(Distractor(_frac(w), tag, exact_value(w)))
    a_dist = pick_distractors(a_text, a_dist, 4)
    # (b) the integrating factor
    if eq.family == "power":
        b_wrongs = [
            ("bernoulli-integrating-factor-uses-p", x**a),
            ("bernoulli-integrating-factor-sign", x ** (-s * a)),
            ("bernoulli-integrating-factor-not-exponentiated", s * a * sympy.log(x)),
        ]
    else:
        b_wrongs = [
            ("bernoulli-integrating-factor-uses-p", sympy.exp(a * x)),
            ("bernoulli-integrating-factor-sign", sympy.exp(-s * a * x)),
            ("bernoulli-integrating-factor-not-exponentiated", s * a * x),
        ]
    b_dist = _options(mu, b_wrongs)
    # (c) the solution
    c_dist = _options(sol, _wrong_solutions(eq))
    # (d) its value
    d_wrongs = []
    for d in c_dist:
        try:
            val = float(sympy.sympify(d.value).subs(x, x1).evalf(30))
        except (TypeError, ValueError):
            continue
        d_wrongs.append((d.misconception, val))
    d_dist = number_options(value, 3, d_wrongs, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    x1_text = f"{float(x1):g}"
    x1_ft = str(x1.p) if x1.q == 1 else f"{x1.p}/{x1.q}"
    domain = " for x > 0" if eq.family == "power" else ""
    stem, stem_latex = say_marked(
        f"Consider the initial value problem «{_ode_text(eq)}, y({eq.x0}) = {_frac(eq.y0)}»{domain}."
    )
    mu_ft = f"x**({a}*part_a)" if eq.family == "power" else f"exp({a}*part_a*x)"
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt_part("The substitution «v = y^k» turns the equation into a linear equation in «v». Find «k»."),
                answer_kind="fraction",
                answer=exact_value(s),
                answer_text=a_text,
                marks=1,
                working=[f"k = 1 - n = {a_text}."],
                distractors=a_dist,
            ),
            dict(
                **prompt_part(
                    "Find the integrating factor «μ(x)» of the linear equation in «v», taking the constant of integration to be 0."
                ),
                answer_kind="expression",
                answer=machine(mu),
                answer_text=plain(mu),
                marks=2,
                follow_through=mu_ft,
                working=[
                    f"The linear equation is v' {_signed_term(s * a, 'v/x' if eq.family == 'power' else 'v')} = ..., so μ = {plain(mu)}."
                ],
                distractors=b_dist,
            ),
            dict(
                **prompt_part("Solve the initial value problem for «y(x)»."),
                answer_kind="expression",
                answer=machine(sol),
                answer_text=plain(sol),
                marks=3,
                working=[f"v = {plain(eq.v)}, so y = {plain(sol)}."],
                distractors=c_dist,
            ),
            dict(
                **prompt_part("Find «y({x1})», correct to 3 decimal places.", x1=x1_text),
                answer_kind="number",
                answer=round(value, 3),
                answer_text=f"{value:.3f}",
                marks=1,
                follow_through=f"Subs(part_c, (x, e), ({x1_ft}, E))",
                working=[f"y({x1_text}) = {value:.3f}."],
                distractors=d_dist,
            ),
        ],
    )
