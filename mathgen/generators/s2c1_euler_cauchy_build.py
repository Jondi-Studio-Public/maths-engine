"""Generates `s2c1_euler_cauchy` questions (easy, medium, hard) and the exam chain `exd_x_s2c1_euler_cauchy`.

``uc2-euler-cauchy`` (``ordinary-differential-equations``, answer ``expression``): the equidimensional equation
``x² y'' + a x y' + b y = 0`` for ``x > 0`` with ``y(1)`` and ``y'(1)`` given. Trying ``y = x^r`` gives
``r² + (a - 1) r + b = 0``; the stem never shows that equation.

* easy: two distinct integer exponents, ``y = c1 x^r1 + c2 x^r2``.
* medium: a repeated exponent, ``y = (c1 + c2 ln x) x^r``.
* hard: complex exponents ``α ± βi``, ``y = x^α (c1 cos(β ln x) + c2 sin(β ln x))``.
* exam (``exd_x_s2c1_euler_cauchy``): any of the three cases: (a) the exponent equation as a polynomial in r,
  (b) the larger root, the repeated root, or β, (c) the solution y(x) for the initial conditions, (d) y at a stated x
  to 3 decimal places (follows through from (c)).

The solution is re-checked by substituting it back into ``x² y'' + a x y' + b y`` at three points and against both
initial values; the wrong answers are the same fit with one named slip in the exponents, the form or the derivative.
"""

from __future__ import annotations

import sympy

from ..core import VerificationError, attempts, check, generator, join_terms, rounds_unambiguously
from ._s2c1_kit import (
    X,
    expr_options,
    machine,
    number_options,
    plain,
    prompt_part,
    say_marked,
    small_fraction,
    solve_constants,
    solves_ode,
)

x = X
r = sympy.Symbol("r")
_R = sympy.Rational
_LN = sympy.log(x)


def _basis(kind: str, p, q, *, wrong: str | None = None):
    """The two basis functions for exponents (p, q): distinct roots p, q; a repeated root p; or α = p, β = q."""
    if kind == "distinct":
        return x**p, x**q
    if kind == "repeated":
        return (x**p, x ** (p + 1)) if wrong == "no-log" else (x**p, x**p * _LN)
    arg = q * x if wrong == "no-log" else q * _LN
    return x**p * sympy.cos(arg), x**p * sympy.sin(arg)


def _fit(kind, p, q, y0, v0, *, wrong=None, swap=False):
    f1, f2 = _basis(kind, p, q, wrong=wrong)
    c1, c2 = solve_constants(f1, f2, 0, 1, y0, v0)
    if swap:
        c1, c2 = c2, c1
    return sympy.expand(c1 * f1 + c2 * f2)


def _roots_of(p_coef, q_coef):
    """(kind, p, q) for the roots of r² + p_coef r + q_coef = 0 when they are rational (or α ± βi with rational α,
    β), else None."""
    disc = p_coef * p_coef - 4 * q_coef
    if disc == 0:
        return "repeated", _R(-p_coef, 2), None
    root = sympy.sqrt(abs(disc))
    if not root.is_Rational:
        return None
    if disc > 0:
        r1, r2 = (-p_coef + root) / 2, (-p_coef - root) / 2
        return "distinct", sympy.nsimplify(r1), sympy.nsimplify(r2)
    return "complex", _R(-p_coef, 2), sympy.nsimplify(root / 2)


class _Case:
    def __init__(self, kind, a, b, p, q, y0, v0, sol):
        self.kind, self.a, self.b, self.p, self.q, self.y0, self.v0, self.sol = kind, a, b, p, q, y0, v0, sol


def _draw(rng, kind: str) -> _Case:
    for _ in attempts(500, "an Euler-Cauchy equation with clean constants"):
        if kind == "distinct":
            p, q = rng.sample([-3, -2, -1, 0, 1, 2, 3, 4], 2)
            if p < q:
                p, q = q, p
            a, b = 1 - (p + q), p * q
        elif kind == "repeated":
            p, q = rng.choice([-2, -1, 0, 1, 2, 3]), None
            a, b = 1 - 2 * p, p * p
        else:
            p, q = rng.choice([-2, -1, 0, 1, 2]), rng.choice([1, 2, 3])
            a, b = 1 - 2 * p, p * p + q * q
        y0, v0 = rng.randint(-4, 5), rng.randint(-6, 6)
        if y0 == 0 and v0 == 0:
            continue
        c1, c2 = solve_constants(*_basis(kind, p, q), 0, 1, y0, v0)
        if c1 == 0 or c2 == 0:
            continue
        if not (small_fraction(c1, 4, 20) and small_fraction(c2, 4, 20)):
            continue
        sol = _fit(kind, sympy.Integer(p), None if q is None else sympy.Integer(q), y0, v0)
        pts = (_R(7, 10), _R(13, 10), _R(19, 10))
        check(solves_ode(sol, (x**2, a * x, b), 0, x, pts), "the solution does not satisfy the equation")
        check(sol.subs(x, 1) == y0 and sympy.diff(sol, x).subs(x, 1) == v0, "initial conditions are not met")
        return _Case(kind, a, b, sympy.Integer(p), None if q is None else sympy.Integer(q), y0, v0, sol)
    raise AssertionError("unreachable")


def _wrongs(c: _Case):
    out = []

    def fit(tag, *args, **kwargs):
        try:
            out.append((tag, _fit(*args, **kwargs)))
        except VerificationError:
            pass  # these exponents make the two solutions dependent: not a usable wrong answer

    kind, a, b, p, q, y0, v0 = c.kind, c.a, c.b, c.p, c.q, c.y0, c.v0
    # slips in the exponent equation: r² + a r + b (the "- 1" forgotten), and two other wrong middle coefficients
    for poly_p, poly_q in ((a, b), (a + 1, b), (1 - a, b)):
        slip = _roots_of(poly_p, poly_q)
        if slip is not None:
            k2, p2, q2 = slip
            fit("euler-cauchy-indicial-slip", k2, p2, q2, y0, v0)
    # the sign of each exponent reversed
    fit("euler-cauchy-sign-of-exponent", kind, -p, None if q is None else (-q if kind == "distinct" else q), y0, v0)
    fit("euler-cauchy-constants-swapped", kind, p, q, y0, v0, swap=True)
    if kind == "distinct":
        fit("euler-cauchy-sign-of-exponent", kind, -p, q, y0, v0)
        fit("euler-cauchy-sign-of-exponent", kind, p, -q, y0, v0)
    if kind != "distinct":
        c1, c2 = solve_constants(*_basis(kind, p, q), 0, 1, y0, v0)
        g1, g2 = _basis(kind, p, q, wrong="no-log")
        out.append(("euler-cauchy-missing-log", sympy.expand(c1 * g1 + c2 * g2)))
        if kind == "repeated":
            out.append(("euler-cauchy-derivative-slip", sympy.expand((y0 + v0 * _LN) * x**p)))
        else:
            out.append(
                (
                    "euler-cauchy-derivative-slip",
                    sympy.expand(x**p * (y0 * sympy.cos(q * _LN) + (v0 / q) * sympy.sin(q * _LN))),
                )
            )
    return out


def _equation(a: int, b: int) -> str:
    return join_terms([(1, "x^2 y''"), (a, "x y'"), (b, "y")])


@generator(
    "s2c1_euler_cauchy",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c1_euler_cauchy(rng, tier="easy"):
    kind = {"easy": "distinct", "medium": "repeated", "hard": "complex"}[tier]
    for _ in attempts(40, "an Euler-Cauchy question with enough distractors"):
        c = _draw(rng, kind)
        dist = expr_options(c.sol, _wrongs(c), 4)
        if len(dist) >= 2:
            break
    template = rng.choice(
        [
            "Solve the initial value problem «{eq} = 0» for «x > 0», with «y(1) = {y0}» and «y'(1) = {v0}».",
            "Find «y(x)» for «x > 0» given «{eq} = 0», «y(1) = {y0}» and «y'(1) = {v0}».",
        ]
    )
    stem, stem_latex = say_marked(template.format(eq=_equation(c.a, c.b), y0=c.y0, v0=c.v0))
    return dict(
        text=stem,
        latex=stem_latex,
        answer=machine(c.sol),
        answer_text=plain(c.sol),
        answer_kind="expression",
        distractors=dist,
        working=[
            f"Try y = x^r: {join_terms([(1, 'r(r - 1)'), (c.a, 'r'), (c.b, '')])} = 0, that is "
            f"{join_terms([(1, 'r²'), (c.a - 1, 'r'), (c.b, '')])} = 0.",
            f"The initial conditions fix the constants, giving y = {plain(c.sol)}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


@generator("exd_x_s2c1_euler_cauchy", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c1_euler_cauchy(rng, tier="exam"):
    """Exam tier: an Euler-Cauchy initial value problem in four connected parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam Euler-Cauchy question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    kind = rng.choice(["distinct", "repeated", "complex"])
    c = _draw(rng, kind)
    a, b = c.a, c.b
    x1 = rng.choice([_R(1, 2), _R(3, 2), _R(2), _R(5, 2), _R(3)])
    value = float(c.sol.subs(x, x1).evalf(30))
    if not (0.01 <= abs(value) <= 500 and rounds_unambiguously(value, 3, 0.1)):
        raise _Redraw

    poly = r**2 + (a - 1) * r + b
    a_wrongs = [
        ("euler-cauchy-indicial-slip", r**2 + a * r + b),
        ("euler-cauchy-indicial-slip", r**2 + (a + 1) * r + b),
        ("euler-cauchy-indicial-sign", r**2 - (a - 1) * r + b),
        ("euler-cauchy-indicial-slip", r**2 + (a - 1) * r - b),
    ]
    a_dist = expr_options(poly, a_wrongs, 4, var=r)

    if kind == "distinct":
        ask, target = "the larger of the two values of r that work", c.p
        wrong_b = [-c.q, c.q, -c.p, c.p + 1]
        b_prompt = "Find the larger value of «r» for which «y = x^r» solves the equation."
    elif kind == "repeated":
        ask, target = "the repeated value of r", c.p
        wrong_b = [-c.p, c.p + 1, c.p - 1, 1 - c.p]
        b_prompt = "The exponent equation has a repeated root. Find it."
    else:
        ask, target = "β", c.q
        wrong_b = [c.q * c.q, c.q + 1, c.q - 1, 2 * c.q]
        b_prompt = "The roots of the exponent equation are «α ± βi» with «β > 0». Find «β»."
    b_dist = number_options(float(target), 0, [("euler-cauchy-root-slip", float(w)) for w in wrong_b], 4, gap=1)
    c_dist = expr_options(c.sol, _wrongs(c), 4)
    d_wrongs = [(d.misconception, float(sympy.sympify(d.value).subs(x, x1).evalf(30))) for d in c_dist]
    d_dist = number_options(value, 3, d_wrongs, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    x1_text = f"{float(x1):g}"
    x1_ft = str(x1.p) if x1.q == 1 else f"{x1.p}/{x1.q}"
    stem, stem_latex = say_marked(
        f"Consider «{_equation(a, b)} = 0» for «x > 0», with «y(1) = {c.y0}» and «y'(1) = {c.v0}»."
    )
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt_part(
                    "Substituting «y = x^r» gives an equation for «r». Write it as a polynomial in «r»: «p(r) = 0»; find «p(r)»."
                ),
                answer_kind="expression",
                answer=machine(poly),
                answer_text=plain(poly),
                marks=2,
                working=[f"{join_terms([(1, 'r(r - 1)'), (a, 'r'), (b, '')])} = {plain(poly)}."],
                distractors=a_dist,
            ),
            dict(
                **prompt_part(b_prompt),
                answer_kind="number",
                answer=int(target),
                answer_text=str(int(target)),
                marks=1,
                working=[f"Solving p(r) = 0 gives {ask} = {int(target)}."],
                distractors=b_dist,
            ),
            dict(
                **prompt_part("Find the solution «y(x)» that meets the initial conditions."),
                answer_kind="expression",
                answer=machine(c.sol),
                answer_text=plain(c.sol),
                marks=3,
                working=[f"y = {plain(c.sol)}."],
                distractors=c_dist,
            ),
            dict(
                **prompt_part("Find «y({x1})», correct to 3 decimal places.", x1=x1_text),
                answer_kind="number",
                answer=round(value, 3),
                answer_text=f"{value:.3f}",
                marks=1,
                follow_through=f"Subs(part_c, (x,), ({x1_ft},))",
                working=[f"y({x1_text}) = {value:.3f}."],
                distractors=d_dist,
            ),
        ],
    )
