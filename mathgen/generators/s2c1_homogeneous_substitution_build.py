"""Generates `s2c1_homogeneous_substitution` questions (easy, medium, hard) and the exam chain
`exd_x_s2c1_homogeneous_substitution`.

``uc2-homogeneous-substitution`` (``ordinary-differential-equations``, answer ``expression``): first-order equations
``dy/dx = F(y/x)`` solved by the substitution ``y = v x`` (so ``dy/dx = v + x dv/dx``), separated and integrated, with
an initial condition fixing the constant. The stem never names the substitution.

* easy: ``dy/dx = y/x + k (y/x)^2``, solved by ``x dv/dx = k v^2``.
* medium: ``dy/dx = (y/x)^2 + m (y/x)`` (partial fractions) or ``dy/dx = (a x^2 + 2 y^2)/(x y)``.
* hard: ``x dy/dx = y + sqrt(x^2 + y^2)``, ``dy/dx = (a x^2 + 3 y^2)/(x y)``, ``dy/dx = y/x + tan(y/x)`` and
  ``dy/dx = y/x + sin(y/x)`` (closed forms with arcsin and arctan).
* exam (``exd_x_s2c1_homogeneous_substitution``): ``dy/dx = (y/x)^2 + m (y/x)`` with ``y(1) = y0``: (a) x dv/dx as a
  function of v, (b) the constant K in v/(v + m - 1) = K x^(m - 1), (c) y(x) (follows through from (b)), (d) y at a
  stated x (follows through from (c)).

Every solution is checked by substituting it into the differential equation numerically at three points near the
initial point, and against the initial condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from ..answers import Distractor, exact_value
from ._s2_options import spot_check
from ..core import attempts, check, generator, pick_distractors, rational_text, rounds_unambiguously
from ._s2c1_kit import (
    X,
    expr_options,
    fraction_options,
    machine,
    neat,
    number_options,
    plain,
    prompt_part,
    agree,
    same_function,
    say_marked,
)

x = X
y = sympy.Symbol("y")
v = sympy.Symbol("v")
_R = sympy.Rational


@dataclass(frozen=True)
class _Eq:
    rhs: sympy.Expr  # dy/dx = rhs(x, y)
    lhs_text: str  # "dy/dx" or "x dy/dx"
    rhs_text: str
    x0: sympy.Rational
    y0: sympy.Rational
    sol: sympy.Expr
    wrongs: tuple  # (misconception id, expression)
    y0_text: str = ""


def _frac(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def _checked(eq: _Eq) -> _Eq:
    sol = eq.sol
    if not spot_check(str(sol), str(eq.rhs)):  # the substitution is the slow second route: one question in four
        return eq
    for k in range(1, 4):
        p = eq.x0 + _R(k, 10)
        got = float(sympy.diff(sol, x).subs(x, p).evalf(30))
        want = float(eq.rhs.subs({x: p, y: sol.subs(x, p)}).evalf(30))
        check(abs(got - want) < 1e-9 * max(1.0, abs(got)), "the solution does not satisfy the differential equation")
    check(abs(float((sol.subs(x, eq.x0) - eq.y0).evalf(30))) < 1e-12, "the initial condition is not met")
    return eq


def _riccati(rng) -> _Eq:
    """dy/dx = y/x + k (y/x)^2, y(1) = y0: y = x y0/(1 - k y0 ln x)."""
    for _ in attempts(100, "a quadratic homogeneous equation"):
        k = rng.choice([-3, -2, -1, 1, 2, 3])
        y0 = _R(rng.choice([1, 2, 3, -1, -2, _R(1, 2), _R(1, 3)]))
        sol = x * y0 / (1 - k * y0 * sympy.log(x))
        wrongs = (
            ("homogeneous-sign-in-separation", x * y0 / (1 + k * y0 * sympy.log(x))),
            ("homogeneous-forgot-back-substitution", y0 / (1 - k * y0 * sympy.log(x))),
            ("homogeneous-reciprocal-slip", x * (1 - k * y0 * sympy.log(x)) / y0),
            ("homogeneous-coefficient-dropped", x * y0 / (1 - y0 * sympy.log(x))),
        )
        rhs = y / x + k * y**2 / x**2
        return _Eq(rhs, "dy/dx", plain(rhs), _R(1), sympy.Rational(y0), sol, wrongs)
    raise AssertionError("unreachable")


def _quadratic_in_v(rng, m: int | None = None) -> _Eq:
    """dy/dx = (y/x)^2 + m (y/x), y(1) = y0: v/(v + m - 1) = K x^(m-1)."""
    m = m or rng.choice([2, 3, 4])
    y0 = _R(rng.choice([1, 2, 3, 4, _R(1, 2), _R(1, 3), _R(1, 4), _R(2, 3), _R(3, 2)]))
    K = y0 / (y0 + m - 1)
    sol = (m - 1) * K * x**m / (1 - K * x ** (m - 1))
    wrongs = [
        ("homogeneous-forgot-back-substitution", sol / x),
        ("homogeneous-constant-error", (m - 1) * (1 / K) * x**m / (1 - x ** (m - 1) / K)),
        ("homogeneous-dropped-product-rule", _dropped_product(m, y0)),
    ]
    if y0 != m - 1:
        K_bad = y0 / (y0 - (m - 1))
        wrongs.insert(2, ("homogeneous-sign-in-partial-fractions", (m - 1) * K_bad * x**m / (1 - K_bad * x ** (m - 1))))
    wrongs = tuple(wrongs)
    rhs = y**2 / x**2 + m * y / x
    return _Eq(rhs, "dy/dx", plain(rhs), _R(1), sympy.Rational(y0), sol, wrongs)


def _dropped_product(m: int, y0):
    """The solution of x v' = v^2 + m v (the product rule forgotten), which is wrong: v/(v + m) = K x^m."""
    K = y0 / (y0 + m)
    return m * K * x ** (m + 1) / (1 - K * x**m)


def _sqrt_family(rng, b: int) -> _Eq:
    """dy/dx = (a x^2 + b y^2)/(x y), y(1) = y0 > 0: y^2 = x^2 (K x^(2(b-1)) - a)/(b-1), K = (b-1) y0^2 + a."""
    for _ in attempts(100, "a square-root homogeneous equation"):
        a = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
        y0 = _R(rng.choice([1, 2, 3, 4, _R(1, 2), _R(3, 2), _R(5, 2), _R(1, 3)]))
        K = (b - 1) * y0**2 + a
        if K <= 0:
            continue
        inner = (K * x ** (2 * (b - 1)) - a) / (b - 1)
        sol = x * sympy.sqrt(inner)
        K_bad = y0**2 - a
        wrongs = (
            ("homogeneous-forgot-back-substitution", sympy.sqrt(inner)),
            ("homogeneous-constant-error", x * sympy.sqrt((K_bad * x ** (2 * (b - 1)) - a) / (b - 1))),
            ("homogeneous-constant-error", x * sympy.sqrt((y0**2 * (b - 1) * x ** (2 * (b - 1)) - a) / (b - 1))),
            ("homogeneous-sign-in-separation", x * sympy.sqrt((K * x ** (2 * (b - 1)) + a) / (b - 1))),
        )
        rhs = (a * x**2 + b * y**2) / (x * y)
        if min(float(inner.subs(x, _R(1) + _R(k, 10)).evalf()) for k in range(0, 4)) <= 0.05:
            continue
        return _Eq(rhs, "dy/dx", plain(rhs), _R(1), sympy.Rational(y0), sol, wrongs)
    raise AssertionError("unreachable")


def _sinh_family(rng) -> _Eq:
    """x dy/dx = y + sqrt(x^2 + y^2), y(1) = y0: y = (K x^2 - 1/K)/2 with y0 = (K - 1/K)/2."""
    K = _R(rng.randint(6, 19), rng.choice([2, 3, 4, 5])) if rng.random() < 0.6 else _R(rng.randint(2, 9))
    y0 = (K - 1 / K) / 2
    sol = (K * x**2 - 1 / (K)) / 2
    wrongs = (
        ("homogeneous-sign-in-separation", (K * x**2 + 1 / K) / 2),
        ("homogeneous-constant-error", (x**2 / K - K) / 2),
        ("homogeneous-forgot-back-substitution", (K * x - 1 / (K * x)) / 2),
        ("homogeneous-constant-error", (2 * y0 * x**2 - 1 / (2 * y0)) / 2),
    )
    rhs = (y + sympy.sqrt(x**2 + y**2)) / x
    return _Eq(rhs, "dy/dx", plain(rhs), _R(1), sympy.Rational(y0), sol, wrongs)


def _trig_family(rng) -> _Eq:
    x0 = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])
    if rng.random() < 0.5:  # dy/dx = y/x + tan(y/x): sin(y/x) = K x, y(x0) = x0 pi/6
        K = _R(1, 2 * x0)
        sol = x * sympy.asin(K * x)
        y0 = x0 * sympy.pi / 6
        wrongs = (
            ("homogeneous-forgot-back-substitution", sympy.asin(K * x)),
            ("homogeneous-constant-error", x * sympy.asin(x / (3 * x0))),
            ("homogeneous-sign-in-separation", x * sympy.acos(K * x)),
            ("homogeneous-coefficient-dropped", x * sympy.asin(K * x) / 2),
        )
        rhs = y / x + sympy.tan(y / x)
    else:  # dy/dx = y/x + sin(y/x): tan(y/(2x)) = K x, y(x0) = x0 pi/2
        K = _R(1, x0)
        sol = 2 * x * sympy.atan(K * x)
        y0 = x0 * sympy.pi / 2
        wrongs = (
            ("homogeneous-forgot-back-substitution", 2 * sympy.atan(K * x)),
            ("homogeneous-coefficient-dropped", x * sympy.atan(K * x)),
            ("homogeneous-constant-error", 2 * x * sympy.atan(x / (2 * x0))),
            ("homogeneous-reciprocal-slip", 2 * x * sympy.atan(1 / (K * x))),
        )
        rhs = y / x + sympy.sin(y / x)
    return _Eq(rhs, "dy/dx", plain(rhs), _R(x0), y0, sol, wrongs)


def _draw(rng, tier: str) -> _Eq:
    if tier == "easy":
        return _checked(_riccati(rng))
    if tier == "medium":
        return _checked(_quadratic_in_v(rng) if rng.random() < 0.5 else _sqrt_family(rng, 2))
    pick = rng.random()
    if pick < 0.3:
        return _checked(_sinh_family(rng))
    if pick < 0.55:
        return _checked(_sqrt_family(rng, 3))
    return _checked(_trig_family(rng))


def _options(answer, wrongs, x0):
    pts = tuple(x0 + _R(k, 10) for k in range(1, 4))
    kept = []
    for tag, w in wrongs:
        w = sympy.sympify(w)
        if w.has(sympy.zoo, sympy.nan, sympy.oo, sympy.I):
            continue
        w = neat(w)
        try:
            if any(abs(complex(w.subs(x, p).evalf(30)).imag) > 1e-9 for p in pts):
                continue
            if same_function(w, answer, x, pts) or any(same_function(w, o, x, pts) for _, o in kept):
                continue
        except Exception:  # noqa: BLE001
            continue
        kept.append((tag, w))
    return pick_distractors(plain(answer), [Distractor(plain(w), tag, machine(w)) for tag, w in kept], 4)


def _pi_text(value) -> str:
    return plain(value).replace("pi", "π")


@generator(
    "s2c1_homogeneous_substitution",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c1_homogeneous_substitution(rng, tier="easy"):
    eq = _draw(rng, tier)
    sol = neat(eq.sol)
    check(agree(sol, eq.sol, x, tuple(eq.x0 + _R(k, 10) for k in range(1, 4))), "neat form differs")
    ic = f"y({eq.x0}) = {_pi_text(eq.y0)}" if eq.y0.has(sympy.pi) else f"y({eq.x0}) = {_frac(eq.y0)}"
    stem, stem_latex = say_marked(
        rng.choice(
            [
                "Solve the initial value problem «{lhs} = {rhs}, {ic}» for «x > 0».",
                "Find «y(x)» for «x > 0» given «{lhs} = {rhs}» and «{ic}».",
            ]
        ).format(lhs=eq.lhs_text, rhs=eq.rhs_text, ic=ic)
    )
    dist = _options(sol, eq.wrongs, eq.x0)
    check(len(dist) >= 2, "too few homogeneous-equation distractors")
    return dict(
        text=stem,
        latex=stem_latex,
        answer=machine(sol),
        answer_text=plain(sol),
        answer_kind="expression",
        distractors=dist,
        working=[
            "The right-hand side is a function of y/x, so put y = vx and dy/dx = v + x dv/dx, then separate and "
            f"integrate. The initial condition fixes the constant, giving y = {plain(sol)}."
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


@generator("exd_x_s2c1_homogeneous_substitution", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c1_homogeneous_substitution(rng, tier="exam"):
    """Exam tier: a homogeneous equation solved by y = vx in four connected parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam homogeneous-equation question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    m = rng.choice([2, 3, 4])
    eq = _checked(_quadratic_in_v(rng, m))
    y0 = eq.y0
    K = y0 / (y0 + m - 1)
    sol = neat((m - 1) * K * x**m / (1 - K * x ** (m - 1)))
    x1 = rng.choice([_R(11, 10), _R(6, 5), _R(5, 4), _R(3, 2)])
    denom = 1 - K * x1 ** (m - 1)
    if denom <= _R(1, 8):
        raise _Redraw
    value = float(sol.subs(x, x1).evalf(30))
    if not (0.01 <= abs(value) <= 400 and rounds_unambiguously(value, 3, 0.1)):
        raise _Redraw

    g = v**2 + (m - 1) * v
    a_wrongs = [
        ("homogeneous-dropped-product-rule", v**2 + m * v),
        ("homogeneous-divided-by-x", g / x),
        ("homogeneous-sign-in-separation", v**2 - (m - 1) * v),
        ("homogeneous-wrong-term", v + (m - 1) * v**2),
    ]
    a_dist = expr_options(g, a_wrongs, 4, var=v)
    b_wrongs = [
        ("homogeneous-constant-error", 1 / K),
        ("homogeneous-sign-in-partial-fractions", y0 / (y0 - (m - 1))),
        ("homogeneous-dropped-product-rule", y0 / (y0 + m)),
        ("homogeneous-constant-error", y0 / (m - 1)),
    ]
    b_dist = fraction_options(K, b_wrongs, 4)
    c_dist = _options(sol, eq.wrongs, _R(1))
    d_wrongs = []
    for d in c_dist:
        try:
            d_wrongs.append((d.misconception, float(sympy.sympify(d.value).subs(x, x1).evalf(30))))
        except (TypeError, ValueError):
            continue
    d_dist = number_options(value, 3, d_wrongs, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    x1_text = f"{float(x1):g}"
    x1_ft = f"{x1.p}/{x1.q}"
    rhs = y**2 / x**2 + m * y / x
    stem, stem_latex = say_marked(
        f"Consider the initial value problem «dy/dx = {plain(rhs)}, y(1) = {_frac(y0)}» for «x > 0»."
    )
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt_part(
                    "The equation depends on «y/x» only. Substitute «y = vx» and write the equation as «x dv/dx = f(v)». "
                    "Find «f(v)»."
                ),
                answer_kind="expression",
                answer=machine(g),
                answer_text=plain(g),
                marks=2,
                working=[f"v + x dv/dx = v^2 + {m}v, so x dv/dx = {plain(g)}."],
                distractors=a_dist,
            ),
            dict(
                **prompt_part(
                    "Separating variables and integrating gives «v/(v + {n}) = K{xp}». Use «y(1) = {y0}» to find «K».".replace(
                        "{n}", str(m - 1)
                    )
                    .replace("{xp}", " x" if m == 2 else f" x^{m - 1}")
                    .replace("{y0}", _frac(y0))
                ),
                answer_kind="fraction",
                answer=exact_value(K),
                answer_text=rational_text(K),
                marks=2,
                working=[f"At x = 1, v = {_frac(y0)}, so K = {rational_text(K)}."],
                distractors=b_dist,
            ),
            dict(
                **prompt_part("Hence find «y(x)»."),
                answer_kind="expression",
                answer=machine(sol),
                answer_text=plain(sol),
                marks=2,
                follow_through=f"{m - 1}*part_b*x**{m}/(1 - part_b*x**{m - 1})",
                working=[
                    f"y = vx with v = {plain((m - 1) * K * x ** (m - 1) / (1 - K * x ** (m - 1)))}, so y = {plain(sol)}."
                ],
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
