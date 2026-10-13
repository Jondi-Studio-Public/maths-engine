"""The classical fourth-order Runge-Kutta step: `r4b_rk4_step` (easy, medium, hard) and the exam chain
`exd_x_r4b_rk4_step`.

``uc2-rk4-step`` (``numerical-and-computational-mathematics``). The differential equation is ``y' = f(x, y)`` with
``f`` a polynomial in x and y; step sizes are 0.2, 0.25 or 0.5 so every slope is a terminating decimal.

* easy: ``f`` linear in x and y, one step, ``y₁`` to 5 decimal places.
* medium: ``f`` nonlinear (a product or a square), one step, ``y₁`` to 5 decimal places.
* hard: ``f`` nonlinear, two steps, ``y₂`` to 5 decimal places.
* exam: ``f`` linear, four parts: the first slope ``k₁``, the second slope ``k₂`` (follow-through from ``k₁``),
  ``y₁`` (follow-through from ``k₁`` and ``k₂``) and ``y₂`` (follow-through from ``y₁``).

All arithmetic is exact (`Fraction`); the answer is rounded once, at the end. A second route evaluates the step from
the Butcher tableau with SymPy rationals.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms
from ._r4b_kit import (
    Redraw,
    dec,
    margin_ok,
    number_distractors,
    number_part,
    sub,
    with_redraw,
)
from ._ucv1_texsay import say_marked

M_EULER = "num2-rk4-euler-step-used"
M_MIDPOINT = "num2-rk4-midpoint-slope-used"
M_EQUAL = "num2-rk4-weights-equal"
M_NO_H = "num2-rk4-step-size-dropped"
M_STALE = "num2-rk4-stage-slope-not-updated"
M_FULL_H = "num2-rk4-half-step-forgotten"
M_X_STUCK = "num2-rk4-stage-x-not-advanced"
M_ONE_STEP = "num2-rk4-stopped-after-one-step"
M_K_SLOPE2 = "num2-rk4-first-slope-reused"
M_K_NOHALF = "num2-rk4-second-slope-half-step-forgotten"
M_K_X = "num2-rk4-second-slope-x-not-advanced"
M_K_Y = "num2-rk4-second-slope-y-not-advanced"
M_K_FIRST_X = "num2-rk4-first-slope-at-the-midpoint"


class Eq:
    """y' = f(x, y) for f = sum of c * x^i * y^j."""

    def __init__(self, terms):
        self.terms = {k: Fraction(v) for k, v in terms.items() if v}

    def __call__(self, x, y):
        return sum(c * Fraction(x) ** i * Fraction(y) ** j for (i, j), c in self.terms.items())

    def text(self) -> str:
        parts = []
        for (i, j), c in sorted(self.terms.items(), key=lambda kv: (-(kv[0][0] + kv[0][1]), -kv[0][0])):
            var = ("x" + ("²" if i == 2 else "" if i == 1 else "") if i else "") + (
                "y" + ("²" if j == 2 else "") if j else ""
            )
            parts.append((int(c), var))
        return join_terms(parts).replace(" - ", " − ").replace("-", "−", 1) if parts else "0"

    def sym(self):
        x, y = sympy.symbols("x y")
        return sum((sympy.Rational(c.numerator, c.denominator) * x**i * y**j for (i, j), c in self.terms.items()), 0)


def rk4_step(f, x, y, h, variant=None):
    """One step; `variant` names a slip. Returns (y_next, (k1, k2, k3, k4))."""
    x, y, h = Fraction(x), Fraction(y), Fraction(h)
    xs = [x, x + h / 2, x + h / 2, x + h]
    if variant == "x_stuck":
        xs = [x, x, x, x]
    half = h if variant == "full_h" else h / 2
    k1 = f(xs[0], y)
    base = y if variant != "stale" else y
    k2 = f(xs[1], y + half * k1) if variant != "stale" else f(xs[1], y)
    k3 = f(xs[2], y + half * k2) if variant != "stale" else f(xs[2], y)
    k4 = f(xs[3], y + h * k3) if variant != "stale" else f(xs[3], y)
    if variant == "euler":
        return y + h * k1, (k1, k2, k3, k4)
    if variant == "midpoint":
        return y + h * k2, (k1, k2, k3, k4)
    if variant == "equal":
        return y + h * (k1 + k2 + k3 + k4) / 4, (k1, k2, k3, k4)
    if variant == "no_h":
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6, (k1, k2, k3, k4)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6, (k1, k2, k3, k4)


def rk4_by_tableau(eq: Eq, x, y, h):
    """The same step from the Butcher tableau, in SymPy rationals."""
    f = eq.sym()
    xs, ys = sympy.symbols("x y")
    rat = lambda v: sympy.Rational(Fraction(v).numerator, Fraction(v).denominator)  # noqa: E731
    x, y, h = rat(x), rat(y), rat(h)
    c = [0, sympy.Rational(1, 2), sympy.Rational(1, 2), 1]
    a = [[], [sympy.Rational(1, 2)], [0, sympy.Rational(1, 2)], [0, 0, 1]]
    b = [sympy.Rational(1, 6), sympy.Rational(1, 3), sympy.Rational(1, 3), sympy.Rational(1, 6)]
    ks = []
    for i in range(4):
        yi = y + h * sum(a[i][j] * ks[j] for j in range(i))
        ks.append(f.subs({xs: x + c[i] * h, ys: yi}))
    out = y + h * sum(b[i] * ks[i] for i in range(4))
    return Fraction(int(out.p), int(out.q))


def run(f, x, y, h, steps, variant=None):
    for _ in range(steps):
        y, _ks = rk4_step(f, x, y, h, variant)
        x += Fraction(h)
    return y


def _h(rng):
    return rng.choice([Fraction(1, 5), Fraction(1, 4), Fraction(1, 2)])


def _h_text(h: Fraction) -> str:
    return {Fraction(1, 5): "0.2", Fraction(1, 4): "0.25", Fraction(1, 2): "0.5"}[h]


def _dec_text(v) -> str:
    v = Fraction(v)
    for dp in range(0, 8):
        if (v * 10**dp).denominator == 1:
            return f"{float(v):.{dp}f}".replace("-", "−")
    return dec(v, 6)


_LINEAR = [
    ({(1, 0): 1, (0, 1): 1}),
    ({(1, 0): 2, (0, 1): -1}),
    ({(1, 0): -1, (0, 1): 2}),
    ({(1, 0): 1, (0, 1): -2}),
    ({(1, 0): 3, (0, 1): 1}),
    ({(1, 0): 1, (0, 1): 3}),
    ({(1, 0): 2, (0, 1): -3}),
    ({(1, 0): -2, (0, 1): -1}),
]
_NONLINEAR = [
    ({(1, 1): 1}),
    ({(2, 0): 1, (0, 1): 1}),
    ({(0, 2): 1, (1, 0): -1}),
    ({(1, 0): 1, (0, 2): 1}),
    ({(1, 1): 2}),
    ({(0, 2): -1, (1, 0): 2}),
    ({(1, 1): 1, (0, 1): -1}),
    ({(2, 0): 1, (0, 2): 1}),
]


def _setup(rng, tier):
    for _ in attempts(1000, "an RK4 draw"):
        terms = rng.choice(_LINEAR if tier == "easy" else _NONLINEAR)
        eq = Eq(terms)
        h = _h(rng)
        x0 = Fraction(rng.choice([0, 0, 1, -1, 2]))
        y0 = Fraction(rng.choice([-2, -1, 1, 2, 3] if tier == "easy" else [-1, 1, 2]))
        steps = 2 if tier == "hard" else 1
        y = y0
        x = x0
        ok = True
        for _s in range(steps):
            y, ks = rk4_step(eq, x, y, h)
            x += h
            if abs(y) > 30 or any(abs(k) > 200 for k in ks):
                ok = False
        if ok and margin_ok(y, 5, 0.12) and y != y0:
            return eq, x0, y0, h, steps
    raise AssertionError("unreachable")


def _ode_text(eq: Eq, x0, y0) -> str:
    return f"y′ = {eq.text()}, y({_dec_text(x0)}) = {_dec_text(y0)}"


@generator(
    "r4b_rk4_step",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4b_rk4_step(rng, tier="easy"):
    return with_redraw(lambda: _single(rng, tier), "an RK4 question with enough distractors")


def _single(rng, tier):
    eq, x0, y0, h, steps = _setup(rng, tier)
    value = run(eq, x0, y0, h, steps)
    check(value == _tableau_run(eq, x0, y0, h, steps), "the tableau route disagrees")
    cands = []
    for tag, variant in (
        (M_EULER, "euler"),
        (M_MIDPOINT, "midpoint"),
        (M_EQUAL, "equal"),
        (M_NO_H, "no_h"),
        (M_STALE, "stale"),
        (M_FULL_H, "full_h"),
        (M_X_STUCK, "x_stuck"),
    ):
        cands.append((tag, run(eq, x0, y0, h, steps, variant)))
    if steps == 2:
        cands.append((M_ONE_STEP, run(eq, x0, y0, h, 1)))
    x_end = x0 + steps * h
    last = "y₂" if steps == 2 else "y₁"
    ask = (
        f"Use two steps of the classical fourth-order Runge–Kutta method with step size «h = {_h_text(h)}» to approximate "
        f"«y({_dec_text(x_end)})». Give the answer to 5 decimal places."
        if steps == 2
        else f"Use one step of the classical fourth-order Runge–Kutta method with step size «h = {_h_text(h)}» to "
        f"approximate «y({_dec_text(x_end)})». Give the answer to 5 decimal places."
    )
    text = say_marked(f"A function «y» satisfies «{_ode_text(eq, x0, y0)}». " + ask)
    dist = number_distractors(value, 5, cands)
    if len(dist) < 3:
        raise Redraw
    part = number_part(
        "",
        value,
        5,
        cands,
        1,
        working=[
            "Find the four slopes k₁ to k₄, then y₁ = y₀ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)"
            + ("; repeat once more from (x₁, y₁)." if steps == 2 else "."),
            f"{last} ≈ {dec(value, 5)}.",
        ],
    )
    return dict(
        text=text[0],
        latex=text[1],
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_kind="number",
        distractors=dist,
        working=part["working"],
    )


def _tableau_run(eq, x0, y0, h, steps):
    x, y = Fraction(x0), Fraction(y0)
    for _ in range(steps):
        y = rk4_by_tableau(eq, x, y, h)
        x += Fraction(h)
    return y


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4b_rk4_step", level=22, presets={"exam": dict(tier="exam")})
def exd_x_r4b_rk4_step(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam RK4 question")


def _ft_step_expr(eq: Eq, x0, h, y_name: str):
    """A SymPy string for the RK4 step from (x0, y_name) in terms of the symbol `y_name`."""
    y = sympy.Symbol(y_name)
    f = eq.sym()
    xs, ys = sympy.symbols("x y")
    rat = lambda v: sympy.Rational(Fraction(v).numerator, Fraction(v).denominator)  # noqa: E731
    x0, h = rat(x0), rat(h)
    k1 = f.subs({xs: x0, ys: y})
    k2 = f.subs({xs: x0 + h / 2, ys: y + h / 2 * k1})
    k3 = f.subs({xs: x0 + h / 2, ys: y + h / 2 * k2})
    k4 = f.subs({xs: x0 + h, ys: y + h * k3})
    return sympy.expand(y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


def _exam(rng):
    for _ in attempts(1000, "an exam RK4 draw"):
        eq = Eq(rng.choice(_LINEAR))
        h = _h(rng)
        x0 = Fraction(rng.choice([0, 0, 1, 2]))
        y0 = Fraction(rng.choice([-2, -1, 1, 2, 3]))
        y1, (k1, k2, k3, k4) = rk4_step(eq, x0, y0, h)
        y2, _ = rk4_step(eq, x0 + h, y1, h)
        if k1 == k2 or not (margin_ok(y1, 5, 0.12) and margin_ok(y2, 5, 0.12) and abs(y2) < 200):
            continue
        if any(d > 4 for d in (_places(k1), _places(k2))):
            continue
        break
    check(y1 == rk4_by_tableau(eq, x0, y0, h), "the tableau route disagrees (y1)")
    check(y2 == rk4_by_tableau(eq, x0 + h, y1, h), "the tableau route disagrees (y2)")
    stem, latex = say_marked(
        f"A function «y» satisfies «{_ode_text(eq, x0, y0)}». It is approximated by the classical fourth-order "
        f"Runge–Kutta method with step size «h = {_h_text(h)}», giving «y₁» at «x₁ = {_dec_text(x0 + h)}» and «y₂» "
        f"at «x₂ = {_dec_text(x0 + 2 * h)}»."
    )
    xh = x0 + h / 2
    k1_dist = [
        (M_K_FIRST_X, eq(xh, y0)),
        ("num2-rk4-first-slope-at-wrong-point", eq(x0 + h, y0)),
        ("num2-rk4-initial-value-for-slope", y0),
        ("num2-rk4-first-slope-y-shifted", eq(x0, y0 + h)),
    ]
    p_a = number_part(
        "Find the first slope «k₁», the slope of the differential equation at the starting point.",
        k1,
        5,
        k1_dist,
        1,
        working=[f"k₁ = f({_dec_text(x0)}, {_dec_text(y0)}) = {_dec_text(k1)}."],
        value_range=(-1000, 1000),
    )
    ft_b = str(_ft_k2(eq, x0, y0, h))
    p_b = number_part(
        "Using your «k₁», find the second slope «k₂», the slope at the midpoint of the step.",
        k2,
        5,
        [
            (M_K_SLOPE2, k1),
            (M_K_NOHALF, eq(xh, y0 + h * k1)),
            (M_K_X, eq(x0, y0 + h / 2 * k1)),
            (M_K_Y, eq(xh, y0)),
        ],
        1,
        ft=ft_b,
        working=[f"k₂ = f({_dec_text(xh)}, {_dec_text(y0)} + {_dec_text(h / 2)}·{_dec_text(k1)}) = {_dec_text(k2)}."],
        value_range=(-1000, 1000),
    )
    ft_c = _ft_c(eq, x0, y0, h)
    cands_c = [
        (M_EULER, y0 + h * k1),
        (M_MIDPOINT, y0 + h * k2),
        (M_EQUAL, rk4_step(eq, x0, y0, h, "equal")[0]),
        (M_NO_H, rk4_step(eq, x0, y0, h, "no_h")[0]),
        (M_STALE, rk4_step(eq, x0, y0, h, "stale")[0]),
        (M_FULL_H, rk4_step(eq, x0, y0, h, "full_h")[0]),
    ]
    p_c = number_part(
        "Using your «k₁» and «k₂», find «y₁» to 5 decimal places.",
        y1,
        5,
        cands_c,
        2,
        ft=ft_c,
        working=[f"k₃ = {_dec_text(k3)} and k₄ = {_dec_text(k4)}, so y₁ = {dec(y1, 5)}."],
        value_range=(-1000, 1000),
    )
    cands_d = [
        (M_EULER, y1 + h * eq(x0 + h, y1)),
        (M_MIDPOINT, rk4_step(eq, x0 + h, y1, h, "midpoint")[0]),
        (M_EQUAL, rk4_step(eq, x0 + h, y1, h, "equal")[0]),
        (M_NO_H, rk4_step(eq, x0 + h, y1, h, "no_h")[0]),
        (M_ONE_STEP, y1),
        (M_X_STUCK, rk4_step(eq, x0 + h, y1, h, "x_stuck")[0]),
    ]
    p_d = number_part(
        "Using your «y₁», take a second step to find «y₂» to 5 decimal places.",
        y2,
        5,
        cands_d,
        2,
        ft=str(_ft_step_expr(eq, x0 + h, h, "part_c")),
        working=[f"The same four-slope step from (x₁, y₁) gives y₂ = {dec(y2, 5)}."],
        value_range=(-1000, 1000),
    )
    for part in (p_a, p_b, p_c, p_d):
        part["prompt"], part["prompt_latex"] = say_marked(part["prompt"])
    return dict(text=stem, latex=latex, parts=[p_a, p_b, p_c, p_d])


def _places(v: Fraction) -> int:
    for dp in range(0, 12):
        if (Fraction(v) * 10**dp).denominator == 1:
            return dp
    return 99


def _ft_k2(eq: Eq, x0, y0, h):
    f = eq.sym()
    xs, ys = sympy.symbols("x y")
    rat = lambda v: sympy.Rational(Fraction(v).numerator, Fraction(v).denominator)  # noqa: E731
    return sympy.expand(f.subs({xs: rat(x0) + rat(h) / 2, ys: rat(y0) + rat(h) / 2 * sympy.Symbol("part_a")}))


def _ft_c(eq: Eq, x0, y0, h):
    f = eq.sym()
    xs, ys = sympy.symbols("x y")
    rat = lambda v: sympy.Rational(Fraction(v).numerator, Fraction(v).denominator)  # noqa: E731
    k1, k2 = sympy.Symbol("part_a"), sympy.Symbol("part_b")
    x0, y0, h = rat(x0), rat(y0), rat(h)
    k3 = f.subs({xs: x0 + h / 2, ys: y0 + h / 2 * k2})
    k4 = f.subs({xs: x0 + h, ys: y0 + h * k3})
    return str(sympy.expand(y0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)))
