"""The improved Euler (Heun) step: `r4h_improved_euler_step` (easy, medium, hard) and the exam chain
`exd_x_r4h_improved_euler_step`.

``uc2-improved-euler-step`` (``numerical-and-computational-mathematics``). For ``y' = f(x, y)`` the step is a predictor
``y* = yₙ + h f(xₙ, yₙ)`` followed by the corrector ``yₙ₊₁ = yₙ + (h/2)(f(xₙ, yₙ) + f(xₙ₊₁, y*))``.

* easy: ``f`` linear in x and y, one step, to 5 decimal places.
* medium: ``f`` nonlinear (a product or a square), one step, to 5 decimal places.
* hard: ``f`` nonlinear, two steps, to 5 decimal places.
* exam: four parts: the first slope ``k₁``, the predictor ``y*`` (follow-through from ``k₁``), the slope at the
  predicted point ``k₂`` (follow-through from ``y*``) and ``y₁`` (follow-through from ``k₁`` and ``k₂``).

All arithmetic is exact (`Fraction`); the answer is rounded once, at the end.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms
from ._r4h_kit import (
    Redraw,
    dec,
    finish_parts,
    ft_symbol,
    margin_ok,
    number_distractors,
    number_part,
    rat,
    single_from,
    with_redraw,
)

MISC = {
    "num2t-heun-euler-step-only": "Stopping at the predictor y + h f(x, y) and reporting it as the next value.",
    "num2t-heun-corrector-slope-at-start": "Evaluating the second slope at the starting y-value instead of at the predicted value.",
    "num2t-heun-average-not-halved": "Adding the two slopes without halving them, y + h(k₁ + k₂).",
    "num2t-heun-end-slope-only": "Stepping with the slope at the predicted end point alone, y + h k₂.",
    "num2t-heun-midpoint-method": "Using the slope at the midpoint (the midpoint method) in place of the average of the end slopes.",
    "num2t-heun-second-slope-x-not-advanced": "Evaluating the second slope at the starting x instead of x + h.",
    "num2t-heun-stopped-after-one-step": "Reporting the value after the first step when two steps were asked for.",
    "num2t-heun-slope-step-size-dropped": "Adding the average slope to y without multiplying by the step size.",
    "num2t-heun-predictor-half-step": "Predicting with a half step, y + (h/2)k₁, instead of a full step.",
    "num2t-heun-first-slope-at-end": "Evaluating the first slope at x + h instead of at the starting point.",
    "num2t-heun-first-slope-initial-value": "Giving the initial y-value where the first slope is asked for.",
    "num2t-heun-predictor-slope-not-multiplied": "Giving the slope k₁ where the predicted value is asked for, without stepping.",
    "num2t-heun-predictor-uses-y-only": "Predicting the next value as y + h, forgetting to multiply the step by the slope.",
    "num2t-heun-second-slope-at-original-y": "Evaluating the second slope at the original y-value instead of the predicted value.",
    "num2t-heun-second-slope-copies-first": "Repeating the first slope as the second slope.",
    "num2t-heun-second-slope-corrected-value": "Evaluating the second slope at the new corrected value of y rather than the predicted one.",
}
M_EULER = "num2t-heun-euler-step-only"
M_STALE = "num2t-heun-corrector-slope-at-start"
M_NOHALF = "num2t-heun-average-not-halved"
M_END = "num2t-heun-end-slope-only"
M_MID = "num2t-heun-midpoint-method"
M_XSTUCK = "num2t-heun-second-slope-x-not-advanced"
M_ONE = "num2t-heun-stopped-after-one-step"
M_NOH = "num2t-heun-slope-step-size-dropped"
M_HALF_PRED = "num2t-heun-predictor-half-step"


class Eq:
    """y' = f(x, y) for f = sum of c * x^i * y^j."""

    def __init__(self, terms):
        self.terms = {k: Fraction(v) for k, v in terms.items() if v}

    def __call__(self, x, y):
        return sum(c * Fraction(x) ** i * Fraction(y) ** j for (i, j), c in self.terms.items())

    def text(self) -> str:
        parts = []
        for (i, j), c in sorted(self.terms.items(), key=lambda kv: (-(kv[0][0] + kv[0][1]), -kv[0][0])):
            var = ("x" + ("²" if i == 2 else "") if i else "") + ("y" + ("²" if j == 2 else "") if j else "")
            parts.append((int(c), var))
        return join_terms(parts).replace(" - ", " − ").replace("-", "−", 1) if parts else "0"

    def sym(self):
        x, y = sympy.symbols("x y")
        return sum((sympy.Rational(c.numerator, c.denominator) * x**i * y**j for (i, j), c in self.terms.items()), 0)


def heun_step(f, x, y, h, variant=None):
    """One step; `variant` names a slip. Returns (y_next, k1, predicted y, k2)."""
    x, y, h = Fraction(x), Fraction(y), Fraction(h)
    k1 = f(x, y)
    pred = y + h * k1
    if variant == "half_pred":
        pred = y + h / 2 * k1
    k2 = f(x + h, pred) if variant != "xstuck" else f(x, pred)
    if variant == "stale":
        k2 = f(x + h, y)
    if variant == "euler":
        return y + h * k1, k1, pred, k2
    if variant == "nohalf":
        return y + h * (k1 + k2), k1, pred, k2
    if variant == "end":
        return y + h * k2, k1, pred, k2
    if variant == "mid":
        return y + h * f(x + h / 2, y + h / 2 * k1), k1, pred, k2
    if variant == "noh":
        return y + (k1 + k2) / 2, k1, pred, k2
    return y + h / 2 * (k1 + k2), k1, pred, k2


def run(f, x, y, h, steps, variant=None):
    for _ in range(steps):
        y = heun_step(f, x, y, h, variant)[0]
        x += Fraction(h)
    return y


_H = {Fraction(1, 10): "0.1", Fraction(1, 5): "0.2", Fraction(1, 4): "0.25", Fraction(1, 2): "0.5"}


def _dec_text(v) -> str:
    v = Fraction(v)
    for dp in range(0, 8):
        if (v * 10**dp).denominator == 1:
            return f"{float(v):.{dp}f}".replace("-", "−")
    return dec(v, 6)


_LINEAR = [
    {(1, 0): 1, (0, 1): 1},
    {(1, 0): 2, (0, 1): -1},
    {(1, 0): -1, (0, 1): 2},
    {(1, 0): 1, (0, 1): -2},
    {(1, 0): 3, (0, 1): 1},
    {(1, 0): 2, (0, 1): -3},
    {(1, 0): -2, (0, 1): -1},
]
_NONLINEAR = [
    {(1, 1): 1},
    {(2, 0): 1, (0, 1): 1},
    {(0, 2): 1, (1, 0): -1},
    {(1, 0): 1, (0, 2): 1},
    {(1, 1): 2},
    {(0, 2): -1, (1, 0): 2},
    {(1, 1): 1, (0, 1): -1},
    {(2, 0): 1, (0, 2): 1},
]


def _setup(rng, tier):
    for _ in attempts(1000, "a Heun draw"):
        eq = Eq(rng.choice(_LINEAR if tier == "easy" else _NONLINEAR))
        h = rng.choice(list(_H))
        x0 = Fraction(rng.choice([0, 0, 1, -1, 2]))
        y0 = Fraction(rng.choice([-2, -1, 1, 2, 3] if tier == "easy" else [-1, 1, 2]))
        steps = 2 if tier == "hard" else 1
        x, y, ok = x0, y0, True
        for _s in range(steps):
            y, k1, _p, k2 = heun_step(eq, x, y, h)
            x += h
            if abs(y) > 30 or abs(k1) > 200 or abs(k2) > 200:
                ok = False
        if ok and margin_ok(y, 5, 0.12) and y != y0 and len(set(_slip_values(eq, x0, y0, h, steps))) >= 5:
            return eq, x0, y0, h, steps
    raise AssertionError("unreachable")


def _slip_values(eq, x0, y0, h, steps):
    return [
        round(float(v), 5)
        for v in (run(eq, x0, y0, h, steps, k) for k in (None, "euler", "stale", "nohalf", "end", "mid"))
    ]


def _ode_text(eq: Eq, x0, y0) -> str:
    return f"y′ = {eq.text()}, y({_dec_text(x0)}) = {_dec_text(y0)}"


@generator(
    "r4h_improved_euler_step",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4h_improved_euler_step(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return with_redraw(lambda: _single(rng, tier), "a Heun question with enough distractors")


def _single(rng, tier):
    eq, x0, y0, h, steps = _setup(rng, tier)
    value = run(eq, x0, y0, h, steps)
    cands = [
        (M_EULER, run(eq, x0, y0, h, steps, "euler")),
        (M_STALE, run(eq, x0, y0, h, steps, "stale")),
        (M_NOHALF, run(eq, x0, y0, h, steps, "nohalf")),
        (M_END, run(eq, x0, y0, h, steps, "end")),
        (M_MID, run(eq, x0, y0, h, steps, "mid")),
        (M_XSTUCK, run(eq, x0, y0, h, steps, "xstuck")),
    ]
    if steps == 2:
        cands.insert(0, (M_ONE, run(eq, x0, y0, h, 1)))
    x_end = x0 + steps * h
    count = "two steps" if steps == 2 else "one step"
    ode, hh, ye = _ode_text(eq, x0, y0), _H[h], _dec_text(x_end)
    stem = rng.choice(
        [
            f"A function «y» satisfies «{ode}». Use {count} of the improved Euler method (Heun's method) with step size "
            f"«h = {hh}» to approximate «y({ye})». Give the answer to 5 decimal places.",
            f"Given «{ode}», approximate «y({ye})» using {count} of Heun's method (the improved Euler method) with "
            f"«h = {hh}». Round the result to 5 decimal places.",
            f"The initial value problem «{ode}» is solved numerically with {count} of the predictor-corrector "
            f"method known as improved Euler, step size «{hh}». What value does it give for «y({ye})»? Answer to 5 "
            "decimal places.",
        ]
    )
    part = number_part("", value, 5, cands, 1, n=4, min_options=3)
    return single_from(
        part,
        stem,
        working=[
            "Predict y* = y + h f(x, y), then correct: y_new = y + (h/2)(f(x, y) + f(x + h, y*))"
            + ("; repeat from the new point." if steps == 2 else "."),
            f"The result is {dec(value, 5)}.",
        ],
    )


@generator("exd_x_r4h_improved_euler_step", level=22, presets={"exam": dict(tier="exam")})
def exd_x_r4h_improved_euler_step(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam Heun question")


def _exam(rng):
    for _ in attempts(1000, "an exam Heun draw"):
        eq = Eq(rng.choice(_LINEAR + _NONLINEAR[:3]))
        h = rng.choice(list(_H))
        x0 = Fraction(rng.choice([0, 0, 1, 2]))
        y0 = Fraction(rng.choice([-2, -1, 1, 2, 3]))
        y1, k1, pred, k2 = heun_step(eq, x0, y0, h)
        if k1 != k2 and margin_ok(y1, 5, 0.12) and abs(y1) < 100 and abs(k2) < 300 and k1 != 0:
            break
    f = eq.sym()
    xs, ys = sympy.symbols("x y")
    x1 = x0 + h
    p_a = number_part(
        "Find the slope «k₁ = f(x₀, y₀)» at the starting point.",
        k1,
        5,
        [
            ("num2t-heun-first-slope-at-end", eq(x1, y0)),
            ("num2t-heun-first-slope-initial-value", y0),
            ("num2t-heun-first-slope-at-end", eq(x0, y0 + h)),
            ("num2t-heun-first-slope-at-end", eq(x0 + h / 2, y0)),
        ],
        1,
        working=[f"k₁ = f({_dec_text(x0)}, {_dec_text(y0)}) = {_dec_text(k1)}."],
        value_range=(-1000, 1000),
    )
    a, c = ft_symbol("part_a"), ft_symbol("part_c")
    p_b = number_part(
        "Using your «k₁», find the predicted value «y*» of the method.",
        pred,
        5,
        [
            ("num2t-heun-predictor-slope-not-multiplied", k1),
            ("num2t-heun-predictor-uses-y-only", y0 + h),
            (M_HALF_PRED, y0 + h / 2 * k1),
            ("num2t-heun-predictor-uses-y-only", y0 - h * k1),
        ],
        1,
        ft=str(rat(y0) + rat(h) * a),
        working=[f"y* = {_dec_text(y0)} + {_dec_text(h)}·{_dec_text(k1)} = {_dec_text(pred)}."],
        value_range=(-1000, 1000),
    )
    b = ft_symbol("part_b")
    p_c = number_part(
        f"Using your «y*», find the slope «k₂ = f(x₁, y*)» at «x₁ = {_dec_text(x1)}».",
        k2,
        5,
        [
            ("num2t-heun-second-slope-at-original-y", eq(x1, y0)),
            (M_XSTUCK, eq(x0, pred)),
            ("num2t-heun-second-slope-copies-first", k1),
            ("num2t-heun-second-slope-corrected-value", eq(x1, y1)),
        ],
        1,
        ft=str(f.subs({xs: rat(x1), ys: b})),
        working=[f"k₂ = f({_dec_text(x1)}, {_dec_text(pred)}) = {_dec_text(k2)}."],
        value_range=(-1000, 1000),
    )
    p_d = number_part(
        "Using your slopes, find the improved Euler value «y₁» to 5 decimal places.",
        y1,
        5,
        [
            (M_EULER, pred),
            (M_NOHALF, y0 + h * (k1 + k2)),
            (M_END, y0 + h * k2),
            (M_NOH, y0 + (k1 + k2) / 2),
            (M_MID, heun_step(eq, x0, y0, h, "mid")[0]),
        ],
        2,
        ft=str(rat(y0) + rat(h) / 2 * (a + c)),
        working=[f"y₁ = {_dec_text(y0)} + ({_dec_text(h)}/2)({_dec_text(k1)} + {_dec_text(k2)}) = {dec(y1, 5)}."],
        value_range=(-1000, 1000),
    )
    ode, hh = _ode_text(eq, x0, y0), _H[h]
    stem = rng.choice(
        [
            f"A function «y» satisfies «{ode}». One step of the improved Euler method (Heun's method) with step size "
            f"«h = {hh}» is taken from «x₀ = {_dec_text(x0)}» to «x₁ = {_dec_text(x1)}».",
            f"For «{ode}», the improved Euler (Heun) method with «h = {hh}» advances the solution from "
            f"«x₀ = {_dec_text(x0)}» to «x₁ = {_dec_text(x1)}» in one step.",
            f"The problem «{ode}» is stepped once, by Heun's method, with «h = {hh}», from «x₀ = {_dec_text(x0)}» "
            f"to «x₁ = {_dec_text(x1)}».",
        ]
    )
    return finish_parts([p_a, p_b, p_c, p_d], stem)
