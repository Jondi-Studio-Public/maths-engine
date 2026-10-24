"""Generates `fy1ca_average_value_mvt` and `fy1ca_average_value_mvt_exam` questions.

One first-year skill, ``ucv1-average-value-mvt``: the average value of a function over an interval,
f_avg = (1/(b - a)) ∫ f(x) dx, and the c-values given by the mean value theorem and Rolle's theorem. Every key is an
exact value (a rational, or a multiple of π, e or a logarithm) graded as an expression.

Tiers (the reasoning escalates):

- ``easy``: an average value from a direct antiderivative: a polynomial, sin x, cos x or e^x over a stated interval.
- ``medium``: the c of Rolle's theorem (a cubic that vanishes at both ends) or of the mean value theorem for
  derivatives (√x or 1/x), where f'(c) must be set equal to a slope and solved.
- ``hard``: an average value that needs a substitution (x e^(x²), x√(x² + k), 1/(1 + x²)), and the c for which
  f(c) equals the average value (the mean value theorem for integrals).
- ``exam`` (topic ``fy1ca_average_value_mvt_exam``): a four-part chain on a storm inflow rate f(t) = √t kL/h: the total
  inflow, the average rate, the time at which the rate equals the average, and how far the peak exceeds the average.

Every key is re-derived by a second route: Simpson's rule gives the average value, and central differences check the
slope condition that defines each c.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy
from sympy import E, Rational, exp, log, pi, sqrt

from ..core import check, generator, latex_answer, phrase
from ._fy1ca_kit import (
    definite,
    lambdified,
    vanishes,
    Redraw,
    build_with_redraws,
    clear_of_rounding,
    close,
    expression_distractors,
    fraction_distractors,
    fraction_part,
    need_distractors,
    number_distractors,
    number_part,
    simpson,
    typed,
)

x = sympy.Symbol("x", positive=True)


def _show(value) -> str:
    text = typed(sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value)
    return text.replace("*pi", "π").replace("pi", "π").replace("*", " × ")


def _nice(candidates):
    kept = []
    for value, tag in candidates:
        if value is None:
            continue
        value = sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value
        if value.is_Rational and value.q > 400:
            continue
        if len(typed(value)) > 22:
            continue
        kept.append((value, tag))
    return kept


def _question(text, answer, wrong, working) -> dict:
    expr = sympy.nsimplify(answer) if not isinstance(answer, sympy.Basic) else answer
    distractors = expression_distractors(expr, _nice(wrong))
    need_distractors(distractors)
    return dict(
        text=text,
        answer=str(expr),
        answer_text=typed(expr),
        answer_kind="expression",
        answer_latex=latex_answer(expr),
        working=working,
        distractors=distractors,
    )


def _average(f, a, b):
    return sympy.simplify(definite(f, x, a, b) / (b - a))


def _numeric_average(f, a, b) -> float:
    g = lambdified(x, f)
    return simpson(g, float(a), float(b), 2400) / float(b - a)


def _average_wrong(f, a, b, avg):
    total = definite(f, x, a, b)
    return [
        (total, "fy1ca-avg-division-forgotten"),
        (total / b, "fy1ca-avg-divide-by-upper-limit"),
        ((f.subs(x, a) + f.subs(x, b)) / sympy.Integer(2), "fy1ca-avg-endpoint-mean"),
        (f.subs(x, (sympy.sympify(a) + sympy.sympify(b)) / 2), "fy1ca-avg-midpoint-value"),
        (total * (b - a), "fy1ca-avg-multiplied"),
        (definite(f, x, 0, b) / (b - a) if a != 0 else None, "fy1ca-avg-lower-limit-ignored"),
    ]


# -- easy ------------------------------------------------------------------------------------------------------------

# (f, text, a, b, a_text, b_text)
_EASY = (
    (x**2, "x²", 0, 3, "0", "3"),
    (x**2, "x²", 1, 4, "1", "4"),
    (3 * x**2 + 2 * x, "3x² + 2x", 1, 3, "1", "3"),
    (4 * x**3, "4x³", 0, 2, "0", "2"),
    (x**3 - x, "x³ - x", 0, 2, "0", "2"),
    (6 * x - x**2, "6x - x²", 0, 6, "0", "6"),
    (sympy.sin(x), "sin x", 0, pi, "0", "π"),
    (sympy.cos(x), "cos x", 0, pi / 2, "0", "π/2"),
    (sympy.sin(x), "sin x", 0, pi / 2, "0", "π/2"),
    (exp(x), "e^x", 0, log(3), "0", "ln 3"),
    (exp(2 * x), "e^(2x)", 0, log(2), "0", "ln 2"),
    (x**3, "x³", 0, 2, "0", "2"),
    (2 * x + 1, "2x + 1", 0, 4, "0", "4"),
    (x**2 + x, "x² + x", 0, 3, "0", "3"),
    (x**2, "x²", 0, 6, "0", "6"),
    (12 * x**2 - 4 * x, "12x² - 4x", 1, 2, "1", "2"),
    (x**4, "x⁴", 0, 2, "0", "2"),
    (5 * x**4 + 3 * x**2, "5x⁴ + 3x²", 0, 2, "0", "2"),
    (x**2 - 4 * x + 5, "x² - 4x + 5", 1, 4, "1", "4"),
    (sympy.sin(2 * x), "sin 2x", 0, pi / 2, "0", "π/2"),
    (sympy.cos(x), "cos x", 0, pi / 6, "0", "π/6"),
    (sympy.sin(x), "sin x", 0, pi / 3, "0", "π/3"),
    (exp(x), "e^x", 0, 2, "0", "2"),
    (exp(-x), "e^(-x)", 0, log(4), "0", "ln 4"),
    (1 / x, "1/x", 1, 5, "1", "5"),
    (1 / x**2, "1/x²", 1, 4, "1", "4"),
    (sqrt(x), "√x", 1, 9, "1", "9"),
)


def _easy(rng: random.Random) -> dict:
    f, ftext, a, b, at, bt = rng.choice(_EASY)
    avg = _average(f, a, b)
    check(close(_numeric_average(f, a, b), float(avg), 1e-6), "numerical average disagrees")
    wrong = _average_wrong(f, a, b, avg)
    text = phrase(
        rng,
        (
            "Find the average value of f(x) = {f} on the interval [{a}, {b}].",
            "Find the mean value of the function f(x) = {f} over {a} ≤ x ≤ {b}.",
        ),
        f=ftext,
        a=at,
        b=bt,
    )
    width = _show(b - a)
    working = [
        f"The average value is (1/(b - a)) times the integral of f from a to b, with b - a = {width}.",
        f"The integral of {ftext} from {at} to {bt} is {_show(definite(f, x, a, b))}.",
        f"Average = {_show(definite(f, x, a, b))} ÷ {width} = {_show(avg)}.",
    ]
    return _question(text, avg, wrong, working)


# -- medium --------------------------------------------------------------------------------------------------------------

_SQRT_PAIRS = ((1, 9), (0, 4), (1, 25), (4, 16), (0, 16), (4, 36), (1, 16), (9, 25))
_RECIP_PAIRS = ((1, 4), (2, 8), (1, 9), (4, 9), (1, 16), (3, 12), (2, 18), (9, 16))


def _mvt_sqrt(rng: random.Random) -> dict:
    a, b = rng.choice(_SQRT_PAIRS)
    f = sqrt(x)
    slope = (sqrt(b) - sqrt(a)) / Rational(b - a)
    c = sympy.nsimplify(((sqrt(a) + sqrt(b)) / 2) ** 2)
    check(a < c < b, "c is outside the interval")
    h = 1e-6
    check(
        close((math.sqrt(float(c) + h) - math.sqrt(float(c) - h)) / (2 * h), float(slope), 1e-6),
        "slope condition fails",
    )
    wrong = [
        (Rational(a + b, 2), "fy1ca-mvt-midpoint-assumed"),
        (slope, "fy1ca-mvt-slope-reported"),
        ((sqrt(a) + sqrt(b)) / 2, "fy1ca-mvt-root-not-squared"),
        (sqrt(a * b), "fy1ca-mvt-geometric-mean"),
        (1 / (2 * slope), "fy1ca-mvt-algebra-slip"),
        (sqrt(c), "fy1ca-mvt-value-for-c"),
    ]
    text = phrase(
        rng,
        (
            "The function f(x) = √x satisfies the conditions of the mean value theorem on [{a}, {b}]. Find the value of c in ({a}, {b}) for which f'(c) = (f({b}) - f({a}))/({b} - {a}).",
            "For f(x) = √x on the interval [{a}, {b}], the mean value theorem guarantees a number c between {a} and {b} with f'(c) = (f({b}) - f({a}))/({b} - {a}). Find c.",
        ),
        a=a,
        b=b,
    )
    working = [
        f"The slope of the chord is (√{b} - √{a})/({b} - {a}) = {_show(slope)}.",
        f"f'(x) = 1/(2√x), so 1/(2√c) = {_show(slope)} gives √c = {_show(sympy.nsimplify(1 / (2 * slope)))}.",
        f"So c = {_show(c)}, and this is between {a} and {b}.",
    ]
    return _question(text, c, wrong, working)


def _mvt_reciprocal(rng: random.Random) -> dict:
    a, b = rng.choice(_RECIP_PAIRS)
    slope = Rational(1, b) - Rational(1, a)
    slope = slope / (b - a)
    c = sqrt(a * b)
    c = sympy.nsimplify(c)
    check(c.is_Rational and a < c < b, "c must be rational and between the ends")
    h = 1e-6
    check(close((1 / (float(c) + h) - 1 / (float(c) - h)) / (2 * h), float(slope), 1e-6), "slope condition fails")
    wrong = [
        (Rational(a + b, 2), "fy1ca-mvt-midpoint-assumed"),
        (slope, "fy1ca-mvt-slope-reported"),
        (Rational(a * b), "fy1ca-mvt-root-not-taken"),
        (-c, "fy1ca-mvt-negative-root"),
        (-1 / slope, "fy1ca-mvt-algebra-slip"),
    ]
    text = phrase(
        rng,
        (
            "The function f(x) = 1/x satisfies the conditions of the mean value theorem on [{a}, {b}]. Find the value of c in ({a}, {b}) for which f'(c) = (f({b}) - f({a}))/({b} - {a}).",
            "For f(x) = 1/x on the interval [{a}, {b}], the mean value theorem guarantees a number c between {a} and {b} with f'(c) = (f({b}) - f({a}))/({b} - {a}). Find c.",
        ),
        a=a,
        b=b,
    )
    working = [
        f"The slope of the chord is (1/{b} - 1/{a})/({b} - {a}) = {_show(slope)}.",
        f"f'(x) = -1/x², so -1/c² = {_show(slope)} gives c² = {_show(a * b)}.",
        f"c = {_show(c)} (the positive root, because c lies between {a} and {b}).",
    ]
    return _question(text, c, wrong, working)


def _rolle(rng: random.Random) -> dict:
    p = rng.choice([3, 6, 9, 12])
    squared_first = rng.random() < 0.5
    if squared_first:
        f = x**2 * (x - p)
        ftext = f"x²(x - {p})"
        c = Rational(2 * p, 3)
        other = Rational(0)
        wrong_factor = Rational(p, 3)
    else:
        f = x * (x - p) ** 2
        ftext = f"x(x - {p})²"
        c = Rational(p, 3)
        other = Rational(p)
        wrong_factor = Rational(2 * p, 3)
    check(vanishes(f.subs(x, 0) - f.subs(x, p)), "Rolle's hypothesis fails")
    check(sympy.solve(sympy.diff(f, x), x).count(c) == 1 and 0 < c < p, "c is not a stationary point in the interval")
    h = 1e-6
    fn = lambdified(x, f)
    check(abs((fn(float(c) + h) - fn(float(c) - h)) / (2 * h)) < 1e-5, "f' does not vanish at c")
    wrong = [
        (other, "fy1ca-rolle-endpoint-root"),
        (wrong_factor, "fy1ca-rolle-wrong-factor"),
        (Rational(p, 2), "fy1ca-mvt-midpoint-assumed"),
        (f.subs(x, c), "fy1ca-mvt-value-for-c"),
        (Rational(3 * p, 2) if squared_first else 3 * Rational(p), "fy1ca-rolle-algebra-slip"),
    ]
    text = phrase(
        rng,
        (
            "The function f(x) = {f} satisfies f(0) = f({p}) = 0. By Rolle's theorem there is a number c in (0, {p}) with f'(c) = 0. Find c.",
            "Show that f(x) = {f} satisfies the conditions of Rolle's theorem on [0, {p}], then find the value of c in (0, {p}) for which f'(c) = 0.",
        ),
        f=ftext,
        p=p,
    )
    fprime = sympy.factor(sympy.diff(f, x))
    working = [
        f"f is a polynomial, so it is continuous and differentiable, and f(0) = f({p}) = 0.",
        f"f'(x) = {_show(fprime)}, which is zero at x = {_show(other)} and x = {_show(c)}.",
        f"The root strictly between 0 and {p} is x = {_show(c)}, so c = {_show(c)}.",
    ]
    return _question(text, c, wrong, working)


def _medium(rng: random.Random) -> dict:
    return rng.choice([_mvt_sqrt, _mvt_reciprocal, _rolle, _rolle])(rng)


# -- hard ----------------------------------------------------------------------------------------------------------------


def _exp_square(rng: random.Random) -> dict:
    b = rng.choice([1, 2])
    f = x * exp(x**2)
    avg = _average(f, 0, b)
    check(close(_numeric_average(f, 0, b), float(avg), 1e-6), "numerical average disagrees")
    wrong = [
        ((exp(b**2) - 1) / 2, "fy1ca-avg-division-forgotten"),
        ((exp(b**2) - 1) / b, "fy1ca-avg-substitution-factor-dropped"),
        ((exp(b**2) - 1) / (4 * b), "fy1ca-avg-substitution-factor-slip"),
        (exp(b**2) * b / 2 - Rational(b, 2), "fy1ca-avg-chain-rule-misapplied"),
        (b * exp(b**2) / 2, "fy1ca-avg-lower-limit-ignored"),
    ]
    text = phrase(
        rng,
        (
            "Find the average value of f(x) = x e^(x²) on the interval [0, {b}].",
            "Find the mean value of the function f(x) = x e^(x²) over 0 ≤ x ≤ {b}.",
        ),
        b=b,
    )
    working = [
        "Use the substitution u = x², du = 2x dx: the integral of x e^(x²) dx is (1/2)e^(x²).",
        f"The integral from 0 to {b} is (1/2)(e^{b * b} - 1).",
        f"Average = (1/{b}) × (1/2)(e^{b * b} - 1) = {_show(avg)}.",
    ]
    return _question(text, avg, wrong, working)


def _root_square_plus(rng: random.Random) -> dict:
    k, b = rng.choice([(9, 4), (16, 3), (25, 12), (144, 5), (16, 9)])
    if not sympy.sqrt(b * b + k).is_Rational:
        raise Redraw
    f = x * sqrt(x**2 + k)
    avg = _average(f, 0, b)
    check(close(_numeric_average(f, 0, b), float(avg), 1e-6), "numerical average disagrees")
    root_k = sympy.sqrt(k)
    wrong = [
        (((b * b + k) ** Rational(3, 2) - root_k**3) / 3, "fy1ca-avg-division-forgotten"),
        (((b * b + k) ** Rational(3, 2) - root_k**3) / (2 * b), "fy1ca-avg-substitution-factor-slip"),
        (((b * b + k) ** Rational(3, 2)) / (3 * b), "fy1ca-avg-lower-limit-ignored"),
        (((b * b + k) ** Rational(3, 2) - root_k**3) * 2 / (3 * b), "fy1ca-avg-power-rule-slip"),
        (((b * b + k) ** Rational(1, 2) - root_k) / b, "fy1ca-avg-power-rule-slip"),
    ]
    text = phrase(
        rng,
        (
            "Find the average value of f(x) = x√(x² + {k}) on the interval [0, {b}].",
            "Find the mean value of the function f(x) = x√(x² + {k}) over 0 ≤ x ≤ {b}.",
        ),
        k=k,
        b=b,
    )
    working = [
        f"Use the substitution u = x² + {k}, du = 2x dx: the integral of x√(x² + {k}) dx is (1/3)(x² + {k})^(3/2).",
        f"The integral from 0 to {b} is (1/3)({b * b + k}^(3/2) - {k}^(3/2)) = {_show(definite(f, x, 0, b))}.",
        f"Average = {_show(definite(f, x, 0, b))} ÷ {b} = {_show(avg)}.",
    ]
    return _question(text, avg, wrong, working)


def _arctan_average(rng: random.Random) -> dict:
    a, b = rng.choice([(0, 1), (-1, 1), (0, sqrt(3)), (-sqrt(3), sqrt(3)), (1, sqrt(3))])
    f = 1 / (1 + x**2)
    avg = sympy.simplify((sympy.atan(b) - sympy.atan(a)) / (b - a))
    check(close(_numeric_average(f, a, b), float(avg), 1e-6), "numerical average disagrees")
    wrong = [
        (sympy.atan(b) - sympy.atan(a), "fy1ca-avg-division-forgotten"),
        (sympy.atan(b) / b if b != 0 else None, "fy1ca-avg-lower-limit-ignored"),
        (sympy.log(1 + b**2) - sympy.log(1 + a**2), "fy1ca-avg-wrong-antiderivative"),
        ((sympy.log(1 + b**2) - sympy.log(1 + a**2)) / (b - a), "fy1ca-avg-wrong-antiderivative"),
        ((f.subs(x, a) + f.subs(x, b)) / 2, "fy1ca-avg-endpoint-mean"),
    ]
    at, bt = (_show(v).replace("sqrt(3)", "√3") for v in (a, b))
    text = phrase(
        rng,
        (
            "Find the average value of f(x) = 1/(1 + x²) on the interval [{a}, {b}].",
            "Find the mean value of the function f(x) = 1/(1 + x²) over {a} ≤ x ≤ {b}.",
        ),
        a=at,
        b=bt,
    )
    working = [
        "The integral of 1/(1 + x²) is arctan x.",
        f"Average = (arctan({bt}) - arctan({at}))/({bt} - ({at})) = {_show(avg)}.",
    ]
    return _question(text, avg, wrong, working)


def _integral_mvt(rng: random.Random) -> dict:
    """The c with f(c) equal to the average value: f = √x gives c = 4b/9."""
    b = rng.choice([9, 18, 27, 36, 45, 81])
    f = sqrt(x)
    avg = _average(f, 0, b)
    c = Rational(4 * b, 9)
    check(vanishes(f.subs(x, c) - avg), "f(c) is not the average value")
    check(0 < c < b, "c is outside the interval")
    wrong = [
        (avg, "fy1ca-mvt-value-for-c"),
        (Rational(b, 2), "fy1ca-mvt-midpoint-assumed"),
        (Rational(2 * b, 3), "fy1ca-mvt-root-not-squared"),
        (avg**2 * 2, "fy1ca-mvt-algebra-slip"),
        (sqrt(c), "fy1ca-mvt-value-for-c"),
    ]
    text = phrase(
        rng,
        (
            "The average value of f(x) = √x on [0, {b}] is attained at some point c in the interval, so that f(c) equals the average value. Find the exact value of c.",
            "By the mean value theorem for integrals, there is a number c in [0, {b}] for which f(c) is the average value of f(x) = √x over that interval. Find c.",
        ),
        b=b,
    )
    working = [
        f"The average value is (1/{b}) times the integral of √x from 0 to {b} = (1/{b}) × (2/3){b}^(3/2) = {_show(avg)}.",
        f"Set √c = {_show(avg)}, so c = ({_show(avg)})².",
        f"c = {_show(c)}.",
    ]
    return _question(text, c, wrong, working)


def _hard(rng: random.Random) -> dict:
    return rng.choice([_exp_square, _root_square_plus, _arctan_average, _integral_mvt])(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fy1ca_average_value_mvt", level=19, presets={t: dict(tier=t) for t in _TIERS})
def fy1ca_average_value_mvt(rng: random.Random, tier: str = "easy") -> dict:
    """Average value (easy), c of Rolle or the mean value theorem (medium), substitution averages and f(c) = average (hard)."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return build_with_redraws(rng, _TIERS[tier], f"a {tier} average-value question")


# -- exam ----------------------------------------------------------------------------------------------------------------


def _exam_build(rng: random.Random) -> dict:
    a, b = rng.choice([(p, q) for p in (1, 4, 9, 16, 25, 36) for q in (4, 9, 16, 25, 36, 49) if p < q])
    total = Rational(2, 3) * (sympy.sqrt(b) ** 3 - sympy.sqrt(a) ** 3)
    total = sympy.nsimplify(total)
    avg = total / (b - a)
    c = avg**2
    peak = sympy.sqrt(b)
    over = (peak - avg) / avg * 100
    f = sympy.sqrt(x)
    check(close(_numeric_average(f, a, b), float(avg), 1e-6), "numerical average disagrees")
    check(a < c < b, "c is outside the interval")
    over_f = float(over)
    if not clear_of_rounding(over_f, 1):
        raise Redraw
    total_f = Fraction(int(total.p), int(total.q))
    avg_f = Fraction(int(avg.p), int(avg.q))
    c_f = Fraction(int(c.p), int(c.q))

    a_dist = fraction_distractors(
        total_f,
        [
            (total * Rational(3, 2), "fy1ca-avg-wrong-antiderivative"),
            (total * Rational(2, 3) * 2, "fy1ca-avg-wrong-antiderivative"),
            (Rational(2, 3) * (b - a), "fy1ca-avg-power-rule-slip"),
            (Rational(2, 3) * (sympy.sqrt(b) - sympy.sqrt(a)), "fy1ca-avg-power-rule-slip"),
            (sympy.sqrt(b) - sympy.sqrt(a), "fy1ca-avg-no-antiderivative"),
        ],
    )
    b_dist = fraction_distractors(
        avg_f,
        [
            (total, "fy1ca-avg-division-forgotten"),
            (total / b, "fy1ca-avg-divide-by-upper-limit"),
            ((sympy.sqrt(a) + sympy.sqrt(b)) / 2, "fy1ca-avg-endpoint-mean"),
            (total * (b - a), "fy1ca-avg-multiplied"),
            (total / (b + a), "fy1ca-avg-divide-by-sum"),
        ],
    )
    c_dist = fraction_distractors(
        c_f,
        [
            (avg, "fy1ca-mvt-value-for-c"),
            (Rational(a + b, 2), "fy1ca-mvt-midpoint-assumed"),
            (sympy.sqrt(c), "fy1ca-mvt-value-for-c"),
            (avg**2 * 2, "fy1ca-mvt-algebra-slip"),
            (avg * 2, "fy1ca-mvt-algebra-slip"),
        ],
    )
    d_dist = number_distractors(
        over_f,
        [
            (float((peak - avg) / peak * 100), "fy1ca-avg-percentage-of-wrong-base"),
            (float(peak / avg * 100), "fy1ca-avg-percentage-of-wrong-base"),
            (float(peak - avg), "fy1ca-avg-difference-not-percentage"),
            (float((peak - avg) / avg), "fy1ca-avg-difference-not-percentage"),
            (float((avg - peak) / avg * 100), "fy1ca-avg-percentage-sign"),
        ],
        1,
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        need_distractors(dist)
    stem = phrase(
        rng,
        (
            "During a storm, rain flows into a farm's rainwater tank at a rate of f(t) = √t kilolitres per hour, where t is the number of hours since the storm began. Consider the period from t = {a} to t = {b}.",
            "The inflow to a rainwater tank after a storm begins is f(t) = √t kilolitres per hour, t hours after the start. Consider the period {a} ≤ t ≤ {b}.",
            "A rainwater tank fills from a catchment channel at f(t) = √t kilolitres per hour, where t is the time in hours since the storm began. Consider the interval from t = {a} to t = {b}.",
        ),
        a=a,
        b=b,
    )
    parts = [
        fraction_part(
            "Find the total volume of water, in kilolitres, that flows into the tank during this period.",
            total_f,
            2,
            [
                f"The integral of √t from {a} to {b} is [(2/3)t^(3/2)] = (2/3)({sympy.sqrt(b) ** 3} - {sympy.sqrt(a) ** 3}) = {_show(total)} kL."
            ],
            a_dist,
        ),
        fraction_part(
            "Hence find the average inflow rate over this period, in kilolitres per hour.",
            avg_f,
            1,
            [f"Average = {_show(total)} ÷ ({b} - {a}) = {_show(avg)} kL/h."],
            b_dist,
            follow_through=f"part_a/{b - a}",
        ),
        fraction_part(
            "At what time t, in hours since the storm began, is the inflow rate equal to the average rate?",
            c_f,
            2,
            [f"√t = {_show(avg)}, so t = ({_show(avg)})² = {_show(c)} hours."],
            c_dist,
            follow_through="part_b**2",
        ),
        number_part(
            "By what percentage does the greatest inflow rate in this period exceed the average rate? Give your answer correct to 1 decimal place.",
            over_f,
            1,
            2,
            [
                f"The greatest rate is f({b}) = {_show(peak)} kL/h, so the excess is ({_show(peak)} - {_show(avg)}) ÷ ({_show(avg)}) × 100 = {over_f:.1f}%."
            ],
            d_dist,
            follow_through=f"({peak} - part_b)/part_b*100",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("fy1ca_average_value_mvt_exam", level=19, presets={"exam": {}})
def fy1ca_average_value_mvt_exam(rng: random.Random) -> dict:
    """A storm inflow rate f(t) = √t: total inflow, average rate, the time the rate equals the average, the peak excess."""
    return build_with_redraws(rng, _exam_build, "an average-value exam chain")
