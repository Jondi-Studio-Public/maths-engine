"""Generates `ucv1_improper_integral` and `exd_x_uc_improper_integral` questions: convergent improper integrals.

One `university-core-v1` skill, ``ucv1-improper-integral`` (domain ``integration``). Every integral converges and
every answer is exact (a fraction, a logarithm, a multiple of pi, a surd, a power of e), so the skill is graded as an
expression. "Does it converge?" is never asked on its own: that is a coin toss.

Tiers:

- ``easy``: a type 1 integral c/x^p from a to infinity, c e^(-kx) or c e^(-x/k) from 0 to infinity, or a type 2
  integral c/x^p from 0 to b with p < 1. One antiderivative and one limit.
- ``medium``: a method has to be chosen first: partial fractions (a logarithm), a substitution (x e^(-kx^2)), the
  type 2 integral c/sqrt(a - x), or an arctan or arcsin standard form (a multiple of pi).
- ``hard``: integration by parts at infinity (x e^(-kx), x^2 e^(-kx), ln x / x^p, x^m ln x on (0, 1]), or a parameter
  to find ("the k for which the integral equals V").
- ``exam`` (topic ``exd_x_uc_improper_integral``): a four-part chain on a probability density function (an
  exponential, a power law, x e^(-lambda x), or k/sqrt(x) on (0, b]) or on a region under y = c/x^p and the solids it
  makes: the constant that makes the integral 1, a tail probability, the mean, then the variance (or the volumes and
  their ratio). Each later part follows through from the earlier results.

Every answer is re-derived by a second route: the integral is evaluated numerically with `mpmath.quad` and compared
with the exact value.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

import mpmath as mp
import sympy
from sympy import Rational, exp, log, pi, sqrt

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, latex_answer, rational_text
from ._ucv1_calc_markup import M, coef, say, tex_power, tex_rational, value_string, value_text

TOPIC = "ucv1_improper_integral"
EXAM_TOPIC = "exd_x_uc_improper_integral"

_TOL = 1e-6


def _q(n) -> Rational:
    n = Fraction(n)
    return Rational(n.numerator, n.denominator)


def _same(a, b) -> bool:
    return abs(float(a) - float(b)) < 1e-9


def _expression_distractors(answer, candidates, n: int = 4) -> list[Distractor]:
    seen = [answer]
    out: list[Distractor] = []
    for value, misconception in candidates:
        value = sympy.sympify(value)
        if not value.is_real or any(_same(value, s) for s in seen):
            continue
        seen.append(value)
        out.append(Distractor(value_text(value), misconception, value_string(value)))
        if len(out) == n:
            break
    return out


def _fraction_distractors(answer, candidates, n: int = 4) -> list[Distractor]:
    seen = [answer]
    out: list[Distractor] = []
    for value, misconception in candidates:
        value = sympy.sympify(value)
        if not value.is_Rational or any(_same(value, s) for s in seen):
            continue
        seen.append(value)
        out.append(Distractor(rational_text(value), misconception, exact_value(value)))
        if len(out) == n:
            break
    return out


def _quad(fn: Callable, lo, hi, dps: int = 20) -> float:
    with mp.workdps(dps):
        return float(mp.quad(fn, [lo, hi]))


def _verify(fn: Callable, lo, hi, value, what: str) -> None:
    """The second route: the integral evaluated numerically."""
    target = float(value)
    got = _quad(fn, lo, hi)
    check(abs(got - target) < _TOL * max(1.0, abs(target)), f"{what}: numeric {got}, exact {target}")


@dataclass
class _Case:
    body: str  # TeX of the integral
    value: sympy.Basic  # the answer
    fn: Callable  # the integrand
    lo: object
    hi: object
    working: list[str]
    wrongs: list = field(default_factory=list)
    integral: sympy.Basic | None = None  # the integral's value, when the answer is a parameter
    sentence: tuple | None = None  # (lead, tail, tex-after-integral) for the "find the parameter" questions


def _pick(rng: random.Random, values, exclude=()):
    return rng.choice([v for v in values if v not in exclude])


def _paren_if_fraction(text: str) -> str:
    return f"({text})" if "/" in text else text


def _scaled_xp(q) -> str:
    """The denominator q·x^q written for a divisor: x, (3)x² ..., never a coefficient of 1."""
    q = Fraction(q)
    if q == 1:
        return "x"
    return f"({_cx(q)}{_xp(q)})"


def _xp(p) -> str:
    """Plain text for x^p: x, x², x³, x^4, x^(3/2)."""
    p = Fraction(p)
    if p == 1:
        return "x"
    if p == 2:
        return "x²"
    if p == 3:
        return "x³"
    if p.denominator == 1:
        return f"x^{p.numerator}"
    return f"x^({p.numerator}/{p.denominator})"


def _xtex(p) -> str:
    return "x" if Fraction(p) == 1 else f"x^{{{tex_power(p)}}}"


def _R(p) -> Rational:
    return _q(p)


def _cx(cf) -> str:
    """A rational coefficient before a variable: "", "-", "3", "(1/2)", "-(1/2)"."""
    cf = Fraction(cf)
    mag = abs(cf)
    if mag == 1:
        body = ""
    elif mag.denominator == 1:
        body = str(mag.numerator)
    else:
        body = f"({mag.numerator}/{mag.denominator})"
    return ("-" if cf < 0 else "") + body


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy_power_tail(rng):
    p = rng.choice([2, 3, 4, 5, 6, Fraction(3, 2), Fraction(5, 2)])
    p = Fraction(p)
    a = rng.choice([1, 1, 2, 3, 4]) if p.denominator == 1 else rng.choice([1, 4, 9])
    c = rng.randint(1, 12)
    P, A, C = _R(p), sympy.Integer(a), sympy.Integer(c)
    q = P - 1
    value = C * A ** (1 - P) / q
    aq = A**q
    return _Case(
        rf"\int_{{{a}}}^{{\infty}}\frac{{{c}}}{{{_xtex(p)}}}\,dx",
        value,
        lambda x: c / x ** float(p),
        a,
        mp.inf,
        [
            f"An antiderivative of {c}/{_xp(p)} is -{c}/{_scaled_xp(q)}, which tends to 0 as x → ∞.",
            f"The integral is 0 - (-{c}/{rational_text(aq) if q == 1 else f'({rational_text(q)} × {rational_text(aq)})'}) "
            f"= {rational_text(value)}.",
        ],
        [
            (C * A ** (1 - P) / P, "divides-by-p-not-p-minus-1"),
            (-value, "calc-sign-slip"),
            (C * A ** (1 - P), "power-rule-divisor-dropped"),
            (C * A**q / q, "lower-limit-power-sign"),
        ],
    )


def _easy_exp(rng):
    c = rng.randint(2, 12)
    if rng.random() < 0.5:
        k = rng.randint(2, 9)
        value = _q(Fraction(c, k))
        return _Case(
            rf"\int_{{0}}^{{\infty}}{c}e^{{-{coef(k)}x}}\,dx",
            value,
            lambda x: c * mp.exp(-k * x),
            0,
            mp.inf,
            [
                f"An antiderivative of {c}e^(-{coef(k)}x) is -({c}/{k})e^(-{coef(k)}x), which tends to 0 as x → ∞."
                if k > 1
                else f"An antiderivative of {c}e^(-x) is -{c}e^(-x), which tends to 0 as x → ∞.",
                f"The integral is 0 - (-({c}/{k})) = {rational_text(value)}."
                if k > 1
                else f"The integral is 0 - (-{c}) = {c}.",
            ],
            [
                (_q(c), "chain-rule-divisor-dropped"),
                (-value, "calc-sign-slip"),
                (_q(c * k), "scaled-the-wrong-way"),
            ],
        )
    k = rng.randint(2, 8)
    value = _q(c * k)
    return _Case(
        rf"\int_{{0}}^{{\infty}}{c}e^{{-\frac{{x}}{{{k}}}}}\,dx",
        value,
        lambda x: c * mp.exp(-x / k),
        0,
        mp.inf,
        [
            f"An antiderivative of {c}e^(-x/{k}) is -{c * k}e^(-x/{k}), which tends to 0 as x → ∞.",
            f"The integral is 0 - (-{c * k}) = {c * k}.",
        ],
        [
            (_q(c), "chain-rule-divisor-dropped"),
            (_q(Fraction(c, k)), "scaled-the-wrong-way"),
            (-value, "calc-sign-slip"),
        ],
    )


_TYPE2_EASY = {
    Fraction(1, 2): [1, 4, 9, 16],
    Fraction(1, 3): [1, 8, 27],
    Fraction(2, 3): [1, 8, 27],
    Fraction(1, 4): [1, 16],
    Fraction(3, 4): [1, 16],
}


def _easy_type2(rng):
    p = rng.choice(sorted(_TYPE2_EASY))
    b = rng.choice(_TYPE2_EASY[p])
    c = rng.randint(1, 10)
    P, B, C = _R(p), sympy.Integer(b), sympy.Integer(c)
    value = C * B ** (1 - P) / (1 - P)
    check(value.is_Rational, "type 2 value not rational")
    denominator = r"\sqrt{x}" if p == Fraction(1, 2) else _xtex(p)
    return _Case(
        rf"\int_{{0}}^{{{b}}}\frac{{{c}}}{{{denominator}}}\,dx",
        value,
        lambda x: c / x ** float(p),
        0,
        b,
        [
            f"The integrand is unbounded at x = 0, so the integral is a limit as t → 0⁺ of the integral from t to {b}.",
            f"An antiderivative of {c}/{_xp(p) if p != Fraction(1, 2) else '√x'} is "
            f"{coef(c)}{_xp(1 - p)}/{_paren_if_fraction(rational_text(1 - P))}, which tends to 0 as x → 0⁺.",
            f"At x = {b} it equals {rational_text(value)}, so the integral is {rational_text(value)}.",
        ],
        [
            (C * B ** (1 - P) / P, "divides-by-p-not-p-minus-1"),
            (C / (1 - P), "forgets-upper-limit-factor"),
            (C * B ** (1 - P) / (1 + P), "power-rule-exponent-sign"),
            (-value, "calc-sign-slip"),
        ],
    )


_EASY = [_easy_power_tail, _easy_power_tail, _easy_exp, _easy_type2]


# -- medium -------------------------------------------------------------------------------------------------------


def _med_partial_fractions(rng):
    a = rng.choice([1, 2, 3])
    r = rng.choice([0, 0, 1])
    k = rng.choice([1, 2, 3, 4])
    if a == 1 and r == 0 and k == 1:
        a = 2  # keeps the wrong answers distinct from the key
    s = r + k
    c = k * rng.randint(1, 3) if rng.random() < 0.5 else rng.randint(1, 6)
    ratio = Rational(a + s, a + r)
    value = Rational(c, k) * log(ratio)
    first = "x" if r == 0 else f"x + {r}"
    body_den = rf"x(x + {s})" if r == 0 else rf"(x + {r})(x + {s})"
    return _Case(
        rf"\int_{{{a}}}^{{\infty}}\frac{{{c}}}{{{body_den}}}\,dx",
        value,
        lambda x: c / ((x + r) * (x + s)),
        a,
        mp.inf,
        [
            f"By partial fractions, {c}/(({first})(x + {s})) = ({rational_text(_q(Fraction(c, k)))})"
            f"(1/({first}) - 1/(x + {s})).",
            f"An antiderivative is ({rational_text(_q(Fraction(c, k)))})ln(({first})/(x + {s})), which tends to 0 "
            "as x → ∞.",
            f"At x = {a} it equals ({rational_text(_q(Fraction(c, k)))})ln({a + r}/{a + s}), so the integral is "
            f"{value_text(value)}.",
        ],
        [
            (-value, "log-quotient-inverted"),
            (c * log(ratio), "partial-fraction-coefficient-dropped"),
            (Rational(c, k) * log(a + s), "forgets-lower-limit-term"),
            (Rational(c, k) * log(Rational(a + s, a)), "partial-fraction-term-dropped"),
        ],
    )


def _med_substitution(rng):
    m = rng.choice([2, 3])
    k = rng.randint(1, 6)
    c = rng.randint(1, 9)
    value = Rational(c, m * k)
    xpow = "x" if m == 2 else "x^{2}"
    upower = "²" if m == 2 else "³"
    plain_x = "x" if m == 2 else "x²"
    return _Case(
        rf"\int_{{0}}^{{\infty}}{coef(c)}{xpow}\,e^{{-{coef(k)}x^{{{m}}}}}\,dx",
        value,
        lambda x: c * x ** (m - 1) * mp.exp(-k * x**m),
        0,
        mp.inf,
        [
            f"Let u = {coef(k)}x{upower}, so du = {m * k}{plain_x if m == 3 else 'x'} dx. The limits stay 0 and ∞.",
            f"The integral becomes ({c}/{m * k}) times the integral of e^(-u) from 0 to ∞, which is 1.",
            f"So the value is {rational_text(value)}.",
        ],
        [
            (_q(Fraction(c, k)), "substitution-factor-dropped"),
            (_q(Fraction(c * m, k)), "scaled-the-wrong-way"),
            (_q(Fraction(c, m)), "substitution-factor-dropped"),
            (-value, "calc-sign-slip"),
        ],
    )


def _med_sqrt_endpoint(rng):
    big_l = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
    c = rng.randint(1, 8)
    value = 2 * c * sqrt(big_l)
    if rng.random() < 0.5:
        body = rf"\int_{{0}}^{{{big_l}}}\frac{{{c}}}{{\sqrt{{{big_l} - x}}}}\,dx"
        fn = lambda x: c / mp.sqrt(big_l - x)  # noqa: E731
        lo, hi = 0, big_l
        steps = [
            f"The integrand is unbounded at x = {big_l}, so the integral is a limit as t → {big_l}⁻.",
            f"An antiderivative of {c}/√({big_l} - x) is -{2 * c}√({big_l} - x), which tends to 0 as x → {big_l}⁻.",
            f"At x = 0 it equals -{2 * c}√{big_l}, so the integral is 0 - (-{2 * c}√{big_l}) = {value_text(value)}.",
        ]
        sign_wrong = (-value, "calc-sign-slip")
    else:
        start = rng.choice([1, 2, 3])
        body = rf"\int_{{{start}}}^{{{start + big_l}}}\frac{{{c}}}{{\sqrt{{x - {start}}}}}\,dx"
        fn = lambda x: c / mp.sqrt(x - start)  # noqa: E731
        lo, hi = start, start + big_l
        steps = [
            f"The integrand is unbounded at x = {start}, so the integral is a limit as t → {start}⁺.",
            f"An antiderivative of {c}/√(x - {start}) is {2 * c}√(x - {start}), which tends to 0 as x → {start}⁺.",
            f"At x = {start + big_l} it equals {2 * c}√{big_l}, so the integral is {value_text(value)}.",
        ]
        sign_wrong = (value * big_l, "calc-power-rule-exponent-dropped")
    return _Case(
        body,
        value,
        fn,
        lo,
        hi,
        steps,
        [
            (c * sqrt(big_l), "sqrt-antiderivative-factor-dropped"),
            (2 * c / sqrt(big_l), "calc-ratio-inverted"),
            sign_wrong,
        ],
    )


def _med_arctan(rng):
    k = rng.randint(1, 6)
    c = rng.randint(1, 9)
    whole_line = rng.random() < 0.4
    value = c * pi / k if whole_line else c * pi / (2 * k)
    lo_tex = r"-\infty" if whole_line else "0"
    arg = "x" if k == 1 else f"x/{k}"
    steps = [
        f"The standard form gives an antiderivative {_cx(Fraction(c, k))}arctan({arg}).",
        f"As x → ∞, arctan({arg}) → π/2 and as x → -∞ it tends to -π/2."
        if whole_line
        else f"As x → ∞, arctan({arg}) tends to π/2, and arctan(0) = 0.",
        f"So the integral is {value_text(value)}.",
    ]
    return _Case(
        rf"\int_{{{lo_tex}}}^{{\infty}}\frac{{{c}}}{{x^{{2}} + {k * k}}}\,dx",
        value,
        lambda x: c / (x * x + k * k),
        -mp.inf if whole_line else 0,
        mp.inf,
        steps,
        [
            (
                c * pi / (2 * k) if whole_line else c * pi / k,
                "symmetry-halving-error" if whole_line else "arctan-limit-misread",
            ),
            (c * pi / 2, "arctan-scale-factor-dropped"),
            (c * pi / (k * k), "arctan-scale-factor-squared"),
            (Rational(c, k), "arctan-limit-misread"),
        ],
    )


def _med_arcsin(rng):
    b = rng.randint(1, 6)
    c = rng.randint(1, 9)
    value = c * pi / 2
    return _Case(
        rf"\int_{{0}}^{{{b}}}\frac{{{c}}}{{\sqrt{{{b * b} - x^{{2}}}}}}\,dx",
        value,
        lambda x: c / mp.sqrt(b * b - x * x),
        0,
        b,
        [
            f"The integrand is unbounded at x = {b}. The standard form gives an antiderivative "
            f"{c}arcsin({'x' if b == 1 else f'x/{b}'}).",
            f"At x = {b} it equals {c}arcsin(1) = {c}π/2, and at x = 0 it equals 0.",
            f"So the integral is {value_text(value)}.",
        ],
        [
            (c * pi, "arcsin-value-misread"),
            (c * pi / 4, "arcsin-value-misread"),
            (Rational(c, b) * pi / 2, "arcsin-scale-factor-misapplied"),
            (Rational(c), "arcsin-value-misread"),
        ],
    )


_MEDIUM = [
    _med_partial_fractions,
    _med_partial_fractions,
    _med_substitution,
    _med_substitution,
    _med_sqrt_endpoint,
    _med_sqrt_endpoint,
    _med_arctan,
    _med_arcsin,
]


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_x_exp(rng):
    k = rng.randint(1, 6)
    c = rng.randint(1, 8)
    value = _q(Fraction(c, k * k))
    ex = f"e^(-{coef(k)}x)"
    bracket = f"-x{ex}" if k == 1 else f"-x{ex}/{k}"
    inv = "" if k == 1 else f"({_cx(Fraction(1, k))}) × "
    return _Case(
        rf"\int_{{0}}^{{\infty}}{coef(c)}x\,e^{{-{coef(k)}x}}\,dx",
        value,
        lambda x: c * x * mp.exp(-k * x),
        0,
        mp.inf,
        [
            f"Integrate by parts with u = x and dv = {ex} dx: the integral is [{bracket}] plus {inv}the integral "
            f"of {ex} dx.",
            f"The bracket is 0 at both ends, since x{ex} → 0 as x → ∞.",
            f"The remaining integral is {rational_text(_q(Fraction(1, k)))}, so the value is {rational_text(value)}.",
        ],
        [
            (_q(Fraction(c, k)), "by-parts-second-integral-factor-dropped"),
            (-value, "calc-sign-slip"),
            (_q(Fraction(c, k**3)), "by-parts-extra-divisor"),
            (_q(Fraction(2 * c, k * k)), "by-parts-extra-factor"),
        ],
    )


def _hard_x2_exp(rng):
    k = rng.randint(1, 5)
    c = rng.randint(1, 6)
    value = _q(Fraction(2 * c, k**3))
    ex = f"e^(-{coef(k)}x)"
    lead = rational_text(_q(Fraction(2 * c, k)))
    return _Case(
        rf"\int_{{0}}^{{\infty}}{coef(c)}x^{{2}}\,e^{{-{coef(k)}x}}\,dx",
        value,
        lambda x: c * x * x * mp.exp(-k * x),
        0,
        mp.inf,
        [
            f"Integrating by parts once with u = x² gives {lead} times the integral of x{ex} dx, the boundary term "
            "being 0.",
            f"A second integration by parts gives {rational_text(_q(Fraction(1, k * k)))} for the integral of x{ex} "
            "dx.",
            f"So the value is {rational_text(value)}.",
        ],
        [
            (_q(Fraction(c, k**3)), "calc-constant-factor-dropped"),
            (_q(Fraction(2 * c, k * k)), "by-parts-second-integral-factor-dropped"),
            (-value, "calc-sign-slip"),
            (_q(Fraction(c, k * k)), "by-parts-second-integral-factor-dropped"),
        ],
    )


def _hard_ln_power(rng):
    p = Fraction(rng.choice([2, 3, 4, 5, Fraction(3, 2)]))
    c = rng.randint(1, 6)
    P = _R(p)
    q = P - 1
    value = c / q**2
    return _Case(
        rf"\int_{{1}}^{{\infty}}\frac{{{coef(c)}\ln x}}{{{_xtex(p)}}}\,dx",
        value,
        lambda x: c * mp.log(x) / x ** float(p),
        1,
        mp.inf,
        [
            f"Integrate by parts with u = ln x and dv = x^(-{rational_text(P)}) dx, so v = "
            f"-{rational_text(1 / q)}x^(-{rational_text(q)}).",
            "The boundary term is 0 at both ends, since ln x grows more slowly than any power of x and ln 1 = 0.",
            f"The remaining integral is {rational_text(1 / q**2)}, so the value is {rational_text(value)}.",
        ],
        [
            (c / q, "by-parts-second-integral-factor-dropped"),
            (-value, "calc-sign-slip"),
            (c / P**2, "divides-by-p-not-p-minus-1"),
            (c / q**3, "by-parts-extra-divisor"),
        ],
    )


def _hard_x_ln_zero_one(rng):
    m = rng.choice([0, 1, 2, 3])
    c = rng.randint(1, 8)
    value = _q(Fraction(-c, (m + 1) ** 2))
    xfactor = "" if m == 0 else (r"x\," if m == 1 else rf"x^{{{m}}}\,")
    return _Case(
        rf"\int_{{0}}^{{1}}{coef(c)}{xfactor}\ln x\,dx",
        value,
        lambda x: c * x**m * mp.log(x),
        0,
        1,
        [
            "Integrate by parts with u = ln x.",
            f"Then v = {_cx(Fraction(c, m + 1))}{_xp(m + 1)}, and the boundary term v ln x is 0 at x = 1 and tends "
            "to 0 as x → 0⁺.",
            f"The remaining integral is minus {rational_text(_q(Fraction(c, m + 1)))} times the integral of "
            f"{_xp(m) if m else '1'} from 0 to 1, so the value is {rational_text(value)}.",
        ],
        [
            (-value, "calc-sign-slip"),
            (_q(Fraction(-c, m + 1)), "by-parts-second-integral-factor-dropped"),
            (_q(0), "zero-times-infinity-is-zero"),
            (_q(Fraction(-c, (m + 1) ** 3)), "by-parts-extra-divisor"),
        ],
    )


def _hard_param_exp(rng):
    k0 = rng.choice([2, 3, 4, 5, 6, Fraction(1, 2), Fraction(3, 2), Fraction(5, 2)])
    c = rng.randint(1, 6)
    K = _R(k0)
    big_v = c / K**2
    return _Case(
        rf"\int_{{0}}^{{\infty}}{coef(c)}x\,e^{{-kx}}\,dx",
        K,
        lambda x: c * x * mp.exp(-float(k0) * x),
        0,
        mp.inf,
        [
            f"By parts, the integral of {coef(c)}x e^(-kx) from 0 to ∞ is {c}/k².",
            f"So {c}/k² = {rational_text(big_v)}, giving k² = {rational_text(c / big_v)}.",
            f"Since k > 0, k = {rational_text(K)}.",
        ],
        [
            (c / big_v, "solves-without-root"),
            (1 / K, "calc-ratio-inverted"),
            (c * K, "scaled-the-wrong-way"),
        ],
        integral=big_v,
        sentence=("Find the value of ", " for which ", "k > 0", tex_rational(big_v)),
    )


def _hard_param_power(rng):
    p0 = rng.choice([2, 3, 4, 5, 6, Fraction(3, 2), Fraction(5, 2), Fraction(7, 2)])
    c = rng.randint(1, 6)
    P = _R(p0)
    big_v = c / (P - 1)
    return _Case(
        rf"\int_{{1}}^{{\infty}}\frac{{{c}}}{{x^{{p}}}}\,dx",
        P,
        lambda x: c / x ** float(p0),
        1,
        mp.inf,
        [
            f"For p > 1 the integral of {c}/x^p from 1 to ∞ is {c}/(p - 1).",
            f"So {c}/(p - 1) = {rational_text(big_v)}, giving p - 1 = {rational_text(c / big_v)}.",
            f"Hence p = {rational_text(P)}.",
        ],
        [
            (c / big_v, "solves-p-minus-one-only"),
            (big_v / c, "calc-ratio-inverted"),
            (1 - c / big_v, "calc-sign-slip"),
        ],
        integral=big_v,
        sentence=("Find the value of ", " for which ", "p > 1", tex_rational(big_v)),
    )


def _hard_param_ln(rng):
    p0 = rng.choice([3, 4, 5, Fraction(3, 2), Fraction(5, 2), Fraction(7, 2)])
    c = rng.randint(1, 4)
    P = _R(p0)
    big_v = c / (P - 1) ** 2
    return _Case(
        rf"\int_{{1}}^{{\infty}}\frac{{{coef(c)}\ln x}}{{x^{{p}}}}\,dx",
        P,
        lambda x: c * mp.log(x) / x ** float(p0),
        1,
        mp.inf,
        [
            f"By parts, for p > 1 the integral of {coef(c)}ln x/x^p from 1 to ∞ is {c}/(p - 1)².",
            f"So (p - 1)² = {rational_text(c / big_v)}, and since p > 1, p - 1 = {rational_text(P - 1)}.",
            f"Hence p = {rational_text(P)}.",
        ],
        [
            (sqrt(Rational(c) / big_v), "solves-p-minus-one-only"),
            (1 + c / big_v, "solves-without-root"),
            (1 + sqrt(big_v / c), "calc-ratio-inverted"),
        ],
        integral=big_v,
        sentence=("Find the value of ", " for which ", "p > 1", tex_rational(big_v)),
    )


_HARD = [
    _hard_x_exp,
    _hard_x2_exp,
    _hard_ln_power,
    _hard_x_ln_zero_one,
    _hard_param_exp,
    _hard_param_power,
    _hard_param_ln,
]

_TIERS = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}
_LEADS = ["Evaluate ", "Find the exact value of ", "Calculate "]


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_improper_integral(rng: random.Random, tier: str = "easy") -> dict:
    """A convergent improper integral with an exact value (or the parameter that gives a stated value)."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    case: _Case = rng.choice(_TIERS[tier])(rng)
    # Second route: numeric quadrature of the integrand (for a parameter question, at the parameter's own value).
    _verify(case.fn, case.lo, case.hi, case.integral if case.integral is not None else case.value, case.body)
    if case.sentence is None:
        text, latex = say(rng.choice(_LEADS), M(case.body), ".")
    else:
        lead, middle, condition, value_tex = case.sentence
        text, latex = say(lead, M(condition), middle, M(f"{case.body} = {value_tex}"), ".")
    distractors = _expression_distractors(case.value, case.wrongs)
    check(len(distractors) >= 2, "fewer than two distractors")
    return dict(
        text=text,
        latex=latex,
        answer=value_string(case.value),
        answer_text=value_text(case.value),
        answer_latex=latex_answer(case.value),
        answer_kind="expression",
        working=case.working,
        distractors=distractors,
    )


# -- exam: a four-part chain ---------------------------------------------------------------------------------------


@dataclass
class _Link:
    prompt: tuple  # (text, latex)
    kind: str  # "fraction" or "expression"
    value: sympy.Basic
    marks: int
    working: list
    wrongs: list
    ft: str | None = None
    sensible: tuple | None = None


def _link_part(link: _Link) -> dict:
    if link.kind == "fraction":
        distractors = _fraction_distractors(link.value, link.wrongs)
        answer, text = exact_value(link.value), rational_text(link.value)
    else:
        distractors = _expression_distractors(link.value, link.wrongs)
        answer, text = value_string(link.value), value_text(link.value)
    check(len(distractors) >= 2, f"fewer than two distractors for {link.prompt[0]!r}")
    part = dict(
        prompt=link.prompt[0],
        prompt_latex=link.prompt[1],
        answer_kind=link.kind,
        answer=answer,
        answer_text=text,
        answer_latex=latex_answer(link.value),
        marks=link.marks,
        working=link.working,
        distractors=distractors,
    )
    if link.ft:
        part["follow_through"] = link.ft
    if link.sensible and link.kind == "fraction":
        part["sensible_range"] = link.sensible
    return part


def _near(got: float, want, what: str) -> None:
    want = float(want)
    check(abs(got - want) < _TOL * max(1.0, abs(want)), f"{what}: numeric {got}, exact {want}")


def _check_pdf(pdf, lo, hi, *, t, upper: bool, prob, mean, var, dps: int = 12) -> None:
    """The second route for the density chains: every part evaluated by numerical integration."""
    _near(_quad(pdf, lo, hi, dps), 1, "total probability")
    _near(_quad(pdf, t, hi, dps) if upper else _quad(pdf, lo, t, dps), prob, "probability")
    m = _quad(lambda x: x * pdf(x), lo, hi, dps)
    _near(m, mean, "mean")
    _near(_quad(lambda x: x * x * pdf(x), lo, hi, dps) - m * m, var, "variance")


def _rate_tex(rate: Fraction) -> str:
    """TeX of the rate times x: 2x, x/3, 3x/2."""
    if rate.denominator == 1:
        return f"{coef(rate.numerator)}x"
    return f"{coef(rate.numerator)}x/{rate.denominator}"


_P_A = say("Find the value of k.")
_P_MEAN = lambda: say("Find the mean ", M("E(X)"), ".")  # noqa: E731
_P_VAR = lambda: say("Find the variance ", M(r"\text{Var}(X)"), ".")  # noqa: E731


def _fam_exponential(rng):
    lam = Fraction(
        rng.choice([Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(1, 5), Fraction(1, 6), 2, 3, 4, 5, 6])
    )
    t = rng.randint(1, 5)
    L = _q(lam)
    k, prob, mean, var = L, exp(-L * t), 1 / L, 1 / L**2
    stem = say(
        "A continuous random variable X has probability density function ",
        M(rf"f(x) = k\,e^{{-{_rate_tex(lam)}}}"),
        " for ",
        M(r"x \ge 0"),
        " and ",
        M("f(x) = 0"),
        " otherwise, where k is a constant.",
    )
    _check_pdf(lambda x: float(k) * mp.exp(-float(lam) * x), 0, mp.inf, t=t, upper=True, prob=prob, mean=mean, var=var)
    links = [
        _Link(
            _P_A,
            "fraction",
            k,
            2,
            [
                f"The integral of k e^(-{rational_text(L)}x) from 0 to ∞ is k/{rational_text(L)}, which must equal 1, so k = {rational_text(k)}."
            ],
            [
                (1 / L, "calc-ratio-inverted"),
                (_q(1), "pdf-normalisation-skipped"),
                (-L, "calc-sign-slip"),
                (L * L, "scaled-the-wrong-way"),
            ],
            sensible=(0, None),
        ),
        _Link(
            say("Find ", M(rf"P(X > {t})"), "."),
            "expression",
            prob,
            2,
            [
                f"P(X > {t}) is the integral of k e^(-{rational_text(L)}x) from {t} to ∞, which is e^(-{rational_text(L * t)})."
            ],
            [
                (1 - prob, "tail-cdf-confusion"),
                (exp(L * t), "calc-sign-slip"),
                (exp(-_q(t)), "rate-dropped-from-exponent"),
                (exp(-t / L), "calc-ratio-inverted"),
            ],
            ft=f"part_a*exp(-({L})*{t})/({L})",
            sensible=(0, 1),
        ),
        _Link(
            _P_MEAN(),
            "fraction",
            mean,
            3,
            [
                f"E(X) is the integral of x k e^(-{rational_text(L)}x) from 0 to ∞. Integrating by parts gives 1/k, so E(X) = {rational_text(mean)}."
            ],
            [
                (L, "calc-ratio-inverted"),
                (1 / L**2, "mean-and-variance-confused"),
                (2 / L, "by-parts-extra-factor"),
                (-mean, "calc-sign-slip"),
            ],
            ft=f"part_a/({L})**2",
            sensible=(0, None),
        ),
        _Link(
            _P_VAR(),
            "fraction",
            var,
            3,
            [
                f"E(X²) = 2/k² = {rational_text(2 / L**2)}, so Var(X) = {rational_text(2 / L**2)} - ({rational_text(mean)})² = {rational_text(var)}."
            ],
            [
                (2 / L**2, "variance-second-moment-not-centred"),
                (mean, "standard-deviation-for-variance"),
                (L**2, "calc-ratio-inverted"),
                (2 / L**2 - mean, "variance-subtracts-mean-not-mean-squared"),
            ],
            ft=f"2*part_a/({L})**3 - (part_a/({L})**2)**2",
            sensible=(0, None),
        ),
    ]
    return stem, links


def _fam_x_exponential(rng):
    lam = Fraction(rng.choice([Fraction(1, 2), Fraction(3, 2), Fraction(5, 2), 2, 3, 4, 5]))
    t = rng.randint(1, 3)
    L = _q(lam)
    k = L * L
    prob = exp(-L * t) * (L * t + 1)
    mean, var = 2 / L, 2 / L**2
    stem = say(
        "A continuous random variable X has probability density function ",
        M(rf"f(x) = k x\,e^{{-{_rate_tex(lam)}}}"),
        " for ",
        M(r"x \ge 0"),
        " and ",
        M("f(x) = 0"),
        " otherwise, where k is a constant.",
    )
    _check_pdf(
        lambda x: float(k) * x * mp.exp(-float(lam) * x), 0, mp.inf, t=t, upper=True, prob=prob, mean=mean, var=var
    )
    links = [
        _Link(
            _P_A,
            "fraction",
            k,
            2,
            [
                f"By parts, the integral of x e^(-{rational_text(L)}x) from 0 to ∞ is 1/({rational_text(L)})² = {rational_text(1 / k)}, so k = {rational_text(k)}."
            ],
            [
                (L, "by-parts-second-integral-factor-dropped"),
                (1 / k, "calc-ratio-inverted"),
                (2 * k, "by-parts-extra-factor"),
                (_q(1), "pdf-normalisation-skipped"),
            ],
            sensible=(0, None),
        ),
        _Link(
            say("Find ", M(rf"P(X > {t})"), "."),
            "expression",
            prob,
            3,
            [
                f"P(X > {t}) is the integral of k x e^(-{rational_text(L)}x) from {t} to ∞. By parts it equals e^(-{rational_text(L * t)})({rational_text(L * t)} + 1)."
            ],
            [
                (L * t * exp(-L * t), "by-parts-second-integral-factor-dropped"),
                (exp(-L * t), "by-parts-boundary-term-dropped"),
                (1 - prob, "tail-cdf-confusion"),
                (exp(-L * t) * (1 - L * t), "calc-sign-slip"),
            ],
            ft=f"part_a*exp(-({L})*{t})*(({L})*{t} + 1)/({L})**2",
            sensible=(0, 1),
        ),
        _Link(
            _P_MEAN(),
            "fraction",
            mean,
            2,
            [
                f"E(X) is the integral of k x² e^(-{rational_text(L)}x) from 0 to ∞, which is 2k/({rational_text(L)})³ = {rational_text(mean)}."
            ],
            [
                (1 / L, "by-parts-second-integral-factor-dropped"),
                (L / 2, "calc-ratio-inverted"),
                (2 / L**2, "by-parts-extra-divisor"),
                (4 / L, "by-parts-extra-factor"),
            ],
            ft=f"2*part_a/({L})**3",
            sensible=(0, None),
        ),
        _Link(
            _P_VAR(),
            "fraction",
            var,
            3,
            [
                f"E(X²) = 6k/({rational_text(L)})⁴ = {rational_text(6 / L**2)}, so Var(X) = {rational_text(6 / L**2)} - ({rational_text(mean)})² = {rational_text(var)}."
            ],
            [
                (6 / L**2, "variance-second-moment-not-centred"),
                (4 / L**2, "mean-and-variance-confused"),
                (L**2 / 2, "calc-ratio-inverted"),
                (6 / L**2 - mean, "variance-subtracts-mean-not-mean-squared"),
            ],
            ft=f"6*part_a/({L})**4 - (2*part_a/({L})**3)**2",
            sensible=(0, None),
        ),
    ]
    return stem, links


def _fam_power_law(rng):
    p = rng.choice([4, 5, 6])
    a = rng.choice([1, 2, 3])
    m = rng.choice([2, 3, 4])
    t = m * a
    P, A = sympy.Integer(p), sympy.Integer(a)
    k = (P - 1) * A ** (P - 1)
    prob = sympy.Integer(m) ** (1 - P)
    mean = (P - 1) * A / (P - 2)
    second = (P - 1) * A**2 / (P - 3)
    var = second - mean**2
    stem = say(
        "A continuous random variable X has probability density function ",
        M(rf"f(x) = \frac{{k}}{{x^{{{p}}}}}"),
        " for ",
        M(rf"x \ge {a}"),
        " and ",
        M("f(x) = 0"),
        " otherwise, where k is a constant.",
    )
    _check_pdf(lambda x: float(k) / x**p, a, mp.inf, t=t, upper=True, prob=prob, mean=mean, var=var)
    links = [
        _Link(
            _P_A,
            "fraction",
            k,
            2,
            [
                f"The integral of k/x^{p} from {a} to ∞ is k/({p - 1} × {a}^{p - 1}), which must equal 1, so k = {rational_text(k)}."
            ],
            [
                ((P - 1) / A ** (P - 1), "lower-limit-power-sign"),
                (P * A ** (P - 1), "divides-by-p-not-p-minus-1"),
                (A ** (P - 1), "power-rule-divisor-dropped"),
                (1 / k, "calc-ratio-inverted"),
            ],
            sensible=(0, None),
        ),
        _Link(
            say("Find ", M(rf"P(X > {t})"), "."),
            "fraction",
            prob,
            2,
            [
                f"P(X > {t}) is the integral of k/x^{p} from {t} to ∞, which is k/({p - 1} × {t}^{p - 1}) = {rational_text(prob)}."
            ],
            [
                (1 - prob, "tail-cdf-confusion"),
                (sympy.Integer(m) ** (-P), "power-law-exponent-off-by-one"),
                (sympy.Integer(m) ** (P - 1), "calc-sign-slip"),
                (_q(Fraction(1, m)), "calc-power-rule-exponent-dropped"),
            ],
            ft=f"part_a*{t}**({1 - p})/{p - 1}",
            sensible=(0, 1),
        ),
        _Link(
            _P_MEAN(),
            "fraction",
            mean,
            3,
            [
                f"E(X) is the integral of k/x^{p - 1} from {a} to ∞, which is k/({p - 2} × {a}^{p - 2}) = {rational_text(mean)}."
            ],
            [
                (A, "power-law-exponent-off-by-one"),
                ((P - 1) / (P - 2), "lower-bound-factor-dropped"),
                ((P - 2) * A / (P - 1), "calc-ratio-inverted"),
                (P * A / (P - 2), "divides-by-p-not-p-minus-1"),
            ],
            ft=f"part_a*{a}**({2 - p})/{p - 2}",
            sensible=(0, None),
        ),
        _Link(
            _P_VAR(),
            "fraction",
            var,
            3,
            [
                f"E(X²) is the integral of k/x^{p - 2} from {a} to ∞, which is {rational_text(second)}, so Var(X) = {rational_text(second)} - ({rational_text(mean)})² = {rational_text(var)}."
            ],
            [
                (second, "variance-second-moment-not-centred"),
                (mean**2, "mean-and-variance-confused"),
                (second - mean, "variance-subtracts-mean-not-mean-squared"),
                (second + mean**2, "calc-sign-slip"),
            ],
            ft=f"part_a*{a}**({3 - p})/{p - 3} - (part_a*{a}**({2 - p})/{p - 2})**2",
            sensible=(0, None),
        ),
    ]
    return stem, links


def _fam_root_density(rng):
    s = rng.randint(2, 8)
    u = rng.randint(1, s - 1)
    S, U = sympy.Integer(s), sympy.Integer(u)
    k = 1 / (2 * S)
    prob = U / S
    mean = S**2 / 3
    second = S**4 / 5
    var = second - mean**2
    stem = say(
        "A continuous random variable X has probability density function ",
        M(r"f(x) = \frac{k}{\sqrt{x}}"),
        " for ",
        M(rf"0 < x \le {s * s}"),
        " and ",
        M("f(x) = 0"),
        " otherwise, where k is a constant.",
    )
    _check_pdf(lambda x: float(k) / mp.sqrt(x), 0, s * s, t=u * u, upper=False, prob=prob, mean=mean, var=var, dps=20)
    links = [
        _Link(
            _P_A,
            "fraction",
            k,
            2,
            [
                f"The integral of k/√x from 0 to {s * s} is 2k√{s * s} = {2 * s}k, which must equal 1, so k = {rational_text(k)}."
            ],
            [
                (1 / S, "sqrt-antiderivative-factor-dropped"),
                (1 / (2 * S**2), "square-root-of-bound-dropped"),
                (2 * S, "calc-ratio-inverted"),
                (-k, "calc-sign-slip"),
            ],
            sensible=(0, None),
        ),
        _Link(
            say("Find ", M(rf"P(X \le {u * u})"), "."),
            "fraction",
            prob,
            2,
            [f"P(X ≤ {u * u}) is the integral of k/√x from 0 to {u * u}, which is 2k√{u * u} = {rational_text(prob)}."],
            [
                (U**2 / S**2, "square-root-of-bound-dropped"),
                (1 - prob, "tail-cdf-confusion"),
                (U / (2 * S), "sqrt-antiderivative-factor-dropped"),
                (S / U, "calc-ratio-inverted"),
            ],
            ft=f"2*part_a*{u}",
            sensible=(0, 1),
        ),
        _Link(
            _P_MEAN(),
            "fraction",
            mean,
            3,
            [f"E(X) is the integral of k√x from 0 to {s * s}, which is (2/3)k × {s}³ = {rational_text(mean)}."],
            [
                (S**2 / 2, "power-rule-divisor-dropped"),
                (2 * S**2 / 3, "calc-constant-factor-dropped"),
                (S**3 / 3, "square-root-of-bound-dropped"),
                (3 * S**2 / 2, "calc-ratio-inverted"),
            ],
            ft=f"2*part_a*{s}**3/3",
            sensible=(0, None),
        ),
        _Link(
            _P_VAR(),
            "fraction",
            var,
            3,
            [
                f"E(X²) is the integral of k x^(3/2) from 0 to {s * s}, which is (2/5)k × {s}⁵ = {rational_text(second)}, so Var(X) = {rational_text(second)} - ({rational_text(mean)})² = {rational_text(var)}."
            ],
            [
                (second, "variance-second-moment-not-centred"),
                (mean**2, "mean-and-variance-confused"),
                (second - mean, "variance-subtracts-mean-not-mean-squared"),
                (second + mean**2, "calc-sign-slip"),
            ],
            ft=f"2*part_a*{s}**5/5 - (2*part_a*{s}**3/3)**2",
            sensible=(0, None),
        ),
    ]
    return stem, links


def _fam_revolve(rng):
    p = Fraction(rng.choice([Fraction(5, 2), 3, Fraction(7, 2), 4, 5]))
    c = rng.randint(1, 6)
    P, C = _q(p), sympy.Integer(c)
    area = C / (P - 1)
    v_x = pi * C**2 / (2 * P - 1)
    v_y = 2 * pi * C / (P - 2)
    ratio = v_x / v_y
    stem = say(
        "The region R lies under the curve ",
        M(rf"y = \frac{{{c}}}{{x^{{p}}}}"),
        " and above the x-axis for ",
        M(r"x \ge 1"),
        ", where ",
        M("p > 2"),
        " is a constant. The area of R is ",
        M(tex_rational(area)),
        " square units.",
    )
    fp = float(p)
    _near(_quad(lambda x: c / x**fp, 1, mp.inf, 12), area, "area")
    _near(_quad(lambda x: mp.pi * (c / x**fp) ** 2, 1, mp.inf, 12), v_x, "volume about the x-axis")
    _near(_quad(lambda x: 2 * mp.pi * x * c / x**fp, 1, mp.inf, 12), v_y, "volume about the y-axis")
    links = [
        _Link(
            say("Find the value of p."),
            "fraction",
            P,
            2,
            [
                f"The area is the integral of {c}/x^p from 1 to ∞, which is {c}/(p - 1). So {c}/(p - 1) = {rational_text(area)}, giving p - 1 = {rational_text(P - 1)}."
            ],
            [
                (P - 1, "solves-p-minus-one-only"),
                (2 - P, "calc-sign-slip"),
                (1 + area / C, "calc-ratio-inverted"),
                (P + 1, "calc-sign-slip"),
            ],
            sensible=(2, None),
        ),
        _Link(
            say("Find the exact volume of the solid formed when R is rotated about the x-axis."),
            "expression",
            v_x,
            3,
            [
                f"The volume is π times the integral of ({c}/x^p)² from 1 to ∞, which is π × {c * c}/(2p - 1) = {value_text(v_x)}."
            ],
            [
                (pi * C**2 / (2 * P), "divides-by-p-not-p-minus-1"),
                (pi * C / (2 * P - 1), "calc-constant-factor-dropped"),
                (C**2 / (2 * P - 1), "volume-pi-dropped"),
                (pi * C / (P - 1), "disc-radius-not-squared"),
            ],
            ft="pi*" + f"{c}**2/(2*part_a - 1)",
            sensible=(0, None),
        ),
        _Link(
            say("Find the exact volume of the solid formed when R is rotated about the y-axis."),
            "expression",
            v_y,
            3,
            [
                f"By cylindrical shells the volume is 2π times the integral of x × {c}/x^p from 1 to ∞, which is 2π × {c}/(p - 2) = {value_text(v_y)}."
            ],
            [
                (2 * pi * C / (P - 1), "shell-radius-factor-omitted"),
                (pi * C / (P - 2), "calc-constant-factor-dropped"),
                (v_x, "rotated-about-wrong-axis"),
                (2 * C / (P - 2), "volume-pi-dropped"),
            ],
            ft=f"2*pi*{c}/(part_a - 2)",
            sensible=(0, None),
        ),
        _Link(
            say("Find the ratio of the volume in part (b) to the volume in part (c)."),
            "fraction",
            ratio,
            2,
            [f"The ratio is ({value_text(v_x)})/({value_text(v_y)}) = {rational_text(ratio)}."],
            [
                (1 / ratio, "calc-ratio-inverted"),
                (2 * ratio, "calc-constant-factor-dropped"),
                (-ratio, "calc-sign-slip"),
                (ratio / C, "calc-constant-factor-dropped"),
            ],
            ft="part_b/part_c",
            sensible=(0, None),
        ),
    ]
    return stem, links


_FAMILIES = [_fam_exponential, _fam_x_exponential, _fam_power_law, _fam_root_density, _fam_revolve]


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_uc_improper_integral(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: a density function or a region under y = c/x^p, with improper integrals in all four parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    stem, links = rng.choice(_FAMILIES)(rng)
    return dict(text=stem[0], latex=stem[1], parts=[_link_part(link) for link in links])
