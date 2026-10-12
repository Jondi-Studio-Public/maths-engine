"""Generates `ucv1_lhopital` and `exd_x_uc_lhopital` questions: limits that need L'Hopital's rule.

One `university-core-v1` skill, ``ucv1-lhopital-rule`` (domain ``limits``). Every answer is exact: a rational, or
a power of e for the 1^infinity forms, so the skill is graded as an expression.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: a 0/0 limit at a point that one application settles (e^(ax) - 1, sin, tan, ln(1 + ax), two exponentials,
  a polynomial over x - c or x^2 - c^2).
- ``medium``: the first application is still indeterminate, so the rule is applied twice (1 - cos, e^(ax) - 1 - ax,
  ln(1 + ax) - ax, x sin(ax) over 1 - cos(bx)), or the form is infinity/infinity at infinity (exponentials,
  logarithms of polynomials, a logarithm against a linear term).
- ``hard``: the limit must be rewritten before the rule applies: 0 times infinity, infinity minus infinity, or
  1^infinity through a logarithm, giving e^k.
- ``exam`` (topic ``exd_x_uc_lhopital``): a four-part chain. A numerator g(px) - qx carries an unknown constant q.
  (a) the q for which the 0/0 limit exists, (b) the limit, (c) a second limit that reuses (b), (d) a 1^infinity
  limit whose exponent is (b) divided by p.

Every answer is re-derived by a second route: the limit function is evaluated in 80-digit arithmetic very close to
the limit point and compared with the exact value (SymPy's own `limit` is far too slow for a generator).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

import mpmath as mp
import sympy
from sympy import Rational, exp

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, latex_answer, rational_text
from ._ucv1_calc_markup import M, coef, poly_tex, say, value_string, value_text

TOPIC = "ucv1_lhopital"
EXAM_TOPIC = "exd_x_uc_lhopital"

_TOL = 1e-6


def _eq(shown: str, value) -> str:
    """`shown = value`, or just the value when `shown` already reads as it (no step that equates a value to itself)."""
    text = rational_text(value)
    return text if shown == text else f"{shown} = {text}"


def _q(n) -> Rational:
    n = Fraction(n)
    return Rational(n.numerator, n.denominator)


def _same(a, b) -> bool:
    return abs(float(a) - float(b)) < 1e-9


def _expression_distractors(answer, candidates, n: int = 4) -> list[Distractor]:
    """Tagged distractors for an expression-graded answer, dropping any that equal the answer or each other."""
    seen = [answer]
    out: list[Distractor] = []
    for value, misconception in candidates:
        value = sympy.sympify(value)
        if any(_same(value, s) for s in seen):
            continue
        seen.append(value)
        out.append(Distractor(value_text(value), misconception, value_string(value)))
        if len(out) == n:
            break
    return out


def _text_shape(text: str) -> str | None:
    if re.fullmatch(r"-?\d+", text.strip()):
        return "integer"
    if re.fullmatch(r"-?\d+/\d+", text.strip()):
        return "fraction"
    return None


def _lone_shape(answer_text: str, distractors) -> bool:
    """Whether the answer is the only integer or fraction among its options, so it can be picked without working."""
    shape = _text_shape(answer_text)
    return shape is not None and all(_text_shape(d.text) != shape for d in distractors)


def _fraction_distractors(answer, candidates, n: int = 4) -> list[Distractor]:
    """The same for a part graded as a fraction (`answer` and every kept candidate rational)."""
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


def _verify(fn: Callable, xs, value, what: str) -> None:
    """The second route: the limit function evaluated in 80-digit arithmetic close to the limit point."""
    target = float(value)
    with mp.workdps(80):
        for x in xs:
            got = float(fn(mp.mpf(x)))
            check(abs(got - target) < _TOL * max(1.0, abs(target)), f"{what}: f({x}) = {got}, expected {target}")


@dataclass
class _Case:
    body: str  # TeX of the whole limit, e.g. \lim_{x\to 0}\frac{...}{...}
    value: sympy.Basic
    fn: Callable
    xs: list
    working: list[str]
    wrongs: list = field(default_factory=list)  # (value, misconception id), most likely first


def _near_zero():
    return [mp.mpf(10) ** -12, mp.mpf(10) ** -14]


def _near_inf():
    return [mp.mpf(10) ** 12, mp.mpf(10) ** 14]


def _near(point: int):
    return [mp.mpf(point) + mp.mpf(10) ** -12, mp.mpf(point) + mp.mpf(10) ** -14]


def _pick(rng: random.Random, values, exclude=()):
    return rng.choice([v for v in values if v not in exclude])


def _kx(a: int) -> str:
    """The argument ax: x, -x, 3x, -3x."""
    return f"{coef(a)}x"


def _ex(a: int) -> str:
    """Plain text for e^(ax): e^x, e^(3x), e^(-3x)."""
    return "e^x" if a == 1 else f"e^({_kx(a)})"


def _one_plus(a: int) -> str:
    """1 + ax with a signed coefficient: 1 + 3x, 1 - 2x, 1 + x."""
    return f"1 {'+' if a > 0 else '-'} {coef(abs(a))}x"


def _add(k: int) -> str:
    """`+ 3` or `- 3`, for appending a signed integer to an expression."""
    return f"+ {k}" if k >= 0 else f"- {-k}"


def _shift(k: int) -> str:
    """x + 2 or x - 2."""
    return f"x {'+' if k > 0 else '-'} {abs(k)}"


# -- easy: 0/0 at a point, one application ----------------------------------------------------------------------

_FIRST = "Numerator and denominator both tend to 0, so L'Hôpital's rule applies."


def _easy_exp(rng):
    a = _pick(rng, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])
    b = _pick(rng, range(1, 8), exclude=(abs(a),))
    value = _q(Fraction(a, b))
    deriv = _ex(a) if a == 1 else f"{coef(a)}{_ex(a)}"
    return _Case(
        rf"\lim_{{x\to 0}}\frac{{e^{{{_kx(a)}}} - 1}}{{{coef(b)}x}}",
        value,
        lambda x: (mp.exp(a * x) - 1) / (b * x),
        _near_zero(),
        [
            _FIRST,
            f"The numerator differentiates to {deriv} and the denominator to {b}.",
            f"The limit of {deriv}/{b} as x → 0 is {rational_text(value)}.",
        ],
        [
            (_q(Fraction(1, b)), "calc-chain-rule-factor-dropped"),
            (_q(Fraction(b, a)), "calc-ratio-inverted"),
            (_q(a), "denominator-coefficient-dropped"),
            (-value, "calc-sign-slip"),
        ],
    )


def _easy_trig(rng):
    a = _pick(rng, [-6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8])
    b = _pick(rng, range(1, 8), exclude=(abs(a),))
    name = rng.choice(["sin", "tan"])
    mpfn = mp.sin if name == "sin" else mp.tan
    value = _q(Fraction(a, b))
    deriv = f"{a}cos({a}x)" if name == "sin" else f"{a}sec²({a}x)"
    return _Case(
        rf"\lim_{{x\to 0}}\frac{{\{name}({_kx(a)})}}{{{coef(b)}x}}",
        value,
        lambda x: mpfn(a * x) / (b * x),
        _near_zero(),
        [
            _FIRST,
            f"The numerator differentiates to {deriv} and the denominator to {b}.",
            f"The limit of {deriv}/{b} as x → 0 is {rational_text(value)}.",
        ],
        [
            (_q(Fraction(1, b)), "calc-chain-rule-factor-dropped"),
            (_q(Fraction(b, a)), "calc-ratio-inverted"),
            (_q(a), "denominator-coefficient-dropped"),
            (-value, "calc-sign-slip"),
        ],
    )


def _easy_ln(rng):
    a = _pick(rng, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])
    b = _pick(rng, range(1, 8), exclude=(abs(a),))
    value = _q(Fraction(a, b))
    return _Case(
        rf"\lim_{{x\to 0}}\frac{{\ln({_one_plus(a)})}}{{{coef(b)}x}}",
        value,
        lambda x: mp.log(1 + a * x) / (b * x),
        _near_zero(),
        [
            _FIRST,
            f"The numerator differentiates to {a}/({_one_plus(a)}) and the denominator to {b}.",
            f"As x → 0 the ratio tends to {_eq(f'{a}/{b}', value)}.",
        ],
        [
            (_q(Fraction(1, b)), "calc-chain-rule-factor-dropped"),
            (_q(Fraction(b, a)), "calc-ratio-inverted"),
            (_q(a), "denominator-coefficient-dropped"),
            (-value, "calc-sign-slip"),
        ],
    )


def _diff_term(b: int) -> str:
    """The derivative of the subtracted term e^(bx), as `- be^(bx)` or `+ |b|e^(bx)`."""
    return f"- {coef(b)}{_ex(b)}" if b > 0 else f"+ {coef(-b)}{_ex(b)}"


def _easy_two_exp(rng):
    a, b = rng.sample([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6], 2)
    c = rng.randint(1, 6)
    value = _q(Fraction(a - b, c))
    return _Case(
        rf"\lim_{{x\to 0}}\frac{{e^{{{_kx(a)}}} - e^{{{_kx(b)}}}}}{{{coef(c)}x}}",
        value,
        lambda x: (mp.exp(a * x) - mp.exp(b * x)) / (c * x),
        _near_zero(),
        [
            _FIRST,
            f"The numerator differentiates to {coef(a)}{_ex(a)} {_diff_term(b)} and the denominator to {c}.",
            f"At x = 0 the ratio is {_eq(f'{a - b}/{c}', value)}.",
        ],
        [
            (_q(Fraction(a + b, c)), "calc-sign-slip"),
            (_q(a - b), "denominator-coefficient-dropped"),
            (_q(Fraction(c, a - b)), "calc-ratio-inverted"),
            (_q(0), "calc-chain-rule-factor-dropped"),
        ],
    )


def _peval(coeffs, x) -> Fraction:
    total = Fraction(0)
    for c in coeffs:
        total = total * x + c
    return total


def _pder(coeffs):
    degree = len(coeffs) - 1
    return [c * (degree - k) for k, c in enumerate(coeffs[:-1])]


def _mp_poly(coeffs):
    def fn(x):
        total = mp.mpf(0)
        for c in coeffs:
            total = total * x + c
        return total

    return fn


def _easy_poly_linear(rng):
    """(x - c)(x^2 + ux + v) over x - c, with x -> c."""
    for _ in attempts(500, "a polynomial limit"):
        c = _pick(rng, [-3, -2, -1, 1, 2, 3, 4])
        u = rng.randint(-3, 3)
        v = rng.randint(-4, 4)
        coeffs = [1, u - c, v - u * c, -v * c]
        d = _pder(coeffs)
        value, at_zero, at_minus = _peval(d, c), _peval(d, 0), _peval(d, -c)
        if u != 0 and v != 0 and len({value, at_zero, at_minus}) == 3:
            break
    slip = _peval([coeffs[0], coeffs[1], coeffs[2]], c)  # every power-rule exponent dropped
    num = _mp_poly(coeffs)
    return _Case(
        rf"\lim_{{x\to {c}}}\frac{{{poly_tex(coeffs)}}}{{{_shift(-c)}}}",
        _q(value),
        lambda x: num(x) / (x - c),
        _near(c),
        [
            f"At x = {c} the numerator and denominator are both 0, so L'Hôpital's rule applies.",
            f"The numerator differentiates to {poly_tex(d)} and the denominator to 1.",
            f"Substituting x = {c} gives {rational_text(value)}.",
        ],
        [
            (_q(at_zero), "evaluated-at-wrong-point"),
            (_q(at_minus), "calc-sign-slip"),
            (_q(slip), "calc-power-rule-exponent-dropped"),
        ],
    )


def _easy_poly_quadratic(rng):
    """(x - c)(mx^2 + ux + v) over x^2 - c^2, with x -> c."""
    for _ in attempts(500, "a polynomial limit"):
        c = rng.randint(1, 4)
        m = rng.choice([1, 2])
        u = rng.randint(-3, 3)
        v = rng.randint(-4, 4)
        coeffs = [m, u - m * c, v - u * c, -v * c]
        n_prime = _peval(_pder(coeffs), c)
        if n_prime != 0:
            break
    value = Fraction(n_prime) / (2 * c)
    num = _mp_poly(coeffs)
    return _Case(
        rf"\lim_{{x\to {c}}}\frac{{{poly_tex(coeffs)}}}{{x^{{2}} - {c * c}}}",
        _q(value),
        lambda x: num(x) / (x * x - c * c),
        _near(c),
        [
            f"At x = {c} the numerator and denominator are both 0, so L'Hôpital's rule applies.",
            f"The numerator differentiates to {poly_tex(_pder(coeffs))} and the denominator to 2x.",
            f"At x = {c} the ratio is {_eq(f'{n_prime}/{2 * c}', value)}.",
        ],
        [
            (_q(Fraction(n_prime, c)), "calc-power-rule-exponent-dropped"),
            (_q(n_prime), "denominator-coefficient-dropped"),
            (-_q(value), "calc-sign-slip"),
        ],
    )


_EASY = [_easy_exp, _easy_trig, _easy_ln, _easy_two_exp, _easy_poly_linear, _easy_poly_quadratic]


# -- medium: two applications, or infinity/infinity ------------------------------------------------------------


def _med_cos(rng):
    a = rng.randint(2, 7)
    b = rng.randint(1, 6)
    value = _q(Fraction(a * a, 2 * b))
    return _Case(
        rf"\lim_{{x\to 0}}\frac{{1 - \cos({a}x)}}{{{coef(b)}x^{{2}}}}",
        value,
        lambda x: (1 - mp.cos(a * x)) / (b * x * x),
        _near_zero(),
        [
            f"The form is 0/0. The first application gives {a}sin({a}x)/({2 * b}x), which is still 0/0.",
            f"The second application gives {a * a}cos({a}x)/{2 * b}.",
            f"At x = 0 the limit is {rational_text(value)}.",
        ],
        [
            (_q(Fraction(a, 2 * b)), "calc-chain-rule-factor-dropped"),
            (_q(0), "stops-after-one-application"),
            (_q(Fraction(a * a, b)), "calc-power-rule-exponent-dropped"),
            (_q(Fraction(2 * b, a * a)), "calc-ratio-inverted"),
        ],
    )


def _med_exp_taylor(rng):
    a = _pick(rng, [-5, -4, -3, -2, -1, 2, 3, 4, 5, 6])
    b = rng.randint(1, 6)
    value = _q(Fraction(a * a, 2 * b))
    return _Case(
        rf"\lim_{{x\to 0}}\frac{{e^{{{_kx(a)}}} - 1 {'-' if a > 0 else '+'} {coef(abs(a))}x}}{{{coef(b)}x^{{2}}}}",
        value,
        lambda x: (mp.exp(a * x) - 1 - a * x) / (b * x * x),
        _near_zero(),
        [
            f"The form is 0/0. The first application gives ({coef(a)}{_ex(a)} {_add(-a)})/({2 * b}x), "
            "which is still 0/0.",
            f"The second application gives {a * a}{_ex(a)}/{2 * b}.",
            f"At x = 0 the limit is {rational_text(value)}.",
        ],
        [
            (_q(Fraction(abs(a), 2 * b)), "calc-chain-rule-factor-dropped"),
            (_q(0), "stops-after-one-application"),
            (_q(Fraction(a * a, b)), "calc-power-rule-exponent-dropped"),
            (-value, "calc-sign-slip"),
        ],
    )


def _med_ln_taylor(rng):
    a = _pick(rng, [-5, -4, -3, -2, -1, 2, 3, 4, 5, 6])
    b = rng.randint(1, 6)
    value = _q(Fraction(-a * a, 2 * b))
    return _Case(
        rf"\lim_{{x\to 0}}\frac{{\ln({_one_plus(a)}) {'-' if a > 0 else '+'} {coef(abs(a))}x}}{{{coef(b)}x^{{2}}}}",
        value,
        lambda x: (mp.log(1 + a * x) - a * x) / (b * x * x),
        _near_zero(),
        [
            f"The form is 0/0. The first application gives ({a}/({_one_plus(a)}) {_add(-a)})/({2 * b}x), "
            "which is still 0/0.",
            f"The second application gives {-a * a}/(({_one_plus(a)})² × {2 * b}).",
            f"At x = 0 the limit is {rational_text(value)}.",
        ],
        [
            (-value, "calc-sign-slip"),
            (_q(0), "stops-after-one-application"),
            (_q(Fraction(-abs(a), 2 * b)), "calc-chain-rule-factor-dropped"),
            (_q(Fraction(-a * a, b)), "calc-power-rule-exponent-dropped"),
        ],
    )


def _med_product(rng):
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    value = _q(Fraction(2 * a, b * b))
    ax, bx = f"{coef(a)}x", f"{coef(b)}x"
    return _Case(
        rf"\lim_{{x\to 0}}\frac{{x\,\sin({ax})}}{{1 - \cos({bx})}}",
        value,
        lambda x: x * mp.sin(a * x) / (1 - mp.cos(b * x)),
        _near_zero(),
        [
            f"The form is 0/0. The first application gives (sin({ax}) + {coef(a)}x cos({ax}))/({coef(b)}sin({bx})), "
            "which is still 0/0.",
            f"The second application gives ({coef(2 * a)}cos({ax}) - {coef(a * a)}x sin({ax}))/({coef(b * b)}cos({bx})).",
            f"At x = 0 the limit is {_eq(f'{2 * a}/{b * b}', value)}.",
        ],
        [
            (_q(Fraction(a, b * b)), "product-rule-term-dropped"),
            (_q(0), "stops-after-one-application"),
            (_q(Fraction(2 * a, b)), "calc-chain-rule-factor-dropped"),
            (_q(Fraction(b * b, 2 * a)), "calc-ratio-inverted"),
        ],
    )


def _med_exp_infinity(rng):
    k = rng.choice([1, 2, 3])
    p, q = rng.sample(range(1, 8), 2)
    s = rng.randint(2, 5)
    value = _q(Fraction(p, q))
    return _Case(
        rf"\lim_{{x\to\infty}}\frac{{{coef(p)}e^{{{coef(k)}x}} + x^{{2}}}}{{{coef(q)}e^{{{coef(k)}x}} + {s}x}}",
        value,
        lambda x: (p * mp.exp(k * x) + x * x) / (q * mp.exp(k * x) + s * x),
        [mp.mpf(200), mp.mpf(300)],
        [
            "Numerator and denominator both tend to ∞, so L'Hôpital's rule applies.",
            f"The first application gives ({coef(p * k)}{_ex(k)} + 2x)/({coef(q * k)}{_ex(k)} + {s}), "
            "which is still ∞/∞.",
            f"The second application gives ({coef(p * k * k)}{_ex(k)} + 2)/({coef(q * k * k)}{_ex(k)}), "
            f"which tends to {p}/{q}.",
        ],
        [
            (_q(Fraction(p * k, q)), "calc-chain-rule-factor-dropped"),
            (_q(Fraction(q, p)), "calc-ratio-inverted"),
            (_q(0), "quotient-rule-instead-of-lhopital"),
        ],
    )


def _med_log_infinity(rng):
    m, n = rng.sample(range(1, 5), 2)
    a = rng.choice([1, 2, 3])
    b = rng.choice([1, 2, 3])
    c = rng.randint(1, 9)
    d = rng.randint(1, 9)
    value = _q(Fraction(m, n))

    def term(coefficient, power, constant):
        return f"{coef(coefficient)}x" + (f"^{{{power}}}" if power > 1 else "") + f" + {constant}"

    def plain(coefficient, power, constant):
        return f"{coef(coefficient)}x" + ("" if power == 1 else f"^{power}") + f" + {constant}"

    def derivative(coefficient, power):
        k = coefficient * power
        return str(k) if power == 1 else f"{k}x" if power == 2 else f"{k}x^{power - 1}"

    big = mp.mpf(10) ** (10**9)
    return _Case(
        rf"\lim_{{x\to\infty}}\frac{{\ln({term(a, m, c)})}}{{\ln({term(b, n, d)})}}",
        value,
        lambda x: mp.log(a * x**m + c) / mp.log(b * x**n + d),
        [big, big**2],
        [
            "Both logarithms tend to ∞, so L'Hôpital's rule applies.",
            f"The numerator differentiates to {derivative(a, m)}/({plain(a, m, c)}) and the denominator to "
            f"{derivative(b, n)}/({plain(b, n, d)}).",
            f"For large x the numerator's derivative behaves like {m}/x and the denominator's like {n}/x, "
            f"so the ratio tends to {m}/{n}.",
        ],
        [
            (_q(Fraction(n, m)), "calc-ratio-inverted"),
            (_q(1), "log-exponent-ignored"),
            (_q(Fraction(a * m, b * n)), "leading-coefficients-kept"),
            (_q(m), "denominator-coefficient-dropped"),
        ],
    )


def _med_log_linear(rng):
    for _ in attempts(200, "a log-against-linear limit"):
        a, c = rng.randint(1, 6), rng.randint(1, 6)
        b, d = rng.randint(1, 6), rng.randint(1, 6)
        if Fraction(b, d) != Fraction(a, c) and b != d:
            break
    value = _q(Fraction(b, d))
    return _Case(
        rf"\lim_{{x\to\infty}}\frac{{{coef(a)}\ln x + {coef(b)}x}}{{{coef(c)}\ln x + {coef(d)}x}}",
        value,
        lambda x: (a * mp.log(x) + b * x) / (c * mp.log(x) + d * x),
        [mp.mpf(10) ** 12, mp.mpf(10) ** 14],
        [
            "Numerator and denominator both tend to ∞, so L'Hôpital's rule applies.",
            f"The first application gives ({a}/x + {b})/({c}/x + {d}).",
            f"As x → ∞ the terms in 1/x vanish, so the limit is {_eq(f'{b}/{d}', value)}.",
        ],
        [
            (_q(Fraction(a, c)), "dominant-term-misjudged"),
            (_q(Fraction(d, b)), "calc-ratio-inverted"),
            (_q(Fraction(a + b, c + d)), "dominant-term-misjudged"),
            (_q(0), "quotient-rule-instead-of-lhopital"),
        ],
    )


_MEDIUM = [
    _med_cos,
    _med_exp_taylor,
    _med_ln_taylor,
    _med_product,
    _med_exp_infinity,
    _med_log_infinity,
    _med_log_linear,
]


# -- hard: rewrite first (0 x infinity, infinity - infinity, 1^infinity) -----------------------------------------


def _hard_x_exp(rng):
    a = _pick(rng, [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
    b = rng.choice([1, 1, 2, 3, 4])
    value = _q(a * b)
    factor = f"{b} × " if b > 1 else ""
    return _Case(
        rf"\lim_{{x\to\infty}} {coef(b)}x\left(e^{{{a}/x}} - 1\right)",
        value,
        lambda x: b * x * (mp.exp(a / x) - 1),
        _near_inf(),
        [
            f"The form is ∞ × 0. Write {coef(b)}x(e^({a}/x) - 1) as {coef(b)}(e^({a}/x) - 1)/(1/x), which is 0/0.",
            f"The numerator differentiates to {-a}e^({a}/x)/x² and the denominator to -1/x².",
            f"The ratio is {coef(a)}e^({a}/x), which tends to {a}, so the limit is {factor}{a} = {a * b}.",
        ],
        [
            (_q(0), "zero-times-infinity-is-zero"),
            (-value, "calc-sign-slip"),
            (_q(a), "calc-constant-factor-dropped"),
            (_q(b), "calc-chain-rule-factor-dropped"),
        ],
    )


def _hard_x_ln(rng):
    a, b = rng.sample([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], 2)
    value = _q(a - b)
    return _Case(
        rf"\lim_{{x\to\infty}} x\,\ln\left(\frac{{{_shift(a)}}}{{{_shift(b)}}}\right)",
        value,
        lambda x: x * mp.log((x + a) / (x + b)),
        _near_inf(),
        [
            f"The form is ∞ × 0. Write x ln(({_shift(a)})/({_shift(b)})) as ln(({_shift(a)})/({_shift(b)}))/(1/x), "
            "which is 0/0.",
            f"The numerator differentiates to 1/({_shift(a)}) - 1/({_shift(b)}) and the denominator to -1/x².",
            f"The ratio is {a - b}x²/(({_shift(a)})({_shift(b)})), which tends to {a - b}.",
        ],
        [
            (_q(0), "zero-times-infinity-is-zero"),
            (_q(b - a), "calc-sign-slip"),
            (_q(a), "one-term-of-difference-only"),
            (_q(a + b), "calc-sign-slip"),
        ],
    )


def _hard_x2_exp(rng):
    a = rng.randint(1, 6)
    value = _q(Fraction(a * a, 2))
    return _Case(
        rf"\lim_{{x\to\infty}} x^{{2}}\left(e^{{{a}/x}} - 1 - \frac{{{a}}}{{x}}\right)",
        value,
        lambda x: x * x * (mp.exp(a / x) - 1 - a / x),
        _near_inf(),
        [
            f"The form is ∞ × 0. Write it as (e^({a}/x) - 1 - {a}/x)/(1/x²), which is 0/0.",
            f"The first application gives ({coef(a)}x/2)(e^({a}/x) - 1), which is again ∞ × 0.",
            f"Writing that as ({a}/2)(e^({a}/x) - 1)/(1/x) and applying the rule again gives "
            f"{_eq(f'({a}/2) × {a}', value)}.",
        ],
        [
            (_q(0), "zero-times-infinity-is-zero"),
            (_q(Fraction(a, 2)), "stops-after-one-application"),
            (_q(a * a), "calc-power-rule-exponent-dropped"),
            (-value, "calc-sign-slip"),
        ],
    )


def _hard_diff_ln(rng):
    a = rng.randint(1, 3)
    b = rng.randint(1, 9)
    value = _q(Fraction(-b, 2))
    ax = f"{coef(a)}x"
    return _Case(
        rf"\lim_{{x\to 0}}\left(\frac{{{b}}}{{{ax}}} - \frac{{{b}}}{{\ln({_one_plus(a)})}}\right)",
        value,
        lambda x: b / (a * x) - b / mp.log(1 + a * x),
        _near_zero(),
        [
            f"The form is ∞ - ∞. Combine over a common denominator: {b}(ln({_one_plus(a)}) - {ax})/"
            f"({ax} ln({_one_plus(a)})), which is 0/0.",
            f"The first application gives {coef(-a * a * b)}x/({_one_plus(a)}) over "
            f"{coef(a)}ln({_one_plus(a)}) + {coef(a * a)}x/({_one_plus(a)}), still 0/0.",
            f"The second application gives {-a * a * b} over {2 * a * a} at x = 0, which is {rational_text(value)}.",
        ],
        [
            (_q(0), "infinity-minus-infinity-is-zero"),
            (_q(Fraction(b, 2)), "calc-sign-slip"),
            (_q(-b), "calc-power-rule-exponent-dropped"),
            (_q(Fraction(-b, 2 * a)), "calc-constant-factor-dropped"),
        ],
    )


def _hard_diff_x2(rng):
    a = rng.randint(1, 6)
    value = _q(Fraction(a * a, 2))
    return _Case(
        rf"\lim_{{x\to\infty}}\left({coef(a)}x - x^{{2}}\ln\left(1 + \frac{{{a}}}{{x}}\right)\right)",
        value,
        lambda x: a * x - x * x * mp.log(1 + a / x),
        _near_inf(),
        [
            f"The form is ∞ - ∞. Factor out x²: x²({a}/x - ln(1 + {a}/x)), which is ∞ × 0.",
            f"Write it as ({a}/x - ln(1 + {a}/x))/(1/x²), which is 0/0, and apply the rule.",
            f"The numerator differentiates to {-a * a}/(x²(x + {a})) and the denominator to -2/x³, so the ratio is "
            f"{a * a}x/(2(x + {a})), which tends to {rational_text(value)}.",
        ],
        [
            (_q(0), "infinity-minus-infinity-is-zero"),
            (-value, "calc-sign-slip"),
            (_q(Fraction(a, 2)), "stops-after-one-application"),
            (_q(a * a), "calc-power-rule-exponent-dropped"),
        ],
    )


def _hard_diff_log_one(rng):
    a = rng.randint(1, 9)
    sign = rng.choice([1, -1])
    value = _q(Fraction(sign * a, 2))
    first, second = (rf"\frac{{{a}}}{{\ln x}}", rf"\frac{{{a}}}{{x - 1}}")
    body = (
        rf"\lim_{{x\to 1}}\left({first} - {second}\right)"
        if sign == 1
        else rf"\lim_{{x\to 1}}\left({second} - {first}\right)"
    )
    s = "" if sign == 1 else "-"
    combined = f"{a}(x - 1 - ln x)" if sign == 1 else f"{a}(ln x - x + 1)"
    return _Case(
        body,
        value,
        lambda x: sign * (a / mp.log(x) - a / (x - 1)),
        _near(1),
        [
            f"The form is ∞ - ∞. Combining gives {combined}/((x - 1) ln x), which is 0/0.",
            f"The first application gives {s}{a}(1 - 1/x)/(ln x + (x - 1)/x), which is still 0/0.",
            f"The second application gives {s}{a}(1/x²)/(1/x + 1/x²), which is {rational_text(value)} at x = 1.",
        ],
        [
            (_q(0), "infinity-minus-infinity-is-zero"),
            (-value, "calc-sign-slip"),
            (_q(sign * a), "stops-after-one-application"),
            (_q(2 * sign * a), "calc-ratio-inverted"),
        ],
    )


def _hard_pow_infinity(rng):
    a = _pick(rng, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    b = rng.choice([2, 3, 4])
    times = rng.random() < 0.5
    k = Fraction(a * b) if times else Fraction(a, b)
    exponent_tex = f"{b}x" if times else rf"\frac{{x}}{{{b}}}"
    exponent_plain = f"{b}x" if times else f"x/{b}"
    inner_tex = rf"1 + \frac{{{a}}}{{x}}" if a > 0 else rf"1 - \frac{{{-a}}}{{x}}"
    inner_plain = f"1 {'+' if a > 0 else '-'} {abs(a)}/x"
    value = exp(_q(k))
    mpk = mp.mpf(k.numerator) / k.denominator
    return _Case(
        rf"\lim_{{x\to\infty}}\left({inner_tex}\right)^{{{exponent_tex}}}",
        value,
        lambda x: (1 + a / x) ** (mpk / a * x),
        _near_inf(),
        [
            f"The form is 1^∞. Let L be the limit and take logarithms: ln L = lim {exponent_plain} × "
            f"ln({inner_plain}), which is ∞ × 0.",
            f"Written as a quotient this is 0/0, and the rule gives {rational_text(k)}/({inner_plain}), "
            f"which tends to {rational_text(k)}.",
            f"So ln L = {rational_text(k)} and L = e^({rational_text(k)}).",
        ],
        [
            (_q(1), "one-to-infinity-is-one"),
            (_q(k), "forgets-to-exponentiate-log-limit"),
            (exp(_q(a)), "exponent-factor-dropped"),
            (exp(-_q(k)), "calc-sign-slip"),
        ],
    )


def _hard_cos_power(rng):
    a = rng.randint(1, 5)
    k = Fraction(-a * a, 2)
    value = exp(_q(k))
    ax = f"{coef(a)}x"
    return _Case(
        rf"\lim_{{x\to 0}}\left(\cos({ax})\right)^{{1/x^{{2}}}}",
        value,
        lambda x: mp.cos(a * x) ** (1 / (x * x)),
        _near_zero(),
        [
            f"The form is 1^∞. Let L be the limit; then ln L = lim ln(cos({ax}))/x², which is 0/0.",
            f"The first application gives -{coef(a)}tan({ax})/(2x), which is still 0/0.",
            f"The second application gives -{coef(a * a)}sec²({ax})/2, so ln L = {rational_text(k)} "
            f"and L = e^({rational_text(k)}).",
        ],
        [
            (_q(1), "one-to-infinity-is-one"),
            (_q(k), "forgets-to-exponentiate-log-limit"),
            (exp(-_q(k)), "calc-sign-slip"),
            (exp(_q(2 * k)), "calc-power-rule-exponent-dropped"),
        ],
    )


def _hard_exp_sum_power(rng):
    for _ in attempts(200, "an exponent of size at least 2"):
        a = _pick(rng, [-4, -3, -2, -1, 1, 2, 3, 4])
        b = _pick(rng, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        if abs(a + b) >= 2 and a != b:
            break
    k = a + b
    value = exp(_q(k))
    linear = f"{'+' if b > 0 else '-'} {coef(abs(b))}x"
    return _Case(
        rf"\lim_{{x\to 0}}\left(e^{{{_kx(a)}}} {linear}\right)^{{1/x}}",
        value,
        lambda x: (mp.exp(a * x) + b * x) ** (1 / x),
        _near_zero(),
        [
            f"The form is 1^∞. Let L be the limit; then ln L = lim ln({_ex(a)} {linear})/x, which is 0/0.",
            f"The rule gives ({coef(a)}{_ex(a)} {_add(b)})/({_ex(a)} {linear}), which is {k} at x = 0.",
            f"So ln L = {k} and L = e^{k}." if k > 0 else f"So ln L = {k} and L = e^({k}).",
        ],
        [
            (_q(1), "one-to-infinity-is-one"),
            (_q(k), "forgets-to-exponentiate-log-limit"),
            (exp(_q(a)), "linear-term-ignored"),
            (exp(_q(a * b)), "sum-product-confusion"),
        ],
    )


def _hard_sinc_power(rng):
    a = rng.randint(1, 4)
    k = Fraction(-a * a, 6)
    value = exp(_q(k))
    ax = f"{coef(a)}x"
    return _Case(
        rf"\lim_{{x\to 0}}\left(\frac{{\sin({ax})}}{{{ax}}}\right)^{{1/x^{{2}}}}",
        value,
        lambda x: (mp.sin(a * x) / (a * x)) ** (1 / (x * x)),
        _near_zero(),
        [
            f"The form is 1^∞. Let L be the limit; then ln L = lim ln(sin({ax})/({ax}))/x², which is 0/0.",
            f"The first application gives ({coef(a)}x cos({ax}) - sin({ax}))/(2x² sin({ax})), still 0/0.",
            f"Two more applications give ln L = {rational_text(k)}, so L = e^({rational_text(k)}).",
        ],
        [
            (_q(1), "one-to-infinity-is-one"),
            (_q(k), "forgets-to-exponentiate-log-limit"),
            (exp(-_q(k)), "calc-sign-slip"),
            (exp(_q(2 * k)), "calc-power-rule-exponent-dropped"),
        ],
    )


_HARD = [
    _hard_x_exp,
    _hard_x_ln,
    _hard_x2_exp,
    _hard_diff_ln,
    _hard_diff_x2,
    _hard_diff_log_one,
    _hard_pow_infinity,
    _hard_cos_power,
    _hard_exp_sum_power,
    _hard_sinc_power,
]

_TIERS = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}

_PROMPTS = [
    ("Evaluate ", "."),
    ("Find ", "."),
    ("Determine the exact value of ", "."),
]


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_lhopital(rng: random.Random, tier: str = "easy") -> dict:
    """A limit that needs L'Hôpital's rule: easy one application, medium two (or ∞/∞), hard rewrite first."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    case: _Case = rng.choice(_TIERS[tier])(rng)
    _verify(case.fn, case.xs, case.value, case.body)
    lead, tail = rng.choice(_PROMPTS)
    text, latex = say(lead, M(case.body), tail)
    distractors = _expression_distractors(case.value, case.wrongs)
    if _lone_shape(value_text(case.value), distractors):
        # no wrong option reads like the answer (the only fraction, say): add the sign slip, a real error of the same shape
        distractors = _expression_distractors(case.value, [(-case.value, "calc-sign-slip"), *case.wrongs])
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

# Each family is f(x) = (g(px) - qx)/x^order. kappa is the limit's coefficient: lim f = kappa * p^order when q = p.
_CHAINS = {"sin": (3, Fraction(-1, 6)), "exp": (2, Fraction(1, 2)), "ln": (2, Fraction(-1, 2))}


def _g(name: str, p: int, x):
    if name == "sin":
        return mp.sin(p * x)
    if name == "exp":
        return mp.exp(p * x) - 1
    return mp.log(1 + p * x)


def _g_tex(name: str, p: int) -> str:
    if name == "sin":
        return rf"\sin({_kx(p)})"
    if name == "exp":
        return rf"e^{{{_kx(p)}}} - 1"
    return rf"\ln({_one_plus(p)})"


def _ratio_tex(name: str, p: int) -> str:
    if name == "sin":
        return rf"\frac{{\sin({_kx(p)})}}{{{_kx(p)}}}"
    if name == "exp":
        return rf"\frac{{e^{{{_kx(p)}}} - 1}}{{{_kx(p)}}}"
    return rf"\frac{{\ln({_one_plus(p)})}}{{{_kx(p)}}}"


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_uc_lhopital(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: find q so that lim (g(px) - qx)/x^n exists, evaluate it, scale it, then a 1^∞ limit."""
    check(tier == "exam", f"unknown tier {tier!r}")
    name = rng.choice(sorted(_CHAINS))
    n, kappa = _CHAINS[name]
    p = _pick(rng, [2, 3, 4, 5, 6] if name == "sin" else [-4, -3, -2, -1, 1, 2, 3, 4, 5])
    r = rng.choice([2, 3, 4, 5, 6])
    limit_b = _q(kappa * p**n)
    exponent_d = limit_b / p
    value_d = exp(exponent_d)

    # Second route for every part: the functions evaluated in 80-digit arithmetic near the limit point.
    with mp.workdps(80):
        for x in _near_zero():
            f_b = (_g(name, p, x) - p * x) / x**n
            check(abs(float(f_b) - float(limit_b)) < 1e-6 * max(1, abs(float(limit_b))), "part (b) disagrees")
            f_c = (_g(name, p, x) - p * x) / (r * x**n)
            check(abs(float(f_c) - float(limit_b / r)) < 1e-6 * max(1, abs(float(limit_b / r))), "part (c) disagrees")
            f_d = (_g(name, p, x) / (p * x)) ** (1 / x ** (n - 1))
            check(abs(float(f_d) - float(value_d)) < 1e-5 * float(value_d), "part (d) disagrees")
        x = mp.mpf(10) ** -10
        for q_bad in (p + 1, p - 1, 0):
            check(abs((_g(name, p, x) - q_bad * x) / x**n) > 1e5, "a constant other than p gives a finite limit")

    g_tex = _g_tex(name, p)
    power = "x^{" + str(n) + "}"
    stem_text, stem_latex = say(
        "Let f(x) = ", M(rf"\frac{{{g_tex} - qx}}{{{power}}}"), " for x ≠ 0, where q is a constant."
    )
    p_a, tex_a = say("Find the value of q for which ", M(r"\lim_{x\to 0} f(x)"), " exists as a finite number.")
    p_b, tex_b = say("For this value of q, evaluate ", M(r"\lim_{x\to 0} f(x)"), ".")
    p_c, tex_c = say(
        "Using the same value of q, evaluate ", M(rf"\lim_{{x\to 0}}\frac{{{g_tex} - qx}}{{{coef(r)}{power}}}"), "."
    )
    root = "x" if n == 2 else f"x^{{{n - 1}}}"
    p_d, tex_d = say(
        "Hence, or otherwise, evaluate ", M(rf"\lim_{{x\to 0}}\left({_ratio_tex(name, p)}\right)^{{1/{root}}}"), "."
    )

    kp = _q(kappa * p**n)
    cands_a = [
        (_q(0), "any-constant-works-at-zero-over-zero"),
        (_q(-p), "calc-sign-slip"),
        (kp, "matches-wrong-order-coefficient"),
        (_q(Fraction(1, p)), "calc-ratio-inverted"),
    ]
    if name == "sin":
        cands_b = [
            (_q(-(p**3)), "factorial-factor-dropped"),
            (_q(Fraction(p**3, 6)), "calc-sign-slip"),
            (_q(Fraction(-(p**3), 3)), "calc-power-rule-exponent-dropped"),
            (_q(0), "stops-after-one-application"),
        ]
        working_b = (
            f"Applying the rule three times, the numerator's third derivative is {-(p**3)}cos({_kx(p)}) and the "
            f"denominator's is 6, so the limit is {rational_text(limit_b)}."
        )
        working_a = (
            f"The form is 0/0. After one application the numerator is {p}cos({_kx(p)}) - q, which tends to {p} - q, "
            f"and the denominator 3x² tends to 0, so {p} - q must be 0 and q = {p}."
        )
    elif name == "exp":
        cands_b = [
            (_q(p * p), "calc-power-rule-exponent-dropped"),
            (_q(Fraction(p, 2)), "calc-chain-rule-factor-dropped"),
            (_q(Fraction(-p * p, 2)), "calc-sign-slip"),
            (_q(0), "stops-after-one-application"),
        ]
        working_b = (
            f"Applying the rule twice, the numerator's second derivative is {p * p}{_ex(p)} and the denominator's "
            f"is 2, so the limit is {rational_text(limit_b)}."
        )
        working_a = (
            f"The form is 0/0. After one application the numerator is {coef(p)}{_ex(p)} - q, which tends to {p} - q, "
            f"and the denominator 2x tends to 0, so {p} - q must be 0 and q = {p}."
        )
    else:
        cands_b = [
            (_q(-p * p), "calc-power-rule-exponent-dropped"),
            (_q(Fraction(p * p, 2)), "calc-sign-slip"),
            (_q(Fraction(-p, 2)), "calc-chain-rule-factor-dropped"),
            (_q(0), "stops-after-one-application"),
        ]
        working_b = (
            f"Applying the rule twice, the numerator's second derivative is {-p * p}/({_one_plus(p)})² and the "
            f"denominator's is 2, so the limit is {rational_text(limit_b)}."
        )
        working_a = (
            f"The form is 0/0. After one application the numerator is {p}/({_one_plus(p)}) - q, which tends to "
            f"{p} - q, and the denominator 2x tends to 0, so {p} - q must be 0 and q = {p}."
        )
    cands_c = [
        (limit_b, "calc-constant-factor-dropped"),
        (limit_b * r, "scaled-the-wrong-way"),
        (_q(0), "stops-after-one-application"),
        (-limit_b / r, "calc-sign-slip"),
    ]
    cands_d = [
        (_q(1), "one-to-infinity-is-one"),
        (exponent_d, "forgets-to-exponentiate-log-limit"),
        (exp(limit_b), "exponent-factor-dropped"),
        (exp(-exponent_d), "calc-sign-slip"),
    ]
    dist_a = _fraction_distractors(_q(p), cands_a)
    dist_b = _fraction_distractors(limit_b, cands_b)
    dist_c = _fraction_distractors(limit_b / r, cands_c)
    dist_d = _expression_distractors(value_d, cands_d)
    check(min(len(dist_a), len(dist_b), len(dist_c), len(dist_d)) >= 2, "too few distractors")

    return dict(
        text=stem_text,
        latex=stem_latex,
        parts=[
            dict(
                prompt=p_a,
                prompt_latex=tex_a,
                answer_kind="fraction",
                answer=exact_value(_q(p)),
                answer_text=rational_text(_q(p)),
                answer_latex=latex_answer(_q(p)),
                marks=2,
                working=[working_a],
                distractors=dist_a,
            ),
            dict(
                prompt=p_b,
                prompt_latex=tex_b,
                answer_kind="fraction",
                answer=exact_value(limit_b),
                answer_text=rational_text(limit_b),
                answer_latex=latex_answer(limit_b),
                marks=3,
                working=[working_b],
                distractors=dist_b,
            ),
            dict(
                prompt=p_c,
                prompt_latex=tex_c,
                answer_kind="fraction",
                answer=exact_value(limit_b / r),
                answer_text=rational_text(limit_b / r),
                answer_latex=latex_answer(limit_b / r),
                marks=2,
                follow_through=f"part_b/{r}",
                working=[f"The constant {r} comes outside the limit: ({rational_text(limit_b)})/{r}."],
                distractors=dist_c,
            ),
            dict(
                prompt=p_d,
                prompt_latex=tex_d,
                answer_kind="expression",
                answer=value_string(value_d),
                answer_text=value_text(value_d),
                answer_latex=latex_answer(value_d),
                marks=3,
                follow_through=f"exp(part_b/({p}))",
                working=[
                    "The form is 1^∞, so take logarithms: ln L = lim ln(ratio)/x^k, where the ratio is "
                    "1 + (g(px) - px)/(px) and ln(1 + u) behaves like u.",
                    f"That makes ln L = (the limit in part (b))/{p} = {rational_text(exponent_d)}, so "
                    f"L = e^({rational_text(exponent_d)}).",
                ],
                distractors=dist_d,
            ),
        ],
    )
