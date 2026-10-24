"""Integrals of powers and products of tan and sec: `sc_integrate_tan_sec` (easy, medium, hard) and
`sc_integrate_tan_sec_exam`.

Skill: ``integrate-tan-sec-powers`` (``integration``, answer ``expression``). The existing ``integrate-trig-powers`` covers
sin^n and cos^n; this one covers the identities and substitutions that tan and sec need:

* easy: ``c tan^2(kx)`` (by tan^2 = sec^2 - 1) and ``c sec(kx) tan(kx)`` (the derivative of sec).
* medium: by substitution, ``sec^n(kx) tan(kx)`` (u = sec, answer in sec) and ``tan^m(kx) sec^2(kx)`` (u = tan, answer in
  tan).
* hard: ``tan^4 x``, ``sec^4 x`` and ``tan^2 x sec^4 x`` (the identity, then u = tan), ``sec^m x tan^3 x`` (split off
  sec x tan x, then u = sec), or the exact value of a definite integral of such a function between 0 and pi/4, pi/3
  or pi/6.
* exam (``sc_integrate_tan_sec_exam``): over 0 to pi/4 the reduction I(n) = 1/(n - 1) - I(n - 2) for I(n) = the integral
  of tan^n. (a) I(s), (b) I(s + 2) from (a), (c) I(s + 4) from (b), (d) a combination of two of them, for s = 1 (answers
  with ln 2) or s = 2 (answers with pi).

Every antiderivative is differentiated back and every exact value is compared with Simpson's rule. An indefinite
integral that could be written in more than one form differing by a constant says which function to answer in (``Give your
answer in terms of tan x``), because the grader compares expressions, not answers up to a constant.
"""

from __future__ import annotations

import re
from sympy import Rational, log, pi, sec, sqrt, tan

from ..core import check, generator, latex_from_inline
from ._spread_kit import uses_rng
from ._r3b_kit import (
    Cand,
    Itg,
    Redraw,
    check_antiderivative,
    check_definite,
    exact_fields,
    expr_text,
    integral_math,
    integral_text,
    key_fields,
    part,
    pick_expr,
    retry,
    say,
    x,
)

_FIND = (
    "Find {I}.",
    "Determine {I}.",
    "Find the indefinite integral {I}.",
)
_EXACT = (
    "Find the exact value of {I}.",
    "Evaluate {I}, giving an exact answer.",
)


def _tidy(line: str) -> str:
    """A worked step with SymPy's spelling made learner-readable: sqrt(3) as √3, 3*pi as 3pi, any other * as ×."""
    line = re.sub(r"sqrt\((\d+)\)", r"√\1", line)
    line = re.sub(r"(?<=[\d)])\*(?=[A-Za-z√(])", "", line)
    return line.replace("*", " × ")


def _arg(k) -> str:
    return "x" if k == 1 else f"{k}x"


def _fn(name: str, n: int, k=1):
    """(plain text, LaTeX) of name^n(kx)."""
    arg = _arg(k)
    if n == 1:
        return f"{name}({arg})", rf"\{name}({arg})"
    return f"{name}^{n}({arg})", rf"\{name}^{{{n}}}({arg})"


def _t(k=1):
    return tan(k * x)


def _s(k=1):
    return sec(k * x)


def _coef(c):
    return ("", "") if c == 1 else (f"{c} ", f"{c}")


def _itg(value, *pieces) -> Itg:
    return Itg(value, text="".join(t for t, _ in pieces), latex="".join(m for _, m in pieces))


def _indef(rng, f: Itg, F, cands, working, note=""):
    check_antiderivative(F, f.value)
    text, latex = say(rng, _FIND, {"I": integral_text(f)}, {"I": integral_math(f)}, suffix=note)
    return dict(
        text=text,
        working=[_tidy(w) for w in working],
        latex=latex,
        distractors=pick_expr(F, True, cands, rng=rng),
        **key_fields(F),
    )


def _no_constant(cands):
    return [Cand(c.expr, c.tag, const=False) for c in cands if c is not None]


def _definite(rng, f: Itg, value, lo, hi, cands, working):
    check_definite(value, f.value, lo, hi)
    text, latex = say(rng, _EXACT, {"I": integral_text(f, lo, hi)}, {"I": integral_math(f, lo, hi)})
    return dict(
        text=text,
        working=[_tidy(w) for w in working],
        latex=latex,
        distractors=pick_expr(value, False, _no_constant(cands), rng=rng),
        **exact_fields(value),
    )


# ------------------------------------------------------------------------------------------------ easy


def _easy_tan_squared(rng):
    k = rng.choice([1, 1, 2, 3])
    c = rng.choice([1, 2, 3, 4, 6])
    f = _itg(c * _t(k) ** 2, _coef(c), _fn("tan", 2, k))
    F = c * (_t(k) / k - x)
    cands = [
        Cand(c * _t(k) / k, "sc-tan-sq-taken-as-sec-sq"),
        Cand(c * (_t(k) / k + x), "sc-tan-sq-identity-sign"),
        Cand(c * (_t(k) - x), "sc-tan-inner-derivative-ignored") if k > 1 else None,
        Cand(c * _t(k) ** 3 / (3 * k), "sc-tan-power-rule-on-trig"),
        Cand(c * (_t(k) / k - 1), "sc-tan-sq-constant-not-integrated"),
    ]
    working = [
        "Use tan^2 A = sec^2 A - 1.",
        f"∫ sec^2({_arg(k)}) dx = tan({_arg(k)})/{k}, and ∫ 1 dx = x, so the result is {c if c != 1 else ''}[tan({_arg(k)})/{k} - x] + C."
        if k > 1
        else f"∫ sec^2 x dx = tan x and ∫ 1 dx = x, so the result is {c if c != 1 else ''}[tan x - x] + C.",
    ]
    return _indef(rng, f, F, cands, working)


def _easy_sec_tan(rng):
    k = rng.choice([1, 2, 3, 4])
    c = rng.choice([1, 2, 3, 5, 6])
    f = _itg(c * _s(k) * _t(k), _coef(c), _fn("sec", 1, k), (" ", " "), _fn("tan", 1, k))
    F = c * _s(k) / k
    cands = [
        Cand(c * _s(k), "sc-sec-tan-inner-derivative-ignored") if k > 1 else None,
        Cand(-c * _s(k) / k, "sc-sec-tan-sign-slip"),
        Cand(c * _t(k) ** 2 / (2 * k), "sc-sec-tan-integrated-as-tan-squared"),
        Cand(c * _s(k) ** 2 / (2 * k), "sc-sec-tan-power-rule"),
        Cand(c * _t(k) / k, "sc-sec-tan-sec-dropped"),
    ]
    working = [
        f"d/dx sec({_arg(k)}) = {k if k > 1 else ''}sec({_arg(k)}) tan({_arg(k)}), so sec({_arg(k)}) tan({_arg(k)}) integrates to sec({_arg(k)})/{k}."
        if k > 1
        else "d/dx sec x = sec x tan x, so sec x tan x integrates to sec x.",
        f"Multiply by {c}." if c != 1 else "Add the constant of integration.",
    ]
    return _indef(rng, f, F, cands, working)


def _easy(rng):
    return rng.choice([_easy_tan_squared, _easy_sec_tan])(rng)


# ------------------------------------------------------------------------------------------------ medium


def _medium_sec_power_tan(rng):
    n = rng.choice([2, 3, 4, 5])
    k = rng.choice([1, 1, 2, 3])
    f = _itg(_s(k) ** n * _t(k), _fn("sec", n, k), (" ", " "), _fn("tan", 1, k))
    F = _s(k) ** n / (n * k)
    cands = [
        Cand(_s(k) ** (n + 1) / ((n + 1) * k), "sc-sec-power-rule-wrong-exponent"),
        Cand(_s(k) ** n / k, "sc-sec-substituted-power-not-divided"),
        Cand(_s(k) ** n / n, "sc-sec-tan-inner-derivative-ignored") if k > 1 else None,
        Cand(-(_s(k) ** n) / (n * k), "sc-sec-tan-sign-slip"),
        Cand(_s(k) ** (n - 1) / ((n - 1) * k), "sc-sec-power-lowered"),
    ]
    working = [
        f"Notice that d/dx sec({_arg(k)}) = {k if k > 1 else ''}sec({_arg(k)}) tan({_arg(k)}). Let u = sec({_arg(k)}), so du = {k if k > 1 else ''}sec({_arg(k)}) tan({_arg(k)}) dx.",
        f"The integral becomes (1/{k}) ∫ u^{n - 1} du = u^{n}/{n * k} + C, that is sec^{n}({_arg(k)})/{n * k} + C."
        if k > 1
        else f"The integral becomes ∫ u^{n - 1} du = u^{n}/{n} + C, that is sec^{n} x/{n} + C.",
    ]
    return _indef(rng, f, F, cands, working, note=f" Give your answer in terms of sec({_arg(k)}).")


def _medium_tan_power_sec2(rng):
    m = rng.choice([1, 2, 3, 4])
    k = rng.choice([1, 1, 2, 3])
    f = _itg(_t(k) ** m * _s(k) ** 2, _fn("tan", m, k), (" ", " "), _fn("sec", 2, k))
    F = _t(k) ** (m + 1) / ((m + 1) * k)
    cands = [
        Cand(_t(k) ** (m + 1) / k, "sc-substituted-power-not-divided"),
        Cand(_t(k) ** (m + 1) / (m * k), "sc-tan-sec-divides-by-old-exponent"),
        Cand(_t(k) ** (m + 1) / (m + 1), "sc-tan-sec-inner-derivative-ignored") if k > 1 else None,
        Cand(-(_t(k) ** (m + 1)) / ((m + 1) * k), "sc-tan-sec-sign-slip"),
    ]
    working = [
        f"Let u = tan({_arg(k)}), so du = {k if k > 1 else ''}sec^2({_arg(k)}) dx.",
        f"The integral becomes (1/{k}) ∫ u^{m} du = u^{m + 1}/{(m + 1) * k} + C, that is tan^{m + 1}({_arg(k)})/{(m + 1) * k} + C."
        if k > 1
        else f"The integral becomes ∫ u^{m} du = u^{m + 1}/{m + 1} + C, that is tan^{m + 1} x/{m + 1} + C.",
    ]
    return _indef(rng, f, F, cands, working, note=f" Give your answer in terms of tan({_arg(k)}).")


def _medium(rng):
    return rng.choice([_medium_sec_power_tan, _medium_tan_power_sec2])(rng)


# ------------------------------------------------------------------------------------------------ hard


def _hard_tan4(rng):
    k = rng.choice([1, 1, 2])
    f = _itg(_t(k) ** 4, _fn("tan", 4, k))
    F = _t(k) ** 3 / (3 * k) - _t(k) / k + x
    cands = [
        Cand(_t(k) ** 3 / (3 * k) - _t(k) / k, "sc-tan-reduction-term-dropped"),
        Cand(_t(k) ** 3 / (3 * k) + _t(k) / k - x, "sc-tan-reduction-sign"),
        Cand(_t(k) ** 5 / (5 * k), "sc-tan-power-rule-on-trig"),
        Cand(_t(k) ** 3 / (3 * k) - _t(k) / k - x, "sc-tan-reduction-sign"),
        Cand(_t(k) ** 3 / 3 - _t(k) + x, "sc-tan-inner-derivative-ignored") if k > 1 else None,
    ]
    working = [
        f"Write tan^4 A = tan^2 A (sec^2 A - 1) = tan^2 A sec^2 A - tan^2 A with A = {_arg(k)}.",
        "Integrate tan^2 A sec^2 A by u = tan A, and tan^2 A = sec^2 A - 1 by the identity.",
        f"∫ tan^4({_arg(k)}) dx = tan^3({_arg(k)})/{3 * k} - tan({_arg(k)})/{k} + x + C."
        if k > 1
        else "∫ tan^4 x dx = tan^3 x/3 - tan x + x + C.",
    ]
    return _indef(rng, f, F, cands, working)


def _hard_sec4(rng):
    k = rng.choice([1, 1, 2, 3])
    f = _itg(_s(k) ** 4, _fn("sec", 4, k))
    F = (_t(k) + _t(k) ** 3 / 3) / k
    cands = [
        Cand((_t(k) + _t(k) ** 3) / k, "sc-sec4-cube-not-divided"),
        Cand(_s(k) ** 5 / (5 * k), "sc-sec-power-rule-on-trig"),
        Cand((_t(k) - _t(k) ** 3 / 3) / k, "sc-sec4-identity-sign"),
        Cand(_t(k) ** 3 / (3 * k), "sc-sec4-term-dropped"),
        Cand(_t(k) + _t(k) ** 3 / 3, "sc-tan-inner-derivative-ignored") if k > 1 else None,
    ]
    working = [
        f"Write sec^4 A = sec^2 A · sec^2 A = (1 + tan^2 A) sec^2 A with A = {_arg(k)}.",
        "Let u = tan A, du = sec^2 A dA, and integrate 1 + u^2 to get u + u^3/3.",
        f"∫ sec^4({_arg(k)}) dx = [tan({_arg(k)}) + tan^3({_arg(k)})/3]/{k} + C."
        if k > 1
        else "∫ sec^4 x dx = tan x + tan^3 x/3 + C.",
    ]
    return _indef(rng, f, F, cands, working)


def _hard_tan2_sec4(rng):
    k = rng.choice([1, 1, 2, 3])
    f = _itg(_t(k) ** 2 * _s(k) ** 4, _fn("tan", 2, k), (" ", " "), _fn("sec", 4, k))
    F = (_t(k) ** 3 / 3 + _t(k) ** 5 / 5) / k
    cands = [
        Cand((_t(k) ** 3 / 3 - _t(k) ** 5 / 5) / k, "sc-sec4-identity-sign"),
        Cand(_t(k) ** 3 / (3 * k), "sc-sec4-term-dropped"),
        Cand((_t(k) ** 3 / 3 + _t(k) ** 5 / 3) / k, "sc-substituted-power-not-divided-by-own-exponent"),
        Cand(_t(k) ** 3 / 3 + _t(k) ** 5 / 5, "sc-tan-inner-derivative-ignored") if k > 1 else None,
    ]
    a = _arg(k)
    working = [
        f"Write sec^4({a}) = (1 + tan^2({a})) sec^2({a}), so the integrand is tan^2({a}) (1 + tan^2({a})) sec^2({a}).",
        f"Let u = tan({a}), du = {k if k > 1 else ''}sec^2({a}) dx: (1/{k}) ∫ u^2 (1 + u^2) du = (u^3/3 + u^5/5)/{k}."
        if k > 1
        else "Let u = tan x, du = sec^2 x dx: ∫ u^2 (1 + u^2) du = u^3/3 + u^5/5.",
        f"∫ tan^2({a}) sec^4({a}) dx = [tan^3({a})/3 + tan^5({a})/5]/{k} + C."
        if k > 1
        else "∫ tan^2 x sec^4 x dx = tan^3 x/3 + tan^5 x/5 + C.",
    ]
    return _indef(rng, f, F, cands, working, note=f" Give your answer in terms of tan({a}).")


def _hard_sec_tan3(rng):
    m = rng.choice([1, 2, 3])
    f = _itg(_s() ** m * _t() ** 3, _fn("sec", m), (" ", " "), _fn("tan", 3))
    F = _s() ** (m + 2) / (m + 2) - _s() ** m / m
    cands = [
        Cand(_s() ** (m + 2) / (m + 2) + _s() ** m / m, "sc-sec-tan3-identity-sign"),
        Cand(_s() ** (m + 2) / (m + 2), "sc-sec-tan3-term-dropped"),
        Cand(
            _s() ** (m + 1) / (m + 1) - _s() ** (m - 1) / (m - 1) if m > 1 else None, "sc-sec-tan3-exponents-off-by-one"
        ),
        Cand(_s() ** (m + 2) / m - _s() ** m / (m + 2), "sc-sec-tan3-denominators-swapped"),
    ]
    working = [
        f"Split off sec x tan x: sec^{m} x tan^3 x = sec^{m - 1} x tan^2 x · sec x tan x = sec^{m - 1} x (sec^2 x - 1) · sec x tan x.",
        f"Let u = sec x, du = sec x tan x dx: ∫ u^{m - 1}(u^2 - 1) du = u^{m + 2}/{m + 2} - u^{m}/{m}.",
        f"The result is sec^{m + 2} x/{m + 2} - sec^{m} x/{m} + C.",
    ]
    return _indef(rng, f, F, cands, working, note=" Give your answer in terms of sec(x).")


_ROOT3 = sqrt(3)


def _hard_definite(rng):
    kind = rng.choice(
        [
            "tan2",
            "sec4",
            "tan3",
            "tan4",
            "sec_tan",
            "tan_sec2",
            "sec3_tan",
            "sec4_sixth",
        ]
    )
    q4, q3, q6 = pi / 4, pi / 3, pi / 6
    if kind == "tan2":
        c = rng.choice([1, 2, 3, 4])
        f = _itg(c * _t() ** 2, _coef(c), _fn("tan", 2))
        hi, value = q4, c * (1 - pi / 4)
        cands = [
            Cand(c * 1, "sc-tan-sq-taken-as-sec-sq"),
            Cand(c * (1 + pi / 4), "sc-tan-sq-identity-sign"),
            Cand(c * (pi / 4 - 1), "sc-definite-sign-slip"),
            Cand(-c * pi / 4, "sc-definite-tan-term-dropped"),
        ]
        working = [
            "tan^2 x = sec^2 x - 1, so ∫ tan^2 x dx = tan x - x.",
            f"Between 0 and pi/4: {c if c != 1 else ''}[(tan(pi/4) - pi/4) - (0 - 0)] = {c if c != 1 else ''}(1 - pi/4).".replace(
                "1[", "["
            ),
        ]
        lo = 0
    elif kind == "sec4":
        f = _itg(_s() ** 4, _fn("sec", 4))
        lo, hi, value = 0, q4, Rational(4, 3)
        cands = [
            Cand(Rational(1), "sc-sec4-term-dropped"),
            Cand(Rational(2), "sc-sec4-cube-not-divided"),
            Cand(Rational(2, 3), "sc-sec4-identity-sign"),
            Cand(Rational(1, 3), "sc-sec4-term-dropped"),
        ]
        working = [
            "sec^4 x = (1 + tan^2 x) sec^2 x, so ∫ sec^4 x dx = tan x + tan^3 x/3.",
            "Between 0 and pi/4: (1 + 1/3) - 0 = 4/3.",
        ]
    elif kind == "tan3":
        f = _itg(_t() ** 3, _fn("tan", 3))
        lo, hi = 0, rng.choice([q4, q3])
        if hi == q4:
            value = Rational(1, 2) - log(2) / 2
            working_end = "tan^2 x/2 + ln|cos x| between 0 and pi/4: (1/2 + ln(1/√2)) - 0 = 1/2 - (ln 2)/2."
        else:
            value = Rational(3, 2) - log(2)
            working_end = "tan^2 x/2 + ln|cos x| between 0 and pi/3: (3/2 + ln(1/2)) - 0 = 3/2 - ln 2."
        cands = [
            Cand(Rational(1, 2) + log(2) / 2 if hi == q4 else Rational(3, 2) + log(2), "sc-tan3-log-sign"),
            Cand(Rational(1, 2) if hi == q4 else Rational(3, 2), "sc-tan3-log-term-dropped"),
            Cand(Rational(1, 4) if hi == q4 else Rational(9, 4), "sc-tan-power-rule-on-trig"),
            Cand(-log(2) / 2 if hi == q4 else -log(2), "sc-tan3-power-term-dropped"),
        ]
        working = [
            "tan^3 x = tan x (sec^2 x - 1) = tan x sec^2 x - tan x, so ∫ tan^3 x dx = tan^2 x/2 + ln|cos x|.",
            working_end,
        ]
    elif kind == "tan4":
        f = _itg(_t() ** 4, _fn("tan", 4))
        lo, hi, value = 0, q4, pi / 4 - Rational(2, 3)
        cands = [
            Cand(Rational(2, 3) - pi / 4, "sc-definite-sign-slip"),
            Cand(Rational(4, 3) - pi / 4, "sc-tan-reduction-sign"),
            Cand(-Rational(2, 3), "sc-tan-reduction-term-dropped"),
            Cand(Rational(1, 5), "sc-tan-power-rule-on-trig"),
            Cand(-Rational(2, 3) - pi / 4, "sc-tan-reduction-sign"),
        ]
        working = [
            "tan^4 x = tan^2 x sec^2 x - tan^2 x, so ∫ tan^4 x dx = tan^3 x/3 - tan x + x.",
            "Between 0 and pi/4: (1/3 - 1 + pi/4) - 0 = pi/4 - 2/3.",
        ]
    elif kind == "sec_tan":
        c = rng.choice([1, 2, 3, 4])
        f = _itg(c * _s() * _t(), _coef(c), _fn("sec", 1), (" ", " "), _fn("tan", 1))
        lo, hi = 0, q3
        value = c * (2 - 1)
        cands = [
            Cand(c * 2, "sc-definite-lower-limit-ignored"),
            Cand(c * sqrt(3), "sc-sec-tan-sec-dropped"),
            Cand(c * 3, "sc-definite-limits-added"),
            Cand(c * Rational(3, 2), "sc-sec-tan-integrated-as-tan-squared"),
        ]
        working = [
            "sec x tan x integrates to sec x.",
            f"Between 0 and pi/3: {c if c != 1 else ''}(sec(pi/3) - sec 0) = {c if c != 1 else ''}(2 - 1) = {value}.".replace(
                "1(", "("
            ),
        ]
    elif kind == "tan_sec2":
        m = rng.choice([1, 2, 3])
        f = _itg(_t() ** m * _s() ** 2, _fn("tan", m), (" ", " "), _fn("sec", 2))
        lo, hi = 0, q3
        value = (sqrt(3) ** (m + 1)) / (m + 1)
        cands = [
            Cand(sqrt(3) ** (m + 1), "sc-substituted-power-not-divided"),
            Cand(sqrt(3) ** (m + 1) / m, "sc-tan-sec-divides-by-old-exponent"),
            Cand(Rational(2) ** (m + 1) / (m + 1), "sc-definite-sec-value-for-tan"),
            Cand(sqrt(3) ** m / (m + 1), "sc-tan-sec-exponent-not-raised"),
        ]
        working = [
            f"Let u = tan x, du = sec^2 x dx. The limits become tan 0 = 0 and tan(pi/3) = √3: ∫ from 0 to √3 of u^{m} du.",
            f"u^{m + 1}/{m + 1} at √3 gives (√3)^{m + 1}/{m + 1}.",
        ]
    elif kind == "sec3_tan":
        f = _itg(_s() ** 3 * _t(), _fn("sec", 3), (" ", " "), _fn("tan", 1))
        lo, hi, value = 0, q3, Rational(8, 3) - Rational(1, 3)
        cands = [
            Cand(Rational(8, 3), "sc-definite-lower-limit-ignored"),
            Cand(Rational(15, 4), "sc-sec-power-rule-wrong-exponent"),
            Cand(Rational(7), "sc-substituted-power-not-divided"),
            Cand(Rational(3), "sc-definite-limits-added"),
        ]
        working = [
            "Let u = sec x, du = sec x tan x dx; sec^3 x tan x dx = u^2 du. The limits become sec 0 = 1 and sec(pi/3) = 2.",
            "∫ from 1 to 2 of u^2 du = (8 - 1)/3 = 7/3.",
        ]
    else:  # sec4_sixth
        f = _itg(_s() ** 4, _fn("sec", 4))
        lo, hi = 0, q6
        value = Rational(10) * sqrt(3) / 27
        cands = [
            Cand(Rational(4) * sqrt(3) / 9, "sc-sec4-cube-not-divided"),
            Cand(sqrt(3) / 3, "sc-sec4-term-dropped"),
            Cand(Rational(8) * sqrt(3) / 27, "sc-sec4-identity-sign"),
        ]
        working = [
            "sec^4 x = (1 + tan^2 x) sec^2 x, so ∫ sec^4 x dx = tan x + tan^3 x/3.",
            "tan(pi/6) = 1/√3, so the value is 1/√3 + 1/(9√3) = 10/(9√3) = 10√3/27.",
        ]
    return _definite(rng, f, value, lo, hi, cands, working)


def _hard(rng):
    return rng.choice([_hard_tan4, _hard_sec4, _hard_tan2_sec4, _hard_sec_tan3, _hard_definite, _hard_definite])(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("sc_integrate_tan_sec", level=14, presets=_PRESETS)
@uses_rng
@retry
def sc_integrate_tan_sec(rng, tier: str = "easy"):
    """Integrals of tan and sec powers and products; the tiers are described in the module docstring."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam

_I = {
    1: log(2) / 2,
    2: 1 - pi / 4,
    3: Rational(1, 2) - log(2) / 2,
    4: pi / 4 - Rational(2, 3),
    5: log(2) / 2 - Rational(1, 4),
    6: Rational(13, 15) - pi / 4,
}


def _reduce(n):
    """I(n) from I(n - 2) by I(n) = 1/(n - 1) - I(n - 2), as a second route to the table above."""
    return Rational(1, n - 1) - _I[n - 2]


@generator("sc_integrate_tan_sec_exam", level=14, presets={"exam": {}})
@uses_rng
@retry
def sc_integrate_tan_sec_exam(rng):
    """Exam chain over 0 to pi/4: I(s), I(s + 2), I(s + 4) by the reduction, then a combination of two of them."""
    s = rng.choice([1, 2])
    for n in (s + 2, s + 4):
        check(_reduce(n) == _I[n], f"I({n}) differs from the reduction")
    for n in (s, s + 2, s + 4):
        check_definite(_I[n], tan(x) ** n, 0, pi / 4)
    p, q = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]), rng.choice([2, 3, 4, 5, 6, 7, 9, 10, 11, 12])
    a, b, c = _I[s], _I[s + 2], _I[s + 4]
    combo = p * c + q * b
    check_definite(combo, p * tan(x) ** (s + 4) + q * tan(x) ** (s + 2), 0, pi / 4)
    stem = rng.choice(
        [
            "Each integral below is taken from 0 to pi/4. Give each answer as an exact value.",
            "Work out the exact value of each of these definite integrals, which all run from 0 to pi/4.",
        ]
    )
    lo, hi = 0, pi / 4

    def prompt(f, lead=""):
        return lead + "Find the exact value of " + integral_text(
            f, lo, hi
        ) + ".", lead + "Find the exact value of " + integral_math(f, lo, hi) + "."

    def fn(n):
        txt, tx = _fn("tan", n)
        return Itg(tan(x) ** n, text=txt, latex=tx)

    pa_p, pa_l = prompt(fn(s))
    if s == 1:
        pa_work = [
            "tan x = sin x / cos x, so ∫ tan x dx = -ln|cos x| = ln|sec x|.",
            "Between 0 and pi/4: ln(sec(pi/4)) - ln(sec 0) = ln √2 = (ln 2)/2.",
        ]
        a_cands = [
            Cand(log(2), "sc-tan1-log-of-sec-value-not-halved"),
            Cand(-log(2) / 2, "sc-definite-sign-slip"),
            Cand(Rational(1, 2), "sc-tan-power-rule-on-trig"),
            Cand(sqrt(2) - 1, "sc-tan-integrated-as-sec"),
        ]
    else:
        pa_work = ["tan^2 x = sec^2 x - 1, so ∫ tan^2 x dx = tan x - x.", "Between 0 and pi/4: (1 - pi/4) - 0."]
        a_cands = [
            Cand(1 + pi / 4, "sc-tan-sq-identity-sign"),
            Cand(Rational(1, 3), "sc-tan-power-rule-on-trig"),
            Cand(pi / 4 - 1, "sc-definite-sign-slip"),
            Cand(Rational(1), "sc-tan-sq-taken-as-sec-sq"),
        ]
    pa = part(pa_p, pa_l, 2, pa_work, exact_fields(a), pick_expr(a, False, _no_constant(a_cands), rng=rng))
    tn = "tan x" if s == 1 else f"tan^{s} x"
    hint = f"Use tan^{s + 2} x = {tn} sec^2 x - {tn} and your answer to (a). "
    pb_p, pb_l = prompt(fn(s + 2), hint)
    pb = part(
        pb_p,
        pb_l,
        2,
        [
            f"∫ {tn} sec^2 x dx = tan^{s + 1} x/{s + 1}, which is 1/{s + 1} between 0 and pi/4.",
            f"So I = 1/{s + 1} - ({expr_text(a, False)}) = {expr_text(b, False)}.",
        ],
        exact_fields(b),
        pick_expr(
            b,
            False,
            _no_constant(
                [
                    Cand(Rational(1, s + 1) + a, "sc-reduction-sign"),
                    Cand(a - Rational(1, s + 1), "sc-definite-sign-slip"),
                    Cand(Rational(1, s + 1), "sc-reduction-term-dropped"),
                    Cand(Rational(1, s + 3), "sc-tan-power-rule-on-trig"),
                ]
            ),
            rng=rng,
        ),
        follow_through=f"1/{s + 1} - part_a",
    )
    hint = f"Use tan^{s + 4} x = tan^{s + 2} x sec^2 x - tan^{s + 2} x and your answer to (b). "
    pc_p, pc_l = prompt(fn(s + 4), hint)
    pc = part(
        pc_p,
        pc_l,
        2,
        [
            f"∫ tan^{s + 2} x sec^2 x dx = tan^{s + 3} x/{s + 3}, which is 1/{s + 3} between 0 and pi/4.",
            f"So I = 1/{s + 3} - ({expr_text(b, False)}) = {expr_text(c, False)}.",
        ],
        exact_fields(c),
        pick_expr(
            c,
            False,
            _no_constant(
                [
                    Cand(Rational(1, s + 3) + b, "sc-reduction-sign"),
                    Cand(b - Rational(1, s + 3), "sc-definite-sign-slip"),
                    Cand(Rational(1, s + 3), "sc-reduction-term-dropped"),
                    Cand(Rational(1, s + 5), "sc-tan-power-rule-on-trig"),
                    Cand(Rational(1, s + 1) - b, "sc-reduction-wrong-denominator"),
                ]
            ),
            rng=rng,
        ),
        follow_through=f"1/{s + 3} - part_b",
    )
    f_combo = Itg(
        p * tan(x) ** (s + 4) + q * tan(x) ** (s + 2),
        text=f"{p} {_fn('tan', s + 4)[0]} + {q} {_fn('tan', s + 2)[0]}",
        latex=f"{p}{_fn('tan', s + 4)[1]} + {q}{_fn('tan', s + 2)[1]}",
    )
    pd_p, pd_l = prompt(f_combo, "Use your answers to (b) and (c). ")
    pd = part(
        pd_p,
        pd_l,
        1,
        [f"Integrate term by term: {p} × ({expr_text(c, False)}) + {q} × ({expr_text(b, False)})."],
        exact_fields(combo),
        pick_expr(
            combo,
            False,
            _no_constant(
                [
                    Cand(c + b, "sc-coefficients-ignored"),
                    Cand(p * c - q * b, "sc-definite-sign-slip"),
                    Cand(p * b + q * c, "sc-coefficients-swapped"),
                    Cand(p * c + q * a, "sc-wrong-earlier-part-used"),
                ]
            ),
            rng=rng,
        ),
        follow_through=f"{p}*part_c + {q}*part_b",
    )
    for each in (pa, pb, pc, pd):
        each["working"] = [_tidy(w) for w in each["working"]]
    return dict(text=stem, latex=latex_from_inline(stem), parts=[pa, pb, pc, pd])
