"""Generates `w5a_residues` and `exd_x_w5a_residues` questions: residues at simple and higher-order poles.

One second-year skill, ``uc2-residues-higher-order-poles`` (domain ``complex-analysis``). Every answer is an exact
complex number typed as an expression. Each question is built backwards from the poles and the numerator, with the
arithmetic done on exact Gaussian rationals; the answer is then confirmed by the mean of f(z)(z - a) round a small
circle about the pole, which is (1/2*pi*i) times the contour integral and a different route from the formulas used to
build it.

Tiers:

- ``easy``: a simple pole. The denominator comes as an expanded quadratic with a conjugate pair of poles (the residue
  is asked at the pole in the upper half-plane, so the learner finds it), as a product of three linear factors with
  Gaussian-integer poles, or as ``z^n - c`` (the residue at one of the n-th roots).
- ``medium``: a pole of order two or three with a polynomial numerator and a second pole elsewhere, so the derivative
  of the rest of the function has to be taken, and divided by (m - 1)!.
- ``hard``: functions that are not rational: ``e^(cz)/((z - a)^m (z - b))``, ``1/(z^k sin(cz))``,
  ``cos(cz)/z^m`` or ``sin(cz)/z^m``, and ``g(z)/sin(pi z)`` at an integer.
- ``exam`` (topic ``exd_x_w5a_residues``): a four-part chain on ``N(z)/((z - a)^2 (z - b)(z - c))`` with real poles: the
  two simple residues, the double-pole residue (which follows from the other two because the residues sum to zero),
  and the integral round a circle enclosing two of the poles.
"""

from __future__ import annotations

import cmath
import math
import random
from fractions import Fraction as Fr

import sympy

from ..core import attempts, check, generator
from ._w5a_support import (
    C,
    Link,
    Rat,
    X,
    atext,
    circle_mean,
    cx_x,
    exam_dict,
    fac_x,
    join_signed,
    frac_x,
    msay,
    numeric_close,
    pmul,
    poly_x,
    ppow,
    prod_x,
    qsym,
    rat_x,
    sdiv,
    single,
)

TOPIC = "w5a_residues"
EXAM_TOPIC = "exd_x_w5a_residues"

_TIDY = 60  # the largest denominator allowed in a residue


def _tidy(c: C) -> bool:
    return c.re.denominator <= _TIDY and c.im.denominator <= _TIDY


def _fline(body: X) -> X:
    return X("f(z) = " + body.p, "f(z) = " + body.t)


def _res_x(pole: C) -> X:
    a = cx_x(pole)
    return X(f"Res(f, {a.p})", rf"\operatorname{{Res}}(f, {a.t})")


def _at_x(pole: C) -> X:
    a = cx_x(pole)
    return X(f"z = {a.p}", f"z = {a.t}")


def _check_residue(rat: Rat, j: int, value: C) -> None:
    """The second route: the mean of f(z)(z - a) round a small circle about pole j."""
    a = complex(rat.poles[j][0])
    gaps = [abs(a - complex(b)) for i, (b, _) in enumerate(rat.poles) if i != j] or [1.0]
    radius = min(0.5, 0.4 * min(gaps))
    numeric_close(circle_mean(rat.at, a, radius), complex(value), "residue")


def _poly(*coeffs) -> list:
    return [C(c) for c in coeffs]


_REAL = [C(k) for k in (-3, -2, -1, 0, 1, 2, 3)]
_GAUSS = [C(a, b) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 1, 2)]


def _numerator(rng, degree: int) -> list:
    coeffs = [rng.randint(-4, 4) for _ in range(degree + 1)]
    if coeffs[-1] == 0:
        coeffs[-1] = rng.choice([1, 2, -1, -2, 3])
    return _poly(*coeffs)


# -- easy ----------------------------------------------------------------------------------------------------------------


def _easy_quadratic(rng):
    """Expanded quadratic denominator with a conjugate pair of poles; the residue at the pole above the real axis."""
    for _ in attempts(500, "an easy quadratic residue"):
        p, q = rng.randint(-3, 3), rng.randint(1, 3)
        num = _numerator(rng, rng.choice([0, 1]))
        up, down = C(p, q), C(p, -q)
        rat = Rat(num, [(up, 1), (down, 1)])
        res = rat.residue(0)
        if res.is_zero() or not _tidy(res):
            continue
        break
    den = poly_x(_poly(p * p + q * q, -2 * p, 1))
    fx = _fline(frac_x(poly_x(num), den))
    other = rat.residue(1)
    pole = rat.poles[0][0]
    lead_num = sum((num[k] * pole**k for k in range(len(num))), C(0))
    wrongs = [
        (other.sym(), "residue-at-wrong-pole"),
        (-res.sym(), "residue-sign-error"),
        ((lead_num / (pole * 2)).sym(), "cx2-denominator-derivative-slip"),
        ((lead_num).sym(), "cx2-residue-numerator-only"),
        (2 * sympy.pi * sympy.I * res.sym(), "cx2-residue-times-2-pi-i"),
    ]
    _check_residue(rat, 0, res)
    working = [
        f"The denominator factors as (z - ({cx_x(up).p}))(z - ({cx_x(down).p})), so the pole above the real axis is z = {cx_x(up).p}.",
        f"Res = N(a)/D'(a), where D'(z) = {join_signed(['2z', str(-2 * p)] if p else ['2z'])}, so D'({cx_x(up).p}) = {cx_x(up * 2 - C(2 * p)).p}.",
        f"N({cx_x(up).p}) = {cx_x(lead_num).p}, so the residue is {atext(res.sym())}.",
    ]
    lead = [
        lambda: msay("Let ", fx, ". Find the residue of f at its pole in the upper half-plane."),
        lambda: msay(
            "The function ", fx, " has two simple poles. Calculate the residue at the one with positive imaginary part."
        ),
        lambda: msay("Find ", _res_x(up), " for ", fx, "."),
    ][rng.randrange(3)]
    return single(lead(), res.sym(), working, wrongs, fallback_tag="residue-sign-error")


def _easy_product(rng):
    """A product of three linear factors with Gaussian-integer poles; the residue at one of them."""
    for _ in attempts(500, "an easy product residue"):
        poles = rng.sample(_REAL + _GAUSS, 3)
        num = _numerator(rng, rng.choice([1, 2]))
        rat = Rat(num, [(a, 1) for a in poles])
        j = rng.randrange(3)
        res = rat.residue(j)
        if res.is_zero() or not _tidy(res):
            continue
        if any(poles[j] == poles[k] for k in range(3) if k != j):
            continue
        break
    pole = poles[j]
    others = [k for k in range(3) if k != j]
    fx = _fline(rat_x(rat))
    nval = sum((num[k] * pole**k for k in range(len(num))), C(0))
    prod_others = C(1)
    for k in others:
        prod_others = prod_others * (pole - poles[k])
    wrongs = [
        (rat.residue(others[0]).sym(), "residue-at-wrong-pole"),
        (-res.sym(), "residue-sign-error"),
        (nval.sym(), "cx2-residue-numerator-only"),
        ((nval / (pole - poles[others[0]])).sym(), "cx2-residue-factor-omitted"),
        (2 * sympy.pi * sympy.I * res.sym(), "cx2-residue-times-2-pi-i"),
    ]
    _check_residue(rat, j, res)
    working = [
        f"Only the factor {fac_x(pole).p} vanishes at z = {cx_x(pole).p}, so Res = N(a) divided by the other factors at a.",
        f"N({cx_x(pole).p}) = {cx_x(nval).p} and the other factors multiply to {cx_x(prod_others).p}, giving {atext(res.sym())}.",
    ]
    lead = [
        lambda: msay("Let ", fx, ". Find the residue of f at ", _at_x(pole), "."),
        lambda: msay("Calculate ", _res_x(pole), " where ", fx, "."),
        lambda: msay("The function ", fx, " has a pole at ", _at_x(pole), ". Find the residue of f there."),
    ][rng.randrange(3)]
    return single(lead(), res.sym(), working, wrongs, fallback_tag="residue-sign-error")


def _easy_root_of_unity(rng):
    """1/(z^n - c) or N/(z^n - c), c = a^n, at the root a."""
    for _ in attempts(500, "an easy z^n - c residue"):
        n = rng.choice([3, 4, 4, 5, 6])
        a = rng.choice([C(1), C(-1), C(2), C(-2), C(0, 1), C(0, -1), C(0, 2), C(0, -2)])
        c = a**n
        num = (
            _poly(rng.choice([1, 1, 2, 3, -1]), rng.choice([0, 0, 1, -1]))
            if rng.random() < 0.5
            else _poly(rng.choice([1, 2, 3, -2]))
        )
        # d/dz (z^n - c) at a = n a^(n-1); residue = N(a) / (n a^(n-1))
        nval = sum((num[k] * a**k for k in range(len(num))), C(0))
        res = nval / (a ** (n - 1) * n)
        if res.is_zero() or not _tidy(res):
            continue
        break
    cx = cx_x(c)
    fx = _fline(frac_x(poly_x(num), poly_x([-c] + [C(0)] * (n - 1) + [C(1)])))
    wrongs = [
        ((nval / (a**n * n)).sym(), "cx2-denominator-derivative-slip"),
        ((nval / (a ** (n - 1))).sym(), "cx2-denominator-derivative-slip"),
        (-res.sym(), "residue-sign-error"),
        ((nval / n).sym(), "cx2-residue-numerator-only"),
        (2 * sympy.pi * sympy.I * res.sym(), "cx2-residue-times-2-pi-i"),
    ]
    # second route: the numeric circle round a
    f = lambda z: sum(complex(num[k]) * z**k for k in range(len(num))) / (z**n - complex(c))  # noqa: E731
    numeric_close(circle_mean(f, complex(a), 0.3), complex(res), "residue")
    working = [
        f"The denominator vanishes at z = {cx_x(a).p} because ({cx_x(a).p})^{n} = {cx.p}, and it has a simple zero there.",
        f"Res = N(a)/D'(a), D'(z) = {n}z^{n - 1}, so Res = {cx_x(nval).p}/({n}({cx_x(a).p})^{n - 1}) = {atext(res.sym())}.",
    ]
    lead = [
        lambda: msay("Let ", fx, ". Find the residue of f at ", _at_x(a), "."),
        lambda: msay("Evaluate ", _res_x(a), " for ", fx, "."),
        lambda: msay("The function ", fx, " has a pole at ", _at_x(a), ". What is the residue of f there?"),
    ][rng.randrange(3)]
    return single(lead(), res.sym(), working, wrongs, fallback_tag="residue-sign-error")


# -- medium: poles of order two and three -----------------------------------------------------------------------------


def _medium_higher(rng):
    for _ in attempts(800, "a higher-order residue"):
        m = rng.choice([2, 2, 3])
        a, b = rng.sample(_REAL + _GAUSS[:8], 2)
        extra = rng.random() < 0.35 and m == 2
        poles = [(a, m), (b, 1)]
        if extra:
            c = rng.choice([p for p in _REAL if p not in (a, b)])
            poles.append((c, 1))
        num = _numerator(rng, rng.choice([1, 2]) if len(poles) > 2 or m == 2 else rng.choice([0, 1]))
        rat = Rat(num, poles)
        ser = rat.laurent(0, 2)
        res = ser[-1]
        if res.is_zero() or not _tidy(res) or ser[-m].is_zero():
            continue
        break
    pole = a
    fx = _fline(rat_x(rat))
    gden = C(1)
    for q, n in poles[1:]:
        gden = gden * (pole - q) ** n
    nprime = [num[k] * k for k in range(1, len(num))] or [C(0)]
    nprime_a = sum((nprime[k] * pole**k for k in range(len(nprime))), C(0))
    num_only = nprime_a / gden if m == 2 else ser[-2] * 0 + nprime_a / gden
    wrongs = [
        (ser[-m].sym(), "cx2-residue-simple-formula"),
        (ser[0].sym(), "cx2-residue-derivative-order"),
        (num_only.sym(), "cx2-residue-numerator-derivative-only"),
        (-res.sym(), "residue-sign-error"),
        (rat.residue(1).sym(), "residue-at-wrong-pole"),
    ]
    if m == 3:
        wrongs.insert(0, ((res * 2).sym(), "cx2-residue-factorial-dropped"))
        wrongs.insert(1, (ser[-2].sym(), "cx2-residue-derivative-order"))
    _check_residue(rat, 0, res)
    order_word = {2: "two", 3: "three"}[m]
    working = [
        f"z = {cx_x(pole).p} is a pole of order {m}: write f(z) = g(z)/{fac_x(pole, m).p} with g analytic there.",
        f"Res = g^({m - 1})(a)/{m - 1}!" if m > 2 else "Res = g'(a), differentiating the rest of the function.",
        f"Working through gives the Laurent coefficient of {fac_x(pole).p}^(-1): {atext(res.sym())}.",
    ]
    lead = [
        lambda: msay("Let ", fx, ". Find the residue of f at ", _at_x(pole), "."),
        lambda: msay("The function ", fx, " is analytic apart from isolated poles. Calculate ", _res_x(pole), "."),
        lambda: msay("Find the residue at ", _at_x(pole), " of ", fx, "."),
    ][rng.randrange(3)]
    del order_word
    return single(lead(), res.sym(), working, wrongs, fallback_tag="residue-sign-error")


# -- hard: functions that are not rational --------------------------------------------------------------------------


def _hard_exp(rng):
    """e^(cz) / ((z - a)^m (z - b)) at a: e^(ca) times a rational (or Gaussian) number."""
    for _ in attempts(500, "an exponential residue"):
        m = rng.choice([2, 3])
        c = rng.choice([1, 2, 3, -1, -2])
        a = rng.choice([0, 1, -1, 2])
        b = rng.choice([x for x in (0, 1, -1, 2, -2, 3) if x != a])
        # g(z) = e^(cz) / (z - b); Taylor series at a: e^(ca) * (sum c^k t^k / k!) * 1/((a - b) + t)
        expo = [C(Fr(c**k, math.factorial(k))) for k in range(m)]
        inv = sdiv([C(1)], [C(a - b), C(1)], m)
        coeff = C(0)
        for k in range(m):
            coeff = coeff + expo[k] * inv[m - 1 - k]
        if coeff.is_zero() or not _tidy(coeff):
            continue
        break
    value = coeff.sym() * sympy.exp(c * a)
    cz = f"{'' if c == 1 else '-' if c == -1 else c}z"
    den = prod_x([fac_x(C(a), m), fac_x(C(b), 1)])
    quo = frac_x(X(f"e^({cz})", rf"e^{{{cz}}}"), den)
    fx = X("f(z) = " + quo.p, "f(z) = " + quo.t)
    pole = C(a)
    wrongs = [
        (coeff.sym(), "cx2-residue-exponential-factor-dropped"),
        (_coef_without_chain(c, a, b, m) * sympy.exp(c * a), "cx2-exp-chain-factor-dropped"),
        (sympy.exp(c * a) / sympy.Integer(a - b) if a != b else 0, "cx2-residue-simple-formula"),
        (-value, "residue-sign-error"),
        (value * math.factorial(m - 1), "cx2-residue-factorial-dropped"),
    ]
    fn = lambda z: cmath.exp(c * z) / ((z - a) ** m * (z - b))  # noqa: E731
    numeric_close(circle_mean(fn, complex(a), 0.4), complex(sympy.N(value, 15)), "residue")
    working = [
        f"Write f(z) = g(z)/{fac_x(pole, m).p} with g(z) = e^({cz})/{fac_x(C(b)).p}.",
        f"The residue is g^({m - 1})(a)/{m - 1}!, using the product rule on e^({cz}) and 1/{fac_x(C(b)).p}.",
        f"This gives {atext(value)}.",
    ]
    lead = [
        lambda: msay("Let ", fx, ". Find the exact residue of f at ", _at_x(pole), "."),
        lambda: msay("Calculate ", _res_x(pole), " for ", fx, "."),
        lambda: msay("Find, in exact form, the residue at ", _at_x(pole), " of ", fx, "."),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="residue-sign-error")


def _coef_without_chain(c, a, b, m):
    """The same Laurent coefficient with the chain-rule factor c left out of the derivative of e^(cz)."""
    expo = [C(Fr(1, math.factorial(k))) for k in range(m)]
    inv = sdiv([C(1)], [C(a - b), C(1)], m)
    total = C(0)
    for k in range(m):
        total = total + expo[k] * inv[m - 1 - k]
    return total.sym()


def _hard_inverse_sine(rng):
    """1 / (z^k sin(cz)) at 0, k = 2 or 4: the first terms of 1/sin."""
    k = rng.choice([2, 4])
    c = rng.choice([1, 2, 3, 4])
    cc = sympy.Integer(c)
    value = cc / 6 if k == 2 else 7 * cc**3 / 360
    cz = "z" if c == 1 else f"{c}z"
    den = X(f"z{'²' if k == 2 else '⁴'} sin({cz})", rf"z^{{{k}}}\sin({cz})")
    fx = X(f"f(z) = 1/({den.p})", rf"f(z) = \frac{{1}}{{{den.t}}}")
    wrongs = [
        (1 / cc, "cx2-series-leading-term-only"),
        (cc**2 / 6 if k == 2 else 7 * cc**4 / 360, "cx2-series-coefficient-slip"),
        (7 * cc**3 / 360 if k == 2 else cc / 6, "cx2-series-wrong-term"),
        (-value, "residue-sign-error"),
        (cc / 3 if k == 2 else 7 * cc**3 / 180, "cx2-series-coefficient-slip"),
    ]
    fn = lambda z: 1 / (z**k * cmath.sin(c * z))  # noqa: E731
    numeric_close(circle_mean(fn, 0j, 0.5 if c <= 2 else 0.3), complex(float(value)), "residue")
    working = [
        f"Near 0, 1/sin({cz}) = (1/({cz}))(1 + ({cz})²/6 + 7({cz})⁴/360 + ...).",
        f"So f(z) = z^(-{k + 1})/{c} times that series, and the z^(-1) term comes from the term in z^{k}.",
        f"The residue is {atext(value)}.",
    ]
    lead = [
        lambda: msay("Find the residue at ", X("z = 0", "z = 0"), " of ", fx, "."),
        lambda: msay("Let ", fx, ". Calculate ", _res_x(C(0)), "."),
        lambda: msay("The function ", fx, " has a pole at the origin. Find its residue there."),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="residue-sign-error")


def _hard_trig_over_power(rng):
    """cos(cz)/z^m (m odd) or sin(cz)/z^m (m even): one term of the Maclaurin series."""
    c = rng.choice([1, 2, 3, -2])
    if rng.random() < 0.5:
        m = rng.choice([3, 5])
        fn_name = "cos"
        value = sympy.Integer((-1) ** ((m - 1) // 2)) * sympy.Integer(c) ** (m - 1) / math.factorial(m - 1)
        fn = lambda z: cmath.cos(c * z) / z**m  # noqa: E731
        slip = sympy.Integer(c) ** (m - 1) / math.factorial(m - 1)
    else:
        m = rng.choice([2, 4, 6])
        fn_name = "sin"
        value = sympy.Integer((-1) ** ((m - 2) // 2)) * sympy.Integer(c) ** (m - 1) / math.factorial(m - 1)
        fn = lambda z: cmath.sin(c * z) / z**m  # noqa: E731
        slip = sympy.Integer(c) ** (m - 1) / math.factorial(m - 1)
    cz = "z" if c == 1 else f"{c}z" if c > 0 else f"({c})z"
    fx = X(
        f"f(z) = {fn_name}({cz})/z{'²' if m == 2 else '³' if m == 3 else '⁴⁵⁶'[m - 4]}",
        rf"f(z) = \frac{{\{fn_name}({cz})}}{{z^{{{m}}}}}",
    )
    wrongs = [
        (slip, "cx2-series-alternating-sign-dropped"),
        (value * math.factorial(m - 1), "cx2-series-factorial-dropped"),
        (sympy.Integer(c) ** m / math.factorial(m) * (-1) ** (m // 2), "cx2-series-wrong-term"),
        (-value, "residue-sign-error"),
        (value / sympy.Integer(c) if c not in (1, -1) else value * 2, "cx2-series-coefficient-slip"),
    ]
    numeric_close(circle_mean(fn, 0j, 0.5), complex(float(value)), "residue")
    working = [
        f"Expand {fn_name}({cz}) as a Maclaurin series and divide by z^{m}.",
        f"The z^(-1) term comes from the term in z^{m - 1} of the series, which has coefficient {atext(value)}.",
    ]
    lead = [
        lambda: msay("Let ", fx, ". Find the residue of f at the origin."),
        lambda: msay("Calculate ", _res_x(C(0)), " for ", fx, "."),
        lambda: msay(
            "The function ", fx, " has an isolated singularity at ", X("z = 0", "z = 0"), ". Find its residue there."
        ),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="residue-sign-error")


def _hard_over_sine_pi(rng):
    """g(z)/sin(pi z) at the integer n: (-1)^n g(n)/pi."""
    for _ in attempts(200, "a sin(pi z) residue"):
        n = rng.choice([0, 1, 2, 3, -1, -2])
        num = _numerator(rng, rng.choice([1, 2]))
        gn = sum((num[k] * C(n) ** k for k in range(len(num))), C(0))
        if gn.is_zero():
            continue
        break
    value = sympy.Integer((-1) ** (abs(n))) * gn.sym() / sympy.pi
    fx = X(f"f(z) = ({poly_x(num).p})/sin(πz)", rf"f(z) = \frac{{{poly_x(num).t}}}{{\sin(\pi z)}}")
    wrongs = [
        (gn.sym() / sympy.pi, "cx2-residue-alternating-sign-dropped"),
        (sympy.Integer((-1) ** abs(n)) * gn.sym(), "cx2-residue-pi-dropped"),
        (-value, "residue-sign-error"),
        (sympy.Integer((-1) ** abs(n)) * gn.sym() * sympy.pi, "cx2-residue-pi-inverted"),
        (
            sympy.Integer((-1) ** abs(n + 1)) * gn.sym() / sympy.pi * 0 + sympy.Integer(1) / sympy.pi,
            "cx2-residue-numerator-only",
        ),
    ]

    def fn(z):
        return sum(complex(num[k]) * z**k for k in range(len(num))) / cmath.sin(math.pi * z)

    numeric_close(circle_mean(fn, complex(n), 0.3), complex(float(sympy.N(value, 15))), "residue")
    working = [
        f"sin(πz) has a simple zero at z = {n} with derivative π cos({n}π) = {(-1) ** abs(n)}π.",
        f"So Res = g({n})/(π cos({n}π)) = {atext(value)}.",
    ]
    lead = [
        lambda: msay("Let ", fx, ". Find the residue of f at ", _at_x(C(n)), "."),
        lambda: msay("Calculate ", _res_x(C(n)), " when ", fx, "."),
        lambda: msay("The function ", fx, " has a pole at ", _at_x(C(n)), ". Find its residue there."),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="residue-sign-error")


_TIERS = {
    "easy": [_easy_quadratic, _easy_product, _easy_root_of_unity],
    "medium": [_medium_higher],
    "hard": [_hard_exp, _hard_inverse_sine, _hard_trig_over_power, _hard_over_sine_pi],
}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5a_residues(rng: random.Random, tier: str = "easy") -> dict:
    """The residue of a complex function at a stated pole, exact."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return rng.choice(_TIERS[tier])(rng)


# -- exam: a four-part chain -------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5a_residues(rng: random.Random, tier: str = "exam") -> dict:
    """N(z) / ((z - a)^2 (z - b)(z - c)) with real poles: two simple residues, the double-pole residue, an integral."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a residue chain"):
        a, b = rng.sample([-2, -1, 0, 1, 2], 2)
        far = max(abs(a), abs(b)) + 2
        c = rng.choice([far, -far])
        num = _numerator(rng, 2)
        rat = Rat(num, [(C(b), 1), (C(c), 1), (C(a), 2)])
        rb, rc, ra = rat.residue(0), rat.residue(1), rat.residue(2)
        if rb.is_zero() or rc.is_zero() or ra.is_zero() or not all(_tidy(r) for r in (rb, rc, ra)):
            continue
        if rb + rc + ra != C(0):
            continue  # holds for degree 2 over degree 4; kept as a guard
        break
    big_r = Fr(max(abs(a), abs(b)) + 1)
    check(max(abs(a), abs(b)) < big_r < abs(c), "circle radius")
    for j, value in ((0, rb), (1, rc), (2, ra)):
        _check_residue(rat, j, value)
    r_b, r_c, r_a = (v.re for v in (rb, rc, ra))
    integral = 2 * sympy.pi * sympy.I * qsym(r_a + r_b)
    den = prod_x([fac_x(C(a), 2), fac_x(C(b)), fac_x(C(c))])
    fx = _fline(frac_x(poly_x(num), den))
    stem = msay("Let ", fx, ".")
    sw = sympy.Rational
    w_a = Link(
        msay("Find the residue of f at ", _at_x(C(b)), "."),
        qsym(r_b),
        "fraction",
        2,
        [f"{fac_x(C(b)).p} is a simple factor: Res = N(b) over the other factors at b = {atext(qsym(r_b))}."],
        [
            (qsym(r_c), "residue-at-wrong-pole"),
            (-qsym(r_b), "residue-sign-error"),
            (qsym(sum((num[k] * C(b) ** k for k in range(3)), C(0)).re), "cx2-residue-numerator-only"),
            (qsym(r_b) * (b - a), "cx2-residue-factor-omitted"),
        ],
    )
    w_b = Link(
        msay("Find the residue of f at ", _at_x(C(c)), "."),
        qsym(r_c),
        "fraction",
        2,
        [f"Res = N(c) over the other factors at c = {atext(qsym(r_c))}."],
        [
            (qsym(r_b), "residue-at-wrong-pole"),
            (-qsym(r_c), "residue-sign-error"),
            (qsym(sum((num[k] * C(c) ** k for k in range(3)), C(0)).re), "cx2-residue-numerator-only"),
            (qsym(r_c) * (c - a), "cx2-residue-factor-omitted"),
        ],
    )
    nprime_a = sum((num[k] * k * C(a) ** (k - 1) for k in range(1, 3)), C(0)).re
    gden = (a - b) * (a - c)
    w_c = Link(
        msay("Find the residue of f at the double pole ", _at_x(C(a)), "."),
        qsym(r_a),
        "fraction",
        3,
        [
            f"The residues of f sum to zero, because f(z) is O(1/z²) for large z, so Res at a = -(Res at b + Res at c) = {atext(qsym(r_a))}.",
            "The same value follows from g'(a), where g(z) = N(z)/((z - b)(z - c)).",
        ],
        [
            (qsym(Fr(sum((num[k] * C(a) ** k for k in range(3)), C(0)).re) / gden), "cx2-residue-simple-formula"),
            (qsym(r_b + r_c), "cx2-residue-sum-sign"),
            (qsym(Fr(nprime_a) / gden), "cx2-residue-numerator-derivative-only"),
            (-qsym(r_a), "residue-sign-error"),
        ],
        ft="-(part_a + part_b)",
    )
    del sw
    w_d = Link(
        msay(
            "Hence evaluate the integral of f(z) round the circle ",
            X(f"|z| = {big_r}", rf"|z| = {big_r}"),
            ", taken once anticlockwise.",
        ),
        integral,
        "expression",
        3,
        [
            f"The circle encloses z = {a} and z = {b} but not z = {c}.",
            f"The integral is 2πi (Res at a + Res at b) = 2πi({join_signed([atext(qsym(r_a)), atext(qsym(r_b))])}) = {atext(integral)}.",
        ],
        [
            (2 * sympy.pi * sympy.I * qsym(r_a + r_b + r_c), "included-pole-outside-contour"),
            (2 * sympy.pi * qsym(r_a + r_b), "forgets-i-in-residue-theorem"),
            (2 * sympy.pi * sympy.I * qsym(r_a), "ignores-pole-enclosed-by-larger-contour"),
            (2 * sympy.pi * sympy.I * qsym(r_c), "pole-outside-the-contour"),
        ],
        ft="2*pi*I*(part_c + part_a)",
    )
    return exam_dict(stem, [w_a, w_b, w_c, w_d])
