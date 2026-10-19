"""Generates `r3b_integrate_exponential` questions (easy, medium, hard) and the exam chain
`r3b_exam_integrate_exponential`.

Skill ``integrate-exponential-linear-argument`` (``integration``, answer ``expression``): antiderivatives of
exponential functions whose argument is linear, typed with the constant of integration.

* easy: a single term ``a e^(kx)`` with ``k`` a whole number other than 0 and 1 (or ``1/2``, ``1/3``), so the
  coefficient has to be divided by ``k``.
* medium: the argument is shifted (``a e^(px + q)``), or the exponential is added to a power term or a constant, or
  two exponentials with opposite signs in the exponent are subtracted.
* hard: the integrand has to be rewritten before it can be integrated: the square of ``e^(px) + q``, the product
  ``e^(px) e^(qx)``, the reciprocal ``a / e^(kx)``, or a base other than ``e`` (``a b^(kx)``).
* exam (``r3b_exam_integrate_exponential``): the gradient function ``a e^(kx) + m`` with an unknown ``a`` fixed by the
  gradient at x = 0; (a) ``a``, (b) the constant of integration from a point, (c) the curve, (d) its exact value at
  x = 1.

Every antiderivative is differentiated back and compared with the integrand at five points, which is a second route
that does not use the integration rules the construction used.
"""

from __future__ import annotations

import sympy
from sympy import E, Integer, Rational, exp, log

from ..core import attempts, check, generator, latex_from_inline
from ._r3b_kit import (
    Cand,
    Itg,
    Redraw,
    check_antiderivative,
    exact_fields,
    frac_fields,
    integral_math,
    integral_text,
    key_fields,
    part,
    plus,
    pick_expr,
    pick_fractions,
    retry,
    say,
    shown,
    x,
)

_FIND = (
    "Find {I}.",
    "Determine {I}.",
    "Find the indefinite integral {I}.",
)


def _cx(c) -> str:
    """A coefficient as written before x: 1 is left out and -1 is a bare minus sign."""
    return "" if c == 1 else "-" if c == -1 else str(c)


def _stem(rng, f: Itg):
    return say(rng, _FIND, {"I": integral_text(f)}, {"I": integral_math(f)})


def _nonzero(rng, lo=-9, hi=9):
    return rng.choice([v for v in range(lo, hi + 1) if v])


def _result(rng, f: Itg, F, cands, working):
    check_antiderivative(F, f.value)
    text, latex = _stem(rng, f)
    return dict(
        text=text,
        latex=latex,
        working=working,
        distractors=pick_expr(F, True, cands, rng=rng),
        **key_fields(F),
    )


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    k = rng.choice([2, 3, 4, 5, -1, -2, -3, -4, Rational(1, 2), Rational(1, 3), Rational(-1, 2)])
    a = _nonzero(rng)
    f = Itg(a * exp(k * x))
    F = Rational(a) / k * exp(k * x)
    cands = [
        Cand(a * exp(k * x), "r3b-exp-integral-not-divided"),
        Cand(a * k * exp(k * x), "r3b-exp-integral-multiplied-by-k"),
        Cand(a * exp(k * x + 1) / (k * x + 1), "r3b-exp-integral-power-rule"),
        Cand(Rational(a) / abs(k) * exp(k * x), "r3b-exp-integral-sign-of-k-lost") if k < 0 else None,
    ]
    working = [
        f"The derivative of e^({shown(k * x)}) is {shown(k)}e^({shown(k * x)}), so divide by {shown(k)}.",
        f"{integral_text(f)} = {shown(F)} + C.",
    ]
    return _result(rng, f, F, cands, working)


# -- medium -------------------------------------------------------------------------------------------------------


def _medium_shift(rng):
    p = rng.choice([2, 3, 4, -1, -2, -3, 5])
    q = _nonzero(rng, -5, 5)
    a = _nonzero(rng, -8, 8)
    f = Itg(a * exp(p * x + q))
    F = Rational(a, p) * exp(p * x + q)
    cands = [
        Cand(a * exp(p * x + q), "r3b-exp-integral-not-divided"),
        Cand(a * p * exp(p * x + q), "r3b-exp-integral-multiplied-by-k"),
        Cand(a * exp(p * x + q + 1) / (p * x + q + 1), "r3b-exp-integral-power-rule"),
        Cand(Rational(a, p) * exp(p * x), "r3b-exp-shift-dropped"),
    ]
    return f, F, cands, [f"Divide by the coefficient {p} of x in the exponent: {shown(F)} + C."]


def _medium_sum(rng):
    k = rng.choice([2, 3, 4, -1, -2, -3])
    a = _nonzero(rng, -8, 8)
    n = rng.choice([0, 1, 2, 3])
    c = _nonzero(rng, -7, 7)
    term = c * x**n if n else Integer(c)
    f = Itg(a * exp(k * x) + term)
    antiderivative_of_term = c * x ** (n + 1) / (n + 1)
    F = Rational(a, k) * exp(k * x) + antiderivative_of_term
    wrong_term = term if n == 0 else c * n * x ** (n - 1)
    cands = [
        Cand(a * exp(k * x) + term, "r3b-exp-integral-not-divided"),
        Cand(Rational(a, k) * exp(k * x) + wrong_term, "r3b-power-term-not-integrated"),
        Cand(a * k * exp(k * x) + antiderivative_of_term, "r3b-exp-integral-multiplied-by-k"),
    ]
    return f, F, cands, [f"Integrate each term: the exponential term gives {shown(Rational(a, k) * exp(k * x))}."]


def _medium_difference(rng):
    p = rng.choice([2, 3, 4, 5])
    q = rng.choice([1, 2, 3])
    a, b = _nonzero(rng, 1, 8), _nonzero(rng, 1, 8)
    f = Itg(a * exp(p * x) - b * exp(-q * x))
    F = Rational(a, p) * exp(p * x) + Rational(b, q) * exp(-q * x)
    cands = [
        Cand(Rational(a, p) * exp(p * x) - Rational(b, q) * exp(-q * x), "r3b-exp-negative-exponent-sign"),
        Cand(a * exp(p * x) - b * exp(-q * x), "r3b-exp-integral-not-divided"),
        Cand(a * p * exp(p * x) + b * q * exp(-q * x), "r3b-exp-integral-multiplied-by-k"),
    ]
    return (
        f,
        F,
        cands,
        [f"The second term has exponent -{_cx(q)}x, so the coefficient {-b} is divided by -{q}, which gives {b}/{q}."],
    )


def _medium(rng):
    f, F, cands, working = rng.choice([_medium_shift, _medium_sum, _medium_difference])(rng)
    return _result(rng, f, F, cands, working)


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_square(rng):
    p = rng.choice([1, 2, 3])
    q = _nonzero(rng, -4, 4)
    base = exp(p * x) + q
    f = Itg(base**2)
    F = exp(2 * p * x) / (2 * p) + 2 * q * exp(p * x) / p + q**2 * x
    cands = [
        Cand(base**3 / (3 * p), "r3b-exp-bracket-power-rule"),
        Cand(exp(2 * p * x) / (2 * p) + q**2 * x, "r3b-square-cross-term-dropped"),
        Cand(exp(2 * p * x) / p + 2 * q * exp(p * x) / p + q**2 * x, "r3b-exp-doubled-exponent-divisor"),
        Cand(exp(2 * p * x) / (2 * p) + 2 * q * exp(p * x) / p + q**2, "r3b-constant-term-not-integrated"),
    ]
    return (
        f,
        F,
        cands,
        [f"Expand first: (e^({shown(p * x)}) {plus(q)})^2 = {shown(exp(2 * p * x) + 2 * q * exp(p * x) + q**2)}."],
    )


def _hard_product(rng):
    p = rng.choice([1, 2, 3, 4])
    q = rng.choice([1, 2, 3, 5, -2, -3])
    if p + q == 0:
        raise Redraw("exponents cancel")
    a = _nonzero(rng, -6, 6)
    f = Itg(
        a * exp(p * x) * exp(q * x),
        text=f"{_cx(a)}e^({_cx(p)}x) × e^({_cx(q)}x)",
        latex=rf"{_cx(a)}e^{{{_cx(p)}x}}\times e^{{{_cx(q)}x}}",
    )
    s = p + q
    F = Rational(a, s) * exp(s * x)
    cands = [
        Cand(Rational(a, p * q) * exp(s * x), "r3b-exp-integrate-factors-separately"),
        Cand(Rational(a, p * q) * exp(p * q * x), "r3b-exp-product-exponents-multiplied"),
        Cand(a * exp(s * x), "r3b-exp-integral-not-divided"),
    ]
    return f, F, cands, [f"Add the exponents: e^({_cx(p)}x) × e^({_cx(q)}x) = e^({s}x), then divide by {s}."]


def _hard_reciprocal(rng):
    k = rng.choice([2, 3, 4, 5])
    a = _nonzero(rng, -9, 9)
    f = Itg(a * exp(-k * x), text=f"{a}/e^({_cx(k)}x)", latex=rf"\frac{{{a}}}{{e^{{{_cx(k)}x}}}}")
    F = Rational(-a, k) * exp(-k * x)
    cands = [
        Cand(Rational(a, k) * exp(-k * x), "r3b-exp-negative-exponent-sign"),
        Cand(Rational(-a, k) * exp(k * x), "r3b-exp-exponent-sign-lost"),
        Cand(a * exp(-k * x), "r3b-exp-integral-not-divided"),
    ]
    return f, F, cands, [f"Write 1/e^({_cx(k)}x) as e^(-{_cx(k)}x), then divide by -{k}."]


def _hard_base(rng):
    b = rng.choice([2, 3, 5])
    k = rng.choice([1, 1, 2, 3])
    a = _nonzero(rng, -6, 6)
    f = Itg(a * b ** (k * x))
    F = Rational(a, k) * b ** (k * x) / log(b)
    cands = [
        Cand(Rational(a, k) * b ** (k * x) * log(b), "r3b-exp-base-multiplied-by-ln"),
        Cand(Rational(a, k) * b ** (k * x + 1) / (k * x + 1), "r3b-exp-integral-power-rule"),
        Cand(Rational(a, k) * b ** (k * x), "r3b-exp-base-ln-omitted"),
        Cand(a * b ** (k * x) / log(b), "r3b-exp-integral-not-divided"),
    ]
    return (
        f,
        F,
        cands,
        [f"For a base b, the integral of b^x is b^x / ln(b); the factor {k} in the exponent divides by {k}."],
    )


def _hard(rng):
    f, F, cands, working = rng.choice([_hard_square, _hard_product, _hard_reciprocal, _hard_base])(rng)
    return _result(rng, f, F, cands, working)


@generator(
    "r3b_integrate_exponential",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def r3b_integrate_exponential(rng, tier="easy"):
    """Antiderivatives of exponentials with a linear argument; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("r3b_exam_integrate_exponential", level=14, presets={"exam": dict(tier="exam")})
@retry
def r3b_exam_integrate_exponential(rng, tier="exam"):
    """Exam chain: an unknown coefficient, the curve through a point, an exact value and a point of given gradient."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "exam constants"):
        k = rng.choice([2, 3, 4, -1, -2, -3])
        m = _nonzero(rng, -6, 6)
        a = _nonzero(rng, -9, 9)
        y0 = _nonzero(rng, -9, 9)
        c = Rational(y0) - Rational(a, k)
        if c != 0 and a + m != 0 and a != y0:
            break
    g = a + m
    g2 = m + 2 * a  # the gradient where e^(kx) = 2
    f_prime = a * exp(k * x) + m
    F = Rational(a, k) * exp(k * x) + m * x + c
    check_antiderivative(F, f_prime)
    check(F.subs(x, 0) == y0 and f_prime.subs(x, 0) == g, "the curve does not meet the stated conditions")
    x4 = log(2) / k
    check(sympy.simplify(f_prime.subs(x, x4) - g2) == 0, "the gradient at the stated x is not the stated gradient")
    value1 = F.subs(x, 1)

    stem = (
        f"A curve y = f(x) has gradient function f'(x) = a e^({_cx(k)}x) {plus(m)}, where a is a constant. "
        f"The gradient of the curve at x = 0 is {g}, and the curve passes through the point (0, {y0})."
    )
    stem_latex = (
        rf"A curve \(y = f(x)\) has gradient function \(f'(x) = a e^{{{_cx(k)}x}} {plus(m)}\), "
        rf"where \(a\) is a constant. The gradient of the curve at \(x = 0\) is \({g}\), "
        rf"and the curve passes through the point \((0, {y0})\)."
    )
    k_a = Rational(a, k)

    pa = part(
        "Find the value of a.",
        "Find the value of \\(a\\).",
        1,
        [f"f'(0) = a {plus(m)} = {g}, so a = {a}."],
        frac_fields(a),
        pick_fractions(
            a,
            [
                (g, "r3b-chain-constant-term-ignored"),
                (g + m, "r3b-chain-constant-term-sign"),
                (-a, "r3b-chain-sign-slip"),
                (Rational(g, k), "r3b-chain-gradient-divided-by-k"),
            ],
            rng=rng,
        ),
    )
    pb = part(
        "Find f(x).",
        "Find \\(f(x)\\).",
        3,
        [
            f"Integrate: f(x) = {shown(k_a * exp(k * x) + m * x)} + c.",
            f"f(0) = {k_a} + c = {y0}, so c = {c} and f(x) = {shown(F)}.",
        ],
        key_fields(F, const=False),
        pick_expr(
            F,
            False,
            [
                Cand(k_a * exp(k * x) + m * x, "r3b-chain-point-not-used", const=False),
                Cand(a * exp(k * x) + m * x + (y0 - a), "r3b-exp-integral-not-divided", const=False),
                Cand(k_a * exp(k * x) + m * x + (y0 + k_a), "r3b-chain-constant-sign", const=False),
                Cand(k_a * exp(k * x) + m + (y0 - k_a), "r3b-constant-term-not-integrated", const=False),
            ],
            rng=rng,
        ),
        follow_through=f"part_a/({k})*exp({k}*x)+({m})*x+({y0})-part_a/({k})",
    )
    pc = part(
        "Find the exact value of f(1).",
        "Find the exact value of \\(f(1)\\).",
        2,
        [f"f(1) = {shown(k_a * exp(k))} {plus(m)} {plus(c)}".replace("+ -", "- ") + f" = {shown(value1)}."],
        exact_fields(value1),
        pick_expr(
            value1,
            False,
            [
                Cand(k_a * E + m + c, "r3b-chain-exponent-k-ignored", const=False),
                Cand(Integer(y0), "r3b-chain-evaluated-at-zero", const=False),
                Cand(a * exp(k) + m, "r3b-chain-gradient-for-value", const=False),
                Cand(k_a * exp(k) + m, "r3b-chain-constant-dropped", const=False),
            ],
            rng=rng,
        ),
        follow_through=f"part_a/({k})*exp({k})+({m})+({y0})-part_a/({k})",
    )
    pd = part(
        f"Find the exact x-coordinate of the point on the curve where the gradient is {g2}.",
        f"Find the exact \\(x\\)-coordinate of the point on the curve where the gradient is \\({g2}\\).",
        2,
        [f"{a}e^({_cx(k)}x) {plus(m)} = {g2} gives e^({_cx(k)}x) = 2, so x = ln(2)/{k}."],
        exact_fields(x4),
        pick_expr(
            x4,
            False,
            [
                Cand(log(2), "r3b-chain-log-not-divided-by-k", const=False),
                Cand(k * log(2), "r3b-chain-log-multiplied-by-k", const=False),
                Cand(log(Rational(2 * a, 1)) / k, "r3b-chain-coefficient-not-divided", const=False),
                Cand(log(Rational(g2, 1)) / k if g2 > 0 else None, "r3b-chain-constant-term-ignored", const=False),
            ],
            rng=rng,
        ),
        follow_through=f"log((({g2})-({m}))/part_a)/({k})",
    )
    return dict(text=stem, latex=latex_from_inline(stem_latex), parts=[pa, pb, pc, pd])
