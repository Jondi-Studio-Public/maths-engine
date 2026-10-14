"""Generates `r4e_riemann_limit` questions (easy, medium, hard) and the exam chain `r4e_riemann_limit_chain`.

Skill `riemann-sum-to-definite-integral` (answer `fraction`): the step from a finite Riemann sum to the definite
integral. The practice skill on rectangle estimates (`p13c_riemann_sum_area`) fixes the number of strips; here n
grows, and the sums are tied to the integral they approach.

For f(x) = p x^2 + q x + r on [0, b] and Delta = b/n, the right-endpoint sum is

    R_n = p b^3 (n+1)(2n+1) / (6 n^2) + q b^2 (n+1) / (2 n) + r b = A + B/n + C/n^2,

with A = p b^3/3 + q b^2/2 + r b (the integral), B = p b^3/2 + q b^2/2 and C = p b^3/6. The left-endpoint sum is
L_n = R_n - (b/n)(f(b) - f(0)).

* easy: R_n is given as a single fraction (N2 n^2 + N1 n + N0) / (6 n^2); find the value the sums approach.
* medium: a limit of sums written out with its sigma, sample points a + c i / n and width c / n (the index running
  i = 1 .. n or i = 0 .. n - 1); recognise the interval [a, a + c] and the function, then evaluate the integral.
* hard: the least number of strips for which a right or left sum of an increasing function is within a stated
  tolerance of the integral, from the exact error.
* exam: R_n built from the sum formulas, its 1/n coefficient, the limit, the excess over the limit at a given n,
  and the least n for a tolerance.

Every key is checked against a direct sum over the strips (exact arithmetic) and against SymPy's integral.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, phrase
from .r4e_common import Redraw, draw, fraction_part, single

x, n_sym = sympy.symbols("x n", positive=True)


def _f_text(p, q, r) -> str:
    return join_terms([(p, "x^2"), (q, "x"), (r, "")])


def _f(p, q, r):
    return lambda t: p * t * t + q * t + r


def right_sum(p, q, r, b, n) -> Fraction:
    """Direct sum of f(i b/n) b/n, i = 1..n, in exact arithmetic."""
    f, d = _f(p, q, r), Fraction(b, n)
    return sum((f(i * d) * d for i in range(1, n + 1)), Fraction(0))


def left_sum(p, q, r, b, n) -> Fraction:
    f, d = _f(p, q, r), Fraction(b, n)
    return sum((f(i * d) * d for i in range(0, n)), Fraction(0))


def integral(p, q, r, b) -> Fraction:
    return Fraction(p * b**3, 3) + Fraction(q * b**2, 2) + r * b


def coefficients(p, q, r, b):
    """(A, B, C) with R_n = A + B/n + C/n^2."""
    return integral(p, q, r, b), Fraction(p * b**3, 2) + Fraction(q * b**2, 2), Fraction(p * b**3, 6)


def _check_coefficients(p, q, r, b) -> None:
    a_, b_, c_ = coefficients(p, q, r, b)
    for n in (1, 2, 5, 12):
        check(right_sum(p, q, r, b, n) == a_ + b_ / n + c_ / n**2, "closed form of the right sum disagrees")
    expr = sympy.integrate(p * x**2 + q * x + r, (x, 0, b))
    check(sympy.Rational(a_.numerator, a_.denominator) == expr, "limit disagrees with SymPy's integral")


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "For f(x) = {f} on the interval [0, {b}], the sum of the areas of n right-endpoint rectangles of equal width is R_n = ({num})/(6n^2).",
    "The n right-endpoint rectangles of equal width on [0, {b}] under the curve y = {f} have total area R_n = ({num})/(6n^2).",
    "Dividing [0, {b}] into n equal strips and using right endpoints for f(x) = {f} gives the sum R_n = ({num})/(6n^2).",
)


def _easy(rng):
    p, q, r, b = rng.choice((1, 2, 3)), rng.choice((0, 1, 2, 3, 4)), rng.choice((0, 1, 2, 3)), rng.choice((1, 2, 3, 4))
    _check_coefficients(p, q, r, b)
    a_, b_, c_ = coefficients(p, q, r, b)
    n2, n1, n0 = (int(v * 6) for v in (a_, b_, c_))
    check(all(v == int(v) for v in (a_ * 6, b_ * 6, c_ * 6)), "coefficients are not sixths")
    num = join_terms([(n2, "n^2"), (n1, "n"), (n0, "")])
    stem = phrase(rng, _EASY_STEMS, f=_f_text(p, q, r), b=b, num=num)
    wrong = [
        (Fraction(n2 + n1 + n0, 6), "riemann-limit-at-n-equals-one"),
        (Fraction(n2 + n1, 6), "riemann-limit-keeps-lower-order-terms"),
        (Fraction(n2), "riemann-limit-ignores-denominator-coefficient"),
        (Fraction(n1, 6), "riemann-limit-takes-wrong-order-coefficient"),
    ]
    working = [
        f"As n grows, {n1}n/(6n^2) and {n0}/(6n^2) tend to 0.",
        f"R_n tends to {n2}/6 = {a_}, the area under the curve.",
    ]
    part = fraction_part("Find the value that R_n approaches as n increases without bound.", a_, 2, working, wrong)
    return single(stem, part)


# -- medium -------------------------------------------------------------------------------------------------------

_MEDIUM_STEMS = (
    "Evaluate the limit, as n tends to infinity, of the sum Σ ({lo} to {hi}) of {term} × ({c}/n), by recognising it as a definite integral.",
    "Find the value that the sum Σ ({lo} to {hi}) of {term} × ({c}/n) approaches as n increases without bound.",
)


def _power_integral(p, k, r, lo, hi) -> Fraction:
    """The integral of p x^k + r from lo to hi."""
    return Fraction(p) * (Fraction(hi) ** (k + 1) - Fraction(lo) ** (k + 1)) / (k + 1) + r * (hi - lo)


def _medium(rng):
    for _ in attempts(500, "a sum whose integral is distinct from its distractors"):
        k = rng.choice((1, 2, 2, 3))
        p, r = rng.choice((1, 2, 3, 4)), rng.choice((0, 0, 1, 2, 3, -1, -2))
        a, c = rng.choice((0, 1, 2, 3, -1, -2)), rng.choice((1, 2, 3, 4))
        if a == 0 or a == c:
            continue
        b = a + c
        key = _power_integral(p, k, r, a, b)
        wrong_ranges = (
            _power_integral(p, k, r, 0, c),
            _power_integral(p, k, r, a, c),
            _power_integral(p, k, r, a, a + 1),
        )
        if len({key, *wrong_ranges}) == 4:
            break
    if rng.random() < 0.5:
        lo, hi, index = "i = 1", "n", "i"
    else:
        lo, hi, index = "i = 0", "n - 1", "i"
    # the sample point and the function are written out; the left sum (i = 0 .. n - 1) has the same limit
    power = "" if k == 1 else f"^{k}"
    inner = join_terms([(p, f"({a} + {c}i/n){power}"), (r, "")])
    term = inner if r == 0 else f"[{inner}]"
    stem = phrase(rng, _MEDIUM_STEMS, lo=lo, hi=hi, term=term, c=c)
    # independent check: the sum itself, at a large n, approaches the key
    n_big = 400
    pts = range(1, n_big + 1) if lo == "i = 1" else range(0, n_big)
    approx = sum(Fraction(p) * (a + Fraction(c * i, n_big)) ** k + r for i in pts) * Fraction(c, n_big)
    top = max(abs(p * Fraction(v) ** k + r) for v in (a, b, 0))
    check(abs(approx - key) <= 4 * Fraction(c, n_big) * top, "the finite sum is not near the integral")
    wrong = list(
        zip(
            wrong_ranges,
            (
                "riemann-limit-drops-lower-limit",
                "riemann-limit-width-as-upper-limit",
                "riemann-limit-width-taken-as-one-over-n",
            ),
        )
    )
    working = [
        f"The sample points are {a} + {c}i/n and the width is {c}/n, so the interval is [{a}, {b}] and f(x) = {p}x{power}{'' if r == 0 else ' ' + join_terms([(r, '')])}.",
        f"The limit is the integral from {a} to {b} of f(x) dx = {key}.",
    ]
    return single(stem, fraction_part("", key, 2, working, wrong))


# -- hard ---------------------------------------------------------------------------------------------------------

_HARD_STEMS = (
    "The function f(x) = {f} is increasing on [0, {b}]. The area under its curve from x = 0 to x = {b} is approximated by the {side}-endpoint rectangle sum S_n with n equal strips.",
    "Let f(x) = {f}, which is increasing for x ≥ 0. The {side}-endpoint rectangle sum S_n, with n equal strips, approximates the area under y = f(x) from x = 0 to x = {b}.",
)


def _error(p, q, r, b, side: str, n: int) -> Fraction:
    """|S_n - I|: B/n + C/n^2 for a right sum and B/n - C/n^2 for a left sum of an increasing function."""
    _, b_, c_ = coefficients(p, q, r, b)
    return b_ / n + c_ / n**2 if side == "right" else b_ / n - c_ / n**2


def _least_n(p, q, r, b, side: str, eps: Fraction, cap: int = 4_000_000):
    """The least n with |S_n - I| < eps. The error falls as n grows from 2, so bisect, then step down."""
    lo, hi = 2, cap
    if _error(p, q, r, b, side, hi) >= eps:
        raise Redraw("no n below the cap")
    while lo < hi:
        mid = (lo + hi) // 2
        if _error(p, q, r, b, side, mid) < eps:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _hard(rng):
    for _ in attempts(500, "a tolerance whose least n is distinct from its distractors"):
        p, q, r = rng.choice((1, 2, 3)), rng.choice((0, 1, 2, 3)), rng.choice((0, 1, 2, 5))
        b = rng.choice((2, 3, 4))
        side = rng.choice(("right", "left"))
        eps = rng.choice((Fraction(1, 2), Fraction(1, 4), Fraction(1, 5), Fraction(1, 10)))
        n_true = _least_n(p, q, r, b, side, eps)
        if not 6 <= n_true <= 150 or eps in (_error(p, q, r, b, side, n_true), _error(p, q, r, b, side, n_true - 1)):
            continue
        direct = right_sum if side == "right" else left_sum
        area = integral(p, q, r, b)
        check(
            abs(direct(p, q, r, b, n_true) - area) < eps <= abs(direct(p, q, r, b, n_true - 1) - area),
            "least n disagrees with the direct sums",
        )
        gap = Fraction(b) * (p * b * b + q * b)  # (b - a)(f(b) - f(a)): the error is at most gap/n
        n_bound = int(gap / eps) + 1
        _, b_, c_ = coefficients(p, q, r, b)
        n_lead = int(b_ / eps) + 1  # only the 1/n term of the error kept
        wrong = [
            (n_bound, "riemann-n-from-error-bound"),
            (n_lead, "riemann-n-drops-smaller-terms"),
            (n_true - 1, "riemann-n-off-by-one-at-threshold"),
            (n_true + 1, "riemann-n-off-by-one-at-threshold"),
        ]
        if len({w for w, _ in wrong} - {n_true}) >= 3:
            break
    stem = phrase(rng, _HARD_STEMS, f=_f_text(p, q, r), b=b, side=side)
    sign = "+" if side == "right" else "-"
    shown = lambda v: str(v)  # noqa: E731
    working = [
        f"With Δ = {b}/n and the sums of i and i^2, the {side}-endpoint sum is off by {shown(b_)}/n {sign} {shown(c_)}/n^2.",
        f"This is below {float(eps):g} first when n = {n_true}.",
    ]
    prompt = f"Find the least number of strips n for which S_n differs from the area under the curve by less than {float(eps):g}."
    return single(stem, fraction_part(prompt, n_true, 3, working, wrong))


@generator(
    "r4e_riemann_limit",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4e_riemann_limit(rng, tier: str = "easy"):
    """From a finite Riemann sum to the integral it approaches; see the module docstring for the ladder."""
    if tier == "easy":
        return draw(_easy, rng)
    if tier == "medium":
        return draw(_medium, rng)
    if tier == "hard":
        return draw(_hard, rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ---------------------------------------------------------------------------------------------------

_CHAIN_STEMS = (
    "Let f(x) = {f}. Dividing [0, {b}] into n equal strips and using right endpoints gives the sum R_n, which can be written R_n = A + B/n + C/n^2 for constants A, B and C. "
    "Use the results Σ i = n(n + 1)/2 and Σ i^2 = n(n + 1)(2n + 1)/6.",
    "The area under y = {f} from x = 0 to x = {b} is approximated by the sum R_n of n right-endpoint rectangles of equal width. "
    "Using Σ i = n(n + 1)/2 and Σ i^2 = n(n + 1)(2n + 1)/6, the sum can be written R_n = A + B/n + C/n^2.",
)


def _chain(rng):
    p, q, r = rng.choice((1, 2, 3)), rng.choice((1, 2, 3, 4)), rng.choice((0, 1, 2, 3))
    b = rng.choice((2, 3, 4))
    _check_coefficients(p, q, r, b)
    a_, b_, c_ = coefficients(p, q, r, b)
    big_n = rng.choice((10, 20, 25, 50))
    excess = b_ / big_n + c_ / big_n**2
    check(right_sum(p, q, r, b, big_n) - a_ == excess, "excess at N disagrees with the direct sum")
    eps = rng.choice((Fraction(1, 2), Fraction(1, 4), Fraction(1, 5), Fraction(1, 10)))
    n_true = _least_n(p, q, r, b, "right", eps)
    if not 6 <= n_true <= 150 or eps in (_error(p, q, r, b, "right", n_true), _error(p, q, r, b, "right", n_true - 1)):
        raise Redraw("least n out of range or on a boundary")
    check(
        right_sum(p, q, r, b, n_true) - a_ < eps <= right_sum(p, q, r, b, n_true - 1) - a_,
        "least n disagrees with the direct sums",
    )
    stem = phrase(rng, _CHAIN_STEMS, f=_f_text(p, q, r), b=b)
    gap = Fraction(b) * (p * b * b + q * b)
    part_a = fraction_part(
        "Find B, the coefficient of 1/n.",
        b_,
        2,
        [
            f"R_n = ({b}/n)·Σ f({b}i/n) = {p * b**3}(n + 1)(2n + 1)/(6n^2) + {q * b**2}(n + 1)/(2n) + {r * b}.",
            f"The 1/n terms give B = {rational(b_)}.",
        ],
        [
            (c_, "riemann-limit-takes-wrong-order-coefficient"),
            (Fraction(p * b**3, 2), "riemann-sum-formula-drops-linear-term"),
            (gap, "riemann-sum-formula-uses-difference-of-ends"),
            (Fraction(p * b**3 + q * b**2, 4), "riemann-sum-formula-halves-twice"),
        ],
        sensible_range=(0, None),
    )
    part_b = fraction_part(
        "Find A, the value that R_n approaches as n increases without bound.",
        a_,
        2,
        [f"As n grows the terms in 1/n and 1/n^2 tend to 0, so A = {rational(a_)}, the area under the curve."],
        [
            (a_ - r * b, "riemann-limit-drops-constant-term"),
            (a_ + b_, "riemann-limit-keeps-lower-order-terms"),
            (right_sum(p, q, r, b, 1), "riemann-limit-at-n-equals-one"),
        ],
    )
    c_num, c_den = c_.numerator, c_.denominator * big_n**2
    part_c = fraction_part(
        f"Hence find the amount by which R_{big_n} exceeds A.",
        excess,
        2,
        [f"R_n - A = B/n + C/n^2 with C = {rational(c_)}; at n = {big_n} this is {rational(excess)}."],
        [
            (b_ / big_n, "riemann-limit-keeps-lower-order-terms"),
            (a_ + excess, "reports-estimate-not-error"),
            (b_ / big_n - c_ / big_n**2, "riemann-left-right-sign"),
        ],
        follow_through=f"part_a/{big_n} + {c_num}/{c_den}",
        sensible_range=(0, None),
    )
    n_lead = int(b_ / eps) + 1
    part_d = fraction_part(
        f"Find the least value of n for which R_n exceeds A by less than {float(eps):g}.",
        n_true,
        2,
        [f"B/n + C/n^2 < {float(eps):g} first holds when n = {n_true}."],
        [
            (n_lead, "riemann-n-drops-smaller-terms"),
            (int(gap / eps) + 1, "riemann-n-from-error-bound"),
            (n_true - 1, "riemann-n-off-by-one-at-threshold"),
            (n_true + 1, "riemann-n-off-by-one-at-threshold"),
        ],
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


def rational(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


@generator("r4e_riemann_limit_chain", level=17, presets={"exam": dict(tier="exam")})
def r4e_riemann_limit_chain(rng, tier: str = "exam"):
    """R_n = A + B/n + C/n^2 for a quadratic: B, the limit A, the excess at a given n, the least n for a tolerance."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
