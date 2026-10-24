"""Generates `r3k_mgf` questions (easy, medium, hard) and the exam chain `exd_x_r3k_mgf`.

``uc2-mgf-moments-identify`` (``probability``, answer ``expression``, exact values only): moments of a random variable
from its moment generating function, and identifying a distribution from its MGF.

* easy: E(X) = M'(0) for an MGF given in closed form (binomial, Poisson, geometric, gamma-type or exponential).
* medium: Var(X) = M''(0) - M'(0)^2 for the same families.
* hard: one of four questions that need the MGF to be recognised or transformed: P(X + Y = k) for independent binomials or
  independent Poissons given by their MGFs; the MGF of Y = aX + b (answer an expression in t); or the third moment
  E(X^3) = M'''(0) of a gamma-type MGF.
* exam (``exd_x_r3k_mgf``): a binomial or Poisson MGF: (a) E(X), (b) Var(X) (follows through from (a)), (c) P(X = k), (d)
  Var(aX + b) (follows through from (b)).

Each moment is read off the MGF by differentiating with SymPy and again from the known moments of the identified
distribution (and, for the probabilities, from a direct sum over the probability mass function).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy

from ..core import VerificationError, attempts, check, generator
from ._r3k_kit import (
    clean_working,
    stored,
    second_route,
    expr_distractors,
    expr_fields,
    frac_distractors,
    frac_fields,
    fr,
    need,
    pretty,
    rat,
)

t = sympy.Symbol("t")


def _ft(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}" if value.denominator != 1 else str(value.numerator)


def _coef_e(c: Fraction, arg: str = "t") -> str:
    """`c e^t` written as `e^t`, `3e^t/5` or `5e^t`."""
    if c.denominator == 1:
        return "e^t" if c.numerator == 1 else f"{c.numerator}e^t"
    return f"e^t/{c.denominator}" if c.numerator == 1 else f"{c.numerator}e^t/{c.denominator}"


@dataclass(frozen=True)
class Family:
    name: str
    text: str
    mgf: sympy.Expr
    mean: sympy.Rational
    var: sympy.Rational
    mean_slips: tuple
    var_slips: tuple
    params: tuple = ()


def _binomial(rng) -> Family:
    n = rng.randint(3, 8)
    p = rng.choice((Fraction(1, 3), Fraction(2, 5), Fraction(1, 4), Fraction(3, 4), Fraction(2, 3), Fraction(3, 5)))
    q = 1 - p
    mgf = (rat(q) + rat(p) * sympy.exp(t)) ** n
    text = f"({_ft(q)} + {_coef_e(p)})^{n}"
    mean, var = n * rat(p), n * rat(p) * rat(q)
    return Family(
        "binomial",
        text,
        mgf,
        mean,
        var,
        ((n, "r3k-mgf-outer-derivative-only"), (n * rat(q), "r3k-mgf-swapped-p-and-q")),
        ((n * rat(p), "r3k-mgf-variance-is-the-mean"), (n * rat(q), "r3k-mgf-swapped-p-and-q")),
        (n, p),
    )


def _poisson(rng) -> Family:
    lam = rng.randint(2, 7)
    mgf = sympy.exp(lam * (sympy.exp(t) - 1))
    text = f"e^({lam}(e^t - 1))"
    return Family(
        "poisson",
        text,
        mgf,
        sympy.Integer(lam),
        sympy.Integer(lam),
        ((lam**2, "r3k-mgf-squared-the-rate"), (lam + lam**2, "r3k-mgf-second-moment-for-mean")),
        ((lam**2, "r3k-mgf-squared-the-rate"), (lam**2 + lam, "r3k-mgf-variance-is-second-moment")),
        (lam,),
    )


def _geometric(rng) -> Family:
    p = rng.choice((Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(1, 5), Fraction(2, 5), Fraction(2, 3)))
    q = 1 - p
    mgf = rat(p) * sympy.exp(t) / (1 - rat(q) * sympy.exp(t))
    text = f"({_coef_e(p)})/(1 - {_coef_e(q)})"
    mean, var = 1 / rat(p), rat(q) / rat(p) ** 2
    return Family(
        "geometric",
        text,
        mgf,
        mean,
        var,
        ((rat(q) / rat(p), "r3k-mgf-failures-for-trials"), (1 / rat(q), "r3k-mgf-swapped-p-and-q")),
        ((1 / rat(p) ** 2, "r3k-mgf-variance-forgot-q"), (rat(p) / rat(q) ** 2, "r3k-mgf-swapped-p-and-q")),
    )


def _gamma(rng) -> Family:
    a = rng.randint(2, 5)
    b = rng.randint(2, 4)
    mgf = (1 - b * t) ** (-a)
    text = f"(1 - {b}t)^(-{a})"
    return Family(
        "gamma",
        text,
        mgf,
        sympy.Integer(a * b),
        sympy.Integer(a * b * b),
        ((a, "r3k-mgf-outer-derivative-only"), (b, "r3k-mgf-scale-for-mean")),
        ((a * b, "r3k-mgf-variance-is-the-mean"), (a * b**2 + a**2 * b**2, "r3k-mgf-variance-is-second-moment")),
    )


def _exponential(rng) -> Family:
    lam = rng.randint(2, 6)
    mgf = sympy.Integer(lam) / (lam - t)
    text = f"{lam}/({lam} - t)"
    return Family(
        "exponential",
        text,
        mgf,
        sympy.Rational(1, lam),
        sympy.Rational(1, lam**2),
        ((lam, "r3k-mgf-rate-for-mean"), (sympy.Rational(1, lam**2), "r3k-mgf-outer-derivative-only")),
        (
            (sympy.Rational(1, lam), "r3k-mgf-variance-is-the-mean"),
            (sympy.Rational(2, lam**2), "r3k-mgf-variance-is-second-moment"),
        ),
    )


_FAMILIES = (_binomial, _poisson, _geometric, _gamma, _exponential)


def _exact(value):
    return value if value.is_Rational else sympy.nsimplify(value)


def _moments_by_diff(mgf):
    d1 = sympy.diff(mgf, t).subs(t, 0)
    d2 = sympy.diff(mgf, t, 2).subs(t, 0)
    return _exact(d1), _exact(d2)


def _check_family(fam: Family):
    d1, d2 = _moments_by_diff(fam.mgf)
    if second_route():
        check(sympy.simplify(d1 - fam.mean) == 0, "M'(0) disagrees with the mean of the identified distribution")
        check(sympy.simplify(d2 - d1**2 - fam.var) == 0, "M''(0) - M'(0)^2 disagrees with the variance")
    return d1, d2


_LEAD = (
    "The random variable X has moment generating function M(t) = {mgf}.",
    "A random variable X has moment generating function M_X(t) = {mgf}, valid for t in an interval about 0.",
)


def _easy(rng):
    fam = rng.choice(_FAMILIES)(rng)
    d1, d2 = _check_family(fam)
    key = fam.mean
    cands = list(fam.mean_slips) + [
        (1, "r3k-mgf-evaluated-the-function"),
        (d2, "r3k-mgf-second-moment-for-mean"),
        (fam.var, "r3k-mgf-variance-for-mean"),
    ]
    dist = need(expr_distractors(key, cands))
    text = f"{rng.choice(_LEAD).format(mgf=fam.text)} Find E(X)."
    return dict(text=text, distractors=dist, working=["E(X) = M'(0).", f"E(X) = {key}."], **expr_fields(key))


def _medium(rng):
    fam = rng.choice(_FAMILIES)(rng)
    d1, d2 = _check_family(fam)
    key = fam.var
    cands = list(fam.var_slips) + [
        (d2, "r3k-mgf-variance-is-second-moment"),
        (d2 + d1**2, "r3k-mgf-variance-wrong-sign"),
        (d1**2, "r3k-mgf-variance-is-mean-squared"),
    ]
    dist = need(expr_distractors(key, cands))
    text = f"{rng.choice(_LEAD).format(mgf=fam.text)} Find Var(X)."
    return dict(
        text=text,
        distractors=dist,
        working=["E(X) = M'(0) and E(X²) = M''(0).", f"Var(X) = M''(0) - M'(0)² = {d2} - {d1**2} = {key}."],
        **expr_fields(key),
    )


# -- hard ---------------------------------------------------------------------------------------------------------


def _product_of_marginals(p, q, n1, n2, k):
    """P(X = k // 2) P(Y = k - k // 2), the slip of multiplying probabilities instead of convolving them."""
    k1 = k // 2
    k2 = k - k1
    if k1 > n1 or k2 > n2:
        return None
    px = Fraction(int(sympy.binomial(n1, k1))) * p**k1 * q ** (n1 - k1)
    py = Fraction(int(sympy.binomial(n2, k2))) * p**k2 * q ** (n2 - k2)
    return px * py


def _hard_binomial_sum(rng):
    p = rng.choice((Fraction(1, 3), Fraction(2, 5), Fraction(1, 4), Fraction(3, 4), Fraction(2, 3)))
    q = 1 - p
    n1, n2 = rng.randint(2, 4), rng.randint(2, 4)
    n = n1 + n2
    k = rng.randint(2, n - 1)
    key = Fraction(int(sympy.binomial(n, k))) * p**k * q ** (n - k)
    # second route: the coefficient of u^k in the generating polynomial (q + p u)^n1 (q + p u)^n2
    u = sympy.Symbol("u")
    coeff = sympy.Poly(sympy.expand((rat(q) + rat(p) * u) ** n1 * (rat(q) + rat(p) * u) ** n2), u).coeff_monomial(u**k)
    check(fr(coeff) == key, "the product of generating functions disagrees")
    cands = [
        (
            Fraction(int(sympy.binomial(n1, min(k, n1)))) * p ** min(k, n1) * q ** (n1 - min(k, n1)),
            "r3k-mgf-sum-uses-one-variable",
        ),
        (Fraction(int(sympy.binomial(n, k))) * q**k * p ** (n - k), "r3k-mgf-swapped-p-and-q"),
        (p**k * q ** (n - k), "r3k-mgf-sum-forgot-the-coefficient"),
        (_product_of_marginals(p, q, n1, n2, k), "r3k-mgf-sum-product-of-probabilities"),
    ]
    cands = [(rat(v), tag) for v, tag in cands if v is not None and 0 < v < 1]
    dist = need(expr_distractors(rat(key), cands))
    text = (
        f"X and Y are independent random variables with moment generating functions M_X(t) = ({_ft(q)} + {_coef_e(p)})^{n1} "
        f"and M_Y(t) = ({_ft(q)} + {_coef_e(p)})^{n2}. Find P(X + Y = {k})."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"M_(X+Y)(t) = M_X(t) M_Y(t) = ({_ft(q)} + {_coef_e(p)})^{n}, the MGF of a binomial distribution with n = {n}, p = {_ft(p)}.",
            f"P(X + Y = {k}) = C({n}, {k}) p^{k} q^{n - k} = {key}.",
        ],
        **expr_fields(rat(key)),
    )


def _hard_poisson_sum(rng):
    l1, l2 = rng.randint(1, 4), rng.randint(1, 4)
    lam = l1 + l2
    k = rng.randint(1, 5)
    key = sympy.Integer(lam) ** k * sympy.exp(-lam) / sympy.factorial(k)
    direct = sum(
        sympy.Integer(l1) ** j
        * sympy.exp(-l1)
        / sympy.factorial(j)
        * sympy.Integer(l2) ** (k - j)
        * sympy.exp(-l2)
        / sympy.factorial(k - j)
        for j in range(k + 1)
    )
    if second_route():
        check(sympy.simplify(direct - key) == 0, "the convolution sum disagrees with the Poisson(l1 + l2) probability")
    cands = [
        (sympy.Integer(l1 * l2) ** k * sympy.exp(-l1 * l2) / sympy.factorial(k), "r3k-mgf-poisson-product-of-rates"),
        (sympy.Integer(lam) ** k * sympy.exp(-lam), "r3k-mgf-poisson-forgot-factorial"),
        (sympy.Integer(l1) ** k * sympy.exp(-l1) / sympy.factorial(k), "r3k-mgf-sum-uses-one-variable"),
        (sympy.Integer(lam) ** k * sympy.exp(-1) / sympy.factorial(k), "r3k-mgf-poisson-wrong-exponent"),
    ]
    dist = need(expr_distractors(key, cands))
    text = (
        f"X and Y are independent random variables with moment generating functions M_X(t) = e^({l1}(e^t - 1)) and "
        f"M_Y(t) = e^({l2}(e^t - 1)). Find P(X + Y = {k}), as an exact expression."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"M_(X+Y)(t) = e^({lam}(e^t - 1)), the MGF of a Poisson distribution with mean {lam}.",
            f"P(X + Y = {k}) = e^(-{lam}) {lam}^{k}/{k}! = {pretty(key)}.",
        ],
        **expr_fields(key),
    )


def _hard_linear(rng):
    fam = rng.choice((_binomial, _poisson))(rng)
    a = rng.choice((2, 3, -1, -2))
    b = rng.choice((-3, -2, -1, 1, 2, 3, 4))
    mx = fam.mgf
    key = sympy.exp(b * t) * mx.subs(t, a * t)
    # second route: the moments of Y from its MGF agree with E(aX + b) and Var(aX + b)
    d1, d2 = _moments_by_diff(key)
    if second_route():
        check(sympy.simplify(d1 - (a * fam.mean + b)) == 0, "E(Y) from M_Y disagrees with a E(X) + b")
        check(sympy.simplify(d2 - d1**2 - a**2 * fam.var) == 0, "Var(Y) from M_Y disagrees with a^2 Var(X)")
    cands = [
        (sympy.exp(b * t) * mx, "r3k-mgf-linear-forgot-the-scale"),
        (mx.subs(t, a * t), "r3k-mgf-linear-forgot-the-shift"),
        (mx.subs(t, a * t + b), "r3k-mgf-linear-shift-inside"),
        (sympy.exp(b * t) * mx**a if a > 0 else None, "r3k-mgf-linear-power-for-scale"),
        (sympy.exp(a * t) * mx.subs(t, b * t), "r3k-mgf-linear-swapped-a-and-b"),
    ]
    dist = need(expr_distractors(key, cands, symbols=(t,)))
    lead = {1: "", -1: "-"}.get(a, str(a))
    sign = f"{lead}X + {b}" if b > 0 else f"{lead}X - {-b}"
    text = (
        f"The random variable X has moment generating function M_X(t) = {fam.text}. Let Y = {sign}. Find the moment "
        f"generating function M_Y(t) of Y."
    )
    return dict(
        text=text,
        distractors=dist,
        working=["M_Y(t) = E(e^(t(aX + b))) = e^(bt) M_X(at).", f"M_Y(t) = {pretty(key)}."],
        **expr_fields(key),
    )


def _hard_third_moment(rng):
    a = rng.randint(2, 4)
    b = rng.randint(2, 3)
    mgf = (1 - b * t) ** (-a)
    d3 = sympy.diff(mgf, t, 3).subs(t, 0)
    key = sympy.Integer(a * (a + 1) * (a + 2) * b**3)
    if second_route():
        check(sympy.nsimplify(d3) == key, "M'''(0) disagrees with a(a+1)(a+2)b^3")
    cands = [
        (a * b**3, "r3k-mgf-third-moment-forgot-factors"),
        (a * (a + 1) * b**3, "r3k-mgf-third-moment-two-factors"),
        (sympy.Integer(a * b) ** 3, "r3k-mgf-third-moment-cube-of-the-mean"),
        (a * (a + 1) * (a + 2) * b**2, "r3k-mgf-third-moment-wrong-power"),
    ]
    dist = need(expr_distractors(key, cands))
    text = f"The random variable X has moment generating function M(t) = (1 - {b}t)^(-{a}). Find E(X³)."
    return dict(
        text=text,
        distractors=dist,
        working=[f"E(X³) = M'''(0); each derivative brings down a factor ({a} + j) and {b}.", f"E(X³) = {key}."],
        **expr_fields(key),
    )


@generator(
    "r3k_mgf",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@clean_working
def r3k_mgf(rng, tier="easy"):
    """Moments and identification from a moment generating function; see the module docstring for the tiers."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice((_hard_binomial_sum, _hard_poisson_sum, _hard_linear, _hard_third_moment))(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_x_r3k_mgf", level=19, presets={"exam": dict(tier="exam")})
@clean_working
def exd_x_r3k_mgf(rng, tier="exam"):
    """E(X), Var(X), P(X = k) and Var(aX + b) for a binomial or Poisson MGF."""
    check(tier == "exam", f"unknown tier {tier!r}")
    # A family whose slips collapse onto the key leaves a part with one distractor: redraw the whole chain.
    for _ in attempts(60, "an exam MGF chain whose parts each keep two distractors"):
        try:
            return _exam_once(rng)
        except VerificationError as err:
            if "distractors" not in str(err):
                raise


def _exam_once(rng):
    fam = rng.choice((_binomial, _poisson))(rng)
    d1, d2 = _check_family(fam)
    mean, var = fam.mean, fam.var
    a = rng.choice((2, 3, 4, -2, -3))
    b = rng.randint(1, 5)
    if fam.name == "binomial":
        n, p_frac = fam.params
        p = rat(p_frac)
        k = rng.randint(1, n - 1)
        pmf = sympy.binomial(n, k) * p**k * (1 - p) ** (n - k)
        # second route: the pmf has mean n p
        check(sum(j * sympy.binomial(n, j) * p**j * (1 - p) ** (n - j) for j in range(n + 1)) == mean, "pmf mean")
        c_kind = "fraction"
        c_fields = frac_fields(fr(pmf))
        c_cands = [
            (sympy.binomial(n, k) * (1 - p) ** k * p ** (n - k), "r3k-mgf-swapped-p-and-q"),
            (p**k * (1 - p) ** (n - k), "r3k-mgf-sum-forgot-the-coefficient"),
            (sympy.binomial(n, k - 1) * p ** (k - 1) * (1 - p) ** (n - k + 1), "r3k-mgf-pmf-wrong-index"),
            (sympy.binomial(n, k) * p**k, "r3k-mgf-pmf-forgot-failures"),
            (sympy.binomial(n, k) * (1 - p) ** (n - k), "r3k-mgf-pmf-forgot-successes"),
        ]
        c_dist = need(
            frac_distractors(fr(pmf), [(fr(v), tag) for v, tag in c_cands], positive=True, at_most=Fraction(1))
        )
        pmf_note = f"X is binomial with n = {n}, p = {p}: P(X = {k}) = C({n}, {k}) p^{k} q^{n - k} = {pmf}."
    else:
        lam = mean
        k = rng.randint(1, 4)
        pmf = lam**k * sympy.exp(-lam) / sympy.factorial(k)
        c_fields = expr_fields(pmf)
        c_cands = [
            (lam**k * sympy.exp(-lam), "r3k-mgf-poisson-forgot-factorial"),
            (lam**k * sympy.exp(-1) / sympy.factorial(k), "r3k-mgf-poisson-wrong-exponent"),
            (sympy.Integer(k) ** lam * sympy.exp(-lam) / sympy.factorial(k), "r3k-mgf-pmf-wrong-index"),
        ]
        c_dist = need(expr_distractors(pmf, c_cands))
        pmf_note = f"X is Poisson with mean {lam}: P(X = {k}) = e^(-{lam}) {lam}^{k}/{k}! = {pretty(pmf)}."
    var_y = a * a * var
    a_dist = need(
        frac_distractors(
            fr(mean),
            [(fr(1), "r3k-mgf-evaluated-the-function"), (fr(d2), "r3k-mgf-second-moment-for-mean")]
            + [(fr(v), tag) for v, tag in fam.mean_slips],
            positive=True,
        )
    )
    # a reply right by follow-through from a wrong earlier part must never be a later part's own distractor
    b_dist = need(
        frac_distractors(
            fr(var),
            [
                (fr(d2), "r3k-mgf-variance-is-second-moment"),
                (fr(d2 + d1**2), "r3k-mgf-variance-wrong-sign"),
                (fr(d1**2), "r3k-mgf-variance-is-mean-squared"),
            ]
            + [(fr(v), tag) for v, tag in fam.var_slips],
            positive=True,
            avoid=[fr(d2) - stored(d) ** 2 for d in a_dist],
        )
    )
    d_dist = need(
        frac_distractors(
            fr(var_y),
            [
                (fr(a * var), "r3k-mgf-linear-forgot-to-square"),
                (fr(a * a * var + b), "r3k-mgf-linear-shift-added-to-variance"),
                (fr(var), "r3k-mgf-linear-forgot-the-scale"),
                (fr(a * a * var + b * b), "r3k-mgf-linear-shift-added-to-variance"),
            ],
            positive=True,
            avoid=[a * a * stored(d) for d in b_dist],
        )
    )
    sign = f"{a}X + {b}"
    return dict(
        text=f"The random variable X has moment generating function M(t) = {fam.text}.",
        parts=[
            dict(
                prompt="Find E(X).",
                marks=2,
                sensible_range=(0, None),
                working=["E(X) = M'(0).", f"E(X) = {mean}."],
                distractors=a_dist,
                **frac_fields(fr(mean)),
            ),
            dict(
                prompt="Find Var(X).",
                marks=3,
                sensible_range=(0, None),
                follow_through=f"{sympy.sstr(d2)} - part_a**2",
                working=[f"E(X²) = M''(0) = {d2}.", f"Var(X) = {d2} - E(X)² = {var}."],
                distractors=b_dist,
                **frac_fields(fr(var)),
            ),
            dict(
                prompt=f"Identify the distribution of X and find P(X = {k}).",
                marks=2,
                working=[pmf_note],
                distractors=c_dist,
                **c_fields,
            ),
            dict(
                prompt=f"Let Y = {sign}. Find Var(Y).",
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{a * a}*part_b",
                working=[f"Var(aX + b) = a² Var(X) = {a * a} × {var} = {var_y}."],
                distractors=d_dist,
                **frac_fields(fr(var_y)),
            ),
        ],
    )
