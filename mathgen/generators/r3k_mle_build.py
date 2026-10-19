"""Generates `r3k_mle` questions (easy, medium, hard) and the exam chain `exd_x_r3k_mle`.

``uc2-mle-method-of-moments`` (``statistics``, answer ``expression``, exact values only): maximum likelihood and method
of moments estimators.

* easy: the maximum likelihood estimate from a data summary for an exponential rate (given the total of the lifetimes),
  a Poisson mean (given a frequency table of counts) or a geometric success probability (given the trial numbers).
* medium: the estimator itself, as an expression in n and a stated statistic, for a density whose likelihood must be
  differentiated: a power density, a Pareto density, a gamma-type density, a shifted exponential, a zero-mean normal
  variance, a Rayleigh-type density or a geometric probability.
* hard: one of four numerical questions: a method of moments estimate for f(x; a) = (a + 1) x^a on (0, 1); the maximum
  likelihood or method of moments estimate for a uniform (0, a) sample (the likelihood has no stationary point); the
  maximum likelihood variance of a normal sample from sums; or the maximum likelihood estimate of an exponential
  tail probability by invariance.
* exam (``exd_x_r3k_mle``): an exponential sample with n and the total T: (a) the log-likelihood as an expression in the
  rate r, (b) the maximum likelihood estimate of r, (c) the estimate of the mean 1/r (follows through from (b)), (d) the
  estimate of P(X > t), to 3 decimal places (follows through from (b)).

Each estimate is found from the closed form and again by solving d(log-likelihood)/d(parameter) = 0 with SymPy (or, for
the uniform, by scanning the likelihood), and each method of moments estimate by solving E(X) = sample mean.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r3k_kit import (
    clean_working,
    expr_distractors,
    expr_fields,
    frac_distractors,
    frac_fields,
    fr,
    need,
    num_distractors,
    num_fields,
    pretty,
    stored,
    rat,
)

n_, T_, L_, W_ = sympy.symbols("n T L W")
par = sympy.Symbol("a")
x = sympy.Symbol("x", positive=True)


def _ft(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


# -- easy: numeric data -------------------------------------------------------------------------------------------


def _easy_exponential(rng):
    n = rng.choice((8, 10, 12, 15, 20, 25))
    for _ in attempts(200, "a total with a readable estimate"):
        total = rng.randint(n, 4 * n)
        est = Fraction(n, total)
        if est.denominator <= 20 and est != 1:
            break
    r = sympy.Symbol("r", positive=True)
    loglik = n * sympy.log(r) - r * total
    solved = sympy.solve(sympy.diff(loglik, r), r)[0]
    check(fr(solved) == est, "the score equation disagrees with n / T")
    cands = [
        (Fraction(total, n), "r3k-mle-mean-for-rate"),
        (Fraction(n - 1, total), "r3k-mle-n-minus-one"),
        (Fraction(1, total), "r3k-mle-forgot-n"),
        (Fraction(n * n, total), "r3k-mle-n-squared"),
    ]
    dist = need(frac_distractors(est, cands, positive=True))
    unit = rng.choice(("hours", "years", "minutes"))
    text = (
        f"A sample of {n} independent lifetimes is modelled by the density f(x; r) = r e^(-rx) for x > 0. The lifetimes "
        f"total {total} {unit}. Find the maximum likelihood estimate of r."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"The log-likelihood is n ln r - r T, with derivative n/r - T, which is zero at r = n/T.",
            f"r̂ = {n}/{total} = {est}.",
        ],
        **expr_fields(sympy.Rational(est.numerator, est.denominator)),
    )


def _easy_poisson(rng):
    ks = [0, 1, 2, 3, 4, 5]
    for _ in attempts(300, "a frequency table with a readable mean"):
        freqs = [rng.randint(1, 9) for _ in ks[:5]] + [rng.randint(0, 3)]
        n = sum(freqs)
        total = sum(k * f for k, f in zip(ks, freqs))
        est = Fraction(total, n)
        if n >= 20 and est.denominator <= 12 and est.numerator != est.denominator:
            break
    m = sympy.Symbol("m", positive=True)
    loglik = sum(f * (k * sympy.log(m) - m - sympy.log(sympy.factorial(k))) for k, f in zip(ks, freqs))
    solved = sympy.solve(sympy.diff(loglik, m), m)[0]
    check(fr(solved) == est, "the score equation disagrees with the sample mean")
    modal = ks[max(range(len(ks)), key=lambda i: freqs[i])]
    cands = [
        (Fraction(total, n - 1), "r3k-mle-n-minus-one"),
        (Fraction(modal), "r3k-mle-mode-for-mean"),
        (Fraction(total, len(ks)), "r3k-mle-ignored-frequencies"),
        (Fraction(sum(ks), len(ks)), "r3k-mle-ignored-frequencies"),
        (Fraction(total), "r3k-mle-total-not-mean"),
    ]
    dist = need(frac_distractors(est, cands, positive=True))
    table = "; ".join(f"{k}: {f}" for k, f in zip(ks, freqs) if f)
    text = (
        f"The number of customers arriving at a kiosk in each of {n} one-minute intervals is modelled as Poisson with mean "
        f"m. The counts observed (count: number of intervals) were {table}. Find the maximum likelihood estimate of m."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[f"The Poisson maximum likelihood estimate is the sample mean: {total}/{n}.", f"m̂ = {est}."],
        **expr_fields(sympy.Rational(est.numerator, est.denominator)),
    )


def _easy_geometric(rng):
    for _ in attempts(300, "a geometric sample with a readable estimate"):
        n = rng.randint(5, 9)
        xs = [rng.randint(1, 7) for _ in range(n)]
        total = sum(xs)
        est = Fraction(n, total)
        if est.denominator <= 25 and est != 1:
            break
    p = sympy.Symbol("p", positive=True)
    loglik = sum(sympy.log(p) + (xi - 1) * sympy.log(1 - p) for xi in xs)
    solved = sympy.solve(sympy.diff(loglik, p), p)[0]
    check(fr(solved) == est, "the score equation disagrees with n / T")
    cands = [
        (Fraction(total, n), "r3k-mle-mean-for-rate"),
        (Fraction(n, total - n) if total != n else None, "r3k-mle-forgot-the-final-success"),
        (Fraction(n - 1, total), "r3k-mle-n-minus-one"),
        (1 - est, "r3k-mle-complement"),
    ]
    dist = need(frac_distractors(est, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"A trainee tries a task until the first success, with the same unknown success probability p on every independent "
        f"attempt. The number of attempts needed on {n} separate occasions was {', '.join(map(str, xs))}. Find the maximum "
        f"likelihood estimate of p."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[f"The likelihood is p^n (1 - p)^(T - n), maximised at p = n/T.", f"p̂ = {n}/{total} = {est}."],
        **expr_fields(sympy.Rational(est.numerator, est.denominator)),
    )


def _easy(rng):
    return rng.choice((_easy_exponential, _easy_poisson, _easy_geometric))(rng)


# -- medium: the estimator as an expression -----------------------------------------------------------------------

# (density text, statistic definition, statistic symbol, log-likelihood per observation, parameter symbol name, estimator,
#  slips)
_FAMILIES = (
    dict(
        name="power",
        probe=-6,
        density="f(x; a) = a x^(a - 1) for 0 < x < 1",
        stat="L = ln x_1 + ln x_2 + ... + ln x_n",
        sym=L_,
        total=lambda n, s: n * sympy.log(par) + (par - 1) * s,
        estimator=-n_ / L_,
        slips=(
            (n_ / L_, "r3k-mle-sign-error"),
            (-L_ / n_, "r3k-mle-inverted"),
            (-(n_ - 1) / L_, "r3k-mle-n-minus-one"),
            (-n_ * L_, "r3k-mle-multiplied"),
        ),
        param="a",
    ),
    dict(
        name="pareto",
        probe=4,
        density="f(x; a) = a/x^(a + 1) for x > 1",
        stat="L = ln x_1 + ln x_2 + ... + ln x_n",
        sym=L_,
        total=lambda n, s: n * sympy.log(par) - (par + 1) * s,
        estimator=n_ / L_,
        slips=(
            (L_ / n_, "r3k-mle-inverted"),
            (n_ / (L_ + n_), "r3k-mle-pareto-exponent-slip"),
            (-n_ / L_, "r3k-mle-sign-error"),
            ((n_ - 1) / L_, "r3k-mle-n-minus-one"),
        ),
        param="a",
    ),
    dict(
        name="gamma2",
        probe=10,
        density="f(x; a) = x e^(-x/a)/a^2 for x > 0",
        stat="T = x_1 + x_2 + ... + x_n",
        sym=T_,
        total=lambda n, s: -2 * n * sympy.log(par) - s / par,
        estimator=T_ / (2 * n_),
        slips=(
            (T_ / n_, "r3k-mle-forgot-the-shape"),
            (2 * T_ / n_, "r3k-mle-doubled-not-halved"),
            (n_ / T_, "r3k-mle-inverted"),
            (T_ / (2 * (n_ - 1)), "r3k-mle-n-minus-one"),
        ),
        param="a",
    ),
    dict(
        name="shifted_exponential",
        probe=9,
        density="f(x; a) = a e^(-a(x - 1)) for x > 1",
        stat="T = x_1 + x_2 + ... + x_n",
        sym=T_,
        total=lambda n, s: n * sympy.log(par) - par * (s - n),
        estimator=n_ / (T_ - n_),
        slips=(
            (n_ / T_, "r3k-mle-ignored-the-shift"),
            ((T_ - n_) / n_, "r3k-mle-inverted"),
            (T_ / n_, "r3k-mle-mean-for-rate"),
            (n_ / (T_ + n_), "r3k-mle-shift-sign"),
        ),
        param="a",
    ),
    dict(
        name="normal_variance",
        probe=20,
        density="f(x; a) = e^(-x²/(2a)) / √(2πa), the normal density with mean 0 and variance a",
        stat="W = x_1² + x_2² + ... + x_n²",
        sym=W_,
        total=lambda n, s: -n * sympy.log(par) / 2 - s / (2 * par),
        estimator=W_ / n_,
        slips=(
            (W_ / (n_ - 1), "r3k-mle-n-minus-one"),
            (sympy.sqrt(W_ / n_), "r3k-mle-sd-for-variance"),
            (W_ / (2 * n_), "r3k-mle-spurious-factor"),
            (n_ / W_, "r3k-mle-inverted"),
        ),
        param="a",
    ),
    dict(
        name="rayleigh",
        probe=20,
        density="f(x; a) = (2x/a) e^(-x²/a) for x > 0",
        stat="W = x_1² + x_2² + ... + x_n²",
        sym=W_,
        total=lambda n, s: -n * sympy.log(par) - s / par,
        estimator=W_ / n_,
        slips=(
            (W_ / (2 * n_), "r3k-mle-spurious-factor"),
            (2 * W_ / n_, "r3k-mle-doubled-not-halved"),
            (n_ / W_, "r3k-mle-inverted"),
            (W_ / (n_ - 1), "r3k-mle-n-minus-one"),
        ),
        param="a",
    ),
)


def _medium(rng):
    fam = rng.choice(_FAMILIES)
    sym = fam["sym"]
    loglik = fam["total"](n_, sym)
    solved = sympy.solve(sympy.diff(loglik, par), par)
    check(
        len(solved) == 1 and sympy.simplify(solved[0] - fam["estimator"]) == 0,
        "the score equation disagrees with the closed form",
    )
    # the stationary point is a maximum: probe the log-likelihood at a typical statistic (n = 5)
    values = {n_: 5, sym: fam["probe"]}
    best = sympy.N(fam["estimator"].subs(values))
    peak = sympy.N(loglik.subs(values).subs(par, best))
    check(
        all(
            peak > sympy.N(loglik.subs(values).subs(par, best * f))
            for f in (sympy.Rational(9, 10), sympy.Rational(11, 10))
        ),
        "not a maximum",
    )
    dist = need(expr_distractors(fam["estimator"], fam["slips"], symbols=(n_, sym)))
    opening = rng.choice(
        (
            "A random sample x_1, x_2, ..., x_n is taken from the distribution with",
            "Independent observations x_1, x_2, ..., x_n are drawn from the distribution with",
            "The observations x_1, x_2, ..., x_n form a random sample from the distribution with",
            "A sample x_1, x_2, ..., x_n of independent values comes from the distribution with",
        )
    )
    request = rng.choice(
        (
            f"Find the maximum likelihood estimator of a, as an expression in n and {sym}.",
            f"Determine the maximum likelihood estimator of a, as an expression in n and {sym}.",
        )
    )
    text = f"{opening} density {fam['density']}, where a > 0 is unknown. Let {fam['stat']}. {request}"
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"The log-likelihood is {pretty(loglik)}; set its derivative with respect to a to zero.",
            f"â = {pretty(fam['estimator'])}.",
        ],
        **expr_fields(fam["estimator"]),
    )


# -- hard: method of moments, the uniform, two-parameter normal, invariance ---------------------------------------


def _hard_mom_power(rng):
    for _ in attempts(300, "a sample mean with a readable moment estimate"):
        den = rng.choice((3, 4, 5, 6, 8, 10))
        num = rng.randint(den // 2 + 1, den - 1)
        xbar = Fraction(num, den)
        est = (2 * xbar - 1) / (1 - xbar)
        if est > 0 and est.denominator <= 6:
            break
    a = sympy.Symbol("a", positive=True)
    mean = sympy.integrate((a + 1) * x ** (a + 1), (x, 0, 1))
    solved = sympy.solve(sympy.Eq(mean, rat(xbar)), a)[0]
    check(fr(solved) == est, "solving E(X) = x-bar disagrees with the closed form")
    cands = [
        (xbar / (1 - xbar), "r3k-mle-mom-wrong-mean-formula"),
        ((1 - xbar) / (2 * xbar - 1), "r3k-mle-inverted"),
        (xbar, "r3k-mle-mom-estimate-is-the-mean"),
        (2 * xbar - 1, "r3k-mle-mom-forgot-to-solve"),
    ]
    dist = need(frac_distractors(est, cands, positive=True))
    n = rng.choice((20, 30, 40, 50))
    text = (
        f"A random sample of size {n} is taken from the distribution with density f(x; a) = (a + 1)x^a for 0 < x < 1, where "
        f"a > -1 is unknown. The sample mean is {_ft(xbar)}. Find the method of moments estimate of a."
    )
    return dict(
        text=text,
        distractors=dist,
        working=["E(X) = (a + 1)/(a + 2). Set this equal to the sample mean and solve for a.", f"ã = {est}."],
        **expr_fields(sympy.Rational(est.numerator, est.denominator)),
    )


def _hard_uniform(rng):
    n = rng.randint(5, 7)
    mle_case = rng.random() < 0.5
    for _ in attempts(300, "a uniform sample"):
        data = sorted(rng.sample(range(1, 30), n))
        total = sum(data)
        mom = Fraction(2 * total, n)
        if mom.denominator <= 7 and max(data) != mom:
            break
    correct = Fraction(max(data)) if mle_case else mom
    # second route: the likelihood theta^-n is decreasing, so it is largest at the smallest admissible theta (the maximum)
    admissible = [t for t in range(max(data), max(data) + 5)]
    check(
        max(admissible, key=lambda t: Fraction(1, t) ** n) == max(data),
        "the likelihood is not maximised at the sample maximum",
    )
    check(Fraction(2 * total, n) == mom, "2 x-bar arithmetic")
    if mle_case:
        cands = [
            (mom, "r3k-mle-uniform-moment-estimate"),
            (Fraction(n + 1, n) * max(data), "r3k-mle-uniform-unbiased-adjustment"),
            (Fraction(total, n), "r3k-mle-mom-estimate-is-the-mean"),
            (Fraction(max(data) + min(data)), "r3k-mle-uniform-midrange"),
        ]
        which = "maximum likelihood"
    else:
        cands = [
            (Fraction(max(data)), "r3k-mle-uniform-sample-maximum"),
            (Fraction(total, n), "r3k-mle-mom-estimate-is-the-mean"),
            (Fraction(n + 1, n) * max(data), "r3k-mle-uniform-unbiased-adjustment"),
            (Fraction(4 * total, n), "r3k-mle-mom-doubled-twice"),
        ]
        which = "method of moments"
    dist = need(frac_distractors(correct, cands, positive=True))
    text = (
        f"A random sample from the uniform distribution on (0, a), where a > 0 is unknown, is {', '.join(map(str, data))}. "
        f"Find the {which} estimate of a."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            "The likelihood a^(-n) is decreasing in a, so it is largest at the smallest a allowed by the data, the sample maximum."
            if mle_case
            else "E(X) = a/2, so set a/2 equal to the sample mean.",
            f"The estimate is {correct}.",
        ],
        **expr_fields(sympy.Rational(correct.numerator, correct.denominator)),
    )


def _hard_normal_variance(rng):
    for _ in attempts(500, "sums with a readable variance"):
        n = rng.choice((8, 10, 12, 16, 20))
        mean = Fraction(rng.randint(2, 9))
        total = mean * n
        var = Fraction(rng.randint(1, 9), rng.choice((1, 2, 3)))
        sq = n * (var + mean**2)
        if total.denominator == 1 and sq.denominator == 1:
            break
    est = sq / n - (total / n) ** 2
    check(est == var, "the maximum likelihood variance disagrees")
    mu, v = sympy.symbols("mu v", positive=True)
    loglik = -sympy.Rational(n, 2) * sympy.log(v) - (rat(sq) - 2 * mu * rat(total) + n * mu**2) / (2 * v)
    sol = sympy.solve([sympy.diff(loglik, mu), sympy.diff(loglik, v)], [mu, v], dict=True)[0]
    check(fr(sol[v]) == var, "solving both score equations disagrees")
    cands = [
        (sq / n, "r3k-mle-variance-ignored-the-mean"),
        (est * Fraction(n, n - 1), "r3k-mle-unbiased-for-mle"),
        (sq / n + (total / n) ** 2, "r3k-mle-variance-wrong-sign"),
        (sq / (n - 1) - (total / n) ** 2, "r3k-mle-n-minus-one"),
    ]
    dist = need(frac_distractors(est, cands, positive=True))
    text = (
        f"A random sample of size {n} from a normal distribution with unknown mean and unknown variance has Σx = {_ft(total)} "
        f"and Σx² = {_ft(sq)}. Find the maximum likelihood estimate of the variance."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"The maximum likelihood estimate of the mean is x̄ = {_ft(total / n)}.",
            f"σ̂² = Σx²/n - x̄² = {_ft(sq / n)} - {_ft((total / n) ** 2)} = {_ft(est)}.",
        ],
        **expr_fields(sympy.Rational(est.numerator, est.denominator)),
    )


def _hard_invariance(rng):
    n = rng.choice((10, 12, 15, 20))
    for _ in attempts(200, "a total with a readable rate"):
        total = rng.randint(n + 2, 3 * n)
        rate = Fraction(n, total)
        if rate.denominator <= 12:
            break
    tt = rng.randint(1, 4)
    correct = sympy.exp(-tt * rat(rate))
    r = sympy.Symbol("r", positive=True)
    loglik = n * sympy.log(r) - r * total
    solved = sympy.solve(sympy.diff(loglik, r), r)[0]
    check(sympy.simplify(sympy.exp(-tt * solved) - correct) == 0, "the invariance computation disagrees")
    mean = Fraction(total, n)
    cands = [
        (sympy.exp(-tt * rat(mean)), "r3k-mle-mean-for-rate"),
        (sympy.exp(-rat(rate)), "r3k-mle-invariance-forgot-the-time"),
        (1 - correct, "r3k-mle-invariance-complement"),
        (rat(rate) * tt, "r3k-mle-invariance-forgot-the-exponential"),
    ]
    dist = need(expr_distractors(correct, cands))
    text = (
        f"The lifetimes of {n} independent components are modelled by the density f(x; r) = r e^(-rx) for x > 0, and total "
        f"{total} hours. Find the maximum likelihood estimate of the probability that a component lasts longer than {tt} "
        f"{'hour' if tt == 1 else 'hours'}."
    )
    return dict(
        text=text,
        distractors=dist,
        working=[
            f"The maximum likelihood estimate of r is n/T = {rate}.",
            f"By invariance the estimate of P(X > {tt}) = e^(-{tt}r) is e^(-{tt}r̂) = {pretty(correct)}.",
        ],
        **expr_fields(correct),
    )


@generator(
    "r3k_mle",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@clean_working
def r3k_mle(rng, tier="easy"):
    """Maximum likelihood and method of moments estimates; see the module docstring for the tiers."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice((_hard_mom_power, _hard_uniform, _hard_normal_variance, _hard_invariance))(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_x_r3k_mle", level=19, presets={"exam": dict(tier="exam")})
@clean_working
def exd_x_r3k_mle(rng, tier="exam"):
    """Log-likelihood, MLE of the rate, MLE of the mean and MLE of a tail probability for an exponential sample."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = rng.choice((6, 8, 10, 12))
    for _ in attempts(300, "a total with a readable rate"):
        total = rng.randint(n + 2, 3 * n)
        rate = Fraction(n, total)
        if rate.denominator <= 12:
            break
    tt = rng.randint(1, 3)
    r = sympy.Symbol("r", positive=True)
    loglik = n * sympy.log(r) - total * r
    solved = sympy.solve(sympy.diff(loglik, r), r)[0]
    check(fr(solved) == rate, "the score equation disagrees with n/T")
    check(sympy.diff(loglik, r, 2).subs(r, solved) < 0, "the stationary point is not a maximum")
    mean = 1 / rate
    tail = math.exp(-tt * float(rate))
    places = 3
    a_cands = [
        (n * sympy.log(r) + total * r, "r3k-mle-loglik-sign-error"),
        (n * r - total * r, "r3k-mle-loglik-dropped-the-log"),
        (sympy.log(r) - total * r, "r3k-mle-loglik-forgot-n"),
        (n * sympy.log(r) - r, "r3k-mle-loglik-forgot-the-total"),
    ]
    a_dist = need(expr_distractors(n * sympy.log(r) - total * r, a_cands, symbols=(r,)))
    b_dist = need(
        frac_distractors(
            rate,
            [
                (Fraction(total, n), "r3k-mle-mean-for-rate"),
                (Fraction(n - 1, total), "r3k-mle-n-minus-one"),
                (Fraction(1, total), "r3k-mle-forgot-n"),
            ],
            positive=True,
        )
    )
    # a reply right by follow-through from a wrong (b) must never be one of (c)'s or (d)'s own distractors
    wrong_b = [stored(d) for d in b_dist]
    c_dist = need(
        frac_distractors(
            mean,
            [
                (rate, "r3k-mle-mean-for-rate"),
                (Fraction(total, n - 1), "r3k-mle-n-minus-one"),
                (Fraction(total), "r3k-mle-forgot-n"),
                (Fraction(total, n + 1), "r3k-mle-n-plus-one"),
                (Fraction(total * n), "r3k-mle-total-times-n"),
            ],
            positive=True,
            avoid=[1 / v for v in wrong_b],
        )
    )
    d_dist = need(
        num_distractors(
            tail,
            [
                (math.exp(-tt * float(mean)), "r3k-mle-mean-for-rate"),
                (math.exp(-float(rate)), "r3k-mle-invariance-forgot-the-time"),
                (1 - tail, "r3k-mle-invariance-complement"),
                (math.exp(-tt * float(Fraction(n - 1, total))), "r3k-mle-n-minus-one"),
                (math.exp(-(tt + 1) * float(rate)), "r3k-mle-invariance-wrong-time"),
                (math.exp(tt * float(rate)), "r3k-mle-invariance-sign"),
            ],
            places,
            avoid=[math.exp(-tt * float(v)) for v in wrong_b],
        )
    )
    plural, single = rng.choice(
        (("batteries", "battery"), ("light bulbs", "bulb"), ("phone chargers", "charger"), ("pumps", "pump"))
    )
    text = (
        f"The lifetimes x_1, ..., x_{n} of {n} independent {plural} (in hours) are modelled by the density f(x; r) = r e^(-rx) "
        f"for x > 0, where r > 0 is unknown. The lifetimes total {total} hours."
    )
    return dict(
        text=text,
        parts=[
            dict(
                prompt="Write down the log-likelihood ℓ(r), as an expression in r.",
                marks=2,
                working=[f"ℓ(r) = Σ ln(r e^(-r x_i)) = {n} ln r - {total} r."],
                distractors=a_dist,
                **expr_fields(n * sympy.log(r) - total * r),
            ),
            dict(
                prompt="Find the maximum likelihood estimate of r.",
                marks=2,
                sensible_range=(0, None),
                working=[f"ℓ'(r) = {n}/r - {total} = 0 gives r̂ = {rate}."],
                distractors=b_dist,
                **frac_fields(rate),
            ),
            dict(
                prompt="Hence find the maximum likelihood estimate of the mean lifetime 1/r.",
                marks=1,
                sensible_range=(0, None),
                follow_through="1/part_b",
                working=[f"By invariance the estimate of 1/r is 1/r̂ = {mean}."],
                distractors=c_dist,
                **frac_fields(mean),
            ),
            dict(
                prompt=f"Hence find the maximum likelihood estimate of the probability that a {single} lasts longer than {tt} {'hour' if tt == 1 else 'hours'}, correct to {places} decimal places.",
                marks=2,
                sensible_range=(0, 1),
                follow_through=f"exp(-{tt}*part_b)",
                working=[f"P(X > {tt}) = e^(-{tt}r), so the estimate is e^(-{tt} × {rate}) = {tail:.{places}f}."],
                distractors=d_dist,
                **num_fields(tail, places),
            ),
        ],
    )
