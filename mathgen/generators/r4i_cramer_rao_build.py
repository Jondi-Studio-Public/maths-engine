"""Generates `r4i_cramer_rao` questions (easy, medium, hard) and the exam chain `exd_x_r4i_cramer_rao`.

``uc2-cramer-rao-lower-bound`` (``statistics``, answer ``fraction``, with a 5 dp decimal in one medium variant and the
exam): the Cramer-Rao lower bound g'(t)^2 / (n I(t)) for the variance of an unbiased estimator, where I is the Fisher
information of one observation. Stems state a model and numerical parameters; they never give I or name the bound.

* easy: the bound for estimating the parameter itself, from a model: Poisson mean, exponential rate, or a Bernoulli
  probability. The learner differentiates the log-likelihood twice (or squares the score) to find I.
* medium: the bound for a function of the parameter: P(X = 0) = e^(-lambda) for a Poisson count (5 dp), the mean 1/r of
  an exponential, p^2 for a Bernoulli probability, or the standard deviation of a normal population (I for the variance
  is stated through the density).
* hard: one of four: the efficiency (bound over variance) of the unbiased estimator (n - 1)/sum of an exponential rate,
  given its variance r^2/(n - 2); the efficiency of a weighted combination of three normal observations; the bound for a
  density that must be differentiated twice (a gamma density with shape 2 or 3 in the rate, or a Pareto exponent); or the
  bound for the odds p/(1 - p) of a Bernoulli probability.
* exam: a Poisson count with lambda and n given. (a) the Fisher information of one observation, (b) the bound for an
  unbiased estimator of lambda (follows through from (a)), (c) the bound for an unbiased estimator of P(X = 0) = e^(-lambda)
  to 5 dp (follows through from (b)), (d) the efficiency of (X1 + X2)/2 (follows through from (b)).

The tests recompute I(theta) as the negative expected second derivative of the log-density (SymPy, exact summation or
integration) and as the variance of the score, and read the numbers back out of the text.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator
from ._r4i_kit import (
    chain_avoid,
    dec,
    frac_distractors,
    frac_fields,
    need,
    num_distractors,
    num_fields,
    redraw,
    retry,
    stable,
    stored,
)

TOPIC = "r4i_cramer_rao"
EXAM = "exd_x_r4i_cramer_rao"


def _ft(v) -> str:
    v = Fraction(v)
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def _pf(v) -> str:
    """A value for use inside a quotient or product: a fraction is bracketed ("(1/2)"), so "1/2/8" never appears."""
    v = Fraction(v)
    return f"({_ft(v)})" if v.denominator != 1 else _ft(v)


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_poisson(rng):
    n = rng.choice((8, 10, 12, 20, 25, 40))
    lam = rng.choice((Fraction(1, 2), Fraction(3, 2), Fraction(2), Fraction(3), Fraction(5, 2), Fraction(4)))
    ans = lam / n
    cands = [
        (Fraction(n) / lam, "r4i-cr-forgot-to-invert"),
        (lam * lam / n, "r4i-cr-wrong-power"),
        (1 / (n * lam), "r4i-cr-info-per-observation"),
        (lam / (n * n), "r4i-cr-n-squared"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = rng.choice(
        [
            f"The number of faults per 100 m of cable is modelled as Poisson with mean λ, and {n} independent lengths are "
            f"inspected. What is the smallest variance that any unbiased estimator of λ can have when λ = {_ft(lam)}?",
            f"A sample of {n} independent counts follows P(X = k) = e^(-λ) λ^k/k! for k = 0, 1, 2, ... . Find the lower bound "
            f"for the variance of an unbiased estimator of λ, evaluated at λ = {_ft(lam)}.",
        ]
    )
    working = [
        "ln f = -λ + k ln λ - ln k!, so the score is k/λ - 1 and I(λ) = -E(-k/λ²) = 1/λ.",
        f"The bound is 1/(n I) = λ/n = {_pf(lam)} ÷ {n} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _easy_exponential(rng):
    n = rng.choice((5, 8, 10, 16, 20, 25))
    r = rng.choice((Fraction(1, 2), Fraction(2), Fraction(3), Fraction(3, 2), Fraction(4), Fraction(5)))
    ans = r * r / n
    cands = [
        (Fraction(n) / (r * r), "r4i-cr-forgot-to-invert"),
        (r / n, "r4i-cr-wrong-power"),
        (1 / (n * r * r), "r4i-cr-info-per-observation"),
        (r * r / (n * n), "r4i-cr-n-squared"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = rng.choice(
        [
            f"The lifetimes of {n} independent components have density f(x) = r e^(-rx) for x > 0. What is the smallest "
            f"variance that an unbiased estimator of the rate r can have when r = {_ft(r)}?",
            f"A sample of {n} independent waiting times has density f(x; r) = r e^(-rx), x > 0. Find the Cramér–Rao lower "
            f"bound for the variance of an unbiased estimator of r, evaluated at r = {_ft(r)}.",
        ]
    )
    working = [
        "ln f = ln r - r x, so the score is 1/r - x and I(r) = -E(-1/r²) = 1/r².",
        f"The bound is 1/(n I) = r²/n = ({_ft(r)})²/{n} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _easy_bernoulli(rng):
    n = rng.choice((10, 20, 25, 40, 50, 100))
    p = rng.choice((Fraction(1, 4), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5), Fraction(1, 10), Fraction(3, 5)))
    ans = p * (1 - p) / n
    cands = [
        (Fraction(n) / (p * (1 - p)), "r4i-cr-forgot-to-invert"),
        (p / n, "r4i-cr-wrong-power"),
        (1 / (n * p * (1 - p)), "r4i-cr-info-per-observation"),
        (p * (1 - p) / (n * n), "r4i-cr-n-squared"),
        (p * p / n, "r4i-cr-wrong-power"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = rng.choice(
        [
            f"Each of {n} independent trials succeeds with the same unknown probability p, and in truth p = {_ft(p)}. What "
            f"is the smallest variance that an unbiased estimator of p can have?",
            f"A quality inspector checks {n} independent items, each defective with probability p; here p = {_ft(p)}. Find "
            f"the lower bound for the variance of an unbiased estimator of p.",
        ]
    )
    working = [
        "ln f = x ln p + (1 - x) ln(1 - p); the score is x/p - (1 - x)/(1 - p) and I(p) = 1/(p(1 - p)).",
        f"The bound is p(1 - p)/n = {_pf(p)} × {_pf(1 - p)} ÷ {n} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _easy(rng):
    return rng.choice((_easy_poisson, _easy_exponential, _easy_bernoulli))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium_poisson_square(rng):
    n = rng.choice((8, 10, 12, 20, 25, 40))
    lam = rng.choice((Fraction(1, 2), Fraction(3, 2), Fraction(2), Fraction(3), Fraction(5, 2)))
    ans = 4 * lam**3 / n
    cands = [
        (lam / n, "r4i-cr-bound-for-parameter"),
        (2 * lam**3 / n, "r4i-cr-derivative-not-squared"),
        (lam**3 / n, "r4i-cr-derivative-ignored-factor"),
        (4 * lam**3, "r4i-cr-forgot-n"),
        (4 * lam**2 / n, "r4i-cr-forgot-info"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"The number of faults in a length of cable is Poisson with mean λ, where λ = {_ft(lam)} in truth. From {n} "
        f"independent lengths we wish to estimate λ², the square of the mean. What is the smallest variance that an "
        f"unbiased estimator of λ² can have?"
    )
    working = [
        "I(λ) = 1/λ, and g(λ) = λ² has g'(λ) = 2λ.",
        f"The bound is g'(λ)²/(nI) = 4λ²×λ/n = 4λ³/n = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium_exp_mean(rng):
    n = rng.choice((5, 8, 10, 16, 20))
    r = rng.choice((Fraction(1, 2), Fraction(2), Fraction(3), Fraction(4), Fraction(1, 3)))
    ans = 1 / (n * r * r)
    cands = [
        (r * r / n, "r4i-cr-bound-for-parameter"),
        (1 / (n * r**4), "r4i-cr-derivative-squared-twice"),
        (1 / (n * r), "r4i-cr-derivative-not-squared"),
        (1 / (r * r), "r4i-cr-forgot-n"),
        (Fraction(1, 1) / (n * r**3), "r4i-cr-derivative-wrong-power"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"The lifetimes of {n} independent components have density f(x) = r e^(-rx), x > 0, with r = {_ft(r)} in truth. "
        f"The mean lifetime is 1/r. What is the smallest variance that an unbiased estimator of the mean lifetime can have?"
    )
    working = [
        "I(r) = 1/r², and g(r) = 1/r has g'(r) = -1/r².",
        f"The bound is g'(r)²/(nI) = (1/r⁴)×r²/n = 1/(n r²) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium_bernoulli_square(rng):
    n = rng.choice((10, 20, 25, 40, 50))
    p = rng.choice((Fraction(1, 4), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5), Fraction(1, 2), Fraction(3, 5)))
    ans = 4 * p**3 * (1 - p) / n
    cands = [
        (p * (1 - p) / n, "r4i-cr-bound-for-parameter"),
        (p**3 * (1 - p) / n, "r4i-cr-derivative-ignored-factor"),
        (2 * p * p * (1 - p) / n, "r4i-cr-derivative-not-squared"),
        (4 * p * p * p * (1 - p), "r4i-cr-forgot-n"),
        (2 * p * p * p * (1 - p) / n, "r4i-cr-derivative-not-squared"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"Each of {n} independent trials succeeds with the same unknown probability p, and in truth p = {_ft(p)}. We want to "
        f"estimate p², the probability that two particular trials both succeed. What is the smallest variance that an "
        f"unbiased estimator of p² can have?"
    )
    working = [
        "I(p) = 1/(p(1 - p)), and g(p) = p² has g'(p) = 2p.",
        f"The bound is (2p)²×p(1 - p)/n = 4p³(1 - p)/n = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium_normal_sd(rng):
    n = rng.choice((8, 10, 16, 20, 25, 50))
    sd = rng.choice((2, 3, 4, 5, 6, 8))
    ans = Fraction(sd * sd, 2 * n)
    cands = [
        (Fraction(2 * sd**4, n), "r4i-cr-bound-for-parameter"),
        (Fraction(sd * sd, n), "r4i-cr-halved-info"),
        (Fraction(sd**4, 2 * n), "r4i-cr-derivative-wrong-power"),
        (Fraction(sd * sd, 4 * n), "r4i-cr-derivative-squared-twice"),
        (Fraction(sd * sd, 2), "r4i-cr-forgot-n"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"A sample of {n} independent measurements is normal with known mean 0 and unknown standard deviation σ; in truth "
        f"σ = {sd}. Writing v = σ² for the variance, the density is f(x; v) = (2πv)^(-1/2) e^(-x²/(2v)). What is the "
        f"smallest variance that an unbiased estimator of σ can have?"
    )
    working = [
        "ln f = -½ ln v - x²/(2v) + const, so I(v) = 1/(2v²).",
        f"σ = v^(1/2) has derivative 1/(2σ) with respect to v, so the bound is (1/(4σ²))×2σ⁴/n = σ²/(2n) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium(rng):
    return rng.choice((_medium_poisson_square, _medium_exp_mean, _medium_bernoulli_square, _medium_normal_sd))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard_efficiency_exp(rng):
    n = rng.choice((4, 5, 6, 8, 10, 12, 20))
    r = rng.choice((Fraction(1, 2), Fraction(2), Fraction(3), Fraction(5)))
    ans = Fraction(n - 2, n)
    cands = [
        (Fraction(n, n - 2), "r4i-cr-efficiency-inverted"),
        (Fraction(n - 1, n), "r4i-cr-variance-shifted"),
        (Fraction(n - 2, n - 1), "r4i-cr-variance-shifted"),
        (Fraction(n), "r4i-cr-forgot-bound"),
        ((n - 2) * r * r / n, "r4i-cr-bound-times-variance"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"A sample of {n} independent lifetimes has density r e^(-rx), x > 0. The estimator T = {n - 1}/(X1 + ... + X{n}) is "
        f"unbiased for r and has variance r²/{n - 2}. Find the efficiency of T at r = {_ft(r)}, that is the Cramér–Rao lower "
        f"bound divided by the variance of T."
    )
    working = [
        f"I(r) = 1/r², so the bound is r²/{n}.",
        f"Efficiency = (r²/{n})/(r²/{n - 2}) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard_efficiency_weights(rng):
    sets = [(1, 2, 3), (1, 1, 2), (1, 3, 4), (2, 3, 5), (1, 2, 5), (1, 1, 3), (2, 2, 3)]
    num = rng.choice(sets)
    tot = sum(num)
    w = [Fraction(v, tot) for v in num]
    sq = sum(v * v for v in w)
    ans = Fraction(1, 3) / sq
    sd2 = rng.choice((4, 9, 16, 25))
    cands = [
        (sq * 3, "r4i-cr-efficiency-inverted"),
        (Fraction(1, 3) * sq, "r4i-cr-bound-times-variance"),
        (1 / sq, "r4i-cr-forgot-n"),
        (Fraction(1, 3) / (sum(w) ** 2), "r4i-cr-square-of-sum-weights"),
        (Fraction(1, 3) / sum(v for v in w), "r4i-cr-weights-not-squared"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True, at_most=Fraction(1)))
    wt = " + ".join(f"({_ft(Fraction(v, tot))})X{i + 1}" for i, v in enumerate(num))
    text = (
        f"Three independent measurements X1, X2, X3 are normal with unknown mean μ and known variance {sd2}. The estimator "
        f"T = {wt} is unbiased for μ. Find its efficiency, the Cramér–Rao lower bound for an unbiased estimator of μ based "
        f"on the three measurements divided by the variance of T."
    )
    working = [
        f"The bound is {sd2}/3, since I(μ) = 1/{sd2} per measurement.",
        f"Var(T) = {sd2}×{_ft(sq)}, so the efficiency is {_pf(Fraction(1, 3))} ÷ {_pf(sq)} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard_density(rng):
    n = rng.choice((5, 8, 10, 12, 20, 25))
    kind = rng.choice(("gamma2", "gamma3", "pareto"))
    th = rng.choice((Fraction(1, 2), Fraction(2), Fraction(3), Fraction(4), Fraction(3, 2)))
    if kind == "pareto":
        info = 1 / (th * th)
        dens = "f(x; a) = a x^(-(a + 1)) for x > 1"
        name = "a"
        score = "1/a - ln x"
        shape = 1
    else:
        shape = 2 if kind == "gamma2" else 3
        info = Fraction(shape) / (th * th)
        dens = f"f(x; θ) = θ² x e^(-θx) for x > 0" if shape == 2 else f"f(x; θ) = (θ³/2) x² e^(-θx) for x > 0"
        name = "θ"
    ans = 1 / (n * info)
    cands = [
        (Fraction(th * th) / n, "r4i-cr-shape-ignored"),
        (Fraction(n) * info, "r4i-cr-forgot-to-invert"),
        (1 / info, "r4i-cr-forgot-n"),
        (th * th / (n * (shape + 1)) if shape > 1 else th**4 / n, "r4i-cr-info-power-wrong"),
        (th / (n * shape), "r4i-cr-wrong-power"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"A sample of {n} independent observations has density {dens}. Find the smallest variance that an unbiased estimator "
        f"of {name} can have when {name} = {_ft(th)}."
    )
    working = [
        f"The second derivative of ln f with respect to {name} is -{shape}/{name}² (not depending on x), so I = {shape}/{name}².",
        f"The bound is 1/(nI) = {name}²/({shape}×{n}) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard_odds(rng):
    n = rng.choice((10, 20, 25, 40, 50))
    p = rng.choice((Fraction(1, 4), Fraction(1, 5), Fraction(1, 3), Fraction(2, 5), Fraction(1, 2), Fraction(3, 5)))
    ans = p / (n * (1 - p) ** 3)
    cands = [
        (p * (1 - p) / n, "r4i-cr-bound-for-parameter"),
        (p / (n * (1 - p)), "r4i-cr-derivative-ignored-factor"),
        (p / (n * (1 - p) ** 2), "r4i-cr-derivative-wrong-power"),
        (p * (1 - p) / (n * (1 - p) ** 2), "r4i-cr-derivative-not-squared"),
        (p / ((1 - p) ** 3), "r4i-cr-forgot-n"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"Each of {n} independent trials succeeds with the same unknown probability p, and in truth p = {_ft(p)}. The odds of "
        f"success are p/(1 - p). What is the smallest variance that an unbiased estimator of the odds can have?"
    )
    working = [
        "I(p) = 1/(p(1 - p)) and g(p) = p/(1 - p) has g'(p) = 1/(1 - p)².",
        f"The bound is g'(p)²/(nI) = p(1 - p)/(n(1 - p)⁴) = p/(n(1 - p)³) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard(rng):
    return rng.choice((_hard_efficiency_exp, _hard_efficiency_weights, _hard_density, _hard_odds))(rng)


@generator(
    TOPIC,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraw
def r4i_cramer_rao(rng, tier="easy"):
    """Cramer-Rao lower bounds and efficiency; see the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------


@generator(EXAM, level=21, presets={"exam": dict(tier="exam")})
@redraw
def exd_x_r4i_cramer_rao(rng, tier="exam"):
    """A Poisson count: Fisher information, the bound for lambda, the bound for P(X = 0), the efficiency of (X1 + X2)/2."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = rng.choice((4, 5, 8, 10))
    lam = rng.choice((Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)))
    lf = float(lam)
    info = 1 / lam
    b = lam / n
    places = 5
    c_exact = lf * math.exp(-2 * lf) / n
    retry(stable(c_exact, places, 0.12), "part (c) near a rounding boundary")
    # the chain carries the stated (b), which is exact, so (c) from b equals the unrounded route
    c = float(dec(c_exact, places))
    eff = b / (lam / 2)
    order = ["a", "b", "c", "d"]
    canon = {"a": info, "b": b, "c": c, "d": eff}
    fns = {
        "b": lambda s: 1 / (n * s["a"]) if s["a"] != 0 else None,
        "c": lambda s: math.exp(-2 * lf) * float(s["b"]),
        "d": lambda s: s["b"] / (lam / 2),
    }
    rounders = {"c": lambda v: float(dec(v, places))}
    a_dist = need(
        frac_distractors(
            info,
            [
                (lam, "r4i-cr-variance-for-information"),
                (1 / (lam * lam), "r4i-cr-wrong-power"),
                (n / lam, "r4i-cr-sample-info-for-one"),
                (lam * n, "r4i-cr-forgot-to-invert"),
            ],
            positive=True,
        )
    )
    wrong = {"a": [stored(v) for v in a_dist]}
    avoid_b = chain_avoid(order, fns, canon, wrong, "b", rounders)
    b_dist = need(
        frac_distractors(
            b,
            [
                (Fraction(n) / lam, "r4i-cr-forgot-to-invert"),
                (lam * lam / n, "r4i-cr-wrong-power"),
                (lam / (n * n), "r4i-cr-n-squared"),
                (Fraction(1, 1) / (n * lam * lam), "r4i-cr-info-squared"),
            ],
            positive=True,
            avoid=[v for v in avoid_b if v is not None],
        )
    )
    wrong["b"] = [stored(v) for v in b_dist]
    avoid_c = [v for v in chain_avoid(order, fns, canon, wrong, "c", rounders) if v is not None]
    c_dist = need(
        num_distractors(
            c,
            [
                (lf * math.exp(-lf) / n, "r4i-cr-derivative-not-squared"),
                (lf / n, "r4i-cr-derivative-ignored"),
                (math.exp(-2 * lf) / n, "r4i-cr-forgot-info"),
                (math.exp(-lf) * float(b) * math.exp(-lf) ** 3, "r4i-cr-derivative-wrong-power"),
            ],
            places,
            avoid=avoid_c,
        )
    )
    wrong["c"] = [float(v.value) for v in c_dist]
    avoid_d = [v for v in chain_avoid(order, fns, canon, wrong, "d", rounders) if v is not None]
    d_dist = need(
        frac_distractors(
            eff,
            [
                (1 / eff, "r4i-cr-efficiency-inverted"),
                (Fraction(1, n), "r4i-cr-forgot-two-observations"),
                (Fraction(2 * n), "r4i-cr-efficiency-inverted"),
                (b * lam / 2, "r4i-cr-bound-times-variance"),
            ],
            positive=True,
            avoid=avoid_d,
        )
    )
    text = rng.choice(
        [
            f"The number of faults X in a length of cable follows P(X = k) = e^(-λ) λ^k/k! for k = 0, 1, 2, ..., and {n} "
            f"independent lengths are inspected. In truth λ = {_ft(lam)}. Throughout, estimators are unbiased.",
            f"A sample of n = {n} independent counts is drawn from a Poisson distribution with mean λ, where in truth "
            f"λ = {_ft(lam)}. All estimators below are unbiased for the quantity they estimate.",
            f"Telephone calls arrive at a Hobart exchange at a Poisson mean of λ per minute, and the counts from {n} "
            f"independent minutes are recorded. In truth λ = {_ft(lam)}. Only unbiased estimators are considered.",
        ]
    )
    parts = [
        dict(
            prompt="Find the Fisher information I(λ) of one observation, at the true value of λ.",
            marks=2,
            sensible_range=(0, None),
            working=["ln f = -λ + k ln λ - ln k!, so I(λ) = -E(-k/λ²) = 1/λ."],
            distractors=a_dist,
            **frac_fields(info),
        ),
        dict(
            prompt="Find the Cramér–Rao lower bound for the variance of an unbiased estimator of λ.",
            marks=1,
            sensible_range=(0, None),
            follow_through=f"1/({n}*part_a)",
            working=[f"The bound is 1/(nI) = {_pf(lam)} ÷ {n} = {_ft(b)}."],
            distractors=b_dist,
            **frac_fields(b),
        ),
        dict(
            prompt=(
                "Find the lower bound for the variance of an unbiased estimator of e^(-λ), the probability that a count is "
                f"zero, correct to {places} decimal places."
            ),
            marks=2,
            sensible_range=(0, None),
            follow_through=f"exp(-2*{_ft(lam)})*part_b",
            working=[f"g(λ) = e^(-λ) has g'(λ)² = e^(-2λ), so the bound is e^(-2λ) × {_pf(b)} = {dec(c, places)}."],
            distractors=c_dist,
            **num_fields(c, places),
        ),
        dict(
            prompt=(
                "The estimator (X1 + X2)/2 uses only the first two observations and is unbiased for λ. Find its efficiency, "
                "the bound from part (b) divided by its variance."
            ),
            marks=2,
            sensible_range=(0, None),
            follow_through=f"part_b/({_ft(lam)}/2)",
            working=[
                f"Var((X1 + X2)/2) = λ/2 = {_ft(lam / 2)}.",
                f"Efficiency = {_pf(b)} ÷ {_pf(lam / 2)} = {_ft(eff)}.",
            ],
            distractors=d_dist,
            **frac_fields(eff),
        ),
    ]
    return dict(text=text, parts=parts)
