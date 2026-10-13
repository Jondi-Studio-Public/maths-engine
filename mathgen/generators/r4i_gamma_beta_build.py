"""Generates `r4i_gamma_beta` questions (easy, medium, hard) and the exam chain `exd_x_r4i_gamma_beta`.

``uc2-gamma-beta-moments`` (``probability``, answer ``fraction``, with a decimal slice in the hard tier): moments of the
gamma and beta families, recognised from the density itself. Stems give a density with its numerical constant written
out (for instance ``f(x) = 12 x (1 - x)^2`` on (0, 1)) and never name the family or its parameters.

* easy: the mean of a beta density, the mean of a gamma density (shape over rate) or its variance (shape over rate squared).
* medium: the variance of a beta density, E(X^2) of a gamma density, E(1/X) of a gamma density (rate over shape minus one),
  or E(X(1 - X)) of a beta density.
* hard: one of five: the constant that normalises a beta kernel x^p (1 - x)^q or a gamma kernel x^p e^(-rx); the upper tail
  P(X > t) of a gamma density with whole shape (a Poisson sum, 4 dp); an exact beta tail probability; or the parameter
  recovered from a stated mean and variance by matching moments (beta alpha, or gamma rate).
* exam: a beta density f(x) = c x^(a-1) (1 - x)^(b-1): (a) the constant c, (b) the mean, (c) the variance (follows through
  from (b), since Var = m (1 - m) / (a + b + 1)), (d) the standard deviation to 3 dp (follows through from (c)).

Every moment is a closed form in exact fractions; the tests integrate the stated density numerically with mpmath and
read the parameters back out of the text.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4i_kit import (
    chain_avoid,
    dec,
    frac_distractors,
    frac_fields,
    int_distractors,
    need,
    num_distractors,
    num_fields,
    redraw,
    retry,
    stable,
    stored,
)

TOPIC = "r4i_gamma_beta"
EXAM = "exd_x_r4i_gamma_beta"
fact = math.factorial


def _ft(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def _coef(c: Fraction) -> str:
    """A constant written in front of a density: `12`, or `(27/2)`."""
    return str(c.numerator) if c.denominator == 1 else f"({c.numerator}/{c.denominator})"


def _pow(base: str, e: int) -> str:
    return base if e == 1 else (f"{base}^{e}" if e else "")


def beta_const(a: int, b: int) -> int:
    return fact(a + b - 1) // (fact(a - 1) * fact(b - 1))


def beta_density(a: int, b: int, c=None) -> str:
    const = c if c is not None else _coef(Fraction(beta_const(a, b)))
    parts = [const, _pow("x", a - 1), _pow("(1 - x)", b - 1)]
    return "f(x) = " + " ".join(p for p in parts if p) + " for 0 < x < 1, and f(x) = 0 otherwise"


def gamma_const(k: int, r: Fraction) -> Fraction:
    return r**k / fact(k - 1)


def _rate_text(r: Fraction) -> str:
    return _ft(r)


def gamma_density(k: int, r: Fraction, c=None) -> str:
    const = c if c is not None else _coef(gamma_const(k, r))
    ex = f"e^(-{_rate_text(r)}x)" if r.denominator == 1 else f"e^(-({_rate_text(r)})x)"
    parts = [const, _pow("x", k - 1), ex]
    return "f(x) = " + " ".join(p for p in parts if p) + " for x > 0, and f(x) = 0 otherwise"


def _ab(rng, lo=2, hi=5):
    for _ in attempts(100, "distinct beta parameters"):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if a != b:
            return a, b
    raise AssertionError("unreachable")


def _kr(rng, kmin=2):
    k = rng.randint(kmin, 5)
    r = rng.choice((Fraction(1), Fraction(2), Fraction(3), Fraction(4), Fraction(1, 2)))
    return k, r


CONTEXTS_BETA = [
    "The proportion of a day's solar output that a Mildura farm uses on site has density",
    "The fraction of its capacity that a Hobart reservoir is filled to on a random day has density",
    "The proportion of a batch of seedlings that survive at a Cairns nursery has density",
    "The share of a month's calls to a Perth helpline that arrive on weekdays has density",
]
CONTEXTS_GAMMA = [
    "The time, in hours, a technician spends on a repair has density",
    "The total rainfall, in cm, in a week at a Darwin station has density",
    "The lifetime, in years, of a pump at a Wagga Wagga bore has density",
    "The time, in minutes, to complete a customer's order at a Geelong cafe has density",
]


def _intro(rng, kind):
    return rng.choice(CONTEXTS_BETA if kind == "beta" else CONTEXTS_GAMMA)


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_beta_mean(rng):
    a, b = _ab(rng)
    ans = Fraction(a, a + b)
    cands = [
        (Fraction(b, a + b), "r4i-gb-swapped-parameters"),
        (Fraction(a, a + b + 1), "r4i-gb-denominator-plus-one"),
        (Fraction(a - 1, a + b - 2), "r4i-gb-mode-for-mean"),
        (Fraction(a - 1, a + b), "r4i-gb-exponent-for-parameter"),
        (Fraction(a, b), "r4i-gb-ratio-not-share"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True, at_most=Fraction(1)))
    text = f"{_intro(rng, 'beta')} {beta_density(a, b)}. Find the mean of this distribution."
    working = [
        f"The density is proportional to x^{a - 1}(1 - x)^{b - 1}, a beta density with parameters {a} and {b}.",
        f"Its mean is {a}/({a}+{b}) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _easy_gamma_mean(rng):
    k, r = _kr(rng)
    ans = k / r
    cands = [
        (k * r, "r4i-gb-rate-times-shape"),
        ((k - 1) / r, "r4i-gb-exponent-for-parameter"),
        (Fraction(k) / (r * r), "r4i-gb-variance-for-mean"),
        (Fraction(1) / r, "r4i-gb-forgot-shape"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = f"{_intro(rng, 'gamma')} {gamma_density(k, r)}. Find the mean of this distribution."
    working = [
        f"The density is proportional to x^{k - 1}e^(-{_rate_text(r)}x), a gamma density with shape {k} and rate {_ft(r)}.",
        f"Its mean is shape over rate = {k}/{_ft(r)} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _easy_gamma_var(rng):
    k, r = _kr(rng)
    ans = k / (r * r)
    cands = [
        (k / r, "r4i-gb-mean-for-variance"),
        (Fraction(k) / r**3, "r4i-gb-wrong-power-of-rate"),
        (Fraction(k - 1) / (r * r), "r4i-gb-exponent-for-parameter"),
        (k * r * r, "r4i-gb-rate-times-shape"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = f"{_intro(rng, 'gamma')} {gamma_density(k, r)}. Find the variance of this distribution."
    working = [
        f"This is a gamma density with shape {k} and rate {_ft(r)}.",
        f"The variance is shape over rate squared = {k}/({_ft(r)})² = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _easy(rng):
    return rng.choice((_easy_beta_mean, _easy_gamma_mean, _easy_gamma_var))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium_beta_var(rng):
    a, b = _ab(rng)
    s = a + b
    ans = Fraction(a * b, s * s * (s + 1))
    cands = [
        (Fraction(a * b, s * (s + 1)), "r4i-gb-missing-square"),
        (Fraction(a * b, s * s * s), "r4i-gb-forgot-plus-one"),
        (Fraction(a, s) * (1 - Fraction(a, s)), "r4i-gb-forgot-division"),
        (Fraction(a * b, s * s * (s - 1)), "r4i-gb-plus-one-sign"),
        (Fraction(a - 1) * (b - 1) / (s * s * (s + 1)), "r4i-gb-exponent-for-parameter"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = f"{_intro(rng, 'beta')} {beta_density(a, b)}. Find the variance of this distribution."
    working = [
        f"This is a beta density with parameters {a} and {b}, so the variance is ab/((a+b)²(a+b+1)).",
        f"Var = {a * b}/({s}²×{s + 1}) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium_gamma_second(rng):
    k, r = _kr(rng)
    ans = Fraction(k * (k + 1)) / (r * r)
    cands = [
        (Fraction(k * k) / (r * r), "r4i-gb-square-of-mean"),
        (k / (r * r), "r4i-gb-variance-for-second-moment"),
        (Fraction(k * (k + 1)) / r, "r4i-gb-forgot-square-of-rate"),
        (Fraction(k * (k - 1)) / (r * r), "r4i-gb-plus-one-sign"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = f"{_intro(rng, 'gamma')} {gamma_density(k, r)}. Find E(X²), the mean of the square of the value."
    working = [
        f"The density has shape {k} and rate {_ft(r)}: E(X) = {_ft(k / r)} and Var(X) = {_ft(k / (r * r))}.",
        f"E(X²) = Var(X) + E(X)² = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium_gamma_inverse(rng):
    k = rng.randint(2, 6)
    r = rng.choice((Fraction(1), Fraction(2), Fraction(3), Fraction(4), Fraction(1, 2)))
    ans = r / (k - 1)
    cands = [
        (r / k, "r4i-gb-shape-not-reduced"),
        (k / r, "r4i-gb-reciprocal-of-mean-wrong-way"),
        (Fraction(1) / (r * (k - 1)), "r4i-gb-rate-inverted"),
        (r * (k - 1), "r4i-gb-inverted-ratio"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = f"{_intro(rng, 'gamma')} {gamma_density(k, r)}. Find E(1/X), the mean of the reciprocal of the value."
    working = [
        "E(1/X) = ∫ c x^(k-2) e^(-rx) dx, which is c times the integral for shape k - 1.",
        f"E(1/X) = rate/(shape - 1) = {_ft(r)}/{k - 1} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium_beta_product(rng):
    a, b = _ab(rng)
    s = a + b
    ans = Fraction(a * b, s * (s + 1))
    cands = [
        (Fraction(a, s) * Fraction(b, s), "r4i-gb-product-of-means"),
        (Fraction(a * b, s * s * (s + 1)), "r4i-gb-variance-for-product"),
        (Fraction(a * b, s * (s - 1)), "r4i-gb-plus-one-sign"),
        (Fraction(a * b, s), "r4i-gb-forgot-division"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = f"{_intro(rng, 'beta')} {beta_density(a, b)}. Find E(X(1 - X)), the mean of x times (1 - x)."
    working = [
        f"E(X(1 - X)) = E(X) - E(X²), and X(1 - X) times the density is a multiple of a beta density with parameters {a + 1} and {b + 1}.",
        f"The result is {a}×{b}/({s}×{s + 1}) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium(rng):
    return rng.choice((_medium_beta_var, _medium_gamma_second, _medium_gamma_inverse, _medium_beta_product))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def beta_tail(a: int, b: int, t: Fraction) -> Fraction:
    """P(X > t) for the beta density with whole parameters a, b: the integral of the expanded polynomial."""
    c = beta_const(a, b)
    total = Fraction(0)
    for j in range(b):
        coef = c * math.comb(b - 1, j) * (-1) ** j
        total += Fraction(coef, a + j) * (1 - t ** (a + j))
    return total


def gamma_tail(k: int, lam_t: float) -> float:
    return math.exp(-lam_t) * sum(lam_t**j / math.factorial(j) for j in range(k))


def _hard_normalise_beta(rng):
    p, q = rng.randint(1, 4), rng.randint(1, 4)
    ans = Fraction(fact(p + q + 1), fact(p) * fact(q))
    cands = [
        (Fraction(fact(p) * fact(q), fact(p + q + 1)), "r4i-gb-constant-inverted"),
        (Fraction(fact(p + q), fact(p) * fact(q)), "r4i-gb-factorial-off-by-one"),
        (Fraction(fact(p + q + 1), fact(p + 1) * fact(q + 1)), "r4i-gb-exponent-for-parameter"),
        (Fraction(fact(p + q + 1), fact(p) * fact(q) * (p + q + 1)), "r4i-gb-forgot-top-factorial"),
        (Fraction(p + q + 1, p * q), "r4i-gb-factorials-dropped"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"{_intro(rng, 'beta').rsplit(' has density', 1)[0]} has a density of the form f(x) = c {_pow('x', p)} "
        f"{_pow('(1 - x)', q)} for 0 < x < 1, and f(x) = 0 otherwise. Find the constant c."
    )
    working = [
        f"∫ x^{p}(1 - x)^{q} dx over (0, 1) = {p}!×{q}!/({p + q + 1})!.",
        f"c is the reciprocal: c = {p + q + 1}!/({p}!×{q}!) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard_normalise_gamma(rng):
    p = rng.randint(1, 4)
    r = rng.choice((Fraction(1), Fraction(2), Fraction(3), Fraction(4), Fraction(1, 2)))
    k = p + 1
    ans = gamma_const(k, r)
    cands = [
        (Fraction(1) / ans, "r4i-gb-constant-inverted"),
        (r**p / fact(p), "r4i-gb-exponent-for-parameter"),
        (r**k / fact(k), "r4i-gb-factorial-off-by-one"),
        (r**k, "r4i-gb-factorial-dropped"),
        (r / fact(p), "r4i-gb-power-of-rate-dropped"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    ex = f"e^(-{_rate_text(r)}x)" if r.denominator == 1 else f"e^(-({_rate_text(r)})x)"
    text = (
        f"{_intro(rng, 'gamma').rsplit(' has density', 1)[0]} has a density of the form f(x) = c {_pow('x', p)} {ex} "
        f"for x > 0, and f(x) = 0 otherwise. Find the constant c."
    )
    working = [
        f"∫ x^{p} e^(-{_rate_text(r)}x) dx over (0, ∞) = {p}!/({_rate_text(r)})^{p + 1}.",
        f"c is the reciprocal: c = ({_rate_text(r)})^{p + 1}/{p}! = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard_gamma_third(rng):
    k, r = _kr(rng)
    ans = Fraction(k * (k + 1) * (k + 2)) / r**3
    cands = [
        (Fraction(k**3) / r**3, "r4i-gb-power-of-mean"),
        (Fraction(k * (k + 1)) / r**2, "r4i-gb-second-moment-for-third"),
        (Fraction(k * (k + 1) * (k + 2)) / r**2, "r4i-gb-wrong-power-of-rate"),
        (Fraction(k * (k - 1) * (k - 2)) / r**3, "r4i-gb-plus-one-sign"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = f"{_intro(rng, 'gamma')} {gamma_density(k, r)}. Find E(X³), the mean of the cube of the value."
    working = [
        f"E(X³) = ∫ c x^{k + 2} e^(-{_rate_text(r)}x) dx, a multiple of the gamma integral with shape {k + 3}.",
        f"E(X³) = k(k + 1)(k + 2)/rate³ = {k}×{k + 1}×{k + 2}/({_rate_text(r)})³ = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard_beta_tail(rng):
    a, b = _ab(rng, 2, 4)
    t = rng.choice((Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(1, 4), Fraction(3, 4)))
    ans = beta_tail(a, b, t)
    check(0 < ans < 1, "tail outside (0, 1)")
    cands = [
        (1 - ans, "r4i-gb-tail-reversed"),
        (beta_tail(b, a, t), "r4i-gb-swapped-parameters"),
        (beta_tail(a, b, 1 - t), "r4i-gb-wrong-limit"),
        (Fraction(beta_const(a, b)) * (1 - t) ** b / b, "r4i-gb-single-term-only"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True, at_most=Fraction(1)))
    text = f"{_intro(rng, 'beta')} {beta_density(a, b)}. Find P(X > {_ft(t)}) exactly, as a fraction."
    working = [
        f"Expand (1 - x)^{b - 1} and integrate the polynomial {beta_const(a, b)} x^{a - 1}(1 - x)^{b - 1} from {_ft(t)} to 1.",
        f"P(X > {_ft(t)}) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard_match_beta(rng):
    a, b = _ab(rng, 1, 6)
    s = a + b
    m = Fraction(a, s)
    v = Fraction(a * b, s * s * (s + 1))
    cands = [
        (Fraction(b), "r4i-gb-swapped-parameters"),
        (m * (m * (1 - m) / v), "r4i-gb-forgot-minus-one"),
        (m * (m * (1 - m) / v - 2), "r4i-gb-minus-two"),
        ((1 - m) * (m * (1 - m) / v - 1), "r4i-gb-swapped-parameters"),
        (m * (1 - m) / v - 1, "r4i-gb-total-for-parameter"),
    ]
    dist = need(frac_distractors(Fraction(a), cands, positive=True))
    text = (
        f"A proportion X has a beta distribution with mean {_ft(m)} and variance {_ft(v)}. Find its first parameter α, "
        f"the one for which the mean is α/(α + β)."
    )
    working = [
        f"α + β + 1 = m(1 - m)/Var = {_ft(m * (1 - m) / v)}, so α + β = {s}.",
        f"α = m(α + β) = {_ft(m)}×{s} = {a}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(Fraction(a)))


def _hard_match_gamma(rng):
    k, r = _kr(rng)
    mean, var = k / r, k / (r * r)
    cands = [
        (mean * mean / var, "r4i-gb-shape-for-rate"),
        (var / mean, "r4i-gb-rate-inverted"),
        (Fraction(mean * mean) / var / 2, "r4i-gb-halved"),
        (1 / mean, "r4i-gb-forgot-shape"),
        (mean * var, "r4i-gb-multiplied"),
    ]
    dist = need(frac_distractors(r, cands, positive=True))
    text = (
        f"A waiting time X, in hours, has a gamma distribution with mean {_ft(mean)} and variance {_ft(var)}. "
        f"Find its rate parameter, the one for which the density contains e^(-rate × x)."
    )
    working = [
        f"Mean = k/r and variance = k/r², so r = mean/variance = {_ft(mean)}/{_ft(var)} = {_ft(r)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(r))


def _hard(rng):
    return rng.choice(
        (
            _hard_normalise_beta,
            _hard_normalise_gamma,
            _hard_gamma_third,
            _hard_beta_tail,
            _hard_match_beta,
            _hard_match_gamma,
        )
    )(rng)


@generator(
    TOPIC,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraw
def r4i_gamma_beta(rng, tier="easy"):
    """Moments, constants and tails of gamma and beta densities; see the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------


@generator(EXAM, level=21, presets={"exam": dict(tier="exam")})
@redraw
def exd_x_r4i_gamma_beta(rng, tier="exam"):
    """A beta density with unknown constant: c, the mean, the variance, then the standard deviation."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a, b = _ab(rng, 2, 5)
    s = a + b
    c = Fraction(beta_const(a, b))
    m = Fraction(a, s)
    v = m * (1 - m) / (s + 1)
    sd = math.sqrt(float(v))
    places = 3
    retry(stable(sd, places, 0.15), "sd near a rounding boundary")
    sd_val = float(dec(sd, places))
    order = ["a", "b", "c", "d"]
    canon = {"a": c, "b": m, "c": v, "d": sd_val}
    fns = {
        "c": lambda x: x["b"] * (1 - x["b"]) / (s + 1),
        "d": lambda x: math.sqrt(float(x["c"])) if x["c"] > 0 else None,
    }
    rounders = {"d": lambda x: float(dec(x, places)) if x is not None else None}
    a_dist = need(
        frac_distractors(
            c,
            [
                (Fraction(fact(a - 1) * fact(b - 1), fact(s - 1)), "r4i-gb-constant-inverted"),
                (Fraction(fact(s), fact(a - 1) * fact(b - 1)), "r4i-gb-factorial-off-by-one"),
                (Fraction(fact(s - 1), fact(a) * fact(b)), "r4i-gb-exponent-for-parameter"),
                (Fraction(fact(s - 1), fact(a - 1)), "r4i-gb-factorials-dropped"),
            ],
            positive=True,
        )
    )
    b_dist = need(
        frac_distractors(
            m,
            [
                (Fraction(b, s), "r4i-gb-swapped-parameters"),
                (Fraction(a - 1, s - 2), "r4i-gb-mode-for-mean"),
                (Fraction(a, s + 1), "r4i-gb-denominator-plus-one"),
                (Fraction(a - 1, s), "r4i-gb-exponent-for-parameter"),
            ],
            positive=True,
            at_most=Fraction(1),
            avoid=[m],
        )
    )
    wrong = {"a": [stored(x) for x in a_dist], "b": [stored(x) for x in b_dist]}
    avoid_c = chain_avoid(order, fns, canon, wrong, "c")
    c_dist = need(
        frac_distractors(
            v,
            [
                (m * (1 - m) / s, "r4i-gb-forgot-plus-one"),
                (m * (1 - m), "r4i-gb-forgot-division"),
                (Fraction(a * b, s * (s + 1)), "r4i-gb-missing-square"),
                (m * (1 - m) / (s - 1), "r4i-gb-plus-one-sign"),
            ],
            positive=True,
            avoid=avoid_c,
        )
    )
    wrong["c"] = [stored(x) for x in c_dist]
    avoid_d = [x for x in chain_avoid(order, fns, canon, wrong, "d", rounders) if x is not None]
    d_dist = need(
        num_distractors(
            sd_val,
            [
                (float(v), "r4i-gb-variance-for-sd"),
                (math.sqrt(float(m * (1 - m) / s)), "r4i-gb-forgot-plus-one"),
                (float(v) / 2, "r4i-gb-halved"),
                (float(m * (1 - m)), "r4i-gb-forgot-division"),
            ],
            places,
            avoid=avoid_d,
        )
    )
    cs = _coef(c)
    dens = beta_density(a, b, "c").replace("f(x) = ", "")
    text = rng.choice(
        [
            f"The proportion X of a Hobart reservoir that is full on a random day has density f(x) = c {_pow('x', a - 1)} "
            f"{_pow('(1 - x)', b - 1)} for 0 < x < 1, and f(x) = 0 otherwise, where c is a constant.",
            f"At a Cairns nursery the proportion X of seedlings that survive is modelled by a density proportional to "
            f"{_pow('x', a - 1)} {_pow('(1 - x)', b - 1)} on (0, 1); write the density as c times this expression.",
            f"A random proportion X in (0, 1) has probability density function f(x) = c {_pow('x', a - 1)} "
            f"{_pow('(1 - x)', b - 1)} on that interval and 0 elsewhere. The constant c is unknown.",
        ]
    ).replace("  ", " ")
    parts = [
        dict(
            prompt="Find the value of c.",
            marks=2,
            sensible_range=(0, None),
            working=[f"c = 1/∫ x^{a - 1}(1 - x)^{b - 1} dx = {s - 1}!/({a - 1}!×{b - 1}!) = {_ft(c)}."],
            distractors=a_dist,
            **frac_fields(c),
        ),
        dict(
            prompt="Find E(X).",
            marks=2,
            sensible_range=(0, 1),
            working=[f"X has a beta distribution with parameters {a} and {b}, so E(X) = {a}/{s} = {_ft(m)}."],
            distractors=b_dist,
            **frac_fields(m),
        ),
        dict(
            prompt="Find Var(X).",
            marks=2,
            sensible_range=(0, 1),
            follow_through=f"part_b*(1 - part_b)/{s + 1}",
            working=[f"Var(X) = m(1 - m)/(a + b + 1) = {_ft(m)}×{_ft(1 - m)}/{s + 1} = {_ft(v)}."],
            distractors=c_dist,
            **frac_fields(v),
        ),
        dict(
            prompt="Find the standard deviation of X, correct to 3 decimal places.",
            marks=1,
            sensible_range=(0, 1),
            follow_through="sqrt(part_c)",
            working=[f"The standard deviation is √{_ft(v)} = {dec(sd, places)}."],
            distractors=d_dist,
            **num_fields(sd_val, places),
        ),
    ]
    return dict(text=text, parts=parts)
