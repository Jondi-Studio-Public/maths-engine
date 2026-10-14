"""Generates `r4i_estimator_bias_mse` questions (easy, medium, hard) and the exam chain `exd_x_r4i_estimator_bias_mse`.

``uc2-estimator-bias-and-mse`` (``statistics``, answer ``fraction``, exact values): bias, variance and mean squared error
of point estimators, from the definitions E(T) - parameter and Var(T) + bias^2. Every stem states numerical values for the
parameters, so each answer is one exact fraction (a negative bias is a signed fraction).

* easy: the bias of an estimator of a mean or variance: sum / (n + k) for a mean, the divide-by-n variance estimator, or a
  weighted combination of three observations whose weights do not sum to 1.
* medium: the mean squared error of a linear estimator a X-bar + b, of a weighted combination of three observations, or of
  sum of squares / (n + k) for a normal variance (the stem recalls that (n - 1) S^2 / sigma^2 has variance 2 (n - 1)).
* hard: one of four: the MSE of the maximum of a uniform (0, theta) sample (the learner integrates the density
  n x^(n-1) / theta^n); the variance of the unbiased rescaling (n + 1) max / n; the shrinkage constant c that minimises the
  MSE of c X-bar; or the MSE of the Poisson estimator (sum + 1) / (n + 2).
* exam: a uniform (0, theta) sample, theta and n given. (a) E(max), (b) the bias of the maximum (follows through from
  (a)), (c) its mean squared error (follows through from (b); the stem supplies Var(max)), (d) the MSE of 2 X-bar divided
  by the MSE of the maximum (follows through from (c)).

The tests recompute each key by exact integration of the stated density (SymPy) or by exact enumeration of the moments of
the weighted sums, and read the numbers back out of the text.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r4i_kit import chain_avoid, frac_distractors, frac_fields, need, redraw, retry, stored

TOPIC = "r4i_estimator_bias_mse"
EXAM = "exd_x_r4i_estimator_bias_mse"


def _ft(v: Fraction) -> str:
    v = Fraction(v)
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def _paren(v: Fraction) -> str:
    v = Fraction(v)
    return f"({_ft(v)})" if v.denominator != 1 or v < 0 else str(v)


def _w(v: Fraction) -> str:
    """A weight written in front of an observation: `X1`, `(1/2)X1`."""
    return "" if v == 1 else f"({_ft(v)})"


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_mean_scaled(rng):
    n = rng.choice((5, 8, 10, 12, 15, 20))
    k = rng.choice((1, 2, 3, -1))
    mu = rng.choice((10, 12, 15, 20, 24, 30, 40))
    ans = Fraction(-k * mu, n + k)
    cands = [
        (-ans, "r4i-bm-sign-reversed"),
        (Fraction(-k * mu, n), "r4i-bm-wrong-denominator"),
        (Fraction(-mu, n + k), "r4i-bm-forgot-multiplier"),
        (Fraction(n * mu, n + k), "r4i-bm-expectation-for-bias"),
    ]
    dist = need(frac_distractors(ans, cands))
    den = f"n + {k}" if k > 0 else f"n - {-k}"
    text = (
        f"A random sample X1, ..., X{n} is drawn from a population with mean {mu}. To estimate the mean, a technician "
        f"uses T = (X1 + X2 + ... + X{n})/{n + k}. Find the bias of T, that is E(T) minus the true mean."
    )
    working = [
        f"E(T) = {n}×{mu}/{n + k} = {_ft(Fraction(n * mu, n + k))}.",
        f"The bias is E(T) - {mu} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _easy_variance(rng):
    n = rng.choice((4, 5, 6, 8, 10, 12, 20))
    s2 = rng.choice((8, 12, 15, 18, 20, 24, 30, 36))
    ans = Fraction(-s2, n)
    cands = [
        (-ans, "r4i-bm-sign-reversed"),
        (Fraction(-s2, n - 1), "r4i-bm-wrong-denominator"),
        (Fraction(s2 * (n - 1), n), "r4i-bm-expectation-for-bias"),
        (Fraction(0), "r4i-bm-claimed-unbiased"),
        (Fraction(-s2, n * n), "r4i-bm-squared-denominator"),
    ]
    dist = need(frac_distractors(ans, cands))
    text = rng.choice(
        [
            f"A sample of size {n} is taken from a population with variance {s2}. The estimator V = (1/{n}) Σ(Xi - X̄)² "
            f"(each squared deviation taken from the sample mean) is used for the variance. Find the bias of V.",
            f"Readings X1, ..., X{n} from a calibrated sensor are independent with variance {s2}. An analyst estimates the "
            f"variance by dividing the sum of squared deviations from the sample mean by {n}. What is the bias of this estimator?",
        ]
    )
    working = [
        f"E(Σ(Xi - X̄)²) = ({n} - 1)×{s2} = {(n - 1) * s2}, so E(V) = {_ft(Fraction((n - 1) * s2, n))}.",
        f"Bias = E(V) - {s2} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _weights(rng):
    for _ in attempts(100, "weights that do not sum to 1"):
        w = [Fraction(rng.choice((1, 2, 3)), rng.choice((2, 3, 4, 5, 6))) for _ in range(3)]
        if sum(w) != 1 and 0 < abs(sum(w) - 1) < 1:
            return w
    raise AssertionError("unreachable")


def _combo(w):
    return " + ".join(f"{_w(x)}X{i + 1}" for i, x in enumerate(w)).replace("(1)", "")


def _easy_weights(rng):
    w = _weights(rng)
    mu = rng.choice((10, 12, 20, 30, 40, 60))
    total = sum(w)
    ans = (total - 1) * mu
    cands = [
        (-ans, "r4i-bm-sign-reversed"),
        (total * mu, "r4i-bm-expectation-for-bias"),
        ((total - 1) * mu / 3, "r4i-bm-averaged"),
        (Fraction(sum(x * x for x in w) - 1) * mu, "r4i-bm-squared-weights"),
    ]
    dist = need(frac_distractors(ans, cands))
    text = (
        f"Three independent measurements X1, X2, X3 each have mean {mu}. The estimator T = {_combo(w)} is proposed for "
        f"the mean. Find the bias of T."
    )
    working = [f"E(T) = ({_ft(total)})×{mu} = {_ft(total * mu)}.", f"Bias = E(T) - {mu} = {_ft(ans)}."]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _easy(rng):
    return rng.choice((_easy_mean_scaled, _easy_variance, _easy_weights))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium_linear(rng):
    n = rng.choice((4, 5, 8, 9, 10, 16, 20, 25))
    a = rng.choice(
        (Fraction(1, 2), Fraction(3, 4), Fraction(4, 5), Fraction(9, 10), Fraction(11, 10), Fraction(5, 4), Fraction(2))
    )
    b = rng.choice((-2, -1, 0, 1, 2, 3))
    mu = rng.choice((6, 10, 12, 20))
    s2 = rng.choice((4, 9, 16, 25, 36))
    bias = (a - 1) * mu + b
    var = a * a * Fraction(s2, n)
    ans = var + bias * bias
    retry(bias != 0, "an unbiased estimator hides the bias step")
    cands = [
        (var, "r4i-bm-variance-only"),
        (bias * bias, "r4i-bm-bias-squared-only"),
        (var + abs(bias), "r4i-bm-bias-not-squared"),
        (a * Fraction(s2, n) + bias * bias, "r4i-bm-multiplier-not-squared"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    bt = f" + {b}" if b > 0 else (f" - {-b}" if b < 0 else "")
    text = (
        f"A random sample of size {n} is taken from a population with mean {mu} and variance {s2}. The estimator "
        f"T = {_paren(a) if a != 1 else ''}X̄{bt} is used for the mean, where X̄ is the sample mean. Find the mean squared "
        f"error of T."
    )
    working = [
        f"E(T) = {_ft(a)}×{mu}{bt} = {_ft(a * mu + b)}, so the bias is {_ft(bias)}.",
        f"Var(T) = ({_ft(a)})²×{s2}/{n} = {_ft(var)}.",
        f"MSE = Var + bias² = {_ft(var)} + {_ft(bias * bias)} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium_weights(rng):
    w = _weights(rng)
    mu = rng.choice((10, 12, 20, 30))
    s2 = rng.choice((4, 9, 16, 25))
    total = sum(w)
    bias = (total - 1) * mu
    var = sum(x * x for x in w) * s2
    ans = var + bias * bias
    cands = [
        (var, "r4i-bm-variance-only"),
        (bias * bias, "r4i-bm-bias-squared-only"),
        (total * total * s2 + bias * bias, "r4i-bm-square-of-sum-weights"),
        (var + abs(bias), "r4i-bm-bias-not-squared"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"Three independent measurements X1, X2, X3 each have mean {mu} and variance {s2}. The estimator "
        f"T = {_combo(w)} is proposed for the mean. Find the mean squared error of T."
    )
    working = [
        f"E(T) = ({_ft(total)})×{mu}, so the bias is {_ft(bias)}.",
        f"Var(T) = ({' + '.join('(' + _ft(x) + ')²' for x in w)})×{s2} = {_ft(var)}.",
        f"MSE = {_ft(var)} + {_ft(bias * bias)} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium_chi(rng):
    n = rng.choice((5, 6, 8, 10, 12, 15))
    k = rng.choice((-1, 0, 1, 2))
    s2 = rng.choice((2, 3, 4, 5, 6))
    q = s2 * s2
    bias2 = Fraction((k + 1) ** 2 * q, (n + k) ** 2)
    var = Fraction(2 * (n - 1) * q, (n + k) ** 2)
    ans = bias2 + var
    cands = [
        (var, "r4i-bm-variance-only"),
        (bias2, "r4i-bm-bias-squared-only"),
        (Fraction(2 * n * q + (k + 1) ** 2 * q, (n + k) ** 2), "r4i-bm-n-for-n-minus-one"),
        (Fraction(2 * (n - 1) * q, (n + k) ** 2) + Fraction((k + 1) * q, n + k), "r4i-bm-bias-not-squared"),
        (Fraction(2 * (n - 1) * q, n - 1), "r4i-bm-unbiased-assumed"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"A sample of size {n} comes from a normal population with variance {s2}. Let Q = Σ(Xi - X̄)², the sum of squared "
        f"deviations from the sample mean, and use V = Q/{n + k} to estimate the variance. Recall that Q/{s2} has mean "
        f"{n - 1} and variance {2 * (n - 1)}. Find the mean squared error of V."
    )
    working = [
        f"E(V) = {s2}×{n - 1}/{n + k}, so the bias is {s2}×({n - 1} - {n + k})/{n + k} = {_ft(Fraction(-(k + 1) * s2, n + k))}.",
        f"Var(V) = {s2}²×{2 * (n - 1)}/{n + k}² = {_ft(var)}.",
        f"MSE = {_ft(var)} + {_ft(bias2)} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _medium(rng):
    return rng.choice((_medium_linear, _medium_weights, _medium_chi))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard_uniform_mse(rng):
    n = rng.choice((3, 4, 5, 6, 8, 9, 10))
    th = rng.choice((6, 8, 10, 12, 15, 20))
    ans = Fraction(2 * th * th, (n + 1) * (n + 2))
    cands = [
        (Fraction(n * th * th, (n + 1) ** 2 * (n + 2)), "r4i-bm-variance-only"),
        (Fraction(th * th, (n + 1) ** 2), "r4i-bm-bias-squared-only"),
        (Fraction(th * th, 3 * n), "r4i-bm-mse-of-other-estimator"),
        (Fraction(2 * th * th, n * (n + 1)), "r4i-bm-moment-shifted"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = rng.choice(
        [
            f"A reading is uniformly distributed on (0, θ) with θ = {th} in truth, though the analyst does not know this. "
            f"From a random sample of {n} readings the maximum M is used to estimate θ. The density of M on (0, θ) is "
            f"n x^(n-1)/θ^n with n = {n}. Find the mean squared error of M as an estimator of θ.",
            f"The waiting times at a depot are uniform on (0, θ), where in fact θ = {th} minutes. The largest of {n} "
            f"independent waits, M, is proposed as an estimator of θ; M has density {n} x^{n - 1}/θ^{n} on (0, θ). "
            f"Find the mean squared error of M.",
        ]
    )
    working = [
        f"E(M) = ∫ x×{n}x^{n - 1}/{th}^{n} dx = {n}×{th}/{n + 1}, so the bias is -{_ft(Fraction(th, n + 1))}.",
        f"E(M²) = {n}×{th}²/{n + 2}, so Var(M) = {_ft(Fraction(n * th * th, (n + 1) ** 2 * (n + 2)))}.",
        f"MSE = Var + bias² = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard_uniform_variance(rng):
    n = rng.choice((3, 4, 5, 6, 8, 9, 10))
    th = rng.choice((6, 8, 10, 12, 15, 20))
    ans = Fraction(th * th, n * (n + 2))
    cands = [
        (Fraction(n * th * th, (n + 1) ** 2 * (n + 2)), "r4i-bm-rescaling-ignored"),
        (Fraction(th * th, 3 * n), "r4i-bm-variance-of-other-estimator"),
        (Fraction(th * th, n * (n + 1)), "r4i-bm-moment-shifted"),
        (Fraction(n + 1, n) * Fraction(n * th * th, (n + 1) ** 2 * (n + 2)), "r4i-bm-rescaling-not-squared"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"A random sample of {n} readings is uniform on (0, θ), where θ = {th} in truth. The maximum M has density "
        f"{n} x^{n - 1}/θ^{n} on (0, θ). The rescaled estimator U = ({n} + 1)M/{n} is unbiased for θ. Find Var(U)."
    )
    working = [
        f"E(M²) = {n}θ²/{n + 2} and E(M) = {n}θ/{n + 1}, so Var(M) = {n}θ²/(({n + 1})²({n + 2})).",
        f"Var(U) = (({n} + 1)/{n})² Var(M) = θ²/({n}×{n + 2}) = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard_shrinkage(rng):
    n = rng.choice((4, 5, 8, 9, 10, 16))
    mu = rng.choice((2, 3, 4, 5, 6))
    s2 = rng.choice((9, 16, 25, 36, 64))
    c = Fraction(mu * mu, 1) / (mu * mu + Fraction(s2, n))
    cands = [
        (1 - c, "r4i-bm-complement"),
        (Fraction(mu * mu, mu * mu + s2), "r4i-bm-forgot-sample-size"),
        (Fraction(mu) / (mu + Fraction(s2, n)), "r4i-bm-mean-not-squared"),
        (Fraction(mu * mu) / (mu * mu + Fraction(s2, n) ** 2), "r4i-bm-variance-squared"),
    ]
    dist = need(frac_distractors(c, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"A random sample of {n} observations comes from a population with mean {mu} and variance {s2}. Consider "
        f"estimators of the form T = c X̄ for a constant c. Find the value of c that minimises the mean squared error of T "
        f"as an estimator of the mean (using the true mean and variance above)."
    )
    working = [
        f"MSE(c) = c²×{s2}/{n} + (c - 1)²×{mu}².",
        f"Setting the derivative to zero: c = {mu}²/({mu}² + {s2}/{n}) = {_ft(c)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(c))


def _hard_poisson(rng):
    n = rng.choice((4, 6, 8, 10, 12))
    lam = rng.choice((Fraction(1), Fraction(2), Fraction(3), Fraction(1, 2), Fraction(3, 2), Fraction(5, 2)))
    bias = (1 - 2 * lam) / (n + 2)
    var = n * lam / (n + 2) ** 2
    ans = bias * bias + var
    cands = [
        (var, "r4i-bm-variance-only"),
        (bias * bias, "r4i-bm-bias-squared-only"),
        ((1 + 2 * lam) ** 2 / (n + 2) ** 2 + var, "r4i-bm-bias-sign-inside"),
        (bias * bias + n * lam / (n + 2), "r4i-bm-variance-denominator-not-squared"),
        ((1 - 2 * lam) ** 2 / (n + 2) ** 2 + n * lam / n**2, "r4i-bm-variance-of-mean-used"),
    ]
    dist = need(frac_distractors(ans, cands, positive=True))
    text = (
        f"Counts X1, ..., X{n} are independent Poisson variables with the same mean λ, and in truth λ = {_ft(lam)}. The "
        f"estimator T = (X1 + ... + X{n} + 1)/({n} + 2) is used for λ. Find the mean squared error of T."
    )
    working = [
        f"E(T) = ({n}×{_ft(lam)} + 1)/{n + 2}, so the bias is {_ft(bias)}.",
        f"Var(T) = {n}×{_ft(lam)}/{n + 2}² = {_ft(var)}.",
        f"MSE = {_ft(var)} + {_ft(bias * bias)} = {_ft(ans)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(ans))


def _hard(rng):
    return rng.choice((_hard_uniform_mse, _hard_uniform_variance, _hard_shrinkage, _hard_poisson))(rng)


@generator(
    TOPIC,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraw
def r4i_estimator_bias_mse(rng, tier="easy"):
    """Bias, variance and mean squared error of point estimators; see the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------


@generator(EXAM, level=21, presets={"exam": dict(tier="exam")})
@redraw
def exd_x_r4i_estimator_bias_mse(rng, tier="exam"):
    """Uniform (0, theta): E(max), its bias, its MSE, then the ratio of the MSE of 2 X-bar to it."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = rng.choice((4, 5, 6, 8, 9, 10))
    th = rng.choice((6, 8, 10, 12, 15, 18, 20))
    ea = Fraction(n * th, n + 1)
    bias = ea - th
    var_max = Fraction(n * th * th, (n + 1) ** 2 * (n + 2))
    mse = bias * bias + var_max
    mse_other = Fraction(th * th, 3 * n)
    ratio = mse_other / mse
    check(mse == Fraction(2 * th * th, (n + 1) * (n + 2)), "MSE disagrees with the closed form")
    order = ["a", "b", "c", "d"]
    canon = {"a": ea, "b": bias, "c": mse, "d": ratio}
    fns = {
        "b": lambda s: s["a"] - th,
        "c": lambda s: s["b"] ** 2 + var_max,
        "d": lambda s: mse_other / s["c"] if s["c"] != 0 else None,
    }
    a_dist = need(
        frac_distractors(
            ea,
            [
                (Fraction(th, 2), "r4i-bm-midpoint-for-expectation"),
                (Fraction((n - 1) * th, n), "r4i-bm-moment-shifted"),
                (Fraction(n * th, n + 2), "r4i-bm-moment-shifted"),
                (Fraction((n + 1) * th, n), "r4i-bm-inverted-ratio"),
            ],
            positive=True,
        )
    )
    wrong = {"a": [stored(x) for x in a_dist]}
    avoid_b = chain_avoid(order, fns, canon, wrong, "b")
    b_dist = need(
        frac_distractors(
            bias,
            [
                (-bias, "r4i-bm-sign-reversed"),
                (ea, "r4i-bm-expectation-for-bias"),
                (Fraction(-th, n), "r4i-bm-wrong-denominator"),
                (Fraction(-th, n + 2), "r4i-bm-wrong-denominator"),
            ],
            avoid=avoid_b,
        )
    )
    wrong["b"] = [stored(x) for x in b_dist]
    avoid_c = chain_avoid(order, fns, canon, wrong, "c")
    c_dist = need(
        frac_distractors(
            mse,
            [
                (var_max, "r4i-bm-variance-only"),
                (bias * bias, "r4i-bm-bias-squared-only"),
                (var_max + abs(bias), "r4i-bm-bias-not-squared"),
                (Fraction(th * th, 3 * n), "r4i-bm-mse-of-other-estimator"),
            ],
            positive=True,
            avoid=avoid_c,
        )
    )
    wrong["c"] = [stored(x) for x in c_dist]
    avoid_d = [x for x in chain_avoid(order, fns, canon, wrong, "d") if x is not None]
    d_dist = need(
        frac_distractors(
            ratio,
            [
                (1 / ratio, "r4i-bm-ratio-inverted"),
                (mse_other, "r4i-bm-forgot-to-divide"),
                (mse_other / var_max, "r4i-bm-variance-instead-of-mse"),
                (ratio * 2, "r4i-bm-doubled"),
            ],
            positive=True,
            avoid=avoid_d,
        )
    )
    text = rng.choice(
        [
            f"A reading is uniformly distributed on (0, θ), where in truth θ = {th}. A random sample of n = {n} readings is "
            f"taken, with maximum M and mean X̄. The density of M is n x^(n-1)/θ^n on (0, θ). You may use "
            f"Var(M) = nθ²/((n + 1)²(n + 2)) and Var(X̄) = θ²/(12n).",
            f"Waiting times at a Perth depot are uniform on (0, θ) with θ = {th} minutes (unknown to the manager). From "
            f"n = {n} independent waits the manager forms the maximum M and the sample mean X̄. M has density "
            f"n x^(n-1)/θ^n on (0, θ), and you may use Var(M) = nθ²/((n + 1)²(n + 2)) and Var(X̄) = θ²/(12n).",
            f"To estimate θ, the endpoint of a uniform distribution on (0, θ), a surveyor takes n = {n} observations. In truth "
            f"θ = {th}. Let M be their maximum (density n x^(n-1)/θ^n) and X̄ their mean. You may use "
            f"Var(M) = nθ²/((n + 1)²(n + 2)) and Var(X̄) = θ²/(12n).",
        ]
    )
    parts = [
        dict(
            prompt="Find E(M).",
            marks=2,
            sensible_range=(0, None),
            working=[f"E(M) = ∫ x×{n}x^{n - 1}/{th}^{n} dx = {n}×{th}/{n + 1} = {_ft(ea)}."],
            distractors=a_dist,
            **frac_fields(ea),
        ),
        dict(
            prompt="Find the bias of M as an estimator of θ.",
            marks=1,
            follow_through=f"part_a - {th}",
            working=[f"Bias = E(M) - θ = {_ft(ea)} - {th} = {_ft(bias)}."],
            distractors=b_dist,
            **frac_fields(bias),
        ),
        dict(
            prompt="Find the mean squared error of M.",
            marks=2,
            sensible_range=(0, None),
            follow_through=f"part_b**2 + {_ft(var_max)}",
            working=[f"MSE = bias² + Var(M) = {_ft(bias * bias)} + {_ft(var_max)} = {_ft(mse)}."],
            distractors=c_dist,
            **frac_fields(mse),
        ),
        dict(
            prompt=(
                "The estimator 2X̄ is unbiased for θ. Find the ratio of the mean squared error of 2X̄ to the mean squared "
                "error of M."
            ),
            marks=2,
            sensible_range=(0, None),
            follow_through=f"{_ft(mse_other)}/part_c",
            working=[
                f"MSE(2X̄) = Var(2X̄) = 4θ²/(12n) = {_ft(mse_other)}.",
                f"The ratio is {_ft(mse_other)}/{_ft(mse)} = {_ft(ratio)}.",
            ],
            distractors=d_dist,
            **frac_fields(ratio),
        ),
    ]
    return dict(text=text, parts=parts)
