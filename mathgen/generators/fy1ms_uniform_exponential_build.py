"""Generates `fy1ms_uniform_exponential` questions and the exam chain `exd_x_fy1ms_uniform_exponential`.

Skill: `ucv1-uniform-exponential-distributions` (statistics-probability). The two continuous distributions a
first-year course meets first: the uniform distribution on an interval [a, b] and the exponential distribution with
rate lambda (mean 1/lambda), with their density, distribution function, probabilities, mean, variance, percentiles
and median, and the memoryless property of the exponential.

Tiers (topic `fy1ms_uniform_exponential`):

* easy   -- the uniform distribution: the height of the density, the probability of an interval, or the mean;
* medium -- the uniform variance or a percentile; or the exponential distribution: the probability of waiting longer
            than t when only the MEAN is given (so the rate is its reciprocal), the distribution function at t, or
            the density at t;
* hard   -- one of: the median of an exponential from its mean; the time that a stated proportion of waiting times
            fall below; the probability of an interval under an exponential; a conditional probability under the
            memoryless property (the elapsed time must be recognised as irrelevant); a conditional probability for a
            uniform distribution (where, unlike the exponential, the elapsed time does matter).

Exam (topic `exd_x_fy1ms_uniform_exponential`): times between calls to a help line, exponential with a stated mean.
(a) the rate (1 mark), (b) the probability the next gap exceeds t (2 marks, follow-through from (a)), (c) the median
gap (1 mark, follow-through from (a)), (d) the probability that a gap exceeds s + t given that it has already lasted
s (2 marks, follow-through from (a)). The mean is chosen so that the rate is a decimal of at most three places, so
the rounded rate carries no error into the later parts.

Every probability is checked a second way by numerical integration of the density (mpmath quadrature), the median
and percentiles by root-finding on the distribution function, so the closed forms are never their own check.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import mpmath

from ..core import attempts, check, generator
from ._fy1ms_common import (
    Redraw,
    build_until,
    clear_of_tie,
    dp_text,
    ft_close,
    num_dist,
    number_part,
)

# ---------------------------------------------------------------------------
# Numeric helpers (exact where possible; otherwise mpmath at 30 digits, then a Fraction for rounding)


def _frac(value) -> Fraction:
    """A mpmath number as an exact Fraction of its first 25 digits."""
    return Fraction(mpmath.nstr(value, 25))


def _mpf(f: Fraction):
    return mpmath.mpf(f.numerator) / f.denominator


def _places(dp: int) -> str:
    return f"{dp} decimal place{'s' if dp != 1 else ''}"


def _same(actual: Fraction, expected, what: str) -> None:
    a = _mpf(actual)
    check(abs(a - expected) <= mpmath.mpf("1e-10") * max(1, abs(expected)), f"{what}: second route disagrees")


def _question(text: str, value, dp: int, cands, working) -> dict:
    part = number_part("", value, dp, cands, working=working)
    return dict(
        text=text,
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_kind="number",
        working=part["working"],
        distractors=part["distractors"],
    )


def _num(f: Fraction) -> str:
    """A rate or a limit as it is printed in a stem: the shortest exact decimal."""
    if f.denominator == 1:
        return str(f.numerator)
    text = f"{float(f):.6f}".rstrip("0")
    check(Fraction(text) == f, "a stem number is not a short decimal")
    return text


# ---------------------------------------------------------------------------
# Uniform distribution

_WIDTHS = (4, 5, 8, 10, 20, 25, 40, 50)  # 1/width has at most three decimal places

_UNIFORM = [
    dict(
        lead="The waiting time X, in minutes, for a tram is uniformly distributed between {a} and {b} minutes.",
        unit="minutes",
        lo=(0, 6),
    ),
    dict(
        lead="A machine fills bottles so that the volume X, in millilitres, is uniformly distributed between {a} and {b} mL.",
        unit="mL",
        lo=(480, 520),
    ),
    dict(
        lead="A delivery driver arrives at a time X, in minutes after noon, that is uniformly distributed between {a} and {b} minutes.",
        unit="minutes",
        lo=(0, 30),
    ),
    dict(
        lead="A random number generator produces a value X that is uniformly distributed between {a} and {b}.",
        unit="",
        lo=(-10, 10),
    ),
    dict(
        lead="A technician cuts a cable at a point X, in centimetres from one end, that is uniformly distributed between {a} and {b} cm.",
        unit="cm",
        lo=(0, 40),
    ),
]


class _U:
    def __init__(self, rng: random.Random):
        self.ctx = rng.choice(_UNIFORM)
        self.a = rng.randint(*self.ctx["lo"])
        self.w = rng.choice(_WIDTHS)
        self.b = self.a + self.w

    def lead(self) -> str:
        return self.ctx["lead"].format(a=self.a, b=self.b)

    def pdf_check(self) -> None:
        """The density integrates to 1 (quadrature, a different route from the closed form 1/(b - a))."""
        with mpmath.workdps(20):
            height = mpmath.mpf(1) / self.w
            check(
                abs(mpmath.quad(lambda _x: height, [self.a, self.b]) - 1) < mpmath.mpf("1e-15"),
                "density does not integrate to 1",
            )


def _u_height(rng: random.Random) -> dict:
    u = _U(rng)
    dp = 3
    value = Fraction(1, u.w)
    u.pdf_check()
    cands = [
        num_dist(Fraction(u.w), dp, "unif-pdf-height-width"),
        num_dist(Fraction(1, abs(u.b)) if u.b else Fraction(1, u.w + 1), dp, "unif-pdf-height-upper-bound"),
        num_dist(Fraction(1, u.w + 1), dp, "unif-continuous-discrete-confused"),
        num_dist(Fraction(1, abs(u.a + u.b)) if u.a + u.b else Fraction(2, u.w), dp, "unif-pdf-height-sum"),
        num_dist(Fraction(u.w, 2), dp, "unif-pdf-height-half-width"),
    ]
    text = (
        f"{u.lead()} Its probability density function is constant on this interval. "
        f"Find the value of the density f(x) for {u.a} ≤ x ≤ {u.b}, correct to {_places(dp)}."
    )
    return _question(
        text,
        value,
        dp,
        cands,
        [
            f"The density is constant, and the total area must be 1, so f(x) = 1/({u.b} − {_p(u.a)}) = 1/{u.w} = {dp_text(value, dp)}."
        ],
    )


def _u_interval(rng: random.Random) -> dict:
    u = _U(rng)
    dp = 3
    c = u.a + rng.randint(0, u.w - 2)
    d = rng.randint(c + 1, u.b)
    if (c, d) == (u.a, u.b):
        raise Redraw("the whole interval")
    value = Fraction(d - c, u.w)
    with mpmath.workdps(20):
        _same(value, mpmath.quad(lambda _x: mpmath.mpf(1) / u.w, [c, d]), "uniform probability")
    cands = [
        num_dist(Fraction(d - c, u.b) if u.b else Fraction(d - c, u.w + 1), dp, "unif-prob-over-upper-bound"),
        num_dist(1 - value, dp, "unif-prob-complement"),
        num_dist(Fraction(d - u.a, u.w), dp, "unif-prob-cdf-not-difference"),
        num_dist(Fraction(d - c, u.w + 1), dp, "unif-continuous-discrete-confused"),
        num_dist(Fraction(d - c), dp, "unif-prob-width-not-divided"),
    ]
    text = f"{u.lead()} Find P({c} < X < {d}), correct to {_places(dp)}."
    return _question(
        text,
        value,
        dp,
        cands,
        [f"P({c} < X < {d}) = ({d} − {_p(c)})/({u.b} − {_p(u.a)}) = {d - c}/{u.w} = {dp_text(value, dp)}."],
    )


def _u_mean(rng: random.Random) -> dict:
    u = _U(rng)
    dp = 1
    value = Fraction(u.a + u.b, 2)
    cands = [
        num_dist(Fraction(u.w, 2), dp, "unif-mean-half-width"),
        num_dist(Fraction(u.b, 2), dp, "unif-mean-half-upper"),
        num_dist(Fraction(u.w, 12), dp, "unif-variance-no-square"),
        num_dist(Fraction(u.a + u.b), dp, "unif-mean-sum-not-halved"),
    ]
    text = f"{u.lead()} Find the mean E(X), correct to {_places(dp)}."
    return _question(text, value, dp, cands, [f"E(X) = (a + b)/2 = ({u.a} + {_p(u.b)})/2 = {dp_text(value, dp)}."])


def _u_variance(rng: random.Random) -> dict:
    u = _U(rng)
    dp = 2
    value = Fraction(u.w**2, 12)
    if not clear_of_tie(value, dp):
        raise Redraw("variance on a rounding tie")
    sd = math.sqrt(u.w**2 / 12)
    cands = [
        num_dist(Fraction(u.w, 12), dp, "unif-variance-no-square"),
        num_dist(Fraction(u.w**2, 2), dp, "unif-variance-divisor-two"),
        num_dist(Fraction(u.w**2), dp, "unif-variance-divisor-forgotten"),
        num_dist(Fraction(repr(sd)), dp, "unif-variance-sd-confused"),
        num_dist(Fraction(u.w**2, 4), dp, "unif-variance-divisor-four"),
    ]
    text = f"{u.lead()} Find the variance of X, correct to {_places(dp)}."
    return _question(text, value, dp, cands, [f"Var(X) = (b − a)²/12 = {u.w}²/12 = {dp_text(value, dp)}."])


def _u_percentile(rng: random.Random) -> dict:
    u = _U(rng)
    dp = 2
    q = rng.choice((Fraction(1, 4), Fraction(3, 4), Fraction(9, 10), Fraction(1, 10), Fraction(1, 5)))
    value = u.a + q * u.w
    with mpmath.workdps(20):
        root = mpmath.findroot(lambda x: (x - u.a) / mpmath.mpf(u.w) - _mpf(q), u.a + 1)
        _same(value, root, "uniform percentile")
    pct = int(q * 100)
    cands = [
        num_dist(q * u.w, dp, "unif-percentile-offset-forgotten"),
        num_dist(u.b - q * u.w, dp, "unif-percentile-from-top"),
        num_dist(q * u.b, dp, "unif-percentile-fraction-of-upper"),
        num_dist(u.a + (1 - q) * u.w, dp, "unif-percentile-complement"),
    ]
    text = f"{u.lead()} Find the value m such that P(X ≤ m) = {_num(q)} (the {pct}th percentile), correct to {_places(dp)}."
    return _question(
        text,
        value,
        dp,
        cands,
        [f"F(m) = (m − {_p(u.a)})/{u.w} = {_num(q)}, so m = {u.a} + {_num(q)} × {u.w} = {dp_text(value, dp)}."],
    )


def _u_conditional(rng: random.Random) -> dict:
    u = _U(rng)
    dp = 3
    d = u.a + rng.randint(1, u.w - 3)
    c = rng.randint(d + 1, u.b - 1)
    value = Fraction(u.b - c, u.b - d)
    if not clear_of_tie(value, dp):
        raise Redraw("probability on a rounding tie")
    with mpmath.workdps(20):
        f = lambda _x: mpmath.mpf(1) / u.w  # noqa: E731
        ratio = mpmath.quad(f, [c, u.b]) / mpmath.quad(f, [d, u.b])
        _same(value, ratio, "uniform conditional probability")
    cands = [
        num_dist(Fraction(u.b - c, u.w), dp, "unif-conditional-ignored"),
        num_dist(1 - value, dp, "unif-conditional-complement"),
        num_dist(Fraction(c - d, u.w), dp, "unif-conditional-numerator-width"),
        num_dist(Fraction(u.b - c, c - d), dp, "unif-conditional-wrong-denominator"),
    ]
    text = (
        f"{u.lead()} Given that X is greater than {d}, find the probability that X is greater than {c}, "
        f"correct to {_places(dp)}."
    )
    return _question(
        text,
        value,
        dp,
        cands,
        [
            f"Once X > {d}, it is uniform on ({d}, {u.b}), an interval of length {u.b - d}.",
            f"P(X > {c} | X > {d}) = ({u.b} − {_p(c)})/({u.b} − {_p(d)}) = {u.b - c}/{u.b - d} = {dp_text(value, dp)}.",
        ],
    )


# ---------------------------------------------------------------------------
# Exponential distribution

_EXPONENTIAL = [
    dict(
        lead="The time X, in minutes, between calls to a help line is exponentially distributed",
        unit="minutes",
        means=(2, 4, 5, 8, 10),
        what="the time between calls",
    ),
    dict(
        lead="The time X, in years, until a pump fails is exponentially distributed",
        unit="years",
        means=(2, 4, 5, 8, 10),
        what="the time until the pump fails",
    ),
    dict(
        lead="The time X, in minutes, that a customer waits at a checkout is exponentially distributed",
        unit="minutes",
        means=(2, 4, 5, 8),
        what="the waiting time",
    ),
    dict(
        lead="The lifetime X, in hours, of an LED lamp is exponentially distributed",
        unit="hours",
        means=(100, 200, 250, 400, 500, 1000),
        what="the lifetime",
    ),
]


class _E:
    """An exponential setting: mean mu (an integer), rate lambda = 1/mu (an exact short decimal)."""

    def __init__(self, rng: random.Random, *, short_rate: bool = False):
        pool = [c for c in _EXPONENTIAL if any(m <= 50 for m in c["means"])] if short_rate else _EXPONENTIAL
        self.ctx = rng.choice(pool)
        means = [m for m in self.ctx["means"] if m <= 50] if short_rate else self.ctx["means"]
        self.mu = rng.choice(means)
        self.lam = Fraction(1, self.mu)
        self.unit = self.ctx["unit"]

    def by_mean(self) -> str:
        return f"{self.ctx['lead']} with mean {self.mu} {self.unit}."

    def by_rate(self) -> str:
        return f"{self.ctx['lead']} with rate λ = {_num(self.lam)} per {self.unit.rstrip('s')}."

    def scale(self, rng: random.Random, lo: float, hi: float) -> Fraction:
        """A time t (an integer, or a half-integer when mu is small) with lambda * t between lo and hi."""
        step = Fraction(1, 2) if self.mu <= 10 else Fraction(self.mu, 4)
        choices = [k * step for k in range(1, 400) if lo <= float(self.lam * k * step) <= hi]
        check(bool(choices), "no time in range")
        return rng.choice(choices)

    def tail(self, t: Fraction) -> Fraction:
        return _frac(mpmath.exp(-_mpf(self.lam) * _mpf(t)))

    def check_interval(self, lo: Fraction, hi: Fraction, value: Fraction) -> None:
        with mpmath.workdps(20):
            lam = _mpf(self.lam)
            area = mpmath.quad(lambda x: lam * mpmath.exp(-lam * x), [_mpf(lo), _mpf(hi)])
            _same(value, area, "exponential probability by quadrature")


def _p(n: int) -> str:
    """A whole number for the middle of a formula: negatives in brackets."""
    return f"({n})" if n < 0 else str(n)


def _amount(t: Fraction, unit: str) -> str:
    """``1 minute`` or ``2.5 minutes``."""
    return f"{_t_text(t)} {unit[:-1] if t == 1 else unit}"


def _t_text(t: Fraction) -> str:
    return str(t.numerator) if t.denominator == 1 else f"{float(t):g}"


def _e_tail_from_mean(rng: random.Random) -> dict:
    e = _E(rng)
    dp = 4
    t = e.scale(rng, 0.25, 3.2)
    value = e.tail(t)
    if not clear_of_tie(value, dp):
        raise Redraw("probability on a rounding tie")
    e.check_interval(t, Fraction(60 * e.mu), value)
    mu = Fraction(e.mu)
    cands = [
        num_dist(_frac(mpmath.exp(-_mpf(mu) * _mpf(t))), dp, "expo-rate-mean-confused"),
        num_dist(1 - value, dp, "expo-tail-complement"),
        num_dist(_frac(mpmath.exp(-_mpf(e.lam))), dp, "expo-time-forgotten"),
        num_dist(max(Fraction(0), 1 - t / mu), dp, "expo-linear-not-exponential"),
    ]
    text = f"{e.by_mean()} Find P(X > {_t_text(t)}), correct to {_places(dp)}."
    return _question(
        text,
        value,
        dp,
        cands,
        [
            f"The rate is λ = 1/{e.mu} = {_num(e.lam)}.",
            f"P(X > t) = e^(−λt) = e^(−{_num(e.lam)} × {_t_text(t)}) = {dp_text(value, dp)}.",
        ],
    )


def _e_cdf(rng: random.Random) -> dict:
    e = _E(rng)
    dp = 4
    t = e.scale(rng, 0.25, 3.0)
    value = 1 - e.tail(t)
    if not clear_of_tie(value, dp):
        raise Redraw("probability on a rounding tie")
    e.check_interval(Fraction(0), t, value)
    pdf = e.lam * e.tail(t)
    cands = [
        num_dist(e.tail(t), dp, "expo-tail-complement"),
        num_dist(pdf, dp, "expo-pdf-as-cdf"),
        num_dist(
            1 - _frac(mpmath.exp(-_mpf(t) / _mpf(e.lam))) if t / e.lam < 30 else Fraction(1),
            dp,
            "expo-rate-mean-confused",
        ),
        num_dist(1 - _frac(mpmath.exp(-_mpf(e.lam))), dp, "expo-time-forgotten"),
    ]
    text = f"{e.by_rate()} Find P(X ≤ {_t_text(t)}), correct to {_places(dp)}."
    return _question(
        text, value, dp, cands, [f"F(t) = 1 − e^(−λt) = 1 − e^(−{_num(e.lam)} × {_t_text(t)}) = {dp_text(value, dp)}."]
    )


def _e_pdf(rng: random.Random) -> dict:
    e = _E(rng)
    dp = 4
    t = e.scale(rng, 0.2, 2.5)
    value = e.lam * e.tail(t)
    if not clear_of_tie(value, dp) or value < Fraction(1, 500):
        raise Redraw("density unsuitable")
    cands = [
        num_dist(e.tail(t), dp, "expo-density-factor-forgotten"),
        num_dist(1 - e.tail(t), dp, "expo-cdf-as-pdf"),
        num_dist(e.lam * _frac(mpmath.exp(-_mpf(e.lam))), dp, "expo-time-forgotten"),
        num_dist(e.lam * (1 - e.tail(t)), dp, "expo-density-times-cdf"),
    ]
    text = f"{e.by_rate()} Find the value of the density f({_t_text(t)}), correct to {_places(dp)}."
    with mpmath.workdps(20):
        lam = _mpf(e.lam)
        h = mpmath.mpf("1e-8")
        slope = (-mpmath.exp(-lam * (_mpf(t) + h)) + mpmath.exp(-lam * (_mpf(t) - h))) / (2 * h)
        _same(value, slope, "density as the derivative of the distribution function")
    return _question(
        text,
        value,
        dp,
        cands,
        [f"f(t) = λe^(−λt) = {_num(e.lam)} × e^(−{_num(e.lam)} × {_t_text(t)}) = {dp_text(value, dp)}."],
    )


def _e_median(rng: random.Random) -> dict:
    e = _E(rng)
    dp = 3
    value = _frac(mpmath.log(2) * e.mu)
    if not clear_of_tie(value, dp):
        raise Redraw("median on a rounding tie")
    with mpmath.workdps(20):
        lam = _mpf(e.lam)
        root = mpmath.findroot(lambda m: 1 - mpmath.exp(-lam * m) - mpmath.mpf(1) / 2, e.mu)
        _same(value, root, "exponential median by root-finding")
    cands = [
        num_dist(Fraction(e.mu), dp, "expo-median-as-mean"),
        num_dist(Fraction(e.mu, 2), dp, "expo-median-halved-mean"),
        num_dist(_frac(mpmath.log(2) * _mpf(e.lam)), dp, "expo-rate-mean-confused"),
        num_dist(_frac(mpmath.log10(2) * e.mu), dp, "expo-log-base-ten"),
    ]
    text = f"{e.by_mean()} Find the median of X, correct to {_places(dp)}."
    return _question(
        text,
        value,
        dp,
        cands,
        [f"Solve 1 − e^(−λm) = 1/2 with λ = 1/{e.mu}: m = (ln 2)/λ = {e.mu} ln 2 = {dp_text(value, dp)}."],
    )


def _e_percentile(rng: random.Random) -> dict:
    e = _E(rng)
    dp = 3
    p = rng.choice((Fraction(3, 10), Fraction(3, 4), Fraction(9, 10), Fraction(19, 20), Fraction(1, 4)))
    value = _frac(-mpmath.log(1 - _mpf(p)) * e.mu)
    if not clear_of_tie(value, dp):
        raise Redraw("percentile on a rounding tie")
    with mpmath.workdps(20):
        lam = _mpf(e.lam)
        root = mpmath.findroot(lambda m: 1 - mpmath.exp(-lam * m) - _mpf(p), e.mu)
        _same(value, root, "exponential percentile by root-finding")
    cands = [
        num_dist(_frac(-mpmath.log(_mpf(p)) * e.mu), dp, "expo-percentile-wrong-tail"),
        num_dist(_frac(-mpmath.log(1 - _mpf(p)) * _mpf(e.lam)), dp, "expo-rate-mean-confused"),
        num_dist(p * e.mu, dp, "expo-percentile-linear"),
        num_dist(_frac(mpmath.log(1 - _mpf(p)) * e.mu), dp, "expo-percentile-sign"),
    ]
    text = (
        f"{e.by_mean()} A proportion {_num(p)} of values of X are less than a time m. Find m, correct to {_places(dp)}."
    )
    return _question(
        text,
        value,
        dp,
        cands,
        [f"Solve 1 − e^(−m/{e.mu}) = {_num(p)}: m = −{e.mu} ln(1 − {_num(p)}) = {dp_text(value, dp)}."],
    )


def _e_memoryless(rng: random.Random) -> dict:
    e = _E(rng)
    dp = 4
    s = e.scale(rng, 0.4, 2.0)
    t = e.scale(rng, 0.25, 1.8)
    value = e.tail(t)
    if not clear_of_tie(value, dp) or s == t:
        raise Redraw("probability unsuitable")
    with mpmath.workdps(20):
        lam = _mpf(e.lam)
        f = lambda x: lam * mpmath.exp(-lam * x)  # noqa: E731
        ratio = mpmath.quad(f, [_mpf(s + t), mpmath.mpf(60 * e.mu)]) / mpmath.quad(f, [_mpf(s), mpmath.mpf(60 * e.mu)])
        _same(value, ratio, "conditional probability by quadrature")
    cands = [
        num_dist(e.tail(s + t), dp, "expo-memoryless-ignored"),
        num_dist(e.tail(s), dp, "expo-memoryless-wrong-time"),
        num_dist(1 - value, dp, "expo-tail-complement"),
        num_dist(_frac(mpmath.exp(_mpf(e.lam) * _mpf(t))), dp, "expo-memoryless-ratio-inverted"),
    ]
    thing = e.ctx["what"]
    text = (
        f"{e.by_rate()} Given that {thing} has already exceeded {_amount(s, e.unit)}, "
        f"find the probability that it exceeds {_amount(s + t, e.unit)} in total, correct to {_places(dp)}."
    )
    return _question(
        text,
        value,
        dp,
        cands,
        [
            f"P(X > {_t_text(s + t)} | X > {_t_text(s)}) = e^(−λ × {_t_text(s + t)})/e^(−λ × {_t_text(s)}) = e^(−λ × {_t_text(t)}).",
            f"By the memoryless property this is e^(−{_num(e.lam)} × {_t_text(t)}) = {dp_text(value, dp)}.",
        ],
    )


def _e_interval(rng: random.Random) -> dict:
    e = _E(rng)
    dp = 4
    s = e.scale(rng, 0.2, 1.2)
    t = e.scale(rng, 0.5, 3.0)
    if t <= s:
        raise Redraw("empty interval")
    value = e.tail(s) - e.tail(t)
    if not clear_of_tie(value, dp):
        raise Redraw("probability on a rounding tie")
    e.check_interval(s, t, value)
    cands = [
        num_dist(e.tail(t - s), dp, "expo-interval-width-only"),
        num_dist(e.tail(t) - e.tail(s), dp, "expo-interval-sign-reversed"),
        num_dist(e.tail(s), dp, "expo-tail-only"),
        num_dist(e.lam * e.tail(s) * (t - s), dp, "expo-interval-density-times-width"),
    ]
    text = f"{e.by_rate()} Find P({_t_text(s)} < X < {_t_text(t)}), correct to {_places(dp)}."
    return _question(
        text,
        value,
        dp,
        cands,
        [
            f"P(s < X < t) = e^(−λs) − e^(−λt) = e^(−{_num(e.lam)} × {_t_text(s)}) − e^(−{_num(e.lam)} × {_t_text(t)}) = {dp_text(value, dp)}."
        ],
    )


# ---------------------------------------------------------------------------
# The generators


def _easy(rng: random.Random) -> dict:
    pick = rng.random()
    return _u_height(rng) if pick < 0.3 else _u_interval(rng) if pick < 0.75 else _u_mean(rng)


def _medium(rng: random.Random) -> dict:
    pick = rng.random()
    if pick < 0.18:
        return _u_variance(rng)
    if pick < 0.36:
        return _u_percentile(rng)
    if pick < 0.6:
        return _e_tail_from_mean(rng)
    return _e_cdf(rng) if pick < 0.8 else _e_pdf(rng)


def _hard(rng: random.Random) -> dict:
    pick = rng.random()
    if pick < 0.2:
        return _e_median(rng)
    if pick < 0.38:
        return _e_percentile(rng)
    if pick < 0.6:
        return _e_memoryless(rng)
    return _e_interval(rng) if pick < 0.8 else _u_conditional(rng)


@generator(
    "fy1ms_uniform_exponential",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1ms_uniform_exponential(rng: random.Random, tier: str = "easy") -> dict:
    """Uniform and exponential distributions: density and probability (easy), variance, percentile and exponential tails (medium), median, memorylessness and conditional probability (hard)."""
    if tier == "easy":
        return build_until(rng, _easy, "an easy uniform question")
    if tier == "medium":
        return build_until(rng, _medium, "a medium distribution question")
    if tier == "hard":
        return build_until(rng, _hard, "a hard distribution question")
    raise AssertionError(f"unknown tier {tier!r}")


@generator("exd_x_fy1ms_uniform_exponential", level=19, presets={"exam": dict(tier="exam")})
def exd_x_fy1ms_uniform_exponential(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: rate, a tail probability, the median, and a memoryless conditional probability (6 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_until(rng, _exam_chain, "an exponential exam question")


def _exam_chain(rng: random.Random) -> dict:
    e = _E(rng, short_rate=True)
    dp_rate, dp_p, dp_m = 3, 4, 3
    for _ in attempts(100, "an exam draw with distinct times"):
        t = e.scale(rng, 0.3, 2.4)
        s = e.scale(rng, 0.4, 2.0)
        t2 = e.scale(rng, 0.25, 1.8)
        if len({t, t2}) == 2:
            break
    lam_text = dp_text(e.lam, dp_rate)
    check(Fraction(lam_text) == e.lam, "the exam rate must be exact at 3 decimal places")
    b_value = e.tail(t)
    d_value = e.tail(t2)
    median = _frac(mpmath.log(2) * e.mu)
    for value, dp in ((b_value, dp_p), (d_value, dp_p), (median, dp_m)):
        if not clear_of_tie(value, dp):
            raise Redraw("a value sits on a rounding tie")
    e.check_interval(t, Fraction(60 * e.mu), b_value)
    with mpmath.workdps(20):
        lam = _mpf(e.lam)
        root = mpmath.findroot(lambda m: 1 - mpmath.exp(-lam * m) - mpmath.mpf(1) / 2, e.mu)
        _same(median, root, "exam median by root-finding")
        f = lambda x: lam * mpmath.exp(-lam * x)  # noqa: E731
        ratio = mpmath.quad(f, [_mpf(s + t2), mpmath.mpf(60 * e.mu)]) / mpmath.quad(f, [_mpf(s), mpmath.mpf(60 * e.mu)])
        _same(d_value, ratio, "exam conditional probability by quadrature")
    ft_close(_frac(mpmath.exp(-_mpf(Fraction(lam_text)) * _mpf(t))), dp_text(b_value, dp_p), dp_p)
    mu = Fraction(e.mu)
    a_cands = [
        num_dist(mu, dp_rate, "expo-rate-mean-confused"),
        num_dist(Fraction(1, e.mu**2), dp_rate, "expo-rate-reciprocal-squared"),
        num_dist(_frac(mpmath.log(2) / _mpf(mu)), dp_rate, "expo-rate-from-median"),
        num_dist(mu / 2, dp_rate, "expo-rate-half-mean"),
    ]
    b_cands = [
        num_dist(_frac(mpmath.exp(-_mpf(mu) * _mpf(t))), dp_p, "expo-rate-mean-confused"),
        num_dist(1 - b_value, dp_p, "expo-tail-complement"),
        num_dist(_frac(mpmath.exp(-_mpf(e.lam))), dp_p, "expo-time-forgotten"),
        num_dist(e.lam * b_value, dp_p, "expo-pdf-as-tail"),
    ]
    c_cands = [
        num_dist(mu, dp_m, "expo-median-as-mean"),
        num_dist(mu / 2, dp_m, "expo-median-halved-mean"),
        num_dist(_frac(mpmath.log(2) * _mpf(e.lam)), dp_m, "expo-rate-mean-confused"),
        num_dist(_frac(mpmath.log10(2) * mu), dp_m, "expo-log-base-ten"),
    ]
    d_cands = [
        num_dist(e.tail(s + t2), dp_p, "expo-memoryless-ignored"),
        num_dist(e.tail(s), dp_p, "expo-memoryless-wrong-time"),
        num_dist(1 - d_value, dp_p, "expo-tail-complement"),
        num_dist(_frac(mpmath.exp(_mpf(e.lam) * _mpf(t2))), dp_p, "expo-memoryless-ratio-inverted"),
    ]
    thing = e.ctx["what"]
    stem = f"{e.by_mean()} Let X be {thing}."
    return dict(
        text=stem,
        parts=[
            number_part(
                "Find the rate λ of the distribution, correct to 3 decimal places.",
                e.lam,
                dp_rate,
                a_cands,
                marks=1,
                sensible_range=(0, None),
                working=[f"λ = 1/mean = 1/{e.mu} = {lam_text}."],
            ),
            number_part(
                f"Hence find P(X > {_t_text(t)}), correct to 4 decimal places.",
                b_value,
                dp_p,
                b_cands,
                marks=2,
                follow_through=f"exp(-part_a*{_ratio(t)})",
                sensible_range=(0, 1),
                working=[f"P(X > t) = e^(−λt) = e^(−{lam_text} × {_t_text(t)}) = {dp_text(b_value, dp_p)}."],
            ),
            number_part(
                "Find the median of X, correct to 3 decimal places.",
                median,
                dp_m,
                c_cands,
                marks=1,
                follow_through="ln(2)/part_a",
                sensible_range=(0, None),
                working=[f"Solve 1 − e^(−λm) = 1/2: m = (ln 2)/λ = {dp_text(median, dp_m)}."],
            ),
            number_part(
                f"Given that {thing} has already exceeded {_amount(s, e.unit)}, find the probability that it exceeds {_amount(s + t2, e.unit)} in total, correct to 4 decimal places.",
                d_value,
                dp_p,
                d_cands,
                marks=2,
                follow_through=f"exp(-part_a*{_ratio(t2)})",
                sensible_range=(0, 1),
                working=[
                    f"By the memoryless property, P(X > {_t_text(s + t2)} | X > {_t_text(s)}) = P(X > {_t_text(t2)}) = e^(−{lam_text} × {_t_text(t2)}) = {dp_text(d_value, dp_p)}."
                ],
            ),
        ],
    )


def _ratio(t: Fraction) -> str:
    """A time as an exact rational literal for a follow-through expression."""
    return f"({t.numerator}/{t.denominator})"
