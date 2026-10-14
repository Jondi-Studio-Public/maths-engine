"""Generates `fy1ms_t_test_two_sample` questions and the exam chain `exd_x_fy1ms_t_test_two_sample`.

Skill: `ucv1-two-sample-and-paired-t-test` (statistics). The arithmetic of a first-year two-sample t procedure
(independent samples, unequal or pooled variances) and of the paired t procedure, from summary data. Critical values
are always printed in the question, as a paper with tables would, so the skill is about the standard error, the
statistic and the interval, not about memorising a table.

Tiers (topic `fy1ms_t_test_two_sample`):

* easy   -- the standard error of the difference of two sample means, sqrt(s1^2/n1 + s2^2/n2), to 3 places;
* medium -- one of: the unequal-variance (Welch) t statistic to 2 places; the pooled sample variance to 2 places;
            the paired t statistic from the mean and standard deviation of the differences;
* hard   -- one of: the pooled t statistic; the paired t statistic from a short list of differences (mean and sample
            standard deviation by hand); a 95% confidence-interval bound for the difference of two means, given t*.

Exam (topic `exd_x_fy1ms_t_test_two_sample`): a pooled two-sample test. (a) the pooled variance (2 marks), (b) the
standard error of the difference (1 mark, follow-through from (a)), (c) the t statistic (2 marks, follow-through
from (b)), (d) the conclusion at the 5% level, a one-mark multiple-choice part. The stem prints the critical values
for three neighbouring degrees of freedom, so reading the right one is part of the work.

Every key is built from exact fractions and checked a second way with mpmath at 40 digits from the raw inputs (the
paired list is checked with the statistics module as well). `verify_critical_values()` re-derives the printed table
from the t distribution; the test file calls it, so the table cannot drift.
"""

from __future__ import annotations

import math
import random
import statistics
from fractions import Fraction

import mpmath

from ..core import VerificationError, attempts, check, generator
from ._fy1ms_common import (
    Redraw,
    build_until,
    choice_part,
    clear_of_tie,
    dp_text,
    ft_close,
    num_dist,
    number_part,
)

# ---------------------------------------------------------------------------
# Critical values of t, two-sided 5% (the 0.975 quantile), 3 decimal places, df 4..40.

T_TWO_SIDED_5 = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 31: 2.040, 32: 2.037, 33: 2.035, 34: 2.032, 35: 2.030, 36: 2.028, 37: 2.026, 38: 2.024, 39: 2.023, 40: 2.021}  # fmt: skip


def _t_cdf(t, nu):
    x = mpmath.mpf(nu) / (nu + t * t)
    return 1 - mpmath.betainc(mpmath.mpf(nu) / 2, mpmath.mpf(1) / 2, 0, x, regularized=True) / 2


def verify_critical_values() -> None:
    """Re-derive every printed critical value from the t distribution (mpmath) and compare at 3 decimals."""
    with mpmath.workdps(30):
        for nu, printed in T_TWO_SIDED_5.items():
            exact = mpmath.findroot(lambda t, nu=nu: _t_cdf(t, nu) - mpmath.mpf("0.975"), 2)
            if abs(float(exact) - printed) > 0.0005 + 1e-9:
                raise VerificationError(f"t table wrong at df={nu}: printed {printed}, exact {exact}")


# ---------------------------------------------------------------------------
# Exact helpers


def _root(x: Fraction, digits: int = 30) -> Fraction:
    """sqrt(x) to `digits` places as a Fraction (truncated, so the error is below 10**-digits)."""
    scaled = x.numerator * 10 ** (2 * digits) // x.denominator
    return Fraction(math.isqrt(scaled), 10**digits)


def _mp(f: Fraction):
    return mpmath.mpf(f.numerator) / f.denominator


def _mp_check(actual: Fraction, expected, what: str) -> None:
    """The exact route must agree with the independent mpmath route to a relative 1e-12."""
    a, b = _mp(actual), expected
    check(abs(a - b) <= mpmath.mpf("1e-12") * max(1, abs(b)), f"{what}: the mpmath route disagrees ({a} vs {b})")


def _t3(df: int) -> str:
    """The printed critical value for `df` degrees of freedom, always with three decimal places."""
    return f"{T_TWO_SIDED_5[df]:.3f}"


def _txt(value: Fraction, dp: int) -> str:
    """Data written to its own precision."""
    return dp_text(value, dp)


# ---------------------------------------------------------------------------
# Contexts: two independent groups. Means and spreads are in units of 10^-dp.

_CONTEXTS = [
    dict(
        key="reaction",
        dp=1,
        mean=(2200, 2900),
        s=(180, 420),
        lead="A researcher compares the reaction times of two groups of adults on the same computer test.",
        g1="adults who had a strong coffee beforehand",
        g2="adults who had water instead",
        measure="reaction time",
        unit="milliseconds",
    ),
    dict(
        key="battery",
        dp=1,
        mean=(110, 190),
        s=(14, 36),
        lead="A consumer magazine tests the battery life of two phone models under the same video-playback routine.",
        g1="phones of model A",
        g2="phones of model B",
        measure="battery life",
        unit="hours",
    ),
    dict(
        key="loaf",
        dp=1,
        mean=(6600, 7400),
        s=(80, 210),
        lead="A food inspector weighs loaves from two bakeries that both advertise a 700 g loaf.",
        g1="loaves from the first bakery",
        g2="loaves from the second bakery",
        measure="mass",
        unit="grams",
    ),
    dict(
        key="plant",
        dp=1,
        mean=(240, 410),
        s=(22, 62),
        lead="A botanist grows seedlings under two fertiliser treatments and measures the height of each seedling after four weeks.",
        g1="seedlings given fertiliser A",
        g2="seedlings given fertiliser B",
        measure="height",
        unit="centimetres",
    ),
    dict(
        key="swim",
        dp=1,
        mean=(1180, 1420),
        s=(28, 76),
        lead="A swimming coach compares the 200 m freestyle times of two training squads.",
        g1="swimmers from the morning squad",
        g2="swimmers from the evening squad",
        measure="time",
        unit="seconds",
    ),
    dict(
        key="marks",
        dp=1,
        mean=(560, 760),
        s=(70, 150),
        lead="A university lecturer compares the final exam marks of students from two tutorial streams.",
        g1="students from stream 1",
        g2="students from stream 2",
        measure="mark",
        unit="marks",
    ),
]

# Paired settings: each subject gives two measurements and the differences are analysed.
_PAIRED = [
    dict(
        lead="A physiotherapist records each client's resting heart rate before and after a six-week fitness program.",
        subject="clients",
        what="difference (before − after) in resting heart rate",
        unit="beats per minute",
        lo=-2,
        hi=14,
    ),
    dict(
        lead="A Year 11 teacher gives the same 50-mark quiz to each student before and after a revision app is introduced.",
        subject="students",
        what="difference (after − before) in quiz mark",
        unit="marks",
        lo=-3,
        hi=11,
    ),
    dict(
        lead="A botanist grows pairs of seedlings, one in standard soil and one in compost-enriched soil, and measures both after four weeks.",
        subject="pairs of seedlings",
        what="difference (compost − standard) in height",
        unit="mm",
        lo=-4,
        hi=16,
    ),
]


class _Draw:
    """Two independent samples: sizes, means and sample standard deviations (all exact Fractions)."""

    def __init__(self, ctx, n1, n2, m1, m2, s1, s2):
        self.ctx, self.n1, self.n2, self.m1, self.m2, self.s1, self.s2 = ctx, n1, n2, m1, m2, s1, s2
        self.dp = ctx["dp"]
        self.diff = m1 - m2
        self.v1, self.v2 = s1 * s1, s2 * s2
        self.df = n1 + n2 - 2
        self.sp2 = ((n1 - 1) * self.v1 + (n2 - 1) * self.v2) / self.df
        self.sp = _root(self.sp2)
        self.se_welch = _root(self.v1 / n1 + self.v2 / n2)
        self.se_pooled = _root(self.sp2 * (Fraction(1, n1) + Fraction(1, n2)))

    def mp_check(self) -> None:
        """The same quantities again from the raw inputs in mpmath, sharing no code with the exact route."""
        with mpmath.workdps(40):
            n1, n2 = mpmath.mpf(self.n1), mpmath.mpf(self.n2)
            s1, s2 = _mp(self.s1), _mp(self.s2)
            sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
            _mp_check(self.sp2, sp2, "pooled variance")
            _mp_check(self.se_welch, mpmath.sqrt(s1**2 / n1 + s2**2 / n2), "Welch standard error")
            _mp_check(self.se_pooled, mpmath.sqrt(sp2 * (1 / n1 + 1 / n2)), "pooled standard error")


def _draw(rng: random.Random, ctx, *, t_range, pooled: bool, n_range=(8, 20)) -> _Draw:
    """Two samples whose sample means differ by about t * (standard error), with t drawn from `t_range`."""
    dp = ctx["dp"]
    unit = Fraction(1, 10**dp)
    for _ in attempts(500, "two t-test samples"):
        n1, n2 = rng.randint(*n_range), rng.randint(*n_range)
        if pooled and n1 == n2 and rng.random() < 0.7:
            n2 = n1 + rng.choice((-3, -2, 2, 3, 4))
            if not n_range[0] <= n2 <= n_range[1]:
                continue
        s1 = rng.randint(*ctx["s"]) * unit
        ratio = Fraction(rng.randint(80, 125), 100) if pooled else Fraction(rng.randint(55, 180), 100)
        s2 = round(s1 * ratio / unit) * unit
        if not ctx["s"][0] * unit <= s2 <= ctx["s"][1] * unit * 2 or s2 == s1 and rng.random() < 0.8:
            continue
        m2 = rng.randint(*ctx["mean"]) * unit
        d = _Draw(ctx, n1, n2, m2, m2, s1, s2)
        se = d.se_pooled if pooled else d.se_welch
        t = Fraction(rng.randint(int(t_range[0] * 100), int(t_range[1] * 100)), 100)
        if rng.random() < 0.5:
            t = -t
        m1 = round((m2 + t * se) / unit) * unit
        if m1 == m2 or m1 <= 0:
            continue
        d = _Draw(ctx, n1, n2, m1, m2, s1, s2)
        d.mp_check()
        return d
    raise AssertionError("unreachable")  # pragma: no cover


def _sentences(d: _Draw) -> str:
    c, dp = d.ctx, d.ctx["dp"]
    return (
        f"{c['lead']} In a random sample of {d.n1} {c['g1']}, the mean {c['measure']} was {_txt(d.m1, dp)} {c['unit']} "
        f"and the sample standard deviation was {_txt(d.s1, dp)} {c['unit']}. "
        f"In an independent random sample of {d.n2} {c['g2']}, the mean {c['measure']} was {_txt(d.m2, dp)} {c['unit']} "
        f"and the sample standard deviation was {_txt(d.s2, dp)} {c['unit']}."
    )


def _hypotheses() -> str:
    return "Let μ₁ and μ₂ be the population means for the first and second group, and test H0: μ₁ = μ₂ against H1: μ₁ ≠ μ₂."


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


def _places(dp: int) -> str:
    return f"{dp} decimal place{'s' if dp != 1 else ''}"


# ---------------------------------------------------------------------------
# Candidate wrong answers shared by several tiers


def _welch_se_cands(d: _Draw, dp: int):
    n1, n2, v1, v2 = d.n1, d.n2, d.v1, d.v2
    return [
        num_dist(d.s1 / _root(Fraction(n1)) + d.s2 / _root(Fraction(n2)), dp, "ttest-se-sds-added"),
        num_dist(v1 / n1 + v2 / n2, dp, "ttest-se-no-root"),
        num_dist(d.s1 / _root(Fraction(n1)), dp, "ttest-se-single-sample"),
        num_dist(_root(v1 / (n1 * n1) + v2 / (n2 * n2)), dp, "ttest-se-sd-over-n"),
        num_dist(_root((v1 + v2) / (n1 + n2)), dp, "ttest-se-total-n"),
    ]


def _t_cands(d: _Draw, t: Fraction, dp: int, se: Fraction, *, pooled: bool):
    n1, n2 = d.n1, d.n2
    out = [
        num_dist(-t, dp, "ttest-sign-reversed"),
        num_dist(d.diff / _root(d.v1 + d.v2), dp, "ttest-diff-over-sds"),
        num_dist(d.diff / (d.s1 / _root(Fraction(n1)) + d.s2 / _root(Fraction(n2))), dp, "ttest-se-sds-added"),
        num_dist(d.diff / (d.v1 / n1 + d.v2 / n2), dp, "ttest-se-no-root"),
    ]
    if pooled:
        out += [
            num_dist(d.diff / _root(d.sp2 * Fraction(1, n1 + n2)), dp, "ttest-pooled-se-one-sum"),
            num_dist(d.diff / (d.sp2 * (Fraction(1, n1) + Fraction(1, n2))), dp, "ttest-pooled-se-unrooted"),
            num_dist(d.diff / (d.sp / _root(Fraction(n1))), dp, "ttest-pooled-se-single-n"),
            num_dist(d.diff / (d.sp2 * _root(Fraction(1, n1) + Fraction(1, n2))), dp, "ttest-pooled-variance-as-sd"),
        ]
    else:
        out.append(num_dist(d.diff / d.se_pooled, dp, "ttest-welch-pooled-confused"))
    return out


# ---------------------------------------------------------------------------
# Easy: the standard error of a difference


def _easy(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    dp = 3
    d = _draw(rng, ctx, t_range=(0.4, 3.0), pooled=False)
    se = d.se_welch
    if not clear_of_tie(se, dp):
        raise Redraw("standard error on a rounding tie")
    text = (
        f"{_sentences(d)} Find the standard error of the difference between the two sample means, "
        f"in {d.ctx['unit']}, correct to {_places(dp)}."
    )
    return _question(
        text,
        se,
        dp,
        _welch_se_cands(d, dp),
        [
            f"SE = √(s₁²/n₁ + s₂²/n₂) = √({_txt(d.s1, d.dp)}²/{d.n1} + {_txt(d.s2, d.dp)}²/{d.n2}) = {dp_text(se, dp)}.",
        ],
    )


# ---------------------------------------------------------------------------
# Medium


def _medium_welch_t(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    dp = 2
    d = _draw(rng, ctx, t_range=(0.5, 3.8), pooled=False)
    t = d.diff / d.se_welch
    _mp_check(t, _mp(d.diff) / mpmath.sqrt(_mp(d.v1) / d.n1 + _mp(d.v2) / d.n2), "Welch t")
    if not clear_of_tie(t, dp) or abs(t) < Fraction(1, 5):
        raise Redraw("t statistic unsuitable")
    text = (
        f"{_sentences(d)} {_hypotheses()} Using the unequal-variance (Welch) statistic, "
        f"calculate t = (x̄₁ − x̄₂)/√(s₁²/n₁ + s₂²/n₂), correct to {_places(dp)}."
    )
    return _question(
        text,
        t,
        dp,
        _t_cands(d, t, dp, d.se_welch, pooled=False),
        [
            f"x̄₁ − x̄₂ = {_txt(d.m1, d.dp)} − {_txt(d.m2, d.dp)} = {_txt(d.diff, d.dp)}.",
            f"SE = {dp_text(d.se_welch, 4)}, so t = {dp_text(t, dp)}.",
        ],
    )


def _medium_pooled_variance(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    dp = 2
    d = _draw(rng, ctx, t_range=(0.5, 3.0), pooled=True)
    if not clear_of_tie(d.sp2, dp):
        raise Redraw("pooled variance on a rounding tie")
    n1, n2, v1, v2 = d.n1, d.n2, d.v1, d.v2
    cands = [
        num_dist((v1 + v2) / 2, dp, "ttest-pooled-simple-average"),
        num_dist(((n1 - 1) * d.s1 + (n2 - 1) * d.s2) / d.df, dp, "ttest-pooled-weights-sds"),
        num_dist(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2), dp, "ttest-pooled-divisor-total-n"),
        num_dist((n1 * v1 + n2 * v2) / (n1 + n2), dp, "ttest-pooled-weights-by-n"),
        num_dist(d.sp, dp, "ttest-pooled-sd-as-variance"),
    ]
    text = (
        f"{_sentences(d)} Assume that the two populations have the same variance. "
        f"Find the pooled sample variance, correct to {_places(dp)}."
    )
    return _question(
        text,
        d.sp2,
        dp,
        cands,
        [
            f"s² = ((n₁ − 1)s₁² + (n₂ − 1)s₂²)/(n₁ + n₂ − 2) = ({n1 - 1} × {dp_text(v1, 2 * d.dp)} + {n2 - 1} × {dp_text(v2, 2 * d.dp)})/{d.df} = {dp_text(d.sp2, dp)}.",
        ],
    )


def _paired_summary(rng: random.Random):
    """(setting, n, mean difference, standard deviation of the differences), to one decimal place."""
    setting = rng.choice(_PAIRED)
    n = rng.randint(8, 25)
    sd = Fraction(rng.randint(25, 80), 10)
    t = Fraction(rng.randint(60, 380), 100) * rng.choice((1, 1, -1))
    mean = round(t * sd / _root(Fraction(n)) * 10) / 10
    return setting, n, Fraction(mean).limit_denominator(10), sd


def _medium_paired_t(rng: random.Random) -> dict:
    setting, n, dbar, sd = _paired_summary(rng)
    dp = 2
    if dbar == 0:
        raise Redraw("no mean difference")
    se = sd / _root(Fraction(n))
    t = dbar / se
    _mp_check(t, _mp(dbar) / (_mp(sd) / mpmath.sqrt(n)), "paired t")
    if not clear_of_tie(t, dp) or abs(t) < Fraction(1, 5):
        raise Redraw("t statistic unsuitable")
    cands = [
        num_dist(-t, dp, "ttest-sign-reversed"),
        num_dist(dbar / sd, dp, "ttest-paired-sd-as-se"),
        num_dist(dbar / (sd / n), dp, "ttest-paired-se-over-n"),
        num_dist(dbar / (sd * sd / _root(Fraction(n))), dp, "ttest-paired-variance-as-sd"),
        num_dist(dbar / (sd / _root(Fraction(n - 1))), dp, "ttest-paired-se-root-df"),
    ]
    text = (
        f"{setting['lead']} For each of {n} {setting['subject']}, the {setting['what']}, in {setting['unit']}, is found. "
        f"The {n} differences have mean {_txt(dbar, 1)} and sample standard deviation {_txt(sd, 1)}. "
        f"Let μ_D be the population mean difference and test H0: μ_D = 0 against H1: μ_D ≠ 0. "
        f"Calculate the paired t statistic, correct to {_places(dp)}."
    )
    return _question(
        text,
        t,
        dp,
        cands,
        [
            f"SE = s_D/√n = {_txt(sd, 1)}/√{n} = {dp_text(se, 4)}, so t = {_txt(dbar, 1)}/{dp_text(se, 4)} = {dp_text(t, dp)}."
        ],
    )


def _medium(rng: random.Random) -> dict:
    pick = rng.random()
    if pick < 0.4:
        return _medium_welch_t(rng)
    return _medium_pooled_variance(rng) if pick < 0.7 else _medium_paired_t(rng)


# ---------------------------------------------------------------------------
# Hard


def _hard_pooled_t(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    dp = 2
    d = _draw(rng, ctx, t_range=(0.6, 3.8), pooled=True)
    t = d.diff / d.se_pooled
    _mp_check(
        t,
        _mp(d.diff)
        / mpmath.sqrt(
            ((d.n1 - 1) * _mp(d.v1) + (d.n2 - 1) * _mp(d.v2)) / d.df * (mpmath.mpf(1) / d.n1 + mpmath.mpf(1) / d.n2)
        ),
        "pooled t",
    )
    if not clear_of_tie(t, dp) or abs(t) < Fraction(1, 5):
        raise Redraw("t statistic unsuitable")
    text = (
        f"{_sentences(d)} Assume that the two populations have the same variance. {_hypotheses()} "
        f"Calculate the pooled two-sample t statistic, correct to {_places(dp)}."
    )
    return _question(
        text,
        t,
        dp,
        _t_cands(d, t, dp, d.se_pooled, pooled=True),
        [
            f"Pooled variance = {dp_text(d.sp2, 4)}, so s = {dp_text(d.sp, 4)}.",
            f"SE = s√(1/{d.n1} + 1/{d.n2}) = {dp_text(d.se_pooled, 4)}, and t = {_txt(d.diff, d.dp)}/{dp_text(d.se_pooled, 4)} = {dp_text(t, dp)}.",
        ],
    )


def _hard_paired_list(rng: random.Random) -> dict:
    setting = rng.choice(_PAIRED)
    n = rng.randint(6, 9)
    dp = 2
    diffs = [rng.randint(setting["lo"], setting["hi"]) for _ in range(n)]
    total = sum(diffs)
    mean = Fraction(total, n)
    ss = sum((x - mean) ** 2 for x in diffs)
    if ss == 0 or total == 0 or mean.denominator > 4 and rng.random() < 0.5:
        raise Redraw("degenerate differences")
    var = ss / (n - 1)
    sd = _root(var)
    se = sd / _root(Fraction(n))
    t = mean / se
    stat_sd = statistics.stdev(diffs)
    check(abs(float(sd) - stat_sd) < 1e-9, "the statistics module disagrees on the standard deviation")
    _mp_check(
        t,
        mpmath.mpf(total) / n / (mpmath.sqrt(mpmath.mpf(ss.numerator) / ss.denominator / (n - 1)) / mpmath.sqrt(n)),
        "paired list t",
    )
    if not clear_of_tie(t, dp) or abs(t) < Fraction(1, 5):
        raise Redraw("t statistic unsuitable")
    sd_n = _root(ss / n)
    raw = _root(Fraction(sum(x * x for x in diffs), n - 1))
    cands = [
        num_dist(-t, dp, "ttest-sign-reversed"),
        num_dist(mean / (sd_n / _root(Fraction(n))), dp, "ttest-paired-sd-divisor-n"),
        num_dist(mean / (raw / _root(Fraction(n))), dp, "ttest-paired-uncentred-spread"),
        num_dist(mean / (sd / n), dp, "ttest-paired-se-over-n"),
        num_dist(mean / (var / _root(Fraction(n))), dp, "ttest-paired-variance-as-sd"),
    ]
    listed = ", ".join(str(x) for x in diffs)
    text = (
        f"{setting['lead']} For each of {n} {setting['subject']}, the {setting['what']}, in {setting['unit']}, is: {listed}. "
        f"Let μ_D be the population mean difference and test H0: μ_D = 0 against H1: μ_D ≠ 0. "
        f"Calculate the paired t statistic, correct to {_places(dp)}."
    )
    return _question(
        text,
        t,
        dp,
        cands,
        [
            f"The mean difference is {total}/{n} = {dp_text(mean, 4)} and the sample standard deviation (divisor n − 1 = {n - 1}) is {dp_text(sd, 4)}.",
            f"t = {dp_text(mean, 4)}/({dp_text(sd, 4)}/√{n}) = {dp_text(t, dp)}.",
        ],
    )


def _hard_ci(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    dp = 2
    d = _draw(rng, ctx, t_range=(0.6, 3.0), pooled=True, n_range=(6, 20))
    tstar = Fraction(str(T_TWO_SIDED_5[d.df]))
    margin = tstar * d.se_pooled
    upper = rng.random() < 0.5
    bound = d.diff + margin if upper else d.diff - margin
    if not clear_of_tie(bound, dp):
        raise Redraw("bound on a rounding tie")
    _mp_check(
        bound,
        _mp(d.diff)
        + (1 if upper else -1)
        * _mp(tstar)
        * mpmath.sqrt(
            ((d.n1 - 1) * _mp(d.v1) + (d.n2 - 1) * _mp(d.v2)) / d.df * (mpmath.mpf(1) / d.n1 + mpmath.mpf(1) / d.n2)
        ),
        "confidence bound",
    )
    sign = 1 if upper else -1
    other = d.diff - sign * margin
    cands = [
        num_dist(other, dp, "ttest-ci-other-bound"),
        num_dist(margin, dp, "ttest-ci-margin-as-bound"),
        num_dist(d.diff + sign * Fraction(196, 100) * d.se_pooled, dp, "ttest-ci-normal-critical"),
        num_dist(-bound, dp, "ttest-sign-reversed"),
        num_dist(
            d.diff + sign * tstar * (d.s1 / _root(Fraction(d.n1)) + d.s2 / _root(Fraction(d.n2))),
            dp,
            "ttest-se-sds-added",
        ),
    ]
    which = "upper" if upper else "lower"
    text = (
        f"{_sentences(d)} Assume that the two populations have the same variance. "
        f"The critical value of t for {d.df} degrees of freedom is {_t3(d.df)} (two-sided, 5% level). "
        f"Find the {which} limit of the 95% confidence interval for μ₁ − μ₂, correct to {_places(dp)}."
    )
    return _question(
        text,
        bound,
        dp,
        cands,
        [
            f"Pooled SE = {dp_text(d.se_pooled, 4)}, margin = {tstar} × {dp_text(d.se_pooled, 4)} = {dp_text(margin, 4)}.",
            f"The {which} limit is {_txt(d.diff, d.dp)} {'+' if upper else '−'} {dp_text(margin, 4)} = {dp_text(bound, dp)}.",
        ],
    )


@generator(
    "fy1ms_t_test_two_sample",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1ms_t_test_two_sample(rng: random.Random, tier: str = "easy") -> dict:
    """Two-sample and paired t procedures: a standard error (easy), a statistic or pooled variance (medium), a pooled or paired statistic or an interval limit (hard)."""
    if tier == "easy":
        return build_until(rng, _easy, "an easy two-sample standard error")
    if tier == "medium":
        return build_until(rng, _medium, "a medium two-sample question")
    if tier == "hard":

        def hard(r: random.Random) -> dict:
            pick = r.random()
            return _hard_pooled_t(r) if pick < 0.4 else _hard_paired_list(r) if pick < 0.7 else _hard_ci(r)

        return build_until(rng, hard, "a hard two-sample question")
    raise AssertionError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain: a pooled two-sample test


@generator("exd_x_fy1ms_t_test_two_sample", level=19, presets={"exam": dict(tier="exam")})
def exd_x_fy1ms_t_test_two_sample(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: pooled variance, standard error, t statistic, conclusion (6 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_until(rng, _exam_chain, "a pooled t-test exam question")


def _exam_chain(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    reject = rng.random() < 0.5
    # the draw is made for a target t on the pooled scale, then the critical value is read for its degrees of freedom
    probe = _draw(rng, ctx, t_range=(0.4, 0.5), pooled=True, n_range=(7, 18))
    tcrit = Fraction(str(T_TWO_SIDED_5[probe.df]))
    lo, hi = (float(tcrit) + 0.5, float(tcrit) + 2.5) if reject else (0.4, float(tcrit) - 0.5)
    d = _redraw_at(rng, ctx, probe, lo, hi)
    dp_var, dp_se, dp_t = 2, 3, 2
    t = d.diff / d.se_pooled
    for value, dp in ((d.sp2, dp_var), (d.se_pooled, dp_se), (t, dp_t)):
        if not clear_of_tie(value, dp):
            raise Redraw("a value sits on a rounding tie")
    if abs(abs(t) - tcrit) < Fraction(3, 10):
        raise Redraw("t too close to the critical value")
    a_text, b_text, c_text = dp_text(d.sp2, dp_var), dp_text(d.se_pooled, dp_se), dp_text(t, dp_t)
    k = Fraction(1, d.n1) + Fraction(1, d.n2)
    ft_close(_root(Fraction(a_text) * k), b_text, dp_se)
    ft_close(d.diff / Fraction(b_text), c_text, dp_t)
    n1, n2, v1, v2 = d.n1, d.n2, d.v1, d.v2
    a_cands = [
        num_dist((v1 + v2) / 2, dp_var, "ttest-pooled-simple-average"),
        num_dist(((n1 - 1) * d.s1 + (n2 - 1) * d.s2) / d.df, dp_var, "ttest-pooled-weights-sds"),
        num_dist(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2), dp_var, "ttest-pooled-divisor-total-n"),
        num_dist((n1 * v1 + n2 * v2) / (n1 + n2), dp_var, "ttest-pooled-weights-by-n"),
        num_dist(d.sp, dp_var, "ttest-pooled-sd-as-variance"),
    ]
    b_cands = [
        num_dist(_root(d.sp2 * Fraction(1, n1 + n2)), dp_se, "ttest-pooled-se-one-sum"),
        num_dist(d.sp2 * k, dp_se, "ttest-pooled-se-unrooted"),
        num_dist(d.sp / _root(Fraction(n1)), dp_se, "ttest-pooled-se-single-n"),
        num_dist(d.sp2 * _root(k), dp_se, "ttest-pooled-variance-as-sd"),
        num_dist(d.s1 / _root(Fraction(n1)) + d.s2 / _root(Fraction(n2)), dp_se, "ttest-se-sds-added"),
    ]
    c_cands = [
        num_dist(-t, dp_t, "ttest-sign-reversed"),
        num_dist(d.diff / d.sp, dp_t, "ttest-pooled-sd-as-standard-error"),
        num_dist(d.diff / d.sp2, dp_t, "ttest-pooled-variance-as-sd"),
        num_dist(d.diff / _root(d.v1 + d.v2), dp_t, "ttest-diff-over-sds"),
        num_dist(d.diff / (d.sp / _root(Fraction(n1))), dp_t, "ttest-pooled-se-single-n"),
    ]
    crit_row = ", ".join(f"{nu} degrees of freedom: {_t3(nu)}" for nu in (d.df - 1, d.df, d.df + 1))
    stem = (
        f"{_sentences(d)} Assume that the two populations are normal with the same variance. "
        f"Test H0: μ₁ = μ₂ against H1: μ₁ ≠ μ₂ at the 5% level of significance. "
        f"Critical values of t (two-sided, 5%): {crit_row}."
    )
    ratio = "greater than" if reject else "less than"
    right = (
        "Reject H0, because |t| is greater than the critical value."
        if reject
        else "Do not reject H0, because |t| is less than the critical value."
    )
    reversed_ = (
        "Do not reject H0, because |t| is greater than the critical value."
        if reject
        else "Reject H0, because |t| is less than the critical value."
    )
    wrongs = [
        (reversed_, "ttest-conclusion-decision-reversed"),
        ("Reject H0, because t is not exactly 0, so the two sample means differ.", "ttest-conclusion-means-differ"),
    ]
    check(ratio in right, "conclusion wording")
    return dict(
        text=stem,
        parts=[
            number_part(
                "Find the pooled sample variance, correct to 2 decimal places.",
                d.sp2,
                dp_var,
                a_cands,
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"s² = ({n1 - 1} × {dp_text(v1, 2 * d.dp)} + {n2 - 1} × {dp_text(v2, 2 * d.dp)})/{d.df} = {a_text}."
                ],
            ),
            number_part(
                "Hence find the standard error of the difference between the sample means, correct to 3 decimal places.",
                d.se_pooled,
                dp_se,
                b_cands,
                marks=1,
                follow_through=f"sqrt(part_a*({k.numerator}/{k.denominator}))",
                sensible_range=(0, None),
                working=[f"SE = √(s²(1/{n1} + 1/{n2})) = √({a_text} × {k.numerator}/{k.denominator}) = {b_text}."],
            ),
            number_part(
                "Calculate the value of the t statistic, correct to 2 decimal places.",
                t,
                dp_t,
                c_cands,
                marks=2,
                follow_through=f"({d.diff.numerator}/{d.diff.denominator})/part_b",
                working=[f"t = (x̄₁ − x̄₂)/SE = {dp_text(d.diff, d.dp)}/{b_text} = {c_text}."],
            ),
            choice_part(
                rng,
                f"Using the critical value for {d.df} degrees of freedom ({_t3(d.df)}), choose the conclusion of the test.",
                right,
                wrongs,
                marks=1,
                working=[f"|t| = {dp_text(abs(t), 2)} is {ratio} {_t3(d.df)}."],
            ),
        ],
    )


def _redraw_at(rng: random.Random, ctx, probe: _Draw, lo: float, hi: float) -> _Draw:
    """The probe's samples with the first mean moved so that |t| (pooled) falls in [lo, hi]."""
    unit = Fraction(1, 10 ** ctx["dp"])
    t = Fraction(rng.randint(int(lo * 100), int(hi * 100)), 100)
    if rng.random() < 0.5:
        t = -t
    m1 = round((probe.m2 + t * probe.se_pooled) / unit) * unit
    if m1 == probe.m2 or m1 <= 0:
        raise Redraw("unusable first mean")
    d = _Draw(ctx, probe.n1, probe.n2, m1, probe.m2, probe.s1, probe.s2)
    d.mp_check()
    return d
