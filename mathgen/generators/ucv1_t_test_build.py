"""Generates `ucv1_one_sample_t_test` questions and the exam chain `exd_x_uc_t_test`.

``ucv1-one-sample-t-test`` (``statistics``): the arithmetic of a first-year one-sample (and paired) t procedure
from summary data, never the reading of a p-value and never a decision on its own. Critical values are always
GIVEN in the question, as a first-year paper with tables does, so the skill is about the standard error, the
statistic and what they imply, not about memorising a table.

Tiers (topic ``ucv1_one_sample_t_test``):

* easy   -- the standard error s/sqrt(n) from n and s, to 3 decimal places (the sample mean and the hypothesised
  mean are in the stem and play no part);
* medium -- the t statistic (x-bar - mu0)/(s/sqrt(n)) to 2 decimal places, or to 3 decimal places when the stem
  gives the sample *variance*, so the standard deviation must be found first;
* hard   -- one of: a 95% confidence-interval bound given the critical value t*; the paired t statistic from a
  short list of differences (mean and sample standard deviation by hand); the sample mean that produced a stated
  t statistic (the formula run backwards).

The exam tier (topic ``exd_x_uc_t_test``) is a four-part chain: (a) the standard error, (b) the t statistic
(follow-through from (a)), (c) the smallest (or largest) sample mean that would have been significant with the
same n and s (follow-through from (a)), (d) the one-mark conclusion of the test as a multiple-choice part. The
stem prints the critical values for three neighbouring degrees of freedom, so reading the right row is part of
the work.

Every key is recomputed a second way (mpmath at 40 digits from the raw inputs) and `check_equivalent`ed.
`verify_critical_values()` re-derives the hard-coded tables from the t distribution with mpmath; the test file
calls it, so the tables cannot drift.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import mpmath

from ..answers import Distractor
from ..core import (
    VerificationError,
    attempts,
    check,
    check_equivalent,
    generator,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

# ---------------------------------------------------------------------------
# Critical values of t, 3 decimal places, df 4..30 (a standard printed table).
# Two-sided 5% (t_0.975), two-sided 1% (t_0.995), one-sided 5% (t_0.95).

T_TWO_SIDED_5 = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}  # fmt: skip
T_TWO_SIDED_1 = {4: 4.604, 5: 4.032, 6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169, 11: 3.106, 12: 3.055, 13: 3.012, 14: 2.977, 15: 2.947, 16: 2.921, 17: 2.898, 18: 2.878, 19: 2.861, 20: 2.845, 21: 2.831, 22: 2.819, 23: 2.807, 24: 2.797, 25: 2.787, 26: 2.779, 27: 2.771, 28: 2.763, 29: 2.756, 30: 2.750}  # fmt: skip
T_ONE_SIDED_5 = {4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725, 21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706, 27: 1.703, 28: 1.701, 29: 1.699, 30: 1.697}  # fmt: skip

#: (table, upper-tail probability, label printed in the stem)
_TABLES = {
    "two5": (T_TWO_SIDED_5, "0.975", "two-sided 5%"),
    "two1": (T_TWO_SIDED_1, "0.995", "two-sided 1%"),
    "one5": (T_ONE_SIDED_5, "0.95", "one-sided 5%"),
}


def _t_cdf(t, nu):
    x = mpmath.mpf(nu) / (nu + t * t)
    return 1 - mpmath.betainc(mpmath.mpf(nu) / 2, mpmath.mpf(1) / 2, 0, x, regularized=True) / 2


def verify_critical_values() -> None:
    """Re-derive every hard-coded critical value from the t distribution (mpmath) and compare at 3 decimals."""
    with mpmath.workdps(30):
        for table, prob, _label in _TABLES.values():
            for nu, printed in table.items():
                exact = mpmath.findroot(lambda t, nu=nu, p=mpmath.mpf(prob): _t_cdf(t, nu) - p, 2)
                if abs(float(exact) - printed) > 0.0005 + 1e-9:
                    raise VerificationError(f"t table wrong at df={nu}, p={prob}: printed {printed}, exact {exact}")


# ---------------------------------------------------------------------------
# Exact helpers


def _root(x: Fraction, digits: int = 30) -> Fraction:
    """sqrt(x) to `digits` places, as a Fraction (truncated: error below 10**-digits)."""
    scaled = x.numerator * 10 ** (2 * digits) // x.denominator
    return Fraction(math.isqrt(scaled), 10**digits)


def _places(dp: int) -> str:
    return f"{dp} decimal place{'s' if dp != 1 else ''}"


def _dp_text(value: Fraction, dp: int) -> str:
    return stated_precision_string(round_half_up(value, dp), dp)


def _unambiguous(value: Fraction, dp: int) -> bool:
    """True when `value` is not close to a rounding tie at `dp` places (so a learner rounding an intermediate
    value still lands on the same last digit)."""
    scaled = abs(value) * 10**dp
    frac = scaled - math.floor(scaled)
    return abs(frac - Fraction(1, 2)) > Fraction(1, 10)


def _mp_check(actual: Fraction, expected_mp, what: str) -> None:
    """The exact Fraction route must agree with the independent mpmath route to 1e-9 (relative)."""
    a, b = float(actual), float(expected_mp)
    check(abs(a - b) <= 1e-9 * max(1.0, abs(b)), f"{what}: independent mpmath route disagrees ({a} vs {b})")


def _num_dist(value: Fraction, dp: int, tag: str) -> Distractor:
    text = _dp_text(value, dp)
    return Distractor(text, tag, float(text))


def _pick(answer_text: str, cands: list[Distractor], dp: int, n: int = 4) -> list[Distractor]:
    """Distractors that clear the typed grader by a full unit in the last place, at most `n`."""
    return pick_distractors(answer_text, cands, n, tolerance=Fraction(1, 10**dp))


# ---------------------------------------------------------------------------
# Contexts (all plausible Australian settings; means and spreads fit the quantity)

_CONTEXTS = [
    dict(
        key="milk",
        dp=1,
        mu0=["600", "750", "1000"],
        s_lo=30,
        s_hi=90,
        n_lo=9,
        n_hi=30,
        lead="A dairy says its cartons are filled with a mean of {mu0} mL.",
        obj="cartons",
        meanphrase="a mean fill of",
        unit="mL",
        unit_sq="mL²",
        thing="mean fill volume of a carton, in mL",
    ),
    dict(
        key="bus",
        dp=1,
        mu0=["28", "32", "35", "40", "45"],
        s_lo=35,
        s_hi=95,
        n_lo=9,
        n_hi=30,
        lead="The timetable for a Brisbane bus route says the journey from Chermside to the city takes a mean of {mu0} minutes.",
        obj="trips",
        meanphrase="a mean journey time of",
        unit="minutes",
        unit_sq="minutes²",
        thing="mean journey time, in minutes",
    ),
    dict(
        key="battery",
        dp=1,
        mu0=["12", "14", "16", "18", "20"],
        s_lo=8,
        s_hi=24,
        n_lo=9,
        n_hi=30,
        lead="A manufacturer states that its phone battery lasts a mean of {mu0} hours of video playback.",
        obj="batteries",
        meanphrase="a mean life of",
        unit="hours",
        unit_sq="hours²",
        thing="mean battery life, in hours",
    ),
    dict(
        key="loaf",
        dp=1,
        mu0=["680", "700", "720", "750"],
        s_lo=60,
        s_hi=150,
        n_lo=9,
        n_hi=30,
        lead="A Melbourne bakery says its sourdough loaves have a mean mass of {mu0} g.",
        obj="loaves",
        meanphrase="a mean mass of",
        unit="g",
        unit_sq="g²",
        thing="mean mass of a loaf, in g",
    ),
    dict(
        key="swim",
        dp=2,
        mu0=["28", "30", "32", "34", "36"],
        s_lo=65,
        s_hi=185,
        n_lo=9,
        n_hi=30,
        lead="A swimming coach believes her squad's mean time for 50 m freestyle is {mu0} seconds.",
        obj="swims",
        meanphrase="a mean time of",
        unit="seconds",
        unit_sq="seconds²",
        thing="mean 50 m freestyle time, in seconds",
    ),
    dict(
        key="solar",
        dp=1,
        mu0=["16.5", "18.0", "20.5", "22.0"],
        s_lo=11,
        s_hi=34,
        n_lo=9,
        n_hi=30,
        lead="A solar installer assumes that a panel on a Perth roof produces a mean of {mu0} kWh per day.",
        obj="panels",
        meanphrase="a mean daily output of",
        unit="kWh",
        unit_sq="kWh²",
        thing="mean daily output of a panel, in kWh",
    ),
]

# Paired contexts: integer differences so the list is short and the arithmetic is by hand.
_PAIRED = [
    dict(
        lead="A physiotherapist records the fall in resting heart rate for {n} clients after a six-week fitness program.",
        what="differences (before − after) in resting heart rate (beats per minute)",
        lo=-2,
        hi=14,
    ),
    dict(
        lead="A Year 11 teacher gives {n} students the same 50-mark quiz before and after using a revision app.",
        what="differences (after − before) in marks",
        lo=-3,
        hi=11,
    ),
    dict(
        lead="A botanist measures {n} paired seedlings, one grown in standard soil and one in compost-enriched soil.",
        what="differences in height after four weeks (compost − standard) in mm",
        lo=-4,
        hi=16,
    ),
]


def _fmt_data(ctx, value: Fraction) -> str:
    return stated_precision_string(value, ctx["dp"])


def _draw_summary(rng: random.Random, ctx, *, t_range, n_choices=None, signed=True):
    """(n, mu0, s, xbar) for a context, with xbar = mu0 + t_target * s / sqrt(n) rounded to the data's precision."""
    dp = ctx["dp"]
    for _ in attempts(500, "a t-test summary"):
        n = rng.choice(n_choices) if n_choices else rng.randint(ctx["n_lo"], ctx["n_hi"])
        mu0 = Fraction(rng.choice(ctx["mu0"]))
        s = Fraction(rng.randint(ctx["s_lo"], ctx["s_hi"]), 10**dp)
        se = s / _root(Fraction(n))
        t_target = Fraction(rng.randint(int(t_range[0] * 100), int(t_range[1] * 100)), 100)
        if signed and rng.random() < 0.45:
            t_target = -t_target
        xbar = round_half_up(mu0 + t_target * se, dp)
        if xbar == mu0 or xbar <= 0:
            continue
        return n, mu0, s, xbar
    raise VerificationError("unreachable")  # pragma: no cover


def _mu_text(ctx, mu0: Fraction) -> str:
    return str(int(mu0)) if mu0 == int(mu0) else _fmt_data(ctx, mu0)


def _lead_text(ctx, mu0: Fraction) -> str:
    return ctx["lead"].format(mu0=_mu_text(ctx, mu0))


def _sentence_full(ctx, n, mu0, xbar, s, *, variance=False) -> str:
    lead = _lead_text(ctx, mu0)
    if variance:
        spread = f"a sample variance of {_dp_text(s * s, ctx['dp'] * 2)} {ctx['unit_sq']}"
    else:
        spread = f"a sample standard deviation of {_fmt_data(ctx, s)} {ctx['unit']}"
    return f"{lead} A random sample of {n} {ctx['obj']} has {ctx['meanphrase']} {_fmt_data(ctx, xbar)} {ctx['unit']} and {spread}."


# ---------------------------------------------------------------------------
# Easy: standard error


def _standard_error_question(rng, ctx) -> dict:
    dp = 3
    for _ in attempts(500, "a standard error with unambiguous rounding"):
        n, mu0, s, xbar = _draw_summary(rng, ctx, t_range=(0.4, 3.2))
        se = s / _root(Fraction(n))
        if _unambiguous(se, dp) and se >= Fraction(1, 20):
            break
    mp_se = mpmath.mpf(str(float(s))) / mpmath.sqrt(n)
    _mp_check(se, mp_se, "standard error")
    answer_text = _dp_text(se, dp)
    root_n1 = _root(Fraction(n - 1))
    cands = [
        _num_dist(s / n, dp, "se-divides-by-n"),
        _num_dist(s / root_n1, dp, "se-uses-n-minus-1"),
        _num_dist(s * _root(Fraction(n)), dp, "se-multiplies-by-root-n"),
        _num_dist(s * s / _root(Fraction(n)), dp, "variance-used-as-sd"),
    ]
    distractors = _pick(answer_text, cands, dp)
    check(len(distractors) >= 2, "too few standard-error distractors")
    text = (
        f"{_sentence_full(ctx, n, mu0, xbar, s)} "
        "Find the standard error of the sample mean, correct to 3 decimal places."
    )
    return dict(
        text=text,
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[f"SE = s/√n = {_fmt_data(ctx, s)}/√{n} = {answer_text}."],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# Medium: the t statistic


def _t_statistic_question(rng, ctx, *, variance: bool) -> dict:
    dp = 3 if variance else 2
    for _ in attempts(500, "a t statistic with unambiguous rounding"):
        n, mu0, s, xbar = _draw_summary(rng, ctx, t_range=(0.5, 3.6))
        diff = xbar - mu0
        se = s / _root(Fraction(n))
        t = diff / se
        if _unambiguous(t, dp) and abs(t) > Fraction(1, 5):
            break
    mp_t = (mpmath.mpf(str(float(xbar))) - mpmath.mpf(str(float(mu0)))) / (mpmath.mpf(str(float(s))) / mpmath.sqrt(n))
    _mp_check(t, mp_t, "t statistic")
    answer_text = _dp_text(t, dp)
    cands: list[Distractor] = []
    if variance:
        var = s * s
        cands = [
            _num_dist(diff / (var / _root(Fraction(n))), dp, "variance-used-as-sd"),
            _num_dist(diff / _root(var), dp, "sd-used-as-standard-error"),
            _num_dist(-t, dp, "t-sign-reversed"),
            _num_dist(diff / (s / Fraction(n)), dp, "se-divides-by-n"),
        ]
    else:
        cands = [
            _num_dist(diff / s, dp, "sd-used-as-standard-error"),
            _num_dist(diff / (s / Fraction(n)), dp, "se-divides-by-n"),
            _num_dist(-t, dp, "t-sign-reversed"),
            _num_dist(diff / (s / _root(Fraction(n - 1))), dp, "se-uses-n-minus-1"),
        ]
    distractors = _pick(answer_text, cands, dp)
    check(len(distractors) >= 2, "too few t-statistic distractors")
    mu_t = _mu_text(ctx, mu0)
    text = (
        f"{_sentence_full(ctx, n, mu0, xbar, s, variance=variance)} "
        f"Test H0: μ = {mu_t} against H1: μ ≠ {mu_t}. "
        f"Calculate the value of the t statistic, correct to {dp} decimal places."
    )
    working = [
        f"x̄ - μ0 = {_fmt_data(ctx, xbar)} - {mu_t} = {_fmt_data(ctx, diff)}.",
        f"SE = s/√n = {_dp_text(se, 4)}, so t = {answer_text}.",
    ]
    return dict(
        text=text,
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=working,
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# Hard: confidence-interval bound / paired t / the sample mean behind a t statistic


def _ci_bound_question(rng, ctx) -> dict:
    dp = 2 if ctx["dp"] >= 1 else 1
    for _ in attempts(500, "a CI bound with unambiguous rounding"):
        n, mu0, s, xbar = _draw_summary(rng, ctx, t_range=(0.5, 2.5), n_choices=list(range(5, 30)))
        tstar = Fraction(str(T_TWO_SIDED_5[n - 1]))
        se = s / _root(Fraction(n))
        margin = tstar * se
        upper = rng.random() < 0.5
        bound = xbar + margin if upper else xbar - margin
        if _unambiguous(bound, dp):
            break
    mp_bound = mpmath.mpf(str(float(xbar))) + (1 if upper else -1) * mpmath.mpf(str(float(tstar))) * mpmath.mpf(
        str(float(s))
    ) / mpmath.sqrt(n)
    _mp_check(bound, mp_bound, "confidence bound")
    answer_text = _dp_text(bound, dp)
    other = xbar - margin if upper else xbar + margin
    sign = 1 if upper else -1
    cands = [
        _num_dist(other, dp, "ci-wrong-bound"),
        _num_dist(margin, dp, "ci-margin-as-bound"),
        _num_dist(xbar + sign * tstar * s, dp, "sd-used-as-standard-error"),
        _num_dist(xbar + sign * tstar * s / n, dp, "se-divides-by-n"),
    ]
    distractors = _pick(answer_text, cands, dp)
    check(len(distractors) >= 2, "too few CI distractors")
    which = "upper" if upper else "lower"
    text = (
        f"{_sentence_full(ctx, n, mu0, xbar, s)} "
        f"For a 95% confidence interval for the true mean, the critical value of t with {n - 1} degrees of freedom "
        f"is t* = {float(tstar):.3f}. Find the {which} limit of the 95% confidence interval for μ, "
        f"correct to {dp} decimal places."
    )
    return dict(
        text=text,
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[
            f"x̄ {'+' if upper else '-'} t*·s/√n = {_fmt_data(ctx, xbar)} {'+' if upper else '-'} {float(tstar):.3f} × {_fmt_data(ctx, s)}/√{n} = {answer_text}."
        ],
        distractors=distractors,
    )


def _paired_question(rng) -> dict:
    dp = 2
    for _ in attempts(1000, "a paired sample"):
        ctx = rng.choice(_PAIRED)
        n = rng.randint(6, 8)
        diffs = [rng.randint(ctx["lo"], ctx["hi"]) for _ in range(n)]
        total = sum(diffs)
        mean = Fraction(total, n)
        ss = sum((Fraction(d) - mean) ** 2 for d in diffs)
        if ss == 0 or len(set(diffs)) < 4:
            continue
        s = _root(ss / (n - 1))
        t = mean / (s / _root(Fraction(n)))
        if _unambiguous(t, dp) and abs(t) > Fraction(1, 2) and mean != 0:
            break
    # second route: mpmath from the raw differences
    mp_d = [mpmath.mpf(d) for d in diffs]
    mp_mean = sum(mp_d) / n
    mp_s = mpmath.sqrt(sum((d - mp_mean) ** 2 for d in mp_d) / (n - 1))
    _mp_check(t, mp_mean / (mp_s / mpmath.sqrt(n)), "paired t")
    answer_text = _dp_text(t, dp)
    s_pop = _root(ss / n)
    cands = [
        _num_dist(mean / (s_pop / _root(Fraction(n))), dp, "sample-sd-divides-by-n"),
        _num_dist(mean / s, dp, "sd-used-as-standard-error"),
        _num_dist(mean / (s / n), dp, "se-divides-by-n"),
        _num_dist(-t, dp, "t-sign-reversed"),
    ]
    distractors = _pick(answer_text, cands, dp)
    check(len(distractors) >= 2, "too few paired-t distractors")
    listed = ", ".join(str(d) for d in diffs)
    text = (
        f"{ctx['lead'].format(n=n)} The {ctx['what']} are: {listed}. "
        "Treating these as a random sample of differences, calculate the t statistic for testing H0: μ_d = 0, "
        f"correct to {dp} decimal places."
    )
    return dict(
        text=text,
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[f"Mean difference = {total}/{n}; s_d = {_dp_text(s, 3)}; t = mean/(s_d/√{n}) = {answer_text}."],
        distractors=distractors,
    )


def _inverse_question(rng, ctx) -> dict:
    dp = ctx["dp"]
    for _ in attempts(500, "an inverse t-statistic question"):
        n = rng.randint(ctx["n_lo"], ctx["n_hi"])
        mu0 = Fraction(rng.choice(ctx["mu0"]))
        s = Fraction(rng.randint(ctx["s_lo"], ctx["s_hi"]), 10**dp)
        t = Fraction(rng.randint(60, 340), 100) * (1 if rng.random() < 0.55 else -1)
        se = s / _root(Fraction(n))
        xbar = mu0 + t * se
        if _unambiguous(xbar, dp) and xbar > 0:
            break
    mp_x = mpmath.mpf(str(float(mu0))) + mpmath.mpf(str(float(t))) * mpmath.mpf(str(float(s))) / mpmath.sqrt(n)
    _mp_check(xbar, mp_x, "inverse sample mean")
    answer_text = _dp_text(xbar, dp)
    cands = [
        _num_dist(mu0 + t * s / n, dp, "se-divides-by-n"),
        _num_dist(mu0 - t * se, dp, "t-sign-reversed"),
        _num_dist(t * se, dp, "inverse-forgets-null-mean"),
        _num_dist(mu0 + t * s, dp, "sd-used-as-standard-error"),
    ]
    distractors = _pick(answer_text, cands, dp)
    check(len(distractors) >= 2, "too few inverse-t distractors")
    mu_t = _mu_text(ctx, mu0)
    t_text = f"{float(t):.2f}"
    text = (
        f"{_lead_text(ctx, mu0)} A one-sample t test of H0: μ = {mu_t} used a random sample of {n} {ctx['obj']} "
        f"with a sample standard deviation of {_fmt_data(ctx, s)} {ctx['unit']} and gave t = {t_text}. "
        f"Find the sample mean, in {ctx['unit']}, correct to {_places(dp)}."
    )
    return dict(
        text=text,
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[f"x̄ = μ0 + t·s/√n = {mu_t} + ({t_text})×{_fmt_data(ctx, s)}/√{n} = {answer_text}."],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# The topic for the three single-answer tiers


@generator(
    "ucv1_one_sample_t_test",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_one_sample_t_test(rng: random.Random, tier: str = "easy") -> dict:
    """One-sample / paired t procedure arithmetic: SE (easy), t statistic (medium), CI bound, paired t or inverse (hard)."""
    ctx = rng.choice(_CONTEXTS)
    if tier == "easy":
        return _standard_error_question(rng, ctx)
    if tier == "medium":
        return _t_statistic_question(rng, ctx, variance=rng.random() < 0.4)
    if tier == "hard":
        pick = rng.random()
        if pick < 0.4:
            return _ci_bound_question(rng, ctx)
        if pick < 0.75:
            return _paired_question(rng)
        return _inverse_question(rng, ctx)
    raise VerificationError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain

_SIDES = [
    # (table key, relation shown, direction of the rejection region, level shown)
    ("two5", "≠", 0, "5%"),
    ("two5", "≠", 0, "5%"),
    ("two1", "≠", 0, "1%"),
    ("one5", ">", 1, "5%"),
    ("one5", "<", -1, "5%"),
]

_DECISION = {
    "≠": ("differs from", "differs from"),
    ">": ("is greater than", "is greater than"),
    "<": ("is less than", "is less than"),
}


def _decision_part(rng: random.Random, reject: bool, level: str, rel: str, mu_t: str, unit: str) -> dict:
    """The one-mark conclusion as a numbered list of statements; the reply is the number of the correct one.

    Typed (a number), so it is graded by the number grader like every other part; the correct statement's position
    is shuffled with `rng`."""
    claim = {"≠": "differs from", ">": "is greater than", "<": "is less than"}[rel]
    reject_text = f"Reject H0: there is evidence at the {level} level that the true mean {claim} {mu_t} {unit}."
    keep_text = (
        f"Do not reject H0: there is not enough evidence at the {level} level that the true mean {claim} {mu_t} {unit}."
    )
    proved_text = f"Do not reject H0: the sample shows that the true mean is exactly {mu_t} {unit}."
    any_text = f"Reject H0: the sample mean is not equal to {mu_t} {unit}, so the true mean {claim} {mu_t} {unit}."
    if reject:
        right, wrong = (
            reject_text,
            [(keep_text, "decision-compared-wrong-way"), (proved_text, "null-accepted-as-proved")],
        )
    else:
        right, wrong = (
            keep_text,
            [(reject_text, "decision-compared-wrong-way"), (any_text, "any-difference-is-significant")],
        )
    order = [right] + [w[0] for w in wrong]
    rng.shuffle(order)
    position = {text: i + 1 for i, text in enumerate(order)}
    listed = " ".join(f"{i + 1}. {text}" for i, text in enumerate(order))
    return dict(
        prompt=f"State the conclusion of the original test. Enter the number of the correct statement. {listed}",
        answer_kind="number",
        answer=position[right],
        answer_text=str(position[right]),
        marks=1,
        working=[f"The correct statement is number {position[right]}: {right}"],
        distractors=[Distractor(str(position[t]), tag, position[t]) for t, tag in wrong],
    )


def _exam_draw(rng: random.Random):
    for _ in attempts(1000, "a t-test exam chain"):
        ctx = rng.choice(_CONTEXTS)
        key, rel, direction, level = rng.choice(_SIDES)
        table = _TABLES[key][0]
        dp_data = ctx["dp"]
        n = rng.randint(max(ctx["n_lo"], 7), 28)
        df = n - 1
        crit = Fraction(str(table[df]))
        mu0 = Fraction(rng.choice(ctx["mu0"]))
        s = Fraction(rng.randint(ctx["s_lo"], ctx["s_hi"]), 10**dp_data)
        se = s / _root(Fraction(n))
        if not _unambiguous(se, 3):
            continue
        # aim the statistic either side of the critical value, clear of it
        reject = rng.random() < 0.5
        mag = crit * (Fraction(rng.randint(112, 175), 100) if reject else Fraction(rng.randint(45, 88), 100))
        if direction == 0:
            signed = mag * (1 if rng.random() < 0.5 else -1)
        elif direction == 1:
            signed = mag if reject else mag * (1 if rng.random() < 0.7 else -1)
        else:
            signed = -mag if reject else mag * (-1 if rng.random() < 0.7 else 1)
        xbar = round_half_up(mu0 + signed * se, dp_data)
        if xbar == mu0 or xbar <= 0:
            continue
        diff = xbar - mu0
        t = diff / se
        # the decision must be what the rounded numbers say, with clearance from the critical value
        is_reject = abs(t) > crit if direction == 0 else (t > crit if direction == 1 else t < -crit)
        if abs(abs(t) - crit) < Fraction(15, 100):
            continue
        se3 = round_half_up(se, 3)
        t_ft = diff / se3
        if not (_unambiguous(t, 2) and abs(t) > Fraction(1, 5)) or abs(t - t_ft) > Fraction(8, 1000):
            continue
        edge = mu0 + (crit * se if direction >= 0 else -crit * se)
        if not _unambiguous(edge, 2):
            continue
        return ctx, key, rel, direction, level, table, n, df, crit, mu0, s, se, xbar, diff, t, is_reject, edge
    raise VerificationError("unreachable")  # pragma: no cover


@generator("exd_x_uc_t_test", level=19, presets={"exam": dict(tier="exam")})
def exd_x_uc_t_test(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: standard error, t statistic, boundary sample mean, conclusion (6 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx, key, rel, direction, level, table, n, df, crit, mu0, s, se, xbar, diff, t, is_reject, edge = _exam_draw(rng)
    dp_data = ctx["dp"]
    mu_t = _mu_text(ctx, mu0)
    unit = ctx["unit"]

    # independent route for the whole chain, from the printed numbers, with mpmath
    mp_se = mpmath.mpf(str(float(s))) / mpmath.sqrt(n)
    mp_t = (mpmath.mpf(str(float(xbar))) - mpmath.mpf(str(float(mu0)))) / mp_se
    mp_edge = mpmath.mpf(str(float(mu0))) + (1 if direction >= 0 else -1) * mpmath.mpf(str(float(crit))) * mp_se
    se_text, t_text, edge_text = _dp_text(se, 3), _dp_text(t, 2), _dp_text(edge, 2)
    _mp_check(round_half_up(se, 3), round(mp_se, 3), "exam SE")
    _mp_check(t, mp_t, "exam t")
    _mp_check(edge, mp_edge, "exam boundary mean")

    # (a) distractors
    root_n1 = _root(Fraction(n - 1))
    a_cands = [
        _num_dist(s / n, 3, "se-divides-by-n"),
        _num_dist(s / root_n1, 3, "se-uses-n-minus-1"),
        _num_dist(s * _root(Fraction(n)), 3, "se-multiplies-by-root-n"),
        _num_dist(s * s / _root(Fraction(n)), 3, "variance-used-as-sd"),
    ]
    a_d = _pick(se_text, a_cands, 3, 3)
    # (b)
    b_cands = [
        _num_dist(diff / s, 2, "sd-used-as-standard-error"),
        _num_dist(-t, 2, "t-sign-reversed"),
        _num_dist(diff / (s / n), 2, "se-divides-by-n"),
        _num_dist(diff / (s / root_n1), 2, "se-uses-n-minus-1"),
    ]
    b_d = _pick(t_text, b_cands, 2, 3)
    # (c)
    sgn = 1 if direction >= 0 else -1
    crit_wrong_df = Fraction(str(table[min(max(n, 4), 30)]))
    c_cands = [
        _num_dist(mu0 - sgn * crit * se, 2, "critical-value-wrong-direction"),
        _num_dist(mu0 + sgn * crit * s, 2, "sd-used-as-standard-error"),
        _num_dist(mu0 + sgn * crit, 2, "critical-value-not-scaled"),
        _num_dist(mu0 + sgn * crit_wrong_df * se, 2, "critical-value-wrong-df"),
        _num_dist(xbar + sgn * crit * se, 2, "critical-mean-from-sample-mean"),
    ]
    c_d = _pick(edge_text, c_cands, 2, 3)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors in the t-test chain")
    d_part = _decision_part(rng, is_reject, level, rel, mu_t, unit)

    rows = ", ".join(f"df = {d}: {table[d]:.3f}" for d in (df - 1, df, df + 1))
    label = _TABLES[key][2]
    stem = (
        f"{_sentence_full(ctx, n, mu0, xbar, s)} Let μ be the true {ctx['thing']}. "
        f"A test of H0: μ = {mu_t} against H1: μ {rel} {mu_t} is carried out at the {level} level of significance. "
        f"Critical values of t ({label}): {rows}."
    )
    if direction >= 0:
        c_prompt = (
            f"If n and s were unchanged, find the smallest sample mean above {mu_t} {unit} that would lead to H0 being "
            "rejected. Give your answer correct to 2 decimal places."
        )
        c_ft = f"{_mu_text(ctx, mu0)} + {float(crit)}*part_a"
    else:
        c_prompt = (
            f"If n and s were unchanged, find the largest sample mean below {mu_t} {unit} that would lead to H0 being "
            "rejected. Give your answer correct to 2 decimal places."
        )
        c_ft = f"{_mu_text(ctx, mu0)} - {float(crit)}*part_a"
    xb, m0 = _dp_text(xbar, dp_data), _dp_text(mu0, dp_data)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the standard error of the sample mean, correct to 3 decimal places.",
                answer_kind="number",
                answer=float(se_text),
                answer_text=se_text,
                marks=1,
                sensible_range=(0, None),
                working=[f"SE = s/√n = {_dp_text(s, dp_data)}/√{n} = {se_text}."],
                distractors=a_d,
            ),
            dict(
                prompt="Hence calculate the value of the t statistic, correct to 2 decimal places.",
                answer_kind="number",
                answer=float(t_text),
                answer_text=t_text,
                marks=2,
                follow_through=f"({xb} - {m0})/part_a",
                working=[f"t = (x̄ - μ0)/SE = ({xb} - {m0})/{se_text} = {t_text}."],
                distractors=b_d,
            ),
            dict(
                prompt=c_prompt,
                answer_kind="number",
                answer=float(edge_text),
                answer_text=edge_text,
                marks=2,
                follow_through=c_ft,
                working=[
                    f"df = {df}, so the critical value is {float(crit):.3f}.",
                    f"x̄ = μ0 {'+' if direction >= 0 else '-'} t*·SE = {edge_text}.",
                ],
                distractors=c_d,
            ),
            d_part,
        ],
    )
