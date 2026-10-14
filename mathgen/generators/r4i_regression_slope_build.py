"""Generates `r4i_regression_slope` questions (easy, medium, hard) and the exam chain `exd_x_r4i_regression_slope`.

``uc2-regression-slope-inference`` (``statistics``, answer ``number``, always a decimal stated to 2, 3 or 4 places):
inference for the slope of a simple linear regression. Data arrive as the summary totals n, Σx, Σy, Σx², Σxy, Σy² (so the
learner forms Sxx, Sxy and Syy), or as raw (x, y) pairs, or as a correlation. A t value, when needed, is stated in the
question; critical values are never recalled.

* easy: the least squares slope; the residual variance s² = SSE / (n - 2); or the t statistic for a correlation r from n.
* medium: the standard error of the slope; the t statistic for H0: slope = 0; or the t statistic for H0: slope = b0.
* hard: from raw pairs, a 95% confidence limit for the slope; from r, the two standard deviations and n, the t statistic
  for the slope (which does not depend on the standard deviations); the half-width of a confidence interval for the mean
  response at a stated x; or the t statistic from raw pairs.
* exam: totals for n pairs. (a) the slope (3 dp), (b) its standard error (4 dp), (c) the t
  statistic for H0: slope = 0 (2 dp, follows through from (a) and (b)), (d) the lower 95% confidence limit for the slope
  (3 dp, follows through from (a) and (b)), using a stated t value.

All values are computed in exact fractions up to the final root; the tests recompute them from the raw pairs rebuilt from the
text where listed, from the totals otherwise, and by a direct least squares fit of the pairs.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4i_kit import chain_avoid, clean, dec, need, num_distractors, num_fields, redraw, stable, stored

TOPIC = "r4i_regression_slope"
EXAM = "exd_x_r4i_regression_slope"

T_CRIT = {3: "3.182", 4: "2.776", 5: "2.571", 6: "2.447", 7: "2.365", 8: "2.306", 9: "2.262", 10: "2.228", 12: "2.179"}

# (x name, x unit, y name, y unit, x-range low, x step choices)
SETTINGS = [
    ("weekly advertising spend", "thousand dollars", "weekly sales", "thousand dollars"),
    ("seasonal rainfall", "hundred mm", "wheat yield", "tonnes"),
    ("weekly study time", "hours", "test score", "marks"),
    ("daily maximum temperature", "degrees Celsius", "cold drinks sold", "dozens"),
    ("distance from the city centre", "km", "weekly rent", "dollars"),
]


class Fit:
    """Exact least squares quantities for pairs of integers."""

    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys
        n = len(xs)
        self.n = n
        self.sx, self.sy = sum(xs), sum(ys)
        self.sxx_raw, self.sxy_raw, self.syy_raw = (
            sum(x * x for x in xs),
            sum(x * y for x, y in zip(xs, ys)),
            sum(y * y for y in ys),
        )
        self.Sxx = Fraction(self.sxx_raw) - Fraction(self.sx**2, n)
        self.Sxy = Fraction(self.sxy_raw) - Fraction(self.sx * self.sy, n)
        self.Syy = Fraction(self.syy_raw) - Fraction(self.sy**2, n)
        self.b = self.Sxy / self.Sxx
        self.a = Fraction(self.sy, n) - self.b * Fraction(self.sx, n)
        self.sse = self.Syy - self.b * self.Sxy
        self.s2 = self.sse / (n - 2)
        self.se = math.sqrt(float(self.s2 / self.Sxx))
        self.t = float(self.b) / self.se if self.se else math.inf
        self.r = float(self.Sxy) / math.sqrt(float(self.Sxx * self.Syy))
        self.xbar = Fraction(self.sx, n)


def _draw_fit(rng, n_choices=(6, 7, 8, 10), tmin=2.0, tmax=14.0):
    for _ in attempts(1000, "a regression data set with a clear slope"):
        n = rng.choice(n_choices)
        xs = sorted(rng.sample(range(2, 26), n))
        slope = rng.choice((2, 3, 4, 5, -2, -3, 6))
        a0 = rng.randint(10, 60)
        noise = rng.choice((3, 4, 5, 6))
        ys = [a0 + slope * x + rng.randint(-noise, noise) for x in xs]
        if min(ys) < 1:
            continue
        fit = Fit(xs, ys)
        if fit.sse <= 0 or not (tmin <= abs(fit.t) <= tmax) or fit.b <= 0:
            continue
        return fit
    raise AssertionError("unreachable")


def _totals(fit):
    return f"n = {fit.n}, Σx = {fit.sx}, Σy = {fit.sy}, Σx² = {fit.sxx_raw}, Σxy = {fit.sxy_raw}, Σy² = {fit.syy_raw}"


def _pairs(fit):
    return ", ".join(f"({x}, {y})" for x, y in zip(fit.xs, fit.ys))


def _f(v, places):
    return dec(float(v), places)


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_slope(rng):
    xn, xu, yn, yu = rng.choice(SETTINGS)
    for _ in attempts(500, "a slope away from a rounding boundary"):
        fit = _draw_fit(rng)
        if stable(float(fit.b), 3, 0.12):
            break
    b = float(fit.b)
    cands = [
        (Fraction(fit.sxy_raw, fit.sxx_raw), "r4i-rs-forgot-centering"),
        (fit.Sxy / fit.Syy, "r4i-rs-regressed-x-on-y"),
        (fit.Sxx / fit.Sxy, "r4i-rs-slope-inverted"),
        (fit.r, "r4i-rs-correlation-for-slope"),
        (Fraction(fit.sxy_raw - fit.sx * fit.sy, fit.sxx_raw - fit.sx**2), "r4i-rs-forgot-division-by-n"),
    ]
    dist = need(num_distractors(b, cands, 3))
    text = (
        f"A regression of {yn} ({yu}) on {xn} ({xu}) is fitted by least squares to a sample with "
        f"{_totals(fit)}. Find the slope of the fitted line, correct to 3 decimal places."
    )
    working = [
        f"Sxx = Σx² - (Σx)²/n = {_f(fit.Sxx, 3)} and Sxy = Σxy - ΣxΣy/n = {_f(fit.Sxy, 3)}.",
        f"The slope is Sxy/Sxx = {dec(b, 3)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(b, 3))


def _easy_residual_variance(rng):
    xn, xu, yn, yu = rng.choice(SETTINGS)
    for _ in attempts(500, "a residual variance away from a rounding boundary"):
        fit = _draw_fit(rng)
        if stable(float(fit.s2), 3, 0.12):
            break
    v = float(fit.s2)
    cands = [
        (fit.sse / fit.n, "r4i-rs-divided-by-n"),
        (fit.sse / (fit.n - 1), "r4i-rs-divided-by-n-minus-one"),
        (fit.Syy / (fit.n - 2), "r4i-rs-total-for-residual"),
        (math.sqrt(v), "r4i-rs-sd-for-variance"),
        (fit.sse, "r4i-rs-forgot-division"),
    ]
    dist = need(num_distractors(v, cands, 3))
    text = (
        f"A regression of {yn} ({yu}) on {xn} ({xu}) is fitted by least squares to a sample with "
        f"{_totals(fit)}. Find the estimate s² of the error variance, the residual sum of squares divided by its degrees of "
        f"freedom, correct to 3 decimal places."
    )
    working = [
        f"Sxx = {_f(fit.Sxx, 3)}, Sxy = {_f(fit.Sxy, 3)}, Syy = {_f(fit.Syy, 3)}.",
        f"SSE = Syy - Sxy²/Sxx = {_f(fit.sse, 3)}, and s² = SSE/(n - 2) = {dec(v, 3)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(v, 3))


def _easy_correlation_t(rng):
    for _ in attempts(500, "a correlation with a stable t"):
        n = rng.choice((8, 10, 12, 15, 20, 25))
        r = Fraction(rng.choice((35, 40, 45, 50, 55, 60, 65, 70, 75, 80)), 100) * rng.choice((1, -1))
        t = float(r) * math.sqrt((n - 2) / (1 - float(r) ** 2))
        if stable(abs(t), 3, 0.12):
            break
    t = float(r) * math.sqrt((n - 2) / (1 - float(r) ** 2))
    rf = float(r)
    cands = [
        (rf * math.sqrt(n - 2) / (1 - rf * rf), "r4i-rs-square-root-missing"),
        (rf * math.sqrt(n) / math.sqrt(1 - rf * rf), "r4i-rs-wrong-df"),
        (rf * math.sqrt(n - 1) / math.sqrt(1 - rf * rf), "r4i-rs-wrong-df"),
        (rf * math.sqrt(n - 2) / math.sqrt(abs(rf)), "r4i-rs-r-for-one-minus-r-squared"),
        (rf * math.sqrt(n - 2), "r4i-rs-denominator-dropped"),
    ]
    dist = need(num_distractors(t, cands, 3, signed=True))
    text = rng.choice(
        [
            f"The sample correlation between two measurements on {n} units is r = {_f(r, 2)}. Find the t statistic for "
            f"testing whether the population correlation is zero, correct to 3 decimal places.",
            f"In a simple linear regression based on {n} pairs of observations the sample correlation coefficient is "
            f"{_f(r, 2)}. Find the value of t for the test of zero slope, correct to 3 decimal places.",
        ]
    )
    working = [
        f"t = r√(n - 2)/√(1 - r²) = {_f(r, 2)}×√{n - 2}/√(1 - {float(r) ** 2:.4f}) = {dec(t, 3)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(t, 3))


def _easy(rng):
    return rng.choice((_easy_slope, _easy_residual_variance, _easy_correlation_t))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium_se(rng):
    xn, xu, yn, yu = rng.choice(SETTINGS)
    for _ in attempts(500, "a standard error away from a rounding boundary"):
        fit = _draw_fit(rng)
        if stable(fit.se, 4, 0.15):
            break
    se = fit.se
    s = math.sqrt(float(fit.s2))
    cands = [
        (s / float(fit.Sxx), "r4i-rs-sxx-root-missing"),
        (float(fit.s2) / math.sqrt(float(fit.Sxx)), "r4i-rs-variance-for-sd"),
        (math.sqrt(float(fit.sse / fit.n / fit.Sxx)), "r4i-rs-divided-by-n"),
        (s * math.sqrt(float(fit.Sxx)), "r4i-rs-multiplied-by-root-sxx"),
        (s / math.sqrt(fit.n), "r4i-rs-mean-formula-used"),
    ]
    dist = need(num_distractors(se, cands, 4))
    text = (
        f"A regression of {yn} ({yu}) on {xn} ({xu}) is fitted by least squares to a sample with {_totals(fit)}. "
        f"Find the standard error of the estimated slope, correct to 4 decimal places."
    )
    working = [
        f"Sxx = {_f(fit.Sxx, 3)}, Sxy = {_f(fit.Sxy, 3)}, Syy = {_f(fit.Syy, 3)}; the slope is {_f(fit.b, 4)}.",
        f"SSE = {_f(fit.sse, 3)}, s² = SSE/{fit.n - 2} = {_f(fit.s2, 4)}.",
        f"SE = s/√Sxx = {dec(se, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(se, 4))


def _medium_t(rng):
    xn, xu, yn, yu = rng.choice(SETTINGS)
    for _ in attempts(500, "a t statistic away from a rounding boundary"):
        fit = _draw_fit(rng)
        if stable(fit.t, 2, 0.12):
            break
    t = fit.t
    s = math.sqrt(float(fit.s2))
    cands = [
        (fit.se / float(fit.b), "r4i-rs-t-inverted"),
        (float(fit.b) / s, "r4i-rs-forgot-sxx"),
        (float(fit.b) * fit.se, "r4i-rs-multiplied"),
        (float(fit.b) / math.sqrt(float(fit.sse / fit.n / fit.Sxx)), "r4i-rs-divided-by-n"),
        (fit.r * math.sqrt(fit.n - 1) / math.sqrt(1 - fit.r**2), "r4i-rs-wrong-df"),
    ]
    dist = need(num_distractors(t, cands, 2))
    text = (
        f"A regression of {yn} ({yu}) on {xn} ({xu}) is fitted by least squares to a sample with {_totals(fit)}. "
        f"Find the t statistic for testing whether the true slope is zero, correct to 2 decimal places."
    )
    working = [
        f"b = Sxy/Sxx = {_f(fit.b, 4)}, SSE = {_f(fit.sse, 3)}, s² = {_f(fit.s2, 4)}, SE = {dec(fit.se, 4)}.",
        f"t = b/SE = {dec(t, 2)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(t, 2))


def _medium_t_nonzero(rng):
    xn, xu, yn, yu = rng.choice(SETTINGS)
    for _ in attempts(800, "a t statistic for a non-zero null slope"):
        fit = _draw_fit(rng)
        b0 = Fraction(rng.choice((1, 2, 3, 4)))
        t = (float(fit.b) - float(b0)) / fit.se
        if stable(abs(t), 2, 0.12) and 0.3 < abs(t) < 12 and b0 != round(fit.b):
            break
    cands = [
        (float(fit.b) / fit.se, "r4i-rs-null-value-ignored"),
        ((float(b0) - float(fit.b)) / fit.se, "r4i-rs-sign-reversed"),
        ((float(fit.b) - float(b0)) / math.sqrt(float(fit.s2)), "r4i-rs-forgot-sxx"),
        ((float(fit.b) - float(b0)) * fit.se, "r4i-rs-multiplied"),
        (fit.se / (float(fit.b) - float(b0)), "r4i-rs-t-inverted"),
    ]
    dist = need(num_distractors(t, cands, 2, signed=True))
    text = (
        f"A regression of {yn} ({yu}) on {xn} ({xu}) is fitted by least squares to a sample with {_totals(fit)}. "
        f"A claim is made that the true slope is {b0}. Find the t statistic for testing this claim, correct "
        f"to 2 decimal places."
    )
    working = [
        f"b = {_f(fit.b, 4)} and SE = {dec(fit.se, 4)}.",
        f"t = (b - {b0})/SE = {dec(t, 2)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(t, 2))


def _medium(rng):
    return rng.choice((_medium_se, _medium_t, _medium_t_nonzero))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard_raw_limit(rng):
    xn, xu, yn, yu = rng.choice(SETTINGS)
    for _ in attempts(1000, "raw data with a stable confidence limit"):
        fit = _draw_fit(rng, n_choices=(6, 7, 8))
        df = fit.n - 2
        tv = float(T_CRIT[df])
        upper = rng.random() < 0.5
        limit = float(fit.b) + (tv if upper else -tv) * fit.se
        if stable(limit, 3, 0.15):
            break
    cands = [
        (float(fit.b) + (-tv if upper else tv) * fit.se, "r4i-rs-wrong-end"),
        (tv * fit.se, "r4i-rs-margin-only"),
        (float(fit.b) + (1.96 if upper else -1.96) * fit.se, "r4i-rs-z-for-t"),
        (float(fit.b) + (1 if upper else -1) * fit.se, "r4i-rs-forgot-t-value"),
        (float(fit.b) + (tv if upper else -tv) * math.sqrt(float(fit.s2)), "r4i-rs-forgot-sxx"),
    ]
    dist = need(num_distractors(limit, cands, 3, signed=True))
    end = "upper" if upper else "lower"
    text = (
        f"A researcher records {xn} ({xu}) and {yn} ({yu}) over {fit.n} observations, giving the pairs {_pairs(fit)}. Fit a "
        f"least squares line of {yn} on {xn}. Using the t value {T_CRIT[df]} for {df} degrees of freedom, find the {end} limit "
        f"of a 95% confidence interval for the true slope, correct to 3 decimal places."
    )
    working = [
        f"Σx = {fit.sx}, Σy = {fit.sy}, Σx² = {fit.sxx_raw}, Σxy = {fit.sxy_raw}, Σy² = {fit.syy_raw}.",
        f"b = {_f(fit.b, 4)}, s² = {_f(fit.s2, 4)}, SE = {dec(fit.se, 4)}.",
        f"The {end} limit is b {'+' if upper else '-'} {T_CRIT[df]}×SE = {dec(limit, 3)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(limit, 3))


def _hard_r_t(rng):
    for _ in attempts(500, "a stable t from r"):
        n = rng.choice((8, 10, 12, 15, 20))
        r = Fraction(rng.choice((40, 45, 50, 55, 60, 65, 70, 75, 80)), 100)
        sx = rng.choice((2, 3, 4, 5, 6))
        sy = rng.choice((8, 10, 12, 15, 20, 25))
        rf = float(r)
        t = rf * math.sqrt((n - 2) / (1 - rf * rf))
        if stable(t, 3, 0.12):
            break
    b = rf * sy / sx
    cands = [
        (b * math.sqrt(n - 2) / math.sqrt(1 - rf * rf), "r4i-rs-slope-times-factor"),
        (rf * math.sqrt(n - 2) / (1 - rf * rf), "r4i-rs-square-root-missing"),
        (rf * math.sqrt(n - 1) / math.sqrt(1 - rf * rf), "r4i-rs-wrong-df"),
        (b / (sy / sx), "r4i-rs-correlation-for-slope"),
        (rf * sy / sx * math.sqrt(n - 2), "r4i-rs-slope-times-factor"),
    ]
    dist = need(num_distractors(t, cands, 3))
    text = (
        f"For {n} pairs of observations the sample correlation is r = {_f(r, 2)}, the standard deviation of x is {sx} and the "
        f"standard deviation of y is {sy}. Find the t statistic for testing whether the true regression slope of y on x is "
        f"zero, correct to 3 decimal places."
    )
    working = [
        f"The slope is b = r s_y/s_x = {b:.4f}, and its t statistic simplifies to r√(n - 2)/√(1 - r²).",
        f"t = {_f(r, 2)}×√{n - 2}/√(1 - {rf * rf:.4f}) = {dec(t, 3)}; the standard deviations cancel.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(t, 3))


def _hard_mean_response(rng):
    xn, xu, yn, yu = rng.choice(SETTINGS)
    for _ in attempts(1000, "a stable mean-response half-width"):
        fit = _draw_fit(rng, n_choices=(6, 7, 8, 10))
        df = fit.n - 2
        if df not in T_CRIT:
            continue
        tv = float(T_CRIT[df])
        x0 = rng.randint(int(min(fit.xs)), int(max(fit.xs)))
        s = math.sqrt(float(fit.s2))
        d2 = float((x0 - fit.xbar) ** 2)
        h = tv * s * math.sqrt(1 / fit.n + d2 / float(fit.Sxx))
        if stable(h, 3, 0.15) and d2 > 0.5:
            break
    cands = [
        (tv * s * math.sqrt(1 + 1 / fit.n + d2 / float(fit.Sxx)), "r4i-rs-prediction-interval"),
        (tv * s * math.sqrt(1 / fit.n), "r4i-rs-forgot-distance-term"),
        (tv * s * math.sqrt(1 / fit.n + math.sqrt(d2) / float(fit.Sxx)), "r4i-rs-distance-not-squared"),
        (tv * s * math.sqrt(d2 / float(fit.Sxx)), "r4i-rs-forgot-one-over-n"),
        (1.96 * s * math.sqrt(1 / fit.n + d2 / float(fit.Sxx)), "r4i-rs-z-for-t"),
    ]
    dist = need(num_distractors(h, cands, 3))
    text = (
        f"A regression of {yn} ({yu}) on {xn} ({xu}) is fitted by least squares to a sample with {_totals(fit)}. Find the "
        f"half-width of a 95% confidence interval for the mean {yn} when {xn} is {x0} {xu}, using the t value {T_CRIT[df]} for "
        f"{df} degrees of freedom. Give the answer correct to 3 decimal places."
    )
    working = [
        f"x̄ = {_ft_(fit.xbar)}, Sxx = {_f(fit.Sxx, 3)}, s² = {_f(fit.s2, 4)}.",
        f"Half-width = t s √(1/n + (x₀ - x̄)²/Sxx) = {dec(h, 3)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(h, 3))


def _ft_(v):
    v = Fraction(v)
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def _hard_raw_t(rng):
    xn, xu, yn, yu = rng.choice(SETTINGS)
    for _ in attempts(1000, "raw data with a stable t"):
        fit = _draw_fit(rng, n_choices=(6, 7, 8))
        if stable(fit.t, 2, 0.12):
            break
    s = math.sqrt(float(fit.s2))
    cands = [
        (fit.se / float(fit.b), "r4i-rs-t-inverted"),
        (float(fit.b) / s, "r4i-rs-forgot-sxx"),
        (float(fit.b) / math.sqrt(float(fit.sse / fit.n / fit.Sxx)), "r4i-rs-divided-by-n"),
        (fit.r * math.sqrt(fit.n - 1) / math.sqrt(1 - fit.r**2), "r4i-rs-wrong-df"),
    ]
    dist = need(num_distractors(fit.t, cands, 2))
    text = (
        f"A researcher records {xn} ({xu}) and {yn} ({yu}) over {fit.n} observations, giving the pairs {_pairs(fit)}. Fit a least "
        f"squares line of {yn} on {xn}, and find the t statistic for testing whether the true slope is zero, correct to 2 "
        f"decimal places."
    )
    working = [
        f"Σx = {fit.sx}, Σy = {fit.sy}, Σx² = {fit.sxx_raw}, Σxy = {fit.sxy_raw}, Σy² = {fit.syy_raw}.",
        f"b = {_f(fit.b, 4)}, SE = {dec(fit.se, 4)}, so t = {dec(fit.t, 2)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(fit.t, 2))


def _hard(rng):
    return rng.choice((_hard_raw_limit, _hard_r_t, _hard_mean_response, _hard_raw_t))(rng)


@generator(
    TOPIC,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraw
def r4i_regression_slope(rng, tier="easy"):
    """Inference for a regression slope; see the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------


@generator(EXAM, level=21, presets={"exam": dict(tier="exam")})
@redraw
def exd_x_r4i_regression_slope(rng, tier="exam"):
    """Totals for n pairs: the slope, its standard error, the t statistic, then a lower confidence limit."""
    check(tier == "exam", f"unknown tier {tier!r}")
    xn, xu, yn, yu = rng.choice(SETTINGS)
    for _ in attempts(2000, "an exam draw whose chain survives rounding"):
        fit = _draw_fit(rng, n_choices=(6, 7, 8, 10), tmin=4.5)
        df = fit.n - 2
        if df not in T_CRIT:
            continue
        tv = float(T_CRIT[df])
        a = clean(float(fit.b), 3)
        b = clean(fit.se, 4)
        c_ex = fit.t
        c = clean(c_ex, 2)
        c_ft = a / b
        d_ex = float(fit.b) - tv * fit.se
        d = clean(d_ex, 3)
        d_ft = a - tv * b
        if (
            stable(float(fit.b), 3, 0.15)
            and stable(fit.se, 4, 0.15)
            and stable(c_ex, 2, 0.15)
            and clean(c_ft, 2) == c
            and stable(d_ex, 3, 0.15)
            and clean(d_ft, 3) == d
        ):
            break
    order = ["a", "b", "c", "d"]
    canon = {"a": a, "b": b, "c": c, "d": d}
    fns = {
        "c": lambda s: s["a"] / s["b"] if s["b"] else None,
        "d": lambda s: s["a"] - tv * s["b"],
    }
    rounders = {"c": lambda v: clean(v, 2), "d": lambda v: clean(v, 3)}
    a_dist = need(
        num_distractors(
            a,
            [
                (Fraction(fit.sxy_raw, fit.sxx_raw), "r4i-rs-forgot-centering"),
                (fit.Sxy / fit.Syy, "r4i-rs-regressed-x-on-y"),
                (fit.Sxx / fit.Sxy, "r4i-rs-slope-inverted"),
                (fit.r, "r4i-rs-correlation-for-slope"),
            ],
            3,
        )
    )
    wrong = {"a": [float(stored(x)) for x in a_dist]}
    avoid_b = [
        v for v in chain_avoid(order, fns, canon, wrong, "b", rounders) if v is not None and not isinstance(v, complex)
    ]
    s = math.sqrt(float(fit.s2))
    b_dist = need(
        num_distractors(
            b,
            [
                (s / float(fit.Sxx), "r4i-rs-sxx-root-missing"),
                (float(fit.s2) / math.sqrt(float(fit.Sxx)), "r4i-rs-variance-for-sd"),
                (math.sqrt(float(fit.sse / fit.n / fit.Sxx)), "r4i-rs-divided-by-n"),
                (s * math.sqrt(float(fit.Sxx)), "r4i-rs-multiplied-by-root-sxx"),
            ],
            4,
            avoid=avoid_b,
        )
    )
    wrong["b"] = [float(stored(x)) for x in b_dist]
    avoid_c = [v for v in chain_avoid(order, fns, canon, wrong, "c", rounders) if v is not None]
    c_dist = need(
        num_distractors(
            c,
            [
                (b / a, "r4i-rs-t-inverted"),
                (a / s, "r4i-rs-forgot-sxx"),
                (a * b, "r4i-rs-multiplied"),
                (fit.r * math.sqrt(fit.n - 1) / math.sqrt(1 - fit.r**2), "r4i-rs-wrong-df"),
            ],
            2,
            avoid=avoid_c,
        )
    )
    wrong["c"] = [float(stored(x)) for x in c_dist]
    avoid_d = [v for v in chain_avoid(order, fns, canon, wrong, "d", rounders) if v is not None]
    d_dist = need(
        num_distractors(
            d,
            [
                (a + tv * b, "r4i-rs-wrong-end"),
                (tv * b, "r4i-rs-margin-only"),
                (a - 1.96 * b, "r4i-rs-z-for-t"),
                (a - b, "r4i-rs-forgot-t-value"),
            ],
            3,
            avoid=avoid_d,
            signed=True,
        )
    )
    context = (
        f"The {xn} ({xu}) and {yn} ({yu}) were recorded over {fit.n} observations, with totals {_totals(fit)}. A least squares line "
        f"of {yn} on {xn} is fitted."
    )
    text = rng.choice(
        [
            context,
            f"An analyst studies how {yn} ({yu}) depends on {xn} ({xu}) across {fit.n} observations. The summary totals are "
            f"{_totals(fit)}, and a least squares regression line is fitted to predict {yn}.",
            f"Data on {xn} ({xu}) and {yn} ({yu}) from {fit.n} observations give {_totals(fit)}. The straight line of best fit for "
            f"{yn} against {xn} is found by least squares.",
        ]
    )
    parts = [
        dict(
            prompt="Find the slope of the fitted line, correct to 3 decimal places.",
            marks=2,
            sensible_range=(0, None),
            working=[f"Sxx = {_f(fit.Sxx, 3)}, Sxy = {_f(fit.Sxy, 3)}, so b = {dec(a, 3)}."],
            distractors=a_dist,
            **num_fields(a, 3),
        ),
        dict(
            prompt="Find the standard error of the slope, correct to 4 decimal places.",
            marks=3,
            sensible_range=(0, None),
            working=[
                f"SSE = Syy - b Sxy = {_f(fit.sse, 3)}, s² = SSE/{df} = {_f(fit.s2, 4)}.",
                f"SE = √(s²/Sxx) = {dec(b, 4)}.",
            ],
            distractors=b_dist,
            **num_fields(b, 4),
        ),
        dict(
            prompt="Find the t statistic for testing whether the true slope is zero, correct to 2 decimal places.",
            marks=1,
            sensible_range=(0, None),
            follow_through="part_a/part_b",
            working=[f"t = b/SE = {dec(a, 3)}/{dec(b, 4)} = {dec(c, 2)}."],
            distractors=c_dist,
            **num_fields(c, 2),
        ),
        dict(
            prompt=(
                f"Using the t value {T_CRIT[df]} for {df} degrees of freedom, find the lower limit of a 95% confidence "
                f"interval for the true slope, correct to 3 decimal places."
            ),
            marks=2,
            follow_through=f"part_a - {T_CRIT[df]}*part_b",
            working=[f"Lower limit = b - {T_CRIT[df]}×SE = {dec(a, 3)} - {T_CRIT[df]}×{dec(b, 4)} = {dec(d, 3)}."],
            distractors=d_dist,
            **num_fields(d, 3),
        ),
    ]
    return dict(text=text, parts=parts)
