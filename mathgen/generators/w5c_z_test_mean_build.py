"""Generates `w5c_z_test_mean` questions and the exam chain `exd_x_w5c_z_test_mean`.

``ucv1-z-test-mean`` (``statistics``): a z-test for a population mean when the population standard deviation is
known. Probabilities are never worked out on a calculator: the question prints a short excerpt of the standard
normal table, Phi(z) = P(Z <= z) to 4 decimal places, around the value of |z| that the question needs, and the
learner reads the p-value (or the z behind a p-value) from it. Every printed table value is computed here from
the normal distribution with mpmath and checked against ``math.erfc``.

Tiers (topic ``w5c_z_test_mean``):

* easy   -- the z statistic from n, the known sigma (or the known variance), the sample mean and the hypothesised
  mean, to 2 decimal places;
* medium -- the p-value to 4 decimal places from the table excerpt, for an upper-tail, lower-tail or two-sided test:
  the tail has to be chosen from H1 and the sign of z, so a z on the "wrong" side of H0 gives a p-value above 0.5;
* hard   -- one of: the smallest sample size at which the same sample mean would be significant (an inequality
  solved for n and rounded up); the probability of a type II error at a stated true mean (a boundary on the sample
  mean, then a probability under the alternative); the sample mean behind a reported two-sided p-value (the table
  read backwards).

The exam tier (topic ``exd_x_w5c_z_test_mean``) is a four-part chain: (a) the z statistic, (b) the p-value
(follow-through from (a)), (c) the sample mean on the boundary of the rejection region (follow-through from (a):
the standard error is recovered as (x-bar - mu0)/z), (d) the one-mark conclusion as a numbered statement.

Every question is built backwards from a chosen z (or a chosen z-shift), and every key is recomputed a second way.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    VerificationError,
    attempts,
    check,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)
from .exd_x_r2k_support import number_part, retry_build

# ---------------------------------------------------------------------------
# The normal table


@lru_cache(maxsize=None)
def phi_table(z: Fraction) -> Fraction:
    """Phi(z) to 4 decimal places for a z with at most 2 decimal places, as a printed table gives it.

    Computed from the error function in double precision (Phi is irrational at these z, so the 4-place rounding is
    never near a tie); `tests/test_w5c_z_test_mean.py` re-derives the table with mpmath at 30 digits."""
    return round_half_up(Fraction(repr(0.5 * math.erfc(-float(z) / math.sqrt(2)))), 4)


def phi_table_nonneg(z: Fraction) -> Fraction:
    """Phi(|z|): what a table that stops at z = 0 gives."""
    return phi_table(abs(z))


def tail_from_table(z: Fraction, tail: str) -> Fraction:
    """The p-value for statistic z and tail 'upper', 'lower' or 'two', read from a table of Phi(|z|)."""
    t = phi_table_nonneg(z)
    if tail == "two":
        return 2 * (1 - t)
    if tail == "upper":
        return 1 - t if z >= 0 else t
    return 1 - t if z <= 0 else t


def table_text(z: Fraction, rng: random.Random | None = None) -> str:
    """The table excerpt printed in a stem: five consecutive values of Phi, with |z| somewhere among them.

    The row the question needs sits at a random place in the five (with `rng`), so its position gives nothing away."""
    centre = abs(z)
    shift = rng.randint(-2, 2) if rng is not None else 0
    start = centre + Fraction(shift, 100) - Fraction(2, 100)
    if start < 0:
        start = Fraction(0)
    rows = []
    for k in range(5):
        zz = round_half_up(start + Fraction(k, 100), 2)
        rows.append((zz, phi_table(zz)))
    cells = ", ".join(f"Φ({stated_precision_string(a, 2)}) = {stated_precision_string(b, 4)}" for a, b in rows)
    return f"From the standard normal table, where Φ(z) = P(Z ≤ z): {cells}."


# ---------------------------------------------------------------------------
# Helpers


def _root(x: Fraction, digits: int = 30) -> Fraction:
    scaled = x.numerator * 10 ** (2 * digits) // x.denominator
    return Fraction(math.isqrt(scaled), 10**digits)


def _txt(value: Fraction, dp: int) -> str:
    return stated_precision_string(round_half_up(value, dp), dp)


def _clear(value: Fraction, dp: int, margin: Fraction = Fraction(1, 10)) -> bool:
    scaled = abs(value) * 10**dp
    frac = scaled - math.floor(scaled)
    return abs(frac - Fraction(1, 2)) > margin


def _near(a: Fraction, b: Fraction, tol: Fraction) -> bool:
    return abs(a - b) <= tol


_SQUARES = [16, 25, 36, 49, 64, 81, 100]
_ANY_N = [16, 20, 24, 25, 30, 36, 40, 45, 49, 50, 60, 64, 75, 80, 100]

# dp: places of the data (sample mean, hypothesised mean); sd_lo/sd_hi: the known sigma in units of 10**-sdp.
_CONTEXTS = [
    dict(
        dp=1,
        sdp=1,
        mu0=["600", "750", "1000"],
        sd_lo=30,
        sd_hi=90,
        lead="A dairy in Gippsland says its cartons hold a mean of {mu0} mL, and years of quality records show that the fill volume has a standard deviation of {sd} mL.",
        lead_var="A dairy in Gippsland says its cartons hold a mean of {mu0} mL, and years of quality records show that the variance of the fill volume is {var} mL².",
        obj="cartons",
        meanphrase="a mean fill of",
        unit="mL",
        thing="mean fill volume, in mL",
    ),
    dict(
        dp=1,
        sdp=1,
        mu0=["28", "32", "35", "40"],
        sd_lo=35,
        sd_hi=75,
        lead="The timetable for a Brisbane bus route gives a mean journey time of {mu0} minutes from Chermside to the city. Past records show that journey times have a standard deviation of {sd} minutes.",
        lead_var="The timetable for a Brisbane bus route gives a mean journey time of {mu0} minutes from Chermside to the city. Past records show that journey times have a variance of {var} minutes².",
        obj="trips",
        meanphrase="a mean journey time of",
        unit="minutes",
        thing="mean journey time, in minutes",
    ),
    dict(
        dp=2,
        sdp=2,
        mu0=["14.00", "16.00", "18.00", "20.00"],
        sd_lo=90,
        sd_hi=220,
        lead="A manufacturer states that its phone battery lasts a mean of {mu0} hours of video playback, and the standard deviation of battery life is known to be {sd} hours.",
        lead_var="A manufacturer states that its phone battery lasts a mean of {mu0} hours of video playback, and the variance of battery life is known to be {var} hours².",
        obj="batteries",
        meanphrase="a mean life of",
        unit="hours",
        thing="mean battery life, in hours",
    ),
    dict(
        dp=1,
        sdp=1,
        mu0=["680", "700", "720", "750"],
        sd_lo=100,
        sd_hi=260,
        lead="A Melbourne bakery says its sourdough loaves have a mean mass of {mu0} g. The mass of its loaves is known to have a standard deviation of {sd} g.",
        lead_var="A Melbourne bakery says its sourdough loaves have a mean mass of {mu0} g. The mass of its loaves is known to have a variance of {var} g².",
        obj="loaves",
        meanphrase="a mean mass of",
        unit="g",
        thing="mean mass of a loaf, in g",
    ),
    dict(
        dp=2,
        sdp=2,
        mu0=["28.00", "30.00", "32.00", "34.00"],
        sd_lo=60,
        sd_hi=150,
        lead="A swimming coach believes her squad's mean time for 50 m freestyle is {mu0} seconds. From many past races, the standard deviation of the squad's times is {sd} seconds.",
        lead_var="A swimming coach believes her squad's mean time for 50 m freestyle is {mu0} seconds. From many past races, the variance of the squad's times is {var} seconds².",
        obj="swims",
        meanphrase="a mean time of",
        unit="seconds",
        thing="mean 50 m freestyle time, in seconds",
    ),
    dict(
        dp=1,
        sdp=1,
        mu0=["58.0", "62.0", "65.0", "70.0"],
        sd_lo=100,
        sd_hi=140,
        lead="Across Queensland, the mean mark on a Year 12 trial examination is {mu0}, with a standard deviation of {sd} marks. A school wants to test whether its own students differ.",
        lead_var="Across Queensland, the mean mark on a Year 12 trial examination is {mu0}, with a variance of {var} marks². A school wants to test whether its own students differ.",
        obj="students",
        meanphrase="a mean mark of",
        unit="marks",
        thing="mean trial examination mark",
    ),
]


def _fmt(value: Fraction, dp: int) -> str:
    return stated_precision_string(value, dp)


def _zs(z: Fraction) -> str:
    """A critical z as printed: 1.645, 2.326, but 1.96 and 2.33 without the trailing zero."""
    text = stated_precision_string(z, 3)
    return text[:-1] if text.endswith("0") else text


def _lead(ctx, mu0: Fraction, sigma: Fraction, *, variance: bool) -> str:
    mu_s = _fmt(mu0, ctx["dp"]) if mu0 != int(mu0) or ctx["dp"] > 0 else str(int(mu0))
    mu_s = mu_s.rstrip("0").rstrip(".") if "." in mu_s and ctx["dp"] == 1 and mu_s.endswith(".0") else mu_s
    if variance:
        return ctx["lead_var"].format(mu0=mu_s, var=_fmt(sigma * sigma, 2 * ctx["sdp"]))
    return ctx["lead"].format(mu0=mu_s, sd=_fmt(sigma, ctx["sdp"]))


def _mu_s(ctx, mu0: Fraction) -> str:
    s = _fmt(mu0, ctx["dp"])
    return s[:-2] if s.endswith(".0") else s


def _sample_sentence(ctx, n, xbar) -> str:
    return f"A random sample of {n} {ctx['obj']} has {ctx['meanphrase']} {_fmt(xbar, ctx['dp'])} {ctx['unit']}."


class Draw:
    """One consistent sample: n, mu0, sigma, xbar, the exact standard error, the exact z and z to 2 places."""

    def __init__(self, ctx, n, mu0, sigma, xbar):
        self.ctx, self.n, self.mu0, self.sigma, self.xbar = ctx, n, mu0, sigma, xbar
        self.se = sigma / _root(Fraction(n))
        self.diff = xbar - mu0
        self.z = self.diff / self.se
        self.z2 = round_half_up(self.z, 2)


def _draw(rng, ctx, *, z_lo=0.4, z_hi=3.2, n_choices=_ANY_N, sign=None, stable_se3=False) -> Draw:
    dp, sdp = ctx["dp"], ctx["sdp"]
    for _ in attempts(2000, "a z-test sample with a stable z"):
        n = rng.choice(n_choices)
        mu0 = Fraction(rng.choice(ctx["mu0"]))
        sigma = Fraction(rng.randint(ctx["sd_lo"], ctx["sd_hi"]), 10**sdp)
        se = sigma / _root(Fraction(n))
        target = Fraction(rng.randint(int(z_lo * 100), int(z_hi * 100)), 100)
        s = sign if sign is not None else rng.choice((-1, 1))
        xbar = round_half_up(mu0 + s * target * se, dp)
        if xbar == mu0 or xbar <= 0:
            continue
        d = Draw(ctx, n, mu0, sigma, xbar)
        if not (
            Fraction(z_lo).limit_denominator(100) - Fraction(1, 10)
            <= abs(d.z2)
            <= Fraction(z_hi).limit_denominator(100) + Fraction(1, 10)
        ):
            continue
        if not _clear(d.z, 2) or abs(d.z2) > Fraction(335, 100):
            continue
        if stable_se3:
            z_ft = d.diff / round_half_up(d.se, 3)
            if round_half_up(z_ft, 2) != d.z2:
                continue
        if abs(d.z2 - d.z) > Fraction(1, 200):
            continue
        return d
    raise VerificationError("unreachable")  # pragma: no cover


def _num_dist(value: Fraction, dp: int, tag: str) -> Distractor:
    text = _txt(value, dp)
    return Distractor(text, tag, float(text))


def _pick(answer_text: str, cands: list[Distractor], dp: int, n: int = 4) -> list[Distractor]:
    return pick_distractors(answer_text, cands, n, tolerance=Fraction(1, 10**dp))


# ---------------------------------------------------------------------------
# Easy: the z statistic

_EASY_TAIL = [("≠", "differs from"), (">", "is greater than"), ("<", "is less than")]


def _easy_question(rng, ctx) -> dict:
    d = _draw(rng, ctx, z_lo=0.4, z_hi=3.0, n_choices=_ANY_N)
    variance = rng.random() < 0.35
    answer_text = _txt(d.z, 2)
    sqrt_n = _root(Fraction(d.n))
    cands = [
        _num_dist(d.diff / d.sigma, 2, "sd-used-as-standard-error"),
        _num_dist(d.diff / (d.sigma / d.n), 2, "se-divides-by-n"),
        _num_dist(-d.z, 2, "z-sign-reversed"),
        _num_dist(d.diff / (d.sigma * sqrt_n), 2, "se-multiplies-by-root-n"),
    ]
    if variance:
        cands.append(_num_dist(d.diff / (d.sigma**2 / sqrt_n), 2, "variance-used-as-sd"))
    distractors = _pick(answer_text, cands, 2)
    check(len(distractors) >= 3, "too few z-statistic distractors")
    lead = _lead(ctx, d.mu0, d.sigma, variance=variance)
    mu_t = _mu_s(ctx, d.mu0)
    shape = rng.randrange(3)
    rel = rng.choice(("≠", ">" if d.diff > 0 else "<"))
    sample = _sample_sentence(ctx, d.n, d.xbar)
    if shape == 0:
        text = f"{lead} {sample} Test H0: μ = {mu_t} against H1: μ {rel} {mu_t}. Calculate the value of the test statistic z, correct to 2 decimal places."
    elif shape == 1:
        text = f"{sample} {lead} Find the z statistic for testing whether the true mean {_REL_WORDS[rel]} {mu_t} {ctx['unit']}. Give your answer correct to 2 decimal places."
    else:
        text = f"{lead} For a z-test of H0: μ = {mu_t}, a researcher collects data. {sample} What is the standardised value of the sample mean under H0? Give z correct to 2 decimal places."
    return dict(
        text=text,
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[
            f"SE = σ/√n = {_txt(d.se, 4)}.",
            f"z = (x̄ − μ0)/SE = ({_fmt(d.xbar, ctx['dp'])} − {_mu_s(ctx, d.mu0)})/{_txt(d.se, 4)} = {answer_text}.",
        ],
        distractors=distractors,
    )


_REL_WORDS = {"≠": "differs from", ">": "is greater than", "<": "is less than"}


# ---------------------------------------------------------------------------
# Medium: the p-value from the table

_TAILS = [
    ("upper", "μ >"),
    ("lower", "μ <"),
    ("two", "μ ≠"),
]


def _p_distractors(d_z: Fraction, tail: str, p: Fraction) -> list[Distractor]:
    t = phi_table_nonneg(d_z)
    cands: list[tuple[Fraction, str]] = []
    if tail == "two":
        cands.append((p / 2, "one-tail-for-two-tails"))
        cands.append((1 - p, "gave-complement-of-tail"))
        cands.append((t, "gave-cumulative-for-tail"))
    else:
        cands.append((2 * p if 2 * p <= 1 else Fraction(-1), "doubled-for-one-tail"))
        cands.append((1 - p, "gave-complement-of-tail"))
        cands.append((t, "gave-cumulative-for-tail"))
    cands.append((t - Fraction(1, 2), "uses-area-from-the-mean"))
    out = []
    for value, tag in cands:
        if value <= 0 or value >= 1:
            continue
        text = _txt(value, 4)
        out.append(Distractor(text, tag, float(text)))
    return out


def _medium_question(rng, ctx) -> dict:
    tail, h1 = rng.choice(_TAILS)
    # one-sided tests usually have the sample on the side H1 names; now and then it falls on the other side
    side = rng.choice((-1, 1)) if tail == "two" or rng.random() < 0.25 else (1 if tail == "upper" else -1)
    d = _draw(rng, ctx, z_lo=0.5, z_hi=3.1, n_choices=_ANY_N, sign=side)
    p = tail_from_table(d.z2, tail)
    variance = rng.random() < 0.3
    answer_text = _txt(p, 4)
    distractors = _pick(answer_text, _p_distractors(d.z2, tail, p), 4, 3)
    check(len(distractors) >= 2, "too few p-value distractors")
    lead = _lead(ctx, d.mu0, d.sigma, variance=variance)
    mu_t = _mu_s(ctx, d.mu0)
    sample = _sample_sentence(ctx, d.n, d.xbar)
    table = table_text(d.z2, rng)
    shape = rng.randrange(3)
    ask = "Find the p-value of the test, correct to 4 decimal places."
    if shape == 0:
        text = f"{lead} {sample} Test H0: μ = {mu_t} against H1: {h1} {mu_t}. {table} {ask}"
    elif shape == 1:
        text = f"{sample} {lead} A z-test of H0: μ = {mu_t} against H1: {h1} {mu_t} is carried out. {table} What is the p-value? Give it correct to 4 decimal places."
    else:
        text = f"{lead} The school collects data to test H0: μ = {mu_t} against H1: {h1} {mu_t}. {sample} {table} Calculate the p-value for this test, correct to 4 decimal places."
    which = {"upper": "P(Z ≥ z)", "lower": "P(Z ≤ z)", "two": "2P(Z ≥ |z|)"}[tail]
    return dict(
        text=text,
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[
            f"z = ({_fmt(d.xbar, ctx['dp'])} − {mu_t})/({_txt(d.sigma, ctx['sdp'])}/√{d.n}) = {_txt(d.z2, 2)}.",
            f"p = {which} = {answer_text}.",
        ],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# Hard

_CRIT = [
    # (tail, critical value z*, label)
    ("upper", Fraction(1645, 1000), "5% level"),
    ("lower", Fraction(1645, 1000), "5% level"),
    ("upper", Fraction(2326, 1000), "1% level"),
    ("lower", Fraction(2326, 1000), "1% level"),
    ("two", Fraction(1960, 1000), "5% level"),
    ("two", Fraction(2576, 1000), "1% level"),
]


def _reject_rule(tail: str, zstar: Fraction) -> str:
    s = _zs(zstar)
    return {"upper": f"z > {s}", "lower": f"z < −{s}", "two": f"|z| > {s}"}[tail]


def _h1(tail: str, mu_t: str) -> str:
    return {"upper": f"μ > {mu_t}", "lower": f"μ < {mu_t}", "two": f"μ ≠ {mu_t}"}[tail]


def _min_n_question(rng, ctx) -> dict:
    dp, sdp = ctx["dp"], ctx["sdp"]
    for _ in attempts(3000, "a minimum sample size question"):
        tail, zstar, label = rng.choice(_CRIT)
        mu0 = Fraction(rng.choice(ctx["mu0"]))
        sigma = Fraction(rng.randint(ctx["sd_lo"], ctx["sd_hi"]), 10**sdp)
        sgn = {"upper": 1, "lower": -1, "two": rng.choice((-1, 1))}[tail]
        step = Fraction(rng.randint(2, 14), 10**dp) * (1 if dp == 1 else Fraction(1, 1))
        delta = sgn * step * rng.choice((1, 2, 3))
        xbar = mu0 + delta
        if xbar <= 0:
            continue
        raw = (zstar * sigma / abs(delta)) ** 2
        frac = raw - math.floor(raw)
        if not (Fraction(15, 100) < frac < Fraction(85, 100)) or not (6 <= raw <= 150):
            continue
        n_min = math.floor(raw) + 1
        break
    # second route: brute force the smallest n that rejects
    smallest = next(n for n in range(2, 400) if abs(delta) / (sigma / _root(Fraction(n))) > zstar)
    check(smallest == n_min, "minimum n: brute-force route disagrees")
    answer_text = str(n_min)
    ratio = zstar * sigma / abs(delta)
    cands = [
        Distractor(str(math.floor(raw)), "rounds-down-instead-of-up", math.floor(raw)),
        Distractor(str(math.ceil(ratio)), "sample-size-not-squared", math.ceil(ratio)),
        Distractor(
            str(math.ceil((zstar * abs(delta) / sigma) ** 2) if (zstar * abs(delta) / sigma) ** 2 > 1 else n_min + 3),
            "sample-size-ratio-inverted",
            math.ceil((zstar * abs(delta) / sigma) ** 2) if (zstar * abs(delta) / sigma) ** 2 > 1 else n_min + 3,
        ),
        Distractor(str(math.ceil(raw) + 1), "sample-size-off-by-one", math.ceil(raw) + 1),
    ]
    distractors = pick_distractors(answer_text, cands, 3)
    check(len(distractors) >= 2, "too few minimum-n distractors")
    mu_t = _mu_s(ctx, mu0)
    lead = _lead(ctx, mu0, sigma, variance=False)
    rule = _reject_rule(tail, zstar)
    shape = rng.randrange(3)
    x_s = _fmt(xbar, dp)
    if shape == 0:
        text = (
            f"{lead} A z-test of H0: μ = {mu_t} against H1: {_h1(tail, mu_t)} at the {label} rejects H0 when {rule}. "
            f"A sample gives a mean of {x_s} {ctx['unit']}. What is the smallest sample size n for which a sample mean of {x_s} {ctx['unit']} would lead to H0 being rejected?"
        )
    elif shape == 1:
        text = (
            f"{lead} Test H0: μ = {mu_t} against H1: {_h1(tail, mu_t)}; H0 is rejected at the {label} when {rule}. "
            f"Suppose the sample mean turns out to be {x_s} {ctx['unit']}. Find the least number of {ctx['obj']} that must be sampled for this result to be significant."
        )
    else:
        text = (
            f"{lead} The sample mean is expected to be {x_s} {ctx['unit']}. For the test of H0: μ = {mu_t} against H1: {_h1(tail, mu_t)}, "
            f"H0 is rejected at the {label} when {rule}. How many {ctx['obj']} must be sampled, at a minimum, for a mean of {x_s} {ctx['unit']} to reject H0?"
        )
    return dict(
        text=text,
        answer=n_min,
        answer_text=answer_text,
        answer_kind="number",
        working=[
            f"Need |x̄ − μ0|/(σ/√n) > {_zs(zstar)}, so √n > {_zs(zstar)} × {_txt(sigma, sdp)}/{_txt(abs(delta), dp)} = {_txt(ratio, 3)}.",
            f"n > {_txt(raw, 2)}, so the smallest whole sample size is {n_min}.",
        ],
        distractors=distractors,
    )


_BETA_CRIT = [
    ("upper", Fraction(196, 100), "2.5%"),
    ("upper", Fraction(233, 100), "1%"),
    ("upper", Fraction(258, 100), "0.5%"),
    ("lower", Fraction(196, 100), "2.5%"),
    ("lower", Fraction(233, 100), "1%"),
]


def _type_two_question(rng, ctx) -> dict:
    dp = ctx["dp"]
    for _ in attempts(5000, "a type II error question"):
        tail, zstar, alpha = rng.choice(_BETA_CRIT)
        n = rng.choice((16, 25, 36, 49, 64, 100))
        mu0 = Fraction(rng.choice(ctx["mu0"]))
        sigma = Fraction(rng.randint(ctx["sd_lo"], ctx["sd_hi"]), 10 ** ctx["sdp"])
        se = sigma / int(math.isqrt(n))
        shift = Fraction(rng.randint(80, 330), 100)  # delta / SE, 2 decimals
        z_beta = zstar - shift
        if not (Fraction(-130, 100) <= z_beta <= Fraction(230, 100)) or z_beta == 0:
            continue
        delta = shift * se
        if (delta * 100).denominator != 1 or delta == 0:
            continue
        sgn = 1 if tail == "upper" else -1
        mu1 = mu0 + sgn * delta
        if mu1 <= 0 or (mu1 * 10**dp).denominator != 1 and (mu1 * 100).denominator != 1:
            continue
        break
    # beta = P(Z < z*-shift) (upper) -- the same for the lower tail by symmetry
    beta = phi_table(z_beta) if z_beta >= 0 else 1 - phi_table(abs(z_beta))
    critical_mean = mu0 + sgn * zstar * se
    # second route: standardise the boundary about mu1
    boundary_z = (critical_mean - mu1) / se * sgn
    check(boundary_z == z_beta, "type II: boundary standardisation disagrees")
    answer_text = _txt(beta, 4)
    power = 1 - beta

    def beta_of(z: Fraction) -> Fraction:
        return phi_table(z)

    cands = [
        (power, "gave-power-for-type-two-error"),
        (
            beta_of(round_half_up(zstar + shift, 2)) if zstar + shift <= Fraction(399, 100) else None,
            "shift-applied-wrong-way",
        ),
        (beta_of(round_half_up(-shift, 2)), "type-two-ignores-critical-value"),
    ]
    distractors = []
    for value, tag in cands:
        if value is None or not (0 < value < 1):
            continue
        t = _txt(value, 4)
        distractors.append(Distractor(t, tag, float(t)))
    distractors = _pick(answer_text, distractors, 4, 3)
    check(len(distractors) >= 2, "too few type II distractors")
    mu_t = _mu_s(ctx, mu0)
    mu1_s = _fmt(mu1, 2) if (mu1 * 10**dp).denominator != 1 else _fmt(mu1, dp)
    lead = _lead(ctx, mu0, sigma, variance=False)
    rule = _reject_rule(tail, zstar)
    unit = ctx["unit"]
    table = table_text(z_beta, rng)
    shape = rng.randrange(3)
    if shape == 0:
        text = (
            f"{lead} A z-test of H0: μ = {mu_t} against H1: {_h1(tail, mu_t)} uses a random sample of {n} {ctx['obj']} and rejects H0 when {rule} (the {alpha} level). "
            f"Suppose the true mean is actually {mu1_s} {unit}. {table} Find the probability that the test fails to reject H0, correct to 4 decimal places."
        )
    elif shape == 1:
        text = (
            f"{lead} The test of H0: μ = {mu_t} against H1: {_h1(tail, mu_t)} takes {n} {ctx['obj']} and rejects H0 when {rule}, the {alpha} level. "
            f"If the true mean is {mu1_s} {unit}, what is the probability of a type II error? {table} Give it correct to 4 decimal places."
        )
    else:
        text = (
            f"{lead} Using a sample of {n} {ctx['obj']}, a z-test rejects H0: μ = {mu_t} in favour of H1: {_h1(tail, mu_t)} when {rule} ({alpha} significance). "
            f"The true mean is in fact {mu1_s} {unit}. {table} Calculate the probability that H0 is not rejected, correct to 4 decimal places."
        )
    return dict(
        text=text,
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[
            f"SE = {_txt(sigma, ctx['sdp'])}/√{n} = {_txt(se, 4)}; H0 is rejected beyond a sample mean of {_txt(critical_mean, 3)} {unit}.",
            f"Under μ = {mu1_s}, the boundary has z = {_txt(z_beta, 2)}.",
            f"P(not reject) = Φ({_txt(z_beta, 2)}) = {answer_text}.",
        ],
        distractors=distractors,
    )


def _inverse_inference_z(p_two: Fraction) -> Fraction:
    """The 2-place z whose one-sided upper tail is `p_two` (used for a distractor): z with Phi(z) = 1 - p_two."""
    q = 1 - float(p_two)
    q = min(max(q, 0.5001), 0.99999)
    lo, hi = 0.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 * math.erfc(-mid / math.sqrt(2)) < q else (lo, mid)
    return round_half_up(Fraction(repr((lo + hi) / 2)), 2)


def _reverse_question(rng, ctx) -> dict:
    dp = ctx["dp"]
    ans_dp = dp
    for _ in attempts(5000, "a reverse p-value question"):
        n = rng.choice((16, 25, 36, 49, 64, 100))
        mu0 = Fraction(rng.choice(ctx["mu0"]))
        sigma = Fraction(rng.randint(ctx["sd_lo"], ctx["sd_hi"]), 10 ** ctx["sdp"])
        se = sigma / int(math.isqrt(n))
        z = Fraction(rng.randint(125, 265), 100)
        above = rng.random() < 0.55
        xbar = mu0 + (z * se if above else -z * se)
        if xbar <= 0 or not _clear(xbar, ans_dp, Fraction(15, 100)) or abs(xbar - mu0) < 3 * Fraction(1, 10**ans_dp):
            continue
        p = 2 * (1 - phi_table(z))
        if p <= 0 or sum(1 for k in range(0, 400) if phi_table(Fraction(k, 100)) == 1 - p / 2) != 1:
            continue
        break
    answer_text = _txt(xbar, ans_dp)
    z_wrong = _inverse_inference_z(p)
    sgn = 1 if above else -1
    cands = [
        (mu0 + sgn * z_wrong * se, "two-sided-p-not-halved"),
        (mu0 - sgn * z * se, "critical-value-wrong-direction"),
        (mu0 + sgn * z * sigma, "sd-used-as-standard-error"),
        (z * se, "inverse-forgets-null-mean"),
    ]
    distractors = _pick(
        answer_text,
        [Distractor(_txt(v, ans_dp), tag, float(_txt(v, ans_dp))) for v, tag in cands],
        ans_dp,
        3,
    )
    check(len(distractors) >= 2, "too few reverse p-value distractors")
    # second route: read z back from the printed p-value and the table, then rebuild the sample mean
    z_back = next(a for a in (Fraction(k, 100) for k in range(0, 400)) if phi_table(a) == 1 - p / 2)
    check(z_back == z, "reverse p-value: z not recovered from the table")
    check(mu0 + sgn * z_back * (sigma / int(math.isqrt(n))) == xbar, "reverse p-value: sample mean not recovered")
    mu_t = _mu_s(ctx, mu0)
    lead = _lead(ctx, mu0, sigma, variance=False)
    table = table_text(z, rng)
    side = "above" if above else "below"
    unit = ctx["unit"]
    shape = rng.randrange(3)
    p_s = _txt(p, 4)
    if shape == 0:
        text = (
            f"{lead} A two-sided z-test of H0: μ = {mu_t} using a random sample of {n} {ctx['obj']} gave a p-value of {p_s}, with the sample mean {side} {mu_t} {unit}. "
            f"{table} Find the sample mean, correct to {ans_dp} decimal place{'s' if ans_dp != 1 else ''}."
        )
    elif shape == 1:
        text = (
            f"{lead} The sample of {n} {ctx['obj']} had a mean {side} {mu_t} {unit}. A two-sided z-test of H0: μ = {mu_t} reported p = {p_s}. "
            f"{table} What was the sample mean? Give it correct to {ans_dp} decimal place{'s' if ans_dp != 1 else ''}."
        )
    else:
        text = (
            f"{lead} A report states only that a two-sided test of H0: μ = {mu_t} on {n} {ctx['obj']} had p = {p_s}, and that the sample mean was {side} {mu_t} {unit}. "
            f"{table} Work out the sample mean, correct to {ans_dp} decimal place{'s' if ans_dp != 1 else ''}."
        )
    return dict(
        text=text,
        answer=float(answer_text),
        answer_text=answer_text,
        answer_kind="number",
        working=[
            f"Each tail has probability {_txt(p / 2, 5)}, so Φ(z) = {_txt(1 - p / 2, 4)} and z = {_txt(z, 2)}.",
            f"x̄ = μ0 {'+' if above else '−'} z·σ/√n = {mu_t} {'+' if above else '−'} {_txt(z, 2)} × {_txt(se, 4)} = {answer_text}.",
        ],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# The topic for the three single-answer tiers


@generator(
    "w5c_z_test_mean",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5c_z_test_mean(rng: random.Random, tier: str = "easy") -> dict:
    """z-test for a mean with known sigma: z (easy), p-value from a table (medium), sample size, type II error or a reversed p-value (hard)."""
    ctx = rng.choice(_CONTEXTS)
    if tier == "easy":
        return _easy_question(rng, ctx)
    if tier == "medium":
        return _medium_question(rng, ctx)
    if tier == "hard":
        pick = rng.random()
        if pick < 0.34:
            return _min_n_question(rng, ctx)
        if pick < 0.67:
            return _type_two_question(rng, ctx)
        return _reverse_question(rng, ctx)
    raise VerificationError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain

_EXAM_TESTS = [
    # (tail, alpha label, critical z*)
    ("upper", "5%", Fraction(1645, 1000)),
    ("upper", "1%", Fraction(2326, 1000)),
    ("lower", "5%", Fraction(1645, 1000)),
    ("lower", "1%", Fraction(2326, 1000)),
    ("two", "5%", Fraction(1960, 1000)),
    ("two", "1%", Fraction(2576, 1000)),
]


def _decision_part(rng, reject: bool, level: str, claim: str, mu_t: str, unit: str) -> dict:
    """The one-mark conclusion as a numbered list of statements; the reply is the number of the correct one."""
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
    position = {t: i + 1 for i, t in enumerate(order)}
    listed = " ".join(f"{i + 1}. {t}" for i, t in enumerate(order))
    return dict(
        prompt=f"State the conclusion of the test at the {level} level. Enter the number of the correct statement. {listed}",
        answer_kind="number",
        answer=position[right],
        answer_text=str(position[right]),
        marks=1,
        working=[f"The correct statement is number {position[right]}: {right}"],
        distractors=[Distractor(str(position[t]), tag, position[t]) for t, tag in wrong],
    )


def _floor_phi_expr(arg: str) -> str:
    """SymPy text for the table value of Phi(arg): the normal distribution function rounded to 4 places."""
    return f"floor(10000*(1 - erfc(({arg})/sqrt(2))/2) + 1/2)/10000"


def _build_exam(rng: random.Random) -> dict | None:
    ctx = rng.choice(_CONTEXTS)
    tail, alpha, zstar = rng.choice(_EXAM_TESTS)
    sign = {"upper": 1, "lower": -1, "two": rng.choice((-1, 1))}[tail]
    reject_target = rng.random() < 0.5
    z_lo, z_hi = (1.3, 3.0) if reject_target else (0.5, 1.5)
    d = _draw(rng, ctx, z_lo=z_lo, z_hi=z_hi, n_choices=_ANY_N, sign=sign, stable_se3=True)
    p = tail_from_table(d.z2, tail)
    alpha_f = Fraction(5, 100) if alpha == "5%" else Fraction(1, 100)
    if abs(p - alpha_f) < alpha_f * Fraction(1, 3):
        return None
    reject = p < alpha_f
    if reject != reject_target and rng.random() < 0.8:
        return None
    if not (Fraction(1, 10000) < p < Fraction(9, 10)):
        return None
    dp, unit = ctx["dp"], ctx["unit"]
    mu_t = _mu_s(ctx, d.mu0)
    boundary = d.mu0 + (1 if d.diff > 0 else -1) * zstar * d.se
    if not _clear(boundary, 2, Fraction(15, 100)):
        return None
    # the follow-through value from the 2-place z must land on the same 2-place boundary
    ft_boundary = d.mu0 + zstar * d.diff / abs(d.z2)
    if round_half_up(ft_boundary, 2) != round_half_up(boundary, 2):
        return None
    z_text, p_text, b_text = _txt(d.z, 2), _txt(p, 4), _txt(boundary, 2)
    sqrt_n = _root(Fraction(d.n))
    a_part = number_part(
        "Calculate the value of the test statistic z, correct to 2 decimal places.",
        d.z,
        2,
        2,
        [f"SE = σ/√n = {_txt(d.se, 4)}.", f"z = ({_fmt(d.xbar, dp)} − {mu_t})/{_txt(d.se, 4)} = {z_text}."],
        [
            (d.diff / d.sigma, "sd-used-as-standard-error"),
            (d.diff / (d.sigma / d.n), "se-divides-by-n"),
            (-d.z, "z-sign-reversed"),
            (d.diff / (d.sigma * sqrt_n), "se-multiplies-by-root-n"),
        ],
    )
    if tail == "two":
        ft_p = f"2*(1 - {_floor_phi_expr('Abs(part_a)')})"
        wrongs_p = [
            (p / 2, "one-tail-for-two-tails"),
            (1 - p, "gave-complement-of-tail"),
            (phi_table_nonneg(d.z2), "gave-cumulative-for-tail"),
        ]
    elif tail == "upper":
        ft_p = f"1 - {_floor_phi_expr('part_a')}"
        wrongs_p = [
            (2 * p, "doubled-for-one-tail"),
            (1 - p, "gave-complement-of-tail"),
            (phi_table_nonneg(d.z2), "gave-cumulative-for-tail"),
            (phi_table_nonneg(d.z2) - Fraction(1, 2), "uses-area-from-the-mean"),
        ]
    else:
        ft_p = _floor_phi_expr("part_a")
        wrongs_p = [
            (2 * p, "doubled-for-one-tail"),
            (1 - p, "gave-complement-of-tail"),
            (phi_table_nonneg(d.z2) - Fraction(1, 2), "uses-area-from-the-mean"),
        ]
    wrongs_p = [(v, t) for v, t in wrongs_p if 0 < v < 1]
    b_part = number_part(
        "Hence find the p-value of the test, using the table. Give it correct to 4 decimal places.",
        p,
        4,
        2,
        [f"p = {_txt(p, 4)}."],
        wrongs_p,
        follow_through=ft_p,
        sensible_range=(0, 1),
    )
    sgn_b = 1 if d.diff > 0 else -1
    side = "above" if d.diff > 0 else "below"
    c_part = number_part(
        f"With n and σ unchanged, find the sample mean {side} {mu_t} {unit} at which H0 would only just be rejected. Give it correct to 2 decimal places.",
        boundary,
        2,
        2,
        [
            f"x̄ = μ0 {'+' if sgn_b > 0 else '−'} z*·σ/√n = {mu_t} {'+' if sgn_b > 0 else '−'} {_zs(zstar)} × {_txt(d.se, 4)} = {b_text}."
        ],
        [
            (d.mu0 - sgn_b * zstar * d.se, "critical-value-wrong-direction"),
            (d.mu0 + sgn_b * zstar * d.sigma, "sd-used-as-standard-error"),
            (d.mu0 + sgn_b * zstar, "critical-value-not-scaled"),
            (d.xbar + sgn_b * zstar * d.se, "critical-mean-from-sample-mean"),
        ],
        follow_through=f"{_mu_s(ctx, d.mu0)} + {float(zstar)}*({_fmt(d.diff, dp)})/Abs(part_a)",
        sensible_range=(0, None),
    )
    if a_part is None or b_part is None or c_part is None:
        return None
    claim = {"upper": "is greater than", "lower": "is less than", "two": "differs from"}[tail]
    d_part = _decision_part(rng, reject, alpha, claim, mu_t, unit)
    h1 = _h1(tail, mu_t)
    stem = (
        f"{_lead(ctx, d.mu0, d.sigma, variance=False)} {_sample_sentence(ctx, d.n, d.xbar)} "
        f"Let μ be the true {ctx['thing']}. A z-test of H0: μ = {mu_t} against H1: {h1} is carried out at the {alpha} level of significance; "
        f"the critical value is z* = {_zs(zstar)}. {table_text(d.z2, rng)}"
    )
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])


@generator("exd_x_w5c_z_test_mean", level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5c_z_test_mean(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: z statistic, p-value from a table, boundary sample mean, conclusion (7 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return retry_build(rng, _build_exam, "a z-test exam chain", tries=400)
