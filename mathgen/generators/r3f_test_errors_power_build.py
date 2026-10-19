"""Type I error, Type II error and power of a z-test for a mean: `r3f_test_errors_power` and its exam chain.

Skill `hypothesis-test-errors-power`. A random sample of n values comes from a normal population with known standard
deviation sigma, so the sample mean has standard deviation se = sigma / sqrt(n). The test of H0: mu = mu0 rejects H0 when
the sample mean falls in a critical region fixed by the significance level alpha (the critical z value is stated).

* a Type I error is rejecting H0 when it is true; its probability is alpha;
* a Type II error is keeping H0 when the true mean is mu1; its probability beta is found from the sampling distribution
  of the mean with mu = mu1; the power of the test is 1 - beta.

Tiers (topic `r3f_test_errors_power`), each with a different piece of reasoning:

* easy   -- the critical value of the sample mean from alpha and the critical z; or alpha itself from a stated critical
            value (a normal probability under H0);
* medium -- beta (or the power) at a stated true mean: first the critical value, then a probability under the true mean,
            for an upper-tail, lower-tail or power question;
* hard   -- a two-sided test's beta (two critical values); the power after the sample size is quadrupled; and the least
            sample size that gives a stated power, from the stated z values.

The exam chain (topic `exd_x_r3f_test_errors_power`) is (a) the critical value, (b) beta at a stated true mean, (c) the
power, (d) the power when the sample size is quadrupled. Parts (b) and (c) carry `follow_through`.

Sample sizes are perfect squares and se is a whole number, so every critical value is an exact decimal of at most three
places and a learner's rounded part answers cannot drift. Every probability is computed by erf and cross-checked by
Simpson's rule on the normal density; the least sample size is found both from the formula and by testing each n.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._r3f_kit import ThinOptions, build_until, clear_of_boundary, number_part, phi, single
from .r3f_linear_combination_build import numeric_normal_prob

Z_ONE_TAIL = {5: Fraction("1.645"), 1: Fraction("2.326")}
Z_TWO_TAIL = {5: Fraction("1.96"), 1: Fraction("2.576")}

NS = (9, 16, 25, 36, 49, 64, 100)


def cdf(z) -> Fraction:
    z = Fraction(z)
    value = phi(z)
    check(abs(float(value) - numeric_normal_prob(None, float(z), 0.0, 1.0)) < 1e-7, "normal cdf disagrees")
    return value


def ok4(value) -> bool:
    return clear_of_boundary(value, 4, Fraction(1, 5))


#: (setting, unit, kind, mu0 choices, quantity name for the stem)
CONTEXTS = {
    "upper": (
        (
            "A school introduces a tutoring programme hoping to raise the mean score on a numeracy test, which has "
            "been {mu0} marks. Scores are normally distributed with standard deviation {sigma} marks.",
            "marks",
            range(50, 91),
        ),
        (
            "A dairy changes its feed hoping to raise the mean daily milk yield per cow, which has been {mu0} litres. "
            "Yields are normally distributed with standard deviation {sigma} litres.",
            "litres",
            range(14, 41),
        ),
        (
            "A coach introduces a new training routine hoping to raise the mean standing long jump of a squad, which "
            "has been {mu0} cm. Jumps are normally distributed with standard deviation {sigma} cm.",
            "cm",
            range(180, 261, 10),
        ),
        (
            "A nursery changes its fertiliser hoping to raise the mean height of seedlings after four weeks, which has "
            "been {mu0} mm. Heights are normally distributed with standard deviation {sigma} mm.",
            "mm",
            range(60, 141, 10),
        ),
    ),
    "lower": (
        (
            "A new kettle design is meant to reduce the mean time to boil, which has been {mu0} seconds. Boiling times "
            "are normally distributed with standard deviation {sigma} seconds.",
            "seconds",
            range(150, 221, 10),
        ),
        (
            "A council changes its traffic lights hoping to reduce the mean waiting time at an intersection, which has "
            "been {mu0} seconds. Waiting times are normally distributed with standard deviation {sigma} seconds.",
            "seconds",
            range(40, 91, 5),
        ),
    ),
    "two": (
        (
            "A machine fills flour bags and is set to a mean mass of {mu0} g, with standard deviation {sigma} g. "
            "The owner tests whether the mean has shifted in either direction.",
            "g",
            (250, 500, 750, 1000),
        ),
        (
            "A press cuts steel rods to a mean length of {mu0} mm, with standard deviation {sigma} mm. The "
            "supervisor tests whether the mean length has changed in either direction.",
            "mm",
            (400, 600, 800, 1200),
        ),
    ),
}


def _setting(rng: random.Random, kind: str):
    for _ in attempts(200, "a standard deviation that is a plausible fraction of the mean"):
        n = rng.choice(NS)
        root = math.isqrt(n)
        se = rng.choice((1, 2, 3, 4, 5))
        sigma = se * root
        text, unit, mus = rng.choice(CONTEXTS[kind])
        mu0 = rng.choice(list(mus))
        if 3 * sigma <= mu0:
            break
    level = rng.choice((5, 5, 1))
    z = (Z_TWO_TAIL if kind == "two" else Z_ONE_TAIL)[level]
    return dict(
        n=n,
        root=root,
        se=se,
        sigma=sigma,
        text=text.format(mu0=mu0, sigma=sigma),
        unit=unit,
        mu0=mu0,
        level=level,
        z=z,
        kind=kind,
    )


def zt(z) -> str:
    """A z value as the decimal a learner reads: 1.645."""
    return f"{float(z):g}"


def _hyp(s) -> str:
    sign = {"upper": ">", "lower": "<", "two": "≠"}[s["kind"]]
    return f"H0: μ = {s['mu0']}, H1: μ {sign} {s['mu0']}"


def _intro(s) -> str:
    side = "two-tailed" if s["kind"] == "two" else "one-tailed"
    return (
        f"{s['text']} A random sample of {s['n']} is taken and a {side} z-test of {_hyp(s)} is carried out at the "
        f"{s['level']}% level, for which the critical z value is {zt(s['z'])}."
    )


def _crit(s):
    """(upper critical value or None, lower critical value or None) for the sample mean."""
    mu0, se, z = Fraction(s["mu0"]), Fraction(s["se"]), s["z"]
    if s["kind"] == "upper":
        return mu0 + z * se, None
    if s["kind"] == "lower":
        return None, mu0 - z * se
    return mu0 + z * se, mu0 - z * se


def _beta(s, mu1) -> Fraction:
    """P(keeping H0 | true mean mu1): the sample mean falls outside the critical region."""
    hi, lo = _crit(s)
    se = Fraction(s["se"])
    mu1 = Fraction(mu1)
    below_hi = Fraction(1) if hi is None else cdf((hi - mu1) / se)
    below_lo = Fraction(0) if lo is None else cdf((lo - mu1) / se)
    return below_hi - below_lo


def _shift(rng, s, low=1, high=3):
    """A true mean on the side the alternative points to, delta from mu0 (a whole number of standard errors or half)."""
    delta = rng.randint(low * s["se"], high * s["se"]) if s["se"] > 1 else rng.randint(low, high)
    if s["kind"] == "lower":
        return s["mu0"] - delta
    if s["kind"] == "two":
        return s["mu0"] + rng.choice((-1, 1)) * delta
    return s["mu0"] + delta


def _num(value, places=3) -> str:
    return f"{float(value):.{places}f}"


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_critical(rng):
    s = _setting(rng, rng.choice(("upper", "lower", "two")))
    hi, lo = _crit(s)
    if s["kind"] == "lower":
        key, side, ask = lo, "lower", "the critical value of the sample mean below which H0 is rejected"
        wrong_side = hi if hi is not None else Fraction(s["mu0"]) + s["z"] * s["se"]
    else:
        key, side = hi, "upper"
        ask = (
            "the critical value of the sample mean above which H0 is rejected"
            if s["kind"] == "upper"
            else "the upper critical value of the sample mean"
        )
        wrong_side = Fraction(s["mu0"]) - s["z"] * s["se"]
    mu0, se, sigma, z = Fraction(s["mu0"]), Fraction(s["se"]), Fraction(s["sigma"]), s["z"]
    sign = "-" if s["kind"] == "lower" else "+"
    part = number_part(
        f"Find {ask}, correct to 3 decimal places.",
        key,
        3,
        2,
        [
            f"The standard deviation of the sample mean is {s['sigma']}/√{s['n']} = {s['se']}.",
            f"Critical value = {s['mu0']} {sign} {zt(z)} x {s['se']} = {_num(key)}.",
        ],
        [
            (wrong_side, "r3f-critical-value-on-the-wrong-side"),
            (mu0 + (z * sigma if sign == "+" else -z * sigma), "r3f-critical-value-uses-sigma-not-the-standard-error"),
            (mu0 + (z if sign == "+" else -z), "r3f-critical-value-uses-z-alone"),
            (mu0 + (se if sign == "+" else -se) * (z + 1), "r3f-critical-value-wrong-multiple"),
        ],
        sensible_range=(0, None),
    )
    return single(f"{_intro(s)} {part['prompt']}", part)


def _easy_alpha(rng):
    s = _setting(rng, rng.choice(("upper", "lower")))
    k = rng.choice((Fraction(1), Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3)))
    mu0, se = Fraction(s["mu0"]), Fraction(s["se"])
    cut = mu0 + k * se if s["kind"] == "upper" else mu0 - k * se
    alpha = 1 - cdf(k)
    if not ok4(alpha):
        raise ThinOptions("alpha near a rounding boundary")
    word = "above" if s["kind"] == "upper" else "below"
    sign = ">" if s["kind"] == "upper" else "<"
    stem = (
        f"{s['text']} A random sample of {s['n']} is taken. A test of H0: μ = {s['mu0']} against H1: μ {sign} "
        f"{s['mu0']} rejects H0 when the sample mean is {word} {float(cut):g}."
    )
    part = number_part(
        "Find the probability of a Type I error, correct to 4 decimal places.",
        alpha,
        4,
        3,
        [
            f"A Type I error occurs when H0 is true but the sample mean is {word} {float(cut):g}; then the sample mean has standard deviation {s['sigma']}/√{s['n']} = {s['se']}.",
            f"z = ({float(cut):g} - {s['mu0']})/{s['se']} = {float(k):g}, so the probability is {float(alpha):.4f}.",
        ],
        [
            (1 - alpha, "r3f-type-one-error-gives-the-complement"),
            (2 * alpha, "r3f-type-one-error-doubled-for-a-one-tailed-test"),
            (1 - cdf(abs(cut - mu0) / Fraction(s["sigma"])), "r3f-type-one-error-uses-sigma-not-the-standard-error"),
            (Fraction(1, 20), "r3f-type-one-error-assumes-five-per-cent"),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    return single(f"{stem} {part['prompt']}", part)


def _easy(rng):
    return rng.choice((_easy_critical, _easy_alpha))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium_beta(rng):
    for _ in attempts(500, "beta clear of a rounding boundary"):
        s = _setting(rng, rng.choice(("upper", "lower")))
        mu1 = _shift(rng, s)
        beta = _beta(s, mu1)
        if ok4(beta) and 0.02 < beta < 0.9:
            break
    hi, lo = _crit(s)
    crit = hi if hi is not None else lo
    se = Fraction(s["se"])
    side = "upper" if s["kind"] == "upper" else "lower"
    power = 1 - beta
    sigma = Fraction(s["sigma"])
    mu1f = Fraction(mu1)
    ignores_crit = cdf(abs(mu1f - Fraction(s["mu0"])) / se)  # standardises the true mean against the null mean
    if rng.random() < 0.4:
        part = number_part(
            f"Find the power of the test when the true mean is {mu1}, correct to 4 decimal places.",
            power,
            4,
            4,
            [
                f"Critical value {_num(crit)}; if \u03bc = {mu1} the sample mean has standard deviation {s['se']}.",
                f"Power = 1 - \u03b2 = {float(power):.4f}.",
            ],
            [
                (beta, "r3f-power-reports-beta"),
                (s_wrong_alpha(s), "r3f-power-reports-alpha"),
                (_reject_prob(s, crit, mu1f, sigma), "r3f-beta-uses-sigma-not-the-standard-error"),
                (ignores_crit, "r3f-power-ignores-the-critical-value"),
            ],
            sensible_range=(0, 1),
            within=(0, 1),
        )
    else:
        part = number_part(
            f"Find the probability of a Type II error when the true mean is {mu1}, correct to 4 decimal places.",
            beta,
            4,
            4,
            [
                f"Critical value {_num(crit)}; if \u03bc = {mu1} the sample mean has standard deviation {s['se']}.",
                f"\u03b2 = P(sample mean on the H0 side of {_num(crit)}) = {float(beta):.4f}.",
            ],
            [
                (power, "r3f-beta-reports-the-power"),
                (s_wrong_alpha(s), "r3f-beta-reports-alpha"),
                (1 - _reject_prob(s, crit, mu1f, sigma), "r3f-beta-uses-sigma-not-the-standard-error"),
                (1 - ignores_crit, "r3f-beta-ignores-the-critical-value"),
            ],
            sensible_range=(0, 1),
            within=(0, 1),
        )
    return single(f"{_intro(s)} {part['prompt']}", part)


def _reject_prob(s, crit, mu1, sd) -> Fraction:
    """P(sample mean is in the rejection region of a one-sided test) when it has mean mu1 and standard deviation sd."""
    inside = cdf((crit - mu1) / sd)
    return 1 - inside if s["kind"] == "upper" else inside


def power_one_sided(delta, se_eff, z) -> Fraction:
    """Power of a one-sided z-test: Phi(delta / se - z), delta the distance of the true mean from mu0 towards H1."""
    return cdf(Fraction(delta) / Fraction(se_eff) - Fraction(z))


def s_wrong_alpha(s) -> Fraction:
    return Fraction(s["level"], 100)


def _medium(rng):
    return _medium_beta(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard_two_sided(rng):
    for _ in attempts(500, "a two-sided beta clear of a rounding boundary"):
        s = _setting(rng, "two")
        mu1 = _shift(rng, s, 1, 2)
        beta = _beta(s, mu1)
        if ok4(beta) and 0.05 < beta < 0.95:
            break
    hi, lo = _crit(s)
    se = Fraction(s["se"])
    mu1f = Fraction(mu1)
    part = number_part(
        f"Find the probability of a Type II error when the true mean is {mu1}, correct to 4 decimal places.",
        beta,
        4,
        4,
        [
            f"Critical values {_num(lo)} and {_num(hi)}; H0 is kept when the sample mean lies between them.",
            f"β = P({_num(lo)} < sample mean < {_num(hi)}) with μ = {mu1} and standard deviation {s['se']}: {float(beta):.4f}.",
        ],
        [
            (cdf((hi - mu1f) / se), "r3f-beta-keeps-only-one-critical-value"),
            (1 - beta, "r3f-beta-reports-the-power"),
            (1 - cdf((lo - mu1f) / se), "r3f-beta-keeps-only-one-critical-value"),
            (cdf((hi - Fraction(s["mu0"])) / se) - cdf((lo - Fraction(s["mu0"])) / se), "r3f-beta-uses-the-null-mean"),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    return single(f"{_intro(s)} {part['prompt']}", part)


def _hard_quadrupled(rng):
    for _ in attempts(500, "a quadrupled-sample power clear of a rounding boundary"):
        s = _setting(rng, rng.choice(("upper", "lower")))
        if s["se"] % 2:
            continue
        mu1 = _shift(rng, s, 1, 2)
        s2 = dict(s, n=4 * s["n"], se=s["se"] // 2)
        power = 1 - _beta(s2, mu1)
        if ok4(power) and 0.1 < power < 0.995:
            break
    old_power = 1 - _beta(s, mu1)
    hi2, lo2 = _crit(s2)
    crit2 = hi2 if hi2 is not None else lo2
    part = number_part(
        f"The sample size is increased to {4 * s['n']}, with the same level of significance and critical z value. "
        f"Find the power of the test when the true mean is {mu1}, correct to 4 decimal places.",
        power,
        4,
        4,
        [
            f"The standard deviation of the sample mean is now {s['sigma']}/√{4 * s['n']} = {s2['se']}, so the critical value is {_num(crit2)}.",
            f"Power = {float(power):.4f}.",
        ],
        [
            (old_power, "r3f-power-ignores-the-larger-sample"),
            (1 - power, "r3f-power-reports-beta"),
            (
                power_one_sided(abs(Fraction(mu1) - s["mu0"]), Fraction(repr(s["se"] / math.sqrt(2))), s["z"]),
                "r3f-power-doubles-instead-of-quadruples",
            ),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    return single(f"{_intro(s)} {part['prompt']}", part)


Z_POWER = {80: Fraction("0.842"), 90: Fraction("1.282"), 95: Fraction("1.645")}


def _hard_sample_size(rng):
    for _ in attempts(1000, "a least sample size clear of integer boundaries"):
        sigma = rng.randint(4, 20)
        level = rng.choice((5, 1))
        z_alpha = Z_ONE_TAIL[level]
        power_pct = rng.choice((80, 90, 95))
        z_beta = Z_POWER[power_pct]
        delta = rng.randint(2, max(3, sigma // 2 + 3))
        need = ((z_alpha + z_beta) * sigma / delta) ** 2
        n_min = math.ceil(need)
        if not (6 <= n_min <= 150) or abs(need - round(need)) < Fraction(1, 5):
            continue
        # test each n directly against the stated rule: power at least p  <=>  z_alpha - delta*sqrt(n)/sigma <= -z_beta
        brute = next(m for m in range(1, 400) if (z_alpha + z_beta) * sigma <= delta * math.sqrt(m))
        if brute == n_min:
            break
    mu0 = rng.randint(40, 90)
    side = rng.choice(("upper", "lower"))
    mu1 = mu0 + delta if side == "upper" else mu0 - delta
    sign = ">" if side == "upper" else "<"
    stem = (
        f"A z-test of H0: μ = {mu0} against H1: μ {sign} {mu0} is carried out at the {level}% level (critical z value "
        f"{zt(z_alpha)}) on a sample from a normal population with standard deviation {sigma}. The test is to have a "
        f"probability of at least 0.{power_pct} of rejecting H0 when the true mean is {mu1}; use z = {zt(z_beta)} for the "
        f"{power_pct}% point."
    )
    part = number_part(
        "Find the smallest sample size that achieves this.",
        n_min,
        0,
        4,
        [
            f"Rejecting at the critical value and having power {power_pct}% needs ({zt(z_alpha)} + {zt(z_beta)}) x {sigma}/√n ≤ {delta}.",
            f"n ≥ (({zt(z_alpha)} + {zt(z_beta)}) x {sigma}/{delta})^2 = {float(need):.2f}, so n = {n_min}.",
        ],
        [
            (math.floor(need), "r3f-sample-size-rounds-down"),
            (math.ceil((z_alpha * sigma / delta) ** 2), "r3f-sample-size-ignores-the-power-requirement"),
            (math.ceil((z_alpha + z_beta) * sigma / delta), "r3f-sample-size-forgets-to-square"),
            (math.ceil(((z_alpha + z_beta) * sigma / delta) ** 2 * 2), "r3f-sample-size-uses-two-tails"),
        ],
        sensible_range=(0, None),
    )
    return single(f"{stem} {part['prompt']}", part)


def _hard(rng):
    return rng.choice((_hard_two_sided, _hard_quadrupled, _hard_sample_size))(rng)


@generator(
    "r3f_test_errors_power",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3f_test_errors_power(rng: random.Random, tier: str = "easy") -> dict:
    """Critical values and alpha (easy), beta and power at a true mean (medium), two sides, larger samples, sample size (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _exam(rng: random.Random) -> dict:
    for _ in attempts(2000, "a chain clear of rounding boundaries"):
        s = _setting(rng, "upper")
        if s["se"] % 2:
            continue
        mu1 = _shift(rng, s, 1, 2)
        hi, _lo = _crit(s)
        beta = _beta(s, mu1)
        s4 = dict(s, n=4 * s["n"], se=s["se"] // 2)
        power4 = 1 - _beta(s4, mu1)
        if ok4(beta) and ok4(power4) and 0.05 < beta < 0.9 and 0.1 < power4 < 0.995 and abs(power4 - (1 - beta)) > 0.03:
            break
    se, mu0 = s["se"], s["mu0"]
    z = s["z"]
    power = 1 - beta
    parts = [
        number_part(
            "Find the critical value of the sample mean above which H0 is rejected, correct to 3 decimal places.",
            hi,
            3,
            2,
            [
                f"The standard deviation of the sample mean is {s['sigma']}/√{s['n']} = {se}.",
                f"Critical value = {mu0} + {zt(z)} x {se} = {_num(hi)}.",
            ],
            [
                (Fraction(mu0) - z * se, "r3f-critical-value-on-the-wrong-side"),
                (Fraction(mu0) + z * s["sigma"], "r3f-critical-value-uses-sigma-not-the-standard-error"),
                (Fraction(mu0) + z, "r3f-critical-value-uses-z-alone"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the probability of a Type II error when the true mean is {mu1}, correct to 4 decimal places.",
            beta,
            4,
            3,
            [
                f"If μ = {mu1} the sample mean has standard deviation {se}.",
                f"β = P(sample mean < {_num(hi)}) = {float(beta):.4f}.",
            ],
            [
                (power, "r3f-beta-reports-the-power"),
                (s_wrong_alpha(s), "r3f-beta-reports-alpha"),
                (cdf((hi - Fraction(mu1)) / Fraction(s["sigma"])), "r3f-beta-uses-sigma-not-the-standard-error"),
                (cdf((Fraction(mu1) - hi) / se), "r3f-beta-uses-the-wrong-tail-of-the-true-mean"),
            ],
            follow_through=f"(1+erf((part_a-{mu1})/({se}*sqrt(2))))/2",
            sensible_range=(0, 1),
            within=(0, 1),
        ),
        number_part(
            f"Find the power of the test when the true mean is {mu1}, correct to 4 decimal places.",
            power,
            4,
            1,
            ["Power = 1 - β."],
            [
                (beta, "r3f-power-reports-beta"),
                (s_wrong_alpha(s), "r3f-power-reports-alpha"),
                (Fraction(1, 2), "r3f-power-assumed-half"),
            ],
            follow_through="1-part_b",
            sensible_range=(0, 1),
            within=(0, 1),
        ),
        number_part(
            f"The sample size is increased to {4 * s['n']}, with the same level of significance and critical z value. "
            f"Find the new power when the true mean is {mu1}, correct to 4 decimal places.",
            power4,
            4,
            4,
            [
                f"The standard deviation of the sample mean is now {s4['se']}, so the critical value is {_num(_crit(s4)[0])}.",
                f"Power = {float(power4):.4f}.",
            ],
            [
                (power, "r3f-power-ignores-the-larger-sample"),
                (1 - power4, "r3f-power-reports-beta"),
                ((power + power4) / 2, "r3f-power-doubles-instead-of-quadruples"),
            ],
            sensible_range=(0, 1),
            within=(0, 1),
        ),
    ]
    return dict(text=_intro(s), parts=parts)


@generator("exd_x_r3f_test_errors_power", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r3f_test_errors_power(rng: random.Random, tier: str = "exam") -> dict:
    """Critical value, Type II error, power, then power with a larger sample."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
