"""Generates `u2_clt_sample_mean` questions and the exam chain `exd_x_u2_clt_sample_mean`.

``ucv1-clt-sample-mean-probability`` (``statistics-probability``, answer ``number``): probabilities for a sample mean
or a sample total, and the sample size needed for a stated precision, using the central limit theorem. The population
is described only by its mean and standard deviation, so nothing but the theorem justifies the normal model.

* easy: one tail of the sample mean, P(mean > c) or P(mean < c), for n a perfect square (the standard error is exact).
* medium: either the probability that the total of n observations exceeds a limit (mean n mu, sd sigma sqrt n), or
  the probability that the sample mean lies within d of mu (a two-sided area).
* hard: the smallest n so that the sample mean is within d of mu with a stated confidence, or so that an upper tail
  is at most a stated size: n = ceil((z sigma / d)^2), with the rounding up checked against n - 1.
* exam: a crate of apples. (a) the standard error, (b) P(mean < c) (follows through from (a), through ``erf``),
  (c) P(total > T) (follows through from (a): the sd of the total is n times the standard error), (d) the smallest
  n for a two-sided 95% bound (follows through from (a): sigma = standard error x sqrt(n)).

Each probability is computed with ``math.erf`` and re-derived with mpmath's normal CDF; questions are redrawn when
the 4-decimal-place value sits near a rounding boundary. Every size n is also confirmed by trying n - 1.
"""

from __future__ import annotations

import math
from fractions import Fraction

import mpmath as mp

from ..core import VerificationError, attempts, check, generator, rational_text
from . import _exd_x_sc4_kit
from ._exd_x_sc4_kit import txt
from ._u2_kit import Redraw, redraws


def num_part(*args, **kwargs):
    """The shared number-part builder, with a shortage of usable distractors turned into a redraw."""
    try:
        return _exd_x_sc4_kit.num_part(*args, **kwargs)
    except VerificationError as error:
        raise Redraw(str(error)) from error


def count_part(*args, **kwargs):
    """The shared count-part builder, with a shortage of usable distractors turned into a redraw."""
    try:
        return _exd_x_sc4_kit.count_part(*args, **kwargs)
    except VerificationError as error:
        raise Redraw(str(error)) from error


TOPIC = "u2_clt_sample_mean"
EXAM_TOPIC = "exd_x_u2_clt_sample_mean"

_SQRT2 = math.sqrt(2.0)

# (population phrase, item, quantity, unit, means, sds)
CONTEXTS = [
    (
        "The masses of Pink Lady apples from an orchard in Batlow",
        "apples",
        "mass",
        "g",
        (170, 180, 190, 200, 210),
        (20, 24, 30, 36),
    ),
    (
        "The waiting times at a regional hospital's emergency department",
        "patients",
        "waiting time",
        "minutes",
        (25, 30, 40, 45, 60),
        (12, 15, 20, 24),
    ),
    (
        "The daily electricity use of households in Adelaide",
        "households",
        "electricity use",
        "kWh",
        (16, 18, 20, 22, 24),
        (4, 6, 8),
    ),
    (
        "The lengths of calls to a Canberra service centre",
        "calls",
        "length",
        "minutes",
        (6, 8, 10, 12, 15),
        (4, 5, 6, 8),
    ),
    ("The weekly rainfall at a station near Cairns", "weeks", "rainfall", "mm", (40, 50, 60, 80), (24, 30, 36, 40)),
]
SIZES = (16, 25, 36, 49, 64, 81, 100)
Z_TABLE = {"95": ("0.95", "1.96"), "99": ("0.99", "2.576"), "90": ("0.90", "1.645"), "98": ("0.98", "2.326")}


def phi(z) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / _SQRT2))


def phi_check(z) -> float:
    """The same value from mpmath's normal CDF, used as a second route."""
    with mp.workdps(30):
        return float(mp.ncdf(mp.mpf(float(z))))


def prob(z) -> Fraction:
    """Phi(z) as an exact Fraction of the float, after agreeing with the second route."""
    first, second = phi(z), phi_check(z)
    check(abs(first - second) < 1e-12, "the normal CDF disagrees between routes")
    return Fraction(first)


def safe_rounding(p: Fraction) -> bool:
    """True when the 4 dp rounding is not sensitive to table or calculator differences."""
    scaled = float(p) * 10000
    return abs((scaled - math.floor(scaled)) - 0.5) > 0.06 and 0.0005 < p < 0.9995


def _plain(value) -> str:
    """A whole number as an integer, otherwise a short decimal."""
    value = Fraction(value)
    return str(int(value)) if value.denominator == 1 else f"{float(value):.2f}".rstrip("0").rstrip(".")


def _z(value) -> str:
    """A z value as the decimal in the table ("1.645"), never as a fraction."""
    return f"{float(Fraction(value)):.3f}".rstrip("0").rstrip(".")


def _amt(value, unit) -> str:
    """'1 minute' but '2 minutes', '1.5 minutes'."""
    return f"{_plain(value)} {unit[:-1] if unit.endswith('s') and Fraction(value) == 1 else unit}"


def _ctx(rng):
    population, items, quantity, unit, mus, sigmas = rng.choice(CONTEXTS)
    return population, items, quantity, unit, rng.choice(mus), rng.choice(sigmas)


def _verb(population):
    """'has' for a singular quantity (use, rainfall), 'have' for a plural one."""
    return "has" if ("rainfall" in population or "electricity use" in population) else "have"


def _describe(population, mu, sigma, unit):
    return f"{population} {_verb(population)} mean {mu} {unit} and standard deviation {sigma} {unit}; the distribution is not known to be normal."


def _z_value(rng, lo=-2.5, hi=2.5):
    """A z-value that is a multiple of 0.25 and not 0."""
    while_values = [Fraction(k, 4) for k in range(int(lo * 4), int(hi * 4) + 1) if k != 0]
    return rng.choice(while_values)


def _question(text, p, working, cands, *, signed=False):
    cands = [(w, m) for w, m in cands if 0.001 < w < 0.999]
    part = num_part("", p, 4, 1, working, cands, key_places=True, signed=signed)
    out = dict(text=text, working=working)
    for key in ("answer_kind", "answer", "answer_text", "answer_latex", "distractors"):
        out[key] = part[key]
    return out


def _tail_cands(z_wrong, p, extra=()):
    return [
        (prob(z_wrong[0]), "clt-sd-not-reduced"),
        (prob(z_wrong[1]), "clt-sd-divided-by-n"),
        (1 - p, "clt-tail-reversed"),
        *extra,
    ]


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(300, "a tail probability away from a rounding boundary"):
        population, items, quantity, unit, mu, sigma = _ctx(rng)
        n = rng.choice(SIZES)
        root = math.isqrt(n)
        se = Fraction(sigma, root)
        z = _z_value(rng, -2.25, 2.25)
        c = mu + z * se
        upper = rng.random() < 0.5
        p = prob(z)
        p = (1 - p) if upper else p
        if safe_rounding(p) and c.denominator in (1, 2, 4, 5, 10, 20, 25) and c > 0:
            break
    zs = (c - mu) / sigma  # forgot to divide sigma by sqrt n
    zn = (c - mu) * n / sigma  # divided sigma by n instead of sqrt n
    zv = (c - mu) * root / (sigma * sigma)  # used the variance where the sd belongs
    side = "more than" if upper else "less than"
    sign = 1 if upper else -1
    cands = [
        (prob(sign * -zs) if not upper else prob(-zs), "clt-sd-not-reduced"),
        (prob(-zn) if upper else prob(zn), "clt-sd-divided-by-n"),
        (1 - p, "clt-tail-reversed"),
        (prob(-zv) if upper else prob(zv), "clt-variance-as-sd"),
    ]
    text = (
        f"{_describe(population, mu, sigma, unit)} A random sample of {n} {items} is taken. "
        f"Use the central limit theorem to find, correct to 4 decimal places, the probability that the sample mean "
        f"{quantity} is {side} {_amt(c, unit)}."
    )
    working = [
        f"The sample mean is approximately normal with mean {mu} and standard deviation {sigma}/√{n} = {_amt(se, unit)}.",
        f"z = ({_plain(c)} - {mu})/{_plain(se)} = {_plain(z)}.",
        f"The probability is {txt(p, 4)}.",
    ]
    return _question(text, p, working, cands)


# -- medium --------------------------------------------------------------------------------------------------------


def _medium_total(rng):
    for _ in attempts(300, "a total-exceeds probability away from a rounding boundary"):
        population, items, quantity, unit, mu, sigma = _ctx(rng)
        n = rng.choice(SIZES)
        root = math.isqrt(n)
        z = _z_value(rng, -2.25, 2.25)
        total = n * mu + z * sigma * root
        upper = rng.random() < 0.6
        p = prob(z)
        p = (1 - p) if upper else p
        if safe_rounding(p) and total.denominator in (1, 2, 4) and total > 0:
            break
    sd_total = sigma * root
    cands = [
        (prob(-(total - n * mu) / sigma) if upper else prob((total - n * mu) / sigma), "clt-total-sd-sigma"),
        (
            prob(-(total - n * mu) / (n * sigma)) if upper else prob((total - n * mu) / (n * sigma)),
            "clt-total-sd-n-sigma",
        ),
        (1 - p, "clt-tail-reversed"),
        (prob(-(total - mu) / sd_total) if upper else prob((total - mu) / sd_total), "clt-total-mean-not-scaled"),
    ]
    side = "exceeds" if upper else "is less than"
    text = (
        f"{_describe(population, mu, sigma, unit)} A random sample of {n} {items} is taken. "
        f"Find, correct to 4 decimal places, the approximate probability that the total {quantity} of the sample "
        f"{side} {_plain(total)} {unit}."
    )
    working = [
        f"The total of {n} observations is approximately normal with mean {n}×{mu} = {n * mu} and standard deviation {sigma}√{n} = {sd_total} {unit}.",
        f"z = ({_plain(total)} - {n * mu})/{sd_total} = {_plain(z)}.",
        f"The probability is {txt(p, 4)}.",
    ]
    return _question(text, p, working, cands)


def _medium_within(rng):
    for _ in attempts(300, "a two-sided probability away from a rounding boundary"):
        population, items, quantity, unit, mu, sigma = _ctx(rng)
        n = rng.choice(SIZES)
        root = math.isqrt(n)
        se = Fraction(sigma, root)
        z = abs(_z_value(rng, 0.5, 2.5))
        d = z * se
        p = 2 * prob(z) - 1
        if safe_rounding(p) and d.denominator in (1, 2, 4, 5, 10, 20, 25):
            break
    one_tail = prob(z)
    cands = [
        (one_tail, "clt-two-sided-one-tail"),
        (2 * prob(d / sigma) - 1, "clt-sd-not-reduced"),
        (1 - p, "clt-tail-reversed"),
        (2 * prob(d * n / sigma) - 1, "clt-sd-divided-by-n"),
    ]
    text = (
        f"{_describe(population, mu, sigma, unit)} A random sample of {n} {items} is taken. "
        f"Find, correct to 4 decimal places, the approximate probability that the sample mean {quantity} is within "
        f"{_amt(d, unit)} of the population mean."
    )
    working = [
        f"The sample mean is approximately normal with mean {mu} and standard deviation {sigma}/√{n} = {_amt(se, unit)}.",
        f"Within {_plain(d)} means |mean - {mu}| < {_plain(d)}, so |z| < {_plain(d)}/{_plain(se)} = {_plain(z)}.",
        f"P(|Z| < {_plain(z)}) = 2Φ({_plain(z)}) - 1 = {txt(p, 4)}.",
    ]
    return _question(text, p, working, cands)


def _medium(rng):
    return rng.choice([_medium_total, _medium_within])(rng)


# -- hard ----------------------------------------------------------------------------------------------------------


def _smallest_n(z: Fraction, sigma, d) -> int:
    """The least n with d sqrt(n)/sigma >= z, found by trying n = 1, 2, ... (bounded)."""
    for n in range(1, 4000):
        if (d * d * n) >= (z * z * sigma * sigma):
            return n
    raise AssertionError("no sample size found below 4000")


def _hard(rng):
    for _ in attempts(400, "a sample size well away from an integer boundary"):
        population, items, quantity, unit, mu, sigma = _ctx(rng)
        key = rng.choice(["95", "99", "90", "98", "up05", "up01", "up10"])
        if key.startswith("up"):
            z = {"up05": Fraction("1.645"), "up01": Fraction("2.326"), "up10": Fraction("1.282")}[key]
        else:
            z = Fraction(Z_TABLE[key][1])
        d = rng.choice([1, 2, 3, 4, 5]) * Fraction(sigma, 10) * rng.choice([1, 2])
        exact = (z * sigma / d) ** 2
        frac = float(exact - math.floor(exact))
        if 6 <= exact <= 600 and 0.1 < frac < 0.9 and d.denominator in (1, 2, 5):
            break
    n = math.ceil(exact)
    check(n == _smallest_n(z, sigma, d), "the sample size disagrees with trying n = 1, 2, ...")
    check(d * d * (n - 1) < z * z * sigma * sigma, "n - 1 should be too small")
    if key.startswith("up"):
        level = {"up05": "0.05", "up01": "0.01", "up10": "0.10"}[key]
        ask = (
            f"Find the smallest sample size n for which the probability that the sample mean {quantity} exceeds the "
            f"population mean by more than {_amt(d, unit)} is at most {level}."
        )
        other = Fraction({"up05": "1.96", "up01": "2.576", "up10": "1.645"}[key])
        z_text = f"P(Z > {_z(z)}) = {level}"
    else:
        level = Z_TABLE[key][0]
        ask = (
            f"Find the smallest sample size n for which the probability that the sample mean {quantity} is within "
            f"{_amt(d, unit)} of the population mean is at least {level}."
        )
        other = Fraction({"95": "1.645", "99": "1.96", "90": "1.282", "98": "2.576"}[key])
        z_text = f"P(|Z| < {_z(z)}) = {level}"
    cands = [
        (math.floor(exact), "clt-n-not-rounded-up"),
        (math.ceil(z * sigma / d), "clt-n-forgot-square"),
        (math.ceil((other * sigma / d) ** 2), "clt-n-wrong-z"),
        (math.ceil(((z * sigma) ** 2) / d), "clt-n-forgot-square"),
    ]
    text = f"{_describe(population, mu, sigma, unit)} Use the central limit theorem. {ask}"
    working = [
        f"We need {z_text}, so the standard error σ/√n = {sigma}/√n must satisfy {_plain(d)} ≥ {_z(z)}×{sigma}/√n.",
        f"Then √n ≥ {_z(z)}×{sigma}/{_plain(d)}, so n ≥ {float(exact):.3f}.",
        f"Rounding up, n = {n} (n = {n - 1} is too small).",
    ]
    part = count_part("", n, 1, working, [(w, m) for w, m in cands if w >= 1], lo=1)
    out = dict(text=text, working=working)
    for k in ("answer_kind", "answer", "answer_text", "answer_latex", "distractors"):
        out[k] = part[k]
    return out


@generator(
    TOPIC,
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraws
def u2_clt_sample_mean(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=18, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_u2_clt_sample_mean(rng, tier="exam"):
    """A crate of apples: standard error, a sample-mean tail, a total tail, then the sample size for a 95% bound."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(400, "an exam question away from every rounding boundary"):
        mu, sigma = rng.choice([170, 180, 190, 200, 210]), rng.choice([20, 24, 30, 36])
        n = rng.choice([16, 25, 36, 49, 64])
        root = math.isqrt(n)
        se = Fraction(sigma, root)
        z_b = -abs(_z_value(rng, 0.5, 2.25))
        c = mu + z_b * se
        z_c = _z_value(rng, 0.5, 2.25)
        total = n * mu + z_c * sigma * root
        p_b = prob(z_b)
        p_c = 1 - prob(z_c)
        d = rng.choice([2, 3, 4, 5])
        exact = (Fraction("1.96") * sigma / d) ** 2
        frac = float(exact - math.floor(exact))
        if (
            (se * 100).denominator == 1
            and safe_rounding(p_b)
            and safe_rounding(p_c)
            and c.denominator in (1, 2, 4)
            and total.denominator in (1, 2, 4)
            and 0.1 < frac < 0.9
            and exact < 800
        ):
            break
    n_d = math.ceil(exact)
    check(n_d == _smallest_n(Fraction("1.96"), sigma, d), "part (d) disagrees with trying n = 1, 2, ...")
    sd_total = sigma * root

    mu_s, c_s, total_s = str(mu), _plain(c), _plain(total)
    part_a = num_part(
        "Find the standard deviation of the sample mean (the standard error), in grams.",
        se,
        2,
        2,
        [f"σ/√n = {sigma}/√{n} = {_plain(se)}."],
        [
            (sigma, "clt-sd-not-reduced"),
            (Fraction(sigma, n), "clt-sd-divided-by-n"),
            (sigma * sigma / Fraction(root), "clt-variance-as-sd"),
            (sigma * root, "clt-total-sd-sigma"),
        ],
        key_places=True,
    )
    part_b = num_part(
        f"Find, correct to 4 decimal places, the approximate probability that the mean mass of the {n} apples is less than {c_s} g.",
        p_b,
        4,
        3,
        [f"z = ({c_s} - {mu})/{_plain(se)} = {_plain(z_b)}.", f"P(mean < {c_s}) = Φ({_plain(z_b)}) = {txt(p_b, 4)}."],
        [
            (prob((c - mu) / sigma), "clt-sd-not-reduced"),
            (1 - p_b, "clt-tail-reversed"),
            (prob((c - mu) * n / sigma), "clt-sd-divided-by-n"),
            (prob(-(c - mu) / se), "clt-tail-reversed"),
        ],
        follow_through=f"(1 + erf((({c_s}) - {mu})/part_a/sqrt(2)))/2",
        key_places=True,
    )
    part_c = num_part(
        f"Find, correct to 4 decimal places, the approximate probability that the total mass of the {n} apples exceeds {total_s} g.",
        p_c,
        4,
        3,
        [
            f"The total has mean {n}×{mu} = {n * mu} g and standard deviation {n}×{_plain(se)} = {sd_total} g.",
            f"z = ({total_s} - {n * mu})/{sd_total} = {_plain(z_c)}; P(total > {total_s}) = {txt(p_c, 4)}.",
        ],
        [
            (1 - prob((total - n * mu) / sigma), "clt-total-sd-sigma"),
            (1 - prob((total - n * mu) / (n * sigma)), "clt-total-sd-n-sigma"),
            (1 - p_c, "clt-tail-reversed"),
            (1 - prob((total - mu) / sd_total), "clt-total-mean-not-scaled"),
        ],
        follow_through=f"(1 - erf((({total_s}) - {n * mu})/({n}*part_a)/sqrt(2)))/2",
        key_places=True,
    )
    part_d = count_part(
        f"The orchard wants the sample mean to be within {d} g of the population mean with probability at least 0.95. "
        "Find the smallest sample size that achieves this.",
        n_d,
        3,
        [
            f"1.96×σ/√n ≤ {d}, so √n ≥ 1.96×{sigma}/{d} and n ≥ {float(exact):.3f}.",
            f"Rounding up gives n = {n_d}.",
        ],
        [
            (math.floor(exact), "clt-n-not-rounded-up"),
            (math.ceil(Fraction("1.96") * sigma / d), "clt-n-forgot-square"),
            (math.ceil((Fraction("1.645") * sigma / d) ** 2), "clt-n-wrong-z"),
            (math.ceil((Fraction("2.576") * sigma / d) ** 2), "clt-n-wrong-z"),
        ],
        follow_through=f"ceiling((1.96*part_a*{root}/{d})**2)",
        lo=1,
    )
    parts = [part_a, part_b, part_c, part_d]
    for part in parts[1:3]:
        part["sensible_range"] = (0, 1)
    return dict(
        text=rng.choice(
            [
                f"The mass of an apple from an orchard in Batlow has mean {mu} g and standard deviation {sigma} g. "
                f"The distribution is not known to be normal. A crate holds {n} apples, which may be treated as a "
                "random sample.",
                f"Apples from a Batlow orchard are packed in crates. The mass of one apple follows an unknown "
                f"distribution with mean {mu} g and standard deviation {sigma} g. A crate holds {n} apples, chosen "
                "at random.",
                f"At an orchard in Batlow the masses of apples have mean {mu} g and standard deviation {sigma} g, "
                f"and the shape of the distribution is unknown. Each crate holds {n} apples picked at random.",
            ]
        ),
        parts=parts,
    )
