"""Simulated samples, worked as numbers: `sc_simulated_proportions` (easy, medium, hard) and
`sc_simulated_proportions_exam`.

Skill: ``simulated-sample-proportion-calculations``. A population has a known proportion p. A simulation draws many random
samples of size n and records, for each, the sample proportion (the count with the trait over n). Every answer is typed:

* easy: a sample proportion from a count; the mean of several simulated sample proportions from their counts.
* medium: the standard deviation of the sample proportion, sqrt(p(1 - p)/n); the gap between the mean of the simulated
  proportions and p.
* hard: the sample size that gives a stated standard deviation; the new standard deviation when the sample size is
  multiplied by k^2 (it divides by k); the sample size needed to cut the standard deviation from one value to another
  (it grows with the square of the ratio); the upper end of the interval p + 2 sd that holds about 95% of sample
  proportions.
* exam (``sc_simulated_proportions_exam``): (a) the proportion in the first sample, (b) the mean of the simulated
  proportions, (c) the standard deviation of the sample proportion, (d) the upper limit p + 2 sd (follows through from
  (c)) and (e) the standard deviation when the samples are four times as large (follows through from (c)).

Every standard deviation is checked against the exact variance of a binomial count divided by n^2, summed over the whole
binomial distribution, which is a different route from the formula p(1 - p)/n. Every other answer is recomputed from the
printed counts.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction

from ..core import VerificationError, attempts, check, generator
from ._r3b_kit import Redraw, retry
from ._r3d_kit import clear_of_boundary, dec, number_fields, number_part, sqrt_of

F = Fraction

# (population, plural noun, the trait as it follows "that")
_CONTEXTS = (
    ("households in a regional town", "households", "have rooftop solar panels"),
    ("students at a large secondary school", "students", "walk to school"),
    ("seeds in a packet of wildflower mix", "seeds", "germinate"),
    ("light globes from a factory", "globes", "are faulty"),
    ("voters in a council area", "voters", "support a new library"),
    ("tickets sold for a cricket match", "tickets", "are bought online"),
    ("cyclists using a city bike path", "cyclists", "wear a helmet"),
    ("customers at a bakery", "customers", "buy a sourdough loaf"),
)


def _feasible_sd(sd, n) -> bool:
    """True when a sample proportion of samples of n can have this standard deviation: sd = √(p(1 - p)/n) with p between
    0.1 and 0.9, so sd²·n = p(1 - p) lies between 0.09 and 0.25."""
    return F(9, 100) <= sd * sd * n <= F(1, 4)


def _thin_means_redraw(build):
    """A draw whose wrong options come out too few or too close to the key is redrawn, not raised."""

    @functools.wraps(build)
    def run(rng, *args, **kwargs):
        try:
            return build(rng, *args, **kwargs)
        except VerificationError as err:
            if "usable distractors" in str(err):
                raise Redraw(str(err)) from err
            raise

    return run


_PS = tuple(F(k, 100) for k in (10, 15, 20, 25, 30, 35, 40, 45, 55, 60, 65, 70))
_NS = (40, 50, 80, 100, 125, 200, 250, 400)  # divisors of a power of ten, so counts over n have short decimals


def _context(rng):
    return rng.choice(_CONTEXTS)


def _pct(p: F) -> str:
    return f"{int(p * 100)}%"


def _intro(ctx, p, n):
    pop, unit, trait = ctx
    return (
        f"{_pct(p)} of the {pop} {trait}. A simulation draws random samples of {n} {unit} from this population "
        f"and records the proportion of {unit} in each sample that {trait}."
    )


@functools.lru_cache(maxsize=None)
def _binomial_variance(n: int, p: F) -> F:
    """The exact variance of X/n for X binomial(n, p), by summing over every outcome (a second route)."""
    var = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) * (F(k, n) - p) ** 2 for k in range(n + 1))
    check(var == p * (1 - p) / n, "the binomial variance differs from p(1 - p)/n")
    return var


def _sd(n: int, p: F):
    """The standard deviation of the sample proportion as an exact-to-forty-digits Fraction."""
    var = _binomial_variance(n, p)
    return sqrt_of(var)


def _counts(rng, n: int, p: F, m: int) -> list[int]:
    """m believable simulated counts: draws near np with the spread of a binomial, kept between 1 and n - 1."""
    sigma = math.sqrt(n * float(p) * float(1 - p))
    out = []
    for _ in range(m):
        out.append(max(1, min(n - 1, round(n * float(p) + rng.gauss(0, sigma)))))
    return out


def _proportions_ok(counts, n):
    return len(set(counts)) >= max(3, len(counts) - 2)


def _dp3(value) -> bool:
    return clear_of_boundary(value, 3)


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    ctx = _context(rng)
    pop, unit, trait = ctx
    p = rng.choice(_PS)
    n = rng.choice(_NS)
    if rng.random() < 0.5:
        c = max(5, min(n - 5, round(n * float(p) + rng.randint(-6, 6) * math.sqrt(n) / 4)))
        value = F(c, n)
        text = (
            f"{_intro(ctx, p, n)} In one simulated sample, {c} of the {n} {unit} {trait}. "
            f"Find the proportion of {unit} in this sample that {trait}, as a decimal correct to 3 decimal places."
        )
        wrong = [
            (F(n - c, n), "sc-complement-for-proportion"),
            (F(c, n - c), "sc-odds-for-proportion"),
            (F(c, 100), "sc-count-over-hundred"),
        ]
        working = [f"The sample proportion is the count over the sample size: {c}/{n} = {dec(value, 3)}."]
        return dict(text=text, working=working, **number_fields(value, 3, wrong))
    m = rng.randint(6, 9)
    for _ in attempts(500, "a set of counts"):
        counts = _counts(rng, n, p, m)
        mean = F(sum(counts), m * n)
        if _proportions_ok(counts, n) and _dp3(mean):
            break
    listed = ", ".join(str(c) for c in counts)
    props = sorted(F(c, n) for c in counts)
    text = (
        f"{_intro(ctx, p, n)} In {m} simulated samples, the numbers of {unit} that {trait} were {listed}. "
        f"Find the mean of the {m} sample proportions, correct to 3 decimal places."
    )
    wrong = [
        (F(sum(counts), n), "sc-mean-not-divided-by-samples"),
        (F(sum(counts), (m - 1) * n), "sc-divides-by-one-fewer"),
        (props[m // 2] if m % 2 else (props[m // 2 - 1] + props[m // 2]) / 2, "sc-median-for-mean"),
        (F(counts[0], n), "sc-uses-one-sample"),
        (F(sum(counts), m), "sc-mean-count-not-proportion"),
    ]
    working = [
        f"Each sample proportion is its count over {n}. The counts total {sum(counts)}.",
        f"The mean proportion is {sum(counts)}/({m} × {n}) = {sum(counts)}/{m * n} = {dec(mean, 3)}.",
    ]
    return dict(text=text, working=working, **number_fields(mean, 3, wrong))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    ctx = _context(rng)
    pop, unit, trait = ctx
    p = rng.choice(_PS)
    if rng.random() < 0.55:
        for _ in attempts(500, "a clear standard deviation"):
            n = rng.choice((25, 40, 50, 64, 80, 100, 150, 200, 250, 400))
            sd = _sd(n, p)
            if _dp3(sd) and sd > F(1, 200):
                break
        text = f"{_intro(ctx, p, n)} Find the standard deviation of the sample proportion, correct to 3 decimal places."
        pq = p * (1 - p)
        wrong = [
            (pq / n, "sc-sd-forgets-square-root"),
            (sqrt_of(pq) / n, "sc-sd-root-of-numerator-only"),
            (sqrt_of(p / n), "sc-sd-forgets-one-minus-p"),
            (sqrt_of(n * pq), "sc-sd-of-count-for-proportion"),
        ]
        working = [
            f"The standard deviation of the sample proportion is √(p(1 - p)/n) with p = {dec(p, 2)} and n = {n}.",
            f"√({dec(p, 2)} × {dec(1 - p, 2)}/{n}) = √{dec(pq / n, 6)} = {dec(sd, 3)}.",
        ]
        return dict(text=text, working=working, **number_fields(sd, 3, wrong))
    n = rng.choice((50, 100, 200))
    m = rng.randint(6, 8)
    for _ in attempts(800, "a set of sample proportions"):
        counts = _counts(rng, n, p, m)
        mean = F(sum(counts), m * n)
        gap = mean - p
        if _proportions_ok(counts, n) and gap != 0 and abs(gap) >= F(1, 200) and _dp3(gap):
            break
    props = [f"{float(F(c, n)):.3f}" for c in counts]
    listed = ", ".join(props)
    text = (
        f"{_intro(ctx, p, n)} Here are the proportions found in {m} simulated samples: {listed}. "
        f"Find the mean of these sample proportions minus the population proportion {dec(p, 2)}, "
        "correct to 3 decimal places."
    )
    wrong = [
        (-gap, "sc-gap-sign-reversed"),
        (mean, "sc-mean-reported-as-gap"),
        (F(sum(counts), (m - 1) * n) - p, "sc-divides-by-one-fewer"),
        (abs(gap) / p, "sc-gap-as-fraction-of-p"),
    ]
    working = [
        f"Add the {m} proportions and divide by {m}: mean = {dec(mean, 4)}.",
        f"Mean minus population proportion = {dec(mean, 4)} - {dec(p, 2)} = {dec(gap, 3)}.",
    ]
    return dict(text=text, working=working, **number_fields(gap, 3, wrong))


# ------------------------------------------------------------------------------------------------ hard

_SDS = (F(1, 100), F(2, 100), F(25, 1000), F(3, 100), F(4, 100), F(5, 100))


def _hard(rng):
    variant = rng.choice(("sample-size", "scaling", "sample-size-ratio", "interval", "simulated-spread"))
    ctx = _context(rng)
    pop, unit, trait = ctx
    if variant == "sample-size":
        for _ in attempts(2000, "a whole sample size"):
            p = rng.choice(_PS)
            sd = rng.choice(_SDS)
            n = p * (1 - p) / sd**2
            if n.denominator == 1 and 30 <= n <= 1500:
                break
        n = int(n)
        _binomial_variance(n, p)  # second route: the variance of the proportion at this n is p(1 - p)/n
        text = (
            f"{_pct(p)} of the {pop} {trait}. A simulation is to use samples large enough that the standard "
            f"deviation of the sample proportion is {dec(sd, 3)}. Find the sample size needed."
        )
        pq = p * (1 - p)
        wrong = [
            (pq / sd, "sc-n-forgets-to-square-sd"),
            (p / sd**2, "sc-n-forgets-one-minus-p"),
            (sqrt_of(pq) / sd, "sc-n-root-not-squared"),
        ]
        working = [
            "sd = √(p(1 - p)/n), so n = p(1 - p)/sd².",
            f"n = {dec(p, 2)} × {dec(1 - p, 2)}/{dec(sd, 3)}² = {dec(pq, 4)}/{dec(sd**2, 6)} = {n}.",
        ]
        return dict(text=text, working=working, **number_fields(F(n), 0, wrong, nonneg=True))
    if variant == "scaling":
        k = rng.choice((2, 3, 4))
        for _ in attempts(500, "a clean starting spread"):
            n = rng.choice((25, 36, 40, 50, 64, 100))
            s1 = F(rng.randint(2, 9) * k * 5 + rng.randint(0, 4) * k, 1000)
            if _feasible_sd(s1, n):
                break
        new_n = n * k * k
        value = s1 / k
        text = (
            f"In a simulation of {unit} from a population of {pop}, samples of {n} had a standard deviation of "
            f"{dec(s1, 3)} for the sample proportion. The simulation is repeated with samples of {new_n}, "
            f"{k * k} times as large. Find the standard deviation of the sample proportion for the larger samples, "
            "correct to 3 decimal places."
        )
        wrong = [
            (s1 / (k * k), "sc-sd-scales-with-n"),
            (s1 * k, "sc-sd-grows-with-n"),
            (s1 * k * k, "sc-sd-grows-with-n"),
        ]
        working = [
            "The standard deviation of the sample proportion is √(p(1 - p)/n), so it is proportional to 1/√n.",
            f"The sample size is multiplied by {k * k}, so the standard deviation is divided by √{k * k} = {k}: {dec(s1, 3)}/{k} = {dec(value, 3)}.",
        ]
        return dict(text=text, working=working, **number_fields(value, 3, wrong, nonneg=True))
    if variant == "sample-size-ratio":
        j = rng.choice((2, 3, 4, 5))
        n1 = rng.choice((25, 40, 50, 100, 120, 150, 200))
        for _ in attempts(500, "a feasible starting spread"):
            s1 = F(rng.randint(4, 30) * j, 1000)  # a multiple of j/1000, so the target spread is exact too
            if _feasible_sd(s1, n1):
                break
        s2 = s1 / j
        value = n1 * j * j
        text = (
            f"Samples of {n1} {unit} from a population of {pop} gave a standard deviation of {dec(s1, 3)} for the "
            f"sample proportion. How large must the samples be for the standard deviation to fall to {dec(s2, 3)}?"
        )
        wrong = [
            (n1 * j, "sc-n-proportional-to-sd-ratio"),
            (n1 * j * j - n1, "sc-n-increase-not-total"),
        ]
        if n1 % (j * j) == 0:
            wrong.append((F(n1, j * j), "sc-n-ratio-inverted"))
        working = [
            "The standard deviation is proportional to 1/√n, so n is proportional to 1/sd².",
            f"The standard deviation falls by a factor of {j}, so n is multiplied by {j}² = {j * j}: {n1} × {j * j} = {value}.",
        ]
        return dict(text=text, working=working, **number_fields(F(value), 0, wrong, nonneg=True))
    if variant == "simulated-spread":
        n = rng.choice((50, 100))
        p = rng.choice(_PS)
        m = rng.randint(5, 7)
        for _ in attempts(1000, "a clear spread"):
            counts = _counts(rng, n, p, m)
            xs = [F(c, n) for c in counts]
            mean = sum(xs) / m
            var = sum((x - mean) ** 2 for x in xs) / m
            sd = sqrt_of(var)
            if _proportions_ok(counts, n) and _dp3(sd) and sd > F(1, 200):
                break
        listed = ", ".join(f"{float(x):.2f}" for x in xs)
        text = (
            f"{_intro(ctx, p, n)} In {m} simulated samples the sample proportions were {listed}. "
            "Find the standard deviation of these sample proportions, dividing the sum of squared deviations from their "
            "mean by the number of samples. Give the answer correct to 3 decimal places."
        )
        wrong = [
            (sqrt_of(var * m / (m - 1)), "sc-spread-divides-by-one-fewer"),
            (var, "sc-spread-variance-for-sd"),
            (sum(abs(x - mean) for x in xs) / m, "sc-mean-deviation-for-sd"),
            (_sd(n, p), "sc-theory-for-data"),
        ]
        working = [
            f"The mean of the {m} proportions is {dec(mean, 4)}.",
            f"Squared deviations from the mean sum to {dec(var * m, 6)}; dividing by {m} gives the variance {dec(var, 6)}.",
            f"The standard deviation is √{dec(var, 6)} = {dec(sd, 3)}.",
        ]
        return dict(text=text, working=working, **number_fields(sd, 3, wrong, nonneg=True))
    for _ in attempts(500, "a clear upper limit"):
        p = rng.choice(_PS)
        n = rng.choice((50, 80, 100, 150, 200, 250, 400))
        sd = _sd(n, p)
        upper = p + 2 * sd
        if _dp3(upper) and upper < 1:
            break
    text = (
        f"{_intro(ctx, p, n)} About 95% of the sample proportions lie within 2 standard deviations of the "
        "population proportion. Find the upper end of this interval, correct to 3 decimal places."
    )
    wrong = [
        (p + sd, "sc-interval-one-sd"),
        (p + 3 * sd, "sc-interval-three-sd"),
        (2 * sd, "sc-interval-width-not-endpoint"),
        (p + 2 * sd * sd, "sc-interval-uses-variance"),
        (p - 2 * sd, "sc-lower-limit-for-upper"),
    ]
    working = [
        f"The standard deviation of the sample proportion is √({dec(p, 2)} × {dec(1 - p, 2)}/{n}) = {dec(sd, 4)}.",
        f"The upper end is p + 2sd = {dec(p, 2)} + 2 × {dec(sd, 4)} = {dec(upper, 3)}.",
    ]
    return dict(text=text, working=working, **number_fields(upper, 3, wrong, nonneg=True))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("sc_simulated_proportions", level=20, presets=_PRESETS)
@retry
@_thin_means_redraw
def sc_simulated_proportions(rng, tier: str = "easy"):
    """Sample proportions, their mean and spread in simulated samples, and how spread depends on sample size."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam


@generator("sc_simulated_proportions_exam", level=20, presets={"exam": {}})
@retry
@_thin_means_redraw
def sc_simulated_proportions_exam(rng):
    """Exam chain: one sample proportion, the mean of eight, the standard deviation, p + 2 sd, and the sd at 4n."""
    ctx = _context(rng)
    pop, unit, trait = ctx
    m = 8
    for _ in attempts(4000, "a clear chain"):
        p = rng.choice(_PS)
        n = rng.choice((50, 80, 100, 125, 200, 250, 400))
        counts = _counts(rng, n, p, m)
        mean = F(sum(counts), m * n)
        sd = _sd(n, p)
        upper = p + 2 * sd
        sd4 = sd / 2
        # (d) and (e) follow through from the learner's own (c), shown to 3 places: an even number of thousandths keeps
        # half of it exact, and a value close to it keeps p + 2 sd on the same rounding.
        k = round(sd * 1000)
        if (
            _proportions_ok(counts, n)
            and _dp3(mean)
            and _dp3(sd)
            and _dp3(upper)
            and _dp3(sd4)
            and upper < 1
            and k % 2 == 0
            and abs(sd * 1000 - k) < F(1, 4)
            and round(sd4 * 1000) == k // 2
        ):
            break
    check(sd == _sd(n, p), "sd")
    first = F(counts[0], n)
    listed = ", ".join(str(c) for c in counts)
    stem = f"{_intro(ctx, p, n)} In {m} simulated samples, the numbers of {unit} that {trait} were {listed}."
    pq = p * (1 - p)
    sorted_props = sorted(F(c, n) for c in counts)
    pa = number_part(
        f"Find the proportion of {unit} that {trait} in the first sample, as a decimal correct to 3 decimal places.",
        first,
        3,
        1,
        [f"The first count is {counts[0]} out of {n}: {counts[0]}/{n} = {dec(first, 3)}."],
        [
            (F(n - counts[0], n), "sc-complement-for-proportion"),
            (F(counts[0], n - counts[0]), "sc-odds-for-proportion"),
            (F(counts[0], 100), "sc-count-over-hundred"),
            (p, "sc-uses-population-proportion"),
        ],
        sensible=(0, 1),
    )
    pb = number_part(
        f"Find the mean of the {m} sample proportions, correct to 3 decimal places.",
        mean,
        3,
        2,
        [f"The counts total {sum(counts)}, so the mean proportion is {sum(counts)}/({m} × {n}) = {dec(mean, 3)}."],
        [
            (F(sum(counts), n), "sc-mean-not-divided-by-samples"),
            (F(sum(counts), (m - 1) * n), "sc-divides-by-one-fewer"),
            ((sorted_props[m // 2 - 1] + sorted_props[m // 2]) / 2, "sc-median-for-mean"),
            (first, "sc-uses-one-sample"),
        ],
        sensible=(0, 1),
    )
    pc = number_part(
        "Find the standard deviation of the sample proportion, using the population proportion, correct to 3 decimal places.",
        sd,
        3,
        2,
        [f"sd = √(p(1 - p)/n) = √({dec(p, 2)} × {dec(1 - p, 2)}/{n}) = {dec(sd, 3)}."],
        [
            (pq / n, "sc-sd-forgets-square-root"),
            (sqrt_of(pq) / n, "sc-sd-root-of-numerator-only"),
            (sqrt_of(p / n), "sc-sd-forgets-one-minus-p"),
            (sqrt_of(n * pq), "sc-sd-of-count-for-proportion"),
        ],
        sensible=(0, None),
    )
    pd = number_part(
        "About 95% of sample proportions lie within 2 standard deviations of the population proportion. "
        "Use your answer to (c) to find the upper end of this interval, correct to 3 decimal places.",
        upper,
        3,
        1,
        [f"Upper end = p + 2sd = {dec(p, 2)} + 2 × {dec(sd, 3)} = {dec(upper, 3)}."],
        [
            (p + sd, "sc-interval-one-sd"),
            (p + 3 * sd, "sc-interval-three-sd"),
            (2 * sd, "sc-interval-width-not-endpoint"),
            (p - 2 * sd, "sc-lower-limit-for-upper"),
        ],
        follow=f"{dec(p, 2)} + 2*part_c",
        sensible=(0, 2),
    )
    pe = number_part(
        f"The simulation is repeated with samples of {4 * n}, four times as large. "
        "Use your answer to (c) to find the standard deviation of the sample proportion for these samples, correct to 3 decimal places.",
        sd4,
        3,
        1,
        [
            f"Four times the sample size halves the standard deviation (√4 = 2): {dec(sd, 3)}/2 = {dec(sd4, 3)}.",
        ],
        [
            (sd / 4, "sc-sd-scales-with-n"),
            (sd * 2, "sc-sd-grows-with-n"),
            (sd * 4, "sc-sd-grows-with-n"),
        ],
        follow="part_c/2",
        sensible=(0, 1),
    )
    return dict(text=stem, parts=[pa, pb, pc, pd, pe])
