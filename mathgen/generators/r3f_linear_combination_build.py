"""Linear combinations of independent random variables: `r3f_linear_combination` and its exam chain.

Skill `linear-combination-random-variables`. For independent X and Y: E(aX + bY) = a E(X) + b E(Y) and
Var(aX + bY) = a^2 Var(X) + b^2 Var(Y), whether the sign between the terms is plus or minus. A sum or difference of
independent normal variables is normal.

Tiers (topic `r3f_linear_combination`), each with a different piece of reasoning:

* easy   -- E(aX + bY), Var(aX + bY) or SD(aX + bY) from stated means, variances or standard deviations;
* medium -- a context where the combination is a total, a difference, n separate copies (variance n sigma^2) or one
            variable multiplied by n (variance n^2 sigma^2); the mean or standard deviation of the result;
* hard   -- a probability for a sum or a difference of independent normal variables: P(total above c), P(X > Y),
            P(|X - Y| < d).

The exam chain (topic `exd_x_r3f_linear_combination`) is (a) the mean of the total of n packets and a carton, (b) its
standard deviation, (c) the probability that the total is over a stated limit, (d) the probability that this carton is
heavier than a second carton with its own mean and standard deviation. Parts (b) to (d) carry `follow_through`.

Every standard deviation is a whole number by construction (Pythagorean triples, perfect-square n), so a learner's
rounded part answers feed later parts exactly. Each probability is computed by erf and cross-checked against Simpson's
rule on the combined normal density; draws near a rounding boundary are discarded.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._r3f_kit import (
    TRIPLES,
    build_until,
    clear_of_boundary,
    number_part,
    phi,
    single,
)


def numeric_normal_prob(lo, hi, mean, sd, n=300) -> float:
    """P(lo < X < hi) for X ~ N(mean, sd^2) by Simpson's rule over the stated range, no erf involved."""
    # Beyond six standard deviations the tail is under 1e-9, so the range is clipped there; 300 panels then agree with erf
    # to better than 1e-8.
    lo = mean - 6 * sd if lo is None else max(lo, mean - 6 * sd)
    hi = mean + 6 * sd if hi is None else min(hi, mean + 6 * sd)
    h = (hi - lo) / (2 * n)

    def pdf(x):
        return math.exp(-((x - mean) ** 2) / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi))

    total = pdf(lo) + pdf(hi)
    for i in range(1, 2 * n):
        total += (4 if i % 2 else 2) * pdf(lo + i * h)
    return total * h / 3


def prob(lo, hi, mean, sd) -> Fraction:
    """P(lo < X < hi) for X ~ N(mean, sd^2) by erf, cross-checked by Simpson's rule."""
    upper = Fraction(1) if hi is None else phi(Fraction(hi - mean) / Fraction(sd))
    lower = Fraction(0) if lo is None else phi(Fraction(lo - mean) / Fraction(sd))
    value = upper - lower
    check(abs(float(value) - numeric_normal_prob(lo, hi, mean, sd)) < 1e-7, "normal probability disagrees")
    return value


def ok4(value) -> bool:
    return clear_of_boundary(value, 4, Fraction(1, 5))


# -- easy ------------------------------------------------------------------------------------------------------


def _signed(coef: int, name: str, first: bool = False) -> str:
    sign = "-" if coef < 0 else ("" if first else "+")
    mag = abs(coef)
    body = name if mag == 1 else f"{mag}{name}"
    return f"{sign}{body}" if first else f" {sign} {body}"


def _combo(a: int, b: int) -> str:
    return _signed(a, "X", True) + _signed(b, "Y")


def _easy_mean(rng):
    a, b = rng.choice((2, 3, 4, 5, -2, -3)), rng.choice((2, 3, 5, -2, -4, -3))
    mx, my = rng.randint(4, 30), rng.randint(4, 30)
    vx, vy = rng.randint(2, 25), rng.randint(2, 25)
    want = a * mx + b * my
    part = number_part(
        f"Find E({_combo(a, b)}).",
        want,
        0,
        1,
        [f"E({_combo(a, b)}) = {a} x {mx} + ({b}) x {my} = {want}."],
        [
            (abs(a) * mx + abs(b) * my, "r3f-lincomb-mean-ignores-the-sign"),
            ((a + b) * (mx + my), "r3f-lincomb-mean-uses-the-sum-of-coefficients"),
            (a * mx - b * my, "r3f-lincomb-mean-flips-the-second-sign"),
            (a * mx + b * my + vx, "r3f-lincomb-mean-adds-a-variance"),
        ],
    )
    stem = (
        f"X and Y are independent random variables with E(X) = {mx}, Var(X) = {vx}, E(Y) = {my} and Var(Y) = {vy}. "
        f"{part['prompt']}"
    )
    return single(stem, part)


def _easy_var(rng):
    a, b = rng.choice((2, 3, 4, 5)), rng.choice((2, 3, 4, -2, -3, -5))
    vx, vy = rng.randint(2, 20), rng.randint(2, 20)
    mx, my = rng.randint(4, 30), rng.randint(4, 30)
    want = a * a * vx + b * b * vy
    part = number_part(
        f"Find Var({_combo(a, b)}).",
        want,
        0,
        2,
        [f"Var({_combo(a, b)}) = {a}^2 x {vx} + ({b})^2 x {vy} = {want}."],
        [
            (a * vx + abs(b) * vy, "r3f-lincomb-variance-coefficient-not-squared"),
            (a * a * vx - b * b * vy, "r3f-lincomb-variances-subtracted"),
            (a * vx + b * vy, "r3f-lincomb-variance-uses-the-signed-coefficient"),
            (a * a * mx + b * b * my, "r3f-lincomb-variance-uses-the-means"),
        ],
        sensible_range=(0, None),
    )
    stem = (
        f"X and Y are independent random variables with E(X) = {mx}, Var(X) = {vx}, E(Y) = {my} and Var(Y) = {vy}. "
        f"{part['prompt']}"
    )
    return single(stem, part)


def _easy_sd(rng):
    for _ in attempts(300, "coefficients and standard deviations with a whole-number result"):
        p, q, r = rng.choice(TRIPLES)
        a = rng.choice([d for d in (1, 2, 3, 4, 5) if p % d == 0])
        b = rng.choice([d for d in (1, 2, 3, 4, 5) if q % d == 0])
        sx, sy = p // a, q // b
        if sx >= 1 and sy >= 1 and (a, b) != (1, 1):
            break
    sign = rng.choice((1, -1))
    mx, my = rng.randint(10, 60), rng.randint(10, 60)
    part = number_part(
        f"Find the standard deviation of {_combo(a, sign * b)}.",
        r,
        0,
        2,
        [f"Var = {a}^2 x {sx}^2 + {b}^2 x {sy}^2 = {r * r}.", f"SD = {r}."],
        [
            (a * sx + b * sy, "r3f-lincomb-adds-standard-deviations"),
            (r * r, "r3f-lincomb-sd-reported-as-variance"),
            (abs(a * sx - b * sy), "r3f-lincomb-subtracts-standard-deviations"),
            (Fraction(repr(math.sqrt(sx * sx + sy * sy))), "r3f-lincomb-ignores-the-coefficients"),
        ],
        sensible_range=(0, None),
    )
    stem = (
        f"X and Y are independent random variables with E(X) = {mx}, SD(X) = {sx}, E(Y) = {my} and SD(Y) = {sy}. "
        f"{part['prompt']}"
    )
    return single(stem, part)


def _easy(rng):
    return rng.choice((_easy_mean, _easy_var, _easy_sd))(rng)


# -- medium ----------------------------------------------------------------------------------------------------

#: (the first quantity, the second, unit, what their sum is, what Y - X is)
PAIRS = (
    (
        "the time a student takes to walk to the bus stop",
        "the time the bus ride takes",
        "minutes",
        "the whole journey",
        "how much longer the bus ride takes than the walk",
    ),
    (
        "the length of the front section of a fence panel",
        "the length of the back section",
        "cm",
        "the length of one assembled panel",
        "how much longer the back section is than the front",
    ),
    (
        "the time Priya takes to complete the first part of a race",
        "the time she takes for the second part",
        "seconds",
        "her finishing time",
        "how much longer the second part takes than the first",
    ),
)


def _pair_sd(rng):
    p, q, r = rng.choice(TRIPLES)
    return (p, q, r) if rng.random() < 0.5 else (q, p, r)


def _medium_pair(rng):
    xd, yd, unit, total, diff = rng.choice(PAIRS)
    sx, sy, r = _pair_sd(rng)
    mx, my = rng.randint(20, 80), rng.randint(20, 80)
    if my <= mx:
        mx, my = my, mx + 1
    stem = (
        f"{xd[0].upper()}{xd[1:]} is normally distributed with mean {mx} {unit} and standard deviation {sx} {unit}. "
        f"{yd[0].upper()}{yd[1:]} is normally distributed with mean {my} {unit} and standard deviation {sy} {unit}. "
        "The two are independent."
    )
    ask = rng.choice(("total_sd", "diff_mean", "diff_sd", "total_mean"))
    if ask == "total_sd":
        part = number_part(
            f"Find the standard deviation of {total}, in {unit}.",
            r,
            0,
            2,
            [f"Var = {sx}^2 + {sy}^2 = {r * r}.", f"SD = {r}."],
            [
                (sx + sy, "r3f-lincomb-adds-standard-deviations"),
                (r * r, "r3f-lincomb-sd-reported-as-variance"),
                (abs(sx - sy), "r3f-lincomb-subtracts-standard-deviations"),
                (max(sx, sy), "r3f-lincomb-uses-the-larger-sd"),
            ],
            sensible_range=(0, None),
        )
    elif ask == "total_mean":
        part = number_part(
            f"Find the mean of {total}, in {unit}.",
            mx + my,
            0,
            1,
            [f"{mx} + {my} = {mx + my}."],
            [
                ((mx + my) / 2, "r3f-lincomb-averages-the-means"),
                (my - mx, "r3f-lincomb-mean-flips-the-second-sign"),
                (mx + my + sx + sy, "r3f-lincomb-mean-adds-the-sds"),
            ],
        )
    elif ask == "diff_mean":
        part = number_part(
            f"Find the mean of {diff}, in {unit}.",
            my - mx,
            0,
            1,
            [f"E(Y - X) = {my} - {mx} = {my - mx}."],
            [
                (mx - my, "r3f-lincomb-mean-reversed-sign"),
                (mx + my, "r3f-lincomb-mean-flips-the-second-sign"),
                (Fraction(my - mx, 2), "r3f-lincomb-averages-the-means"),
            ],
        )
    else:
        part = number_part(
            f"Find the standard deviation of {diff}, in {unit}.",
            r,
            0,
            3,
            [f"Var(Y - X) = Var(Y) + Var(X) = {sy}^2 + {sx}^2 = {r * r}.", f"SD = {r}."],
            [
                (abs(sy - sx), "r3f-lincomb-subtracts-standard-deviations"),
                (Fraction(repr(math.sqrt(abs(sy * sy - sx * sx)))), "r3f-lincomb-variances-subtracted"),
                (sx + sy, "r3f-lincomb-adds-standard-deviations"),
                (r * r, "r3f-lincomb-sd-reported-as-variance"),
            ],
            sensible_range=(0, None),
        )
    return single(f"{stem} {part['prompt']}", part)


def _medium_copies(rng):
    n = rng.choice((4, 9, 16))
    root = math.isqrt(n)
    sigma = rng.randint(2, 12)
    mu = rng.randint(20, 90)
    noun = rng.choice(
        (
            ("tile", "tiles", "g", "A builder stacks {n} tiles. The mass of one tile"),
            ("brick", "bricks", "g", "A wall section uses {n} bricks. The mass of a brick"),
        )
    )
    thing, things, unit, lead = noun
    lead = lead.format(n=n)
    if rng.random() < 0.5:
        # n separate items: variances add
        part = number_part(
            f"Find the standard deviation of the total mass of the {n} {things}, in {unit}. The {things} are independent.",
            sigma * root,
            0,
            3,
            [f"Var(total) = {n} x {sigma}^2 = {n * sigma * sigma}.", f"SD = {sigma} x {root} = {sigma * root}."],
            [
                (n * sigma, "r3f-copies-treated-as-a-multiple"),
                (n * sigma * sigma, "r3f-lincomb-sd-reported-as-variance"),
                (sigma, "r3f-copies-sd-unchanged"),
            ],
            sensible_range=(0, None),
        )
        stem = f"{lead} is normally distributed with mean {mu} {unit} and standard deviation {sigma} {unit}."
    else:
        part = number_part(
            f"A single {thing} is weighed and its mass is multiplied by {n} to estimate the mass of {n} {things}. Find the standard deviation of that estimate, in {unit}.",
            n * sigma,
            0,
            3,
            [f"Var({n}X) = {n}^2 x {sigma}^2.", f"SD = {n} x {sigma} = {n * sigma}."],
            [
                (sigma * root, "r3f-multiple-treated-as-copies"),
                (n * sigma * sigma, "r3f-lincomb-sd-reported-as-variance"),
                (sigma, "r3f-copies-sd-unchanged"),
                (n * n * sigma * sigma, "r3f-multiple-variance-reported-as-sd"),
            ],
            sensible_range=(0, None),
        )
        stem = f"{lead} is normally distributed with mean {mu} {unit} and standard deviation {sigma} {unit}."
    return single(f"{stem} {part['prompt']}", part)


def _medium(rng):
    return rng.choice((_medium_pair, _medium_copies))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard_total(rng):
    for _ in attempts(500, "a total probability clear of a rounding boundary"):
        n = rng.choice((4, 9))
        root = math.isqrt(n)
        mu, sigma = rng.randint(30, 120), rng.randint(2, 10)
        tm, tsd = n * mu, root * sigma
        c = tm + rng.choice((-2, -1, 1, 2, 3)) * tsd + rng.choice((0, 0, sigma // 2))
        p = prob(c, None, tm, tsd)
        if ok4(p) and 0.03 < p < 0.97:
            break
    word = rng.choice((("tomato", "g"), ("orange", "g"), ("potato", "g")))
    stem = (
        f"The mass of a {word[0]} is normally distributed with mean {mu} {word[1]} and standard deviation {sigma} "
        f"{word[1]}, independently of the others. A batch has {n} of them. "
    )
    part = number_part(
        f"Find the probability that the total mass of the {n} is more than {c} {word[1]}, correct to 4 decimal places.",
        p,
        4,
        4,
        [f"Total ~ N({tm}, {tsd}^2): mean {n} x {mu}, SD {sigma} x {root}.", f"P(T > {c}) = {float(p):.4f}."],
        [
            (prob(c, None, tm, sigma), "r3f-total-probability-uses-one-items-sd"),
            (1 - p, "r3f-cdf-gives-complement"),
            (prob(c, None, tm, n * sigma), "r3f-copies-treated-as-a-multiple"),
            (prob(c, None, mu, sigma), "r3f-total-probability-uses-one-items-mean-and-sd"),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    return single(stem + part["prompt"], part)


def _hard_difference(rng):
    for _ in attempts(800, "P(X > Y) clear of a rounding boundary"):
        sx, sy, r = _pair_sd(rng)
        mx = rng.randint(30, 90)
        my = mx + rng.choice((-1, 1)) * rng.randint(1, 2) * (r // 2 or 1)
        p = prob(0, None, mx - my, r)
        if ok4(p) and 0.05 < p < 0.95:
            break
    pairs = (
        ("Lena's time for a 100 m sprint", "Maya's time", "seconds"),
        ("The mass of a mango from farm A", "the mass of a mango from farm B", "g"),
    )
    a, b, unit = rng.choice(pairs)
    stem = (
        f"{a} is normally distributed with mean {mx} {unit} and standard deviation {sx} {unit}; {b} is normally "
        f"distributed with mean {my} {unit} and standard deviation {sy} {unit}. The two are independent. "
    )
    part = number_part(
        "Find the probability that the first is greater than the second, correct to 4 decimal places.",
        p,
        4,
        4,
        [f"D = X - Y ~ N({mx - my}, {r}^2): variances add, {sx}^2 + {sy}^2 = {r * r}.", f"P(D > 0) = {float(p):.4f}."],
        [
            (prob(0, None, mx - my, abs(sx - sy) or 1), "r3f-lincomb-subtracts-standard-deviations"),
            (prob(0, None, mx - my, sx + sy), "r3f-lincomb-adds-standard-deviations"),
            (1 - p, "r3f-cdf-gives-complement"),
            (Fraction(1, 2), "r3f-difference-probability-assumed-half"),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    return single(stem + part["prompt"], part)


def _hard_within(rng):
    for _ in attempts(800, "P(|X - Y| < d) clear of a rounding boundary"):
        sx, sy, r = _pair_sd(rng)
        mx = rng.randint(30, 90)
        my = mx + rng.choice((0, 0, 1)) * (r // 2 or 1)
        d = r * rng.choice((1, 1, 2)) // 1 + rng.choice((0, 1, 2))
        p = prob(-d, d, mx - my, r)
        if ok4(p) and 0.2 < p < 0.97:
            break
    stem = (
        f"Two machines fill tins independently. Machine A fills with mean {mx} mL and standard deviation {sx} mL; "
        f"machine B fills with mean {my} mL and standard deviation {sy} mL, both normally distributed. "
    )
    part = number_part(
        f"Find the probability that the fills of one tin from each machine differ by less than {d} mL, correct to 4 decimal places.",
        p,
        4,
        4,
        [f"D = A - B ~ N({mx - my}, {r}^2).", f"P(-{d} < D < {d}) = {float(p):.4f}."],
        [
            (prob(-d, d, mx - my, abs(sx - sy) or 1), "r3f-lincomb-subtracts-standard-deviations"),
            (prob(-d, d, mx - my, sx + sy), "r3f-lincomb-adds-standard-deviations"),
            (prob(None, d, mx - my, r), "r3f-within-probability-one-sided"),
            (1 - p, "r3f-cdf-gives-complement"),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    return single(stem + part["prompt"], part)


def _hard(rng):
    return rng.choice((_hard_total, _hard_difference, _hard_within))(rng)


@generator(
    "r3f_linear_combination",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3f_linear_combination(rng: random.Random, tier: str = "easy") -> dict:
    """E and Var of aX + bY (easy), totals, differences and copies (medium), probabilities for combinations (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _exam(rng: random.Random) -> dict:
    for _ in attempts(2000, "a carton chain whose probabilities are clear of rounding boundaries"):
        n = rng.choice((4, 9, 16))
        root = math.isqrt(n)
        mu = rng.randint(40, 120)
        # sigma_T = root * sigma must be one leg of a triple so the second carton's sd gives a whole-number combined sd
        p, q, r = rng.choice(TRIPLES)
        if rng.random() < 0.5:
            p, q = q, p  # either leg can be the total's sd
        if p % root:
            continue
        sigma = p // root
        carton = rng.randint(30, 80)
        tm, tsd = n * mu + carton, p
        sd_b = q
        # Any whole offset, not a few fixed multiples of the sd, so the z-scores (and the probabilities) vary widely.
        mu_b = tm + rng.choice((-1, 1)) * rng.randint(max(1, r // 5), (3 * r) // 2)
        limit = tm + rng.randint(-tsd, (9 * tsd) // 5)
        p_over = prob(limit, None, tm, tsd)
        p_heavier = prob(0, None, tm - mu_b, r)
        if (
            ok4(p_over)
            and ok4(p_heavier)
            and 0.02 < p_over < 0.9
            and 0.05 < p_heavier < 0.95
            and limit != tm
            and sigma >= 2
        ):
            break
    stem = phrase(
        rng,
        [
            "A carton holds {n} packets of rice and weighs {carton} g when empty. The mass of a packet is normally "
            "distributed with mean {mu} g and standard deviation {sigma} g, independently of the other packets. "
            "Let T be the total mass in grams of a full carton.",
            "Each carton holds {n} packets of rice and weighs {carton} g when empty. Packet masses are independent "
            "and normally distributed, each with mean {mu} g and standard deviation {sigma} g. Let T be the total "
            "mass in grams of a full carton.",
            "A supplier fills a carton, which holds {n} packets of rice and weighs {carton} g when empty. The mass "
            "of each packet is independent of the others and normally distributed with mean {mu} g and standard "
            "deviation {sigma} g. Let T be the total mass in grams of a full carton.",
        ],
        n=n,
        carton=carton,
        mu=mu,
        sigma=sigma,
    )
    parts = [
        number_part(
            "Find the mean of T, in grams.",
            tm,
            0,
            1,
            [f"E(T) = {n} x {mu} + {carton} = {tm}."],
            [
                (mu * n, "r3f-total-mean-forgets-the-carton"),
                (mu + carton, "r3f-total-mean-one-packet-only"),
                (n * (mu + carton), "r3f-total-mean-carton-counted-per-packet"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find the standard deviation of T, in grams.",
            tsd,
            0,
            2,
            [f"Var(T) = {n} x {sigma}^2 = {n * sigma * sigma}; the carton adds nothing.", f"SD(T) = {tsd}."],
            [
                (n * sigma, "r3f-copies-treated-as-a-multiple"),
                (n * sigma * sigma, "r3f-lincomb-sd-reported-as-variance"),
                (sigma, "r3f-copies-sd-unchanged"),
                (root * sigma + carton, "r3f-total-sd-adds-the-carton"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the probability that a full carton weighs more than {limit} g, correct to 4 decimal places.",
            p_over,
            4,
            3,
            [f"T ~ N({tm}, {tsd}^2).", f"P(T > {limit}) = {float(p_over):.4f}."],
            [
                (prob(limit, None, tm, sigma), "r3f-total-probability-uses-one-items-sd"),
                (1 - p_over, "r3f-cdf-gives-complement"),
                (prob(limit, None, tm, n * sigma), "r3f-copies-treated-as-a-multiple"),
            ],
            follow_through=f"1-(1+erf(({limit}-part_a)/(part_b*sqrt(2))))/2",
            sensible_range=(0, 1),
            within=(0, 1),
        ),
        number_part(
            f"A second carton has a total mass that is normally distributed with mean {mu_b} g and standard deviation "
            f"{sd_b} g, independently of the first. Find the probability that the first carton is heavier than the second, "
            "correct to 4 decimal places.",
            p_heavier,
            4,
            4,
            [
                f"D = T - (second carton) ~ N({tm - mu_b}, {r}^2), since {tsd}^2 + {sd_b}^2 = {r * r}.",
                f"P(D > 0) = {float(p_heavier):.4f}.",
            ],
            [
                (prob(0, None, tm - mu_b, abs(tsd - sd_b) or 1), "r3f-lincomb-subtracts-standard-deviations"),
                (prob(0, None, tm - mu_b, tsd + sd_b), "r3f-lincomb-adds-standard-deviations"),
                (1 - p_heavier, "r3f-cdf-gives-complement"),
                (Fraction(1, 2), "r3f-difference-probability-assumed-half"),
            ],
            follow_through=f"(1+erf((part_a-{mu_b})/(sqrt(part_b**2+{sd_b}**2)*sqrt(2))))/2",
            sensible_range=(0, 1),
            within=(0, 1),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r3f_linear_combination", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r3f_linear_combination(rng: random.Random, tier: str = "exam") -> dict:
    """Mean and standard deviation of a total, a probability for it, then a comparison with a second carton."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
