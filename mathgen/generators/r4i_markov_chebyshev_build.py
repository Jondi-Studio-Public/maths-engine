"""Generates `r4i_markov_chebyshev` questions (easy, medium, hard) and the exam chain `exd_x_r4i_markov_chebyshev`.

``uc2-markov-chebyshev-bound`` (``probability``, answer ``fraction``, with a decimal and a whole-number slice in the
hard tier): distribution-free bounds on tail probabilities from a mean and a standard deviation. No stem names an
inequality; the learner decides which information is usable.

* easy: one tail bound. Either P(X >= a) for a non-negative variable with a stated mean (a / mean is not a multiple), or
  P(|X - mean| >= d) from the mean and standard deviation.
* medium: one of three bounds that need a small step first: a guaranteed lower bound for an interval that is not centred
  on the mean (the nearer end fixes the distance); a bound on P(|X| >= a) from the second moment sigma^2 + mean^2; or a
  bound on how far a sample mean of n observations strays (the variance of the mean is sigma^2 / n).
* hard: one of three: the smallest n that guarantees P(|sample mean - mean| < d) >= 1 - alpha; the half-width
  sigma / sqrt(alpha) of the shortest symmetric interval guaranteed to hold X with probability 1 - alpha (a decimal, 2 dp);
  or the better of the bounds from the mean alone and from the mean with the standard deviation, for a non-negative X.
* exam: a waiting time at a service counter. (a) the bound on P(X >= a) from the mean, (b) the bound on P(|X - mean| >= d),
  (c) the guaranteed lower bound for P(|X - mean| < d) (follows through from (b)), (d) the smallest n that guarantees the
  sample mean within e of the mean with probability at least 0.95 (follows through from (b), since sigma^2 = (b) d^2).

All values are exact fractions or a closed-form root; the tests recompute them from the text with SymPy and by checking
that the bound is attained by a two-point distribution (so no smaller bound can be guaranteed).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4i_kit import (
    chain_avoid,
    dec,
    frac_distractors,
    frac_fields,
    int_distractors,
    int_fields,
    need,
    num_distractors,
    num_fields,
    redraw,
    retry,
    stored,
)

TOPIC = "r4i_markov_chebyshev"
EXAM = "exd_x_r4i_markov_chebyshev"

# (the variable, its unit, a non-negative quantity?)
NONNEG = [
    ("The time a customer waits at a Services Australia counter", "minutes"),
    ("The daily rainfall recorded at a Townsville weather station", "mm"),
    ("The size of a file uploaded to a school server", "MB"),
    ("The delivery time of a parcel sent from Adelaide", "hours"),
    ("The daily electricity use of a Hobart household", "kWh"),
]
ANY = [
    ("The error in a bench scale reading", "g"),
    ("The change in a share price over a day", "dollars"),
    ("The difference between a forecast and the actual overnight temperature", "degrees"),
]


def _v(text):
    return text


def _unit_text(value, unit):
    return f"{value:g} {unit}"


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_markov(rng):
    what, unit = rng.choice(NONNEG)
    for _ in attempts(200, "a Markov bound that is not a unit fraction"):
        mean = rng.choice((4, 5, 6, 8, 9, 10, 12, 15, 20, 25))
        a = rng.randint(2 * mean, 6 * mean)
        bound = Fraction(mean, a)
        if bound.numerator != 1 and bound.denominator <= 120:
            break
    cands = [
        (Fraction(a, mean), "r4i-mc-inverted-ratio"),
        (1 - bound, "r4i-mc-complement"),
        (bound * bound, "r4i-mc-squared"),
        (Fraction(mean, a + mean), "r4i-mc-shifted-threshold"),
        (Fraction(mean, a - mean), "r4i-mc-shifted-threshold"),
    ]
    dist = need(frac_distractors(bound, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{what} cannot be negative and has mean {mean} {unit}; nothing else is known about its distribution. "
        f"What is the smallest upper bound that can be guaranteed for the probability that it is at least {a} {unit}?"
    )
    working = [
        "A non-negative variable satisfies P(X ≥ a) ≤ E(X)/a.",
        f"The bound is {mean}/{a} = {bound.numerator}/{bound.denominator}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(bound))


def _easy_chebyshev(rng):
    what, unit = rng.choice(NONNEG + ANY)
    for _ in attempts(200, "a Chebyshev bound that is not a unit fraction"):
        sd = rng.choice((2, 3, 4, 5, 6, 8))
        d = rng.randint(2 * sd - 1, 5 * sd)
        bound = Fraction(sd * sd, d * d)
        if bound.numerator != 1 and bound < 1 and bound.denominator <= 120:
            break
    mean = rng.choice((20, 24, 30, 36, 40, 50, 60))
    cands = [
        (Fraction(sd, d), "r4i-mc-forgot-square"),
        (Fraction(sd * sd, d), "r4i-mc-squared-numerator-only"),
        (1 - bound, "r4i-mc-complement"),
        (Fraction(sd * sd, 2 * d * d), "r4i-mc-halved"),
        (Fraction(d * d, sd * sd), "r4i-mc-inverted-ratio"),
    ]
    dist = need(frac_distractors(bound, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{what} has mean {mean} {unit} and standard deviation {sd} {unit}; its distribution is otherwise unknown. "
        f"What is the smallest upper bound that can be guaranteed for the probability that it differs from its mean by at "
        f"least {d} {unit}?"
    )
    working = [
        f"P(|X - μ| ≥ d) ≤ σ²/d² = {sd}²/{d}² = {bound.numerator}/{bound.denominator}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(bound))


def _easy(rng):
    return rng.choice((_easy_markov, _easy_chebyshev))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium_interval(rng):
    what, unit = rng.choice(NONNEG + ANY)
    for _ in attempts(300, "an off-centre interval"):
        sd = rng.choice((2, 3, 4, 5))
        mean = rng.choice((30, 40, 50, 60, 80))
        d1, d2 = sorted(rng.sample(range(sd + 1, 4 * sd + 1), 2))
        bound = 1 - Fraction(sd * sd, d1 * d1)
        if 0 < bound < 1 and d2 > d1 and bound.denominator <= 100:
            break
    lo_first = rng.random() < 0.5
    lo, hi = (mean - d1, mean + d2) if lo_first else (mean - d2, mean + d1)
    mid = Fraction(d1 + d2, 2)
    cands = [
        (1 - Fraction(sd * sd, d2 * d2), "r4i-mc-farther-end"),
        (1 - Fraction(sd * sd, mid * mid), "r4i-mc-half-width"),
        (Fraction(sd * sd, d1 * d1), "r4i-mc-complement"),
        (1 - Fraction(sd, d1), "r4i-mc-forgot-square"),
    ]
    dist = need(frac_distractors(bound, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{what} has mean {mean} {unit} and standard deviation {sd} {unit}; its distribution is otherwise unknown. "
        f"What is the largest lower bound that can be guaranteed for the probability that it lies strictly between "
        f"{lo} and {hi} {unit}?"
    )
    working = [
        f"The nearer end is {d1} {unit} from the mean, so the interval contains the symmetric one |X - {mean}| < {d1}.",
        f"P(|X - {mean}| < {d1}) ≥ 1 - {sd}²/{d1}² = {bound.numerator}/{bound.denominator}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(bound))


def _medium_second_moment(rng):
    what, unit = rng.choice(ANY)
    for _ in attempts(300, "a second-moment bound below 1"):
        mean = rng.choice((2, 3, 4, 5))
        sd = rng.choice((2, 3, 4, 5, 6))
        second = mean * mean + sd * sd
        a = rng.randint(math.isqrt(second) + 2, math.isqrt(second) + 8)
        bound = Fraction(second, a * a)
        if bound < 1 and bound.numerator != 1 and bound.denominator <= 250:
            break
    cands = [
        (Fraction(sd * sd, a * a), "r4i-mc-variance-for-second-moment"),
        (Fraction(mean * mean, a * a), "r4i-mc-mean-squared-only"),
        (Fraction((mean + sd) ** 2, a * a), "r4i-mc-sum-then-square"),
        (Fraction(second, a), "r4i-mc-forgot-square"),
    ]
    dist = need(frac_distractors(bound, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{what} has mean {mean} {unit} and standard deviation {sd} {unit}; its distribution is otherwise unknown. "
        f"What is the smallest upper bound that can be guaranteed for the probability that its size |X| is at least "
        f"{a} {unit}?"
    )
    working = [
        f"E(X²) = Var(X) + E(X)² = {sd}² + {mean}² = {second}.",
        f"X² is non-negative, so P(X² ≥ {a}²) ≤ E(X²)/{a}² = {second}/{a * a} = {bound.numerator}/{bound.denominator}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(bound))


def _medium_sample_mean(rng):
    what, unit = rng.choice(NONNEG + ANY)
    for _ in attempts(300, "a sample-mean bound that is not a unit fraction"):
        sd = rng.choice((4, 5, 6, 8, 10, 12))
        n = rng.choice((9, 16, 20, 25, 30, 36, 40, 50))
        d = rng.randint(sd // 2 + 1, sd)
        bound = Fraction(sd * sd, n * d * d)
        if bound.numerator != 1 and bound < 1 and bound.denominator <= 250:
            break
    mean = rng.choice((20, 25, 40, 50))
    cands = [
        (Fraction(sd * sd, d * d), "r4i-mc-forgot-sample-size"),
        (Fraction(sd * sd, n * n * d * d), "r4i-mc-n-squared"),
        (Fraction(sd, n * d), "r4i-mc-forgot-square"),
        (1 - bound, "r4i-mc-complement"),
    ]
    dist = need(frac_distractors(bound, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{what} has mean {mean} {unit} and standard deviation {sd} {unit}; its distribution is otherwise unknown. "
        f"The mean of {n} independent observations is taken. What is the smallest upper bound that can be guaranteed for "
        f"the probability that this sample mean differs from {mean} {unit} by at least {d} {unit}?"
    )
    working = [
        f"The sample mean has variance σ²/n = {sd}²/{n}.",
        f"P(|X̄ - {mean}| ≥ {d}) ≤ ({sd}²/{n})/{d}² = {bound.numerator}/{bound.denominator}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(bound))


def _medium(rng):
    return rng.choice((_medium_interval, _medium_second_moment, _medium_sample_mean))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard_sample_size(rng):
    what, unit = rng.choice(NONNEG + ANY)
    for _ in attempts(400, "a sample size away from an integer boundary"):
        sd = rng.choice((6, 8, 10, 12, 15, 20))
        d = rng.choice((2, 3, 4, 5))
        alpha = rng.choice((Fraction(1, 20), Fraction(1, 10), Fraction(1, 100), Fraction(1, 50)))
        exact = Fraction(sd * sd) / (alpha * d * d)
        if exact.denominator != 1 and 20 < exact < 4000:
            break
    n = math.ceil(exact)
    check(Fraction(sd * sd, n * d * d) <= alpha < Fraction(sd * sd, (n - 1) * d * d), "n is not the least")
    z = {Fraction(1, 20): "1.645", Fraction(1, 10): "1.282", Fraction(1, 100): "2.326", Fraction(1, 50): "2.054"}[alpha]
    normal = math.ceil((float(z) * sd / d) ** 2)
    cands = [
        (normal, "r4i-mc-normal-instead-of-chebyshev"),
        (math.floor(exact), "r4i-mc-round-down"),
        (math.ceil(sd * sd / (d * d) / float(alpha) ** 0.5), "r4i-mc-root-of-alpha"),
        (math.ceil(sd / (d * float(alpha))), "r4i-mc-forgot-square"),
        (math.ceil(Fraction(sd * sd, d * d)), "r4i-mc-forgot-alpha"),
    ]
    dist = need(frac_distractors(Fraction(n), [(Fraction(v), t) for v, t in cands], positive=True))
    mean = rng.choice((20, 30, 50, 100))
    pct = int(100 * (1 - alpha))
    text = (
        f"{what} has mean {mean} {unit} and standard deviation {sd} {unit}; its distribution is otherwise unknown. "
        f"How many independent observations are needed so that the probability that their mean lies strictly within "
        f"{d} {unit} of {mean} {unit} is guaranteed to be at least {pct}%? Give the smallest whole number that is certain to work."
    )
    working = [
        f"P(|X̄ - {mean}| ≥ {d}) ≤ {sd}²/(n×{d}²) and we need this ≤ {float(alpha):g}.",
        f"n ≥ {sd * sd}/({float(alpha):g}×{d * d}) = {float(exact):.3f}, so n = {n}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(Fraction(n)))


def _hard_halfwidth(rng):
    what, unit = rng.choice(NONNEG + ANY)
    sd = rng.choice((3, 4, 5, 6, 8, 10, 12, 15))
    k = rng.choice((2, 4, 5, 10))
    alpha = Fraction(1, k * k)
    c = Fraction(k * sd)
    cands = [
        (Fraction(k * k * sd), "r4i-mc-forgot-root"),
        (Fraction(k * sd * sd), "r4i-mc-variance-for-sd"),
        (Fraction(sd, k), "r4i-mc-inverted-root"),
        (Fraction(sd * (k - 1)), "r4i-mc-minus-one"),
        (Fraction(2 * k * sd), "r4i-mc-two-tails-doubled"),
    ]
    dist = need(frac_distractors(c, cands, positive=True))
    mean = rng.choice((20, 30, 50, 100))
    text = (
        f"{what} has mean {mean} {unit} and standard deviation {sd} {unit}; its distribution is otherwise unknown. "
        f"Find the smallest number c such that the probability that the value lies strictly between {mean} - c and "
        f"{mean} + c is guaranteed to be at least {_ft_pct(1 - alpha)}."
    )
    working = [
        f"P(|X - {mean}| ≥ c) ≤ {sd}²/c², and we need this ≤ 1/{k * k}.",
        f"c² ≥ {k * k}×{sd}², so c = {k}×{sd} = {k * sd}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(c))


def _ft_pct(p: Fraction) -> str:
    pct = p * 100
    return f"{pct.numerator}%" if pct.denominator == 1 else f"{float(pct):g}%"


def _hard_best_of_two(rng):
    what, unit = rng.choice(NONNEG)
    for _ in attempts(500, "mean-only and mean-and-sd bounds that differ"):
        mean = rng.choice((10, 12, 15, 20, 25, 30))
        sd = rng.choice((3, 4, 5, 6, 8, 10))
        a = rng.randint(mean + 2, 4 * mean)
        markov = Fraction(mean, a)
        cheb = Fraction(sd * sd, (a - mean) ** 2)
        if (
            markov < 1
            and cheb < 1
            and markov != cheb
            and markov.numerator != 1
            and cheb.numerator != 1
            and cheb.denominator <= 300
        ):
            break
    best = min(markov, cheb)
    worse = max(markov, cheb)
    cands = [
        (worse, "r4i-mc-took-the-weaker-bound"),
        (Fraction(sd * sd, a * a), "r4i-mc-distance-from-zero"),
        (markov + cheb if markov + cheb < 1 else Fraction(markov * cheb), "r4i-mc-combined-bounds"),
        (1 - best, "r4i-mc-complement"),
        (Fraction(sd * sd, 2 * (a - mean) ** 2), "r4i-mc-halved"),
    ]
    dist = need(frac_distractors(best, cands, positive=True, at_most=Fraction(1)))
    text = (
        f"{what} cannot be negative. It has mean {mean} {unit} and standard deviation {sd} {unit}; its distribution is "
        f"otherwise unknown. Using only this information, what is the smallest upper bound you can guarantee for the "
        f"probability that it is at least {a} {unit}?"
    )
    working = [
        f"From the mean alone: P(X ≥ {a}) ≤ {mean}/{a} = {markov.numerator}/{markov.denominator}.",
        f"From the standard deviation: X ≥ {a} means X - {mean} ≥ {a - mean}, so P ≤ {sd}²/{a - mean}² = {cheb.numerator}/{cheb.denominator}.",
        f"The smaller of the two is {best.numerator}/{best.denominator}.",
    ]
    return dict(text=text, working=working, distractors=dist, **frac_fields(best))


def _hard(rng):
    return rng.choice((_hard_sample_size, _hard_halfwidth, _hard_best_of_two))(rng)


@generator(
    TOPIC,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraw
def r4i_markov_chebyshev(rng, tier="easy"):
    """Distribution-free tail bounds from a mean and a standard deviation; see the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------


@generator(EXAM, level=21, presets={"exam": dict(tier="exam")})
@redraw
def exd_x_r4i_markov_chebyshev(rng, tier="exam"):
    """A service counter: Markov bound, Chebyshev bound, its complement, then the bound for a sample mean."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "an exam draw with distinct bounds"):
        mean = rng.choice((8, 10, 12, 15, 20))
        sd = rng.choice((3, 4, 5, 6))
        a = rng.randint(3 * mean, 5 * mean)
        d = rng.randint(2 * sd, 4 * sd)
        n = rng.choice((4, 5, 8, 9, 10, 16, 20, 25))
        ok = Fraction(mean, a)
        ob = Fraction(sd * sd, d * d)
        od = ob / n
        if (
            ok.numerator != 1
            and ob.numerator != 1
            and od.numerator != 1
            and len({ok, ob, od, 1 - ob}) == 4
            and od.denominator <= 300
        ):
            break
    oc = 1 - ob
    order = ["a", "b", "c", "d"]
    canon = {"a": ok, "b": ob, "c": oc, "d": od}
    fns = {"c": lambda s: 1 - s["b"], "d": lambda s: s["b"] / n}
    a_dist = need(
        frac_distractors(
            ok,
            [
                (Fraction(a, mean), "r4i-mc-inverted-ratio"),
                (1 - ok, "r4i-mc-complement"),
                (Fraction(mean, a - mean), "r4i-mc-shifted-threshold"),
                (Fraction(mean, a + mean), "r4i-mc-shifted-threshold"),
            ],
            positive=True,
            at_most=Fraction(1),
        )
    )
    b_dist = need(
        frac_distractors(
            ob,
            [
                (Fraction(sd, d), "r4i-mc-forgot-square"),
                (Fraction(sd * sd, d), "r4i-mc-squared-numerator-only"),
                (Fraction(sd * sd, 2 * d * d), "r4i-mc-halved"),
                (1 - ob, "r4i-mc-complement"),
            ],
            positive=True,
            at_most=Fraction(1),
            avoid=[oc],
        )
    )
    wrong = {"a": [stored(x) for x in a_dist], "b": [stored(x) for x in b_dist]}
    avoid_c = chain_avoid(order, fns, canon, wrong, "c")
    avoid_d = chain_avoid(order, fns, canon, wrong, "d")
    c_dist = need(
        frac_distractors(
            oc,
            [
                (ob, "r4i-mc-forgot-complement"),
                (1 - 2 * ob, "r4i-mc-tails-counted-twice"),
                (oc / 2, "r4i-mc-halved"),
                ((oc + 1) / 2, "r4i-mc-averaged-with-one"),
            ],
            positive=True,
            at_most=Fraction(1),
            avoid=avoid_c,
        )
    )
    d_dist = need(
        frac_distractors(
            od,
            [
                (ob, "r4i-mc-forgot-sample-size"),
                (ob / (n * n), "r4i-mc-n-squared"),
                (
                    ob / Fraction(math.isqrt(n)) if math.isqrt(n) ** 2 == n else Fraction(sd, n * d),
                    "r4i-mc-root-n-for-n",
                ),
                (ob * n, "r4i-mc-multiplied-by-n"),
            ],
            positive=True,
            at_most=Fraction(1),
            avoid=avoid_d,
        )
    )
    text = rng.choice(
        [
            f"The time a customer waits at a Services Australia counter cannot be negative. It has mean {mean} minutes and "
            f"standard deviation {sd} minutes, and nothing else is known about its distribution.",
            f"Nothing is known about the shape of the distribution of X, the waiting time in minutes at a busy Services "
            f"Australia counter, except that X cannot be negative, E(X) = {mean} and the standard deviation is {sd}.",
            f"A manager records how long customers wait, in minutes, at a Services Australia counter. The waits are "
            f"non-negative with an average of {mean} and a spread (standard deviation) of {sd}; no model is assumed for them.",
        ]
    )
    parts = [
        dict(
            prompt=f"Find the smallest upper bound that can be guaranteed for P(X ≥ {a}), the probability a customer waits at least {a} minutes.",
            marks=1,
            sensible_range=(0, 1),
            working=[f"P(X ≥ {a}) ≤ {mean}/{a} = {ok.numerator}/{ok.denominator}."],
            distractors=a_dist,
            **frac_fields(ok),
        ),
        dict(
            prompt=f"Find the smallest upper bound that can be guaranteed for the probability that the wait differs from {mean} minutes by at least {d} minutes.",
            marks=2,
            sensible_range=(0, 1),
            working=[f"P(|X - {mean}| ≥ {d}) ≤ {sd}²/{d}² = {ob.numerator}/{ob.denominator}."],
            distractors=b_dist,
            **frac_fields(ob),
        ),
        dict(
            prompt=f"Hence find the largest lower bound that can be guaranteed for the probability that the wait is strictly between {mean - d} and {mean + d} minutes.",
            marks=1,
            sensible_range=(0, 1),
            follow_through="1 - part_b",
            working=[
                f"This is the complement of the event in part (b): 1 - {ob.numerator}/{ob.denominator} = {oc.numerator}/{oc.denominator}."
            ],
            distractors=c_dist,
            **frac_fields(oc),
        ),
        dict(
            prompt=(
                f"The mean waiting time of {n} independent customers is recorded. Find the smallest upper bound that can be "
                f"guaranteed for the probability that this sample mean differs from {mean} minutes by at least {d} minutes."
            ),
            marks=2,
            sensible_range=(0, 1),
            follow_through=f"part_b/{n}",
            working=[
                f"The sample mean has variance σ²/{n}, so the bound in part (b) is divided by {n}.",
                f"P(|X̄ - {mean}| ≥ {d}) ≤ {ob.numerator}/{ob.denominator} ÷ {n} = {od.numerator}/{od.denominator}.",
            ],
            distractors=d_dist,
            **frac_fields(od),
        ),
    ]
    return dict(text=text, parts=parts)
