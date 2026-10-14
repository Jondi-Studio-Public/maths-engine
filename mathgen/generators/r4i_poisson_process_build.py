"""Generates `r4i_poisson_process` questions (easy, medium, hard) and the exam chain `exd_x_r4i_poisson_process`.

``uc2-poisson-process-interarrival`` (``stochastic-processes``, answer ``number``, every question states its rounding):
waiting times and counts for a Poisson process, with the time units deliberately mixed (a rate per hour and minutes, a
rate per minute and seconds). No stem says "exponential", "Erlang", "thinning" or "memoryless".

* easy: the chance the next arrival is more than t away; the same given that a wait of s has already passed; the standard
  deviation of the time to the k-th arrival (minutes, 2 dp).
* medium: the chance the k-th arrival is within t (a Poisson tail sum); exactly m arrivals from two independent sources;
  the chance the next k arrivals take longer than a number of seconds when the rate is per minute.
* hard: the chance exactly two of the next three arrivals come from source A (a race); given n arrivals in a session, the
  chance exactly m fell in its first part (binomial); the chance that some call arrives but none of the urgent kind.
* exam: a call centre. (a) the mean time between calls in minutes, (b) the standard deviation of the time to the k-th call
  (follows through from (a)), (c) the chance of no call in the first t minutes (from (a)), (d) the chance the k-th call
  arrives within t minutes (from (a)). The rate divides 60 to at most two decimals, so (a) is exact and the chain does not
  depend on rounding.

All keys are closed forms in `math.exp`; the tests recompute them from the text with mpmath and by simulating the process.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4i_kit import (
    chain_avoid,
    clean,
    dec,
    need,
    num_distractors,
    num_fields,
    redraw,
    retry,
    stable,
    stored,
)

TOPIC = "r4i_poisson_process"
EXAM = "exd_x_r4i_poisson_process"

# (sentence subject, singular noun)
SCENES = [
    ("Calls to a council helpline", "call"),
    ("Emails to a shared support inbox", "email"),
    ("Customers at a self-serve kiosk", "customer"),
    ("Faults on a long fibre cable", "fault"),
    ("Orders on a small web shop", "order"),
    ("Patients at a country clinic's walk-in desk", "patient"),
]
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}


def pmf(m: int, mu: float) -> float:
    return math.exp(-mu) * mu**m / math.factorial(m)


def cdf(m: int, mu: float) -> float:
    """P(N <= m) for N Poisson(mu)."""
    return sum(pmf(j, mu) for j in range(m + 1))


def _p4(rng_value: float) -> bool:
    return 0.02 < rng_value < 0.98 and stable(rng_value, 4, 0.15)


def _scene(rng):
    return rng.choice(SCENES)


def _rate_time(rng, lo=0.5, hi=3.2):
    """A rate per hour and a time in minutes with mu = rate * t / 60 in [lo, hi]."""
    for _ in attempts(500, "a rate and a time with a middling mean count"):
        lam = rng.choice((2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24, 30))
        t = rng.choice((5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 45, 50, 75, 90, 100))
        mu = lam * t / 60
        if lo <= mu <= hi:
            return lam, t, mu
    raise AssertionError("unreachable")


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_survival(rng):
    subject, noun = _scene(rng)
    for _ in attempts(300, "a survival probability that is stable at 4 places"):
        lam, t, x = _rate_time(rng, 0.4, 3.5)
        p = math.exp(-x)
        if _p4(p):
            break
    cands = [
        (1 - p, "r4i-pp-complement"),
        (x * math.exp(-x), "r4i-pp-pmf-for-wait"),
        (math.exp(-t / lam), "r4i-pp-mean-as-rate"),
        (math.exp(-x / 2), "r4i-pp-halved-exponent"),
    ]
    dist = need(num_distractors(p, cands, 4))
    text = (
        f"{subject} occur at random, independently, at an average rate of {lam} per hour. What is the probability that the "
        f"wait from now until the next {noun} is longer than {t} minutes? Give your answer correct to 4 decimal places."
    )
    working = [
        f"The rate per minute is {lam}/60, so the mean count in {t} minutes is {lam}×{t}/60 = {x:g}.",
        f"The wait exceeds {t} minutes exactly when there is no {noun} in that time: e^(-{x:g}) = {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


def _easy_memoryless(rng):
    subject, noun = _scene(rng)
    for _ in attempts(300, "a conditional wait that is stable"):
        lam, t, x = _rate_time(rng, 0.4, 3.0)
        s = rng.choice((5, 10, 15, 20, 30, 45))
        p = math.exp(-x)
        if _p4(p):
            break
    xs = lam * s / 60
    cands = [
        (math.exp(-(x + xs)), "r4i-pp-forgot-memoryless"),
        (math.exp(-xs), "r4i-pp-waited-time-only"),
        (1 - p, "r4i-pp-complement"),
        (math.exp(-(x + xs)) / (1 - math.exp(-xs)), "r4i-pp-conditioned-on-arrival"),
    ]
    dist = need(num_distractors(p, cands, 4))
    text = (
        f"{subject} occur at random, independently, at an average rate of {lam} per hour. No {noun} has been seen for the "
        f"last {s} minutes. What is the probability that the wait from now until the next {noun} is longer than {t} "
        f"minutes? Give your answer correct to 4 decimal places."
    )
    working = [
        f"The process has no memory, so the {s} minutes already waited do not matter.",
        f"The mean count in the next {t} minutes is {lam}×{t}/60 = {x:g}, so the probability is e^(-{x:g}) = {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


def _easy_kth_sd(rng):
    subject, noun = _scene(rng)
    for _ in attempts(300, "a standard deviation that is stable at 2 places"):
        lam = rng.choice((2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15))
        k = rng.randint(2, 8)
        sd = 60 * math.sqrt(k) / lam
        if stable(sd, 2, 0.15) and math.isqrt(k) ** 2 != k:
            break
    mean = 60 * k / lam
    cands = [
        (mean, "r4i-pp-mean-for-sd"),
        (3600 * k / lam**2, "r4i-pp-variance-for-sd"),
        (math.sqrt(k) / lam, "r4i-pp-units-not-converted"),
        (60 * math.sqrt(k - 1) / lam, "r4i-pp-k-minus-one"),
        (60 / lam, "r4i-pp-single-interarrival-sd"),
    ]
    dist = need(num_distractors(sd, cands, 2))
    text = (
        f"{subject} occur at random, independently, at an average rate of {lam} per hour. Starting now, find the standard "
        f"deviation, in minutes, of the time until the {ORD[k]} {noun}. Give your answer correct to 2 decimal places."
    )
    working = [
        f"The time to the {ORD[k]} {noun} is a sum of {k} independent waits, each with mean and standard deviation 60/{lam} minutes.",
        f"The variances add: SD = √{k} × 60/{lam} = {dec(sd, 2)} minutes.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(sd, 2))


def _easy(rng):
    return rng.choice((_easy_survival, _easy_memoryless, _easy_kth_sd))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium_kth_within(rng):
    subject, noun = _scene(rng)
    for _ in attempts(500, "a k-th arrival probability that is stable"):
        lam, t, x = _rate_time(rng, 1.0, 4.0)
        k = rng.randint(2, 5)
        p = 1 - cdf(k - 1, x)
        if _p4(p):
            break
    cands = [
        (cdf(k - 1, x), "r4i-pp-complement"),
        (pmf(k, x), "r4i-pp-pmf-for-cdf"),
        (1 - cdf(k, x), "r4i-pp-off-by-one-count"),
        (1 - math.exp(-x), "r4i-pp-exponential-for-erlang"),
        (1 - cdf(k - 2, x), "r4i-pp-off-by-one-count"),
    ]
    dist = need(num_distractors(p, cands, 4))
    text = (
        f"{subject} occur at random, independently, at an average rate of {lam} per hour. What is the probability that the "
        f"{ORD[k]} {noun} from now arrives within {t} minutes? Give your answer correct to 4 decimal places."
    )
    terms = " + ".join(f"e^(-{x:g})×{x:g}^{j}/{j}!" for j in range(k))
    working = [
        f"The {ORD[k]} {noun} is within {t} minutes exactly when at least {k} {noun}s occur in {t} minutes; the mean count is {x:g}.",
        f"P = 1 - [{terms}] = {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


def _medium_two_streams(rng):
    subject, noun = _scene(rng)
    for _ in attempts(800, "a merged-count probability that is stable"):
        l1 = rng.choice((2, 3, 4, 5, 6, 8))
        l2 = rng.choice((3, 4, 5, 6, 9, 10, 12))
        t = rng.choice((10, 15, 20, 30, 40, 45))
        mu = (l1 + l2) * t / 60
        if not 1.0 <= mu <= 5.0:
            continue
        m = max(1, round(mu) + rng.choice((-1, 0, 0, 1)))
        p = pmf(m, mu)
        if _p4(p) and p > 0.03:
            break
    mu1, mu2 = l1 * t / 60, l2 * t / 60
    cands = [
        (pmf(m, mu1), "r4i-pp-one-stream-only"),
        (pmf(m, mu2), "r4i-pp-one-stream-only"),
        (pmf(m, mu / 2), "r4i-pp-averaged-rates"),
        (1 - cdf(m - 1, mu), "r4i-pp-at-least"),
        (pmf(m, mu1) * pmf(m, mu2), "r4i-pp-product-of-streams"),
    ]
    dist = need(num_distractors(p, cands, 4))
    text = (
        f"{subject} come from two independent sources. The first produces them at random at an average rate of {l1} per "
        f"hour and the second at an average rate of {l2} per hour. What is the probability that the total number arriving "
        f"during the next {t} minutes is exactly {m}? Give your answer correct to 4 decimal places."
    )
    working = [
        f"Independent random streams combine into one random stream with rate {l1} + {l2} = {l1 + l2} per hour.",
        f"The mean count in {t} minutes is {(l1 + l2)}×{t}/60 = {mu:g}, so P(exactly {m}) = e^(-{mu:g})×{mu:g}^{m}/{m}! = {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


def _medium_seconds(rng):
    subject, noun = _scene(rng)
    for _ in attempts(800, "a seconds-and-minutes probability that is stable"):
        r = rng.choice((0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.5))
        t = rng.choice((30, 45, 60, 90, 120, 150, 180, 240))
        k = rng.randint(2, 5)
        mu = r * t / 60
        if not 0.8 <= mu <= 4.0:
            continue
        p = cdf(k - 1, mu)
        if _p4(p):
            break
    cands = [
        (1 - p, "r4i-pp-complement"),
        (cdf(k, mu), "r4i-pp-at-most-k"),
        (pmf(k - 1, mu), "r4i-pp-pmf-for-cdf"),
        (cdf(k - 1, r * t), "r4i-pp-units-not-converted"),
    ]
    dist = need(num_distractors(p, cands, 4))
    text = (
        f"{subject} occur at random, independently, at an average rate of {r:g} per minute. Starting now, what is the "
        f"probability that the next {k} {noun}s together take longer than {t} seconds to arrive? Give your answer correct "
        f"to 4 decimal places."
    )
    working = [
        f"{t} seconds is {t / 60:g} {'minute' if t == 60 else 'minutes'}, so the mean count in that time is {r:g}×{t / 60:g} = {mu:g}.",
        f"The {k} {noun}s take longer than that exactly when fewer than {k} occur in it: P(N ≤ {k - 1}) = {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


def _medium(rng):
    return rng.choice((_medium_kth_within, _medium_two_streams, _medium_seconds))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard_race(rng):
    subject, noun = _scene(rng)
    for _ in attempts(500, "a race probability that is stable"):
        a = rng.choice((2, 3, 4, 5, 6, 8, 9))
        b = rng.choice((2, 3, 4, 5, 6, 7, 10, 12))
        if a == b:
            continue
        q = a / (a + b)
        p = 3 * q * q * (1 - q)
        if _p4(p) and p > 0.05:
            break
    cands = [
        (q * q * (1 - q), "r4i-pp-forgot-arrangements"),
        (3 * (1 - q) ** 2 * q, "r4i-pp-wrong-source"),
        (q * q, "r4i-pp-ignored-third-arrival"),
        (3 * q * q * (1 - q) + q**3, "r4i-pp-at-least"),
    ]
    dist = need(num_distractors(p, cands, 4))
    text = (
        f"{subject} come from two independent sources, A and B. A produces them at random at an average rate of {a} per "
        f"hour and B at an average rate of {b} per hour. Of the next three {noun}s overall, what is the probability that "
        f"exactly two come from A? Give your answer correct to 4 decimal places."
    )
    working = [
        f"Each new {noun} comes from A with probability {a}/({a}+{b}) = {a}/{a + b}, independently of the others.",
        f"P = 3×({a}/{a + b})²×({b}/{a + b}) = {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


def _hard_given_count(rng):
    subject, noun = _scene(rng)
    for _ in attempts(500, "a conditional count probability that is stable"):
        total = rng.choice((30, 40, 45, 60, 90, 120))
        s = rng.choice((10, 15, 20, 30, 45))
        if s >= total or total % s:
            continue
        n = rng.randint(4, 10)
        q = s / total
        m = max(1, round(n * q) + rng.choice((-1, 0, 0, 1)))
        p = math.comb(n, m) * q**m * (1 - q) ** (n - m)
        if _p4(p) and p > 0.03:
            break
    lam = rng.choice((4, 6, 8, 10, 12))
    mu = n * q
    cands = [
        (math.comb(n, m) * (1 - q) ** m * q ** (n - m), "r4i-pp-wrong-fraction"),
        (q**m * (1 - q) ** (n - m), "r4i-pp-forgot-arrangements"),
        (pmf(m, mu), "r4i-pp-poisson-for-conditional"),
        (sum(math.comb(n, j) * q**j * (1 - q) ** (n - j) for j in range(m, n + 1)), "r4i-pp-at-least"),
        (pmf(m, lam * s / 60), "r4i-pp-ignored-the-condition"),
    ]
    dist = need(num_distractors(p, cands, 4))
    text = (
        f"{subject} occur at random, independently, at an average rate of {lam} per hour. In one {total}-minute session "
        f"exactly {n} {noun}s occurred. Given this, what is the probability that exactly {m} of them occurred in the first "
        f"{s} minutes? Give your answer correct to 4 decimal places."
    )
    working = [
        f"Given {n} occurrences in the session, each is equally likely to fall anywhere in it, independently of the others.",
        f"So the number in the first {s} minutes is Binomial({n}, {s}/{total}): P = C({n},{m})×({s}/{total})^{m}×(1-{s}/{total})^{n - m} = {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


def _hard_thinned(rng):
    subject, noun = _scene(rng)
    for _ in attempts(800, "a thinned probability that is stable"):
        lam, t, x = _rate_time(rng, 0.8, 3.5)
        u = rng.choice((0.1, 0.2, 0.25, 0.3, 0.4))
        p = math.exp(-u * x) - math.exp(-x)
        if _p4(p) and p > 0.05:
            break
    cands = [
        (math.exp(-u * x), "r4i-pp-none-urgent-only"),
        (math.exp(-x), "r4i-pp-no-arrivals-at-all"),
        (math.exp(-(1 - u) * x) - math.exp(-x), "r4i-pp-wrong-type"),
        (math.exp(-u * x) * (1 - math.exp(-x)), "r4i-pp-independence-assumed"),
        (1 - math.exp(-u * x), "r4i-pp-complement"),
    ]
    dist = need(num_distractors(p, cands, 4))
    text = (
        f"{subject} occur at random, independently, at an average rate of {lam} per hour, and each one is independently "
        f"urgent with probability {u:g}. What is the probability that during the next {t} minutes at least one {noun} "
        f"occurs but none of them is urgent? Give your answer correct to 4 decimal places."
    )
    working = [
        f"The mean count in {t} minutes is {x:g}. Urgent ones occur at random with mean {u:g}×{x:g} = {u * x:g}.",
        f"P(none urgent) = e^(-{u * x:g}), which includes the case of no {noun} at all, e^(-{x:g}).",
        f"P = e^(-{u * x:g}) - e^(-{x:g}) = {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


def _hard(rng):
    return rng.choice((_hard_race, _hard_given_count, _hard_thinned))(rng)


@generator(
    TOPIC,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraw
def r4i_poisson_process(rng, tier="easy"):
    """Waiting times and counts for a Poisson process; see the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------


@generator(EXAM, level=21, presets={"exam": dict(tier="exam")})
@redraw
def exd_x_r4i_poisson_process(rng, tier="exam"):
    """A call centre: mean gap, standard deviation of the k-th gap sum, a no-call probability, a k-th call probability."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "an exam draw whose chain is stable"):
        lam = rng.choice((2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24))
        k = rng.randint(2, 4)
        t = rng.choice((5, 6, 8, 10, 12, 15, 20, 25, 30, 40))
        mean = Fraction(60, lam)
        a = float(mean)
        x = t / a
        if not 0.6 <= x <= 3.5:
            continue
        b = clean(math.sqrt(k) * a, 2)
        c = clean(math.exp(-x), 4)
        d = clean(1 - cdf(k - 1, x), 4)
        if (
            stable(math.sqrt(k) * a, 2, 0.15)
            and stable(math.exp(-x), 4, 0.15)
            and stable(1 - cdf(k - 1, x), 4, 0.15)
            and 0.03 < c < 0.97
            and 0.03 < d < 0.97
        ):
            break
    order = ["a", "b", "c", "d"]
    canon = {"a": a, "b": b, "c": c, "d": d}
    fns = {
        "b": lambda s: math.sqrt(k) * s["a"],
        "c": lambda s: math.exp(-t / s["a"]),
        "d": lambda s: 1 - cdf(k - 1, t / s["a"]),
    }
    rounders = {"b": lambda v: clean(v, 2), "c": lambda v: clean(v, 4), "d": lambda v: clean(v, 4)}
    a_dist = need(
        num_distractors(
            a,
            [
                (float(lam), "r4i-pp-rate-for-mean"),
                (1 / lam, "r4i-pp-mean-in-hours"),
                (60.0 * lam, "r4i-pp-rate-times-60"),
                (30.0 / lam, "r4i-pp-half-the-mean"),
            ],
            2,
        )
    )
    wrong = {"a": [float(stored(v)) for v in a_dist]}
    avoid_b = [v for v in chain_avoid(order, fns, canon, wrong, "b", rounders) if v is not None]
    b_dist = need(
        num_distractors(
            b,
            [
                (k * a, "r4i-pp-mean-for-sd"),
                (k * a * a, "r4i-pp-variance-for-sd"),
                (a, "r4i-pp-single-interarrival-sd"),
                (math.sqrt(k - 1) * a, "r4i-pp-k-minus-one"),
            ],
            2,
            avoid=avoid_b,
        )
    )
    wrong["b"] = [float(stored(v)) for v in b_dist]
    avoid_c = [v for v in chain_avoid(order, fns, canon, wrong, "c", rounders) if v is not None]
    c_dist = need(
        num_distractors(
            c,
            [
                (1 - c, "r4i-pp-complement"),
                (x * math.exp(-x), "r4i-pp-pmf-for-wait"),
                (math.exp(-a * t / 60), "r4i-pp-mean-as-rate"),
                (math.exp(-x / 2), "r4i-pp-halved-exponent"),
            ],
            4,
            avoid=avoid_c,
        )
    )
    wrong["c"] = [float(stored(v)) for v in c_dist]
    avoid_d = [v for v in chain_avoid(order, fns, canon, wrong, "d", rounders) if v is not None]
    d_dist = need(
        num_distractors(
            d,
            [
                (cdf(k - 1, x), "r4i-pp-complement"),
                (pmf(k, x), "r4i-pp-pmf-for-cdf"),
                (1 - cdf(k, x), "r4i-pp-off-by-one-count"),
                (1 - math.exp(-x), "r4i-pp-exponential-for-erlang"),
            ],
            4,
            avoid=avoid_d,
        )
    )
    text = rng.choice(
        [
            f"Calls reach a help desk at random, independently of one another, at an average rate of {lam} per hour.",
            f"A help desk receives phone calls at random and independently, at a constant average rate of {lam} calls every hour.",
            f"Incoming calls at a help desk form a random stream with no pattern, averaging {lam} per hour.",
        ]
    )
    xs = f"{x:g}"
    series = " + ".join(f"{xs}^{j}/{j}!" for j in range(k))
    parts = [
        dict(
            prompt="Find the mean time between successive calls, in minutes, correct to 2 decimal places.",
            marks=1,
            sensible_range=(0, None),
            working=[f"The rate is {lam} per hour, so the mean gap is 60/{lam} = {dec(a, 2)} minutes."],
            distractors=a_dist,
            **num_fields(a, 2),
        ),
        dict(
            prompt=f"Find the standard deviation, in minutes, of the time from now until the {ORD[k]} call, correct to 2 decimal places.",
            marks=2,
            sensible_range=(0, None),
            follow_through=f"sqrt({k})*part_a",
            working=[
                f"Each gap has standard deviation {dec(a, 2)} minutes, and the {k} gaps add independently: √{k}×{dec(a, 2)} = {dec(b, 2)}."
            ],
            distractors=b_dist,
            **num_fields(b, 2),
        ),
        dict(
            prompt=f"Find the probability that no call arrives in the first {t} minutes, correct to 4 decimal places.",
            marks=2,
            sensible_range=(0, 1),
            follow_through=f"exp(-{t}/part_a)",
            working=[
                f"The mean number of calls in {t} minutes is {t}/{dec(a, 2)} = {xs}, so P = e^(-{xs}) = {dec(c, 4)}."
            ],
            distractors=c_dist,
            **num_fields(c, 4),
        ),
        dict(
            prompt=f"Find the probability that the {ORD[k]} call arrives within {t} minutes, correct to 4 decimal places.",
            marks=3,
            sensible_range=(0, 1),
            follow_through="1 - exp(-"
            + f"{t}/part_a)*("
            + " + ".join(f"({t}/part_a)**{j}/{math.factorial(j)}" for j in range(k))
            + ")",
            working=[
                f"The {ORD[k]} call is within {t} minutes exactly when at least {k} calls occur; the mean count is {xs}.",
                f"P = 1 - e^(-{xs})×({series}) = {dec(d, 4)}.",
            ],
            distractors=d_dist,
            **num_fields(d, 4),
        ),
    ]
    return dict(text=text, parts=parts)
