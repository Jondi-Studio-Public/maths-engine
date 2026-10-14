"""Generates `r4i_clt_sums` questions (easy, medium, hard) and the exam chain `exd_x_r4i_clt_sums`.

``uc2-clt-approximation-sums`` (``statistics``, answer ``number``): the normal approximation to a SUM of independent
identically distributed discrete counts, where the continuity correction decides the answer. (The first-year
``ucv1-clt-sample-mean-probability`` handles means and totals of continuous measurements; this skill is about
integer-valued totals and the inverse problems they lead to.)

* easy: a total of n independent counts whose distribution is given as a table. The learner finds the mean and variance
  of one count from the table, scales to the total (mean n m, variance n v) and finds P(S >= k) or P(S <= k) with the
  continuity correction, to 4 decimal places.
* medium: a Poisson total (n days, rate L per day, so the total has mean and variance nL) and one of six events whose
  wording fixes the continuity-corrected boundaries: at least, more than, at most, fewer than, between inclusive,
  strictly between.
* hard: an inverse problem, one of two kinds. (i) The smallest whole number c with P(S > c) <= alpha for a discrete total
  (the boundary is c + 1/2). (ii) The largest number of passengers n so that the total mass of n people exceeds a load
  limit with probability at most alpha: m n + z s sqrt(n) <= L, a quadratic in the square root of n.
* exam: parcels left at each stop of a courier round. (a) the mean of the total, (b) its standard deviation (follows
  nothing), (c) P(S >= k) to 4 dp (follows through from (a) and (b)), (d) the smallest c with P(S > c) <= 0.05 (follows
  through from (a) and (b)).

Probabilities use ``math.erf``; the tests re-derive each with mpmath and by summing the exact distribution. A question is
redrawn when its 4 dp value sits near a rounding boundary or its inverse answer near an integer boundary.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4i_kit import (
    chain_avoid,
    clean,
    dec,
    int_distractors,
    int_fields,
    need,
    num_distractors,
    num_fields,
    phi,
    redraw,
    retry,
    stable,
    stored,
)

TOPIC = "r4i_clt_sums"
EXAM = "exd_x_r4i_clt_sums"
INF = math.inf

# (setting, trials, description of one count, what is counted, counted values)
PMF_CONTEXTS = [
    ("A courier van is loaded for a round in Perth.", "stops", "left at a stop", "parcels", (0, 1, 2, 3)),
    (
        "A bakery in Fremantle sells loaves one customer at a time.",
        "customers",
        "bought by a customer",
        "loaves",
        (1, 2, 3, 4),
    ),
    ("A library self-checkout in Hobart records each visit.", "visits", "borrowed in a visit", "books", (0, 1, 2, 3)),
    (
        "A ferry ticket machine at Circular Quay serves one passenger at a time.",
        "passengers",
        "bought by a passenger",
        "tickets",
        (1, 2, 3, 4),
    ),
]
POISSON_CONTEXTS = [
    ("The number of calls to a Darwin helpline each day", "day", "days", "calls"),
    ("The number of faults logged on a Brisbane tram line each week", "week", "weeks", "faults"),
    ("The number of customers entering a Geelong cafe each hour", "hour", "hours", "customers"),
    ("The number of emails arriving at a Canberra office each day", "day", "days", "emails"),
]
EVENTS = ("at least", "more than", "at most", "fewer than", "between", "strictly between")


def _pmf(rng, values):
    """Probabilities that are multiples of 0.05, each at least 0.1, summing to 1."""
    for _ in attempts(200, "a table with a readable variance"):
        k = len(values)
        cuts = sorted(rng.sample(range(2, 20), k - 1))
        probs = [Fraction(b - a, 20) for a, b in zip([0] + cuts, cuts + [20])]
        mean = sum(v * p for v, p in zip(values, probs))
        var = sum(v * v * p for v, p in zip(values, probs)) - mean * mean
        if min(probs) >= Fraction(1, 10) and var >= Fraction(3, 5):
            return probs, mean, var
    raise AssertionError("unreachable")


def _table_text(values, probs):
    return ", ".join(f"P(X = {v}) = {float(p):.2f}" for v, p in zip(values, probs))


def _cdf(lo, hi, mu, sd):
    top = 1.0 if hi == INF else phi((hi - mu) / sd)
    bottom = 0.0 if lo == -INF else phi((lo - mu) / sd)
    return top - bottom


def _event(kind, a, b):
    """(phrase, inclusive integer bounds (lo, hi)) for the event named by `kind`."""
    if kind == "at least":
        return f"is at least {a}", (a, INF)
    if kind == "more than":
        return f"is more than {a}", (a + 1, INF)
    if kind == "at most":
        return f"is at most {a}", (-INF, a)
    if kind == "fewer than":
        return f"is fewer than {a}", (-INF, a - 1)
    if kind == "between":
        return f"is between {a} and {b} inclusive", (a, b)
    return f"is strictly between {a} and {b}", (a + 1, b - 1)


def _shift(bound, delta):
    return bound if bound in (INF, -INF) else bound + delta


def _correct(bounds):
    lo, hi = bounds
    return _shift(lo, -0.5), _shift(hi, 0.5)


def _event_candidates(kind, a, b, bounds, mu, sd, sd_one, p):
    """Wrong probabilities for one event: no correction, the correction the wrong way, strictness ignored."""
    lo, hi = bounds
    out = [
        (_cdf(lo, hi, mu, sd), "r4i-clt-no-continuity-correction"),
        (_cdf(_shift(lo, 0.5), _shift(hi, -0.5), mu, sd), "r4i-clt-continuity-wrong-way"),
    ]
    if kind in ("more than", "fewer than", "strictly between"):
        inc = {"more than": (a, INF), "fewer than": (-INF, a), "strictly between": (a, b)}[kind]
        lo2, hi2 = _correct(inc)
        out.append((_cdf(lo2, hi2, mu, sd), "r4i-clt-strict-inequality-ignored"))
    out.append((_cdf(*_correct(bounds), mu, sd_one), "r4i-clt-sd-of-one-observation"))
    out.append((_cdf(*_correct(bounds), mu, sd * sd), "r4i-clt-variance-as-sd"))
    out.append((1 - p, "r4i-clt-tail-reversed"))
    return out


def _bounds_text(bounds):
    lo, hi = _correct(bounds)
    if hi == INF:
        return f"P(S ≥ {lo})"
    if lo == -INF:
        return f"P(S ≤ {hi})"
    return f"P({lo} ≤ S ≤ {hi})"


def _pick_event(rng, mu, sd, kinds):
    """An event and integer bounds placed near the mean whose probability is stable at 4 dp."""
    for _ in attempts(400, "an event with a stable 4 dp probability"):
        kind = rng.choice(kinds)
        z1 = rng.choice([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        a = round(mu + z1 * sd)
        b = a + rng.randint(2, max(3, round(1.6 * sd)))
        if kind in ("between", "strictly between"):
            a = round(mu - rng.choice([0.5, 1.0, 1.5]) * sd)
            b = round(mu + rng.choice([0.5, 1.0, 1.5]) * sd)
            if b - a < 4:
                continue
        phrase, bounds = _event(kind, a, b)
        p = _cdf(*_correct(bounds), mu, sd)
        if stable(p, 4) and 0.02 < p < 0.98:
            return kind, a, b, phrase, bounds, p
    raise AssertionError("unreachable")


# -- easy ------------------------------------------------------------------------------------------------------


def _easy(rng):
    setting, ones, desc, what, values = rng.choice(PMF_CONTEXTS)
    probs, m, v = _pmf(rng, values)
    n = rng.choice((40, 48, 60, 75, 80, 100))
    mu, var = float(n * m), float(n * v)
    sd = math.sqrt(var)
    kind, a, b, phrase, bounds, p = _pick_event(rng, mu, sd, ("at least", "at most"))
    sd_one = math.sqrt(float(v))
    cands = _event_candidates(kind, a, b, bounds, mu, sd, sd_one, p)
    dist = need(num_distractors(p, cands, 4, at_most=1.0))
    text = (
        f"{setting} The number of {what} {desc} is a random variable X with {_table_text(values, probs)}, and the "
        f"{ones} are independent. Over {n} {ones}, let S be the total number of {what}. "
        f"Use a normal approximation, with a continuity correction, to estimate the probability that S {phrase}. "
        f"Give the answer correct to 4 decimal places."
    )
    working = [
        f"E(X) = {float(m):.2f} and Var(X) = {float(v):.4f}, so S has mean {n}×{float(m):.2f} = {mu:.2f} and variance {n}×{float(v):.4f} = {var:.2f}.",
        f"S takes whole values, so use the boundary {_bounds_text(bounds)} with the continuity correction.",
        f"z = ({_correct(bounds)[0] if _correct(bounds)[1] == INF else _correct(bounds)[1]} - {mu:.2f})/{sd:.4f}; the probability is {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


# -- medium ----------------------------------------------------------------------------------------------------


def _medium(rng):
    setting, one, ones, what = rng.choice(POISSON_CONTEXTS)
    lam = rng.choice((1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 6.0))
    n = rng.choice((20, 24, 30, 36, 40, 50))
    mu = n * lam
    sd = math.sqrt(mu)
    kind, a, b, phrase, bounds, p = _pick_event(rng, mu, sd, EVENTS)
    cands = _event_candidates(kind, a, b, bounds, mu, sd, math.sqrt(lam), p)
    dist = need(num_distractors(p, cands, 4, at_most=1.0))
    text = (
        f"{setting} has a Poisson distribution with mean {lam:g}, independently from one {one} to the next. "
        f"Let S be the total number of {what} over {n} {ones}. Use a normal approximation to S, with a continuity "
        f"correction, to estimate the probability that S {phrase}. Give the answer correct to 4 decimal places."
    )
    working = [
        f"The mean and the variance of S are both {n}×{lam:g} = {mu:g}, so the standard deviation is {sd:.4f}.",
        f"S is a whole number, so the event becomes {_bounds_text(bounds)} after the continuity correction.",
        f"Standardising with mean {mu:g} and standard deviation {sd:.4f} gives the probability {dec(p, 4)}.",
    ]
    return dict(text=text, working=working, distractors=dist, **num_fields(p, 4))


# -- hard ------------------------------------------------------------------------------------------------------

Z_TAIL = {"0.05": 1.645, "0.01": 2.326, "0.10": 1.282, "0.025": 1.96}


def _hard_smallest_c(rng):
    setting, one, ones, what = rng.choice(POISSON_CONTEXTS)
    lam = rng.choice((2.0, 2.5, 3.0, 4.0, 5.0))
    n = rng.choice((20, 25, 30, 40, 50))
    mu, sd = n * lam, math.sqrt(n * lam)
    alpha = rng.choice(list(Z_TAIL))
    z = Z_TAIL[alpha]
    exact = mu + z * sd - 0.5
    retry(0.12 < exact - math.floor(exact) < 0.88, "inverse answer too near an integer")
    c = math.ceil(exact)
    # the inverse problem is solved by trying c - 1 and c under the same approximation
    check(
        1 - phi((c + 0.5 - mu) / sd) <= float(alpha) + 5e-4 and 1 - phi((c - 0.5 - mu) / sd) > float(alpha) - 5e-4,
        "c is not the least",
    )
    cands = [
        (math.ceil(mu + z * sd), "r4i-clt-no-continuity-correction"),
        (math.floor(exact), "r4i-clt-round-down"),
        (math.ceil(mu + z * sd + 0.5), "r4i-clt-continuity-wrong-way"),
        (math.ceil(mu + z * math.sqrt(lam) - 0.5), "r4i-clt-sd-of-one-observation"),
        (math.ceil(mu + 1.96 * sd - 0.5) if alpha != "0.025" else math.ceil(mu + 1.645 * sd - 0.5), "r4i-clt-wrong-z"),
    ]
    dist = need(int_distractors(c, cands))
    text = (
        f"{setting} has a Poisson distribution with mean {lam:g}, independently from one {one} to the next. A manager "
        f"wants the smallest whole number c for which the probability that the total number of {what} over {n} {ones} "
        f"exceeds c is at most {alpha}. Use a normal approximation with a continuity correction, and take the "
        f"standard normal value with an upper tail of {alpha} to be {z}."
    )
    working = [
        f"The total S has mean {mu:g} and standard deviation √{mu:g} = {sd:.4f}.",
        f"P(S > c) = P(S ≥ c + 1) ≈ P(Z > (c + 0.5 - {mu:g})/{sd:.4f}) ≤ {alpha}, so c + 0.5 ≥ {mu:g} + {z}×{sd:.4f}.",
        f"c ≥ {exact:.3f}, so the smallest whole number is c = {c}.",
    ]
    return dict(text=text, working=working, distractors=dist, **int_fields(c))


def _hard_largest_n(rng):
    thing, load, unit = rng.choice(
        [("passengers", "A lift in a Melbourne office building", "kg"), ("riders", "A cable car near Cairns", "kg")]
    )
    m = rng.choice((72, 75, 78, 80, 82))
    s = rng.choice((10, 12, 14, 15))
    alpha = rng.choice(["0.05", "0.01", "0.10"])
    z = Z_TAIL[alpha]
    limit = rng.choice(range(600, 1301, 50))
    root = (-z * s + math.sqrt(z * z * s * s + 4 * m * limit)) / (2 * m)
    exact = root * root
    retry(0.12 < exact - math.floor(exact) < 0.88 and exact >= 4, "n too near an integer")
    n = math.floor(exact)
    check(n * m + z * s * math.sqrt(n) <= limit < (n + 1) * m + z * s * math.sqrt(n + 1), "n is not the largest")
    cands = [
        (math.floor(limit / m), "r4i-clt-ignored-spread"),
        (math.floor(limit / (m + z * s)), "r4i-clt-sd-of-total-n-sigma"),
        (math.ceil(exact), "r4i-clt-round-up"),
        (
            math.floor(((-1.96 * s + math.sqrt(3.8416 * s * s + 4 * m * limit)) / (2 * m)) ** 2)
            if alpha != "0.025"
            else None,
            "r4i-clt-wrong-z",
        ),
    ]
    dist = need(int_distractors(n, cands))
    text = (
        f"{load} is rated for a total load of {limit} {unit}. The mass of each of the {thing} using it is independent, "
        f"with mean {m} {unit} and standard deviation {s} {unit}. What is the largest number of {thing} that can ride "
        f"together so that the probability that their total mass exceeds {limit} {unit} is at most {alpha}? "
        f"Use a normal approximation and take the standard normal value with an upper tail of {alpha} to be {z}."
    )
    working = [
        f"For n {thing} the total has mean {m}n and standard deviation {s}√n.",
        f"We need {m}n + {z}×{s}√n ≤ {limit}, a quadratic in √n with positive root √n = {root:.3f}.",
        f"So n ≤ {exact:.3f} and the largest whole number is n = {n}.",
    ]
    return dict(text=text, working=working, distractors=dist, **int_fields(n))


def _hard(rng):
    return rng.choice((_hard_smallest_c, _hard_largest_n))(rng)


@generator(
    TOPIC,
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraw
def r4i_clt_sums(rng, tier="easy"):
    """Normal approximation to a sum of independent counts, with the continuity correction; see the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------


@generator(EXAM, level=21, presets={"exam": dict(tier="exam")})
@redraw
def exd_x_r4i_clt_sums(rng, tier="exam"):
    """A courier round: mean and sd of the total, P(S >= k), then the smallest c with P(S > c) <= 0.05."""
    check(tier == "exam", f"unknown tier {tier!r}")
    values = (0, 1, 2, 3)
    probs, m, v = _pmf(rng, values)
    n = rng.choice((40, 50, 60, 64, 80, 100))
    mu_exact, var_exact = n * m, n * v
    a_val = clean(float(mu_exact), 2)
    check(Fraction(repr(a_val)) == mu_exact or abs(a_val - float(mu_exact)) < 0.005, "mean not exact")
    b_exact = math.sqrt(float(var_exact))
    b_val = clean(b_exact, 2)
    retry(stable(b_exact, 2, 0.15), "sd near a rounding boundary")
    # the chain follows through from the stated (rounded) mean and sd, so the key is computed from them and must
    # agree with the unrounded route
    z0 = rng.choice([-1.0, -0.5, 0.5, 1.0, 1.5])
    k = round(a_val + z0 * b_val)
    p_c = 1 - phi((k - 0.5 - a_val) / b_val)
    p_c_exact = 1 - phi((k - 0.5 - float(mu_exact)) / b_exact)
    retry(stable(p_c, 4) and dec(p_c, 4) == dec(p_c_exact, 4) and 0.02 < p_c < 0.98, "part (c) unstable")
    z = 1.645
    c_raw = a_val + z * b_val - 0.5
    c_raw_exact = float(mu_exact) + z * b_exact - 0.5
    retry(0.15 < c_raw - math.floor(c_raw) < 0.85 and math.ceil(c_raw) == math.ceil(c_raw_exact), "part (d) unstable")
    c = math.ceil(c_raw)

    order = ["a", "b", "c", "d"]
    canon = {"a": a_val, "b": b_val, "c": clean(p_c, 4), "d": c}
    fns = {
        "c": lambda s: 1 - phi((k - 0.5 - s["a"]) / s["b"]),
        "d": lambda s: math.ceil(s["a"] + z * s["b"] - 0.5),
    }
    rounders = {"c": lambda x: clean(x, 4), "d": int}
    sd_one = math.sqrt(float(v))
    a_dist = need(
        num_distractors(
            a_val,
            [
                (n * float(m) + float(m), "r4i-clt-off-by-one-trial"),
                (float(m), "r4i-clt-mean-of-one-observation"),
                (n * float(sum(x * x * p for x, p in zip(values, probs))), "r4i-clt-second-moment-for-mean"),
                (float(n) * float(m) / 2, "r4i-clt-halved-total"),
            ],
            2,
        )
    )
    b_dist = need(
        num_distractors(
            b_val,
            [
                (float(var_exact), "r4i-clt-variance-as-sd"),
                (sd_one, "r4i-clt-sd-of-one-observation"),
                (n * sd_one, "r4i-clt-sd-of-total-n-sigma"),
                (
                    math.sqrt(n * float(sum(x * x * p for x, p in zip(values, probs)))),
                    "r4i-clt-second-moment-for-variance",
                ),
            ],
            2,
        )
    )
    wrong = {"a": [stored(d) for d in a_dist], "b": [stored(d) for d in b_dist]}
    wrong = {key: [float(x) for x in vals] for key, vals in wrong.items()}
    avoid_c = chain_avoid(order, fns, canon, wrong, "c", rounders)
    avoid_d = chain_avoid(order, fns, canon, wrong, "d", rounders)
    c_dist = need(
        num_distractors(
            canon["c"],
            [
                (1 - phi((k - a_val) / b_val), "r4i-clt-no-continuity-correction"),
                (1 - phi((k + 0.5 - a_val) / b_val), "r4i-clt-continuity-wrong-way"),
                (1 - phi((k - 0.5 - a_val) / sd_one), "r4i-clt-sd-of-one-observation"),
                (phi((k - 0.5 - a_val) / b_val), "r4i-clt-tail-reversed"),
            ],
            4,
            avoid=avoid_c,
            at_most=1.0,
        )
    )
    d_dist = need(
        int_distractors(
            c,
            [
                (math.ceil(a_val + z * b_val), "r4i-clt-no-continuity-correction"),
                (math.floor(c_raw), "r4i-clt-round-down"),
                (math.ceil(a_val + z * b_val + 0.5), "r4i-clt-continuity-wrong-way"),
                (math.ceil(a_val + 1.96 * b_val - 0.5), "r4i-clt-wrong-z"),
            ],
            avoid=avoid_d,
        )
    )
    mean_txt = dec(float(m), 2)
    table = _table_text(values, probs)
    text = rng.choice(
        [
            f"A courier van's round in Perth has {n} stops. The number of parcels left at a stop is a random variable X "
            f"with {table}, independent from stop to stop. Let S be the total number of parcels left on the round.",
            f"On each of {n} independent stops, a Perth courier leaves X parcels, where {table}. "
            f"The total number of parcels delivered on the round is S.",
            f"A delivery driver in Perth makes {n} stops in a shift. At every stop the parcels left follow the same "
            f"distribution: {table}, and stops do not affect each other. Write S for the total number of parcels left.",
        ]
    )
    parts = [
        dict(
            prompt="Find E(S), correct to 2 decimal places.",
            marks=1,
            sensible_range=(0, None),
            working=[f"E(X) = {mean_txt}, so E(S) = {n}×{mean_txt} = {dec(a_val, 2)}."],
            distractors=a_dist,
            **num_fields(a_val, 2),
        ),
        dict(
            prompt="Find the standard deviation of S, correct to 2 decimal places.",
            marks=2,
            sensible_range=(0, None),
            working=[
                f"Var(X) = E(X²) - E(X)² = {float(v):.4f}, so Var(S) = {n}×{float(v):.4f} = {float(var_exact):.2f}.",
                f"The standard deviation is √{float(var_exact):.2f} = {dec(b_val, 2)}.",
            ],
            distractors=b_dist,
            **num_fields(b_val, 2),
        ),
        dict(
            prompt=f"Use a normal approximation with a continuity correction to estimate P(S ≥ {k}), correct to 4 decimal places.",
            marks=3,
            sensible_range=(0, 1),
            follow_through=f"1 - (1 + erf((({k} - 0.5) - part_a)/part_b/sqrt(2)))/2",
            working=[
                f"S ≥ {k} becomes S > {k - 0.5} after the continuity correction.",
                f"z = ({k - 0.5} - {dec(a_val, 2)})/{dec(b_val, 2)}, so the probability is {dec(canon['c'], 4)}.",
            ],
            distractors=c_dist,
            **num_fields(canon["c"], 4),
        ),
        dict(
            prompt=(
                "The depot wants the smallest whole number c such that P(S > c) ≤ 0.05. Using the same approximation and "
                "the standard normal value 1.645 for an upper tail of 0.05, find c."
            ),
            marks=3,
            follow_through=f"ceiling(part_a + {z}*part_b - 0.5)",
            working=[
                f"P(S > c) = P(S ≥ c + 1) ≈ P(Z > (c + 0.5 - {dec(a_val, 2)})/{dec(b_val, 2)}) ≤ 0.05.",
                f"c ≥ {dec(a_val, 2)} + {z}×{dec(b_val, 2)} - 0.5 = {c_raw:.3f}, so c = {c}.",
            ],
            distractors=d_dist,
            **int_fields(c),
        ),
    ]
    return dict(text=text, parts=parts)
