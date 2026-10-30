"""Generates `w5b_mm1_queue`, `w5b_eoq_inventory` (easy, medium, hard) and the exam chain `exd_x_w5b_mm1_queue_eoq`.

``uc2-mm1-queue-and-eoq`` (``optimisation``, answer ``fraction``): the M/M/1 queue and the economic order quantity.
Two topics feed every tier through separate sources, so a learner cannot tell from the tier which model comes next.

M/M/1 (arrival rate lam, service rate mu, utilisation rho = lam / mu < 1; P(n) = (1 - rho) rho^n; L = rho / (1 - rho);
Lq = rho^2 / (1 - rho); W = 1 / (mu - lam); Wq = rho / (mu - lam); P(wait > 0) = rho):

* easy: rho, L and the probability the server is idle.
* medium: Wq and W in a stated unit, Lq through Little's law, P(n = k) and P(at least k in the system).
* hard: the largest whole-number lam for which the mean wait in the queue stays within a target, the smallest
  whole-number mu for a target L or a target tail probability, and the time saved by a faster server.

EOQ (annual demand D, ordering cost S per order, holding cost H per unit per year, Q* = sqrt(2 D S / H)) is built
backwards from Q*, S and H with D = Q*^2 H / (2 S) a whole number:

* easy: Q*, the annual cost at Q*, the number of orders per year.
* medium: the reorder point (daily demand over the lead time, on 250 working days), the time between orders, the
  annual cost of a stated lot size and the extra cost over the optimum.
* hard: the best lot size and its cost when lots come in multiples of m, the demand D from the minimum annual cost, and
  the total annual cost including purchases when holding cost is a share of the unit price.
* exam: typed three-part chains with follow-through, drawn from both models.

Every value is an exact Fraction. Each queue quantity is checked against a numeric sum of the geometric
distribution, each EOQ against the first-order condition D S / Q^2 = H / 2 and its neighbouring lot sizes, and each
target by scanning the whole numbers on either side of the boundary.
"""

from __future__ import annotations

import re
from fractions import Fraction
from math import gcd

from ..answers import exact_value
from ..core import VerificationError, attempts, check, generator, phrase
from ._spread_kit import uses_rng
from ._w5b_uniform_rank_kit import balanced

WORKING_DAYS = 250
SUFFIX = " Give an exact answer, a whole number or a fraction in lowest terms."


def ftext(v):
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def frac_options(value, slips, keep=4, *, lo=None, hi=None):
    """Wrong options for an exact answer, placed by `balanced` so the key has no fixed rank among them."""
    return balanced(Fraction(value), slips, keep, kind="fraction", lo=lo, hi=hi)


def frac_q(text, value, working, slips, *, keep=4, lo=0, hi=None, suffix=SUFFIX):
    if hi is None and Fraction(value) <= 1 and re.search(r"probab|utilisation|fraction of time|P\(", text):
        hi = 1  # a probability or a utilisation never exceeds 1
    options = frac_options(value, slips, keep, lo=lo, hi=hi)
    check(len(options) >= 2, "too few distractors")
    return dict(
        text=text + suffix,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=ftext(value),
        answer_latex=ftext(value),
        working=working,
        distractors=options,
    )


def frac_part(prompt, value, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3, lo=0, hi=None):
    if hi is None and sensible_range == (0, 1):
        hi = 1
    options = frac_options(value, slips, keep, lo=lo, hi=hi)
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}")
    part = dict(
        prompt=prompt + " (lowest terms)",
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=ftext(value),
        marks=marks,
        working=working,
        distractors=options,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


# -- the M/M/1 model ---------------------------------------------------------------------------------------------


def mm1(lam, mu):
    """Closed forms as exact Fractions; times are in hours when the rates are per hour."""
    lam, mu = Fraction(lam), Fraction(mu)
    rho = lam / mu
    check(0 < rho < 1, "the queue must be stable")
    return dict(
        rho=rho,
        p0=1 - rho,
        L=rho / (1 - rho),
        Lq=rho * rho / (1 - rho),
        W=1 / (mu - lam),
        Wq=rho / (mu - lam),
    )


def mm1_by_series(lam, mu, terms=4000):
    """The same quantities from the stationary distribution P(n) = (1 - rho) rho^n summed numerically (floats)."""
    rho = float(Fraction(lam) / Fraction(mu))
    p, mean_system, mean_queue = 1 - rho, 0.0, 0.0
    prob = p
    for n in range(1, terms):
        prob *= rho
        mean_system += n * prob
        mean_queue += (n - 1) * prob
    return dict(L=mean_system, Lq=mean_queue, W=mean_system / float(lam), Wq=mean_queue / float(lam))


def checked_mm1(lam, mu):
    exact = mm1(lam, mu)
    numeric = mm1_by_series(lam, mu)
    for key, approx in numeric.items():
        check(abs(float(exact[key]) - approx) <= 1e-7 * max(1.0, abs(approx)), f"{key}: series and formula disagree")
    return exact


_SERVICE_TIMES = (2, 3, 4, 5, 6, 10, 12)  # minutes per customer: mu = 60 / s per hour

_SETUPS = (
    "In an M/M/1 queue customers arrive at rate λ = {lam} per hour and the server works at rate μ = {mu} per hour.",
    "A single-server queue is modelled as M/M/1 with arrival rate λ = {lam} per hour and service rate μ = {mu} per "
    "hour.",
    "Jobs reach a single print server at rate λ = {lam} per hour and are served at rate μ = {mu} per hour, "
    "both exponential (an M/M/1 queue).",
)
_TIME_SETUPS = (
    "Customers reach a single-server desk at {lam} per hour (Poisson arrivals), and each service takes an "
    "exponential time with mean {s} minutes. Treat it as an M/M/1 queue.",
    "Calls reach a help line with one operator at rate {lam} per hour; call lengths are exponential with mean {s} "
    "minutes. Model it as an M/M/1 queue.",
)


def _setup(rng, lam, mu, s=None):
    if s is not None:
        return phrase(rng, _TIME_SETUPS, lam=lam, s=s)
    return phrase(rng, _SETUPS, lam=lam, mu=mu)


def _draw_queue(rng, lo_rho=Fraction(3, 10), hi_rho=Fraction(9, 10), max_den=None):
    """(lam, mu, s) with s the mean service time in minutes when the stem states it that way, else None."""
    for _ in attempts(2000, "a stable queue with a utilisation in range"):
        if rng.random() < 0.35:
            s = rng.choice(_SERVICE_TIMES)
            mu = 60 // s
            lam = rng.randint(2, mu - 1) if mu > 3 else 2
        else:
            s = None
            mu = rng.randint(6, 48)
            lam = rng.randint(3, mu - 2)
        rho = Fraction(lam, mu)
        if lo_rho <= rho <= hi_rho and rho.denominator >= 5 and (max_den is None or rho.denominator <= max_den):
            return lam, mu, s


def fmt_minutes(v):
    """ "1 minute" or "n minutes" (a fraction such as 7/2 is plural)."""
    return f"{ftext(v)} minute" + ("" if Fraction(v) == 1 else "s")


def fmt_hours(v):
    return f"{ftext(v)} hour" + ("" if Fraction(v) == 1 else "s")


def _unit(rng):
    return rng.choice(("minutes", "minutes", "hours"))


def _time(value_hours, unit):
    return Fraction(value_hours) * (60 if unit == "minutes" else 1)


# -- easy queue ----------------------------------------------------------------------------------------------------

_Q_RHO = (
    "{setup} What is the utilisation ρ of the server?",
    "{setup} Find the long-run fraction of time the server is busy.",
)
_Q_L = (
    "{setup} Find L, the mean number of customers in the system (queueing and in service).",
    "{setup} On average, how many customers are in the system at a random moment (L)?",
)
_Q_P0 = (
    "{setup} What is the probability that the system is empty at a random moment?",
    "{setup} Find P(0), the long-run probability that no customer is in the system.",
)


def _queue_easy(rng):
    kind = rng.choice(("rho", "L", "p0"))
    for _ in attempts(500, "a queue whose utilisation needs reducing or whose L is not whole"):
        lam, mu, s = _draw_queue(rng)
        g = gcd(lam, mu)
        if g > 1 or rng.random() < 0.4:
            break
    q = checked_mm1(lam, mu)
    setup = _setup(rng, lam, mu, s)
    if kind == "rho":
        return frac_q(
            phrase(rng, _Q_RHO, setup=setup),
            q["rho"],
            [f"ρ = λ/μ = {lam}/{mu} = {ftext(q['rho'])}."],
            [
                (Fraction(mu, lam), "mm1-utilisation-inverted"),
                (q["p0"], "mm1-utilisation-complement"),
                (q["L"], "mm1-utilisation-as-l"),
                (Fraction(lam, lam + mu), "mm1-utilisation-sum-denominator"),
            ],
        )
    if kind == "L":
        return frac_q(
            phrase(rng, _Q_L, setup=setup),
            q["L"],
            [f"ρ = {ftext(q['rho'])}, so L = ρ/(1 - ρ) = {lam}/({mu} - {lam}) = {ftext(q['L'])}."],
            [
                (q["rho"], "mm1-l-as-utilisation"),
                (q["Lq"], "mm1-l-as-lq"),
                (q["L"] + 1, "mm1-l-one-too-many"),
                (Fraction(mu, mu - lam), "mm1-l-one-too-many"),
                (Fraction(lam, mu + lam), None),
            ],
        )
    return frac_q(
        phrase(rng, _Q_P0, setup=setup),
        q["p0"],
        [f"P(0) = 1 - ρ = 1 - {ftext(q['rho'])} = {ftext(q['p0'])}."],
        [
            (q["rho"], "mm1-idle-as-busy"),
            (1 / q["p0"], "mm1-idle-inverted"),
            (q["p0"] ** 2, "mm1-idle-squared"),
            (q["L"], "mm1-utilisation-as-l"),
        ],
    )


# -- medium queue --------------------------------------------------------------------------------------------------

_Q_WQ = (
    "{setup} Find the mean time a customer spends waiting in the queue before service starts, Wq, in {unit}.",
    "{setup} What is the average wait before service begins (not counting service), in {unit}?",
)
_Q_W = (
    "{setup} Find W, the mean total time a customer spends in the system, in {unit}.",
    "{setup} How long does a customer spend in the system on average, queueing and being served, in {unit}?",
)
_Q_LQ = (
    "{setup} Use Little's law with the mean wait in the queue to find Lq, the mean number of customers waiting "
    "(not in service).",
    "{setup} Find the mean number of customers waiting in the queue, not counting the one in service.",
)
_Q_PN = (
    "{setup} What is the probability that exactly {k} customers are in the system?",
    "{setup} Find P({k}), the long-run probability of exactly {k} customers in the system.",
)
_Q_TAIL = (
    "{setup} What is the probability that at least {k} customers are in the system?",
    "{setup} Find the long-run probability that {k} or more customers are present in the system.",
)


def _queue_medium(rng):
    kind = rng.choice(("Wq", "W", "Lq", "pn", "tail"))
    lam, mu, s = _draw_queue(rng, Fraction(2, 5), Fraction(17, 20), max_den=24 if kind in ("pn", "tail") else None)
    q = checked_mm1(lam, mu)
    setup = _setup(rng, lam, mu, s)
    rho = q["rho"]
    if kind in ("Wq", "W"):
        unit = _unit(rng)
        hours = q[kind]
        value = _time(hours, unit)
        other = q["W" if kind == "Wq" else "Wq"]
        slips = [
            (_time(other, unit), "mm1-wq-w-swapped"),
            (_time(Fraction(1, mu), unit), "mm1-w-as-service-time"),
            (_time(hours, "hours" if unit == "minutes" else "minutes"), "mm1-time-unit-not-converted"),
            (_time(q["L"] / mu, unit), None),
        ]
        formula = "ρ/(μ - λ)" if kind == "Wq" else "1/(μ - λ)"
        return frac_q(
            phrase(rng, _Q_WQ if kind == "Wq" else _Q_W, setup=setup, unit=unit),
            value,
            [
                f"{kind} = {formula} = {fmt_hours(hours)}"
                + (f" = {fmt_minutes(value)}." if unit == "minutes" else "."),
            ],
            slips,
        )
    if kind == "Lq":
        value = lam * q["Wq"]
        check(value == q["Lq"], "Little's law and the closed form for Lq disagree")
        return frac_q(
            phrase(rng, _Q_LQ, setup=setup),
            value,
            [
                f"Wq = ρ/(μ - λ) = {ftext(q['Wq'])} hours, so Lq = λ Wq = {lam} × {ftext(q['Wq'])} = {ftext(value)}.",
            ],
            [
                (q["L"], "mm1-lq-as-l"),
                (q["rho"], "mm1-lq-as-utilisation"),
                (q["L"] - 1, "mm1-lq-minus-one"),
                (lam * q["Wq"] * 60, "mm1-time-unit-not-converted"),
            ],
        )
    if kind == "pn":
        k = rng.randint(2, 3)
        value = q["p0"] * rho**k
        return frac_q(
            phrase(rng, _Q_PN, setup=setup, k=k),
            value,
            [f"P({k}) = (1 - ρ) ρ^{k} = ({ftext(q['p0'])})({ftext(rho)})^{k} = {ftext(value)}."],
            [
                (rho**k, "mm1-pn-forgets-idle-factor"),
                (q["p0"] * rho ** (k - 1), "mm1-pn-exponent-off-by-one"),
                (q["p0"] ** k * rho, "mm1-pn-swaps-rho-and-complement"),
                (q["p0"] * rho ** (k + 1), "mm1-pn-exponent-off-by-one"),
            ],
        )
    k = rng.randint(2, 4)
    value = rho**k
    return frac_q(
        phrase(rng, _Q_TAIL, setup=setup, k=k),
        value,
        [f"P(N ≥ {k}) = Σ over n ≥ {k} of (1 - ρ) ρ^n = ρ^{k} = {ftext(value)}."],
        [
            (rho ** (k - 1), "mm1-tail-exponent-off-by-one"),
            (q["p0"] * rho**k, "mm1-tail-as-point-probability"),
            (1 - rho**k, "mm1-tail-as-complement"),
            (rho ** (k + 1), "mm1-tail-exponent-off-by-one"),
        ],
    )


# -- hard queue ----------------------------------------------------------------------------------------------------

_Q_MAXLAM = (
    "A single server works at rate μ = {mu} per hour (an M/M/1 queue). Arrivals are Poisson at rate λ per hour. "
    "The mean wait in the queue, Wq, must not exceed {T} minutes. What is the largest whole-number λ that "
    "meets this?",
    "An M/M/1 queue has service rate μ = {mu} per hour. Management wants the mean time customers wait before "
    "service to be at most {T} minutes. Find the largest whole number of arrivals per hour, λ, the queue can take.",
)
_Q_MINMU_L = (
    "Customers arrive at rate λ = {lam} per hour at an M/M/1 queue. The mean number in the system, L, must be at "
    "most {ell}. What is the smallest whole-number service rate μ per hour that achieves this?",
    "An M/M/1 queue receives λ = {lam} arrivals per hour. Find the smallest whole-number service rate μ (per hour) "
    "that keeps the mean number of customers in the system, L, at or below {ell}.",
)
_Q_MINMU_TAIL = (
    "An M/M/1 queue has arrival rate λ = {lam} per hour. The probability of {k} or more customers being in the "
    "system must be at most {pct}%. Find the smallest whole-number service rate μ per hour that achieves this.",
    "Arrivals at an M/M/1 queue come at λ = {lam} per hour. What is the smallest whole-number service rate μ "
    "per hour for which P(at least {k} customers in the system) is at most {pct}%?",
)
_Q_SAVED = (
    "An M/M/1 queue has arrival rate λ = {lam} per hour and service rate μ = {mu} per hour. A new server raises the "
    "service rate to {mu2} per hour. By how many minutes does the mean time in the system, W, fall?",
    "Customers arrive at λ = {lam} per hour. Replacing a server with rate μ = {mu} per hour by one with rate "
    "{mu2} per hour changes the M/M/1 queue. Find the reduction in W, in minutes.",
)
_Q_MINMU_W = (
    "Arrivals at an M/M/1 queue are λ = {lam} per hour. The mean time in the system, W, must not exceed {w} "
    "minutes. Find the smallest whole-number service rate μ per hour that achieves this.",
    "An M/M/1 queue with λ = {lam} arrivals per hour is to have mean time in system W of at most {w} minutes. "
    "What is the smallest whole-number μ (per hour) that works?",
)


def _ceil(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def _floor(x: Fraction) -> int:
    return x.numerator // x.denominator


def _min_mu_tail(lam, power, p, tail):
    """The smallest whole mu above lam with tail(mu) <= p, for a tail that falls as mu grows. A float guess at the
    boundary, then exact Fraction comparisons on the whole numbers around it; the neighbour below must fail."""
    guess = int(lam * float(p) ** (-1.0 / power))
    low = max(lam + 1, guess - 3)
    for m in range(low, guess + 6):
        if tail(m) <= p:
            check(m == lam + 1 or tail(m - 1) > p, "the whole number below the boundary also meets the target")
            return m
    raise VerificationError("no whole-number boundary near the float guess")


def _queue_hard(rng):
    kind = rng.choice(("maxlam", "minmu_L", "minmu_tail", "saved", "minmu_W"))
    if kind == "maxlam":
        for _ in attempts(3000, "a wait target with a non-integer boundary"):
            mu = rng.randint(8, 40)
            T = rng.randint(1, 12)
            boundary = Fraction(T * mu * mu, 60 + T * mu)
            lam_max = _floor(boundary)
            if boundary.denominator != 1 and 3 <= lam_max < mu - 1 and Fraction(lam_max, mu) >= Fraction(2, 5):
                break

        def wq_minutes(lam):
            return 60 * mm1(lam, mu)["Wq"]

        check(wq_minutes(lam_max) <= T < wq_minutes(lam_max + 1), "the boundary scan disagrees with the closed form")
        w_only = _floor(mu - Fraction(60, T)) if mu > Fraction(60, T) else None
        return frac_q(
            phrase(rng, _Q_MAXLAM, mu=mu, T=T),
            lam_max,
            [
                f"Wq = 60 λ/(μ(μ - λ)) minutes ≤ {T} gives λ ≤ {T}μ²/(60 + {T}μ) = {ftext(boundary)}.",
                f"The largest whole number is {lam_max}; at λ = {lam_max + 1} the wait is {fmt_minutes(wq_minutes(lam_max + 1))}.",
            ],
            [
                (lam_max + 1, "mm1-target-rounds-up"),
                (w_only, "mm1-target-uses-w-not-wq"),
                (lam_max - 1, None),
                (mu - 1, "mm1-target-edge-of-stability"),
            ],
            lo=1,
        )
    if kind == "minmu_L":
        for _ in attempts(3000, "an L target with a non-integer boundary"):
            lam = rng.randint(5, 45)
            ell = (
                Fraction(rng.choice((3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18)), 2)
                if rng.random() < 0.7
                else Fraction(rng.randint(2, 7))
            )
            boundary = lam * (ell + 1) / ell
            if boundary.denominator != 1 and _ceil(boundary) >= lam + 2:
                break
        mu_min = _ceil(boundary)
        check(mm1(lam, mu_min)["L"] <= ell < mm1(lam, mu_min - 1)["L"], "the L boundary scan disagrees")
        return frac_q(
            phrase(rng, _Q_MINMU_L, lam=lam, ell=ftext_decimal(ell)),
            mu_min,
            [
                f"L = λ/(μ - λ) ≤ {ftext_decimal(ell)} gives μ ≥ λ(1 + 1/{ftext_decimal(ell)}) = {ftext(boundary)}.",
                f"The smallest whole number is {mu_min}.",
            ],
            [
                (mu_min - 1, "mm1-target-rounds-down"),
                (_ceil(lam * ell), "mm1-l-target-inverted"),
                (lam + _ceil(ell), "mm1-l-target-adds-ell"),
                (mu_min + 1, "mm1-target-rounds-up"),
            ],
            lo=lam + 1,
        )
    if kind == "minmu_tail":
        for _ in attempts(3000, "a tail target with a clear integer answer"):
            lam = rng.randint(6, 30)
            k = rng.randint(2, 4)
            pct = rng.choice((1, 2, 4, 5, 8, 10, 20, 25))
            p = Fraction(pct, 100)
            mu_min = _min_mu_tail(lam, k, p, lambda m: Fraction(lam, m) ** k)
            if mu_min > lam + 2:
                break
        wrong_exp = _min_mu_tail(lam, k - 1, p, lambda m: Fraction(lam, m) ** (k - 1))
        wrong_up = _min_mu_tail(lam, k + 1, p, lambda m: Fraction(lam, m) ** (k + 1))
        return frac_q(
            phrase(rng, _Q_MINMU_TAIL, lam=lam, k=k, pct=pct),
            mu_min,
            [
                f"P(N ≥ {k}) = (λ/μ)^{k} ≤ {pct}/100 requires μ ≥ λ × (100/{pct})^(1/{k}).",
                f"Checking whole numbers: μ = {mu_min} gives ({lam}/{mu_min})^{k} ≤ {pct}%, but μ = {mu_min - 1} does not.",
            ],
            [
                (mu_min - 1, "mm1-target-rounds-down"),
                (wrong_exp, "mm1-tail-exponent-off-by-one"),
                (wrong_up, "mm1-tail-exponent-off-by-one"),
                (mu_min + 1, "mm1-target-rounds-up"),
            ],
            lo=lam + 1,
        )
    if kind == "saved":
        for _ in attempts(2000, "a faster server that still leaves a visible wait"):
            lam, mu, _s = _draw_queue(rng, Fraction(3, 5), Fraction(19, 20))
            mu2 = mu + rng.randint(2, 12)
            if mu2 > mu + 1:
                break
        w1, w2 = checked_mm1(lam, mu)["W"], checked_mm1(lam, mu2)["W"]
        value = 60 * (w1 - w2)
        return frac_q(
            phrase(rng, _Q_SAVED, lam=lam, mu=mu, mu2=mu2),
            value,
            [
                f"W = 1/(μ - λ): {fmt_minutes(60 * w1)} before, {fmt_minutes(60 * w2)} after.",
                f"The mean time falls by {fmt_minutes(value)}.",
            ],
            [
                (w1 - w2, "mm1-time-unit-not-converted"),
                (60 * (w1 - w2) * Fraction(lam, mu), None),
                (60 * (Fraction(1, mu) - Fraction(1, mu2)), "mm1-w-as-service-time"),
                (60 * (mm1(lam, mu)["Wq"] - mm1(lam, mu2)["Wq"]), "mm1-wq-w-swapped"),
            ],
        )
    for _ in attempts(3000, "a W target with a non-integer boundary"):
        lam = rng.randint(4, 40)
        w = rng.choice((2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15))
        boundary = lam + Fraction(60, w)
        if boundary.denominator != 1:
            break
    mu_min = _ceil(boundary)
    check(60 * mm1(lam, mu_min)["W"] <= w < 60 * mm1(lam, mu_min - 1)["W"], "the W boundary scan disagrees")
    return frac_q(
        phrase(rng, _Q_MINMU_W, lam=lam, w=w),
        mu_min,
        [
            f"W = 1/(μ - λ) hours ≤ {w}/60 gives μ ≥ λ + 60/{w} = {ftext(boundary)}; the smallest whole number is {mu_min}."
        ],
        [
            (mu_min - 1, "mm1-target-rounds-down"),
            (_floor(boundary), None),
            (lam + _ceil(Fraction(1, w)), "mm1-time-unit-not-converted"),
            (_ceil(Fraction(60 * lam, w)), None),
        ],
        lo=lam + 1,
    )


def ftext_decimal(v: Fraction) -> str:
    """A target such as 5/2 written as a decimal, 2.5 (denominators here are 1 or 2)."""
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    check(v.denominator == 2, "only halves are written as decimals")
    return f"{v.numerator // 2}.5"


@generator(
    "w5b_mm1_queue",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
def w5b_mm1_queue(rng, tier="easy"):
    """M/M/1 queue questions; the tiers are described in the module docstring."""
    if tier == "easy":
        return _queue_easy(rng)
    if tier == "medium":
        return _queue_medium(rng)
    if tier == "hard":
        return _queue_hard(rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- the EOQ model -------------------------------------------------------------------------------------------------


def total_cost(D, S, H, Q):
    """Annual ordering plus holding cost for lot size Q."""
    return Fraction(D * S, 1) / Q + Fraction(H * Q, 2)


def checked_eoq(D, S, H, Q):
    """Verify Q = sqrt(2 D S / H) by the first-order condition and by its neighbours, return the minimum cost."""
    check(2 * D * S == H * Q * Q, "Q is not the EOQ: the first-order condition D S / Q^2 = H / 2 fails")
    best = total_cost(D, S, H, Q)
    check(best < total_cost(D, S, H, Q - 1) and best < total_cost(D, S, H, Q + 1), "a neighbouring lot size is cheaper")
    check(best == Q * H, "cost at the EOQ differs from Q* H")
    return best


_SS = (15, 20, 25, 30, 40, 45, 50, 60, 75, 80, 90, 100, 120, 150)


def _tidy_lot(rng, D, S, H, Q, max_den=20):
    """A lot size away from Q whose annual cost has a small denominator (so the answer is a readable fraction): a
    divisor of 2 D S between half and double the EOQ, at least 15 units from it. Any lot size away from Q otherwise."""
    near = [x for x in range(max(10, Q // 2), 2 * Q + 1) if abs(x - Q) >= 15 and (2 * D * S) % x == 0]
    near = [x for x in near if total_cost(D, S, H, x).denominator <= max_den]
    if near:
        return rng.choice(near)
    fallback = None
    for _ in range(400):
        Q2 = max(10, Q + rng.choice((-1, 1)) * rng.choice((20, 25, 30, 40, 50, 60, 75, 100)))
        if Q2 != Q and abs(Q2 - Q) >= 15:
            fallback = Q2
            break
    check(fallback is not None, "no lot size away from the EOQ")
    return fallback


def _draw_eoq(rng, lo=Fraction(800), hi=Fraction(60000)):
    """(Q, S, H, D) with D = Q^2 H / (2 S) a whole number."""
    for _ in attempts(5000, "an EOQ with a whole-number demand"):
        Q = rng.randint(30, 600)
        S = rng.choice(_SS)
        H = rng.randint(2, 40)
        if (Q * Q * H) % (2 * S) == 0 and lo <= Fraction(Q * Q * H, 2 * S) <= hi:
            return Q, S, H, Q * Q * H // (2 * S)


_E_SETUP = (
    "A firm uses {D} units of a part a year. Each order costs ${S} to place and holding a unit for a year costs ${H}.",
    "Annual demand for an item is {D} units. Placing an order costs ${S}, and it costs ${H} to hold one unit in "
    "stock for a year.",
    "A shop sells {D} units of a product each year, ordering costs ${S} per order and holding costs ${H} per unit "
    "per year.",
)

_E_Q = (
    "{setup} Using the EOQ model, what is the optimal order quantity Q*?",
    "{setup} Find the economic order quantity.",
)
_E_TC = (
    "{setup} Orders are placed at the economic order quantity. What is the total annual ordering plus holding cost "
    "in dollars?",
    "{setup} Find the minimum total annual cost of ordering and holding, in dollars.",
)
_E_N = (
    "{setup} How many orders per year does the economic order quantity policy place?",
    "{setup} Using the EOQ, find the number of orders placed in a year.",
)


def _eoq_easy(rng):
    kind = rng.choice(("Q", "TC", "N"))
    Q, S, H, D = _draw_eoq(rng)
    best = checked_eoq(D, S, H, Q)
    setup = phrase(rng, _E_SETUP, D=D, S=S, H=H)
    if kind == "Q":
        return frac_q(
            phrase(rng, _E_Q, setup=setup),
            Q,
            [f"Q* = √(2DS/H) = √(2 × {D} × {S}/{H}) = √{Q * Q} = {Q}."],
            [
                (D * S // H if (D * S) % H == 0 else None, "eoq-drops-factor-two"),
                (Fraction(2 * D * S, H), "eoq-no-square-root"),
                (_isqrt_exact(2 * D * H, S), "eoq-swaps-s-and-h"),
                (_isqrt_exact(D * S, 2 * H), "eoq-halves-instead"),
            ],
            lo=1,
        )
    if kind == "TC":
        return frac_q(
            phrase(rng, _E_TC, setup=setup),
            best,
            [
                f"Q* = {Q}. Ordering: DS/Q* = {D * S}/{Q} = {ftext(Fraction(D * S, Q))}; holding: HQ*/2 = {ftext(Fraction(H * Q, 2))}.",
                f"Total = {ftext(best)}.",
            ],
            [
                (Fraction(D * S, Q), "eoq-cost-only-ordering"),
                (Fraction(H * Q, 2), "eoq-cost-only-holding"),
                (Fraction(D * S, Q) + H * Q, "eoq-holding-without-half"),
                (Fraction(D * S, Q) * 2 + H * Q, None),
            ],
        )
    value = Fraction(D, Q)
    return frac_q(
        phrase(rng, _E_N, setup=setup),
        value,
        [f"Q* = {Q}, so the number of orders is D/Q* = {D}/{Q} = {ftext(value)}."],
        [
            (Fraction(Q, D), "eoq-orders-inverted"),
            (Fraction(D, 2 * Q), "eoq-orders-halved"),
            (Fraction(D, Q) + 1, None),
        ],
    )


def _isqrt_exact(num, den):
    """√(num/den) when that is a whole number, else None."""
    ratio = Fraction(num, den)
    if ratio.denominator != 1:
        return None
    r = round(ratio.numerator**0.5)
    return r if r * r == ratio.numerator else None


# -- medium EOQ ----------------------------------------------------------------------------------------------------

_E_ROP = (
    "{setup} The firm works {days} days a year, and an order takes {lead} working days to arrive. At what stock "
    "level should it reorder (demand over the lead time)?",
    "{setup} With {days} working days a year and a lead time of {lead} working days, what is the reorder point?",
)
_E_CYCLE = (
    "{setup} The firm works {days} days a year. Orders are placed at the EOQ. How many working days pass between "
    "orders?",
    "{setup} On {days} working days a year, find the time in working days between successive orders at the EOQ.",
)
_E_LOT = (
    "{setup} The firm instead orders {Q2} units each time. What is its total annual ordering plus holding cost, in "
    "dollars?",
    "{setup} If every order is for {Q2} units, find the annual ordering plus holding cost in dollars.",
)
_E_EXTRA = (
    "{setup} The firm currently orders {Q2} units at a time rather than the EOQ. How much more per year, in "
    "dollars, does this cost in ordering plus holding than the EOQ policy?",
    "{setup} By how many dollars per year does ordering {Q2} units each time exceed the minimum annual ordering "
    "plus holding cost?",
)


def _eoq_medium(rng):
    kind = rng.choice(("rop", "cycle", "lot", "extra"))
    Q, S, H, D = _draw_eoq(rng)
    best = checked_eoq(D, S, H, Q)
    setup = phrase(rng, _E_SETUP, D=D, S=S, H=H)
    days = WORKING_DAYS
    if kind == "rop":
        for _ in range(60):
            lead = rng.randint(3, 21)
            if (D * lead) % days == 0 or rng.random() < 0.15:
                break
            Q, S, H, D = _draw_eoq(rng)
            best = checked_eoq(D, S, H, Q)
            setup = phrase(rng, _E_SETUP, D=D, S=S, H=H)
        value = Fraction(D, days) * lead
        return frac_q(
            phrase(rng, _E_ROP, setup=setup, days=days, lead=lead),
            value,
            [f"Daily demand is {D}/{days}; over {lead} days it is {D} × {lead}/{days} = {ftext(value)} units."],
            [
                (Fraction(D, days), "eoq-rop-forgets-lead-time"),
                (Fraction(D * lead, 365), "eoq-rop-wrong-days"),
                (Fraction(Q * lead, days), "eoq-rop-uses-eoq"),
                (Fraction(D, days * lead), None),
            ],
        )
    if kind == "cycle":
        value = Fraction(days * Q, D)
        return frac_q(
            phrase(rng, _E_CYCLE, setup=setup, days=days),
            value,
            [f"Orders per year = D/Q* = {D}/{Q}; the time between orders is {days} × {Q}/{D} = {ftext(value)} days."],
            [
                (Fraction(D, Q), "eoq-orders-inverted"),
                (Fraction(365 * Q, D), "eoq-rop-wrong-days"),
                (Fraction(days * D, Q), None),
                (Fraction(days * Q, 2 * D), None),
            ],
        )
    Q2 = _tidy_lot(rng, D, S, H, Q)
    cost2 = total_cost(D, S, H, Q2)
    if kind == "lot":
        return frac_q(
            phrase(rng, _E_LOT, setup=setup, Q2=Q2),
            cost2,
            [
                f"Ordering: {D} × {S}/{Q2} = {ftext(Fraction(D * S, Q2))}; holding: {H} × {Q2}/2 = {ftext(Fraction(H * Q2, 2))}."
            ],
            [
                (Fraction(D * S, Q2), "eoq-cost-only-ordering"),
                (Fraction(H * Q2, 2), "eoq-cost-only-holding"),
                (Fraction(D * S, Q2) + H * Q2, "eoq-holding-without-half"),
                (best, "eoq-cost-at-optimum-not-lot"),
            ],
        )
    value = cost2 - best
    check(value > 0, "a non-optimal lot must cost more")
    return frac_q(
        phrase(rng, _E_EXTRA, setup=setup, Q2=Q2),
        value,
        [f"Cost at {Q2} is {ftext(cost2)}; the minimum is {ftext(best)}; the difference is {ftext(value)}."],
        [
            (cost2, "eoq-cost-at-lot-not-extra"),
            (abs(Fraction(D * S, Q2) - Fraction(D * S, Q)), "eoq-extra-ordering-only"),
            (abs(Fraction(H * Q2, 2) - Fraction(H * Q, 2)), "eoq-extra-holding-only"),
            (best, "eoq-cost-at-optimum-not-lot"),
        ],
    )


# -- hard EOQ ------------------------------------------------------------------------------------------------------

_E_MULT = (
    "{setup} The supplier only ships whole multiples of {m} units. Which order quantity (a multiple of {m}) gives "
    "the lowest total annual ordering plus holding cost?",
    "{setup} Orders must be a multiple of {m} units. Find the feasible order quantity with the smallest annual "
    "ordering plus holding cost.",
)
_E_MULTCOST = (
    "{setup} Orders must be a multiple of {m} units. What is the lowest total annual ordering plus holding cost "
    "among the feasible order quantities, in dollars?",
    "{setup} The supplier ships only in multiples of {m} units. Find the smallest achievable annual ordering plus "
    "holding cost, in dollars.",
)
_E_INVERSE = (
    "Ordering costs ${S} per order and holding costs ${H} per unit per year. At the economic order quantity the "
    "total annual ordering plus holding cost is ${TC}. What is the annual demand D?",
    "An item has order cost ${S} and annual holding cost ${H} per unit. The minimum annual ordering plus holding "
    "cost over all lot sizes is ${TC}. Find the annual demand in units.",
)
_E_PRICE = (
    "A firm buys a part at ${c} each and needs {D} a year. Each order costs ${S}, and holding a unit for a year "
    "costs {pct}% of its price. Orders are placed at the EOQ. What is the total annual cost in dollars, "
    "including the purchase cost of the {D} parts?",
    "A part costs ${c}; annual demand is {D}. Ordering costs ${S} per order and the annual holding cost is {pct}% of "
    "the unit price. Find the total yearly cost, purchases plus ordering plus holding, when ordering at the EOQ.",
)


def _eoq_hard(rng):
    kind = rng.choice(("mult_q", "mult_cost", "inverse", "price"))
    if kind in ("mult_q", "mult_cost"):
        for _ in attempts(3000, "a multiple that straddles the EOQ"):
            Q, S, H, D = _draw_eoq(rng)
            m = rng.choice((25, 40, 50, 60, 75, 80, 100))
            if Q % m != 0 and Q > m:
                break
        best = checked_eoq(D, S, H, Q)
        lo_lot, hi_lot = (Q // m) * m, (Q // m + 1) * m
        by_scan = min(range(m, 3 * Q + m, m), key=lambda x: (total_cost(D, S, H, x), x))
        winner = min((lo_lot, hi_lot), key=lambda x: (total_cost(D, S, H, x), x))
        check(by_scan == winner, "the best feasible lot is not one of the two multiples around Q*")
        nearer = lo_lot if Q - lo_lot <= hi_lot - Q else hi_lot
        setup = phrase(rng, _E_SETUP, D=D, S=S, H=H)
        if kind == "mult_q":
            other = hi_lot if winner == lo_lot else lo_lot
            return frac_q(
                phrase(rng, _E_MULT, setup=setup, m=m),
                winner,
                [
                    f"Q* = {Q} lies between {lo_lot} and {hi_lot}. Costs: {ftext(total_cost(D, S, H, lo_lot))} and {ftext(total_cost(D, S, H, hi_lot))}.",
                    f"The cheaper is {winner}.",
                ],
                [
                    (other, "eoq-nearest-lot-not-cheapest" if winner != nearer else "eoq-other-neighbour"),
                    (nearer, "eoq-nearest-lot-not-cheapest"),
                    (Q, "eoq-ignores-multiple-constraint"),
                    (hi_lot + m, None),
                ],
                lo=1,
            )
        value = total_cost(D, S, H, winner)
        return frac_q(
            phrase(rng, _E_MULTCOST, setup=setup, m=m),
            value,
            [f"The best multiple is {winner}, with cost {D} × {S}/{winner} + {H} × {winner}/2 = {ftext(value)}."],
            [
                (best, "eoq-cost-at-optimum-not-lot"),
                (total_cost(D, S, H, hi_lot if winner == lo_lot else lo_lot), "eoq-other-neighbour"),
                (value + (value - best), None),
                (value - best, "eoq-cost-at-lot-not-extra"),
            ],
        )
    if kind == "inverse":
        Q, S, H, D = _draw_eoq(rng)
        best = checked_eoq(D, S, H, Q)
        check(Fraction(best * best, 2 * S * H) == D, "D from the minimum cost disagrees")
        return frac_q(
            phrase(rng, _E_INVERSE, S=S, H=H, TC=ftext(best)),
            D,
            [f"At the EOQ the cost is Q*H, so Q* = {ftext(best)}/{H} = {Q}; then D = Q*² H/(2S) = {D}."],
            [
                (best * best // (S * H) if (best * best) % (S * H) == 0 else None, "eoq-inverse-drops-two"),
                (Q, "eoq-inverse-stops-at-q"),
                (best // S if best % S == 0 else None, None),
                (Fraction(2 * best * best, S * H), "eoq-inverse-doubles"),
            ],
            lo=1,
        )
    for _ in attempts(5000, "a unit price and holding share giving a whole-number EOQ"):
        Q = rng.randint(30, 400)
        S = rng.choice(_SS)
        c = rng.choice((5, 8, 10, 12, 15, 20, 25, 30, 40, 50))
        pct = rng.choice((10, 12, 15, 20, 25, 30))
        H = Fraction(c * pct, 100)
        D2 = Fraction(Q * Q) * H / (2 * S)
        if D2.denominator == 1 and 800 <= D2 <= 40000:
            D = int(D2)
            break
    check(2 * D * S == H * Q * Q, "the price-based EOQ is not Q")
    cost = c * D + total_cost(D, S, H, Q)
    check(total_cost(D, S, H, Q) == Q * H, "cost at the EOQ differs from Q* H")
    return frac_q(
        phrase(rng, _E_PRICE, c=c, D=D, S=S, pct=pct),
        cost,
        [
            f"H = {pct}% of {c} = {ftext(H)} per unit per year. Q* = √(2 × {D} × {S}/{ftext(H)}) = {Q}.",
            f"Annual cost = {c} × {D} + {D} × {S}/{Q} + {ftext(H)} × {Q}/2 = {ftext(cost)}.",
        ],
        [
            (cost - c * D, "eoq-forgets-purchase-cost"),
            (c * D + total_cost(D, S, H, Q) * 2, None),
            (c * D + Fraction(D * S, Q), "eoq-cost-only-ordering"),
            (c * D + Fraction(H * Q, 2), "eoq-cost-only-holding"),
        ],
    )


@generator(
    "w5b_eoq_inventory",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
def w5b_eoq_inventory(rng, tier="easy"):
    """Economic order quantity questions; the tiers are described in the module docstring."""
    if tier == "easy":
        return _eoq_easy(rng)
    if tier == "medium":
        return _eoq_medium(rng)
    if tier == "hard":
        return _eoq_hard(rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam ----------------------------------------------------------------------------------------------------------


def _queue_exam(rng):
    shape = rng.choice(("rho_L_W", "Wq_W_L", "p0_pn_tail"))
    lam, mu, s = _draw_queue(rng, Fraction(2, 5), Fraction(17, 20), max_den=24 if shape == "p0_pn_tail" else None)
    q = checked_mm1(lam, mu)
    rho = q["rho"]
    setup = _setup(rng, lam, mu, s)
    if shape == "rho_L_W":
        w_min = 60 * q["W"]
        return dict(
            text=setup,
            parts=[
                frac_part(
                    "Find the utilisation ρ.",
                    rho,
                    2,
                    [f"ρ = λ/μ = {ftext(rho)}."],
                    [(q["p0"], "mm1-utilisation-complement"), (Fraction(mu, lam), "mm1-utilisation-inverted")],
                    sensible_range=(0, 1),
                ),
                frac_part(
                    "Hence find L, the mean number of customers in the system.",
                    q["L"],
                    2,
                    [f"L = ρ/(1 - ρ) = {ftext(q['L'])}."],
                    [(rho, "mm1-l-as-utilisation"), (q["Lq"], "mm1-l-as-lq"), (q["L"] + 1, "mm1-l-one-too-many")],
                    follow_through="part_a/(1 - part_a)",
                    sensible_range=(0, None),
                ),
                frac_part(
                    "Using Little's law, find W, the mean time a customer spends in the system, in minutes.",
                    w_min,
                    3,
                    [f"W = L/λ = {ftext(q['L'])}/{lam} = {fmt_hours(q['L'] / lam)} = {fmt_minutes(w_min)}."],
                    [
                        (q["L"] / lam, "mm1-time-unit-not-converted"),
                        (60 * q["Wq"], "mm1-wq-w-swapped"),
                        (60 * q["L"], "mm1-w-as-l"),
                    ],
                    follow_through=f"60*part_b/{lam}",
                    sensible_range=(0, None),
                ),
            ],
        )
    if shape == "Wq_W_L":
        wq_min = 60 * q["Wq"]
        w_min = wq_min + Fraction(60, mu)
        check(w_min == 60 * q["W"], "W = Wq + 1/mu disagrees with the closed form")
        return dict(
            text=setup,
            parts=[
                frac_part(
                    "Find Wq, the mean time waiting in the queue before service, in minutes.",
                    wq_min,
                    2,
                    [f"Wq = ρ/(μ - λ) = {fmt_hours(q['Wq'])} = {fmt_minutes(wq_min)}."],
                    [
                        (60 * q["W"], "mm1-wq-w-swapped"),
                        (q["Wq"], "mm1-time-unit-not-converted"),
                        (Fraction(60, mu), "mm1-w-as-service-time"),
                    ],
                    sensible_range=(0, None),
                ),
                frac_part(
                    "Hence find W, the mean time in the system, in minutes (waiting plus service).",
                    w_min,
                    2,
                    [f"W = Wq + 1/μ = {ftext(wq_min)} + {ftext(Fraction(60, mu))} = {fmt_minutes(w_min)}."],
                    [
                        (wq_min, "mm1-w-as-wq"),
                        (wq_min + Fraction(1, mu), "mm1-time-unit-not-converted"),
                        (wq_min + Fraction(60, lam), None),
                    ],
                    follow_through=f"part_a + {ftext(Fraction(60, mu))}",
                    sensible_range=(0, None),
                ),
                frac_part(
                    "Using Little's law, find L, the mean number of customers in the system.",
                    q["L"],
                    3,
                    [f"L = λ W = {lam} × {ftext(w_min)}/60 = {ftext(q['L'])}."],
                    [
                        (Fraction(lam) * w_min, "mm1-time-unit-not-converted"),
                        (q["Lq"], "mm1-l-as-lq"),
                        (rho, "mm1-l-as-utilisation"),
                    ],
                    follow_through=f"{lam}*part_b/60",
                    sensible_range=(0, None),
                ),
            ],
        )
    k = rng.randint(2, 3)
    return dict(
        text=setup,
        parts=[
            frac_part(
                "Find the probability that the system is empty.",
                q["p0"],
                2,
                [f"P(0) = 1 - ρ = {ftext(q['p0'])}."],
                [(rho, "mm1-idle-as-busy"), (1 / q["p0"], "mm1-idle-inverted"), (q["p0"] ** 2, "mm1-idle-squared")],
                sensible_range=(0, 1),
            ),
            frac_part(
                f"Hence find the probability that exactly {k} customers are in the system.",
                q["p0"] * rho**k,
                3,
                [f"P({k}) = P(0) ρ^{k} = {ftext(q['p0'] * rho**k)}."],
                [
                    (rho**k, "mm1-pn-forgets-idle-factor"),
                    (q["p0"] * rho ** (k - 1), "mm1-pn-exponent-off-by-one"),
                    (q["p0"] ** k * rho, "mm1-pn-swaps-rho-and-complement"),
                ],
                follow_through=f"part_a*(1 - part_a)**{k}",
                sensible_range=(0, 1),
            ),
            frac_part(
                f"Find the probability that at least {k} customers are in the system.",
                rho**k,
                2,
                [f"P(N ≥ {k}) = ρ^{k} = {ftext(rho**k)}."],
                [
                    (rho ** (k - 1), "mm1-tail-exponent-off-by-one"),
                    (q["p0"] * rho**k, "mm1-tail-as-point-probability"),
                    (1 - rho**k, "mm1-tail-as-complement"),
                ],
                follow_through=f"(1 - part_a)**{k}",
                sensible_range=(0, 1),
            ),
        ],
    )


def _eoq_exam(rng):
    shape = rng.choice(("Q_TC_N", "Q_cycle_rop", "Q_cost_saving"))
    Q, S, H, D = _draw_eoq(rng)
    best = checked_eoq(D, S, H, Q)
    setup = phrase(rng, _E_SETUP, D=D, S=S, H=H)
    q_part = frac_part(
        "Find the economic order quantity Q*.",
        Q,
        2,
        [f"Q* = √(2DS/H) = √{Q * Q} = {Q}."],
        [
            (_isqrt_exact(2 * D * H, S), "eoq-swaps-s-and-h"),
            (Fraction(2 * D * S, H), "eoq-no-square-root"),
            (_isqrt_exact(D * S, 2 * H), "eoq-halves-instead"),
            (D * S // H if (D * S) % H == 0 else None, "eoq-drops-factor-two"),
        ],
        sensible_range=(0, None),
    )
    if shape == "Q_TC_N":
        return dict(
            text=setup,
            parts=[
                q_part,
                frac_part(
                    "Hence find the total annual ordering plus holding cost, in dollars.",
                    best,
                    3,
                    [f"DS/Q* + HQ*/2 = {ftext(Fraction(D * S, Q))} + {ftext(Fraction(H * Q, 2))} = {ftext(best)}."],
                    [
                        (Fraction(D * S, Q), "eoq-cost-only-ordering"),
                        (Fraction(H * Q, 2), "eoq-cost-only-holding"),
                        (Fraction(D * S, Q) + H * Q, "eoq-holding-without-half"),
                    ],
                    follow_through=f"{D * S}/part_a + {H}*part_a/2",
                    sensible_range=(0, None),
                ),
                frac_part(
                    "Find the number of orders placed per year.",
                    Fraction(D, Q),
                    2,
                    [f"D/Q* = {D}/{Q} = {ftext(Fraction(D, Q))}."],
                    [(Fraction(Q, D), "eoq-orders-inverted"), (Fraction(D, 2 * Q), "eoq-orders-halved")],
                    follow_through=f"{D}/part_a",
                    sensible_range=(0, None),
                ),
            ],
        )
    if shape == "Q_cycle_rop":
        lead = rng.randint(3, 21)
        for _ in range(60):
            if (D * lead) % WORKING_DAYS == 0:
                break
            lead = rng.randint(3, 21)
        rop = Fraction(D * lead, WORKING_DAYS)
        cycle = Fraction(WORKING_DAYS * Q, D)
        return dict(
            text=setup + f" The firm works {WORKING_DAYS} days a year.",
            parts=[
                q_part,
                frac_part(
                    "Find the number of working days between orders when ordering at Q*.",
                    cycle,
                    3,
                    [f"{WORKING_DAYS} × Q*/D = {WORKING_DAYS} × {Q}/{D} = {ftext(cycle)}."],
                    [
                        (Fraction(D, Q), "eoq-orders-inverted"),
                        (Fraction(365 * Q, D), "eoq-rop-wrong-days"),
                        (Fraction(WORKING_DAYS * D, Q), None),
                    ],
                    follow_through=f"{WORKING_DAYS}*part_a/{D}",
                    sensible_range=(0, None),
                ),
                frac_part(
                    f"Orders take {lead} working days to arrive. Find the reorder point in units.",
                    rop,
                    2,
                    [f"Daily demand {D}/{WORKING_DAYS} times {lead} days = {ftext(rop)}."],
                    [
                        (Fraction(D, WORKING_DAYS), "eoq-rop-forgets-lead-time"),
                        (Fraction(D * lead, 365), "eoq-rop-wrong-days"),
                        (Fraction(Q * lead, WORKING_DAYS), "eoq-rop-uses-eoq"),
                    ],
                    sensible_range=(0, None),
                ),
            ],
        )
    Q2 = _tidy_lot(rng, D, S, H, Q)
    cost2 = total_cost(D, S, H, Q2)
    return dict(
        text=setup + f" The firm currently orders {Q2} units at a time.",
        parts=[
            q_part,
            frac_part(
                "Hence find the minimum total annual ordering plus holding cost, in dollars.",
                best,
                2,
                [f"DS/Q* + HQ*/2 = {ftext(best)}."],
                [
                    (Fraction(D * S, Q), "eoq-cost-only-ordering"),
                    (Fraction(H * Q, 2), "eoq-cost-only-holding"),
                    (Fraction(D * S, Q) + H * Q, "eoq-holding-without-half"),
                ],
                follow_through=f"{D * S}/part_a + {H}*part_a/2",
                sensible_range=(0, None),
            ),
            frac_part(
                f"By how many dollars per year does ordering {Q2} units at a time exceed that minimum?",
                cost2 - best,
                3,
                [f"Cost at {Q2}: {D * S}/{Q2} + {H} × {Q2}/2 = {ftext(cost2)}; excess {ftext(cost2 - best)}."],
                [
                    (cost2, "eoq-cost-at-lot-not-extra"),
                    (best, "eoq-cost-at-optimum-not-lot"),
                    (abs(Fraction(D * S, Q2) - Fraction(D * S, Q)), "eoq-extra-ordering-only"),
                ],
                follow_through=f"{ftext(cost2)} - part_b",
                sensible_range=(0, None),
            ),
        ],
    )


@generator("exd_x_w5b_mm1_queue_eoq", level=20, presets={"exam": dict(tier="exam")})
@uses_rng
def exd_x_w5b_mm1_queue_eoq(rng, tier="exam"):
    """Exam chains: M/M/1 (ρ, L, W by Little; Wq, W, L; P(0), P(k), tail) or EOQ (Q*, cost, orders, cycle, reorder)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    if rng.random() < 0.5:
        return _queue_exam(rng)
    return _eoq_exam(rng)
