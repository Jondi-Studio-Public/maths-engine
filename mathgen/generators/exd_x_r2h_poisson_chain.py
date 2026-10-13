"""Exam question on a Poisson process: a count probability, then conditioning and a second, disjoint period.

The stem gives an arrival rate per hour, a period in minutes and the value of e^(-lambda t) as 1/E (E close to the
true e^(mu), mu = lambda t the mean count, which the learner works out from the rate and the period). Four parts:

(a) P(N = k)                                             fraction, 2 marks
(b) P(N >= 1)                                            fraction, 1 mark
(c) P(N = k | N >= 1)                                    fraction, 2 marks, follow_through part_a / part_b
(d) exactly k arrivals now and none in the next c * t    fraction, 2 marks, follow_through part_a / E^c
    (disjoint periods are independent, and e^(-c lambda t) = (1/E)^c)

Every key is an exact Fraction built from mu and E; the test re-derives them from the rendered stem.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, generator, phrase
from .exd_x_r2h_support import build_part

_MINUTES = (10, 12, 15, 20, 24, 30, 36, 40, 45, 48, 60)
_MEANS = tuple(Fraction(n, 2) for n in range(3, 13))  # mu = 1.5, 2, ..., 6
_COUNTS = tuple(range(2, 10))
_SCALES = (1, 2, 3)

# (opening sentence with {lam}, plural noun, singular noun, plural verb, singular verb)
_CONTEXTS = (
    (
        "Calls reach a school office switchboard as a Poisson process, at an average of {lam} calls per hour.",
        "calls",
        "call",
        "arrive",
        "arrives",
    ),
    (
        "Cars pass a speed camera on a country road as a Poisson process, at an average of {lam} cars per hour.",
        "cars",
        "car",
        "pass",
        "passes",
    ),
    (
        "Customers enter a bakery in Fremantle as a Poisson process, at an average of {lam} customers per hour.",
        "customers",
        "customer",
        "enter",
        "enters",
    ),
    (
        "Emails arrive in a shared inbox as a Poisson process, at an average of {lam} emails per hour.",
        "emails",
        "email",
        "arrive",
        "arrives",
    ),
    (
        "Trams stop at a Melbourne tram stop as a Poisson process, at an average of {lam} trams per hour.",
        "trams",
        "tram",
        "stop",
        "stops",
    ),
    (
        "Faults are logged on a mine's conveyor belt as a Poisson process, at an average of {lam} faults per hour.",
        "faults",
        "fault",
        "are logged",
        "is logged",
    ),
)
_LEAD = (
    "Let N be the number of {things} in a {mins}-minute period, so N is Poisson with mean lambda t. "
    "For this question take e^(-lambda t) to be exactly 1/{E}.",
    "Consider a {mins}-minute period and let N count the {things} in it; N has a Poisson distribution with mean "
    "lambda t. Take e^(-lambda t) to be exactly 1/{E}.",
    "N is the number of {things} in a {mins}-minute period (mean lambda t). In this question, e^(-lambda t) is "
    "taken to be exactly 1/{E}.",
)
_PROMPT_A = (
    "Find the probability that exactly {k} {things} {verb} in the {mins}-minute period. Give your answer as a "
    "fraction.",
    "Find P(N = {k}). Give your answer as a fraction.",
)
_PROMPT_B = (
    "Find the probability that at least one {thing} {verb1} in the {mins}-minute period. Give your answer as a "
    "fraction.",
    "Find P(N >= 1). Give your answer as a fraction.",
)
_PROMPT_C = (
    "Given that at least one {thing} {verb1}, find the probability that exactly {k} {verb}. Give your answer as a "
    "fraction.",
    "Hence find P(N = {k} | N >= 1). Give your answer as a fraction.",
)
_PROMPT_D = (
    "Counts in disjoint periods are independent. Find the probability that exactly {k} {things} {verb} in this "
    "period and that no {thing} {verb1} in the following {next} minutes. Give your answer as a fraction.",
    "Periods that do not overlap are independent. Find the probability that exactly {k} {things} {verb} in the "
    "period and that no {thing} {verb1} in the {next} minutes after it. Give your answer as a fraction.",
)


def _poisson(mu: Fraction, k: int, e_inv: Fraction) -> Fraction:
    return mu**k / math.factorial(k) * e_inv


def _prob(value: Fraction) -> Fraction | None:
    """`value` when it could be a probability (strictly between 0 and 1), else None: a distractor above 1 is no
    one's plausible answer."""
    return value if 0 < value < 1 else None


def _build(rng: random.Random) -> dict:
    mu = rng.choice(_MEANS)
    options = [
        (m, Fraction(mu) * 60 / m) for m in _MINUTES if (mu * 60 / m).denominator == 1 and 2 <= mu * 60 / m <= 60
    ]
    check(bool(options), "no whole hourly rate for this mean")
    mins, lam = rng.choice(options)
    lam = int(lam)
    k = rng.choice(_COUNTS)
    c = rng.choice(_SCALES)
    big_e = round(math.exp(mu))
    e_inv = Fraction(1, big_e)
    check(mu == Fraction(lam * mins, 60), "mean does not match the rate and period")

    context, things, thing, verb, verb1 = rng.choice(_CONTEXTS)
    stem = context.format(lam=lam) + " " + phrase(rng, _LEAD, things=things, mins=mins, E=big_e)

    pk = _poisson(mu, k, e_inv)
    p_at_least_one = 1 - e_inv
    p_cond = pk / p_at_least_one
    p_none_c = e_inv**c
    p_joint = pk * p_none_c
    # Independent route: the whole distribution sums to a tail we can bound, and the identities the parts rest on.
    check(0 < pk < 1 and 0 < p_cond < 1 and 0 < p_joint < 1, "a part answer is not a probability")
    check(p_cond * p_at_least_one == pk, "conditional probability does not rebuild the joint")

    at_least_two = 1 - e_inv * (1 + mu)
    cumulative = sum(_poisson(mu, j, e_inv) for j in range(k + 1))
    a_wrong = [
        (_prob(mu**k * e_inv), "poisson-forgets-factorial"),
        (_prob(_poisson(Fraction(lam), k, e_inv)), "poisson-rate-not-mean"),
        (_prob(_poisson(mu, k - 1, e_inv)), "poisson-off-by-one-count"),
        (_prob(cumulative), "poisson-cumulative-for-exact"),
        (_prob(_poisson(mu, k + 1, e_inv)), "poisson-off-by-one-count"),
        (_prob(mu**k / math.factorial(k - 1) * e_inv), "poisson-factorial-off-by-one"),
        (_prob(1 - cumulative + _poisson(mu, k, e_inv)), "exactly-read-as-at-least"),
    ]
    b_wrong = [
        (e_inv, "none-instead-of-at-least-one"),
        (1 - mu * e_inv, "at-least-one-complement-slip"),
        (at_least_two, "at-least-one-complement-slip"),
        (_prob(mu * e_inv), "exactly-one-for-at-least-one"),
    ]
    c_wrong = [
        (pk, "conditional-ignored"),
        (pk * p_at_least_one, "conditional-as-product"),
        (pk / at_least_two, "at-least-one-complement-slip"),
        (_prob(p_at_least_one / pk), "conditional-ratio-inverted"),
    ]
    d_wrong = [
        (pk, "next-window-ignored"),
        (pk + p_none_c, "adds-instead-of-multiplies"),
        (pk * p_at_least_one, "at-least-one-for-none"),
        (_prob(pk * e_inv) if c > 1 else None, "next-window-ignored"),
    ]
    nxt = mins * c
    pa, pb, pc, pd_ = (Fraction(pk), p_at_least_one, p_cond, p_joint)
    parts = [
        build_part(
            phrase(rng, _PROMPT_A, k=k, things=things, mins=mins, verb=verb),
            pa,
            kind="fraction",
            marks=2,
            working=[
                f"lambda t = {lam} x {mins}/60 = {mu}, so P(N = {k}) = e^(-{mu}) x {mu}^{k} / {k}!.",
                f"= {mu}^{k} / {math.factorial(k)} x 1/{big_e} = {pa}.",
            ],
            wrong=a_wrong,
            sensible_range=(0, 1),
        ),
        build_part(
            phrase(rng, _PROMPT_B, thing=thing, mins=mins, verb1=verb1),
            pb,
            kind="fraction",
            marks=1,
            working=[f"P(N >= 1) = 1 - P(N = 0) = 1 - e^(-lambda t) = 1 - 1/{big_e} = {pb}."],
            wrong=b_wrong,
            sensible_range=(0, 1),
        ),
        build_part(
            phrase(rng, _PROMPT_C, thing=thing, k=k, verb=verb, verb1=verb1),
            pc,
            kind="fraction",
            marks=2,
            working=[f"P(N = {k} | N >= 1) = P(N = {k}) / P(N >= 1) = ({pa}) / ({pb}) = {pc}."],
            wrong=c_wrong,
            follow_through="part_a/part_b",
            sensible_range=(0, 1),
        ),
        build_part(
            phrase(rng, _PROMPT_D, k=k, things=things, thing=thing, next=nxt, verb=verb, verb1=verb1),
            pd_,
            kind="fraction",
            marks=2,
            working=[
                f"No arrivals in {nxt} minutes has probability e^(-{c} lambda t) = (1/{big_e})^{c}.",
                f"Independent periods multiply: ({pa}) x (1/{big_e})^{c} = {pd_}.",
            ],
            wrong=d_wrong,
            follow_through=f"part_a/{big_e}**{c}",
            sensible_range=(0, 1),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r2h_poisson_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2h_poisson_chain(rng: random.Random, tier: str = "exam") -> dict:
    """A Poisson count probability, an at-least-one probability, a conditional one, and a disjoint-period joint one."""
    check(tier == "exam", "exd_x_r2h_poisson_chain: only the exam tier exists")
    return _build(rng)
