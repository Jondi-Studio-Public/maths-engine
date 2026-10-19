"""Binomial tails and ranges by the complement: `r3a_binomial_at_least` and its exam chain.

Skill `binomial-at-least-complement`. X ~ Bin(n, p) with n from 10 to 30 and a small success probability, so the
event in the question is built from the tail the learner has to choose to sum or to take the complement of.

Tiers (topic `r3a_binomial_at_least`):

* easy   -- an event with a short sum or a one-line complement: at least one, at most one, or at most two;
* medium -- a tail or range that needs several point probabilities: P(X >= k) by the complement (k 2 to 4),
  P(X > k) read strictly, or P(a <= X <= b) around the mean;
* hard   -- a conditional probability built from two tails: P(X >= k | X >= j), or P(X <= b | X >= a).

The exam chain (topic `exd_x_r3a_binomial_at_least`) is (a) the lower tail P(X <= k - 1), (b) P(X >= k) from (a),
(c) P(X >= k | X >= 1) from (b), (d) the chance that the event X >= k happens on at least one of m independent
days, from (b). Parts (b) to (d) carry `follow_through`.

Every key is held as an exact Fraction and re-derived by an independent route: the whole distribution is built one
trial at a time (Pascal's recurrence), which uses no binomial coefficient and no power. A draw is rejected when a rounded
answer is close to a rounding boundary or when feeding a learner's rounded earlier answers forward would move a
later key by more than one unit in its last place.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, round_half_up
from ._r3a_kit import build_until, clear_of_boundary, dec, number_part, pmf, single, span

#: (setting with {n} and {p}, the count X)
_CONTEXTS = (
    (
        "A factory machine produces {n} bolts in a minute, each independently faulty with probability {p}.",
        "the number of faulty bolts",
    ),
    (
        "Each of {n} flights leaving an airport one morning is independently delayed with probability {p}.",
        "the number of delayed flights",
    ),
    (
        "A charity rings {n} households, and each independently agrees to donate with probability {p}.",
        "the number of households that agree to donate",
    ),
    (
        "A nursery sows {n} seeds, and each independently fails to germinate with probability {p}.",
        "the number of seeds that fail to germinate",
    ),
    (
        "A laboratory runs {n} tests, and each independently returns a false positive with probability {p}.",
        "the number of false positives",
    ),
    (
        "A bus company sells {n} tickets for one trip, and each ticket holder independently fails to turn up with probability {p}.",
        "the number of ticket holders who fail to turn up",
    ),
)
_UNIT = (Fraction(0), Fraction(1))
_PS = tuple(Fraction(k, 20) for k in range(1, 8))


def _distribution_by_recurrence(n: int, p: Fraction) -> list[Fraction]:
    """The whole distribution of Bin(n, p) built one trial at a time: no binomial coefficient, no power."""
    dist = [Fraction(1)]
    for _ in range(n):
        nxt = [Fraction(0)] * (len(dist) + 1)
        for k, mass in enumerate(dist):
            nxt[k] += mass * (1 - p)
            nxt[k + 1] += mass * p
        dist = nxt
    return dist


def _draw(rng: random.Random, fine: bool = False):
    setting, x_word = rng.choice(_CONTEXTS)
    n = rng.randint(10, 30)
    p = Fraction(rng.randint(5, 35), 100) if fine else rng.choice(_PS)
    return n, p, setting.format(n=n, p=dec(p)), x_word


def _stem(setting: str, x_word: str) -> str:
    return f"{setting} Let X be {x_word}."


def _ok(value: Fraction) -> bool:
    return Fraction(2, 100) < value < Fraction(98, 100) and clear_of_boundary(value, 4, Fraction(1, 5))


# -- easy ------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    variant = rng.choice(("at_least_one", "at_most_one", "at_most_two"))
    for _ in attempts(1000, "an event with a stable probability"):
        n, p, setting, x_word = _draw(rng)
        q = 1 - p
        dist = _distribution_by_recurrence(n, p)
        if variant == "at_least_one":
            value, ask = 1 - q**n, "at least one"
            check(value == sum(dist[1:]), "complement disagrees with the distribution")
        elif variant == "at_most_one":
            value, ask = q**n + n * p * q ** (n - 1), "at most one"
            check(value == dist[0] + dist[1], "two-term sum disagrees with the distribution")
        else:
            value, ask = q**n + n * p * q ** (n - 1) + Fraction(n * (n - 1), 2) * p**2 * q ** (n - 2), "at most two"
            check(value == sum(dist[:3]), "three-term sum disagrees with the distribution")
        if _ok(value):
            break
    prompt = f"Find the probability that X is {ask}, correct to 4 decimal places."
    if variant == "at_least_one":
        slips = [
            (q**n, "r3a-at-least-one-gives-none"),
            (n * p * q ** (n - 1), "r3a-at-least-one-gives-exactly-one"),
            (1 - p**n, "r3a-at-least-one-complements-all-successes"),
        ]
        working = [f"P(X >= 1) = 1 - P(X = 0) = 1 - {dec(q)}^{n}."]
    elif variant == "at_most_one":
        slips = [
            (q**n, "r3a-at-most-one-omits-exactly-one"),
            (1 - q**n - n * p * q ** (n - 1), "r3a-reports-the-complement-tail"),
            (n * p * q ** (n - 1), "r3a-at-most-one-omits-none"),
        ]
        working = [f"P(X <= 1) = P(X = 0) + P(X = 1) = {dec(q)}^{n} + {n} x {dec(p)} x {dec(q)}^{n - 1}."]
    else:
        slips = [
            (q**n + n * p * q ** (n - 1), "r3a-at-most-two-stops-at-one"),
            (1 - value, "r3a-reports-the-complement-tail"),
            (Fraction(n * (n - 1), 2) * p**2 * q ** (n - 2), "r3a-at-most-reduced-to-one-point"),
        ]
        working = [f"P(X <= 2) = P(X = 0) + P(X = 1) + P(X = 2) = {float(value):.4f}."]
    part = number_part(prompt, value, 4, 1, working, slips, within=_UNIT)
    return single(f"{_stem(setting, x_word)} {part['prompt']}", part)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium(rng: random.Random) -> dict:
    variant = rng.choice(("at_least_k", "more_than_k", "between"))
    for _ in attempts(2000, "a tail with a stable probability"):
        n, p, setting, x_word = _draw(rng)
        dist = _distribution_by_recurrence(n, p)
        mean = n * p
        if variant == "at_least_k":
            k = rng.randint(2, 4)
            value, event = sum(dist[k:]), f"at least {k}"
            lower, slips = sum(dist[:k]), None
        elif variant == "more_than_k":
            k = rng.randint(2, 5)
            value, event = sum(dist[k + 1 :]), f"more than {k}"
        else:
            centre = round(mean)
            a, b = max(centre - 1, 0), centre + rng.randint(1, 2)
            value, event = sum(dist[a : b + 1]), f"between {a} and {b} inclusive"
            if b - a < 2:
                continue
        check(value == (span(n, p, k, n) if variant == "at_least_k" else value), "tail disagrees")
        if _ok(value) and (variant == "between" or value > Fraction(1, 10)):
            break
    q = 1 - p
    prompt = f"Find the probability that X is {event}, correct to 4 decimal places."
    if variant == "at_least_k":
        check(value == 1 - span(n, p, 0, k - 1), "complement disagrees with the sum")
        slips = [
            (1 - span(n, p, 0, k), "r3a-complement-boundary-off-by-one"),
            (span(n, p, 0, k - 1), "r3a-reports-the-complement-tail"),
            (pmf(n, p, k), "r3a-tail-reduced-to-one-point"),
        ]
        working = [f"P(X >= {k}) = 1 - P(X <= {k - 1}) = 1 - ({' + '.join(f'P(X = {j})' for j in range(k))})."]
    elif variant == "more_than_k":
        check(value == 1 - span(n, p, 0, k), "complement disagrees with the sum")
        slips = [
            (1 - span(n, p, 0, k - 1), "r3a-strict-inequality-read-as-inclusive"),
            (span(n, p, 0, k), "r3a-reports-the-complement-tail"),
            (1 - span(n, p, 0, k + 1), "r3a-complement-boundary-off-by-one"),
        ]
        working = [f"P(X > {k}) = P(X >= {k + 1}) = 1 - P(X <= {k})."]
    else:
        slips = [
            (pmf(n, p, a) + pmf(n, p, b), "r3a-range-sums-only-the-ends"),
            (span(n, p, a + 1, b - 1), "r3a-range-drops-the-end-values"),
            (span(n, p, 0, b), "r3a-range-ignores-the-lower-bound"),
        ]
        working = [f"P({a} <= X <= {b}) = sum of P(X = j) for j from {a} to {b}."]
    part = number_part(prompt, value, 4, 2, working, slips, within=_UNIT)
    return single(f"{_stem(setting, x_word)} {part['prompt']}", part)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard(rng: random.Random) -> dict:
    variant = rng.choice(("tail_given_tail", "range_given_tail"))
    for _ in attempts(3000, "a conditional probability with a stable value"):
        n, p, setting, x_word = _draw(rng)
        dist = _distribution_by_recurrence(n, p)
        j = rng.randint(1, 2)
        if variant == "tail_given_tail":
            k = j + rng.randint(1, 3)
            num, den = sum(dist[k:]), sum(dist[j:])
            event = f"X is at least {k}, given that X is at least {j}"
        else:
            j = rng.randint(1, 2)
            k = j + rng.randint(1, 3)
            num, den = sum(dist[j : k + 1]), sum(dist[j:])
            event = f"X is at most {k}, given that X is at least {j}"
        if den <= Fraction(1, 5) or num == 0:
            continue
        value = num / den
        if _ok(value):
            break
    q = 1 - p
    prompt = f"Find the probability that {event}. Give your answer correct to 4 decimal places."
    if variant == "tail_given_tail":
        slips = [
            (num, "r3a-conditional-ignores-the-condition"),
            (den, "r3a-reports-the-probability-of-the-condition"),
            (sum(dist[k:]) / sum(dist[:j]) if sum(dist[:j]) else None, "r3a-conditional-divides-by-the-complement"),
            (sum(dist[j:k]) / den, "r3a-conditional-takes-the-wrong-event"),
        ]
        working = [f"P(X >= {k} | X >= {j}) = P(X >= {k}) / P(X >= {j}), because X >= {k} already implies X >= {j}."]
    else:
        slips = [
            (num, "r3a-conditional-ignores-the-condition"),
            (num / sum(dist[: k + 1]), "r3a-conditional-divides-by-the-wrong-tail"),
            (sum(dist[k + 1 :]) / den, "r3a-conditional-takes-the-wrong-event"),
            (den, "r3a-reports-the-probability-of-the-condition"),
        ]
        working = [f"P(X <= {k} | X >= {j}) = P({j} <= X <= {k}) / P(X >= {j})."]
    part = number_part(prompt, value, 4, 3, working, slips, within=_UNIT)
    return single(f"{_stem(setting, x_word)} {part['prompt']}", part)


@generator(
    "r3a_binomial_at_least",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3a_binomial_at_least(rng: random.Random, tier: str = "easy") -> dict:
    """Binomial tails by the complement: a one-line complement, a several-term tail or range, a conditional."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _frac(value: Fraction) -> str:
    return f"({value.numerator}/{value.denominator})"


def _exam(rng: random.Random) -> dict:
    """The lower tail, the upper tail from it, a conditional tail, then the chance it happens on some day."""
    for _ in attempts(5000, "a chain whose rounded answers carry forward cleanly"):
        n, p, setting, x_word = _draw(rng, fine=True)
        q = 1 - p
        k = rng.randint(3, 4)
        days = rng.randint(3, 5)
        dist = _distribution_by_recurrence(n, p)
        lower = sum(dist[:k])
        upper = sum(dist[k:])
        none = q**n
        cond = upper / (1 - none)
        some_day = 1 - (1 - upper) ** days
        if not all(_ok(v) for v in (lower, upper, cond, some_day)):
            continue
        # a learner's rounded earlier answers must carry forward to within one unit of the last place
        b_round = round_half_up(upper, 4)
        c_from_b = b_round / (1 - none)
        d_from_b = 1 - (1 - b_round) ** days
        if abs(round_half_up(1 - round_half_up(lower, 4), 4) - round_half_up(upper, 4)) > Fraction(1, 10000):
            continue
        if abs(c_from_b - cond) > Fraction(1, 20000) or abs(d_from_b - some_day) > Fraction(1, 20000):
            continue
        break
    check(upper == span(n, p, k, n) and lower == span(n, p, 0, k - 1), "tails disagree with the formula route")
    stem = f"{_stem(setting, x_word)} The same situation arises on each of {days} independent days."
    parts = [
        number_part(
            f"Find P(X <= {k - 1}), correct to 4 decimal places.",
            lower,
            4,
            2,
            [f"P(X <= {k - 1}) = " + " + ".join(f"P(X = {j})" for j in range(k)) + f" = {float(lower):.4f}."],
            [
                (span(n, p, 0, k), "r3a-lower-tail-includes-the-boundary"),
                (span(n, p, 0, k - 2), "r3a-lower-tail-stops-one-short"),
                (pmf(n, p, k - 1), "r3a-tail-reduced-to-one-point"),
            ],
            sensible_range=(0, 1),
            within=_UNIT,
        ),
        number_part(
            f"Find P(X >= {k}), correct to 4 decimal places.",
            upper,
            4,
            1,
            [f"P(X >= {k}) = 1 - P(X <= {k - 1}) = 1 - {float(lower):.4f}."],
            [
                (lower, "r3a-reports-the-complement-tail"),
                (1 - span(n, p, 0, k), "r3a-complement-boundary-off-by-one"),
                (pmf(n, p, k), "r3a-tail-reduced-to-one-point"),
            ],
            follow_through="1-part_a",
            sensible_range=(0, 1),
            within=_UNIT,
        ),
        number_part(
            f"Given that X is at least 1, find the probability that X is at least {k}, correct to 4 decimal places.",
            cond,
            4,
            2,
            [f"P(X >= {k} | X >= 1) = P(X >= {k}) / (1 - {dec(q)}^{n}) = {float(upper):.4f} / {float(1 - none):.4f}."],
            [
                (upper, "r3a-conditional-ignores-the-condition"),
                (1 - none, "r3a-reports-the-probability-of-the-condition"),
                (upper / none, "r3a-conditional-divides-by-the-complement"),
            ],
            follow_through=f"part_b/(1-{_frac(q)}**{n})",
            sensible_range=(0, 1),
            within=_UNIT,
        ),
        number_part(
            f"Find the probability that X is at least {k} on at least one of the {days} days, correct to 4 decimal "
            "places.",
            some_day,
            4,
            2,
            [
                f"P(never) = (1 - {float(upper):.4f})^{days}, so P(at least one day) = 1 - (1 - {float(upper):.4f})^{days}."
            ],
            [
                (upper**days, "r3a-repeated-event-every-day"),
                ((1 - upper) ** days, "r3a-repeated-event-never"),
                (days * upper, "r3a-repeated-event-adds-the-probabilities"),
            ],
            follow_through=f"1-(1-part_b)**{days}",
            sensible_range=(0, 1),
            within=_UNIT,
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r3a_binomial_at_least", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r3a_binomial_at_least(rng: random.Random, tier: str = "exam") -> dict:
    """The lower tail, the upper tail from it, a conditional tail, then the chance it happens on some day."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
