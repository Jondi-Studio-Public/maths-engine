"""Exam questions: is a Bernoulli or binomial model reasonable when items are drawn without replacement?

The classification skills ask whether a described setting is suited to a Bernoulli or a binomial model. A real paper
attaches the working that settles it: how far the success probability moves, or how wrong the independence
assumption makes a probability. Both chains here end in the same classification, but only after the numbers are in.

Bernoulli chain: a batch of N items, K of them successes; the first m items are examined and set aside and are all
successes; one more is drawn. (a) the starting probability K/N, (b) the probability for the next draw (K-m)/(N-m),
(c) the percentage fall from (a) to (b), (d) whether the starting probability can still be used as a fixed Bernoulli
parameter, under an acceptance rule stated in the question (the fall is under t%).

Binomial chain: n items are drawn one after another without replacement from N with K successes. (a) the probability
that all n are successes if the draws are treated as independent, p^n, (b) the exact probability without
replacement, (c) the percentage by which the independent model overstates it, (d) whether a binomial model is
reasonable, under a stated rule (overstatement under t%).

Every key is an exact rational built from N, K, m (or n); the rounded percentage is the only rounded key, and it is
taken from the exact fractions, which the follow-through reproduces. Draws are rejected when the percentage sits
within 0.3 of the acceptance threshold or close to a rounding boundary, so the classification and the rounded figure
are unambiguous.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, rational_text
from .exd_x_r2k_support import retry_build, POSITIVE, clear_of_boundary, fraction_part, word_part, number_part, rd, tx

_CONTEXTS = (
    ("A batch of {N} components from a factory contains {K} that are faulty.", "component", "components", "faulty"),
    ("A raffle drum holds {N} tickets, {K} of which are winning tickets.", "ticket", "tickets", "winning"),
    ("A carton holds {N} eggs, {K} of which are cracked.", "egg", "eggs", "cracked"),
    ("A bag holds {N} jelly beans, {K} of which are red.", "jelly bean", "jelly beans", "red"),
    ("A crate holds {N} apples, {K} of which are bruised.", "apple", "apples", "bruised"),
    ("A depot holds {N} parcels, {K} of which are marked fragile.", "parcel", "parcels", "fragile"),
)
_BERNOULLI_THRESHOLDS = (1, 2, 5)
_BINOMIAL_THRESHOLDS = (2, 5, 10)

SUITED = "yes"
NOT_SUITED = "no"


_BERNOULLI_SIZES = (40, 60, 80, 100, 120, 150, 200, 250, 300, 400)
_BINOMIAL_SIZES = (
    20,
    24,
    25,
    30,
    32,
    36,
    40,
    45,
    48,
    50,
    56,
    60,
    64,
    72,
    75,
    80,
    90,
    96,
    100,
    120,
    125,
    144,
    150,
    160,
    180,
    200,
)


def _all(n: int) -> str:
    return "both" if n == 2 else f"all {n}"


_PROPORTIONS = tuple(
    Fraction(a, b)
    for a, b in (
        (1, 10),
        (3, 20),
        (1, 5),
        (1, 4),
        (3, 10),
        (1, 3),
        (7, 20),
        (2, 5),
        (9, 20),
        (1, 6),
        (1, 8),
        (3, 8),
        (5, 12),
        (3, 16),
        (5, 16),
        (7, 16),
        (1, 12),
        (7, 12),
        (5, 8),
    )
)


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def _pick_size(rng: random.Random, sizes: tuple[int, ...]) -> tuple[int, int]:
    """(N, K): a population and its number of successes, with a simple success proportion."""
    for _ in attempts(100, "a population size"):
        N = rng.choice(sizes)
        p = rng.choice(_PROPORTIONS)
        K = p * N
        if K.denominator == 1 and 6 <= K < N:
            return N, int(K)
    raise AssertionError("unreachable")


def _build_k_bernoulli_suitability(rng: random.Random) -> dict | None:
    """Starting probability, probability after removals, the percentage fall, and whether Bernoulli still fits."""
    want_suited = rng.random() < 0.5
    t = rng.choice(_BERNOULLI_THRESHOLDS)
    for _ in attempts(3000, "a population with a clear acceptance decision"):
        N, K = _pick_size(rng, _BERNOULLI_SIZES)
        m = rng.randint(4, 30)
        j = -(-m * K // N) + rng.randint(0, 2)  # at least the expected number, so the probability falls
        if j * N <= m * K or j > min(m, K - 1) or m >= N // 2:
            continue
        a = Fraction(K, N)
        b = Fraction(K - j, N - m)
        fall = (a - b) / a * 100
        if abs(fall - t) < Fraction(3, 10) or not clear_of_boundary(fall, 1) or (fall < t) != want_suited:
            continue
        break
    sentence, item, plural, adj = rng.choice(_CONTEXTS)
    stem = (
        f"{sentence.format(N=N, K=K)} The first {m} {plural} are examined and set aside, and {j} of them are {adj}. "
        f"One more {item} is then selected at random from those remaining. A Bernoulli model that keeps the original "
        f"probability of success is accepted for this selection when that probability falls by less than {t}% "
        f"of its original value."
    )
    pct = rd(fall, 1)
    suited = fall < t

    pa = fraction_part(
        f"Find the probability that {_article(item)} {item} chosen at random from the whole collection is {adj}.",
        a,
        1,
        [f"P(success) = {K}/{N} = {rational_text(a)}."],
        [
            (1 - a, "gave-complement-for-event"),
            (Fraction(K, N - 1), "wrong-denominator-for-proportion"),
            (Fraction(m, N), "wrong-count-in-probability"),
            (Fraction(j, m), "used-sample-proportion-for-population"),
        ],
        sensible_range=(0, 1),
    )
    pb = fraction_part(
        f"Find the probability that the next {item} is {adj}, given that {j} of the {m} already removed were {adj}.",
        b,
        2,
        [f"Now {K - j} of the {N - m} remaining are {adj}: {K - j}/{N - m} = {rational_text(b)}."],
        [
            (Fraction(K, N - m), "removed-from-total-only"),
            (Fraction(K - j, N), "removed-from-successes-only"),
            (Fraction(K - m, N - m) if K - m >= 1 else None, "removed-all-as-successes"),
            (Fraction(K - 1, N - 1), "removed-one-not-all"),
        ],
        sensible_range=(0, 1),
    )
    pc = number_part(
        "Hence find the percentage by which the probability of success has fallen from its original value, "
        "correct to 1 decimal place.",
        fall,
        1,
        2,
        [f"Fall = ({rational_text(a)} - {rational_text(b)}) / {rational_text(a)} x 100 = {tx(fall, 1)}%."],
        [
            ((a - b) / b * 100, "divides-by-final"),
            (b / a * 100, "new-as-percent-of-original"),
            ((a - b) * 100, "gave-change-not-percent"),
        ],
        follow_through="(part_a - part_b)/part_a*100",
        sensible_range=(0, 100),
    )
    if pa is None or pb is None or pc is None:
        return None
    if suited:
        key, wrongs = SUITED, [(NOT_SUITED, "dependence-assumed-regardless-of-size")]
        why = f"The probability falls by {pct}%, under {t}%, so the original probability can still be used."
    else:
        key, wrongs = NOT_SUITED, [(SUITED, "ignored-removal-effect")]
        why = f"The probability falls by {pct}%, which is not under {t}%, so a fixed probability is not reasonable."
    pd = word_part(
        "Is a Bernoulli model with the original success probability suited to this selection? Answer yes or no.",
        key,
        wrongs,
        1,
        [why],
    )
    return dict(text=stem, parts=[pa, pb, pc, pd])


def _build_k_binomial_suitability(rng: random.Random) -> dict | None:
    """p^n, the exact probability without replacement, the overstatement, and whether binomial still fits."""
    want_suited = rng.random() < 0.5
    t = rng.choice(_BINOMIAL_THRESHOLDS)
    for _ in attempts(3000, "a population with a clear acceptance decision"):
        N, K = _pick_size(rng, _BINOMIAL_SIZES)
        n = rng.randint(2, 4)
        p = Fraction(K, N)
        a = p**n
        b = Fraction(1)
        for i in range(n):
            b *= Fraction(K - i, N - i)
        over = (a - b) / b * 100
        if abs(over - t) < Fraction(3, 10) or not clear_of_boundary(over, 1) or (over < t) != want_suited:
            continue
        break
    sentence, item, plural, adj = rng.choice(_CONTEXTS)
    stem = (
        f"{sentence.format(N=N, K=K)} A sample of {n} {plural} is chosen at random, one after another and without "
        f"replacement. Let X be the number of {plural} in the sample that are {adj}. A binomial model for X is accepted "
        f"when it overstates the probability that {_all(n)} are {adj} by less than {t}% of the exact value."
    )
    pct = rd(over, 1)
    suited = over < t

    pa = fraction_part(
        f"Treating the {n} draws as independent, each with the same probability, find the probability that {_all(n)} "
        f"{plural} in the sample are {adj}.",
        a,
        1,
        [f"p = {K}/{N} = {rational_text(p)}; P(X = {n}) = p^{n} = {rational_text(a)}."],
        [
            (p * n if p * n < 1 else None, "multiplied-by-n-not-powered"),
            (p ** (n - 1), "wrong-exponent-on-probability"),
            ((1 - p) ** n, "swapped-success-failure"),
            (1 - a, "gave-complement-for-event"),
        ],
        sensible_range=(0, 1),
    )
    pb = fraction_part(
        f"Find the exact probability that {_all(n)} {plural} in the sample are {adj}, allowing for the fact that "
        f"none is replaced.",
        b,
        2,
        ["P = " + " x ".join(f"{K - i}/{N - i}" for i in range(n)) + f" = {rational_text(b)}."],
        [
            (_prod(K, N, n, shrink_total=False), "removed-from-successes-only"),
            (_prod(K, N, n, shrink_successes=False), "removed-from-total-only"),
            (_prod(K, N, n - 1), "drops-one-trial-from-sequence"),
            (a, "ignored-removal-effect"),
        ],
        sensible_range=(0, 1),
    )
    pc = number_part(
        "Hence find the percentage by which the independent model overstates the exact probability, "
        "correct to 1 decimal place.",
        over,
        1,
        2,
        [f"Overstatement = ({rational_text(a)} - {rational_text(b)}) / {rational_text(b)} x 100 = {tx(over, 1)}%."],
        [
            ((a - b) / a * 100, "percent-of-wrong-base"),
            (a / b * 100, "new-as-percent-of-original"),
            ((a - b) * 100, "gave-change-not-percent"),
        ],
        follow_through="(part_a - part_b)/part_b*100",
        sensible_range=POSITIVE,
    )
    if pa is None or pb is None or pc is None:
        return None
    if suited:
        key, wrongs = SUITED, [(NOT_SUITED, "dependence-assumed-regardless-of-size")]
        why = f"The overstatement is {pct}%, under {t}%, so the binomial model is reasonable."
    else:
        key, wrongs = NOT_SUITED, [(SUITED, "ignored-removal-effect")]
        why = f"The overstatement is {pct}%, which is not under {t}%, so the binomial model is not reasonable."
    pd = word_part("Is a binomial model suited to X? Answer yes or no.", key, wrongs, 1, [why])
    return dict(text=stem, parts=[pa, pb, pc, pd])


def _prod(K: int, N: int, n: int, shrink_successes: bool = True, shrink_total: bool = True) -> Fraction:
    out = Fraction(1)
    for i in range(n):
        out *= Fraction(K - (i if shrink_successes else 0), N - (i if shrink_total else 0))
    return out


@generator("exd_x_r2k_bernoulli_suitability", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2k_bernoulli_suitability(rng: random.Random, tier: str = "exam") -> dict:
    """Starting probability, probability after removals, the percentage fall, and whether Bernoulli still fits."""
    check(tier == "exam", "exd_x_r2k_bernoulli_suitability: only the exam tier exists")
    return retry_build(rng, _build_k_bernoulli_suitability, "exd_x_r2k_bernoulli_suitability")


@generator("exd_x_r2k_binomial_suitability", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2k_binomial_suitability(rng: random.Random, tier: str = "exam") -> dict:
    """p^n, the exact probability without replacement, the overstatement, and whether binomial still fits."""
    check(tier == "exam", "exd_x_r2k_binomial_suitability: only the exam tier exists")
    return retry_build(rng, _build_k_binomial_suitability, "exd_x_r2k_binomial_suitability")
