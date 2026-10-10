"""Generates `exd_x_s1_complement` questions: an "at least one" probability that is best found through the
complement.

Two forms, both asking for a single fraction in lowest terms:

* sampling without replacement -- a batch of ``n`` items of which ``d`` are faulty, ``k`` items chosen at
  random (``k`` is 2 or 3); find P(at least one faulty).  The answer is ``1 - C(n-d, k) / C(n, k)``.
* independent components -- two or three components that fail independently with stated, different
  probabilities; find P(at least one fails).  The answer is ``1 - prod(1 - p_i)``.

The learner has to choose the method (the complement of "none are faulty") and carry out a multi-step
counting or product calculation, rather than subtract one fraction from one.

Each question is cross-checked with an independent route: exhaustive enumeration of the sample space for the
sampling form, and enumeration of all fail/work patterns for the component form.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction
from math import comb

from ..core import check, check_equivalent, generator, pick_distractors, rational_text

_ITEMS = [
    ("components", "faulty"),
    ("bulbs", "defective"),
    ("tablets", "damaged"),
    ("screws", "faulty"),
]

_WORDS = {2: "Two", 3: "Three"}

_PARTS = ["pump", "sensor", "valve", "relay"]


def _payload(value: Fraction) -> dict:
    return {"p": value.numerator, "q": value.denominator}


def _enumerate_at_least_one(n: int, d: int, k: int) -> Fraction:
    """Brute-force P(at least one faulty) over every k-subset of n items, d of them faulty."""
    faulty = set(range(d))
    hits = 0
    total = 0
    for subset in itertools.combinations(range(n), k):
        total += 1
        if faulty.intersection(subset):
            hits += 1
    return Fraction(hits, total)


def _enumerate_patterns(probs: list[Fraction]) -> Fraction:
    """Sum the probability of every fail/work pattern with at least one failure."""
    total = Fraction(0)
    for pattern in itertools.product([True, False], repeat=len(probs)):
        if not any(pattern):
            continue
        weight = Fraction(1)
        for failed, p in zip(pattern, probs):
            weight *= p if failed else 1 - p
        total += weight
    return total


def _sampling(rng: random.Random) -> dict:
    noun, bad = rng.choice(_ITEMS)
    k = rng.choice([2, 3])
    n = rng.randint(8, 14)
    d = rng.randint(2, 4)
    check(n - d >= k, "not enough good items for the sample")
    none = Fraction(comb(n - d, k), comb(n, k))
    answer = 1 - none
    check_equivalent(answer, _enumerate_at_least_one(n, d, k), "enumeration disagrees with the complement")
    with_replacement = 1 - Fraction(n - d, n) ** k
    cands = [
        none,  # forgot to take the complement
        with_replacement,  # treats the draws as independent
        Fraction(k * d, n),  # adds the single-draw probabilities
        Fraction(d, n),  # one draw only
        1 - Fraction(comb(d, k), comb(n, k)),  # complements the event 'all chosen are faulty'
    ]
    text = (
        f"A box holds {n} {noun}, of which {d} are {bad}. {_WORDS[k]} of the {noun} are chosen at random without replacement. "
        f"Find the probability that at least one chosen item is {bad}."
    )
    return dict(answer=answer, text=text, cands=cands)


def _components(rng: random.Random) -> dict:
    count = rng.choice([2, 3])
    dens = rng.sample([3, 4, 5, 6, 8, 10], count)
    probs = []
    for den in dens:
        nums = [x for x in range(1, den) if x != den - x]
        probs.append(Fraction(rng.choice(nums), den))
    names = rng.sample(_PARTS, count)
    answer = 1 - _product(1 - p for p in probs)
    check_equivalent(answer, _enumerate_patterns(probs), "pattern enumeration disagrees with the complement")
    never = _product(1 - p for p in probs)
    cands = [
        never,  # forgot to take the complement
        1 - _product(probs),  # complements the product of failures
        sum(probs, Fraction(0)),  # adds the failure probabilities
        _product(probs),  # all fail
    ]
    cands = [c for c in cands if 0 < c < 1]
    parts = ", ".join(f"{nm} {rational_text(p)}" for nm, p in zip(names, probs))
    text = (
        f"A machine has {count} independent parts. In one day the probabilities of failure are: {parts}. "
        "Find the probability that at least one part fails during the day."
    )
    return dict(answer=answer, text=text, cands=cands)


def _product(values) -> Fraction:
    out = Fraction(1)
    for v in values:
        out *= v
    return out


@generator(
    "exd_x_s1_complement",
    level=10,
    presets={"exam": dict(tier="exam")},
)
def exd_x_s1_complement(rng: random.Random, tier: str = "exam") -> dict:
    """At-least-one probability found through the complement; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(50):
        built = _sampling(rng) if rng.random() < 0.5 else _components(rng)
        answer: Fraction = built["answer"]
        built["cands"] = [c for c in built["cands"] if 0 < c < 1]
        check(0 < answer < 1, "probability out of range")
        answer_text = rational_text(answer)
        distractors = pick_distractors(answer_text, [rational_text(c) for c in built["cands"]])
        if len(distractors) >= 2:
            break
    else:
        raise RuntimeError("could not build enough distractors")
    return dict(
        text=built["text"],
        latex=f"plain:{built['text']}",
        answer=_payload(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            "P(at least one) = 1 - P(none).",
            f"P(none) = {rational_text(1 - answer)}, so the answer is {answer_text}.",
        ],
        distractors=distractors,
    )
