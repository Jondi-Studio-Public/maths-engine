"""Exam source for `discrete-expected-value`: a distribution that is not handed over whole.

Two compositions, neither reachable by weighting listed probabilities:

* ``square`` -- probabilities are proportional to ``k + s`` over ``k = 1..m``
  with the constant left unstated, and the quantity asked for is ``E[X^2]``.
  The learner must normalise first, then take the expectation of a function of X.
* ``ratio`` -- two probabilities are given and the remaining mass is shared
  between two outcomes in a stated ratio.

Answers are exact fractions, computed with `fractions.Fraction` from the drawn values.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, phrase, pick_distractors

_SQUARE = [
    "A random variable X takes the values {ks} with P(X = k) = c(k + {s}), where c is a constant. Find E[X^2].",
    "The probability that X equals k is proportional to k + {s} for k = {ks}. Find E[X^2].",
]
_RATIO = [
    "X takes the values {vs}. P(X = {v1}) = {p1} and P(X = {v2}) = {p2}. The remaining probability is shared "
    "between X = {v3} and X = {v4} in the ratio {r}:{t}. Find E[X].",
    "A game pays {vs} dollars. The first two payouts have probabilities {p1} and {p2}; the rest of the "
    "probability is split between the last two payouts in the ratio {r}:{t}. Find the expected payout.",
]


def _text(fr: Fraction) -> str:
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def _square(rng: random.Random) -> dict:
    for _ in range(100):
        m = rng.randint(3, 5)
        s = rng.randint(1, 4)
        ks = list(range(1, m + 1))
        weights = [k + s for k in ks]
        total = sum(weights)
        probs = [Fraction(w, total) for w in weights]
        check(sum(probs) == 1, "probabilities must sum to 1")
        answer = sum(k * k * p for k, p in zip(ks, probs))
        mean = sum(k * p for k, p in zip(ks, probs))
        cands = [
            mean * mean,  # squares the mean
            Fraction(sum(k * k * w for k, w in zip(ks, weights))),  # forgets to normalise
            Fraction(sum(k * k for k in ks), m),  # equal weights
            answer - mean * mean,  # variance
        ]
        if len({answer, *cands}) == len(cands) + 1:
            break
    else:
        raise RuntimeError("exd_x_s0_expval: no collision-free square draw")
    text = phrase(rng, _SQUARE, ks=", ".join(map(str, ks)), s=s)
    at = _text(answer)
    return dict(
        text=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=at,
        answer_latex=at,
        working=[f"c = 1/{total}; E[X^2] = {sum(k * k * w for k, w in zip(ks, weights))}/{total} = {at}."],
        distractors=pick_distractors(at, [_text(c) for c in cands]),
    )


def _ratio(rng: random.Random) -> dict:
    for _ in range(200):
        den = rng.choice([12, 15, 20, 24, 30])
        a, b = rng.randint(1, den // 3), rng.randint(1, den // 3)
        r, t = rng.sample(range(1, 6), 2)
        vs = sorted(rng.sample(range(-6, 13), 4))
        p1, p2 = Fraction(a, den), Fraction(b, den)
        rest = 1 - p1 - p2
        p3, p4 = rest * Fraction(r, r + t), rest * Fraction(t, r + t)
        probs = [p1, p2, p3, p4]
        check(sum(probs) == 1, "probabilities must sum to 1")
        answer = sum(v * p for v, p in zip(vs, probs))
        cands = [
            vs[0] * p1 + vs[1] * p2 + vs[2] * p4 + vs[3] * p3,  # ratio applied the wrong way round
            vs[0] * p1 + vs[1] * p2 + (vs[2] + vs[3]) * rest / 2,  # shares the rest equally
            vs[0] * p1 + vs[1] * p2 + vs[2] * Fraction(r, r + t) + vs[3] * Fraction(t, r + t),  # ratio as probabilities
            Fraction(sum(vs), 4),  # unweighted mean
        ]
        if len({answer, *cands}) == len(cands) + 1:
            break
    else:
        raise RuntimeError("exd_x_s0_expval: no collision-free ratio draw")
    text = phrase(
        rng,
        _RATIO,
        vs=", ".join(map(str, vs)),
        v1=vs[0],
        v2=vs[1],
        v3=vs[2],
        v4=vs[3],
        p1=_text(p1),
        p2=_text(p2),
        r=r,
        t=t,
    )
    at = _text(answer)
    return dict(
        text=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=at,
        answer_latex=at,
        working=[f"Remaining probability = {_text(rest)}; split {r}:{t}. E = {at}."],
        distractors=pick_distractors(at, [_text(c) for c in cands]),
    )


@generator(
    "exd_x_s0_expval",
    level=12,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s0_expval(rng: random.Random, tier: str = "exam") -> dict:
    """E[X^2] of a proportional distribution, or E[X] with a ratio-split tail. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _square(rng) if rng.random() < 0.5 else _ratio(rng)
