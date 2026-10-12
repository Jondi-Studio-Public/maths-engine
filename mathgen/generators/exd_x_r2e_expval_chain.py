"""Exam source for `discrete-expected-value`: a four-part chain on one distribution with an unknown constant.

The probabilities of a discrete random variable ``X`` are stated as multiples of an unknown constant ``c``, either
by a formula ``P(X = x) = c(x + s)`` over consecutive values or by a list of multiples such as ``c, 3c, 2c, 4c``.

- (a) the constant ``c``, from the probabilities summing to 1 (2 marks);
- (b) ``E(X)`` (2 marks, ``follow_through`` from (a));
- (c) ``E(X^2)`` (2 marks, ``follow_through`` from (a));
- (d) ``Var(X) = E(X^2) - E(X)^2`` (2 marks, ``follow_through`` from (b) and (c)).

## Construction

The weights are drawn first and the constant is their total's reciprocal, so every answer is an exact
fraction, computed with `fractions.Fraction`. The values are chosen so that ``Var(X)`` is positive. A learner who
forgets to normalise still gets a coherent set of wrong answers, and the follow-through expressions are written
with this question's weights so a wrong ``c`` carries into (b) and (c).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, latex_answer, pick_distractors, rational_text


def _text(f: Fraction) -> str:
    return rational_text(f)


def _tag(value: Fraction, tag: str) -> Distractor:
    return Distractor(_text(value), tag, exact_value(value))


@generator("exd_x_r2e_expval_chain", level=12, presets={"exam": dict(tier="exam")})
def expval_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Normalise a distribution given with an unknown constant, then find E(X), E(X^2) and Var(X)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    formula = rng.random() < 0.5
    for _ in attempts(2000, "a distribution with a moderate total weight"):
        if formula:
            m = rng.randint(4, 6)
            s = rng.randint(1, 5)
            lo = rng.randint(0, 3)
            values = list(range(lo, lo + m))
            weights = [v + s for v in values]
        else:
            m = rng.randint(4, 5)
            values = sorted(rng.sample(range(-3, 10), m))
            weights = [rng.randint(1, 6) for _ in values]
        total = sum(weights)
        if not 8 <= total <= 32 or len(set(weights)) == 1 or sum(v * w for v, w in zip(values, weights)) == 0:
            continue
        break

    c = Fraction(1, total)
    probs = [c * w for w in weights]
    check(sum(probs) == 1, "probabilities must sum to 1")
    mean = sum(v * p for v, p in zip(values, probs))
    second = sum(v * v * p for v, p in zip(values, probs))
    var = second - mean * mean
    check(var > 0, "variance must be positive")
    check(var == sum((v - mean) ** 2 * p for v, p in zip(values, probs)), "variance by definition disagrees")

    if formula:
        stem = (
            f"A discrete random variable X takes the values {', '.join(map(str, values))}, and "
            f"P(X = x) = c(x + {s}) for each of these values, where c is a constant."
        )
    else:
        pieces = [f"P(X = {v}) = {'' if w == 1 else w}c" for v, w in zip(values, weights)]
        stem = f"A discrete random variable X has {', '.join(pieces[:-1])} and {pieces[-1]}, where c is a constant."

    raw_mean = sum(v * w for v, w in zip(values, weights))
    raw_second = sum(v * v * w for v, w in zip(values, weights))
    a_cands = [
        _tag(Fraction(total), "inverted-ratio"),
        _tag(Fraction(1, m), "assumes-equally-likely"),
        _tag(Fraction(1, total + 1), "off-by-one-count"),
        _tag(Fraction(1, total - 1), "off-by-one-count"),
    ]
    b_cands = [
        _tag(Fraction(raw_mean), "forgets-to-normalise"),
        _tag(Fraction(sum(values), m), "unweighted-mean-for-expected-value"),
        _tag(second, "second-moment-for-expected-value"),
        _tag(Fraction(sum(v * w for v, w in zip(values, weights)), m), "divides-by-the-number-of-values"),
    ]
    c_cands = [
        _tag(mean * mean, "squares-the-mean"),
        _tag(Fraction(raw_second), "forgets-to-normalise"),
        _tag(Fraction(sum(v * v for v in values), m), "unweighted-mean-for-expected-value"),
        _tag(mean, "first-moment-for-second-moment"),
    ]
    d_cands = [
        _tag(second, "forgets-to-subtract-the-mean-squared"),
        _tag(second - mean, "subtracts-the-mean-not-its-square"),
        _tag(mean * mean - second, "sign-reversed-result"),
        _tag(mean, "first-moment-for-second-moment"),
    ]
    picks = [
        pick_distractors(_text(c), a_cands, 3),
        pick_distractors(_text(mean), b_cands, 3),
        pick_distractors(_text(second), c_cands, 3),
        pick_distractors(_text(var), d_cands, 3),
    ]
    check(all(len(p) >= 2 for p in picks), f"a part has too few distractors: {[len(p) for p in picks]}")
    sum_vw = " + ".join(f"{v}*{w}" for v, w in zip(values, weights))
    sum_v2w = " + ".join(f"{v * v}*{w}" for v, w in zip(values, weights))
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of c.",
                answer_kind="fraction",
                answer=exact_value(c),
                answer_text=_text(c),
                answer_latex=latex_answer(_text(c)),
                marks=2,
                sensible_range=(0, 1),
                working=[f"The probabilities sum to 1: {total}c = 1, so c = {_text(c)}."],
                distractors=picks[0],
            ),
            dict(
                prompt="Hence find E(X).",
                answer_kind="fraction",
                answer=exact_value(mean),
                answer_text=_text(mean),
                answer_latex=latex_answer(_text(mean)),
                marks=2,
                follow_through=f"part_a*({sum_vw})",
                working=[f"E(X) = sum of x P(X = x) = c({raw_mean}) = {_text(mean)}."],
                distractors=picks[1],
            ),
            dict(
                prompt="Find E(X^2).",
                answer_kind="fraction",
                answer=exact_value(second),
                answer_text=_text(second),
                answer_latex=latex_answer(_text(second)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*({sum_v2w})",
                working=[f"E(X^2) = sum of x^2 P(X = x) = c({raw_second}) = {_text(second)}."],
                distractors=picks[2],
            ),
            dict(
                prompt="Find Var(X).",
                answer_kind="fraction",
                answer=exact_value(var),
                answer_text=_text(var),
                answer_latex=latex_answer(_text(var)),
                marks=2,
                sensible_range=(0, None),
                follow_through="part_c - part_b**2",
                working=[f"Var(X) = E(X^2) - (E(X))^2 = {_text(second)} - ({_text(mean)})^2 = {_text(var)}."],
                distractors=picks[3],
            ),
        ],
    )
