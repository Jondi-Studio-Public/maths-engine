"""Generates `exd_x_s1_variance` questions: variance of a finite distribution that is not handed over as a
ready-made table.

Two forms, each answered as an exact fraction:

* a distribution table is given, but the quantity wanted is ``Var(aX + b)`` for a stated linear
  transform -- the learner must know how variance scales and shifts, not just compute it;
* the probabilities are only described as proportional to ``x`` (or ``x^2``) on ``x = 1..n`` -- the
  learner must normalise first, then compute ``Var(X)``.

The first form is cross-checked by pushing the transformed values through the definition of variance
directly; the second by building the explicit pmf and applying the centred definition.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, check_equivalent, generator, pick_distractors, rational_text


def _payload(value: Fraction) -> dict:
    return {"p": value.numerator, "q": value.denominator}


def _moments(values: list[int], weights: list[int]) -> tuple[Fraction, Fraction, Fraction]:
    total = sum(weights)
    mean = Fraction(sum(v * w for v, w in zip(values, weights)), total)
    second = Fraction(sum(v * v * w for v, w in zip(values, weights)), total)
    return mean, second, second - mean * mean


def _centred(values: list, weights: list[int]) -> Fraction:
    total = sum(weights)
    mean = sum(Fraction(v) * w for v, w in zip(values, weights)) / total
    return sum((Fraction(v) - mean) ** 2 * w for v, w in zip(values, weights)) / total


def _transform_form(rng: random.Random) -> dict:
    values = sorted(rng.sample(range(-6, 7), rng.choice([3, 4])))
    weights = [rng.randint(1, 5) for _ in values]
    total = sum(weights)
    a = rng.choice([-4, -3, -2, 2, 3, 4])
    b = rng.choice([x for x in range(-5, 6) if x != 0])
    _, _, var = _moments(values, weights)
    check(var > 0, "degenerate distribution")
    answer = a * a * var
    new_values = [a * v + b for v in values]
    check_equivalent(answer, _centred(new_values, weights), "direct variance of aX+b disagrees with a^2 Var(X)")
    cands = [
        var,  # ignores the transform
        abs(a) * var,  # scales by a instead of a^2
        a * a * var + b,  # shifts the variance
        a * var + b,  # treats Var like a mean
        (a * a + 1) * var if b else var,
    ]
    table = ", ".join(f"P(X={v})={w}/{total}" for v, w in zip(values, weights))
    shift = f"+ {b}" if b > 0 else f"- {-b}"
    text = f"A random variable X has {table}. Let Y = {a}X {shift}. Find Var(Y)."
    return dict(answer=answer, text=text, cands=cands)


def _proportional_form(rng: random.Random) -> dict:
    n = rng.randint(4, 6)
    power = rng.choice([1, 2])
    values = list(range(1, n + 1))
    weights = [v**power for v in values]
    mean, second, var = _moments(values, weights)
    check_equivalent(var, _centred(values, weights), "centred variance disagrees")
    uniform_mean = Fraction(n + 1, 2)
    uniform_var = Fraction(n * n - 1, 12)
    cands = [
        uniform_var,  # treats the outcomes as equally likely
        second,  # second moment only
        mean * mean,  # squared mean only
        second - uniform_mean**2,  # right second moment, uniform mean
    ]
    shape = "x" if power == 1 else "x^2"
    text = f"The random variable X takes the values 1, 2, ..., {n}, with P(X=x) proportional to {shape}. Find Var(X)."
    return dict(answer=var, text=text, cands=cands)


@generator(
    "exd_x_s1_variance",
    level=14,
    presets={"exam": dict(tier="exam")},
)
def exd_x_s1_variance(rng: random.Random, tier: str = "exam") -> dict:
    """Variance of a transformed or only-proportionally-specified finite distribution."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(50):
        built = _transform_form(rng) if rng.random() < 0.5 else _proportional_form(rng)
        answer: Fraction = built["answer"]
        check(answer > 0, "variance must be positive")
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
        working=[f"Var = {answer_text}."],
        distractors=distractors,
    )
