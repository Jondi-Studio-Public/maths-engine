"""Exam-tier source for `non_uniform_discrete_random_variable`: the distribution is given by a formula with an
unknown constant, ``P(X = x) = k * f(x)``, and the question asks for the variance.

The learner has to compose three ideas: use "probabilities sum to 1" to find ``k``, compute ``E(X)``, then compute
``E(X^2)`` and combine them as ``E(X^2) - E(X)^2``. The answer is an exact fraction.

Built from the chosen form and support, so ``k`` is exact. The key is cross-checked by a second route (the
definition ``Var(X) = sum p * (x - mu)^2``). Draws whose variance is a whole number are skipped so the answer is
always a genuine fraction.

Distractors, each a named slip:

* ``E(X^2)`` -- forgets to subtract the squared mean.
* ``E(X^2) - E(X)`` -- subtracts the mean instead of its square.
* the variance of the uniform distribution on the same values -- ignores the stated probabilities.
* ``E(X)`` -- reports the mean instead of the variance.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, latex_answer, pick_distractors, rational_text


def _draw_form(rng: random.Random):
    """(label shown in the prompt, support, weights): a random positive formula on a random run of integers."""
    kind = rng.choice(["up", "down", "square", "square_plus"])
    size = rng.randint(3, 5)
    if kind == "up":
        c = rng.choice([0, 1, 2, 3])
        start = rng.choice([0, 1, 2] if c else [1, 2])  # a zero weight would make P(X = x) = 0
        xs = list(range(start, start + size))
        label = "kx" if c == 0 else f"k(x + {c})"
        return label, xs, [x + c for x in xs]
    if kind == "down":
        start = rng.choice([0, 1])
        xs = list(range(start, start + size))
        top = xs[-1] + rng.choice([1, 2, 3])
        return f"k({top} - x)", xs, [top - x for x in xs]
    xs = list(range(1, 1 + min(size, 4)))
    if kind == "square":
        return "kx^2", xs, [x * x for x in xs]
    xs = list(range(0, min(size, 4)))
    return "k(x^2 + 1)", xs, [x * x + 1 for x in xs]


def _stats(xs, ws):
    total = sum(ws)
    probs = [Fraction(w, total) for w in ws]
    mean = sum(p * x for p, x in zip(probs, xs))
    ex2 = sum(p * x * x for p, x in zip(probs, xs))
    return probs, mean, ex2


@generator(
    "exd_x_s3_rv",
    level=11,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s3_rv(rng: random.Random, tier: str = "exam"):
    """Variance of a discrete random variable whose probabilities are k times a formula in x."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        label, xs, ws = _draw_form(rng)
        check(all(w > 0 for w in ws), "every weight must be positive")
        probs, mean, ex2 = _stats(xs, ws)
        var = ex2 - mean * mean
        if var.denominator != 1:
            break
    else:
        raise RuntimeError("no draw gave a fractional variance")

    # Independent route: the definition of variance.
    var_def = sum(p * (x - mean) ** 2 for p, x in zip(probs, xs))
    check(var_def == var, "variance by definition disagrees with E(X^2) - E(X)^2")
    check(sum(probs) == 1, "probabilities must sum to 1")

    n = len(xs)
    uniform_mean = Fraction(sum(xs), n)
    uniform_var = Fraction(sum(x * x for x in xs), n) - uniform_mean**2
    cands = [ex2, ex2 - mean, uniform_var, mean]
    answer_text = rational_text(var)
    distractors = pick_distractors(answer_text, [rational_text(c) for c in cands])
    check(len(distractors) == 3, "need three distinct wrong options")

    values = ", ".join(str(x) for x in xs[:-1]) + f" and {xs[-1]}"
    text = (
        f"The random variable X takes the values {values}, with P(X = x) = {label} for each of these values, "
        f"where k is a constant. Find Var(X), giving your answer as a fraction."
    )
    working = [
        f"The probabilities sum to 1, so k = 1/{sum(ws)}.",
        f"E(X) = {rational_text(mean)} and E(X^2) = {rational_text(ex2)}.",
        f"Var(X) = E(X^2) - E(X)^2 = {answer_text}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": var.numerator, "q": var.denominator},
        answer_text=answer_text,
        answer_latex=latex_answer(var),
        working=working,
        distractors=distractors,
    )
