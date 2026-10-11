"""Exam-tier source for inverse proportion ``y = k/x``: relations the learner has to set up before using.

The practice tiers scale one input and read off the output. Here the proportionality is still ``y = k/x``, but the
question asks about a condition linking ``x`` and ``y``, percentage changes in the input, or a job whose
workforce changes part-way through, so the learner has to decide how to use the constant of proportionality.

Three question shapes, each built backwards from chosen whole numbers so the key is exact:

- ``crossing``: one data point is given, and the learner finds the positive ``x`` at which ``y`` equals a stated
  multiple of ``x``. With ``y = c x`` and ``y = k/x`` this is ``c x^2 = k``.
- ``percent_chain``: ``x`` is increased or decreased by one percentage and then by another; find the new ``y``.
  The two scale factors multiply and ``y`` is divided by their product.
- ``workers``: ``N`` workers finish a job in ``D`` days; after ``d`` days some leave. The total work is the
  product ``N * D`` (worker-days). How many further days do the remaining workers need?

Every key is cross-checked from the printed data with exact fractions (the constant, or the worker-days
remaining). Distractors, each a named slip:

- ``crossing``: leaving ``x^2`` as the answer (no square root); dividing ``k`` by ``c`` and stopping there;
  multiplying the root by ``c``.
- ``percent_chain``: treating the relation as direct and multiplying ``y`` by the factors; applying the net
  percentage change to ``y`` with the opposite sign; undoing only the first change.
- ``workers``: forgetting that the first days already did some of the work; restarting the whole job with the
  smaller workforce; restarting the whole job with the smaller workforce and then subtracting the days already
  worked.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import VerificationError, check_equivalent, check_pairwise_distinct, generator, phrase, pick_distractors

_RESAMPLE = 300

_CROSSING = (
    "y is inversely proportional to x, so y = k/x. When x = {x1}, y = {y1}. For what positive value of x is "
    "y equal to {c} times x?",
    "The variable y varies inversely as x, so y = k/x, and y = {y1} when x = {x1}. Find the positive value of x "
    "at which y is {c} times as large as x.",
)
_PERCENT = (
    "y is inversely proportional to x, so y = k/x. When x = {x1}, y = {y1}. x is {c1} by {p1}% and the result is "
    "then {c2} by {p2}%. What is the new value of y?",
    "The variable y varies inversely as x, so y = k/x, and y = {y1} when x = {x1}. The value of x is {c1} by "
    "{p1}%, and the new value is then {c2} by {p2}%. Find the final value of y.",
)
_WORKERS = (
    "{n} workers, all working at the same rate, can finish a job in {total} days. After {d} days, {leave} of "
    "the workers leave and the rest carry on at the same rate. How many more days do the remaining workers "
    "need to finish the job?",
    "A job would take {n} workers {total} days if they all kept working at the same rate. After {d} days, "
    "{leave} workers are reassigned elsewhere. How many further days will the remaining workers need to "
    "complete the job?",
)


def _crossing(rng: random.Random) -> dict:
    for _ in range(_RESAMPLE):
        c = rng.randint(2, 4)
        r = rng.randint(2, 8)  # the key
        k = c * r * r
        divisors = [d for d in range(1, k + 1) if k % d == 0 and 2 <= d <= 40 and 2 <= k // d <= 60]
        if not divisors:
            continue
        x1 = rng.choice(divisors)
        y1 = k // x1
        wrong = [r * r, c * r * r, c * r]
        printed = {x1, y1, c}
        if set(wrong) & printed or r in printed:
            continue
        try:
            check_pairwise_distinct(r, *wrong)
        except VerificationError:
            continue
        break
    else:
        raise VerificationError("crossing: no collision-free draw")
    # Independent route: at the key, y = k / x must equal c * x, with k recovered from the printed data.
    check_equivalent(Fraction(x1 * y1, r), Fraction(c * r), "crossing: y at the key is not c times x")
    text = phrase(rng, _CROSSING, x1=x1, y1=y1, c=c)
    working = [
        f"k = x × y = {x1} × {y1} = {k}.",
        f"y = {k}/x and y = {c}x, so {c}x = {k}/x, giving x^2 = {k // c}.",
        f"The positive solution is x = {r}.",
    ]
    return _build(text, r, wrong, working)


def _percent_chain(rng: random.Random) -> dict:
    for _ in range(_RESAMPLE):
        p1, p2 = rng.choice([10, 20, 25, 50]), rng.choice([10, 20, 25, 50])
        s1, s2 = rng.choice([1, -1]), rng.choice([1, -1])
        if s1 == s2 and p1 == p2:
            continue
        f1, f2 = 1 + Fraction(s1 * p1, 100), 1 + Fraction(s2 * p2, 100)
        factors = [1 / (f1 * f2), f1 * f2, 1 - Fraction(s1 * p1 + s2 * p2, 100), 1 / f1]
        lcm = 1
        for f in factors:
            lcm = lcm * f.denominator // math.gcd(lcm, f.denominator)
        y1 = lcm * rng.randint(1, 3)
        if y1 > 600 or y1 < 10:
            continue
        x1 = rng.randint(2, 9) * 10
        values = [y1 * f for f in factors]
        if len(set(values)) != 4 or any(v.denominator != 1 for v in values):
            continue
        answer, wrong = int(values[0]), [int(v) for v in values[1:]]
        printed = {x1, y1, p1, p2}
        if set(wrong) & printed or answer in printed:
            continue
        break
    else:
        raise VerificationError("percent_chain: no collision-free draw")
    # Independent route: change x itself, then use k = x1 * y1.
    x_new = x1 * f1 * f2
    check_equivalent(Fraction(x1 * y1) / x_new, Fraction(answer), "percent_chain: key disagrees with k / x_new")
    text = phrase(
        rng,
        _PERCENT,
        x1=x1,
        y1=y1,
        c1="increased" if s1 > 0 else "decreased",
        p1=p1,
        c2="increased" if s2 > 0 else "decreased",
        p2=p2,
    )
    working = [
        f"x is multiplied by {f1} and then by {f2}, so y is divided by ({f1 * f2}).",
        f"New y = {y1} ÷ ({f1 * f2}) = {answer}.",
    ]
    return _build(text, answer, wrong, working)


def _workers(rng: random.Random) -> dict:
    for _ in range(_RESAMPLE):
        n = rng.randint(6, 20)
        left = rng.randint(2, n - 2)  # workers remaining
        leave = n - left
        unit = left // math.gcd(n, left)  # days must be a multiple of this so n * days / left is whole
        further = unit * rng.randint(1, 4)  # days still needed, as a multiple that keeps every candidate whole
        d = unit * rng.randint(1, 4)
        # n * further / left is the answer; make `further` the work remaining expressed in days at n workers
        remaining_days = further
        total = d + remaining_days
        if total > 60:
            continue
        work = n * total
        answer = Fraction(n * remaining_days, left)
        if answer.denominator != 1 or answer <= 0:
            continue
        restart = Fraction(work, left)
        if restart.denominator != 1:
            continue
        wrong = [remaining_days, int(restart), int(restart) - d]
        answer = int(answer)
        printed = {n, total, d, leave}
        if set(wrong) & printed or answer in printed or min(wrong) <= 0:
            continue
        try:
            check_pairwise_distinct(answer, *wrong)
        except VerificationError:
            continue
        break
    else:
        raise VerificationError("workers: no collision-free draw")
    # Independent route: worker-days left divided by the new workforce.
    worker_days_left = n * total - n * d
    check_equivalent(
        Fraction(worker_days_left, left), Fraction(answer), "workers: key disagrees with worker-days / workers"
    )
    text = phrase(rng, _WORKERS, n=n, total=total, d=d, leave=leave)
    working = [
        f"The whole job is {n} × {total} = {n * total} worker-days.",
        f"After {d} days, {n} × {d} = {n * d} worker-days are done, leaving {worker_days_left}.",
        f"{left} workers need {worker_days_left} / {left} = {answer} more days.",
    ]
    return _build(text, answer, wrong, working)


def _build(text: str, answer: int, wrong: list[int], working: list[str]) -> dict:
    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(str(answer), [str(w) for w in wrong], n=3),
    )


@generator(
    "exd_x_f2_inverse_proportion",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f2_inverse_proportion(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier inverse-proportion question: a crossing point, percentage changes, or a work-rate problem."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}; this source only has 'exam'")
    shape = rng.choice(["crossing", "percent_chain", "workers"])
    if shape == "crossing":
        return _crossing(rng)
    if shape == "percent_chain":
        return _percent_chain(rng)
    return _workers(rng)
