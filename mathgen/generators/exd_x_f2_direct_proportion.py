"""Exam-tier source for direct proportion ``y = kx``: relations the learner has to set up before using.

The practice tiers scale one input and read off the output. Here the proportionality is still ``y = kx``, but the
question asks about a combination of the two variables, a change in the input expressed as a percentage, or the
size of a change, so the learner has to decide how to use the constant of proportionality.

Three question shapes, each built backwards from chosen whole numbers so the key is exact:

- ``constraint``: one data point is given, and the learner finds ``x`` when ``x + y`` (or ``y - x``) takes a
  stated value. The relation turns this into one equation in one unknown.
- ``percent_chain``: ``x`` is increased or decreased by one percentage and then by another; find the new ``y``.
  The two scale factors multiply (they do not add).
- ``change``: find how much ``y`` changes when ``x`` moves from one stated value to another.

Every key is cross-checked from the printed data with exact fractions: the constant ``k`` is recovered and the
unknown recomputed by a different route from the one used to build it. Distractors, each a named slip:

- ``constraint``: assuming ``x = y`` and halving; giving the value of ``y`` instead of ``x``; subtracting the known
  value of ``x`` from the stated total (or reporting the difference itself as ``x``).
- ``percent_chain``: adding the two percentages instead of multiplying the factors; applying only the first
  change; applying only the second.
- ``change``: reporting the final value of ``y`` instead of the change; reporting the starting value of ``y``;
  reporting the change in ``x`` and forgetting ``k``.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import VerificationError, check_equivalent, check_pairwise_distinct, generator, phrase, pick_distractors

_RESAMPLE = 300

_CONSTRAINT = (
    "y is directly proportional to x, so y = kx. When x = {x1}, y = {y1}. Find the value of x for which {cond}.",
    "The variable y varies directly as x, so y = kx, and y = {y1} when x = {x1}. For what value of x is {cond}?",
)
_PERCENT = (
    "y is directly proportional to x, so y = kx. When x = {x1}, y = {y1}. x is {c1} by {p1}% and the result is "
    "then {c2} by {p2}%. What is the new value of y?",
    "The variable y varies directly as x, so y = kx, and y = {y1} when x = {x1}. The value of x is {c1} by {p1}%, "
    "and the new value is then {c2} by {p2}%. Find the final value of y.",
)
_CHANGE = (
    "y is directly proportional to x, so y = kx. When x = {x1}, y = {y1}. By how much does y increase when x "
    "increases from {p} to {q}?",
    "The variable y varies directly as x, so y = kx, and y = {y1} when x = {x1}. Find the increase in y when x "
    "goes up from {p} to {q}.",
)


def _constraint(rng: random.Random) -> dict:
    for _ in range(_RESAMPLE):
        m, n = rng.randint(1, 7), rng.randint(1, 7)
        if m == n or math.gcd(m, n) != 1:
            continue
        a, t = rng.randint(1, 4), rng.randint(2, 9)
        x1, y1 = m * a, n * a
        use_sum = n < m or rng.random() < 0.5
        if use_sum:
            total = (m + n) * t
            cond = f"x + y = {total}"
            wrong = [Fraction(total, 2), Fraction(n * t), Fraction(total - x1)]
        else:
            if n < m:
                continue
            total = (n - m) * t
            cond = f"y - x = {total}"
            wrong = [Fraction(total, 2), Fraction(n * t), Fraction(total)]
        answer = m * t
        if any(w.denominator != 1 or w <= 0 for w in wrong):
            continue
        wrong_i = [int(w) for w in wrong]
        printed = {x1, y1, total}
        if set(wrong_i) & printed or answer in printed:
            continue
        try:
            check_pairwise_distinct(answer, *wrong_i)
        except VerificationError:
            continue
        break
    else:
        raise VerificationError("constraint: no collision-free draw")
    # Independent route: recover k from the printed data, then check the stated condition holds at the key.
    k = Fraction(y1, x1)
    x_key, y_key = Fraction(answer), k * answer
    check_equivalent(
        x_key + y_key if use_sum else y_key - x_key, Fraction(total), "constraint: condition fails at the key"
    )
    text = phrase(rng, _CONSTRAINT, x1=x1, y1=y1, cond=cond)
    lhs = f"x + ({k})x" if use_sum else f"({k})x - x"
    working = [
        f"k = y / x = {y1} / {x1} = {k}.",
        f"y = ({k})x, so the condition becomes {lhs} = {total}.",
        f"Hence x = {answer}.",
    ]
    return _build(text, answer, wrong_i, working)


def _percent_chain(rng: random.Random) -> dict:
    for _ in range(_RESAMPLE):
        p1, p2 = rng.choice([10, 20, 25, 50]), rng.choice([10, 20, 25, 50])
        s1, s2 = rng.choice([1, -1]), rng.choice([1, -1])
        if s1 == s2 and p1 == p2:
            continue
        f1, f2 = 1 + Fraction(s1 * p1, 100), 1 + Fraction(s2 * p2, 100)
        factors = [f1 * f2, 1 + Fraction(s1 * p1 + s2 * p2, 100), f1, f2]
        lcm = 1
        for f in factors:
            lcm = lcm * f.denominator // math.gcd(lcm, f.denominator)
        y1 = lcm * rng.randint(1, 3)
        if y1 > 400 or y1 < 10:
            continue
        x1 = rng.randint(2, 9) * 10
        values = [y1 * f for f in factors]
        answer, wrong = values[0], values[1:]
        if len(set(values)) != 4 or any(v.denominator != 1 for v in values):
            continue
        printed = {x1, y1, p1, p2}
        if set(int(w) for w in wrong) & printed or int(answer) in printed:
            continue
        break
    else:
        raise VerificationError("percent_chain: no collision-free draw")
    # Independent route: apply each change to x itself, then use k = y1 / x1.
    x_new = x1 * f1 * f2
    check_equivalent(Fraction(y1, x1) * x_new, answer, "percent_chain: key disagrees with k * x_new")
    answer_i, wrong_i = int(answer), [int(w) for w in wrong]
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
        f"y is multiplied by the same factors as x: {f1} then {f2}.",
        f"New y = {y1} × {f1} × {f2} = {answer_i}.",
    ]
    return _build(text, answer_i, wrong_i, working)


def _change(rng: random.Random) -> dict:
    for _ in range(_RESAMPLE):
        m, n = rng.randint(1, 6), rng.randint(1, 9)
        if math.gcd(m, n) != 1 or m == n:
            continue
        a = rng.randint(1, 4)
        x1, y1 = m * a, n * a
        p = rng.randint(1, 6) * m
        q = p + rng.randint(1, 5) * m
        k = Fraction(n, m)
        answer = k * (q - p)
        wrong = [k * q, k * p, Fraction(q - p)]
        if any(v.denominator != 1 for v in [answer] + wrong) or len({answer, *wrong}) != 4:
            continue
        printed = {x1, y1, p, q}
        if {int(w) for w in wrong} & printed or int(answer) in printed:
            continue
        break
    else:
        raise VerificationError("change: no collision-free draw")
    # Independent route: the two values of y from the printed data, then their difference.
    y_p, y_q = Fraction(y1, x1) * p, Fraction(y1, x1) * q
    check_equivalent(y_q - y_p, answer, "change: key disagrees with y(q) - y(p)")
    text = phrase(rng, _CHANGE, x1=x1, y1=y1, p=p, q=q)
    working = [
        f"k = y / x = {y1} / {x1} = {k}.",
        f"y at x = {p} is {y_p}; y at x = {q} is {y_q}.",
        f"The increase is {y_q} - {y_p} = {int(answer)}.",
    ]
    return _build(text, int(answer), [int(w) for w in wrong], working)


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
    "exd_x_f2_direct_proportion",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f2_direct_proportion(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier direct-proportion question: a condition on x and y, percentage changes, or a change in y."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}; this source only has 'exam'")
    shape = rng.choice(["constraint", "percent_chain", "change"])
    if shape == "constraint":
        return _constraint(rng)
    if shape == "percent_chain":
        return _percent_chain(rng)
    return _change(rng)
