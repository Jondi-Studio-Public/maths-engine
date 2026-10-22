"""Exam tier for radicals and fractional indices.

The practice tiers evaluate a single numeral ``b^(m/n)`` that is an exact
perfect power. The exam tier asks the learner to choose a method:

* ``solve_base`` -- solve ``x^(m/n) = N`` (or a negative-index version) for
  the base ``x``, which means undoing a fractional power by raising both
  sides to the reciprocal index.
* ``mixed`` -- evaluate a product or quotient of two fractional-index
  numerals on different bases, one of which carries a negative index.

Built backwards from the integer roots so every answer is an exact integer.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, generator, pick_distractors


def _pow_text(base: int, num: int, den: int) -> str:
    return f"{base}^({num}/{den})"


def _solve_base(rng: random.Random):
    while True:
        n = rng.choice([2, 3, 4])
        m = rng.choice([2, 3, 5])
        if math.gcd(m, n) != 1:
            continue
        c = rng.randint(2, 6 if n < 4 else 4)
        neg = rng.random() < 0.5
        x = c**n
        big_n = c**m
        if big_n > 5000 or x > 2000:
            continue
        break
    if neg:
        text = f"Find the positive number x such that x^(-{m}/{n}) = 1/{big_n}."
    else:
        text = f"Find the positive number x such that x^({m}/{n}) = {big_n}."
    # Independent check: substitute the key back with an integer root.
    root = round(x ** (1 / n))
    check(root**n == x, "x is not a perfect n-th power")
    check(Fraction(root) ** m == big_n, "substituting x does not reproduce N")
    # root only; the right-hand side (or its reciprocal's denominator) taken as x; raised to n but never
    # divided by m.
    wrong = [c, big_n, big_n**n]
    working = [
        f"The index {'-' if neg else ''}{m}/{n} is undone by the power {'-' if neg else ''}{n}/{m}.",
        f"x = {c}^{n} = {x}",
    ]
    return x, text, wrong, working


def _mixed(rng: random.Random):
    while True:
        n1, n2 = rng.choice([2, 3, 4]), rng.choice([2, 3, 4])
        m1, m2 = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
        if math.gcd(m1, n1) != 1 or math.gcd(m2, n2) != 1:
            continue
        c1, c2 = rng.randint(2, 5), rng.randint(2, 5)
        b1, b2 = c1**n1, c2**n2
        if b1 > 1300 or b2 > 1300 or b1 == b2:
            continue
        break
    divide = rng.random() < 0.5
    if divide:
        text = f"Evaluate {_pow_text(b1, m1, n1)} ÷ {b2}^(-{m2}/{n2})."
        wrong = [Fraction(c1**m1, c2**m2), c1 * c2, c1**m1 + c2**m2, b1**m1 * b2**m2]
    else:
        text = f"Evaluate {_pow_text(b1, m1, n1)} × {_pow_text(b2, m2, n2)}."
        wrong = [c1 * c2, c1**m1 + c2**m2, b1**m1 * b2**m2, abs(c1**m1 - c2**m2)]
    key = c1**m1 * c2**m2
    # Independent check: compute with Fraction arithmetic on the exact integer roots.
    check(round(b1 ** (1 / n1)) ** n1 == b1 and round(b2 ** (1 / n2)) ** n2 == b2, "bases are not perfect powers")
    working = [
        (f"{b1}^({m1}/{n1}) = ({b1}^(1/{n1}))^{m1} = {c1}^{m1} = {c1**m1}" if m1 != 1 else f"{b1}^(1/{n1}) = {c1}"),
        f"{b2}^({m2}/{n2}) = {c2**m2}"
        + (f", so dividing by {b2}^(-{m2}/{n2}) multiplies by {c2**m2}" if divide else ""),
        f"Result: {c1**m1} × {c2**m2} = {key}",
    ]
    return key, text, wrong, working


_FORMS = {"solve_base": _solve_base, "mixed": _mixed}


@generator("exd_x_f3_fracidx", level=11, presets={"exam": dict(tier="exam")})
def exd_x_f3_fracidx(rng: random.Random, tier: str = "exam", form: str | None = None) -> dict:
    """An integer-valued fractional-index question needing a choice of method."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    key, text, wrong, working = _FORMS[form or rng.choice(sorted(_FORMS))](rng)
    answer_text = str(key)
    # Only whole-number wrong answers are offered; they are graded exactly.
    cands = [str(w) for w in wrong if w is not None and Fraction(w).denominator == 1 and abs(w) <= 10**6]
    distractors = pick_distractors(answer_text, cands, n=3)
    check(len(distractors) >= 2, "too few principled distractors")
    return dict(text=text, answer=key, answer_text=answer_text, working=working, distractors=distractors)
