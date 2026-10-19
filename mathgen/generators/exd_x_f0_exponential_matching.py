"""Exam-tier source for `f10-exponential-relations-equations`: exponential equations whose two sides are
powers of DIFFERENT bases, with an unknown in a linear exponent on both sides.

The learner has to notice that the bases are powers of one prime (a base may be a reciprocal such as 1/8),
rewrite each side over that prime, and then solve the resulting linear equation. The answer is an exact
rational number.

Built backwards: the common prime `c`, the signed base-exponents `m` and `n` (so the bases are c^m and c^n),
and the linear exponents are drawn first; the key is then the solution of the linear equation, recomputed with
`Fraction` arithmetic and cross-checked by substituting it back into both sides with exact sympy powers.

  two_sides -- a^(p1 x + q1) = b^(p2 x + q2)
  product   -- a^(p1 x + q1) x b^(p2 x + q2) = c^k (the right-hand side shown as a number)
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Rational

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    fmt_signed,
    generator,
    join_terms,
    pick_distractors,
    rational_text,
)

_PRIMES = (2, 3)
_MAX_POWER = {2: 4, 3: 3}


def _scaled(k, inner: str) -> str:
    """k(inner) as an exponent: the bracket and the 1 are dropped when k is 1, and -(inner) is written for k = -1."""
    return inner if k == 1 else fmt_signed(k, f"({inner})")


def _base_text(c: int, k: int) -> str:
    """The base c**k as a learner would read it: 8, or (1/8) for a negative k."""
    return str(c**k) if k > 0 else f"(1/{c ** (-k)})"


def _exp_text(p: int, q: int) -> str:
    return join_terms([(p, "x"), (q, "")])


def _value_text(c: int, k: int) -> str:
    return str(c**k) if k >= 0 else f"1/{c ** (-k)}"


def _draw_power(rng: random.Random, c: int) -> int:
    return rng.choice([-1, 1]) * rng.randint(1, _MAX_POWER[c])


def _linear(rng: random.Random) -> tuple[int, int]:
    return rng.choice([-1, 1]) * rng.randint(1, 3), rng.randint(-6, 6)


def _two_sides(rng: random.Random) -> dict:
    for _ in range(300):
        c = rng.choice(_PRIMES)
        m, n = _draw_power(rng, c), _draw_power(rng, c)
        p1, q1 = _linear(rng)
        p2, q2 = _linear(rng)
        den = m * p1 - n * p2
        if m == n or den == 0:
            continue
        x = Fraction(n * q2 - m * q1, den)
        wrong = {
            "bases-ignored": Fraction(q2 - q1, p1 - p2) if p1 != p2 else None,
            "constants-sign": Fraction(n * q2 + m * q1, den),
            "bases-swapped": Fraction(m * q2 - n * q1, n * p1 - m * p2) if n * p1 != m * p2 else None,
            "reciprocal": 1 / x if x != 0 else None,
        }
        values = [x] + [w for w in wrong.values() if w is not None]
        if abs(x.numerator) > 30 or abs(x.denominator) > 12 or len(values) < 4 or len(set(values)) != len(values):
            continue
        break
    else:
        check(False, "no usable two-sided exponential equation")
    # substitute back, comparing the two sides directly
    xs = Rational(x.numerator, x.denominator)
    lhs = Rational(c) ** (m * (p1 * xs + q1))
    rhs = Rational(c) ** (n * (p2 * xs + q2))
    check_equivalent(lhs, rhs, "substituting the solution does not balance the two sides")
    text = f"Solve {_base_text(c, m)}^({_exp_text(p1, q1)}) = {_base_text(c, n)}^({_exp_text(p2, q2)}) for x."
    working = [
        f"Write both bases as powers of {c}: {c}^({_scaled(m, _exp_text(p1, q1))}) = {c}^({_scaled(n, _exp_text(p2, q2))}).",
        f"Equate the exponents and solve: x = {rational_text(x)}.",
    ]
    return _pack(text, x, working, [w for w in wrong.values() if w is not None])


def _product(rng: random.Random) -> dict:
    for _ in range(300):
        c = rng.choice(_PRIMES)
        m, n = _draw_power(rng, c), _draw_power(rng, c)
        p1, q1 = _linear(rng)
        p2, q2 = _linear(rng)
        k = rng.choice([-1, 1]) * rng.randint(1, _MAX_POWER[c] + 2)
        den = m * p1 + n * p2
        if den == 0 or m == n:
            continue
        x = Fraction(k - m * q1 - n * q2, den)
        wrong = {
            "constants-sign": Fraction(k + m * q1 + n * q2, den),
            "second-base-unconverted": Fraction(k - m * q1 - q2, m * p1 + p2) if m * p1 + p2 != 0 else None,
            "value-as-exponent": Fraction(c**k - m * q1 - n * q2, den) if k > 0 else None,
            "reciprocal": 1 / x if x != 0 else None,
        }
        values = [x] + [w for w in wrong.values() if w is not None]
        if abs(x.numerator) > 30 or abs(x.denominator) > 12 or len(values) < 4 or len(set(values)) != len(values):
            continue
        break
    else:
        check(False, "no usable product exponential equation")
    xs = Rational(x.numerator, x.denominator)
    lhs = Rational(c) ** (m * (p1 * xs + q1)) * Rational(c) ** (n * (p2 * xs + q2))
    check_equivalent(lhs, Rational(c) ** k, "substituting the solution does not satisfy the product equation")
    text = (
        f"Solve {_base_text(c, m)}^({_exp_text(p1, q1)}) × {_base_text(c, n)}^({_exp_text(p2, q2)}) = "
        f"{_value_text(c, k)} for x."
    )
    working = [
        f"Write every number as a power of {c} and add the exponents: "
        f"{join_terms([(m, f'({_exp_text(p1, q1)})'), (n, f'({_exp_text(p2, q2)})')])} = {k}.",
        f"Solve the linear equation: x = {rational_text(x)}.",
    ]
    return _pack(text, x, working, [w for w in wrong.values() if w is not None])


def _pack(text: str, x: Fraction, working: list[str], wrong_values: list[Fraction]) -> dict:
    text += " Give x as an integer or a fraction in lowest terms."
    answer_text = rational_text(x)
    check_pairwise_distinct(x, *wrong_values, msg="an exponential distractor collides with the key or another")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": x.numerator, "q": x.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [rational_text(w) for w in wrong_values]),
    )


@generator(
    "exd_x_f0_exponential_matching",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_f0_exponential_matching(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "this topic only builds the exam tier")
    return rng.choice([_two_sides, _product])(rng)
