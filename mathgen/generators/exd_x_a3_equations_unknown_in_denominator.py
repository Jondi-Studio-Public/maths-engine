"""Exam source for `linear-equations-with-fractions`: the unknown sits in the denominator.

The practice tiers clear constant denominators. Here the unknown is inside the denominators, so the learner has to
choose a method (cross-multiply or multiply through by the common denominator), notice that the result is linear, and
keep the excluded value in mind. Three forms:

  * ``cross``  -- ``a/(x + b) = c/(x + d)``
  * ``ratio``  -- ``(x + a)/(x + b) = p/q``
  * ``shift``  -- ``a/(x + b) + c = d/(x + b)``  (one shared denominator and a loose constant)

Each is built BACKWARDS from an integer answer ``x0`` that is not an excluded value, with a non-degenerate
coefficient (so the cleared equation is genuinely linear and the answer unique), and re-solved with SymPy.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import gcd

from sympy import Eq, Rational, Symbol, denom, solve, together

from ..core import GRADER_TOLERANCE, check, generator, join_terms, phrase, pick_distractors, rational_text

_X = Symbol("x")


def _den(b: int) -> str:
    return f"(x + {b})" if b > 0 else f"(x - {-b})"


def _frac(num: int | str, b: int) -> str:
    return f"{num}/{_den(b)}"


def _txt(v: Fraction) -> str:
    return rational_text(Rational(v.numerator, v.denominator))


def _finish(text: str, lhs, rhs, x0: int, cands: list[Fraction], working: list[str]) -> dict | None:
    """Independent re-solve, then distractors from `cands`."""
    eq = together(lhs - rhs)
    sols = solve(Eq(eq.as_numer_denom()[0], 0), _X)
    sols = [s for s in sols if denom(together(lhs - rhs)).subs(_X, s) != 0]
    check(sols == [x0], f"expected unique solution {x0}, got {sols}")
    ans = Fraction(x0)
    texts = [_txt(c) for c in cands if c != ans]
    d = pick_distractors(_txt(ans), texts, 3, tolerance=GRADER_TOLERANCE)
    if len(d) < 3:
        return None
    return dict(
        text=text,
        latex="plain:" + text,
        answer=x0,
        answer_text=f"x = {x0}",
        distractors=[f"x = {v}" for v in d],
        working=working,
    )


def _cross(rng: random.Random) -> dict | None:
    a = rng.choice([2, 3, 4, 5, 6, 7])
    c = rng.choice([v for v in (2, 3, 4, 5, 6, 7, -2, -3, -4, -5) if v != a])
    x0 = rng.choice([v for v in range(-8, 9) if v != 0])
    t = rng.choice([v for v in (-3, -2, -1, 1, 2, 3) if v != 0])
    b = a * t - x0  # x0 + b = a*t
    d = c * t - x0  # a*(x0 + d) = c*(x0 + b)
    if x0 + b == 0 or x0 + d == 0 or b == d or b == 0 or d == 0 or abs(b) > 15 or abs(d) > 15:
        return None
    lhs = Rational(a) / (_X + b)
    rhs = Rational(c) / (_X + d)
    text = phrase(
        rng,
        ["Solve {l} = {r}", "Find x if {l} = {r}", "Solve for x: {l} = {r}"],
        l=_frac(a, b),
        r=_frac(c, d),
    )
    f = Fraction
    cands = [
        f(c * d - a * b, a - c),  # a(x + b) = c(x + d): each numerator over the wrong denominator
        f(c * b - a * d, a + c) if a + c else f(0),  # adds the x terms instead of subtracting
        f(c * b + a * d, a - c),  # sign slip on one constant
        f(-b),  # an excluded value (makes a denominator zero)
    ]
    return _finish(
        text,
        lhs,
        rhs,
        x0,
        cands,
        [f"Cross-multiply: {a}(x + {d}) = {c}(x + {b})", f"x = {x0}"],
    )


def _ratio(rng: random.Random) -> dict | None:
    p = rng.choice([2, 3, 4, 5])
    q = rng.choice([v for v in (3, 4, 5, 6, 7, 8) if v != p and gcd(p, v) == 1])
    x0 = rng.choice([v for v in range(-8, 9) if v != 0])
    t = rng.choice([1, 2, 3, -1, -2])
    a = p * t - x0  # x0 + a = p*t
    b = q * t - x0  # x0 + b = q*t, so (x0+a)/(x0+b) = p/q
    if x0 + b == 0 or a == b or a == 0 or b == 0 or abs(a) > 15 or abs(b) > 15:
        return None
    lhs = (_X + a) / (_X + b)
    rhs = Rational(p, q)
    text = phrase(
        rng,
        ["Solve ({a})/{b} = {p}/{q}", "Find x if ({a})/{b} = {p}/{q}"],
        a=_den(a)[1:-1],
        b=_den(b),
        p=p,
        q=q,
    )
    f = Fraction
    cands = [
        f(q * b - p * a, p - q),  # p(x + a) = q(x + b): cross-multiplied the wrong way round
        f(p * b - a, q - p),  # forgets to multiply the constant a by q
        f(p * b + q * a, q - p),  # sign slip on one constant
        f(-b),  # an excluded value (makes the denominator zero)
    ]
    return _finish(
        text,
        lhs,
        rhs,
        x0,
        cands,
        [f"Cross-multiply: {q}(x + {a}) = {p}(x + {b})", f"x = {x0}"],
    )


def _shift(rng: random.Random) -> dict | None:
    c = rng.choice([v for v in (-4, -3, -2, 1, 2, 3, 4)])
    x0 = rng.choice([v for v in range(-8, 9) if v != 0])
    u = x0 + rng.choice([v for v in range(-6, 7) if v != 0 and v != -x0])  # u = x0 + b, nonzero
    b = u - x0
    if u == 0 or c == 0 or b == 0:
        return None
    # a/(x+b) + c = d/(x+b)  ->  (d - a) = c(x + b) -> d - a = c*u
    a = rng.choice([v for v in range(1, 9)])
    d = a + c * u
    if d == 0:
        return None
    lhs = Rational(a) / (_X + b) + c
    rhs = Rational(d) / (_X + b)
    text = phrase(
        rng,
        ["Solve {l}", "Find x if {l}"],
        l=f"{_frac(a, b)} {'+' if c > 0 else '-'} {abs(c)} = {_frac(d, b)}",
    )
    f = Fraction
    cands = [
        f(d - a, c),  # writes c in place of c(x + b): forgets to multiply the loose constant by the denominator
        f(d + a, c) - b,  # adds the numerators instead of subtracting
        f(d - a, c) + b,  # sign slip on b
        f(a - d, c) - b,  # subtracts in the wrong order
    ]
    return _finish(
        text,
        lhs,
        rhs,
        x0,
        cands,
        [f"Multiply by {_den(b)}: {a} + {c}{_den(b)} = {d}", f"x = {x0}"],
    )


_BUILDERS = {"cross": _cross, "ratio": _ratio, "shift": _shift}


@generator(
    "exd_x_a3_equations_unknown_in_denominator",
    level=7,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_a3_equations_unknown_in_denominator(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    while True:
        out = _BUILDERS[rng.choice(sorted(_BUILDERS))](rng)
        if out is not None:
            return out
