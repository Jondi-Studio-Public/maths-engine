"""Exam source for `variables-on-both-sides`: equations whose unknown is a parameter, or that must be built first.

The practice tiers hand the learner an equation and ask for ``x``. This source changes the task:

  * ``given``    -- ``k(x + a) = n*x + c`` has a stated solution; the learner substitutes it and solves for the
                    parameter ``k`` (the unknown has moved from ``x`` to a coefficient).
  * ``same``     -- two equations share a solution; the learner solves the plain one first, then uses that value to
                    find the parameter in the other (two chained solves).
  * ``polygons`` -- two regular polygons with side lengths ``x + a`` and ``x + b`` have equal perimeters; the learner
                    builds ``m(x + a) = n(x + b)``, solves, then evaluates a perimeter (build, solve, interpret).

Everything is built BACKWARDS from the chosen integer answer, so the key is exact for every seed, and each
question is re-checked with SymPy before it is returned.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import gcd

from sympy import Eq, Rational, Symbol, solve

from ..core import (
    GRADER_TOLERANCE,
    check,
    check_equivalent,
    generator,
    join_terms,
    phrase,
    pick_distractors,
    rational_text,
)

_X = Symbol("x")
_LETTERS = ("k", "m", "p", "c")
_POLYGONS = {3: "equilateral triangle", 4: "square", 5: "regular pentagon", 6: "regular hexagon", 8: "regular octagon"}


def _bracket(a: int) -> str:
    return f"(x + {a})" if a > 0 else f"(x - {-a})"


def _lin(coef: int, const: int) -> str:
    return join_terms([(coef, "x"), (const, "")])


def _safe_fraction(num: int, den: int = 1) -> Fraction | None:
    """``Fraction(num, den)``, or None when the slip being modelled would divide by zero."""
    return Fraction(num, den) if den else None


def _article(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def _frac_text(value: Fraction) -> str:
    return rational_text(Rational(value.numerator, value.denominator))


def _pick(candidates: list[Fraction], answer: Fraction, n: int = 3, *, nonneg: bool = False) -> list[str] | None:
    texts = [
        _frac_text(c) for c in candidates if c is not None and c != answer and (c >= 0 or answer < 0 or not nonneg)
    ]
    out = pick_distractors(_frac_text(answer), texts, n, tolerance=GRADER_TOLERANCE)
    return out if len(out) == n else None


def _given(rng: random.Random) -> dict | None:
    """``L(x + a) = n x + c`` with solution ``x0``; find ``L``."""
    letter = rng.choice(_LETTERS)
    L = rng.choice([v for v in range(-6, 8) if v not in (0, 1, -1)])
    n = rng.choice([v for v in range(-5, 8) if v not in (0, L)])
    x0 = rng.choice([v for v in range(-6, 7) if v != 0])
    a = rng.choice([v for v in range(-7, 8) if v != 0 and x0 + v != 0])
    c = L * (x0 + a) - n * x0
    if c == 0:
        return None
    text = phrase(
        rng,
        [
            "The equation {letter}{br} = {rhs} has the solution x = {x0}. Find the value of {letter}.",
            "x = {x0} is the solution of {letter}{br} = {rhs}. What is the value of {letter}?",
        ],
        letter=letter,
        br=_bracket(a),
        rhs=_lin(n, c),
        x0=x0,
    )
    # Independent check: substitute, then solve for the parameter.
    sym = Symbol(letter)
    solved = solve(Eq(sym * (x0 + a), n * x0 + c), sym)
    check(len(solved) == 1)
    check_equivalent(solved[0], L)
    x_sol = solve(Eq(L * (_X + a), n * _X + c), _X)
    check(x_sol == [x0])
    f = _safe_fraction
    cands = [
        f(n * x0 + c - a, x0),  # brackets only the x: k*x + a
        f(n * x0 - c, x0 + a),  # sign slip on the constant
        f(-n * x0 + c, a - x0),  # substitutes -x0
        f(n * x0 + c, x0) - a,  # divides by x0, then subtracts a
        f(x0),  # reports the given solution
    ]
    d = _pick(cands, f(L))
    if d is None:
        return None
    return dict(
        text=text,
        latex="plain:" + text,
        answer=L,
        answer_text=f"{letter} = {L}",
        distractors=[f"{letter} = {v}" for v in d],
        working=[
            f"Substitute x = {x0}: {letter}({x0} + {a}) = {n}({x0}) + {c}",
            f"{letter} = {L}",
        ],
    )


def _same(rng: random.Random) -> dict | None:
    """``L x + c1 = n x + c2`` shares its solution with ``p x - q = r``; find ``L``."""
    letter = rng.choice(_LETTERS)
    L = rng.choice([v for v in range(-6, 8) if v not in (0, 1, -1)])
    n = rng.choice([v for v in range(-5, 8) if v not in (0, L)])
    x0 = rng.choice([v for v in range(-7, 8) if v != 0])
    c1 = rng.choice([v for v in range(-9, 10) if v != 0])
    c2 = L * x0 + c1 - n * x0
    p = rng.choice([2, 3, 4, 5, 6])
    q = rng.choice([v for v in range(-9, 10) if v != 0])
    r = p * x0 + q
    if c2 == 0 or r == 0:
        return None
    first = f"{letter}x {'+' if c1 > 0 else '-'} {abs(c1)} = {_lin(n, c2)}"
    second = f"{p}x {'+' if q > 0 else '-'} {abs(q)} = {r}"
    text = phrase(
        rng,
        [
            "The equations {first} and {second} have the same solution. Find the value of {letter}.",
            "Both {second} and {first} are solved by the same value of x. Find the value of {letter}.",
        ],
        first=first,
        second=second,
        letter=letter,
    )
    check(solve(Eq(p * _X + q, r), _X) == [x0])
    sym = Symbol(letter)
    solved = solve(Eq(sym * x0 + c1, n * x0 + c2), sym)
    check(len(solved) == 1)
    check_equivalent(solved[0], L)
    f = _safe_fraction
    x_bad = f(r + q, p)  # sign slip while solving the second equation
    cands = [
        (n * x_bad + c2 - c1) / x_bad if x_bad != 0 else f(0),  # that wrong solution carried through
        f(n * x0 + c2 + c1, x0),  # moves c1 across without changing its sign
        f(n * x0 - c2 - c1, x0),
        f(x0),  # reports the shared solution instead of the parameter
        f(r, p) - q,
    ]
    d = _pick(cands, f(L))
    if d is None:
        return None
    return dict(
        text=text,
        latex="plain:" + text,
        answer=L,
        answer_text=f"{letter} = {L}",
        distractors=[f"{letter} = {v}" for v in d],
        working=[
            f"Solve {second}: x = {x0}",
            f"Substitute x = {x0} into the other equation and solve for {letter}: {letter} = {L}",
        ],
    )


def _polygons(rng: random.Random) -> dict | None:
    """Equal perimeters of two regular polygons with side lengths ``x + a`` and ``x + b``; give the perimeter."""
    m, n = rng.sample(sorted(_POLYGONS), 2)
    g = gcd(m, n)
    t = rng.choice([1, 2, 3])
    s1, s2 = (n // g) * t, (m // g) * t  # m * s1 == n * s2
    x0 = rng.choice([v for v in range(1, 9)])
    a, b = s1 - x0, s2 - x0
    if a == b or a == 0 or b == 0:
        return None
    text = phrase(
        rng,
        [
            "{A_cap} has side length {sa} cm and {B_art} has side length {sb} cm. Their perimeters are equal. "
            "Find the perimeter in cm.",
            "The sides of {A} and {B} measure {sa} cm and {sb} cm respectively, and the two shapes have the same "
            "perimeter. What is that perimeter, in cm?",
        ],
        A=_article(_POLYGONS[m]),
        B=_article(_POLYGONS[n]),
        A_cap=_article(_POLYGONS[m]).capitalize(),
        B_art=_article(_POLYGONS[n]),
        sa=_bracket(a),
        sb=_bracket(b),
    )
    sol = solve(Eq(m * (_X + a), n * (_X + b)), _X)
    check(sol == [x0])
    perim = m * s1
    check(perim == n * s2 == m * (x0 + a))
    f = _safe_fraction
    swapped = f(m * b - n * a, n - m)  # coefficients m and n swapped
    cands = [
        f(s1),  # one side, not the perimeter
        f(x0),  # the unknown, not the perimeter
        n * (swapped + a),  # swapped coefficients throughout
        f(m * x0 + a),  # perimeter: brackets only the x
        f(m * (x0 - a)),  # sign slip on the constant
    ]
    d = _pick(cands, f(perim), nonneg=True)
    if d is None:
        return None
    return dict(
        text=text,
        latex="plain:" + text,
        answer=perim,
        answer_text=str(perim),
        distractors=d,
        working=[
            f"{m}({_bracket(a)[1:-1]}) = {n}({_bracket(b)[1:-1]})",
            f"x = {x0}, so the perimeter is {m} x {s1} = {perim} cm",
        ],
    )


_BUILDERS = {"given": _given, "same": _same, "polygons": _polygons}


@generator(
    "exd_x_a3_parametric_both_sides",
    level=7,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_a3_parametric_both_sides(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    while True:
        kind = rng.choice(sorted(_BUILDERS))
        out = _BUILDERS[kind](rng)
        if out is not None:
            return out
