"""Exam-tier factorial-notation questions: combine or solve with factorials of a variable.

The practice tiers evaluate two small factorials, telescope ``n!/m!`` for numbers, and simplify one ratio
``(n + a)!/(n + b)!`` to a product. The exam tier needs more than telescoping one ratio:

* ``factor_out``: a sum or difference of two factorials over a third, ``((n + a)! +/- (n + b)!)/(n + c)!``.
  The learner must see that the numerator has a common factorial to factor out before cancelling.
* ``two_ratios``: two telescoping ratios added or subtracted, which then have to be expanded and collected
  into one polynomial in ``n``.
* ``solve``: an equation ``(n + a)!/(n + b)! = N`` to solve for the integer ``n``, where the ratio telescopes
  to a product of two or three consecutive terms.

The two simplify kinds ask for the value of the simplified expression at a stated n, never for "the simplified
expression": the grader compares by value, so a restated prompt would otherwise be marked correct.

Answers are built from the polynomial forms, then re-evaluated from the definition of the factorial at several
integer values of ``n`` (and, for the equation, by searching the integers) before the question is returned.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction
from math import factorial

import sympy

from ..core import check, generator, pick_distractors

_n = sympy.symbols("n")
_PROBE_VALUES = (3, 4, 5, 7, 9)  # all >= 2, so every factorial of n - 2 is defined


def _arg(shift: int) -> str:
    if shift == 0:
        return "n"
    return f"n + {shift}" if shift > 0 else f"n - {-shift}"


def _fact(shift: int) -> str:
    return "n!" if shift == 0 else f"({_arg(shift)})!"


def _run(first: int, last: int):
    """The product (n + first)(n + first + 1)...(n + last); the empty product is 1."""
    out = sympy.Integer(1)
    for t in range(first, last + 1):
        out *= _n + t
    return out


def _fact_at(shift: int, n_value: int) -> int:
    return factorial(n_value + shift)


def _verify_by_definition(expr, fn) -> None:
    """The expression must equal the factorial form at every probe value of n."""
    for value in _PROBE_VALUES:
        expected = fn(value)
        check(
            Fraction(int(sympy.numer(expr.subs(_n, value))), int(sympy.denom(expr.subs(_n, value)))) == expected,
            "expression disagrees with the factorial form",
        )


def _pretty(expr) -> str:
    """A learner-style spelling of a sympy expression: ``2n*(n + 1)`` and ``n^2``, never ``2*n**2``."""
    # A multiplication sign stays only before a bracket, where "n(n + 2)" would read as a function call.
    return re.sub(r"\*(?!\()", "", str(expr).replace("**", "^"))


def _w(expr) -> str:
    """The same spelling for use inside a worked-solution sentence, with a real multiplication sign."""
    return _pretty(expr).replace("*", " × ")


def _finish_value(rng: random.Random, expression_text: str, expr, candidates: list, working: list[str]) -> dict:
    """Ask for the simplified expression's value at a stated n, so that restating the prompt cannot be the answer.

    The grader compares by value, so asking for "the simplified expression" would accept the printed expression
    typed back. A number at n = 4..16 only comes out of the simplification (or a long evaluation), and the answer
    space is wide.
    """
    n_value = rng.randint(4, 16)
    key = int(sympy.expand(expr).subs(_n, n_value))
    wrong: list[int] = []
    for cand in candidates:
        value = sympy.cancel(cand).subs(_n, n_value)
        if value == int(value) and int(value) != key and int(value) not in wrong:
            wrong.append(int(value))
    check(len(wrong) >= 1, "no usable distractor")
    distractors = pick_distractors(str(key), [str(w) for w in wrong])
    check(len(distractors) >= 1, "no usable distractor")
    text = (
        f"Simplify {expression_text} for integers n >= 2, then find the value of the simplified expression "
        f"when n = {n_value}."
    )
    working = [*working, f"At n = {n_value}, {_w(sympy.factor(expr))} = {key}."]
    return dict(
        text=text,
        latex=text,
        answer=str(key),
        answer_text=str(key),
        answer_latex=str(key),
        working=working,
        distractors=distractors,
    )


_FACTOR_OUT_SHAPES = (
    # (a, b, c): ((n + a)! +/- (n + b)!) / (n + c)!, with a > b >= c
    (2, 1, 0),
    (2, 1, 1),
    (3, 2, 1),
    (3, 2, 2),
    (1, 0, -1),
    (1, 0, 0),
    (2, 0, 0),
    (3, 1, 0),
    (2, 1, -1),
)


def _factor_out(rng: random.Random) -> dict:
    while True:
        a, b, c = rng.choice(_FACTOR_OUT_SHAPES)
        sign = rng.choice((1, -1))
        first, second = _run(c + 1, a), _run(c + 1, b)
        expr = sympy.expand(first + sign * second)
        if expr.has(_n):
            break
    op = "+" if sign == 1 else "-"
    expression_text = f"({_fact(a)} {op} {_fact(b)})/{_fact(c)}"
    _verify_by_definition(expr, lambda v: Fraction(_fact_at(a, v) + sign * _fact_at(b, v), _fact_at(c, v)))
    candidates = [
        first,  # the second factorial dropped
        first - sign * second,  # the sign of the second term reversed
        first + sign,  # the second ratio replaced by 1
    ]
    working = [
        f"Factor {_fact(b)} out of the numerator: {_fact(a)} = {_fact(b)} × {_w(sympy.factor(_run(b + 1, a)))}. "
        f"Then cancel with the denominator to get {_w(sympy.factor(expr))}."
    ]
    return _finish_value(rng, expression_text, expr, candidates, working)


def _two_ratios(rng: random.Random) -> dict:
    while True:
        s1, s2 = rng.choice((0, 1, 2)), rng.choice((0, 1, 2))
        d1, d2 = rng.choice((1, 2)), rng.choice((1, 2))
        sign = rng.choice((1, -1))
        p1, p2 = _run(s1 - d1 + 1, s1), _run(s2 - d2 + 1, s2)
        expr = sympy.expand(p1 + sign * p2)
        if (s1, d1) != (s2, d2) and expr.has(_n):
            break
    op = "+" if sign == 1 else "-"
    expression_text = f"{_fact(s1)}/{_fact(s1 - d1)} {op} {_fact(s2)}/{_fact(s2 - d2)}"
    _verify_by_definition(
        expr,
        lambda v: (
            Fraction(_fact_at(s1, v), _fact_at(s1 - d1, v)) + sign * Fraction(_fact_at(s2, v), _fact_at(s2 - d2, v))
        ),
    )
    candidates = [
        p1 - sign * p2,  # sign of the second ratio reversed
        p1,  # second ratio dropped
        p1 / (_n + s1) + sign * p2,  # one factor lost from the first ratio
        p1 + sign * p2 / (_n + s2),  # one factor lost from the second ratio
    ]
    working = [
        f"{_fact(s1)}/{_fact(s1 - d1)} = {_w(sympy.factor(p1))} and {_fact(s2)}/{_fact(s2 - d2)} = {_w(sympy.factor(p2))}; "
        f"combining gives {_w(sympy.factor(expr))}."
    ]
    return _finish_value(rng, expression_text, expr, candidates, working)


def _solve_for_n(rng: random.Random) -> dict:
    gap = rng.choice((2, 2, 3))
    b = rng.choice((0, 1, 2))
    a = b + gap
    n0 = rng.randint(1, 8)
    value = 1
    for t in range(b + 1, a + 1):
        value *= n0 + t
    # Independent route: search the non-negative integers with the factorials evaluated directly.
    solutions = [m for m in range(0, 60) if factorial(m + a) == value * factorial(m + b)]
    check(solutions == [n0], "the equation must have exactly one non-negative integer solution")
    text = f"Given that {_fact(a)}/{_fact(b)} = {value} and that n is a non-negative integer, find n."
    candidates = [n0 + b + 1, n0 + a, n0 + 1]
    if gap == 2:
        candidates.append(-(n0 + 2 * b + 3))  # the negative root of the quadratic, which n >= 0 excludes
    working = [
        f"{_fact(a)}/{_fact(b)} = {_w(sympy.factor(_run(b + 1, a)))} = {value}; the left side grows with n, "
        f"and n = {n0} gives {value}."
    ]
    # Not routed through _finish: the answer is a number and so are the wrong values.
    wrong = [str(c) for c in candidates if c != n0]
    distractors = pick_distractors(str(n0), wrong)
    check(len(distractors) >= 1, "no usable distractor")
    return dict(
        text=text,
        latex=text,
        answer=str(n0),
        answer_text=str(n0),
        answer_latex=str(n0),
        working=working,
        distractors=distractors,
    )


@generator(
    "exd_x_h2_factorial_notation",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_h2_factorial_notation(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier factorial notation: factor out, combine two ratios, or solve a factorial equation."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_factor_out, _two_ratios, _solve_for_n))(rng)
