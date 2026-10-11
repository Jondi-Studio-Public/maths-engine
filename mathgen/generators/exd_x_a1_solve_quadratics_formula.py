"""Exam-tier source for `solve-quadratics-formula`: equations that must be rearranged into standard form first.

Two shapes, chosen per seed:

* ``brackets`` -- ``(px + q)(rx + s) = tx + u``. The learner expands, collects everything on one side, then
  reads ``a``, ``b`` and ``c`` from the result (which are none of the numbers printed in the prompt).
* ``reciprocal`` -- ``px + k/x = m``. The unknown is in a denominator; multiplying through by ``x`` turns it
  into ``px^2 - mx + k = 0``.

The discriminant of the standard form is always positive and not a perfect square, so the roots are
irrational and are asked for to 2 decimal places. Roots are re-computed with ``sympy.solve`` on the
equation as printed (an independent route from the standard-form coefficients used to build it).
"""

from __future__ import annotations

import math
import random

import sympy
from sympy import Eq, symbols

from ..core import (
    GRADER_TOLERANCE,
    check,
    fmt_signed,
    generator,
    pick_distractors,
    rounds_unambiguously,
)

x = symbols("x")


def _bracket(a: int, b: int) -> str:
    return f"({fmt_signed(a, 'x')} {'+' if b >= 0 else '-'} {abs(b)})"


def _usable(a: int, b: int, c: int) -> bool:
    disc = b * b - 4 * a * c
    if disc <= 0 or math.isqrt(disc) ** 2 == disc or max(abs(a), abs(b), abs(c)) > 40:
        return False
    sq = math.sqrt(disc)
    roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
    return all(rounds_unambiguously(r, 2) for r in roots) and abs(roots[1] - roots[0]) > 0.2


def _fmt(values) -> str:
    return " or ".join(f"x = {v:.2f}" for v in sorted(values))


def _roots(a: int, b: int, c: int, disc_sign: int = -1):
    sq = math.sqrt(b * b + disc_sign * 4 * a * c)
    return [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]


@generator("exd_x_a1_solve_quadratics_formula", level=9, presets={"exam": dict(tier="exam")})
def exd_x_a1_solve_quadratics_formula(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: rearrange to standard form, then use the quadratic formula (2 d.p.)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    if rng.random() < 0.5:
        while True:
            p, r = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
            q, s = rng.randint(-6, 6), rng.randint(-6, 6)
            t, u = rng.randint(-6, 6), rng.randint(-9, 9)
            if 0 in (q, s, t, u):
                continue
            a, b, c = p * r, p * s + q * r - t, q * s - u
            if _usable(a, b, c):
                break
        equation = f"{_bracket(p, q)}{_bracket(r, s)} = {fmt_signed(t, 'x')} {'+' if u >= 0 else '-'} {abs(u)}"
        lhs = (p * x + q) * (r * x + s)
        rhs = t * x + u
    else:
        while True:
            p = rng.choice([1, 2, 3])
            k = rng.randint(1, 9) * rng.choice([1, -1])
            m = rng.randint(2, 12) * rng.choice([1, -1])
            a, b, c = p, -m, k
            if _usable(a, b, c):
                break
        lead = fmt_signed(p, "x")
        equation = f"{lead} {'+' if k > 0 else '-'} {abs(k)}/x = {m}"
        lhs = p * x + sympy.Integer(k) / x
        rhs = sympy.Integer(m)

    solved = sorted(float(v) for v in sympy.solve(Eq(lhs, rhs), x))
    key = sorted(round(v, 2) for v in _roots(a, b, c))
    check(
        len(solved) == 2 and all(abs(sv - kv) < 0.006 for sv, kv in zip(solved, key)), "roots disagree with sympy.solve"
    )
    answer_text = _fmt(key)

    cands = [
        _fmt(_roots_wrong_sign(a, b, c)),  # +b instead of -b
        _fmt([(-b - math.sqrt(b * b - 4 * a * c)) / a, (-b + math.sqrt(b * b - 4 * a * c)) / a]),  # forgot to double a
    ]
    if b * b + 4 * a * c >= 0:
        cands.append(_fmt(_roots(a, b, c, +1)))  # discriminant b^2 + 4ac
    text = f"Solve {equation} using the quadratic formula (2 d.p.)"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=answer_text,
        answer_latex=" \\text{ or } ".join(f"x = {v:.2f}" for v in key),
        working=[
            f"Standard form: {a}x^2 + ({b})x + ({c}) = 0.",
            f"x = (-({b}) +/- sqrt({b * b - 4 * a * c})) / (2 x {a})",
        ],
        distractors=pick_distractors(answer_text, cands, unordered=True, tolerance=GRADER_TOLERANCE),
    )


def _roots_wrong_sign(a: int, b: int, c: int):
    sq = math.sqrt(b * b - 4 * a * c)
    return [(b - sq) / (2 * a), (b + sq) / (2 * a)]
