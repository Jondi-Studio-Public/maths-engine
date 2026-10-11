"""Exam-tier source for solving quadratic equations by the formula when the equation is not yet in standard form.

The practice tiers hand over ``ax^2 + bx + c = 0``. The exam question disguises the quadratic, so the learner
must first decide how to turn it into standard form (expand, collect terms, clear a denominator), then apply
the formula and leave the larger root exactly. Three disguises:

- ``product``: ``(px + q)(rx + s) = tx + u``
- ``both_sides``: ``a1 x^2 + b1 x + c1 = a2 x^2 + b2 x + c2``
- ``reciprocal``: ``ax + c/x = d`` with ``x`` not zero (multiply through by ``x``)

Built backwards from the target ``Ax^2 + Bx + C = 0`` (non-square positive discriminant) by drawing the pieces
that will be rearranged. The independent second route is completing the square: the larger root is rebuilt from
the vertex and ``sqrt(-k/A)`` rather than from the formula.

Distractors are named slips: the sign of ``b`` read the wrong way in the numerator; ``2a`` replaced by ``a`` in
the denominator; only the ``-b`` term divided by ``2a``; and a rearrangement slip (a constant carried across the equals
sign, or through the multiplication by ``x``, with the wrong sign), used when the resulting equation still has real roots.
"""

from __future__ import annotations

import math
import random

from sympy import Rational, Symbol, latex, sqrt

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    pick_distractors,
    symbolically_equal,
)

_X = Symbol("x")


def _display(expr) -> str:
    return str(expr).replace("**", "^").replace("*", "")


def _poly(a: int, b: int, c: int) -> str:
    return join_terms([(a, "x^2"), (b, "x"), (c, "")])


def _larger_root(a: int, b: int, c: int):
    d = b * b - 4 * a * c
    r1 = (-b + sqrt(d)) / (2 * a)
    r2 = (-b - sqrt(d)) / (2 * a)
    return max((r1, r2), key=lambda r: float(r.evalf()))


def _is_square(n: int) -> bool:
    return n >= 0 and math.isqrt(n) ** 2 == n


def _nz(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([n for n in range(lo, hi + 1) if n != 0])


def _draw(rng: random.Random) -> dict:
    shape = rng.choice(["product", "both_sides", "reciprocal"])
    if shape == "product":
        p, r = _nz(rng, 1, 3), _nz(rng, 1, 3)
        q, s = _nz(rng, -5, 5), _nz(rng, -5, 5)
        t, u = _nz(rng, -6, 6), _nz(rng, -9, 9)
        A, B, C = p * r, p * s + q * r - t, q * s - u
        slip_c = q * s + u
        text = f"Solve ({_poly_lin(p, q)})({_poly_lin(r, s)}) = {_poly_lin(t, u)}."
    elif shape == "both_sides":
        a1, a2 = _nz(rng, 1, 5), _nz(rng, -3, 3)
        b1, b2 = _nz(rng, -7, 7), _nz(rng, -7, 7)
        c1, c2 = _nz(rng, -8, 8), _nz(rng, -8, 8)
        A, B, C = a1 - a2, b1 - b2, c1 - c2
        slip_c = c1 + c2
        text = f"Solve {_poly(a1, b1, c1)} = {_poly(a2, b2, c2)}."
    else:
        a = _nz(rng, 1, 4)
        c = _nz(rng, -9, 9)
        d = _nz(rng, -9, 9)
        # a x + c/x = d  ->  a x^2 - d x + c = 0
        A, B, C = a, -d, c
        slip_c = -c
        text = f"Solve {join_terms([(a, 'x')])} {'+' if c > 0 else '-'} {abs(c)}/x = {d}, where x is not 0."
    return dict(shape=shape, A=A, B=B, C=C, slip_c=slip_c, text=text)


def _poly_lin(a: int, b: int) -> str:
    return join_terms([(a, "x"), (b, "")])


def _exam(rng: random.Random) -> dict:
    while True:
        d = _draw(rng)
        A, B, C = d["A"], d["B"], d["C"]
        disc = B * B - 4 * A * C
        if A > 0 and disc > 0 and not _is_square(disc) and disc < 400:
            break
    root = _larger_root(A, B, C)

    # independent second route: complete the square
    vertex = Rational(-B, 2 * A)
    height = C - Rational(B * B, 4 * A)  # A(x - vertex)^2 + height = 0
    offset = sqrt(-height / A)
    by_squares = vertex + offset  # offset > 0, so this is the larger root whatever the sign of A
    check_equivalent(by_squares, root, "completing the square disagrees with the formula")
    check("sqrt" in str(root), "root should be a surd")
    # the stated equation must be equivalent to the standard form (not just the key to the key)
    check(A * _X**2 + B * _X + C != 0, "degenerate standard form")

    answer_text = f"x = {_display(root)}"
    sq = sqrt(disc)
    cands = [
        (B + sq) / (2 * A),  # sign of b the wrong way round
        (-B + sq) / A,  # 2a replaced by a
        -Rational(B, 2 * A) + sq,  # only -b divided by 2a
    ]
    slip_disc = B * B - 4 * A * d["slip_c"]
    if slip_disc > 0 and not _is_square(slip_disc):
        cands.insert(0, _larger_root(A, B, d["slip_c"]))  # constant moved across without changing sign
    # a slip that happens to give the key for these numbers (the sign of b when b = 0) is not a distractor
    cands = [c for c in cands if not symbolically_equal(root, c)]
    check(len(cands) >= 2, "too few distinct distractors")
    for c in cands:
        check_distinct(root, c, msg="distractor equals the key")
    return dict(
        text=d["text"] + " Give the larger root exactly.",
        latex=f"plain:{d['text']} Give the larger root exactly.",
        answer=root,
        answer_text=answer_text,
        answer_latex="x = " + latex(root),
        distractors=pick_distractors(answer_text, [f"x = {_display(c)}" for c in cands]),
    )


@generator("exd_x_a2_quadratic_formula_rearranged", level=12, presets={"exam": dict(tier="exam")})
def exd_x_a2_quadratic_formula_rearranged(rng: random.Random, tier: str = "exam") -> dict:
    """Rearrange a disguised quadratic to standard form, then solve by the formula for the larger root."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
