"""Exam-tier source for finding the equation of a quadratic from conditions that are not points or roots.

The practice tiers give a vertex and a point, two roots and a point, or three points. The exam question states
the conditions as FEATURES of the graph, so the learner must choose a form (turning-point form, factored
form) and turn each feature into an equation. Three shapes:

- ``tangent``: the graph touches the x-axis at ``x = h`` and passes through ``(m, n)``, so
  ``y = a(x - h)^2`` and ``a = n / (m - h)^2``.
- ``axis_two_points``: the axis of symmetry is ``x = h`` and the graph passes through two points, so
  ``y = a(x - h)^2 + k`` with ``a`` and ``k`` found from a pair of equations.
- ``intercepts_extremum``: the x-intercepts are ``x = r1`` and ``x = r2`` and the graph has a stated maximum
  or minimum value ``M``. The turning point is at the midpoint, ``a(r1 - r2)^2 / 4 = -M``.

Built backwards from integer ``a``, ``h``, ``k`` (or ``r1``, ``r2``, ``a``) so every stated number is a whole
number and the key is exact. The independent second route solves the stated conditions with ``sympy.solve`` as
a linear system for ``(a, b, c)`` and compares with the key built from the construction.

Distractors are named slips: the sign of ``h`` read the wrong way in turning-point form, the leading
coefficient's sign flipped, ``a`` computed without squaring ``(m - h)``, the turning-point height left out, the
extreme value used as the leading coefficient, and factored form written with the roots' signs reversed or with leading coefficient 1.
"""

from __future__ import annotations

import random

from sympy import Symbol, expand, linsolve, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
)

_X = Symbol("x")


def _coefs_text(a: int, b: int, c: int) -> str:
    return "y = " + join_terms([(a, "x^2"), (b, "x"), (c, "")])


def _expr(a, b, c):
    return a * _X**2 + b * _X + c


def _vertex_coefs(a: int, h: int, k: int) -> tuple[int, int, int]:
    return a, -2 * a * h, a * h * h + k


def _nz(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([n for n in range(lo, hi + 1) if n != 0])


def _exam(rng: random.Random) -> dict:
    shape = rng.choice(["tangent", "axis_two_points", "intercepts_extremum"])
    a_s, b_s, c_s = symbols("a b c")
    if shape == "tangent":
        while True:
            a, h = _nz(rng, -4, 4), _nz(rng, -5, 5)
            m = rng.randint(-6, 6)
            if m != h and m != 0:
                break
        n = a * (m - h) ** 2
        key = _vertex_coefs(a, h, 0)
        text = (
            f"A quadratic graph touches the x-axis at x = {h} and passes through the point ({m}, {n}). "
            f"Find its equation in the form y = ax^2 + bx + c."
        )
        wrong = [
            _vertex_coefs(a, -h, 0),  # sign of h read the wrong way
            _vertex_coefs(-a, h, 0),  # sign of a flipped
            _vertex_coefs(a * (m - h), h, 0),  # (m - h) not squared when finding a
        ]
        eqs = [
            (a_s * h * h + b_s * h + c_s, 0),  # passes through (h, 0)
            (2 * a_s * h + b_s, 0),  # turning point at x = h
            (a_s * m * m + b_s * m + c_s, n),
        ]
    elif shape == "axis_two_points":
        while True:
            a, h, k = _nz(rng, -3, 3), rng.randint(-4, 4), rng.randint(-6, 6)
            x1, x2 = rng.sample(range(-5, 6), 2)
            if abs(x1 - h) != abs(x2 - h) and h not in (x1, x2):
                break
        y1 = a * (x1 - h) ** 2 + k
        y2 = a * (x2 - h) ** 2 + k
        key = _vertex_coefs(a, h, k)
        text = (
            f"A quadratic graph has axis of symmetry x = {h} and passes through the points ({x1}, {y1}) "
            f"and ({x2}, {y2}). Find its equation in the form y = ax^2 + bx + c."
        )
        wrong = [
            _vertex_coefs(a, h, 0),  # turning-point height left out
            _vertex_coefs(a, -h, k),  # sign of h read the wrong way
            _vertex_coefs(-a, h, k),  # sign of a flipped
        ]
        eqs = [
            (2 * a_s * h + b_s, 0),
            (a_s * x1 * x1 + b_s * x1 + c_s, y1),
            (a_s * x2 * x2 + b_s * x2 + c_s, y2),
        ]
    else:
        a = _nz(rng, -3, 3)
        half = rng.randint(1, 4)  # half the distance between the intercepts
        mid = rng.randint(-4, 4)
        r1, r2 = mid - half, mid + half
        extreme = -a * half * half
        key = (a, -a * (r1 + r2), a * r1 * r2)
        kind = "minimum" if a > 0 else "maximum"
        text = (
            f"A quadratic graph has x-intercepts at x = {r1} and x = {r2} and a {kind} value of {extreme}. "
            f"Find its equation in the form y = ax^2 + bx + c."
        )
        wrong = [
            (extreme, -extreme * (r1 + r2), extreme * r1 * r2),  # extreme value used as the leading coefficient
            (-a, a * (r1 + r2), -a * r1 * r2),  # sign of a flipped
            (a, a * (r1 + r2), a * r1 * r2),  # roots' signs reversed in the factored form
            (1, -(r1 + r2), r1 * r2),  # factored form written with leading coefficient 1
        ]
        eqs = [
            (a_s * r1 * r1 + b_s * r1 + c_s, 0),
            (a_s * r2 * r2 + b_s * r2 + c_s, 0),
            (a_s * mid * mid + b_s * mid + c_s, extreme),
        ]

    # independent second route: solve the three stated conditions as a linear system
    sol = list(linsolve([lhs - rhs for lhs, rhs in eqs], a_s, b_s, c_s))
    check(len(sol) == 1, "conditions do not determine a unique quadratic")
    check_equivalent(_expr(*sol[0]), _expr(*key), "linear-system solution disagrees with the constructed key")
    check(key[0] != 0, "leading coefficient is zero")

    answer_expr = expand(_expr(*key))
    answer_text = _coefs_text(*key)
    texts = []
    for w in wrong:
        if w == key:
            continue
        check_distinct(answer_expr, _expr(*w), msg="distractor equals the key")
        texts.append(_coefs_text(*w))
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex="y = " + latex_answer(answer_expr),
        distractors=pick_distractors(answer_text, texts),
    )


@generator("exd_x_a2_quadratic_from_conditions", level=11, presets={"exam": dict(tier="exam")})
def exd_x_a2_quadratic_from_conditions(rng: random.Random, tier: str = "exam") -> dict:
    """Find a quadratic from graph features: tangency, axis of symmetry, or intercepts with an extreme value."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
