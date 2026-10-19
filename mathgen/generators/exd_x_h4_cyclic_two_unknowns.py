"""Exam-tier source for "the opposite angles of a cyclic quadrilateral are supplementary".

The practice tiers use one unknown. An exam question gives all four angles of a
cyclic quadrilateral ABCD as expressions in TWO unknowns, x and y. Each pair of
opposite angles adds to 180 degrees, which gives one linear equation per pair,
so the learner solves a 2 x 2 system and then evaluates a named angle.

The solution (x, y) and the four angle sizes are drawn first; the constant term
of each expression is then set so that the opposite pairs sum to 180 exactly.
The system's determinant is checked to be non-zero, so (x, y) is the only
solution, and the key is re-derived by solving the system with sympy.
"""

from __future__ import annotations

import random

from sympy import Eq, Integer, Symbol, solve

from ..core import check, generator, join_terms, phrase, pick_distractors

_X, _Y = Symbol("x"), Symbol("y")


def _expr(cx: int, cy: int, k: int) -> str:
    return join_terms([(cx, "x"), (cy, "y"), (k, "")])


def _bracketed(expr: str) -> str:
    """``expr`` in brackets, except a single bare term (``3x``), which needs none."""
    return f"({expr})" if " " in expr or expr.startswith("-") else expr


def _draw(rng):
    for _ in range(2000):
        x, y = rng.randint(5, 30), rng.randint(5, 30)
        a_val, b_val = rng.randint(45, 135), rng.randint(45, 135)
        vals = {"A": a_val, "C": 180 - a_val, "B": b_val, "D": 180 - b_val}
        coef = {n: (rng.randint(0, 4), rng.randint(0, 4)) for n in "ABCD"}
        if any(coef[n] == (0, 0) for n in "ABCD"):
            continue
        const = {n: vals[n] - coef[n][0] * x - coef[n][1] * y for n in "ABCD"}
        if any(abs(const[n]) > 70 for n in "ABCD"):
            continue
        m = [
            [coef["A"][0] + coef["C"][0], coef["A"][1] + coef["C"][1]],
            [coef["B"][0] + coef["D"][0], coef["B"][1] + coef["D"][1]],
        ]
        if m[0][0] * m[1][1] - m[0][1] * m[1][0] == 0:
            continue
        if len({coef[n] for n in "ABCD"}) < 4:
            continue
        return x, y, vals, coef, const
    raise AssertionError("no draw")  # pragma: no cover


@generator(
    "exd_x_h4_cyclic_two_unknowns",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_h4_cyclic_two_unknowns(rng: random.Random, tier: str = "exam") -> dict:
    """Cyclic quadrilateral with all four angles in two unknowns."""
    check(tier == "exam", f"unknown tier {tier!r}")
    x, y, vals, coef, const = _draw(rng)
    sums = [Eq(sum(coef[n][0] * _X + coef[n][1] * _Y + const[n] for n in pair), 180) for pair in ("AC", "BD")]
    sol = solve(sums, [_X, _Y], dict=True)
    check(sol == [{_X: Integer(x), _Y: Integer(y)}], "cyclic_two_unknowns: system solution disagrees")

    asked = rng.choice("ABCD")
    key = vals[asked]
    text = phrase(
        rng,
        (
            "ABCD is a cyclic quadrilateral with angle A = {a} degrees, angle B = {b} degrees, "
            "angle C = {c} degrees and angle D = {d} degrees. Find angle {asked}, in degrees.",
            "The angles of cyclic quadrilateral ABCD are A = {a} degrees, B = {b} degrees, C = {c} degrees "
            "and D = {d} degrees. Calculate the size of angle {asked}, in degrees.",
        ),
        a=_bracketed(_expr(*coef["A"], const["A"])),
        b=_bracketed(_expr(*coef["B"], const["B"])),
        c=_bracketed(_expr(*coef["C"], const["C"])),
        d=_bracketed(_expr(*coef["D"], const["D"])),
        asked=asked,
    )
    other = "BD" if asked in "AC" else "AC"
    wrong = [180 - key, vals[other[0]], vals[other[1]], x, y, 360 - key]
    answer_text = str(key)
    distractors = pick_distractors(answer_text, [str(w) for w in wrong if 0 < w < 360])
    check(len(distractors) >= 2, "too few distractors")
    working = [
        "Opposite angles of a cyclic quadrilateral are supplementary: A + C = 180 and B + D = 180.",
        "These give two linear equations in x and y.",
        f"Solving gives x = {x} and y = {y}.",
        f"Angle {asked} = {key} degrees.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
