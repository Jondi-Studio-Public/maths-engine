"""Exam tier for `constants-coefficients-terms`: coefficients that depend on an unknown constant.

The hard tier asks for the coefficient or constant of a plain polynomial. Here each coefficient and the constant term
of a quadratic expression is itself written in terms of an unknown constant k, such as 3kx² + (k + 4)x + (9 - 2k).
One of the three is given as a number. Part (a) identifies that coefficient (or the constant term) and solves for k,
and parts (b) and (c) identify and evaluate the other two. Parts (b) and (c) carry `follow_through` from (a).

Built backwards from k, so every answer is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator
from .exd_x_sc3_support import number_part, redraw, tagged

F = Fraction
_NAMES = ["the coefficient of x²", "the coefficient of x", "the constant term"]
_ASKS = ["Find the coefficient of x².", "Find the coefficient of x.", "Find the constant term."]


def _times(e: int, var: str = "k") -> str:
    return var if e == 1 else f"{e}{var}"


@generator("exd_x_sc3_coefficients_chain", level=8, presets={"exam": dict(tier="exam")})
def exd_x_sc3_coefficients_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Solve for k from one coefficient of 3kx² + (k + 4)x + (9 - 2k), then give the other two."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return redraw(_build, rng, "coefficients that are all non-zero with usable wrong answers")


def _build(rng: random.Random) -> dict:
    k = rng.choice([v for v in range(-8, 9) if v != 0])
    m = rng.randint(2, 5)
    lin_plus = rng.random() < 0.5  # (k + c) or (c - k)
    c = rng.randint(1, 9)
    d = rng.randint(2, 12)
    e = rng.randint(1, 3)
    sg = rng.choice([-1, 1])  # constant term d + sg * e * k
    x2, x1, x0 = m * k, (k + c if lin_plus else c - k), d + sg * e * k
    check(x1 != 0 and x0 != 0, "a coefficient is zero")
    values = [x2, x1, x0]
    lin_text = f"k + {c}" if lin_plus else f"{c} - k"
    const_text = f"{d} {'+' if sg > 0 else '-'} {_times(e)}"
    expr = f"{_times(m)}x² + ({lin_text})x + ({const_text})"
    given = rng.randrange(3)
    stem = f"In the expression {expr}, k is a constant. {_NAMES[given].capitalize()} is {values[given]}."

    # Part (a): solve for k from the given coefficient, three ways to go wrong.
    v = values[given]
    if given == 0:
        a_cands = [
            (F(v * m), "multiplies-instead-of-dividing"),
            (F(v - m), "subtracts-instead-of-dividing"),
            (F(v), "reports-the-given-value-as-the-unknown"),
        ]
        a_work = [f"{m}k = {v}, so k = {v} ÷ {m} = {k}."]
    elif given == 1:
        if lin_plus:
            a_cands = [
                (F(v + c), "sign-error-moving-term"),
                (F(c - v), "sign-error-moving-term"),
                (F(v), "reports-the-given-value-as-the-unknown"),
            ]
            a_work = [f"k + {c} = {v}, so k = {v} - {c} = {k}."]
        else:
            a_cands = [
                (F(v - c), "forgets-to-negate"),
                (F(v + c), "sign-error-moving-term"),
                (F(v), "reports-the-given-value-as-the-unknown"),
            ]
            a_work = [f"{c} - k = {v}, so k = {c} - {v} = {k}."]
    else:
        a_cands = [
            (F(v + d, sg * e), "sign-error-moving-term"),
            (F(v - d), "forgot-to-divide-by-coefficient"),
            (F(v, e), "ignored-constant-term"),
        ]
        if sg < 0:
            a_cands.insert(2, (F(v - d, e), "forgets-to-negate"))
        a_work = [f"{d} {'+' if sg > 0 else '-'} {_times(e)} = {v}, so {_times(e)} = {sg * (v - d)} and k = {k}."]

    # Parts (b) and (c): the other two, each with its own typical slip.
    slips = [
        [
            (F(m + k), "adds-instead-of-multiplying"),
            (F(k), "reports-the-unknown-not-the-answer"),
            (F(m), "ignores-the-constant-in-the-term"),
        ],
        [
            (F(c - k if lin_plus else c + k), "sign-error-in-a-bracket"),
            (F(k), "reports-the-unknown-not-the-answer"),
            (F(c), "ignores-the-constant-in-the-term"),
        ],
        [
            (F(d - sg * e * k), "sign-error-in-a-bracket"),
            (F(d), "ignores-the-constant-in-the-term"),
            (F(sg * e * k), "drops-the-number-from-the-constant-term"),
        ],
    ]
    ft = [f"{m}*part_a", f"part_a+{c}" if lin_plus else f"{c}-part_a", f"{d}{sg * e:+d}*part_a"]
    working = [
        [f"Coefficient of x² = {m}k = {m} × {k} = {x2}."],
        [f"Coefficient of x = {lin_text} = {x1} (with k = {k})."],
        [f"Constant term = {const_text} = {x0} (with k = {k})."],
    ]
    others = [i for i in range(3) if i != given]
    parts = [
        number_part(
            "Find the value of k.",
            F(k),
            0,
            2,
            a_work,
            tagged(F(k), a_cands, 0),
        )
    ]
    for i in others:
        parts.append(
            number_part(
                _ASKS[i],
                F(values[i]),
                0,
                1,
                working[i],
                tagged(F(values[i]), slips[i], 0),
                follow_through=ft[i],
            )
        )
    return dict(text=stem, parts=parts)
