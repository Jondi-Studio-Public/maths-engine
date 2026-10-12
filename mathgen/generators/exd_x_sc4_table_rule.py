"""Exam tier for "generate and interpret input-output tables": infer a linear rule from a gapped table, then use it.

The practice tiers apply a stated rule to an input. Here the rule is NOT stated. The table gives four input and
output pairs whose inputs step by 2, 3 or 4, so the learner must divide the change in output by the step to find
the rate, then find the output at input 0, then use both to predict an output far outside the table and to run
the rule backwards from an output. Parts (b), (c) and (d) carry ``follow_through`` from the earlier parts.

Built backwards from the rate, the starting output and the inputs, so every key is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, draw_until, generator, paren_negative, phrase
from ._exd_x_sc4_kit import num_part

_LETTERS = (("x", "y"), ("n", "T"), ("a", "b"))


@generator("exd_x_sc4_table_rule", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc4_table_rule(rng: random.Random, tier: str = "exam") -> dict:
    """Infer a linear rule from a table with gaps in the inputs, then use it. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    inp, out = rng.choice(_LETTERS)
    m = draw_until(rng, lambda r: r.randint(-15, 18), lambda v: abs(v) >= 2, what="a rate")
    c = draw_until(rng, lambda r: r.randint(-20, 30), lambda v: v != 0, what="a starting output")
    x1 = rng.randint(1, 4)
    step = rng.randint(2, 4)
    xs = [x1 + step * i for i in range(4)]
    ys = [m * x + c for x in xs]
    x0 = rng.randint(11, 25)
    x_target = rng.randint(6, 30)
    y_target = m * x_target + c
    check(y_target != 0 and m * x0 + c != 0, "keep the asked values away from zero")
    y0 = m * x0 + c

    pairs = "; ".join(f"{inp} = {x} gives {out} = {y}" for x, y in zip(xs, ys))
    stem = phrase(
        rng,
        (
            "A linear rule changes an input {i} into an output {o}. These pairs are known: {pairs}.",
            "The output {o} is a linear function of the input {i}. Some of its values are known: {pairs}.",
            "A table for a linear rule lists these input and output pairs: {pairs}.",
        ),
        i=inp,
        o=out,
        pairs=pairs,
    )

    xl, yl = xs[-1], ys[-1]
    parts = [
        num_part(
            f"Find the change in {out} for each increase of 1 in {inp}.",
            Fraction(m),
            0,
            2,
            [
                f"Change in {out} over the first two entries = {ys[1]} - {paren_negative(ys[0])} = {ys[1] - ys[0]}.",
                f"Change in {inp} = {xs[1]} - {xs[0]} = {step}, so the change per unit is {ys[1] - ys[0]} ÷ {step} = {m}.",
            ],
            [
                (ys[1] - ys[0], "change-not-divided-by-step"),
                (Fraction(ys[0], xs[0]), "output-over-input-as-rate"),
                (Fraction(1, m), "inverted-ratio"),
            ],
            signed=True,
        ),
        num_part(
            f"Hence find the output when the input is 0.",
            Fraction(c),
            0,
            1,
            [
                f"{out} = {m}{inp} + c, and {inp} = {xs[0]} gives {ys[0]}, so c = {ys[0]} - {paren_negative(m)}({xs[0]}) = {c}."
            ],
            [
                (ys[0], "first-output-as-intercept"),
                (ys[0] + m * xs[0], "sign-slip-in-intercept"),
                (ys[0] - m, "assumed-first-input-is-one"),
            ],
            follow_through=f"{ys[0]} - part_a*{xs[0]}",
            signed=True,
        ),
        num_part(
            f"Use your rule to find the output when the input is {x0}.",
            Fraction(y0),
            0,
            2,
            [f"{out} = {m}({x0}) + {paren_negative(c)} = {y0}."],
            [
                (m * x0, "dropped-constant-term"),
                (yl + (x0 - xl), "continued-pattern-by-one"),
                (m * x0 - c, "sign-slip-in-intercept"),
            ],
            follow_through=f"part_a*{x0} + part_b",
            signed=True,
        ),
        num_part(
            f"Which input gives an output of {y_target}?",
            Fraction(x_target),
            0,
            2,
            [
                f"{y_target} = {m}{inp} + {paren_negative(c)}, so {inp} = ({y_target} - {paren_negative(c)}) ÷ {paren_negative(m)} = {x_target}."
            ],
            [
                (Fraction(y_target, m), "dropped-constant-term"),
                (Fraction(y_target + c, m), "sign-error-moving-term"),
                ((y_target - c) * m, "multiplied-instead-of-divided"),
            ],
            follow_through=f"({y_target} - part_b)/part_a",
            signed=True,
        ),
    ]
    marks = [2, 1, 2, 2]
    for part, mark in zip(parts, marks):
        part["marks"] = mark
    return dict(text=stem, parts=parts)
