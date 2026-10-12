"""Exam source of `combine-like-terms`: collect the like terms in a perimeter, then use the result.

The practice tiers combine the coefficients of one variable. Here the sides of a triangle or quadrilateral are
expressions in two variables with a constant, so the learner has to sort three kinds of term across several
brackets and then use the simplified perimeter Ax + By + C:

    (a) the coefficient of x in the simplified perimeter                          (1 mark)
    (b) the coefficient of y                                                      (1 mark)
    (c) the constant term                                                         (1 mark)
    (d) the perimeter for stated values of x and y, using (a), (b) and (c)         (2 marks, follow-through)

Each coefficient is asked for as a number rather than asking for the whole expression, because the expression
grader accepts any equivalent form and the sum of the sides as printed is equivalent to the simplified perimeter.
"""

from __future__ import annotations

import sympy
from sympy import Symbol

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, join_terms, latex_answer, paren_negative, pick_distractors

x, y = Symbol("x"), Symbol("y")
_SHAPES = [
    (3, "A triangular paddock has side lengths"),
    (3, "A triangular shade sail has side lengths"),
    (4, "A quadrilateral courtyard has side lengths"),
    (4, "A quadrilateral vegetable garden has side lengths"),
]


def ignore(values):
    """The total a learner reaches by dropping every minus sign."""
    return sum(abs(v) for v in values)


def _dist(value, tag):
    return Distractor(str(value), tag, int(value))


def _part(prompt, key, marks, working, candidates, follow_through=None):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=key,
        answer_text=str(key),
        answer_latex=latex_answer(str(key)),
        marks=marks,
        working=working,
        distractors=pick_distractors(str(key), candidates, 3),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


@generator("exd_x_sc1_perimeter_collect", level=8, presets={"exam": dict(tier="exam")})
def exd_x_sc1_perimeter_collect(rng, tier="exam"):
    sides_n, opening = rng.choice(_SHAPES)
    for _ in attempts(1000, "sides that stay positive and collect to three nonzero terms"):
        sides = []
        for _s in range(sides_n):
            a = rng.randint(1, 8)
            b = rng.choice([0, 0, 1, 2, 3, 4, 5, 6, -1, -2, -3])
            c = rng.choice([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, -1, -2, -3, -4, -5])
            sides.append((a, b, c))
        cx, cy = rng.randint(2, 9), rng.randint(1, 7)
        A, B, C = (sum(s[i] for s in sides) for i in range(3))
        values = [a * cx + b * cy + c for a, b, c in sides]
        if min(values) < 2 or min(A, B) < 1 or C == 0:
            continue
        negatives = sum(1 for s in sides for t in s if t < 0)
        if negatives == 0 or sum(1 for s in sides if s[1] != 0) < 2:
            continue
        total = A * cx + B * cy + C
        if total > 400 or total in {A, B, C, cx, cy} or abs(C) > 40:
            continue
        break

    perimeter = sum(a * x + b * y + c for a, b, c in sides)
    check(sympy.expand(perimeter) == A * x + B * y + C, "collected perimeter disagrees with sympy")
    check(total == sum(values) == perimeter.subs({x: cx, y: cy}), "perimeter value disagrees")

    xs, ys, cs = [s[0] for s in sides], [s[1] for s in sides], [s[2] for s in sides]
    a_part = _part(
        "The perimeter simplifies to Ax + By + C. Find the value of A.",
        A,
        1,
        [f"Add the x terms: {' + '.join(paren_negative(v) for v in xs)} = {A}."],
        [
            _dist(ignore(xs), "ignored-sign"),
            _dist(A + B, "combined-unlike-terms"),
            _dist(A - xs[-1], "dropped-a-term"),
            _dist(A + C, "combined-unlike-terms"),
            _dist(A - xs[0], "dropped-a-term"),
        ],
    )
    b_part = _part(
        "Find the value of B.",
        B,
        1,
        [f"Add the y terms: {' + '.join(paren_negative(v) for v in ys)} = {B}."],
        [
            _dist(ignore(ys), "ignored-sign"),
            _dist(B + C, "combined-unlike-terms"),
            _dist(B - ys[-1], "dropped-a-term"),
            _dist(A + B, "combined-unlike-terms"),
            _dist(B - ys[0], "dropped-a-term"),
        ],
    )
    c_part = _part(
        "Find the value of C.",
        C,
        1,
        [f"Add the constants: {' + '.join(paren_negative(v) for v in cs)} = {C}."],
        [
            _dist(ignore(cs), "ignored-sign"),
            _dist(C + A, "combined-unlike-terms"),
            _dist(sum(v for v in cs if v > 0), "dropped-a-term"),
            _dist(C + B, "combined-unlike-terms"),
            _dist(C - cs[-1], "dropped-a-term"),
        ],
    )
    d_part = _part(
        f"Hence find the perimeter, in metres, when x = {cx} and y = {cy}.",
        total,
        2,
        [
            f"P = {A}x + {paren_negative(B)}y + {paren_negative(C)} = {A} × {cx} + {paren_negative(B)} × {cy} + {paren_negative(C)} = {total}."
        ],
        [
            _dist(A * cy + B * cx + C, "swapped-the-variables"),
            _dist(A * cx + B * cy, "dropped-constant-term"),
            _dist(A + cx + B + cy + C, "coefficient-added-not-multiplied"),
        ],
        follow_through=f"part_a*{cx} + part_b*{cy} + part_c",
    )
    shown = ", ".join(join_terms([(a, "x"), (b, "y"), (c, "")]) for a, b, c in sides[:-1])
    last = join_terms([(sides[-1][0], "x"), (sides[-1][1], "y"), (sides[-1][2], "")])
    stem = f"{opening} {shown} and {last}, all in metres."
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])
