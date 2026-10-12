"""Exam chain for vector equations of lines: where two lines meet, and how far along the first line that is.

Two lines in the plane, line m: ``r = A + s e`` and line n: ``r = C + t d``, meet at a point P. Line m is given either by two
points A and B (``e = B - A``) or by its vector equation. Set the way a senior exam sets it:

(a) the parameter s at P                                              number,   2 marks
(b) the parameter t at P, using s in one component equation           fraction, 1 mark,  follow-through (a)
(c) the x-coordinate (or y) of P                                      fraction, 1 mark,  follow-through (a)
(d) the distance AP, correct to 1 decimal place                       number,   2 marks, follow-through (a)

## Construction: choose P's parameters first

``s`` and ``t`` are non-zero whole numbers, the directions ``e`` and ``d`` have no zero component and are not
parallel, ``A`` is drawn, and ``C = A + s e - t d``. Then ``A + s e = C + t d`` is the unique common point. The
parameters are re-derived by Cramer's rule from the printed vectors, and P and the distance by substitution.

## Distractors

* (a): the two parameters swapped, the sign wrong, only one component equation solved with t forgotten.
* (b): s given for t, the sign wrong, the division by the wrong direction component.
* (c): the base point of line n given, the wrong parameter put into a line.
* (d): the length of the direction vector (s forgotten), the components added, the distance from C.
"""

from __future__ import annotations

from fractions import Fraction
from math import hypot

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    check,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_POINTS = (
    "Line m passes through the points A({ax}, {ay}) and B({bx}, {by}). It is written r = A + s(B - A). "
    "Line n has vector equation r = ({cx}, {cy}) + t({dx}, {dy}). The two lines meet at the point P.",
    "Line m is the line through A({ax}, {ay}) and B({bx}, {by}), with equation r = A + s(B - A), and line n is the line "
    "r = ({cx}, {cy}) + t({dx}, {dy}). The lines intersect at the point P.",
)
_VECTORS = (
    "Line m passes through the point A({ax}, {ay}) and has vector equation r = ({ax}, {ay}) + s({ex}, {ey}). "
    "Line n has vector equation r = ({cx}, {cy}) + t({dx}, {dy}). The two lines meet at the point P.",
    "The line m, r = ({ax}, {ay}) + s({ex}, {ey}), passes through A({ax}, {ay}). The line n is "
    "r = ({cx}, {cy}) + t({dx}, {dy}). The lines intersect at the point P.",
)


def _part(prompt, value, kind, marks, working, wrong, follow=None, places=None):
    value = Fraction(value)
    if places is None:
        text = str(value) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        cands = [
            Distractor(
                str(Fraction(v))
                if Fraction(v).denominator == 1
                else f"{Fraction(v).numerator}/{Fraction(v).denominator}",
                tag,
                exact_value(Fraction(v)),
            )
            for v, tag in wrong
        ]
        answer = exact_value(value) if kind == "fraction" else int(value)
        tolerance = 0
    else:
        shown = round_half_up(value, places)
        text = stated_precision_string(shown, places)
        cands = [
            Distractor(
                stated_precision_string(round_half_up(Fraction(v), places), places),
                tag,
                exact_value(round_half_up(Fraction(v), places)),
            )
            for v, tag in wrong
        ]
        answer = float(shown)
        tolerance = stated_precision_tolerance(places)
    part = dict(
        prompt=prompt,
        answer_kind=kind,
        answer=answer,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, cands, tolerance=tolerance),
    )
    if follow:
        part["follow_through"] = follow
    return part


def _cross(u, v) -> int:
    return u[0] * v[1] - u[1] * v[0]


@generator("exd_x_r2c_vector_line", level=12, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_vector_line(rng, tier: str = "exam"):
    """Two lines in the plane: the parameter at their intersection, the other parameter, a coordinate, a distance."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    nonzero = [v for v in range(-4, 5) if v != 0]
    for _ in range(1000):  # bounded rejection loop
        e = (rng.choice(nonzero), rng.choice(nonzero))
        d = (rng.choice(nonzero), rng.choice(nonzero))
        s, t = rng.choice([-4, -3, -2, 2, 3, 4]), rng.choice([-3, -2, -1, 1, 2, 3])
        a = (rng.randint(-8, 8), rng.randint(-8, 8))
        far = abs(s) * hypot(*e)
        if _cross(e, d) != 0 and abs(s) != abs(t) and abs((far * 10) % 1 - 0.5) > 1e-6:
            break
    else:  # pragma: no cover
        check(False, "no usable draw")
    c = (a[0] + s * e[0] - t * d[0], a[1] + s * e[1] - t * d[1])
    p = (a[0] + s * e[0], a[1] + s * e[1])

    # Independent check: Cramer's rule on A + s e = C + t d recovers s and t, and P is on both lines.
    diff = (c[0] - a[0], c[1] - a[1])
    s_cramer, t_cramer = Fraction(_cross(diff, d), _cross(e, d)), Fraction(_cross(diff, e), _cross(e, d))
    check(s_cramer == s and t_cramer == t, "Cramer's rule disagrees with the construction")
    check((c[0] + t * d[0], c[1] + t * d[1]) == p, "P is not on line n")
    length_e = hypot(*e)
    distance = Fraction(abs(s)) * Fraction(length_e).limit_denominator(10**9)

    if rng.random() < 0.5:
        b = (a[0] + e[0], a[1] + e[1])
        stem = phrase(rng, _POINTS, ax=a[0], ay=a[1], bx=b[0], by=b[1], cx=c[0], cy=c[1], dx=d[0], dy=d[1])
    else:
        stem = phrase(rng, _VECTORS, ax=a[0], ay=a[1], ex=e[0], ey=e[1], cx=c[0], cy=c[1], dx=d[0], dy=d[1])

    part_a = _part(
        "Find the value of the parameter s at the point P.",
        s,
        "number",
        2,
        [
            f"Equate the components: {join_terms([(a[0], ''), (e[0], 's')])} = {join_terms([(c[0], ''), (d[0], 't')])} "
            f"and {join_terms([(a[1], ''), (e[1], 's')])} = {join_terms([(c[1], ''), (d[1], 't')])}.",
            f"Eliminating t gives s = {s} (and t = {t}).",
        ],
        [
            (t, "gave-the-other-parameter"),
            (-s, "sign-error-moving-term"),
            (Fraction(c[0] - a[0], e[0]), "ignored-second-parameter"),
            (Fraction(c[1] - a[1], e[1]), "ignored-second-parameter"),
        ],
    )
    # t from the x-equation, with the learner's own s.
    part_b = _part(
        "Hence find the value of the parameter t at P.",
        t,
        "fraction",
        1,
        [f"{a[0]} + {paren_negative(e[0])}·{paren_negative(s)} = {join_terms([(c[0], ''), (d[0], 't')])}, so t = {t}."],
        [
            (s, "gave-the-other-parameter"),
            (-t, "sign-error-moving-term"),
            (Fraction(a[0] + e[0] * s - c[0], e[0]), "divided-by-wrong-coefficient"),
        ],
        follow=f"({a[0]} + {e[0]}*part_a - ({c[0]}))/({d[0]})",
    )
    part_c = _part(
        "Hence find the x-coordinate of P.",
        p[0],
        "fraction",
        1,
        [f"x = {a[0]} + {paren_negative(e[0])}·{paren_negative(s)} = {p[0]}."],
        [
            (c[0], "wrong-parameter-substituted"),
            (a[0] + e[0] * t, "wrong-parameter-substituted"),
            (c[0] + d[0] * s, "wrong-parameter-substituted"),
            (p[1], "swaps-x-and-y"),
        ],
        follow=f"{a[0]} + ({e[0]})*part_a",
    )
    part_d = _part(
        "Find the distance AP, correct to 1 decimal place.",
        distance,
        "number",
        2,
        [f"AP = |{s}|·|({e[0]}, {e[1]})| = {abs(s)}·√{e[0] ** 2 + e[1] ** 2}."],
        [
            (Fraction(length_e).limit_denominator(10**9), "forgot-multiplier-on-parameter"),
            (abs(s) * (abs(e[0]) + abs(e[1])), "speed-as-sum-of-components"),
            (abs(t) * Fraction(hypot(*d)).limit_denominator(10**9), "wrong-parameter-substituted"),
        ],
        follow=f"sqrt(part_a**2*{e[0] ** 2 + e[1] ** 2})",
        places=1,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
