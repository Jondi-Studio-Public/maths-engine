"""Exam-tier source for `directional-derivative`: the rate of change of a surface towards another point.

The surface is a cubic-in-two-variables height function (a term in x^2 y, a cross term, a y^2 term and a linear
term), so each partial derivative needs the product structure, not a power rule alone. The direction is given as
a second point, not a unit vector, so the learner must form the displacement and normalise it. Four parts:

* (a) and (b) the two partial derivatives at the start point,
* (c) the rate of change towards the second point (follow-through from (a) and (b)),
* (d) the greatest rate of change at the start point, the length of the gradient (follow-through).

Built backwards from the coefficients, the start point and a displacement whose length is a whole number
(a Pythagorean triple, or twice one), so the unit vector has a clean denominator. Partials are checked with SymPy.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    numeric_answer,
    paren_negative,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_TRIPLES = [(3, 4), (4, 3), (5, 12), (12, 5), (8, 15), (15, 8), (7, 24)]
_SCALES = [1, 1, 1, 2]
_SETTINGS = [
    "The height of a hillside, in metres, at a point (x, y) kilometres east and north of a lookout near Katoomba is",
    "The height of a sand dune, in metres, at a point (x, y) hundreds of metres east and north of a camp near "
    "Dunk Island is",
    "The temperature of a flat metal plate, in degrees Celsius, at a point (x, y) centimetres from one corner is",
]
_UNITS = ["metres per kilometre", "metres per hundred metres", "degrees Celsius per centimetre"]
_X, _Y = sympy.symbols("x y")


def _dp2(value: Fraction) -> Fraction:
    return round_half_up(value, 2)


def _text(value: Fraction) -> str:
    return stated_precision_string(_dp2(value), 2)


def _sqrt_fraction(n: int) -> Fraction:
    return Fraction(str(sympy.N(sympy.sqrt(n), 40)))


def _clear(value: Fraction) -> bool:
    return abs((value * 100) % 1 - Fraction(1, 2)) > Fraction(1, 20)


@generator("exd_x_u3_directional_toward", level=18, presets={"exam": dict(tier="exam")})
def exd_x_u3_directional_toward(rng, tier: str = "exam") -> dict:
    """Exam tier: partial derivatives, then the rate towards a second point, then the steepest rate."""
    check(tier == "exam", f"unknown tier {tier!r}")
    setting_index = rng.randrange(len(_SETTINGS))
    for _ in attempts(1000, "a surface and direction with a clear rounding"):
        a = rng.choice([-3, -2, -1, 1, 2, 3])
        b = rng.randint(-4, 4)
        c = rng.randint(-4, 4)
        d = rng.randint(-5, 5)
        x0, y0 = rng.randint(-3, 3), rng.randint(-3, 3)
        u, v = rng.choice(_TRIPLES)
        k = rng.choice(_SCALES)
        sx, sy = rng.choice([1, -1]), rng.choice([1, -1])
        dx, dy = sx * u * k, sy * v * k
        length = int(sympy.sqrt(dx * dx + dy * dy))
        g1 = 2 * a * x0 * y0 + b * y0 + d
        g2 = a * x0 * x0 + b * x0 + 2 * c * y0
        if b == 0 or x0 == 0 or y0 == 0 or g1 == 0 or g2 == 0 or g1 == g2 or abs(g1) > 60 or abs(g2) > 60:
            continue
        directional = Fraction(g1 * dx + g2 * dy, length)
        steepest = _sqrt_fraction(g1 * g1 + g2 * g2)
        if _clear(directional) and _clear(steepest) and directional != 0:
            break

    f = a * _X**2 * _Y + b * _X * _Y + c * _Y**2 + d * _X
    check(sympy.diff(f, _X).subs({_X: x0, _Y: y0}) == g1, "df/dx disagrees with SymPy")
    check(sympy.diff(f, _Y).subs({_X: x0, _Y: y0}) == g2, "df/dy disagrees with SymPy")
    # Independent route: the derivative of f along the line from P to Q, at the start of the line.
    t = sympy.symbols("t")
    along = f.subs({_X: x0 + t * sympy.Rational(dx, length), _Y: y0 + t * sympy.Rational(dy, length)})
    check(sympy.diff(along, t).subs(t, 0) == sympy.Rational(directional.numerator, directional.denominator), "D_u f")

    qx, qy = x0 + dx, y0 + dy
    expr = join_terms([(a, "x^2y"), (b, "xy"), (c, "y^2"), (d, "x")])
    unit = _UNITS[setting_index]
    stem = f"{_SETTINGS[setting_index]} f(x, y) = {expr}. Consider the point P({x0}, {y0}) and the point Q({qx}, {qy})."

    def whole(value: int, tag: str | None = None):
        return Distractor(str(value), tag, value)

    def dec(value: Fraction, tag: str | None = None):
        n = _dp2(value)
        return Distractor(stated_precision_string(n, 2), tag, n)

    a_cand = [
        whole(g1 - b * y0, "gradient-term-omitted"),  # cross term left out
        whole(g2),
        whole(2 * a * x0 + b * y0 + d),  # product factor y dropped from the x^2 y term
        whole(g1 - d),  # linear term's derivative left out
        whole(g1 - 2 * (2 * a * x0 * y0), "substitution-sign-error"),  # sign of the x^2 y term lost
        whole(-g1, "substitution-sign-error"),
    ]
    b_cand = [
        whole(g2 - b * x0, "gradient-term-omitted"),
        whole(g1),
        whole(a * x0 + b * x0 + 2 * c * y0),  # x^2 y differentiated as if x were not squared
        whole(g2 - 2 * c * y0),  # the y^2 term's derivative left out
        whole(-g2, "substitution-sign-error"),
    ]
    c_cand = [
        dec(Fraction(g1 * dx + g2 * dy), "direction-not-normalised"),
        dec(-directional, "direction-reversed"),
        dec(Fraction(g1 * dy + g2 * dx, length), "gradient-components-swapped"),
        dec(steepest, "gradient-length-for-directional"),
    ]
    d_cand = [
        dec(abs(directional)),
        dec(Fraction(g1 * g1 + g2 * g2)),  # square root never taken
        dec(Fraction(abs(g1) + abs(g2))),
        dec(Fraction(abs(g1 * dx + g2 * dy))),  # dot product with the displacement, never divided by its length
    ]
    a_text, b_text, c_text, d_text = str(g1), str(g2), _text(directional), _text(steepest)
    a_d = pick_distractors(a_text, a_cand, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_cand, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(c_text, c_cand, 3, tolerance=GRADER_TOLERANCE)
    d_d = pick_distractors(d_text, d_cand, 3, tolerance=GRADER_TOLERANCE)
    check(
        all(len(x) >= 2 for x in (a_d, b_d, c_d, d_d)), f"too few distractors {[len(x) for x in (a_d, b_d, c_d, d_d)]}"
    )

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the partial derivative of f with respect to x at P.",
                answer_kind="number",
                answer=g1,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                working=[f"f_x = {sympy.sstr(sympy.diff(f, _X))}, so f_x({x0}, {y0}) = {g1}."],
                distractors=a_d,
            ),
            dict(
                prompt="Find the partial derivative of f with respect to y at P.",
                answer_kind="number",
                answer=g2,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                working=[f"f_y = {sympy.sstr(sympy.diff(f, _Y))}, so f_y({x0}, {y0}) = {g2}."],
                distractors=b_d,
            ),
            dict(
                prompt=f"Using your answers to (a) and (b), find the rate of change of f at P in the direction "
                f"from P towards Q, in {unit}, correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(_dp2(directional)),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=3,
                follow_through=f"({dx}*part_a + {dy}*part_b)/{length}",
                working=[
                    f"PQ = ({dx}, {dy}), of length {length}, so the unit vector is ({dx}/{length}, {dy}/{length}).",
                    f"D = ({paren_negative(g1)} × {paren_negative(dx)} + {paren_negative(g2)} × {paren_negative(dy)}) ÷ {length} = {c_text} (2 d.p.).",
                ],
                distractors=c_d,
            ),
            dict(
                prompt="Using your answers to (a) and (b), find the greatest rate of change of f at P in any "
                f"direction, in {unit}, correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(_dp2(steepest)),
                answer_text=d_text,
                answer_latex=latex_answer(d_text),
                marks=2,
                sensible_range=(0, None),
                follow_through="sqrt(part_a**2 + part_b**2)",
                working=[
                    f"The greatest rate is the length of the gradient: √({paren_negative(g1)}² + {paren_negative(g2)}²) = {d_text} (2 d.p.)."
                ],
                distractors=d_d,
            ),
        ],
    )
