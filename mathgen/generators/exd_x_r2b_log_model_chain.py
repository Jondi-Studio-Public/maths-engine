"""Generates `exd_x_r2b_log_model_chain` questions: a multi-part exam chain on a logarithmic model.

Skill: senior-use-logarithmic-functions-and-their-derivatives-to-solve (exam tier).

A response is modelled by `V(x) = a ln(x) + b` for `x > 0`, where `b` is stated and the positive constant `a` is not.
The rate of change `V'(x) = a / x` is given at one value of x, which fixes `a`; a second rate then fixes a second x:

(a) find a from the rate at x0                                    fraction, 2 marks
(b) the x at which the rate has a second stated value             fraction, 1 mark,  x1 = a / r1
(c) V at that x, to 2 decimal places                              number,   2 marks, a ln(x1) + b
(d) the average rate of change of V from x0 to x1, to 3 places    number,   2 marks, from V(x1) and V(x0)

## Construction: choose a and the two x-values first

`a`, `x0` and `x1` are drawn and the rates are read off, `r0 = a / x0` and `r1 = a / x1`, each needing at most three
decimal places so that it can be stated exactly. The numeric keys (c) and (d) are drawn only when they sit clear of a
rounding boundary, and the route a learner takes after rounding (c) first, or rounding both values of V first, must
give the same last digit as the key.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import log

from ..core import check, generator, phrase, round_half_up
from .exd_x_r2b_support import Redraw, clear_of_boundary, draw, fraction_part, number_part, real

_X0 = (4, 5, 10, 20, 25, 50)
_CONTEXTS = (
    ("A patient's response score V, in points, to a drug dose", "the dose in milligrams", "mg"),
    ("The yield index V, in points, of a crop", "the fertiliser applied in kilograms per hectare", "kg/ha"),
    ("The sales index V, in points, of a product", "the advertising spend in thousands of dollars", "$1000"),
    ("The brightness rating V, in points, of a lamp", "the power supplied in watts", "watt"),
)
_PATTERNS = (
    "{subject} is modelled by V(x) = a ln(x) {b}, where x > 0 is {xdesc} and a is a positive constant. "
    "When x = {x0}, the rate of change of V is {r0} {pts} per {unit}.",
    "{subject} is given by V(x) = a ln(x) {b} for x > 0, where x is {xdesc} and a is a positive constant. "
    "The rate of change of V with respect to x is {r0} {pts} per {unit} when x = {x0}.",
)


def _dec(value: Fraction) -> str:
    """The shortest decimal for a rational with a terminating expansion."""
    for places in range(0, 7):
        scaled = value * 10**places
        if scaled.denominator == 1:
            return f"{scaled.numerator / 10**places:.{places}f}"
    raise Redraw("not a terminating decimal")


def _points(rate: Fraction) -> str:
    return "point" if rate == 1 else "points"


def _build(rng):
    a = rng.randint(4, 40)
    x0 = rng.choice(_X0)
    b = rng.choice([v for v in range(-20, 31) if v])
    choices = [c for c in range(2 * x0, 401) if (1000 * a) % c == 0 and (c % x0 == 0 or c > 3 * x0)]
    if not choices:
        raise Redraw("no second x")
    x1 = rng.choice(choices)
    r0, r1 = Fraction(a, x0), Fraction(a, x1)

    v1 = real(a * log(x1) + b)
    v0 = real(a * log(x0) + b)
    avg = (v1 - v0) / (x1 - x0)
    if not (clear_of_boundary(v1, 2) and clear_of_boundary(avg, 3)):
        raise Redraw("a key too close to a rounding tie")
    # The routes a learner may take after rounding must read the same last digit as the key.
    v1r = round_half_up(v1, 2)
    v0r = round_half_up(v0, 2)
    key_d = round_half_up(avg, 3)
    for route in ((v1r - v0) / (x1 - x0), (v1r - v0r) / (x1 - x0)):
        if round_half_up(route, 3) != key_d:
            raise Redraw("rounding V first changes the last digit of the average rate")
    # An independent route: integrate the rate, the change in V is a ln(x1 / x0).
    check(abs(float(v1 - v0) - a * math.log(x1 / x0)) < 1e-9, "average rate disagrees with a ln(x1/x0)")

    subject, xdesc, unit = rng.choice(_CONTEXTS)
    stem = phrase(
        rng,
        _PATTERNS,
        pts=_points(r0),
        subject=subject,
        xdesc=xdesc,
        unit=unit,
        b=f"+ {b}" if b > 0 else f"- {-b}",
        x0=x0,
        r0=_dec(r0),
    )

    # (a) a = r0 x0
    a_wrong = [
        (r0 / x0, "inverted-ratio"),  # a = r0 / x0
        (r0, "forgets-the-multiplier"),  # the rate read as the coefficient
        (r0 * x0 * x0, "log-derivative-wrong-form"),  # V' = a / x^2
    ]
    part_a = fraction_part(
        "Find the value of a.",
        a,
        2,
        ["V'(x) = a/x.", f"V'({x0}) = a/{x0} = {_dec(r0)}, so a = {a}."],
        a_wrong,
        sensible_range=(0, None),
    )

    # (b) x1 = a / r1
    b_wrong = [
        (a * r1, "multiplied-instead-of-divided"),
        (r1 / a, "inverted-ratio"),
        (Fraction(a) / (r1 * r1), "log-derivative-wrong-form"),  # V' = a / x^2
    ]
    part_b = fraction_part(
        f"Find the value of x for which the rate of change of V is {_dec(r1)} {_points(r1)} per {unit}.",
        x1,
        1,
        [f"a/x = {_dec(r1)}, so x = {a}/{_dec(r1)} = {x1}."],
        b_wrong,
        follow_through=f"part_a/Rational({r1.numerator}, {r1.denominator})",
        sensible_range=(0, None),
    )

    # (c) V(x1)
    c_wrong = [
        (real(a * log(x1)), "dropped-constant-term"),
        (real(a * log(x1, 10) + b), "log-base-confused"),
        (v0, "used-initial-value"),
    ]
    part_c = number_part(
        "Hence find the value of V at this value of x.",
        v1,
        2,
        2,
        [f"V({x1}) = {a} ln({x1}) {'+' if b >= 0 else '-'} {abs(b)} = {v1r}."],
        c_wrong,
        follow_through=f"part_a*log(part_b) + {b if b >= 0 else f'({b})'}",
    )

    # (d) the average rate of change
    d_wrong = [
        ((v1 - v0) / x1, "divided-by-endpoint-not-width"),
        (r1, "instantaneous-for-average"),
        (v1 - v0, "total-not-divided-by-width"),
    ]
    part_d = number_part(
        f"Hence find the average rate of change of V, in points per {unit}, as x increases from {x0} to this value.",
        avg,
        3,
        2,
        [
            f"V({x0}) = {a} ln({x0}) {'+' if b >= 0 else '-'} {abs(b)} = {round_half_up(v0, 2)}.",
            f"Average rate = (V({x1}) - V({x0})) / ({x1} - {x0}) = {key_d}.",
        ],
        d_wrong,
        follow_through=f"(part_c - (part_a*log({x0}) + {b if b >= 0 else f'({b})'}))/(part_b - {x0})",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2b_log_model_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_log_model_chain(rng, tier: str = "exam"):
    """A logarithmic model with an unknown coefficient: a, the x at a second rate, V there, an average rate."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
