"""Generates `exd_x_r2a_vector_motion_chain` questions: a multi-part exam chain on a particle moving in the plane.

Skill: senior-differentiate-and-integrate-a-vector-function-with-respect (exam tier). The acceleration vector is
given with the initial velocity and position. The learner integrates component by component, then uses the result:

(a) the x-component of the velocity v(t)                          expression, 2 marks
(b) the y-component of the velocity v(t)                          expression, 2 marks
(c) the speed at t = T, to 1 decimal place                        number,     1 mark,  from the learner's (a) and (b)
(d) the distance from the origin at t = T, to 1 decimal place     number,     2 marks, integrates the learner's (a), (b)

## Construction

The velocity components are drawn as ``A t^2 + B t + C`` with integer A, B, C (so the acceleration is
``2A t + B``), and the starting position is an integer pair. Speed and distance are square roots of exact rationals,
so each answer is computed once with exact arithmetic, rounded half up, and kept at least a small margin away from a
rounding boundary. The independent check recomputes both from ``sympy`` integration.

## Follow-through

(c) evaluates the learner's two velocity components at T. (d) recovers each component's A, B, C from the learner's
expression (its values at t = 1, -1 and 0), integrates, adds the starting position and takes the length.
"""

from __future__ import annotations

import random

from sympy import Integer, Poly, Rational, sqrt

from ..core import check, generator, join_terms
from .exd_x_r2a_common import (
    T,
    Redraw,
    at,
    build_with_redraw,
    expr_part,
    number_part,
    paren,
    shield,
    shield,
    part_prompt,
    pretty,
    prose,
    rounding_is_safe,
    same_function,
)

_PATTERNS = (
    "A particle moves in the plane. Its acceleration at time t seconds is a(t) = ({ax}, {ay}) m/s². At t = 0 its "
    "velocity is ({u1}, {u2}) m/s and its position is ({p1}, {p2}) m.",
    "The acceleration of a drone flying over a paddock is a(t) = ({ax}, {ay}) m/s², where t is in seconds. When "
    "t = 0 it has velocity ({u1}, {u2}) m/s and is at the point ({p1}, {p2}) metres from the origin.",
    "A radio-controlled car on a flat oval track has acceleration a(t) = ({ax}, {ay}) m/s² at t seconds. At t = 0 its "
    "velocity is ({u1}, {u2}) m/s and its position vector is ({p1}, {p2}) m.",
)


def _poly(c2, c1, c0):
    return c2 * T**2 + c1 * T + c0


def _definite(poly, upper) -> Rational:
    """The integral of the polynomial `poly` in t from 0 to `upper`, from its exact antiderivative."""
    return Poly(poly, T).integrate().as_expr().subs(T, upper)


def _position_ft(label: str, start: int, upper: int) -> str:
    """Follow-through text for the position at t = `upper`: the start plus the integral of the quadratic in t a learner
    gave in part `label`, read from its values at t = 1, -1 and 0 (A = (v1 + v-1)/2 - v0, B = (v1 - v-1)/2, C = v0)."""
    u = Rational(upper)
    at_one = u**3 / 6 + u**2 / 4
    at_minus_one = u**3 / 6 - u**2 / 4
    at_zero = u - u**3 / 3
    return (
        f"({paren(start)} + {paren(at_one)}*{at(label, 1, 't')} + {paren(at_minus_one)}*{at(label, -1, 't')}"
        f" + {paren(at_zero)}*{at(label, 0, 't')})"
    )


def _build(rng: random.Random):
    big_t = rng.choice([2, 3])
    ax = [rng.choice([-2, -1, 1, 2]), rng.choice([m for m in range(-5, 6) if m != 0]), rng.randint(-5, 5)]
    ay = [rng.choice([-2, -1, 1, 2]), rng.choice([m for m in range(-5, 6) if m != 0]), rng.randint(-5, 5)]
    # ax = (A, B, C): velocity component A t^2 + B t + C, acceleration 2A t + B
    p = [rng.randint(-8, 8), rng.randint(-8, 8)]
    if ax[2] == 0 and ay[2] == 0:
        raise Redraw
    vx, vy = _poly(*ax), _poly(*ay)
    vxT, vyT = vx.subs(T, big_t), vy.subs(T, big_t)
    speed = sqrt(vxT**2 + vyT**2)
    xT = Rational(p[0]) + Rational(ax[0] * big_t**3, 3) + Rational(ax[1] * big_t**2, 2) + ax[2] * big_t
    yT = Rational(p[1]) + Rational(ay[0] * big_t**3, 3) + Rational(ay[1] * big_t**2, 2) + ay[2] * big_t
    check(
        xT == p[0] + _definite(vx, big_t) and yT == p[1] + _definite(vy, big_t),
        "positions disagree with the exact antiderivative",
    )
    dist = sqrt(xT**2 + yT**2)
    disp = sqrt((xT - p[0]) ** 2 + (yT - p[1]) ** 2)
    if not (rounding_is_safe(speed, 1) and rounding_is_safe(dist, 1)):
        raise Redraw
    if speed < 1 or dist < 1 or abs(vxT) > 40 or abs(vyT) > 40 or abs(xT) > 80 or abs(yT) > 80:
        raise Redraw

    def accel(c):
        return join_terms([(2 * c[0], "t"), (c[1], "")])

    wrong_x = [
        ("forgot-initial-condition", _poly(ax[0], ax[1], 0)),
        ("integration-power-not-divided", _poly(2 * ax[0], ax[1], ax[2])),
        ("differentiated-instead-of-integrated", Integer(2 * ax[0])),
        ("wrong-initial-condition-used", _poly(ax[0], ax[1], 0) + p[0]),
    ]
    wrong_y = [
        ("forgot-initial-condition", _poly(ay[0], ay[1], 0)),
        ("integration-power-not-divided", _poly(2 * ay[0], ay[1], ay[2])),
        ("differentiated-instead-of-integrated", Integer(2 * ay[0])),
        ("wrong-initial-condition-used", _poly(ay[0], ay[1], 0) + p[1]),
    ]
    rng.shuffle(wrong_x)
    rng.shuffle(wrong_y)

    def vel_part(label_word, comp, accel_c, wrong):
        return expr_part(
            part_prompt("Find " + label_word + ", the {c}-component of the velocity at time t.", c=(comp, comp)),
            _poly(*(ax if comp == "x" else ay)),
            2,
            [
                f"Integrate the {comp}-component of the acceleration: v_{comp}(t) = ∫({accel(accel_c)}) dt.",
                f"Use the initial velocity to find the constant: v_{comp}(t) = {pretty(_poly(*accel_c))}.",
            ],
            wrong,
            var=T,
        )

    part_a = vel_part("v_x(t)", "x", ax, wrong_x)
    part_b = vel_part("v_y(t)", "y", ay, wrong_y)

    vxft, vyft = at("a", big_t, "t"), at("b", big_t, "t")
    part_c = number_part(
        part_prompt("Hence find the speed when t = {T}, correct to 1 decimal place.", T=(str(big_t), str(big_t))),
        speed,
        1,
        1,
        [f"v({big_t}) = ({vxT}, {vyT}) m/s.", f"Speed = √({vxT}² + {vyT}²)."],
        [
            ("speed-as-sum-of-components", abs(vxT) + abs(vyT)),
            ("forgets-square-root", vxT**2 + vyT**2),
            (
                "acceleration-magnitude-for-speed",
                sqrt((2 * ax[0] * big_t + ax[1]) ** 2 + (2 * ay[0] * big_t + ay[1]) ** 2),
            ),
            ("used-one-component-only", abs(vxT)),
        ],
        follow_through=f"sqrt({shield(vxft + '**2', 't')} + {shield(vyft + '**2', 't')})",
        sensible_range=(0, None),
        unit="m/s",
    )
    x_ft = _position_ft("a", p[0], big_t)
    y_ft = _position_ft("b", p[1], big_t)
    part_d = number_part(
        part_prompt(
            "Hence find the distance from the origin when t = {T}, correct to 1 decimal place.",
            T=(str(big_t), str(big_t)),
        ),
        dist,
        1,
        2,
        [
            f"Integrate v again and use the starting position: r(t) = ({p[0]} + ∫v_x dt, {p[1]} + ∫v_y dt).",
            f"r({big_t}) = ({xT}, {yT}); distance = √(({xT})² + ({yT})²).",
        ],
        [
            ("forgot-initial-condition", disp),
            ("velocity-used-as-position", speed),
            ("constant-speed-assumed", speed * big_t),
            ("coordinates-added-not-squared", abs(xT) + abs(yT)),
        ],
        follow_through=f"sqrt({shield(x_ft + '**2', 't')} + {shield(y_ft + '**2', 't')})",
        sensible_range=(0, None),
        unit="m",
    )
    u1, u2 = ax[2], ay[2]
    stem, stem_latex = prose(
        rng,
        _PATTERNS,
        ax=(accel(ax), accel(ax)),
        ay=(accel(ay), accel(ay)),
        u1=(str(u1), str(u1)),
        u2=(str(u2), str(u2)),
        p1=(str(p[0]), str(p[0])),
        p2=(str(p[1]), str(p[1])),
    )
    check(
        Poly(vx, T).diff(T).as_expr() == 2 * ax[0] * T + ax[1],
        "velocity does not differentiate to the acceleration",
    )
    return dict(text=stem, latex=stem_latex, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2a_vector_motion_chain", level=16, presets={"exam": {}})
def exd_x_r2a_vector_motion_chain(rng: random.Random):
    """Integrate an acceleration vector: the velocity components, the speed and the distance from the origin."""
    return build_with_redraw(rng, _build)
