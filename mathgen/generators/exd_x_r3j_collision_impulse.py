"""Generates `exd_x_r3j_collision_impulse` questions: two bodies collide and couple, then the impulse and force.

Skill: `senior-examine-momentum-force-resultant-force-action-and-reaction` (exam tier). The practice tiers ask
for one momentum or one force; a senior mechanics exam chains them through a collision on a straight line:

(a) total momentum before the collision                                number, 1 mark
(b) common velocity after the bodies couple                            number, 2 marks, part_a / (m1 + m2)
(c) impulse on one of the bodies                                       number, 2 marks, m_x (part_b - v_x)
(d) average force on the OTHER body during the contact (Newton III)    number, 2 marks, -part_c / t

## Construction: choose the outcome first

The two masses, the common velocity v and the slower body's velocity are drawn, then the faster body's velocity
is solved from conservation of momentum and kept only when it is a whole number of m/s. The faster body is
always the one coming up from behind, so the bodies do collide, and the contact time is 0.5, 0.25, 0.2 or 0.1 s
so that the average force is a whole number of newtons. Which body is faster, which way "right" points and which
body part (c) asks about are all drawn, so the signs of (c) and (d) vary.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_x_r3_distinct import distinct_answers
from ._exd_x_r3j_kit import number_part

_SETTINGS = [
    # (stem opening, body noun, mass factor in kg, largest mass number, largest speed in m/s)
    ("Two loaded dynamics trolleys run on a level laboratory track.", "trolley", 1, 5, 6),
    ("Two cargo carts roll along a level warehouse aisle.", "cart", 10, 9, 3),
    ("Two trolleys of different mass are tested on a frictionless air track.", "trolley", 1, 5, 5),
]
_TIMES = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 5), Fraction(1, 10)]


def _speed_phrase(v: int) -> str:
    if v == 0:
        return "is at rest"
    return f"moves {'right' if v > 0 else 'left'} at {abs(v)} m/s"


@generator("exd_x_r3j_collision_impulse", level=17, presets={"exam": {}})
@distinct_answers
def exd_x_r3j_collision_impulse(rng):
    opening, noun, factor, kmax, vmax = rng.choice(_SETTINGS)
    for _ in attempts(2000, "masses and velocities with a whole-number approach velocity"):
        k1, k2 = rng.randint(2, kmax), rng.randint(2, kmax)
        v = rng.randint(-vmax + 1, vmax - 1)
        slow = rng.randint(-vmax, vmax)
        num = (k1 + k2) * v - k2 * slow  # fast = num / k1, with k1 the fast body's mass number
        if v == 0 or num % k1 != 0:
            continue
        fast = num // k1
        if fast <= v or abs(fast) > vmax or v <= slow:
            continue
        break
    fast_is_a = rng.random() < 0.5
    mass = {"A": 0, "B": 0}
    vel = {"A": 0, "B": 0}
    fast_name, slow_name = ("A", "B") if fast_is_a else ("B", "A")
    mass[fast_name], mass[slow_name] = k1 * factor, k2 * factor
    vel[fast_name], vel[slow_name] = fast, slow
    v_after = v
    m_total = (k1 + k2) * factor
    p_before = mass["A"] * vel["A"] + mass["B"] * vel["B"]
    check(p_before == m_total * v_after, "momentum is not conserved")

    x_name = rng.choice(["A", "B"])
    y_name = "B" if x_name == "A" else "A"
    t = rng.choice(_TIMES)
    impulse_x = mass[x_name] * (v_after - vel[x_name])
    check(impulse_x == -mass[y_name] * (v_after - vel[y_name]), "impulses are not equal and opposite")
    force_y = Fraction(-impulse_x) / t
    check(force_y.denominator == 1, "average force is not a whole number of newtons")
    force_y = int(force_y)

    stem = (
        f"{opening} {noun.capitalize()} A has mass {mass['A']} kg and {_speed_phrase(vel['A'])}. "
        f"{noun.capitalize()} B has mass {mass['B']} kg and {_speed_phrase(vel['B'])}. "
        f"They collide and couple together, and the contact lasts {t.numerator / t.denominator:g} s. "
        "Take right as the positive direction."
    )

    ma, mb = mass["A"], mass["B"]
    va, vb = vel["A"], vel["B"]
    mx, vx = mass[x_name], vel[x_name]
    my, vy = mass[y_name], vel[y_name]
    slow_v = vel[slow_name]
    # (a) slips
    mixed = mass[fast_name] * vel[fast_name] + (
        mass[slow_name] * abs(slow_v) if slow_v < 0 else -mass[slow_name] * slow_v
    )
    parts = [
        number_part(
            "Find the total momentum of the two bodies before the collision, in kg·m/s.",
            p_before,
            0,
            1,
            [f"p = {ma}×({va}) + {mb}×({vb}) = {p_before} kg·m/s."],
            [
                (mixed, "momentum-combined-ignoring-direction"),
                (mass[fast_name] * vel[fast_name], "momentum-of-one-object"),
                (mass[slow_name] * vel[slow_name], "momentum-of-one-object"),
                (ma * va - mb * vb, "momentum-combined-ignoring-direction"),
                (mass[fast_name] * vel[slow_name] + mass[slow_name] * vel[fast_name], "masses-and-velocities-swapped"),
                (mass[fast_name] + vel[fast_name] + mass[slow_name] + vel[slow_name], "added-instead-of-multiplied"),
            ],
        ),
        number_part(
            f"Find the velocity of the coupled {noun}s immediately after the collision, in m/s.",
            Fraction(v_after),
            0,
            2,
            [f"Momentum is conserved: ({ma} + {mb})v = {p_before}, so v = {v_after} m/s."],
            [
                (Fraction(p_before, mass[fast_name]), "divided-by-one-mass"),
                (Fraction(p_before, mass[slow_name]), "divided-by-one-mass"),
                (Fraction(va + vb, 2), "averaged-the-velocities"),
                (Fraction(mixed, m_total), "momentum-combined-ignoring-direction"),
                (Fraction(vel[fast_name]), "initial-velocity-for-final-velocity"),
                (Fraction(vel[slow_name]), "initial-velocity-for-final-velocity"),
            ],
            follow_through=f"part_a/{m_total}",
        ),
        number_part(
            f"Find the impulse on {noun} {x_name} during the collision, in N·s.",
            impulse_x,
            0,
            2,
            [f"J = m(v − u) = {mx}×({v_after} − ({vx})) = {impulse_x} N·s."],
            [
                (-impulse_x, "impulse-on-the-other-object"),
                (mx * v_after, "final-momentum-for-impulse"),
                (mx * (v_after + vx), "added-instead-of-subtracted"),
                (-mx * vx, "initial-momentum-for-impulse"),
            ],
            follow_through=f"{mx}*(part_b - ({vx}))",
        ),
        number_part(
            f"Find the average force exerted on {noun} {y_name} during the contact, in newtons.",
            force_y,
            0,
            2,
            [
                f"By Newton's third law the impulse on {noun} {y_name} is {-impulse_x} N·s.",
                f"F = J/t = {-impulse_x} ÷ {t.numerator / t.denominator:g} = {force_y} N.",
            ],
            [
                (-force_y, "impulse-on-the-other-object"),
                (Fraction(-impulse_x), "impulse-for-force"),
                (Fraction(-impulse_x) * t, "multiplied-instead-of-divided"),
                (Fraction(impulse_x) * t, "multiplied-instead-of-divided"),
            ],
            follow_through=f"-part_c/({t.numerator}/{t.denominator})",
        ),
    ]
    check(my * (v_after - vy) == -impulse_x, "force part disagrees with Newton III")
    return dict(text=stem, parts=parts)
