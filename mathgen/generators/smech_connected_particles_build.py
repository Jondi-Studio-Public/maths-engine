"""Connected particles: `smech_connected_particles` (easy, medium, hard) and `smech_connected_particles_exam`.

Skill: connected-particles-pulleys-and-towing. Two bodies joined by a light inextensible string or a rigid tow bar
share one acceleration. The skill is the choice of body: take the whole system to find the acceleration, then take a
single body to find the tension.

Tiers (a new decision each time, not bigger numbers):

* easy: a car tows a trailer along a level road. Treat the pair as one body to find the common acceleration.
* medium: a new question shape for the same idea. Either the tension in the tow bar (isolate the trailer), or a block
  on a smooth table joined over a smooth pulley to a hanging mass (the driving force is only the hanging weight).
* hard: both masses hang over the pulley (the weights oppose each other), or the tension on a table-and-pulley system.
  The tension is neither mass's weight, which is the usual slip.
* exam: the block on the table is now on a rough surface. Friction, acceleration, tension, then the speed gained.

g = 9.8 m/s² wherever it appears, and the prompt says so.

## Construction

Masses are drawn and the resulting accelerations and tensions are computed exactly as `Fraction`s. A draw is kept only
when every quantity asked for is exact to the stated number of decimal places, so no answer is a rounding of a
rounding. The speed in the exam chain is the one rounded value, and is kept clear of a rounding boundary.
Each answer is checked a second way by applying F = ma to each body separately and adding.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, draw_until, generator, phrase
from ._smech_kit import G, G_NOTE, clear_of_boundary, dec, is_exact, number_fields, number_part

_VEHICLES = (("car", "trailer"), ("truck", "trailer"), ("ute", "caravan"), ("tractor", "trailer"))


# ------------------------------------------------------------------------------------------------ easy / medium


def _towing(rng, ask: str):
    car, trailer = rng.choice(_VEHICLES)
    big, small = draw_until(
        rng,
        lambda r: (r.choice((800, 1000, 1200, 1500, 2000)), r.choice((200, 250, 300, 400, 500, 600))),
        lambda d: d[0] > d[1],
        what="a car heavier than its trailer",
    )
    a = draw_until(
        rng,
        lambda r: r.choice((Fraction(1, 2), Fraction(3, 4), Fraction(3, 2), Fraction(5, 4), Fraction(1))),
        lambda x: ((big + small) * x).denominator == 1,
        what="an acceleration that gives a whole net force",
    )
    net = (big + small) * a
    r1, r2 = rng.choice((200, 250, 300, 400)), rng.choice((100, 150, 200, 250))
    drive = int(net) + r1 + r2
    base = (
        f"A {car} of mass {big} kg tows a {trailer} of mass {small} kg along a straight level road. The {car} "
        f"has a driving force of {drive} N. Resistances of {r1} N on the {car} and {r2} N on the {trailer} act "
        f"against the motion."
    )
    if ask == "acceleration":
        text = base + " Find the acceleration of the pair in m/s², correct to 2 decimal places."
        value, dp = a, 2
        wrong = [
            (Fraction(int(net), big), "mech-system-mass-omitted"),
            (Fraction(drive - r1, big + small), "mech-ignored-resistance"),
            (Fraction(drive, big + small), "mech-ignored-resistance"),
            (Fraction(int(net)), "mech-forgot-divide-by-mass"),
        ]
        check(Fraction(drive - r1 - r2, big + small) == a, "second route disagrees")
        working = [
            f"Whole system: {drive} - {r1} - {r2} = ({big} + {small})a.",
            f"a = {int(net)} / {big + small} = {dec(a)} m/s².",
        ]
    else:
        tension = small * a + r2
        text = base + f" Find the tension in the tow bar in newtons, correct to 2 decimal places."
        value, dp = tension, 2
        wrong = [
            (small * a, "mech-ignored-resistance"),
            (small * a - r2, "mech-resistance-added"),
            (Fraction(drive - r1), "mech-forgot-acceleration-of-car"),
            (Fraction(int(net)), "mech-forgot-divide-by-mass"),
        ]
        check(drive - r1 - big * a == tension, "the car's equation gives a different tension")
        working = [
            f"Whole system: a = {int(net)} / {big + small} = {dec(a)} m/s².",
            f"{trailer.capitalize()}: T - {r2} = {small} × {dec(a)}, so T = {dec(tension)} N.",
        ]
    return dict(text=text, working=working, **number_fields(value, dp, wrong))


def _draw_pulley(rng, need):
    """Two masses (kg) such that every quantity in `need` (a function of m1, m2) is exact to 2 places."""
    return draw_until(
        rng,
        lambda r: (r.randint(2, 12), r.randint(2, 12)),
        lambda d: d[0] != d[1] and all(is_exact(q, 2) for q in need(*d)),
        what="two masses with exact answers to 2 decimal places",
    )


def _table(rng, ask: str):
    m1, m2 = _draw_pulley(rng, lambda x, y: (y * G / (x + y), x * y * G / (x + y)))
    a = m2 * G / (m1 + m2)
    tension = m1 * a
    check(tension == m2 * (G - a), "the hanging mass gives a different tension")
    base = (
        f"A block of mass {m1} kg rests on a smooth horizontal table. It is joined by a light inextensible string, "
        f"passing over a smooth pulley at the edge of the table, to a mass of {m2} kg hanging freely. The system is "
        f"released from rest. {G_NOTE}"
    )
    if ask == "acceleration":
        text = base + " Find the acceleration of the block in m/s², correct to 2 decimal places."
        value = a
        wrong = [
            (m2 * G / m1, "mech-system-mass-omitted"),
            (m2 * G, "mech-forgot-divide-by-mass"),
            (m1 * G / (m1 + m2), "mech-masses-swapped"),
            (G, "mech-free-fall-assumed"),
        ]
        working = [f"System: {m2}g = ({m1} + {m2})a.", f"a = {m2} × {dec(G)} / {m1 + m2} = {dec(a)} m/s²."]
    else:
        text = base + " Find the tension in the string in newtons, correct to 2 decimal places."
        value = tension
        wrong = [
            (m2 * G, "mech-tension-equals-weight"),
            (m1 * G, "mech-tension-equals-weight"),
            (a, "mech-forgot-mass-in-resultant"),
            (m2 * a, "mech-wrong-body-mass"),
        ]
        working = [f"a = {m2}g / {m1 + m2} = {dec(a)} m/s².", f"Block: T = {m1} × {dec(a)} = {dec(tension)} N."]
    return dict(text=text, working=working, **number_fields(value, 2, wrong))


def _atwood(rng, ask: str):
    m1, m2 = _draw_pulley(rng, lambda x, y: (abs(y - x) * G / (x + y), 2 * x * y * G / (x + y)))
    light, heavy = min(m1, m2), max(m1, m2)
    a = (heavy - light) * G / (heavy + light)
    tension = 2 * light * heavy * G / (heavy + light)
    check(tension == light * (G + a) and tension == heavy * (G - a), "the two bodies give different tensions")
    base = (
        f"Masses of {light} kg and {heavy} kg are joined by a light inextensible string passing over a smooth "
        f"fixed pulley and both hang freely. They are released from rest. {G_NOTE}"
    )
    if ask == "acceleration":
        text = base + " Find the magnitude of the acceleration in m/s², correct to 2 decimal places."
        value = a
        wrong = [
            ((heavy - light) * G / heavy, "mech-system-mass-omitted"),
            ((heavy - light) * G, "mech-forgot-divide-by-mass"),
            ((heavy + light) * G / (heavy - light), "mech-ratio-inverted"),
            ((heavy - light) * G / light, "mech-system-mass-omitted"),
        ]
        working = [f"System: ({heavy} - {light})g = ({heavy} + {light})a.", f"a = {dec(a)} m/s²."]
    else:
        text = base + " Find the tension in the string in newtons, correct to 2 decimal places."
        value = tension
        wrong = [
            (heavy * G, "mech-tension-equals-weight"),
            (light * G, "mech-tension-equals-weight"),
            ((heavy + light) * G / 2, "mech-tension-averaged"),
            (light * (G - a), "mech-weight-and-acceleration-opposed"),
        ]
        working = [
            f"a = {dec(a)} m/s².",
            f"Lighter mass: T - {light}g = {light}a, so T = {light}({dec(G)} + {dec(a)}) = {dec(tension)} N.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong))


def _easy(rng):
    return _towing(rng, "acceleration")


def _medium(rng):
    return _towing(rng, "tension") if rng.random() < 0.5 else _table(rng, "acceleration")


def _hard(rng):
    pick = rng.choice(("atwood_a", "atwood_t", "table_t"))
    return {
        "atwood_a": lambda: _atwood(rng, "acceleration"),
        "atwood_t": lambda: _atwood(rng, "tension"),
        "table_t": lambda: _table(rng, "tension"),
    }[pick]()


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("smech_connected_particles", level=17, presets={t: {"tier": t} for t in _TIERS})
def smech_connected_particles(rng, tier: str = "easy"):
    """Connected bodies: the system first for the acceleration, then one body for the tension."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam

_EXAM_STEMS = (
    "A block of mass {m1} kg rests on a rough horizontal table. The coefficient of friction between the block "
    "and the table is {mu}. The block is joined by a light inextensible string, passing over a smooth pulley "
    "at the edge of the table, to a mass of {m2} kg hanging freely. The system is released from rest and the "
    "string stays taut. {g}",
    "A block of mass {m1} kg is placed on a rough horizontal table, and the coefficient of friction between the "
    "block and the table is {mu}. A light inextensible string runs from the block over a smooth pulley at the "
    "edge of the table to a mass of {m2} kg hanging freely. Both are released from rest and the string "
    "stays taut. {g}",
    "A light inextensible string joins a block of mass {m1} kg, resting on a rough horizontal table, over a "
    "smooth pulley at the edge of the table to a mass of {m2} kg hanging freely. The coefficient of friction "
    "between the block and the table is {mu}. The system is released from rest and the string stays taut. {g}",
)

_MUS = (Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(3, 10))


@generator("smech_connected_particles_exam", level=17, presets={"exam": {}})
def smech_connected_particles_exam(rng):
    """A block on a rough table joined over a smooth pulley to a hanging mass: friction, acceleration, tension, speed."""
    for _ in attempts(2000, "a rough-table system with exact answers"):
        m1, m2 = rng.randint(2, 12), rng.randint(2, 12)
        mu = rng.choice(_MUS)
        d = rng.choice((Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3)))
        friction = mu * m1 * G
        if not (m2 > mu * m1 and m2 > 2 and is_exact(friction, 2)):
            continue
        a = (m2 * G - friction) / (m1 + m2)
        tension = m2 * (G - a)
        speed_sq = 2 * a * d
        speed = math.sqrt(speed_sq)
        if (
            is_exact(a, 2)
            and is_exact(tension, 2)
            and a > Fraction(1, 5)
            and clear_of_boundary(Fraction(speed).limit_denominator(10**9), 2)
        ):
            break
    # Second route: F = ma for each body separately, solved together.
    check(tension - friction == m1 * a and m2 * G - tension == m2 * a, "the bodies' own equations disagree")
    check(abs(speed * speed - float(speed_sq)) < 1e-9, "speed does not square back")
    stem = phrase(rng, _EXAM_STEMS, m1=m1, m2=m2, mu=dec(mu), g=G_NOTE)
    part_a = number_part(
        "Find the friction force on the block, in newtons, correct to 2 decimal places.",
        friction,
        2,
        1,
        [f"R = {m1}g, so F = {dec(mu)} × {m1} × {dec(G)} = {dec(friction)} N."],
        [
            (mu * m2 * G, "mech-wrong-body-mass"),
            (mu * m1, "mech-forgot-g"),
            (m1 * G, "mech-friction-without-coefficient"),
            (friction * G, "mech-g-applied-twice"),
        ],
        sensible=(0, None),
    )
    part_b = number_part(
        "Find the acceleration of the system, in m/s², correct to 2 decimal places.",
        a,
        2,
        3,
        [f"System: {m2}g - {dec(friction)} = ({m1} + {m2})a.", f"a = {dec(a)} m/s²."],
        [
            (m2 * G / (m1 + m2), "mech-ignored-friction"),
            ((m2 * G + friction) / (m1 + m2), "mech-friction-direction-reversed"),
            ((m2 * G - friction) / m1, "mech-system-mass-omitted"),
            ((m2 * G - friction) / m2, "mech-system-mass-omitted"),
        ],
        follow=f"({m2} * 9.8 - part_a) / {m1 + m2}",
        sensible=(0, None),
    )
    part_c = number_part(
        "Find the tension in the string, in newtons, correct to 2 decimal places.",
        tension,
        2,
        2,
        [f"Hanging mass: {m2}g - T = {m2}a.", f"T = {m2}({dec(G)} - {dec(a)}) = {dec(tension)} N."],
        [
            (m2 * G, "mech-tension-equals-weight"),
            (m2 * (G + a), "mech-weight-and-acceleration-opposed"),
            (m1 * a, "mech-friction-omitted-from-block"),
            (a, "mech-forgot-mass-in-resultant"),
        ],
        follow=f"{m2} * (9.8 - part_b)",
        sensible=(0, None),
    )
    part_d = number_part(
        f"Find the speed of the system after the hanging mass has fallen {dec(d)} m, in m/s, "
        f"correct to 2 decimal places.",
        Fraction(speed).limit_denominator(10**9),
        2,
        2,
        [f"v² = 0 + 2 × {dec(a)} × {dec(d)} = {dec(speed_sq)}.", f"v = {round(speed, 2)} m/s."],
        [
            (speed_sq, "mech-forgot-square-root"),
            (a * d, "mech-forgot-two-in-kinematics"),
            (Fraction(math.sqrt(float(a * d))).limit_denominator(10**9), "mech-forgot-two-in-kinematics"),
            (
                Fraction(math.sqrt(float(2 * (m2 * G / (m1 + m2)) * d))).limit_denominator(10**9),
                "mech-ignored-friction",
            ),
        ],
        follow=f"sqrt(2 * part_b * {dec(d)})",
        sensible=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
