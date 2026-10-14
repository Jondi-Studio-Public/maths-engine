"""Particles on a rough inclined plane: `smech_inclined_plane` (easy, medium, hard) and `smech_inclined_plane_exam`.

Skill: inclined-plane-friction. The plane's slope is given by a 3-4-5 triangle (sin θ = 3/5 or 4/5), so sin θ and cos θ
are exact and no calculator rounding enters the working. The prompt gives sin θ, or a ramp that rises 3 m for every
5 m of its length, and the learner resolves the weight into the components along and perpendicular to the plane.

Tiers, each a different decision:

* easy: resolve only. The normal reaction, the weight component along the slope, or the greatest friction μR.
  Choosing cos for the perpendicular component and sin for the one along the slope is the whole question.
* medium: equilibrium on the slope. Either the coefficient of friction when the block is about to slip (μ = tan θ,
  the mass cancels), or the friction acting on a block that stays at rest, which is mg sin θ and not μR.
* hard: the block moves. Its acceleration down the slope, or the force up the slope that keeps it moving at
  constant speed. Friction is limiting and its direction follows the motion.
* exam: a four-part chain: normal reaction, greatest friction, acceleration down the slope, distance travelled.

g = 9.8 m/s², stated in every prompt that uses it. Every answer is built from exact `Fraction`s and kept only when it
is exact to the places the prompt asks for; the exam's chain is also solved a second way, by applying F = ma along the
slope with the forces as signed components.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, draw_until, generator, phrase
from ._smech_kit import G, G_NOTE, dec, is_exact, number_fields, number_part

_SLOPES = ((Fraction(3, 5), Fraction(4, 5)), (Fraction(4, 5), Fraction(3, 5)))  # (sin, cos)
_MU = (Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(1, 2), Fraction(2, 5), Fraction(3, 10))
_THINGS = ("crate", "box", "block", "suitcase", "trolley")
_SLOPE_TEXT = (
    "a rough plane inclined at an angle θ to the horizontal, where sin θ = {s}",
    "a rough ramp whose slope makes an angle θ with the horizontal, where sin θ = {s}",
)


def _slope_phrase(rng, s: Fraction) -> str:
    return phrase(rng, _SLOPE_TEXT, s=f"{s.numerator}/{s.denominator}")


def _mass(rng):
    return rng.choice((2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25))


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    thing = rng.choice(_THINGS)
    sin, cos = rng.choice(_SLOPES)
    mu = rng.choice(_MU)
    m = draw_until(rng, _mass, lambda x: is_exact(x * G * cos * mu, 2), what="a mass with exact friction")
    ask = rng.choice(("normal", "along", "friction"))
    where = _slope_phrase(rng, sin)
    head = f"A {thing} of mass {m} kg rests on {where}. "
    if ask == "normal":
        text = (
            head
            + f"Find the normal reaction of the plane on the {thing}, in newtons, correct to 2 decimal places. "
            + G_NOTE
        )
        value = m * G * cos
        wrong = [
            (m * G * sin, "mech-sin-cos-swapped"),
            (m * G, "mech-weight-not-resolved"),
            (m * cos, "mech-forgot-g"),
            (m * G * cos * cos, "mech-component-resolved-twice"),
        ]
        working = [f"Perpendicular to the plane: R = {m}g cos θ = {m} × {dec(G)} × {dec(cos)} = {dec(value)} N."]
    elif ask == "along":
        text = (
            head
            + f"Find the component of the {thing}'s weight acting down the slope, in newtons, correct to 2 decimal places. "
            + G_NOTE
        )
        value = m * G * sin
        wrong = [
            (m * G * cos, "mech-sin-cos-swapped"),
            (m * G, "mech-weight-not-resolved"),
            (m * sin, "mech-forgot-g"),
            (m * G * sin * sin, "mech-component-resolved-twice"),
        ]
        working = [f"Along the slope: {m}g sin θ = {m} × {dec(G)} × {dec(sin)} = {dec(value)} N."]
    else:
        text = (
            head + f"The coefficient of friction between the {thing} and the plane is {dec(mu)}. Find the greatest "
            f"friction force the plane can exert, in newtons, correct to 2 decimal places. {G_NOTE}"
        )
        value = mu * m * G * cos
        wrong = [
            (mu * m * G * sin, "mech-sin-cos-swapped"),
            (mu * m * G, "mech-weight-not-resolved"),
            (m * G * cos, "mech-friction-without-coefficient"),
            (mu * m * cos, "mech-forgot-g"),
        ]
        working = [
            f"R = {m}g cos θ = {dec(m * G * cos)} N.",
            f"F = μR = {dec(mu)} × {dec(m * G * cos)} = {dec(value)} N.",
        ]
    check(is_exact(value, 2), "answer is not exact to 2 places")
    return dict(text=text, working=working, **number_fields(value, 2, wrong))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    thing = rng.choice(_THINGS)
    m = _mass(rng)
    if rng.random() < 0.5:
        sin, cos = _SLOPES[0]  # tan θ = 3/4 = 0.75 is exact
        text = (
            f"A {thing} of mass {m} kg is on a rough plane inclined at an angle θ to the horizontal, where "
            f"sin θ = 3/5. The {thing} is on the point of slipping down the plane. Find the coefficient of friction "
            f"between the {thing} and the plane, correct to 2 decimal places."
        )
        value = sin / cos
        check(m * G * sin == value * m * G * cos, "limiting equilibrium does not balance")
        wrong = [
            (cos / sin, "mech-tan-inverted"),
            (sin, "mech-sin-used-as-mu"),
            (cos, "mech-cos-used-as-mu"),
            (m * G * sin / (m * G), "mech-sin-used-as-mu"),
        ]
        working = [
            f"Along the slope: F = {m}g sin θ. Perpendicular: R = {m}g cos θ.",
            "Limiting friction: F = μR, so μ = sin θ / cos θ = tan θ = 0.75.",
        ]
        return dict(text=text, working=working, **number_fields(value, 2, wrong))
    sin, cos = _SLOPES[0]  # tan θ = 3/4, so a coefficient of 0.8 or more holds the block
    mu = rng.choice((Fraction(4, 5), Fraction(9, 10), Fraction(1)))
    check(mu > sin / cos, "friction holds the block only if mu exceeds tan θ")
    m = draw_until(rng, _mass, lambda x: is_exact(x * G * sin, 2), what="a mass with an exact weight component")
    text = (
        f"A {thing} of mass {m} kg rests on a rough plane inclined at an angle θ to the horizontal, where "
        f"sin θ = {sin.numerator}/{sin.denominator}. The coefficient of friction between the {thing} and the plane "
        f"is {dec(mu)}, and the {thing} stays at rest. {G_NOTE} Find the friction force acting on the {thing}, "
        f"in newtons, correct to 2 decimal places."
    )
    value = m * G * sin
    check(value < mu * m * G * cos, "the block would slip")
    wrong = [
        (mu * m * G * cos, "mech-limiting-friction-assumed"),
        (m * G * cos, "mech-sin-cos-swapped"),
        (m * G, "mech-weight-not-resolved"),
        (mu * m * G * sin, "mech-sin-cos-swapped"),
    ]
    working = [
        f"At rest, the forces along the slope balance: F = {m}g sin θ = {dec(value)} N.",
        f"This is less than the greatest friction μR = {dec(mu * m * G * cos)} N, so the {thing} does not slip.",
    ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong))


# ------------------------------------------------------------------------------------------------ hard


def _hard(rng):
    thing = rng.choice(_THINGS)
    sin, cos = rng.choice(_SLOPES)
    for _ in attempts(1000, "a sliding block with an exact answer"):
        mu = rng.choice(_MU)
        m = _mass(rng)
        down = sin - mu * cos
        if down > Fraction(1, 10) and is_exact(G * down, 2) and is_exact(m * G * (sin + mu * cos), 2):
            break
    slope = f"sin θ = {sin.numerator}/{sin.denominator}"
    if rng.random() < 0.5:
        text = (
            f"A {thing} of mass {m} kg is released from rest on a rough plane inclined at an angle θ to the "
            f"horizontal, where {slope}. The coefficient of friction between the {thing} and the plane is {dec(mu)}. "
            f"{G_NOTE} Find the acceleration of the {thing} down the plane, in m/s², correct to 2 decimal places."
        )
        value = G * down
        check(m * value == m * G * sin - mu * m * G * cos, "equation of motion along the slope disagrees")
        wrong = [
            (G * sin, "mech-ignored-friction"),
            (G * (sin + mu * cos), "mech-friction-direction-reversed"),
            (G * (sin - mu), "mech-weight-not-resolved"),
            (G * (cos - mu * sin), "mech-sin-cos-swapped"),
        ]
        working = [
            f"R = {m}g cos θ, so F = μR = {dec(mu * m * G * cos)} N.",
            f"Down the slope: {m}g sin θ - F = {m}a, so a = {dec(G)}({dec(sin)} - {dec(mu)} × {dec(cos)}) = {dec(value)} m/s².",
        ]
    else:
        text = (
            f"A {thing} of mass {m} kg is pulled up a rough plane inclined at an angle θ to the horizontal, "
            f"where {slope}, by a force parallel to the plane. The coefficient of friction is {dec(mu)} and the "
            f"{thing} moves up at constant speed. {G_NOTE} Find the pulling force, in newtons, "
            f"correct to 2 decimal places."
        )
        value = m * G * (sin + mu * cos)
        wrong = [
            (m * G * (sin - mu * cos), "mech-friction-direction-reversed"),
            (m * G * sin, "mech-ignored-friction"),
            (mu * m * G * cos, "mech-ignored-weight-component"),
            (m * G * (sin + mu), "mech-weight-not-resolved"),
        ]
        working = [
            "Constant speed, so the forces along the slope balance.",
            f"P = {m}g sin θ + μ × {m}g cos θ = {dec(m * G * sin)} + {dec(mu * m * G * cos)} = {dec(value)} N.",
        ]
    check(is_exact(value, 2), "answer is not exact to 2 places")
    return dict(text=text, working=working, **number_fields(value, 2, wrong))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("smech_inclined_plane", level=17, presets={t: {"tier": t} for t in _TIERS})
def smech_inclined_plane(rng, tier: str = "easy"):
    """A block on a rough slope: resolve the weight, then equilibrium, then motion."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam


@generator("smech_inclined_plane_exam", level=17, presets={"exam": {}})
def smech_inclined_plane_exam(rng):
    """Released from rest on a rough slope: normal reaction, greatest friction, acceleration, distance in t seconds."""
    sin, cos = rng.choice(_SLOPES)
    for _ in attempts(2000, "a sliding block with exact answers"):
        mu = rng.choice(_MU)
        m = _mass(rng)
        t = rng.choice((2, 3, 4, 5))
        normal = m * G * cos
        friction = mu * normal
        a = (m * G * sin - friction) / m
        dist = a * t * t / 2
        if a > Fraction(1, 10) and all(is_exact(x, 2) for x in (normal, friction, a, dist)):
            break
    # Second route: signed components along the slope (down the slope positive).
    check(m * a == m * G * sin - friction, "F = ma along the slope disagrees")
    check(dist * 2 == a * t * t and a < G * sin, "distance or acceleration out of range")
    thing = rng.choice(_THINGS)
    stem = (
        f"A {thing} of mass {m} kg is released from rest on a rough plane inclined at an angle θ to the horizontal, "
        f"where sin θ = {sin.numerator}/{sin.denominator}. The coefficient of friction between the {thing} and the "
        f"plane is {dec(mu)}. {G_NOTE}"
    )
    part_a = number_part(
        f"Find the normal reaction of the plane on the {thing}, in newtons, correct to 2 decimal places.",
        normal,
        2,
        1,
        [f"R = {m}g cos θ = {m} × {dec(G)} × {dec(cos)} = {dec(normal)} N."],
        [
            (m * G * sin, "mech-sin-cos-swapped"),
            (m * G, "mech-weight-not-resolved"),
            (m * cos, "mech-forgot-g"),
            (m * G * cos * cos, "mech-component-resolved-twice"),
        ],
        sensible=(0, None),
    )
    part_b = number_part(
        "Find the friction force acting on it as it slides, in newtons, correct to 2 decimal places.",
        friction,
        2,
        1,
        [f"The {thing} is sliding, so friction is limiting: F = μR = {dec(mu)} × {dec(normal)} = {dec(friction)} N."],
        [
            (mu * m * G * sin, "mech-sin-cos-swapped"),
            (mu * m * G, "mech-weight-not-resolved"),
            (normal, "mech-friction-without-coefficient"),
            (m * G * sin, "mech-limiting-friction-assumed"),
        ],
        follow=f"{dec(mu)} * part_a",
        sensible=(0, None),
    )
    part_c = number_part(
        "Find the acceleration of the " + thing + " down the plane, in m/s², correct to 2 decimal places.",
        a,
        2,
        3,
        [
            f"Down the slope: {m}g sin θ - F = {m}a.",
            f"a = ({dec(m * G * sin)} - {dec(friction)}) / {m} = {dec(a)} m/s².",
        ],
        [
            (G * sin, "mech-ignored-friction"),
            ((m * G * sin + friction) / m, "mech-friction-direction-reversed"),
            (m * G * sin - friction, "mech-forgot-divide-by-mass"),
            ((m * G * cos - friction) / m, "mech-sin-cos-swapped"),
        ],
        follow=f"({dec(m * G * sin)} - part_b) / {m}",
        sensible=(0, None),
    )
    part_d = number_part(
        f"Find the distance the {thing} slides in the first {t} seconds, in metres, correct to 2 decimal places.",
        dist,
        2,
        2,
        [f"s = ½at² = ½ × {dec(a)} × {t}² = {dec(dist)} m."],
        [
            (a * t * t, "mech-forgot-half-in-kinematics"),
            (a * t / 2, "mech-forgot-square-on-time"),
            (a * t, "mech-speed-used-as-distance"),
            (G * sin * t * t / 2, "mech-ignored-friction"),
        ],
        follow=f"0.5 * part_c * {t}**2",
        sensible=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
