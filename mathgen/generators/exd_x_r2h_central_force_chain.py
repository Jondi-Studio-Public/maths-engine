"""Exam question on conserved quantities for a particle in a central potential, used to say where it can go.

The stem gives a particle of mass m in the plane under V = c (x^2 + y^2), so V depends on the position only through
r^2 = x^2 + y^2: rotating about the origin is a symmetry, and Noether's theorem makes the angular momentum L_z
conserved; there is no explicit time dependence, so the energy is conserved too. The particle starts at (x0, y0)
with velocity (vx0, vy0). Four parts:

(a) L_z = m (x0 vy0 - y0 vx0)                                             number (whole), 2 marks
(b) E = (m/2)(vx0^2 + vy0^2) + c (x0^2 + y0^2)                            fraction, 2 marks
(c) the speed perpendicular to the radius at distance r1, L_z / (m r1)     fraction, 1 mark, follow_through part_a / (m r1)
(d) the radial kinetic energy there, E - c r1^2 - (m/2) v_perp^2           fraction, 2 marks, follow_through on part_b and part_c

The draw keeps (d) positive, so the particle can reach distance r1. Every key is also found by working in polar
coordinates: the radial kinetic energy is E - c r1^2 - L_z^2 / (2 m r1^2).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, paren_negative, phrase
from .exd_x_r2h_support import build_part

_MASSES = (1, 2, 3, 4)
_COEFFS = (1, 2, 3, 4, 5, 6)
_STEM = (
    "A particle of mass {m} kg moves in the xy-plane with Lagrangian L = (m/2)(x'^2 + y'^2) - V(x, y), where "
    "V(x, y) = {c}(x^2 + y^2) joules, with x and y in metres. At t = 0 it is at ({x0}, {y0}) with velocity "
    "({vx}, {vy}) m/s. V depends on x and y only through x^2 + y^2 and L has no explicit time dependence, so "
    "Noether's theorem gives two conserved quantities: the angular momentum L_z about the origin and the energy E.",
    "A particle of mass {m} kg obeys L = (m/2)(x'^2 + y'^2) - V(x, y) in the xy-plane, with V(x, y) = "
    "{c}(x^2 + y^2) joules (x and y in metres). It starts at ({x0}, {y0}) moving with velocity ({vx}, {vy}) m/s. "
    "Rotations about the origin leave V unchanged and L has no explicit time dependence, so by Noether's theorem "
    "the angular momentum L_z and the energy E are both conserved.",
)
_PROMPT_A = (
    "Find the initial value of L_z, in kg m^2/s.",
    "Calculate L_z for the particle at t = 0, in kg m^2/s.",
)
_PROMPT_B = (
    "Find the energy E of the particle, in joules. Give your answer as a fraction if it is not a whole number.",
    "Find the conserved energy E in joules, as a fraction if it is not a whole number.",
)
_PROMPT_C = (
    "Later the particle is {r} m from the origin. Using your L_z, find the component of its velocity perpendicular "
    "to the radius there (positive anticlockwise), in m/s, as a fraction if it is not a whole number.",
    "At a later time the particle is {r} m from the origin. Use your L_z to find the velocity component perpendicular to "
    "the radius at that moment (positive anticlockwise), in m/s, as a fraction if it is not a whole number.",
)
_PROMPT_D = (
    "At that moment, find the kinetic energy of the radial motion, in joules, as a fraction if it is not a whole "
    "number.",
    "Using your E and your perpendicular speed, find the radial kinetic energy at that moment, in joules, as a "
    "fraction if it is not a whole number.",
)


def _wrongs(m, c, x0, y0, vx, vy, r1, lz, energy, v_perp):
    """Tagged wrong values for parts (a) to (d); wrong energies must be positive."""
    kinetic0 = Fraction(m * (vx * vx + vy * vy), 2)
    a_wrong = [
        (m * (x0 * vy + y0 * vx), "sign-error-in-cross-product"),
        (m * (x0 * vx + y0 * vy), "dot-product-for-cross-product"),
        (x0 * vy - y0 * vx, "forgets-mass-in-angular-momentum"),
        (m * (y0 * vx - x0 * vy), "cross-product-reversed"),
        (m * x0 * vy, "drops-one-term-of-cross-product"),
    ]
    b_wrong = [
        (kinetic0, "kinetic-energy-only"),
        (Fraction(m * (vx * vx + vy * vy)) + c * (x0 * x0 + y0 * y0), "forgets-half-in-kinetic-energy"),
        (kinetic0 + 2 * c * (x0 * x0 + y0 * y0), "doubles-potential-energy"),
        (Fraction(c * (x0 * x0 + y0 * y0)), "potential-energy-only"),
    ]
    c_wrong = [
        (Fraction(lz, m * r1 * r1), "angular-speed-for-tangential-speed"),
        (Fraction(lz, r1), "forgets-mass-in-angular-momentum"),
        (Fraction(lz * r1, m), "multiplies-instead-of-dividing"),
        (Fraction(lz, m), "ignores-the-radius"),
    ]
    d_wrong = [
        (energy - c * r1 * r1, "ignores-tangential-kinetic-energy"),
        (energy - Fraction(m, 2) * v_perp**2, "ignores-potential-at-new-radius"),
        (energy - c * r1 * r1 - m * v_perp**2, "forgets-half-in-kinetic-energy"),
        (energy - c * r1 * r1 + Fraction(m, 2) * v_perp**2, "adds-tangential-energy"),
    ]
    b_wrong = [(v, t) for v, t in b_wrong if v > 0]
    d_wrong = [(v, t) for v, t in d_wrong if v > 0]
    return a_wrong, b_wrong, c_wrong, d_wrong


def _enough(key, wrong) -> bool:
    return len({Fraction(v) for v, _ in wrong if v != key}) >= 3


def _build(rng: random.Random) -> dict:
    for _ in attempts(2000, "a start and a later radius with positive radial energy"):
        m, c = rng.choice(_MASSES), rng.choice(_COEFFS)
        x0, y0 = rng.randint(-4, 4), rng.randint(-4, 4)
        vx, vy = rng.randint(-4, 4), rng.randint(-4, 4)
        r1 = rng.randint(2, 8)
        lz = m * (x0 * vy - y0 * vx)
        if lz == 0 or (x0, y0) == (0, 0) or (vx, vy) == (0, 0):
            continue
        energy = Fraction(m * (vx * vx + vy * vy), 2) + c * (x0 * x0 + y0 * y0)
        v_perp = Fraction(lz, m * r1)
        k_radial = energy - c * r1 * r1 - Fraction(m, 2) * v_perp**2
        if k_radial <= 0 or v_perp == 0:
            continue
        a_wrong, b_wrong, c_wrong, d_wrong = _wrongs(m, c, x0, y0, vx, vy, r1, lz, energy, v_perp)
        if all(_enough(k, w) for k, w in ((lz, a_wrong), (energy, b_wrong), (v_perp, c_wrong), (k_radial, d_wrong))):
            break
    # Second route, in polar form: E - c r^2 - L^2 / (2 m r^2).
    check(
        k_radial == energy - c * r1 * r1 - Fraction(lz * lz, 2 * m * r1 * r1),
        "radial energy disagrees with the polar form",
    )
    check(abs(v_perp * r1 * m) == abs(lz), "perpendicular speed does not rebuild L_z")

    cc = "" if c == 1 else c  # a coefficient of 1 is not written
    stem = phrase(rng, _STEM, m=m, c=c, x0=x0, y0=y0, vx=vx, vy=vy)
    parts = [
        build_part(
            phrase(rng, _PROMPT_A),
            Fraction(lz),
            kind="number",
            marks=2,
            working=[
                "L_z = m (x vy - y vx) is conserved because V is unchanged by rotation about the origin.",
                f"L_z = {m} × ({paren_negative(x0)} × {paren_negative(vy)} - {paren_negative(y0)} × {paren_negative(vx)}) = {lz}.",
            ],
            wrong=a_wrong,
        ),
        build_part(
            phrase(rng, _PROMPT_B),
            energy,
            kind="fraction",
            marks=2,
            working=[
                "E = kinetic + potential energy is conserved because L has no explicit time dependence.",
                f"E = ({m}/2)({paren_negative(vx)}^2 + {paren_negative(vy)}^2) + {cc}({paren_negative(x0)}^2 + {paren_negative(y0)}^2) = {energy}.",
            ],
            wrong=b_wrong,
            sensible_range=(0, None),
        ),
        build_part(
            phrase(rng, _PROMPT_C, r=r1),
            v_perp,
            kind="fraction",
            marks=1,
            working=[f"L_z = m r v_perp, so v_perp = {lz} / ({m} × {r1}) = {v_perp}."],
            wrong=c_wrong,
            follow_through=f"part_a/({m}*{r1})",
        ),
        build_part(
            phrase(rng, _PROMPT_D),
            k_radial,
            kind="fraction",
            marks=2,
            working=[
                f"E = V + (m/2) v_perp^2 + radial kinetic energy, with V = {c * r1 * r1}.",
                f"Radial kinetic energy = {energy} - {c * r1 * r1} - ({m}/2)({v_perp})^2 = {k_radial}.",
            ],
            wrong=d_wrong,
            follow_through=f"part_b - {c * r1 * r1} - {m}*part_c**2/2",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r2h_central_force_chain", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r2h_central_force_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Conserved L_z and E for a particle in a quadratic central potential, then its motion at a later radius."""
    check(tier == "exam", "exd_x_r2h_central_force_chain: only the exam tier exists")
    return _build(rng)
