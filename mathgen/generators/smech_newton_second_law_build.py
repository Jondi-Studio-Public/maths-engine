"""Newton's second law with a resultant of several forces: `smech_newton_second_law` (easy, medium, hard) and
`smech_newton_second_law_exam` (a four-part exam chain with force vectors).

Skill: newtons-second-law-resultant-force. Where the neighbouring skills ask for a resultant (or its acceleration)
from forces that are all stated, this one runs the law both ways: the forces and the motion are known and a force,
a mass or an acceleration is missing.

Tiers differ in the decision they ask for, not in the size of the numbers:

* easy: forces along one line. Add the forces with their signs, then use F = ma to find the acceleration, the
  unknown resistance or the mass.
* medium: the motion is described in words (speed gained in a time, or a stopping distance). The acceleration has to
  be found from the kinematics before F = ma gives the force.
* hard: the motion is vertical, so the weight (g = 9.8 m/s²) joins the forces and the sign of the acceleration
  decides whether the tension is above or below the weight.
* exam: force vectors in two dimensions. A particle is accelerated by two known forces and one unknown force P;
  find the components of P, then its magnitude and its direction.

## Construction

Every question is built backwards from chosen values, so every quantity is exact until it is shown: the mass and
the acceleration are drawn so that m×a is a whole number of newtons. Each answer is also recomputed by a second
route (complex-number arithmetic for the vector chain, a kinematics identity for the medium tier).
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction

from ..core import attempts, check, draw_until, generator, paren_negative, phrase
from ._smech_kit import G, G_NOTE, clear_of_boundary, dec, is_exact, number_fields, number_part, shown

_THINGS = ("trolley", "sled", "crate", "go-kart", "wheelbarrow", "barrow")
_ACCEL = (Fraction(1, 2), Fraction(4, 5), Fraction(6, 5), Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3))


def _draw_motion(rng, masses=(2, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40)):
    """A mass and an acceleration whose product is a whole number of newtons."""
    return draw_until(
        rng,
        lambda r: (r.choice(masses), r.choice(_ACCEL)),
        lambda d: (d[0] * d[1]).denominator == 1,
        what="a mass and acceleration with a whole-newton product",
    )


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    thing = rng.choice(_THINGS)
    m, a = _draw_motion(rng)
    net = int(m * a)
    variant = rng.choice(("acceleration", "resistance", "mass"))
    for _ in attempts(200, "two genuine pushes that balance the net force"):
        res = rng.randint(6, 40)
        p1 = rng.randint(10, 60)
        p2 = net + res - p1
        if p2 < 5:  # keep the two pushes both genuine
            p1, p2 = net + res - 10, 10
        if p1 > 0:  # a small net force and resistance can leave no room for a positive first push: redraw
            break
    check(p1 + p2 - res == net and p1 > 0 and p2 > 0, "forces do not balance the construction")
    forces = f"two forces in the direction of motion, {p1} N and {p2} N, and a resistance of {res} N against it"
    if variant == "acceleration":
        text = f"A {thing} of mass {m} kg moves along a horizontal track under {forces}. Find its acceleration in m/s², correct to 1 decimal place."
        value, dp = a, 1
        wrong = [
            (Fraction(p1 + p2 - res), "mech-forgot-divide-by-mass"),
            (Fraction(p1 + p2 + res, m), "mech-resistance-added"),
            (Fraction(p1 - res, m), "mech-ignored-one-force"),
            (Fraction(p1 + p2 - res) * m, "mech-multiplied-by-mass"),
        ]
        working = [f"Resultant = {p1} + {p2} - {res} = {net} N.", f"a = {net} / {m} = {dec(a)} m/s²."]
    elif variant == "resistance":
        text = (
            f"A {thing} of mass {m} kg is driven along a horizontal track by two forces of {p1} N and {p2} N "
            f"in the direction of motion. A constant resistance acts against the motion and the {thing} "
            f"accelerates at {dec(a)} m/s². Find the resistance in newtons."
        )
        value, dp = Fraction(res), 0
        wrong = [
            (Fraction(p1 + p2 + net), "mech-resistance-added"),
            (Fraction(net), "mech-resultant-for-resistance"),
            (Fraction(p1 + p2) - a, "mech-forgot-mass-in-resultant"),
            (Fraction(p1 - net), "mech-ignored-one-force"),
        ]
        working = [f"m a = {m} × {dec(a)} = {net} N.", f"{p1} + {p2} - R = {net}, so R = {res} N."]
    else:
        text = (
            f"A {thing} is pulled along a horizontal track by two forces of {p1} N and {p2} N in the direction "
            f"of motion against a resistance of {res} N. Its acceleration is {dec(a)} m/s². Find its mass in kilograms."
        )
        value, dp = Fraction(m), 0
        wrong = [
            (Fraction(p1 + p2) / a, "mech-ignored-resistance"),
            (Fraction(p1 + p2 + res) / a, "mech-resistance-added"),
            (Fraction(net) * a, "mech-multiplied-by-mass"),
            (Fraction(p1 - res) / a, "mech-ignored-one-force"),
        ]
        working = [f"Resultant = {p1} + {p2} - {res} = {net} N.", f"m = {net} / {dec(a)} = {m} kg."]
    check(
        (variant != "acceleration" or Fraction(net, m) == a)
        and (variant != "resistance" or p1 + p2 - m * a == res)
        and (variant != "mass" or Fraction(net) / a == m),
        "second route disagrees",
    )
    return dict(text=text, working=working, **number_fields(value, dp, wrong))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    thing = rng.choice(("car", "van", "truck", "bus"))
    variant = rng.choice(("from_rest", "braking", "distance"))
    if variant == "from_rest":
        m = rng.choice((600, 800, 900, 1000, 1200, 1500))
        a = Fraction(rng.choice((Fraction(1, 2), Fraction(3, 2), 2, Fraction(5, 2), 3)))
        t = rng.choice((4, 5, 8, 10))
        v = a * t
        res = rng.choice((150, 200, 250, 300, 400))
        force = m * a + res
        text = (
            f"A {thing} of mass {m} kg starts from rest on a straight horizontal road and reaches {dec(v)} m/s after "
            f"{t} seconds. The acceleration is constant and a resistance of {res} N acts against the motion "
            f"throughout. Find the driving force in newtons."
        )
        wrong = [
            (Fraction(m * a), "mech-ignored-resistance"),
            (m * a - res, "mech-resistance-added"),
            (m * v + res, "mech-speed-used-as-acceleration"),
            (m * Fraction(v) + res * t, "mech-speed-used-as-acceleration"),
        ]
        check(Fraction(v, t) == a and force - res == m * a, "second route disagrees")
        working = [f"a = {v} / {t} = {dec(a)} m/s².", f"F - {res} = {m} × {dec(a)}, so F = {force} N."]
    elif variant == "braking":
        m = rng.choice((800, 1000, 1200, 1500, 1600, 2000))
        d, a = rng.choice(((25, 8), (40, 5), (50, 4), (80, Fraction(5, 2)), (100, 2)))
        a = Fraction(a)
        u = math.isqrt(int(2 * a * d))
        if u * u != 2 * a * d:
            u = None
        check(u is not None, "stopping distance does not give a whole speed")
        force = m * a
        text = (
            f"A {thing} of mass {m} kg is travelling at {u} m/s when the brakes are applied. It is brought to "
            f"rest after travelling {d} m with a constant braking force. Find the braking force in newtons."
        )
        wrong = [
            (Fraction(m * u * u, d), "mech-forgot-two-in-kinematics"),
            (Fraction(m * u, d), "mech-speed-used-as-acceleration"),
            (Fraction(m * u * u, 2), "mech-kinetic-energy-for-force"),
            (Fraction(m * u, 2 * d), "mech-speed-used-as-acceleration"),
        ]
        check(Fraction(u * u, 2 * d) == a, "second route disagrees")
        working = [f"0 = {u}² + 2a({d}), so a = -{dec(a)} m/s².", f"F = {m} × {dec(a)} = {force} N."]
    else:
        m = rng.choice((20, 25, 40, 50, 60, 80))
        a, t = draw_until(
            rng,
            lambda r: (
                Fraction(r.choice((Fraction(1, 2), Fraction(3, 4), Fraction(5, 4), Fraction(3, 2), 2))),
                r.choice((4, 6, 8, 10)),
            ),
            lambda d: (d[0] * d[1] * d[1] / 2).denominator == 1 and (m * d[0]).denominator == 1,
            what="an acceleration and time that give a whole distance and a whole force",
        )
        s = a * t * t / 2
        res = rng.choice((10, 15, 20, 30))
        force = m * a + res
        thing = rng.choice(("sled", "crate", "trolley"))
        text = (
            f"A {thing} of mass {m} kg is pushed from rest along level ground by a constant horizontal force. "
            f"It travels {s} m in the first {t} seconds against a constant resistance of {res} N. "
            f"Find the pushing force in newtons."
        )
        wrong = [
            (m * Fraction(s) / t + res, "mech-speed-used-as-acceleration"),
            (m * Fraction(s) / (t * t) + res, "mech-forgot-half-in-kinematics"),
            (Fraction(m * a), "mech-ignored-resistance"),
            (m * a - res, "mech-resistance-added"),
        ]
        check(Fraction(2 * s, t * t) == a, "second route disagrees")
        working = [f"{s} = ½ a ({t})², so a = {dec(a)} m/s².", f"F - {res} = {m} × {dec(a)}, so F = {force} N."]
    check(Fraction(force).denominator == 1, "the force is not a whole number of newtons")
    return dict(text=text, working=working, **number_fields(Fraction(force), 0, wrong))


# ------------------------------------------------------------------------------------------------ hard


def _hard(rng):
    variant = rng.choice(("lift_up", "lower_down", "find_accel"))
    m = rng.choice((20, 25, 40, 50, 60, 80, 100, 250, 500))
    a = rng.choice((Fraction(1, 2), Fraction(3, 2), Fraction(1), Fraction(2), Fraction(5, 2)))
    if variant == "lift_up":
        tension = m * (G + a)
        text = (
            f"A crane lifts a load of mass {m} kg vertically with a cable. The load accelerates upward at "
            f"{dec(a)} m/s². {G_NOTE} Find the tension in the cable in newtons, correct to 1 decimal place."
        )
        value, dp = tension, 1
        wrong = [
            (m * (G - a), "mech-weight-and-acceleration-opposed"),
            (m * a, "mech-ignored-weight"),
            (m * G, "mech-ignored-acceleration"),
            (G + a, "mech-forgot-mass-in-resultant"),
        ]
        check(tension - m * G == m * a, "second route disagrees")
        working = [f"T - {m}g = {m} × {dec(a)}.", f"T = {m}({dec(G)} + {dec(a)}) = {dec(tension)} N."]
    elif variant == "lower_down":
        tension = m * (G - a)
        text = (
            f"A lift of mass {m} kg is held by a cable and is accelerating downward at {dec(a)} m/s². {G_NOTE} "
            f"Find the tension in the cable in newtons, correct to 1 decimal place."
        )
        value, dp = tension, 1
        wrong = [
            (m * (G + a), "mech-weight-and-acceleration-opposed"),
            (m * a, "mech-ignored-weight"),
            (m * G, "mech-ignored-acceleration"),
            (G - a, "mech-forgot-mass-in-resultant"),
        ]
        check(m * G - tension == m * a, "second route disagrees")
        working = [f"{m}g - T = {m} × {dec(a)}.", f"T = {m}({dec(G)} - {dec(a)}) = {dec(tension)} N."]
    else:
        tension = m * (G + a)
        text = (
            f"A cable supports a load of mass {m} kg. The tension in the cable is {dec(tension)} N and the load is "
            f"moving vertically. {G_NOTE} Find the upward acceleration of the load in m/s², correct to 1 decimal place."
        )
        value, dp = a, 1
        wrong = [
            (tension / m, "mech-ignored-weight"),
            (G - a, "mech-weight-and-acceleration-opposed"),
            (tension - m * G, "mech-forgot-divide-by-mass"),
            (a - G, "mech-weight-and-acceleration-opposed"),
        ]
        check(tension / m - G == a, "second route disagrees")
        working = [f"T - {m}g = {m}a.", f"a = {dec(tension)}/{m} - {dec(G)} = {dec(a)} m/s²."]
    check(is_exact(value, dp), "answer is not exact at its stated precision")
    return dict(text=text, working=working, **number_fields(value, dp, wrong))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("smech_newton_second_law", level=17, presets=_PRESETS)
def smech_newton_second_law(rng, tier: str = "easy"):
    """F = ma with several forces: one line, then kinematics first, then vertical motion with weight."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam

_EXAM_STEMS = (
    "A particle of mass {m} kg moves on a smooth horizontal surface. Relative to east and north unit vectors, "
    "three forces act on it: F1 = ({f1x}, {f1y}) N, F2 = ({f2x}, {f2y}) N and an unknown force P. "
    "Its acceleration is ({ax}, {ay}) m/s².",
    "Three forces act on a {m} kg puck sliding on smooth ice: F1 = ({f1x}, {f1y}) N, F2 = ({f2x}, {f2y}) N "
    "and a third force P that is not known. The puck's acceleration is ({ax}, {ay}) m/s², where the components "
    "are east and north.",
    "A {m} kg trolley moves on a smooth horizontal floor under three forces, written as (east, north) components: "
    "F1 = ({f1x}, {f1y}) N, F2 = ({f2x}, {f2y}) N and an unknown force P. Its acceleration is "
    "({ax}, {ay}) m/s².",
)


def _angle_clear(deg: float) -> bool:
    return abs((deg % 1) - 0.5) >= 0.1


@generator("smech_newton_second_law_exam", level=17, presets={"exam": {}})
def smech_newton_second_law_exam(rng):
    """Two known force vectors and an unknown force P give a stated acceleration: P's components, size, direction."""
    for _ in attempts(1000, "a force vector with a clear magnitude and direction"):
        m = rng.choice((2, 4, 5, 8, 10))
        ax = rng.choice((-3, -2, Fraction(-3, 2), Fraction(-1, 2), Fraction(1, 2), Fraction(3, 2), 2, 3))
        ay = rng.choice((Fraction(1, 2), Fraction(3, 2), 1, 2, Fraction(5, 2), 3))
        f1x, f1y, f2x, f2y = (rng.randint(-9, 9) for _ in range(4))
        px, py = m * Fraction(ax) - f1x - f2x, m * Fraction(ay) - f1y - f2y
        if px.denominator != 1 or py.denominator != 1:
            continue
        px, py = int(px), int(py)
        if px == 0 or py <= 0 or abs(px) > 40 or abs(py) > 40 or f1x == 0 or f2y == 0:
            continue
        mag = math.hypot(px, py)
        theta = math.degrees(math.atan2(py, px))
        if not (clear_of_boundary(Fraction(mag).limit_denominator(10**9), 1) and _angle_clear(theta)):
            continue
        if min(abs(theta - 45), abs(theta - 135)) < 1:  # the complement and the reflection would equal the answer
            continue
        # Each component part needs two wrong values that differ from the key and from each other.
        slips_a = {shown(v, 0) for v in (ax - f1x - f2x, f1x + f2x - m * ax, m * ax - f1x, m * ax + f1x + f2x)} - {px}
        slips_b = {shown(v, 0) for v in (ay - f1y - f2y, f1y + f2y - m * ay, m * ay - f2y, m * ay + f1y + f2y)} - {py}
        if len(slips_a) < 2 or len(slips_b) < 2:
            continue
        break
    # Second route: the same components from complex arithmetic.
    unknown = m * complex(float(ax), float(ay)) - complex(f1x, f1y) - complex(f2x, f2y)
    check(abs(unknown - complex(px, py)) < 1e-9, "complex route disagrees on P")
    check(
        abs(abs(unknown) - mag) < 1e-9 and abs(math.degrees(cmath.phase(unknown)) - theta) < 1e-9, "magnitude or angle"
    )
    ang = round(theta)
    q_angle = Fraction(math.atan(abs(py / px)) * 180 / math.pi).limit_denominator(10**6)
    a_txt = lambda v: str(v) if Fraction(v).denominator == 1 else str(float(v))  # noqa: E731
    stem = phrase(
        rng,
        _EXAM_STEMS,
        m=m,
        f1x=f1x,
        f1y=f1y,
        f2x=f2x,
        f2y=f2y,
        ax=a_txt(ax),
        ay=a_txt(ay),
    )
    ax_f, ay_f = Fraction(ax), Fraction(ay)
    res_mag = m * math.hypot(float(ax_f), float(ay_f))
    res_angle = round(math.degrees(math.atan2(float(ay_f), float(ax_f))))
    part_a = number_part(
        "Find the east component of P, in newtons.",
        Fraction(px),
        0,
        2,
        [
            f"Resultant force = m a = {m} × {a_txt(ax)} = {a_txt(m * ax_f)} N east.",
            f"P = {a_txt(m * ax_f)} - ({f1x}) - ({f2x}) = {px} N.",
        ],
        [
            (ax_f - f1x - f2x, "mech-forgot-mass-in-resultant"),
            (f1x + f2x - m * ax_f, "mech-resultant-sign-reversed"),
            (m * ax_f - f1x, "mech-ignored-one-force"),
            (m * ax_f + f1x + f2x, "mech-added-known-forces-to-resultant"),
        ],
    )
    part_b = number_part(
        "Find the north component of P, in newtons.",
        Fraction(py),
        0,
        1,
        [f"P = {a_txt(m * ay_f)} - ({f1y}) - ({f2y}) = {py} N."],
        [
            (ay_f - f1y - f2y, "mech-forgot-mass-in-resultant"),
            (f1y + f2y - m * ay_f, "mech-resultant-sign-reversed"),
            (m * ay_f - f2y, "mech-ignored-one-force"),
            (m * ay_f + f1y + f2y, "mech-added-known-forces-to-resultant"),
        ],
    )
    part_c = number_part(
        "Find the magnitude of P, in newtons, correct to 1 decimal place.",
        Fraction(mag).limit_denominator(10**9),
        1,
        2,
        [f"|P| = √({paren_negative(px)}² + {paren_negative(py)}²) = {round(mag, 1)} N."],
        [
            (Fraction(abs(px)), "mech-used-one-component"),
            (Fraction(abs(py)), "mech-used-one-component"),
            (Fraction(abs(px) + abs(py)), "mech-added-components"),
            (Fraction(res_mag).limit_denominator(10**9), "mech-resultant-for-unknown"),
        ],
        follow="sqrt(part_a**2 + part_b**2)",
        sensible=(0, None),
    )
    part_d = number_part(
        "Find the direction of P as an angle between 0° and 180° anticlockwise from east, to the nearest degree.",
        Fraction(ang),
        0,
        2,
        [f"tan θ = {py}/{px}, and P points {'north-west' if px < 0 else 'north-east'}, so θ = {ang}°."],
        [
            (abs(90 - ang), "mech-angle-from-wrong-axis"),
            (q_angle, "mech-quadrant-ignored"),
            (180 - ang, "mech-angle-reflected"),
            (Fraction(res_angle), "mech-resultant-for-unknown"),
        ],
        follow=("180 + " if px < 0 else "") + "atan(part_b / part_a) * 180 / pi",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
