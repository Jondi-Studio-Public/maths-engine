"""The Cartesian path of a projectile, y = x tan(a) - g x² / (2V² cos²(a)): `r3d_projectile_cartesian` (easy, medium,
hard) and `r3d_projectile_cartesian_exam`.

Skill: projectile-cartesian-elimination. Eliminating t from x = V cos(a) t and y = h + V sin(a) t - g t²/2 gives the path
as a quadratic in x. The skill works with that equation directly. (The vector-component treatment of a projectile
is the neighbouring `senior-apply-vector-calculus-to-motion-in-a-plane`; this skill does not repeat it.)

Tiers differ in the decision they ask for, not in the size of the numbers:

* easy: use the path to find the height at a stated horizontal distance, or the coefficient Q in y = Px - Qx².
* medium: read the range and the greatest height from the path (and the quadratic formula for a stated height
  or the vertex), and find the launch angle from a stated range on level ground.
* hard: aim. The launch angle (lower or higher) that makes a ball of known speed pass through a stated point needs
  the path as a quadratic in tan(a) (using sec² = 1 + tan²); and the speed at 45° needed to pass through a point.
* exam: launched from a height at 45°. (a) the path y = h + x - g x²/V² as an expression, (b) the greatest height,
  (c) the horizontal range to the ground, (d) the angle below the horizontal at which it lands.

## Construction

g = 9.8 m/s² and air resistance is ignored; every prompt says so. Angles are 30°, 45°, 60° or an angle whose tangent is
3/4, 4/3, 5/12 or 12/5, so that trigonometric values are exact. Every answer is recomputed by following the particle
through time, solving the horizontal and vertical equations separately (a different route from the path equation).
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms
from ._r3d_kit import G, G_NOTE, X, clear_of_boundary, dec, exact, expression_part, fit_dp, number_fields, number_part

F = Fraction
_g = sympy.Rational(98, 10)
_R3 = sympy.sqrt(3)

#: (label used in the stem, tan, cos², sin2a, sin²a) as exact SymPy values
_ANGLES = (
    ("30°", _R3 / 3, sympy.Rational(3, 4), _R3 / 2, sympy.Rational(1, 4)),
    ("45°", sympy.Integer(1), sympy.Rational(1, 2), sympy.Integer(1), sympy.Rational(1, 2)),
    ("60°", _R3, sympy.Rational(1, 4), _R3 / 2, sympy.Rational(3, 4)),
    (
        "an angle whose tangent is 3/4",
        sympy.Rational(3, 4),
        sympy.Rational(16, 25),
        sympy.Rational(24, 25),
        sympy.Rational(9, 25),
    ),
    (
        "an angle whose tangent is 4/3",
        sympy.Rational(4, 3),
        sympy.Rational(9, 25),
        sympy.Rational(24, 25),
        sympy.Rational(16, 25),
    ),
    (
        "an angle whose tangent is 5/12",
        sympy.Rational(5, 12),
        sympy.Rational(144, 169),
        sympy.Rational(120, 169),
        sympy.Rational(25, 169),
    ),
    (
        "an angle whose tangent is 12/5",
        sympy.Rational(12, 5),
        sympy.Rational(25, 169),
        sympy.Rational(120, 169),
        sympy.Rational(144, 169),
    ),
)
_SPEEDS = (14, 16, 18, 20, 24, 25, 28, 30)
_SPEEDS_NICE = (7, 14, 21, 28, 35, 42)  # V² / 9.8 is a whole number

_LAUNCH = (
    "A footballer kicks a ball from level ground with speed {V} m/s at {angle} above the horizontal.",
    "A cricket ball is thrown from ground level with speed {V} m/s at {angle} above the horizontal.",
)


def _R(v) -> sympy.Rational:
    v = F(v)
    return sympy.Rational(v.numerator, v.denominator)


def _a2(V, cos2):
    """g / (2 V² cos²a): the coefficient of x² in the path."""
    return _g / (2 * sympy.Integer(V) ** 2 * cos2)


def _fly(V, tan, cos2, x_target):
    """Second route: follow the ball through time. Returns (t, y) when its horizontal position is `x_target`."""
    vx = float(V) * math.sqrt(float(cos2))
    vy = vx * float(tan)
    t = x_target / vx
    return t, vy * t - 4.9 * t * t


def _launch(rng, V, angle):
    return _LAUNCH[rng.randrange(len(_LAUNCH))].format(V=V, angle=angle)


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    variant = rng.choice(("height_at", "height_at", "coefficient"))
    for _ in attempts(1000, "a clear value"):
        label, tan, cos2, sin2a, sin2 = rng.choice(_ANGLES)
        V = rng.choice(_SPEEDS)
        a2 = _a2(V, cos2)
        rng_range = V**2 * sin2a / _g
        if variant == "height_at":
            x = rng.choice((4, 5, 6, 8, 10, 12, 15, 16, 20, 25))
            if x >= rng_range * sympy.Rational(9, 10):
                continue
            value = exact(sympy.N(x * tan - a2 * x**2, 40))
            ok = value > F(1, 2) and clear_of_boundary(value, 2)
            dp = 2
        else:
            value = exact(sympy.N(a2, 40))
            dp = 4
            ok = value > 0 and clear_of_boundary(value, 4, F(1, 10))
        if ok:
            break
    launch = _launch(rng, V, label)
    if variant == "height_at":
        t, y = _fly(V, tan, cos2, x)
        check(abs(y - float(value)) < 5e-9, "height differs from the time route")
        text = (
            f"{launch} Ignore air resistance. {G_NOTE} Find the height of the ball above the ground when it has "
            f"travelled {x} m horizontally, in metres, correct to 2 decimal places."
        )
        T = exact(sympy.N(tan, 40))
        wrong = [
            (exact(sympy.N(x * tan - _g * x**2 / (2 * sympy.Integer(V) ** 2), 40)), "r3d-cos-squared-dropped"),
            (exact(sympy.N(x * tan - _g * x**2 / (sympy.Integer(V) ** 2 * cos2), 40)), "r3d-factor-two-dropped"),
            (exact(sympy.N(x * tan + a2 * x**2, 40)), "r3d-sign-of-gravity-term"),
            (exact(sympy.N(x * tan - _g * x**2 / (2 * sympy.Integer(V) * cos2), 40)), "r3d-speed-not-squared"),
            (exact(sympy.N(x * tan, 40)), "r3d-gravity-term-dropped"),
        ]
        working = [
            "Eliminating t from x = V cos(a) t and y = V sin(a) t - 4.9t² gives y = x tan(a) - g x²/(2V² cos²(a)).",
            f"With V = {V} and the given angle, y = {float(T):.4f}x - {float(exact(sympy.N(a2, 40))):.5f}x²; at x = {x}, y = {float(value):.2f} m.",
        ]
    else:
        text = (
            f"{launch} Ignore air resistance. {G_NOTE} The path of the projectile is y = Px - Qx² (x and y in metres, "
            f"x horizontal and y vertical). Find Q, correct to 4 decimal places."
        )
        wrong = [
            (exact(sympy.N(_g / (2 * sympy.Integer(V) ** 2), 40)), "r3d-cos-squared-dropped"),
            (exact(sympy.N(_g / (sympy.Integer(V) ** 2 * cos2), 40)), "r3d-factor-two-dropped"),
            (exact(sympy.N(_g / (2 * sympy.Integer(V) ** 2 * sympy.sqrt(cos2)), 40)), "r3d-cos-not-squared"),
            (exact(sympy.N(_g / (2 * sympy.Integer(V) * cos2), 40)), "r3d-speed-not-squared"),
            (exact(sympy.N(_g * sympy.Integer(V) ** 2 / (2 * cos2), 40)), "r3d-speed-in-numerator"),
        ]
        working = [
            "Eliminating t from x = V cos(a) t and y = V sin(a) t - 4.9t² gives y = x tan(a) - g x²/(2V² cos²(a)).",
            f"Q = 9.8/(2 × {V}² × cos²(a)) = {float(value):.4f}.",
        ]
    return dict(text=text, working=working, **number_fields(value, dp, wrong, nonneg=True))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    variant = rng.choice(("range", "greatest_height", "return_height", "angle_for_range"))
    if variant == "angle_for_range":
        for _ in attempts(1000, "a clear angle"):
            V = rng.choice(_SPEEDS_NICE)
            unit = F(V * V) / G  # V² / g in metres
            ratio = rng.choice((F(1, 2), F(3, 5), F(4, 5), F(9, 10), F(3, 4), F(7, 10)))
            R = ratio * unit
            if R.denominator != 1 and (R * 10).denominator != 1:
                continue
            angle = sympy.asin(_R(ratio)) / 2 * 180 / sympy.pi
            value = exact(sympy.N(angle, 40))
            if clear_of_boundary(value, 1):
                break
        # second route: launch at this angle and find where the ball lands
        th = math.radians(float(value))
        flight = 2 * V * math.sin(th) / 9.8
        check(abs(V * math.cos(th) * flight - float(R)) < 0.05, "range differs from the time route")
        text = (
            f"A ball is kicked from level ground with speed {V} m/s and lands {dec(R, 2)} m away on the same level. "
            f"Ignore air resistance. {G_NOTE} Find the smaller of the two possible angles of projection above the "
            f"horizontal, in degrees, correct to 1 decimal place."
        )
        s2 = float(ratio)
        wrong = [
            (exact(sympy.N(sympy.asin(_R(ratio)) * 180 / sympy.pi, 40)), "r3d-double-angle-not-halved"),
            (90 - value, "r3d-larger-angle-given"),
            (exact(sympy.N(sympy.asin(_R(ratio) / 2) * 180 / sympy.pi, 40)), "r3d-double-angle-misused"),
            (exact(sympy.N(sympy.atan(_R(ratio)) * 180 / sympy.pi, 40)), "r3d-tan-for-sin"),
        ]
        working = [
            f"On level ground R = V² sin(2a)/g, so sin(2a) = gR/V² = 9.8 × {dec(R, 2)}/{V}² = {dec(ratio, 3)}.",
            f"2a = {math.degrees(math.asin(s2)):.2f}° (or its supplement), so the smaller angle is a = {float(value):.1f}°.",
        ]
        return dict(text=text, working=working, **number_fields(value, 1, wrong, nonneg=True))
    for _ in attempts(2000, "a clear value"):
        label, tan, cos2, sin2a, sin2 = rng.choice(_ANGLES)
        V = rng.choice(_SPEEDS)
        a2 = _a2(V, cos2)
        if variant == "range":
            value = exact(sympy.N(V**2 * sin2a / _g, 40))
        elif variant == "greatest_height":
            value = exact(sympy.N(V**2 * sin2 / (2 * _g), 40))
        else:
            R0 = V**2 * sin2a / _g
            H0 = V**2 * sin2 / (2 * _g)
            h = rng.choice((F(1, 2), 1, F(3, 2), 2, F(5, 2), 3, 4, 5))
            if h >= H0 * sympy.Rational(8, 10):
                continue
            disc = tan**2 - 4 * a2 * _R(h)
            value = exact(sympy.N((tan + sympy.sqrt(disc)) / (2 * a2), 40))
        if value > F(1, 2) and clear_of_boundary(value, 2):
            break
    launch = _launch(rng, V, label)
    vx, vy = V * float(sympy.sqrt(cos2)), V * float(sympy.sqrt(cos2)) * float(tan)
    if variant == "range":
        flight = 2 * vy / 9.8
        check(abs(vx * flight - float(value)) < 5e-9, "range differs from the time route")
        text = f"{launch} Ignore air resistance. {G_NOTE} Find the horizontal range of the ball on level ground, in metres, correct to 2 decimal places."
        wrong = [
            (exact(sympy.N(V**2 * sin2a / (2 * _g), 40)), "r3d-range-halved"),
            (exact(sympy.N(V**2 * sympy.sqrt(sin2) / _g, 40)), "r3d-single-angle-sine"),
            (exact(sympy.N(V**2 * sin2 / _g, 40)), "r3d-sin-squared-for-double-angle"),
            (exact(sympy.N(V * sympy.sqrt(sin2a) / _g, 40)), "r3d-speed-not-squared"),
        ]
        working = [
            "The ball is on the ground again when y = 0: x tan(a) - g x²/(2V² cos²(a)) = 0, so x = 2V² sin(a) cos(a)/g.",
            f"R = V² sin(2a)/g = {V}² sin(2a)/9.8 = {float(value):.2f} m.",
        ]
    elif variant == "greatest_height":
        top = vy * vy / (2 * 9.8)
        check(abs(top - float(value)) < 5e-9, "height differs from the time route")
        text = f"{launch} Ignore air resistance. {G_NOTE} Find the greatest height the ball reaches, in metres, correct to 2 decimal places."
        wrong = [
            (exact(sympy.N(V**2 * sin2 / _g, 40)), "r3d-factor-two-dropped"),
            (exact(sympy.N(V**2 * sin2a / (2 * _g), 40)), "r3d-double-angle-for-square"),
            (exact(sympy.N(V**2 * sympy.sqrt(sin2) / (2 * _g), 40)), "r3d-sine-not-squared"),
            (exact(sympy.N(V**2 * (1 - sin2) / (2 * _g), 40)), "r3d-cosine-for-sine"),
        ]
        working = [
            "The vertex of y = x tan(a) - Qx² is at x = tan(a)/(2Q), where y = tan²(a)/(4Q).",
            f"With Q = g/(2V² cos²(a)): H = V² sin²(a)/(2g) = {float(value):.2f} m.",
        ]
    else:
        t_flight, y_check = _fly(V, tan, cos2, float(value))
        check(abs(y_check - float(h)) < 5e-8, "return height differs from the time route")
        text = (
            f"{launch} Ignore air resistance. {G_NOTE} On its way down the ball is again at a height of "
            f"{dec(h, 2)} m above the ground. Find its horizontal distance from the launch point at that moment, in metres, "
            f"correct to 2 decimal places."
        )
        small = exact(sympy.N((tan - sympy.sqrt(disc)) / (2 * a2), 40))
        wrong = [
            (small, "r3d-smaller-root-taken"),
            (exact(sympy.N(tan / (2 * a2), 40)), "r3d-vertex-for-root"),
            (
                exact(sympy.N((tan + sympy.sqrt(tan**2 + 4 * a2 * _R(h))) / (2 * a2), 40)),
                "r3d-height-sign-in-discriminant",
            ),
            (exact(sympy.N(tan / a2, 40)), "r3d-range-for-return-height"),
        ]
        working = [
            f"Set y = {dec(h, 2)}: Qx² - x tan(a) + {dec(h, 2)} = 0 with Q = {float(exact(sympy.N(a2, 40))):.5f} and tan(a) = {float(exact(sympy.N(tan, 40))):.4f}.",
            f"The ball is on the way down at the larger root, x = {float(value):.2f} m.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong, nonneg=True))


# ------------------------------------------------------------------------------------------------ hard


def _hard(rng):
    variant = rng.choice(("lower_angle", "higher_angle", "speed_at_45"))
    if variant == "speed_at_45":
        for _ in attempts(2000, "a clear speed"):
            x = rng.choice((10, 15, 20, 25, 30, 35, 40))
            y = F(rng.choice((10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80)), 10)
            if y >= x * F(7, 10):
                continue
            V2 = _g * x**2 / (x - _R(y))
            value = exact(sympy.N(sympy.sqrt(V2), 40))
            if clear_of_boundary(value, 2) and value > 5:
                break
        # second route: launch at 45° with this speed and read the height at x
        vv = float(value)
        vx = vv * math.sqrt(0.5)
        t = x / vx
        check(abs(vx * t - x) < 1e-9 and abs(vx * t - 4.9 * t * t - float(y)) < 5e-8, "speed does not reach the point")
        text = (
            f"A ball is kicked from level ground at 45° above the horizontal and just passes over the top of a wall "
            f"{dec(y, 2)} m high that stands {x} m from the kicker, on its way up or down. Ignore air resistance. "
            f"{G_NOTE} Find the speed of the kick, in metres per second, correct to 2 decimal places."
        )
        wrong = [
            (exact(sympy.N(sympy.sqrt(_g * x**2 / (2 * (x - _R(y)))), 40)), "r3d-cos-squared-misused"),
            (exact(sympy.N(sympy.sqrt(_g * x**2 / (x + _R(y))), 40)), "r3d-height-sign"),
            (exact(sympy.N(_g * x**2 / (x - _R(y)), 40)), "r3d-speed-squared-reported"),
            (exact(sympy.N(sympy.sqrt(_g * x / 1), 40)), "r3d-height-ignored"),
        ]
        working = [
            "At 45°, tan(a) = 1 and 2cos²(a) = 1, so y = x - g x²/V².",
            f"Putting x = {x} and y = {dec(y, 2)}: V² = g x²/(x - y) = 9.8 × {x}²/({x} - {dec(y, 2)}), so V = {float(value):.2f} m/s.",
        ]
        return dict(text=text, working=working, **number_fields(value, 2, wrong, nonneg=True))
    for _ in attempts(5000, "a reachable point with clear angles"):
        V = rng.choice((14, 18, 20, 24, 25, 28, 30))
        x = rng.choice((8, 10, 12, 15, 16, 18, 20, 24, 25, 30, 35, 40))
        y = F(rng.choice(range(5, 160, 5)), 10)
        a = _g * x**2 / (2 * sympy.Integer(V) ** 2)
        disc = x**2 - 4 * a * (a + _R(y))
        if disc <= 0 or y >= x:
            continue
        T1 = (x - sympy.sqrt(disc)) / (2 * a)
        T2 = (x + sympy.sqrt(disc)) / (2 * a)
        lo = exact(sympy.N(sympy.atan(T1) * 180 / sympy.pi, 40))
        hi = exact(sympy.N(sympy.atan(T2) * 180 / sympy.pi, 40))
        if hi > 80 or not (clear_of_boundary(lo, 1) and clear_of_boundary(hi, 1)) or abs(hi - lo) < 5:
            continue
        break
    pick_low = variant == "lower_angle"
    value = lo if pick_low else hi
    other = hi if pick_low else lo
    for ang, name in ((value, "chosen"), (other, "other")):
        th = math.radians(float(ang))
        _, yy = _fly(V, math.tan(th), math.cos(th) ** 2, x)
        check(abs(yy - float(y)) < 5e-3, f"the {name} angle does not reach the point")
    which = "smaller" if pick_low else "larger"
    text = (
        f"A ball is kicked from level ground with speed {V} m/s and passes through a point {x} m horizontally "
        f"from the kicker and {dec(y, 2)} m above the ground. Ignore air resistance. {G_NOTE} There are two possible angles "
        f"of projection above the horizontal. Find the {which} one, in degrees, correct to 1 decimal place."
    )
    noncos = (
        exact(sympy.N(sympy.atan((x - sympy.sqrt(x**2 - 4 * a * _R(y))) / (2 * a)) * 180 / sympy.pi, 40))
        if x**2 - 4 * a * _R(y) > 0
        else None
    )
    wrong = [
        (other, "r3d-other-angle-given"),
        (90 - value, "r3d-complementary-angle"),
        (exact(sympy.N(sympy.atan(_R(y) / x) * 180 / sympy.pi, 40)), "r3d-line-of-sight-angle"),
    ]
    if noncos is not None and pick_low:
        wrong.append((noncos, "r3d-sec-squared-dropped"))
    wrong.append(
        (
            exact(sympy.N(sympy.asin(min(1, float(T1 if pick_low else T2))) * 180 / sympy.pi, 40))
            if abs(float(T1 if pick_low else T2)) <= 1
            else other + 3,
            "r3d-sin-for-tan",
        )
    )
    working = [
        f"With T = tan(a), y = xT - (g x²/(2V²))(1 + T²). Here g x²/(2V²) = {float(exact(sympy.N(a, 40))):.4f}, giving {float(exact(sympy.N(a, 40))):.4f}T² - {x}T + {float(exact(sympy.N(a + _R(y), 40))):.4f} = 0.",
        f"T = {float(exact(sympy.N(T1, 40))):.4f} or {float(exact(sympy.N(T2, 40))):.4f}, so the angles are {float(lo):.1f}° and {float(hi):.1f}°. The {which} is {float(value):.1f}°.",
    ]
    return dict(text=text, working=working, **number_fields(value, 1, wrong, nonneg=True))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("r3d_projectile_cartesian", level=18, presets=_PRESETS)
def r3d_projectile_cartesian(rng, tier: str = "easy"):
    """The path y = x tan(a) - g x²/(2V² cos²(a)): heights, range, greatest height, launch angle, speed to reach a point."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam

_EXAM_STEMS = (
    "A ball is thrown from the top of a cliff {h} m above level ground. It leaves the thrower's hand at {V} m/s at 45° "
    "above the horizontal and moves under gravity only. Take x as the horizontal distance from the cliff face and y as "
    "the height above the ground, both in metres. " + G_NOTE,
    "A stone is thrown from a platform {h} m above the ground with a speed of {V} m/s at 45° above the horizontal. "
    "Ignore air resistance. Let x metres be the horizontal distance from the platform and y metres the height above the "
    "ground. " + G_NOTE,
    "A cricketer throws a ball from a viewing platform {h} m above the ground. It leaves her hand at {V} m/s at 45° "
    "above the horizontal, and only gravity acts on it. Let x be the horizontal distance from the platform and y "
    "the height above the ground, both in metres. " + G_NOTE,
)


@generator("r3d_projectile_cartesian_exam", level=18, presets={"exam": {}})
def r3d_projectile_cartesian_exam(rng):
    """Thrown at 45° from a height: the path as an expression, the greatest height, the range, the angle of landing."""
    for _ in attempts(1000, "a clear chain"):
        V = rng.choice((14, 21, 28, 35))
        h = rng.choice((4, 5, 6, 8, 10, 12, 15, 18, 20, 25))
        a2 = _g / sympy.Integer(V) ** 2  # coefficient of x² (45°: 2cos² = 1)
        xv = 1 / (2 * a2)  # x at the vertex
        top = h + 1 / (4 * a2)
        R = (1 + sympy.sqrt(1 + 4 * a2 * h)) / (2 * a2)
        land = sympy.atan(sympy.Abs(1 - 2 * a2 * R)) * 180 / sympy.pi
        top_v, R_v, land_v = exact(sympy.N(top, 40)), exact(sympy.N(R, 40)), exact(sympy.N(land, 40))
        if clear_of_boundary(top_v, 2) and clear_of_boundary(R_v, 2) and clear_of_boundary(land_v, 1):
            break
    # second route: follow the ball through time to the ground
    vx = V / math.sqrt(2)
    t_land = (vx + math.sqrt(vx * vx + 2 * 9.8 * h)) / 9.8
    check(abs(vx * t_land - float(R_v)) < 5e-8, "range differs from the time route")
    vy_land = vx - 9.8 * t_land
    check(
        abs(math.degrees(math.atan(abs(vy_land) / vx)) - float(land_v)) < 5e-7,
        "landing angle differs from the time route",
    )
    check(
        abs(h + vx * (vx / 9.8) - 0.5 * 9.8 * (vx / 9.8) ** 2 - float(top_v)) < 5e-8,
        "greatest height differs from the time route",
    )
    a2f = F(98, 10) / (V * V)
    a2_den = 1 / a2f  # a2 = 1/a2_den
    stem = _EXAM_STEMS[rng.randrange(len(_EXAM_STEMS))].format(h=h, V=V)
    inv = int(a2_den) if a2_den.denominator == 1 else None
    check(inv is not None, "x² coefficient is not a unit fraction")
    path = sympy.Integer(h) + X - X**2 / inv
    path_text = f"{h} + x - x²/{inv}"
    wrong_a = [
        (h + X - 2 * X**2 / inv, f"{h} + x - 2x²/{inv}", "r3d-factor-two-misplaced"),
        (h + X - X**2 / (2 * inv), f"{h} + x - x²/{2 * inv}", "r3d-cos-squared-dropped"),
        (X - X**2 / inv, f"x - x²/{inv}", "r3d-launch-height-dropped"),
        (h + X + X**2 / inv, f"{h} + x + x²/{inv}", "r3d-sign-of-gravity-term"),
    ]
    part_a = expression_part(
        "Show that the path of the object is a parabola and find y in terms of x.",
        path,
        path_text,
        3,
        [
            f"With tan 45° = 1 and 2cos²45° = 1, y = {h} + x tan 45° - g x²/(2V² cos² 45°) = {h} + x - 9.8x²/{V}².",
            f"9.8/{V}² = 1/{inv}, so y = {path_text}.",
        ],
        wrong_a,
        X,
    )
    xv_f = exact(sympy.N(xv, 40))
    part_b = number_part(
        "Find the greatest height of the object above the ground, in metres, correct to 2 decimal places.",
        top_v,
        2,
        2,
        [f"The vertex of y = {path_text} is at x = {inv}/2 = {dec(xv_f, 3)}.", f"y = {float(top_v):.2f} m."],
        [
            (F(h), "r3d-launch-height-for-maximum"),
            (exact(sympy.N(sympy.Integer(V) ** 2 / (4 * _g), 40)), "r3d-launch-height-dropped"),
            (exact(sympy.N(top + sympy.Integer(V) ** 2 / (4 * _g), 40)), "r3d-height-counted-twice"),
            (exact(sympy.N(h + V**2 / (2 * _g), 40)), "r3d-factor-two-dropped"),
        ],
        follow=f"Subs(part_a, (x,), ({dec(xv_f, 3)},))",
        sensible=(0, None),
        nonneg=True,
    )
    part_c = number_part(
        "Find the horizontal distance from the starting point at which the object reaches the ground, in metres, correct to 2 decimal places.",
        R_v,
        2,
        2,
        [
            f"Set y = 0: x²/{inv} - x - {h} = 0, so x = [1 + √(1 + {4 * h}/{inv})]/(2/{inv}).",
            f"x = {float(R_v):.2f} m.",
        ],
        [
            (exact(sympy.N(sympy.sqrt(1 + 4 * a2 * h) / (2 * a2), 40)), "r3d-quadratic-formula-slip"),
            (exact(sympy.N(1 / a2, 40)), "r3d-level-ground-range"),
            (exact(sympy.N((1 + sympy.sqrt(1 + 4 * a2 * h)) / a2, 40)), "r3d-denominator-without-two"),
            (exact(sympy.N(sympy.sqrt(sympy.Integer(h) / a2), 40)), "r3d-linear-term-ignored"),
        ],
        sensible=(0, None),
        nonneg=True,
    )
    part_d = number_part(
        "Find the angle below the horizontal at which the object lands, in degrees, correct to 1 decimal place.",
        land_v,
        1,
        2,
        [
            "The gradient of the path is dy/dx = 1 - 2x/" + str(inv) + ".",
            f"At x = {float(R_v):.2f} the gradient is {float(exact(sympy.N(1 - 2 * a2 * R, 40))):.3f}, so the angle is {float(land_v):.1f}° below the horizontal.",
        ],
        [
            (exact(sympy.N(sympy.atan(1 / sympy.Abs(1 - 2 * a2 * R)) * 180 / sympy.pi, 40)), "r3d-complementary-angle"),
            (F(45), "r3d-launch-angle-for-landing"),
            (exact(sympy.N(sympy.atan(2 * a2 * R) * 180 / sympy.pi, 40)), "r3d-gradient-term-dropped"),
            (exact(sympy.N(sympy.atan(sympy.Abs(1 - a2 * R)) * 180 / sympy.pi, 40)), "r3d-derivative-slip"),
        ],
        follow=f"atan(abs(1 - 2*part_c/{inv}))*180/pi",
        sensible=(0, 90),
        nonneg=True,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
