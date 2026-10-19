"""Projectile motion with resistance proportional to velocity: `w6_projectile_resistance` (easy, medium, hard) and
`w6_projectile_resistance_exam`.

Skill: projectile-with-linear-resistance. A particle is projected from level ground at V m/s at an angle α above the
horizontal, and the resistance per unit mass is kv, opposite to the velocity (k per second). With axes horizontal and
vertical up the equations decouple:

    x'' = -k x'                 so  x' = u cos α · e^(-kt)
    y'' = -g - k y'             so  y' = (u sin α + g/k) e^(-kt) - g/k

The stem gives V and tan α for a Pythagorean triple, so both components are whole numbers the learner has to find.
g = 9.8 m/s² and every prompt says so.

* easy: no integration. The horizontal deceleration at launch, the vertical deceleration at launch (weight and
  resistance both act downward), the downward acceleration at a stated downward vertical speed.
* medium: each component integrated on its own. The horizontal speed or horizontal distance after a time, the time to
  the greatest height t = (1/k) ln(1 + k u_y / g), the vertical velocity after a time (upward positive).
* hard: the greatest height H = (1/k)[u_y - (g/k) ln(1 + k u_y / g)], the horizontal distance when it is at the
  greatest height, u_x u_y/(g + k u_y), and the speed there, g u_x/(g + k u_y).
* exam: (a) the vertical deceleration at launch, (b) the time to the greatest height, (c) the greatest height,
  (d) the horizontal distance travelled by then. Parts (b) to (d) follow through from (a).

Every closed form is checked against a fourth-order Runge-Kutta integration of the two-dimensional equations of
motion, run to the moment the vertical velocity is zero, which is a different route from the integrals behind the
formulae. The horizontal distance and speed at the top also have a second closed form (u_x u_y/(g + k u_y)).
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r3d_kit import G, G_NOTE, clear_of_boundary, dec, exact, fit_dp, number_fields, number_part, secs

F = Fraction
_g = sympy.Rational(98, 10)


def _launches():
    """Every (horizontal, vertical, speed, tan α) with whole components, a whole speed from 20 to 50 m/s and an angle
    between 20 and 70 degrees, so the learner has to find both components from V and tan α."""
    out = []
    for ux in range(6, 49):
        for uy in range(6, 49):
            speed = math.isqrt(ux * ux + uy * uy)
            if speed * speed != ux * ux + uy * uy or not 20 <= speed <= 50:
                continue
            g = math.gcd(ux, uy)
            if not 0.35 <= uy / ux <= 2.8:
                continue
            out.append((ux, uy, speed, f"{uy // g}/{ux // g}"))
    return tuple(out)


_LAUNCHES = _launches()
_KS = (F(1, 20), F(1, 10), F(3, 20), F(1, 5), F(1, 4), F(3, 10), F(7, 20), F(2, 5), F(9, 20), F(1, 2))
_BODIES = (
    "A ball",
    "A cricket ball",
    "A pebble",
    "A small stone",
    "A tennis ball",
    "A golf ball",
)


def _sym(f: Fraction):
    return sympy.Rational(f.numerator, f.denominator)


def _stem(rng, ux, uy, V, tan, k, extra=""):
    body = rng.choice(_BODIES)
    kd = dec(k, 3)
    return (
        f"{body} is projected from level ground at {V} m/s at an angle α above the horizontal, where tan α = {tan}. "
        f"The resistance per unit mass has magnitude kv and acts opposite to the velocity, where v m/s is the speed "
        f"and k = {kd} per second. {G_NOTE}{extra}"
    )


def _rk4_to_top(ux: float, uy: float, k: float, dt: float = 2e-3):
    """RK4 for x'' = -k x', y'' = -g - k y' until y' = 0. Returns (t, x, y, x')."""
    g = float(G)

    def f(s):
        x, y, vx, vy = s
        return (vx, vy, -k * vx, -g - k * vy)

    s = (0.0, 0.0, ux, uy)
    t = 0.0
    for _ in range(2_000_000):
        k1 = f(s)
        k2 = f(tuple(a + dt * b / 2 for a, b in zip(s, k1, strict=True)))
        k3 = f(tuple(a + dt * b / 2 for a, b in zip(s, k2, strict=True)))
        k4 = f(tuple(a + dt * b for a, b in zip(s, k3, strict=True)))
        n = tuple(a + dt * (p + 2 * q + 2 * r + w) / 6 for a, p, q, r, w in zip(s, k1, k2, k3, k4, strict=True))
        if n[3] <= 0:
            frac = s[3] / (s[3] - n[3])
            return (
                t + dt * frac,
                *(a + (b - a) * frac for a, b in zip(s[:2], n[:2], strict=True)),
                s[2] + (n[2] - s[2]) * frac,
            )
        s, t = n, t + dt
    raise RuntimeError("no top")


def _closed(ux, uy, k):
    """(t_top, H, x_top, vx_top) as exact SymPy-evaluated Fractions."""
    kk, u, w = _sym(k), _sym(F(uy)), _sym(F(ux))
    t = sympy.log(1 + kk * u / _g) / kk
    H = (u - _g / kk * sympy.log(1 + kk * u / _g)) / kk
    x = w * u / (_g + kk * u)
    vx = _g * w / (_g + kk * u)
    return exact(sympy.N(t, 40)), exact(sympy.N(H, 40)), exact(x), exact(vx)


def _check_numeric(ux, uy, k):
    t, H, x, vx = _closed(ux, uy, k)
    tr, xr, yr, vxr = _rk4_to_top(float(ux), float(uy), float(k))
    check(
        abs(tr - float(t)) < 5e-3
        and abs(yr - float(H)) < 5e-3
        and abs(xr - float(x)) < 5e-3
        and abs(vxr - float(vx)) < 5e-3,
        "closed forms differ from the numerical integration",
    )


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    variant = rng.choice(("horizontal", "vertical", "falling"))
    for _ in attempts(500, "an exact answer"):
        ux, uy, V, tan = rng.choice(_LAUNCHES)
        k = rng.choice(_KS)
        if variant == "horizontal":
            value = k * ux
        elif variant == "vertical":
            value = G + k * uy
        else:
            w = F(rng.choice((4, 5, 6, 8, 10, 12, 15)))
            value = G - k * w
            if value <= 0:
                continue
        if fit_dp(value, 1, 3) is not None:
            break
    kd = dec(k, 3)
    if variant == "horizontal":
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + " Find the magnitude of the horizontal acceleration at the instant of projection, in metres per second squared."
        )
        wrong = [
            (k * V, "w6-proj-speed-for-component"),
            (k * uy, "w6-proj-components-swapped"),
            (G + k * ux, "w6-proj-gravity-on-horizontal"),
            (F(ux), "r3d-rate-dropped"),
        ]
        working = [
            f"Resolving, the horizontal component of the initial velocity is {V} cos α = {ux} m/s.",
            f"Horizontally the only force is resistance, so the deceleration is {kd} × {ux} = {dec(value, 3)} m/s².",
        ]
    elif variant == "vertical":
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + " Find the magnitude of the vertical acceleration at the instant of projection, in metres per second squared."
        )
        wrong = [
            (G - k * uy, "r3d-resistance-subtracted"),
            (G + k * V, "w6-proj-speed-for-component"),
            (G + k * ux, "w6-proj-components-swapped"),
            (G, "r3d-resistance-ignored"),
            (k * uy, "r3d-resistance-only"),
        ]
        working = [
            f"The vertical component of the initial velocity is {V} sin α = {uy} m/s.",
            f"Weight and resistance both act downward, so the deceleration is 9.8 + {kd} × {uy} = {dec(value, 3)} m/s².",
        ]
    else:
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + f" At one instant on the way down its vertical speed is {int(w)} m/s. Find the magnitude of its vertical acceleration then, in metres per second squared."
        )
        wrong = [
            (G + k * w, "r3d-resistance-added"),
            (G, "r3d-resistance-ignored"),
            (k * w, "r3d-resistance-only"),
            (G - k * uy, "w6-proj-components-swapped"),
            (G - k * V, "w6-proj-speed-for-component"),
        ]
        working = [
            "Going down, the weight acts downward and the resistance acts upward.",
            f"a = 9.8 - {kd} × {int(w)} = {dec(value, 3)} m/s² downward.",
        ]
    return dict(text=text, working=working, **number_fields(value, fit_dp(value, 1, 3), wrong, nonneg=True))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    variant = rng.choice(("vx", "x", "time_top", "vy"))
    for _ in attempts(800, "a clear value"):
        ux, uy, V, tan = rng.choice(_LAUNCHES)
        k = rng.choice(_KS)
        kk, w, u = _sym(k), _sym(F(ux)), _sym(F(uy))
        t = F(rng.choice((1, 2, 3, 4, 5)))
        tt = _sym(t)
        if variant == "vx":
            expr = w * sympy.exp(-kk * tt)
        elif variant == "x":
            expr = w / kk * (1 - sympy.exp(-kk * tt))
        elif variant == "time_top":
            expr = sympy.log(1 + kk * u / _g) / kk
        else:
            expr = (u + _g / kk) * sympy.exp(-kk * tt) - _g / kk
        value = exact(sympy.N(expr, 40))
        if clear_of_boundary(value, 2) and abs(value) > F(1, 2):
            break
    _check_numeric(ux, uy, k)
    kd = dec(k, 3)
    if variant == "vx":
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + f" Find its horizontal speed {secs(t)} after projection, in metres per second, correct to 2 decimal places."
        )
        wrong = [
            (F(ux), "r3d-resistance-ignored"),
            (ux * exact(sympy.N(sympy.exp(-tt), 40)), "r3d-rate-dropped"),
            (ux * exact(sympy.N(sympy.exp(kk * tt), 40)), "r3d-exponent-sign"),
            (V * exact(sympy.N(sympy.exp(-kk * tt), 40)), "w6-proj-speed-for-component"),
            (uy * exact(sympy.N(sympy.exp(-kk * tt), 40)), "w6-proj-components-swapped"),
        ]
        working = [
            f"Horizontally x'' = -kx', so x' = {ux} e^(-{kd}t).",
            f"At t = {int(t)}, x' = {ux} e^(-{kd} × {int(t)}) = {float(value):.2f} m/s.",
        ]
    elif variant == "x":
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + f" Find its horizontal distance from the point of projection {secs(t)} after projection, in metres, correct to 2 decimal places."
        )
        wrong = [
            (ux * t, "r3d-resistance-ignored"),
            (ux * exact(sympy.N(sympy.exp(-kk * tt), 40)) * t, "w6-proj-speed-times-time"),
            (ux * (1 - exact(sympy.N(sympy.exp(-kk * tt), 40))), "r3d-factor-g-over-k-dropped"),
            (ux / k * exact(sympy.N(sympy.exp(-kk * tt), 40)), "r3d-decay-for-growth"),
        ]
        working = [
            f"x' = {ux} e^(-{kd}t), so x = ({ux}/{kd})(1 - e^(-{kd}t)), with x = 0 at t = 0.",
            f"At t = {int(t)}, x = {float(value):.2f} m.",
        ]
    elif variant == "time_top":
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + " Find the time taken to reach its greatest height, in seconds, correct to 2 decimal places."
        )
        a0 = G + k * uy
        wrong = [
            (uy / G, "r3d-resistance-ignored"),
            (uy / a0, "r3d-constant-deceleration-assumed"),
            (exact(sympy.N(sympy.log(a0 / G), 40)), "r3d-rate-dropped"),
            (exact(sympy.N(sympy.log(1 + kk * u) / kk, 40)), "r3d-gravity-dropped"),
            (V / G, "w6-proj-speed-for-component"),
        ]
        working = [
            f"Vertically y'' = -(g + ky'), so t = (1/k) ln((g + k u_y)/g) with u_y = {uy}.",
            f"t = (1/{kd}) ln({dec(a0, 3)}/9.8) = {float(value):.2f} s.",
        ]
    else:
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + f" Find its vertical velocity {secs(t)} after projection, taking upward as positive, in metres per second, correct to 2 decimal places."
        )
        wrong = [
            (uy - G * t, "r3d-resistance-ignored"),
            (-value, "w6-proj-sign-of-velocity"),
            (u * exact(sympy.N(sympy.exp(-kk * tt), 40)), "w6-proj-gravity-term-dropped"),
            ((uy + G / k) * exact(sympy.N(sympy.exp(-kk * tt), 40)) + G / k, "r3d-resistance-subtracted"),
            ((V + G / k) * exact(sympy.N(sympy.exp(-kk * tt), 40)) - G / k, "w6-proj-speed-for-component"),
        ]
        working = [
            f"Vertically y'' = -g - ky', so y' = (u_y + g/k) e^(-kt) - g/k with g/k = {dec(G / k, 3)}.",
            f"At t = {int(t)}, y' = ({uy} + {dec(G / k, 3)}) e^(-{kd} × {int(t)}) - {dec(G / k, 3)} = {float(value):.2f} m/s.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong))


# ------------------------------------------------------------------------------------------------ hard


def _hard(rng):
    variant = rng.choice(("height", "x_top", "speed_top"))
    for _ in attempts(800, "a clear value"):
        ux, uy, V, tan = rng.choice(_LAUNCHES)
        k = rng.choice(_KS)
        t, H, x, vx = _closed(ux, uy, k)
        value = {"height": H, "x_top": x, "speed_top": vx}[variant]
        if clear_of_boundary(value, 2) and value > F(1, 2):
            break
    _check_numeric(ux, uy, k)
    kk, u, w = _sym(k), _sym(F(uy)), _sym(F(ux))
    kd = dec(k, 3)
    a0 = G + k * uy
    if variant == "height":
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + " Find the greatest height it reaches above the ground, in metres, correct to 2 decimal places."
        )
        wrong = [
            (F(uy) ** 2 / (2 * G), "r3d-resistance-ignored"),
            (F(uy) ** 2 / (2 * a0), "r3d-constant-deceleration-assumed"),
            (exact(sympy.N((u + _g / kk * sympy.log(1 + kk * u / _g)) / kk, 40)), "r3d-log-term-sign"),
            (exact(sympy.N(u / kk, 40)), "w6-proj-limit-for-height"),
            (exact(sympy.N((u - sympy.log(1 + kk * u / _g)) / kk, 40)), "r3d-factor-g-over-k-dropped"),
            (F(V) ** 2 / (2 * G), "w6-proj-speed-for-component"),
        ]
        working = [
            f"Vertically v dv/dy = -(g + kv), so H = ∫ v/(g + kv) dv from 0 to {uy}, with u_y = {uy}.",
            f"v/(g + kv) = (1/k)(1 - g/(g + kv)), so H = (1/k)[u_y - (g/k) ln((g + k u_y)/g)] = {float(value):.2f} m.",
        ]
    elif variant == "x_top":
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + " Find its horizontal distance from the point of projection when it is at its greatest height, in metres, correct to 2 decimal places."
        )
        t_top = t
        wrong = [
            (ux * t_top, "w6-proj-no-drag-range-used"),
            (F(ux) * F(uy) / G, "r3d-resistance-ignored"),
            (F(ux) / k, "w6-proj-limit-for-height"),
            (ux * (1 - exact(sympy.N(sympy.exp(-t_top), 40))), "r3d-rate-dropped"),
        ]
        working = [
            f"First t = (1/{kd}) ln({dec(a0, 3)}/9.8) at the top; horizontally x = ({ux}/{kd})(1 - e^(-{kd}t)).",
            f"Since e^(-kt) = g/(g + k u_y), x = u_x u_y/(g + k u_y) = {ux} × {uy}/{dec(a0, 3)} = {float(value):.2f} m.",
        ]
    else:
        text = (
            _stem(rng, ux, uy, V, tan, k)
            + " Find its speed when it is at its greatest height, in metres per second, correct to 2 decimal places."
        )
        wrong = [
            (F(ux), "r3d-resistance-ignored"),
            (F(ux) * G / (G + k * V), "w6-proj-speed-for-component"),
            (F(ux) * exact(sympy.N(sympy.exp(-t), 40)), "r3d-rate-dropped"),
            (F(ux) * a0 / G, "r3d-ratio-inverted"),
            (F(uy) * G / a0, "w6-proj-components-swapped"),
        ]
        working = [
            f"At the top the speed is the horizontal speed, u_x e^(-kt) with e^(-kt) = g/(g + k u_y) = 9.8/{dec(a0, 3)}.",
            f"Speed = {ux} × 9.8/{dec(a0, 3)} = {float(value):.2f} m/s.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong, nonneg=True))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("w6_projectile_resistance", level=19, presets=_PRESETS)
def w6_projectile_resistance(rng, tier: str = "easy"):
    """Projectile at an angle with resistance kv: components, horizontal and vertical motion, height, distance at the top."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam


@generator("w6_projectile_resistance_exam", level=19, presets={"exam": {}})
def w6_projectile_resistance_exam(rng):
    """Projectile with resistance kv: the launch deceleration, time to the top, greatest height and distance there."""
    for _ in attempts(1000, "a clear chain"):
        ux, uy, V, tan = rng.choice(_LAUNCHES)
        k = rng.choice(_KS)
        a0 = G + k * uy
        t_top, H, x_top, _vx = _closed(ux, uy, k)
        if (
            fit_dp(a0, 1, 2) is not None
            and clear_of_boundary(t_top, 2)
            and clear_of_boundary(H, 1)
            and clear_of_boundary(x_top, 1)
        ):
            break
    _check_numeric(ux, uy, k)
    kk, u = _sym(k), _sym(F(uy))
    kd = dec(k, 3)
    stem = _stem(rng, ux, uy, V, tan, k)
    part_a = number_part(
        "Find the magnitude of the vertical acceleration at the instant of projection, in metres per second squared.",
        a0,
        fit_dp(a0, 1, 2),
        1,
        [
            f"The vertical component of the initial velocity is {V} sin α = {uy} m/s.",
            f"Weight and resistance both act downward, so the deceleration is 9.8 + {kd} × {uy} = {dec(a0, 3)} m/s².",
        ],
        [
            (G - k * uy, "r3d-resistance-subtracted"),
            (G + k * V, "w6-proj-speed-for-component"),
            (G + k * ux, "w6-proj-components-swapped"),
            (k * uy, "r3d-resistance-only"),
        ],
        sensible=(0, None),
        nonneg=True,
    )
    part_b = number_part(
        "Find the time taken to reach the greatest height, in seconds, correct to 2 decimal places.",
        t_top,
        2,
        2,
        [
            "Vertically y'' = -(g + ky'), so t = (1/k) ln((g + k u_y)/g).",
            f"t = (1/{kd}) ln({dec(a0, 3)}/9.8) = {float(t_top):.2f} s.",
        ],
        [
            (F(uy) / G, "r3d-resistance-ignored"),
            (F(uy) / a0, "r3d-constant-deceleration-assumed"),
            (exact(sympy.N(sympy.log(a0 / G), 40)), "r3d-rate-dropped"),
            (exact(sympy.N(sympy.log(1 + kk * u) / kk, 40)), "r3d-gravity-dropped"),
        ],
        follow=f"log(part_a/9.8)/{kd}",
        sensible=(0, None),
        nonneg=True,
    )
    part_c = number_part(
        "Find the greatest height reached above the ground, in metres, correct to 1 decimal place.",
        H,
        1,
        3,
        [
            "v dv/dy = -(g + kv) and v/(g + kv) = (1/k)(1 - g/(g + kv)).",
            f"H = (1/k)[u_y - (g/k) ln((g + k u_y)/g)] = (1/{kd})[{uy} - {dec(G / k, 3)} ln({dec(a0, 3)}/9.8)] = {float(H):.1f} m.",
        ],
        [
            (F(uy) ** 2 / (2 * G), "r3d-resistance-ignored"),
            (F(uy) ** 2 / (2 * a0), "r3d-constant-deceleration-assumed"),
            (exact(sympy.N((u + _g / kk * sympy.log(1 + kk * u / _g)) / kk, 40)), "r3d-log-term-sign"),
            (F(uy) * t_top / 2, "r3d-mean-speed-used"),
        ],
        follow=f"({uy} - 9.8*log(part_a/9.8)/{kd})/{kd}",
        sensible=(0, None),
        nonneg=True,
    )
    part_d = number_part(
        "Find its horizontal distance from the point of projection when it is at its greatest height, in metres, correct to 1 decimal place.",
        x_top,
        1,
        2,
        [
            f"Horizontally x'' = -kx', so x = ({ux}/{kd})(1 - e^(-{kd}t)), and at the top e^(-kt) = g/(g + k u_y) = 9.8/{dec(a0, 3)}.",
            f"x = {ux} × {uy}/{dec(a0, 3)} = {float(x_top):.1f} m.",
        ],
        [
            (ux * t_top, "w6-proj-no-drag-range-used"),
            (F(ux) / k, "w6-proj-limit-for-height"),
            (F(ux) * F(uy) / G, "r3d-resistance-ignored"),
            (ux * (1 - exact(sympy.N(sympy.exp(-t_top), 40))), "r3d-rate-dropped"),
        ],
        follow=f"{ux}*{uy}/part_a",
        sensible=(0, None),
        nonneg=True,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
