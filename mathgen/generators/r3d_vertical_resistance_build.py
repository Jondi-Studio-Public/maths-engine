"""Vertical motion with resistance proportional to speed: `r3d_vertical_resistance` (easy, medium, hard) and
`r3d_vertical_resistance_exam`.

Skill: vertical-motion-with-resistance. A body of mass m moves vertically against air resistance of magnitude cv
newtons (v is its speed in m/s), so k = c/m per second is the resistance per unit mass. Falling, the acceleration is
g - kv (downward positive); rising, the deceleration is g + kv. g = 9.8 m/s² and every prompt says so.

Tiers differ in the decision they ask for, not in the size of the numbers:

* easy: no integration. The terminal speed (resistance equals weight), the acceleration at a stated speed, the
  deceleration at the instant of launch of an object thrown upwards. The resistance is given in newtons, so
  k = c/m has to be formed first.
* medium: dv/dt integrated. The speed after a time falling from rest, v = V(1 - e^(-kt)), the time to reach the greatest
  height going up, t = (1/k) ln(1 + ku/g), the time to reach a stated fraction of the terminal speed.
* hard: v dv/dx integrated, which needs v/(g + kv) split into partial fractions: the greatest height when thrown
  upwards, the distance fallen to reach a fraction of the terminal speed, and the terminal speed worked out backwards
  from one measured acceleration.
* exam: thrown upwards. (a) the deceleration at launch, (b) the time to the greatest height, (c) the greatest height,
  (d) how much lower that is than the greatest height with no air resistance.

## Construction

(m, c) pairs are chosen so that k = c/m is one of 0.1, 0.2, 0.25, 0.35, 0.4 or 0.5 and the terminal speed g/k is a
sensible one for the object (a parachutist 49 m/s, a ball 20 to 40 m/s). Closed forms are checked against a
fourth-order Runge-Kutta integration of the equation of motion run until the stated event (the speed reaches zero,
or a stated fraction of the terminal speed), which is a different route from the integrals the closed forms come from.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r3d_kit import G, G_NOTE, clear_of_boundary, dec, exact, fit_dp, number_fields, number_part, secs

F = Fraction
_g = sympy.Rational(98, 10)

#: (mass kg, resistance coefficient N per m/s, noun phrase, upward speed choices, k)
_OBJECTS = (
    (F(80), F(16), "A parachutist of mass 80 kg", F(1, 5)),
    (F(60), F(12), "A skydiver of mass 60 kg", F(1, 5)),
    (F(70), F(7), "A diver of mass 70 kg in a streamlined position", F(1, 10)),
    (F(2, 5), F(1, 25), "A hailstone of mass 0.4 kg", F(1, 10)),
    (F(1, 2), F(1, 10), "A ball of mass 0.5 kg", F(1, 5)),
    (F(1, 5), F(1, 20), "A ball of mass 0.2 kg", F(1, 4)),
    (F(2, 5), F(7, 50), "A ball of mass 0.4 kg", F(7, 20)),
    (F(1, 2), F(1, 5), "A ball of mass 0.5 kg", F(2, 5)),
    (F(3), F(3, 2), "A ball of mass 3 kg", F(1, 2)),
)
_UP_OBJECTS = tuple(o for o in _OBJECTS if "ball" in o[2])
_DOWN_OBJECTS = _OBJECTS


def _newtons(c: Fraction) -> str:
    return "v newtons" if c == 1 else f"{dec(c, 3)}v newtons"


def _rk4(accel, v0: float, *, until_t: float | None = None, until_v: float | None = None, dt: float = 5e-3):
    """Fourth-order Runge-Kutta for v' = accel(v), x' = v, from v(0) = v0 and x(0) = 0, run to `until_t`, or to the
    moment the speed reaches `until_v` (crossing it from either side). Returns (t, x, v) at the event, the last step
    corrected back to the event by linear interpolation."""
    t = x = 0.0
    v = v0
    for _ in range(2_000_000):
        if until_t is not None and t >= until_t - 1e-12:
            return t, x, v
        k1 = accel(v)
        k2 = accel(v + dt * k1 / 2)
        k3 = accel(v + dt * k2 / 2)
        k4 = accel(v + dt * k3)
        v_new = v + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if until_v is not None and (v - until_v) * (v_new - until_v) <= 0:
            frac = (v - until_v) / (v - v_new)
            h = dt * frac
            return t + h, x + h * (v + (v + (v_new - v) * frac)) / 2, until_v
        x += dt * (v + v_new) / 2
        v, t = v_new, t + dt
    raise RuntimeError("integration did not reach the event")


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    variant = rng.choice(("terminal", "acceleration", "launch"))
    for _ in attempts(500, "an exact answer"):
        m, c, noun, k = rng.choice(_DOWN_OBJECTS if variant != "launch" else _UP_OBJECTS)
        V = G / k
        if variant == "terminal":
            value = m * G / c
            check(value == V, "terminal speed route disagrees")
        elif variant == "acceleration":
            v = F(rng.choice((4, 6, 8, 10, 12, 15, 16, 20)))
            if v >= V:
                continue
            value = G - k * v
        else:
            u = F(rng.choice((10, 12, 15, 16, 20, 24, 25, 30)))
            value = G + k * u
        dp = fit_dp(value, 1, 3)
        if dp is not None and value != 0:
            break
    res = _newtons(c)
    if variant == "terminal":
        text = (
            f"{noun} falls vertically. The air resistance on it is {res} when its speed is v m/s. {G_NOTE} "
            f"Find its terminal speed, in metres per second."
        )
        wrong = [
            (m * G * c, "r3d-multiplied-for-divided"),
            (c / (m * G), "r3d-ratio-inverted"),
            (G / c, "r3d-mass-dropped"),
            (m * G, "r3d-weight-for-speed"),
            (c / m, "r3d-resistance-per-mass-for-speed"),
        ]
        working = [
            "At terminal speed the acceleration is zero, so the resistance equals the weight: cV = mg.",
            f"V = mg/c = {dec(m, 3)} × 9.8/{dec(c, 3)} = {dec(value, 3)} m/s.",
        ]
        dp = fit_dp(value, 1, 3)
    elif variant == "acceleration":
        text = (
            f"{noun} falls vertically. The air resistance on it is {res} when its speed is v m/s. {G_NOTE} "
            f"Find its downward acceleration when its speed is {dec(v, 2)} m/s, in metres per second squared."
        )
        wrong = [
            (G, "r3d-resistance-ignored"),
            (G + k * v, "r3d-resistance-added"),
            (k * v, "r3d-resistance-only"),
            (G - c * v, "r3d-mass-ignored"),
            (m * G - c * v, "r3d-force-for-acceleration"),
        ]
        working = [
            f"m a = mg - cv, so a = g - (c/m)v with c/m = {dec(k, 3)}.",
            f"a = 9.8 - {dec(k, 3)} × {dec(v, 2)} = {dec(value, 3)} m/s².",
        ]
    else:
        text = (
            f"{noun} is thrown vertically upward at {dec(u, 2)} m/s. The air resistance on it is {res} when its speed "
            f"is v m/s. {G_NOTE} Find the magnitude of its deceleration at the instant it is thrown, in metres "
            f"per second squared."
        )
        wrong = [
            (G - k * u, "r3d-resistance-subtracted"),
            (G, "r3d-resistance-ignored"),
            (k * u, "r3d-resistance-only"),
            (G + c * u, "r3d-mass-ignored"),
            (m * (G + k * u), "r3d-force-for-acceleration"),
        ]
        working = [
            f"Weight and resistance both act downward: m a = mg + cv, so a = g + (c/m)v with c/m = {dec(k, 3)}.",
            f"a = 9.8 + {dec(k, 3)} × {dec(u, 2)} = {dec(value, 3)} m/s².",
        ]
    return dict(text=text, working=working, **number_fields(value, dp, wrong, nonneg=True))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    variant = rng.choice(("speed_after", "time_to_top", "time_to_fraction"))
    for _ in attempts(500, "a clear value"):
        m, c, noun, k = rng.choice(_DOWN_OBJECTS if variant != "time_to_top" else _UP_OBJECTS)
        V = G / k
        kk = sympy.Rational(k.numerator, k.denominator)
        if variant == "speed_after":
            t = F(rng.choice((2, 3, 4, 5, 6, 8, 10)))
            expr = sympy.Rational(V.numerator, V.denominator) * (
                1 - sympy.exp(-kk * sympy.Rational(t.numerator, t.denominator))
            )
        elif variant == "time_to_top":
            u = F(rng.choice((10, 12, 15, 16, 20, 24, 25, 30)))
            expr = sympy.log(1 + kk * sympy.Rational(u.numerator, u.denominator) / _g) / kk
        else:
            frac = rng.choice((F(1, 2), F(3, 4), F(4, 5), F(9, 10)))
            expr = -sympy.log(1 - sympy.Rational(frac.numerator, frac.denominator)) / kk
        value = exact(sympy.N(expr, 40))
        if clear_of_boundary(value, 2) and value > F(1, 2):
            break
    res = _newtons(c)
    kd = dec(k, 3)
    # second route: Runge-Kutta
    if variant == "speed_after":
        _, _, v = _rk4(lambda s: float(G) - float(k) * s, 0.0, until_t=float(t))
        check(abs(v - float(value)) < 5e-3, "speed differs from the numerical integration")
        text = (
            f"{noun} is released from rest and falls vertically. The air resistance on it is {res} when its speed is "
            f"v m/s. {G_NOTE} Find its speed {secs(t)} after release, in metres per second, correct to "
            f"2 decimal places."
        )
        wrong = [
            (V, "r3d-terminal-speed-for-speed"),
            (G * t, "r3d-resistance-ignored"),
            (V * (1 - exact(sympy.N(sympy.exp(-sympy.Rational(t.numerator, t.denominator)), 40))), "r3d-rate-dropped"),
            (
                exact(
                    sympy.N(
                        sympy.Rational(V.numerator, V.denominator)
                        * sympy.exp(-kk * sympy.Rational(t.numerator, t.denominator)),
                        40,
                    )
                ),
                "r3d-decay-for-growth",
            ),
            (
                exact(
                    sympy.N(
                        sympy.Rational(V.numerator, V.denominator)
                        * (sympy.exp(kk * sympy.Rational(t.numerator, t.denominator)) - 1),
                        40,
                    )
                ),
                "r3d-exponent-sign",
            ),
        ]
        working = [
            f"k = c/m = {kd}. dv/dt = g - kv, so v = (g/k)(1 - e^(-kt)) with g/k = {dec(V, 3)}.",
            f"At t = {int(t)}, v = {dec(V, 3)}(1 - e^(-{kd} × {int(t)})) = {float(value):.2f} m/s.",
        ]
    elif variant == "time_to_top":
        t_rk, _, _ = _rk4(lambda s: -(float(G) + float(k) * s), float(u), until_v=0.0)
        check(abs(t_rk - float(value)) < 5e-3, "time differs from the numerical integration")
        text = (
            f"{noun} is thrown vertically upward at {dec(u, 2)} m/s. The air resistance on it is {res} when its speed "
            f"is v m/s. {G_NOTE} Find the time taken to reach its greatest height, in seconds, correct to 2 decimal places."
        )
        wrong = [
            (u / G, "r3d-resistance-ignored"),
            (
                exact(sympy.N(sympy.log(1 + sympy.Rational(u.numerator, u.denominator) / _g) / kk, 40)),
                "r3d-rate-dropped",
            ),
            (
                exact(sympy.N(sympy.log(1 + kk * sympy.Rational(u.numerator, u.denominator)) / kk, 40)),
                "r3d-gravity-dropped",
            ),
            (u / (G + k * u), "r3d-constant-deceleration-assumed"),
            (
                exact(sympy.N(sympy.log(1 - kk * sympy.Rational(u.numerator, u.denominator) / _g + 1) / kk, 40))
                if k * u < G
                else -1,
                "r3d-resistance-subtracted",
            ),
        ]
        working = [
            f"k = c/m = {kd}. Going up, dv/dt = -(g + kv), so t = (1/k) ln((g + ku)/g).",
            f"t = (1/{kd}) ln((9.8 + {kd} × {dec(u, 2)})/9.8) = {float(value):.2f} s.",
        ]
    else:
        target = V * frac
        t_rk, _, _ = _rk4(lambda s: float(G) - float(k) * s, 0.0, until_v=float(target))
        check(abs(t_rk - float(value)) < 5e-3, "time differs from the numerical integration")
        text = (
            f"{noun} is released from rest and falls vertically. The air resistance on it is {res} when its speed is "
            f"v m/s. {G_NOTE} Find the time taken for its speed to reach {f'{float(frac * 100):g}'}% of its terminal speed, "
            f"in seconds, correct to 2 decimal places."
        )
        wrong = [
            (
                exact(sympy.N(-sympy.log(sympy.Rational(frac.numerator, frac.denominator)) / kk, 40)),
                "r3d-fraction-not-complement",
            ),
            (
                exact(
                    sympy.N(
                        sympy.Rational(frac.numerator, frac.denominator)
                        * sympy.Rational(V.numerator, V.denominator)
                        / _g,
                        40,
                    )
                ),
                "r3d-resistance-ignored",
            ),
            (exact(sympy.N(-sympy.log(1 - sympy.Rational(frac.numerator, frac.denominator)), 40)), "r3d-rate-dropped"),
            (exact(sympy.N(sympy.log(1 - sympy.Rational(frac.numerator, frac.denominator)) / kk, 40)), "r3d-log-sign"),
        ]
        working = [
            f"v = V(1 - e^(-kt)) with V = {dec(V, 3)} and k = {kd}; v = fV gives e^(-kt) = 1 - f = {dec(1 - frac, 4)}.",
            f"t = -ln({dec(1 - frac, 4)})/{kd} = {float(value):.2f} s.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong, nonneg=True))


# ------------------------------------------------------------------------------------------------ hard


def _hard(rng):
    variant = rng.choice(("greatest_height", "distance_to_fraction", "terminal_from_acceleration"))
    if variant == "terminal_from_acceleration":
        for _ in attempts(500, "an exact terminal speed"):
            k = rng.choice((F(1, 10), F(1, 5), F(1, 4), F(7, 20), F(2, 5), F(1, 2)))
            v = F(rng.choice((6, 8, 10, 12, 14, 16, 20)))
            alpha = G - k * v
            if alpha <= 0 or fit_dp(alpha, 1, 2) is None:
                continue
            V = G / k
            if fit_dp(V, 1, 3) is not None:
                break
        m, c, noun, _ = rng.choice([o for o in _DOWN_OBJECTS if o[3] == k] or _DOWN_OBJECTS)
        check(G * v / (G - alpha) == V, "terminal speed route disagrees")
        text = (
            f"An object of mass {dec(m, 3)} kg falls vertically against air resistance proportional to its speed. {G_NOTE} "
            f"When its speed is {dec(v, 2)} m/s its downward acceleration is {dec(alpha, 3)} m/s². Find its terminal "
            f"speed, in metres per second."
        )
        wrong = [
            (v + alpha, "r3d-speed-plus-acceleration"),
            (G * v / alpha, "r3d-acceleration-for-resistance"),
            (G / alpha, "r3d-speed-dropped"),
            (v * G / (G + alpha), "r3d-resistance-added"),
            (v * alpha / (G - alpha), "r3d-ratio-misplaced"),
        ]
        working = [
            f"a = g - kv, so {dec(alpha, 3)} = 9.8 - k({dec(v, 2)}) and k = {dec(k, 3)}.",
            f"Terminal speed = g/k = 9.8/{dec(k, 3)} = {dec(V, 3)} m/s.",
        ]
        return dict(text=text, working=working, **number_fields(V, fit_dp(V, 1, 3), wrong, nonneg=True))
    for _ in attempts(500, "a clear value"):
        m, c, noun, k = rng.choice(_UP_OBJECTS if variant == "greatest_height" else _DOWN_OBJECTS)
        V = G / k
        kk = sympy.Rational(k.numerator, k.denominator)
        if variant == "greatest_height":
            u = F(rng.choice((10, 12, 15, 16, 20, 24, 25, 30)))
            uu = sympy.Rational(u.numerator, u.denominator)
            expr = (uu - _g / kk * sympy.log(1 + kk * uu / _g)) / kk
        else:
            frac = rng.choice((F(1, 2), F(3, 4), F(4, 5), F(9, 10)))
            ff = sympy.Rational(frac.numerator, frac.denominator)
            VV = sympy.Rational(V.numerator, V.denominator)
            expr = VV / kk * (-sympy.log(1 - ff) - ff)
        value = exact(sympy.N(expr, 40))
        if clear_of_boundary(value, 2) and value > F(1, 2):
            break
    res = _newtons(c)
    if variant == "greatest_height":
        _, x_rk, _ = _rk4(lambda s: -(float(G) + float(k) * s), float(u), until_v=0.0)
        check(abs(x_rk - float(value)) < 5e-3, "height differs from the numerical integration")
        text = (
            f"{noun} is thrown vertically upward at {dec(u, 2)} m/s. The air resistance on it is {res} when its speed "
            f"is v m/s. {G_NOTE} Find the greatest height it reaches above the point of projection, in metres, "
            f"correct to 2 decimal places."
        )
        wrong = [
            (u * u / (2 * G), "r3d-resistance-ignored"),
            (u * u / (2 * (G + k * u)), "r3d-constant-deceleration-assumed"),
            (exact(sympy.N((uu + _g / kk * sympy.log(1 + kk * uu / _g)) / kk, 40)), "r3d-log-term-sign"),
            (exact(sympy.N(uu / kk, 40)), "r3d-log-term-dropped"),
            (exact(sympy.N((uu - sympy.log(1 + kk * uu / _g)) / kk, 40)), "r3d-factor-g-over-k-dropped"),
        ]
        working = [
            f"k = c/m = {dec(k, 3)}. Going up, v dv/dx = -(g + kv), so x = ∫ v/(g + kv) dv from 0 to {dec(u, 2)}.",
            f"v/(g + kv) = (1/k)(1 - g/(g + kv)), so H = (1/k)[u - (g/k) ln((g + ku)/g)] = {float(value):.2f} m.",
        ]
    else:
        target = V * frac
        _, x_rk, _ = _rk4(lambda s: float(G) - float(k) * s, 0.0, until_v=float(target))
        check(abs(x_rk - float(value)) < 5e-3, "distance differs from the numerical integration")
        pct = f"{float(frac * 100):g}"
        text = (
            f"{noun} is released from rest and falls vertically. The air resistance on it is {res} when its speed is "
            f"v m/s. {G_NOTE} Find the distance it falls while its speed rises to {pct}% of its terminal speed, "
            f"in metres, correct to 2 decimal places."
        )
        wrong = [
            (exact(sympy.N(VV / kk * -sympy.log(1 - ff), 40)), "r3d-linear-term-dropped"),
            ((frac * V) ** 2 / (2 * G), "r3d-resistance-ignored"),
            (exact(sympy.N(VV / kk * (-sympy.log(1 - ff) + ff), 40)), "r3d-linear-term-sign"),
            (exact(sympy.N(VV * (-sympy.log(1 - ff) - ff), 40)), "r3d-rate-dropped"),
        ]
        working = [
            f"V = g/k = {dec(V, 3)}. v dv/dx = g - kv = k(V - v), so x = (1/k)∫ v/(V - v) dv = (1/k)[-V ln(1 - v/V) - v].",
            f"With v = {pct}% of V: x = {float(value):.2f} m.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong, nonneg=True))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("r3d_vertical_resistance", level=18, presets=_PRESETS)
def r3d_vertical_resistance(rng, tier: str = "easy"):
    """Vertical motion with resistance cv: terminal speed, speed and time by dv/dt, height and distance by v dv/dx."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam

_EXAM_STEMS = (
    "{noun} is thrown vertically upward at {u} m/s. The air resistance on it has magnitude {res} when its speed is "
    "v m/s. {note}",
    "{noun} is launched vertically upward with a speed of {u} m/s. While it is moving the air resistance on it has "
    "magnitude {res}, where v m/s is its speed. {note}",
    "{noun} is thrown vertically upward at {u} m/s, and air resistance of magnitude {res} acts against the motion, "
    "where v m/s is its speed. {note}",
)


@generator("r3d_vertical_resistance_exam", level=18, presets={"exam": {}})
def r3d_vertical_resistance_exam(rng):
    """Thrown upward against resistance cv: the deceleration at launch, the time to the top, the greatest height, and
    how much lower that is than the height with no air resistance."""
    for _ in attempts(500, "a clear chain"):
        m, c, noun, k = rng.choice(_UP_OBJECTS)
        u = F(rng.choice((15, 16, 20, 24, 25, 30)))
        a0 = G + k * u
        kk = sympy.Rational(k.numerator, k.denominator)
        uu = sympy.Rational(u.numerator, u.denominator)
        t_up = exact(sympy.N(sympy.log(1 + kk * uu / _g) / kk, 40))
        height = exact(sympy.N((uu - _g / kk * sympy.log(1 + kk * uu / _g)) / kk, 40))
        free = u * u / (2 * G)
        if (
            fit_dp(a0, 1, 2) is not None
            and clear_of_boundary(t_up, 2)
            and clear_of_boundary(height, 1)
            and clear_of_boundary(free - height, 1)
        ):
            break
    t_rk, x_rk, _ = _rk4(lambda s: -(float(G) + float(k) * s), float(u), until_v=0.0)
    check(
        abs(t_rk - float(t_up)) < 5e-3 and abs(x_rk - float(height)) < 5e-3, "closed forms differ from the integration"
    )
    res = _newtons(c)
    stem = _EXAM_STEMS[rng.randrange(len(_EXAM_STEMS))].format(noun=noun, u=dec(u, 2), res=res, note=G_NOTE)
    kd = dec(k, 3)
    dp_a = fit_dp(a0, 1, 2)
    part_a = number_part(
        "Find the magnitude of the deceleration of the object at the instant it is thrown, in metres per second squared.",
        a0,
        dp_a,
        1,
        [
            f"Weight and resistance both act downward: m a = mg + cv, so a = g + (c/m)v, with c/m = {kd}.",
            f"a = 9.8 + {kd} × {dec(u, 2)} = {dec(a0, 3)} m/s².",
        ],
        [
            (G - k * u, "r3d-resistance-subtracted"),
            (G, "r3d-resistance-ignored"),
            (k * u, "r3d-resistance-only"),
            (G + c * u, "r3d-mass-ignored"),
        ],
        sensible=(0, None),
        nonneg=True,
    )
    part_b = number_part(
        "Find the time taken to reach the greatest height, in seconds, correct to 2 decimal places.",
        t_up,
        2,
        2,
        [
            f"Going up, dv/dt = -(g + kv), so t = (1/k) ln((g + ku)/g), where g + ku = {dec(a0, 3)}.",
            f"t = (1/{kd}) ln({dec(a0, 3)}/9.8) = {float(t_up):.2f} s.",
        ],
        [
            (u / G, "r3d-resistance-ignored"),
            (u / a0, "r3d-constant-deceleration-assumed"),
            (exact(sympy.N(sympy.log(a0 / G), 40)), "r3d-rate-dropped"),
            (exact(sympy.N(sympy.log(1 + kk * uu) / kk, 40)), "r3d-gravity-dropped"),
        ],
        follow=f"log(part_a/9.8)/{kd}",
        sensible=(0, None),
        nonneg=True,
    )
    part_c = number_part(
        "Find the greatest height reached above the point of projection, in metres, correct to 1 decimal place.",
        height,
        1,
        2,
        [
            "v dv/dx = -(g + kv) and v/(g + kv) = (1/k)(1 - g/(g + kv)).",
            f"H = (1/k)[u - (g/k) ln((g + ku)/g)] = (1/{kd})[{dec(u, 2)} - {dec(G / k, 3)} ln({dec(a0, 3)}/9.8)] = {float(height):.1f} m.",
        ],
        [
            (free, "r3d-resistance-ignored"),
            (u * u / (2 * a0), "r3d-constant-deceleration-assumed"),
            (exact(sympy.N((uu + _g / kk * sympy.log(1 + kk * uu / _g)) / kk, 40)), "r3d-log-term-sign"),
            (u * t_up / 2, "r3d-mean-speed-used"),
        ],
        follow=f"({dec(u, 2)} - 9.8*log(part_a/9.8)/{kd})/{kd}",
        sensible=(0, None),
        nonneg=True,
    )
    part_d = number_part(
        "Find how much lower this is than the greatest height the object would reach if there were no air resistance, "
        "in metres, correct to 1 decimal place.",
        free - height,
        1,
        1,
        [
            f"Without resistance the greatest height is u²/(2g) = {dec(u, 2)}²/19.6 = {float(free):.1f} m.",
            f"The difference is {float(free):.1f} - {float(height):.1f} = {float(free - height):.1f} m.",
        ],
        [
            (free, "r3d-whole-height-for-difference"),
            (height, "r3d-height-for-difference"),
            (free + height, "r3d-sum-for-difference"),
            (free - u * u / (2 * a0), "r3d-constant-deceleration-assumed"),
        ],
        follow=f"{dec(u, 2)}**2/19.6 - part_c",
        sensible=(0, None),
        nonneg=True,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
