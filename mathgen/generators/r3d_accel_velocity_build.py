"""Acceleration as a function of velocity, a = -kv and a = -kv²: `r3d_accel_velocity` (easy, medium, hard) and
`r3d_accel_velocity_exam`.

Skill: acceleration-function-of-velocity. A resisting acceleration that depends on the speed is solved two ways:
with a = dv/dt for velocity against time, and with a = v dv/dx for velocity against distance. Neither is a constant
acceleration formula.

Tiers differ in the decision they ask for, not in the size of the numbers:

* easy: a = -kv. The velocity after a time (an exponential decay) or the time for the speed to fall to a stated value
  (a logarithm).
* medium: a = -kv² gives v = v0/(1 + kv0t), so the speed after a time and the time to slow to a stated speed are
  rational. For a = -kv, writing a = v dv/dx cancels v, so the speed falls linearly with distance and the
  distance to stop (or to reach a stated speed) is exact.
* hard: the quantity needs a second integration or the other form of the acceleration: the distance travelled
  in a stated time for either law, the speed after a stated distance for a = -kv², the time to cover a stated distance.
* exam: a boat (a = -kv) or a cyclist (a = -kv²) slowing down. (a) v(t) as an expression, (b) the speed at a stated
  time, (c) the distance travelled to that time, (d) a quantity following from it: the limiting distance for the boat,
  or the time to halve the speed for the cyclist.

## Construction

Every value is an exact SymPy expression (exp, log and rational functions) turned into a Fraction only at display,
rounded half-up, and kept away from a rounding boundary. Each model is also checked by substituting the claimed
velocity back into its differential equation with SymPy (dv/dt + kv = 0 or dv/dt + kv² = 0).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r3d_kit import (
    T,
    clear_of_boundary,
    dec,
    exact,
    expression_part,
    number_fields,
    number_part,
    residual_zero,
    secs,
    simpson,
)

F = Fraction
_t = T


def _R(v) -> sympy.Rational:
    v = F(v)
    return sympy.Rational(v.numerator, v.denominator)


def _exp_text(coef, rate) -> str:
    """'20e^(-0.2t)' for coef × e^(−rate·t), rate may be negative for growth."""
    sign = "-" if rate > 0 else ""
    mag = abs(F(rate))
    rate_txt = "t" if mag == 1 else f"{dec(mag, 4)}t"
    c = dec(coef, 4)
    return f"{c}e^({sign}{rate_txt})"


def _lin_text(c0, c1) -> str:
    """'20 - 4t'."""
    return f"{dec(c0, 4)} {'-' if c1 < 0 else '+'} {dec(abs(c1), 4)}t"


_BOAT = (
    "A boat's engine is switched off while it is moving at {v0} m/s. From then on the water gives it an acceleration "
    "of -{k}v m/s², where v m/s is its speed",
    "A trolley is moving at {v0} m/s along a track when its drive is cut. Friction in the axles then gives it an "
    "acceleration of -{k}v m/s², where v m/s is its speed",
    "A skater glides across the ice at {v0} m/s and stops pushing. Her acceleration is then -{k}v m/s², where v m/s "
    "is her speed",
)
_CYCLIST = (
    "A car is coasting in neutral at {v0} m/s on a level road. Air resistance gives it an acceleration of -{k}v² m/s², "
    "where v m/s is the speed",
    "A cyclist stops pedalling while travelling at {v0} m/s on level ground. Air resistance then gives an "
    "acceleration of -{k}v² m/s², where v m/s is the speed",
)


def _linear(rng):
    """(v0, k) for a = -kv."""
    return F(rng.choice((6, 8, 10, 12))), F(rng.choice((F(1, 10), F(1, 5), F(1, 4), F(2, 5), F(1, 2))))


def _quadratic(rng):
    """(v0, k) for a = -kv², k per metre."""
    return F(rng.choice((10, 12, 15, 20, 25))), F(rng.choice((F(1, 100), F(1, 50), F(1, 25), F(1, 20))))


def _cyc(rng, v0, k) -> str:
    """The cyclist or car sentence; a speed of 20 m/s or more is a car."""
    patterns = _CYCLIST[:1] if v0 >= 20 else _CYCLIST
    return patterns[rng.randrange(len(patterns))].format(v0=dec(v0, 2), k=dec(k, 4))


def _fmt(rng, patterns, **slots):
    return patterns[rng.randrange(len(patterns))].format(**slots)


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    variant = rng.choice(("speed_after", "time_to_speed"))
    for _ in attempts(500, "a clear decay"):
        v0, k = _linear(rng)
        if variant == "speed_after":
            t = F(rng.choice((1, 2, 3, 4, 5, 6, 8, 10)))
            value = sympy.N(_R(v0) * sympy.exp(-_R(k) * _R(t)), 40)
        else:
            frac = rng.choice((F(1, 2), F(1, 4), F(1, 5), F(3, 4), F(1, 10)))
            v1 = v0 * frac
            if v1.denominator != 1 and (v1 * 10).denominator != 1:
                continue
            value = sympy.N(sympy.log(_R(v0 / v1)) / _R(k), 40)
        value = exact(value)
        if clear_of_boundary(value, 2) and value > F(1, 2):
            break
    intro = _fmt(rng, _BOAT, v0=dec(v0, 2), k=dec(k, 3))
    # second route: the claimed velocity satisfies dv/dt = -kv and v(0) = v0
    vt = _R(v0) * sympy.exp(-_R(k) * _t)
    check(
        residual_zero(sympy.diff(vt, _t) + _R(k) * vt, _t) and vt.subs(_t, 0) == _R(v0), "v does not solve dv/dt = -kv"
    )
    if variant == "speed_after":
        text = f"{intro}. Find the speed {secs(t)} later, in metres per second, correct to 2 decimal places."
        wrong = [
            (exact(sympy.N(_R(v0) * sympy.exp(_R(k) * _R(t)), 40)), "r3d-exponent-sign"),
            (exact(sympy.N(_R(v0) * sympy.exp(-_R(t)), 40)), "r3d-rate-dropped"),
            (v0 - k * v0 * t, "r3d-constant-deceleration-assumed"),
            (v0 / (1 + k * v0 * t), "r3d-quadratic-law-used"),
            (exact(sympy.N(_R(v0) * sympy.exp(-_R(k)), 40)), "r3d-time-dropped"),
        ]
        working = [
            f"a = dv/dt = -{dec(k, 3)}v, so dv/v = -{dec(k, 3)} dt and ln v = -{dec(k, 3)}t + C.",
            f"v = {dec(v0, 2)}e^(-{dec(k, 3)}t); at t = {int(t)}, v = {float(value):.2f} m/s.",
        ]
    else:
        text = (
            f"{intro}. Find the time taken for the speed to fall to {dec(v1, 2)} m/s, in seconds, correct to "
            f"2 decimal places."
        )
        wrong = [
            (-value, "r3d-log-ratio-inverted"),
            (exact(sympy.N(sympy.log(_R(v0 / v1)), 40)), "r3d-rate-dropped"),
            ((v0 - v1) / (k * v0), "r3d-constant-deceleration-assumed"),
            (exact(sympy.N(sympy.log(_R(v0 / v1)) * _R(k), 40)), "r3d-rate-multiplied"),
            ((1 / v1 - 1 / v0) / k, "r3d-quadratic-law-used"),
        ]
        working = [
            f"v = {dec(v0, 2)}e^(-{dec(k, 3)}t) = {dec(v1, 2)}, so e^(-{dec(k, 3)}t) = {dec(v1 / v0, 3)}.",
            f"t = ln({dec(v0 / v1, 3)})/{dec(k, 3)} = {float(value):.2f} s.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong, nonneg=True))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    variant = rng.choice(("quadratic_speed", "quadratic_time", "linear_distance"))
    if variant == "linear_distance":
        v0, k = _linear(rng)
        to_rest = rng.random() < 0.5
        v1 = F(0) if to_rest else v0 * F(rng.choice((F(1, 2), F(1, 4), F(3, 4))))
        value = (v0 - v1) / k
        dp = 1 if value.denominator in (1, 2, 5, 10) and (value * 10).denominator == 1 else 2
        value_dp = value
        intro = _fmt(rng, _BOAT, v0=dec(v0, 2), k=dec(k, 3))
        vx = v0 - k * _t  # v against distance: dv/dx = -k
        check(sympy.diff(_R(v0) - _R(k) * T, T) + _R(k) == 0, "dv/dx is not -k")
        if to_rest:
            text = f"{intro}. Find the greatest distance it could ever travel, in metres."
            wrong = [
                (v0 / (2 * k), "r3d-constant-deceleration-assumed"),
                (v0 * v0 / (2 * k), "r3d-kinetic-energy-style"),
                (1 / k, "r3d-speed-dropped"),
                (v0 * k, "r3d-rate-multiplied"),
            ]
            working = [
                f"a = v dv/dx = -{dec(k, 3)}v, so dv/dx = -{dec(k, 3)} and v = {dec(v0, 2)} - {dec(k, 3)}x.",
                f"v = 0 when x = {dec(v0, 2)}/{dec(k, 3)} = {dec(value, 3)} m.",
            ]
        else:
            text = f"{intro}. Find the distance travelled while the speed falls to {dec(v1, 2)} m/s, in metres."
            wrong = [
                ((v0 * v0 - v1 * v1) / (2 * k * v0), "r3d-constant-deceleration-assumed"),
                ((v0 * v0 - v1 * v1) / (2 * k), "r3d-kinetic-energy-style"),
                (v1 / k, "r3d-final-speed-for-change"),
                (v0 / k, "r3d-stopping-distance-for-partial"),
            ]
            working = [
                f"a = v dv/dx = -{dec(k, 3)}v, so dv/dx = -{dec(k, 3)} and v = {dec(v0, 2)} - {dec(k, 3)}x.",
                f"v = {dec(v1, 2)} when x = ({dec(v0, 2)} - {dec(v1, 2)})/{dec(k, 3)} = {dec(value, 3)} m.",
            ]
        del vx, value_dp
        return dict(text=text, working=working, **number_fields(value, dp if dp else 2, wrong, nonneg=True))
    for _ in attempts(500, "a clear rational answer"):
        v0, k = _quadratic(rng)
        if variant == "quadratic_speed":
            t = F(rng.choice((2, 3, 4, 5, 6, 8, 10)))
            value = v0 / (1 + k * v0 * t)
        else:
            frac = rng.choice((F(1, 2), F(1, 4), F(1, 5), F(2, 5)))
            v1 = v0 * frac
            if (v1 * 10).denominator != 1:
                continue
            value = (1 / v1 - 1 / v0) / k
        if clear_of_boundary(value, 2):
            break
    intro = _cyc(rng, v0, k)
    vt = _R(v0) / (1 + _R(k) * _R(v0) * _t)
    check(
        residual_zero(sympy.diff(vt, _t) + _R(k) * vt**2, _t) and vt.subs(_t, 0) == _R(v0),
        "v does not solve dv/dt = -kv²",
    )
    if variant == "quadratic_speed":
        text = f"{intro}. Find the speed {secs(t)} later, in metres per second, correct to 2 decimal places."
        wrong = [
            (exact(sympy.N(_R(v0) * sympy.exp(-_R(k) * _R(t)), 40)), "r3d-linear-law-used"),
            (v0 - k * v0 * v0 * t, "r3d-constant-deceleration-assumed"),
            (v0 / (1 + k * t), "r3d-initial-speed-dropped"),
            (v0 / (1 + k * v0 * t) ** 2, "r3d-denominator-squared"),
        ]
        working = [
            f"a = dv/dt = -{dec(k, 4)}v², so dv/v² = -{dec(k, 4)} dt and -1/v = -{dec(k, 4)}t + C.",
            f"1/v = 1/{dec(v0, 2)} + {dec(k, 4)}t, so v = {dec(v0, 2)}/(1 + {dec(k * v0, 4)}t) = {float(value):.2f} m/s at t = {int(t)}.",
        ]
    else:
        text = (
            f"{intro}. Find the time taken for the speed to fall to {dec(v1, 2)} m/s, in seconds, correct to "
            f"2 decimal places."
        )
        wrong = [
            (exact(sympy.N(sympy.log(_R(v0 / v1)) / _R(k), 40)), "r3d-linear-law-used"),
            ((v0 - v1) / (k * v0 * v0), "r3d-constant-deceleration-assumed"),
            (1 / v1 - 1 / v0, "r3d-rate-dropped"),
            ((v0 - v1) / (k * v1), "r3d-final-speed-only"),
        ]
        working = [
            f"1/v = 1/{dec(v0, 2)} + {dec(k, 4)}t, so t = (1/{dec(v1, 2)} - 1/{dec(v0, 2)})/{dec(k, 4)}.",
            f"t = {float(value):.2f} s.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong, nonneg=True))


# ------------------------------------------------------------------------------------------------ hard


def _hard(rng):
    variant = rng.choice(
        ("quad_distance", "quad_speed_after_distance", "lin_distance_in_time", "quad_time_for_distance")
    )
    for _ in attempts(500, "a clear value"):
        if variant == "lin_distance_in_time":
            v0, k = _linear(rng)
            t = F(rng.choice((2, 3, 4, 5, 6, 8)))
            expr = _R(v0) / _R(k) * (1 - sympy.exp(-_R(k) * _R(t)))
        else:
            v0, k = _quadratic(rng)
            t = F(rng.choice((2, 4, 5, 6, 8, 10)))
            d = F(rng.choice((10, 20, 25, 30, 40, 50, 60, 80, 100)))
            if variant == "quad_distance":
                expr = sympy.log(1 + _R(k) * _R(v0) * _R(t)) / _R(k)
            elif variant == "quad_speed_after_distance":
                expr = _R(v0) * sympy.exp(-_R(k) * _R(d))
            else:
                expr = (sympy.exp(_R(k) * _R(d)) - 1) / (_R(k) * _R(v0))
        value = exact(sympy.N(expr, 40))
        if clear_of_boundary(value, 2) and value > F(1, 2):
            break
    if variant == "lin_distance_in_time":
        intro = _fmt(rng, _BOAT, v0=dec(v0, 2), k=dec(k, 3))
        text = f"{intro}. Find the distance travelled in the first {secs(t)}, in metres, correct to 2 decimal places."
        vt = _R(v0) * sympy.exp(-_R(k) * _t)
        check(abs(simpson(vt, _t, 0, t) - float(value)) < 1e-8, "distance differs from the integral of v")
        wrong = [
            (v0 * t, "r3d-no-deceleration"),
            (v0 / k, "r3d-stopping-distance-for-partial"),
            (exact(sympy.N(_R(v0) * (1 - sympy.exp(-_R(k) * _R(t))), 40)), "r3d-rate-dropped"),
            (exact(sympy.N(_R(v0) * _R(t) * sympy.exp(-_R(k) * _R(t)), 40)), "r3d-speed-times-time"),
        ]
        working = [
            f"v = {dec(v0, 2)}e^(-{dec(k, 3)}t), so x = ∫v dt = {dec(v0 / k, 3)}(1 - e^(-{dec(k, 3)}t)).",
            f"At t = {int(t)}, x = {float(value):.2f} m.",
        ]
    elif variant == "quad_distance":
        intro = _cyc(rng, v0, k)
        text = f"{intro}. Find the distance travelled in the first {secs(t)}, in metres, correct to 2 decimal places."
        vt = _R(v0) / (1 + _R(k) * _R(v0) * _t)
        check(abs(simpson(vt, _t, 0, t) - float(value)) < 1e-8, "distance differs from the integral of v")
        wrong = [
            (exact(sympy.N(sympy.log(1 + _R(k) * _R(v0) * _R(t)), 40)), "r3d-rate-dropped"),
            (v0 * t, "r3d-no-deceleration"),
            (exact(sympy.N(_R(v0) / (1 + _R(k) * _R(v0) * _R(t)) * _R(t), 40)), "r3d-speed-times-time"),
            (exact(sympy.N(sympy.log(1 + _R(k) * _R(t)) / _R(k), 40)), "r3d-initial-speed-dropped"),
        ]
        working = [
            f"v = {dec(v0, 2)}/(1 + {dec(k * v0, 4)}t), so x = ∫v dt = (1/{dec(k, 4)}) ln(1 + {dec(k * v0, 4)}t).",
            f"At t = {int(t)}, x = {float(value):.2f} m.",
        ]
    elif variant == "quad_speed_after_distance":
        intro = _cyc(rng, v0, k)
        text = f"{intro}. Find the speed after a distance of {int(d)} metres, in metres per second, correct to 2 decimal places."
        check(
            residual_zero(_R(v0) * sympy.diff(sympy.exp(-_R(k) * T), T) / sympy.exp(-_R(k) * T) + _R(k) * _R(v0), T),
            "dv/dx is not -kv",
        )
        wrong = [
            (v0 / (1 + k * v0 * d), "r3d-distance-used-as-time"),
            (v0 - k * v0 * d, "r3d-constant-deceleration-assumed"),
            (exact(sympy.N(_R(v0) * sympy.exp(_R(k) * _R(d)), 40)), "r3d-exponent-sign"),
            (exact(sympy.N(_R(v0) * sympy.exp(-_R(d)), 40)), "r3d-rate-dropped"),
        ]
        working = [
            f"a = v dv/dx = -{dec(k, 4)}v², so dv/dx = -{dec(k, 4)}v and ln v = -{dec(k, 4)}x + C.",
            f"v = {dec(v0, 2)}e^(-{dec(k, 4)}x); at x = {int(d)}, v = {float(value):.2f} m/s.",
        ]
    else:
        intro = _cyc(rng, v0, k)
        text = f"{intro}. Find the time taken to travel {int(d)} metres, in seconds, correct to 2 decimal places."
        wrong = [
            (F(int(d)) / v0, "r3d-no-deceleration"),
            (exact(sympy.N(sympy.log(1 + _R(k) * _R(d)) / (_R(k) * _R(v0)), 40)), "r3d-log-for-exponential"),
            (exact(sympy.N((sympy.exp(_R(k) * _R(d)) - 1) / _R(k), 40)), "r3d-initial-speed-dropped"),
            (exact(sympy.N(sympy.exp(_R(k) * _R(d)) / (_R(k) * _R(v0)), 40)), "r3d-constant-dropped"),
        ]
        working = [
            f"v = {dec(v0, 2)}e^(-{dec(k, 4)}x), so dt/dx = 1/v = e^({dec(k, 4)}x)/{dec(v0, 2)}.",
            f"t = ∫dx/v = (e^({dec(k, 4)}x) - 1)/({dec(k, 4)} × {dec(v0, 2)}); at x = {int(d)}, t = {float(value):.2f} s.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong, nonneg=True))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("r3d_accel_velocity", level=18, presets=_PRESETS)
def r3d_accel_velocity(rng, tier: str = "easy"):
    """a = -kv and a = -kv²: speed against time, time to a speed, distance to stop, then distance, speed after a distance."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam


@generator("r3d_accel_velocity_exam", level=18, presets={"exam": {}})
def r3d_accel_velocity_exam(rng):
    """a boat (a = -kv) or a cyclist (a = -kv²): v(t), the speed at a time, the distance to it, then a limiting quantity."""
    linear = rng.random() < 0.5
    if linear:
        for _ in attempts(500, "a clear exponential chain"):
            v0 = F(rng.choice((6, 8, 10, 12)))
            k = F(rng.choice((F(1, 5), F(1, 4), F(2, 5), F(1, 2))))
            t1 = F(rng.choice((2, 3, 4, 5, 6)))
            v1 = exact(sympy.N(_R(v0) * sympy.exp(-_R(k) * _R(t1)), 40))
            dist = (v0 - v1) / k
            limit = v0 / k
            if (
                clear_of_boundary(v1, 2)
                and clear_of_boundary(dist, 1)
                and limit.denominator == 1
                and (k * v0).denominator != 1
            ):
                break
        vt = _R(v0) * sympy.exp(-_R(k) * _t)
        check(
            residual_zero(sympy.diff(vt, _t) + _R(k) * vt, _t) and vt.subs(_t, 0) == _R(v0),
            "v does not solve dv/dt = -kv",
        )
        check(abs(simpson(vt, _t, 0, t1) - float(dist)) < 1e-8, "distance differs from ∫v dt")
        intro = _fmt(rng, _BOAT, v0=dec(v0, 2), k=dec(k, 3))
        ans_text = _exp_text(v0, k)
        wrong_a = [
            (_R(v0) * sympy.exp(_R(k) * _t), _exp_text(v0, -k), "r3d-exponent-sign"),
            (
                _R(v0) - _R(k) * _R(v0) * _t,
                _lin_text(v0, -(k * v0)),
                "r3d-constant-deceleration-assumed",
            ),
            (_R(v0 * k) * sympy.exp(-_R(k) * _t), _exp_text(v0 * k, k), "r3d-acceleration-for-speed"),
            (_R(v0 / k) * sympy.exp(-_R(k) * _t), _exp_text(v0 / k, k), "r3d-amplitude-divided"),
        ]
        part_a = expression_part(
            "Find an expression for the speed v, in metres per second, at time t seconds after the resistance starts to act.",
            vt,
            ans_text,
            3,
            [f"dv/dt = -{dec(k, 3)}v, so ln v = -{dec(k, 3)}t + C and v = {ans_text}, using v(0) = {dec(v0, 2)}."],
            wrong_a,
            _t,
        )
        part_b = number_part(
            f"Find the speed when t = {int(t1)}, in metres per second, correct to 2 decimal places.",
            v1,
            2,
            1,
            [f"v = {dec(v0, 2)}e^(-{dec(k, 3)} × {int(t1)}) = {float(v1):.2f} m/s."],
            [
                (exact(sympy.N(_R(v0) * sympy.exp(_R(k) * _R(t1)), 40)), "r3d-exponent-sign"),
                (v0 - k * v0 * t1, "r3d-constant-deceleration-assumed"),
                (exact(sympy.N(_R(v0) * sympy.exp(-_R(t1)), 40)), "r3d-rate-dropped"),
                (v0 * k * t1, "r3d-rate-multiplied"),
            ],
            sensible=(0, None),
            nonneg=True,
        )
        part_c = number_part(
            f"Find the distance travelled in the first {secs(t1)}, in metres, correct to 1 decimal place.",
            dist,
            1,
            2,
            [
                "Writing a = v dv/dx = -kv gives dv/dx = -k, so the speed falls by k for every metre.",
                f"Distance = (v0 - v)/k = ({dec(v0, 2)} - {float(v1):.2f})/{dec(k, 3)} = {float(dist):.1f} m.",
            ],
            [
                (v0 * t1, "r3d-no-deceleration"),
                (v0 / k, "r3d-stopping-distance-for-partial"),
                ((v0 + v1) / 2 * t1, "r3d-mean-speed-used"),
                ((v0 - v1) * k, "r3d-rate-multiplied"),
                (v0 - v1, "r3d-speed-change-for-distance"),
                (v1 * t1, "r3d-speed-times-time"),
            ],
            follow=f"({dec(v0, 2)} - part_b)/{dec(k, 3)}",
            sensible=(0, None),
            nonneg=True,
        )
        part_d = number_part(
            "Find the greatest distance that could ever be travelled, to the nearest metre.",
            limit,
            0,
            1,
            [
                "The remaining distance to rest is v/k, so the total is the distance so far plus v/k = v0/k.",
                f"Limit = {dec(v0, 2)}/{dec(k, 3)} = {int(limit)} m.",
            ],
            [
                (limit / 2, "r3d-constant-deceleration-assumed"),
                (limit * k, "r3d-rate-multiplied"),
                (dist, "r3d-distance-so-far-for-limit"),
                (v0 * v0 / (2 * k), "r3d-kinetic-energy-style"),
            ],
            follow=f"part_c + part_b/{dec(k, 3)}",
            sensible=(0, None),
            nonneg=True,
        )
        return dict(text=intro + ".", parts=[part_a, part_b, part_c, part_d])
    for _ in attempts(500, "a clear rational chain"):
        v0 = F(rng.choice((10, 12, 15, 20, 25)))
        k = F(rng.choice((F(1, 100), F(1, 50), F(1, 25), F(1, 20))))
        t1 = F(rng.choice((2, 4, 5, 6, 8, 10)))
        v1 = v0 / (1 + k * v0 * t1)
        dist = exact(sympy.N(sympy.log(1 + _R(k) * _R(v0) * _R(t1)) / _R(k), 40))
        half = 1 / (k * v0)
        if (
            clear_of_boundary(v1, 2)
            and (k * v0).denominator != 1
            and clear_of_boundary(dist, 0)
            and half.denominator in (1, 2, 4, 5, 10, 20, 25, 50, 100)
            and clear_of_boundary(half, 2)
        ):
            break
    vt = _R(v0) / (1 + _R(k) * _R(v0) * _t)
    check(
        residual_zero(sympy.diff(vt, _t) + _R(k) * vt**2, _t) and vt.subs(_t, 0) == _R(v0),
        "v does not solve dv/dt = -kv²",
    )
    check(abs(simpson(vt, _t, 0, t1) - float(dist)) < 1e-8, "distance differs from ∫v dt")
    intro = _cyc(rng, v0, k)
    kv = k * v0
    ans_text = f"{dec(v0, 2)}/(1 + {dec(kv, 4)}t)"
    wrong_a = [
        (_R(v0) * sympy.exp(-_R(k) * _t), _exp_text(v0, k), "r3d-linear-law-used"),
        (_R(v0) / (1 + _R(k) * _t), f"{dec(v0, 2)}/(1 + {dec(k, 4)}t)", "r3d-initial-speed-dropped"),
        (_R(v0) / (1 + _R(kv) * _t) ** 2, f"{dec(v0, 2)}/(1 + {dec(kv, 4)}t)²", "r3d-denominator-squared"),
        (_R(v0) / (1 - _R(kv) * _t), f"{dec(v0, 2)}/(1 - {dec(kv, 4)}t)", "r3d-constant-sign"),
    ]
    part_a = expression_part(
        "Find an expression for the speed v, in metres per second, at time t seconds after the resistance starts to act.",
        vt,
        ans_text,
        3,
        [
            f"dv/dt = -{dec(k, 4)}v², so -1/v = -{dec(k, 4)}t + C and 1/v = 1/{dec(v0, 2)} + {dec(k, 4)}t.",
            f"v = {ans_text}.",
        ],
        wrong_a,
        _t,
    )
    part_b = number_part(
        f"Find the speed when t = {int(t1)}, in metres per second, correct to 2 decimal places.",
        v1,
        2,
        1,
        [f"v = {dec(v0, 2)}/(1 + {dec(kv, 4)} × {int(t1)}) = {float(v1):.2f} m/s."],
        [
            (exact(sympy.N(_R(v0) * sympy.exp(-_R(k) * _R(t1)), 40)), "r3d-linear-law-used"),
            (v0 / (1 + k * t1), "r3d-initial-speed-dropped"),
            (v0 - k * v0 * v0 * t1, "r3d-constant-deceleration-assumed"),
            (v0 / (1 + kv * t1) ** 2, "r3d-denominator-squared"),
        ],
        follow=f"Subs(part_a, (t,), ({int(t1)},))",
        sensible=(0, None),
        nonneg=True,
    )
    part_c = number_part(
        f"Find the distance travelled in the first {secs(t1)}, to the nearest metre.",
        dist,
        0,
        2,
        [
            "Writing a = v dv/dx = -kv² gives dv/dx = -kv, so v = v0 e^(-kx) and x = (1/k) ln(v0/v).",
            f"x = ln({dec(v0, 2)}/{float(v1):.2f})/{dec(k, 4)} = {float(dist):.0f} m.",
        ],
        [
            (v0 * t1, "r3d-no-deceleration"),
            (exact(sympy.N(sympy.log(_R(v0 / v1)), 40)), "r3d-rate-dropped"),
            ((v0 + v1) / 2 * t1, "r3d-mean-speed-used"),
            (exact(sympy.N(sympy.log(_R(v0 / v1)) * _R(k), 40)), "r3d-rate-multiplied"),
            (v1 * t1, "r3d-speed-times-time"),
            (exact(sympy.N(sympy.log(_R(v0 / v1)) / _R(k) * 2, 40)), "r3d-doubled"),
        ],
        follow=f"log({dec(v0, 2)}/part_b)/{dec(k, 4)}",
        sensible=(0, None),
        nonneg=True,
    )
    part_d = number_part(
        "Find the time taken for the speed to fall to half its initial value, in seconds, correct to 2 decimal places.",
        half,
        2,
        2,
        [
            f"1/v = 1/{dec(v0, 2)} + {dec(k, 4)}t. Setting v = {dec(v0 / 2, 3)} gives 2/{dec(v0, 2)} = 1/{dec(v0, 2)} + {dec(k, 4)}t.",
            f"t = 1/({dec(k, 4)} × {dec(v0, 2)}) = {float(half):.2f} s.",
        ],
        [
            (exact(sympy.N(sympy.log(2) / _R(k), 40)), "r3d-linear-law-used"),
            (1 / k, "r3d-initial-speed-dropped"),
            (half / 2, "r3d-half-time-for-half-speed"),
            (half * 2, "r3d-doubled"),
        ],
        nonneg=True,
    )
    return dict(text=intro + ".", parts=[part_a, part_b, part_c, part_d])
