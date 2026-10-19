"""Simple harmonic motion by calculus: `r3d_shm_calculus` (easy, medium, hard) and `r3d_shm_calculus_exam`.

Skill: shm-displacement-velocity-calculus. The displacement of a particle in simple harmonic motion is differentiated
once for the velocity and twice for the acceleration, and the pieces (n, the amplitude, the greatest speed and the
greatest acceleration) are read from the result. The neighbouring `simple-harmonic-angular-frequency` skill only asks
for omega from a spring constant and a mass; here the displacement function is the starting point.

Tiers differ in the decision they ask for, not in the size of the numbers:

* easy: x = A cos(nt) or A sin(nt). Differentiate (with the factor n from the chain rule) and evaluate the velocity or
  the acceleration at a time that is a multiple of pi, or write the velocity function itself.
* medium: the displacement is not in the form A cos(nt) yet, or n and A are hidden. A sum p sin(nt) + q cos(nt) has
  to be recognised as an amplitude sqrt(p² + q²) before the greatest speed is n times it; the period comes from the
  greatest acceleration and the amplitude; the amplitude (or period) comes from the greatest speed and the greatest
  acceleration together.
* hard: x = a cos(nt + e) with a phase. Find the phase from the position and velocity at t = 0 (the quadrant has
  to be decided), the greatest acceleration from the same two readings, or the first time the particle is at rest.
* exam: x = A cos(nt + e) with e = ±pi/6 or ±5pi/6. (a) the constant k in x'' = -kx, (b) the period, (c) the greatest
  speed, (d) the velocity at t = 0, each following through from the one before.

## Construction

Times are multiples of pi, chosen so that sin and cos take the values 1/2, 1 or -1/2, -1, so every velocity and
acceleration is an exact multiple of A, n and n². Each value is also recomputed with SymPy (differentiate, then
substitute), a different route from the closed form it is written from.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, phrase
from ._r3d_kit import (
    T,
    clear_of_boundary,
    dec,
    exact,
    fit_dp,
    number_fields,
    number_part,
    phase_text,
    pi_text,
)

F = Fraction
_HALF = F(1, 2)

# angle (as a multiple of pi) -> sin or cos value, only where the value is a non-zero rational
_SIN = {F(1, 6): _HALF, F(1, 2): F(1), F(5, 6): _HALF, F(7, 6): -_HALF, F(3, 2): F(-1), F(11, 6): -_HALF}
_COS = {F(1, 3): _HALF, F(2, 3): -_HALF, F(1): F(-1), F(4, 3): -_HALF, F(5, 3): _HALF, F(0): F(1)}

#: (sentence naming x, amplitudes, values of n)
_CONTEXTS = (
    (
        "A buoy rises and falls with the swell. Its height above the mean water level is x metres at time t seconds",
        (F(2, 5), F(3, 5), F(4, 5), F(6, 5), F(3, 2), F(2)),
        (1, 2, F(3, 2)),
    ),
    (
        "A mass hanging from a spring oscillates vertically. Its displacement from the equilibrium position is x "
        "metres at time t seconds",
        (F(1, 20), F(2, 25), F(1, 10), F(3, 25), F(3, 20), F(1, 5)),
        (4, 5, 6, 8, 10),
    ),
    (
        "The head of a piston moves to and fro in a cylinder. Its displacement from the middle of the stroke is x "
        "metres at time t seconds",
        (F(1, 25), F(1, 20), F(3, 50)),
        (50, 60, 80, 100),
    ),
)


def _coef(c) -> str:
    """A coefficient for an expression: nothing for 1, '-' for -1, else the exact decimal."""
    c = F(c)
    if c == 1:
        return ""
    if c == -1:
        return "-"
    return dec(c, 4) + " "


def _arg(n) -> str:
    n = F(n)
    return "t" if n == 1 else f"{dec(n, 2)}t"


def _trig(c, name: str, n, phase: str = "") -> str:
    """'0.8 cos(3t)' with the coefficient and frequency written as a student would."""
    arg = _arg(n) if not phase else f"{_arg(n)} {phase}"
    return f"{_coef(c)}{name}({arg})".strip()


def _fn_text(c, name, n) -> str:
    return _trig(c, name, n)


def _sym_trig(c, name, n):
    f = sympy.cos if name == "cos" else sympy.sin
    return sympy.Rational(c.numerator, c.denominator) * f(sympy.Rational(n.numerator, n.denominator) * T)


def _bracketed(value) -> str:
    """A decimal with its minus sign in brackets, so a square of it reads correctly."""
    text = dec(value, 3)
    return f"({text})" if value < 0 else text


def _draw_motion(rng):
    sentence, amps, ns = rng.choice(_CONTEXTS)
    return sentence, F(rng.choice(amps)), F(rng.choice(ns))


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    variant = rng.choice(("velocity", "acceleration"))
    name = rng.choice(("cos", "sin"))
    for _ in attempts(500, "an amplitude, frequency and time with an exact answer"):
        sentence, amp, n = _draw_motion(rng)
        own = _COS if name == "cos" else _SIN
        other = _SIN if name == "cos" else _COS
        # d/dt of A cos(nt) is -A n sin(nt); of A sin(nt) is A n cos(nt). Second derivative is -n² x either way.
        table = other if variant == "velocity" else own
        theta = rng.choice(
            sorted(k for k in table if abs(table[k]) == _HALF)
        )  # a value of ±1/2 keeps the slips distinct
        trig_t = table[theta]
        if variant == "velocity":
            value = (-amp * n * trig_t) if name == "cos" else (amp * n * trig_t)
        else:
            value = -amp * n * n * trig_t
        dp = fit_dp(value, 1, 3)
        if dp is None or value == 0:
            continue
        break
    fn = _fn_text(amp, name, n)
    t0 = theta / n
    A, N = sympy.Rational(amp.numerator, amp.denominator), sympy.Rational(n.numerator, n.denominator)
    x = A * (sympy.cos(N * T) if name == "cos" else sympy.sin(N * T))
    # a number
    t_text = pi_text(t0)
    if variant == "velocity":
        v_sym = sympy.diff(x, T).subs(T, sympy.pi * sympy.Rational(t0.numerator, t0.denominator))
        check(exact(sympy.simplify(v_sym)) == value, "velocity differs from the SymPy derivative")
        sign = -1 if name == "cos" else 1
        same = exact(
            sympy.N(
                sign
                * A
                * N
                * (sympy.cos if name == "cos" else sympy.sin)(sympy.pi * theta.numerator / theta.denominator),
                40,
            )
        )
        wrong = [
            (value / n, "r3d-chain-factor-missing"),
            (-value, "r3d-derivative-sign"),
            (same, "r3d-trig-not-switched"),
            (value * n, "r3d-chain-factor-repeated"),
        ]
        working = [
            f"v = dx/dt = {_trig(sign * amp * n, 'sin' if name == 'cos' else 'cos', n)}.",
            f"At t = {t_text}, {dec(n, 2)}t = {pi_text(theta)}, so v = {dec(value, 3)} m/s.",
        ]
        ask = "Find the velocity of the object when t = " + t_text + " seconds, in metres per second."
    else:
        a_sym = sympy.diff(x, T, 2).subs(T, sympy.pi * sympy.Rational(t0.numerator, t0.denominator))
        check(exact(sympy.simplify(a_sym)) == value, "acceleration differs from the SymPy second derivative")
        first = exact(sympy.N(sympy.diff(x, T).subs(T, sympy.pi * sympy.Rational(t0.numerator, t0.denominator)), 40))
        wrong = [
            (-amp * n * trig_t, "r3d-chain-factor-missing"),
            (amp * n * n * trig_t, "r3d-derivative-sign"),
            (first, "r3d-stopped-after-first-derivative"),
            (amp * trig_t, "r3d-evaluated-displacement"),
        ]
        working = [
            f"a = d²x/dt² = {_trig(-amp * n * n, name, n)}.",
            f"At t = {t_text}, {dec(n, 2)}t = {pi_text(theta)}, so a = {dec(value, 3)} m/s².",
        ]
        ask = "Find the acceleration of the object when t = " + t_text + " seconds, in metres per second squared."
    text = f"{sentence}, where x = {fn}. {ask}"
    return dict(text=text, working=working, **number_fields(value, dp, wrong))


# ------------------------------------------------------------------------------------------------ medium

_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 6, 10), (15, 8, 17), (12, 5, 13), (4, 3, 5), (6, 8, 10), (7, 24, 25))


def _medium(rng):
    variant = rng.choice(
        ("sum_speed", "sum_accel", "period_from_accel", "amplitude_from_extremes", "period_from_extremes")
    )
    sentence_only, _, _ = _draw_motion(rng)
    if variant in ("sum_speed", "sum_accel"):
        for _ in attempts(500, "a sum of sine and cosine with an exact answer"):
            p0, q0, r0 = rng.choice(_TRIPLES)
            scale = rng.choice((F(1, 10), F(1, 5), F(1, 2)))
            n = rng.choice((2, 3, 4, 5, 6)) if scale >= F(1, 5) else rng.choice((4, 5, 8, 10))
            sp, sq = rng.choice((1, -1)), rng.choice((1, -1))
            p, q = sp * p0 * scale, sq * q0 * scale
            amp = r0 * scale
            value = n * amp if variant == "sum_speed" else n * n * amp
            dp = fit_dp(value, 1, 3)
            if dp is not None:
                break
        order = rng.choice(("sin_first", "cos_first"))
        terms = [(p, "sin"), (q, "cos")] if order == "sin_first" else [(q, "cos"), (p, "sin")]
        body = ""
        for c, name in terms:
            piece = _trig(abs(c), name, n)
            body += (("-" + piece) if c < 0 else piece) if not body else (" - " if c < 0 else " + ") + piece
        # second route: the largest value of |x'| (or |x''|) found by sampling one period
        top = 0.0
        for i in range(2000):
            tt = 2 * math.pi * i / (2000 * n)
            xp = n * (float(p) * math.cos(n * tt) - float(q) * math.sin(n * tt))
            xpp = -n * n * (float(p) * math.sin(n * tt) + float(q) * math.cos(n * tt))
            top = max(top, abs(xp if variant == "sum_speed" else xpp))
        check(abs(top - float(value)) / float(value) < 1e-5, "sampled maximum disagrees with n·R")
        if variant == "sum_speed":
            ask = "Find the greatest speed of the particle, in metres per second."
            wrong = [
                (n * (abs(p) + abs(q)), "r3d-amplitudes-added"),
                (n * max(abs(p), abs(q)), "r3d-one-term-only"),
                (amp, "r3d-chain-factor-missing"),
                (n * n * amp, "r3d-acceleration-for-speed"),
            ]
            working = [
                f"x is a sine wave of amplitude √({dec(abs(p), 3)}² + {dec(abs(q), 3)}²) = {dec(amp, 3)}.",
                f"Greatest speed = n × amplitude = {n} × {dec(amp, 3)} = {dec(value, 3)} m/s.",
            ]
        else:
            ask = "Find the greatest acceleration of the particle, in metres per second squared."
            wrong = [
                (n * n * (abs(p) + abs(q)), "r3d-amplitudes-added"),
                (n * n * max(abs(p), abs(q)), "r3d-one-term-only"),
                (n * amp, "r3d-speed-for-acceleration"),
                (n * amp * n * n, "r3d-omega-cubed"),
            ]
            working = [
                f"x has amplitude √({dec(abs(p), 3)}² + {dec(abs(q), 3)}²) = {dec(amp, 3)}.",
                f"Greatest acceleration = n² × amplitude = {n}² × {dec(amp, 3)} = {dec(value, 3)} m/s².",
            ]
        text = f"A particle moves along a straight line so that its displacement from O is x = {body} metres at time t seconds. {ask}"
        return dict(text=text, working=working, **number_fields(value, dp, wrong))
    if variant == "period_from_accel":
        for _ in attempts(500, "an amplitude and frequency"):
            amp = F(rng.choice((F(1, 5), F(1, 4), F(1, 2), F(3, 5), F(4, 5), F(3, 2), 2)))
            n = rng.choice((2, 3, 4, 5, 6))
            q = amp * n * n
            if fit_dp(q, 1, 3) is not None:
                break
        value = 2 * sympy.pi / n
        check(abs(exact(value) - F(2 * math.pi / n).limit_denominator(10**9)) < F(1, 10**8), "period route disagrees")
        check(clear_of_boundary(value, 2), "period too near a rounding boundary")
        text = (
            f"A particle moves with displacement x = {dec(amp, 3)} cos(nt) metres at time t seconds, where n is a "
            f"positive constant. Its greatest acceleration is {dec(q, 3)} m/s². Find the period of the motion, in "
            f"seconds, correct to 2 decimal places."
        )
        wrong = [
            (2 * sympy.pi / (q / amp), "r3d-omega-squared-for-omega"),
            (sympy.pi / n, "r3d-half-period"),
            (F(n) / (2 * sympy.pi), "r3d-frequency-for-period"),
            (2 * sympy.pi * n, "r3d-period-inverted"),
        ]
        working = [
            f"a = -{dec(amp, 3)}n² cos(nt), so the greatest acceleration is {dec(amp, 3)}n² = {dec(q, 3)}.",
            f"n² = {dec(q / amp, 3)}, so n = {n}.",
            f"Period = 2π/n = {float(value):.2f} s.",
        ]
        return dict(text=text, working=working, **number_fields(value, 2, wrong))
    # amplitude or period from the greatest speed and greatest acceleration
    for _ in attempts(500, "an amplitude and frequency"):
        amp = F(rng.choice((F(1, 5), F(1, 4), F(2, 5), F(1, 2), F(4, 5), F(3, 2), 2, F(5, 2))))
        n = rng.choice((F(1, 2), 2, 3, 4, 5, F(3, 2)))
        v_max, a_max = amp * n, amp * n * n
        if fit_dp(v_max, 1, 3) is not None and fit_dp(a_max, 1, 3) is not None:
            break
    given = (
        f"A particle moves in simple harmonic motion about O. Its greatest speed is {dec(v_max, 3)} m/s and its "
        f"greatest acceleration is {dec(a_max, 3)} m/s²."
    )
    if variant == "amplitude_from_extremes":
        value = amp
        check(v_max**2 / a_max == amp, "amplitude route disagrees")
        dp = fit_dp(value, 1, 3)
        wrong = [
            (v_max / a_max, "r3d-period-ratio-for-amplitude"),
            (a_max / v_max, "r3d-omega-for-amplitude"),
            (v_max * v_max / (a_max * a_max), "r3d-ratio-squared-twice"),
            (v_max, "r3d-speed-for-amplitude"),
            (a_max, "r3d-acceleration-for-amplitude"),
            (v_max / n / n, "r3d-omega-twice-divided"),
        ]
        text = given + " Find the amplitude of the motion, in metres."
        working = [
            "Greatest speed V = an and greatest acceleration Q = an², so n = Q/V.",
            f"n = {dec(a_max / v_max, 3)} and the amplitude a = V/n = {dec(value, 3)} m.",
        ]
        return dict(text=text, working=working, **number_fields(value, dp, wrong))
    value = 2 * sympy.pi * v_max / a_max
    check(abs(exact(value) - F(2 * math.pi) * (v_max / a_max)) < F(1, 10**8), "period route disagrees")
    check(clear_of_boundary(value, 2), "period too near a rounding boundary")
    wrong = [
        (2 * sympy.pi * a_max / v_max, "r3d-period-inverted"),
        (sympy.pi * v_max / a_max, "r3d-half-period"),
        (2 * sympy.pi * v_max * v_max / (a_max * a_max), "r3d-ratio-squared-twice"),
        (v_max / a_max, "r3d-omega-for-period"),
        (2 * sympy.pi * v_max, "r3d-period-from-speed-only"),
        (2 * sympy.pi / v_max, "r3d-period-from-speed-only"),
    ]
    text = given + " Find the period of the motion, in seconds, correct to 2 decimal places."
    working = [
        "Greatest speed V = an and greatest acceleration Q = an², so n = Q/V.",
        f"n = {dec(a_max / v_max, 3)} and the period = 2π/n = {float(value):.2f} s.",
    ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong))


# ------------------------------------------------------------------------------------------------ hard


def _hard(rng):
    variant = rng.choice(("phase", "greatest_acceleration", "first_rest"))
    if variant == "phase":
        for _ in attempts(1000, "a start position and velocity with a clear phase"):
            n = rng.choice((2, 3, 4, 5))
            x0 = F(rng.randint(-6, 6), 10)
            u = F(rng.randint(-6, 6), 10)  # -v0/n
            if x0 == 0 or u == 0:
                continue
            v0 = -n * u
            eps = math.atan2(float(u), float(x0))
            if not clear_of_boundary(F(eps).limit_denominator(10**12), 3):
                continue
            break
        value = sympy.atan2(sympy.Rational(u.numerator, u.denominator), sympy.Rational(x0.numerator, x0.denominator))
        check(abs(float(sympy.N(value, 20)) - eps) < 1e-12, "phase routes disagree")
        # the second route: put the phase back and recover x(0) and v(0)
        amp_val = math.hypot(float(x0), float(u))
        check(
            abs(amp_val * math.cos(eps) - float(x0)) < 1e-12 and abs(-amp_val * n * math.sin(eps) - float(v0)) < 1e-12,
            "phase does not reproduce x(0), v(0)",
        )
        wrong = [
            (math.atan(float(u / x0)), "r3d-quadrant-ignored"),
            (math.atan2(-float(u), float(x0)), "r3d-phase-sign"),
            (math.atan2(float(-v0), float(x0)), "r3d-chain-factor-missing"),
            (math.atan2(float(x0), float(u)), "r3d-sin-cos-swapped"),
        ]
        wrong = [(F(w).limit_denominator(10**12), tag) for w, tag in wrong]
        text = (
            f"A particle moves along a straight line with displacement x = a cos({n}t + ε) metres at time t seconds, "
            f"where a > 0 and -π < ε ≤ π. At t = 0 the particle is at x = {dec(x0, 2)} and its velocity is "
            f"{dec(v0, 2)} m/s. Find ε in radians, correct to 3 decimal places."
        )
        working = [
            f"x(0) = a cos ε = {dec(x0, 2)} and v(0) = -{n}a sin ε = {dec(v0, 2)}, so a sin ε = {dec(u, 2)}.",
            f"tan ε = {dec(u, 2)}/{dec(x0, 2)}, and (cos ε, sin ε) has signs ({'+' if x0 > 0 else '-'}, {'+' if u > 0 else '-'}), so ε = {eps:.3f}.",
        ]
        return dict(text=text, working=working, **number_fields(F(eps).limit_denominator(10**12), 3, wrong))
    if variant == "greatest_acceleration":
        for _ in attempts(500, "a start position and velocity with an exact amplitude"):
            p0, q0, r0 = rng.choice(_TRIPLES)
            scale = rng.choice((F(1, 10), F(1, 5)))
            n = rng.choice((2, 3, 4, 5, 6))
            x0 = rng.choice((1, -1)) * p0 * scale
            u = rng.choice((1, -1)) * q0 * scale
            v0 = n * u
            amp = r0 * scale
            value = n * n * amp
            dp = fit_dp(value, 1, 3)
            if dp is not None:
                break
        check(
            sympy.sqrt(
                sympy.Rational(x0.numerator, x0.denominator) ** 2 + sympy.Rational(u.numerator, u.denominator) ** 2
            )
            == sympy.Rational(amp.numerator, amp.denominator),
            "amplitude route disagrees",
        )
        wrong = [
            (n * n * abs(x0), "r3d-start-value-for-greatest"),
            (n * n * (abs(x0) + abs(u)), "r3d-amplitudes-added"),
            (n * amp, "r3d-speed-for-acceleration"),
            (abs(v0) * n, "r3d-start-velocity-scaled"),
        ]
        text = (
            f"A particle moves along a straight line in simple harmonic motion with acceleration -{n * n}x m/s², where "
            f"x metres is its displacement from O. At t = 0 it is at x = {dec(x0, 3)} with velocity {dec(v0, 3)} m/s. "
            f"Find the greatest acceleration of the particle, in metres per second squared."
        )
        working = [
            f"n = {n}. The amplitude satisfies a² = x0² + (v0/n)² = {_bracketed(x0)}² + {_bracketed(u)}², so a = {dec(amp, 3)} m.",
            f"Greatest acceleration = n²a = {n * n} × {dec(amp, 3)} = {dec(value, 3)} m/s².",
        ]
        return dict(text=text, working=working, **number_fields(value, dp, wrong))
    # first time at rest
    phases = [F(k, d) for d, ks in ((6, (1, 2, 3, 4, 5)), (4, (1, 3)), (3, (1, 2))) for k in ks] + [
        F(-k, d) for d, ks in ((6, (1, 2, 3, 4, 5)), (4, (1, 3)), (3, (1, 2))) for k in ks
    ]
    phases = sorted(set(phases))
    for _ in attempts(500, "a phase and frequency with a clear first rest"):
        n = rng.choice((2, 3, 4, 5, 6))
        eps = rng.choice(phases)  # multiple of pi, in (-1, 1)
        first = (-eps if eps < 0 else 1 - eps) / n  # multiple of pi
        value = first * sympy.pi
        if clear_of_boundary(exact(value), 3):
            break
    amp = rng.choice((F(1, 2), F(3, 5), F(4, 5), F(3, 2), 2))
    v = lambda tt: -float(amp) * n * math.sin(n * tt + float(eps) * math.pi)  # noqa: E731
    tt = float(first) * math.pi
    check(
        abs(v(tt)) < 1e-9 and all(abs(v(tt * k / 50)) > 1e-9 for k in range(1, 50)),
        "first rest time disagrees with the velocity",
    )
    wrong = [
        (abs(eps) * sympy.pi / n, "r3d-phase-sign"),
        (sympy.pi / n, "r3d-phase-ignored"),
        ((1 - eps) * sympy.pi / n if eps < 0 else (1 + abs(eps)) * sympy.pi / n, "r3d-next-rest-taken"),
        (eps * sympy.pi / n if eps > 0 else (1 + eps) * sympy.pi / n, "r3d-first-positive-wrong-branch"),
    ]
    text = (
        f"A particle moves along a straight line with displacement x = {dec(amp, 2)} cos({n}t{phase_text(eps)}) metres at time t seconds. "
        f"Find the first time after t = 0 at which the particle is momentarily at rest, in seconds, correct to 3 decimal places."
    )
    working = [
        f"v = -{dec(amp * n, 3)} sin({n}t{phase_text(eps)}), which is zero when {n}t{phase_text(eps)} = kπ.",
        f"The smallest t > 0 has {n}t = {pi_text(first * n)}, so t = {float(first) * math.pi:.3f} s.",
    ]
    return dict(text=text, working=working, **number_fields(value, 3, wrong))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("r3d_shm_calculus", level=17, presets=_PRESETS)
def r3d_shm_calculus(rng, tier: str = "easy"):
    """Velocity and acceleration of x = A cos(nt) by differentiating, then amplitude, period, phase and rest times."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam

_EXAM_CONTEXTS = (
    (
        (
            "The height of a buoy above the mean water level is x = {A} cos({n}t{eps}) metres at time t seconds. "
            "Upward is the positive direction.",
            "A float on a lake moves up and down with the waves. Its height above the still-water level is "
            "x = {A} cos({n}t{eps}) metres at time t seconds, with upward positive.",
        ),
        (F(2, 5), F(3, 5), F(4, 5), F(6, 5), F(3, 2), F(2)),
        (2, 3, 4, 5),
    ),
    (
        (
            "A mass hangs from a spring and oscillates vertically. Its height above the equilibrium position is "
            "x = {A} cos({n}t{eps}) metres at time t seconds, with upward positive.",
            "A load on the end of a spring bobs up and down. Its height above its rest position is "
            "x = {A} cos({n}t{eps}) metres at time t seconds. Upward is the positive direction.",
        ),
        (F(1, 20), F(2, 25), F(1, 10), F(3, 20), F(1, 5)),
        (4, 5, 6, 7, 8, 9, 10, 12),
    ),
)


@generator("r3d_shm_calculus_exam", level=17, presets={"exam": {}})
def r3d_shm_calculus_exam(rng):
    """k in x'' = -kx, then the period, the greatest speed and the velocity at t = 0, each from the part before."""
    stems, amps, ns = rng.choice(_EXAM_CONTEXTS)
    for _ in attempts(500, "an amplitude and frequency with exact speeds"):
        amp = F(rng.choice(amps))
        n = rng.choice(ns)
        eps = rng.choice((F(1, 6), F(-1, 6), F(5, 6), F(-5, 6)))
        sin_eps = _HALF if eps in (F(1, 6), F(5, 6)) else -_HALF
        k = n * n
        vmax = amp * n
        v0 = -vmax * sin_eps
        dp_c, dp_d = fit_dp(vmax, 1, 3), fit_dp(v0, 1, 3)
        period = 2 * sympy.pi / n
        if dp_c is None or dp_d is None or not clear_of_boundary(exact(period), 2):
            continue
        break
    # Second route: SymPy differentiates x twice and divides by x.
    A_, N_, E_ = (
        sympy.Rational(amp.numerator, amp.denominator),
        sympy.Integer(n),
        sympy.pi * sympy.Rational(eps.numerator, eps.denominator),
    )
    x = A_ * sympy.cos(N_ * T + E_)
    check(sympy.simplify(sympy.diff(x, T, 2) + k * x) == 0, "x'' is not -kx")
    check(exact(sympy.diff(x, T).subs(T, 0)) == v0, "v(0) disagrees with the derivative")
    eps_text = pi_text(eps)
    eps_show = phase_text(eps)
    stem = phrase(rng, stems, A=dec(amp, 3), n=n, eps=eps_show)
    part_a = number_part(
        "The acceleration satisfies d²x/dt² = -kx, where k is a constant. Find k.",
        F(k),
        0,
        1,
        [
            f"x'' = -{n}² x, since differentiating twice multiplies cos by -{n}² each time the argument has {n}t.",
            f"k = {k}.",
        ],
        [
            (F(n), "r3d-omega-not-squared"),
            (amp * k, "r3d-amplitude-left-in-constant"),
            (F(n**3), "r3d-omega-cubed"),
            (amp * n, "r3d-speed-for-constant"),
            (F(2 * n), "r3d-doubled-not-squared"),
            (F(n + k), "r3d-sum-for-product"),
        ],
    )
    part_b = number_part(
        "Find the period of the motion, in seconds, correct to 2 decimal places.",
        exact(period),
        2,
        2,
        [f"n = √k = {n}.", f"Period = 2π/n = {float(period):.2f} s."],
        [
            (2 * sympy.pi / k, "r3d-period-from-k"),
            (sympy.pi / n, "r3d-half-period"),
            (2 * sympy.pi * n, "r3d-period-inverted"),
        ],
        follow="2*pi/sqrt(part_a)",
        sensible=(0, None),
    )
    part_c = number_part(
        "Find the greatest speed of the object, in metres per second.",
        vmax,
        dp_c,
        2,
        [
            f"v = -{dec(vmax, 3)} sin({n}t{phase_text(eps)}), so the greatest speed is {dec(amp, 3)}n = {dec(vmax, 3)} m/s."
        ],
        [
            (amp * k, "r3d-acceleration-for-speed"),
            (F(n), "r3d-amplitude-dropped"),
            (amp / n, "r3d-integrated-not-differentiated"),
            (amp, "r3d-chain-factor-missing"),
            (amp * (n + 1), "r3d-sum-for-product"),
        ],
        follow=f"{dec(amp, 3)}*sqrt(part_a)",
        sensible=(0, None),
    )
    part_d = number_part(
        "Find the velocity of the object at t = 0, in metres per second (positive means moving upward).",
        v0,
        dp_d,
        2,
        [f"v(0) = -{dec(vmax, 3)} sin({eps_text}) = {dec(v0, 3)} m/s."],
        [
            (-v0, "r3d-derivative-sign"),
            (vmax, "r3d-phase-ignored"),
            (amp * n * math.sqrt(3) / 2 * (1 if v0 > 0 else -1), "r3d-sin-cos-swapped"),
        ],
        follow=f"-part_c*({sin_eps.numerator}/{sin_eps.denominator})",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
