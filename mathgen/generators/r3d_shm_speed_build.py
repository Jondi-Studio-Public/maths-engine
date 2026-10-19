"""Speed and position in simple harmonic motion, v² = n²(a² - x²): `r3d_shm_speed` (easy, medium, hard) and
`r3d_shm_speed_exam`.

Skill: shm-speed-at-position. A particle with acceleration -n²x has v² = n²(a² - x²), where a is the amplitude. The
skill is about using that relation and reading n from whatever the question gives: a stated acceleration constant, the
greatest speed, the period, the acceleration at a point, or two speeds at two positions.

Tiers differ in the decision they ask for, not in the size of the numbers:

* easy: n² is stated (acceleration -kx) and so is the amplitude. Find the speed at a stated displacement, or the
  distance from the centre at a stated speed.
* medium: n is hidden. It comes from the greatest speed and the amplitude, from a period written as a multiple of pi,
  or from the acceleration at one displacement, and the speed at another displacement follows.
* hard: the amplitude comes from the period and one speed; the time to travel between two stated points (a/2 and a, or
  O and a/2) comes from n found from a speed; the greatest acceleration comes from two speeds at two positions.
* exam: two speeds at two positions give (a) n, then (b) the amplitude, (c) the greatest speed and (d) the time to go
  from O to a/2.

## Construction

Amplitude, displacement and speed come from Pythagorean triples (5-3-4, 13-5-12, 17-8-15, 25-7-24 and the like,
scaled by 1/10, 1/5 or 1/2), so every speed is exact. Times are multiples of pi/n (the sine of pi/6 is 1/2), shown to
3 decimal places and kept away from a rounding boundary. Speeds are recomputed from v² = n²(a² - x²) in exact
fractions, and the time from numerical solution of x = a sin(nt) by arcsine.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r3d_kit import clear_of_boundary, dec, exact, fit_dp, number_fields, number_part, pi_text, sqrt_of

F = Fraction

#: (amplitude, displacement, speed factor): amplitude² = displacement² + speed factor²
_TRIPLES = (
    (5, 3, 4),
    (5, 4, 3),
    (10, 6, 8),
    (10, 8, 6),
    (13, 5, 12),
    (13, 12, 5),
    (15, 9, 12),
    (15, 12, 9),
    (17, 8, 15),
    (17, 15, 8),
    (20, 12, 16),
    (20, 16, 12),
    (25, 7, 24),
    (25, 15, 20),
    (25, 20, 15),
    (25, 24, 7),
    (26, 10, 24),
    (26, 24, 10),
)
#: amplitudes that offer two different positions with a whole speed factor
_TWO_POSITIONS = {
    5: (3, 4),
    10: (6, 8),
    13: (5, 12),
    15: (9, 12),
    17: (8, 15),
    20: (12, 16),
    25: (7, 15, 20, 24),
    26: (10, 24),
}
_SCALES = (F(1, 10), F(1, 5), F(1, 2))
_FRAME = "A particle moves in simple harmonic motion along a straight line about the centre O"
_EASY_FRAMES = (
    _FRAME,
    "A particle oscillates in simple harmonic motion on a line through the centre O",
    "A particle performs simple harmonic motion along a straight line, centred on O",
)


def _draw(rng, ns=(2, 3, 4), top_speed=F(24)):
    """(amplitude, displacement, speed factor, n) with a bounded greatest speed, all in exact fractions."""
    for _ in attempts(500, "a triple and frequency with a moderate speed"):
        r, p, q = rng.choice(_TRIPLES)
        f = rng.choice(_SCALES)
        n = rng.choice(ns)
        if n * r * f <= top_speed:
            return r * f, p * f, q * f, F(n)
    raise AssertionError("unreachable")


def _speed(n, a, x) -> Fraction:
    return n * sqrt_of(a * a - x * x)


def _timed(value, dp=3) -> bool:
    return clear_of_boundary(value, dp)


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    variant = rng.choice(("speed", "distance"))
    frame = rng.choice(_EASY_FRAMES)
    for _ in attempts(500, "an exact answer"):
        a, x, s, n = _draw(rng)
        v = n * s
        dp = fit_dp(v if variant == "speed" else x, 1, 3)
        if dp is not None:
            break
    k = n * n
    check(_speed(n, a, x) == v, "speed differs from v² = n²(a² - x²)")
    accel = f"its acceleration is -{dec(k, 2)}x m/s², where x metres is its displacement from O"
    if variant == "speed":
        text = (
            f"{frame}; {accel}. The amplitude is {dec(a, 3)} m. Find the speed of the particle when x = {dec(x, 3)}, "
            f"in metres per second."
        )
        value = v
        wrong = [
            (n * (a - x), "r3d-difference-for-squares"),
            (sqrt_of(a * a - x * x), "r3d-factor-n-dropped"),
            (n * sqrt_of(a * a + x * x), "r3d-sum-for-difference"),
            (k * sqrt_of(a * a - x * x), "r3d-n-squared-for-speed"),
        ]
        working = [
            f"The acceleration is -n²x with n² = {dec(k, 2)}, so n = {dec(n, 2)}.",
            f"v² = n²(a² - x²) = {dec(k, 2)}({dec(a, 3)}² - {dec(x, 3)}²), so v = {dec(v, 3)} m/s.",
        ]
    else:
        text = (
            f"{frame}; {accel}. The amplitude is {dec(a, 3)} m. At one instant the speed of the particle is "
            f"{dec(v, 3)} m/s. Find its distance from O at that instant, in metres."
        )
        value = x
        wrong = [
            (a - v / n, "r3d-difference-for-squares"),
            (sqrt_of(a * a - v * v), "r3d-factor-n-dropped"),
            (sqrt_of(a * a - v * v / n), "r3d-n-squared-for-speed"),
            (sqrt_of(a * a + (v / n) ** 2), "r3d-sum-for-difference"),
        ]
        working = [
            f"n² = {dec(k, 2)}, so n = {dec(n, 2)}. From v² = n²(a² - x²): x² = a² - (v/n)².",
            f"x² = {dec(a, 3)}² - {dec(v / n, 3)}², so x = {dec(x, 3)} m.",
        ]
    return dict(text=text, working=working, **number_fields(value, dp, wrong))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    variant = rng.choice(("greatest_speed", "period", "acceleration_at_point"))
    for _ in attempts(500, "an exact answer"):
        a, x, s, n = _draw(rng, ns=(2, 3, 4, 5))
        v = n * s
        dp = fit_dp(v, 1, 3)
        top = n * a
        if (
            dp is not None
            and fit_dp(top, 1, 3) is not None
            and fit_dp(n * x, 1, 3) is not None
            and fit_dp(n * n * x, 1, 3) is not None
        ):
            break
    check(_speed(n, a, x) == v, "speed differs from v² = n²(a² - x²)")
    ask = f"Find the speed of the particle when x = {dec(x, 3)}, in metres per second."
    if variant == "greatest_speed":
        given = f"its greatest speed is {dec(top, 3)} m/s and the amplitude of the motion is {dec(a, 3)} m"
        wrong = [
            (top * sqrt_of(a * a - x * x), "r3d-speed-for-n"),
            (top * (a - x) / a, "r3d-difference-for-squares"),
            (top - x, "r3d-subtracted-position"),
            (sqrt_of(a * a - x * x), "r3d-factor-n-dropped"),
        ]
        working = [
            f"The greatest speed is na, so n = {dec(top, 3)}/{dec(a, 3)} = {dec(n, 2)}.",
            f"v² = n²(a² - x²) gives v = {dec(n, 2)} × √({dec(a, 3)}² - {dec(x, 3)}²) = {dec(v, 3)} m/s.",
        ]
    elif variant == "period":
        given = f"the period of the motion is {pi_text(F(2) / n)} seconds and its amplitude is {dec(a, 3)} m"
        wrong = [
            (sqrt_of(a * a - x * x) * F(2) / n, "r3d-period-for-frequency"),
            (sqrt_of(a * a - x * x) * n * n, "r3d-n-squared-for-speed"),
            (sqrt_of(a * a - x * x), "r3d-factor-n-dropped"),
            (n * (a - x), "r3d-difference-for-squares"),
        ]
        working = [
            f"Period = 2π/n = {pi_text(F(2) / n)}, so n = {dec(n, 2)}.",
            f"v = n√(a² - x²) = {dec(n, 2)} × √({dec(a, 3)}² - {dec(x, 3)}²) = {dec(v, 3)} m/s.",
        ]
    else:
        x2 = F(rng.choice((F(1, 2), F(3, 4), F(1, 4)))) * a
        # a second displacement, closer to the end, at which the acceleration is quoted
        accel = n * n * x2
        for _ in attempts(100, "a quoted acceleration with an exact decimal"):
            if fit_dp(accel, 1, 3) is not None:
                break
            x2 = F(rng.choice((F(1, 2), F(3, 4), F(1, 4)))) * a
            accel = n * n * x2
        check(fit_dp(accel, 1, 3) is not None, "acceleration is not an exact decimal")
        given = (
            f"the magnitude of its acceleration is {dec(accel, 3)} m/s² when it is {dec(x2, 3)} m from O, and its "
            f"amplitude is {dec(a, 3)} m"
        )
        wrong = [
            (sqrt_of(a * a - x * x) * accel / x2, "r3d-n-squared-for-speed"),
            (sqrt_of(a * a - x * x), "r3d-factor-n-dropped"),
            (sqrt_of(accel / x2) * (a - x), "r3d-difference-for-squares"),
            (accel * sqrt_of(a * a - x * x) / a, "r3d-acceleration-ratio-misused"),
        ]
        working = [
            f"The acceleration is -n²x, so n² = {dec(accel, 3)}/{dec(x2, 3)} = {dec(n * n, 2)} and n = {dec(n, 2)}.",
            f"v = n√(a² - x²) = {dec(n, 2)} × √({dec(a, 3)}² - {dec(x, 3)}²) = {dec(v, 3)} m/s.",
        ]
    text = f"{_FRAME}; {given}. {ask}"
    return dict(text=text, working=working, **number_fields(v, dp, wrong))


# ------------------------------------------------------------------------------------------------ hard


def _hard(rng):
    variant = rng.choice(("amplitude", "time", "greatest_acceleration"))
    if variant == "amplitude":
        for _ in attempts(500, "an exact amplitude"):
            a, x, s, n = _draw(rng, ns=(2, 3, 4, 5))
            v = n * s
            if fit_dp(a, 1, 3) is not None and fit_dp(v, 1, 3) is not None and fit_dp(x, 1, 3) is not None:
                break
        dp = fit_dp(a, 1, 3)
        check(sqrt_of(x * x + (v / n) ** 2) == a, "amplitude differs from a² = x² + (v/n)²")
        wrong = [
            (x + v / n, "r3d-sum-for-sum-of-squares"),
            (sqrt_of(x * x + v * v), "r3d-factor-n-dropped"),
            (sqrt_of(v * v / n + x * x), "r3d-n-squared-for-speed"),
            (v / n, "r3d-speed-term-only"),
        ]
        text = (
            f"{_FRAME}. Its period is {pi_text(F(2) / n)} seconds. When it is {dec(x, 3)} m from O its speed is "
            f"{dec(v, 3)} m/s. Find the amplitude of the motion, in metres."
        )
        working = [
            f"Period = 2π/n = {pi_text(F(2) / n)}, so n = {dec(n, 2)}.",
            f"a² = x² + (v/n)² = {dec(x, 3)}² + {dec(v / n, 3)}², so a = {dec(a, 3)} m.",
        ]
        return dict(text=text, working=working, **number_fields(a, dp, wrong))
    if variant == "time":
        legs = (
            (
                "O",
                "a/2",
                F(1, 6),
                "a point halfway to the end of the path",
                "from O to the point halfway between O and the end of the path",
            ),
            (
                "a/2",
                "a",
                F(1, 3),
                "",
                "from the point halfway between O and the end of the path to the end of the path",
            ),
            (
                "-a/2",
                "a/2",
                F(1, 3),
                "",
                "from the point halfway on one side of O to the point halfway on the other side",
            ),
        )
        for _ in attempts(500, "a clear time"):
            a, x, s, n = _draw(rng, ns=(2, 3, 4, 5))
            v = n * s
            leg = rng.choice(legs)
            t = leg[2] * sympy.pi / sympy.Rational(n.numerator, n.denominator)
            if fit_dp(v, 1, 3) is not None and _timed(exact(t)):
                break
        # second route: the time to reach x is asin(x/a)/n, so a leg takes the difference of two such times
        position = {"O": 0.0, "a/2": 0.5, "-a/2": -0.5, "a": 1.0}
        elapsed = (math.asin(position[leg[1]]) - math.asin(position[leg[0]])) / float(n)
        check(abs(elapsed - float(t)) < 1e-9, "time differs from the arcsine route")
        wrong = [
            (leg[2] * 2 * sympy.pi / sympy.Rational(n.numerator, n.denominator), "r3d-time-fraction-of-period-wrong"),
            (sympy.pi / (2 * sympy.Rational(n.numerator, n.denominator)), "r3d-whole-quarter-period"),
            (leg[2] * sympy.pi, "r3d-factor-n-dropped"),
            (
                F(1, 2) * sympy.pi / sympy.Rational(n.numerator, n.denominator)
                - leg[2] * sympy.pi / sympy.Rational(n.numerator, n.denominator),
                "r3d-complementary-time",
            ),
        ]
        text = (
            f"{_FRAME}. The amplitude is {dec(a, 3)} m and, when the particle is {dec(x, 3)} m from O, its speed is "
            f"{dec(v, 3)} m/s. Find the time the particle takes to travel {leg[4]}, in seconds, correct to "
            f"3 decimal places."
        )
        working = [
            f"n = v/√(a² - x²) = {dec(v, 3)}/{dec(s, 3)} = {dec(n, 2)}.",
            f"With x = a sin(nt), the particle is at a/2 when nt = π/6 and at a when nt = π/2, so the time is "
            f"{pi_text(leg[2])}/{dec(n, 2)} = {float(t):.3f} s.",
        ]
        return dict(text=text, working=working, **number_fields(exact(t), 3, wrong))
    # greatest acceleration from two speeds
    for _ in attempts(500, "an exact greatest acceleration"):
        r = rng.choice(sorted(_TWO_POSITIONS))
        p1, p2 = rng.sample(_TWO_POSITIONS[r], 2)
        f = rng.choice(_SCALES)
        n = F(rng.choice((1, 2, 3, 4)))
        a = r * f
        x1, x2 = p1 * f, p2 * f
        q1, q2 = sqrt_of(r * r - p1 * p1) * f, sqrt_of(r * r - p2 * p2) * f
        v1, v2 = n * q1, n * q2
        top = n * n * a
        if max(v1, v2) <= 24 and fit_dp(top, 1, 3) and all(fit_dp(z, 1, 3) is not None for z in (v1, v2, x1, x2)):
            break
    n2 = (v1 * v1 - v2 * v2) / (x2 * x2 - x1 * x1)
    check(n2 == n * n and sqrt_of(x1 * x1 + v1 * v1 / n2) == a, "two-speed solution disagrees")
    dp = fit_dp(top, 1, 3)
    wrong = [
        (n * a, "r3d-speed-for-acceleration"),
        (n2 * x1, "r3d-start-value-for-greatest"),
        (n2 * x2, "r3d-start-value-for-greatest"),
        (n2, "r3d-constant-for-greatest-acceleration"),
        (sqrt_of(n2) * a * a, "r3d-amplitude-squared"),
        ((v1 - v2) / (x2 - x1) * a, "r3d-linear-for-squares"),
    ]
    text = (
        f"{_FRAME}. When the particle is {dec(x1, 3)} m from O its speed is {dec(v1, 3)} m/s, and when it is {dec(x2, 3)} m "
        f"from O its speed is {dec(v2, 3)} m/s. Find the greatest acceleration of the particle, in metres per second squared."
    )
    working = [
        f"v² = n²(a² - x²) at both points: n² = (v1² - v2²)/(x2² - x1²) = {dec(n2, 3)}.",
        f"a² = x1² + v1²/n² gives a = {dec(a, 3)} m, so the greatest acceleration is n²a = {dec(top, 3)} m/s².",
    ]
    return dict(text=text, working=working, **number_fields(top, dp, wrong))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("r3d_shm_speed", level=17, presets=_PRESETS)
def r3d_shm_speed(rng, tier: str = "easy"):
    """v² = n²(a² - x²): a speed or position given n, then n found from the data, then the amplitude, a time or two speeds."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam

_EXAM_STEMS = (
    "A particle moves in simple harmonic motion along a straight line about the centre O. When it is {x1} m from O "
    "its speed is {v1} m/s, and when it is {x2} m from O its speed is {v2} m/s.",
    "A trolley on a spring-loaded track moves in simple harmonic motion about O. At a displacement of {x1} m from O "
    "its speed is {v1} m/s, and at a displacement of {x2} m from O its speed is {v2} m/s.",
    "A mass on a horizontal spring moves in simple harmonic motion about its equilibrium position O. When it is {x1} m "
    "from O its speed is {v1} m/s, and when it is {x2} m from O its speed is {v2} m/s.",
)


@generator("r3d_shm_speed_exam", level=17, presets={"exam": {}})
def r3d_shm_speed_exam(rng):
    """Two speeds at two positions give n, then the amplitude, the greatest speed and the time from O to a/2."""
    for _ in attempts(1000, "two speeds with an exact solution and a clear time"):
        r = rng.choice(sorted(_TWO_POSITIONS))
        p1, p2 = rng.sample(_TWO_POSITIONS[r], 2)
        f = rng.choice(_SCALES)
        n = F(rng.choice((F(1, 2), F(3, 2), 2, 3, 4)))
        a = r * f
        x1, x2 = p1 * f, p2 * f
        v1 = n * sqrt_of(r * r - p1 * p1) * f
        v2 = n * sqrt_of(r * r - p2 * p2) * f
        vmax = n * a
        t = sympy.pi / (6 * sympy.Rational(n.numerator, n.denominator))
        if (
            max(v1, v2) <= 24
            and all(fit_dp(z, 1, 3) is not None for z in (v1, v2, x1, x2, n, a, vmax))
            and clear_of_boundary(exact(t), 3)
        ):
            break
    n2 = (v1 * v1 - v2 * v2) / (x2 * x2 - x1 * x1)
    check(n2 == n * n and sqrt_of(x1 * x1 + v1 * v1 / n2) == a, "two-speed solution disagrees")
    text = _EXAM_STEMS[rng.randrange(len(_EXAM_STEMS))].format(
        x1=dec(x1, 3), v1=dec(v1, 3), x2=dec(x2, 3), v2=dec(v2, 3)
    )
    dp_n, dp_a, dp_v = fit_dp(n, 1, 3), fit_dp(a, 1, 3), fit_dp(vmax, 1, 3)
    part_a = number_part(
        "The acceleration is -n²x. Find n.",
        n,
        dp_n,
        2,
        [
            "v² = n²(a² - x²) holds at both positions; subtracting eliminates a².",
            f"n² = (v1² - v2²)/(x2² - x1²) = {dec(n2, 4)}, so n = {dec(n, 3)}.",
        ],
        [
            (n2, "r3d-omega-squared-for-omega"),
            ((v1 - v2) / (x2 - x1), "r3d-linear-for-squares"),
            ((v1 * v1 - v2 * v2) / (x2 - x1) / 1, "r3d-difference-not-squared"),
            (sqrt_of((v1 * v1 + v2 * v2) / (x1 * x1 + x2 * x2)), "r3d-sum-for-difference"),
            (v1 / x1, "r3d-speed-over-displacement"),
            (v2 / x2, "r3d-speed-over-displacement"),
        ],
        sensible=(0, None),
    )
    part_b = number_part(
        "Find the amplitude of the motion, in metres.",
        a,
        dp_a,
        2,
        [f"a² = x1² + v1²/n² = {dec(x1, 3)}² + {dec(v1 / n, 3)}², so a = {dec(a, 3)} m."],
        [
            (x1 + v1 / n, "r3d-sum-for-sum-of-squares"),
            (sqrt_of(x1 * x1 + v1 * v1), "r3d-factor-n-dropped"),
            (v1 / n, "r3d-speed-term-only"),
            (sqrt_of(x1 * x1 + v1 * v1 / n), "r3d-n-squared-for-speed"),
        ],
        follow=f"sqrt(({x1.numerator}/{x1.denominator})**2 + (({v1.numerator}/{v1.denominator})/part_a)**2)",
        sensible=(0, None),
    )
    part_c = number_part(
        "Find the greatest speed of the particle, in metres per second.",
        vmax,
        dp_v,
        1,
        [f"The greatest speed is at O: v = na = {dec(n, 3)} × {dec(a, 3)} = {dec(vmax, 3)} m/s."],
        [
            (n * n * a, "r3d-acceleration-for-speed"),
            (a / n, "r3d-period-ratio-for-amplitude"),
            (n * (a - x1), "r3d-difference-for-squares"),
            (v1, "r3d-given-speed-for-greatest"),
        ],
        follow="part_a*part_b",
        sensible=(0, None),
    )
    part_d = number_part(
        "Find the time the particle takes to travel from O to the point halfway between O and the end of its path, "
        "in seconds, correct to 3 decimal places.",
        exact(t),
        3,
        2,
        [
            "Taking x = a sin(nt), the particle is at x = a/2 when nt = π/6.",
            f"t = π/(6 × {dec(n, 3)}) = {float(t):.3f} s.",
        ],
        [
            (sympy.pi / (2 * sympy.Rational(n.numerator, n.denominator)), "r3d-whole-quarter-period"),
            (sympy.pi / 6, "r3d-factor-n-dropped"),
            (sympy.pi / (3 * sympy.Rational(n.numerator, n.denominator)), "r3d-time-fraction-of-period-wrong"),
            (sympy.pi / (6 * sympy.Rational(n.numerator, n.denominator)) * n, "r3d-omega-multiplied-for-time"),
        ],
        follow="pi/(6*part_a)",
        sensible=(0, None),
    )
    return dict(text=text, parts=[part_a, part_b, part_c, part_d])
