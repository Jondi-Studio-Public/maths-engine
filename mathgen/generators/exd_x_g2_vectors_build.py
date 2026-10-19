"""Exam-tier question sources for seven vector skills.

Each topic here is the exam tier of one existing vector skill: harder than that skill's hard tier in KIND, not
in number range. Every question is built BACKWARDS from a chosen integer answer or a chosen root, so the key is
exact for every seed and the solution a learner must find is unique.

- ``exd_x_g2_projectile_landing``: projectile motion where the learner first solves for a time from a
  condition (landing, greatest height, impact) and only then reads off a position, height or speed.
- ``exd_x_g2_position_collision``: two particles with different position functions; find where they collide
  (solve a quadratic in one component, check it against the other, substitute).
- ``exd_x_g2_scalar_pair``: two unknown scalars in ``p*u + q*w = r``; the learner picks a method to solve the
  2 x 2 system and reports one scalar.
- ``exd_x_g2_line_motion_event``: motion along a line where the learner finds the instant of an event
  (momentary rest, return to the start) before asking for a position or velocity component.
- ``exd_x_g2_complex_vector_system``: two simultaneous vector equations in ``z1`` and ``z2``.
- ``exd_x_g2_velocity_composition``: velocities relative to a moving medium combined with time, then a
  magnitude.
- ``exd_x_g2_parallel_perp_condition``: a parallel or perpendicular condition on a vector built from a scalar
  parameter, including one where the second vector is itself built from the first two.

All answers are integers, so the typed-number grader requires an exact match. Distractors are principled wrong
answers (a named slip) and are filtered against the answer by ``pick_distractors``.
"""

from __future__ import annotations

import functools
from fractions import Fraction
from math import isqrt

from ..core import (
    GRADER_TOLERANCE,
    check,
    fmt_signed,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    pick_distractors,
    rational_text,
)

_MAX_ATTEMPTS = 500


class _Resample(Exception):
    """A draw left fewer than two usable distractors; the caller draws again."""


def _retrying(fn):
    """Re-run a generator body on a fresh draw when a question has too few distinct wrong answers."""

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in range(50):
            try:
                return fn(rng, *args, **kwargs)
            except _Resample:
                continue
        raise RuntimeError(f"{fn.__name__}: no draw with enough distractors")

    return wrapper


# --- shared helpers ------------------------------------------------------------------------------------------


def _ij(x, y) -> str:
    """A vector written in ``i``/``j`` form, e.g. ``3i - 2j``."""
    return join_terms([(x, "i"), (y, "j")])


def _pair(x, y) -> str:
    return f"({x}, {y})"


def _cx(x, y) -> str:
    """A complex number ``x + yi`` written the way a student would."""
    return join_terms([(x, ""), (y, "i")])


def _plain(value) -> str | None:
    """Candidate as text, or None when it is not a whole number (the answers here are whole numbers, and an
    option written as a fraction or decimal would give the key away by its format)."""
    f = Fraction(value).limit_denominator(10**6)
    return str(f.numerator) if f.denominator == 1 else None


def _number_result(answer: int, candidates, text: str, working: list[str]) -> dict:
    """Assemble a number-kind question. ``candidates`` are values (ints, Fractions or floats)."""
    candidates = [c for c in candidates if c is not None]
    answer_text = str(int(answer))
    distractors = pick_distractors(
        answer_text, [t for t in (_plain(c) for c in candidates) if t is not None], n=3, tolerance=GRADER_TOLERANCE
    )
    if len(distractors) < 2:
        raise _Resample
    return dict(
        text=text,
        answer=int(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(int(answer)),
        distractors=distractors,
        working=working,
    )


def _pair_result(x: int, y: int, candidates, text: str, working: list[str]) -> dict:
    """Assemble a pair-kind question answered as ``x = .., y = ..`` (the form the skill's grader reads)."""
    answer_text = f"x = {x}, y = {y}"
    distractors = pick_distractors(
        answer_text,
        [f"x = {cx}, y = {cy}" for cx, cy in candidates if (cx, cy) != (x, y)],
        n=3,
        tolerance=GRADER_TOLERANCE,
    )
    if len(distractors) < 2:
        raise _Resample
    return dict(
        text=text,
        answer={"x": x, "y": y},
        answer_text=answer_text,
        answer_latex=rf"x = {x},\; y = {y}",
        distractors=distractors,
        working=working,
    )


def _dot(a, b) -> int:
    return a[0] * b[0] + a[1] * b[1]


def _cross(a, b) -> int:
    return a[0] * b[1] - a[1] * b[0]


def _hours(n: int) -> str:
    return "hour" if n == 1 else "hours"


def _seconds(n: int) -> str:
    return "second" if n == 1 else "seconds"


# --- 1. projectile: solve for a time first -------------------------------------------------------------------

_G = 10
_TRIPLES = ((3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6), (9, 12), (12, 9), (8, 15), (15, 8))


def _projectile_landing(rng):
    for _ in range(_MAX_ATTEMPTS):
        T = rng.randint(2, 6)
        vy = rng.randint(1, 9) * 2
        y0 = _G * T * T // 2 - vy * T
        if not 1 <= y0 <= 60:
            continue
        vx = rng.choice([v for v in range(-9, 10) if v != 0])
        x0 = rng.randint(-9, 9)
        break
    else:
        raise RuntimeError("no landing draw")
    # y(t) = y0 + vy t - 5 t^2 has roots with product -y0/5 < 0, so exactly one positive root, namely T.
    check(y0 + vy * T - _G * T * T // 2 == 0 and y0 > 0, "landing time is not the unique positive root")
    answer = x0 + vx * T
    cands = [
        vx * T,  # forgot the starting i-component
        Fraction(x0) + Fraction(vx * vy, 5),  # dropped the launch height when solving for the time
        Fraction(x0) + Fraction(vx * vy, 10),  # used the time to the highest point, not the landing time
        T,  # reported the time of flight
    ]
    text = (
        f"A ball is launched from position vector r(0) = {_ij(x0, y0)} m with velocity v(0) = {_ij(vx, vy)} m/s. "
        f"Its acceleration is a(t) = -{_G}j m/s², and the ground is where the j-component of position is 0. "
        f"Find the i-component of the ball's position, in m, at the instant it lands."
    )
    working = [
        f"j-component of position: {y0} + {vy}t - 5t² = 0, so t = {T} (the only positive root).",
        f"i-component at t = {T}: {x0} + {paren_negative(vx)}×{T} = {answer}.",
    ]
    return _number_result(answer, cands, text, working)


def _projectile_height(rng):
    vy = rng.choice([10, 20, 30])
    y0 = rng.randint(1, 30)
    vx = rng.choice([v for v in range(-9, 10) if v != 0])
    x0 = rng.randint(-9, 9)
    t_top = Fraction(vy, _G)
    top = Fraction(y0) + vy * t_top - Fraction(_G, 2) * t_top**2
    check(top.denominator == 1 and top == y0 + Fraction(vy * vy, 2 * _G), "greatest height is not whole")
    answer = int(top)
    cands = [
        y0 + vy * vy // _G,  # kept the launch speed times the time but forgot the -5t^2 term
        vy * vy // (2 * _G),  # dropped the launch height
        int(t_top),  # reported the time to the highest point
    ]
    text = (
        f"A ball is launched from position vector r(0) = {_ij(x0, y0)} m with velocity v(0) = {_ij(vx, vy)} m/s. "
        f"Its acceleration is a(t) = -{_G}j m/s². Find the greatest value of the j-component of the ball's "
        f"position, in m."
    )
    working = [
        f"The j-component of velocity is {vy} - 10t, which is 0 at t = {int(t_top)}.",
        f"Greatest j-component: {y0} + {vy}×{int(t_top)} - 5×{int(t_top)}² = {answer}.",
    ]
    return _number_result(answer, cands, text, working)


def _projectile_speed(rng):
    for _ in range(_MAX_ATTEMPTS):
        a, n = rng.choice(_TRIPLES)
        T = rng.randint(1, 5)
        vy = _G * T - n
        y0 = T * (n - 5 * T)
        if vy >= 1 and 1 <= y0 <= 80:
            break
    else:
        raise RuntimeError("no impact-speed draw")
    vx = a * rng.choice([-1, 1])
    x0 = rng.randint(-9, 9)
    check(y0 + vy * T - 5 * T * T == 0, "landing time is not T")
    answer = isqrt(a * a + n * n)
    check(answer * answer == a * a + n * n, "impact speed is not whole")
    launch = (vx * vx + vy * vy) ** 0.5
    wrong_sign = (vx * vx + (vy + _G * T) ** 2) ** 0.5
    cands = [a + n, n, launch, wrong_sign]
    text = (
        f"A stone is thrown from position vector r(0) = {_ij(x0, y0)} m with velocity v(0) = {_ij(vx, vy)} m/s. "
        f"Its acceleration is a(t) = -{_G}j m/s², and the ground is where the j-component of position is 0. "
        f"Find the speed of the stone, in m/s, when it hits the ground."
    )
    working = [
        f"Landing: {y0} + {vy}t - 5t² = 0 gives t = {T}.",
        f"Velocity at t = {T}: {_ij(vx, vy - _G * T)} m/s.",
        f"Speed = √({paren_negative(vx)}² + {paren_negative(vy - _G * T)}²) = {answer} m/s.",
    ]
    return _number_result(answer, cands, text, working)


@generator(
    "exd_x_g2_projectile_landing",
    level=14,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
@_retrying
def exd_x_g2_projectile_landing(rng, tier: str = "exam"):
    return rng.choice([_projectile_landing, _projectile_height, _projectile_speed])(rng)


# --- 2. two particles: where do they collide? ----------------------------------------------------------------


def _poly_t(a, b, c) -> str:
    return join_terms([(a, "t^2"), (b, "t"), (c, "")])


def _lin_t(a, c) -> str:
    return join_terms([(a, "t"), (c, "")])


@generator(
    "exd_x_g2_position_collision",
    level=14,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
@_retrying
def exd_x_g2_position_collision(rng, tier: str = "exam"):
    for _ in range(_MAX_ATTEMPTS):
        e = rng.choice([1, 2])
        t_star = rng.randint(1, 4)
        t_other = rng.choice([0, -1, -2, -3])
        m = rng.choice([-3, -2, -1, 1, 2, 3])
        u1 = rng.randint(-5, 5)  # straight-line particle: x velocity
        v1 = rng.randint(-5, 5)  # straight-line particle: y velocity
        p1 = rng.randint(-9, 9)
        q1 = rng.randint(-9, 9)
        # x-component difference (curved minus straight) is e (t - t*)(t - t_other), the y-difference m (t - t*).
        f = u1 - e * (t_star + t_other)
        p2 = p1 + e * t_star * t_other
        g = v1 + m
        q2 = q1 - m * t_star
        if all(abs(c) <= 15 for c in (f, p2, g, q2)):
            break
    else:
        raise RuntimeError("no collision draw")

    ax = lambda t: u1 * t + p1  # noqa: E731
    ay = lambda t: v1 * t + q1  # noqa: E731
    bx = lambda t: e * t * t + f * t + p2  # noqa: E731
    by = lambda t: g * t + q2  # noqa: E731
    check(ax(t_star) == bx(t_star) and ay(t_star) == by(t_star), "particles do not meet at t*")
    check(not (ax(0) == bx(0) and ay(0) == by(0)), "particles already coincide at t = 0")
    x, y = ax(t_star), ay(t_star)

    cands = [
        (y, x),  # components in the wrong order
        (ax(t_other), ay(t_other)),  # used the non-positive root of the quadratic
        (p1, q1),  # reported where the straight-line particle starts
        (-x, y),  # sign slip on the i-component
    ]
    straight_is_p = rng.random() < 0.5
    lin = f"({_lin_t(u1, p1)}, {_lin_t(v1, q1)})"
    cur = f"({_poly_t(e, f, p2)}, {_lin_t(g, q2)})"
    pos = {"P": lin, "Q": cur} if straight_is_p else {"P": cur, "Q": lin}
    text = (
        f"Two particles move in a plane. Their position vectors, in metres at time t seconds (t ≥ 0), are "
        f"r_P(t) = {pos['P']} and r_Q(t) = {pos['Q']}. "
        f"The particles collide. Find the position vector of the point of collision, as a coordinate pair."
    )
    working = [
        f"Equate the i-components: {_poly_t(e, f, p2)} = {_lin_t(u1, p1)}, which factorises as "
        f"{e if e != 1 else ''}(t - {t_star})(t {'+' if t_other < 0 else '-'} {abs(t_other)}) = 0 (taking the curved "
        f"particle's component minus the other's).",
        f"The only t ≥ 0 root is t = {t_star}; the j-components also agree there.",
        f"Position at t = {t_star}: ({x}, {y}).",
    ]
    return _pair_result(x, y, cands, text, working)


# --- 3. two unknown scalars: p u + q w = r -------------------------------------------------------------------


@generator(
    "exd_x_g2_scalar_pair",
    level=14,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
@_retrying
def exd_x_g2_scalar_pair(rng, tier: str = "exam"):
    for _ in range(_MAX_ATTEMPTS):
        u = (rng.randint(-5, 5), rng.randint(-5, 5))
        w = (rng.randint(-5, 5), rng.randint(-5, 5))
        det = _cross(u, w)
        p = rng.choice([s for s in range(-6, 7) if s != 0])
        q = rng.choice([s for s in range(-6, 7) if s not in (0, p)])
        r = (p * u[0] + q * w[0], p * u[1] + q * w[1])
        if det == 0 or 0 in u or 0 in w or r == (0, 0) or 0 in r:
            continue
        ask = rng.choice(["p", "q"])
        answer = p if ask == "p" else q
        other = q if ask == "p" else p
        # Cramer's rule, the independent route: p = (r x w)/(u x w), q = (u x r)/(u x w).
        check(Fraction(_cross(r, w), det) == p and Fraction(_cross(u, r), det) == q, "scalars do not solve the system")
        first_axis = Fraction(r[0], u[0]) if ask == "p" else Fraction(r[0], w[0])
        cands = [
            other,  # reported the other scalar
            -answer,  # sign slip
            first_axis,  # used only the i-equation, as if the other term vanished
            p + q,  # reported the sum of the scalars
        ]
        if answer in cands[:1] + cands[3:]:
            # keep the pool varied but legal: check_distinct below is the real guard, so just resample here
            continue
        if any(Fraction(c) == answer for c in cands):
            continue
        if answer in (*u, *w, *r):
            continue  # a vector copied out of the prompt must not read as the key
        break
    else:
        raise RuntimeError("no scalar-pair draw")
    text = (
        f"Write {_ij(*r)} in the form p·u + q·w, where p and q are scalars, u is the vector {_ij(*u)} and w is "
        f"the vector {_ij(*w)}. Find the value of {ask}."
    )
    working = [
        f"i: {u[0]}p {'+' if w[0] >= 0 else '-'} {abs(w[0])}q = {r[0]}; j: {u[1]}p {'+' if w[1] >= 0 else '-'} "
        f"{abs(w[1])}q = {r[1]}.",
        f"Solving the pair simultaneously gives p = {p}, q = {q}.",
    ]
    return _number_result(answer, cands, text, working)


# --- 4. motion on a line: find the instant of an event first -------------------------------------------------

_DIRECTIONS = ((3, 4), (4, 3), (1, 2), (2, 1), (2, 3), (3, 2), (1, 1))
_OFFSETS = [v for v in range(-9, 10) if v != 0]


def _coef_d(coeff: int) -> str:
    """``coeff`` times ``d`` without a printed 1: ``d``, ``-d``, ``3d``."""
    return {1: "d", -1: "-d"}.get(coeff, f"{coeff}d")


def _fmt_accel(coeff: int, with_t: bool) -> str:
    var = "t·d" if with_t else "d"
    return f"{coeff}{var}" if coeff not in (1, -1) else ("-" if coeff == -1 else "") + var


@generator(
    "exd_x_g2_line_motion_event",
    level=14,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
@_retrying
def exd_x_g2_line_motion_event(rng, tier: str = "exam"):
    kind = rng.choice(["rest_variable", "rest_constant", "return_start"])
    dx, dy = rng.choice(_DIRECTIONS)
    axis = rng.choice(["i", "j"])
    dc, other_dc = (dx, dy) if axis == "i" else (dy, dx)
    x0, y0 = rng.choice(_OFFSETS), rng.choice(_OFFSETS)
    r0c, other_r0c = (x0, y0) if axis == "i" else (y0, x0)
    d_text = _ij(dx, dy)

    for _ in range(_MAX_ATTEMPTS):
        if kind == "rest_variable":
            H = rng.choice([-12, -6, 6, 12])
            ts = rng.randint(1, 3)
            f0 = -H * ts * ts // 2
            S = Fraction(H * ts**3, 6) + f0 * ts
            wrong_no_f0 = Fraction(H * ts**3, 6)
            wrong_power = Fraction(H * ts**3, 2) + f0 * ts
            check(Fraction(H * ts * ts, 2) + f0 == 0, "velocity is not zero at the event time")
            accel_clause = f"Its acceleration vector is {_fmt_accel(H, True)} m/s² at time t seconds (t ≥ 0)."
            when = "at the instant (t > 0) when the particle is momentarily at rest"
            cands_scalar = [wrong_no_f0, wrong_power]
        elif kind == "rest_constant":
            G0 = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            ts = rng.randint(1, 5)
            f0 = -G0 * ts
            S = Fraction(G0 * ts * ts, 2) + f0 * ts
            wrong_no_f0 = Fraction(G0 * ts * ts, 2)
            wrong_power = Fraction(G0 * ts * ts, 1) + f0 * ts  # forgot the 1/2 in the displacement formula
            check(G0 * ts + f0 == 0, "velocity is not zero at the event time")
            accel_clause = f"Its acceleration vector is {_fmt_accel(G0, False)} m/s² at time t seconds (t ≥ 0)."
            when = "at the instant (t > 0) when the particle is momentarily at rest"
            cands_scalar = [wrong_no_f0, wrong_power]
        else:
            G0 = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            ts = rng.randint(2, 8)
            f0 = -G0 * ts // 2 if (G0 * ts) % 2 == 0 else None
            if f0 is None:
                continue
            S = None
            accel_clause = f"Its acceleration vector is {_fmt_accel(G0, False)} m/s² at time t seconds (t ≥ 0)."
            when = "when it returns to its starting position (t > 0)"
        if f0 == 0 or abs(f0) > 30:
            continue
        if kind != "return_start" and S.denominator != 1:
            continue
        break
    else:
        raise RuntimeError("no line-motion draw")

    if kind == "return_start":
        # Displacement along d is G0 t²/2 + f0 t = 0 at t = -2 f0 / G0 = ts; velocity then is f0 + G0 ts = -f0.
        check(Fraction(G0 * ts * ts, 2) + f0 * ts == 0 and f0 + G0 * ts == -f0, "return time is wrong")
        answer = -f0 * dc
        cands = [f0 * dc, 0, -f0 * other_dc, G0 * ts * dc]
        text = (
            f"A particle moves in a straight line, always parallel to the vector d = {d_text}. "
            f"{accel_clause} At time t = 0 its velocity vector is {_coef_d(f0)} m/s. "
            f"Find the {axis}-component of the particle's velocity vector, in m/s, {when}."
        )
        working = [
            f"Displacement along d: {G0}t²/2 + {paren_negative(f0)}t = 0 gives t = {ts}.",
            f"Scalar velocity at t = {ts}: {paren_negative(f0)} + {paren_negative(G0)}×{ts} = {-f0}.",
            f"{axis}-component of velocity = {-f0}×{dc} = {answer}.",
        ]
        return _number_result(answer, cands, text, working)

    S = int(S)
    answer = r0c + S * dc
    cands = [
        r0c + int(cands_scalar[0]) * dc if cands_scalar[0].denominator == 1 else None,  # dropped initial velocity
        S * dc,  # dropped the starting position
        other_r0c + S * other_dc,  # wrong axis
        r0c + int(cands_scalar[1]) * dc if cands_scalar[1].denominator == 1 else None,  # mis-integrated
    ]
    text = (
        f"A particle moves in a straight line, always parallel to the vector d = {d_text}. "
        f"{accel_clause} At time t = 0 its velocity vector is {_coef_d(f0)} m/s "
        f"and its position vector is r(0) = {_ij(x0, y0)} m. "
        f"Find the {axis}-component of the particle's position vector, in m, {when}."
    )
    working = [
        f"Scalar velocity is zero at t = {ts}.",
        f"Scalar displacement along d by then: S = {S}.",
        f"{axis}-component of position = {r0c} + {paren_negative(S)}×{dc} = {answer}.",
    ]
    return _number_result(answer, cands, text, working)


# --- 5. complex numbers as vectors: two simultaneous vector equations ----------------------------------------


@generator(
    "exd_x_g2_complex_vector_system",
    level=14,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
@_retrying
def exd_x_g2_complex_vector_system(rng, tier: str = "exam"):
    coeffs = [(1, 2), (1, 3), (1, -1), (2, 1), (3, 1), (1, -2), (2, -1), (3, 2), (2, 3), (2, -3)]
    for _ in range(_MAX_ATTEMPTS):
        z1 = (rng.randint(-6, 6), rng.randint(-6, 6))
        z2 = (rng.randint(-6, 6), rng.randint(-6, 6))
        m1, m2 = rng.choice(coeffs)
        if z1 == z2 or (0, 0) in (z1, z2):
            continue
        R = (z1[0] + z2[0], z1[1] + z2[1])
        S = (m1 * z1[0] + m2 * z2[0], m1 * z1[1] + m2 * z2[1])
        if R == (0, 0) or S == (0, 0):
            continue
        break
    else:
        raise RuntimeError("no vector-system draw")
    # Independent route with Python's own complex numbers: eliminate z1 to get z2 = (S - m1 R)/(m2 - m1).
    c1, c2 = complex(*z1), complex(*z2)
    check(c1 + c2 == complex(*R) and m1 * c1 + m2 * c2 == complex(*S), "vectors do not satisfy the system")
    check(abs((complex(*S) - m1 * complex(*R)) / (m2 - m1) - c2) < 1e-9, "elimination disagrees")
    ask = rng.choice(["z1", "z2"])
    target, other = (z1, z2) if ask == "z1" else (z2, z1)
    cands = [
        (target[1], target[0]),  # components in the wrong order
        other,  # solved for the other vector
        (-target[0], -target[1]),  # sign slip on both components
        R,  # reported the given resultant
    ]
    second = join_terms([(m1, "z1"), (m2, "z2")])
    text = (
        f"Two vectors from the origin represent the complex numbers z1 and z2. Their resultant is "
        f"z1 + z2 = {_cx(*R)}, and they also satisfy {second} = {_cx(*S)}. "
        f"Find {ask}, as a coordinate pair."
    )
    working = [
        f"Subtract {'the first equation' if m1 == 1 else f'{m1} times the first equation'} from the second: {fmt_signed(m2 - m1, "z2")} = {_cx(S[0] - m1 * R[0], S[1] - m1 * R[1])}.",
        f"z2 = {_pair(*z2)}, then z1 = R - z2 = {_pair(*z1)}.",
    ]
    return _pair_result(target[0], target[1], cands, text, working)


# --- 6. velocities relative to a moving medium, with time ----------------------------------------------------


def _draw_ground(rng):
    a, b = rng.choice(_TRIPLES)
    k = rng.choice([1, 1, 2, 3])
    return (a * k * rng.choice([-1, 1]), b * k * rng.choice([-1, 1]))


@generator(
    "exd_x_g2_velocity_composition",
    level=14,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
@_retrying
def exd_x_g2_velocity_composition(rng, tier: str = "exam"):
    variant = rng.choice(["speed", "distance", "two_winds"])
    if variant in ("speed", "distance"):
        g = _draw_ground(rng)
        w = (rng.randint(-8, 8), rng.randint(-8, 8))
        air = (g[0] - w[0], g[1] - w[1])
        if 0 in air or w == (0, 0):
            return exd_x_g2_velocity_composition(rng)
        speed = isqrt(g[0] ** 2 + g[1] ** 2)
        check(speed * speed == g[0] ** 2 + g[1] ** 2, "ground speed is not whole")
        air_speed = (air[0] ** 2 + air[1] ** 2) ** 0.5
        if variant == "speed":
            t = 1
            answer = speed
            cands = [air_speed, abs(g[0]) + abs(g[1]), ((air[0] - w[0]) ** 2 + (air[1] - w[1]) ** 2) ** 0.5]
            question = "Find the plane's speed relative to the ground, in km/h."
        else:
            t = rng.randint(2, 5)
            answer = speed * t
            cands = [speed, air_speed * t, (abs(g[0]) + abs(g[1])) * t]
            question = f"Find how far the plane is from its starting point after {t} {_hours(t)}, in km."
        text = (
            f"A plane flies with velocity {_ij(*air)} km/h relative to the air. The wind blows with velocity "
            f"{_ij(*w)} km/h. {question}"
        )
        working = [
            f"Velocity relative to the ground = {_ij(*air)} + {'(' + _ij(*w) + ')' if _ij(*w).startswith('-') else _ij(*w)} = {_ij(*g)} km/h.",
            f"Ground speed = √({paren_negative(g[0])}² + {paren_negative(g[1])}²) = {speed} km/h"
            + (f"; distance = {speed}×{t} = {answer} km." if t > 1 else "."),
        ]
        return _number_result(answer, cands, text, working)

    # two_winds: same air velocity, two stages with different winds
    for _ in range(_MAX_ATTEMPTS):
        D = _draw_ground(rng)
        t1, t2 = rng.randint(1, 3), rng.randint(1, 3)
        w1 = (rng.randint(-8, 8), rng.randint(-8, 8))
        w2 = (rng.randint(-8, 8), rng.randint(-8, 8))
        rx, ry = D[0] - t1 * w1[0] - t2 * w2[0], D[1] - t1 * w1[1] - t2 * w2[1]
        n = t1 + t2
        if rx % n or ry % n or w1 == w2:
            continue
        air = (rx // n, ry // n)
        if 0 in air or abs(air[0]) > 40 or abs(air[1]) > 40:
            continue
        break
    else:
        raise RuntimeError("no two-wind draw")
    total = (t1 * (air[0] + w1[0]) + t2 * (air[0] + w2[0]), t1 * (air[1] + w1[1]) + t2 * (air[1] + w2[1]))
    check(total == D, "displacement does not match the chosen total")
    answer = isqrt(D[0] ** 2 + D[1] ** 2)
    check(answer * answer == D[0] ** 2 + D[1] ** 2, "distance is not whole")
    g1 = (air[0] + w1[0], air[1] + w1[1])
    g2 = (air[0] + w2[0], air[1] + w2[1])
    cands = [
        (air[0] ** 2 + air[1] ** 2) ** 0.5 * n,  # ignored the wind
        t1 * (g1[0] ** 2 + g1[1] ** 2) ** 0.5 + t2 * (g2[0] ** 2 + g2[1] ** 2) ** 0.5,  # added the two path lengths
        ((t1 * g1[0] + g2[0]) ** 2 + (t1 * g1[1] + g2[1]) ** 2) ** 0.5,  # forgot the second duration
    ]
    text = (
        f"A plane flies with constant velocity {_ij(*air)} km/h relative to the air. For {t1} {_hours(t1)} the "
        f"wind has velocity {_ij(*w1)} km/h, and for the next {t2} {_hours(t2)} it has velocity {_ij(*w2)} km/h. "
        f"Find how far the plane is from its starting point at the end, in km."
    )
    working = [
        f"Ground velocities: {_ij(*g1)} km/h, then {_ij(*g2)} km/h.",
        f"Total displacement = {t1}×({_ij(*g1)}) + {t2}×({_ij(*g2)}) = {_ij(*D)} km.",
        f"Distance = √({paren_negative(D[0])}² + {paren_negative(D[1])}²) = {answer} km.",
    ]
    return _number_result(answer, cands, text, working)


# --- 7. parallel / perpendicular conditions on a parametrised vector -----------------------------------------


@generator(
    "exd_x_g2_parallel_perp_condition",
    level=14,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
@_retrying
def exd_x_g2_parallel_perp_condition(rng, tier: str = "exam"):
    variant = rng.choice(["comb_perp", "comb_par", "derived_perp"])
    for _ in range(_MAX_ATTEMPTS):
        u = (rng.randint(-6, 6), rng.randint(-6, 6))
        v = (rng.randint(-6, 6), rng.randint(-6, 6))
        if variant == "derived_perp":
            mult = rng.choice([2, 3])
            w = (mult * u[0] - v[0], mult * u[1] - v[1])
            if 0 in u or 0 in v or w == (0, 0) or _dot(v, w) == 0 or _cross(u, v) == 0:
                continue
            num = -_dot(u, w)
            k = Fraction(num, _dot(v, w))
            if k.denominator != 1 or k == 0:
                continue
            k = int(k)
            c = (u[0] + k * v[0], u[1] + k * v[1])
            check(c != (0, 0) and _dot(c, w) == 0, "u + kv is not perpendicular to w")
            if k in (*u, *v, mult):
                continue  # a vector copied out of the prompt must not read as the key
            break
        k = rng.choice([s for s in range(-7, 8) if s != 0])
        c = (u[0] + k * v[0], u[1] + k * v[1])
        if c == (0, 0) or 0 in u or 0 in v or _cross(u, v) == 0:
            continue
        scale = rng.choice([-3, -2, -1, 1, 2, 3])
        if variant == "comb_perp":
            w = (-scale * c[1], scale * c[0])
            if _dot(v, w) == 0:
                continue
            check(_dot(c, w) == 0, "u + kv is not perpendicular to w")
        else:
            w = (scale * c[0], scale * c[1])
            if _cross(v, w) == 0:
                continue
            check(_cross(c, w) == 0, "u + kv is not parallel to w")
        if k in (*u, *v, *w):
            continue  # a vector copied out of the prompt must not read as the key
        break
    else:
        raise RuntimeError("no parallel/perpendicular draw")

    perp_formula = Fraction(-_dot(u, w), _dot(v, w)) if _dot(v, w) else None
    par_formula = Fraction(-_cross(u, w), _cross(v, w)) if _cross(v, w) else None
    answer = k
    if variant == "comb_par":
        cands = [-k, perp_formula, Fraction(-u[0], v[0])]
        cond = "parallel to"
        rel = f"the vector {_pair(*w)}"
        text = f"u is the vector {_pair(*u)} and v is the vector {_pair(*v)}. Find the value of k for which u + kv is {cond} {rel}."
        working = [
            f"u + kv = ({u[0]} + {paren_negative(v[0])}k, {u[1]} + {paren_negative(v[1])}k).",
            f"Parallel to w means the cross product is 0, which gives k = {k}.",
        ]
    else:
        w_swap = (w[1], w[0])
        cands = [
            -k,
            par_formula,  # applied the parallel condition instead
            Fraction(-_dot(v, w), _dot(u, w)) if _dot(u, w) else None,  # inverted the ratio
            Fraction(-_dot(u, w_swap), _dot(v, w_swap)) if _dot(v, w_swap) else None,  # swapped w's components
            Fraction(-u[1], v[1]),  # made only the j-component vanish
        ]
        if variant == "derived_perp":
            # w built with the wrong sign (mu + v), and w built as m(u - v): each followed through correctly
            for w_bad in ((mult * u[0] + v[0], mult * u[1] + v[1]), (mult * (u[0] - v[0]), mult * (u[1] - v[1]))):
                cands.append(Fraction(-_dot(u, w_bad), _dot(v, w_bad)) if _dot(v, w_bad) else None)
        if variant == "comb_perp":
            text = (
                f"u is the vector {_pair(*u)} and v is the vector {_pair(*v)}. Find the value of k for which u + kv "
                f"is perpendicular to the vector {_pair(*w)}."
            )
            working = [
                f"(u + kv)·w = 0 gives {_dot(u, w)} + {paren_negative(_dot(v, w))}k = 0, so k = {k}.",
            ]
        else:
            text = (
                f"u is the vector {_pair(*u)} and v is the vector {_pair(*v)}. Let w be the vector {mult}u - v. "
                f"Find the value of k for which u + kv is perpendicular to w."
            )
            working = [
                f"w = {_pair(*w)}.",
                f"(u + kv)·w = 0 gives {_dot(u, w)} + {paren_negative(_dot(v, w))}k = 0, so k = {k}.",
            ]
    return _number_result(answer, cands, text, working)
