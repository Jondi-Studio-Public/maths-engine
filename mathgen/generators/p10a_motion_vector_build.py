"""Vector calculus, motion in a line and a plane.

Senior Specialist Mathematics, domain `vectors`, framework `acara-senior-v8-4`.

- p10a_vector_motion_line_accel (ACMSM114, senior-determine-equations-of-motion-of-
  a-particle-travelling): a particle confined to a straight line always parallel to
  a fixed vector `d`, with acceleration `g(t)*d` (constant `G0` easy/medium, variable
  `H*t` hard). Antidifferentiate twice (speed-along-d, then displacement-along-d) and
  report ONE named axis component of the resulting position vector.

- p10a_projectile_motion_vector (ACMSM115, senior-apply-vector-calculus-to-motion-in-
  a-plane): constant-acceleration projectile motion (`a(t) = -g*j`). Easy asks for a
  velocity component; medium/hard ask for a position component (double integration).
  **Scope**: this build covers PROJECTILE motion only.
  Circular motion (trigonometric position components), also named in the ACMSM115
  descriptor, is out of scope.

Both skills use `answer: number` and ask for exactly ONE named axis's component,
never a full vector expression -- the workaround `p4c_vectors_component_build.py`
already established for `senior-express-a-vector-in-component-form-using-the`, made
necessary by a live, unfixed defect in `mathgen/grading/typed.py`'s expression parser: a
numeral immediately followed by a bare `j` (`4j`, `2j`) is silently read as SymPy's
imaginary unit, not the vector `j` component. Neither skill below ever
asks the learner to type a vector.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import diff, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    pick_distractors,
)

_t = symbols("t")


def _prose(expr) -> str:
    """Learner-facing rendering: ``**`` -> ``^``, ``*`` -> the middle dot
    (never a bare asterisk -- `mathgen/text_quality.py`'s `star_operator`
    check would flag a literal ``*``). Same idiom as
    `p9c_difference_quotient_build.py`'s own `_prose`; not shared centrally
    because it is two lines and used only by a handful of topics."""
    return str(expr).replace("**", "^").replace("*", "·")


def _seconds(n: int) -> str:
    """ "second"/"seconds" for a stated time count -- `mathgen/text_quality.py`'s
    `singular_count_plural` check flags a bare "1 seconds"."""
    return "second" if n == 1 else "seconds"


# --- Skill 1: motion in a straight line, constant/variable acceleration ----

_DIRECTION_POOL = ((3, 4), (4, 3), (1, 2), (2, 1), (2, 3), (3, 2), (1, 1))
_G0_POOL = (-6, -4, -2, 2, 4, 6)
_H_POOL = (-12, -6, 6, 12)
_SCALAR_POOL = tuple(range(-9, 0)) + tuple(range(1, 10))  # [-9,-1] u [1,9]
_T_POOL_CONST = (1, 2, 3, 4)
_T_POOL_VAR = (1, 2, 3)
_MAX_ATTEMPTS = 300


@generator(
    "p10a_vector_motion_line_accel",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p10a_vector_motion_line_accel(rng, tier="easy"):
    """Equations of motion of a particle confined to a straight line parallel
    to a fixed vector `d`, acceleration `g(t)*d`.

    easy/medium: g(t) = G0 (nonzero even constant). hard: g(t) = H*t (H a
    nonzero multiple of 6). easy: starts at the origin. medium/hard: starts
    at a stated nonzero r0. Reports ONE named axis's component of the
    position vector at a stated time T.
    """
    for _attempt in range(_MAX_ATTEMPTS):
        dx, dy = rng.choice(_DIRECTION_POOL)
        f0 = rng.choice(_SCALAR_POOL)
        asked_axis = rng.choice(["i", "j"])

        if tier == "hard":
            H = rng.choice(_H_POOL)
            T = rng.choice(_T_POOL_VAR)
            F_expr = H * _t**2 / 2 + f0
            S_expr = H * _t**3 / 6 + f0 * _t
            g_expr = H * _t
        else:
            G0 = rng.choice(_G0_POOL)
            T = rng.choice(_T_POOL_CONST)
            F_expr = G0 * _t + f0
            S_expr = G0 * _t**2 / 2 + f0 * _t
            g_expr = G0

        if tier == "easy":
            x0 = y0 = 0
        else:
            x0 = rng.choice(_SCALAR_POOL)
            y0 = rng.choice(_SCALAR_POOL)

        S_T_frac = Fraction(S_expr.subs(_t, T))
        check(S_T_frac.denominator == 1, f"S(T) not integer: tier={tier}")
        S_T = int(S_T_frac)

        d_component = dx if asked_axis == "i" else dy
        other_d_component = dy if asked_axis == "i" else dx
        r0_component = x0 if asked_axis == "i" else y0
        other_r0_component = y0 if asked_axis == "i" else x0

        answer = r0_component + S_T * d_component

        # Misconception 1: forgot-initial-scalar-velocity -- drops the
        # `+f0*t` term when forming S(t), as if the particle started from
        # rest along the line. Always exercisable, every tier.
        if tier == "hard":
            s_wrong1_frac = Fraction(H * T**3, 6)
        else:
            s_wrong1_frac = Fraction(G0 * T**2, 2)
        check(s_wrong1_frac.denominator == 1, f"S_wrong1 not integer: tier={tier}")
        cand_forgot_initial = r0_component + int(s_wrong1_frac) * d_component

        # Misconception 3: wrong-axis-component -- the fully-correct value
        # for the OTHER axis (both the direction component AND the offset
        # swapped together). Always exercisable.
        cand_wrong_axis = other_r0_component + S_T * other_d_component

        candidates = [cand_forgot_initial, cand_wrong_axis]

        # Misconception 2: forgot-position-offset -- medium/hard only
        # (collides with the answer at easy, where r0 = (0,0)).
        if tier != "easy":
            cand_forgot_offset = S_T * d_component
            candidates.append(cand_forgot_offset)

        # Misconception 4: power-rule-no-divide -- hard only (mis-integrates
        # H*t, dividing by 2 instead of 6).
        if tier == "hard":
            s_wrong4_frac = Fraction(H * T**3, 2) + f0 * T
            check(s_wrong4_frac.denominator == 1, f"S_wrong4 not integer: tier={tier}")
            cand_power_rule = r0_component + int(s_wrong4_frac) * d_component
            candidates.append(cand_power_rule)

        if answer not in candidates:
            break
    else:
        raise RuntimeError(f"could not draw a collision-free p10a_vector_motion_line_accel question: tier={tier}")

    check_distinct(answer, *candidates, msg=f"a distractor equals the answer: tier={tier}, dx={dx}, dy={dy}")

    # Independent check: differentiation route, wholly separate from the
    # closed-form-then-substitute construction above.
    check_equivalent(diff(F_expr, _t), g_expr, f"F'(t) != g(t): tier={tier}")
    check_equivalent(F_expr.subs(_t, 0), f0, f"F(0) != f0: tier={tier}")
    check_equivalent(diff(S_expr, _t), F_expr, f"S'(t) != F(t): tier={tier}")
    check_equivalent(S_expr.subs(_t, 0), 0, f"S(0) != 0: tier={tier}")

    recomputed = (x0 if asked_axis == "i" else y0) + S_expr.subs(_t, T) * (dx if asked_axis == "i" else dy)
    check_equivalent(recomputed, answer, f"independent final-number check failed: tier={tier}")

    d_text = join_terms([(dx, "i"), (dy, "j")])
    g_prose = _prose(g_expr)
    if tier == "easy":
        offset_clause = "it starts at the origin"
    else:
        r0_text = join_terms([(x0, "i"), (y0, "j")])
        offset_clause = f"its position vector is r(0) = {r0_text} m"

    question_text = (
        f"A particle moves in a straight line, always parallel to the vector d = {d_text}. "
        f"Its acceleration vector is {g_prose}·d m/s² at time t seconds (t ≥ 0). "
        f"At time t = 0, its velocity vector is {f0}d m/s, and {offset_clause}. "
        f"Find the {asked_axis}-component of the particle's position vector, "
        f"in metres, at time t = {T} {_seconds(T)}."
    )

    answer_text = str(int(answer))
    distractors = pick_distractors(answer_text, candidates, n=len(candidates))
    working = [
        f"S({T}) = {S_T} (scalar displacement along d since t = 0).",
        f"{asked_axis}-component of r({T}) = {r0_component} + {paren_negative(S_T)}×{d_component} = {answer}.",
    ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=int(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(int(answer)),
        distractors=distractors,
        working=working,
    )


# --- Skill 2: projectile motion in a plane -----------------------------

_G = 10  # m/s^2, fixed every seed so g/2 = 5 stays an exact integer.
_VX0_POOL = tuple(range(-9, 0)) + tuple(range(1, 10))
_VY0_POOL = tuple(range(1, 10))
_T_POOL2 = (1, 2, 3)


@generator(
    "p10a_projectile_motion_vector",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p10a_projectile_motion_vector(rng, tier="easy"):
    """Projectile motion in a plane: a(t) = -g*j (g = 10, fixed). Easy asks
    for a velocity component; medium/hard ask for a position component
    (double integration). hard adds a nonzero initial position.

    Scope: projectile motion (constant acceleration vector) only -- circular
    motion is out of scope for this build (see module docstring).
    """
    question_kind = "velocity" if tier == "easy" else "position"
    for _attempt in range(_MAX_ATTEMPTS):
        vx0 = rng.choice(_VX0_POOL)
        vy0 = rng.choice(_VY0_POOL)
        T = rng.choice(_T_POOL2)
        asked_axis = rng.choice(["i", "j"])

        if tier == "hard":
            x0 = rng.choice(_SCALAR_POOL)
            y0 = rng.choice(_SCALAR_POOL)
        else:
            x0 = y0 = 0

        vx_T = vx0
        vy_T = vy0 - _G * T
        x_T = x0 + vx0 * T
        y_T_frac = Fraction(y0) + Fraction(vy0 * T) - Fraction(_G, 2) * T**2
        check(y_T_frac.denominator == 1, f"y(T) not integer: tier={tier}")
        y_T = int(y_T_frac)

        if question_kind == "velocity":
            answer = vx_T if asked_axis == "i" else vy_T
            other_answer = vy_T if asked_axis == "i" else vx_T
        else:
            answer = x_T if asked_axis == "i" else y_T
            other_answer = y_T if asked_axis == "i" else x_T

        candidates = [other_answer]  # misconception 3: wrong-axis-component

        # misconception 2: dropped-initial-component. Always exercisable.
        if question_kind == "velocity":
            dropped = 0 if asked_axis == "i" else -_G * T
        else:
            if asked_axis == "i":
                dropped = x0
            else:
                dropped_frac = Fraction(y0) - Fraction(_G, 2) * T**2
                check(dropped_frac.denominator == 1, f"dropped(y) not integer: tier={tier}")
                dropped = int(dropped_frac)
        candidates.append(dropped)

        # misconception 1: sign-error-on-gravity. Any question touching the
        # y-axis (both velocity and position).
        if asked_axis == "j":
            if question_kind == "velocity":
                sign_err = vy0 + _G * T
            else:
                sign_err_frac = Fraction(y0) + Fraction(vy0 * T) + Fraction(_G, 2) * T**2
                check(sign_err_frac.denominator == 1, f"sign_err not integer: tier={tier}")
                sign_err = int(sign_err_frac)
            candidates.append(sign_err)

        # misconception 4: forgot-position-offset. hard tier + position
        # questions only (collides with the answer at medium, r0 = (0,0)).
        if tier == "hard" and question_kind == "position":
            if asked_axis == "i":
                forgot_offset = vx0 * T
            else:
                forgot_offset_frac = Fraction(vy0 * T) - Fraction(_G, 2) * T**2
                check(forgot_offset_frac.denominator == 1, f"forgot_offset not integer: tier={tier}")
                forgot_offset = int(forgot_offset_frac)
            candidates.append(forgot_offset)

        if answer not in candidates:
            break
    else:
        raise RuntimeError(f"could not draw a collision-free p10a_projectile_motion_vector question: tier={tier}")

    check_distinct(answer, *candidates, msg=f"a distractor equals the answer: tier={tier}, vx0={vx0}, vy0={vy0}")

    # Independent check: differentiation route.
    x_expr = x0 + vx0 * _t
    y_expr = y0 + vy0 * _t - Fraction(_G, 2) * _t**2
    vy_expr = vy0 - _G * _t

    check_equivalent(diff(x_expr, _t), vx0, f"x'(t) != vx0: tier={tier}")
    check_equivalent(x_expr.subs(_t, 0), x0, f"x(0) != x0: tier={tier}")
    check_equivalent(diff(y_expr, _t), vy_expr, f"y'(t) != vy(t): tier={tier}")
    check_equivalent(y_expr.subs(_t, 0), y0, f"y(0) != y0: tier={tier}")
    check_equivalent(diff(vy_expr, _t), -_G, f"vy'(t) != -g: tier={tier}")

    if question_kind == "velocity":
        recomputed = vx0 if asked_axis == "i" else vy_expr.subs(_t, T)
    else:
        recomputed = x_expr.subs(_t, T) if asked_axis == "i" else y_expr.subs(_t, T)
    check_equivalent(recomputed, answer, f"independent final-number check failed: tier={tier}")

    v0_text = join_terms([(vx0, "i"), (vy0, "j")])
    if tier == "hard":
        r0_text = join_terms([(x0, "i"), (y0, "j")])
        location_clause = f"from position vector r(0) = {r0_text} m"
    else:
        location_clause = "from the origin"
    unit = "m/s" if question_kind == "velocity" else "m"

    question_text = (
        f"A projectile is launched with velocity vector v(0) = {v0_text} m/s, {location_clause}. "
        f"Its acceleration vector is a(t) = -{_G}j m/s² (gravity). "
        f"Find the {asked_axis}-component of the particle's {question_kind} vector, "
        f"in {unit}, at time t = {T} {_seconds(T)}."
    )

    answer_text = str(int(answer))
    distractors = pick_distractors(answer_text, candidates, n=len(candidates))
    working = [
        f"{question_kind} {asked_axis}-component at t = {T}: {answer_text}.",
    ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=int(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(int(answer)),
        distractors=distractors,
        working=working,
    )
