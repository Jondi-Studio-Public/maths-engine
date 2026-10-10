"""Velocity/acceleration to displacement, motion in a line.

Two skills, one file, one author -- a single-vs-double antidifferentiation
pair sharing the same "particle on a line" physical setting:

- ``senior_determine_displacement_given_velocity_in_linear_motion_problems``
  (``senior-determine-displacement-given-velocity-in-linear-motion-problems``,
  ACMMM123, topic ``p10a_velocity_to_displacement_line``) -- ONE
  antidifferentiation step: given ``v(t)`` and one stated position at one
  instant, find ``s(t)``.
- ``senior_determine_positions_given_acceleration_and_initial_values_of``
  (``senior-determine-positions-given-acceleration-and-initial-values-of``,
  ACMMM135, topic ``p10a_acceleration_to_position_line``) -- TWO
  antidifferentiation steps: given ``a(t)`` and BOTH velocity and position at
  one shared instant, find ``s(t)``.

**The ACMMM122 boundary** (``senior-determine-f-left-x-right-given-f-x``,
``mathgen/generators/p10a_antiderivative_ic_build.py``, built in
parallel). That skill is PURE, context-free antidifferentiation with no
physical setting at all -- no particle, no velocity, no time in seconds. Both
skills below apply the identical technique inside a stated linear-motion word
problem, so they never collapse into ACMMM122's abstract-notation shape.

**The ACMMM133 boundary**
(``senior-calculate-total-change-by-integrating-instantaneous-or-marginal``,
``mathgen/generators/p6c_total_change_build.py``, already shipped). That
skill is a DEFINITE integral over a stated interval with NO initial
condition -- its answer is always a bare number ("total change from t=a to
t=b"). Both skills below are INDEFINITE antidifferentiation resolved by an
initial condition: the answer is always a function of ``t`` (never a bare
number), and every question states a position/velocity at one specific
instant rather than a change over an interval.

**A pinned-design defect found and fixed here.** The ``sign-error-on-constant``-family misconceptions
compute ``candidate - true_answer = 2*G(a)`` (skill 1) / ``2*G2(a_ic)`` (skill 2), and ``G``/``G2`` are
antiderivatives built with NO constant term, so ``G(0)`` and ``G2(0)`` are identically zero for ANY
coefficients -- not merely occasionally, by construction. Skill 1's easy tier fixes ``a = 0``; skill 2's easy
AND medium tiers fix ``a_ic = 0``. At those tiers the sign-error candidate is therefore ALWAYS equal to the
true answer, and no amount of resampling can change that (the collision does not depend on `s0`/`v0` at all).
This is the same reasoning already applied to skill 2's hard-tier-only constraint ("hard tier ONLY, since
a_ic = 0 at easy/medium makes C1 trivial and this collapses to the true value") -- applied here to the sibling
misconception that was not also restricted. Fix: the sign-error-on-constant misconception (skill 1) is
offered at medium/hard only (never easy); the sign-error-on-second-constant misconception (skill 2) is offered
at hard only (never easy/medium) -- i.e. it now shares hard-only scope with skill 2's other misconception, for the
identical reason.

A second, narrower instance in skill 1's own hard-tier ``misread-worded-sign``:
``candidate - true_answer = abs(s0) - s0``, which is zero whenever
``s0 > 0`` (a "right of the origin" question), not merely "always distinct
since s0 != 0" as might be assumed. Fix: that misconception is offered only
when the drawn `s0` is negative ("left of the origin").
"""

from __future__ import annotations

from sympy import Rational, diff, expand, symbols

from ..core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors

t = symbols("t")


def _typed(expr) -> str:
    """Learner-facing typed form for answer_text/distractors: ** -> ^."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """Learner-facing prose form for question text: ** -> ^, * -> the middle dot."""
    return str(expr).replace("**", "^").replace("*", "·")


# --- Skill 1: velocity -> displacement, one antidifferentiation step --------

_N_POOL = tuple(v for v in range(-6, 7) if v != 0 and 2 <= abs(v) <= 6)
_K_POOL = tuple(v for v in range(-9, 10) if v != 0)
_M_MEDIUM_POOL = (-9, -6, -3, 3, 6, 9)
_M_HARD_POOL = tuple(v for v in range(-8, 9) if v != 0 and v % 3 != 0)
_A_POOL_S1 = (1, 2, 3, 4)


def _draw_skill1(rng, tier):
    """Draw (M, N, K, a, s0) satisfying the resample guard. `G(a)` depends
    only on the coefficients/`a`, so a coefficient collision (`G(a) == 0` at
    medium/hard) cannot be fixed by redrawing `s0` alone -- redraw
    coefficients/`a` first in that case, then `s0` up to 50 times against the
    resulting `G(a)`. At easy tier `a = 0` is fixed and `G(a) == 0`
    identically (see module docstring), so only `C != 0` can ever bind."""
    for _coeff_attempt in range(10):
        if tier == "easy":
            M, a = 0, 0
        elif tier == "medium":
            M = rng.choice(_M_MEDIUM_POOL)
            a = rng.choice(_A_POOL_S1)
        else:
            M = rng.choice(_M_HARD_POOL)
            a = rng.choice(_A_POOL_S1)
        N = rng.choice(_N_POOL)
        K = rng.choice(_K_POOL)
        G = Rational(M, 3) * t**3 + Rational(N, 2) * t**2 + K * t
        G_a = G.subs(t, a)
        if tier != "easy" and G_a == 0:
            continue  # this coefficient/`a` draw structurally collides; redraw
        for _s0_attempt in range(50):
            if tier == "easy":
                s0 = rng.randint(-15, 15)
            elif tier == "medium":
                s0 = rng.randint(-20, 20)
            else:
                s0 = rng.choice([v for v in range(-20, 21) if v != 0])
            C = s0 - G_a
            if C != 0:
                return M, N, K, a, s0, G, G_a, C
    check(False, f"skill1 resample guard did not converge for tier={tier}")


@generator(
    "p10a_velocity_to_displacement_line",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p10a_velocity_to_displacement_line(rng, tier="easy"):
    """v(t) = M*t^2 + N*t + K m/s (t >= 0). At t = a, s(a) = s0 m. Find s(t).

    easy:   M = 0 (linear velocity), a = 0 (fixed), s0 a plain signed integer.
    medium: M nonzero, divisible by 3 (keeps G(t)'s cubic coefficient an
            integer), a a nonzero small integer -- the escalation is a
            nonzero IC point, not magnitude.
    hard:   M nonzero, NOT divisible by 3 (the answer's cubic coefficient is
            a genuine fraction) -- a fraction power-rule escalation; s0 is
            stated in worded left/right-of-origin form instead of a signed
            number -- a sign-reading escalation.
    """
    M, N, K, a, s0, G, G_a, C = _draw_skill1(rng, tier)

    v_expr = M * t**2 + N * t + K
    S = expand(G + C)

    # Independent check: differentiation route, never a second integration.
    check_equivalent(diff(S, t), v_expr, f"S does not differentiate back to v(t): M={M}, N={N}, K={K}")
    check_equivalent(S.subs(t, a), s0, f"S(a) does not equal the stated initial position: a={a}, s0={s0}")

    v_prose = _prose(v_expr)
    if tier != "hard":
        question_text = (
            f"A particle moves along a straight line with velocity v(t) = {v_prose} m/s "
            f"at time t seconds (t ≥ 0). At time t = {a}, the particle's position is "
            f"{s0} m. Find an expression for the particle's position s(t) at time t."
        )
    else:
        side = "left" if s0 < 0 else "right"
        question_text = (
            f"A particle moves along a straight line with velocity v(t) = {v_prose} m/s "
            f"at time t seconds (t ≥ 0). At time t = {a}, the particle is {abs(s0)} m "
            f"to the {side} of the origin. Find an expression for the particle's position "
            f"s(t) at time t."
        )

    answer_text = _typed(S)

    # Misconceptions -- see module docstring for the tier-restriction fixes.
    candidates_text = []

    forgot_constant = G
    check_distinct(S, forgot_constant, msg=f"forgot-constant equals answer: M={M}, N={N}, K={K}, a={a}")
    candidates_text.append(_typed(forgot_constant))

    wrong_G = M * t**3 + N * t**2 + K * t
    wrong_C = s0 - wrong_G.subs(t, a)
    power_rule_no_divide = expand(wrong_G + wrong_C)
    check_distinct(S, power_rule_no_divide, msg=f"power-rule-no-divide equals answer: M={M}, N={N}, K={K}, a={a}")
    candidates_text.append(_typed(power_rule_no_divide))

    if tier != "easy":
        # Only distinct once G(a) != 0, guaranteed by the resample guard at
        # medium/hard (never at easy, where a = 0 forces G(a) == 0 always).
        sign_error_on_constant = expand(G + (s0 + G_a))
        check_distinct(
            S, sign_error_on_constant, msg=f"sign-error-on-constant equals answer: M={M}, N={N}, K={K}, a={a}"
        )
        candidates_text.append(_typed(sign_error_on_constant))

    if tier == "hard" and s0 < 0:
        # Only distinct when s0 < 0 ("left of origin"): abs(s0) - s0 == 0
        # whenever s0 >= 0, so this misconception cannot fire on a "right of
        # the origin" question -- see module docstring.
        wrong_C2 = abs(s0) - G_a
        misread_worded_sign = expand(G + wrong_C2)
        check_distinct(
            S, misread_worded_sign, msg=f"misread-worded-sign equals answer: M={M}, N={N}, K={K}, a={a}, s0={s0}"
        )
        candidates_text.append(_typed(misread_worded_sign))

    distractors = pick_distractors(answer_text, candidates_text)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": str(S),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
    }


# --- Skill 2: acceleration -> position, two antidifferentiation steps ------

_Q_POOL = tuple(v for v in range(-9, 10) if v != 0)
_P_POOL = tuple(v for v in range(-6, 7) if v != 0)
_A_IC_HARD_POOL = (1, 2, 3)
_V0_POOL = tuple(v for v in range(-9, 10) if v != 0)
_S0_POOL = tuple(v for v in range(-15, 16) if v != 0)


def _draw_skill2(rng, tier):
    """Draw (P, Q, a_ic, v0, s0) satisfying the resample guard. At easy/medium
    `a_ic = 0` is fixed, so `G1(a_ic) == G2(a_ic) == 0` identically and
    `C1 == v0`, `C2 == s0` automatically -- both nonzero by construction
    since `v0`/`s0` are drawn from nonzero pools, so no loop is needed there.
    At hard, `a_ic` is a nonzero draw and all four guard conditions can
    collide with the coefficients, so redraw coefficients/`a_ic` first, then
    `v0`/`s0` up to 50 times against the resulting G1/G2."""
    for _coeff_attempt in range(10):
        if tier == "easy":
            P, a_ic = 0, 0
        elif tier == "medium":
            P = rng.choice(_P_POOL)
            a_ic = 0
        else:
            P = rng.choice(_P_POOL)
            a_ic = rng.choice(_A_IC_HARD_POOL)
        Q = rng.choice(_Q_POOL)
        G1 = Rational(P, 2) * t**2 + Q * t
        G1_a = G1.subs(t, a_ic)
        if tier == "hard" and G1_a == 0 and a_ic != 0:
            # Possible only at hard, where a_ic != 0 and coefficients vary;
            # a coincidental G1(a_ic) == 0 needs a coefficient/a_ic redraw
            # since it does not depend on v0/s0 at all.
            continue
        for _vs_attempt in range(50):
            v0 = rng.choice(_V0_POOL)
            s0 = rng.choice(_S0_POOL)
            C1 = v0 - G1_a
            if C1 == 0:
                continue
            G2 = Rational(P, 6) * t**3 + Rational(Q, 2) * t**2 + C1 * t
            G2_a = G2.subs(t, a_ic)
            C2 = s0 - G2_a
            if C2 == 0:
                continue
            if tier == "hard" and G2_a == 0:
                continue
            return P, Q, a_ic, v0, s0, G1, G1_a, C1, G2, G2_a, C2
    check(False, f"skill2 resample guard did not converge for tier={tier}")


def _draw_skill2_split(rng):
    """Split-instant draw: velocity v0 stated at t = 0, position s0 stated at
    a LATER instant t = b. C1 == v0 (V = G1 + v0); C2 = s0 - G2(b), where
    G2 = P/6 t^3 + Q/2 t^2 + v0*t. Resample so G2(b) != 0 (else the
    "position given at t = 0" misconception collapses onto the answer) and
    C2 != 0. Up to 50 tries, then fail loudly."""
    for _attempt in range(50):
        P = rng.choice(_P_POOL)
        Q = rng.choice(_Q_POOL)
        v0 = rng.choice(_V0_POOL)
        b = rng.choice((1, 2, 3))
        s0 = rng.choice(_S0_POOL)
        G1 = Rational(P, 2) * t**2 + Q * t
        C1 = v0
        G2 = Rational(P, 6) * t**3 + Rational(Q, 2) * t**2 + v0 * t
        G2_b = G2.subs(t, b)
        C2 = s0 - G2_b
        if C1 == 0 or G2_b == 0 or C2 == 0:
            continue
        return P, Q, b, v0, s0, G1, C1, G2, G2_b, C2
    check(False, "skill2 split resample guard did not converge")


def _acceleration_to_position_split(rng):
    P, Q, b, v0, s0, G1, C1, G2, G2_b, C2 = _draw_skill2_split(rng)

    a_expr = P * t + Q
    V = expand(G1 + C1)
    S = expand(G2 + C2)

    check_equivalent(diff(V, t), a_expr, f"V does not differentiate back to a(t): P={P}, Q={Q}")
    check_equivalent(V.subs(t, 0), v0, f"V(0) does not equal the stated initial velocity: v0={v0}")
    check_equivalent(diff(S, t), V, f"S does not differentiate back to V(t): P={P}, Q={Q}")
    check_equivalent(S.subs(t, b), s0, f"S(b) does not equal the stated position: b={b}, s0={s0}")

    a_prose = _prose(a_expr)
    question_text = (
        f"A particle moves along a straight line with acceleration a(t) = {a_prose} m/s² "
        f"at time t seconds (t ≥ 0). At time t = 0, its velocity is {v0} m/s. "
        f"At time t = {b}, its position is {s0} m. Find an expression for its position s(t) at time t."
    )
    answer_text = _typed(S)

    candidates_text = []

    forgot_second_constant = G2
    check_distinct(S, forgot_second_constant, msg=f"split forgot-second-constant equals answer: P={P}, Q={Q}, b={b}")
    candidates_text.append(_typed(forgot_second_constant))

    wrong_G2 = Rational(P, 6) * t**3 + Rational(Q, 2) * t**2  # no v0*t term
    forgot_first_constant = expand(wrong_G2 + (s0 - wrong_G2.subs(t, b)))
    check_distinct(S, forgot_first_constant, msg=f"split forgot-first-constant equals answer: P={P}, Q={Q}, b={b}")
    candidates_text.append(_typed(forgot_first_constant))

    sign_error_on_second_constant = expand(G2 + (s0 + G2_b))
    check_distinct(S, sign_error_on_second_constant, msg=f"split sign-error equals answer: P={P}, Q={Q}, b={b}")
    candidates_text.append(_typed(sign_error_on_second_constant))

    position_at_zero = expand(G2 + s0)  # treated the position as given at t = 0
    check_distinct(S, position_at_zero, msg=f"split position-at-zero equals answer: P={P}, Q={Q}, b={b}")
    candidates_text.append(_typed(position_at_zero))

    distractors = pick_distractors(answer_text, candidates_text)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": str(S),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
    }


@generator(
    "p10a_acceleration_to_position_line",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p10a_acceleration_to_position_line(rng, tier="easy", split=False):
    """a(t) = P*t + Q m/s^2 (t >= 0). At t = a_ic, v(a_ic) = v0 m/s and
    s(a_ic) = s0 m (the SAME instant for both). Find s(t).

    easy:   P = 0 (constant acceleration), a_ic = 0 (fixed).
    medium: P nonzero (variable acceleration), a_ic = 0 (fixed) -- the
            escalation is a richer a(t), not magnitude.
    hard:   as medium, but a_ic is a nonzero small integer -- BOTH constants
            of integration must be solved against a nonzero instant, the
            genuinely new reasoning step.
    hard + split=True: velocity is stated at t = 0 and position at a LATER
            instant t = b, so the two constants are solved at different
            instants. Any other tier with split=True raises ValueError.
    """
    if split:
        if tier != "hard":
            raise ValueError("split=True is only meaningful with tier='hard'")
        return _acceleration_to_position_split(rng)
    P, Q, a_ic, v0, s0, G1, G1_a, C1, G2, G2_a, C2 = _draw_skill2(rng, tier)

    a_expr = P * t + Q
    V = expand(G1 + C1)
    S = expand(G2 + C2)

    # Independent check: differentiation route, both steps.
    check_equivalent(diff(V, t), a_expr, f"V does not differentiate back to a(t): P={P}, Q={Q}")
    check_equivalent(V.subs(t, a_ic), v0, f"V(a_ic) does not equal the stated initial velocity: a_ic={a_ic}, v0={v0}")
    check_equivalent(diff(S, t), V, f"S does not differentiate back to V(t): P={P}, Q={Q}")
    check_equivalent(S.subs(t, a_ic), s0, f"S(a_ic) does not equal the stated initial position: a_ic={a_ic}, s0={s0}")

    a_prose = _prose(a_expr)
    question_text = (
        f"A particle moves along a straight line with acceleration a(t) = {a_prose} m/s² "
        f"at time t seconds (t ≥ 0). At time t = {a_ic}, its velocity is {v0} m/s and its "
        f"position is {s0} m. Find an expression for its position s(t) at time t."
    )

    answer_text = _typed(S)

    # Misconceptions -- see module docstring for the tier-restriction fix.
    candidates_text = []

    forgot_second_constant = G2
    check_distinct(S, forgot_second_constant, msg=f"forgot-second-constant equals answer: P={P}, Q={Q}, a_ic={a_ic}")
    candidates_text.append(_typed(forgot_second_constant))

    wrong_G2 = Rational(P, 6) * t**3 + Rational(Q, 2) * t**2  # no C1*t term
    C2_prime = s0 - wrong_G2.subs(t, a_ic)
    forgot_first_constant = expand(wrong_G2 + C2_prime)
    check_distinct(S, forgot_first_constant, msg=f"forgot-first-constant equals answer: P={P}, Q={Q}, a_ic={a_ic}")
    candidates_text.append(_typed(forgot_first_constant))

    if tier == "hard":
        # Only distinct once G2(a_ic) != 0, guaranteed by the resample guard
        # at hard only (never at easy/medium, where a_ic = 0 forces
        # G2(a_ic) == 0 always -- see module docstring).
        sign_error_on_second_constant = expand(G2 + (s0 + G2_a))
        check_distinct(
            S,
            sign_error_on_second_constant,
            msg=f"sign-error-on-second-constant equals answer: P={P}, Q={Q}, a_ic={a_ic}",
        )
        candidates_text.append(_typed(sign_error_on_second_constant))

        C1_prime = v0 + G1_a
        wrong_G2_prime = Rational(P, 6) * t**3 + Rational(Q, 2) * t**2 + C1_prime * t
        C2_double_prime = s0 - wrong_G2_prime.subs(t, a_ic)
        sign_error_on_first_constant = expand(wrong_G2_prime + C2_double_prime)
        check_distinct(
            S,
            sign_error_on_first_constant,
            msg=f"sign-error-on-first-constant equals answer: P={P}, Q={Q}, a_ic={a_ic}",
        )
        candidates_text.append(_typed(sign_error_on_first_constant))

    distractors = pick_distractors(answer_text, candidates_text)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": str(S),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
    }
