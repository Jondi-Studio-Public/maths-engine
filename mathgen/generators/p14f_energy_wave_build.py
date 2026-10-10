"""Generates `p14f_energy_conservation_check` and `p14f_wave_parameter_relation` questions: spring-mass energy
differences and wave speed, frequency and wavelength relations.

Two mathematical-physics skills.

``ucv1-energy-conservation-check`` (topic ``p14f_energy_conservation_check``)
computes the numeric DIFFERENCE in total mechanical energy
E = (1/2)k x^2 + (1/2) m v^2 between two independently supplied states of a
horizontal spring-mass system. It reuses the same k, m as the shipping
``simple-harmonic-angular-frequency`` (which computes omega = sqrt(k/m) for
ONE state) but is a different formula (energy, not frequency) over a
different observable count (two combined states, never omega).

``ucv1-wave-parameter-relation`` (topic ``p14f_wave_parameter_relation``) solves for wave SPEED v from the
other two of {wavelength, frequency, wave speed} -- the one unknown the shipping
``wave-frequency-from-speed-wavelength`` neighbour never asks for at any tier (its hard tier already solves
for wavelength; easy/medium already solve for frequency). Every tier here also withholds the formula v =
f*lambda from the prompt (the neighbour always prints it) except the exam tier, which must state it to make
the verification-before-compute check meaningful.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import (
    check,
    check_conversion,
    check_distinct,
    check_equivalent,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_RESAMPLE_ATTEMPTS = 300
_DP1_TOL = stated_precision_tolerance(1)


def _draw_nonzero(rng, lo, hi):
    while True:
        v = rng.randint(lo, hi)
        if v != 0:
            return v


def _energy(k, x, m, v) -> Fraction:
    """E = (1/2)k x^2 + (1/2) m v^2, exact via Fraction (never float)."""
    return Fraction(k, 2) * x * x + Fraction(m, 2) * v * v


def _fmt1(value) -> str:
    return stated_precision_string(Fraction(value), 1)


def _gap_ok(values, tol=_DP1_TOL) -> bool:
    fracs = [Fraction(v) for v in values]
    for i in range(len(fracs)):
        for j in range(i + 1, len(fracs)):
            if abs(fracs[i] - fracs[j]) <= tol:
                return False
    return True


def _all_distinct(values) -> bool:
    fracs = [Fraction(v) for v in values]
    return len(set(fracs)) == len(fracs)


def _num_str(value) -> str:
    """Plain distractor display: integer if exact, else a decimal string.
    The canonical answers on this topic never carry a decimal point, so any
    of these differing in VALUE is already enough to clear the grader."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return str(float(value))


# ---------------------------------------------------------------------------
# ucv1-energy-conservation-check
# ---------------------------------------------------------------------------

_DIRECT_TEMPLATES = (
    "A mass m = {m} kg on a spring of stiffness k = {k} N/m is measured at "
    "two states. State 1: displacement x1 = {x1} m, velocity v1 = {v1} m/s. "
    "State 2: displacement x2 = {x2} m, velocity v2 = {v2} m/s. Using "
    "E = (1/2)k x^2 + (1/2) m v^2, find the change in total mechanical "
    "energy from state 1 to state 2 (state 2 minus state 1), in J, to 1 "
    "decimal place.",
    "A spring-mass oscillator has mass m = {m} kg and spring constant "
    "k = {k} N/m. At state 1 its displacement is x1 = {x1} m and its "
    "velocity is v1 = {v1} m/s; at state 2 its displacement is x2 = {x2} m "
    "and its velocity is v2 = {v2} m/s. Compute state 2's total mechanical "
    "energy minus state 1's, in J, to 1 decimal place, using "
    "E = (1/2)k x^2 + (1/2) m v^2.",
    "For a mass m = {m} kg attached to a spring of stiffness k = {k} N/m, "
    "the total mechanical energy is E = (1/2)k x^2 + (1/2) m v^2. Given "
    "state 1 (x1 = {x1} m, v1 = {v1} m/s) and state 2 (x2 = {x2} m, "
    "v2 = {v2} m/s), report E at state 2 minus E at state 1, in J, to 1 "
    "decimal place.",
)


def _build_direct(rng, mk_hi, xv_hi, allow_negative):
    lo = -xv_hi if allow_negative else 0
    for _ in range(_RESAMPLE_ATTEMPTS):
        m = rng.randint(1, mk_hi)
        k = rng.randint(1, mk_hi)
        x1 = rng.randint(lo, xv_hi)
        v1 = rng.randint(lo, xv_hi)
        x2 = rng.randint(lo, xv_hi)
        v2 = rng.randint(lo, xv_hi)

        E1 = _energy(k, x1, m, v1)
        E2 = _energy(k, x2, m, v2)
        diff = E2 - E1

        # drops-the-half: forgets both factors of 1/2.
        drops_half = Fraction(k * x2 * x2 + m * v2 * v2 - (k * x1 * x1 + m * v1 * v1))
        # sign-flip: reports E1 - E2 instead of E2 - E1.
        sign_flip = -diff
        # swapped-roles: uses m where k belongs and vice versa, throughout.
        swapped_diff = _energy(m, x2, k, v2) - _energy(m, x1, k, v1)
        # single-state-only: reports just E2, forgetting to subtract E1.
        single_state = E2

        cands = [drops_half, sign_flip, swapped_diff, single_state]
        if _gap_ok([diff] + cands):
            break
    else:
        raise RuntimeError("p14f_energy_conservation_check: no collision-free direct draw found")

    # Independent check: recompute both states via sympy.Rational, a
    # separate arithmetic engine from the Fraction construction above.
    E1_sym = sympy.Rational(k, 2) * x1**2 + sympy.Rational(m, 2) * v1**2
    E2_sym = sympy.Rational(k, 2) * x2**2 + sympy.Rational(m, 2) * v2**2
    check_equivalent(sympy.Rational(E1.numerator, E1.denominator), E1_sym, "direct: state 1 energy mismatch")
    check_equivalent(sympy.Rational(E2.numerator, E2.denominator), E2_sym, "direct: state 2 energy mismatch")
    check_equivalent(
        sympy.Rational(diff.numerator, diff.denominator), E2_sym - E1_sym, "direct: energy difference mismatch"
    )
    check_distinct(
        diff,
        drops_half,
        sign_flip,
        swapped_diff,
        msg=f"direct: a distractor equals the true difference (m={m},k={k},x1={x1},v1={v1},x2={x2},v2={v2})",
    )

    template = rng.choice(_DIRECT_TEMPLATES)
    text = template.format(m=m, k=k, x1=x1, v1=v1, x2=x2, v2=v2)
    answer_text = _fmt1(diff)
    distractors = pick_distractors(answer_text, [_fmt1(c) for c in cands], n=3)
    check(len(distractors) >= 1, "direct tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(diff),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"E1 = (1/2)({k})({x1})^2 + (1/2)({m})({v1})^2 = {_fmt1(E1)} J.",
            f"E2 = (1/2)({k})({x2})^2 + (1/2)({m})({v2})^2 = {_fmt1(E2)} J.",
            f"ΔE = E2 - E1 = {answer_text} J.",
        ],
        distractors=distractors,
    )


_IMPLICIT_TEMPLATES = (
    "A mass m = {m} kg is attached to a spring of stiffness k = {k} N/m. "
    "{implicit_desc} The other state (state {other_no}) has displacement "
    "x = {xo} m and velocity v = {vo} m/s. Using E = (1/2)k x^2 + (1/2) m "
    "v^2, find the change in total mechanical energy from state 1 to state "
    "2 (state 2 minus state 1), in J, to 1 decimal place.",
    "A spring-mass system has mass m = {m} kg and stiffness k = {k} N/m. "
    "{implicit_desc} Its other measured state (state {other_no}) has "
    "displacement x = {xo} m and velocity v = {vo} m/s. Compute state 2's "
    "total mechanical energy minus state 1's, in J, to 1 decimal place, "
    "using E = (1/2)k x^2 + (1/2) m v^2.",
)


def _implicit_desc(state_no, branch, value):
    if branch == "turning_point":
        return (
            f"State {state_no} is at a turning point (momentarily at rest), with displacement x{state_no} = {value} m."
        )
    return (
        f"State {state_no} is passing through equilibrium (zero displacement), with velocity v{state_no} = {value} m/s."
    )


def _build_implicit(rng, mk_hi, xv_hi):
    for _ in range(_RESAMPLE_ATTEMPTS):
        m = rng.randint(1, mk_hi)
        k = rng.randint(1, mk_hi)
        implicit_state = rng.choice((1, 2))
        branch = rng.choice(("turning_point", "equilibrium"))
        imp_value = _draw_nonzero(rng, -xv_hi, xv_hi)
        xo = _draw_nonzero(rng, -xv_hi, xv_hi)
        vo = _draw_nonzero(rng, -xv_hi, xv_hi)

        if branch == "turning_point":
            E_imp = Fraction(k, 2) * imp_value * imp_value
            E_imp_wrong_branch = Fraction(m, 2) * imp_value * imp_value
            swapped_E_imp = Fraction(m, 2) * imp_value * imp_value
        else:
            E_imp = Fraction(m, 2) * imp_value * imp_value
            E_imp_wrong_branch = Fraction(k, 2) * imp_value * imp_value
            swapped_E_imp = Fraction(k, 2) * imp_value * imp_value

        E_other = _energy(k, xo, m, vo)
        swapped_E_other = _energy(m, xo, k, vo)

        if implicit_state == 1:
            E1, E2 = E_imp, E_other
            E1_wrong_branch, E2_wrong_branch = E_imp_wrong_branch, E_other
            swapped_E1, swapped_E2 = swapped_E_imp, swapped_E_other
        else:
            E1, E2 = E_other, E_imp
            E1_wrong_branch, E2_wrong_branch = E_other, E_imp_wrong_branch
            swapped_E1, swapped_E2 = swapped_E_other, swapped_E_imp

        diff = E2 - E1
        # wrong-branch: assumes the WRONG term vanishes at the implicit state.
        wrong_branch_diff = E2_wrong_branch - E1_wrong_branch
        # drops-the-half: forgets both factors of 1/2.
        drops_half = 2 * diff
        # sign-flip: reports E1 - E2 instead of E2 - E1.
        sign_flip = -diff
        # swapped-roles: uses m where k belongs and vice versa, throughout.
        swapped_diff = swapped_E2 - swapped_E1
        # single-state-only: reports just E2, forgetting to subtract E1.
        single_state = E2

        cands = [drops_half, sign_flip, swapped_diff, wrong_branch_diff, single_state]
        if _gap_ok([diff] + cands):
            break
    else:
        raise RuntimeError("p14f_energy_conservation_check: no collision-free implicit draw found")

    E_other_sym = sympy.Rational(k, 2) * xo**2 + sympy.Rational(m, 2) * vo**2
    if branch == "turning_point":
        E_imp_sym = sympy.Rational(k, 2) * imp_value**2
    else:
        E_imp_sym = sympy.Rational(m, 2) * imp_value**2
    expected_diff_sym = (E_other_sym - E_imp_sym) if implicit_state == 1 else (E_imp_sym - E_other_sym)
    check_equivalent(
        sympy.Rational(diff.numerator, diff.denominator), expected_diff_sym, "implicit: energy difference mismatch"
    )
    check_distinct(
        diff,
        drops_half,
        sign_flip,
        swapped_diff,
        wrong_branch_diff,
        msg=f"implicit: a distractor equals the true difference (m={m},k={k},state={implicit_state},branch={branch})",
    )

    other_no = 2 if implicit_state == 1 else 1
    desc = _implicit_desc(implicit_state, branch, imp_value)
    template = rng.choice(_IMPLICIT_TEMPLATES)
    text = template.format(m=m, k=k, implicit_desc=desc, other_no=other_no, xo=xo, vo=vo)
    answer_text = _fmt1(diff)
    distractors = pick_distractors(answer_text, [_fmt1(c) for c in cands], n=3)
    check(len(distractors) >= 1, "implicit tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(diff),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The implicit state's energy uses only its nonzero term: E = {_fmt1(E_imp)} J.",
            f"The other state's energy is E = {_fmt1(E_other)} J.",
            f"ΔE = E2 - E1 = {answer_text} J.",
        ],
        distractors=distractors,
    )


_INVERSE_TEMPLATES = (
    "A mass m = {m} kg on a spring of stiffness k = {k} N/m starts at "
    "state 1: displacement x1 = {x1} m, velocity v1 = {v1} m/s. At state 2 "
    "its displacement is x2 = {x2} m. Using E = (1/2)k x^2 + (1/2) m v^2, "
    "the total mechanical energy difference from state 1 to state 2 (state "
    "2 minus state 1) is {diff} J. Find the speed, not velocity, at state "
    "2, in m/s, to 1 decimal place.",
    "A spring-mass oscillator (mass m = {m} kg, stiffness k = {k} N/m) is "
    "at x1 = {x1} m, v1 = {v1} m/s in state 1. In state 2 its displacement "
    "is x2 = {x2} m, and the energy difference (state 2 minus state 1) is "
    "{diff} J. Using E = (1/2)k x^2 + (1/2) m v^2, find the speed, not "
    "velocity, at state 2, in m/s, to 1 decimal place.",
    "For a mass m = {m} kg attached to a spring of stiffness k = {k} N/m: "
    "state 1 has displacement x1 = {x1} m and velocity v1 = {v1} m/s; "
    "state 2 has displacement x2 = {x2} m. The energy change from state 1 "
    "to state 2, E2 minus E1, is {diff} J. Using E = (1/2)k x^2 + (1/2) m "
    "v^2, determine the speed, not velocity, at state 2, in m/s, to 1 "
    "decimal place.",
)


def _speed_from_sq(sq: Fraction) -> Fraction:
    """A wrong candidate speed from a wrong v^2 value -- never negative
    under the square root, since only the displayed distractor value
    matters here, not physical validity."""
    magnitude = math.sqrt(abs(float(sq)))
    return round_half_up(Fraction(magnitude).limit_denominator(10**9), 1)


def _build_inverse(rng, mk_hi, xv_hi):
    for _ in range(_RESAMPLE_ATTEMPTS):
        m = rng.randint(1, mk_hi)
        k = rng.randint(1, mk_hi)
        x1 = _draw_nonzero(rng, -xv_hi, xv_hi)
        v1 = _draw_nonzero(rng, -xv_hi, xv_hi)
        x2 = _draw_nonzero(rng, -xv_hi, xv_hi)
        v2 = rng.randint(1, xv_hi)  # the speed itself; positive by construction

        E1 = _energy(k, x1, m, v1)
        E2 = _energy(k, x2, m, v2)
        diff = E2 - E1  # always exact on the 0.5 grid, hence exact at 1 d.p.

        E2_target = E1 + diff
        base = E2_target - Fraction(k, 2) * x2 * x2  # = (1/2) m v2^2, exact

        # drops-the-half: forgets the factor of 2 when isolating v2^2.
        drops_half_sq = base / m
        # sign-flip: uses E1 - diff as the state-2 energy target.
        e2_target_signflip = E1 - diff
        sign_flip_sq = (e2_target_signflip - Fraction(k, 2) * x2 * x2) / m
        # swapped-roles: swaps which coefficient (k vs m) multiplies which term.
        swapped_sq = (E2_target - Fraction(m, 2) * x2 * x2) / k
        # single-state-only: forgets to subtract the potential-energy term.
        single_state_sq = Fraction(2, m) * E2_target

        drops_half_speed = _speed_from_sq(drops_half_sq)
        sign_flip_speed = _speed_from_sq(sign_flip_sq)
        swapped_speed = _speed_from_sq(swapped_sq)
        single_state_speed = _speed_from_sq(single_state_sq)

        cands = [drops_half_speed, sign_flip_speed, swapped_speed, single_state_speed]
        if _gap_ok([Fraction(v2)] + cands):
            break
    else:
        raise RuntimeError("p14f_energy_conservation_check: no collision-free inverse draw found")

    # Independent re-solve: from the same public inputs alone (m, k, x1, v1,
    # x2, the stated difference), re-derive v2^2 via a fresh rearrangement,
    # not the construct-first value.
    v2_sq_check = Fraction(2, m) * (E1 + diff - Fraction(k, 2) * x2 * x2)
    check(v2_sq_check >= 0, "inverse: re-solved v2^2 must be non-negative")
    check_equivalent(
        sympy.Rational(v2_sq_check.numerator, v2_sq_check.denominator),
        sympy.Integer(v2 * v2),
        "inverse: re-solved v2^2 disagrees with the constructed speed",
    )
    check_distinct(
        Fraction(v2),
        drops_half_speed,
        sign_flip_speed,
        swapped_speed,
        msg=f"inverse: a distractor equals the true speed (m={m},k={k},x1={x1},v1={v1},x2={x2},v2={v2})",
    )

    template = rng.choice(_INVERSE_TEMPLATES)
    diff_text = _fmt1(diff)
    text = template.format(m=m, k=k, x1=x1, v1=v1, x2=x2, diff=diff_text)
    answer_text = _fmt1(v2)
    distractors = pick_distractors(answer_text, [_fmt1(c) for c in cands], n=3)
    check(len(distractors) >= 1, "inverse tier expects at least 1 distinct distractor")

    # Bracket a negative difference so "+" stays visible without a bare sign
    # straight after it ("+ -2" is flagged as signed_after_operator) -- the
    # same discipline as core.paren_negative, applied to the pre-formatted
    # 1-d.p. string rather than the raw Fraction.
    diff_disp = f"({diff_text})" if diff < 0 else diff_text

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(Fraction(v2)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"E1 = (1/2)({k})({x1})^2 + (1/2)({m})({v1})^2 = {_fmt1(E1)} J.",
            f"E2 = E1 + ΔE = {_fmt1(E1)} + {diff_disp} = {_fmt1(E2_target)} J.",
            f"(1/2)({m})v2^2 = E2 - (1/2)({k})({x2})^2 = {_fmt1(base)} J, so v2 = {answer_text} m/s.",
        ],
        distractors=distractors,
    )


@generator(
    "p14f_energy_conservation_check",
    level=16,
    presets={
        "easy": dict(variant="direct", mk_hi=4, xv_hi=4, allow_negative=False),
        "medium": dict(variant="direct", mk_hi=8, xv_hi=6, allow_negative=True),
        "hard": dict(variant="implicit", mk_hi=8, xv_hi=6),
        "exam": dict(variant="inverse", mk_hi=8, xv_hi=14),
    },
)
def p14f_energy_conservation_check(rng, variant="direct", mk_hi=8, xv_hi=6, allow_negative=True):
    """Numeric difference E(state 2) - E(state 1) for E = (1/2)k x^2 +
    (1/2) m v^2 on a horizontal spring-mass system. ``variant="direct"``
    (easy/medium) supplies both states fully. ``variant="implicit"`` (hard)
    supplies one state via a structural description (turning point or
    equilibrium crossing) instead of both numbers. ``variant="inverse"``
    (exam) gives the difference and asks for the missing speed at state 2."""
    if variant == "direct":
        return _build_direct(rng, mk_hi, xv_hi, allow_negative)
    if variant == "implicit":
        return _build_implicit(rng, mk_hi, xv_hi)
    if variant == "inverse":
        return _build_inverse(rng, mk_hi, xv_hi)
    raise ValueError(f"unknown variant {variant!r}")


# ---------------------------------------------------------------------------
# ucv1-wave-parameter-relation
# ---------------------------------------------------------------------------

_SOLVE_V_TEMPLATES = (
    "A travelling wave has frequency f = {f} Hz and wavelength λ = {lam} m. Find its speed v, in m/s.",
    "A wave oscillates at f = {f} Hz with wavelength λ = {lam} m. Determine the wave's speed v, in m/s.",
    "A wave has frequency {f} Hz and wavelength {lam} m. Calculate how fast the wave travels, in m/s.",
)


def _build_solve_v(rng, hi):
    for _ in range(_RESAMPLE_ATTEMPTS):
        f = rng.randint(2, hi)  # f != 1 keeps the inverted-relation distractor distinct
        lam = rng.randint(1, hi)
        v = f * lam
        # inverted-relation: uses lambda/f instead of f*lambda.
        inverted = Fraction(lam, f)
        cands = [inverted, Fraction(f), Fraction(lam), Fraction(v + 1)]
        if _all_distinct([Fraction(v)] + cands):
            break
    else:
        raise RuntimeError("p14f_wave_parameter_relation: no collision-free solve_v draw found")

    v_sym, f_sym, lam_sym = sympy.symbols("v f lam")
    solved = sympy.solve(sympy.Eq(v_sym, f_sym * lam_sym), v_sym)
    v_check = solved[0].subs({f_sym: f, lam_sym: lam})
    check_equivalent(sympy.Integer(v), v_check, "solve_v: sympy re-solve disagrees")
    check_distinct(Fraction(v), inverted, msg=f"solve_v: inverted-relation distractor equals v (f={f}, lam={lam})")

    template = rng.choice(_SOLVE_V_TEMPLATES)
    text = template.format(f=f, lam=lam)
    answer_text = str(v)
    # The key is a whole number, so every option is: the inverted relation
    # (a fraction like 4/7) would give the key away by its decimal point and
    # is replaced by the whole-number slip f + lambda ("adds instead of multiplies").
    whole = [c for c in cands + [Fraction(f + lam)] if c.denominator == 1]
    distractors = pick_distractors(answer_text, [_num_str(c) for c in whole], n=3)
    check(len(distractors) >= 1, "solve_v tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=v,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"v = f × λ = {f} × {lam} = {v} m/s."],
        distractors=distractors,
    )


_SOLVE_V_PERIOD_TEMPLATES = (
    "A wave completes one full cycle every T = {T} s and has wavelength λ = {lam} m. Find its speed v, in m/s.",
    "A wave's period is T = {T} s and its wavelength is λ = {lam} m. Determine the wave's speed, in m/s.",
    "Every T = {T} seconds, a wave completes one cycle; its wavelength is {lam} m. Calculate the wave's speed, in m/s.",
)


def _build_solve_v_period(rng, hi):
    for _ in range(_RESAMPLE_ATTEMPTS):
        v = rng.randint(1, hi)
        T = rng.randint(2, hi)  # T != 1 keeps forgot-reciprocal distinct
        lam = v * T
        # forgot-reciprocal: uses T directly as if it were f.
        forgot_reciprocal = Fraction(lam * T)
        inverted = Fraction(T, lam)
        cands = [forgot_reciprocal, inverted, Fraction(T), Fraction(lam)]
        if _all_distinct([Fraction(v)] + cands):
            break
    else:
        raise RuntimeError("p14f_wave_parameter_relation: no collision-free solve_v_period draw found")

    v_sym, T_sym, lam_sym = sympy.symbols("v T lam")
    solved = sympy.solve(sympy.Eq(v_sym * T_sym, lam_sym), v_sym)
    v_check = solved[0].subs({T_sym: T, lam_sym: lam})
    check_equivalent(sympy.Integer(v), v_check, "solve_v_period: sympy re-solve disagrees")
    check_distinct(
        Fraction(v), forgot_reciprocal, msg=f"solve_v_period: forgot-reciprocal distractor equals v (T={T}, lam={lam})"
    )

    template = rng.choice(_SOLVE_V_PERIOD_TEMPLATES)
    text = template.format(T=T, lam=lam)
    answer_text = str(v)
    # Whole-number options only, like the key (the inverted relation T / lambda
    # is a fraction); lambda + T ("adds instead of dividing") tops the list up.
    whole = [c for c in cands + [Fraction(lam + T)] if c.denominator == 1]
    distractors = pick_distractors(answer_text, [_num_str(c) for c in whole], n=3)
    check(len(distractors) >= 1, "solve_v_period tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=v,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"f = 1 / T = 1 / {T} Hz.", f"v = λ / T = {lam} / {T} = {v} m/s."],
        distractors=distractors,
    )


_SOLVE_LAMBDA_UNIT_TEMPLATES = (
    "A wave travels at v = {v_kmh} km/h and completes one cycle every "
    "T = {T} s. Find its wavelength λ, in metres, to 2 decimal places.",
    "A wave's speed is v = {v_kmh} km/h and its period is T = {T} s. "
    "Determine the wavelength, in metres, to 2 decimal places.",
    "Travelling at {v_kmh} km/h, a wave completes one cycle every {T} "
    "seconds. Calculate its wavelength, in metres, to 2 decimal places.",
)

_KMH_FACTOR = Fraction(18, 5)  # v(km/h) = v(m/s) * 18/5, i.e. * 3.6
_CONV_FACTORS = {"m/s": Fraction(1), "km/h": Fraction(5, 18)}
_DP2_TOL = stated_precision_tolerance(2)


def _disp2(value) -> str:
    """Display a value at exactly 2 decimal places, rounding half-up --
    unlike `stated_precision_string`, never raises when `value` is not
    exactly on the 2-d.p. grid (the wrong-factor distractor here sits on a
    1/50 = 0.02 grid but `lam_inverted` need not terminate at all), so this
    is scoped to a DISTRACTOR-and-canonical DISPLAY where every value is
    rounded for presentation, not a graded value constructed to already sit
    on the grid. Formatting the always-exact-integer canonical `lam` the
    same way keeps every option's SHAPE identical (a decimal point on every
    option), so the answer is never the only "clean" integer-looking choice
    (test_answer_is_not_the_only_option_of_its_shape)."""
    return f"{float(round_half_up(Fraction(value), 2)):.2f}"


def _build_solve_lambda_unit(rng, hi):
    for _ in range(_RESAMPLE_ATTEMPTS):
        v_mps = rng.randint(1, hi)
        T = rng.randint(1, hi)
        v_kmh = _KMH_FACTOR * v_mps
        lam = v_mps * T
        # unit-conversion-error: divides by 60 (a time-unit factor) instead of 3.6.
        wrong_factor_mps = Fraction(v_kmh, 60)
        lam_wrong_factor = wrong_factor_mps * T
        # inverted-relation (no conversion at all): uses the km/h number directly.
        lam_no_conversion = v_kmh * T
        # divides by T instead of multiplying.
        lam_inverted = Fraction(v_mps, T)
        cands = [lam_wrong_factor, lam_no_conversion, lam_inverted]
        # 2-d.p. DISPLAYED gap, not bare value inequality: every option is
        # shown at 2 decimal places below (so the correct answer is not the
        # only integer-shaped option), so the grader-clearing margin must be
        # checked on that same grid, and on the ROUNDED display values (two
        # candidates rounding to the same 2-d.p. string would otherwise still
        # pass a check on their un-rounded Fractions).
        displayed = [round_half_up(Fraction(v), 2) for v in ([Fraction(lam)] + cands)]
        if _gap_ok(displayed, tol=_DP2_TOL) and _all_distinct(displayed):
            break
    else:
        raise RuntimeError("p14f_wave_parameter_relation: no collision-free solve_lambda_unit draw found")

    check_conversion(v_kmh, "km/h", "m/s", Fraction(v_mps), _CONV_FACTORS, label="solve_lambda_unit: km/h to m/s")
    lam_sym, v_sym, T_sym = sympy.symbols("lam v T")
    solved = sympy.solve(sympy.Eq(lam_sym, v_sym * T_sym), lam_sym)
    lam_check = solved[0].subs({v_sym: v_mps, T_sym: T})
    check_equivalent(sympy.Integer(lam), lam_check, "solve_lambda_unit: sympy re-solve disagrees")
    check_distinct(
        Fraction(lam),
        lam_wrong_factor,
        lam_no_conversion,
        lam_inverted,
        msg=f"solve_lambda_unit: a distractor equals lambda (v_mps={v_mps}, T={T})",
    )

    template = rng.choice(_SOLVE_LAMBDA_UNIT_TEMPLATES)
    v_kmh_text = _fmt1(v_kmh)
    text = template.format(v_kmh=v_kmh_text, T=T)
    # Every option is shown at 2 d.p. -- never a bare integer -- so the answer
    # is not identifiable by being the only "clean" option on this tier
    # (lam is always an exact integer by construction; the distractors are
    # not, and _num_str's mixed integer/long-float shapes made the answer
    # stand out).
    answer_text = _disp2(lam)
    distractors = pick_distractors(answer_text, [_disp2(c) for c in cands], n=3)
    check(len(distractors) >= 1, "solve_lambda_unit tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=lam,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Convert v = {v_kmh_text} km/h to m/s: v = {v_kmh_text} / 3.6 = {v_mps} m/s.",
            f"λ = v × T = {v_mps} × {T} = {answer_text} m.",
        ],
        distractors=distractors,
    )


_MEDIUM_CHANGE_TEMPLATES = (
    "A wave travels through medium 1 with speed v1 = {v1} m/s and "
    "wavelength λ1 = {lam1} m. It then passes into medium 2, where its "
    "frequency is unchanged but its speed becomes v2 = {v2} m/s. Find its "
    "new wavelength λ2, in m, to 2 decimal places.",
    "A wave's speed is v1 = {v1} m/s and its wavelength is λ1 = {lam1} m "
    "in medium 1. Its frequency is unchanged crossing into medium 2, "
    "where its speed is v2 = {v2} m/s. Determine the wavelength in "
    "medium 2, in m, to 2 decimal places.",
    "In medium 1 a wave has speed v1 = {v1} m/s and wavelength λ1 = "
    "{lam1} m. The same wave enters medium 2 with unchanged frequency and "
    "speed v2 = {v2} m/s. Calculate its wavelength there, in m, to 2 "
    "decimal places.",
)


def _build_medium_change(rng, hi):
    """exam: a wave crosses into a new medium -- frequency is the invariant
    (never printed as a number), speed changes to a stated v2. The learner
    must derive the frequency from the FIRST medium's v1, lambda1 pair, then
    apply it to the SECOND medium's stated v2 to find lambda2 -- two chained
    relations, never a single v = f*lambda substitution the way the
    neighbour skill (or this skill's own easy tier) is, and never a printed
    candidate answer to pick between."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        f = rng.randint(2, hi)
        lam1 = rng.randint(1, hi)
        lam2 = rng.randint(1, hi)
        if lam2 == lam1:
            continue
        v1 = f * lam1
        v2 = f * lam2
        # assumed-wavelength-unchanged: forgets that only FREQUENCY is the
        # invariant across the boundary, reports lambda1 as if it carried
        # over unchanged.
        assumed_unchanged = Fraction(lam1)
        # confused-v2-for-frequency: skips re-deriving f and divides v2 by
        # lambda1 directly.
        confused_v2 = Fraction(v2, lam1)
        # inverted-final-step: computes f / v2 (a reciprocal) instead of
        # v2 / f.
        inverted_final = Fraction(f, v2)
        cands = [assumed_unchanged, confused_v2, inverted_final]
        displayed = [round_half_up(Fraction(v), 2) for v in ([Fraction(lam2)] + cands)]
        if _gap_ok(displayed, tol=_DP2_TOL) and _all_distinct(displayed):
            break
    else:
        raise RuntimeError("p14f_wave_parameter_relation: no collision-free medium_change draw found")

    # Independent check: a FRESH chained symbolic solve (f from state 1,
    # then lambda2 from state 2), never reusing the construct-first f/lam2
    # the draw above used.
    f_sym, lam1_sym, v1_sym, v2_sym, lam2_sym = sympy.symbols("f lam1 v1 v2 lam2")
    f_solved = sympy.solve(sympy.Eq(v1_sym, f_sym * lam1_sym), f_sym)[0].subs({v1_sym: v1, lam1_sym: lam1})
    check_equivalent(sympy.Integer(f), f_solved, "medium_change: re-solved f disagrees")
    lam2_solved = sympy.solve(sympy.Eq(v2_sym, f_sym * lam2_sym), lam2_sym)[0].subs({f_sym: f_solved, v2_sym: v2})
    check_equivalent(sympy.Integer(lam2), lam2_solved, "medium_change: re-solved lambda2 disagrees")
    check_distinct(
        Fraction(lam2),
        assumed_unchanged,
        confused_v2,
        inverted_final,
        msg=f"medium_change: a distractor equals lambda2 (f={f}, lam1={lam1}, lam2={lam2})",
    )

    template = rng.choice(_MEDIUM_CHANGE_TEMPLATES)
    text = template.format(v1=v1, lam1=lam1, v2=v2)
    answer_text = _disp2(lam2)
    distractors = pick_distractors(answer_text, [_disp2(c) for c in cands], n=3)
    check(len(distractors) >= 1, "medium_change tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=lam2,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"f = v1 / λ1 = {v1} / {lam1} = {f} Hz (frequency is unchanged across the boundary).",
            f"λ2 = v2 / f = {v2} / {f} = {answer_text} m.",
        ],
        distractors=distractors,
    )


@generator(
    "p14f_wave_parameter_relation",
    level=16,
    presets={
        "easy": dict(variant="solve_v", hi=12),
        "medium": dict(variant="solve_v_period", hi=20),
        "hard": dict(variant="solve_lambda_unit", hi=9),
        "exam": dict(variant="medium_change", hi=20),
    },
)
def p14f_wave_parameter_relation(rng, variant="solve_v", hi=12):
    """Solve for one of {wavelength, frequency, wave speed} from the other two -- always wave SPEED here (the
    unknown the shipping ``wave-frequency-from-speed-wavelength`` neighbour never asks for).
    ``variant="solve_v"`` (easy): v from f and lambda, formula not printed. ``variant="solve_v_period"``
    (medium): v from period T and lambda (a reciprocal step the neighbour never needs).
    ``variant="solve_lambda_unit"`` (hard): lambda from v (given in km/h) and T -- a unit conversion
    compounded with the period relation. ``variant="medium_change"`` (exam): the wave crosses into a new
    medium with the SAME frequency (the invariant, never printed as a number) but a stated new speed v2; the
    learner must derive f from the first medium's (v1, lambda1) pair, then apply it to the second medium's v2
    to find lambda2 -- two chained relations, never a single-substitution kernel and never a printed candidate
    answer."""
    if variant == "solve_v":
        return _build_solve_v(rng, hi)
    if variant == "solve_v_period":
        return _build_solve_v_period(rng, hi)
    if variant == "solve_lambda_unit":
        return _build_solve_lambda_unit(rng, hi)
    if variant == "medium_change":
        return _build_medium_change(rng, hi)
    raise ValueError(f"unknown variant {variant!r}")
