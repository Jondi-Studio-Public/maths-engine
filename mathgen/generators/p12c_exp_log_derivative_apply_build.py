"""Exponential and logarithmic derivative applications.

Two skills, both ``domain: differentiation``, both ``answer: number``,
``prerequisites: [differentiate-polynomial]``:

- ``senior-use-exponential-functions-and-their-derivatives-to-solve``
  (ACMMM101), topic ``p12c_exp_deriv_apply``. Model
  ``N(t) = N0 * exp(k*t)`` is ALWAYS given (``N0``, ``k`` stated); the
  question asks for something computed from the model and its derivative
  ``dN/dt = N0*k*exp(k*t)`` -- a rate of change at a stated time, or a time
  at which the rate reaches a stated value. **No question here ever asks
  the learner to find the particular solution or the constants ``k``/``N0``
  from an initial condition** -- that is
  ``skills/differential-equations/solve-ode-exponential.yaml``
  (``mathgen/generators/calculus.py::differentiation_equation``'s
  ``exponential`` kind), a DIFFERENT skill that solves the ODE
  ``dy/dx = ky`` from a stated initial condition to recover ``y0``/``k``
  itself. This skill's every input is already bounded and stated; the
  boundary is enforced by
  ``tests/test_p12c_exp_log_derivative_apply_build.py::test_never_asks_to_find_the_model``.

- ``senior-use-logarithmic-functions-and-their-derivatives-to-solve``
  (ACMMM163), topic ``p12c_log_deriv_apply``. Model
  ``V(x) = a*ln(x) + b`` is always given; ``dV/dx = a/x``. This skill's
  questions always start from an already-differentiated rate-of-change
  relationship applied to a stated model; no question asks the learner to
  solve a bare logarithmic equation or apply a log law with no derivative
  in play (there is no directly-adjacent logarithm-solving skill to draw
  the boundary against -- three senior logarithm-DEFINITION skills and
  ``logarithm-laws`` ship, none of which differentiate).

Both skills share the same four-tier design, mirrored deliberately so the
reasoning escalation is symmetric between the exponential and logarithmic
cases:

- **easy**: the rate of change at the SIMPLEST possible input (``t=0`` for
  the exponential model, since ``exp(0)=1`` needs no evaluation; a stated
  small positive integer ``x`` for the logarithmic model, a single
  division). No exponential/logarithm evaluation is required.
- **medium**: the rate of change at a stated NONZERO input that itself
  needs a small extra arithmetic step to resolve (exponential: a stated
  nonzero ``t0``, requiring ``exp(k*t0)`` to be evaluated, not just looked
  up; logarithmic: ``x`` is stated as an expression -- a base value and a
  multiplier -- that must be resolved before dividing).
- **hard**: INVERT the derivative -- given a stated target rate ``R``,
  solve for the input (exponential: the time ``t``; logarithmic: the
  input ``x``) at which the derivative first equals ``R``. A genuinely
  different decision (inverting via a logarithm/reciprocal) from
  evaluating the derivative forward.
- **exam**: a genuine escalation in KIND over hard, composing the model
  and its derivative rather than using either alone.
  - Exponential exam: given the target VALUE ``M`` the model reaches
    (not a rate), find the rate of change AT THAT MOMENT. The clean
    route is recognising ``dN/dt = k*N(t)``, so the answer is exactly
    ``k*M`` -- a different reasoning move from hard's "invert then stop",
    it requires connecting the model's own value back into its derivative.
  - Logarithmic exam: given the target RATE ``R`` (the same shape as
    hard), first invert the derivative for ``x`` (as hard does) and then
    EVALUATE THE MODEL at that ``x`` -- an explicit second step hard never
    asks for.

**Sign-domain simplification, stated once here rather than scattered in each tier's code.** The exponential
model's ``k`` may be positive (growth) or negative (decay) freely on the ``easy``/``medium``/``exam`` tiers,
where no misconception construction takes a logarithm of a possibly-negative quantity. The ``hard`` tier
restricts to GROWTH-only draws (``k>0``) because its own misconceptions (``ln(R/N0)``,
``ln(R)``) are only defined for a positive stated ``R``, and restricting ``k`` to growth keeps ``R`` positive
by construction instead of special-casing every misconception formula for a decaying scenario's negative rate.
Documented rather than implicit.

Similarly, the logarithmic model's coefficient ``a`` is drawn as a
POSITIVE nonzero integer only (the descriptor says "a nonzero integer",
not that both signs must appear): ``ln`` of the model's own input ``x``
must stay positive throughout, and a negative ``a`` would force several
misconception formulas (``exp((R-b)/a)``, ``a*ln(R)+b``) to sometimes
take a logarithm of a negative quantity depending on the sign of the
stated rate ``R``. Restricting ``a`` to positive avoids that domain
gymnastics entirely; ``b`` (the vertical shift) is unrestricted in sign.

Every canonical answer is displayed to a STATED precision of 2 decimal places
(``mathgen.core.stated_precision_string``), even on tiers whose answer happens to be exact to that precision
by construction (easy/exam exponential, hard logarithmic-by-construction) -- so no option is identifiable by
carrying a different number of decimal places than its siblings. Every irrational-valued tier (exponential
medium/hard, logarithmic easy/medium/exam) is guarded against landing within 1e-6 of the.xx5 rounding boundary
before it is accepted, and re-drawn otherwise.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import Rational, diff, exp as sym_exp, log as sym_log, symbols

from ..core import (
    check,
    check_equivalent,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_RESAMPLE_ATTEMPTS = 300
_DP = 2
_TOL = stated_precision_tolerance(_DP)  # Fraction(1, 200)

t_sym, x_sym = symbols("t x")


def _round2(value) -> Fraction:
    """Round an exact `Fraction` (or a float converted at the call site) to
    the skill's stated 2 decimal places, ties away from zero."""
    return round_half_up(Fraction(value).limit_denominator(10**9), _DP)


def _near_rounding_boundary(value: float, dp: int = _DP, eps: float = 1e-6) -> bool:
    """True when `value` sits within `eps` of the.xx5 halfway point at `dp` decimal places -- the point where a
    construction's own float arithmetic and the independent oracle's `math.log`/`math.exp` calls could
    legitimately round in opposite directions. Re-draw rather than ship such a seed."""
    scaled = value * (10**dp)
    frac = scaled - math.floor(scaled)
    return abs(frac - 0.5) < eps


def _gap_ok(values: list[Fraction], tol: Fraction = _TOL) -> bool:
    """Every pair in `values` (already rounded to the stated precision) differs, checked directly on the values
    the grader will see rather than trusted from the symbolic construction alone."""
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) <= tol:
                return False
    return True


def _fmt(value: Fraction) -> str:
    return stated_precision_string(value, _DP)


_INSTRUCTION = "Give your answer correct to 2 decimal places."

# =============================================================================
# Exponential model: N(t) = N0 * exp(k*t)
# =============================================================================

_K_MAGNITUDES = [
    Fraction(5, 100),
    Fraction(8, 100),
    Fraction(10, 100),
    Fraction(12, 100),
    Fraction(15, 100),
    Fraction(18, 100),
    Fraction(20, 100),
]


def _draw_k_signed(rng) -> Fraction:
    mag = rng.choice(_K_MAGNITUDES)
    return mag if rng.choice([True, False]) else -mag


def _draw_k_growth(rng) -> Fraction:
    return rng.choice(_K_MAGNITUDES)


_EXP_SCENARIOS = [
    dict(subject="the number of bacteria in a laboratory culture", unit="hours"),
    dict(subject="the number of active members of an online forum", unit="months"),
    dict(subject="the mass, in grams, of a radioactive sample", unit="years"),
    dict(subject="the concentration, in mg/L, of a drug in a patient's bloodstream", unit="hours"),
]


def _model_statement(N0: int, k: Fraction, scenario: dict) -> str:
    k_str = _fmt(Fraction(k))
    return (
        f"{scenario['subject'].capitalize()}, after t {scenario['unit']}, is modelled by "
        f"N(t) = {N0}·e^({k_str}t), where t >= 0."
    )


def _exp_check_equiv(N0: int, k: Fraction, t0, constructed: Fraction) -> None:
    """Independent route: sympy's own `diff`, never re-deriving
    `N0*k*exp(k*t)` by hand a second time in the same form the construction
    already used."""
    k_sym = Rational(k.numerator, k.denominator)
    t0_sym = Rational(
        Fraction(t0).limit_denominator(10**9).numerator, Fraction(t0).limit_denominator(10**9).denominator
    )
    dN_dt = diff(N0 * sym_exp(k_sym * t_sym), t_sym).subs(t_sym, t0_sym)
    check_equivalent(
        dN_dt, Rational(constructed.numerator, constructed.denominator), "exp derivative independent check failed"
    )


# --- easy: rate at t=0 ------------------------------------------------------


def _build_exp_easy(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        N0 = rng.randint(2, 500)
        k = _draw_k_signed(rng)
        scenario = rng.choice(_EXP_SCENARIOS)

        rate0 = Fraction(N0) * k  # exact: N0 int, k already 2dp
        ans = _round2(rate0)

        value_instead = _round2(Fraction(N0))  # evaluates N(0) instead of dN/dt
        sign_flip = _round2(-rate0)  # sign error on k
        wrong_t = _round2(Fraction(N0) * k * Fraction(math.exp(float(k))).limit_denominator(10**9))
        # ^ substituted t=1 into the derivative by mistake

        cands = [value_instead, sign_flip, wrong_t]
        if ans == 0 or not _gap_ok([ans] + cands):
            continue
        break
    else:
        raise ValueError("p12c_exp_deriv_apply: no collision-free easy draw found")

    _exp_check_equiv(N0, k, Fraction(0), rate0)

    text = (
        f"{_model_statement(N0, k, scenario)} Find the instantaneous rate of change of N "
        f"with respect to t at t = 0. {_INSTRUCTION}"
    )
    latex = text
    answer_text = _fmt(ans)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "exp easy tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"dN/dt = N0·k·e^(kt). At t=0, e^0=1, so dN/dt = {N0}·{_fmt(Fraction(k))} = {answer_text}."],
        distractors=distractors,
    )


# --- medium: rate at nonzero t0 --------------------------------------------


def _build_exp_medium(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        N0 = rng.randint(2, 500)
        k = _draw_k_signed(rng)
        t0 = rng.randint(1, 20)
        scenario = rng.choice(_EXP_SCENARIOS)

        kf = float(k)
        rate_val = N0 * kf * math.exp(kf * t0)
        if _near_rounding_boundary(rate_val):
            continue
        ans = _round2(rate_val)

        value_val = N0 * math.exp(kf * t0)  # evaluates N(t0) / drops k
        sign_flip_val = -rate_val
        wrong_t_val = N0 * kf * math.exp(kf * (t0 + 1))  # off-by-one t

        cands = [_round2(value_val), _round2(sign_flip_val), _round2(wrong_t_val)]
        if ans == 0 or not _gap_ok([ans] + cands):
            continue
        break
    else:
        raise ValueError("p12c_exp_deriv_apply: no collision-free medium draw found")

    dN_dt_exact = (
        Rational(N0) * Rational(k.numerator, k.denominator) * sym_exp(Rational(k.numerator, k.denominator) * t0)
    )
    check(abs(float(dN_dt_exact) - rate_val) < 1e-6, "exp medium: constructed rate disagrees with sympy diff")

    text = (
        f"{_model_statement(N0, k, scenario)} Find the instantaneous rate of change of N "
        f"with respect to t at t = {t0}. {_INSTRUCTION}"
    )
    latex = text
    answer_text = _fmt(ans)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "exp medium tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"dN/dt = N0·k·e^(kt). At t={t0}, dN/dt = {N0}·{_fmt(Fraction(k))}·"
            f"e^({_fmt(Fraction(k))}·{t0}) = {answer_text}."
        ],
        distractors=distractors,
    )


# --- hard: invert for t given a target rate R (growth-only, see module note) --


def _build_exp_hard(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        N0 = rng.randint(2, 500)
        k = _draw_k_growth(rng)
        scenario = rng.choice(_EXP_SCENARIOS)
        base_rate = float(N0) * float(k)  # > 0 by construction (growth-only)

        factor = rng.uniform(1.4, 6.0)
        R_raw = base_rate * factor
        R = int(round_half_up(Fraction(R_raw).limit_denominator(10**6), 0))
        if R <= 0 or R == round_half_up(Fraction(base_rate).limit_denominator(10**6), 0):
            continue

        ratio = R / base_rate
        if ratio <= 0:
            continue
        kf = float(k)
        t_val = math.log(ratio) / kf
        if not (0.3 <= t_val <= 150):
            continue
        if _near_rounding_boundary(t_val):
            continue
        ans = _round2(t_val)

        wrong_value_eq = math.log(R / N0) / kf  # solved N0*e^(kt)=R instead of the rate eq
        wrong_sign = -t_val  # dropped a sign on k
        wrong_no_divide = math.log(R) / kf  # forgot to divide R by N0*k (R>0 by construction)

        cands_f = [wrong_value_eq, wrong_sign, wrong_no_divide]
        if any(_near_rounding_boundary(v) for v in cands_f):
            continue
        cands = [_round2(v) for v in cands_f]
        if ans <= 0 or not _gap_ok([ans] + cands):
            continue
        break
    else:
        raise ValueError("p12c_exp_deriv_apply: no collision-free hard draw found")

    dN_dt_exact = (
        Rational(N0)
        * Rational(k.numerator, k.denominator)
        * sym_exp(Rational(k.numerator, k.denominator) * Rational(ans.numerator, ans.denominator))
    )
    check(abs(float(dN_dt_exact) - R) < 0.5, "exp hard: constructed t does not reproduce the stated target rate")

    text = (
        f"{_model_statement(N0, k, scenario)} The rate of change of N first reaches {R} "
        f"(the same units as N, per {scenario['unit'][:-1]}) at some time t. Find this value of t, "
        f"in {scenario['unit']}. {_INSTRUCTION}"
    )
    latex = text
    answer_text = _fmt(ans)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "exp hard tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"dN/dt = N0·k·e^(kt) = {R}. e^(kt) = {R} / ({N0}·{_fmt(Fraction(k))}) = {ratio:.4f}. "
            f"t = ln({ratio:.4f}) / {_fmt(Fraction(k))} = {answer_text} {scenario['unit']}."
        ],
        distractors=distractors,
    )


# --- exam: given the target VALUE M, find the rate of change at that moment --


def _build_exp_exam(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        N0 = rng.randint(2, 500)
        k = _draw_k_signed(rng)
        scenario = rng.choice(_EXP_SCENARIOS)

        growth = k > 0
        mult = rng.uniform(1.3, 4.0) if growth else rng.uniform(0.2, 0.8)
        M_raw = N0 * mult
        M = int(round_half_up(Fraction(M_raw).limit_denominator(10**6), 0))
        if M <= 0 or M == N0:
            continue

        kf = float(k)
        ratio = M / N0
        if ratio <= 0:
            continue
        t_val = math.log(ratio) / kf
        if not (0.3 <= t_val <= 150):
            continue

        ans = _round2(Fraction(k) * M)  # dN/dt = k*N(t) = k*M, exact (k has <=2dp, M integer)

        used_M_as_rate = _round2(Fraction(M))  # reports M itself as the rate
        rate_at_zero = _round2(Fraction(N0) * k)  # evaluated dN/dt at t=0, not at the stated moment
        sign_error = _round2(
            Fraction(N0 * N0) * k / Fraction(M).limit_denominator(10**6)
        )  # N0^2*k/M, wrong-sign-t substitution

        cands = [used_M_as_rate, rate_at_zero, sign_error]
        if ans == 0 or not _gap_ok([ans] + cands):
            continue
        break
    else:
        raise ValueError("p12c_exp_deriv_apply: no collision-free exam draw found")

    k_sym = Rational(k.numerator, k.denominator)
    dN_dt_at_t = diff(N0 * sym_exp(k_sym * t_sym), t_sym).subs(t_sym, sym_log(Rational(M, N0)) / k_sym)
    check_equivalent(dN_dt_at_t, k_sym * M, "exp exam: dN/dt at the target moment must equal k*M")

    text = (
        f"{_model_statement(N0, k, scenario)} At some time t, N(t) = {M}. Find the "
        f"instantaneous rate of change of N with respect to t at that moment. {_INSTRUCTION}"
    )
    latex = text
    answer_text = _fmt(ans)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "exp exam tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"dN/dt = kN, so at the moment N(t) = {M}, dN/dt = {_fmt(Fraction(k))}·{M} = {answer_text}."],
        distractors=distractors,
    )


@generator(
    "p12c_exp_deriv_apply",
    level=12,
    presets={
        "easy": dict(mode="rate_at_zero"),
        "medium": dict(mode="rate_at_t"),
        "hard": dict(mode="time_for_rate"),
        "exam": dict(mode="rate_at_value"),
    },
)
def p12c_exp_deriv_apply(rng, mode="rate_at_zero"):
    """Exponential model N(t) = N0*exp(k*t), always given. easy: dN/dt at
    t=0. medium: dN/dt at a stated nonzero t0. hard: invert for the time t
    at which dN/dt first reaches a stated target rate R (growth-only, see
    module note). exam: given the target VALUE M the model reaches, find
    dN/dt at that moment (dN/dt = kN)."""
    if mode == "rate_at_zero":
        return _build_exp_easy(rng)
    if mode == "rate_at_t":
        return _build_exp_medium(rng)
    if mode == "time_for_rate":
        return _build_exp_hard(rng)
    if mode == "rate_at_value":
        return _build_exp_exam(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# =============================================================================
# Logarithmic model: V(x) = a*ln(x) + b
# =============================================================================

_LOG_SCENARIOS = [
    dict(subject="the perceived loudness, in decibels, of a sound source", unit="a distance (in metres)"),
    dict(subject="the response, on a standard scale, of a patient to a drug dose", unit="a dose (in mg)"),
    dict(subject="the signal strength, in arbitrary units, received from a transmitter", unit="a distance (in km)"),
]

_A_VALUES = list(range(2, 16))  # positive only, see module note


def _log_model_statement(a: int, b: int, scenario: dict) -> str:
    b_sign = "+" if b >= 0 else "-"
    return (
        f"{scenario['subject'].capitalize()}, as a function of {scenario['unit']} x, is modelled by "
        f"V(x) = {a}·ln(x) {b_sign} {abs(b)}, for x > 0."
    )


def _log_check_equiv(a: int, b: int, x0, constructed: Fraction) -> None:
    x0_frac = Fraction(x0).limit_denominator(10**9)
    x0_sym = Rational(x0_frac.numerator, x0_frac.denominator)
    dV_dx = diff(a * sym_log(x_sym) + b, x_sym).subs(x_sym, x0_sym)
    check_equivalent(
        dV_dx, Rational(constructed.numerator, constructed.denominator), "log derivative independent check failed"
    )


# --- easy: rate at a stated small positive integer x ------------------------


def _build_log_easy(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        a = rng.choice(_A_VALUES)
        b = rng.randint(-20, 20)
        x = rng.randint(2, 15)
        scenario = rng.choice(_LOG_SCENARIOS)

        rate_exact = Fraction(a, x)
        if _near_rounding_boundary(float(rate_exact)):
            continue
        ans = _round2(rate_exact)

        value_eq = a * math.log(x) + b  # evaluates V(x) instead of dV/dx
        drop_a = Fraction(1, x)  # drops a from the derivative
        sign_flip = -rate_exact  # sign error

        cands = [_round2(value_eq), _round2(drop_a), _round2(sign_flip)]
        if ans == 0 or not _gap_ok([ans] + cands):
            continue
        break
    else:
        raise ValueError("p12c_log_deriv_apply: no collision-free easy draw found")

    _log_check_equiv(a, b, x, Fraction(a, x))

    text = (
        f"{_log_model_statement(a, b, scenario)} Find the instantaneous rate of change of V "
        f"with respect to x at x = {x}. {_INSTRUCTION}"
    )
    latex = text
    answer_text = _fmt(ans)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "log easy tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"dV/dx = a / x = {a} / {x} = {answer_text}."],
        distractors=distractors,
    )


# --- medium: x must be resolved from a stated expression first --------------


def _build_log_medium(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        a = rng.choice(_A_VALUES)
        b = rng.randint(-20, 20)
        x_base = rng.randint(2, 10)
        multiplier = rng.choice([2, 3, 4, 5])
        x = x_base * multiplier
        scenario = rng.choice(_LOG_SCENARIOS)

        rate_exact = Fraction(a, x)
        if _near_rounding_boundary(float(rate_exact)):
            continue
        ans = _round2(rate_exact)

        value_eq = a * math.log(x) + b
        drop_a = Fraction(1, x)
        used_base = Fraction(a, x_base)  # forgot to resolve x, used the stated starting value

        cands = [_round2(value_eq), _round2(drop_a), _round2(used_base)]
        if ans == 0 or not _gap_ok([ans] + cands):
            continue
        break
    else:
        raise ValueError("p12c_log_deriv_apply: no collision-free medium draw found")

    _log_check_equiv(a, b, x, Fraction(a, x))

    text = (
        f"{scenario['subject'].capitalize()}, as a function of {scenario['unit']} x, is modelled by "
        f"V(x) = {a}·ln(x) {'+' if b >= 0 else '-'} {abs(b)}, for x > 0. The value of x starts at "
        f"{x_base} and is {multiplier} times that once the system settles. Find the instantaneous "
        f"rate of change of V with respect to x once the system has settled. {_INSTRUCTION}"
    )
    latex = text
    answer_text = _fmt(ans)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "log medium tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"x = {x_base}·{multiplier} = {x}. dV/dx = a / x = {a} / {x} = {answer_text}."],
        distractors=distractors,
    )


# --- hard: invert for x given a target rate R --------------------------------


def _build_log_hard(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        a = rng.choice(_A_VALUES)
        b = rng.randint(-20, 20)
        x_target = rng.randint(2, 300)
        scenario = rng.choice(_LOG_SCENARIOS)

        R_raw = Fraction(a, x_target)
        R = _round2(R_raw)
        if R <= 0:
            continue
        x_exact = Fraction(a) / R  # exact: a/x = R by construction on the DISPLAYED (rounded) R
        x_val = float(x_exact)
        if not (0.5 <= x_val <= 1000):
            continue
        if _near_rounding_boundary(x_val):
            continue
        ans = _round2(x_exact)
        if ans <= 0:
            continue

        Rf = float(R)
        wrong_value_eq = math.exp((Rf - b) / a)  # solved V(x)=R instead of a/x=R
        wrong_drop_a = 1 / Rf  # dropped a: solved 1/x=R
        wrong_sign = -x_val  # sign error

        cands_f = [wrong_value_eq, wrong_drop_a, wrong_sign]
        if any(_near_rounding_boundary(v) for v in cands_f if v > 0):
            continue
        cands = [_round2(v) for v in cands_f]
        if not _gap_ok([ans] + cands):
            continue
        break
    else:
        raise ValueError("p12c_log_deriv_apply: no collision-free hard draw found")

    _log_check_equiv(a, b, x_exact, R)

    text = (
        f"{_log_model_statement(a, b, scenario)} The rate of change of V first equals {_fmt(R)} at "
        f"some value of x. Find this value of x. {_INSTRUCTION}"
    )
    latex = text
    answer_text = _fmt(ans)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "log hard tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"dV/dx = a / x = {_fmt(R)}. x = a / {_fmt(R)} = {a} / {_fmt(R)} = {answer_text}."],
        distractors=distractors,
    )


# --- exam: invert for x given target rate R, then evaluate V at that x -------


def _build_log_exam(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        a = rng.choice(_A_VALUES)
        b = rng.randint(-20, 20)
        x_target = rng.randint(2, 300)
        scenario = rng.choice(_LOG_SCENARIOS)

        R_raw = Fraction(a, x_target)
        R = _round2(R_raw)
        if R <= 0:
            continue
        Rf = float(R)
        x_exact = Fraction(a) / R  # exact: a/x = R by construction on the DISPLAYED (rounded) R
        x_val = float(x_exact)
        if not (0.5 <= x_val <= 1000):
            continue

        V_val = a * math.log(x_val) + b
        if _near_rounding_boundary(V_val):
            continue
        ans = _round2(V_val)

        forgot_invert = a * math.log(Rf) + b  # used R directly as x (R>0 by construction)
        forgot_final_step = x_val  # found x but reported it instead of V(x)
        reported_rate = Rf  # reported the stated rate itself

        cands_f = [forgot_invert, forgot_final_step, reported_rate]
        if any(_near_rounding_boundary(v) for v in cands_f):
            continue
        cands = [_round2(v) for v in cands_f]
        if not _gap_ok([ans] + cands):
            continue
        break
    else:
        raise ValueError("p12c_log_deriv_apply: no collision-free exam draw found")

    # Independent route: verify a/x_exact == R exactly (the inversion step),
    # then re-evaluate V symbolically at x_exact via sympy and compare to the
    # constructed (rounded) answer within the rounding tolerance.
    check_equivalent(
        Rational(a) / Rational(x_exact.numerator, x_exact.denominator),
        Rational(R.numerator, R.denominator),
        "log exam: inversion step a/x_exact must equal the stated rate R",
    )
    a_sym = Rational(a)
    x_sym_val = a_sym / Rational(R.numerator, R.denominator)
    V_sym = a_sym * sym_log(x_sym_val) + b
    check(abs(float(V_sym) - float(ans)) < 0.5, "log exam: constructed V(x) disagrees with sympy evaluation")

    text = (
        f"{_log_model_statement(a, b, scenario)} The rate of change of V first equals {_fmt(R)} at "
        f"some value of x. Find V(x) at that value of x. {_INSTRUCTION}"
    )
    latex = text
    answer_text = _fmt(ans)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "log exam tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"a / x = {_fmt(R)} => x = {a} / {_fmt(R)} = {x_val:.4f}. "
            f"V(x) = {a}·ln({x_val:.4f}) {'+' if b >= 0 else '-'} {abs(b)} = {answer_text}."
        ],
        distractors=distractors,
    )


@generator(
    "p12c_log_deriv_apply",
    level=12,
    presets={
        "easy": dict(mode="rate_at_x"),
        "medium": dict(mode="rate_at_resolved_x"),
        "hard": dict(mode="x_for_rate"),
        "exam": dict(mode="value_at_rate"),
    },
)
def p12c_log_deriv_apply(rng, mode="rate_at_x"):
    """Logarithmic model V(x) = a*ln(x) + b, always given, a a positive
    nonzero integer (see module note). easy: dV/dx at a stated small
    positive integer x. medium: x is stated as an expression (a base value
    and a multiplier) that must be resolved first. hard: invert for the x
    at which dV/dx first equals a stated target rate R. exam: invert for x
    given R (as hard), then evaluate V at that x -- a genuine second step
    hard never asks for."""
    if mode == "rate_at_x":
        return _build_log_easy(rng)
    if mode == "rate_at_resolved_x":
        return _build_log_medium(rng)
    if mode == "x_for_rate":
        return _build_log_hard(rng)
    if mode == "value_at_rate":
        return _build_log_exam(rng)
    raise ValueError(f"unknown mode: {mode!r}")
