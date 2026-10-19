"""Exam-tier sources for three integration skills.

Topics registered here:

``exd_x_h5_velocity_two_positions``
    ``v(t) = A t^2 + k t + D`` with an unknown constant ``k``; the position is given at two different times,
    so ``k`` and the constant of integration must both be found before ``s(t)`` can be written.
``exd_x_h5_antiderivative_linear_power``
    ``f'(x) = c (p x + q)^n`` -- a power of a linear factor (a reverse chain rule), with ``f(a) = b``.
``exd_x_h5_acceleration_rest_or_positions``
    ``a(t) = A t + B`` where the initial velocity is not stated: it must be recovered from either a
    "momentarily at rest at t = T" condition or from a second position, before the position function follows.

Every question is built backwards from the chosen answer, so the key is exact; each generator re-checks the
key by differentiating it back and substituting the stated conditions.
"""

from __future__ import annotations

from sympy import Rational, diff, expand, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
)

_t, _x = symbols("t x")


def _typed(expr) -> str:
    """Learner-typed rendering: ``**`` becomes ``^``."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """Prompt rendering: ``**`` becomes ``^`` and ``*`` the middle dot."""
    return str(expr).replace("**", "^").replace("*", "·")


def _w(expr) -> str:
    """Working rendering: ``**`` becomes ``^`` and the implied product sign is dropped (3t^2, not 3·t^2)."""
    return str(expr).replace("**", "^").replace("*", "")


def _signed(value) -> str:
    return f" + {value}" if value > 0 else f" - {-value}"


def _finish(text, answer_expr, candidates, tolerance_msg, working=None):
    answer_text = _typed(answer_expr)
    cands = []
    for cand in candidates:
        check_distinct(answer_expr, cand, msg=tolerance_msg)
        cands.append(_typed(cand))
    distractors = pick_distractors(answer_text, cands, n=3)
    check(len(distractors) >= 2, "expected at least two distractors")
    return {
        "text": text,
        "latex": text,
        "answer": str(answer_expr),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": list(working or []),
    }


# --------------------------------------------------------------------------------------------------
# Velocity with an unknown parameter, position known at two times.
# --------------------------------------------------------------------------------------------------

_TIME_PAIRS = ((0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1, 4), (2, 4))
_VEL_OPENERS = (
    "A particle moves along a straight line with velocity v(t) = {v} m/s at time t seconds (t ≥ 0), where k is a constant.",
    "The velocity of a particle on a straight line is v(t) = {v} m/s, t ≥ 0, where k is a constant.",
)
_VEL_CONDITIONS = (
    "Its position is {p1} m at t = {t1} and {p2} m at t = {t2}.",
    "The particle is at position {p1} m when t = {t1} and at position {p2} m when t = {t2}.",
)
_VEL_ASKS = (
    "Find an expression for the position s(t).",
    "Find s(t), the position at time t.",
)


@generator("exd_x_h5_velocity_two_positions", level=13, presets={"exam": dict(tier="exam")})
def exd_x_h5_velocity_two_positions(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _attempt in range(100):
        m = rng.choice([v for v in range(-3, 4) if v != 0])
        half_k = rng.choice([v for v in range(-5, 6) if v != 0])
        d = rng.choice([v for v in range(-6, 7) if v != 0])
        c = rng.randint(-9, 9)
        t1, t2 = rng.choice(_TIME_PAIRS)
        s_expr = m * _t**3 + half_k * _t**2 + d * _t + c
        p1, p2 = int(s_expr.subs(_t, t1)), int(s_expr.subs(_t, t2))
        if p1 == p2:
            continue

        # Wrong answers, each from a named slip.
        def fit(a_coef, dt, drop_k=False):
            """Fit k and the constant to the two positions for the form  a_coef t^3 + (k) t^2 + dt t + C."""
            # position difference removes C; unknown t^2 coefficient `q`
            q = Rational(p2 - p1 - a_coef * (t2**3 - t1**3) - dt * (t2 - t1), t2**2 - t1**2)
            cc = p1 - a_coef * t1**3 - q * t1**2 - dt * t1
            return a_coef * _t**3 + (0 if drop_k else q) * _t**2 + dt * _t + cc

        a_cubic = 3 * m
        v_expr = a_cubic * _t**2 + 2 * half_k * _t + d
        check_equivalent(diff(s_expr, _t), v_expr, "s(t) must differentiate back to v(t)")
        check_equivalent(s_expr.subs(_t, t1), p1, "s(t1) must match the first stated position")
        check_equivalent(s_expr.subs(_t, t2), p2, "s(t2) must match the second stated position")
        ignore_k = expand(m * _t**3 + d * _t + (p1 - m * t1**3 - d * t1))
        no_divide = expand(fit(a_cubic, d))
        # Reads k from the second position alone, with the constant of integration taken to be zero.
        q_wrong = Rational(p2 - m * t2**3 - d * t2, t2**2)
        no_constant = m * _t**3 + q_wrong * _t**2 + d * _t
        cands = [ignore_k, no_divide, no_constant, expand(s_expr - c)]
        ok = []
        for cand in cands:
            try:
                check_distinct(s_expr, cand, msg="x")
            except Exception:
                continue
            ok.append(cand)
        if len({_typed(x) for x in ok}) >= 3:
            break
    else:
        raise RuntimeError("could not draw a velocity question with distinct distractors")

    v_text = f"{a_cubic}t^2 + kt{_signed(d)}"
    text = " ".join(
        [
            phrase(rng, _VEL_OPENERS, v=v_text),
            phrase(rng, _VEL_CONDITIONS, p1=p1, p2=p2, t1=t1, t2=t2),
            phrase(rng, _VEL_ASKS),
        ]
    )
    working = [
        f"Integrate: s(t) = {_w(m * _t**3)} + (k/2)t^2{_signed(d)}t + C.",
        f"s({t2}) - s({t1}) = {p2} - {paren_negative(p1)} = {p2 - p1}, so "
        f"{'' if m == 1 else paren_negative(m)}({t2**3} - {t1**3}) + (k/2)({t2**2} - {t1**2}) + {paren_negative(d)}({t2} - {t1}) = {p2 - p1}, "
        f"giving k/2 = {half_k}.",
        f"s({t1}) = {p1} then gives C = {c}.",
        f"s(t) = {_w(expand(s_expr))}.",
    ]
    return _finish(text, expand(s_expr), ok, "a distractor equals the answer", working)


# --------------------------------------------------------------------------------------------------
# f'(x) = c (p x + q)^n with f(a) = b.
# --------------------------------------------------------------------------------------------------


_ANTIDERIV_PROMPTS = (
    "The gradient function of a curve is f'(x) = {fp}, and f({a}) = {b}. Find f(x). Any equivalent form is accepted.",
    "A function f satisfies f'(x) = {fp} and f({a}) = {b}. Determine f(x), in any equivalent form.",
    "Given that f'(x) = {fp} with f({a}) = {b}, find the rule for f(x). Any equivalent form is accepted.",
)


@generator("exd_x_h5_antiderivative_linear_power", level=13, presets={"exam": dict(tier="exam")})
def exd_x_h5_antiderivative_linear_power(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _attempt in range(100):
        n = rng.choice((2, 3, 4, 5))
        p = rng.choice((-3, -2, 2, 3, 4))
        q = rng.choice([v for v in range(-5, 6) if v != 0])
        lead = rng.choice([v for v in range(-4, 5) if v != 0])
        a = rng.randint(-2, 2)
        b = rng.randint(-12, 12)
        lin = p * _x + q
        coeff = lead * p * (n + 1)  # makes the antiderivative's leading coefficient the integer `lead`
        f_prime = coeff * lin**n
        g = lead * lin ** (n + 1)
        cst = b - g.subs(_x, a)
        f_expr = g + cst
        check_equivalent(diff(f_expr, _x), f_prime, "F must differentiate back to f'")
        check_equivalent(f_expr.subs(_x, a), b, "F(a) must equal the stated value")
        if cst == 0 or p * a + q == 0:
            continue
        no_chain = coeff // (n + 1) * lin ** (n + 1)
        no_chain = no_chain + (b - no_chain.subs(_x, a))
        no_const = g
        no_divide = coeff * lin ** (n + 1)
        no_divide = no_divide + (b - no_divide.subs(_x, a))
        sign_const = g + (b + g.subs(_x, a))
        cands = [no_chain, no_const, no_divide, sign_const]
        ok = []
        for cand in cands:
            try:
                check_distinct(f_expr, cand, msg="x")
            except Exception:
                continue
            ok.append(cand)
        if len({_typed(x) for x in ok}) >= 3:
            break
    else:
        raise RuntimeError("could not draw an antiderivative question with distinct distractors")

    lin_text = f"({p}x{_signed(q)})"
    fprime_text = f"{coeff}{lin_text}^{n}" if coeff != 1 else f"{lin_text}^{n}"
    if coeff == -1:
        fprime_text = f"-{lin_text}^{n}"
    text = phrase(rng, _ANTIDERIV_PROMPTS, fp=fprime_text, a=a, b=b)
    answer_expr = f_expr
    # Keep the key in the compact form  lead*(p x + q)^(n+1) + C  (equivalent forms are accepted).
    lead_text = "" if lead == 1 else ("-" if lead == -1 else str(lead))
    lead_times = {1: "", -1: "-"}.get(lead, f"{lead} × ")
    working = [
        f"Reverse the chain rule: f(x) = {lead_text}{lin_text}^{n + 1} + C, since {lead} × {p} × {n + 1} = {coeff}.",
        f"f({a}) = {b}: {lead_times}{paren_negative(lin.subs(_x, a))}^{n + 1} + C = {b}, so C = {cst}.",
        f"f(x) = {lead_text}{lin_text}^{n + 1}{_signed(cst)}.",
    ]
    out = _finish(text, answer_expr, ok, "a distractor equals the answer", working)
    compact = f"{_typed(lead * lin ** (n + 1))}{_signed(cst)}"
    out["answer"] = str(answer_expr)
    out["answer_text"] = compact
    out["answer_latex"] = latex_answer(compact)
    out["distractors"] = pick_distractors(compact, [_typed(c) for c in ok], n=3)
    return out


# --------------------------------------------------------------------------------------------------
# a(t) = A t + B with the initial velocity recovered from a rest or a second position.
# --------------------------------------------------------------------------------------------------

_ACC_OPENERS = (
    "A particle moves along a straight line with acceleration a(t) = {a} m/s² at time t seconds (t ≥ 0).",
    "The acceleration of a particle on a straight line is a(t) = {a} m/s², where t ≥ 0 is the time in seconds.",
)
_ACC_ASKS = (
    "Find an expression for its position s(t).",
    "Find s(t), the position at time t.",
)


@generator("exd_x_h5_acceleration_rest_or_positions", level=13, presets={"exam": dict(tier="exam")})
def exd_x_h5_acceleration_rest_or_positions(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _attempt in range(300):
        a_coef = rng.choice([v for v in range(-6, 7) if v != 0])
        b_coef = rng.choice([v for v in range(-6, 7) if v != 0])
        variant = rng.choice(("rest", "positions"))
        c2 = rng.randint(-9, 9)
        if variant == "rest":
            rest_t = rng.choice((1, 2, 3, 4))
            c1 = -(Rational(a_coef, 2) * rest_t**2 + b_coef * rest_t)
            t_pos = rng.choice([v for v in (0, 1, 2, 3, 4) if v != rest_t])
        else:
            t1, t_pos = rng.choice(((0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1, 4)))
            c1 = rng.choice([v for v in range(-8, 9) if v != 0])
        if Rational(c1).q != 1 or c1 == 0:
            continue
        c1 = int(c1)
        s_expr = Rational(a_coef, 6) * _t**3 + Rational(b_coef, 2) * _t**2 + c1 * _t + c2
        v_expr = diff(s_expr, _t)
        pos_at = s_expr.subs(_t, t_pos)
        if pos_at.q != 1:
            continue
        if variant == "positions":
            p1 = s_expr.subs(_t, t1)
            if p1.q != 1 or p1 == pos_at:
                continue
        check_equivalent(diff(v_expr, _t), a_coef * _t + b_coef, "v must differentiate back to a")
        if variant == "rest":
            check_equivalent(v_expr.subs(_t, rest_t), 0, "the particle must be at rest at the stated time")
        # Slips.
        g_s = Rational(a_coef, 6) * _t**3 + Rational(b_coef, 2) * _t**2
        zero_v0 = g_s + (pos_at - g_s.subs(_t, t_pos))
        no_c2 = g_s + c1 * _t
        sign_c1 = g_s - c1 * _t + (pos_at - (g_s - c1 * _t).subs(_t, t_pos))
        cands = [zero_v0, no_c2, sign_c1]
        if variant == "rest":
            # Reads "at rest at t = T" as v(0) = 0 only after fitting the position (same as zero_v0), so add
            # the slip of taking a(T) = 0 instead of v(T) = 0.
            c1_wrong = -(a_coef * rest_t + b_coef)
            wrong_v = g_s + c1_wrong * _t
            cands.append(wrong_v + (pos_at - wrong_v.subs(_t, t_pos)))
        ok = []
        for cand in cands:
            cand = expand(cand)
            try:
                check_distinct(s_expr, cand, msg="x")
            except Exception:
                continue
            ok.append(cand)
        if len({_typed(x) for x in ok}) >= 3:
            break
    else:
        raise RuntimeError("could not draw an acceleration question with distinct distractors")

    a_text = _prose(a_coef * _t + b_coef).replace("·t", "t")
    if variant == "rest":
        cond = (
            f"The particle is momentarily at rest when t = {rest_t}, and its position is {pos_at} m when t = {t_pos}."
        )
    else:
        cond = f"Its position is {p1} m when t = {t1} and {pos_at} m when t = {t_pos}."
    text = " ".join([phrase(rng, _ACC_OPENERS, a=a_text), cond, phrase(rng, _ACC_ASKS)])
    v_part = Rational(a_coef, 2) * _t**2 + b_coef * _t
    working = [
        f"Integrate: v(t) = {_w(v_part)} + C₁ and s(t) = {_w(g_s)} + C₁t + C₂.",
    ]
    if variant == "rest":
        working.append(f"At rest when t = {rest_t}: {_w(v_part.subs(_t, rest_t))} + C₁ = 0, so C₁ = {c1}.")
    else:
        working.append(
            f"s({t_pos}) - s({t1}) = {pos_at - p1}: {_w(g_s.subs(_t, t_pos) - g_s.subs(_t, t1))} "
            f"+ C₁({t_pos} - {t1}) = {pos_at - p1}, so C₁ = {c1}."
        )
    working.append(f"s({t_pos}) = {pos_at}: {_w(g_s.subs(_t, t_pos) + c1 * t_pos)} + C₂ = {pos_at}, so C₂ = {c2}.")
    working.append(f"s(t) = {_w(expand(s_expr))}.")
    return _finish(text, expand(s_expr), ok, "a distractor equals the answer", working)
