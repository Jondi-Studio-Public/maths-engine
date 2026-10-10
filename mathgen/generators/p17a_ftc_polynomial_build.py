"""Generates `ucv1-ftc-polynomial` questions: evaluate a definite integral of a polynomial using the fundamental
theorem of calculus, `answer: number`.

`ucv1-ftc-polynomial`, `answer: number`.

This skill keeps only the INVERSE direction of the fundamental theorem: the
integral's value is GIVEN and the learner recovers an unknown coefficient or
limit inside it (an equation in the unknown, including choosing the correct
root in a stated domain). Forward evaluation is the shipping `definite-
integral` skill and is out of scope here.

One registered topic, `p17a_ftc_inverse`, four difficulty presets:

- easy: unknown COEFFICIENT c of x^2 in p(x) = c*x^2 + m*x, integral from a
  to b (a possibly nonzero) equals a stated value V; solve the LINEAR
  equation in c. Every equation here (true and every misconception) is
  linear in c, so every candidate is an EXACT rational -- display key and every distractor with `rational_text`
  (int or reduced fraction), never a decimal.
- medium: unknown UPPER limit k, LOWER limit fixed at 0, domain k > 0.
  p(x) = 2*A2*x + Cx (linear, known A2, Cx) so F is quadratic in k.
Constructed via `_linear_case`: pick A2, V with
  `sign(V) == sign(A2)`, so the equation `A2*k^2 + Cx*k - D = 0`'s product of
  roots `-V/A2` is guaranteed NEGATIVE -- exactly one real root is positive
  (the key, generically irrational) and the other negative (the honest
  OLD Vieta-from-two-chosen-roots construction made the key/rejected-root
  pair "nice" (.00/.50) on 162-174/200 draws, an identifiable shape). Two
  further misconceptions are guaranteed in-domain by the SAME sign argument
  (`used-integrand`, `dropped-power-rule-factor` -- see `_linear_case`'s
  question": only 1 in-domain distractor on 189/200 medium draws before).
- hard: unknown limit k, lower limit L a NONZERO known integer, domain
  defect 3: the OLD hard reused medium's quadratic-in-k shape with
  only the lower limit changed, an "one more element, not a change in kind"
  regression -- `measure_tiers.py` read hard's shape as `=` against medium).
  F is now CUBIC in k, built by `_quadratic_case` as
  `A*(u - r_u)*(u^2 + p*u - m)` with `u = k - L`, `r_u < 0` a HIDDEN
  fixed-small-denominator hidden/shown root is exactly what makes a rejected
  root suspiciously "nice"), and `m > 0` so the quadratic factor's product of
  roots `-m` is negative: exactly one of its two roots is positive (`u1`,
  giving the domain-unique key `k1 = L + u1`, generically irrational), the
  other (`u2`) gives the honest rejected root `k2 = L + u2` (also generically
  irrational). `r_u` itself is a genuine third real root of the same cubic
  but is deliberately never surfaced as an option.
- exam: a condition on TWO integrals -- the first (from L to k, unknown) is
  stated as an exact MULTIPLE of a second, fully-specified integral (from a2
  to b2, both known) -- so the learner must evaluate the second integral,
  multiply, and then solve exactly as hard. The FIRST integral's integrand is
  a non-linear integrand on some draws too", so exam is not merely hard's
  linear predecessor with a multiplier bolted on). The multiplier is
  displayed as an exact fraction (it is chosen as V / W where W is the
  second integral's own value, so it is not itself constrained to be
  "nice").

Every tier's key uniqueness is proved twice, independently of the
Vieta/Vieta-like construction that picked it: once via `sympy.Poly(...,
domain="QQ").real_roots()` over the constructed integer coefficients (an
exact root-isolation algorithm, not the closed-form quadratic-factor
computation used to construct the draw), and once via `sympy.integrate` over
the DISPLAYED integrand, confirming it reproduces the stated value V. See
`_close` for the numeric-at-40-digits equality used to compare a `Poly`
root (occasionally a `CRootOf`) against the closed-form root without
depending on `simplify` being able to recognise algebraic equality of two
different radical/CRootOf representations.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import sympy
from sympy import Eq, Poly, Rational, Symbol, integrate, solve, sqrt

from mathgen.core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    fmt_signed,
    generator,
    join_terms,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_x = Symbol("x")
_k = Symbol("k", real=True)
_c_sym = Symbol("c_unknown")

_MAX_DRAW_ATTEMPTS = 200
_DP = 2  # stated decimal places for the limit tiers (medium/hard/exam)
_TOL = stated_precision_tolerance(_DP)

# Every member's reduced denominator is <= 4, and the generator additionally rejects any draw where
# that denominator happens to equal |W| (the reference integral's own value) by coincidence, so
# `denominator(r) == |I_ref|` never occurs and a learner can never read the reference integral's
# value off the multiplier's own denominator.
_EXAM_RATIOS = (Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(3, 2), Fraction(2, 1), Fraction(3, 1))


# ===========================================================================
# shared display helpers
# ===========================================================================


def _append_signed_term(prefix: str, coef: int, var: str) -> str:
    """`prefix` plus a correctly-signed ` + term`/` - term`, e.g.
    `_append_signed_term("c x^2", 3, "x") == "c x^2 + 3x"`;
    `_append_signed_term("c x^2", -1, "x") == "c x^2 - x"`;
    `_append_signed_term("c x^2", 0, "x") == "c x^2"`."""
    if coef == 0:
        return prefix
    term = fmt_signed(coef, var)
    if term.startswith("-"):
        return f"{prefix} - {term[1:]}"
    return f"{prefix} + {term}"


def _coeff_poly_text(m: int) -> str:
    """'c x^2' plus the known linear term, e.g. 'c x^2 + 3x'/'c x^2 - 4x'."""
    return _append_signed_term("c x^2", m, "x")


def _linear_poly_text(B: int, C: int) -> str:
    return join_terms([(B, "x"), (C, "")])


def _quad_poly_text(A3: int, B: int, C: int) -> str:
    return join_terms([(A3, "x^2"), (B, "x"), (C, "")])


def _round_grid(value, d: int) -> Fraction | None:
    """Round an exact sympy real number (rational or algebraic) to `d`
    decimal places on the exact `10**-d` grid, using Python `Decimal` at 50
    significant digits of precision (far beyond what a rounding decision at
    2dp ever needs). Returns None if the value sits within 1e-9 of the
    halfway rounding boundary (a genuine ambiguity a real calculator would
    also stumble on) -- the caller drops the candidate rather than trust it.
    """
    hp = sympy.N(value, 50)
    dec = Decimal(str(hp))
    quantum = Decimal(1).scaleb(-d)
    scaled = dec / quantum
    frac = abs(scaled) % 1  # Decimal % keeps the dividend's sign; a negative tie must be seen too
    dist_from_half = abs(frac - Decimal("0.5"))
    if dist_from_half < Decimal("0.000000001"):
        return None
    rounded = dec.quantize(quantum, rounding=ROUND_HALF_UP)
    return Fraction(str(rounded))


def _fmt2(value: Fraction) -> str:
    return stated_precision_string(value, _DP)


def _close(a, b, digits: int = 40) -> bool:
    """Numeric equality at `digits` significant figures -- used to compare a
    `Poly.real_roots()` result (occasionally a `CRootOf`, which `simplify`
    cannot always recognise as algebraically equal to a different radical
    representation of the same root) against a closed-form root computed a
    different way. 40 digits is far past the 2dp a candidate is ever
    rounded to, so two DIFFERENT roots always disagree well before this
    threshold; only the SAME root passes."""
    return abs(sympy.N(a, digits) - sympy.N(b, digits)) < sympy.Float(f"1e-{digits - 5}")


_PRECISION_SENTENCE = "Give your answer correct to 2 decimal places."


# ===========================================================================
# easy: unknown coefficient c (route (a): exact rational display)
# ===========================================================================


# Lower limit fixed at 0 (as in the ladder's own example), upper limit `b`
#: a multiple of 6 (so K1 = b**3/3 and K2 = b**2/2 are both exact integers),
#: with the LINEAR coefficient `m` in turn forced to a multiple of `|b|`
#: (verified directly with `Fraction` arithmetic, not by hand, in
#: `/tmp/find_b_m.py` during development -- every value of `m` a multiple of
#: `b` keeps BOTH the "swapped limits" and "used the integrand" misconception
#: candidates on the integer grid) and `c` forced to a multiple of 3 (keeps
#: "dropped factor" == c/3 on the grid). "forgot to subtract F(lower)"
#: degenerates to the SAME value as the true equation here (F(0) = 0
#: already), so it is not exercisable at this tier -- documented, not a
#: defect (the same situation as the medium tier's fixed lower limit).
#: Forcing every option to a bare integer (never a fraction) is what keeps
#: the key from being the only option of its shape.
_EASY_B_CHOICES = (6, -6, 12, -12)
_EASY_C_CHOICES = tuple(3 * n for n in range(-6, 7) if n != 0)


def _easy_m_choices(b: int) -> list[int]:
    return [b * n for n in range(-4, 5)]


def _easy_draw(rng):
    a = 0
    b = rng.choice(_EASY_B_CHOICES)
    m = rng.choice(_easy_m_choices(b))
    c = rng.choice(_EASY_C_CHOICES)
    return a, b, m, c


def _easy_build(rng):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        a, b, m, c = _easy_draw(rng)
        K1 = Fraction(b**3 - a**3, 3)
        K2 = Fraction(b**2 - a**2, 2)
        check(K1.denominator == 1 and K2.denominator == 1, f"easy: K1/K2 not integer for a={a}, b={b}")
        V = c * K1 + m * K2

        # Independent check: sympy's own `integrate`, over a genuinely
        # unknown symbol, solved for c -- a different route from the K1/K2
        # Fraction arithmetic used to construct V.
        F_expr = integrate(_c_sym * _x**2 + m * _x, (_x, a, b))
        solved = solve(Eq(F_expr, Rational(V.numerator, V.denominator)), _c_sym)
        check(len(solved) == 1, f"easy: expected a unique linear solution, got {solved}")
        check_equivalent(
            solved[0],
            Rational(c.numerator, c.denominator),
            f"easy independent check failed: a={a}, b={b}, m={m}, c={c}",
        )

        raw_candidates = {
            # swapped limits: computed F(a) - F(b) = V instead of F(b) - F(a) = V.
            "swapped-limits": (-V - m * K2) / K1,
            # used the integrand p(x) instead of the antiderivative F(x).
            "used-integrand": (V - m * (b - a)) / (b**2 - a**2),
            # dropped the 1/(n+1) factor on the unknown's own term only.
            "dropped-factor": (V - m * K2) / (b**3 - a**3),
        }
        # Every option must be a BARE INTEGER, the same shape as c (never a
        # fraction): a lone integer among fractions (or vice versa) is
        # pickable by form alone (`test_answer_is_not_the_only_option_of_its_shape`).
        check(
            all(v.denominator == 1 for v in raw_candidates.values()),
            f"easy: a misconception candidate is not an exact integer: a={a}, b={b}, m={m}, c={c}, "
            f"candidates={raw_candidates}",
        )
        candidates = {k: int(v) for k, v in raw_candidates.items()}

        try:
            check_pairwise_distinct(
                c, *candidates.values(), msg=f"easy misconception collision: a={a}, b={b}, m={m}, c={c}"
            )
        except Exception:
            continue

        answer_text = str(c)
        cand_texts = {k: str(v) for k, v in candidates.items()}
        if len(set(cand_texts.values()) | {answer_text}) != len(cand_texts) + 1:
            continue

        text = phrase(
            rng,
            (
                "The integral from {a} to {b} of ({poly}) dx equals {V}. Find c.",
                "Given that the integral from {a} to {b} of ({poly}) dx is {V}, find the value of c.",
                "The definite integral of ({poly}) dx, evaluated from x = {a} to x = {b}, equals {V}. What is c?",
            ),
            a=a,
            b=b,
            poly=_coeff_poly_text(m),
            V=rational_text(V),
        )
        distractors = pick_distractors(answer_text, list(cand_texts.values()), n=3)
        F_text = _append_signed_term("c x^3/3", m, "x^2/2")
        return dict(
            text=text,
            answer=c,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"F(x) = {F_text} is an antiderivative of {_coeff_poly_text(m)}.",
                f"F({b}) - F({a}) = {rational_text(V)} gives a linear equation in c "
                f"whose solution is c = {answer_text}.",
            ],
            distractors=distractors,
        )
    raise AssertionError("could not draw a collision-free easy ftc-polynomial question")


# ===========================================================================
# limit tiers (medium/hard/exam): route (b), decimal display
#
# Both `_linear_case` and `_quadratic_case` below share the same construction
# idea: build the true equation so its
# relevant roots' PRODUCT is forced negative (or, for the cubic, the
# relevant quadratic FACTOR's roots' product is forced negative). That
# guarantees, by a sign argument alone -- never by picking specific "nice"
# roots -- that exactly one real root lands in the stated domain, and that
# root is generically irrational (no small-denominator bias), fixing defect
# 2. The same sign argument, applied to two named misconceptions, guarantees
# at least two in-domain distractors on every returned draw, fixing defect 1.
# ===========================================================================


def _in_domain_root(expr, domain_lo, avoid):
    """First real root of `expr` (a sympy expression in `_k`) strictly
    greater than `domain_lo`, rounded to `_DP` places, at least `_TOL` away
    from every value in `avoid` -- or `None` if no such root exists. Used for
    the OPPORTUNISTIC misconceptions (no sign-argument guarantee)."""
    roots = [r for r in Poly(expr, _k, domain="QQ").real_roots() if r.evalf(40) > domain_lo]
    for r in roots:
        cand = _round_grid(r, _DP)
        if cand is None:
            continue
        if any(abs(cand - a) <= _TOL for a in avoid):
            continue
        return cand
    return None


def _linear_case(rng, L: int) -> dict | None:
    """One attempt at a linear-integrand limit case at lower limit `L`:
    p(x) = 2*A2*x + Cx (F(x) = A2*x^2 + Cx*x), F(k) - F(L) = V. Returns
    `None` if this attempt should be discarded (caller retries).

    Constructed in `u = k - L` as `A2*u^2 + E*u - V = 0` with `sign(V) == sign(A2)` forced: the product of
    roots `-V/A2` is then guaranteed negative, so exactly one real root is positive (`u1`, the domain-unique
    key `k1 = L + u1`, generically irrational) and the other negative (`u2`, the honest rejected root `k2 = L
    + u2`, also generically irrational). The SAME sign argument, applied to the "used the integrand" and
    "dropped the power-rule factor" misconception equations below, guarantees each of THOSE also has exactly
    one root with `u > 0` -- two guaranteed in-domain distractors on every draw.
    """
    # |A2| == 1 makes "used the integrand" (k = L + V/(2*A2), an EXACT half-
    # defect 2 found this pushed the distractor-nice rate to 35% against an
    # 8% key-nice rate. Excluding |A2| == 1 lifts the typical denominator of
    # V/(2*A2) instead.
    A2 = rng.choice([3, -3, 4, -4, 5, -5, 6, -6])
    sign = 1 if A2 > 0 else -1
    V = sign * rng.randint(4, 90)
    E = rng.randint(-12, 12)
    Cx = E - 2 * A2 * L

    disc = E * E + 4 * A2 * V
    if disc <= 0:
        return None
    sqrt_disc = sqrt(disc)
    u_a = (Rational(-E) + sqrt_disc) / (2 * A2)
    u_b = (Rational(-E) - sqrt_disc) / (2 * A2)
    u1, u2 = (u_a, u_b) if float(u_a) > 0 else (u_b, u_a)
    check(float(u1) > 0 and float(u2) < 0, f"linear-case: sign guarantee violated A2={A2}, E={E}, V={V}")
    k1 = L + u1
    k2 = L + u2

    # Independent check 1: sympy's own Poly root-finder over the constructed
    # integer coefficients (A2, Cx, D) -- a different route from the
    # closed-form quadratic-formula computation above.
    D = A2 * L * L + Cx * L + V
    roots = Poly(A2 * _k**2 + Cx * _k - D, _k, domain="QQ").real_roots()
    check(len(roots) == 2, f"linear-case: expected two real roots, got {roots}")
    in_domain = [r for r in roots if r.evalf(40) > L]
    check(len(in_domain) == 1, f"linear-case: expected exactly one root in domain, got {in_domain}")
    check(_close(in_domain[0], k1), f"linear-case: domain root mismatch: {in_domain[0]} vs {k1}")

    # Independent check 2: sympy's own `integrate`, over the DISPLAYED
    # integrand, reproduces V.
    p_expr = 2 * A2 * _x + Cx
    F_check = integrate(p_expr, (_x, L, k1))
    check(_close(F_check, V), "linear-case: sympy integral disagrees with constructed V")

    k1_2dp = _round_grid(k1, _DP)
    k2_2dp = _round_grid(k2, _DP)
    if k1_2dp is None or k2_2dp is None:
        return None
    avoid = [k1_2dp, k2_2dp]

    candidates: dict[str, Fraction] = {"rejected-root": k2_2dp}

    # used the integrand p(x) instead of the antiderivative F(x):
    # p(k) - p(L) = V  <=>  2*A2*(k - L) = V  <=>  k = L + V/(2*A2).
    # Guaranteed in-domain: sign(V) == sign(A2) forces V/(2*A2) > 0.
    integrand_root = L + Fraction(V, 2 * A2)
    integrand_2dp = _round_grid(Rational(integrand_root.numerator, integrand_root.denominator), _DP)
    if integrand_2dp is not None and all(abs(integrand_2dp - a) > _TOL for a in avoid):
        candidates["used-integrand"] = integrand_2dp
        avoid.append(integrand_2dp)

    # dropped the power-rule factor: uses 2*A2 (the integrand's own x-term
    # coefficient) as the antiderivative's x^2 coefficient instead of A2.
    # In u = k - L: 2*A2*u^2 + (2*A2*L + E)*u - V = 0, whose product of roots
    # -V/(2*A2) is negative by the same sign argument as above.
    dropped_expr = 2 * A2 * (_k - L) ** 2 + (2 * A2 * L + E) * (_k - L) - V
    dropped_roots = [r for r in Poly(dropped_expr, _k, domain="QQ").real_roots() if r.evalf(40) > L]
    if dropped_roots:
        cand = _round_grid(dropped_roots[0], _DP)
        if cand is not None and all(abs(cand - a) > _TOL for a in avoid):
            candidates["dropped-power-rule-factor"] = cand
            avoid.append(cand)

    # forgot to subtract F(lower): F(k) = V directly. Degenerate (same as
    # the true equation) when L == 0; live and opportunistic otherwise.
    if L != 0:
        forgot_expr = A2 * _k**2 + Cx * _k - V
        cand = _in_domain_root(forgot_expr, L, avoid)
        if cand is not None:
            candidates["forgot-lower"] = cand
            avoid.append(cand)

    # swapped limits: F(L) - F(k) = V (opportunistic).
    F_L = A2 * L * L + Cx * L
    swapped_expr = A2 * _k**2 + Cx * _k - (F_L - V)
    cand = _in_domain_root(swapped_expr, L, avoid)
    if cand is not None:
        candidates["swapped-limits"] = cand
        avoid.append(cand)

    if len(candidates) - 1 < 2:  # exclude rejected-root from the "in-domain" count
        return None

    return dict(kind="linear", A2=A2, Cx=Cx, V=V, L=L, k1_2dp=k1_2dp, candidates=candidates)


def _quadratic_case(rng, L: int) -> dict | None:
    """Returns `None` if this attempt should be discarded (caller retries).

    p(x) = 3*A*x^2 + 2*B*x + C, F(x) = A*x^3 + B*x^2 + C*x. Constructed as `A*(u - r_u)*(u^2 + p*u - m)` with
    `u = k - L`, `r_u < 0` a HIDDEN construction root, and `m > 0` so the quadratic factor's product of roots
    `-m` is negative: exactly one of its two roots is positive (`u1`, the domain-unique key `k1 = L + u1`,
    generically irrational), the other (`u2`) gives the honest rejected root `k2 = L + u2` (also generically
    irrational). `r_u` is a genuine third real root of the cubic but is never surfaced.
    """
    d = rng.choice([1, 1, 2])
    num = -rng.randint(1, 9 * d)  # r_u = num/d < 0
    p = rng.randint(-15, 15)
    m = rng.randint(1, 40)
    base = rng.choice([1, -1, 2, -2])
    A = base * d

    disc = p * p + 4 * m
    if disc <= 0:
        return None
    # Rejecting it forces the key genuinely irrational, generically.
    if sympy.sqrt(disc) == int(sympy.sqrt(disc)):
        return None
    sqrt_disc = sqrt(disc)
    u_a = (Rational(-p) + sqrt_disc) / 2
    u_b = (Rational(-p) - sqrt_disc) / 2
    u1, u2 = (u_a, u_b) if float(u_a) > 0 else (u_b, u_a)
    check(float(u1) > 0 and float(u2) < 0, f"quadratic-case: sign guarantee violated p={p}, m={m}")

    Bp = A * p - base * num  # B' = A*p - A*r_u  (A*r_u = base*num, exact)
    Cp = -A * m - base * num * p  # C' = -A*m - A*r_u*p
    V = base * (-num) * m  # V = -A*r_u*m; sign(V) == sign(A)

    B = Bp - 3 * A * L
    C = 3 * A * L * L - 2 * Bp * L + Cp

    k1 = L + u1
    k2 = L + u2

    # Independent check 1: sympy's own Poly root-finder over the constructed
    # integer coefficients -- a different route from the closed-form
    # quadratic-factor computation above.
    D = A * L**3 + B * L**2 + C * L + V
    roots = Poly(A * _k**3 + B * _k**2 + C * _k - D, _k, domain="QQ").real_roots()
    check(len(roots) == 3, f"quadratic-case: expected three real roots, got {roots}")
    in_domain = [r for r in roots if r.evalf(40) > L]
    check(len(in_domain) == 1, f"quadratic-case: expected exactly one root in domain, got {in_domain}")
    check(_close(in_domain[0], k1), f"quadratic-case: domain root mismatch: {in_domain[0]} vs {k1}")

    # Independent check 2: sympy's own `integrate`, over the DISPLAYED
    # quadratic integrand, reproduces V.
    p_expr = 3 * A * _x**2 + 2 * B * _x + C
    F_check = integrate(p_expr, (_x, L, k1))
    check(_close(F_check, V), "quadratic-case: sympy integral disagrees with constructed V")

    k1_2dp = _round_grid(k1, _DP)
    k2_2dp = _round_grid(k2, _DP)
    if k1_2dp is None or k2_2dp is None:
        return None
    avoid = [k1_2dp, k2_2dp]

    candidates: dict[str, Fraction] = {"rejected-root": k2_2dp}

    # used the integrand p(x) instead of the antiderivative F(x):
    # p(k) - p(L) = V  <=>  3*A*u^2 + (6*A*L + 2*B)*u - V = 0, whose product
    # of roots -V/(3*A) is negative by the same sign argument as above.
    used_expr = 3 * A * (_k - L) ** 2 + (6 * A * L + 2 * B) * (_k - L) - V
    used_roots = [r for r in Poly(used_expr, _k, domain="QQ").real_roots() if r.evalf(40) > L]
    if used_roots:
        cand = _round_grid(used_roots[0], _DP)
        if cand is not None and all(abs(cand - a) > _TOL for a in avoid):
            candidates["used-integrand"] = cand
            avoid.append(cand)

    # forgot to subtract F(lower): F(k) = V directly (opportunistic; live
    # here since L != 0 at both hard and exam).
    forgot_expr = A * _k**3 + B * _k**2 + C * _k - V
    cand = _in_domain_root(forgot_expr, L, avoid)
    if cand is not None:
        candidates["forgot-lower"] = cand
        avoid.append(cand)

    # swapped limits: F(L) - F(k) = V (opportunistic).
    F_L = A * L**3 + B * L**2 + C * L
    swapped_expr = A * _k**3 + B * _k**2 + C * _k - (F_L - V)
    cand = _in_domain_root(swapped_expr, L, avoid)
    if cand is not None:
        candidates["swapped-limits"] = cand
        avoid.append(cand)

    # dropped the power-rule factor: uses 3*A (the integrand's own x^2-term
    # coefficient) as the antiderivative's x^3 coefficient instead of A
    # (opportunistic).
    dropped_expr = 3 * A * _k**3 + B * _k**2 + C * _k - (3 * A * L**3 + B * L**2 + C * L + V)
    cand = _in_domain_root(dropped_expr, L, avoid)
    if cand is not None:
        candidates["dropped-power-rule-factor"] = cand
        avoid.append(cand)

    if len(candidates) - 1 < 2:  # exclude rejected-root from the "in-domain" count
        return None

    return dict(kind="quadratic", A=A, B=B, C=C, V=V, L=L, k1_2dp=k1_2dp, candidates=candidates)


def _limit_build(rng, tier: str):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        if tier == "medium":
            L = 0
            case = _linear_case(rng, L)
        elif tier == "hard":
            L = rng.choice([-3, -2, -1, 1, 2, 3])
            case = _quadratic_case(rng, L)
        else:  # exam
            L = rng.choice([-3, -2, -1, 1, 2, 3])
            quad = rng.choice([True, False])
            case = _quadratic_case(rng, L) if quad else _linear_case(rng, L)

        if case is None:
            continue

        L = case["L"]
        V = case["V"]
        k1_2dp = case["k1_2dp"]
        candidates = dict(case["candidates"])
        quad = case["kind"] == "quadratic"

        if quad:
            A, B, C = case["A"], case["B"], case["C"]
            poly_text = _quad_poly_text(3 * A, 2 * B, C)
            p_expr = 3 * A * _x**2 + 2 * B * _x + C
        else:
            A2, Cx = case["A2"], case["Cx"]
            poly_text = _linear_poly_text(2 * A2, Cx)
            p_expr = 2 * A2 * _x + Cx

        if tier == "exam":
            # The displayed ratio `r` is drawn from the small fixed set FIRST, the reference
            # integral W is evaluated from a2/b2, and the target value the first integral must equal
            # is then DEFINED as `r * W` (never the reverse: `denominator(r)` is therefore never
            # derived from W, so it can never equal `|W|` except by a coincidence this loop
            # explicitly rejects).
            a2 = L + rng.choice([6, 7, 8, 9])
            b2 = a2 + rng.choice([1, 2, 3])
            W = integrate(p_expr, (_x, a2, b2))
            W = int(W)
            if W == 0:
                continue

            multiplier = rng.choice(_EXAM_RATIOS)
            if multiplier.denominator == abs(W):
                continue  # would give away |I_ref| by coincidence

            V_new = multiplier * W  # exact Fraction; may be non-integer
            V_sym = Rational(V_new.numerator, V_new.denominator)

            if quad:
                D = A * L**3 + B * L**2 + C * L + V_sym
                expr = A * _k**3 + B * _k**2 + C * _k - D
            else:
                D = A2 * L * L + Cx * L + V_sym
                expr = A2 * _k**2 + Cx * _k - D
            roots = Poly(expr, _k, domain="QQ").real_roots()
            in_domain = [r for r in roots if r.evalf(40) > L]
            if len(in_domain) != 1:
                continue
            k1 = in_domain[0]
            k1_2dp = _round_grid(k1, _DP)
            if k1_2dp is None:
                continue

            # Independent check: sympy's own `integrate`, over the DISPLAYED
            # integrand, reproduces the target `r * W` exactly.
            F_check = integrate(p_expr, (_x, L, k1))
            check(_close(F_check, V_sym), "exam: sympy integral disagrees with the target ratio value")

            V = V_new
            avoid = [k1_2dp]
            candidates = {}

            other_roots = [r for r in roots if r.evalf(40) <= L]
            if other_roots:
                cand = _round_grid(other_roots[0], _DP)
                if cand is not None and all(abs(cand - a) > _TOL for a in avoid):
                    candidates["rejected-root"] = cand
                    avoid.append(cand)

            # used the integrand p(x) instead of the antiderivative F(x).
            if quad:
                used_expr = 3 * A * (_k - L) ** 2 + (6 * A * L + 2 * B) * (_k - L) - V_sym
            else:
                used_expr = 2 * A2 * (_k - L) - V_sym
            cand = _in_domain_root(used_expr, L, avoid)
            if cand is not None:
                candidates["used-integrand"] = cand
                avoid.append(cand)

            # forgot to subtract F(lower): F(k) = V directly (live: L != 0
            # at exam).
            if quad:
                forgot_expr = A * _k**3 + B * _k**2 + C * _k - V_sym
            else:
                forgot_expr = A2 * _k**2 + Cx * _k - V_sym
            cand = _in_domain_root(forgot_expr, L, avoid)
            if cand is not None:
                candidates["forgot-lower"] = cand
                avoid.append(cand)

            # swapped limits: F(L) - F(k) = V.
            if quad:
                F_L = A * L**3 + B * L**2 + C * L
                swapped_expr = A * _k**3 + B * _k**2 + C * _k - (F_L - V_sym)
            else:
                F_L = A2 * L * L + Cx * L
                swapped_expr = A2 * _k**2 + Cx * _k - (F_L - V_sym)
            cand = _in_domain_root(swapped_expr, L, avoid)
            if cand is not None:
                candidates["swapped-limits"] = cand
                avoid.append(cand)

            # dropped the power-rule factor.
            if quad:
                dropped_expr = 3 * A * _k**3 + B * _k**2 + C * _k - (3 * A * L**3 + B * L**2 + C * L + V_sym)
            else:
                E = Cx + 2 * A2 * L
                dropped_expr = 2 * A2 * (_k - L) ** 2 + (2 * A2 * L + E) * (_k - L) - V_sym
            cand = _in_domain_root(dropped_expr, L, avoid)
            if cand is not None:
                candidates["dropped-power-rule-factor"] = cand
                avoid.append(cand)

            # forgot the stated multiplier: F(k) - F(L) = W (unmultiplied).
            if quad:
                D_nomult = A * L**3 + B * L**2 + C * L + W
                nomult_expr = A * _k**3 + B * _k**2 + C * _k - D_nomult
            else:
                D_nomult = A2 * L * L + Cx * L + W
                nomult_expr = A2 * _k**2 + Cx * _k - D_nomult
            cand = _in_domain_root(nomult_expr, L, avoid)
            if cand is not None:
                candidates["forgot-multiplier"] = cand
                avoid.append(cand)
        else:
            a2 = b2 = W = multiplier = None

        in_domain_count = len([name for name in candidates if name != "rejected-root"])
        if in_domain_count < 2:
            continue

        answer_text = _fmt2(k1_2dp)
        cand_texts = {name: _fmt2(v) for name, v in candidates.items()}
        if answer_text in cand_texts.values():
            continue

        if tier == "medium":
            text = phrase(
                rng,
                (
                    "The integral from 0 to k of ({poly}) dx equals {V}, where k > 0. Find k. {prec}",
                    "Given that the integral from 0 to k of ({poly}) dx is {V} and k > 0, find k. {prec}",
                    "The definite integral of ({poly}) dx from x = 0 to x = k equals {V}, with "
                    "k > 0. What is k? {prec}",
                ),
                poly=poly_text,
                V=V,
                prec=_PRECISION_SENTENCE,
            )
        elif tier == "hard":
            text = phrase(
                rng,
                (
                    "The integral from {L} to k of ({poly}) dx equals {V}, where k > {L}. Find k. {prec}",
                    "Given that the integral from {L} to k of ({poly}) dx is {V} and k > {L}, find k. {prec}",
                    "The definite integral of ({poly}) dx from x = {L} to x = k equals {V}, with "
                    "k > {L}. What is k? {prec}",
                ),
                L=L,
                poly=poly_text,
                V=V,
                prec=_PRECISION_SENTENCE,
            )
        else:  # exam
            text = phrase(
                rng,
                (
                    "The integral from {L} to k of ({poly}) dx equals {mult} times the integral "
                    "from {a2} to {b2} of ({poly}) dx, where k > {L}. Find k. {prec}",
                    "Given that the integral from {L} to k of ({poly}) dx is {mult} times the "
                    "integral from {a2} to {b2} of ({poly}) dx, with k > {L}, find k. {prec}",
                    "The definite integral of ({poly}) dx from x = {L} to x = k is {mult} times "
                    "the definite integral of the same function from x = {a2} to x = {b2}, with "
                    "k > {L}. What is k? {prec}",
                ),
                L=L,
                poly=poly_text,
                mult=rational_text(multiplier),
                a2=a2,
                b2=b2,
                prec=_PRECISION_SENTENCE,
            )

        distractors = pick_distractors(answer_text, list(cand_texts.values()), n=3, tolerance=_TOL)
        if len(distractors) < 2:
            continue

        if quad:
            F_text = join_terms([(A, "x^3"), (B, "x^2"), (C, "x")])
        else:
            F_text = join_terms([(A2, "x^2"), (Cx, "x")])
        working = [
            f"p(x) = {poly_text}; an antiderivative is F(x) = {F_text}.",
        ]
        V_text = rational_text(V) if tier == "exam" else str(V)
        if tier == "exam":
            working.append(
                f"The second integral evaluates to {W}, so the first equals "
                f"{rational_text(multiplier)} x {W} = {V_text}."
            )
        working.append(
            f"F(k) - F({L}) = {V_text} has exactly one root with k > {L}: "
            f"k = {answer_text} (the domain excludes the other real root(s))."
        )

        return dict(
            text=text,
            answer=numeric_answer(k1_2dp),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=working,
            distractors=distractors,
        )
    raise AssertionError(f"could not draw a collision-free {tier} ftc-polynomial question")


# ===========================================================================
# registered topic
# ===========================================================================


@generator(
    "p17a_ftc_inverse",
    level=19,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p17a_ftc_inverse(rng, tier="easy"):
    if tier == "easy":
        return _easy_build(rng)
    return _limit_build(rng, tier)
