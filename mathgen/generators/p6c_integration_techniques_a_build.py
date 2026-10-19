"""Integration by parts and partial fractions.

- ``integrate_by_parts``
  (``senior-integrate-by-parts``, ACMSM123, domain integration,
  answer: expression) -- given bounded, fully specified inputs, integrate by
  parts. Return the resulting value, expression, relation, or classification.

- ``use_partial_fractions_where_necessary_for_integration_in``
  (``senior-use-partial-fractions-where-necessary-for-integration-in``,
  ACMSM122, domain integration, answer: expression) -- given bounded, fully
  specified inputs, use partial fractions where necessary for integration in
  simple cases. Return the resulting value, expression, relation, or
  classification.
"""

from __future__ import annotations

from sympy import symbols, sin, cos, log, diff, pi, Rational, simplify

from ..calc_cache import integrate

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    pick_distractors,
    symbolically_equal,
)


# Display helpers: convert ** to ^ for typed, ** to ^ and * to · for prose
def _typed(expr) -> str:
    """Convert expression to learner-facing typed form: ** -> ^, and sympy's log -> the ln a learner writes."""
    return str(expr).replace("**", "^").replace("log(", "ln(")


def _prose(expr) -> str:
    """Convert expression to prose form: ** -> ^, log -> ln, * to ·."""
    typed_str = str(expr).replace("**", "^").replace("log(", "ln(")
    return typed_str.replace("*", "·")


def _is_non_terminating_rational(value) -> bool:
    """True if `value` is a rational number whose reduced denominator has a
    prime factor other than 2 or 5 -- its decimal expansion never
    terminates, so a learner who reports a correctly-rounded decimal is
    typing something that is not exactly the canonical answer. False for a
    non-rational value (pi/log/sqrt-valued): those always need the
    exact-form instruction regardless of this check, and are handled by
    the plain "contains pi/log/sqrt" condition at each call site, not this
    function.
    """
    if not value.is_rational:
        return False
    denom = Rational(value).q
    for prime in (2, 5):
        while denom % prime == 0:
            denom //= prime
    return denom != 1


# Module-scope symbol
x = symbols("x")


@generator(
    "integrate_by_parts",
    level=17,
    presets={
        "easy": dict(difficulty="easy"),
        "medium": dict(difficulty="medium"),
        "hard": dict(difficulty="hard"),
    },
)
def integrate_by_parts(rng, difficulty="easy"):
    """
    Integrate by parts: compute definite integrals using the by-parts formula.

    easy: u = x, dv = sin(k*x) dx, bounds [0, pi/k]
    medium: u = x, dv = sin(k*x) dx, bounds [pi/(2*k), pi/k]
    hard: u = log(x), dv = x dx, bounds [n, m] where n, m >= 1
    """

    if difficulty == "easy":
        kind = "trig"
        k = rng.randint(2, 6)  # k >= 2 ensures wrong_v is distinct
        c = rng.randint(1, 5)
        f = c * x * sin(k * x)

        # By-parts formula: u*v - integral(v*du), then multiply by c
        u = x
        dv = sin(k * x)
        du = 1
        v = -cos(k * x) / k

        # Compute antiderivative (for x*sin(k*x), then multiply by c)
        integral_v_du = integrate(v, x)
        F_base = u * v - integral_v_du
        F = c * F_base

        # Bounds: hi = π/(2k) avoids misconception collapse at bounds
        lo = 0
        hi = pi / (2 * k)

        # Answer: F(hi) - F(lo)
        ans = F.subs(x, hi) - F.subs(x, lo)

        # Independent checks: F is genuine antiderivative of f, answer matches sympy
        check_equivalent(diff(F, x), f)
        check_equivalent(ans, integrate(f, (x, lo, hi)))

        # Misconceptions (all three checked unconditionally)
        # sign_error: u*v + integral(v*du) instead of u*v - integral(v*du)
        F_sign_base = u * v + integral_v_du
        F_sign = c * F_sign_base
        sign_error = F_sign.subs(x, hi) - F_sign.subs(x, lo)

        # dropped_integral_term: only u*v without the integral term
        F_drop_base = u * v
        F_drop = c * F_drop_base
        dropped_integral_term = F_drop.subs(x, hi) - F_drop.subs(x, lo)

        # wrong_v_antiderivative: v without dividing by k
        v_wrong = -cos(k * x)
        F_v_base = u * v_wrong - integrate(v_wrong * du, x)
        F_v = c * F_v_base
        wrong_v_antiderivative = F_v.subs(x, hi) - F_v.subs(x, lo)

        check_distinct(ans, sign_error, dropped_integral_term, wrong_v_antiderivative)

        candidates = [
            _typed(sign_error),
            _typed(dropped_integral_term),
            _typed(wrong_v_antiderivative),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        exact_note = ". Give your answer as an exact value." if _is_non_terminating_rational(ans) else ""
        prompt = f"Find ∫ from x = 0 to x = π/{2 * k} of {_prose(f)} dx{exact_note}"
        working = [
            f"u = x, dv = {_prose(sin(k * x))} dx  ⇒  du = dx, v = {_prose(-cos(k * x) / k)}",
            f"F(x) = {_prose(F)}",
            f"F(π/{2 * k}) - F(0) = {_prose(ans)}",
        ]

        return {
            "text": prompt,
            "latex": prompt,
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": latex_answer(ans),
            "distractors": distractors,
            "working": working,
        }

    elif difficulty == "medium":
        kind = "trig_offset"
        k = rng.randint(2, 6)  # k >= 2 ensures wrong_v is distinct
        c = rng.randint(1, 5)
        f = c * x * sin(k * x)

        # By-parts formula, then multiply by c
        u = x
        dv = sin(k * x)
        du = 1
        v = -cos(k * x) / k

        # Compute antiderivative
        integral_v_du = integrate(v, x)
        F_base = u * v - integral_v_du
        F = c * F_base

        # Bounds: both nonzero
        lo = pi / (2 * k)
        hi = pi / k

        # Answer
        ans = F.subs(x, hi) - F.subs(x, lo)

        # Independent checks
        check_equivalent(diff(F, x), f)
        check_equivalent(ans, integrate(f, (x, lo, hi)))

        # Misconceptions
        # swapped_uv: swapped the roles of u and v in the by-parts formula
        u_swap = sin(k * x)
        v_swap = x
        du_swap = k * cos(k * x)
        F_swap = c * (u_swap * v_swap - integrate(v_swap * du_swap, x))
        swapped_uv = F_swap.subs(x, hi) - F_swap.subs(x, lo)

        v_wrong = -cos(k * x)
        F_v_base = u * v_wrong - integrate(v_wrong * du, x)
        F_v = c * F_v_base
        wrong_v_antiderivative = F_v.subs(x, hi) - F_v.subs(x, lo)

        # only_upper_bound: forgot to subtract lower bound
        only_upper = c * F.subs(x, hi)

        check_distinct(ans, swapped_uv, wrong_v_antiderivative, only_upper)

        candidates = [
            _typed(swapped_uv),
            _typed(wrong_v_antiderivative),
            _typed(only_upper),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        prompt = f"Find ∫ from x = π/{2 * k} to x = π/{k} of {_prose(f)} dx. Give your answer as an exact value in terms of pi."
        working = [
            f"u = x, dv = {_prose(sin(k * x))} dx  ⇒  du = dx, v = {_prose(-cos(k * x) / k)}",
            f"F(x) = {_prose(F)}",
            f"F(π/{k}) - F(π/{2 * k}) = {_prose(ans)}",
        ]

        return {
            "text": prompt,
            "latex": prompt,
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": latex_answer(ans),
            "distractors": distractors,
            "working": working,
        }

    elif difficulty == "hard":
        kind = "log"
        n = rng.randint(1, 3)
        m = rng.randint(n + 1, n + 4)
        c = rng.randint(1, 4)
        f = c * x * log(x)

        # By-parts: u = log(x), dv = x dx
        u = log(x)
        dv = x
        du = 1 / x
        v = x**2 / 2

        # Compute antiderivative
        integral_v_du = integrate(v * du, x)
        F = c * (u * v - integral_v_du)

        # Bounds
        lo = n
        hi = m
        check(lo >= 1 and hi >= 1)  # ensure log stays real

        # Answer
        ans = F.subs(x, hi) - F.subs(x, lo)

        # Independent checks
        check_equivalent(diff(F, x), f)
        check_equivalent(ans, integrate(f, (x, lo, hi)))

        # Misconceptions
        # swapped_uv: swapped the roles of u and v in the by-parts formula
        u_swap = x
        v_swap = log(x)
        du_swap = 1
        # For log as dv: antiderivative is x*log(x) - x
        F_swap = c * (u_swap * v_swap - integrate(v_swap * du_swap, x))
        swapped_uv = F_swap.subs(x, hi) - F_swap.subs(x, lo)

        # only_upper_bound: forgot to subtract lower bound
        only_upper = c * F.subs(x, hi)

        # wrong_v_antiderivative: forgot /2 when integrating x dx
        v_wrong = x**2
        F_v = c * (u * v_wrong - integrate(v_wrong * du, x))
        wrong_v_antiderivative = F_v.subs(x, hi) - F_v.subs(x, lo)

        check_distinct(ans, swapped_uv, only_upper, wrong_v_antiderivative)

        candidates = [
            _typed(swapped_uv),
            _typed(only_upper),
            _typed(wrong_v_antiderivative),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        prompt = f"Find ∫ from x = {n} to x = {m} of {_prose(f)} dx. Give your answer as an exact value."
        working = [
            f"u = ln(x), dv = {_prose(x)} dx  ⇒  du = 1/x dx, v = {_prose(x**2 / 2)}",
            f"F(x) = {_prose(F)}",
            f"F({m}) - F({n}) = {_prose(ans)}",
        ]

        return {
            "text": prompt,
            "latex": prompt,
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": latex_answer(ans),
            "distractors": distractors,
            "working": working,
        }


@generator(
    "use_partial_fractions_where_necessary_for_integration_in",
    level=17,
    presets={
        "easy": dict(difficulty="easy"),
        "medium": dict(difficulty="medium"),
        "hard": dict(difficulty="hard"),
    },
)
def use_partial_fractions_where_necessary_for_integration_in(rng, difficulty="easy"):
    """
    Integrate using partial fractions: definite integrals of rational functions.

    easy: 1/((x-p)*(x-q)), two distinct linear factors
    medium: x/((x-p)*(x-q)), two distinct linear factors with linear numerator
    hard: 1/((x-p)*(x-q)*(x-r)), three distinct linear factors
    """

    if difficulty == "easy":
        kind = "two_factor"
        p = rng.randint(1, 5)
        d = rng.randint(2, 5)
        q = p + d
        f = 1 / ((x - p) * (x - q))

        # Partial fractions
        A = Rational(1, p - q)
        B = -A
        F = A * log(x - p) + B * log(x - q)

        # Bounds: both > q; resample if F(lo) = 0
        while True:
            o1 = rng.randint(1, 4)
            o2 = rng.randint(1, 4)
            lo = q + o1
            hi = q + o1 + o2

            # Verify bounds constraint
            check(lo > max(p, q))
            check(hi > max(p, q))

            # If F(lo) is zero, resample
            if not symbolically_equal(F.subs(x, lo), 0):
                break

        # Answer
        ans = F.subs(x, hi) - F.subs(x, lo)

        # Independent checks
        check_equivalent(diff(F, x), f)
        check_equivalent(ans, integrate(f, (x, lo, hi)))

        # Misconceptions
        # For easy (constant numerator): A = -B, so sign_error would equal swapped_residues.
        # Use different misconceptions: wrong_both_same_sign (used A for both A and B)
        F_swap = B * log(x - p) + A * log(x - q)
        swapped_residues = F_swap.subs(x, hi) - F_swap.subs(x, lo)

        # Use A for both residues (common mistake: forget to negate one)
        F_both_same = A * log(x - p) + A * log(x - q)
        wrong_both_same_sign = F_both_same.subs(x, hi) - F_both_same.subs(x, lo)

        only_upper_bound = F.subs(x, hi)
        check_distinct(ans, swapped_residues, wrong_both_same_sign, only_upper_bound)

        candidates = [
            _typed(swapped_residues),
            _typed(wrong_both_same_sign),
            _typed(only_upper_bound),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        prompt = f"Find ∫ from x = {lo} to x = {hi} of 1/((x - {p})·(x - {q})) dx. Give your answer as an exact value."
        working = [f"A = {A}, B = {B}", f"F(x) = {_prose(F)}", f"F({hi}) - F({lo}) = {_prose(ans)}"]

        return {
            "text": prompt,
            "latex": prompt,
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": latex_answer(ans),
            "distractors": distractors,
            "working": working,
        }

    elif difficulty == "medium":
        kind = "linear_numerator"
        p = rng.randint(1, 5)
        d = rng.randint(2, 5)
        q = p + d
        f = x / ((x - p) * (x - q))

        # Partial fractions: x = A*(x-q) + B*(x-p)
        # x = p: p = A*(p-q) => A = p/(p-q)
        # x = q: q = B*(q-p) => B = q/(q-p)
        A = Rational(p, p - q)
        B = Rational(q, q - p)
        F = A * log(x - p) + B * log(x - q)

        # Verify residue solution
        check_equivalent(diff(F, x), f)

        # Bounds; resample if F(lo) = 0
        while True:
            o1 = rng.randint(1, 4)
            o2 = rng.randint(1, 4)
            lo = q + o1
            hi = q + o1 + o2

            check(lo > max(p, q))
            check(hi > max(p, q))

            # If F(lo) is zero, resample
            if not symbolically_equal(F.subs(x, lo), 0):
                break

        # Answer
        ans = F.subs(x, hi) - F.subs(x, lo)

        # Independent check
        check_equivalent(ans, integrate(f, (x, lo, hi)))

        # Misconceptions
        # For medium (linear numerator): A = -B (same issue as easy)
        # Use wrong_both_same_sign instead of sign_error to avoid collision
        F_swap = B * log(x - p) + A * log(x - q)
        swapped_residues = F_swap.subs(x, hi) - F_swap.subs(x, lo)

        # Use A for both residues (common mistake)
        F_both_same = A * log(x - p) + A * log(x - q)
        wrong_both_same_sign = F_both_same.subs(x, hi) - F_both_same.subs(x, lo)

        only_upper_bound = F.subs(x, hi)
        check_distinct(ans, swapped_residues, wrong_both_same_sign, only_upper_bound)

        candidates = [
            _typed(swapped_residues),
            _typed(wrong_both_same_sign),
            _typed(only_upper_bound),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        prompt = f"Find ∫ from x = {lo} to x = {hi} of {_prose(x)}/((x - {p})·(x - {q})) dx. Give your answer as an exact value."
        working = [f"A = {A}, B = {B}", f"F(x) = {_prose(F)}", f"F({hi}) - F({lo}) = {_prose(ans)}"]

        return {
            "text": prompt,
            "latex": prompt,
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": latex_answer(ans),
            "distractors": distractors,
            "working": working,
        }

    elif difficulty == "hard":
        kind = "three_factor"
        # Resample parameters if misconceptions collide
        max_attempts = 10
        for attempt in range(max_attempts):
            p = rng.randint(1, 4)
            d1 = rng.randint(2, 4)
            d2 = rng.randint(2, 4)
            q = p + d1
            r = q + d2
            f = 1 / ((x - p) * (x - q) * (x - r))

            # Partial fractions: 1 = A*(x-q)*(x-r) + B*(x-p)*(x-r) + C*(x-p)*(x-q)
            # x = p: 1 = A*(p-q)*(p-r) => A = 1/((p-q)*(p-r))
            # x = q: 1 = B*(q-p)*(q-r) => B = 1/((q-p)*(q-r))
            # x = r: 1 = C*(r-p)*(r-q) => C = 1/((r-p)*(r-q))
            A = Rational(1, (p - q) * (p - r))
            B = Rational(1, (q - p) * (q - r))
            C = Rational(1, (r - p) * (r - q))
            F = A * log(x - p) + B * log(x - q) + C * log(x - r)

            # Verify residue solution
            check_equivalent(diff(F, x), f)

            # Bounds; resample if F(lo) = 0
            bounds_ok = False
            for _ in range(5):
                o1 = rng.randint(1, 3)
                o2 = rng.randint(1, 3)
                lo = r + o1
                hi = r + o1 + o2

                check(lo > max(p, q, r))
                check(hi > max(p, q, r))

                # If F(lo) is not zero, bounds are valid
                if not symbolically_equal(F.subs(x, lo), 0):
                    bounds_ok = True
                    break

            if not bounds_ok:
                continue

            # Answer
            ans = F.subs(x, hi) - F.subs(x, lo)

            # Independent check
            check_equivalent(ans, integrate(f, (x, lo, hi)))

            # Misconceptions
            F_swap = C * log(x - p) + B * log(x - q) + A * log(x - r)
            swapped_residues = F_swap.subs(x, hi) - F_swap.subs(x, lo)

            F_both_same = A * log(x - p) + A * log(x - q) + A * log(x - r)
            wrong_all_same_coeff = F_both_same.subs(x, hi) - F_both_same.subs(x, lo)

            only_upper_bound = F.subs(x, hi)

            # Try to verify misconceptions are distinct
            try:
                check_distinct(ans, swapped_residues, wrong_all_same_coeff, only_upper_bound)
            except:
                # Misconceptions collided, resample parameters
                if attempt == max_attempts - 1:
                    raise
                continue

            # Success! Generate the question
            candidates = [
                _typed(swapped_residues),
                _typed(wrong_all_same_coeff),
                _typed(only_upper_bound),
            ]
            distractors = pick_distractors(_typed(ans), candidates)

            prompt = f"Find ∫ from x = {lo} to x = {hi} of 1/((x - {p})·(x - {q})·(x - {r})) dx. Give your answer as an exact value."
            working = [f"A = {A}, B = {B}, C = {C}", f"F(x) = {_prose(F)}", f"F({hi}) - F({lo}) = {_prose(ans)}"]

            return {
                "text": prompt,
                "latex": prompt,
                "answer": str(ans),
                "answer_text": _typed(ans),
                "answer_latex": latex_answer(ans),
                "distractors": distractors,
                "working": working,
            }
