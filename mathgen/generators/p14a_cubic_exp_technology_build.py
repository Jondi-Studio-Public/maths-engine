"""Generates `p14a_solve_cubic_technology` and `p14a_solve_exp_technology` questions: solve a cubic or an
exponential equation numerically using technology.

Two senior skills, both `answer: number` (decimal, always to 3 decimal places), both
prerequisite `function-notation`, both `capability_gates: [technology]`:

- ``solve_cubic_technology`` (ACMMM019, senior-solve-cubic-equations-using-technology-and-
  algebraically-in): every tier names "the largest real root" -- well-defined at every tier
  since a real cubic always has at least one real root. Distinct from the shipping neighbour
  ``factorise_cubic_linear_factor`` (ACMMM018), whose own canonical answer is ALWAYS a factored
  (x - p)(x^2 + qx + r) expression, never a root and never the solved equation -- this skill's
  answer is always a single numeric root value, never a factorisation.

- ``solve_exp_technology`` (ACMMM067, senior-solve-equations-involving-exponential-functions-
  using-technology-and): base e, and EVERY tier carries an additive constant `+ c` on the
  exponential term before it can be isolated -- the shipping neighbour
  ``solve_indices_with_logarithms`` (a^x = b / c*a^x = b / a^(x+m) = b, never an additive
  constant) never has that extra isolation step at any tier.

Neither generator calls `sympy.solve`/`sympy.nsolve`.
"""

from __future__ import annotations

import math
from fractions import Fraction

from functools import cache


@cache
def _np():
    import numpy

    return numpy


from ..core import (
    check,
    check_pairwise_distinct,
    fmt_signed,
    generator,
    join_terms,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
)


def _display(value) -> str:
    """Display any Fraction/int/float root at EXACTLY 3 decimal places.

    `Fraction(value)` on a Python float (numpy.float64 included -- it subclasses
    `float`) is the float's own exact binary value, not an approximation, so
    rounding it with `round_half_up` reproduces ordinary decimal rounding to
    3 d.p.; on an exact Fraction/int input it is simply exact. One formatting
    route for every tier of both skills, following a uniform display
    convention.
    """
    return stated_precision_string(round_half_up(Fraction(value), 3), 3)


def _floats_distinct(*values, tol=0.01, msg="values not sufficiently distinct"):
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            check(abs(values[i] - values[j]) > tol, msg)


# =============================================================================
# Skill A: solve_cubic_technology (ACMMM019)
# =============================================================================


def _verify_largest_real_root(coeffs, claimed_root, tol=1e-4) -> None:
    """Independent check for the 'largest real root' tiers (easy/medium/hard).

    Two legs: (1) the CUBIC RESIDUAL at the claimed root, evaluated with plain
    float arithmetic (never sympy); (2) an independent re-derivation of "the
    largest real root" via `numpy.roots` on the coefficient array, filtering to
    real parts (imaginary part within a numeric tolerance of zero) and taking
    the max.

    For easy/medium, construction never calls numpy at all (easy expands three chosen integer roots; medium
    multiplies a linear and an irreducible quadratic factor), so BOTH legs here are a genuinely different
    route from construction. For hard, construction already used `numpy.roots` to find the root -- there leg
    (2) is not independent of construction, and leg (1), the residual check, is the one genuinely independent
    leg.
    """
    a, b, c, d = (float(x) for x in coeffs)
    x = float(claimed_root)
    residual = a * x**3 + b * x**2 + c * x + d
    check(abs(residual) < 1e-6, f"cubic residual {residual} not near zero for claimed root {x}")
    roots = _np().roots([a, b, c, d])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    check(len(real_roots) >= 1, "numpy.roots found no real root")
    largest = max(real_roots)
    check(abs(largest - x) < tol, f"numpy's largest real root {largest} != claimed {x}")


def _verify_domain_root(coeffs, claimed_root, lo, hi, tol=1e-4) -> None:
    """Independent check for the exam tier: the cubic residual, plus an independent re-derivation via
    `numpy.roots` that EXACTLY ONE real root lies in the open interval (lo, hi) and it matches the claimed
    root.
    """
    a, b, c, d = (float(x) for x in coeffs)
    x = float(claimed_root)
    residual = a * x**3 + b * x**2 + c * x + d
    check(abs(residual) < 1e-6, f"cubic residual {residual} not near zero for claimed root {x}")
    roots = _np().roots([a, b, c, d])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    in_domain = [r for r in real_roots if lo < r < hi]
    check(len(in_domain) == 1, f"expected exactly one root in ({lo}, {hi}), got {in_domain}")
    check(abs(in_domain[0] - x) < tol, f"domain root {in_domain[0]} != claimed {x}")


_KEY_POOL = [i for i in range(-10, 13) if i != 0]  # -10..-1, 1..12 -- wide, nonzero


def _build_easy_three_integer_roots(rng) -> dict:
    resamples = 0
    while True:
        key = rng.choice(_KEY_POOL)
        lower_pool = [v for v in range(-14, key) if v != 0]
        if len(lower_pool) < 2:
            resamples += 1
            continue
        other1, other2 = rng.sample(lower_pool, 2)
        break
    r1, r2, r3 = key, other1, other2
    b = -(r1 + r2 + r3)
    c = r1 * r2 + r1 * r3 + r2 * r3
    d = -(r1 * r2 * r3)
    coeffs = (1, b, c, d)

    largest = max(r1, r2, r3)
    others = [v for v in (r1, r2, r3) if v != largest]
    check_pairwise_distinct(largest, *others, msg="the three integer roots must be pairwise distinct")

    _verify_largest_real_root(coeffs, largest)

    cubic_text = join_terms([(1, "x^3"), (b, "x^2"), (c, "x"), (d, "")])
    answer_text = _display(largest)
    candidates = [_display(v) for v in others]

    text = f"{cubic_text} = 0 has three integer roots. Find the largest one, to 3 decimal places."
    working = [
        f"The three integer roots are {r1}, {r2}, {r3}.",
        f"The largest is {largest}.",
        f"To 3 decimal places: {answer_text}.",
    ]
    return dict(
        text=text,
        answer=float(largest),
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates, n=4),
    )


_MEDIUM_A_CHOICES = [2, 3, 4]


def _build_medium_rational_root_theorem(rng) -> dict:
    resamples = 0
    while True:
        q = rng.choice(_MEDIUM_A_CHOICES)
        p = rng.choice([v for v in range(-9, 10) if v != 0 and v % q != 0])
        if math.gcd(abs(p), q) != 1:
            resamples += 1
            continue
        m = rng.randint(-4, 4)
        n = rng.randint(1, 8)
        if m * m - 4 * n >= 0:
            resamples += 1
            continue
        A = q
        B = q * m - p
        C = q * n - p * m
        D = -p * n
        coeffs = (A, B, C, D)
        root_exact = Fraction(p, q)
        # Direct verification the quadratic factor has no real roots (never
        # just assumed from the m/n draw above).
        discriminant = m * m - 4 * n
        if discriminant >= 0:
            resamples += 1
            continue
        roots = _np().roots([float(A), float(B), float(C), float(D)])
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
        if len(real_roots) != 1:
            resamples += 1
            continue
        if not rounds_unambiguously(float(root_exact), 3):
            resamples += 1
            continue
        break

    _verify_largest_real_root(coeffs, root_exact)

    # Distractors: (1) misapplying the rational root theorem by testing only
    # integer divisors of the constant term, ignoring the leading coefficient
    # -- the numerator p treated as if it were the whole (integer) root;
    # (2) the negated correct root.
    wrong_integer_root = Fraction(p, 1)
    negated_root = -root_exact
    check_pairwise_distinct(
        root_exact,
        wrong_integer_root,
        negated_root,
        msg="medium-tier distractors collide with the answer or each other",
    )

    answer_text = _display(root_exact)
    candidates = [_display(wrong_integer_root), _display(negated_root)]

    A3, B2, C1, D0 = coeffs
    cubic_text = join_terms([(A3, "x^3"), (B2, "x^2"), (C1, "x"), (D0, "")])
    text = f"{cubic_text} = 0. Find the largest real root, to 3 decimal places."
    linear_factor_text = join_terms([(q, "x"), (-p, "")])
    quadratic_factor_text = join_terms([(1, "x^2"), (m, "x"), (n, "")])
    working = [
        f"({linear_factor_text})({quadratic_factor_text}) = 0 (the quadratic factor has discriminant "
        f"{discriminant} < 0, so no real roots).",
        f"The only real root is x = {p}/{q}.",
        f"To 3 decimal places: {answer_text}.",
    ]
    return dict(
        text=text,
        answer=float(root_exact),
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates, n=4),
    )


def _integer_divisors(n: int) -> list[int]:
    n = abs(n)
    divs = [k for k in range(1, n + 1) if n % k == 0]
    return divs + [-k for k in divs]


def _build_hard_technology_irrational(rng) -> dict:
    resamples = 0
    while True:
        b = rng.randint(-8, 8)
        c = rng.randint(-8, 8)
        d = rng.choice([v for v in range(-8, 9) if v != 0])
        has_rational_root = any((k**3 + b * k**2 + c * k + d) == 0 for k in _integer_divisors(d))
        if has_rational_root:
            resamples += 1
            continue
        coeffs = (1.0, float(b), float(c), float(d))
        roots = _np().roots(coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
        if len(real_roots) < 1:
            resamples += 1
            continue
        root_exact = max(real_roots)
        if not rounds_unambiguously(root_exact, 3):
            resamples += 1
            continue

        real_roots_sorted = sorted(real_roots)
        if len(real_roots_sorted) == 3:
            candidates_val = [real_roots_sorted[1], real_roots_sorted[0]]
        else:
            # Only one real root: two plausible arithmetic-slip candidates,
            # each re-computed via the SAME numeric route (numpy.roots) on a
            # cubic with one transcribed sign error -- a genuinely different
            # construction from the real one, confirmed distinct below.
            sign_error_c = _np().roots([1.0, float(b), float(-c), float(d)])
            sign_error_b = _np().roots([1.0, float(-b), float(c), float(d)])
            cand1_roots = [r.real for r in sign_error_c if abs(r.imag) < 1e-6]
            cand2_roots = [r.real for r in sign_error_b if abs(r.imag) < 1e-6]
            candidates_val = [max(cand1_roots), max(cand2_roots)]

        all_values = [root_exact, *candidates_val]
        all_distinct = all(
            abs(all_values[i] - all_values[j]) > 0.01
            for i in range(len(all_values))
            for j in range(i + 1, len(all_values))
        )
        if not all_distinct:
            resamples += 1
            continue
        break

    _verify_largest_real_root(coeffs, root_exact)
    _floats_distinct(root_exact, *candidates_val, tol=0.01, msg="hard-tier distractor too close to the answer")

    answer_text = _display(root_exact)
    candidates = [_display(v) for v in candidates_val]

    cubic_text = join_terms([(1, "x^3"), (b, "x^2"), (c, "x"), (d, "")])
    text = f"{cubic_text} = 0. Use technology to find the largest real root, to 3 decimal places."
    working = [
        "No integer divisor of the constant term is a root, so there is no rational root here.",
        f"Solving numerically, the largest real root is {answer_text}.",
    ]
    return dict(
        text=text,
        answer=float(root_exact),
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates, n=4),
    )


def _build_exam_physical_domain(rng) -> dict:
    """An open box's volume V(x) = x(p - 2x)(q - 2x) is a hump on its full physically-valid domain (0, min(p,
    q)/2): it rises to an interior local maximum and falls back to zero, so a target volume strictly between 0
    and that maximum is generically reached at TWO points inside the full valid domain, not one -- resampling
    for "exactly one root in (0, min(p,q)/2)" would loop forever. Structural uniqueness instead comes from a
    NARROWER stated domain (0, c) with c strictly less than the local-maximum location: V is strictly
    increasing throughout (0, c) (it has not yet turned over), so V(x) = V0 has exactly one solution there by
    construction, for any V0 in (0, V(c)) -- proved directly below via numpy.roots' count, not merely assumed
    from the calculus.
    """
    shape = rng.choice(["standard", "rearranged", "corner_first"])
    resamples = 0
    while True:
        p = rng.randint(12, 40)
        q = rng.randint(12, 40)
        S, P = p + q, p * q
        # The smaller critical point of V(x) = 4x^3 - 2Sx^2 + Px is where
        # V'(x) = 12x^2 - 4Sx + P first hits zero -- the local MAXIMUM (V'
        # goes + -> - there); S^2 - 3P = p^2 - pq + q^2 = (p - q/2)^2 +
        # 3q^2/4 > 0 always, so this is always real.
        x_crit = (S - math.sqrt(S * S - 3 * P)) / 6
        c = math.floor(x_crit * 2) / 2 - 0.5  # a clean 0.5-cm bound, safely below x_crit
        if c <= 1.0:
            resamples += 1
            continue
        x0 = rng.uniform(0.2 * c, 0.7 * c)
        v0 = round(x0 * (p - 2 * x0) * (q - 2 * x0))
        if v0 <= 0:
            resamples += 1
            continue
        # 4x^3 - 2(p+q)x^2 + pq x - v0 = 0
        coeffs = (4.0, -2.0 * S, float(P), -float(v0))
        roots = _np().roots(coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
        in_domain = [r for r in real_roots if 0 < r < c]
        if len(in_domain) != 1:
            resamples += 1
            continue
        root_exact = in_domain[0]
        if not rounds_unambiguously(root_exact, 3):
            resamples += 1
            continue
        out_of_domain = [r for r in real_roots if r not in in_domain]
        if out_of_domain:
            candidates_val = out_of_domain
        else:
            # Only one real root at all (the domain one): the domain bound
            # itself is a genuine common mistake -- stating the stated limit
            # as if it were the answer.
            candidates_val = [c]
        if any(abs(root_exact - v) <= 0.01 for v in candidates_val):
            resamples += 1
            continue
        break

    _verify_domain_root(coeffs, root_exact, 0, c)
    _floats_distinct(root_exact, *candidates_val, tol=0.01, msg="exam-tier distractor too close to the answer")

    answer_text = _display(root_exact)
    candidates = [_display(v) for v in candidates_val]

    c_str = _display(c) if not float(c).is_integer() else str(int(c))
    if shape == "standard":
        text = (
            f"An open box is made by cutting equal squares of side length x cm from each corner "
            f"of a {p} cm by {q} cm rectangular sheet of metal and folding up the sides. The volume "
            f"of the resulting box is {v0} cm^3. It is known that 0 < x < {c_str}. Find the value "
            "of x, to 3 decimal places."
        )
    elif shape == "rearranged":
        text = (
            f"A rectangular sheet of metal, {p} cm by {q} cm, has equal squares of side length x "
            "cm cut from each corner; the sides are then folded up to form an open box. If the "
            f"box's volume is {v0} cm^3 and 0 < x < {c_str}, find x, to 3 decimal places."
        )
    else:  # corner_first
        text = (
            f"Squares of side length x cm are cut from each corner of a {p} cm by {q} cm sheet "
            f"of metal, and the flaps are folded up to make an open box with volume {v0} cm^3. "
            f"Given that 0 < x < {c_str}, find the value of x, to 3 decimal places."
        )
    working = [
        f"V(x) = x({p} - 2x)({q} - 2x) = {v0} has exactly one solution in 0 < x < {c_str}.",
        f"Solving numerically: x = {answer_text}.",
    ]
    return dict(
        text=text,
        answer=float(root_exact),
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates, n=4),
    )


@generator(
    "p14a_solve_cubic_technology",
    level=12,
    presets={
        "easy": dict(variant="three_integer_roots"),
        "medium": dict(variant="rational_root_theorem"),
        "hard": dict(variant="technology_irrational"),
        "exam": dict(variant="physical_domain"),
    },
)
def p14a_solve_cubic_technology(rng, variant="three_integer_roots"):
    """Solve cubic equations using technology, and algebraically in cases
    where a linear factor is easily obtained. Every tier names 'the largest
    real root' (except the exam tier, which names the physically valid root
    in a stated domain) -- a real cubic always has at least one real root, so
    this naming is always well-defined."""
    if variant == "three_integer_roots":
        return _build_easy_three_integer_roots(rng)
    if variant == "rational_root_theorem":
        return _build_medium_rational_root_theorem(rng)
    if variant == "technology_irrational":
        return _build_hard_technology_irrational(rng)
    if variant == "physical_domain":
        return _build_exam_physical_domain(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# =============================================================================
# Skill B: solve_exp_technology (ACMMM067)
# =============================================================================


def _dynamic_tol(derivative_estimate: float) -> float:
    """A residual tolerance tied to the 3-decimal-place display: a claimed x
    is only ever wrong by up to half the last displayed place (0.0005), and
    that error propagates through the exponential's own local derivative
    (first-order: delta_lhs ~= f'(x) * delta_x). 6e-4 is a 20% safety margin
    over the 5e-4 half-unit, plus a small additive floor for near-zero
    derivatives."""
    return abs(derivative_estimate) * 6e-4 + 1e-9


def _verify_exp_additive(A, k, c, d, claimed_x) -> None:
    lhs = A * math.exp(k * claimed_x) + c
    tol = _dynamic_tol(A * k * math.exp(k * claimed_x))
    check(abs(lhs - d) < tol, f"additive-isolate residual {lhs - d} exceeds tolerance {tol}")


def _verify_exp_linear_exponent(A, k, m, c, d, claimed_x) -> None:
    lhs = A * math.exp(k * claimed_x + m) + c
    tol = _dynamic_tol(A * k * math.exp(k * claimed_x + m))
    check(abs(lhs - d) < tol, f"linear-exponent residual {lhs - d} exceeds tolerance {tol}")


def _verify_hidden_quadratic(Aq, Bq, Cq, claimed_x) -> None:
    lhs = Aq * math.exp(2 * claimed_x) + Bq * math.exp(claimed_x) + Cq
    derivative = 2 * Aq * math.exp(2 * claimed_x) + Bq * math.exp(claimed_x)
    tol = _dynamic_tol(derivative)
    check(abs(lhs) < tol, f"hidden-quadratic residual {lhs} exceeds tolerance {tol}")


def _build_easy_additive_isolate(rng) -> dict:
    resamples = 0
    while True:
        A = rng.randint(2, 9)
        k = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        c = rng.choice([v for v in range(-15, 16) if v != 0])
        T = rng.randint(5, 300)
        d = A * T + c
        x_exact = math.log(T) / k
        if rounds_unambiguously(x_exact, 3):
            break
        resamples += 1

    check(Fraction(d - c, A) == T, "construction did not exactly reproduce T")

    _verify_exp_additive(A, k, c, d, x_exact)

    exponent = fmt_signed(k, "x")
    sign = "+" if c > 0 else "-"
    lhs_text = f"{A} \u00d7 e^({exponent}) {sign} {abs(c)}"
    text = f"Solve {lhs_text} = {d} for x. Give your answer to 3 decimal places."
    answer_text = _display(x_exact)

    # Misconceptions: forgot to subtract c before dividing (uses d directly);
    # forgot to divide by A after subtracting c (uses d - c directly); sign
    # error on k (uses -k).
    candidates_val = []
    if d > 0:
        candidates_val.append(math.log(d) / k)
    if d - c > 0:
        candidates_val.append(math.log(d - c) / k)
    candidates_val.append(math.log(T) / (-k))
    candidates_val = [v for v in candidates_val if abs(v - x_exact) > 0.01]

    working = [
        f"{A}e^({exponent}) = {d} {'- ' if c > 0 else '+ '}{abs(c)} = {d - c}",
        f"e^({exponent}) = {T}",
        f"{exponent} = ln({T})",
        f"x = {answer_text}",
    ]
    return dict(
        text=text,
        answer=float(x_exact),
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [_display(v) for v in candidates_val], n=4),
    )


def _build_medium_linear_exponent_additive(rng) -> dict:
    resamples = 0
    while True:
        A = rng.randint(2, 9)
        k = rng.choice([-3, -2, -1, 1, 2, 3])
        m = rng.choice([v for v in range(-6, 7) if v != 0])
        c = rng.choice([v for v in range(-15, 16) if v != 0])
        T = rng.randint(5, 300)
        d = A * T + c
        x_exact = (math.log(T) - m) / k
        if rounds_unambiguously(x_exact, 3):
            break
        resamples += 1

    _verify_exp_linear_exponent(A, k, m, c, d, x_exact)

    exponent_inner = join_terms([(k, "x"), (m, "")])
    sign = "+" if c > 0 else "-"
    lhs_text = f"{A} \u00d7 e^({exponent_inner}) {sign} {abs(c)}"
    text = f"Solve {lhs_text} = {d} for x. Give your answer to 3 decimal places."
    answer_text = _display(x_exact)

    candidates_val = []
    # forgot c: uses d instead of T
    if d > 0:
        candidates_val.append((math.log(d) - m) / k)
    # forgot the m shift entirely
    candidates_val.append(math.log(T) / k)
    # sign error on m
    candidates_val.append((math.log(T) + m) / k)
    candidates_val = [v for v in candidates_val if abs(v - x_exact) > 0.01]

    working = [
        f"{A}e^({exponent_inner}) = {d} {'- ' if c > 0 else '+ '}{abs(c)} = {d - c}",
        f"e^({exponent_inner}) = {T}",
        f"{exponent_inner} = ln({T})",
        f"x = (ln({T}) - ({m})) / {k}",
        f"x = {answer_text}",
    ]
    return dict(
        text=text,
        answer=float(x_exact),
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [_display(v) for v in candidates_val], n=4),
    )


_U_POOL = list(range(1, 31))


def _build_hard_hidden_quadratic(rng) -> dict:
    shape = rng.choice(["standard", "rearranged", "solution_first"])
    resamples = 0
    while True:
        u1, u2 = rng.sample(_U_POOL, 2)
        s = rng.choice([1, 2, 3])
        Aq = s
        Bq = -s * (u1 + u2)
        Cq = s * u1 * u2
        x1, x2 = math.log(u1), math.log(u2)
        if abs(x1 - x2) <= 0.05:
            resamples += 1
            continue
        larger_u, smaller_u = max(u1, u2), min(u1, u2)
        larger_x, smaller_x = math.log(larger_u), math.log(smaller_u)
        if not rounds_unambiguously(larger_x, 3):
            resamples += 1
            continue
        break

    roots_u = _np().roots([float(Aq), float(Bq), float(Cq)])
    positive_u = [r.real for r in roots_u if abs(r.imag) < 1e-6 and r.real > 0]
    check(len(positive_u) == 2, f"expected exactly two positive u-roots, got {positive_u}")
    check(abs(max(positive_u) - larger_u) < 1e-4, "numpy's larger positive u-root disagrees with construction")

    _verify_hidden_quadratic(Aq, Bq, Cq, larger_x)

    eq_text = join_terms([(Aq, "e^(2x)"), (Bq, "e^x"), (Cq, "")])
    # Every shape below states the SAME quadratic-in-e^x equation; only the
    # surrounding clause structure differs, so `_verify_hidden_quadratic`
    # above (already run against Aq/Bq/Cq/larger_x) covers every shape.
    if shape == "standard":
        text = f"{eq_text} = 0 has exactly two solutions for x. Give the larger solution for x, to 3 decimal places."
    elif shape == "rearranged":
        text = (
            f"The equation {eq_text} = 0 has exactly two real solutions for x; give the larger "
            "one, to 3 decimal places."
        )
    else:  # solution_first
        text = f"Give the larger of the two real solutions for x to {eq_text} = 0, to 3 decimal places."
    answer_text = _display(larger_x)

    candidates_val = [smaller_x]
    # Misconception: treats the equation as linear in e^x, ignoring the
    # squared term -- solves Bq*u + Cq = 0 for u.
    if Bq != 0:
        u_linear = -Cq / Bq
        if u_linear > 0:
            candidates_val.append(math.log(u_linear))
    candidates_val = [v for v in candidates_val if abs(v - larger_x) > 0.01]

    working = [
        f"Let u = e^x: {Aq}u^2 {'+' if Bq >= 0 else '-'} {abs(Bq)}u {'+' if Cq >= 0 else '-'} {abs(Cq)} = 0",
        f"u = {smaller_u} or u = {larger_u} (both positive, since e^x > 0 always).",
        f"x = ln({larger_u}) = {answer_text} (the larger solution).",
    ]
    return dict(
        text=text,
        answer=float(larger_x),
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [_display(v) for v in candidates_val], n=4),
    )


def _bisection_root(f, lo, hi, iters=80):
    flo, fhi = f(lo), f(hi)
    check(flo * fhi < 0, "bisection requires a sign change at the endpoints")
    for _ in range(iters):
        mid = (lo + hi) / 2
        fmid = f(mid)
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return (lo + hi) / 2


def _newton_root(f, fprime, x0, iters=50):
    x = x0
    for _ in range(iters):
        x = x - f(x) / fprime(x)
    return x


def _build_exam_transcendental(rng) -> dict:
    shape = rng.choice(["standard", "rearranged", "solution_first"])
    resamples = 0
    while True:
        b = rng.randint(1, 8)
        x0 = rng.uniform(0.3, 3.7)
        frac = x0 - math.floor(x0)
        if frac < 0.15 or frac > 0.85:
            resamples += 1
            continue
        a = round(math.exp(x0) + b * x0)
        f = lambda x: math.exp(x) - a + b * x  # noqa: E731
        lo, hi = math.floor(x0), math.floor(x0) + 1
        if f(lo) * f(hi) >= 0:
            resamples += 1
            continue
        root_bisect = _bisection_root(f, lo, hi)
        if not rounds_unambiguously(root_bisect, 3):
            resamples += 1
            continue
        break

    fprime = lambda x: math.exp(x) + b  # noqa: E731
    root_newton = _newton_root(f, fprime, hi, iters=50)
    check(abs(root_bisect - root_newton) < 1e-6, "bisection and Newton's method disagree")

    tol = _dynamic_tol(fprime(root_bisect))
    check(abs(f(root_bisect)) < tol, "transcendental residual exceeds tolerance")

    answer_text = _display(root_bisect)
    bx_text = fmt_signed(b, "x")  # b is always a positive int here: "x" or "8x", never "1x"
    # Every shape below states the SAME equation e^x = {a} - {bx_text} (the
    # "rearranged" shape is algebraically identical, just isolated to the
    # left instead of subtracted on the right); `a`, `bx_text`, `lo`, `hi`
    # are all drawn once above and simply reused, so the "transcendental
    # residual exceeds tolerance" check above always checked the same
    # underlying equation regardless of which surface text is chosen below.
    if shape == "standard":
        text = (
            f"e^x = {a} - {bx_text} has exactly one solution for x in the interval [{lo}, {hi}]. "
            "Use technology to find it, to 3 decimal places."
        )
    elif shape == "rearranged":
        text = (
            f"e^x + {bx_text} = {a} has exactly one solution for x in the interval [{lo}, {hi}]. "
            "Use technology to find it, to 3 decimal places."
        )
    else:  # solution_first
        text = (
            f"There is exactly one solution for x in the interval [{lo}, {hi}] to "
            f"e^x = {a} - {bx_text}. Use technology to find it, to 3 decimal places."
        )

    candidates_val = []
    # Misconception: ignores the linear term entirely (treats it as e^x = a).
    if a > 0:
        candidates_val.append(math.log(a))
    # Misconception: uses the small-x linear approximation e^x ~= 1 + x.
    lin_denom = 1 + b
    if lin_denom != 0:
        candidates_val.append((a - 1) / lin_denom)
    candidates_val = [v for v in candidates_val if abs(v - root_bisect) > 0.01]

    working = [
        f"f(x) = e^x - {a} + {bx_text}",
        f"f({lo}) and f({hi}) have opposite signs, so exactly one root lies in [{lo}, {hi}].",
        f"Solving numerically: x = {answer_text}.",
    ]
    return dict(
        text=text,
        answer=float(root_bisect),
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [_display(v) for v in candidates_val], n=4),
    )


@generator(
    "p14a_solve_exp_technology",
    level=12,
    presets={
        "easy": dict(variant="additive_isolate"),
        "medium": dict(variant="linear_exponent_additive"),
        "hard": dict(variant="hidden_quadratic"),
        "exam": dict(variant="transcendental"),
    },
)
def p14a_solve_exp_technology(rng, variant="additive_isolate"):
    """Solve equations involving exponential functions using technology, and
    algebraically in simple cases. Base e throughout; every tier carries an
    additive constant on the exponential term before it can be isolated."""
    if variant == "additive_isolate":
        return _build_easy_additive_isolate(rng)
    if variant == "linear_exponent_additive":
        return _build_medium_linear_exponent_additive(rng)
    if variant == "hidden_quadratic":
        return _build_hard_hidden_quadratic(rng)
    if variant == "transcendental":
        return _build_exam_transcendental(rng)
    raise ValueError(f"unknown variant: {variant!r}")
