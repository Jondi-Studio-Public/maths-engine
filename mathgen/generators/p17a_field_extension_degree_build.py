"""Generates `p17a_field_extension_degree` questions: the degree [Q(alpha):Q] of a field extension, found by
deciding reducibility over Q.

Every polynomial prompt now instead names alpha by REAL-ROOT ORDER -- "the largest real root of p(x) = 0" /
"the smallest real root of p(x) = 0" -- drawn independently of which factor alpha actually belongs to, so the
naming phrase never leaks the key. The words "rational" and "irrational" never appear in a polynomial prompt.

**The root must be provably located, never just claimed.** Every branch
either (a) supplies a p all of whose real roots share the SAME degree (p
irreducible, or a product of irreducible factors that are ALL the same
degree), in which case naming either extreme is honest regardless of its
exact value, or (b) forces one factor's root(s) OUTSIDE the Cauchy bound of
the other factor -- `_cauchy_bound`, `1 + max(|c_i|)` over the non-leading
coefficients of a monic polynomial strictly bounds every one of its real (or
complex) roots -- so "the largest/smallest real root" is a CERTIFIED,
integer-arithmetic-only separation between the two factors' roots, without
ever computing an irrational root's numeric value. Every branch also proves
its polynomial has at least one real root: odd-degree monic polynomials
always do (IVT, free); the irreducible even-degree (quartic/biquadratic)
branches force a negative constant term, so p(0) < 0 < p(x) for large |x|
certifies a real root exists by IVT.

**Construct-to-target**, unchanged from 1602: every tier draws the target
degree class first (an equal-weight `rng.choice` over that tier's degree
set) and then builds a polynomial (or a/b pair) that REALISES that class.

Ladder (per-tier key values and reasoning DECISION are unchanged from the
original 1602 build; only how alpha is NAMED changed):

- easy (cubics, {1, 2, 3} even thirds): an irreducible cubic (rational-root
  test finds nothing -- degree 3, real root guaranteed by IVT since a cubic
  is odd degree); a rational root forced OUTSIDE the Cauchy bound of an
  irreducible quadratic factor, alpha named as the extreme belonging to the
  quadratic factor (degree 2); the same construction, alpha named as the
  extreme belonging to the linear factor (degree 1). The DECISION: does the
  rational-root test find a root at all, and which named root does the
  question actually ask about.
- medium (quartics, {2, 3, 4} even thirds): an Eisenstein-irreducible quartic
  with a negative constant term (degree 4, real root certified by IVT); a
  rational root forced outside the Cauchy bound of an irreducible cubic
  factor, alpha named as the extreme belonging to the cubic (degree 3); a
  product of two distinct irreducible quadratics, alpha named as either
  extreme (both factors' roots share degree 2). The DECISION: the
  rational-root test alone no longer settles irreducibility (a quartic can
  be a product of two quadratics with NO rational root), so the learner must
  also try a quadratic factorisation.
- hard (biquadratics x^4 + px^2 + q, {1, 2, 4} even thirds): an
  Eisenstein-irreducible biquadratic with a negative constant term (degree
  4); (x^2-a)(x^2-b) for distinct non-square positive a, b -- the
  rational-root test finds NOTHING (every root is irrational) yet the
  quartic is reducible into two quadratics, the standard disguise
  (degree 2); (x^2-r^2)(x^2-s) for an integer r forced so r^2 > s, alpha
  named as the extreme belonging to the r-factor (degree 1). The DECISION: a
  quartic with zero rational roots is not automatically irreducible -- the
  learner must look for the biquadratic quadratic-in-x^2 factorisation the
  rational-root test cannot see.
- exam ({2, 3, 4} weighted so no value exceeds ~40% of draws, measured at
  200 seeds): a genuinely new KIND of question, not a bigger version of
  hard's -- alpha = sqrt(a) + sqrt(b) or the direct compound-field framing
  [Q(sqrt a, sqrt b):Q], for distinct non-square positive integers a, b
  (degree 4 when a*b is not a perfect square, degree 2 when it is -- sqrt(ab)
  rational collapses the extension; these prompts state alpha EXPLICITLY as
  a number, never "a root of p", so the root-naming defect never applied to
  them), mixed with a bigger-number reprise of medium's rational-root-
  forced-outside-a-cubic's-Cauchy-bound case (degree 3) so the tier reaches
  three honest values. The DECISION: reducibility now has to be judged from
  TWO stated numbers with no polynomial to test roots of at all.
"""

from __future__ import annotations

import math

from mathgen.core import check, check_pairwise_distinct, generator, join_terms, latex_answer, phrase

# ---------------------------------------------------------------------------
# Shared arithmetic helpers (integer-only; no sympy in make(), per SS "Speed").
# ---------------------------------------------------------------------------


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _quadratic_irreducible(b: int, c: int) -> bool:
    """x^2 + bx + c, D = b^2 - 4c > 0 (real, distinct roots) and D not a
    perfect square -- the same discriminant test used elsewhere in this package."""
    D = b * b - 4 * c
    return D > 0 and not _is_perfect_square(D)


def _eval_poly(coeffs: list[int], x: int) -> int:
    """Horner evaluation; `coeffs` highest-degree first."""
    total = 0
    for c in coeffs:
        total = total * x + c
    return total


def _divisors(n: int) -> list[int]:
    n = abs(n)
    out = []
    for d in range(1, n + 1):
        if n % d == 0:
            out.append(d)
            out.append(-d)
    return out


def _has_rational_root(coeffs: list[int]) -> bool:
    """`coeffs` is a MONIC integer polynomial (leading coefficient 1); any
    rational root is then an integer divisor of the constant term."""
    check(coeffs[0] == 1, "rational-root test assumes a monic polynomial")
    const = coeffs[-1]
    if const == 0:
        return True
    for d in _divisors(const):
        if _eval_poly(coeffs, d) == 0:
            return True
    return False


def _poly_mul(a: list[int], b: list[int]) -> list[int]:
    """Convolve two coefficient lists (highest-degree first) -> their
    product's coefficients, highest-degree first."""
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _linear_factor_str(r: int) -> str:
    """The factor (x - r) as a student would write it, sign-collapsed via
    `join_terms` so a negative r renders as "(x + 4)", never "(x - -4)"."""
    return f"({join_terms([(1, 'x'), (-r, '')])})"


def _poly_expr_str(coeffs: list[int]) -> str:
    """coeffs highest degree first -> a `join_terms`-built display string,
    e.g. [1, 0, -2, 3] -> 'x^3 - 2x + 3'."""
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        var = "" if power == 0 else ("x" if power == 1 else f"x^{power}")
        terms.append((c, var))
    return join_terms(terms)


def _eisenstein_ok(coeffs: list[int], prime: int) -> bool:
    """`coeffs` monic, highest-degree first. Eisenstein's criterion at
    `prime`: every non-leading coefficient divisible by `prime`, and
    `prime^2` does NOT divide the constant term."""
    check(coeffs[0] == 1, "Eisenstein test assumes a monic polynomial")
    if any(c % prime != 0 for c in coeffs[1:]):
        return False
    return coeffs[-1] % (prime * prime) != 0


def _cauchy_bound(coeffs: list[int]) -> int:
    """`coeffs` monic, highest-degree first. Cauchy's bound: every real (or complex) root r of a monic polynomial
    satisfies |r| < 1 + max(|c_i|) over the non-leading coefficients."""
    check(coeffs[0] == 1, "Cauchy bound assumes a monic polynomial")
    return 1 + max(abs(c) for c in coeffs[1:])


_PRIMES = (2, 3, 5)

# ---------------------------------------------------------------------------
# Root-order naming, shared by every polynomial branch in every tier.
# "Largest"/"smallest" never states rationality; the branch's own
# construction (below) proves, via Cauchy's bound or via every real root of
# an irreducible factor sharing one degree, which factor the named extreme
# belongs to.
# ---------------------------------------------------------------------------

_ORDER_TEMPLATES = {
    "largest": (
        "Let alpha be the largest real root of p(x) = 0, where p(x) = {p_expr}. Find [Q(alpha) : Q].",
        "Given p(x) = {p_expr}, let alpha denote the largest real root of p(x) = 0. Determine [Q(alpha) : Q].",
    ),
    "smallest": (
        "Let alpha be the smallest real root of p(x) = 0, where p(x) = {p_expr}. Find [Q(alpha) : Q].",
        "Given p(x) = {p_expr}, let alpha denote the smallest real root of p(x) = 0. Determine [Q(alpha) : Q].",
    ),
}


def _order_text(rng, order: str, p_expr: str) -> str:
    return phrase(rng, _ORDER_TEMPLATES[order], p_expr=p_expr)


def _linear_vs_other_extremes(rng, other_coeffs: list[int], magnitude_slack: int = 4):
    """Draw an integer r with |r| forced to or beyond `other_coeffs`' Cauchy
    bound (monic, highest-degree first), so r is PROVABLY the unique global
    extreme on its own sign's side, and every real root of `other_coeffs` --
    however many there are, and whatever their exact irrational values --
    lies on the OPPOSITE extreme. Returns (r, bound, r_order, other_order)."""
    bound = _cauchy_bound(other_coeffs)
    magnitude = bound + rng.randint(0, magnitude_slack)
    sign = rng.choice((1, -1))
    r = sign * magnitude
    check(abs(r) >= bound, "r must be forced to or beyond the other factor's Cauchy bound")
    r_order = "largest" if sign > 0 else "smallest"
    other_order = "smallest" if sign > 0 else "largest"
    return r, bound, r_order, other_order


# ---------------------------------------------------------------------------
# easy: cubics, degree in {1, 2, 3}
# ---------------------------------------------------------------------------

_EASY_QUAD_B = list(range(-5, 6))
_EASY_QUAD_C = list(range(-5, 6))


def _irreducible_cubic(rng, lo: int, hi: int, tries: int = 200) -> list[int]:
    for _ in range(tries):
        coeffs = [1, rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)]
        if coeffs[-1] != 0 and not _has_rational_root(coeffs):
            return coeffs
    raise AssertionError("could not build an irreducible cubic")


def _mixed_cubic(rng, b_pool, c_pool, tries: int = 200):
    """(x - r) * (x^2 + bx + c), quadratic irreducible over Q, with r forced
    to or beyond the quadratic's Cauchy bound. Returns
    (coeffs, r, b, c, bound, r_order, other_order)."""
    for _ in range(tries):
        b = rng.choice(b_pool)
        c = rng.choice(c_pool)
        if not _quadratic_irreducible(b, c):
            continue
        r, bound, r_order, other_order = _linear_vs_other_extremes(rng, [1, b, c])
        coeffs = _poly_mul([1, -r], [1, b, c])
        check(coeffs[0] == 1, "leading coefficient must stay monic")
        check(not _has_rational_root([1, b, c]), "quadratic factor must have no rational root")
        return coeffs, r, b, c, bound, r_order, other_order
    raise AssertionError("could not build a mixed cubic with a Cauchy-separated root")


def p17a_easy(rng) -> dict:
    branch = rng.choice(("irreducible", "quadratic_factor_root", "linear_factor_root"))
    if branch == "irreducible":
        coeffs = _irreducible_cubic(rng, -6, 6)
        check(not _has_rational_root(coeffs), "easy irreducible branch must have no rational root")
        p_expr = _poly_expr_str(coeffs)
        order = rng.choice(("largest", "smallest"))
        text = _order_text(rng, order, p_expr)
        degree = 3
        other = (1, 2)
        working = [
            f"p(x) = {p_expr}: no integer divisor of the constant term {coeffs[-1]} is a root, "
            "so the rational-root test finds nothing and p(x) is irreducible over Q.",
            f"An odd-degree polynomial always has a real root, and every real root of an "
            f"irreducible cubic has degree 3, so the {order} real root of p(x) = 0 has degree 3.",
        ]
    else:
        coeffs, r, b, c, bound, r_order, other_order = _mixed_cubic(rng, _EASY_QUAD_B, _EASY_QUAD_C)
        check(_eval_poly(coeffs, r) == 0, "the forced root must actually be a root of the cubic")
        p_expr = _poly_expr_str(coeffs)
        quad_expr = _poly_expr_str([1, b, c])
        D = b * b - 4 * c
        factor_line = (
            f"p(x) = {p_expr} = {_linear_factor_str(r)}({quad_expr}); the quadratic factor has "
            f"discriminant D = {D}, not a perfect square, so it is irreducible over Q, and "
            f"Cauchy's bound puts both its roots strictly inside (-{bound}, {bound})."
        )
        if branch == "quadratic_factor_root":
            order = other_order
            text = _order_text(rng, order, p_expr)
            degree = 2
            other = (1, 3)
            working = [
                factor_line,
                f"alpha = {r} lies outside that range, so the {order} real root of p(x) = 0 "
                "belongs to the quadratic factor, giving [Q(alpha):Q] = 2.",
            ]
        else:
            order = r_order
            text = _order_text(rng, order, p_expr)
            degree = 1
            other = (2, 3)
            working = [
                factor_line,
                f"alpha = {r} lies outside that range and is itself a root of p(x), so the "
                f"{order} real root of p(x) = 0 is alpha = {r}, giving [Q(alpha):Q] = 1.",
            ]
    return _finish(rng, text, degree, other, working)


# ---------------------------------------------------------------------------
# medium: quartics, degree in {2, 3, 4}
# ---------------------------------------------------------------------------

_MEDIUM_QUAD_B = list(range(-6, 7))
_MEDIUM_QUAD_C = list(range(-6, 7))


def _eisenstein_general_quartic(rng, prime_pool, m_range, tries: int = 200):
    """Constant term is always forced negative so p(0) < 0 < p(x) for large
    |x| certifies a real root by IVT (an irreducible quartic need not have
    one otherwise, e.g. x^4 + 2x^2 + 2)."""
    for _ in range(tries):
        prime = rng.choice(prime_pool)
        c3 = prime * rng.randint(-3, 3)
        c2 = prime * rng.randint(-3, 3)
        c1 = prime * rng.randint(-3, 3)
        m = rng.randint(1, m_range)
        if m % prime == 0:
            continue
        c0 = -prime * m
        coeffs = [1, c3, c2, c1, c0]
        if _eisenstein_ok(coeffs, prime):
            return coeffs, prime
    raise AssertionError("could not build an Eisenstein-irreducible quartic")


def _cubic_vs_linear(rng, cubic_lo, cubic_hi, tries: int = 200):
    """(x - r) * cubic (cubic irreducible), r forced to or beyond the
    cubic's Cauchy bound. Returns (coeffs, r, cubic, bound, r_order,
    other_order)."""
    for _ in range(tries):
        cubic = _irreducible_cubic(rng, cubic_lo, cubic_hi)
        r, bound, r_order, other_order = _linear_vs_other_extremes(rng, cubic)
        coeffs = _poly_mul([1, -r], cubic)
        return coeffs, r, cubic, bound, r_order, other_order
    raise AssertionError("could not build a cubic-vs-linear Cauchy-separated quartic")


def _two_irreducible_quadratics(rng, b_pool, c_pool, tries: int = 200):
    for _ in range(tries):
        b1, c1 = rng.choice(b_pool), rng.choice(c_pool)
        b2, c2 = rng.choice(b_pool), rng.choice(c_pool)
        if (b1, c1) == (b2, c2):
            continue
        if not (_quadratic_irreducible(b1, c1) and _quadratic_irreducible(b2, c2)):
            continue
        coeffs = _poly_mul([1, b1, c1], [1, b2, c2])
        return coeffs, (b1, c1), (b2, c2)
    raise AssertionError("could not build a product of two distinct irreducible quadratics")


def p17a_medium(rng) -> dict:
    branch = rng.choice(("eisenstein", "cubic_factor_root", "two_quadratics"))
    if branch == "eisenstein":
        coeffs, prime = _eisenstein_general_quartic(rng, _PRIMES, 6)
        check(_eisenstein_ok(coeffs, prime), "medium eisenstein branch must actually satisfy the criterion")
        check(coeffs[-1] < 0, "medium eisenstein branch must force a negative constant term")
        p_expr = _poly_expr_str(coeffs)
        order = rng.choice(("largest", "smallest"))
        text = _order_text(rng, order, p_expr)
        degree = 4
        other = (2, 3)
        working = [
            f"p(x) = {p_expr} is Eisenstein at the prime {prime}: every coefficient below the "
            f"leading term is divisible by {prime}, and {prime}^2 does not divide the constant "
            f"term {coeffs[-1]}, so p(x) is irreducible over Q.",
            f"p(0) = {coeffs[-1]} < 0 and p(x) -> +infinity as x -> +/-infinity, so by the "
            f"intermediate value theorem p(x) has a real root; every root of the irreducible "
            f"p(x) has degree 4, so the {order} real root of p(x) = 0 has degree 4.",
        ]
    elif branch == "cubic_factor_root":
        coeffs, r, cubic, bound, r_order, other_order = _cubic_vs_linear(rng, -5, 5)
        check(_eval_poly(coeffs, r) == 0, "the forced root must actually be a root of the quartic")
        check(not _has_rational_root(cubic), "the cubic factor must have no rational root")
        p_expr = _poly_expr_str(coeffs)
        cubic_expr = _poly_expr_str(cubic)
        order = other_order
        text = _order_text(rng, order, p_expr)
        degree = 3
        other = (2, 4)
        working = [
            f"p(x) = {p_expr} = {_linear_factor_str(r)}({cubic_expr}); no integer divisor of the "
            f"cubic factor's constant term {cubic[-1]} is a root, so that cubic is irreducible "
            f"over Q, and Cauchy's bound puts every one of its real roots strictly inside "
            f"(-{bound}, {bound}).",
            f"alpha = {r} lies outside that range, so the {order} real root of p(x) = 0 belongs "
            "to the cubic factor, giving [Q(alpha):Q] = 3.",
        ]
    else:
        coeffs, (b1, c1), (b2, c2) = _two_irreducible_quadratics(rng, _MEDIUM_QUAD_B, _MEDIUM_QUAD_C)
        check(
            _quadratic_irreducible(b1, c1) and _quadratic_irreducible(b2, c2),
            "both quadratic factors must be irreducible",
        )
        p_expr = _poly_expr_str(coeffs)
        q1_expr, q2_expr = _poly_expr_str([1, b1, c1]), _poly_expr_str([1, b2, c2])
        D1, D2 = b1 * b1 - 4 * c1, b2 * b2 - 4 * c2
        order = rng.choice(("largest", "smallest"))
        text = _order_text(rng, order, p_expr)
        degree = 2
        other = (3, 4)
        working = [
            f"The rational-root test on p(x) = {p_expr} finds nothing, but p(x) = "
            f"({q1_expr})({q2_expr}), a product of two quadratics with discriminants "
            f"D1 = {D1} and D2 = {D2}, neither a perfect square, so both factors are "
            "irreducible over Q, and both have real roots (D1, D2 > 0).",
            f"Every root of p(x) belongs to one of those two quadratics, so the {order} real "
            "root of p(x) = 0 has degree 2.",
        ]
    return _finish(rng, text, degree, other, working)


# ---------------------------------------------------------------------------
# hard: biquadratics x^4 + px^2 + q, degree in {1, 2, 4}
# ---------------------------------------------------------------------------

_HARD_NONSQUARE_POOL = [n for n in range(2, 41) if not _is_perfect_square(n)]
_HARD_R_POOL = list(range(2, 8))


def _eisenstein_biquadratic(rng, prime_pool, m_range, tries: int = 200):
    """Constant term always forced negative, same IVT argument as the
    medium-tier general quartic."""
    for _ in range(tries):
        prime = rng.choice(prime_pool)
        p_coeff = prime * rng.randint(-3, 3)
        m = rng.randint(1, m_range)
        if m % prime == 0:
            continue
        q_coeff = -prime * m
        coeffs = [1, 0, p_coeff, 0, q_coeff]
        if _eisenstein_ok(coeffs, prime):
            return coeffs, prime
    raise AssertionError("could not build an Eisenstein-irreducible biquadratic")


def _two_nonsquare_quadratic_factors(rng, pool, tries: int = 200):
    for _ in range(tries):
        a, b = rng.sample(pool, 2)
        coeffs = [1, 0, -(a + b), 0, a * b]
        return coeffs, a, b
    raise AssertionError("could not build (x^2-a)(x^2-b)")


def _rational_root_vs_quadratic_factor(rng, r_pool, nonsquare_pool, tries: int = 200):
    """(x^2 - r^2)(x^2 - s), r forced so r^2 > s -- an exact, integer-only
    separation: sqrt(s) < r, so +/- r are the two global extremes and
    +/- sqrt(s) sit strictly between them. Returns (coeffs, r_signed, s,
    order)."""
    for _ in range(tries):
        r = rng.choice(r_pool)
        candidates = [s for s in nonsquare_pool if s < r * r]
        if not candidates:
            continue
        s = rng.choice(candidates)
        sign = rng.choice((1, -1))
        r_signed = sign * r
        coeffs = [1, 0, -(r * r + s), 0, r * r * s]
        order = "largest" if sign > 0 else "smallest"
        return coeffs, r_signed, s, order
    raise AssertionError("could not build a rational-root-dominant biquadratic")


def p17a_hard(rng) -> dict:
    branch = rng.choice(("eisenstein", "two_quadratic_factors", "linear_factor_root"))
    if branch == "eisenstein":
        coeffs, prime = _eisenstein_biquadratic(rng, _PRIMES, 6)
        check(_eisenstein_ok(coeffs, prime), "hard eisenstein branch must actually satisfy the criterion")
        check(coeffs[-1] < 0, "hard eisenstein branch must force a negative constant term")
        p_expr = _poly_expr_str(coeffs)
        order = rng.choice(("largest", "smallest"))
        text = _order_text(rng, order, p_expr)
        degree = 4
        other = (1, 2)
        working = [
            f"p(x) = {p_expr} is Eisenstein at the prime {prime}: the x^3 and x coefficients are "
            f"0 (divisible by {prime}), the x^2 coefficient {coeffs[2]} is divisible by {prime}, "
            f"and {prime}^2 does not divide the constant term {coeffs[-1]}, so p(x) is "
            "irreducible over Q.",
            f"p(0) = {coeffs[-1]} < 0 and p(x) -> +infinity as x -> +/-infinity, so by the "
            f"intermediate value theorem p(x) has a real root; every root of the irreducible "
            f"p(x) has degree 4, so the {order} real root of p(x) = 0 has degree 4.",
        ]
    elif branch == "two_quadratic_factors":
        coeffs, a, b = _two_nonsquare_quadratic_factors(rng, _HARD_NONSQUARE_POOL)
        check(
            not _has_rational_root(coeffs), "the rational-root test must find nothing even though the quartic factors"
        )
        p_expr = _poly_expr_str(coeffs)
        order = rng.choice(("largest", "smallest"))
        text = _order_text(rng, order, p_expr)
        degree = 2
        other = (1, 4)
        working = [
            f"The rational-root test on p(x) = {p_expr} finds nothing, but p(x) factors directly "
            f"as (x^2 - {a})(x^2 - {b}); {a} and {b} are not perfect squares, so both quadratic "
            "factors are irreducible over Q.",
            "Every root of p(x) is +/- sqrt(" + f"{a}) or +/- sqrt({b})" + ", each of degree 2, so "
            f"the {order} real root of p(x) = 0 has degree 2.",
        ]
    else:
        coeffs, r, s, order = _rational_root_vs_quadratic_factor(rng, _HARD_R_POOL, _HARD_NONSQUARE_POOL)
        check(_eval_poly(coeffs, r) == 0, "r must actually be a root")
        check(_eval_poly(coeffs, -r) == 0, "-r must actually be a root (biquadratic symmetry)")
        check(r * r > s, "r^2 must exceed s so +/- r are the biquadratic's global extremes")
        p_expr = _poly_expr_str(coeffs)
        rr = abs(r)
        text = _order_text(rng, order, p_expr)
        degree = 1
        other = (2, 4)
        working = [
            f"p(x) = {p_expr} = (x - {rr})(x + {rr})(x^2 - {s}); the roots are +/-{rr} and "
            f"+/-sqrt({s}), and {rr}^2 = {rr * rr} > {s}, so sqrt({s}) < {rr}.",
            f"alpha = {r} is then the {order} of the four real roots, giving [Q(alpha):Q] = 1.",
        ]
    return _finish(rng, text, degree, other, working)


# ---------------------------------------------------------------------------
# exam: sqrt(a) + sqrt(b) / [Q(sqrt a, sqrt b):Q], mixed with a bigger-number
# rational-root-times-irreducible-cubic reprise, degree in {2, 3, 4}
# ---------------------------------------------------------------------------

_EXAM_NONSQUARE_POOL = [n for n in range(2, 61) if not _is_perfect_square(n)]
_EXAM_ROOTS = [r for r in range(-10, 11) if r != 0]

# Explicit pool of (a, b) pairs, both non-square, with a*b a perfect square
# (so sqrt(ab) is rational and [Q(sqrt a, sqrt b):Q] collapses to 2) -- a
# random rejection-sample over all pairs finds these too rarely to rely on.
_EXAM_SQUARE_PRODUCT_PAIRS = [
    (a, b) for a in _EXAM_NONSQUARE_POOL for b in _EXAM_NONSQUARE_POOL if a < b and _is_perfect_square(a * b)
]
check(len(_EXAM_SQUARE_PRODUCT_PAIRS) >= 10, "exam square-product pool is too small")

_ALPHA_SUM_TEMPLATES = (
    "Let alpha = sqrt({a}) + sqrt({b}). Find [Q(alpha) : Q].",
    "Let a = {a} and b = {b}, and let alpha = sqrt(a) + sqrt(b). Find [Q(alpha) : Q].",
)
_COMPOUND_TEMPLATES = (
    "Let a = {a} and b = {b}. Find [Q(sqrt(a), sqrt(b)) : Q].",
    "Find the degree of the field extension [Q(sqrt({a}), sqrt({b})) : Q].",
)


def _compound_degree(a: int, b: int) -> int:
    check(
        not _is_perfect_square(a) and not _is_perfect_square(b),
        "compound branch requires both a and b to be non-square",
    )
    return 2 if _is_perfect_square(a * b) else 4


def _draw_ab(rng, want_square: bool, tries: int = 200):
    if want_square:
        return rng.choice(_EXAM_SQUARE_PRODUCT_PAIRS)
    for _ in range(tries):
        a, b = rng.sample(_EXAM_NONSQUARE_POOL, 2)
        if not _is_perfect_square(a * b):
            return a, b
    raise AssertionError("could not draw a, b of the requested class")


def _alpha_sum_working(a: int, b: int, degree: int, *, names_alpha: bool) -> list[str]:
    """`names_alpha` distinguishes the two exam phrasings: the `alpha = sqrt(a)
    + sqrt(b)` templates (True, so the closing line and `_finish`'s own
    summary both talk about alpha) from the direct `[Q(sqrt a, sqrt b):Q]`
    templates (False, where alpha is never introduced in the prompt at all --
    the working must not invent it either)."""
    ab = a * b
    label = "[Q(alpha):Q]" if names_alpha else f"[Q(sqrt({a}), sqrt({b})):Q]"
    if degree == 4:
        first = (
            f"a = {a}, b = {b}; the product ab = {ab} is not a perfect square, so sqrt({a}) and "
            f"sqrt({b}) generate genuinely independent quadratic extensions of Q, and "
            f"[Q(sqrt({a}), sqrt({b})):Q] = 4."
        )
        if not names_alpha:
            # `_finish` itself appends "{label} = 4." as the closing line; a
            # second, textually-identical sentence here would be a verbatim
            # duplicate a learner would notice, so this branch stops at the
            # one line above.
            return [first]
        return [first, f"alpha = sqrt({a}) + sqrt({b}) already generates the whole field, so {label} = 4."]
    k = math.isqrt(ab)
    first = (
        f"a = {a}, b = {b}; the product ab = {ab} = {k}^2 is a perfect square, so sqrt({ab}) = "
        f"{k} is rational, and sqrt({b}) = {k}/sqrt({a}) already lies in Q(sqrt({a}))."
    )
    if not names_alpha:
        return [first]
    return [first, f"alpha = sqrt({a}) + sqrt({b}) is then a rational multiple of sqrt({a}), so {label} = 2."]


def p17a_exam(rng) -> dict:
    # Weighted so no answer value exceeds ~40% of draws over the whole tier, under a "none
    # above 50%" bound. The last two slices are BOTH polynomial ("largest/smallest real root of p(x)
    # = 0") prompts on purpose: the cubic-vs-linear reprise alone always named a cubic-factor root
    # (degree 3), which made THAT masked phrase a single-key giveaway within this tier even though
    # "a root" itself was already dropped; mixing in a bigger-number Eisenstein-quartic reprise
    # (degree 4) under the SAME order templates gives every polynomial phrase in this tier at least
    # two keys (3 and 4), same as easy/medium/ hard's shared templates already have from their own
    # multi-branch use.
    roll = rng.random()
    label = "[Q(alpha):Q]"
    if roll < 0.20:
        a, b = _draw_ab(rng, want_square=False)
        degree = _compound_degree(a, b)
        check(degree == 4, "non-square a*b must give degree 4")
        text = phrase(rng, _ALPHA_SUM_TEMPLATES, a=a, b=b)
        other = (2, 3)
        working = _alpha_sum_working(a, b, degree, names_alpha=True)
    elif roll < 0.35:
        a, b = _draw_ab(rng, want_square=True)
        degree = _compound_degree(a, b)
        check(degree == 2, "square a*b must give degree 2")
        text = phrase(rng, _ALPHA_SUM_TEMPLATES, a=a, b=b)
        other = (3, 4)
        working = _alpha_sum_working(a, b, degree, names_alpha=True)
    elif roll < 0.50:
        a, b = _draw_ab(rng, want_square=False)
        degree = _compound_degree(a, b)
        check(degree == 4, "non-square a*b must give degree 4")
        text = phrase(rng, _COMPOUND_TEMPLATES, a=a, b=b)
        other = (2, 3)
        working = _alpha_sum_working(a, b, degree, names_alpha=False)
        label = f"[Q(sqrt({a}), sqrt({b})):Q]"
    elif roll < 0.65:
        a, b = _draw_ab(rng, want_square=True)
        degree = _compound_degree(a, b)
        check(degree == 2, "square a*b must give degree 2")
        text = phrase(rng, _COMPOUND_TEMPLATES, a=a, b=b)
        other = (3, 4)
        working = _alpha_sum_working(a, b, degree, names_alpha=False)
        label = f"[Q(sqrt({a}), sqrt({b})):Q]"
    elif roll < 0.90:
        coeffs, r, cubic, bound, r_order, other_order = _cubic_vs_linear(rng, -9, 9)
        check(_eval_poly(coeffs, r) == 0, "the forced root must actually be a root of the quartic")
        check(not _has_rational_root(cubic), "the cubic factor must have no rational root")
        p_expr = _poly_expr_str(coeffs)
        cubic_expr = _poly_expr_str(cubic)
        order = other_order
        text = _order_text(rng, order, p_expr)
        degree = 3
        other = (2, 4)
        working = [
            f"p(x) = {p_expr} = {_linear_factor_str(r)}({cubic_expr}); no integer divisor of the "
            f"cubic factor's constant term {cubic[-1]} is a root, so that cubic is irreducible "
            f"over Q, and Cauchy's bound puts every one of its real roots strictly inside "
            f"(-{bound}, {bound}).",
            f"alpha = {r} lies outside that range, so the {order} real root of p(x) = 0 belongs "
            "to the cubic factor, giving [Q(alpha):Q] = 3.",
        ]
    else:
        coeffs, prime = _eisenstein_general_quartic(rng, _PRIMES, 10)
        check(_eisenstein_ok(coeffs, prime), "exam eisenstein reprise must actually satisfy the criterion")
        check(coeffs[-1] < 0, "exam eisenstein reprise must force a negative constant term")
        p_expr = _poly_expr_str(coeffs)
        order = rng.choice(("largest", "smallest"))
        text = _order_text(rng, order, p_expr)
        degree = 4
        other = (2, 3)
        working = [
            f"p(x) = {p_expr} is Eisenstein at the prime {prime}: every coefficient below the "
            f"leading term is divisible by {prime}, and {prime}^2 does not divide the constant "
            f"term {coeffs[-1]}, so p(x) is irreducible over Q.",
            f"p(0) = {coeffs[-1]} < 0 and p(x) -> +infinity as x -> +/-infinity, so by the "
            f"intermediate value theorem p(x) has a real root; every root of the irreducible "
            f"p(x) has degree 4, so the {order} real root of p(x) = 0 has degree 4.",
        ]
    return _finish(rng, text, degree, other, working, label=label)


# ---------------------------------------------------------------------------
# Shared finish: assemble the returned dict, with the two remaining values as
# distractors and the pairwise-distinctness check every candidate must clear.
# ---------------------------------------------------------------------------


def _finish(rng, text: str, degree: int, other: tuple, working: list[str], label: str = "[Q(alpha):Q]") -> dict:
    check(degree not in other, "the answer must not appear among its own distractors")
    check(len(set(other)) == 2, "exactly two distinct distractor values are expected")
    check_pairwise_distinct(degree, *other, msg="answer and distractors must all be pairwise distinct")
    answer_text = str(degree)
    distractors = [str(v) for v in other]
    return dict(
        text=text,
        answer=degree,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[*working, f"{label} = {answer_text}."],
        distractors=distractors,
    )


@generator(
    "p17a_field_extension_degree",
    level=19,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def field_extension_degree(rng, tier="easy"):
    """Given a rational polynomial of degree at most 4 and a specified root
    alpha, or two rational numbers a and b, return [Q(alpha):Q] or
    [Q(sqrt a, sqrt b):Q] by deciding reducibility over Q. See the module
    docstring for the per-tier ladder."""
    if tier == "easy":
        return p17a_easy(rng)
    if tier == "medium":
        return p17a_medium(rng)
    if tier == "hard":
        return p17a_hard(rng)
    return p17a_exam(rng)
