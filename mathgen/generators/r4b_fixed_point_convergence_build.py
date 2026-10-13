"""Fixed-point iteration and its convergence: `r4b_fixed_point_convergence` (easy, medium, hard) and the exam chain
`exd_x_r4b_fixed_point_convergence`.

``uc2-fixed-point-iteration-convergence`` (``numerical-and-computational-mathematics``). The iteration function is
``g(x) = (x² + c)/d`` where ``c = pq`` and ``d = p + q`` for integers ``0 < p < q``, so its two fixed points are p and
q and ``g'(x) = 2x/d`` is 2p/d < 1 at p and 2q/d > 1 at q: the iteration is drawn to p and pushed away from q.

* easy: the rate at which the error shrinks, ``|g'(p)|``, at the fixed point the iteration (from a stated start
  between 0 and q) converges to. The learner must find both fixed points and test them.
* medium: the number of iterations the a priori bound ``λⁿ/(1 − λ)·|x₁ − x₀| ≤ tol`` needs on a stated interval
  ``[0, m]``, with ``λ = max |g'|`` on the interval.
* hard: the iteration ``x ↦ x − k f(x)`` for ``f(x) = (x − p)(x − q)``: the value of k that makes the convergence to
  p quadratic (``g'(p) = 0``).
* exam: for ``g(x) = (x² + c)/d``, (a) the smaller fixed point, (b) ``g'`` there, (c) ``g'`` at the other fixed point,
  (d) the least iteration count from the a priori bound with the learner's contraction constant.

Fixed points and rates are checked with SymPy (``solve``, ``diff``); the iteration count is found by exact `Fraction`
comparison of the bound and cross-checked with the logarithm formula.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r4b_kit import (
    X,
    as_single,
    fraction_part,
    integer_part,
    sub,
    with_redraw,
)
from ._ucv1_texsay import say_marked

M_OTHER = "num2-fixed-point-rate-at-the-other-fixed-point"
M_TWO = "num2-fixed-point-derivative-factor-two-dropped"
M_START = "num2-fixed-point-derivative-at-the-start"
M_RECIP = "num2-fixed-point-rate-inverted"
M_VALUE = "num2-fixed-point-value-for-rate"
M_STEP1 = "num2-fixed-point-bound-first-step-ignored"
M_DENOM = "num2-fixed-point-bound-denominator-dropped"
M_LAMBDA = "num2-fixed-point-lambda-at-the-fixed-point"
M_OFF = "num2-fixed-point-iteration-count-off-by-one"
M_DERIV_Q = "num2-fixed-point-newton-constant-from-other-root"
M_DERIV_X = "num2-fixed-point-newton-constant-from-x-term-only"
M_DERIV_0 = "num2-fixed-point-newton-constant-at-zero"
M_RECIPK = "num2-fixed-point-newton-constant-inverted"
M_SIGN_K = "num2-fixed-point-newton-constant-sign"
M_OTHER_ROOT = "num2-fixed-point-other-root-reported"
M_SUM = "num2-fixed-point-sum-of-roots-reported"
M_PRODUCT = "num2-fixed-point-constant-term-reported"
M_ONE_MINUS = "num2-fixed-point-rate-complement"


def g_text(c: int, d: int) -> str:
    return f"g(x) = (x² + {c})/{d}"


def _roots(rng, lo=1, hi=5, gap=2):
    for _ in attempts(200, "two fixed points"):
        p = rng.randint(lo, hi)
        q = p + rng.randint(gap, 8)
        return p, q
    raise AssertionError("unreachable")


def _check_fixed_points(p, q, c, d):
    """g(x) = (x² + c)/d: both p and q are fixed points, and g'(x) = 2x/d is below 1 at p and above 1 at q."""
    check(all(x * x + c == d * x for x in (p, q)) and p != q, "p and q are not both fixed points")
    check(abs(Fraction(2 * p, d)) < 1 < abs(Fraction(2 * q, d)), "p must attract and q repel")
    # the quadratic x² − d·x + c has exactly the roots p and q
    check(p + q == d and p * q == c, "fixed points disagree with the sum and product of roots")


def bound_count(lam: Fraction, delta: Fraction, tol: Fraction):
    """Least n with lam^n/(1 - lam) * delta <= tol (exact), and the real-valued crossing for a margin check."""
    n = 0
    while lam**n * delta / (1 - lam) > tol:
        n += 1
        check(n < 400, "the bound never reaches the tolerance")
    real = math.log(float(tol * (1 - lam) / delta)) / math.log(float(lam))
    return n, real


def _count_ok(n: int, real: float) -> bool:
    return 3 <= n <= 60 and 0.04 < n - real < 0.96


@generator(
    "r4b_fixed_point_convergence",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4b_fixed_point_convergence(rng, tier="easy"):
    return with_redraw(lambda: _single(rng, tier), "a fixed-point question with enough distractors")


def _single(rng, tier):
    if tier == "easy":
        p, q = _roots(rng, 1, 8, 2)
        c, d = p * q, p + q
        _check_fixed_points(p, q, c, d)
        x0 = rng.choice([v for v in range(1, q) if v != p])
        rate = Fraction(2 * p, d)
        text = say_marked(
            f"The iteration «xₙ₊₁ = g(xₙ)», with «{g_text(c, d)}» and starting value «x₀ = {x0}», converges to a fixed "
            "point of «g». Find «|g′(x)|» at that fixed point, as a fraction."
        )
        cands = [
            (M_OTHER, Fraction(2 * q, d)),
            (M_TWO, Fraction(p, d)),
            (M_START, Fraction(2 * x0, d)),
            (M_RECIP, Fraction(d, 2 * p)),
            (M_VALUE, Fraction(p)),
            (M_ONE_MINUS, 1 - rate),
        ]
        part = fraction_part(
            "",
            rate,
            cands,
            1,
            working=[
                f"Fixed points: x = (x² + {c})/{d} gives x² − {d}x + {c} = 0, so x = {p} or x = {q}.",
                f"g′(x) = 2x/{d}: it is {Fraction(2 * p, d)} at x = {p} (below 1, so the iteration is drawn in) and "
                f"{Fraction(2 * q, d)} at x = {q}.",
            ],
        )
        return _q(text, part)
    if tier == "medium":
        for _ in attempts(500, "an interval and tolerance"):
            p = rng.randint(2, 6)
            q = p + rng.randint(3, 8)
            c, d = p * q, p + q
            m = rng.randint(p, (d - 1) // 2)
            lam = Fraction(2 * m, d)
            x1 = Fraction(c, d)
            k = rng.choice([3, 4, 5, 6])
            tol = Fraction(1, 10**k)
            n, real = bound_count(lam, abs(x1), tol)
            if not _count_ok(n, real):
                continue
            n_alt1, _ = bound_count(lam, Fraction(1), tol)
            lam_p = Fraction(2 * p, d)
            if lam_p >= 1:
                continue
            break
        _check_fixed_points(p, q, c, d)
        # second route: the logarithm formula
        want = math.ceil(math.log(float(tol * (1 - lam) / x1)) / math.log(float(lam)))
        check(int(want) == n, "iteration count disagrees with the logarithm formula")
        # slips
        no_denom = 0
        while lam**no_denom * x1 > tol:
            no_denom += 1
        lam_half = Fraction(m, d)
        n_half, _ = bound_count(lam_half, x1, tol)
        n_p, _ = bound_count(lam_p, x1, tol)
        cands = [
            (M_STEP1, n_alt1),
            (M_DENOM, no_denom),
            (M_LAMBDA, n_p),
            (M_OFF, n - 1),
            (M_OFF, n + 1),
            ("num2-fixed-point-derivative-factor-two-dropped", n_half),
        ]
        text = say_marked(
            f"On the interval «[0, {m}]» the function «{g_text(c, d)}» maps the interval into itself. The iteration "
            f"«xₙ₊₁ = g(xₙ)» starts at «x₀ = 0». Using the a priori error bound with the largest value of «|g′|» on "
            f"the interval as the contraction constant, find the least number of iterations «n» that guarantees "
            f"«|xₙ − p| ≤ 10⁻{_sup(k)}», where «p» is the fixed point in the interval."
        )
        part = integer_part(
            "",
            n,
            cands,
            1,
            working=[
                f"g′(x) = 2x/{d} is increasing, so λ = max|g′| = g′({m}) = {lam}.",
                f"x₁ = g(0) = {x1}, so |x₁ − x₀| = {x1}.",
                f"The bound λⁿ/(1 − λ)·|x₁ − x₀| first drops to 10⁻{_sup(k)} or below at n = {n}.",
            ],
        )
        return _q(text, part)
    # hard
    for _ in attempts(500, "a quadratic-convergence question"):
        p = rng.randint(-4, 5)
        q = p + rng.randint(2, 7)
        if p == 0 or q == 0 or p + q == 0:
            continue
        break
    m = rng.choice([2, 3, 4])
    d = p + q
    c = p * q
    quad = _quad_text(m, m * d, m * c)
    smaller, larger = min(p, q), max(p, q)
    k_val = Fraction(1, m * (smaller - larger))
    gx = X - sympy.Rational(k_val.numerator, k_val.denominator) * sympy.Poly(m * (X - p) * (X - q), X).as_expr()
    check(sympy.diff(gx, X).subs(X, smaller) == 0, "g'(p) is not zero for this k")
    check(sympy.diff(gx, X, 2).subs(X, smaller) != 0, "the convergence would be better than quadratic")
    text = say_marked(
        f"The roots of «{quad} = 0» are found by the iteration «xₙ₊₁ = xₙ − k({quad})», where «k» is a nonzero "
        f"constant. Find the value of «k» for which the iteration converges quadratically to the smaller root, "
        "as a fraction."
    )
    cands = [
        (M_SIGN_K, -k_val),
        (M_DERIV_Q, Fraction(1, m * (2 * larger - d))),
        (M_DERIV_X, Fraction(1, 2 * m * smaller)),
        (M_DERIV_0, Fraction(1, -m * d)),
        (M_RECIPK, Fraction(m * (smaller - larger))),
    ]
    part = fraction_part(
        "",
        k_val,
        cands,
        1,
        working=[
            f"Here g(x) = x − k·f(x) with f(x) = {quad}, whose roots are {smaller} and {larger}.",
            f"g′(x) = 1 − k(f′(x)), so g′({smaller}) = 1 − k({m * (smaller - larger)}) = 0 gives k = {k_val}.",
        ],
    )
    return _q(text, part)


def _quad_text(a: int, b: int, c: int) -> str:
    """a x² − b x + c as printed, with the signs of b and c respected (b here is the coefficient of -x)."""
    out = "x²" if a == 1 else f"{a}x²"
    if b != 0:
        out += f" {'−' if b > 0 else '+'} {abs(b)}x"
    if c != 0:
        out += f" {'+' if c > 0 else '−'} {abs(c)}"
    return out


def _sup(k: int) -> str:
    return str(k).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def _q(text_pair, part):
    text, latex = text_pair
    d = as_single(part, text)
    d["latex"] = latex
    if len(d["distractors"]) < 3:
        from ._r4b_kit import Redraw

        raise Redraw
    return d


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4b_fixed_point_convergence", level=22, presets={"exam": dict(tier="exam")})
def exd_x_r4b_fixed_point_convergence(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam fixed-point question")


def _exam(rng):
    for _ in attempts(1000, "an exam fixed-point draw"):
        p = rng.randint(2, 6)
        q = p + rng.randint(3, 8)
        c, d = p * q, p + q
        lam = Fraction(2 * p, d)
        x1 = Fraction(c, d)
        k = rng.choice([3, 4, 5])
        tol = Fraction(1, 10**k)
        n, real = bound_count(lam, abs(x1), tol)
        if _count_ok(n, real) and 0.3 < lam < 0.9:
            break
    _check_fixed_points(p, q, c, d)
    lam_q = Fraction(2 * q, d)
    stem, latex = say_marked(f"The iteration «xₙ₊₁ = g(xₙ)» uses «{g_text(c, d)}» and starts at «x₀ = 0».")
    p_a = integer_part(
        "Find the smaller fixed point «p» of «g».",
        p,
        [(M_OTHER_ROOT, q), (M_SUM, d), (M_PRODUCT, c), ("num2-fixed-point-sign-slip-in-root", abs(q - p))],
        1,
        working=[f"x = (x² + {c})/{d} gives x² − {d}x + {c} = 0, so x = {p} or x = {q}; the smaller is {p}."],
        value_range=(0, None),
    )
    p_b = fraction_part(
        "Using your «p», find «|g′(p)|» as a fraction.",
        lam,
        [(M_OTHER, lam_q), (M_TWO, Fraction(p, d)), (M_RECIP, Fraction(d, 2 * p)), (M_ONE_MINUS, 1 - lam)],
        2,
        ft=f"2*part_a/{d}",
        working=[f"g′(x) = 2x/{d}, so |g′({p})| = {lam}."],
        value_range=(0, 1),
    )
    p_c = fraction_part(
        "Using your «p», find «|g′(q)|» as a fraction at the other fixed point «q = {d} − p».".replace("{d}", str(d)),
        lam_q,
        [(M_OTHER, lam), (M_TWO, Fraction(q, d)), (M_RECIP, Fraction(d, 2 * q)), (M_START, Fraction(2 * c, d * d))],
        2,
        ft=f"2*({d} - part_a)/{d}",
        working=[f"q = {q}, so |g′(q)| = {lam_q}, which is above 1: the iteration moves away from q."],
        value_range=(0, None),
    )
    no_denom = 0
    while lam**no_denom * x1 > tol:
        no_denom += 1
    n_p = bound_count(lam_q if lam_q < 1 else Fraction(9, 10), x1, tol)[0]
    n_alt1, _ = bound_count(lam, Fraction(1), tol)
    p_d = integer_part(
        f"Use your «|g′(p)|» as the contraction constant «λ» in the a priori bound «λⁿ/(1 − λ)·|x₁ − x₀|». "
        f"Find the least «n» that guarantees «|xₙ − p| ≤ 10⁻{_sup(k)}».",
        n,
        [
            (M_STEP1, n_alt1),
            (M_DENOM, no_denom),
            (M_OFF, n - 1),
            (M_OFF, n + 1),
            (M_TWO, bound_count(Fraction(p, d), x1, tol)[0]),
        ],
        2,
        ft=f"ceiling(log(Rational(1, {10**k})*(1 - part_b)/Rational({x1.numerator}, {x1.denominator}))/log(part_b))",
        working=[f"x₁ = g(0) = {x1}; the bound λⁿ/(1 − λ)·{x1} with λ = {lam} first reaches 10⁻{_sup(k)} at n = {n}."],
        value_range=(0, 400),
    )
    for part in (p_a, p_b, p_c, p_d):
        part["prompt"], part["prompt_latex"] = say_marked(part["prompt"])
    return dict(text=stem, latex=latex, parts=[p_a, p_b, p_c, p_d])
