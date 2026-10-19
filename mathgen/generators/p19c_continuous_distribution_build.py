"""Generates `p19c_density_probability` and `p19c_continuous_uniform_moments` questions on probability density
functions and continuous uniform distributions.

Two continuous-random-variable skills.

Two generators, one module:

- ``p19c_density_probability`` (``senior-understand-the-concepts-of-a-
  probability-density-function``, ACMMM165, ``response_family:
  scalar_expression``, ``answer: number``). **THIS SKILL IS NARROWED.** The
  raw descriptor names three clauses: (1) probability density function and
  probabilities from integrals, (2) the cumulative distribution function,
  (3) "examine simple types of continuous random variables and use them in
  appropriate contexts". The skill
  drops (2) and (3) entirely -- no CDF value is ever built or graded, and no
  "type" classification is ever asked. Every tier is exactly: given a
  probability density function stated algebraically on a bounded interval
  [c, d] and two bounds [p, q] strictly inside that interval (the exam tier
  puts one bound AT an endpoint of the support instead), compute the
  probability that X falls between them by evaluating the definite integral
  of the density over that range.

  Construction never prints an arbitrary polynomial and calls it a density:
  every density in this generator is built from a normalising constant ``k``
  solved algebraically so the integral over the full stated support is
  EXACTLY 1, with an EXACT ``Fraction`` construction route (integer bounds,
  integer arithmetic throughout -- no rounding until the final display), and
  is provably nonnegative on its support by the shape of the family chosen
  (see ``_check_density`` below). The independent in-generator check
  (``_verify_density_and_probability``) re-derives the SAME density's full-
  support integral and the SAME bounded probability via ``sympy.integrate``
  acting symbolically on ``sympy.Rational`` -- an entirely different code
  path from the plain-``Fraction`` construction route.

  Density family per tier (``c``, ``d`` integers, ``L = d - c``). The kept
  kernel's own wording contrasts p, q ("still directly stated") against k
  ("requiring the normalising constant to be solved for and shown, NOT
  pre-supplied") -- so from medium on, the rendered density shows the
  literal letter ``k`` (never its numeric value), and the ``working`` field
  shows the equation solved for it; only easy's uniform ``k = 1/L`` is
  immediate by inspection and shown numerically:
  - easy: ``f(x) = k`` (uniform) on [c, d], ``k = 1/L`` shown numerically;
    p, q are two of the stated "nice" integers, strictly inside (c, d).
  - medium: ``f(x) = k*(x - c)`` (a linear ramp from 0 at x=c) -- ``k`` is
    shown as the LETTER ``k``, solved from ``k*L**2/2 = 1`` -> ``k = 2/L**2``
    (the `working` field states this); p, q strictly inside (c, d). A
    "k forgotten" distractor (k=1, never normalised) is now exercisable.
  - hard: ``f(x) = k*(x - c)**2`` (a genuine quadratic) -- same "k is a
    letter, solved and shown in `working`" treatment, from ``k*L**3/3 = 1``
    -> ``k = 3/L**3``; p, q strictly inside (c, d).
  - exam: same quadratic family as hard (k a letter, solved and shown), but
    one of p, q is set EXACTLY to an endpoint of the support (p = c, or
    q = d) -- the learner must recognise the density is genuinely 0 outside
    [c, d] and integrate from the support's own edge, rather than assuming
    both bounds sit symmetrically inside the interval the way every lower
    tier's bounds do. A genuine reasoning step over hard (which endpoint),
    not a bigger polynomial.

  **Boundary against `definite-integral`** (`skills/integration/definite-
  integral.yaml`): that skill evaluates a bare polynomial definite integral
  with NO probability semantics -- the integrand is never stated or implied
  to be a density, never constrained nonnegative or normalised, and the
  result is never framed as a probability. Every prompt here ALWAYS names
  the function as "the probability density function of a continuous random
  variable X" and ALWAYS frames the computed value as "the probability that
  X falls between p and q" -- never a bare "evaluate the integral of f(x) dx"
  with no probability language.

- ``p19c_continuous_uniform_moments`` (``senior-recognise-the-expected-
  value-variance-and-standard-deviation``, ACMMM166, ``response_family:
  ordered_structures``, ``answer: ordered_list``). Given a continuous random
  variable UNIFORMLY distributed on a stated interval [a, b], compute the
  ordered triple (E(X), Var(X), SD(X)) using the three CLOSED-FORM uniform-
  distribution formulas: E(X) = (a+b)/2, Var(X) = (b-a)**2/12, SD(X) =
  sqrt(Var(X)), evaluated (not left as an unevaluated ``sqrt(...)``) to a
  stated precision. Never a discrete weighted-sum computation.

  Difficulty ladder:
  - easy: a, b small nonnegative integers, b - a a multiple of 6 (the actual
    discriminator: gap = 6k makes Var(X) = gap**2/12 = 3k**2 an EXACT
    integer, so all three values round to tidy 2-d.p. decimals -- an even
    gap alone, e.g. gap=2, gives Var(X) = 1/3 = 0.333..., which does NOT).
  - medium: b - a odd AND not a multiple of 3, so Var(X) shares no common
    factor with 12 and is genuinely non-terminating in decimal (e.g.
    gap=1 -> 1/12 = 0.08333...), needing the stated 2-d.p. rounding.
  - hard: a is negative (the interval straddles or sits entirely below
    zero), testing E(X)/spread with a negative midpoint.
  - exam: the interval is stated indirectly through a short scenario ("a
    component's lifetime ... is guaranteed to be at least a hours and never
    exceeds b hours") rather than printed as a bare "[a, b]" -- the learner
    must read a, b out of the described context before applying the same
    three closed-form formulas. A genuine reasoning escalation (extracting
    the interval from prose), not bigger numbers.

  **Boundary against `discrete-expected-value` / `finite-distribution-
  variance`** (`skills/probability/discrete-expected-value.yaml`,
  `skills/statistics-probability/finite-distribution-variance.yaml`): both
  are DISCRETE, finite-outcome-list random variables (a table of outcomes
  and probabilities) computed by a probability-weighted sum. This skill is
  always CONTINUOUS and UNIFORM on a stated interval, computed by the three
  closed-form formulas -- never a table of discrete outcomes/probabilities.

  Independent in-generator check (``_verify_uniform_moments``) recomputes
  the SAME triple from ``a``, ``b`` via ``sympy.Rational``/``sympy.sqrt``,
  an entirely different code path from the plain-``Fraction`` construction.
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    rational_text,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
)

_x = sp.symbols("x")

_MAX_ATTEMPTS = 500
_DP1 = 3  # skill 1 stated precision (decimal places)
_DP2 = 2  # skill 2 stated precision (decimal places)


# ============================================================================
# Skill 1: p19c_density_probability
# ============================================================================


def _to_sp(value: int) -> sp.Integer:
    return sp.Integer(value)


def _density_expr(shape: str, k: Fraction, c: int) -> sp.Expr:
    k_sp = sp.Rational(k.numerator, k.denominator)
    if shape == "uniform":
        return k_sp
    if shape == "linear":
        return k_sp * (_x - _to_sp(c))
    return k_sp * (_x - _to_sp(c)) ** 2


def _sp_to_fraction(expr) -> Fraction:
    r = sp.nsimplify(sp.Rational(expr))
    return Fraction(int(r.p), int(r.q))


def _check_density(shape: str, k: Fraction, c: int, d: int) -> None:
    """Nonnegativity is provable by the shape of the family chosen, not
    assumed: a positive constant is nonnegative everywhere; k*(x - c) is
    nonnegative on [c, d] because (x - c) >= 0 there and k > 0; k*(x - c)**2
    is nonnegative everywhere because a square is never negative. Checked
    both algebraically (k > 0) and by sampling the density across the full
    support on an integer grid, so a future shape change that broke this
    would be caught even if the algebraic argument in a docstring went
    stale."""
    check(k > 0, f"normalising constant is not positive: k={k}")
    f = _density_expr(shape, k, c)
    for sample in range(c, d + 1):
        value = f.subs(_x, sample)
        check(value >= 0, f"density is negative at x={sample}: f(x)={value}")


def _verify_density_and_probability(shape: str, k: Fraction, c: int, d: int, p: int, q: int, raw: Fraction) -> None:
    """Independent check: rebuild the density symbolically with sympy and
    (a) confirm it integrates to EXACTLY 1 over its full stated support --
    the "is it a genuine density" proof -- and (b) recompute the bounded
    probability via ``sympy.integrate`` and assert it matches the plain-
    ``Fraction`` construction route exactly."""
    f = _density_expr(shape, k, c)
    total = sp.integrate(f, (_x, _to_sp(c), _to_sp(d)))
    check(sp.nsimplify(total) == 1, f"density does not integrate to 1 over [{c}, {d}]: got {total}")
    prob = sp.integrate(f, (_x, _to_sp(p), _to_sp(q)))
    check_equivalent(
        _sp_to_fraction(prob), raw, f"independent sympy integral mismatch: shape={shape} c={c} d={d} p={p} q={q}"
    )


def _density_str(shape: str, k: Fraction, c: int, *, show_k: bool = True) -> str:
    """``show_k=False`` renders the literal letter ``k`` in place of its
    numeric value -- the medium/hard/exam tiers never pre-supply the
    normalising constant ("requiring the normalising
    constant to be solved for and shown, not pre-supplied"); only easy's
    uniform density (whose ``k = 1/L`` is immediate by inspection) shows a
    numeric coefficient."""
    base = "x" if c == 0 else f"(x - {c})"
    if not show_k:
        if shape == "uniform":
            return "k"
        if shape == "linear":
            return f"k{base}"
        return f"k{base}^2"
    k_s = rational_text(k)
    if shape == "uniform":
        return k_s
    # Parenthesise a fractional coefficient so "1/32x" (ambiguous: 1/(32x)
    # or (1/32)*x?) never renders -- always "(1/32)x", never the bare form.
    k_disp = f"({k_s})" if "/" in k_s else k_s
    if shape == "linear":
        return f"{k_disp}{base}"
    return f"{k_disp}{base}^2"


_FACT_TEMPLATES = (
    "A continuous random variable X has probability density function f(x) = {fexpr} for "
    "{c} <= x <= {d} (and f(x) = 0 elsewhere). Find the probability that X falls between "
    "{p} and {q}, correct to {dp} decimal places.",
    "The random variable X has probability density function f(x) = {fexpr}, defined on the "
    "interval {c} <= x <= {d}. Compute the probability that X lies between x = {p} and "
    "x = {q}, correct to {dp} decimal places.",
    "Let X be a continuous random variable with probability density function f(x) = {fexpr} "
    "for x in [{c}, {d}]. Determine the probability that X falls between {p} and {q}, "
    "correct to {dp} decimal places.",
)

_FACT_TEMPLATES_SYMBOLIC_K = (
    "A continuous random variable X has probability density function f(x) = {fexpr} for "
    "{c} <= x <= {d} (and f(x) = 0 elsewhere), where k is the constant for which f is a valid "
    "probability density function. Find the probability that X falls between {p} and {q}, "
    "correct to {dp} decimal places.",
    "The random variable X has probability density function f(x) = {fexpr}, defined on the "
    "interval {c} <= x <= {d}, where k is the constant that makes f a valid probability "
    "density function on this interval. Compute the probability that X lies between x = {p} "
    "and x = {q}, correct to {dp} decimal places.",
    "Let X be a continuous random variable with probability density function f(x) = {fexpr} "
    "for x in [{c}, {d}], where k is the constant for which f is a valid probability density "
    "function. Determine the probability that X falls between {p} and {q}, correct to {dp} "
    "decimal places.",
)

_EXAM_TEMPLATES_SYMBOLIC_K = (
    "A continuous random variable X has probability density function f(x) = {fexpr} for "
    "{c} <= x <= {d} (and f(x) = 0 elsewhere, so the support of X is exactly [{c}, {d}]), "
    "where k is the constant for which f is a valid probability density function. Find the "
    "probability that X falls between {p} and {q}, correct to {dp} decimal places.",
    "The random variable X has probability density function f(x) = {fexpr}, defined on the "
    "interval {c} <= x <= {d} and equal to 0 outside it, where k is the constant that makes f "
    "a valid probability density function on this interval. Compute the probability that X "
    "lies between x = {p} and x = {q}, correct to {dp} decimal places.",
    "Let X be a continuous random variable with probability density function f(x) = {fexpr} "
    "for x in [{c}, {d}] (X can never take a value outside this interval), where k is the "
    "constant for which f is a valid probability density function. Determine the probability "
    "that X falls between {p} and {q}, correct to {dp} decimal places.",
)


def _prob_uniform(c: int, d: int, p: int, q: int) -> Fraction:
    return Fraction(q - p, d - c)


def _prob_linear(c: int, d: int, p: int, q: int) -> Fraction:
    L = d - c
    u, v = q - c, p - c
    return Fraction(u * u - v * v, L * L)


def _prob_quadratic(c: int, d: int, p: int, q: int) -> Fraction:
    L = d - c
    u, v = q - c, p - c
    return Fraction(u**3 - v**3, L**3)


def _prob_linear_wrong_k(c: int, d: int, p: int, q: int) -> Fraction:
    """Wrong-k-magnitude distractor for the linear tier: applies k = 1/L --
    the UNIFORM family's normalising constant, one level below linear -- to
    the correct linear integral shape (integral of (x - c) dx from p to q is
    (u^2 - v^2)/2), instead of the real k = 2/L**2. A genuinely different
    magnitude from `k_forgotten`'s old literal k=1, which could exceed 1 for
    a wide interval; this stays bounded because it is still 1/L of a
    quantity less than L**2/2."""
    L = d - c
    u, v = q - c, p - c
    return Fraction(u * u - v * v, 2 * L)


def _prob_quadratic_wrong_k(c: int, d: int, p: int, q: int) -> Fraction:
    """Wrong-k-magnitude distractor for the quadratic tiers: applies
    k = 2/L**2 -- the LINEAR family's normalising constant, one level below
    quadratic -- to the correct quadratic integral shape (integral of
    (x - c)**2 dx from p to q is (u^3 - v^3)/3), instead of the real
    k = 3/L**3."""
    L = d - c
    u, v = q - c, p - c
    return Fraction(2 * (u**3 - v**3), 3 * L * L)


def _finish1(
    text: str, answer_rounded: Fraction, dp: int, distractor_candidates: list, working: list | None = None
) -> dict:
    answer_text = stated_precision_string(answer_rounded, dp)
    survivors = [d for d in distractor_candidates if d is not None]
    distractors = pick_distractors(answer_text, survivors)
    out = dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer_rounded),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )
    if working:
        out["working"] = working
    return out


def _drop_if_collides1(raw: Fraction, dp: int, answer_rounded: Fraction) -> str | None:
    """Drop a distractor candidate that collides with the true (rounded)
    answer, OR that rounds outside the open interval (0, 1) -- a negative or >=1 "probability" is rejectable on
    sight without doing the actual integral, which is a giveaway, not a
    genuine distractor (every real probability from this generator's
    construction lies strictly inside (0, 1))."""
    rounded = round_half_up(raw, dp)
    if rounded <= 0 or rounded >= 1:
        return None
    if rounded == answer_rounded:
        return None
    return stated_precision_string(rounded, dp)


def _pd_easy(rng) -> dict:
    dp = _DP1
    for _attempt in range(_MAX_ATTEMPTS):
        c = rng.randint(0, 4)
        L = rng.randint(4, 12)
        d = c + L
        p, q = sorted(rng.sample(range(c + 1, d), 2))
        k = Fraction(1, L)
        raw = _prob_uniform(c, d, p, q)
        if not rounds_unambiguously(float(raw), dp):
            continue
        answer_rounded = round_half_up(raw, dp)
        if not (0 < answer_rounded < 1):
            continue

        complement_raw = Fraction(1) - raw
        # Forgets to subtract the lower bound: uses (q - c)/L, the cumulative
        # probability up to q, instead of (q - p)/L -- always a genuinely
        # different value (equal to the true answer only if p == c, which
        # cannot happen here since p is strictly interior).
        forgot_lower_raw = Fraction(q - c, L)
        # Wrong bound pairing: measures from the wrong end of the interval,
        # (d - q)/L instead of (q - p)/L.
        wrong_bound_raw = Fraction(d - q, L)

        cands = [
            _drop_if_collides1(complement_raw, dp, answer_rounded),
            _drop_if_collides1(forgot_lower_raw, dp, answer_rounded),
            _drop_if_collides1(wrong_bound_raw, dp, answer_rounded),
        ]
        if len({x for x in cands if x is not None}) < 3:
            continue
        break
    else:
        check(False, "easy: could not draw a non-degenerate question with >=3 in-range distractors")

    _check_density("uniform", k, c, d)
    _verify_density_and_probability("uniform", k, c, d, p, q, raw)

    fexpr = _density_str("uniform", k, c)
    text = phrase(rng, _FACT_TEMPLATES, fexpr=fexpr, c=str(c), d=str(d), p=str(p), q=str(q), dp=str(dp))
    return _finish1(text, answer_rounded, dp, cands)


def _pd_medium(rng) -> dict:
    dp = _DP1
    for _attempt in range(_MAX_ATTEMPTS):
        c = rng.randint(0, 3)
        L = rng.randint(5, 10)
        d = c + L
        p, q = sorted(rng.sample(range(c + 1, d), 2))
        k = Fraction(2, L * L)
        raw = _prob_linear(c, d, p, q)
        if not rounds_unambiguously(float(raw), dp):
            continue
        answer_rounded = round_half_up(raw, dp)
        if not (0 < answer_rounded < 1):
            continue

        complement_raw = Fraction(1) - raw
        uniform_instead_raw = Fraction(q - p, L)
        # Wrong k magnitude: k = 1/L (uniform family, one level below) on
        # the correct linear integral shape, instead of the real k=2/L**2.
        wrong_k_raw = _prob_linear_wrong_k(c, d, p, q)

        cands = [
            _drop_if_collides1(complement_raw, dp, answer_rounded),
            _drop_if_collides1(uniform_instead_raw, dp, answer_rounded),
            _drop_if_collides1(wrong_k_raw, dp, answer_rounded),
        ]
        if len({x for x in cands if x is not None}) < 3:
            continue
        break
    else:
        check(False, "medium: could not draw a non-degenerate question with >=3 in-range distractors")

    _check_density("linear", k, c, d)
    _verify_density_and_probability("linear", k, c, d, p, q, raw)

    fexpr = _density_str("linear", k, c, show_k=False)
    text = phrase(rng, _FACT_TEMPLATES_SYMBOLIC_K, fexpr=fexpr, c=str(c), d=str(d), p=str(p), q=str(q), dp=str(dp))
    working = [
        f"Solve for k: the integral of f(x) = k(x - {c}) over [{c}, {d}] must equal 1, so "
        f"k*{L}^2/2 = 1, giving k = {rational_text(k)}.",
    ]
    return _finish1(text, answer_rounded, dp, cands, working)


def _pd_hard(rng) -> dict:
    dp = _DP1
    for _attempt in range(_MAX_ATTEMPTS):
        c = rng.randint(0, 3)
        L = rng.randint(5, 10)
        d = c + L
        p, q = sorted(rng.sample(range(c + 1, d), 2))
        k = Fraction(3, L**3)
        raw = _prob_quadratic(c, d, p, q)
        if not rounds_unambiguously(float(raw), dp):
            continue
        answer_rounded = round_half_up(raw, dp)
        if not (0 < answer_rounded < 1):
            continue

        complement_raw = Fraction(1) - raw
        linear_instead_raw = Fraction((q - c) ** 2 - (p - c) ** 2, L * L)
        # Wrong k magnitude: k = 2/L**2 (linear family, one level below) on
        # the correct quadratic integral shape, instead of the real
        # k = 3/L**3.
        wrong_k_raw = _prob_quadratic_wrong_k(c, d, p, q)

        cands = [
            _drop_if_collides1(complement_raw, dp, answer_rounded),
            _drop_if_collides1(linear_instead_raw, dp, answer_rounded),
            _drop_if_collides1(wrong_k_raw, dp, answer_rounded),
        ]
        if len({x for x in cands if x is not None}) < 3:
            continue
        break
    else:
        check(False, "hard: could not draw a non-degenerate question with >=3 in-range distractors")

    _check_density("quadratic", k, c, d)
    _verify_density_and_probability("quadratic", k, c, d, p, q, raw)

    fexpr = _density_str("quadratic", k, c, show_k=False)
    text = phrase(rng, _FACT_TEMPLATES_SYMBOLIC_K, fexpr=fexpr, c=str(c), d=str(d), p=str(p), q=str(q), dp=str(dp))
    working = [
        f"Solve for k: the integral of f(x) = k(x - {c})^2 over [{c}, {d}] must equal 1, so "
        f"k*{L}^3/3 = 1, giving k = {rational_text(k)}.",
    ]
    return _finish1(text, answer_rounded, dp, cands, working)


def _pd_exam(rng) -> dict:
    """Same quadratic family as hard, but one bound is set EXACTLY to an
    endpoint of the support (p = c, or q = d) -- the learner must recognise
    the density is 0 outside [c, d] and integrate from the support's own
    edge, rather than assuming both bounds sit symmetrically interior the
    way every lower tier's bounds do."""
    dp = _DP1
    for _attempt in range(_MAX_ATTEMPTS):
        c = rng.randint(0, 3)
        L = rng.randint(5, 10)
        d = c + L
        k = Fraction(3, L**3)
        branch = rng.choice(("lower", "upper"))
        if branch == "lower":
            p = c
            q = rng.randint(c + 1, d - 1)
            offset = q - c
        else:
            q = d
            p = rng.randint(c + 1, d - 1)
            offset = d - p
        # With one bound pinned to the support's
        # own endpoint, the probability is exactly (offset/L)**3, so an
        # interior offset that happens to land exactly at the midpoint
        # (offset/L == 1/2, only possible when L is even) always gives
        # exactly 0.125 regardless of L or c -- measured concentrating 46/200
        # exam-tier draws (23%, over the composition's 15% ceiling) on that
        # single value. Redraw rather than accept an exact-half interior
        # offset; every other offset/L ratio is far less likely to collide
        # across different (c, L) draws, so excluding only this one case is
        # enough (re-measured below at the true fix).
        if 2 * offset == L:
            continue
        raw = _prob_quadratic(c, d, p, q)
        if not rounds_unambiguously(float(raw), dp):
            continue
        answer_rounded = round_half_up(raw, dp)
        if not (0 < answer_rounded < 1):
            continue

        complement_raw = Fraction(1) - raw
        linear_instead_raw = Fraction((q - c) ** 2 - (p - c) ** 2, L * L)
        # Wrong k magnitude: k = 2/L**2 (linear family, one level below) on
        # the correct quadratic integral shape, instead of the real
        # k = 3/L**3.
        wrong_k_raw = _prob_quadratic_wrong_k(c, d, p, q)

        cands = [
            _drop_if_collides1(complement_raw, dp, answer_rounded),
            _drop_if_collides1(linear_instead_raw, dp, answer_rounded),
            _drop_if_collides1(wrong_k_raw, dp, answer_rounded),
        ]
        if len({x for x in cands if x is not None}) < 3:
            continue
        break
    else:
        check(False, "exam: could not draw a non-degenerate boundary question with >=3 in-range distractors")

    _check_density("quadratic", k, c, d)
    _verify_density_and_probability("quadratic", k, c, d, p, q, raw)

    fexpr = _density_str("quadratic", k, c, show_k=False)
    text = phrase(rng, _EXAM_TEMPLATES_SYMBOLIC_K, fexpr=fexpr, c=str(c), d=str(d), p=str(p), q=str(q), dp=str(dp))
    working = [
        f"Solve for k: the integral of f(x) = k(x - {c})^2 over [{c}, {d}] must equal 1, so "
        f"k*{L}^3/3 = 1, giving k = {rational_text(k)}.",
    ]
    return _finish1(text, answer_rounded, dp, cands, working)


@generator(
    "p19c_density_probability",
    level=15,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19c_density_probability(rng, tier="easy"):
    """See module docstring for the full ladder and the scope.

    easy:   f(x) = k (uniform) on [c, d]; p, q strictly interior.
    medium: f(x) = k*(x - c) (linear ramp), k solved and shown; p, q interior.
    hard:   f(x) = k*(x - c)**2 (quadratic); p, q interior.
    exam:   same quadratic family, one of p/q set exactly to an endpoint
            of the support (c or d).
    """
    if tier == "easy":
        return _pd_easy(rng)
    if tier == "medium":
        return _pd_medium(rng)
    if tier == "hard":
        return _pd_hard(rng)
    return _pd_exam(rng)


# ============================================================================
# Skill 2: p19c_continuous_uniform_moments
# ============================================================================


def _sqrt_of_fraction(value: Fraction) -> Fraction:
    """Evaluate the square root of a nonnegative `Fraction` to 30 significant
    figures via sympy (the only irrational step) and capture it as an exact
    `Fraction` of that high-precision decimal string -- the same "carry the
    transcendental step as a `Fraction`, then round once at the end"
    discipline the `p13c_exponential_pdf_numint` precedent uses for its own
    `math.exp` step. Shared by the genuine SD(X) computation and by any
    distractor row that recomputes SD from its OWN (wrong) Var, so that row
    stays internally coherent end to end."""
    check(value >= 0, f"cannot take the square root of a negative value: {value}")
    if value == 0:
        return Fraction(0)
    sp_value = sp.sqrt(sp.Rational(value.numerator, value.denominator))
    return Fraction(str(sp_value.evalf(30)))


def _uniform_moments(a: int, b: int) -> tuple[Fraction, Fraction, Fraction]:
    """Plain-``Fraction`` construction route: E(X)=(a+b)/2, Var(X)=(b-a)**2/12,
    SD(X)=sqrt(Var(X))."""
    e = Fraction(a + b, 2)
    var = Fraction((b - a) ** 2, 12)
    sd = _sqrt_of_fraction(var)
    return e, var, sd


def _uniform_wrong_rows(a: int, b: int) -> list[tuple[Fraction, Fraction, Fraction]]:
    """Three distractor rows, each applying ONE coherent wrong METHOD to ALL THREE quantities, so each row is
    what one confused learner would actually produce.

    - `wrong_divisor`: Var(X) = (b-a)**2/6 instead of /12 (the wrong
      uniform-specific constant, used consistently), so SD is recomputed
      from THAT wrong Var; paired with E(X) = (b-a)/2 (mean-vs-range
      confusion) so E also differs from the key in this same row.
    - `sd_is_var_over_2`: SD(X) = Var(X)/2 (never takes the square root),
      paired with E(X) = (a+b)/3 (an arithmetic-mean-of-three-numbers slip
      applied here). A simpler version would take Var(X) here as
      the KEY's own (correct) Var(X); that would make this option's Var column
      value literally equal the key's every single time (a systematic,
      not merely occasional, column-wise repeat). Kept internally coherent
      instead by pairing the wrong-SD
      step with a DIFFERENT wrong divisor for Var, /4 instead of /12 (a
      distinct divisor slip from `wrong_divisor`'s /6), so this option's Var
      differs from the key's on every draw rather than by luck.
    - `e_is_b`: E(X) = b (endpoint instead of midpoint), paired with a
      squaring slip in Var -- (b-a) instead of (b-a)**2 -- before dividing
      by 12, and SD recomputed from THAT wrong Var.
    """
    L = b - a
    var_wd = Fraction(L * L, 6)
    row_wrong_divisor = (Fraction(L, 2), var_wd, _sqrt_of_fraction(var_wd))

    var_svo = Fraction(L * L, 4)
    row_sd_is_var_over_2 = (Fraction(a + b, 3), var_svo, var_svo / 2)

    var_eb = Fraction(L, 12)
    row_e_is_b = (Fraction(b), var_eb, _sqrt_of_fraction(var_eb))

    return [row_wrong_divisor, row_sd_is_var_over_2, row_e_is_b]


def _uniform_columns_pairwise_distinct(dp: int, rows: list[tuple[Fraction, Fraction, Fraction]]) -> bool:
    """True iff, once rounded and rendered to `dp` places, every column (E,
    Var, SD independently) across all `rows` (key + distractors) is
    pairwise distinct -- no column-wise repeat, hence no majority-vote tell.
    Compares the RENDERED strings, not the raw `Fraction`s: two different
    exact values that round to the same displayed decimal are just as much
    a majority tell as two values that are exactly equal."""
    for column in zip(*rows):
        strings = [stated_precision_string(round_half_up(v, dp), dp) for v in column]
        if len(set(strings)) != len(strings):
            return False
    return True


def _assert_uniform_columns_distinct(
    dp: int, e: Fraction, var: Fraction, sd: Fraction, cands: list[tuple[Fraction, Fraction, Fraction]]
) -> None:
    """Defense-in-depth: assert, in the generator itself (not only in a
    test), that every column's 4 rendered values (key + 3 distractors) are
    pairwise distinct -- using `check_pairwise_distinct` on the RENDERED
    (rounded-to-`dp`) strings, since a majority-vote tell arises from what a
    learner actually SEES, not from raw `Fraction` equality. The retry loop
    above already redraws (a, b) until this holds, so this should never
    fire; it exists so a future change that breaks the invariant fails loud
    here rather than only in a test."""
    rows = [(e, var, sd)] + cands
    for label, column in zip(("E", "Var", "SD"), zip(*rows)):
        strings = [stated_precision_string(round_half_up(v, dp), dp) for v in column]
        check_pairwise_distinct(
            *strings, msg=f"{label}(X) column has a repeated rendered value across key+distractors: {strings}"
        )


def _verify_uniform_moments(a: int, b: int, e: Fraction, var: Fraction, sd: Fraction) -> None:
    """Independent check: rebuild all three values from a, b via
    sympy.Rational/sympy.sqrt (a different code path from the plain-
    Fraction construction above) and assert exact/near-exact agreement."""
    a_sp, b_sp = sp.Integer(a), sp.Integer(b)
    e_sp = (a_sp + b_sp) / sp.Integer(2)
    var_sp = (b_sp - a_sp) ** 2 / sp.Integer(12)
    sd_sp = sp.sqrt(var_sp)
    check_equivalent(e, Fraction(str(sp.nsimplify(e_sp))), f"E(X) independent check failed a={a} b={b}")
    check_equivalent(var, Fraction(str(sp.nsimplify(var_sp))), f"Var(X) independent check failed a={a} b={b}")
    check(abs(sd - Fraction(str(sd_sp.evalf(30)))) < Fraction(1, 10**15), f"SD(X) independent check failed a={a} b={b}")


def _fmt_triple(e: Fraction, var: Fraction, sd: Fraction, dp: int) -> str:
    e_s = stated_precision_string(round_half_up(e, dp), dp)
    var_s = stated_precision_string(round_half_up(var, dp), dp)
    sd_s = stated_precision_string(round_half_up(sd, dp), dp)
    return f"({e_s}, {var_s}, {sd_s})"


_UNIFORM_TEMPLATES = (
    "A continuous random variable X is uniformly distributed on the interval [{a}, {b}]. "
    "Find E(X), Var(X) and SD(X), each correct to {dp} decimal places, giving your answer "
    "as the ordered triple (E(X), Var(X), SD(X)).",
    "X is a continuous random variable, uniformly distributed over [{a}, {b}]. Compute the "
    "expected value E(X), the variance Var(X), and the standard deviation SD(X), each "
    "correct to {dp} decimal places, as the ordered triple (E(X), Var(X), SD(X)).",
    "Suppose X is uniformly distributed on the interval from {a} to {b}. Determine E(X), "
    "Var(X) and SD(X), each correct to {dp} decimal places, in that order as (E(X), Var(X), SD(X)).",
)

_EXAM_UNIFORM_TEMPLATES = (
    "A component's lifetime (in hours), X, is uniformly distributed: it is guaranteed to "
    "last at least {a} hours and at most {b} hours. Find E(X), Var(X) and SD(X), each "
    "correct to {dp} decimal places, as the ordered triple (E(X), Var(X), SD(X)).",
    "A machine's output (in kilograms), X, is uniformly distributed between a guaranteed "
    "minimum of {a} kilograms and a maximum of {b} kilograms. Compute E(X), "
    "Var(X) and SD(X), each correct to {dp} decimal places, as (E(X), Var(X), SD(X)).",
    "A delivery time (in minutes), X, is uniformly distributed: it is never less than {a} "
    "minutes and never more than {b} minutes. Determine E(X), Var(X) and SD(X), each correct "
    "to {dp} decimal places, in that order as (E(X), Var(X), SD(X)).",
)


def _cum_finish(text: str, e: Fraction, var: Fraction, sd: Fraction, dp: int, distractor_triples: list) -> dict:
    answer_text = _fmt_triple(e, var, sd, dp)
    answer = [
        numeric_answer(round_half_up(e, dp)),
        numeric_answer(round_half_up(var, dp)),
        numeric_answer(round_half_up(sd, dp)),
    ]
    survivors = []
    for triple in distractor_triples:
        if triple is None:
            continue
        s = _fmt_triple(*triple, dp)
        if s == answer_text:
            continue
        survivors.append(s)
    distractors = pick_distractors(answer_text, survivors, unordered=False)
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
    )


def _cum_easy(rng) -> dict:
    """gap must be a multiple of 6 for Var(X) = gap**2/12 to be an EXACT
    integer (gap=6k -> Var = 3k**2), which is what genuinely makes E(X),
    Var(X) and SD(X) round to "clean" 2-d.p. decimals -- an even gap alone
    (e.g. gap=2 -> Var=1/3=0.333...) does NOT terminate, so "even" in the
    skill's own difficulty wording is honoured here as "multiple of
    6" (a superset of even), the actual discriminator against medium's
    non-terminating Var(X)."""
    dp = _DP2
    for _attempt in range(_MAX_ATTEMPTS):
        a = rng.randint(0, 20)
        gap = rng.choice((6, 12, 18))
        b = a + gap
        e, var, sd = _uniform_moments(a, b)
        if not (
            rounds_unambiguously(float(e), dp)
            and rounds_unambiguously(float(var), dp)
            and rounds_unambiguously(float(sd), dp)
        ):
            continue
        cands = _uniform_wrong_rows(a, b)
        if not _uniform_columns_pairwise_distinct(dp, [(e, var, sd)] + cands):
            continue
        break
    else:
        check(False, "easy: could not draw a question with 4 pairwise-distinct values per column")

    _verify_uniform_moments(a, b, e, var, sd)
    _assert_uniform_columns_distinct(dp, e, var, sd, cands)

    text = phrase(rng, _UNIFORM_TEMPLATES, a=str(a), b=str(b), dp=str(dp))
    return _cum_finish(text, e, var, sd, dp, cands)


def _cum_medium(rng) -> dict:
    """gap odd AND not a multiple of 3, so gap**2 shares no factor of 2 or 3
    with 12 -- Var(X) = gap**2/12 is genuinely non-terminating in decimal
    (e.g. gap=1 -> 1/12 = 0.08333...), the actual discriminator against
    easy's exact-integer Var(X)."""
    dp = _DP2
    for _attempt in range(_MAX_ATTEMPTS):
        a = rng.randint(0, 20)
        gap = rng.choice((1, 5, 7, 11))
        b = a + gap
        e, var, sd = _uniform_moments(a, b)
        if not (
            rounds_unambiguously(float(e), dp)
            and rounds_unambiguously(float(var), dp)
            and rounds_unambiguously(float(sd), dp)
        ):
            continue
        cands = _uniform_wrong_rows(a, b)
        if not _uniform_columns_pairwise_distinct(dp, [(e, var, sd)] + cands):
            continue
        break
    else:
        check(False, "medium: could not draw a question with 4 pairwise-distinct values per column")

    _verify_uniform_moments(a, b, e, var, sd)
    _assert_uniform_columns_distinct(dp, e, var, sd, cands)

    text = phrase(rng, _UNIFORM_TEMPLATES, a=str(a), b=str(b), dp=str(dp))
    return _cum_finish(text, e, var, sd, dp, cands)


def _cum_hard(rng) -> dict:
    dp = _DP2
    for _attempt in range(_MAX_ATTEMPTS):
        a = rng.randint(-15, -1)
        gap = rng.randint(2, 14)
        b = a + gap
        e, var, sd = _uniform_moments(a, b)
        if not (
            rounds_unambiguously(float(e), dp)
            and rounds_unambiguously(float(var), dp)
            and rounds_unambiguously(float(sd), dp)
        ):
            continue
        cands = _uniform_wrong_rows(a, b)
        if not _uniform_columns_pairwise_distinct(dp, [(e, var, sd)] + cands):
            continue
        break
    else:
        check(False, "hard: could not draw a question with 4 pairwise-distinct values per column")

    _verify_uniform_moments(a, b, e, var, sd)
    _assert_uniform_columns_distinct(dp, e, var, sd, cands)

    text = phrase(rng, _UNIFORM_TEMPLATES, a=str(a), b=str(b), dp=str(dp))
    return _cum_finish(text, e, var, sd, dp, cands)


def _cum_exam(rng) -> dict:
    """Interval stated indirectly through a short scenario -- the learner
    reads a, b off the described context before applying the same three
    closed-form formulas."""
    dp = _DP2
    for _attempt in range(_MAX_ATTEMPTS):
        # a >= 2 so the scenario never renders a singular-vs-plural unit
        # trap ("1 hours"/"1 minutes"/"1 kilograms").
        a = rng.randint(2, 20)
        gap = rng.randint(2, 20)
        b = a + gap
        e, var, sd = _uniform_moments(a, b)
        if not (
            rounds_unambiguously(float(e), dp)
            and rounds_unambiguously(float(var), dp)
            and rounds_unambiguously(float(sd), dp)
        ):
            continue
        cands = _uniform_wrong_rows(a, b)
        if not _uniform_columns_pairwise_distinct(dp, [(e, var, sd)] + cands):
            continue
        break
    else:
        check(False, "exam: could not draw a question with 4 pairwise-distinct values per column")

    _verify_uniform_moments(a, b, e, var, sd)
    _assert_uniform_columns_distinct(dp, e, var, sd, cands)

    text = phrase(rng, _EXAM_UNIFORM_TEMPLATES, a=str(a), b=str(b), dp=str(dp))
    return _cum_finish(text, e, var, sd, dp, cands)


@generator(
    "p19c_continuous_uniform_moments",
    level=15,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19c_continuous_uniform_moments(rng, tier="easy"):
    """See module docstring for the full ladder.

    easy:   a, b small nonnegative integers, b-a small and even.
    medium: b-a odd, so Var(X) is typically a non-terminating decimal.
    hard:   a negative (interval straddles or sits below zero).
    exam:   interval stated indirectly through a short scenario.
    """
    if tier == "easy":
        return _cum_easy(rng)
    if tier == "medium":
        return _cum_medium(rng)
    if tier == "hard":
        return _cum_hard(rng)
    return _cum_exam(rng)
