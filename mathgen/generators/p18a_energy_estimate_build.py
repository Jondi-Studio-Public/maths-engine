"""For a supplied single separated heat-equation mode on a stated interval, with
its boundary conditions and the sign of the diffusion coefficient stated in
the question, evaluate the energy E(t) = the integral of the square of the
solution over the interval, as a closed-form expression in t. Scope:
one mode, one value --
never a bound, a multi-mode sum, or a justification of the assumptions.

Starts from ``ucv1-heat-equation-mode`` (which computes the mode itself):
here the mode is always GIVEN in full (or, at the exam tier, given only
through its initial condition), and the one new step is squaring it and
integrating over the interval.

Key shape (Dirichlet/Neumann/mixed, n or n+1/2 >= 1/2):
    E(t) = (A^2 * L / 2) * exp(-2 k lambda t),  lambda = (m*pi/L)^2
so that L/2 (the integral of sin^2/cos^2 of a full set of half-periods) is
the SAME constant for every family and every interval offset -- the
escalation across tiers lives in reading which lambda and which family the
stated boundary conditions give, never in the integral itself. The n = 0
Neumann mode is a genuine change in kind: u is constant, so
E(t) = A^2 * L, with no decay term at all.

Independent check (`_verify_energy`, not a restatement of `_energy_key`):
integrates sin^2/cos^2 of the GENERIC linear argument via the explicit
double-angle antiderivative (theta/2 -/+ sin(2*theta)/4, bounds substituted
directly) rather than calling `sympy.integrate`'s general dispatcher -- a
different, cheap route (sub-millisecond) that still re-derives L/2 (or L for
the constant mode) numerically for THIS draw's own m/L/a, not a hardcoded
constant.
"""

from __future__ import annotations

from fractions import Fraction as Fr

import sympy as sp
from sympy import Rational, cos, exp, pi, simplify, sin, symbols, sympify

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
    symbolically_equal,
)

_MAX_ATTEMPTS = 500

_X, _T = symbols("x t")

_A_POOL = tuple(range(2, 10))  # amplitude, A >= 2 so "forgot to square" can never
# coincide with the answer (A = 1 would collapse it)
_K_POOL = tuple(range(1, 7))  # diffusion coefficient, k > 0 (stated in the prompt)
_N_POOL = tuple(range(1, 6))  # mode number, n >= 1 (Dirichlet / Neumann)
_N0_POOL = tuple(range(0, 5))  # mixed-condition mode number, n >= 0 (m = n + 1/2 >= 1/2)
_L_POOL = tuple(range(2, 10))  # interval length
_OFFSET_POOL = tuple(v for v in range(-6, 7) if v != 0)  # interval start a != 0


def _fmt(value) -> str:
    """Canonical display string for `answer_text`/a distractor: Python's "**"
    power notation rewritten to "^" (grader-parseable per
    `mathgen/grading/typed.py::_parse_expression`, required by `double_star`).

    `sympify`, never `simplify`: every value this generator builds is already
    a fully-reduced `Rational * exp(Rational * pi**2 * t)` product straight
    from sympy's own auto-simplification on construction (`str()` before and
    after a `simplify()` pass is identical, checked directly) -- `simplify`'s
    generic exptrigsimp pass on an `exp(pi**2 * ...)` expression measured
    3-65 ms *per call* here for no display change at all, and was this
    generator's whole speed problem (see the module docstring)."""
    return str(sympify(value)).replace("**", "^")


def _working_fmt(text: str) -> str:
    """Rewrite a bare "*" to an interpunct for the `working` field --, `working` is checked by every
    `text_quality` pattern regardless of its declared scope, so an `answer_text` string ("25*exp(...)")
    embedded verbatim into a worked step trips `star_operator` exactly as a hand-written "3*x" would."""
    return text.replace("*", "·")


def _lambda_value(family, m, L):
    """`lambda = (m*pi/L)^2` for every decaying family; 0 for the constant
    (n = 0 Neumann) mode, which never decays."""
    if family == "neumann-zero":
        return sp.Integer(0)
    return (m * pi / L) ** 2


def _energy_key(A, family, m, L, k):
    """The construction route: plug the memorised L/2 (or L, for the constant
    mode) directly into the formula."""
    if family == "neumann-zero":
        return sympify(A**2 * L)
    lam = _lambda_value(family, m, L)
    return sympify(Rational(1, 2) * A**2 * L * exp(-2 * k * lam * _T))


def _antideriv_bounds(m, L, a, *, cosine: bool) -> sp.Expr:
    """Independent route: the explicit double-angle antiderivative of
    sin^2/cos^2(m*pi*(x-a)/L) evaluated at the stated bounds [a, a+L] --
    never `sympy.integrate`'s general dispatcher, and never the L/2 constant
    `_energy_key` assumes. `theta(x) = m*pi*(x-a)/L`, `dtheta/dx = m*pi/L`, so
    the x-antiderivative is `(L/(m*pi)) * F(theta)` with
    `F(u) = u/2 -/+ sin(2u)/4` (cos^2 takes the "+" sign)."""
    scale = L / (m * pi)
    sign = 1 if cosine else -1

    def F(u):
        return scale * (u / 2 + sign * sin(2 * u) / 4)

    theta_hi = m * pi * ((a + L) - a) / L  # = m*pi (theta at x = a + L)
    theta_lo = m * pi * (a - a) / L  # = 0     (theta at x = a)
    return simplify(F(theta_hi) - F(theta_lo))


def _verify_energy(A, family, m, L, k, a=0):
    """Independent proof: re-derive E(t) from a second computational route
    and check it against `_energy_key`'s formula-based answer."""
    if family == "neumann-zero":
        # u(x,t) = A (constant): E(t) = the integral of A^2 over [0, L] = A^2*L,
        # by direct definition -- nothing further to re-derive independently.
        check(
            symbolically_equal(_energy_key(A, family, m, L, k), A**2 * L),
            "neumann-zero: constant-mode energy is not A^2 L",
        )
        return
    cosine = family == "neumann"
    half_period_integral = _antideriv_bounds(m, L, a, cosine=cosine)
    check(
        symbolically_equal(half_period_integral, L / 2), f"{family}: independent double-angle integral did not give L/2"
    )
    lam = _lambda_value(family, m, L)
    independent = A**2 * exp(-2 * k * lam * _T) * half_period_integral
    check(
        symbolically_equal(independent, _energy_key(A, family, m, L, k)),
        f"{family}: independent energy re-derivation disagrees with the formula route",
    )


def _verify_boundary(family, m, L, a=0):
    """Independent check, from GENERIC symbols (never the numeric mode
    construction builds): the stated boundary condition actually holds for
    this family's spatial factor."""
    n_s = symbols("n_s", integer=True, nonnegative=True)
    L_s = symbols("L_s", positive=True)
    a_s = symbols("a_s", real=True)
    if family == "dirichlet":
        spatial = sin(n_s * pi * (_X - a_s) / L_s)
        check(symbolically_equal(spatial.subs(_X, a_s), 0), "boundary check failed at x=a (Dirichlet)")
        check(symbolically_equal(spatial.subs(_X, a_s + L_s), 0), "boundary check failed at x=a+L (Dirichlet)")
    elif family == "neumann":
        spatial = cos(n_s * pi * (_X - a_s) / L_s)
        d = sp.diff(spatial, _X)
        check(symbolically_equal(d.subs(_X, a_s), 0), "boundary check failed at x=a (Neumann)")
        check(symbolically_equal(d.subs(_X, a_s + L_s), 0), "boundary check failed at x=a+L (Neumann)")
    elif family == "mixed":
        spatial = sin((n_s + Rational(1, 2)) * pi * (_X - a_s) / L_s)
        check(symbolically_equal(spatial.subs(_X, a_s), 0), "mixed: not Dirichlet at x=a")
        d = sp.diff(spatial, _X)
        check(symbolically_equal(d.subs(_X, a_s + L_s), 0), "mixed: not Neumann at x=a+L")
    elif family == "neumann-zero":
        pass  # a constant trivially satisfies u_x = 0 everywhere
    else:
        raise ValueError(family)


def _energy_pair(A, family, m, L, k):
    """The answer as a plain `(c, pi_power, rate)` tuple: value =
    `c * exp(-rate * pi**pi_power * t)`, or (by convention) `rate = 0` with
    `pi_power = 0` for a value that does not decay at all (the constant
    n = 0 Neumann mode). Pure `fractions.Fraction`/`int` arithmetic -- no
    sympy involved -- so this is cheap to build and cheap to compare; see
    `_offer`'s docstring for why comparing these tuples element-wise is a
    COMPLETE distinctness test for this family."""
    A, L, k = Fr(A), Fr(L), Fr(k)
    if family == "neumann-zero":
        return (A**2 * L, 0, Fr(0))
    m = Fr(m)
    lam_over_pi2 = (m / L) ** 2  # lambda = lam_over_pi2 * pi**2
    return (Fr(1, 2) * A**2 * L, 2, 2 * k * lam_over_pi2)


def _misconception_pairs(A, family, m, L, k):
    """Every real slip this generator offers, as `(c, pi_power, rate)`
    tuples in `_energy_pair`'s representation -- one dict entry per
    misconception, keyed by its label."""
    A, L, k = Fr(A), Fr(L), Fr(k)
    if family == "neumann-zero":
        return {
            "forgot-square": (A * L, 0, Fr(0)),
            "half-wrongly-applied": (Fr(1, 2) * A**2 * L, 0, Fr(0)),
        }
    m = Fr(m)
    lam_over_pi2 = (m / L) ** 2  # lambda = lam_over_pi2 * pi**2
    half = Fr(1, 2)
    return {
        "forgot-square": (A * L * half, 2, 2 * k * lam_over_pi2),
        "missing-half": (A**2 * L, 2, 2 * k * lam_over_pi2),
        "exponent-not-doubled": (A**2 * L * half, 2, k * lam_over_pi2),
        # lambda kept linear in pi (never squared): the exponent decays as
        # pi**1, not pi**2 -- a different power of a transcendental, so a
        # different KIND of value, never a coincidental numeric collision.
        "lambda-unsquared": (A**2 * L * half, 1, 2 * k * (m / L)),
        # lambda missing pi entirely: the exponent decays as pi**0 (a plain
        # rational rate, no pi at all) -- again a different power of pi.
        "lambda-no-pi": (A**2 * L * half, 0, 2 * k * lam_over_pi2),
    }


def _pair_to_value(pair):
    """Build the sympy display value for a `(c, pi_power, rate)` tuple --
    called only for the (at most 5) candidates that survive distinctness,
    never inside the hot pairwise-comparison loop."""
    c, pi_power, rate = pair
    if rate == 0:
        return sympify(c)
    return sympify(c * exp(-rate * pi**pi_power * _T))


def _offer(answer_pair, pair, offered, formatted, label):
    """`pair` is a `(c, pi_power, rate)` tuple of plain
    `fractions.Fraction`/`int` values -- never a sympy expression built from
    `pi`/`exp`. Since pi is transcendental, two candidates
    `c1*exp(-r1*pi**p1*t)` and `c2*exp(-r2*pi**p2*t)` with r1, r2 != 0 are
    equal as functions of t iff p1 == p2 AND c1 == c2 AND r1 == r2: a ratio
    `pi**(p1-p2)` of two DIFFERENT integer powers of pi is irrational, so
    p1 != p2 alone already rules out equality whenever r1, r2 are both
    nonzero rationals. Every rate this generator ever builds for a decaying
    candidate is strictly positive (k in 1..6, lambda > 0 for every decaying
    family), and every non-decaying candidate is built with `rate = 0` AND
    `pi_power = 0` by convention (never any other `pi_power` at rate 0) --
    so plain Python/Fraction tuple equality is a COMPLETE distinctness
    criterion here, with no symbolic exp/pi arithmetic needed at all
    (this was the whole cost of `make()`, see the module
    docstring's history)."""
    if pair == answer_pair or pair in offered:
        return
    check_distinct(answer_pair, pair, msg=f"{label} collided with the answer")
    check_pairwise_distinct(*offered, pair, msg=f"{label} collided with another candidate")
    offered.append(pair)
    formatted.append(_fmt(_pair_to_value(pair)))


def _offer_misconceptions(answer, answer_pair, A, family, m, L, k, offered, formatted):
    # The ONE symbolic cross-check per draw: `answer_pair`
    # (the fast Fraction-tuple route used for distinctness) and `answer` (the
    # sympy route `_fmt` formats into the GRADED `answer_text`) are two
    # independent implementations of the same formula -- tie them together
    # once here so a slip in either one is caught, never silently accepted
    # (without this, a disagreement between the two routes could let a
    # distractor collide with the real graded key while `_offer`'s own
    # Fraction-tuple comparison sees no collision at all).
    check_equivalent(_pair_to_value(answer_pair), answer, "pair/key routes disagree")
    for label, pair in _misconception_pairs(A, family, m, L, k).items():
        _offer(answer_pair, pair, offered, formatted, label)


# ---------------------------------------------------------------------------
# Prompt text (hand-built per parameter, never `str(sympy_expr)` -- the
# `(x - a)` offset expands into an ugly `sin(pi*(3x/7 + 6/7))` under sympy's
# own auto-simplification, which no learner should have to read).
# ---------------------------------------------------------------------------


def _x_term(a) -> str:
    """`x`, `(x - a)` for a > 0, `(x + |a|)` for a < 0 -- never the doubled
    sign `(x - (-3))` a bare `f"(x - ({a}))"` produces for a negative offset."""
    if a == 0:
        return "x"
    if a > 0:
        return f"(x - {a})"
    return f"(x + {-a})"


def _coef_prefix_suffix(frac: Fr) -> tuple[str, str]:
    """The `("", "")`/`("7·", "")`/`("", "/12")`/`("7·", "/12")` pieces that
    surround `pi` for a reduced coefficient `frac` -- drop a numerator of 1
    and a denominator of 1 rather than print `1·pi` or `pi/1` (this is also how a
    mixed-condition `n + 1/2` wavenumber and an `n == L` mode reduce down to `pi·x/10`, `7·pi·x/12` or
    `pi·x` instead of `(0 + 1/2)·pi·x/5` or `5·pi·x/5`)."""
    num, den = frac.numerator, frac.denominator
    prefix = "" if num == 1 else f"{num}·"
    suffix = "" if den == 1 else f"/{den}"
    return prefix, suffix


def _sin_cos_arg_text(trig: str, m, L, a=0) -> str:
    """`sin`/`cos` of the REDUCED wavenumber `m/L` (never the raw, possibly
    non-reduced `m` and `L` separately) times `pi·(x [- a])`."""
    prefix, suffix = _coef_prefix_suffix(Fr(m) / L)
    return f"{trig}({prefix}pi·{_x_term(a)}{suffix})"


def _exp_arg_text(k, m, L) -> str:
    """`exp(-k*lambda*t)` with `lambda = (m*pi/L)^2`, displayed as a single
    REDUCED coefficient of `pi^2·t` (e.g. `exp(-45·pi^2·t/4)`) rather than
    the raw `k·m^2·pi^2·t/L^2` -- never `1^2` for `m == 1`."""
    prefix, suffix = _coef_prefix_suffix(Fr(k) * Fr(m) ** 2 / Fr(L) ** 2)
    return f"exp(-{prefix}pi^2·t{suffix})"


def _mode_text_dirichlet(A, n, L, k, a=0):
    return f"{A}·{_sin_cos_arg_text('sin', n, L, a)}·{_exp_arg_text(k, n, L)}"


def _mode_text_neumann(A, n, L, k):
    return f"{A}·{_sin_cos_arg_text('cos', n, L)}·{_exp_arg_text(k, n, L)}"


def _mode_text_mixed(A, n, L, k):
    m = n + Fr(1, 2)
    return f"{A}·{_sin_cos_arg_text('sin', m, L)}·{_exp_arg_text(k, m, L)}"


_EASY_PATTERNS = [
    "The 1-D heat equation u_t = k u_xx holds on [0, {L}] with k = {k} (k > 0) and Dirichlet "
    "ends u(0,t) = u({L},t) = 0. One separated mode is u(x,t) = {mode}. Find its energy "
    "E(t), the integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
    "On the interval [0, {L}], u(x,t) satisfies u_t = k u_xx (k = {k}, k > 0) with fixed "
    "(Dirichlet) ends u(0,t) = u({L},t) = 0. Given the mode u(x,t) = {mode}, compute its "
    "energy E(t) = the integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
    "A rod occupies [0, {L}] with Dirichlet ends u(0,t) = u({L},t) = 0 and obeys "
    "u_t = k u_xx, k = {k} (k > 0). Its temperature is the single mode u(x,t) = {mode}. "
    "Find E(t), the integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
]

_MEDIUM_PATTERNS = [
    "The 1-D heat equation u_t = k u_xx holds on [0, {L}] with k = {k} (k > 0) and insulated "
    "(Neumann) ends u_x(0,t) = u_x({L},t) = 0. One separated mode is u(x,t) = {mode}. Find "
    "its energy E(t), the integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
    "On [0, {L}], u(x,t) satisfies u_t = k u_xx (k = {k}, k > 0) with insulated ends "
    "u_x(0,t) = u_x({L},t) = 0. Given the mode u(x,t) = {mode}, compute its energy "
    "E(t) = the integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
    "A rod [0, {L}] has insulated ends u_x(0,t) = u_x({L},t) = 0 and obeys u_t = k u_xx, "
    "k = {k} (k > 0). Its temperature is the single mode u(x,t) = {mode}. Find E(t), the "
    "integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
]

_HARD_MIXED_PATTERNS = [
    "The 1-D heat equation u_t = k u_xx holds on [0, {L}] with k = {k} (k > 0), and mixed "
    "boundary conditions u(0,t) = 0 (fixed) and u_x({L},t) = 0 (insulated). One separated "
    "mode is u(x,t) = {mode}. Find its energy E(t), the integral of u(x,t)^2 over [0, {L}], "
    "as a closed-form expression in t.",
    "On [0, {L}], u(x,t) satisfies u_t = k u_xx (k = {k}, k > 0) with mixed boundary "
    "conditions: u(0,t) = 0 and u_x({L},t) = 0. Given the mode u(x,t) = {mode}, compute its "
    "energy E(t) = the integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
    "A rod [0, {L}] has one fixed end (u(0,t) = 0) and one insulated end (u_x({L},t) = 0), "
    "and obeys u_t = k u_xx, k = {k} (k > 0). Its temperature is the single mode "
    "u(x,t) = {mode}. Find E(t), the integral of u(x,t)^2 over [0, {L}], as a closed-form "
    "expression in t.",
]

_HARD_OFFSET_PATTERNS = [
    "The 1-D heat equation u_t = k u_xx holds on [{a}, {b}] (an interval of length {L}) with "
    "k = {k} (k > 0) and Dirichlet ends u({a},t) = u({b},t) = 0. One separated mode is "
    "u(x,t) = {mode}. Find its energy E(t), the integral of u(x,t)^2 over [{a}, {b}], as a "
    "closed-form expression in t.",
    "On the interval [{a}, {b}] (length {L}), u(x,t) satisfies u_t = k u_xx (k = {k}, k > 0) "
    "with fixed ends u({a},t) = u({b},t) = 0. Given the mode u(x,t) = {mode}, compute its "
    "energy E(t) = the integral of u(x,t)^2 over [{a}, {b}], as a closed-form expression in t.",
    "A rod occupies [{a}, {b}] (length {L}) with Dirichlet ends u({a},t) = u({b},t) = 0 and "
    "obeys u_t = k u_xx, k = {k} (k > 0). Its temperature is the single mode u(x,t) = {mode}. "
    "Find E(t), the integral of u(x,t)^2 over [{a}, {b}], as a closed-form expression in t.",
]

_EXAM_IC_PATTERNS = [
    "The 1-D heat equation u_t = k u_xx holds on [0, {L}] with k = {k} (k > 0) and Dirichlet "
    "ends u(0,t) = u({L},t) = 0. At t = 0, the temperature is u(x,0) = {ic}; this is itself "
    "one separated mode. Write its time evolution u(x,t), then find its energy E(t), the "
    "integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
    "On [0, {L}], u(x,t) satisfies u_t = k u_xx (k = {k}, k > 0) with fixed ends "
    "u(0,t) = u({L},t) = 0, and its initial profile u(x,0) = {ic} is one separated mode. "
    "Determine u(x,t) first, then compute its energy E(t) = the integral of u(x,t)^2 over "
    "[0, {L}], as a closed-form expression in t.",
    "A rod [0, {L}] has Dirichlet ends u(0,t) = u({L},t) = 0 and obeys u_t = k u_xx, "
    "k = {k} (k > 0); at t = 0 its temperature is the single mode u(x,0) = {ic}. Decide the "
    "mode's decay rate, then find E(t), the integral of u(x,t)^2 over [0, {L}], as a "
    "closed-form expression in t.",
]

_EXAM_ZERO_PATTERNS = [
    "The 1-D heat equation u_t = k u_xx holds on [0, {L}] with k = {k} (k > 0) and insulated "
    "(Neumann) ends u_x(0,t) = u_x({L},t) = 0. One separated mode is the constant mode "
    "u(x,t) = {A} (mode number n = 0, which never decays). Find its energy E(t), the "
    "integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
    "On [0, {L}], u(x,t) satisfies u_t = k u_xx (k = {k}, k > 0) with insulated ends "
    "u_x(0,t) = u_x({L},t) = 0. The n = 0 mode is constant in space and time: u(x,t) = {A}. "
    "Compute its energy E(t) = the integral of u(x,t)^2 over [0, {L}], as a closed-form "
    "expression in t.",
    "A rod [0, {L}] has insulated ends u_x(0,t) = u_x({L},t) = 0 and obeys u_t = k u_xx, "
    "k = {k} (k > 0). Its n = 0 mode never decays: u(x,t) = {A} for all t. Find E(t), the "
    "integral of u(x,t)^2 over [0, {L}], as a closed-form expression in t.",
]


def _p18a_energy_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        A = rng.choice(_A_POOL)
        k = rng.choice(_K_POOL)
        n = rng.choice(_N_POOL)
        L = rng.choice(_L_POOL)
        family = "dirichlet"
        _verify_boundary(family, n, L)
        answer = _energy_key(A, family, n, L, k)
        _verify_energy(A, family, n, L, k)
        answer_pair = _energy_pair(A, family, n, L, k)

        offered, formatted = [], []
        _offer_misconceptions(answer, answer_pair, A, family, n, L, k, offered, formatted)
        if len(formatted) < 2:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 2:
            continue

        mode = _mode_text_dirichlet(A, n, L, k)
        text = phrase(rng, _EASY_PATTERNS, L=L, k=k, mode=mode)
        working = [
            f"E(t) = A^2·(L/2)·exp(-2·k·lambda·t), lambda = (n·pi/L)^2 = ({n}·pi/{L})^2. "
            f"E(t) = {_working_fmt(answer_text)}."
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p18a_energy_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        A = rng.choice(_A_POOL)
        k = rng.choice(_K_POOL)
        n = rng.choice(_N_POOL)
        L = rng.choice(_L_POOL)
        family = "neumann"
        _verify_boundary(family, n, L)
        answer = _energy_key(A, family, n, L, k)
        _verify_energy(A, family, n, L, k)
        answer_pair = _energy_pair(A, family, n, L, k)

        offered, formatted = [], []
        _offer_misconceptions(answer, answer_pair, A, family, n, L, k, offered, formatted)
        if len(formatted) < 2:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 2:
            continue

        mode = _mode_text_neumann(A, n, L, k)
        text = phrase(rng, _MEDIUM_PATTERNS, L=L, k=k, mode=mode)
        working = [
            f"The integral of cos^2(n·pi·x/L) over [0, L] is still L/2. E(t) = A^2·(L/2)·"
            f"exp(-2·k·lambda·t), lambda = ({n}·pi/{L})^2. E(t) = {_working_fmt(answer_text)}."
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p18a_energy_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        variant = rng.choice(["mixed", "offset"])
        A = rng.choice(_A_POOL)
        k = rng.choice(_K_POOL)
        L = rng.choice(_L_POOL)

        if variant == "mixed":
            n = rng.choice(_N0_POOL)
            m = n + Rational(1, 2)
            family = "mixed"
            _verify_boundary(family, m, L)
            answer = _energy_key(A, family, m, L, k)
            _verify_energy(A, family, m, L, k)
            answer_pair = _energy_pair(A, family, m, L, k)

            offered, formatted = [], []
            _offer_misconceptions(answer, answer_pair, A, family, m, L, k, offered, formatted)
            if len(formatted) < 2:
                continue
            answer_text = _fmt(answer)
            distractors = pick_distractors(answer_text, formatted)
            if len(distractors) < 2:
                continue

            mode = _mode_text_mixed(A, n, L, k)
            text = phrase(rng, _HARD_MIXED_PATTERNS, L=L, k=k, mode=mode)
            working = [
                f"Dirichlet-at-0/Neumann-at-L forces lambda from (n + 1/2), not n: "
                f"lambda = (({n} + 1/2)·pi/{L})^2. The integral of sin^2 is still L/2. "
                f"E(t) = {_working_fmt(answer_text)}."
            ]
            return dict(
                text=text,
                latex=text,
                answer=answer_text,
                answer_text=answer_text,
                working=working,
                distractors=distractors,
            )
        else:
            n = rng.choice(_N_POOL)
            a = rng.choice(_OFFSET_POOL)
            family = "dirichlet"
            _verify_boundary(family, n, L, a)
            answer = _energy_key(A, family, n, L, k)
            _verify_energy(A, family, n, L, k, a)
            answer_pair = _energy_pair(A, family, n, L, k)

            offered, formatted = [], []
            _offer_misconceptions(answer, answer_pair, A, family, n, L, k, offered, formatted)
            if len(formatted) < 2:
                continue
            answer_text = _fmt(answer)
            distractors = pick_distractors(answer_text, formatted)
            if len(distractors) < 2:
                continue

            mode = _mode_text_dirichlet(A, n, L, k, a)
            text = phrase(rng, _HARD_OFFSET_PATTERNS, a=a, b=a + L, L=L, k=k, mode=mode)
            working = [
                f"Shifting the interval to [{a}, {a + L}] does not change the integral of "
                f"sin^2 over a full set of half-periods -- it is still L/2. "
                f"E(t) = {_working_fmt(answer_text)}."
            ]
            return dict(
                text=text,
                latex=text,
                answer=answer_text,
                answer_text=answer_text,
                working=working,
                distractors=distractors,
            )
    raise RuntimeError("hard: could not draw a usable question")


def _p18a_energy_exam(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        variant = rng.choice(["ic", "zero"])
        A = rng.choice(_A_POOL)
        k = rng.choice(_K_POOL)
        L = rng.choice(_L_POOL)

        if variant == "ic":
            n = rng.choice(_N_POOL)
            family = "dirichlet"
            _verify_boundary(family, n, L)
            answer = _energy_key(A, family, n, L, k)
            _verify_energy(A, family, n, L, k)
            answer_pair = _energy_pair(A, family, n, L, k)

            offered, formatted = [], []
            _offer_misconceptions(answer, answer_pair, A, family, n, L, k, offered, formatted)
            if len(formatted) < 2:
                continue
            answer_text = _fmt(answer)
            distractors = pick_distractors(answer_text, formatted)
            if len(distractors) < 2:
                continue

            ic = f"{A}·{_sin_cos_arg_text('sin', n, L)}"
            mode = _mode_text_dirichlet(A, n, L, k)
            text = phrase(rng, _EXAM_IC_PATTERNS, L=L, k=k, ic=ic)
            working = [
                f"Dirichlet + u_t = k u_xx forces u(x,t) = {mode} before integrating. "
                f"E(t) = {_working_fmt(answer_text)}."
            ]
            return dict(
                text=text,
                latex=text,
                answer=answer_text,
                answer_text=answer_text,
                working=working,
                distractors=distractors,
            )
        else:
            family = "neumann-zero"
            answer = _energy_key(A, family, 0, L, k)
            _verify_energy(A, family, 0, L, k)
            answer_pair = _energy_pair(A, family, 0, L, k)

            offered, formatted = [], []
            _offer_misconceptions(answer, answer_pair, A, family, 0, L, k, offered, formatted)
            if len(formatted) < 2:
                continue
            answer_text = _fmt(answer)
            distractors = pick_distractors(answer_text, formatted)
            if len(distractors) < 2:
                continue

            text = phrase(rng, _EXAM_ZERO_PATTERNS, L=L, k=k, A=A)
            working = [
                f"n = 0 gives lambda = 0: the mode never decays and is constant in space, "
                f"so E(t) = A^2·L (no 1/2 factor, since the constant squares to a "
                f"constant integrand). E(t) = {_working_fmt(answer_text)}."
            ]
            return dict(
                text=text,
                latex=text,
                answer=answer_text,
                answer_text=answer_text,
                working=working,
                distractors=distractors,
            )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p18a_energy_estimate",
    level=19,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p18a_energy_estimate(rng, tier="easy"):
    if tier == "easy":
        return _p18a_energy_easy(rng)
    if tier == "medium":
        return _p18a_energy_medium(rng)
    if tier == "hard":
        return _p18a_energy_hard(rng)
    if tier == "exam":
        return _p18a_energy_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
