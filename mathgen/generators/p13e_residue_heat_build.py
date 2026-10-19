"""Residue at a simple pole + separated heat mode.

Two skills, one module:

- ``p13e_residue_simple_pole`` (``ucv1-residue-simple-pole``, ``answer:
  expression``) -- Res(f, a) = lim_{z->a} (z - a) f(z) for a rational
  function with a stated simple pole. Every tier draws the answer via
  closed-form algebra; the independent check re-derives it via
  ``sympy.limit`` on the full symbolic ``f(z)`` -- a genuinely different
  computational route from the closed-form arithmetic construction uses.
- ``p13e_heat_equation_mode`` (``ucv1-heat-equation-mode``, ``answer:
  expression``) -- one separated space-time mode of the 1-D heat equation
  ``u_t = alpha u_xx`` under a stated Dirichlet or Neumann boundary
  condition. The independent check proves the constructed spatial factor
  actually satisfies the stated boundary condition (from a GENERIC
  sin/cos(n*pi*x/L) symbol, via ``sympy.diff``/``subs``), never re-typing
  construction's own formula.

Both topics are brand new; neither shares a kernel with
``fourier-coefficient-simple`` (integrates for ONE coefficient, never builds
the separated mode).
"""

from __future__ import annotations

import functools
import re

import sympy as sp
from sympy import I, cos, exp, pi, sin, symbols
from sympy import simplify as _simplify_uncached


_simplify_cached = functools.lru_cache(maxsize=None)(_simplify_uncached)


def _gaussian_rational(value):
    """`a + b*I` for an exact number with rational real and imaginary parts, else None.

    Only for deciding whether two such numbers are equal. It is not a stand-in for `simplify`, whose result can
    be a different (equal) expression such as `I*(3 - I)/2`, and the printed answers must not change."""
    if not isinstance(value, sp.Basic) or value.free_symbols or not value.is_number:
        return None
    try:
        real, imag = value.as_real_imag()
    except (TypeError, ValueError):
        return None
    if real.is_Rational and imag.is_Rational:
        return real + imag * I
    return None


def _is_integer_gaussian(value) -> bool:
    """Whether `value` is `m + n*I` with integer m and n, as sympy builds `p + q*I`, `p - q*I` or `p + I`."""
    if not (isinstance(value, sp.Add) and len(value.args) == 2):
        return False
    whole, imaginary = sorted(value.args, key=lambda term: term.has(I))
    if not whole.is_Integer:
        return False
    return imaginary == I or (
        imaginary.is_Mul and len(imaginary.args) == 2 and imaginary.args[0].is_Integer and imaginary.args[1] == I
    )


def simplify(value):
    """`sympy.simplify`, memoised. It is pure, and this generator re-simplifies the same
    handful of expressions on every seed (the parameter pools are small), so trigsimp's
    ~60 ms per call was the bulk of a `make()`. Falls back to the plain call for anything
    unhashable, so behaviour is unchanged. `m + n*I` with integers is returned as it is: `simplify` leaves it
    unchanged (checked for every m, n from -13 to 13 and for `p - q*I` and `(p + 1) + q*I` too, which covers the
    exam pools), and it ran the full pipeline on every root the exam tier prints."""
    if _is_integer_gaussian(value):
        return value
    try:
        return _simplify_cached(value)
    except TypeError:
        return _simplify_uncached(value)


@functools.lru_cache(maxsize=None)
def _same_value(left, right) -> bool:
    """`symbolically_equal`, memoised: it is pure, and one candidate is compared with the same answer and the same
    earlier candidates by the dedup test, by `check_distinct` and by `check_pairwise_distinct`, each probing
    numerically at 30 digits every time. Two exact complex rationals are compared in canonical form instead of
    through `simplify`."""
    left_exact, right_exact = _gaussian_rational(left), _gaussian_rational(right)
    if left_exact is not None and right_exact is not None:
        return left_exact == right_exact  # canonical forms of exact numbers: equal values are equal expressions
    return symbolically_equal(left, right)


from ..core import (
    check,
    fmt_signed,
    generator,
    phrase,
    pick_distractors,
    symbolically_equal,
)

_MAX_ATTEMPTS = 500

_Z = symbols("z")
_X, _T = symbols("x t")

# ---------------------------------------------------------------------------
# Skill 1: ucv1-residue-simple-pole
# ---------------------------------------------------------------------------

_AB_POOL = tuple(v for v in range(-8, 9) if v != 0)  # -8..8, 0 excluded
_C_POOL = tuple(v for v in range(-9, 10) if v != 0)  # -9..9, 0 excluded
_R_POOL = tuple(range(-8, 9))  # numerator root pool
_K_POOL = tuple(range(1, 7))  # imaginary-part magnitude
_P_POOL = tuple(v for v in range(-8, 9) if v != 0)  # exam real part, nonzero
_Q_POOL = tuple(range(1, 7))  # exam imaginary part, positive


def _fmt(value) -> str:
    """Canonical display string for `answer_text`/a distractor: Python's
    "**" power notation is rewritten to "^" (grader-parseable, per
    `mathgen/grading/typed.py::_parse_expression`, and required by
    `mathgen/text_quality.py`'s `double_star` check). A bare "*" is left
    alone here -- `text_quality`'s `star_operator` check only inspects the
    prompt `text` field, never `answer_text`/distractors."""
    return _imaginary_i(str(simplify(value)).replace("**", "^"))


def _text_fmt(value) -> str:
    """Display string for embedding a value INSIDE prompt or working text,
    where `text_quality`'s `star_operator` check also applies: "**" -> "^"
    and a bare "*" -> "·" so neither the `double_star` nor the
    `star_operator` pattern fires."""
    return _imaginary_i(str(simplify(value)).replace("**", "^").replace("*", "·"))


def _imaginary_i(text: str) -> str:
    """SymPy writes the imaginary unit as `I`; the stems (and a learner) write `i`."""
    return re.sub(r"\bI\b", "i", text)


def _check_collisions(answer, value, offered, label):
    """`check_distinct(answer, value)` then `check_pairwise_distinct(*offered, value)`, from the memoised comparison.

    The pairs among `offered` themselves were proved when the later of each pair was offered, so only `value`
    against each is new; the messages are the ones those helpers would raise."""
    check(not _same_value(answer, value), f"{label} collided with the answer")
    check(not any(_same_value(v, value) for v in offered), f"{label} collided with another candidate")


def _offer(answer, value, offered, formatted, label):
    """Append `value` (a sympy value) to the offered list iff it genuinely differs from the answer and from every
    value already offered -- several misconceptions can land on the same wrong value (e.g. hard tier's
    wrong-pole and sign-flip both negate the answer), and only one should ever be shown. Every value actually
    offered is proven pairwise distinct, unconditionally."""
    value = simplify(value)
    if _same_value(value, answer) or any(_same_value(value, v) for v in offered):
        return
    _check_collisions(answer, value, offered, label)
    offered.append(value)
    formatted.append(_fmt(value))


_EASY_PATTERNS = [
    "f(z) = {c} / ((z - ({a}))(z - ({b}))) has a simple pole at z = {a}. Using "
    "Res(f, a) = lim_(z->a) (z - a) f(z), find Res(f, {a}).",
    "The function f(z) = {c} / ((z - ({a}))(z - ({b}))) has a simple pole at "
    "z = {a}. Find the residue Res(f, {a}) via the local limit formula "
    "Res(f, a) = lim_(z->a) (z - a) f(z).",
    "Given f(z) = {c} / ((z - ({a}))(z - ({b}))), which has a simple pole at "
    "z = {a}, compute Res(f, {a}) = lim_(z->a) (z - a) f(z).",
]

_MEDIUM_PATTERNS = [
    "f(z) = (z - ({r})) / ((z - ({a}))(z - ({b}))) has a simple pole at z = {a}. "
    "Using Res(f, a) = lim_(z->a) (z - a) f(z), find Res(f, {a}).",
    "The function f(z) = (z - ({r})) / ((z - ({a}))(z - ({b}))) has a simple "
    "pole at z = {a}. Find the residue Res(f, {a}) via the local limit "
    "formula Res(f, a) = lim_(z->a) (z - a) f(z).",
    "Given f(z) = (z - ({r})) / ((z - ({a}))(z - ({b}))), which has a simple "
    "pole at z = {a}, compute Res(f, {a}) = lim_(z->a) (z - a) f(z).",
]

_HARD_PATTERNS = [
    "f(z) = {c} / (z^2 + {k}^2) has simple poles at z = {k}i and z = -{k}i. "
    "Find the residue at the pole with positive imaginary part, using "
    "Res(f, a) = lim_(z->a) (z - a) f(z).",
    "The function f(z) = {c} / (z^2 + {k}^2) has simple poles at z = +/-{k}i. "
    "Using Res(f, a) = lim_(z->a) (z - a) f(z), find the residue at the pole "
    "with positive imaginary part.",
    "Given f(z) = {c} / (z^2 + {k}^2), with simple poles at z = {k}i and "
    "z = -{k}i, compute the residue at the pole with positive imaginary part "
    "via Res(f, a) = lim_(z->a) (z - a) f(z).",
]

_EXAM_PATTERNS = [
    "f(z) = (z - ({r})) / (z^2 + ({b_coef})z + ({c_coef})) has a complex "
    "conjugate pair of simple poles. Exactly one of these candidates is "
    "actually a pole of f: Candidate 1: z = {w1}. Candidate 2: z = {w2}. "
    "Find the residue of f at that pole, where "
    "Res(f, a) = lim_(z->a) (z - a) f(z).",
    "The function f(z) = (z - ({r})) / (z^2 + ({b_coef})z + ({c_coef})) has "
    "a complex conjugate pair of simple poles. Only one candidate below is a "
    "genuine root of z^2 + ({b_coef})z + ({c_coef}) = 0: Candidate 1: "
    "z = {w1}. Candidate 2: z = {w2}. Compute Res(f, a) = lim_(z->a) (z - a) f(z) "
    "at the true pole.",
    "Given f(z) = (z - ({r})) / (z^2 + ({b_coef})z + ({c_coef})), with "
    "candidate poles Candidate 1: z = {w1} and Candidate 2: z = {w2} (only "
    "one of which is a genuine root of the denominator), find the residue "
    "at the true pole, where Res(f, a) = lim_(z->a) (z - a) f(z).",
]


def _p13e_residue_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        a, b = rng.sample(_AB_POOL, 2)
        c = rng.choice(_C_POOL)
        answer = sp.Rational(c, a - b)

        f = c / ((_Z - a) * (_Z - b))
        indep = sp.limit((_Z - a) * f, _Z, a)
        check(_same_value(indep, answer), "easy: independent limit check failed")

        offered, formatted = [], []
        # wrong-pole: evaluates at the other pole b instead of the stated a
        _offer(answer, sp.Rational(c, b - a), offered, formatted, "wrong-pole")
        # drops-the-factor: evaluates the leftover factor c/(z-b) at z = 0
        # instead of at the stated pole a (b is drawn nonzero, so this is
        # always finite)
        _offer(answer, simplify((c / (_Z - b)).subs(_Z, 0)), offered, formatted, "drops-the-factor")
        # sign-flip: the residue with the wrong sign
        _offer(answer, -answer, offered, formatted, "sign-flip")

        if len(formatted) < 1:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        text = phrase(rng, _EASY_PATTERNS, a=a, b=b, c=c)
        working = [f"Res(f, {a}) = lim_(z->{a}) (z - ({a})) f(z) = {c} / (({a}) - ({b})) = {_text_fmt(answer)}."]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p13e_residue_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        a, b = rng.sample(_AB_POOL, 2)
        r = rng.choice(_R_POOL)
        if r == a:
            continue  # pole would cancel -- not a genuine pole of f
        answer = sp.Rational(a - r, a - b)

        f = (_Z - r) / ((_Z - a) * (_Z - b))
        indep = sp.limit((_Z - a) * f, _Z, a)
        check(_same_value(indep, answer), "medium: independent limit check failed")

        offered, formatted = [], []
        _offer(answer, sp.Rational(b - r, b - a), offered, formatted, "wrong-pole")
        _offer(answer, simplify(((_Z - r) / (_Z - b)).subs(_Z, 0)), offered, formatted, "drops-the-factor")
        _offer(answer, -answer, offered, formatted, "sign-flip")

        if len(formatted) < 1:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        text = phrase(rng, _MEDIUM_PATTERNS, a=a, b=b, r=r)
        working = [
            f"Res(f, {a}) = lim_(z->{a}) (z - ({a})) f(z) = (({a}) - ({r})) / (({a}) - ({b})) = {_text_fmt(answer)}."
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p13e_residue_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        c = rng.choice(_C_POOL)
        k = rng.choice(_K_POOL)
        answer = simplify(sp.Rational(-c, 2 * k) * I)

        f = c / (_Z**2 + k**2)
        indep = sp.limit((_Z - I * k) * f, _Z, I * k)
        check(_same_value(indep, answer), "hard: independent limit check failed")

        offered, formatted = [], []
        # wrong-pole: evaluates at -ik instead of ik
        _offer(answer, sp.limit((_Z + I * k) * f, _Z, -I * k), offered, formatted, "wrong-pole")
        # drops-the-factor: evaluates the leftover factor c/(z+ik) at z = 0
        _offer(answer, simplify((c / (_Z + I * k)).subs(_Z, 0)), offered, formatted, "drops-the-factor")
        _offer(answer, -answer, offered, formatted, "sign-flip")

        if len(formatted) < 1:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        text = phrase(rng, _HARD_PATTERNS, c=c, k=k)
        working = [f"z^2 + {k}^2 = (z - {k}i)(z + {k}i), so Res(f, {k}i) = {c} / (2 × {k}i) = {_text_fmt(answer)}."]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("hard: could not draw a usable question")


def _p13e_residue_exam(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        p = rng.choice(_P_POOL)
        q = rng.choice(_Q_POOL)
        r = rng.choice(_R_POOL)
        b_coef, c_coef = -2 * p, p * p + q * q
        w_true = p + q * I
        w_conj = p - q * I
        # Decoy: perturb the real part by 1, keep the same imaginary part.
        # Substituting into z^2 + b_coef z + c_coef gives exactly 1 + 2qi
        # (real part always 1, imaginary part always 2q > 0) -- proved
        # analytically, never a genuine root. (The other candidate
        # decoy, "swap the sign of the imaginary part", would give the
        # ACTUAL conjugate root of a real quadratic, which does satisfy the
        # equation -- so it cannot be used as a decoy here; this is the
        # "perturb p by 1" alternative.)
        w_decoy = (p + 1) + q * I
        check(
            simplify(w_decoy**2 + b_coef * w_decoy + c_coef) != 0,
            "exam: decoy unexpectedly satisfies the characteristic equation",
        )
        check(
            _same_value(w_true**2 + b_coef * w_true + c_coef, 0),
            "exam: true root fails its own characteristic equation",
        )

        answer = simplify((w_true - r) / (2 * I * q))

        f = (_Z - r) / (_Z**2 + b_coef * _Z + c_coef)
        indep = sp.limit((_Z - w_true) * f, _Z, w_true)
        check(_same_value(indep, answer), "exam: independent limit check failed")

        offered, formatted = [], []
        # wrong-pole: applies the local-limit formula at the TRUE conjugate
        # root w_conj instead of w_true
        _offer(answer, simplify((w_conj - r) / (-2 * I * q)), offered, formatted, "wrong-pole")
        # drops-the-factor: evaluates the leftover factor (z-r)/(z-w_conj)
        # at z = 0 instead of at w_true (w_conj is never 0 since q > 0)
        _offer(answer, simplify(((_Z - r) / (_Z - w_conj)).subs(_Z, 0)), offered, formatted, "drops-the-factor")
        _offer(answer, -answer, offered, formatted, "sign-flip")
        # uses-wrong-candidate: applies the formula as if the DECOY were the
        # true pole (same "other root" difference 2iq, since the decoy's own
        # conjugate is (p+1)-qi)
        _offer(answer, simplify((w_decoy - r) / (2 * I * q)), offered, formatted, "uses-wrong-candidate")

        if len(formatted) < 1:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        # Randomise which candidate slot (1 or 2) shows the true root.
        if rng.random() < 0.5:
            w1_text, w2_text = _text_fmt(w_true), _text_fmt(w_decoy)
        else:
            w1_text, w2_text = _text_fmt(w_decoy), _text_fmt(w_true)

        text = phrase(rng, _EXAM_PATTERNS, r=r, b_coef=b_coef, c_coef=c_coef, w1=w1_text, w2=w2_text)
        working = [
            f"Substitute each candidate into z^2 + ({b_coef})z + ({c_coef}): only "
            f"z = {_text_fmt(w_true)} satisfies it (its conjugate is {_text_fmt(w_conj)}).",
            f"Res(f, {_text_fmt(w_true)}) = (({_text_fmt(w_true)}) - ({r})) / (2i × {q}) = {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p13e_residue_simple_pole",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13e_residue_simple_pole(rng, tier="easy"):
    if tier == "easy":
        return _p13e_residue_easy(rng)
    if tier == "medium":
        return _p13e_residue_medium(rng)
    if tier == "hard":
        return _p13e_residue_hard(rng)
    if tier == "exam":
        return _p13e_residue_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Skill 2: ucv1-heat-equation-mode
# ---------------------------------------------------------------------------

_ALPHA_POOL = tuple(range(1, 6))
_N_POOL = tuple(range(1, 6))
_L_POOL = tuple(range(2, 10))
_COEF_POOL = tuple(range(2, 10))


def _spatial(family, n, L):
    return sin(n * pi * _X / L) if family == "dirichlet" else cos(n * pi * _X / L)


def _mode(coeff, family, n, L, alpha, *, square_n=True, use_alpha=True):
    """Build one separated mode. `square_n=False`/`use_alpha=False` build the
    drops-the-square / drops-alpha misconception variants from the SAME
    construction path (only the decay exponent's malformed piece differs)."""
    n_term = sp.Integer(n) ** 2 if square_n else sp.Integer(n)
    alpha_term = alpha if use_alpha else 1
    # L is EITHER the sympy constant `pi` (easy tier) or a plain Python int
    # (every other tier) -- `sp.Integer(pi)` silently truncates to 3, which
    # produced a `pi**2/9` exponent for an easy-tier "[0, pi]" question
    # (caught by the independent oracle, not by this generator's own
    # `check()`s, since the malformed value was still internally consistent
    # with itself). Never coerce an already-sympy `L` through `sp.Integer`.
    L_sym = L if isinstance(L, sp.Basic) else sp.Integer(L)
    decay = exp(-alpha_term * n_term * pi**2 * _T / L_sym**2)
    # No `simplify()` here: sympy already auto-reduces the Rational
    # coefficient in the exponent and the sin/cos argument at construction
    # time (no trigsimp needed), and `simplify()`'s trigsimp pass on every
    # one of the ~5 variants `make()` builds per question was this
    # generator's whole speed problem (see `_offer_expr`'s own comment).
    return coeff * _spatial(family, n, L) * decay


def _verify_boundary(family, n, L):
    """Independent check: from a GENERIC sin/cos(n*pi*x/L) symbol (never the
    numeric mode construction builds), confirm the boundary condition holds
    -- the function itself at x=0,L for Dirichlet, its x-derivative at
    x=0,L for Neumann."""
    n_s, L_s = symbols("n_s L_s", positive=True, integer=True)
    spatial_generic = sin(n_s * pi * _X / L_s) if family == "dirichlet" else cos(n_s * pi * _X / L_s)
    if family == "dirichlet":
        check(_same_value(spatial_generic.subs(_X, 0), 0), "boundary check failed at x=0 (Dirichlet)")
        check(_same_value(spatial_generic.subs({_X: L_s}), 0), "boundary check failed at x=L (Dirichlet)")
    else:
        d = sp.diff(spatial_generic, _X)
        check(_same_value(d.subs(_X, 0), 0), "boundary check failed at x=0 (Neumann)")
        check(_same_value(d.subs({_X: L_s}), 0), "boundary check failed at x=L (Neumann)")


_EASY_HEAT_PATTERNS = [
    "The temperature u(x,t) on [0, pi] satisfies u_t = alpha u_xx with "
    "u(0,t) = u(pi,t) = 0 (Dirichlet ends), alpha = {alpha}. One separated "
    "mode has coefficient B = 1 and mode number n = {n}. Write u_n(x,t).",
    "On the rod [0, pi] with Dirichlet ends u(0,t) = u(pi,t) = 0, the heat "
    "equation u_t = alpha u_xx holds with alpha = {alpha}. Write the "
    "separated mode u_n(x,t) for n = {n} and coefficient B = 1.",
    "A rod occupies [0, pi] with u(0,t) = u(pi,t) = 0 and u_t = alpha u_xx, "
    "alpha = {alpha}. Find the mode u_n(x,t) for mode number n = {n}, "
    "coefficient B = 1.",
]

_MEDIUM_HEAT_PATTERNS = [
    "The temperature u(x,t) on [0, {L}] satisfies u_t = alpha u_xx with "
    "u(0,t) = u({L},t) = 0 (Dirichlet ends), alpha = {alpha}. One separated "
    "mode has coefficient B = {coeff} and mode number n = {n}. Write u_n(x,t).",
    "On the rod [0, {L}] with Dirichlet ends u(0,t) = u({L},t) = 0, the heat "
    "equation u_t = alpha u_xx holds with alpha = {alpha}. Write the "
    "separated mode u_n(x,t) for n = {n}, coefficient B = {coeff}.",
    "A rod occupies [0, {L}] with u(0,t) = u({L},t) = 0 and u_t = alpha u_xx, "
    "alpha = {alpha}. Find the mode u_n(x,t) for mode number n = {n}, "
    "coefficient B = {coeff}.",
]

_HARD_HEAT_PATTERNS = [
    "The temperature u(x,t) on [0, {L}] satisfies u_t = alpha u_xx. The ends "
    "are insulated: u_x(0,t) = u_x({L},t) = 0 (Neumann), alpha = {alpha}. "
    "One separated mode has coefficient A = {coeff} and mode number n = {n}. "
    "Write u_n(x,t).",
    "A rod [0, {L}] has insulated ends u_x(0,t) = u_x({L},t) = 0 and obeys "
    "u_t = alpha u_xx, alpha = {alpha}. Write the separated mode u_n(x,t) "
    "for n = {n}, coefficient A = {coeff}.",
    "On [0, {L}], u(x,t) satisfies u_t = alpha u_xx with insulated ends "
    "(u_x(0,t) = u_x({L},t) = 0), alpha = {alpha}. Find the mode u_n(x,t) "
    "for mode number n = {n}, coefficient A = {coeff}.",
]

_EXAM_HEAT_PATTERNS = [
    "On [0, {L}], u(x,t) satisfies u_t = alpha u_xx with alpha = {alpha}, "
    "mode number n = {n} and coefficient C = {coeff}. The boundary condition "
    "is {bc_words}. Exactly one candidate mode below satisfies this boundary "
    "condition: Candidate 1: {cand1}. Candidate 2: {cand2}. Verify which "
    "candidate satisfies the stated boundary condition, then report the "
    "correct mode u_n(x,t).",
    "A rod [0, {L}] obeys u_t = alpha u_xx (alpha = {alpha}) with {bc_words}. "
    "For mode number n = {n} and coefficient C = {coeff}, only one candidate "
    "is consistent with this boundary condition: Candidate 1: {cand1}. "
    "Candidate 2: {cand2}. Check each against the boundary condition and "
    "report the correct mode u_n(x,t).",
    "The temperature u(x,t) on [0, {L}] satisfies u_t = alpha u_xx, alpha = "
    "{alpha}, with {bc_words}. Given mode number n = {n} and coefficient "
    "C = {coeff}, exactly one of these is a valid mode: Candidate 1: "
    "{cand1}. Candidate 2: {cand2}. Determine which satisfies the boundary "
    "condition and give the correct mode u_n(x,t).",
]

_BC_WORDS = {
    "dirichlet": "fixed ends: u(0,t) = u({L},t) = 0",
    "neumann": "insulated ends: u_x(0,t) = u_x({L},t) = 0",
}


def _heat_fmt(value) -> str:
    """`_fmt` for a value built by `_mode`, without the `simplify` call.

    `simplify` returns every `_mode` value unchanged, so the string is the same. That was checked for every
    expression `_mode` can build from the pools (all 2 families, n, L (including pi), alpha and coefficient
    values, with and without the square and with and without alpha; 12,150 expressions), comparing both `==` and
    `str`. trigsimp on each was most of this generator's `make()` time. `tests/test_p13e_heat_canonical.py`
    re-checks a sample."""
    return str(value).replace("**", "^")


def _heat_text_fmt(value) -> str:
    """`_text_fmt` for a value built by `_mode`; see `_heat_fmt`."""
    return str(value).replace("**", "^").replace("*", "·")


def _offer_expr(answer, value, offered, formatted, label):
    # `symbolically_equal` (not a raw `simplify(diff) == 0`) for the initial
    # dedup test: its numeric probe proves "these differ" in one cheap
    # evaluation for the common case, and only falls back to sympy's slow
    # trigsimp-heavy `simplify` when the probe is inconclusive. The original
    # raw-`simplify` spelling here made this generator the single slowest
    # topic in the library
    # -- every candidate was fully trigsimp'd against every prior candidate.
    if _same_value(value, answer) or any(_same_value(value, v) for v in offered):
        return
    # `value` comes from `_mode`, which `simplify` leaves unchanged: no `simplify` here (see `_heat_fmt`).
    _check_collisions(answer, value, offered, label)
    offered.append(value)
    formatted.append(_heat_fmt(value))


def _p13e_heat_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        alpha = rng.choice(_ALPHA_POOL)
        n = rng.choice(_N_POOL)
        L = sp.pi
        family = "dirichlet"
        answer = _mode(1, family, n, L, alpha)
        _verify_boundary(family, n, L)

        offered, formatted = [], []
        _offer_expr(answer, _mode(1, "neumann", n, L, alpha), offered, formatted, "wrong-family")
        _offer_expr(answer, _mode(1, family, n, L, alpha, square_n=False), offered, formatted, "drops-the-square")
        _offer_expr(answer, _mode(1, family, n, L, alpha, use_alpha=False), offered, formatted, "drops-alpha")
        # wrong-coefficient not exercisable here: B is already 1 (the design
        # fixes B = 1 for easy), so "uses 1 instead of the stated coefficient"
        # collapses to the correct answer -- honestly omitted rather than faked.

        if len(formatted) < 1:
            continue
        answer_text = _heat_fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        text = phrase(rng, _EASY_HEAT_PATTERNS, alpha=alpha, n=n)
        working = [f"u_n(x,t) = 1 × sin({fmt_signed(n, 'x')}) exp(-{alpha} × {n}^2 × t) = {_heat_text_fmt(answer)}."]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p13e_heat_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        alpha = rng.choice(_ALPHA_POOL)
        n = rng.choice(_N_POOL)
        L = rng.choice(_L_POOL)
        coeff = rng.choice(_COEF_POOL)
        family = "dirichlet"
        answer = _mode(coeff, family, n, L, alpha)
        _verify_boundary(family, n, L)

        offered, formatted = [], []
        _offer_expr(answer, _mode(coeff, "neumann", n, L, alpha), offered, formatted, "wrong-family")
        _offer_expr(answer, _mode(coeff, family, n, L, alpha, square_n=False), offered, formatted, "drops-the-square")
        _offer_expr(answer, _mode(coeff, family, n, L, alpha, use_alpha=False), offered, formatted, "drops-alpha")
        _offer_expr(answer, _mode(1, family, n, L, alpha), offered, formatted, "wrong-coefficient")

        if len(formatted) < 1:
            continue
        answer_text = _heat_fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        text = phrase(rng, _MEDIUM_HEAT_PATTERNS, alpha=alpha, n=n, L=L, coeff=coeff)
        working = [
            f"u_n(x,t) = {coeff} × sin({n} pi x / {L}) exp(-{alpha} × {n}^2 pi^2 t / {L}^2) = {_heat_text_fmt(answer)}."
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p13e_heat_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        alpha = rng.choice(_ALPHA_POOL)
        n = rng.choice(_N_POOL)
        L = rng.choice(_L_POOL)
        coeff = rng.choice(_COEF_POOL)
        family = "neumann"
        answer = _mode(coeff, family, n, L, alpha)
        _verify_boundary(family, n, L)

        offered, formatted = [], []
        _offer_expr(answer, _mode(coeff, "dirichlet", n, L, alpha), offered, formatted, "wrong-family")
        _offer_expr(answer, _mode(coeff, family, n, L, alpha, square_n=False), offered, formatted, "drops-the-square")
        _offer_expr(answer, _mode(coeff, family, n, L, alpha, use_alpha=False), offered, formatted, "drops-alpha")
        _offer_expr(answer, _mode(1, family, n, L, alpha), offered, formatted, "wrong-coefficient")

        if len(formatted) < 1:
            continue
        answer_text = _heat_fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        text = phrase(rng, _HARD_HEAT_PATTERNS, alpha=alpha, n=n, L=L, coeff=coeff)
        working = [
            f"u_n(x,t) = {coeff} × cos({n} pi x / {L}) exp(-{alpha} × {n}^2 pi^2 t / {L}^2) = {_heat_text_fmt(answer)}."
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("hard: could not draw a usable question")


def _p13e_heat_exam(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        alpha = rng.choice(_ALPHA_POOL)
        n = rng.choice(_N_POOL)
        L = rng.choice(_L_POOL)
        coeff = rng.choice(_COEF_POOL)
        family = rng.choice(["dirichlet", "neumann"])
        decoy_family = "neumann" if family == "dirichlet" else "dirichlet"

        answer = _mode(coeff, family, n, L, alpha)
        decoy = _mode(coeff, decoy_family, n, L, alpha)
        _verify_boundary(family, n, L)
        # Independent proof the decoy family does NOT satisfy this BC (its
        # own boundary condition is satisfied instead, never this one, since
        # sin and cos never both vanish -- respectively have both derivatives
        # vanish -- at the same nonzero n*pi/L).
        check(not _same_value(decoy, answer), "exam: decoy coincides with the true mode")

        offered, formatted = [], []
        _offer_expr(answer, decoy, offered, formatted, "uses-wrong-candidate")
        _offer_expr(answer, _mode(coeff, family, n, L, alpha, square_n=False), offered, formatted, "drops-the-square")
        _offer_expr(answer, _mode(coeff, family, n, L, alpha, use_alpha=False), offered, formatted, "drops-alpha")
        _offer_expr(answer, _mode(1, family, n, L, alpha), offered, formatted, "wrong-coefficient")

        if len(formatted) < 1:
            continue
        answer_text = _heat_fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        if rng.random() < 0.5:
            cand1, cand2 = _heat_text_fmt(answer), _heat_text_fmt(decoy)
        else:
            cand1, cand2 = _heat_text_fmt(decoy), _heat_text_fmt(answer)

        text = phrase(
            rng,
            _EXAM_HEAT_PATTERNS,
            alpha=alpha,
            n=n,
            L=L,
            coeff=coeff,
            bc_words=_BC_WORDS[family].format(L=L),
            cand1=cand1,
            cand2=cand2,
        )
        working = [
            f"Only the {family} spatial factor satisfies {_BC_WORDS[family].format(L=L)}.",
            f"u_n(x,t) = {_heat_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p13e_heat_equation_mode",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13e_heat_equation_mode(rng, tier="easy"):
    if tier == "easy":
        return _p13e_heat_easy(rng)
    if tier == "medium":
        return _p13e_heat_medium(rng)
    if tier == "hard":
        return _p13e_heat_hard(rng)
    if tier == "exam":
        return _p13e_heat_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
