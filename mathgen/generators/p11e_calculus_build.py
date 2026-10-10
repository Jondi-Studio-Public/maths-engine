"""Generates `derivative_definition` and `double_integral_rectangle` questions: a derivative from the limit
definition, and a double integral over a rectangle.

The calculus pair.

Two `university-core-v1` skills, one generator module, two structurally
unrelated calculus operations that happen to sit either side of a
single-variable/multivariable boundary:

- ``ucv1-derivative-definition`` (``single-variable-calculus``, topic
  ``derivative_definition``): evaluate f'(a) at a stated point ``a`` for a
  stated CUBIC polynomial ``f``, using the limit definition of the
  derivative -- never the power rule as the named method, never an
  expression as the answer. This is deliberately CUBIC at every tier, never
  a plain quadratic: the closest neighbour on the shelf,
  ``senior-examine-the-behaviour-of-the-difference-quotient-frac``
  (ACMMM081, `mathgen/generators/p9c_difference_quotient_build.py`), is
  hard-capped to a quadratic by its own generator (``a = rng.randrange(1,
  6)`` always squared, never cubic) and reads a numeric TREND table of
  shrinking-h values rather than presenting the symbolic limit definition
  directly, so a cubic row does something that skill structurally cannot on
  top of the presentational difference. It is also never a duplicate of
  ``differentiate-polynomial`` (`mathgen/generators/calculus.py`'s
  ``differentiation`` topic): that skill's answer is the derivative
  EXPRESSION itself (``answer: expression``), formed by the power rule and
  never evaluated at a point; this skill's answer is a NUMBER, the value of
  the derivative at a stated point, formed by the limit definition.

- ``ucv1-double-integral-rectangle`` (``multivariable-and-vector-calculus``,
  topic ``double_integral_rectangle``): evaluate the iterated double
  integral of a SEPARABLE polynomial ``f(x, y) = g(x) h(y)`` over an
  axis-aligned rectangle, by factoring into two one-variable integrals and
  multiplying. Every answer is an exact integer by construction (even
  interval widths, and for the hard tier's quadratic factor a leading
  coefficient that is a multiple of 3), never a change of variables and
  never a non-rectangular region. Not a duplicate of either shelf-mate in
  `multivariable-and-vector-calculus` (`directional-derivative`,
  `jacobian-two-variable`, both in `mathgen/generators/
  university_foundations.py`): neither of those is an integral at all, one
  evaluates a directional derivative and the other a Jacobian matrix.
"""

from __future__ import annotations

from fractions import Fraction

import sympy
from sympy import expand, symbols

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

# Module-scope symbols, shared by both topics below (never reused across a
# single question's own construction and its independent check with a
# different meaning).
_x, _h = symbols("x h")
_y = symbols("y")

_MAX_DRAW_ATTEMPTS = 500


def _poly_display(expr) -> str:
    """Learner-facing rendering of a sympy polynomial: ``**`` -> ``^``, the bare multiplication asterisk dropped
    entirely (``mathgen/text_quality.py``'s ``star_operator`` check would flag a literal ``*``). Mirrors
    `mathgen/generators/university_foundations.py`'s own `_poly_display` -- not imported from there."""
    return str(expr).replace("**", "^").replace("*", "")


# ---------------------------------------------------------------------------
# ucv1-derivative-definition: evaluate f'(a) from the limit definition


def _nonzero_randint(rng, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


_R_LO, _R_HI = -6, 6
_S_LO, _S_HI = -6, 6
_Q_LO, _Q_HI = -6, 6
_A_LO, _A_HI = -4, 4
_HARD_P_CHOICES = (2, 3, -2, -3)


def _draw_cubic(rng, tier: str):
    """(p, q, r, s, a) for f(x) = p*x^3 + q*x^2 + r*x + s, evaluated at a.

    r is drawn nonzero at every tier: r = 0 would make the `drops-linear-term`
    misconception (the derivative formula with r dropped) numerically
    IDENTICAL to the true answer, which is a duplicate-correct-option defect
    rather than a wrong-but-plausible distractor.
    """
    if tier == "easy":
        p, q = 1, 0
    elif tier == "medium":
        p, q = 1, _nonzero_randint(rng, _Q_LO, _Q_HI)
    else:
        p = rng.choice(_HARD_P_CHOICES)
        q = _nonzero_randint(rng, _Q_LO, _Q_HI)
    r = _nonzero_randint(rng, _R_LO, _R_HI)
    s = rng.randint(_S_LO, _S_HI)
    a = rng.randint(_A_LO, _A_HI)
    return p, q, r, s, a


def _cubic_expr(p: int, q: int, r: int, s: int):
    return p * _x**3 + q * _x**2 + r * _x + s


def _limit_derivative_at(f_expr, a: int) -> int:
    """f'(a) via the limit definition: expand the difference quotient (which
    algebraically cancels the common factor of h for any polynomial -- the
    raw, uncancelled quotient would raise a division error at h = 0), then
    substitute x = a and h = 0 on the CANCELLED expression. Never
    `sympy.limit`, which is measurably slower under the per-topic sweep load
    for a case that always cancels cleanly."""
    difference_quotient = expand((f_expr.subs(_x, _x + _h) - f_expr) / _h)
    at_a = difference_quotient.subs(_x, a)
    limit_value = at_a.subs(_h, 0)
    check(limit_value.is_Integer, f"limit is not an integer at a={a}: {f_expr}")
    return int(limit_value)


@generator(
    "derivative_definition",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def derivative_definition(rng, tier="easy"):
    """Evaluate f'(a) at a stated point from the limit definition, for a
    cubic f -- never a plain quadratic (see the module docstring's boundary
    against ACMMM081).

    easy:   f(x) = x^3 + r*x + s (monic, no x^2 term). Answer 3*a^2 + r.
    medium: f(x) = x^3 + q*x^2 + r*x + s, q != 0. Answer 3*a^2 + 2*q*a + r.
    hard:   f(x) = p*x^3 + q*x^2 + r*x + s, p in {2, 3, -2, -3}, q != 0.
            Answer 3*p*a^2 + 2*q*a + r.
    """
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        p, q, r, s, a = _draw_cubic(rng, tier)
        f_expr = _cubic_expr(p, q, r, s)
        answer = _limit_derivative_at(f_expr, a)

        # Independent check: the closed-form power-rule value computed
        # directly from the coefficients -- plain Python-int arithmetic, a
        # genuinely different route from the symbolic difference quotient
        # above, never `sympy.diff` (the same symbolic-differentiation
        # machinery under another name).
        power_rule_value = 3 * p * a * a + 2 * q * a + r
        check_equivalent(
            answer,
            power_rule_value,
            f"independent power-rule check failed: p={p}, q={q}, r={r}, a={a}",
        )

        f_at_a = p * a**3 + q * a**2 + r * a + s
        f_at_a_plus_1 = p * (a + 1) ** 3 + q * (a + 1) ** 2 + r * (a + 1) + s
        candidates = {
            "evaluates-function-not-derivative": f_at_a,
            "off-by-one-point": 3 * p * (a + 1) * (a + 1) + 2 * q * (a + 1) + r,
            "h-not-taken-to-zero": f_at_a_plus_1 - f_at_a,
            "drops-linear-term": 3 * p * a * a + 2 * q * a,
        }
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=(f"misconception collision: tier={tier}, p={p}, q={q}, r={r}, s={s}, a={a}"),
            )
        except VerificationError:
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free derivative_definition question")

    answer_text = str(answer)
    f_text = _poly_display(f_expr)
    question_text = f"f(x) = {f_text}. Using the limit definition f'(a) = lim(h→0) [f(a+h) - f(a)] / h, find f'({a})."
    # n=4: all four misconceptions are pairwise-distinct from the answer and
    # from each other by construction, so pick_distractors's default n=3 cap
    # would silently and DETERMINISTICALLY drop `drops-linear-term` on every
    # single seed (it is always fourth in candidate order and the first
    # three never collide) -- every misconception should be exercisable in
    # the actually-offered options, not just checked in-generator.
    distractors = pick_distractors(answer_text, [str(value) for value in candidates.values()], n=4)
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer),
        working=[
            f"(f(a+h) - f(a))/h expands to a polynomial in h with the common factor of h "
            f"cancelled algebraically; substituting h = 0 at a = {a} gives f'({a}) = {answer_text}.",
        ],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# ucv1-double-integral-rectangle: iterated integral of a separable polynomial


_SHIFT_LO, _SHIFT_HI = -3, 3
_COEF_LO, _COEF_HI = -3, 3
_BOUND_LO, _BOUND_HI = -3, 3
_WIDTH_CHOICES = (2, 4)
# Easy tier only: the design fixes g(x) = x, h(y) = y, a1 = a2 = 0 at this tier, leaving only
# (w1, w2) to vary the question. With w1, w2 each drawn from the same {2, 4} pool as medium/hard
# that is only 4 distinct ordered pairs -- and so only 4 distinct rendered questions over any number
# of seeds, short of the shared `_MIN_DISTINCT = 10` floor that every registered `*_build` topic is
# expected to meet, with no exemption. Widened to four even
# choices for this tier only -- medium and hard keep the literal {2, 4}.
_WIDTH_CHOICES_EASY = (2, 4, 6, 8)
_HARD_LEADING_CHOICES = (3, 6, -3, -6)


def _antiderivative(p: int, q: int, r: int, t) -> Fraction:
    """F(t) = (p/3)t^3 + (q/2)t^2 + r*t for g(x) = p*x^2 + q*x + r -- the
    closed-form power-rule antiderivative, evaluated at a SINGLE point (not
    yet a definite integral). Shared by `_definite_integral` below (the net
    difference over an interval, always an integer by construction) and the
    `forgets-lower-bound` misconception (a single endpoint value, NOT
    guaranteed to be an integer -- see `double_integral_rectangle`'s own
    resampling comment)."""
    t = Fraction(t)
    return Fraction(p, 3) * t**3 + Fraction(q, 2) * t**2 + Fraction(r) * t


def _definite_integral(p: int, q: int, r: int, lo: int, hi: int) -> Fraction:
    """Exact definite integral of p*t^2 + q*t + r from lo to hi via the
    closed-form antiderivative F(t) = (p/3)t^3 + (q/2)t^2 + r*t -- the actual
    term-by-term power-rule "manual" route this skill teaches."""
    return _antiderivative(p, q, r, hi) - _antiderivative(p, q, r, lo)


def _draw_rectangle_integral(rng, tier: str):
    """One resampled draw of (gp, gq, gr, h_const, a1, a2, w1, w2, Ix, Iy).

    g(x) = gp*x^2 + gq*x + gr; h(y) = y + h_const (unit y-coefficient at
    every tier). w1/w2 are even so `\\int x dx` over an integer-bounded
    interval of that width is always an integer; the hard tier's `gp` is a
    nonzero multiple of 3 so `\\int gp*x^2 dx` is too, regardless of parity.
    """
    if tier == "easy":
        gp, gq, gr = 0, 1, 0
        h_const = 0
        a1, a2 = 0, 0
    elif tier == "medium":
        gp, gq = 0, 1
        gr = _nonzero_randint(rng, _COEF_LO, _COEF_HI)
        h_const = _nonzero_randint(rng, _SHIFT_LO, _SHIFT_HI)
        a1 = rng.randint(_BOUND_LO, _BOUND_HI)
        a2 = rng.randint(_BOUND_LO, _BOUND_HI)
    else:
        gp = rng.choice(_HARD_LEADING_CHOICES)
        gq = rng.randint(_COEF_LO, _COEF_HI)
        gr = rng.randint(_COEF_LO, _COEF_HI)
        h_const = _nonzero_randint(rng, _SHIFT_LO, _SHIFT_HI)
        a1 = rng.randint(_BOUND_LO, _BOUND_HI)
        a2 = rng.randint(_BOUND_LO, _BOUND_HI)
    width_choices = _WIDTH_CHOICES_EASY if tier == "easy" else _WIDTH_CHOICES
    w1 = rng.choice(width_choices)
    w2 = rng.choice(width_choices)
    return gp, gq, gr, h_const, a1, a2, w1, w2


_DOUBLE_INTEGRAL_TEMPLATES = (
    "f(x, y) = ({g})({h}) over the rectangle {a1} ≤ x ≤ {x_hi}, {a2} ≤ y ≤ {y_hi}. Evaluate "
    "the double integral of f over this rectangle using iterated integration.",
    "f(x, y) = ({g})({h}) over the rectangle {a1} ≤ x ≤ {x_hi}, {a2} ≤ y ≤ {y_hi}. Compute the "
    "double integral of f over this rectangle by iterated integration.",
    "f(x, y) = ({g})({h}) over the rectangle {a1} ≤ x ≤ {x_hi}, {a2} ≤ y ≤ {y_hi}. Find the "
    "value of the double integral of f over this rectangle using iterated integration.",
)


@generator(
    "double_integral_rectangle",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def double_integral_rectangle(rng, tier="easy"):
    """Evaluate the iterated double integral of a separable polynomial
    f(x, y) = g(x) h(y) over an axis-aligned rectangle: factor into
    Ix = int g(x) dx, Iy = int h(y) dy, and multiply.

    easy:   g(x) = x, h(y) = y, rectangle anchored at the origin
            (a1 = a2 = 0).
    medium: g(x) = x + q, h(y) = y + s, both nonzero shifts, general
            (possibly negative) rectangle corner.
    hard:   g(x) = p*x^2 + q*x + r with p a nonzero multiple of 3, h(y) as
            in medium -- a true second-order factor.
    """
    resample_reasons: list[str] = []
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        gp, gq, gr, h_const, a1, a2, w1, w2 = _draw_rectangle_integral(rng, tier)

        ix_fraction = _definite_integral(gp, gq, gr, a1, a1 + w1)
        iy_fraction = _definite_integral(0, 1, h_const, a2, a2 + w2)
        check(ix_fraction.denominator == 1, f"Ix is not an exact integer: gp={gp}, gq={gq}, gr={gr}, a1={a1}, w1={w1}")
        check(iy_fraction.denominator == 1, f"Iy is not an exact integer: h_const={h_const}, a2={a2}, w2={w2}")
        ix, iy = int(ix_fraction), int(iy_fraction)

        if ix == 0 or iy == 0:
            resample_reasons.append("zero")
            continue
        answer = ix * iy
        if abs(ix) >= 100 or abs(iy) >= 100 or abs(answer) >= 10000:
            resample_reasons.append("capped")
            continue

        # Independent check: sympy's own general double-integral machinery
        # over the FULL 2-variable expression -- a genuinely different code
        # path from the closed-form arithmetic above.
        g_expr = gp * _x**2 + gq * _x + gr
        h_expr = _y + h_const
        f_expr = g_expr * h_expr
        sympy_value = sympy.integrate(f_expr, (_x, a1, a1 + w1), (_y, a2, a2 + w2))
        check_equivalent(
            sympy_value,
            answer,
            f"independent double-integral check failed: tier={tier}, gp={gp}, gq={gq}, "
            f"gr={gr}, h_const={h_const}, a1={a1}, a2={a2}, w1={w1}, w2={w2}",
        )

        # Building the candidates (and, for medium/hard, checking that
        # `forgets-lower-bound` -- a SINGLE antiderivative endpoint, not the
        # net difference -- happens to land on an integer at this draw, which
        # is not guaranteed the way the true definite integral is) happens
        # inside the same resampling try/except as the pairwise-distinctness
        # check below, so any of these is just another reason to redraw
        # rather than a hard failure: every distractor must be formatted
        # like the answer (a plain integer), never a stray fraction.
        try:
            if tier == "easy":
                # Both `forgets-lower-bound` and `corner-value-times-area`
                # degenerate at a1 = a2 = 0 (the lower-bound antiderivative
                # value is already 0, and g(0)*h(0) = 0 since gr = h_const = 0
                # at this tier) -- use the two tier-specific misconceptions
                # instead.
                candidates = {
                    "sum-instead-of-product": ix + iy,
                    "drops-half-factor": (w1 * w1) * (w2 * w2),
                }
            else:
                forgets_lower_bound_raw = _antiderivative(gp, gq, gr, a1 + w1) * _antiderivative(0, 1, h_const, a2 + w2)
                check(
                    forgets_lower_bound_raw.denominator == 1,
                    f"forgets-lower-bound candidate is not an integer: a1={a1}, a2={a2}, w1={w1}, w2={w2}",
                )
                corner_value = gp * a1 * a1 + gq * a1 + gr
                corner_value *= a2 + h_const
                candidates = {
                    "sum-instead-of-product": ix + iy,
                    "forgets-lower-bound": int(forgets_lower_bound_raw),
                    "corner-value-times-area": corner_value * w1 * w2,
                    "integrates-one-variable-only": ix * w2,
                }
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=(
                    f"misconception collision: tier={tier}, gp={gp}, gq={gq}, gr={gr}, "
                    f"h_const={h_const}, a1={a1}, a2={a2}, w1={w1}, w2={w2}"
                ),
            )
        except VerificationError:
            resample_reasons.append("collision")
            continue
        break
    else:  # pragma: no cover - defensive; the draw space clears this easily
        raise VerificationError("could not draw a collision-free double_integral_rectangle question")

    answer_text = str(answer)
    g_expr = gp * _x**2 + gq * _x + gr
    h_expr = _y + h_const
    g_text = _poly_display(g_expr)
    h_text = _poly_display(h_expr)
    question_text = phrase(
        rng,
        _DOUBLE_INTEGRAL_TEMPLATES,
        g=g_text,
        h=h_text,
        a1=a1,
        x_hi=a1 + w1,
        a2=a2,
        y_hi=a2 + w2,
    )
    # n=4: medium/hard draw exactly 4 pairwise-distinct misconceptions, and
    # the default n=3 cap would deterministically drop the fourth
    # (`integrates-one-variable-only`) on every seed for the same reason as
    # `derivative_definition` above; easy tier only ever offers 2 candidates,
    # so n=4 is a no-op there.
    distractors = pick_distractors(answer_text, [str(value) for value in candidates.values()], n=4)
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer),
        working=[
            f"Ix = ∫ g(x) dx from {a1} to {a1 + w1} = {ix}; Iy = ∫ h(y) dy from "
            f"{a2} to {a2 + w2} = {iy}; the double integral is Ix × Iy = {answer_text}.",
        ],
        distractors=distractors,
    )
