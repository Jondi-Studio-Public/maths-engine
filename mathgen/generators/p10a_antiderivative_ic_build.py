"""Pure antidifferentiation with an initial condition.

``senior_determine_f_left_x_right_given_f_x``
(``senior-determine-f-left-x-right-given-f-x``, ACMMM122, topic
``p10a_antiderivative_given_fprime``, ``answer: expression``) -- given
f'(x) = A*x^2 + B*x + D and an initial condition f(a) = b, find f(x) by
term-by-term antidifferentiation and solving for the constant of
integration.

**Deliberately PURE antidifferentiation -- no motion, no velocity, no particle, no physical context of any
kind.** This is the abstract-notation neighbour of the motion skills built elsewhere; keeping it context-free
is what stops it from re-testing the SAME operation those skills already cover in a physical setting.

1. The ``sign-error-on-constant`` misconception (isolates C with the sign
   flipped: ``G(x) + (b + G(a))``) is claimed for "every tier unless marked",
   but at the EASY tier the initial-condition point ``a`` is pinned to 0, and
   ``G(0) = (A/3)*0**3 + (B/2)*0**2 + D*0 = 0`` identically for EVERY choice
   of A, B, D -- not merely often. When ``G(a) == 0``, the sign-flipped
   candidate ``b + G(a)`` collapses to exactly ``b``, the same value as the
   true ``C = b - G(a) = b - 0 = b``, so the "sign error" is invisible: the
   candidate IS the answer, not a distinct wrong answer, on every single
   easy-tier seed with no possible resample fixing it (resampling ``b``
   cannot change ``G(a)``, and ``a`` is fixed at 0 by the tier's own spec,
   not resampled). This generator therefore does not offer
   ``sign-error-on-constant`` at the easy tier at all.
2. The ``ignored-negative-base-sign`` misconception (hard tier only:
   ``wrong_C2 = b - G(|a|)``) collides with the true answer whenever
   ``G(a) == G(|a|)`` -- which happens for a genuinely reachable, non-rare
   draw, not just a knife-edge coincidence: for ``a = -3`` this holds exactly
   whenever ``D == -3*A`` (e.g. A=1, D=-3, a=-3), and the hard tier's own
   pools contain multiple such (A, D) pairs. The original resample guard
   only names two collision conditions (``C == 0``, ``G(a) == 0``); this
   generator adds a third, structural one -- ``G(a) == G(|a|)`` at the hard
   tier -- resampling the COEFFICIENTS (never ``b``, which this collision
   does not depend on) until it clears, the same "redraw a/coefficients"
   escape hatch the original guard already describes for the other two
   conditions.
3. **A distractor-vs-distractor collision, found by reviewing the
   rendered output rather than by the two named conditions.** The original
   guard only protects each misconception against colliding with the TRUE
   ANSWER; it says nothing about two DIFFERENT misconceptions colliding with
   EACH OTHER. ``sign-error-on-constant``'s candidate,
   ``G(x) + (b + G(a))``, is the exact same STRING as ``forgot-constant``'s
   candidate, ``G(x)``, whenever ``b == -G(a)`` -- both are then neither
   distinct from the answer (``check_distinct`` passes on each individually)
   nor from each other, so ``pick_distractors``'s own string dedupe drops the
   duplicate and medium/hard ship only 2 distractors instead of 3 on those
   seeds (measured ~1% of medium-tier seeds over a 2000-seed sweep). Fixed by
   also excluding ``b == -G(a)`` from the ``b``-resample loop below.
"""

from __future__ import annotations

from sympy import Rational, diff, expand, symbols

from ..core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors

_x = symbols("x")


def _typed(expr) -> str:
    """Learner-typed-answer rendering: ``**`` -> ``^`` only (bare ``*`` kept),
    matching ``p9c_difference_quotient_build.py``/``p5e_first_principles_build.py``.
    The real grader's own ``_parse_expression`` reverses this substitution."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """Question-text rendering: ``**`` -> ``^``, ``*`` -> the middle dot
    (never a bare asterisk -- ``mathgen/text_quality.py``'s ``star_operator``
    check would flag one). Matches the same two files' own ``_prose``."""
    return str(expr).replace("**", "^").replace("*", "·")


# --- Tier coefficient/point pools (pinned exactly) ------------------------

_EASY_B_POOL = tuple(v for v in range(-6, 7) if v <= -2 or v >= 2)  # [-6,-2] u [2,6]
_NONZERO_1_TO_9 = tuple(v for v in range(-9, 10) if v != 0)  # [-9,-1] u [1,9]
_MEDIUM_A_POOL = (-9, -6, -3, 3, 6, 9)
_MEDIUM_POINT_POOL = (-3, -2, -1, 1, 2, 3)
_HARD_A_POOL = tuple(v for v in range(-8, 9) if v != 0 and v % 3 != 0)
_HARD_POINT_POOL = (-3, -2, -1)
_B_RANGE = range(-20, 21)


def _draw_coeffs(rng, tier):
    if tier == "easy":
        a_coef = 0
        b_coef = rng.choice(_EASY_B_POOL)
        d_coef = rng.choice(_NONZERO_1_TO_9)
        a_pt = 0
    elif tier == "medium":
        a_coef = rng.choice(_MEDIUM_A_POOL)
        b_coef = rng.choice(_NONZERO_1_TO_9)
        d_coef = rng.choice(_NONZERO_1_TO_9)
        a_pt = rng.choice(_MEDIUM_POINT_POOL)
    elif tier == "hard":
        a_coef = rng.choice(_HARD_A_POOL)
        b_coef = rng.choice(_NONZERO_1_TO_9)
        d_coef = rng.choice(_NONZERO_1_TO_9)
        a_pt = rng.choice(_HARD_POINT_POOL)
    else:
        raise ValueError(f"unknown tier {tier!r}")
    return a_coef, b_coef, d_coef, a_pt


@generator(
    "p10a_antiderivative_given_fprime",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p10a_antiderivative_given_fprime(rng, tier="easy"):
    """Given f'(x) = A*x^2 + B*x + D and f(a) = b, find f(x).

    G(x) = (A/3)*x^3 + (B/2)*x^2 + D*x (term-by-term power rule, no constant).
    C = b - G(a); F(x) = G(x) + C.

    easy:   A = 0, a = 0 (fixed). Single power-rule term (D*x) plus a
            linear f'(x) term; the constant is solved at the trivial point
            a = 0.
    medium: A nonzero, divisible by 3 (integer cubic coefficient); a nonzero
            -- the learner must actually substitute a nonzero point into a
            cubic to solve for C, not just evaluate at 0.
    hard:   A nonzero, NOT divisible by 3 (the answer's leading coefficient
            is a proper fraction -- fraction power-rule arithmetic, a
            reasoning escalation, not a magnitude one); a forced negative --
            sign-careful substitution of a negative point into a cubic, a
            second, independent escalation.
    """

    def _build(a_coef, b_coef, d_coef, a_pt):
        f_prime_expr = a_coef * _x**2 + b_coef * _x + d_coef
        g_expr = Rational(a_coef, 3) * _x**3 + Rational(b_coef, 2) * _x**2 + d_coef * _x
        g_a = g_expr.subs(_x, a_pt)
        return f_prime_expr, g_expr, g_a

    def _coeffs_structurally_valid(tier, g_expr, g_a, a_pt):
        """Whether this (A, B, D, a) draw avoids the two collisions that
        depend only on the COEFFICIENTS and the point -- never on `b` -- so
        resampling `b` alone could never fix either:
        - medium/hard: G(a) == 0 would make sign-error-on-constant collide
          with the true answer (not checked
          at easy, where G(a) == 0 always and that misconception is simply
          never offered);
        - hard only: G(a) == G(|a|) would make ignored-negative-base-sign
          collide with the true answer -- a genuinely reachable
          draw (e.g. A=1, D=-3, a=-3), not a knife-edge coincidence.
        """
        if tier != "easy" and g_a == 0:
            return False
        if tier == "hard" and g_a == g_expr.subs(_x, abs(a_pt)):
            return False
        return True

    a_coef, b_coef, d_coef, a_pt = _draw_coeffs(rng, tier)
    f_prime_expr, g_expr, g_a = _build(a_coef, b_coef, d_coef, a_pt)
    for _coeff_attempt in range(50):
        if _coeffs_structurally_valid(tier, g_expr, g_a, a_pt):
            break
        a_coef, b_coef, d_coef, a_pt = _draw_coeffs(rng, tier)
        f_prime_expr, g_expr, g_a = _build(a_coef, b_coef, d_coef, a_pt)
    check(
        _coeffs_structurally_valid(tier, g_expr, g_a, a_pt),
        f"resample guard failed to find structurally valid coefficients: tier={tier}",
    )

    # `b` alone decides the two remaining collisions: C == 0 would make
    # forgot-constant collide with the true answer; b == -G(a) makes
    # sign-error-on-constant's own candidate (G(x) + (b + G(a))) collapse to
    # exactly G(x) -- the SAME STRING as forgot-constant -- so the two
    # distinct misconceptions silently merge into one offered option at
    # medium/hard, dropped by pick_distractors's own string dedupe (neither
    # ever equals the true answer, so check_distinct alone never catches
    # this; measured at ~1% of medium-tier seeds over a 2000-seed sweep).
    b_val = None
    for attempt in range(50):
        candidate_b = rng.randrange(_B_RANGE.start, _B_RANGE.stop)
        if candidate_b - g_a != 0 and candidate_b + g_a != 0:
            b_val = candidate_b
            break
    check(b_val is not None, f"resample guard failed to find a valid draw: tier={tier}")

    c_const = b_val - g_a
    f_expr = expand(g_expr + c_const)

    # Independent check -- the DIFFERENTIATION route, never a second
    # integration: differentiate the constructed F(x) back and confirm it
    # matches f'(x), then confirm F(a) equals the stated initial condition.
    check_equivalent(
        diff(f_expr, _x),
        f_prime_expr,
        msg=f"antiderivative does not differentiate back to f'(x): tier={tier}, "
        f"a_coef={a_coef}, b_coef={b_coef}, d_coef={d_coef}, a_pt={a_pt}, b_val={b_val}",
    )
    check_equivalent(
        f_expr.subs(_x, a_pt),
        b_val,
        msg=f"F(a) does not equal the stated initial condition b: tier={tier}, "
        f"a_coef={a_coef}, b_coef={b_coef}, d_coef={d_coef}, a_pt={a_pt}, b_val={b_val}",
    )

    f_prime_prose = _prose(f_prime_expr)
    question_text = f"f'(x) = {f_prime_prose}. If f({a_pt}) = {b_val}, find f(x)."
    answer_text = _typed(f_expr)

    # Misconceptions -- every candidate is a polynomial expression in x.
    candidates_text = []

    forgot_constant = expand(g_expr)
    check_distinct(
        f_expr, forgot_constant, msg=f"forgot_constant equals answer: tier={tier}, a_pt={a_pt}, b_val={b_val}"
    )
    candidates_text.append(_typed(forgot_constant))

    wrong_g = a_coef * _x**3 + b_coef * _x**2 + d_coef * _x
    wrong_g_a = wrong_g.subs(_x, a_pt)
    wrong_c = b_val - wrong_g_a
    power_rule_no_divide = expand(wrong_g + wrong_c)
    check_distinct(
        f_expr, power_rule_no_divide, msg=f"power_rule_no_divide equals answer: tier={tier}, a_pt={a_pt}, b_val={b_val}"
    )
    candidates_text.append(_typed(power_rule_no_divide))

    if tier != "easy":
        sign_error_on_constant = expand(g_expr + (b_val + g_a))
        check_distinct(
            f_expr,
            sign_error_on_constant,
            msg=f"sign_error_on_constant equals answer: tier={tier}, a_pt={a_pt}, b_val={b_val}",
        )
        candidates_text.append(_typed(sign_error_on_constant))

    if tier == "hard":
        wrong_c2 = b_val - g_expr.subs(_x, abs(a_pt))
        ignored_negative_base_sign = expand(g_expr + wrong_c2)
        check_distinct(
            f_expr,
            ignored_negative_base_sign,
            msg=f"ignored_negative_base_sign equals answer: tier={tier}, a_pt={a_pt}, b_val={b_val}",
        )
        candidates_text.append(_typed(ignored_negative_base_sign))

    distractors = pick_distractors(answer_text, candidates_text)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": str(f_expr),
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": [
            f"Integrate term by term (power rule): f(x) = {_prose(g_expr)} + C.",
            f"Use f({a_pt}) = {b_val}: {_prose(g_expr.subs(_x, a_pt))} + C = {b_val}, so C = {c_const}.",
            f"f(x) = {_prose(f_expr)}.",
        ],
    }
