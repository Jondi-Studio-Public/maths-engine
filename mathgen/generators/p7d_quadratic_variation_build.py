"""Generates `p7d_examine_quadratic_variation` questions: use y = kx^2 to find a new value of y or x.

Quadratic variation.

- ``p7d_examine_quadratic_variation``
  (``senior-examine-examples-of-quadratically-related-variables``, ACMMM006,
  domain quadratics, answer: number) -- given bounded, fully specified
  inputs, examine a quadratically related pair of variables (y = kx^2) and
  compute a new value, a new positive independent-variable value, or a
  scaling factor.

This skill sits beside two others (direct and inverse variation, own
generator module) at measured 0.50 semantic similarity. All three follow the
identical "Variation phrasing conventions": every question states its relation
as a formula, never a bare "varies directly/inversely/quadratically"
phrase, so a learner can tell the three apart from the question text alone.
Distractors are built by applying the *other two* relations' arithmetic
(direct: exponent 1, inverse: exponent -1) to the same drawn numbers, plus
one generic "no change"/"doubling" style distractor to round out three
options per tier.

easy: given (x1, y1) and a new x2 = x1 * t, find y2 (direct substitution
into y = kx^2).
medium: given (x1, y1) and a new y2, find the new positive x2 (the unknown
moves to the independent-variable side; x is introduced explicitly as a
positive length so "the positive root" is not an unstated assumption).
hard: kinetic energy E = kv^2 -- given the speed is multiplied by t, find
the factor by which the kinetic energy is multiplied (t an integer 2-5,
independent of the stated (v1, E1) pair, which is included only for
bounded-input honesty).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors


# ==============================================================================
# easy: y2 given (x1, y1) and x2 = x1 * t
# ==============================================================================


def _verify_easy(x1: int, y1: int, x2: int, y2_claimed: int) -> None:
    """Independent check for the easy tier: reconstruct k from the stated
    (x1, y1) alone, then recompute y2 from k and the stated x2 -- a route
    that never reuses the t/s multiplication the construction used."""
    k = Fraction(y1, x1 * x1)
    y2_check = k * (x2 * x2)
    check_equivalent(y2_check, y2_claimed, f"easy: reconstruction disagrees for x1={x1}, y1={y1}, x2={x2}")


def _build_easy(rng) -> dict:
    a = rng.randint(2, 9)  # x1
    t = rng.randint(2, 4)  # x-scale factor
    s = rng.randint(1, 9)
    x1 = a
    y1 = t * s
    x2 = a * t

    # Correct: y = kx^2, so y2 = y1 * t^2 = t^3 * s.
    y2 = y1 * t * t
    # "as if direct" (forgot to square): y2 = y1 * t = t^2 * s.
    as_if_direct = y1 * t
    # "as if inverse": y2 = y1 / t = s. Exact by construction (y1 = t * s).
    as_if_inverse = s
    # Generic "forgot to change y at all" error.
    no_change = y1

    # Independent check: reconstruct k from the stated (x1, y1) alone, then
    # recompute y2 from k and the stated x2 -- a different route from the
    # t/s multiplication above.
    _verify_easy(x1, y1, x2, y2)

    check_distinct(
        y2,
        as_if_direct,
        as_if_inverse,
        no_change,
        msg=f"easy: a distractor equals the answer for x1={x1}, y1={y1}, t={t}, s={s}",
    )
    check_distinct(as_if_direct, as_if_inverse, msg=f"easy: as_if_direct collides with as_if_inverse for t={t}, s={s}")
    check_distinct(as_if_direct, no_change, msg=f"easy: as_if_direct collides with no_change for t={t}, s={s}")
    check_distinct(as_if_inverse, no_change, msg=f"easy: as_if_inverse collides with no_change for t={t}, s={s}")

    text = (
        f"A quantity y varies according to the relation y = kx^2, where k is a "
        f"constant. When x = {x1}, y = {y1}. If x increases to {x2}, find the "
        f"new value of y."
    )
    answer_text = str(y2)
    candidates = [str(as_if_direct), str(as_if_inverse), str(no_change)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, f"easy: fewer than 3 distinct distractors for x1={x1}, y1={y1}, x2={x2}")

    k_display = Fraction(y1, x1 * x1)
    working = [
        "Stated relation: y = kx^2.",
        f"k = y / x^2 = {y1} / {x1}^2 = {k_display} (from the given data point).",
        f"New y = k × x^2 = {k_display} × {x2}^2 = {y2}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=y2,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )


# ==============================================================================
# medium: new positive x2 given (x1, y1) and a new y2
# ==============================================================================


def _verify_medium(x1: int, y1: int, x2: int, y2: int) -> None:
    """Independent check for the medium tier: confirm x2 > 0 and that the
    ratio condition (x2/x1)^2 == y2/y1 holds directly -- the definition of
    "positive root of a squared ratio" -- without calling any square-root
    routine the construction used."""
    check(x2 > 0, f"medium: x2 must be positive, got {x2}")
    lhs = Fraction(x2, x1) ** 2
    rhs = Fraction(y2, y1)
    check(lhs == rhs, f"medium: ratio condition failed for x1={x1}, y1={y1}, x2={x2}, y2={y2}")


def _build_medium(rng) -> dict:
    t = rng.randint(2, 4)  # true x-scale factor
    a2 = rng.randint(1, 9)
    x1 = t * t * a2
    y1 = a2
    y2 = a2 * t * t

    # Correct positive root: x2 = x1 * t = t^3 * a2.
    x2 = a2 * t**3
    # "as if direct" (used the value-ratio directly instead of its square root).
    as_if_direct = a2 * t**4
    # "as if inverse".
    as_if_inverse = a2
    # Generic "forgot to change x at all" error.
    no_change = x1

    # Independent check: confirm x2 > 0 and that the ratio condition
    # (x2/x1)^2 == y2/y1 -- the definition of "positive root of a squared
    # ratio" -- holds directly, without calling any square-root routine.
    _verify_medium(x1, y1, x2, y2)

    check_distinct(
        x2,
        as_if_direct,
        as_if_inverse,
        no_change,
        msg=f"medium: a distractor equals the answer for x1={x1}, y1={y1}, y2={y2}, t={t}",
    )
    check_distinct(
        as_if_direct, as_if_inverse, msg=f"medium: as_if_direct collides with as_if_inverse for t={t}, a2={a2}"
    )
    check_distinct(as_if_direct, no_change, msg=f"medium: as_if_direct collides with no_change for t={t}, a2={a2}")
    check_distinct(as_if_inverse, no_change, msg=f"medium: as_if_inverse collides with no_change for t={t}, a2={a2}")

    text = (
        f"A quantity y varies according to y = kx^2 for a constant k, where x "
        f"is a positive length (so x > 0). When x = {x1}, y = {y1}. If y "
        f"increases to {y2}, find the new positive value of x."
    )
    answer_text = str(x2)
    candidates = [str(as_if_direct), str(as_if_inverse), str(no_change)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, f"medium: fewer than 3 distinct distractors for x1={x1}, y1={y1}, y2={y2}")

    k = Fraction(y1, x1 * x1)
    working = [
        "Stated relation: y = kx^2, with x a positive length.",
        f"k = y / x^2 = {y1} / {x1}^2 = {k} (from the given data point).",
        f"x is positive, so x = sqrt(y2 / k) = sqrt({y2} / {k}) = {x2}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=x2,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )


# ==============================================================================
# hard: kinetic energy scale factor, E = kv^2
# ==============================================================================


def _verify_hard(v1: int, e1: int, t: int, factor_claimed: int) -> None:
    """Independent check for the hard tier: re-derive k from the stated
    (v1, E1) pair, recompute the new energy from k and v2 = v1 * t, and
    confirm the ratio E2/E1 equals the claimed scale factor -- a route that
    never reuses the t*t multiplication the construction used."""
    k = Fraction(e1, v1 * v1)
    v2 = v1 * t
    e2 = k * (v2 * v2)
    factor_check = Fraction(e2, e1)
    check_equivalent(
        factor_check, factor_claimed, f"hard: scale-factor reconstruction disagrees for v1={v1}, e1={e1}, t={t}"
    )


def _build_hard(rng) -> dict:
    t = rng.randint(2, 5)
    correct = t * t
    forgot_square = t
    cubic = t**3
    doubling = 2 * t
    if doubling == correct:
        # Collides only at t == 2 (2*2 == 2^2); swap in the fallback candidate
        # instead of shipping a collision.
        doubling = correct + 1

    v1 = rng.randint(2, 9)
    k_context = rng.randint(1, 5)
    e1 = k_context * v1 * v1
    v2 = v1 * t

    # Independent check: re-derive k from the stated (v1, E1) pair, recompute
    # the new energy from k and v2, and confirm the ratio E2/E1 equals the
    # claimed scale factor -- a route that never reuses the t*t multiplication
    # above.
    _verify_hard(v1, e1, t, correct)

    check_distinct(correct, forgot_square, cubic, doubling, msg=f"hard: a distractor equals the answer for t={t}")
    check_distinct(forgot_square, cubic, msg=f"hard: forgot_square collides with cubic for t={t}")
    check_distinct(forgot_square, doubling, msg=f"hard: forgot_square collides with doubling for t={t}")
    check_distinct(cubic, doubling, msg=f"hard: cubic collides with doubling for t={t}")

    text = (
        f"Kinetic energy is directly proportional to the square of speed, so "
        f"E = kv^2 for a constant k. For example, when v = {v1} m/s, E = {e1} J. "
        f"If the speed is multiplied by {t}, by what factor is the kinetic "
        f"energy multiplied?"
    )
    answer_text = str(correct)
    candidates = [str(forgot_square), str(cubic), str(doubling)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, f"hard: fewer than 3 distinct distractors for t={t}")

    k_display = Fraction(e1, v1 * v1)
    working = [
        "Stated relation: E = kv^2.",
        f"k = E / v^2 = {e1} / {v1}^2 = {k_display} (from the stated data point; the scale factor below does not depend on it).",
        f"New speed = {t} × v, so new E = k × ({t} × v)^2 = {t}^2 × (k × v^2) = {correct} × E.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=correct,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "p7d_examine_quadratic_variation",
    level=13,
    presets={
        "easy": dict(variant="quadratic_forward"),
        "medium": dict(variant="quadratic_inverse_solve"),
        "hard": dict(variant="kinetic_energy_scale_factor"),
    },
)
def p7d_examine_quadratic_variation(rng, variant="quadratic_forward"):
    """Examine examples of quadratically related variables (y = kx^2)."""
    if variant == "quadratic_forward":
        return _build_easy(rng)
    if variant == "quadratic_inverse_solve":
        return _build_medium(rng)
    if variant == "kinetic_energy_scale_factor":
        return _build_hard(rng)
    raise ValueError(f"unknown variant: {variant!r}")
