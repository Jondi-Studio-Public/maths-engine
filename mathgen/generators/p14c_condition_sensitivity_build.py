"""Generates `p14c_condition_sensitivity` questions: how sensitive a computation is to input error, via the
condition number, answered as a fraction.

``ucv1-condition-number-sensitivity`` (numerical-and-computational-mathematics, topic
``p14c_condition_sensitivity``, ``answer: fraction``).

**Kernel.** The standard first-order conditioning bound: for a supplied
condition number ``kappa`` (exact rational, ``kappa >= 1`` always -- a
condition number can never be less than 1) and a supplied relative input
error ``rel_in`` (exact rational), the bound on the relative output error is

    rel_out_bound = kappa * rel_in.

Every quantity is stated in the question (directly at easy/medium, via an
absolute error and the input's own magnitude at hard, inverted at exam); the
learner substitutes and multiplies (or, at exam tier, divides).

Boundary against the prerequisite ``condition-number-diagonal``
:
that skill COMPUTES kappa from a stated diagonal matrix (``kappa_2(A) =
||A||_2 ||A^-1||_2``). This skill is handed kappa directly -- it never
derives it from a matrix -- and USES it, with a stated relative input
error, to bound the relative output error. The two answers are never the
same computation: one is a norm ratio of matrix entries, the other is a
product of two already-given scalars.

``kappa`` is drawn strictly greater than 1 on every tier: at kappa = 1 every
misconception collapses onto the true answer (``rel_in / kappa ==
kappa * rel_in`` and ``kappa**2 * rel_in`` all equal ``rel_in`` when
kappa = 1), so kappa = 1 is not merely atypical, it makes every distractor a
second correct answer.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

_MAX_DRAW_ATTEMPTS = 500

_GIVEN_TEMPLATES = (
    "A computation has condition number kappa = {kappa} and relative input "
    "error {rel_in}. Using the first-order bound rel_out_bound = kappa × "
    "rel_in, find the bound on the relative output error (as a fraction).",
    "For a problem with condition number kappa = {kappa}, a relative input "
    "error of {rel_in} produces a relative output error bounded by "
    "kappa × rel_in. Find this bound (as a fraction).",
    "Given condition number kappa = {kappa} and relative input error "
    "{rel_in}, compute the bound on relative output error rel_out_bound = "
    "kappa × rel_in (as a fraction).",
)

_HARD_TEMPLATES = (
    "A computation has condition number kappa = {kappa}. The input has true "
    "value {input_val} and absolute error {abs_err}. Using the bound "
    "rel_out_bound = kappa × (absolute error / |input value|), find the "
    "bound on the relative output error (as a fraction).",
    "For a problem with condition number kappa = {kappa}, the input value is "
    "{input_val} with an absolute error of {abs_err}. Find the bound on the "
    "relative output error, rel_out_bound = kappa × (absolute error / "
    "|input value|) (as a fraction).",
    "Given condition number kappa = {kappa}, input value {input_val} and "
    "absolute input error {abs_err}, compute rel_out_bound = kappa × "
    "(absolute error / |input value|) (as a fraction).",
)

_EXAM_TEMPLATES = (
    "A computation has condition number kappa = {kappa}. The first-order "
    "bound rel_out_bound = kappa × rel_in works out to {bound} for some "
    "relative input error rel_in > 0. Find rel_in (as a fraction).",
    "For a problem with condition number kappa = {kappa}, the bound on "
    "relative output error is rel_out_bound = kappa × rel_in = {bound} for "
    "some rel_in > 0. Find the relative input error rel_in (as a fraction).",
    "Given condition number kappa = {kappa}, the relative output error bound "
    "kappa × rel_in equals {bound}. Find the relative input error rel_in "
    "(rel_in > 0, as a fraction).",
)


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


def _bound_candidates(kappa: Fraction, rel_in: Fraction) -> dict:
    """Given-directly-tier misconception values, each from a plausible WRONG
    formula -- never a restatement of the correct one."""
    return {
        "divides-instead-of-multiplies": rel_in / kappa,
        "omits-kappa": rel_in,
        "uses-kappa-squared": kappa * kappa * rel_in,
    }


def _draw_given(rng, *, kappa_hi, rel_in_den_hi, force_fraction_kappa):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        if force_fraction_kappa:
            kappa_num = rng.randint(3, kappa_hi)
            kappa_den = rng.randint(2, 4)
            if kappa_num == kappa_den:
                continue
            kappa = Fraction(kappa_num, kappa_den)
            if kappa <= 1:
                continue
        else:
            kappa = Fraction(rng.randint(2, kappa_hi))
        rel_in_den = rng.randint(10, rel_in_den_hi)
        rel_in_num = rng.randint(1, 9)
        rel_in = Fraction(rel_in_num, rel_in_den)

        answer = kappa * rel_in
        # Independent route: recompute via numerator/denominator products
        # done in a different order (cross-multiply then reduce) rather than
        # Fraction's own __mul__.
        direct = Fraction(kappa.numerator * rel_in.numerator, kappa.denominator * rel_in.denominator)
        check_equivalent(answer, direct, "conditioning bound recomputation mismatch")

        # A condition number can never be less than 1 -- a real mathematical
        # invariant (kappa = ||A|| ||A^-1|| >= ||A A^-1|| = 1 by the
        # sub-multiplicative norm inequality), checked explicitly rather than
        # assumed. The draw additionally requires kappa > 1 (never exactly
        # 1): at kappa = 1 every misconception below collapses onto the true
        # answer (rel_in / 1 == 1 * rel_in == 1*1*rel_in), so kappa = 1 would
        # ship a question with no wrong option at all.
        check(kappa >= 1, "condition number kappa must be >= 1")

        candidates = _bound_candidates(kappa, rel_in)
        values_to_check = [answer, *candidates.values()]
        try:
            check_pairwise_distinct(*values_to_check, msg=f"conditioning collision: kappa={kappa}, rel_in={rel_in}")
            check(kappa > 1, "kappa must be strictly greater than 1 to avoid the kappa=1 collision")
        except VerificationError:
            continue
        return kappa, rel_in, answer, candidates
    raise VerificationError("could not draw a collision-free given-directly conditioning question")


def _build_easy(rng) -> dict:
    kappa, rel_in, answer, candidates = _draw_given(
        rng,
        kappa_hi=8,
        rel_in_den_hi=20,
        force_fraction_kappa=False,
    )
    payload, answer_text = _fraction_payload(answer)
    text = phrase(rng, _GIVEN_TEMPLATES, kappa=rational_text(kappa), rel_in=rational_text(rel_in))
    distractor_strs = [rational_text(v) for v in candidates.values()]
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[f"rel_out_bound = kappa × rel_in = {rational_text(kappa)} × {rational_text(rel_in)} = {answer_text}."],
        distractors=pick_distractors(answer_text, distractor_strs, n=5),
    )


def _build_medium(rng) -> dict:
    kappa, rel_in, answer, candidates = _draw_given(
        rng,
        kappa_hi=9,
        rel_in_den_hi=24,
        force_fraction_kappa=True,
    )
    payload, answer_text = _fraction_payload(answer)
    text = phrase(rng, _GIVEN_TEMPLATES, kappa=rational_text(kappa), rel_in=rational_text(rel_in))
    distractor_strs = [rational_text(v) for v in candidates.values()]
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[f"rel_out_bound = kappa × rel_in = {rational_text(kappa)} × {rational_text(rel_in)} = {answer_text}."],
        distractors=pick_distractors(answer_text, distractor_strs, n=5),
    )


def _build_hard(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        kappa = Fraction(rng.randint(2, 9))
        input_val = rng.choice([v for v in range(-12, 13) if abs(v) >= 2])
        abs_err = Fraction(rng.randint(1, 9), rng.randint(10, 30))
        rel_in = abs_err / abs(input_val)

        answer = kappa * rel_in
        # Independent route: recompute rel_in via cross-multiplied division
        # (abs_err.numerator, abs_err.denominator * |input_val|) instead of
        # Fraction's own __truediv__, then recombine.
        direct_rel_in = Fraction(abs_err.numerator, abs_err.denominator * abs(input_val))
        check(direct_rel_in == rel_in, "hard-tier relative-error recomputation mismatch")
        direct = Fraction(kappa.numerator * direct_rel_in.numerator, kappa.denominator * direct_rel_in.denominator)
        check_equivalent(answer, direct, "hard-tier conditioning bound recomputation mismatch")
        # Invariant: a condition number can never be less than 1.
        check(kappa >= 1, "condition number kappa must be >= 1")

        candidates = _bound_candidates(kappa, rel_in)
        candidates["uses-absolute-not-relative"] = kappa * abs_err
        values_to_check = [answer, *candidates.values()]
        try:
            check_pairwise_distinct(
                *values_to_check,
                msg=f"hard-tier conditioning collision: kappa={kappa}, input_val={input_val}, abs_err={abs_err}",
            )
        except VerificationError:
            continue

        payload, answer_text = _fraction_payload(answer)
        text = phrase(
            rng,
            _HARD_TEMPLATES,
            kappa=rational_text(kappa),
            input_val=input_val,
            abs_err=rational_text(abs_err),
        )
        distractor_strs = [rational_text(v) for v in candidates.values()]
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"rel_in = |abs error| / |input value| = {rational_text(abs_err)} / {abs(input_val)} = {rational_text(rel_in)}.",
                f"rel_out_bound = kappa × rel_in = {rational_text(kappa)} × {rational_text(rel_in)} = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=6),
        )
    raise VerificationError("could not draw a collision-free hard-tier conditioning question")


def _build_exam(rng) -> dict:
    """Inverse tier: kappa and the resulting bound rel_out_bound are stated;
    the learner recovers the relative input error rel_in = rel_out_bound /
    kappa. Linear in rel_in (kappa > 1 is enforced by construction, so the
    division is always well defined), so the solution is unique by
    construction -- no second algebraic root to rule out. Proved directly
    below by recomputing rel_in from the stated bound and checking the
    forward computation lands back on it (round-trip), the same discipline
    ``p13f_ridge_coefficient_build.py``'s ``_build_exam`` uses for its own
    linear inversion."""
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        kappa = Fraction(rng.randint(2, 9))
        rel_in_num = rng.randint(1, 9)
        rel_in_den = rng.randint(10, 30)
        rel_in = Fraction(rel_in_num, rel_in_den)
        bound = kappa * rel_in
        check(bound != 0, "exam-tier bound must be nonzero to invert")
        check(kappa >= 1, "condition number kappa must be >= 1")

        recovered_rel_in = bound / kappa
        check_equivalent(recovered_rel_in, rel_in, "recovered rel_in does not match the drawn rel_in")
        # Round-trip: recompute the forward bound from the recovered value
        # and confirm it lands back on the stated bound.
        check_equivalent(kappa * recovered_rel_in, bound, "forward round-trip from recovered rel_in failed")

        candidates = {
            "divides-instead-of-multiplies": bound * kappa,
            "omits-kappa": bound,
            "uses-kappa-squared": bound / (kappa * kappa),
        }
        try:
            check_pairwise_distinct(
                rel_in, *candidates.values(), msg=f"exam-tier conditioning collision: kappa={kappa}, rel_in={rel_in}"
            )
        except VerificationError:
            continue

        payload, answer_text = _fraction_payload(rel_in)
        bound_text = rational_text(bound)
        text = phrase(rng, _EXAM_TEMPLATES, kappa=rational_text(kappa), bound=bound_text)
        distractor_strs = [rational_text(v) for v in candidates.values()]
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"rel_out_bound = kappa × rel_in, so rel_in = rel_out_bound / kappa = {bound_text} / {rational_text(kappa)} = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=3),
        )
    raise VerificationError("could not draw a collision-free exam-tier conditioning question")


@generator(
    "p14c_condition_sensitivity",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def condition_sensitivity(rng, tier="easy"):
    """Bound relative output error from a supplied condition number and
    relative input error (the skill's stated
    contract): rel_out_bound = kappa * rel_in.

    easy/medium: kappa and rel_in given directly (medium forces kappa itself
    to be a genuine fraction).
    hard: rel_in is not given directly -- an absolute input error and the
    input's own magnitude are given, and rel_in must be derived first.
    exam: inverse -- kappa and the resulting bound are stated; recover
    rel_in.
    """
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
