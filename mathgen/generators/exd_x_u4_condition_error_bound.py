"""Exam-tier source for `condition-number-diagonal`: a condition number put to work on measurement error.

The practice tiers give `diag(...)` and ask for its 2-norm condition number. The exam question describes a
calibration matrix, sometimes by its inverse, and uses the number the way an exam does:

1. find `kappa_2(A)` for a signed three-entry diagonal matrix (given as `A`, or as `A^-1`),
2. bound the relative error in the solution of `Ax = b` caused by a stated relative error in `b`
   (`||dx|| / ||x|| <= kappa ||db|| / ||b||`, quoted in the question), and
3. work back from a target relative error in `x` to the largest relative error in `b` that guarantees it.

Parts (b) and (c) carry follow-through. For a diagonal matrix the 2-norm condition number is the ratio of the
largest to the smallest absolute diagonal entry; the independent route builds it from the singular values of
`A` and of `A^-1` with SymPy's matrix norm instead.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Matrix, Rational, diag, sqrt

from mathgen.core import check, check_equivalent, generator, rational_text

from ._u4_kit import fraction_distractors, fraction_payload, survives_nsimplify

_SETTINGS = [
    "A controller solves Ax = b for the readings x of three sensors, where A = {matrix}.",
    "A calibration step recovers three quantities x from the linear system Ax = b, with A = {matrix}.",
    "Three channels of a data logger are scaled by A = {matrix}, and the true signal x is recovered from Ax = b.",
]
_INVERSE_SETTINGS = [
    "A controller solves Ax = b for the readings x of three sensors. Its stored matrix is A⁻¹ = {matrix}.",
    "A calibration step recovers three quantities x from the linear system Ax = b, and the stored matrix is the "
    "inverse A⁻¹ = {matrix}.",
]


def _matrix_text(entries: list[Fraction]) -> str:
    return "diag(" + ", ".join(rational_text(e) for e in entries) + ")"


def _percent(value: Fraction) -> str:
    return f"{value.numerator}" if value.denominator == 1 else f"{value.numerator / value.denominator:g}"


def _br(value) -> str:
    """A fraction in brackets, so it can be divided or multiplied: ``(1/2)``; a whole number is left alone."""
    return f"({value})" if "/" in str(value) else str(value)


@generator("exd_x_u4_condition_error_bound", level=17, presets={"exam": dict(tier="exam")})
def exd_x_u4_condition_error_bound(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: condition number of a signed 3x3 diagonal matrix, then error bounds in both directions."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(1000):
        magnitudes = rng.sample(range(1, 13), 3)
        entries = [m * rng.choice([1, -1]) for m in magnitudes]
        if len({abs(e) for e in entries}) < 3:
            continue
        kappa = Fraction(max(magnitudes), min(magnitudes))
        eps_tenths = rng.choice([1, 2, 3, 4, 5, 6, 8, 9, 12, 15, 25])  # relative error in b, in tenths of a percent
        tau = rng.choice([2, 3, 4, 5, 6, 8, 10])  # target relative error in x, in percent
        eps = Fraction(eps_tenths, 10)
        bound = kappa * eps
        allowed = tau / kappa
        if min(entries) > 0 and rng.random() < 0.5:
            continue  # at least half the matrices carry a negative entry
        if 1 in (kappa.denominator, bound.denominator, allowed.denominator) and rng.random() < 0.5:
            continue
        if not survives_nsimplify(kappa, bound, allowed):
            continue
        break
    else:
        check(False, "no usable matrix and tolerances in 1000 tries")

    # Independent route: singular values of A and of A^-1, whose product of largest values is the condition number.
    matrix = diag(*[Rational(e) for e in entries])
    singular = sorted(sqrt(ev) for ev in (matrix.T * matrix).eigenvals())
    inverse_singular = sorted(sqrt(ev) for ev in (matrix.inv().T * matrix.inv()).eigenvals())
    check_equivalent(singular[-1] * inverse_singular[-1], Rational(kappa.numerator, kappa.denominator))
    check(Matrix(matrix).det() != 0, "matrix is singular")

    inverse_form = rng.random() < 0.4
    if inverse_form:
        shown = [Fraction(1, e) for e in entries]
        setting = rng.choice(_INVERSE_SETTINGS).format(matrix=_matrix_text(shown))
    else:
        setting = rng.choice(_SETTINGS).format(matrix=_matrix_text([Fraction(e) for e in entries]))
    stem = (
        f"{setting} For this kind of system, ‖δx‖/‖x‖ ≤ κ₂(A) · ‖δb‖/‖b‖, where κ₂(A) = ‖A‖₂‖A⁻¹‖₂. "
        "Give exact values as fractions or integers."
    )
    largest, smallest = max(abs(e) for e in entries), min(abs(e) for e in entries)
    signed_ratio = Fraction(max(entries), min(entries))
    a_candidates = [
        (Fraction(smallest, largest), "condition-number-inverted"),
        (signed_ratio, "condition-number-signed"),
        (Fraction(sum(abs(e) for e in entries)), "condition-number-sum"),
        (Fraction(largest, 1) - smallest, None),
    ]
    b_candidates = [
        (eps / kappa, "error-bound-divided"),
        (eps * Fraction(sum(abs(e) for e in entries)), "condition-number-sum"),
        (eps, None),
        (kappa, None),
        (eps * largest, None),
    ]
    c_candidates = [
        (tau * kappa, "error-allowance-multiplied"),
        (kappa / tau, "inverted-ratio"),
        (Fraction(tau), None),
        (tau / (kappa * kappa), None),
    ]
    kappa_text, bound_text, allowed_text = (rational_text(v) for v in (kappa, bound, allowed))
    eps_text = _percent(eps)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the 2-norm condition number κ₂(A).",
                answer_kind="fraction",
                answer=fraction_payload(kappa)[0],
                answer_text=kappa_text,
                marks=2,
                sensible_range=(1, None),
                working=[
                    f"The singular values are the absolute values of the diagonal entries: largest {largest}, "
                    f"smallest {smallest} (for A, whichever way it is given).",
                    f"κ₂(A) = {largest}/{smallest} = {kappa_text}.",
                ],
                distractors=fraction_distractors(kappa, a_candidates),
            ),
            dict(
                prompt=f"The relative error in b is at most {eps_text}%. Using your κ₂(A), find the largest relative "
                "error, in percent, that this can cause in x.",
                answer_kind="fraction",
                answer=fraction_payload(bound)[0],
                answer_text=bound_text,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{eps.numerator}/{eps.denominator} * part_a",
                working=[f"The bound is κ₂(A) × {eps_text}% = {kappa_text} × {eps_text} = {bound_text} percent."],
                distractors=fraction_distractors(bound, b_candidates),
            ),
            dict(
                prompt=f"The relative error in x must not exceed {tau}%. Using your κ₂(A), find the largest relative "
                "error in b, in percent, that guarantees this.",
                answer_kind="fraction",
                answer=fraction_payload(allowed)[0],
                answer_text=allowed_text,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{tau}/part_a",
                working=[
                    f"κ₂(A) × (relative error in b) ≤ {tau}, so the error in b is at most {tau} ÷ {_br(kappa_text)} = {allowed_text} percent."
                ],
                distractors=fraction_distractors(allowed, c_candidates),
            ),
        ],
    )
