"""The effect of interest rate and repayment amount on the number of periods needed to repay a flat-interest
loan.

One skill: ``senior-investigate-the-effect-of-the-interest-rate-and``
(ACMEM175), ``domain: financial-discrete``, ``answer: number``,
``prerequisites: [simple-interest]``.

This is the deliberate INVERSE of the shipping ``flat-loan-repayment``
(``mathgen/generators/curriculum_expansion.py::flat_loan_repayment``), which
computes the monthly repayment given principal, rate and TERM. Here the
principal ``P``, flat annual rate ``r`` (%) and the repayment amount are
GIVEN and the term is the unknown, using the same flat-interest total-
repayable model: ``Total = P * (1 + r * n / 100)``, ``repayment = Total /
(number of repayment periods)``.

Construct-to-target, exact by algebra (never a search-and-resample for
exactness): every tier picks its ``denom`` term and a scale factor ``k``
first, so that ``P = 1200 * n * k`` (years-unit tiers) makes the derived
repayment ``M = k * denom`` an exact integer BY CONSTRUCTION -- no float,
no rounding. Every tier then independently re-derives its unknown by the
FORWARD linear solve from the stated ``P``, ``r`` and repayment alone
(``n = P / (12*M - P*r/100)`` or the months/rate equivalent), proving
construction and the direct algebraic route agree (`check_equivalent`).

Four tiers, genuine reasoning escalation:

- ``years`` (easy): solve for the whole number of YEARS (2-6), single
  rearrangement.
- ``months`` (medium): solve for the whole number of MONTHS (a two-digit
  count, 13-95) -- a genuinely different unit, not merely a bigger easy
  question.
- ``term_difference`` (hard): two repayment scenarios share the same
  principal and rate; the question asks for the DIFFERENCE, in years,
  between the two resulting terms -- solve twice, subtract, a different
  reasoning shape from a single-shot solve.
- ``max_rate`` (exam): given the desired term and the repayment amount,
  find the MAXIMUM flat annual rate that still clears the loan within that
  term -- solves for ``r`` instead of ``n``, a genuinely different unknown.

Misconceptions checked, per tier, via ``check_pairwise_distinct`` on the
underlying ``Fraction``/``int`` values (never the display string) so no two
options can collide after rounding: years/months unit confusion (reporting
the months count as if it were years, or vice versa), ignoring interest
entirely (dividing the bare principal by the repayment), and a
circular-reasoning trap that plugs a wrong assumed term into the total-
repayable formula before dividing back out.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    pick_distractors,
    round_half_up,
)

_RESAMPLE_ATTEMPTS = 300


def _year_noun(n: int) -> str:
    return "year" if n == 1 else "years"


def _floor_fraction(value: Fraction) -> int:
    """Floor a non-negative Fraction to an int. Used for the "ignored
    interest" and "circular reasoning trap" misconceptions: both are
    provably strictly LESS than the correct term (see module note below),
    so flooring -- rather than rounding to nearest -- guarantees the
    misconception's displayed value never collides with the correct answer,
    by construction, instead of relying on resampling to dodge a collision
    that a small-term draw hits on almost every attempt."""
    return math.floor(value)


# =============================================================================
# Shared algebra
# =============================================================================


def _n_from_P_r_M(P: Fraction, r: Fraction, M: Fraction) -> Fraction:
    """Forward, independent solve for the number of YEARS ``n`` given
    principal ``P``, flat annual rate ``r`` (%) and monthly repayment ``M``,
    from ``M * 12 * n = P * (1 + r*n/100)`` rearranged to
    ``n = P / (12*M - P*r/100)``. Used only to VERIFY construction, never to
    build the question."""
    denom = 12 * M - P * r / 100
    check(denom != 0, "n_from_P_r_M: degenerate denominator")
    return P / denom


def _r_from_P_n_M(P: Fraction, n: Fraction, M: Fraction) -> Fraction:
    """Forward, independent solve for the flat annual rate ``r`` given
    principal ``P``, term ``n`` (years) and monthly repayment ``M``, from
    the same identity rearranged for ``r``:
    ``r = 100 * (12*M*n - P) / (P*n)``."""
    check(P != 0 and n != 0, "r_from_P_n_M: degenerate denominator")
    return 100 * (12 * M * n - P) / (P * n)


# =============================================================================
# easy: years
# =============================================================================

_EASY_R = [3, 4, 5, 6, 7, 8, 9, 10]
_EASY_N = [2, 3, 4, 5, 6]
_EASY_K = list(range(3, 13))

_YEARS_TEMPLATES = [
    "A ${P} flat-interest loan charges {r}% per year in interest. The borrower "
    "repays ${M} per month. How many years will it take to repay the loan in full?",
    "Sam takes out a ${P} loan at a flat interest rate of {r}% p.a. Repayments "
    "are ${M} each month. How many years does it take to pay off the loan?",
    "A flat-interest loan of ${P} is charged at {r}% per annum. Monthly "
    "repayments of ${M} are made until the loan is cleared. Find the number of "
    "years required to repay the loan.",
]


def _build_years(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        r = rng.choice(_EASY_R)
        n = rng.choice(_EASY_N)
        k = rng.choice(_EASY_K)
        denom = 100 + r * n
        P = 1200 * n * k
        M = k * denom
        # Bounds wide enough that every (n, k) combination in the ranges above passes -- P tops out
        # at 1200*6*12 = 86400, M at 12*(100+10*6) = 1920 -- so `n`'s own draw is never filtered
        # unevenly by principal size.
        if not (5000 <= P <= 90000 and 100 <= M <= 3000):
            continue

        Pf, rf, Mf, nf = Fraction(P), Fraction(r), Fraction(M), Fraction(n)

        # Two misconception distractors, both guaranteed distinct from `n` by construction rather
        # than by resampling -- `months_as_years` (12n) is always >= 24 while `n` is at most 6, and
        # `ignored_interest` = floor(100n/(100+rn)) is ALWAYS strictly less than n whenever r > 0
        # (100n < n(100+rn) for r,n > 0), so flooring can never round it back up to n.
        months_as_years = Fraction(12 * n)  # unit confusion
        ignored_interest = Fraction(_floor_fraction(Pf / (12 * Mf)))

        try:
            check_pairwise_distinct(
                nf,
                months_as_years,
                ignored_interest,
                msg="years-tier misconception collides with the answer or another option",
            )
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p12b_loan_term_from_repayment: no collision-free years draw found")

    # Independent forward check: prove the direct algebraic solve agrees.
    n_check = _n_from_P_r_M(Pf, rf, Mf)
    check_equivalent(n_check, nf, "years-tier: forward solve must reproduce the constructed term")

    template = _YEARS_TEMPLATES[rng.randrange(len(_YEARS_TEMPLATES))]
    text = template.format(P=f"{P:,}", r=r, M=f"{M:,}")
    latex = text
    answer_text = str(n)

    distractors = pick_distractors(
        answer_text,
        [str(int(months_as_years)), str(int(ignored_interest))],
        n=2,
    )
    check(len(distractors) == 2, "years tier expects 2 distinct distractors")

    return dict(
        text=text,
        latex=latex,
        answer=n,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Total repayable = P(1 + rn/100). Solve M × 12n = P(1 + rn/100) for n: "
            f"n = P / (12M - Pr/100) = {P} / (12 × {M} - {P} × {r}/100) = {n} years."
        ],
        distractors=distractors,
    )


# =============================================================================
# medium: months
# =============================================================================

_MEDIUM_R = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_MEDIUM_M = list(range(13, 96))
_MEDIUM_K = [1, 2, 3]

_MONTHS_TEMPLATES = [
    "A ${P} flat-interest loan charges {r}% per year in interest. The borrower "
    "repays ${M} per month. How many MONTHS will it take to repay the loan in full?",
    "Priya borrows ${P} at a flat interest rate of {r}% p.a. She repays ${M} "
    "each month. How many months does it take her to clear the loan?",
    "A flat-interest loan of ${P} is charged at {r}% per annum, repaid in "
    "monthly instalments of ${M}. Find the exact number of months needed to "
    "repay the loan in full.",
]


def _build_months(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        r = rng.choice(_MEDIUM_R)
        m = rng.choice(_MEDIUM_M)
        if m % 12 == 0:
            continue  # keep the answer genuinely a "months" count, not years in disguise
        k = rng.choice(_MEDIUM_K)
        denom = 1200 + r * m
        P = 1200 * m * k
        M = k * denom
        if not (5000 <= P <= 60000 and 100 <= M <= 3000):
            continue

        Pf, rf, Mf, mf = Fraction(P), Fraction(r), Fraction(M), Fraction(m)

        years_as_months = round_half_up(mf / 12, 0)  # unit confusion, reversed
        # Both are provably strictly less than m by the same argument as the
        # years tier (1200 in place of 100): flooring guarantees distinctness.
        total_wrong_n1 = Pf * (1 + rf * Fraction(1, 12) / 100)  # circular trap: 1 month
        circular_trap = Fraction(_floor_fraction(total_wrong_n1 / Mf))
        ignored_interest = Fraction(_floor_fraction(Pf / Mf))

        try:
            check_pairwise_distinct(
                mf,
                years_as_months,
                circular_trap,
                ignored_interest,
                msg="months-tier misconception collides with the answer or another option",
            )
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p12b_loan_term_from_repayment: no collision-free months draw found")

    # Independent forward check via the years-unit solve, converted to months.
    n_years_check = _n_from_P_r_M(Pf, rf, Mf)
    check_equivalent(n_years_check * 12, mf, "months-tier: forward solve (x12) must reproduce the constructed term")

    template = _MONTHS_TEMPLATES[rng.randrange(len(_MONTHS_TEMPLATES))]
    text = template.format(P=f"{P:,}", r=r, M=f"{M:,}")
    latex = text
    answer_text = str(m)

    distractors = pick_distractors(
        answer_text,
        [str(int(years_as_months)), str(int(circular_trap)), str(int(ignored_interest))],
        n=3,
    )
    check(len(distractors) >= 1, "months tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=m,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Total repayable = P(1 + r(m/12)/100). Solve M × m = P(1 + r(m/12)/100) for m "
            f"(equivalently, n = P/(12M - Pr/100) years, then m = 12n): m = {m} months."
        ],
        distractors=distractors,
    )


# =============================================================================
# hard: term_difference
# =============================================================================

_HARD_R = [3, 4, 5, 6, 7, 8, 9, 10]
_HARD_N = [2, 3, 4, 5, 6]
_HARD_K = [1, 2, 3]

_DIFF_TEMPLATES = [
    "A ${P} flat-interest loan charges {r}% per year in interest. Repaying "
    "${M1} per month clears the loan in one plan; repaying ${M2} per month "
    "clears it in another. How many fewer years does the higher repayment "
    "take, compared with the lower one?",
    "Two repayment plans exist for the same ${P} loan at a flat rate of {r}% "
    "p.a.: Plan A repays ${M1} per month, Plan B repays ${M2} per month. Find "
    "the difference, in years, between the two plans' repayment terms.",
    "Deng is comparing two ways to repay a ${P} flat-interest loan charged at "
    "{r}% per annum: ${M1} per month, or ${M2} per month. By how many years is "
    "the loan cleared sooner under the higher monthly repayment?",
]


def _build_term_difference(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        r = rng.choice(_HARD_R)
        # Draw the DIFFERENCE itself uniformly first, then pick a compatible
        # higher-term value -- an unconstrained rng.sample over _HARD_N
        # produces a skewed difference distribution (diff=1 outnumbers
        # diff=4 five-to-one, since there are far more adjacent pairs than
        # extreme ones), which the 200-seed concentration measure caught at
        # 39% for diff=1. Construct-to-target on the ANSWER itself instead.
        diff = rng.choice([1, 2, 3, 4])
        n_higher = rng.choice([v for v in _HARD_N if v + diff <= max(_HARD_N)])
        n_lower = n_higher + diff
        n1, n2 = n_lower, n_higher  # n1: lower repayment/longer term, n2: higher repayment/shorter term
        k = rng.choice(_HARD_K)
        P = 1200 * n1 * n2 * k
        M1 = n2 * k * (100 + r * n1)  # repayment that clears the loan in n1 years
        M2 = n1 * k * (100 + r * n2)  # repayment that clears the loan in n2 years
        if not (5000 <= P <= 90000 and 100 <= M1 <= 3500 and 100 <= M2 <= 3500):
            continue
        if M1 == M2:
            continue

        answer = abs(n1 - n2)
        Pf, rf = Fraction(P), Fraction(r)
        M1f, M2f = Fraction(M1), Fraction(M2)

        summed_instead = n1 + n2  # added instead of subtracted
        ignored_second = max(n1, n2)  # forgot to solve the second scenario
        months_not_years = 12 * answer  # unit confusion, easy-tier's reversal

        try:
            check_pairwise_distinct(
                answer,
                summed_instead,
                ignored_second,
                months_not_years,
                msg="term_difference misconception collides with the answer or another option",
            )
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p12b_loan_term_from_repayment: no collision-free term_difference draw found")

    # Independent forward check: re-derive BOTH terms from the stated P, r and
    # each repayment separately, then subtract -- never trust the construction
    # direction (n2*k*(100+r*n1) etc.) alone.
    n1_check = _n_from_P_r_M(Pf, rf, M1f)
    n2_check = _n_from_P_r_M(Pf, rf, M2f)
    check_equivalent(n1_check, Fraction(n1), "term_difference: forward solve for plan A must reproduce n1")
    check_equivalent(n2_check, Fraction(n2), "term_difference: forward solve for plan B must reproduce n2")
    check_equivalent(
        abs(n1_check - n2_check),
        Fraction(answer),
        "term_difference: |n1_check - n2_check| must equal the constructed difference",
    )

    higher_M, lower_M = (M1, M2) if M1 > M2 else (M2, M1)
    template = _DIFF_TEMPLATES[rng.randrange(len(_DIFF_TEMPLATES))]
    text = template.format(P=f"{P:,}", r=r, M1=f"{lower_M:,}", M2=f"{higher_M:,}")
    latex = text
    answer_text = str(answer)

    distractors = pick_distractors(
        answer_text,
        [str(int(summed_instead)), str(int(ignored_second)), str(int(months_not_years))],
        n=3,
    )
    check(len(distractors) >= 1, "term_difference tier expects at least 1 distinct distractor")

    n_lower_M, n_higher_M = (n1, n2) if M1 < M2 else (n2, n1)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Solve n for each plan via n = P / (12M - Pr/100): the ${lower_M:,}/month plan takes "
            f"{n_lower_M} {_year_noun(n_lower_M)}, the ${higher_M:,}/month plan takes {n_higher_M} "
            f"{_year_noun(n_higher_M)}. Difference = {max(n_lower_M, n_higher_M)} - "
            f"{min(n_lower_M, n_higher_M)} = {answer} {_year_noun(answer)}."
        ],
        distractors=distractors,
    )


# =============================================================================
# exam: max_rate
# =============================================================================

_EXAM_R = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
_EXAM_N = [2, 3, 4, 5, 6]
_EXAM_K = list(range(3, 13))

_MAX_RATE_TEMPLATES = [
    "A borrower can afford ${M} per month and wants to clear a ${P} loan within "
    "{n} years. What is the MAXIMUM flat annual interest rate (%) the loan "
    "could charge for this to be possible?",
    "A ${P} loan is to be repaid within {n} years at ${M} per month under a "
    "flat-interest scheme. Find the highest flat annual rate (%) for which the "
    "loan is still fully repaid within {n} years.",
    "Wei can repay ${M} per month towards a ${P} loan and needs it cleared "
    "within {n} years. Assuming flat interest, what is the maximum annual "
    "interest rate (%) the loan can carry?",
]


def _build_max_rate(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        r = rng.choice(_EXAM_R)
        n = rng.choice(_EXAM_N)
        k = rng.choice(_EXAM_K)
        denom = 100 + r * n
        P = 1200 * n * k
        M = k * denom
        if not (5000 <= P <= 60000 and 100 <= M <= 3000):
            continue

        Pf, nf, Mf = Fraction(P), Fraction(n), Fraction(M)

        # All three are provably POSITIVE and provably != r, by algebra on
        # the identity r = 1200M/P - 100/n (the rearranged construction):
        # `forgot_subtract_P` = 1200M/P = r + 100/n (omits the -100/n term
        # entirely, always > r by exactly 100/n, never zero since n is
        # finite); `forgot_divide_by_n` = r*n (never equal to r since n>=2);
        # `base_is_total` = rn*100/(100+rn) (always positive since r,n>0).
        # An earlier "forgot to annualise" formula, 100(Mn-P)/(Pn), was
        # dropped: it reduces to 100k*n*(rn-1100)/(Pn), which is NEGATIVE
        # for every draw this generator can make (rn <= 15*6 = 90 << 1100)
        # -- a defect the render caught (every exam-tier row showed a
        # negative "maximum interest rate", implausible on sight, so it was
        forgot_subtract_P = 1200 * Mf / Pf  # omits the -100/n term
        forgot_divide_by_n = 100 * (12 * Mf * nf - Pf) / Pf  # un-annualised rate
        base_is_total = 100 * (12 * Mf * nf - Pf) / (12 * Mf * nf)  # divided by total, not P

        try:
            check_pairwise_distinct(
                Fraction(r),
                forgot_subtract_P,
                forgot_divide_by_n,
                base_is_total,
                msg="max_rate misconception collides with the answer or another option",
            )
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p12b_loan_term_from_repayment: no collision-free max_rate draw found")

    # Independent forward check: re-derive r from the stated P, n and M alone.
    r_check = _r_from_P_n_M(Pf, nf, Mf)
    check_equivalent(r_check, Fraction(r), "max_rate: forward solve for r must reproduce the constructed rate")

    template = _MAX_RATE_TEMPLATES[rng.randrange(len(_MAX_RATE_TEMPLATES))]
    text = template.format(P=f"{P:,}", n=n, M=f"{M:,}")
    latex = text
    answer_text = str(r)

    def _fmt(value: Fraction) -> str:
        return str(int(round_half_up(value, 0)))

    distractors = pick_distractors(
        answer_text,
        [_fmt(forgot_subtract_P), _fmt(forgot_divide_by_n), _fmt(base_is_total)],
        n=3,
    )
    check(len(distractors) >= 1, "max_rate tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=latex,
        answer=r,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Solve M × 12n = P(1 + rn/100) for r: r = 100(12Mn - P) / (Pn) = "
            f"100(12 × {M} × {n} - {P}) / ({P} × {n}) = {r}%."
        ],
        distractors=distractors,
    )


@generator(
    "p12b_loan_term_from_repayment",
    level=12,
    presets={
        "easy": dict(mode="years"),
        "medium": dict(mode="months"),
        "hard": dict(mode="term_difference"),
        "exam": dict(mode="max_rate"),
    },
)
def p12b_loan_term_from_repayment(rng, mode="years"):
    """The number of repayment periods needed to clear a flat-interest loan
    given its principal, flat annual rate and repayment amount -- the
    deliberate inverse of ``flat_loan_repayment``. easy: solve for whole
    YEARS. medium: solve for whole MONTHS, a genuinely different unit.
    hard: two repayment plans share the same principal and rate; find the
    DIFFERENCE in years between their terms. exam: given the desired term
    and repayment amount, find the MAXIMUM flat annual rate that still
    clears the loan in time -- solves for the rate instead of the term."""
    if mode == "years":
        return _build_years(rng)
    if mode == "months":
        return _build_months(rng)
    if mode == "term_difference":
        return _build_term_difference(rng)
    if mode == "max_rate":
        return _build_max_rate(rng)
    raise ValueError(f"unknown mode: {mode!r}")
