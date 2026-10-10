"""Logarithms as indices, solving indices with logs.

Two generators:

- ``logarithms_as_indices`` (``senior-define-logarithms-as-indices-a-x-b-is``,
  ACMMM151) -- define logarithms as indices using the equivalence a^x = b <=> x = log_a(b).
  Three tiers: easy (find exponent), medium (find base result), hard (negative exponent).

- ``solve_indices_with_logarithms`` (``senior-solve-equations-involving-indices-using-
  logarithms``, ACMMM155) -- solve a^x = b (with variations) for x by applying logarithms,
  where b is not a clean integer power of a. Three tiers: easy (direct), medium (coefficient),
  hard (linear exponent).

Both skills use exact Fraction arithmetic for skill C and decimal 3-place rounding for skill D.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, pick_distractors


# =============================================================================
# Skill C: logarithms_as_indices
# =============================================================================


def _find_exponent(a: int, b: int, lo: int = 0, hi: int = 15) -> int | None:
    """Brute-force search for exponent (never trusting the construction)."""
    for candidate in range(lo, hi + 1):
        if a**candidate == b:
            return candidate
    return None


def _repeated_multiply(a: int, x: int) -> int:
    """Verify exponentiation via repeated multiplication (never Python's **)."""
    result = 1
    for _ in range(x):
        result *= a
    return result


def _find_exponent_signed(a: int, target: Fraction, lo: int = -10, hi: int = 10) -> int | None:
    """Brute-force search for signed exponent using Fraction arithmetic."""
    for candidate in range(lo, hi + 1):
        value = Fraction(a) ** candidate
        if value == target:
            return candidate
    return None


def _build_easy_find_x(rng) -> dict:
    """Easy: find x given a^x = b (index form given, unknown is exponent)."""
    a = rng.choice([2, 3, 5])
    x = rng.randint(2, 4)
    b = a**x

    text = f"{a}^x = {b}. Using the definition of a logarithm, find the value of x."
    answer = x
    answer_text = str(x)

    # Verify via independent check
    _find_exponent(a, b) == x or check(False, "easy independent check failed")

    # Misconceptions:
    # - states-b-instead-of-x: wrong1 = b (the computed power instead of exponent)
    # - off-by-one-exponent: wrong2 = x + 1
    # Both are always distinct from x for the range a>=2, x>=2 (e.g., a=2, x=2 gives b=4 != 2=x)
    wrong1 = str(b)
    wrong2 = str(x + 1)

    candidates = [wrong1, wrong2]

    return dict(
        text=text,
        latex=f"{a}^x = {b}",
        answer=answer,
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_medium_find_b(rng) -> dict:
    """Medium: find b given log_a(b) = x (log form given, unknown is b)."""
    a = rng.randint(2, 6)
    x = rng.randint(2, 5)
    # Enforce a != x for distractor distinctness
    while x == a:
        x = rng.randint(2, 5)

    target_b = a**x

    text = f"Suppose log_{a}(b) = {x}. Using the definition of a logarithm, find the value of b."
    answer = target_b
    answer_text = str(target_b)

    # Verify via independent check
    check(_repeated_multiply(a, x) == target_b, "medium independent check failed")

    # Misconceptions:
    # - swaps-a-and-x-roles: wrong1 = x ** a (raises x to power a)
    # - treats-the-exponent-as-a-multiplier: wrong2 = a * x
    # Enforce a != x to ensure swaps case is distinct.
    # Also exclude a==2, x==2 together to ensure multiplication case is distinct (2**2 = 2*2 = 4)
    check(a != x, "a should not equal x by re-sampling")
    check(not (a == 2 and x == 2), "a==2 and x==2 should be re-sampled")

    wrong1 = str(x**a)
    wrong2 = str(a * x)

    candidates = [wrong1, wrong2]

    return dict(
        text=text,
        latex=f"\\log_{{{a}}}(b) = {x}",
        answer=answer,
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_hard_find_negative_x(rng) -> dict:
    """Hard: find x given a^x = 1/a^k (unit fraction, unknown is negative exponent)."""
    a = rng.choice([2, 3, 5])
    k = rng.randint(1, 3)
    x = -k
    b = Fraction(1, a**k)
    b_str = str(b)  # Renders as "1/8" etc.

    text = f"{a}^x = {b_str}. Using the definition of a logarithm, find the value of x."
    answer = x  # Negative Python int
    answer_text = str(x)

    # Verify via independent check
    check(_find_exponent_signed(a, b) == -k, "hard independent check failed")

    # Misconceptions:
    # - off-by-one-exponent: wrong1 = -k - 1 (one step further negative)
    # - ignores-the-negative-sign: wrong2 = k (magnitude only)
    wrong1 = str(-k - 1)
    wrong2 = str(k)

    candidates = [wrong1, wrong2]

    return dict(
        text=text,
        latex=f"{a}^x = {b_str}",
        answer=answer,
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "logarithms_as_indices",
    level=11,
    presets={
        "easy": dict(variant="find_x"),
        "medium": dict(variant="find_b"),
        "hard": dict(variant="find_negative_x"),
    },
)
def logarithms_as_indices(rng, variant="find_x"):
    """Define logarithms as indices using the equivalence a^x = b <=> x = log_a(b)."""
    if variant == "find_x":
        return _build_easy_find_x(rng)
    if variant == "find_b":
        return _build_medium_find_b(rng)
    if variant == "find_negative_x":
        return _build_hard_find_negative_x(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# =============================================================================
# Skill D: solve_indices_with_logarithms
# =============================================================================


def _verify_log_solution(target: float, a: int, x_rounded: float) -> None:
    """Independent check using base-10 change-of-base (different code path from
    math.log(target, a) construction)."""
    x_alt = math.log10(target) / math.log10(a)
    check(abs(x_alt - x_rounded) < 0.001, "base-10 change-of-base recomputation disagrees with constructed solution")


def _verify_log_solution_hard(target: float, a: int, k: int, x_rounded: float) -> None:
    """Hard tier verification: log_a(target) is computed independently, then -k
    is subtracted and rounded, matching the hard-tier construction."""
    x_alt = math.log10(target) / math.log10(a)
    x_alt_adjusted = round(x_alt - k, 3)
    check(
        abs(x_alt_adjusted - x_rounded) < 0.001, "base-10 change-of-base recomputation with linear adjustment disagrees"
    )


def _is_clean_power(a: int, target: float, tolerance: float = 1e-10) -> bool:
    """Check if target is a clean integer power of a (for guard against degenerate cases)."""
    if target <= 0:
        return False
    try:
        log_val = math.log(target, a)
        rounded_exp = round(log_val)
        if abs(log_val - rounded_exp) > 0.01:  # Not close to an integer exponent
            return False
        return abs(a**rounded_exp - target) < tolerance
    except (ValueError, ZeroDivisionError):
        return False


def _build_easy_direct(rng) -> dict:
    """Easy: solve a^x = b for x (single log step)."""
    a = rng.randint(2, 9)
    b = rng.randint(10, 500)

    # Guard: re-sample b if it's a clean power of a
    while _is_clean_power(a, b):
        b = rng.randint(10, 500)

    x_exact = math.log(b, a)
    x_rounded = round(x_exact, 3)

    text = f"Solve {a}^x = {b} for x. Give your answer to 3 decimal places."
    answer = x_rounded  # Plain float
    answer_text = f"{x_rounded:.3f}"

    # Verify via independent check
    _verify_log_solution(b, a, x_rounded)

    # Misconceptions:
    # - swaps-base-and-argument: wrong1 = log_b(a) instead of log_a(b)
    # - treats-index-as-linear-division: wrong2 = b / a
    wrong1 = math.log(a, b)
    wrong2 = b / a

    # Guard: ensure candidates don't collide with answer_text
    candidates_raw = [
        f"{wrong1:.3f}",
        f"{round(wrong2, 3):.3f}",
    ]
    candidates = [c for c in candidates_raw if c != answer_text]

    return dict(
        text=text,
        latex=rf"{a}^x = {b} \ \text{{Give your answer to 3 decimal places}}",
        answer=answer,
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates_raw),
    )


def _build_medium_coefficient(rng) -> dict:
    """Medium: solve c * a^x = b for x (isolate exponential term first)."""
    a = rng.randint(2, 9)
    c = rng.randint(2, 9)
    b = rng.randint(10, 500)

    target = b / c

    # Guard: re-sample b if target is a clean power of a
    attempts = 0
    while _is_clean_power(a, target) and attempts < 100:
        b = rng.randint(10, 500)
        target = b / c
        attempts += 1

    x_exact = math.log(target, a)
    x_rounded = round(x_exact, 3)

    text = f"Solve {c} × {a}^x = {b} for x. Give your answer to 3 decimal places."
    answer = x_rounded
    answer_text = f"{x_rounded:.3f}"

    # Verify via independent check
    _verify_log_solution(target, a, x_rounded)

    # Misconceptions:
    # - swaps-base-and-argument: wrong1 = log_target(a)
    # - treats-index-as-linear-division: wrong2 = target / a
    # - forgets-the-coefficient-adjustment: wrong3 = log_a(b) instead of log_a(b/c)
    wrong1 = math.log(a, target)
    wrong2 = target / a
    wrong3 = round(math.log(b, a), 3)

    candidates_raw = [
        f"{wrong1:.3f}",
        f"{round(wrong2, 3):.3f}",
        f"{wrong3:.3f}",
    ]
    candidates = [c for c in candidates_raw if c != answer_text]

    return dict(
        text=text,
        latex=rf"{c} \cdot {a}^x = {b} \ \text{{Give your answer to 3 decimal places}}",
        answer=answer,
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates_raw),
    )


def _build_hard_linear_exponent(rng) -> dict:
    """Hard: solve a^(x + k) = b for x (linear expression in exponent, rearrange after log)."""
    a = rng.randint(2, 9)
    k = rng.randint(1, 5)
    b = rng.randint(10, 500)

    # Guard: re-sample b if it's a clean power of a
    while _is_clean_power(a, b):
        b = rng.randint(10, 500)

    x_exact = math.log(b, a) - k
    x_rounded = round(x_exact, 3)

    text = f"Solve {a}^(x + {k}) = {b} for x. Give your answer to 3 decimal places."
    answer = x_rounded
    answer_text = f"{x_rounded:.3f}"

    # Verify via independent check (hard-tier specific, with linear adjustment)
    _verify_log_solution_hard(b, a, k, x_rounded)

    # Misconceptions:
    # - swaps-base-and-argument: wrong1 = log_b(a)
    # - treats-index-as-linear-division: wrong2 = b / a
    # - forgets-the-linear-adjustment: wrong3 = log_a(b) (without the -k)
    wrong1 = math.log(a, b)
    wrong2 = b / a
    wrong3 = round(math.log(b, a), 3)

    candidates_raw = [
        f"{wrong1:.3f}",
        f"{round(wrong2, 3):.3f}",
        f"{wrong3:.3f}",
    ]
    candidates = [c for c in candidates_raw if c != answer_text]

    return dict(
        text=text,
        latex=rf"{a}^{{(x + {k})}} = {b} \ \text{{Give your answer to 3 decimal places}}",
        answer=answer,
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates_raw),
    )


@generator(
    "solve_indices_with_logarithms",
    level=11,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="coefficient"),
        "hard": dict(variant="linear_exponent"),
    },
)
def solve_indices_with_logarithms(rng, variant="direct"):
    """Solve equations involving indices using logarithms."""
    if variant == "direct":
        return _build_easy_direct(rng)
    if variant == "coefficient":
        return _build_medium_coefficient(rng)
    if variant == "linear_exponent":
        return _build_hard_linear_exponent(rng)
    raise ValueError(f"unknown variant: {variant!r}")
