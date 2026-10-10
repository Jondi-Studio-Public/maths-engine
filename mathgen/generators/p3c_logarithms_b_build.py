"""Natural logarithm and geometric nth-term formula, both senior ``acara-senior-v8-4``, domain
``functions-sequences``, ``answer: number``.

Two generators:

- ``natural_logarithm_define`` (``senior-define-the-natural-logarithm-ln-x-log-ex``, ACMMM159)
  -- assess the definitional relationship between ln and e as inverse operations:
  ln(e^k) = k exactly (easy), general natural-log evaluation ln(x) (medium, requiring the actual
  ln function), and the reverse direction e^k given ln(x) = k (hard).

- ``geometric_nth_term_formula`` (``senior-use-the-formula-t-n-r-n-1``, ACMMM073)
  -- apply or algebraically invert the closed-form geometric nth-term formula t_n = r^(n-1) * t_1
  with integer inputs: easy finds t_n, medium finds r, hard handles negative r.

Both generators use exact arithmetic and format ``answer_text`` explicitly.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, pick_distractors


# --- natural_logarithm_define ------------------------------------------------


def _verify_ln_of_e_power(k: int, expected: int) -> None:
    """Independent check for easy tier: numeric round trip e^k -> ln(...) recovers k."""
    recomputed = math.log(math.e**k)
    check(abs(recomputed - k) < 1e-9, "numeric round trip e**k -> ln(...) does not recover k")
    check(recomputed == expected, f"expected {expected}, but got {recomputed}")


def _verify_ln_general(x: int, expected: float) -> None:
    """Independent check for medium tier: change-of-base via base-10 logs."""
    x_alt = math.log10(x) / math.log10(math.e)
    check(abs(x_alt - expected) < 0.001, "base-10 change-of-base recomputation disagrees")


def _verify_e_to_the_k(x_rounded: float, k: float) -> None:
    """Independent check for hard tier: ln of computed x recovers the stated k."""
    check(
        abs(math.log(x_rounded) - k) < 0.001,
        "ln of the computed x does not recover the stated k within rounding tolerance",
    )


@generator(
    "natural_logarithm_define",
    level=11,
    presets={
        "easy": dict(variant="ln_of_e_power"),
        "medium": dict(variant="ln_general"),
        "hard": dict(variant="e_to_the_k"),
    },
)
def natural_logarithm_define(
    rng,
    variant="ln_of_e_power",
):
    """Define the natural logarithm as ln x = log_e x. Assess the definitional
    relationship and its immediate inverse-function consequence in both directions."""

    if variant == "ln_of_e_power":
        # Easy: ln(e^k) = k exactly (integer answer, no decimal point)
        k = rng.randint(2, 5)  # Fixed to skip k=1 to avoid distractor collision
        answer = k
        answer_text = str(k)
        text = f"Evaluate ln(e^{k})."

        _verify_ln_of_e_power(k, answer)

        # Misconceptions:
        # The key is a whole number, so the distractors are whole numbers too
        # (a decimal point would give the key away).
        # 1. treats-e-to-the-k-as-the-answer: computes e^k directly (to a whole number)
        # 2. multiplies-by-e: reads ln(e^k) as k * e
        # 3. squares-the-exponent
        candidates = [
            str(round(math.e**k)),
            str(round(math.e * k)),
            str(k * k),
        ]

        return dict(
            text=text,
            latex=rf"\ln(e^{{{k}}})",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"ln(e^{k}) = {k} (by definition of natural logarithm)."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "ln_general":
        # Medium: ln(x) for general x, to 3 decimal places
        x = rng.randint(2, 20)
        value_exact = math.log(x)
        value_rounded = round(value_exact, 3)
        answer = value_rounded
        answer_text = f"{value_rounded:.3f}"
        text = f"Evaluate ln({x}). Give your answer to 3 decimal places."

        _verify_ln_general(x, value_rounded)

        # Misconceptions:
        # 1. uses-log-base-10-instead-of-e: computes log10(x)
        wrong3 = round(math.log10(x), 3)
        # 2. off-by-a-sign: negates the result
        wrong4 = round(-value_exact, 3)

        candidates = [
            f"{wrong3:.3f}",
            f"{wrong4:.3f}",
        ]

        return dict(
            text=text,
            latex=rf"\ln({x}) \ \text{{Give your answer to 3 decimal places}}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"ln({x}) ≈ {answer_text} (to 3 decimal places)."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "e_to_the_k":
        # Hard: given ln(x) = k, find x, to 3 decimal places
        # Note: k=1 is excluded to avoid distractor collision
        k = rng.choice([2, 3, 4, 0.5, 1.5, 2.5])
        x_exact = math.e**k
        x_rounded = round(x_exact, 3)
        answer = x_rounded
        answer_text = f"{x_rounded:.3f}"
        text = f"ln(x) = {k}. Find the value of x, to 3 decimal places."

        _verify_e_to_the_k(x_rounded, k)

        # Misconceptions:
        # 1. uses-k-directly-ignoring-the-exponentiation: states k as the answer
        wrong5 = k
        # 2. multiplies-e-by-k-instead-of-exponentiating: e*k
        wrong6 = round(math.e * k, 3)

        candidates = [
            f"{wrong5:.3f}",
            f"{wrong6:.3f}",
        ]

        return dict(
            text=text,
            latex=rf"x = e^{{{k}}} \ \text{{Find the value of x, to 3 decimal places}}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"If ln(x) = {k}, then x = e^{k} ≈ {answer_text}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    raise ValueError(f"unknown variant: {variant!r}")


# --- geometric_nth_term_formula -----------------------------------------------


def _iterative_geometric_term(t1: int, r: int, n: int) -> int:
    """Independent check for easy/hard: compute via multiplication loop, never **."""
    value = t1
    for _ in range(n - 1):
        value *= r
    return value


def _find_ratio(t1: int, t_n: int, n: int, hi: int = 12) -> int | None:
    """Independent check for medium: brute-force search for positive integer r."""
    for candidate in range(1, hi + 1):
        if candidate ** (n - 1) * t1 == t_n:
            return candidate
    return None


def _find_n(t1: int, r: int, t_n: int, hi: int = 12) -> int | None:
    """Independent check for find_n: brute-force search for the position n by repeated multiplication, never by
    solving r**(n-1) == t_n/t1 with logs or exponent inversion."""
    value = t1
    for n in range(1, hi + 1):
        if value == t_n:
            return n
        value *= r
    return None


def _verify_easy_geometric_term(t1: int, r: int, n: int, expected: int) -> None:
    """Independent check for easy tier: iterative multiplication."""
    recomputed = _iterative_geometric_term(t1, r, n)
    check(recomputed == expected, "iterative geometric term computation mismatch")


def _verify_medium_geometric_ratio(t1: int, t_n: int, n: int, expected: int) -> None:
    """Independent check for medium tier: brute-force search."""
    recomputed = _find_ratio(t1, t_n, n)
    check(recomputed == expected, "brute-force ratio search failed to find correct r")


def _verify_hard_geometric_term(t1: int, r: int, n: int, expected: int) -> None:
    """Independent check for hard tier: iterative multiplication with negative r."""
    recomputed = _iterative_geometric_term(t1, r, n)
    check(recomputed == expected, "iterative geometric term with negative ratio mismatch")


def _verify_find_n(t1: int, r: int, t_n: int, expected: int) -> None:
    """Independent check for find_n: brute-force search."""
    recomputed = _find_n(t1, r, t_n)
    check(recomputed == expected, "brute-force search for n failed to find the correct position")


@generator(
    "geometric_nth_term_formula",
    level=11,
    presets={
        "easy": dict(variant="find_t_n"),
        "medium": dict(variant="find_r"),
        "hard": dict(variant="negative_ratio"),
    },
)
def geometric_nth_term_formula(
    rng,
    variant="find_t_n",
):
    """Use the formula t_n = r^(n-1) * t_1 to find the nth term or common ratio
    of a geometric sequence with integer inputs."""

    if variant == "find_t_n":
        # Easy: direct substitution to find t_n
        r = rng.randint(2, 4)
        t1 = rng.randint(2, 9)
        n = rng.randint(3, 5)
        answer = r ** (n - 1) * t1
        answer_text = str(answer)
        text = f"A geometric sequence has t1 = {t1} and common ratio r = {r}. Using t_n = r^(n-1) × t1, find t_{n}."

        _verify_easy_geometric_term(t1, r, n, answer)

        # Misconceptions:
        # 1. adds-instead-of-multiplying-repeatedly: arithmetic sequence formula
        wrong1 = t1 + r * (n - 1)
        # 2. off-by-one-exponent: uses n instead of n-1
        wrong2 = r**n * t1
        # 3. forgets-t1-factor: drops t1
        wrong3 = r ** (n - 1)

        candidates = [
            str(wrong1),
            str(wrong2),
            str(wrong3),
        ]

        return dict(
            text=text,
            latex=rf"t_{{{n}}} = {r}^{{{n - 1}}} \times {t1}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"t_{n} = {r}^{n - 1} × {t1} = {r ** (n - 1)} × {t1} = {answer}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "find_r":
        # Medium: solve for r (the common ratio)
        r = rng.randint(2, 5)
        n = rng.randint(3, 5)
        t1 = rng.randint(2, 9)
        t_n = r ** (n - 1) * t1
        answer = r
        answer_text = str(answer)
        text = f"A geometric sequence has t1 = {t1}, t_{n} = {t_n}, and n = {n}. Using t_n = r^(n-1) × t1, find the common ratio r."

        _verify_medium_geometric_ratio(t1, t_n, n, answer)

        # Misconceptions (different from easy, since unknown is r):
        # 1. uses-the-full-ratio-without-taking-the-root: uses t_n/t1 directly
        wrong4 = t_n // t1
        # 2. off-by-one-ratio: r+1
        wrong5 = r + 1

        candidates = [
            str(wrong4),
            str(wrong5),
        ]

        return dict(
            text=text,
            latex=rf"r = \sqrt[{{{n - 1}}}]{{\frac{{t_{n}}}{{t_1}}}}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"t_n = r^(n-1) × t1, so r^{n - 1} = t_n / t1 = {t_n} / {t1} = {t_n / t1}, so r = {answer}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "negative_ratio":
        # Hard: find t_n with negative r (sign mix escalation)
        r = rng.choice([-2, -3, -4])
        t1 = rng.randint(2, 9)
        n = rng.randint(3, 6)
        answer = r ** (n - 1) * t1
        answer_text = str(answer)
        text = f"A geometric sequence has t1 = {t1} and common ratio r = {r}. Using t_n = r^(n-1) × t1, find t_{n}."

        _verify_hard_geometric_term(t1, r, n, answer)

        # Misconceptions (adapted for negative r):
        # 1. ignores-the-sign: uses abs(r)^(n-1)*t1 instead
        wrong6_candidate = abs(r) ** (n - 1) * t1
        # 2. off-by-one-exponent: uses r^n instead of r^(n-1)
        wrong7 = r**n * t1
        # 3. forgets-t1-factor: uses r^(n-1) only
        wrong8 = r ** (n - 1)

        candidates = [
            str(wrong6_candidate),
            str(wrong7),
            str(wrong8),
        ]

        return dict(
            text=text,
            latex=rf"t_{{{n}}} = ({r})^{{{n - 1}}} \times {t1}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"t_{n} = ({r})^{n - 1} × {t1} = {r ** (n - 1)} × {t1} = {answer}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif variant == "find_n":
        # Hard: find the POSITION n given t1, r, and the term's VALUE -- a third unknown, distinct
        # from "find_t_n" (finds t_n) and "find_r" (finds r). r is a fixed small positive integer
        # (never negative), so n is unique -- the even-root sign ambiguity "find_r" has cannot occur
        # here.
        #
        # The term's value is now stated WITHOUT its index; only t1 and r are given directly, and
        # the learner must search for which position produces that value.
        r = rng.randint(2, 5)
        t1 = rng.randint(2, 9)
        n = rng.randint(3, 6)
        t_n = r ** (n - 1) * t1
        answer = n
        answer_text = str(n)
        text = (
            f"The number {t_n} is a term of the geometric sequence with t1 = {t1} and "
            f"common ratio r = {r}. Using t_n = r^(n-1) × t1, which term number is it?"
        )

        _verify_find_n(t1, r, t_n, answer)

        # Misconceptions:
        # 1. reports-the-exponent-not-n: states n-1 (the exponent) instead of n
        wrong9 = n - 1
        # 2. off-by-one-the-other-way: n+1
        wrong10 = n + 1

        candidates = [
            str(wrong9),
            str(wrong10),
        ]

        return dict(
            text=text,
            latex=rf"{t_n} = {r}^{{n-1}} \times {t1}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"t_n / t1 = {t_n} / {t1} = {r ** (n - 1)} = {r}^{n - 1}, so n = {n}."],
            distractors=pick_distractors(answer_text, candidates),
        )

    raise ValueError(f"unknown variant: {variant!r}")
