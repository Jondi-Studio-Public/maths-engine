"""- ``square_roots_perfect_squares`` (AC9M7N01) -- given a stated natural number,
  determine whether it is a perfect square and, if it is, state its square root.
  The question is binary classification: either the number is a perfect square and
  the answer is its root as a string (e.g., "6" for 36), or it is not and the
  answer is the literal string "not a perfect square".

- ``exponent_laws_positive_zero`` (AC9M8N02) -- apply exponent laws for positive
  integer and zero exponents to a numerical expression to produce an equivalent
  simplified form. The expression combines two exponent-law applications (product
  rule, quotient rule, or both) on a small positive-integer base. The canonical
  answer is always a sympy Pow object rendered via str(), never a hand-typed "^"
  (sympify parses "^" as bitwise XOR, not exponentiation).
"""

from __future__ import annotations

import math
from sympy import sympify, Pow, symbols, Integer
from ..core import check, generator, latex_answer, pick_distractors

# -- square_roots_perfect_squares -----------------------------------------------


def _get_nearby_perfect_square_roots(n: int) -> list[int]:
    """Get the roots of the two perfect squares nearest to n.

    Returns a list of roots (not the squares themselves), ordered by distance
    from n. Always returns exactly 2 values (lower and upper neighbor roots),
    even if one edge case occurs.
    """
    sqrt_n = math.isqrt(n)

    # The two perfect squares nearest to n are sqrt_n**2 and (sqrt_n+1)**2
    # (or sqrt_n**2 and (sqrt_n-1)**2 if n itself is already a perfect square)
    lower_root = max(1, sqrt_n - 1) if n == sqrt_n * sqrt_n else sqrt_n
    upper_root = sqrt_n + 1 if n == sqrt_n * sqrt_n else sqrt_n + 1

    # Ensure lower < upper
    if lower_root >= upper_root:
        lower_root = upper_root - 1

    return [lower_root, upper_root]


def _verify_perfect_square(n: int, predicted: str) -> None:
    """Independent check: verify n's perfect-square status and root (if applicable).

    Construction picks predicted first, then builds n to match. This check uses
    a different route: math.isqrt(n)**2 == n to verify perfect-square status,
    and math.isqrt(n) to get the root independently.
    """
    sqrt_n = math.isqrt(n)
    is_perfect_square = sqrt_n * sqrt_n == n

    if predicted == "not a perfect square":
        check(
            not is_perfect_square,
            f"n={n} predicted as not perfect square, but {sqrt_n}**2 = {sqrt_n * sqrt_n}",
        )
    else:
        # predicted should be the string of the root
        check(
            is_perfect_square,
            f"n={n} predicted as perfect square, but {sqrt_n}**2 = {sqrt_n * sqrt_n} != {n}",
        )
        predicted_root = int(predicted)
        check(
            predicted_root == sqrt_n,
            f"n={n}, predicted root={predicted_root}, but math.isqrt({n})={sqrt_n}",
        )


def _sample_perfect_square_case(rng, predicted: str, root_range: tuple[int, int]) -> int:
    """Sample a number n matching the predicted perfect-square status.

    For predicted = "yes" (a root string): construct n = k**2 for random k.
    For predicted = "not a perfect square": construct n as k**2 +/- offset.
    """
    root_lo, root_hi = root_range

    if predicted == "not a perfect square":
        # Build n as k**2 +/- small offset, confirmed not itself a perfect square
        attempts = 0
        while attempts < 100:
            k = rng.randint(root_lo, root_hi)
            base = k * k
            # Choose offset direction; only go negative if base > 1
            if base > 1:
                offset_direction = rng.choice([-1, 1])
            else:
                offset_direction = 1
            offset_magnitude = rng.randint(1, min(3, k if k > 3 else 2))
            offset = offset_direction * offset_magnitude
            n = base + offset

            # Verify n is positive and not a perfect square
            if n > 0 and math.isqrt(n) ** 2 != n:
                return n
            attempts += 1

        raise ValueError(f"Cannot construct non-perfect-square in root range {root_range}")
    else:
        # predicted is the string of a root
        k = rng.randint(root_lo, root_hi)
        n = k * k
        return n


_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _square_root_power_form(rng):
    """Number stated as a product of prime powers; decide if it is a square."""
    primes = [2, 3, 5, 7]
    perfect = rng.random() < 0.5
    while True:
        chosen = sorted(rng.sample(primes, rng.randint(2, 3)))
        if perfect:
            exps = [2 * rng.randint(1, 3) for _ in chosen]
        else:
            exps = [rng.randint(1, 6) for _ in chosen]
            # near-miss: exactly one odd exponent
            idx = rng.randrange(len(chosen))
            exps = [e if e % 2 == 0 else e + 1 for e in exps]
            exps[idx] += 1 if exps[idx] < 6 else -1
        n = math.prod(p**e for p, e in zip(chosen, exps))
        if n <= 2_000_000:
            break
    odd = [e % 2 for e in exps]
    check(perfect == (sum(odd) == 0))
    root = math.isqrt(n)
    check((root * root == n) == perfect, "square_roots power_form: parity disagrees with isqrt")
    shown = " × ".join(f"{p}{str(e).translate(_SUP)}" for p, e in zip(chosen, exps))
    text = f"Is {shown} a perfect square? If yes, what is its square root?"
    if perfect:
        answer = str(root)
        p0 = chosen[0]
        candidates = [str(root * p0), str(root // p0), str(n // 2), "not a perfect square"]
    else:
        answer = "not a perfect square"
        half_down = math.prod(p ** (e // 2) for p, e in zip(chosen, exps))
        half_up = math.prod(p ** ((e + 1) // 2) for p, e in zip(chosen, exps))
        candidates = [str(half_down), str(half_up), str(root), str(n // 2)]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer),
        working=[],
        distractors=pick_distractors(answer, candidates, n=3),
    )


@generator(
    "square_roots_perfect_squares",
    level=4,
    presets={
        "easy": dict(root_range=(1, 12), force_non_perfect=False),
        "medium": dict(root_range=(1, 15), force_non_perfect=None),  # 50/50 mix
        "hard": dict(root_range=(1, 20), force_non_perfect=None),  # near-misses for non-perfect, 2/3 non-perfect
    },
)
def square_roots_perfect_squares(rng, root_range=(1, 12), force_non_perfect=None, power_form=False):
    """Given a natural number n, determine whether it is a perfect square and,
    if it is, state its square root. Answer is either the string of the root
    (e.g., "6" for n=36) or the literal string "not a perfect square".

    ``power_form=True`` states the number as a product of prime powers
    (e.g. 2⁴ × 3³) so the learner must judge perfection from the parity of the
    exponents. The default path is unchanged.
    """
    if power_form:
        return _square_root_power_form(rng)
    # Choose predicted answer first
    if force_non_perfect is None:
        # For hard: 2/3 non-perfect, 1/3 perfect (to get more near-miss traps)
        if root_range[1] > 15:  # hard tier
            is_non_perfect = rng.random() < 2 / 3
        else:
            # medium: 50/50 mix
            is_non_perfect = rng.random() < 0.5
    else:
        is_non_perfect = force_non_perfect

    if is_non_perfect:
        # For hard: force near-miss placement (n = k**2 +/- 1) on ~80% of non-perfect seeds
        # For medium/easy: random non-perfect-square construction
        if root_range[1] > 15:  # hard tier
            force_near_miss = rng.random() < 0.8  # 80% near-miss, 20% random non-perfect
            if force_near_miss:
                k = rng.randint(root_range[0], root_range[1])
                base = k * k
                offset = rng.choice([-1, 1])
                n = base + offset
                # Ensure n > 0 and not a perfect square
                attempts = 0
                while (n <= 0 or math.isqrt(n) ** 2 == n) and attempts < 10:
                    k = rng.randint(root_range[0], root_range[1])
                    base = k * k
                    offset = rng.choice([-1, 1])
                    n = base + offset
                    attempts += 1
            else:
                # 20% of hard tier non-perfect seeds are random non-perfect
                n = _sample_perfect_square_case(rng, "not a perfect square", root_range)
        else:
            # medium/easy: random non-perfect-square construction
            n = _sample_perfect_square_case(rng, "not a perfect square", root_range)

        predicted = "not a perfect square"
    else:
        # Perfect square case
        k = rng.randint(root_range[0], root_range[1])
        n = k * k
        predicted = str(k)

    _verify_perfect_square(n, predicted)

    # Build question text
    question_text = f"Is {n} a perfect square? If yes, what is its square root?"

    # Build distractors
    nearby_roots = _get_nearby_perfect_square_roots(n)
    nearby_root_strs = [str(r) for r in nearby_roots]

    if predicted == "not a perfect square":
        # Distractors are the nearby roots
        distractor_candidates = nearby_root_strs
    else:
        # Distractors are the nearby roots plus "not a perfect square"
        distractor_candidates = nearby_root_strs + ["not a perfect square"]

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=predicted,
        answer_text=predicted,
        answer_latex=latex_answer(predicted),
        working=[],
        distractors=pick_distractors(predicted, distractor_candidates, n=3),
    )


# -- exponent_laws_positive_zero -----------------------------------------------


def _verify_exponent_law_expression(original_value: int, answer_expr, answer_value: int) -> None:
    """Independent check: verify the simplified expression equals the original numerically.

    Construction uses exponent-law addition/subtraction to derive the final exponent.
    This check independently evaluates the original, unsimplified expression and
    compares it to the simplified form, using a different route (direct numeric
    evaluation) from the law-based derivation.
    """
    check(
        original_value == answer_value,
        f"original numeric value {original_value} != simplified answer value {answer_value}",
    )


def _build_easy_expression(rng, base: int) -> tuple[int, int]:
    """Build a product-rule-only expression.

    Returns (original_value, final_exponent) where the expression is base**m * base**n
    and final_exponent = m + n.
    """
    m = rng.randint(1, 5)
    n = rng.randint(1, 5)
    original_value = (base**m) * (base**n)
    final_exponent = m + n
    return original_value, final_exponent, (base, m, n, None, None, "product")


def _build_medium_expression(rng, base: int) -> tuple[int, int]:
    """Build a quotient-rule-only expression.

    Returns (original_value, final_exponent) where the expression is base**m / base**n,
    m >= n, and final_exponent = m - n.
    """
    m = rng.randint(1, 8)
    n = rng.randint(0, m)  # Includes zero-exponent case (n=0) and equal-exponent case (n=m)
    original_value = (base**m) // (base**n) if n > 0 else base**m
    final_exponent = m - n
    return original_value, final_exponent, (base, m, n, None, None, "quotient")


def _build_hard_expression(rng, base: int) -> tuple[int, int]:
    """Build a chained expression combining product and quotient.

    Example: (base**m * base**n) / base**p
    Returns (original_value, final_exponent) where final_exponent >= 0 and p > 0.

    Ensures the expression is always chained (both product AND quotient).
    """
    # Build the chain: (base**m * base**n) / base**p
    # After simplification: base**(m + n - p) = base**final_exponent
    # Ensure p > 0 (always divide) and final_exponent >= 0

    m = rng.randint(1, 8)
    n = rng.randint(1, 8)
    temp_sum = m + n

    # Choose p such that 0 < p < temp_sum (to get m + n - p > 0)
    # and ensure final_exponent >= 0
    max_p = temp_sum - 1  # Ensure final_exponent >= 1
    min_p = 1

    if min_p > max_p:
        # If m+n=1, we can't have 0 < p < 1, so we need to regenerate
        m = rng.randint(2, 8)
        n = rng.randint(2, 8)
        temp_sum = m + n
        max_p = temp_sum - 1

    p = rng.randint(min_p, max_p)
    final_exponent = temp_sum - p

    original_value = ((base**m) * (base**n)) // (base**p)
    return original_value, final_exponent, (base, m, n, p, None, "chain")


@generator(
    "exponent_laws_positive_zero",
    level=5,
    presets={
        "easy": dict(base_range=(2, 5), kind="product"),
        "medium": dict(base_range=(2, 5), kind="quotient"),
        "hard": dict(base_range=(2, 5), kind="chain"),
    },
)
def exponent_laws_positive_zero(rng, base_range=(2, 5), kind="product"):
    """Apply exponent laws for positive integer and zero exponents to a
    numerical expression to produce an equivalent simplified form.

    The expression combines two exponent-law applications. The canonical answer
    is a sympy Pow object rendered via str(), ensuring sympify parses it correctly
    as exponentiation (** not ^).
    """
    base = rng.randint(*base_range)

    if kind == "product":
        original_value, final_exponent, (base, m, n, p, _, rule_type) = _build_easy_expression(rng, base)
        question_text = f"Simplify: {base}**{m} × {base}**{n}"
        original_expr_for_check = f"{base}**{m} × {base}**{n}"
    elif kind == "quotient":
        original_value, final_exponent, (base, m, n, p, _, rule_type) = _build_medium_expression(rng, base)
        question_text = f"Simplify: {base}**{m} ÷ {base}**{n}"
        original_expr_for_check = f"{base}**{m} ÷ {base}**{n}"
    elif kind == "chain":
        original_value, final_exponent, (base, m, n, p, _, rule_type) = _build_hard_expression(rng, base)
        if p > 0:
            question_text = f"Simplify: ({base}**{m} × {base}**{n}) ÷ {base}**{p}"
            original_expr_for_check = f"({base}**{m} × {base}**{n}) ÷ {base}**{p}"
        else:
            question_text = f"Simplify: {base}**{m} × {base}**{n}"
            original_expr_for_check = f"{base}**{m} × {base}**{n}"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    # Build the answer as a sympy Pow object (to ensure str() renders as **base**exponent, not numeric)
    answer_expr = Pow(Integer(base), Integer(final_exponent), evaluate=False)
    answer_value = int(base**final_exponent)  # Numeric value for independent check

    # Verify the independent check
    _verify_exponent_law_expression(original_value, answer_expr, answer_value)

    # Build answer string using sympy Pow rendering (uses ** not ^)
    answer_str = str(answer_expr)  # This will be "base**exponent"

    # Build misconception distractors
    distractors_list = []

    # Misconception 1: adds-instead-of-multiplying-bases (if applicable)
    if kind == "product":
        wrong_bases_expr = (2 * base) ** final_exponent
        distractors_list.append(str(wrong_bases_expr))

    # Misconception 2: wrong-law (quotient instead of product, or vice versa)
    if kind == "product":
        # Apply quotient rule instead
        wrong_exponent = m - n
        if wrong_exponent >= 0:
            wrong_law_expr = base**wrong_exponent
            distractors_list.append(str(wrong_law_expr))
    elif kind == "quotient":
        # Apply product rule instead
        wrong_exponent = m + n
        wrong_law_expr = base**wrong_exponent
        distractors_list.append(str(wrong_law_expr))
    elif kind == "chain":
        # Apply only product rule (m + n)
        wrong_exponent = m + n
        wrong_law_expr = base**wrong_exponent
        distractors_list.append(str(wrong_law_expr))

    # Misconception 3: zero-exponent-as-zero
    if final_exponent == 0 or (kind == "quotient" and n == 0):
        zero_expr = 0
        distractors_list.append(str(zero_expr))
    elif kind == "quotient" or kind == "chain":
        # Also add base**0 = 0 misconception as a possibility
        if rng.random() < 0.5:
            distractors_list.append("0")

    # Ensure all distractors are numeric and different from the answer
    valid_distractors = []
    for dist_str in distractors_list:
        try:
            dist_value = int(dist_str)
            if dist_value != answer_value and str(dist_value) not in valid_distractors:
                valid_distractors.append(str(dist_value))
        except (ValueError, TypeError):
            pass

    # Pick up to 3 distractors
    final_distractors = valid_distractors[:3]

    # Ensure we have exactly 3 distractors; if not, add random wrong exponents
    while len(final_distractors) < 3:
        random_exp = rng.randint(0, final_exponent + 5)
        if random_exp != final_exponent:
            random_dist = base**random_exp
            random_dist_str = str(random_dist)
            if random_dist_str not in final_distractors:
                final_distractors.append(random_dist_str)

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer_str,
        answer_text=answer_str,
        answer_latex=answer_str,
        working=[],
        distractors=pick_distractors(answer_str, final_distractors, n=3),
    )
