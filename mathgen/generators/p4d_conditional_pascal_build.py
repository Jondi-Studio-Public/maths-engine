"""Conditional probability notation + Pascal's triangle.

- ``senior_use_the_notation_p_a_vert_b`` (``senior-use-the-notation-p-a-vert-b``,
  ACMMM057, domain statistics-probability, answer: fraction) -- use conditional
  probability notation P(A|B), given bounded fully specified inputs, and return the
  resulting probability value.

- ``senior_use_pascals_triangle_and_its_properties`` (``senior-use-pascal-s-triangle-and-its-properties``,
  ACMMM048, domain statistics-probability, answer: number) -- use Pascal's triangle
  and its properties, given bounded fully specified inputs, and return the resulting
  value.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, pick_distractors, rational_text


# ============================================================================
# Skill E: senior_use_the_notation_p_a_vert_b (conditional probability)
# ============================================================================


def _verb(n: int, singular: str, plural: str) -> str:
    """Subject-verb agreement for a count that can be exactly one."""
    return singular if n == 1 else plural


def _person_word(n: int) -> str:
    """Noun for a count of people, singular vs plural."""
    return "person" if n == 1 else "people"


def _verify_conditional_enumeration(N: int, n_b: int, n_ab: int, answer: Fraction) -> None:
    """Independent check via explicit enumeration instead of direct division.
    Build items, filter to B, count A intersect B, divide by |B|."""
    items = list(range(N))
    in_b = set(items[:n_b])
    in_a_and_b = set(items[:n_ab])
    filtered = [i for i in items if i in in_b]
    count_in_both = len([i for i in filtered if i in in_a_and_b])
    computed = Fraction(count_in_both, len(filtered))
    check(computed == answer, f"enumeration: N={N}, n_b={n_b}, n_ab={n_ab}, expected {answer}, got {computed}")


@generator(
    "conditional_probability_notation",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def conditional_probability_notation(rng, tier="easy"):
    """Use conditional probability notation P(A|B) and return the value."""
    # Sample base parameters
    N = rng.randint(20, 60)
    n_b = rng.randint(4, N // 2)

    # Guard 2: reject if N == 2 * n_b (conditions-on-complement collision)
    attempts_guard2 = 0
    while N == 2 * n_b:
        attempts_guard2 += 1
        if attempts_guard2 > 10:
            # Resample N/n_b pair
            N = rng.randint(20, 60)
            n_b = rng.randint(4, N // 2)
            attempts_guard2 = 0
        else:
            N = rng.randint(20, 60)
            n_b = rng.randint(4, N // 2)

    # Sample n_ab with guard 1: reject if n_b * n_b == n_ab * N
    n_ab = rng.randint(1, n_b - 1)
    attempts_guard1 = 0
    while n_b * n_b == n_ab * N:
        attempts_guard1 += 1
        if attempts_guard1 > 50:
            raise ValueError(f"Could not find valid n_ab for N={N}, n_b={n_b} after 50 attempts")
        n_ab = rng.randint(1, n_b - 1)

    answer = Fraction(n_ab, n_b)
    answer_text = rational_text(answer)
    _verify_conditional_enumeration(N, n_b, n_ab, answer)

    if tier == "easy":
        # State n_ab and n_b directly
        question_text = (
            f"In a group of {N} people, {n_b} like Sport. "
            f"Of those {n_b}, {n_ab} {_person_word(n_ab)} also {_verb(n_ab, 'likes', 'like')} Music. "
            f"If one of the Sport-likers is chosen at random, "
            f"what is the probability they also like Music?"
        )
    elif tier == "medium":
        # Present a table, learner must compute n_b
        n2 = n_b - n_ab
        question_text = (
            f"In a group of {N} people: {n_ab} {_person_word(n_ab)} {_verb(n_ab, 'likes', 'like')} both Sport and Music; "
            f"{n2} {_person_word(n2)} {_verb(n2, 'likes', 'like')} Sport but not Music; "
            f"{N - n_b} {_person_word(N - n_b)} {_verb(N - n_b, 'does', 'do')} not like Sport. "
            f"If one of the Sport-likers is chosen at random, "
            f"what is the probability they also like Music?"
        )
    else:  # hard
        # Present P(A ∩ B) and P(B) as fractions, learner divides fractions
        prob_ab = Fraction(n_ab, N)
        prob_b = Fraction(n_b, N)
        prob_ab_text = rational_text(prob_ab)
        prob_b_text = rational_text(prob_b)
        question_text = (
            f'Let A = "likes Music" and B = "likes Sport". '
            f"Given: P(A ∩ B) = {prob_ab_text} and P(B) = {prob_b_text}. "
            f"Using the formula P(A|B) = P(A ∩ B) / P(B), "
            f"what is P(A|B)?"
        )

    # Distractors
    # 1. reports-joint-not-conditional: Fraction(n_ab, N) -- safe unconditionally
    dist1_text = rational_text(Fraction(n_ab, N))

    # 2. reports-marginal-of-conditioning-event: Fraction(n_b, N) -- excluded by guard 1
    dist2_text = rational_text(Fraction(n_b, N))

    # 3. conditions-on-complement: Fraction(n_ab, N - n_b) -- excluded by guard 2
    dist3_text = rational_text(Fraction(n_ab, N - n_b))

    candidates = [dist1_text, dist2_text, dist3_text]
    distractors = pick_distractors(answer_text, candidates)

    return dict(
        text=question_text,
        latex=question_text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Group: {N} people. Sport: {n_b}. Both: {n_ab}.",
            f"P(Music|Sport) = (count in both) / (count in Sport) = {n_ab} / {n_b}.",
            f"Simplified: {answer_text}.",
        ],
        distractors=distractors,
    )


# ============================================================================
# Skill F: senior_use_pascals_triangle_and_its_properties (Pascal's triangle)
# ============================================================================


def _pascal_row(n: int) -> list[int]:
    """Build Pascal's triangle by additive recurrence, not closed form."""
    row = [1]
    for _ in range(n):
        row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]
    return row


def _verify_pascal_formula(n: int, k: int, expected: int) -> None:
    """Independent check: pascal_row recurrence must equal math.comb."""
    computed = _pascal_row(n)[k]
    check(computed == expected, f"pascal_row(n={n})[{k}] = {computed}, expected {expected}")
    check(computed == math.comb(n, k), f"pascal_row(n={n})[{k}] = {computed}, math.comb({n}, {k}) = {math.comb(n, k)}")


@generator(
    "pascals_triangle_properties",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def pascals_triangle_properties(rng, tier="easy"):
    """Use Pascal's triangle and its properties to find a value."""

    if tier == "easy":
        # Direct lookup: sample n from [4, 6, 7] (exclude 5), k from [2, n-2]
        # with symmetric-tie exclusion for odd n
        n = rng.choice([4, 6, 7])
        k = rng.randint(2, n - 2)

        # Symmetric-tie exclusion for odd n
        if n % 2 == 1 and k == (n - 1) // 2:
            if k + 1 <= n - 2:
                k = k + 1
            else:
                k = k - 1

        answer = math.comb(n, k)
        _verify_pascal_formula(n, k, answer)

        question_text = f"In Pascal's triangle, what is the value at position {k} in row {n}?"

        # Distractors
        # 1. off-by-one-k
        if k + 1 <= n - 2:
            dist1 = math.comb(n, k + 1)
        else:
            dist1 = math.comb(n, k - 1)

        # 2. permutation-not-combination
        dist2 = math.perm(n, k)

        candidates = [str(dist1), str(dist2)]
        distractors = pick_distractors(str(answer), candidates)

    elif tier == "medium":
        # Forward addition: state parent1 and parent2, learner adds
        n = rng.randint(4, 8)
        k = rng.randint(1, n - 1)

        parent1 = math.comb(n - 1, k - 1)
        parent2 = math.comb(n - 1, k)
        answer = parent1 + parent2

        check(answer == math.comb(n, k), f"parent1={parent1} + parent2={parent2} must equal comb({n}, {k})")
        _verify_pascal_formula(n, k, answer)

        question_text = (
            f"In Pascal's triangle, row {n - 1} has values {parent1} and {parent2} "
            f"at positions {k - 1} and {k}. "
            f"Using Pascal's addition rule, what is the value at position {k} in row {n}?"
        )

        # Distractors
        # 1. subtracts-instead-of-adds
        dist1 = abs(parent1 - parent2)

        # 2. reports-one-parent-only
        dist2 = parent2

        candidates = [str(dist1), str(dist2)]
        distractors = pick_distractors(str(answer), candidates)

    else:  # hard
        # Inverse addition: given child and one parent, find the other parent
        n = rng.randint(4, 8)
        k = rng.randint(1, n - 1)

        # Guard: exclude n == 2 * k
        while n == 2 * k:
            k = k + 1 if k + 1 <= n - 1 else k - 1

        child = math.comb(n, k)
        parent_given = math.comb(n - 1, k)
        missing = child - parent_given

        check(missing == math.comb(n - 1, k - 1), f"missing must equal comb({n - 1}, {k - 1})")
        _verify_pascal_formula(n - 1, k - 1, missing)

        question_text = (
            f"In Pascal's triangle, row {n} has value {child} at position {k}. "
            f"Row {n - 1} has value {parent_given} at position {k}. "
            f"Using Pascal's addition rule, what is the value at position {k - 1} in row {n - 1}?"
        )

        answer = missing

        # Distractors
        # 1. adds-instead-of-subtracts
        dist1 = child + parent_given

        # 2. reports-given-parent-as-the-answer (excluded by n != 2*k guard)
        dist2 = parent_given

        candidates = [str(dist1), str(dist2)]
        distractors = pick_distractors(str(answer), candidates)

    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"Using Pascal's triangle: row {n}, position {k}.",
            f"Answer: {answer}.",
        ],
        distractors=distractors,
    )
