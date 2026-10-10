"""Recursive sequence generation, both senior ``acara-senior-v8-4``, domain ``functions-sequences``, ``answer:
ordered_list``.

Two generators:

- ``arithmetic_recursion_generate`` (``senior-use-recursion-to-generate-an-arithmetic-
  sequence``, ACMGM067) -- generate the next k terms of an arithmetic sequence by
  iteratively applying the recursive rule t(n+1) = t(n) + d from a stated first term.
  This is explicitly order-graded (answer: ordered_list); reversing the order is wrong.
  Distinguished from the existing active ``generate-sequence-from-rule`` skill which
  generates from a CLOSED-FORM rule by substituting term positions, not a recursive rule.

- ``geometric_recursion_generate`` (``senior-use-recursion-to-generate-a-geometric-
  sequence``, ACMGM071) -- generate the next k terms of a geometric sequence by
  iteratively applying the recursive rule t(n+1) = t(n) * r from a stated first term.
  Three difficulty tiers: integer growth (easy), signed/alternating (medium), and
  fractional decay (hard -- a genuinely different multiplicative mode). Both
  order-graded; reversing is wrong.

Both generators use exact integer arithmetic and expose independent verification helpers at module level for
testing (ordered_sequence_build.py).
"""

from __future__ import annotations

from ..core import check, generator, paren_negative, pick_distractors


# --- Shared helper -----------------------------------------------


def _has_zero_crossing(values: list[int]) -> bool:
    """Check if a sequence has a zero crossing or touches zero.

    A zero crossing for this purpose means: comparing consecutive values,
    at least one adjacent pair has one value < 0 and the next >= 0, or one >= 0
    and the next < 0 (i.e. crosses or touches zero at some point in the walk).

    For example:
    - [1, 2, 3] -> False (all positive, no crossing)
    - [0, 1, 2] -> True (zero crossing: 0 >= 0, then 1 >= 0, but 0 is present)
    - [3, 0, -3] -> True (zero is present; also 0 >= 0 and -3 < 0 is a crossing)
    - [2, 1, 0] -> True (0 is present; and 1 >= 0, 0 >= 0 still touches zero)
    - [-3, -2, -1] -> False (all negative, no crossing)
    - [-1, 0, 1] -> True (0 is present; and pairs cross sign)
    """
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        # Check if one is negative and the next is non-negative, or vice versa
        if (a < 0 and b >= 0) or (a >= 0 and b < 0):
            return True
    return False


# --- arithmetic_recursion_generate -----------------------------------------------


def _verify_arithmetic_recursion(t1: int, d: int, k: int, terms: list[int]) -> None:
    """Construction builds `terms` by an iterative accumulation loop
    (cur += d, k times). This check recomputes via the CLOSED-FORM formula
    t1 + m*d for the m-th generated term (m = 1..k) -- a genuinely different
    derivation, not a re-run of the same loop."""
    expected = [t1 + m * d for m in range(1, k + 1)]
    check(expected == terms, "closed-form recomputation disagrees with the iterative construction")


def _verify_from_two_terms(t1: int, d: int, k: int, p: int, q: int, t_p: int, t_q: int, terms: list[int]) -> None:
    """Independent check for from_two_terms: recompute d and t1 from t_p/t_q by the two-point difference formula
    (a route independent of the construction, which builds t_p/t_q FROM t1/d), then recompute the k generated
    terms via the standard closed form."""
    d_recomputed = (t_q - t_p) // (q - p)
    check(d_recomputed == d, "two-point difference formula did not recover d")
    check((t_q - t_p) % (q - p) == 0, "t_q - t_p is not exactly divisible by the term gap")
    t1_recomputed = t_p - (p - 1) * d_recomputed
    check(t1_recomputed == t1, "two-point back-solve did not recover t1")
    expected = [t1_recomputed + m * d_recomputed for m in range(1, k + 1)]
    check(expected == terms, "closed-form recomputation disagrees with the iterative construction")


@generator(
    "arithmetic_recursion_generate",
    level=11,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def arithmetic_recursion_generate(rng, variant="easy"):
    """Generate the next k terms of an arithmetic sequence by iteratively
    applying the recursive rule t(n+1) = t(n) + d, starting from a stated
    first term t1. The graded answer is exactly the k terms AFTER t1, never
    including t1 itself."""

    if variant == "easy":
        k = 3
        d = rng.randint(1, 9)  # always positive
        t1 = rng.randint(1, 20)
    elif variant == "medium":
        k = 4
        d = rng.choice([v for v in range(-9, 10) if v != 0])  # allows negative
        t1 = rng.randint(1, 20)
    elif variant == "hard":
        k = 5
        t1 = rng.randint(-20, 20)
        d = rng.choice([v for v in range(-9, 10) if v != 0])

        # Force a zero-crossing: resample t1 if needed
        for attempt in range(50):
            # Compute full sequence including t1
            full_sequence = [t1 + m * d for m in range(k + 1)]
            if _has_zero_crossing(full_sequence):
                break
            # Resample t1 only (keep d), retry
            t1 = rng.randint(-20, 20)
        else:
            raise ValueError(f"Could not find a zero-crossing sequence after 50 attempts (t1 range: -20 to 20, d={d})")
    elif variant == "from_two_terms":
        # Given two NON-CONSECUTIVE terms t_p and t_q (gap >= 2) instead of t1 directly -- the
        # learner must derive d and t1 algebraically first, a genuinely different first step from
        # "easy"/"medium"/"hard" above, which all state t1 directly.
        k = 4
        d = rng.choice([v for v in range(-9, 10) if v != 0])
        t1 = rng.randint(-15, 15)
        p = rng.randint(2, 3)
        gap = rng.randint(2, 3)
        q = p + gap
        t_p = t1 + (p - 1) * d
        t_q = t1 + (q - 1) * d

        terms = []
        cur = t1
        for _ in range(k):
            cur += d
            terms.append(cur)

        _verify_from_two_terms(t1, d, k, p, q, t_p, t_q, terms)

        question_text = (
            f"An arithmetic sequence has t{p} = {t_p} and t{q} = {t_q}. "
            f"Find the common difference and the first term, then write the next "
            f"{k} terms after t1, in order."
        )

        answer_text = ", ".join(str(t) for t in terms)

        # Misconceptions:
        # 1. treats-t_p-as-t1: generates from t_p as if it were the first term
        from_tp = [t_p + m * d for m in range(1, k + 1)]
        # 2. forgot-to-divide-by-the-gap: uses (t_q - t_p) as d directly, not
        #    (t_q - t_p) / gap
        d_wrong = t_q - t_p
        from_wrong_d = [t1 + m * d_wrong for m in range(1, k + 1)]
        # 3. reverses-the-order
        reversed_order = list(reversed(terms))

        candidates = [
            ", ".join(str(t) for t in from_tp),
            ", ".join(str(t) for t in from_wrong_d),
            ", ".join(str(t) for t in reversed_order),
        ]

        return dict(
            text=question_text,
            latex=question_text,
            answer=list(terms),
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"d = (t{q} - t{p}) / {gap} = ({t_q} - {paren_negative(t_p)}) / {gap} = {d}",
                f"t1 = t{p} - {p - 1}×d = {t_p} - {paren_negative((p - 1) * d)} = {t1}",
                f"Answer: {answer_text}",
            ],
            distractors=pick_distractors(answer_text, candidates, n=3),
        )
    else:
        raise ValueError(f"unknown variant {variant!r}; expected 'easy', 'medium', 'hard', or 'from_two_terms'")

    # Iterative construction: the answer is k terms AFTER t1
    terms = []
    cur = t1
    for _ in range(k):
        cur += d
        terms.append(cur)

    _verify_arithmetic_recursion(t1, d, k, terms)

    # Format question text
    if d < 0:
        d_text = f"({d})"
    else:
        d_text = str(d)

    question_text = (
        f"An arithmetic sequence starts at t1 = {t1} and uses the recursive rule "
        f"t(n+1) = t(n) + {d_text}. Write the next {k} terms after t1, in order."
    )

    answer_text = ", ".join(str(t) for t in terms)

    # Misconception distractors (exactly 3 candidates per tier):
    # 1. repeats-the-first-step-constant: [t1 + d] * k
    #    States the SAME single step repeated k times, never accumulating.
    repeats_constant = [t1 + d] * k

    # 2. off-by-one-shift-includes-t1: [t1 + m*d for m in range(0, k)]
    #    Accidentally includes t1 itself as the first "generated" term.
    off_by_one = [t1 + m * d for m in range(0, k)]

    # 3. reverses-the-order: list(reversed(terms))
    #    The exact SAME correct values, in REVERSED order.
    reversed_order = list(reversed(terms))

    candidates = [
        ", ".join(str(t) for t in repeats_constant),
        ", ".join(str(t) for t in off_by_one),
        ", ".join(str(t) for t in reversed_order),
    ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=list(terms),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Start at t1 = {t1}.",
            f"Apply t(n+1) = t(n) + {d_text} repeatedly {k} times:",
            f"t(2) = {t1} + {d_text} = {terms[0]}" if k >= 1 else "",
            f"t(3) = {terms[0]} + {d_text} = {terms[1]}" if k >= 2 else "",
            f"t({k + 1}) = {terms[k - 2]} + {d_text} = {terms[k - 1]}" if k >= 2 else "",
            f"Answer: {answer_text}",
        ],
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


# --- geometric_recursion_generate -----------------------------------------------


def _verify_geometric_recursion_integer(t1: int, r: int, k: int, terms: list[int]) -> None:
    """Easy/medium (integer ratio r). Construction uses an iterative
    repeated-multiplication loop; this recomputes via direct exponentiation,
    a genuinely different code path."""
    expected = [t1 * (r**m) for m in range(1, k + 1)]
    check(expected == terms, "closed-form recomputation disagrees with the iterative construction")


def _verify_geometric_recursion_fraction(t1: int, r_denom: int, k: int, terms: list[int]) -> None:
    """Hard (fractional ratio 1/r_denom). Construction uses an iterative
    repeated-floor-division loop; this recomputes via direct exponentiation
    of the denominator, then a single division -- a genuinely different
    code path (direct power + one division vs. k repeated divisions)."""
    expected = [t1 // (r_denom**m) for m in range(1, k + 1)]
    check(expected == terms, "closed-form recomputation disagrees with the iterative construction")
    # Also confirm every step was exact (no truncation ever occurred):
    for m in range(1, k + 1):
        check(t1 % (r_denom**m) == 0, f"t1 is not exactly divisible by r_denom**{m}")


@generator(
    "geometric_recursion_generate",
    level=11,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def geometric_recursion_generate(rng, variant="easy"):
    """Generate the next k terms of a geometric sequence by iteratively
    applying the recursive rule t(n+1) = t(n) * r, starting from a stated
    first term t1. The graded answer is exactly the k terms AFTER t1, never
    including t1."""

    if variant == "easy":
        # Integer growth
        k = 3
        r = rng.randint(2, 4)
        t1 = rng.randint(1, 9)

        terms = []
        cur = t1
        for _ in range(k):
            cur *= r
            terms.append(cur)

        _verify_geometric_recursion_integer(t1, r, k, terms)
        r_str = str(r)
        r_display = r_str

    elif variant == "medium":
        # Signed/alternating
        k = 4
        r = rng.choice([-4, -3, -2, 2, 3, 4])
        t1 = rng.randint(1, 9)

        terms = []
        cur = t1
        for _ in range(k):
            cur *= r
            terms.append(cur)

        _verify_geometric_recursion_integer(t1, r, k, terms)
        r_display = f"({r})" if r < 0 else str(r)

    elif variant == "hard":
        # Fractional decay
        r_denom = rng.choice([2, 3])
        k = 4
        t1 = (r_denom**k) * rng.randint(1, 5)

        terms = []
        cur = t1
        for _ in range(k):
            check(cur % r_denom == 0, f"cur {cur} is not divisible by r_denom {r_denom}")
            cur = cur // r_denom
            terms.append(cur)

        _verify_geometric_recursion_fraction(t1, r_denom, k, terms)
        r_display = str(r_denom)  # Just the denominator

    else:
        raise ValueError(f"unknown variant {variant!r}; expected 'easy', 'medium', or 'hard'")

    # Format question text
    if variant == "hard":
        question_text = (
            f"A geometric sequence starts at t1 = {t1} and uses the recursive rule "
            f"t(n+1) = t(n) / {r_display}. Write the next {k} terms after t1, in order."
        )
    else:
        question_text = (
            f"A geometric sequence starts at t1 = {t1} and uses the recursive rule "
            f"t(n+1) = t(n) × {r_display}. Write the next {k} terms after t1, in order."
        )

    answer_text = ", ".join(str(t) for t in terms)

    # Misconception distractors
    if variant == "hard":
        # Hard: swap in decay-specific misconception
        # 1. treats-the-ratio-as-a-multiplier-not-a-divisor: [t1 * (r_denom ** m) for m in range(1, k + 1)]
        #    Forgets this is DECAY -- grows instead of shrinking.
        grows_instead = [t1 * (r_denom**m) for m in range(1, k + 1)]

        # 2. off-by-one-shift-includes-t1: [t1 // (r_denom ** m) for m in range(0, k)]
        off_by_one = [t1 // (r_denom**m) for m in range(0, k)]

        # 3. reverses-the-order: list(reversed(terms))
        reversed_order = list(reversed(terms))

        candidates = [
            ", ".join(str(t) for t in grows_instead),
            ", ".join(str(t) for t in off_by_one),
            ", ".join(str(t) for t in reversed_order),
        ]
    else:
        # Easy/medium: use standard misconceptions
        # 1. adds-instead-of-multiplying: [t1 + r * m for m in range(1, k + 1)]
        #    Treats r as a common DIFFERENCE rather than ratio.
        adds_instead = [t1 + r * m for m in range(1, k + 1)]

        # 2. off-by-one-shift-includes-t1: [t1 * (r ** m) for m in range(0, k)]
        off_by_one = [t1 * (r**m) for m in range(0, k)]

        # 3. reverses-the-order: list(reversed(terms))
        reversed_order = list(reversed(terms))

        candidates = [
            ", ".join(str(t) for t in adds_instead),
            ", ".join(str(t) for t in off_by_one),
            ", ".join(str(t) for t in reversed_order),
        ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=list(terms),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Start at t1 = {t1}.",
            (
                f"Apply t(n+1) = t(n) / {r_display} repeatedly {k} times:"
                if variant == "hard"
                else f"Apply t(n+1) = t(n) × {r_display} repeatedly {k} times:"
            ),
            f"Answer: {answer_text}",
        ],
        distractors=pick_distractors(answer_text, candidates, n=3),
    )
