"""Generates `probability_complements_compounds` and `random_equal_likelihood_outcomes` questions on
complementary events and equally likely outcomes.

Two skills:

1. f10-probability-complements-compounds (AC9M8P01, AC9M8P02): Given the
   probability of one event in a complementary pair (as a reduced or unreduced
   fraction), compute the probability of the complementary event as 1 - p.

   - Easy: p is reduced, question explicitly asks "what is probability does not happen"
   - Medium: p is unreduced, learner must recognize and express complement in lowest terms
   - Hard: two-outcome scenario with no word "not", learner must recognize complementary event

   Distractors (all genuine probabilities in [0, 1]): forgot-to-complement (p), misreads-numerator
   (1-Fraction(n±1,d), either direction), then slips of two or the ends of the scale as fallbacks.

2. f10-random-equal-likelihood-outcomes (AC9M5P01): Given 4 spinners with
   explicit sector lists, classify which has equally likely outcomes.

   - Easy: 2 distinct labels, small equal counts (4-6 total)
   - Medium: 3 distinct labels, one option close-but-unequal (e.g. 3/3/2 of 8)
   - Hard: 4 distinct labels, larger counts (8-12), superficially balanced
"""

from __future__ import annotations

import re
from fractions import Fraction

from ..core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors


# ============================================================================
# Skill 1: probability_complements_compounds
# ============================================================================


def _format_fraction(f: Fraction) -> str:
    """Format a Fraction as 'n/d', always in lowest terms. Always shows n/d form for consistency."""
    f = Fraction(f).limit_denominator()
    return f"{f.numerator}/{f.denominator}"


# Easy and medium tiers use events with hand-written grammatical negations
_EASY_MEDIUM_EVENTS = [
    ("a bus arrives on time", "a bus does not arrive on time"),
    ("a customer chooses the vegetarian option", "a customer does not choose the vegetarian option"),
    ("a plant seed germinates", "a plant seed does not germinate"),
    ("a package is delivered within 2 days", "a package is not delivered within 2 days"),
]


@generator(
    "probability_complements_compounds",
    level=8,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def probability_complements_compounds(rng, tier="easy"):
    """Calculate the probability of a complementary event.

    Tiers:
    - easy: p is reduced, explicit "does not happen" wording
    - medium: p is unreduced display, implicit reduction in answer
    - hard: two-outcome scenario, no word "not", must recognize complementary event

    Exclusion: p != 1/2 on all tiers (required for forgot-to-complement distractor uniqueness).
    """
    # Valid (n, d) pairs for reduced fractions with d in {2,3,4,5,6,8,10}
    # Exclude 2*n == d (i.e., p != 1/2)
    # Only include pairs where gcd(n, d) == 1 (reduced form)
    from math import gcd

    valid_pairs = []
    for d in [2, 3, 4, 5, 6, 8, 10]:
        for n in range(1, d):
            if 2 * n != d and gcd(n, d) == 1:  # exclude p == 1/2 and unreduced
                valid_pairs.append((n, d))

    # Draw a base (n, d) pair
    n0, d0 = rng.choice(valid_pairs)
    p_reduced = Fraction(n0, d0)

    # Compute complement (always reduces automatically)
    c = 1 - p_reduced

    if tier == "easy":
        # Easy and medium use everyday events with no fixed probability expectation
        event_name, event_negation = rng.choice(_EASY_MEDIUM_EVENTS)

        # p displayed as reduced, explicit complement wording
        p_display = _format_fraction(p_reduced)
        question_text = (
            f"The probability that {event_name} is {p_display}. What is the probability that {event_negation}?"
        )
        answer_text = _format_fraction(c)

    elif tier == "medium":
        # p displayed as unreduced
        event_name, event_negation = rng.choice(_EASY_MEDIUM_EVENTS)

        k = rng.choice([2, 3])
        n = n0 * k
        d = d0 * k
        p_display = f"{n}/{d}"
        question_text = (
            f"The probability that {event_name} is {p_display}. What is the probability that {event_negation}?"
        )
        answer_text = _format_fraction(c)  # Always reduced

    else:  # hard
        # Two-outcome scenario, no word "not"
        # Only use outcomes with inherently variable probability (no fair coin/standard die/standard deck)
        outcome_pairs = [
            ("winning", "losing", "match"),
            ("passing", "failing", "test"),
            ("rain", "no rain", "day"),
            ("success", "failure", "task"),
            ("approval", "rejection", "proposal"),
        ]
        outcome_1, outcome_2, scenario = rng.choice(outcome_pairs)

        p_display = _format_fraction(p_reduced)
        question_text = (
            f"A {scenario} can only end in exactly one of two ways: {outcome_1} or {outcome_2}. "
            f"The probability of {outcome_1} is {p_display}. What is the probability of {outcome_2}?"
        )
        answer_text = _format_fraction(c)

        # Verify no word "not" appears in the question
        check(
            re.search(r"\bnot\b", question_text) is None,
            f"hard tier question should not contain 'not': {question_text}",
        )

    # Build distractors
    # 1. forgot-to-complement: value p (distinct from c iff p != 1/2, which we exclude)
    check_distinct(c, p_reduced, msg="forgot-to-complement must differ from answer")

    # 2. misreads-numerator: shift n by ±1
    if n0 + 1 <= d0 - 1:
        shifted_n = n0 + 1
    else:
        shifted_n = n0 - 1
    misreads_value = 1 - Fraction(shifted_n, d0)
    check_distinct(c, misreads_value, msg="misreads-numerator must differ from answer")
    # Verify the difference is exactly 1/d
    check(
        abs(c - misreads_value) == Fraction(1, d0),
        f"misreads-numerator should differ by 1/{d0}: c={c}, misreads={misreads_value}",
    )

    # 3. Nudge the numerator the other way (a slip of one in the opposite direction). Stays a
    # probability; the "adds instead" (1 + p) and reciprocal (d/n) slips exceed 1, which a learner
    # rejects at sight, so they are not offered.
    other_shifted_n = n0 - 1 if shifted_n == n0 + 1 else n0 + 1
    other_value = 1 - Fraction(other_shifted_n, d0)
    check_distinct(c, other_value, msg="opposite numerator slip must differ from answer")

    # Fallbacks, all genuine in-range wrong values: a slip of two, or the other end of the scale.
    fallbacks = [c + Fraction(2, d0), c - Fraction(2, d0), Fraction(1, d0), Fraction(d0 - 1, d0)]
    candidates = [p_reduced, misreads_value, other_value, *fallbacks]
    candidates = [_format_fraction(v) for v in candidates if 0 <= v <= 1 and v != c]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "need three in-range distractors")
    check(all(0 <= Fraction(x) <= 1 for x in distractors), "every distractor must be a probability")

    # Independent check: sum to 1 (different route from 1 - p computation)
    check_equivalent(c + p_reduced, Fraction(1, 1), msg="complement should sum to 1 with the original event")

    comp_den = d0 * k if tier == "medium" else d0
    comp_num = comp_den - (n0 * k if tier == "medium" else n0)
    working = [
        "The two events cannot both happen and one of them must happen, so their probabilities add to 1.",
        f"1 − {p_display} = {comp_den}/{comp_den} − {p_display} = {comp_num}/{comp_den}"
        + (f", which simplifies to {answer_text}." if f"{comp_num}/{comp_den}" != answer_text else "."),
    ]

    return dict(
        text=question_text,
        working=working,
        latex=f"plain:{question_text}",
        answer={"p": int(c.numerator), "q": int(c.denominator)},
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
    )


# ============================================================================
# Skill 2: random_equal_likelihood_outcomes
# ============================================================================

_COLORS = ["red", "blue", "green", "yellow", "orange", "purple"]
_FRUITS = ["apple", "banana", "cherry", "date"]
_SHAPES = ["circle", "square", "triangle", "star"]


def _build_spinner(rng, k: int, n_total: int, force_balanced: bool = False) -> dict:
    """Build a spinner with k distinct labels and n_total sectors.

    Args:
        rng: random generator
        k: number of distinct labels
        n_total: total number of sectors
        force_balanced: if True, create equal-count spinner; else random counts

    Returns:
        dict with 'labels' (the full repeated list), 'counts' (counts per label),
        and 'display_text' (formatted for the question)
    """
    # Sample k distinct labels without replacement
    shuffled = list(_COLORS)
    rng.shuffle(shuffled)
    labels_pool = shuffled[:k]

    if force_balanced:
        # Every label appears exactly n_total // k times
        if n_total % k != 0:
            # Adjust n_total to be divisible by k
            n_total = (n_total // k) * k
        m = n_total // k
        counts = [m] * k
    else:
        # Random distribution summing to n_total, not all equal
        # Use a retry loop to ensure at least two parts differ
        while True:
            counts = []
            remaining = n_total
            for i in range(k - 1):
                c = rng.randint(1, remaining - (k - i - 1))  # Leave room for others
                counts.append(c)
                remaining -= c
            counts.append(remaining)
            if len(set(counts)) > 1:  # At least two different counts
                break

    # Build the full label list by repeating each label its count of times
    full_labels = []
    for label, count in zip(labels_pool, counts):
        full_labels.extend([label] * count)

    # Shuffle the label list
    rng.shuffle(full_labels)

    # Format for display
    display_text = ", ".join(full_labels)

    return {
        "labels": full_labels,
        "counts": counts,
        "display_text": display_text,
    }


def _is_balanced(label_list: list[str]) -> bool:
    """Check if all distinct labels appear the same number of times."""
    from collections import Counter

    counter = Counter(label_list)
    return len(set(counter.values())) == 1


@generator(
    "random_equal_likelihood_outcomes",
    level=5,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def random_equal_likelihood_outcomes(rng, tier="easy"):
    """Classify which spinner has equally likely outcomes.

    Tiers:
    - easy: 2 labels, 4-6 total sectors, one balanced option, three obviously unbalanced
    - medium: 3 labels, one close-but-unequal option (e.g. 3/3/2), must count
    - hard: 4 labels, 8-12 total, all superficially look balanced, only one is equal
    """
    spinners = []
    answer_index = None

    if tier == "easy":
        # 2 labels, 4-6 sectors per option
        k = 2
        n = rng.randint(4, 6)

        # Balanced option
        balanced = _build_spinner(rng, k, n, force_balanced=True)
        spinners.append(balanced)
        answer_index = 0

        # Three unbalanced options
        for _ in range(3):
            unbalanced = _build_spinner(rng, k, n, force_balanced=False)
            spinners.append(unbalanced)

    elif tier == "medium":
        # 3 labels, but force one close-but-unequal count
        k = 3
        # Use n that's NOT divisible by k so we can have exactly max-min=1
        n_base = rng.randint(2, 3)  # 2 or 3
        n_balanced_base = n_base * k  # e.g., 6 or 9 (divisible by k)
        # For unbalanced spinners, use n_balanced_base + 1 so we can have close-but-unequal
        n_unbalanced = n_balanced_base + 1  # e.g., 7 or 10 (NOT divisible by k)

        # Balanced option (all labels have equal count)
        balanced = _build_spinner(rng, k, n_balanced_base, force_balanced=True)
        spinners.append(balanced)
        answer_index = 0

        # Three unbalanced options
        # One must have close-but-unequal counts (max - min == 1)
        for i in range(3):
            if i == 0:
                # Force close-but-unequal with max - min == 1
                # Using n_unbalanced which is NOT divisible by k
                base = n_unbalanced // k
                remainder = n_unbalanced % k  # This is >= 1
                # Create counts with remainder labels getting base+1, rest getting base
                counts = [base + 1] * remainder + [base] * (k - remainder)
                # This guarantees: sum = (base+1)*remainder + base*(k-remainder) = base*k + remainder = n_unbalanced
                # And: max = base+1, min = base, so max - min = 1
                # Shuffle counts
                rng.shuffle(counts)

                # Build spinner from counts
                shuffled = list(_COLORS)
                rng.shuffle(shuffled)
                labels_pool = shuffled[:k]
                full_labels = []
                for label, count in zip(labels_pool, counts):
                    full_labels.extend([label] * count)
                rng.shuffle(full_labels)

                spinner = {
                    "labels": full_labels,
                    "counts": counts,
                    "display_text": ", ".join(full_labels),
                }
                spinners.append(spinner)

                # Verify this is actually unequal
                check(not _is_balanced(spinner["labels"]), f"medium tier forced-unequal option should be unbalanced")
                # Verify max - min == 1
                check(max(counts) - min(counts) == 1, f"close-but-unequal counts should differ by 1: {counts}")
            else:
                # Random unbalanced
                unbalanced = _build_spinner(rng, k, n_unbalanced, force_balanced=False)
                spinners.append(unbalanced)

    else:  # hard
        # 4 labels, 8-12 sectors
        k = 4
        n = rng.randint(8, 12)

        # Balanced option
        if n % 4 == 0:
            balanced = _build_spinner(rng, k, n, force_balanced=True)
        else:
            # Adjust to nearest multiple of 4
            n = (n // 4) * 4
            if n == 0:
                n = 4
            balanced = _build_spinner(rng, k, n, force_balanced=True)
        spinners.append(balanced)
        answer_index = 0

        # Three unbalanced options
        for _ in range(3):
            unbalanced = _build_spinner(rng, k, n, force_balanced=False)
            spinners.append(unbalanced)

    # Verify exactly one spinner is balanced
    balanced_count = sum(1 for s in spinners if _is_balanced(s["labels"]))
    check(balanced_count == 1, f"exactly one spinner should be balanced, got {balanced_count}")
    check(_is_balanced(spinners[answer_index]["labels"]), f"answer spinner at index {answer_index} should be balanced")

    # Shuffle spinners and track answer position
    shuffled_indices = list(range(len(spinners)))
    rng.shuffle(shuffled_indices)
    shuffled_spinners = [spinners[i] for i in shuffled_indices]
    answer_index = shuffled_indices.index(answer_index)

    # Format options
    options = [f"Spinner {chr(65 + i)}: {s['display_text']}" for i, s in enumerate(shuffled_spinners)]
    answer_text = options[answer_index]
    distractors = [options[i] for i in range(len(options)) if i != answer_index]

    # Question text with options formatted inline (no newlines for text_quality check)
    options_text = " | ".join(options)
    question_text = f"Which of these four spinners has equally likely outcomes? {options_text}"

    # Independent check: re-derive balance from rendered labels using Counter
    from collections import Counter

    for i, option in enumerate(options):
        # Parse the label list from the option text (after "Spinner X: ")
        match = re.search(r"Spinner \w: (.+)$", option, re.MULTILINE)
        check(match, f"could not parse option: {option}")
        labels_text = match.group(1)
        labels = [label.strip() for label in labels_text.split(",")]

        counter = Counter(labels)
        is_balanced = len(set(counter.values())) == 1
        if i == answer_index:
            check(is_balanced, f"answer option (index {i}) should be balanced")
        else:
            check(not is_balanced, f"distractor option (index {i}) should not be balanced")

    working = ["Equally likely outcomes means every colour appears the same number of times. Count each colour."]
    for i, spinner in enumerate(shuffled_spinners):
        tally = Counter(spinner["labels"])
        tally_text = ", ".join(f"{colour} {count}" for colour, count in tally.items())
        verdict = "equal" if i == answer_index else "not equal"
        working.append(f"Spinner {chr(65 + i)}: {tally_text} ({verdict}).")
    working.append(f"Only Spinner {chr(65 + answer_index)} has the same count for every colour.")

    return dict(
        text=question_text,
        working=working,
        latex=f"plain:{question_text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=distractors,
    )
