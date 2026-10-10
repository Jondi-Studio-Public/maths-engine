"""Ordered-structures pilot generators: the first two
generators whose graded answer is a typed *ordered* sequence
(``mathgen/grading/typed.py::grade_ordered_list``), rather than a scalar or an
unordered set/list.

- ``large_number_ordering`` (AC9M3N01, prerequisite ``place-value-to-100``) --
  order a set of distinct natural numbers greater than 10,000, smallest
  first. The original specification also covers reading/writing/
  representing the numeral, which is real curriculum content but is not
  itself gradable as a typed ordered-list response -- excluded here and
recorded; only
  ordering by value is graded.
- ``day_time_sequencing`` (AC9MFM02, prerequisite ``tell-time-to-hour``) --
  order a shuffled window of familiar day-parts or days of the week into
  their correct chronological sequence. The original "match each to a
  supplied familiar event or action" clause is realised as question framing:
  each label is shown paired with a short familiar event (e.g. "morning --
  eat breakfast"), shuffled in display order. The learner's one typed
  response is the label sequence, not the events -- matching a label to its
  event is established by the display itself, not by a second inference the
  learner must perform, so this stays a single graded field (a list of
  labels), never a second one. No week-wrap: a generated day-of-week window
  never runs past Sunday, since calendar arithmetic is excluded
  and wrapping would introduce cyclic-order reasoning this skill does not
  assess.
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, pick_distractors, reversed_and_adjacent_swap

# -- large_number_ordering -------------------------------------------------

# Inclusive (lo, hi) natural-number range for each digit length this
# generator ever samples from. 5-digit starts at 10,001 (the descriptor's
# "greater than 10,000"); 7-digit stops at 9,999,999 (the stated hard
# ceiling).
_LENGTH_RANGES = {
    5: (10_001, 99_999),
    6: (100_000, 999_999),
    7: (1_000_000, 9_999_999),
}


def _verify_large_number_order(sampled: list[int], ordered: list[int]) -> None:
    """Independent check, deliberately not a re-run of the numeric sort that
    built ``ordered`` (``sorted(sampled, key=lambda n: (len(str(n)), n))``):
    this walks adjacent pairs and asserts ascending order using only (a) an
    integer digit-count comparison and (b) for equal-length numbers, a plain
    Python string comparison of their decimal representations -- never a
    numeric ``<`` between the two numbers themselves. Digit-count comparison
    and equal-length decimal-string comparison are each independently valid
    orderings of positive integers, but neither is "run the same sort again"
    -- a construction bug that produced the wrong numeric order while still
    preserving digit-count-then-lexicographic shape would still be caught,
    because this check only trusts digit length and raw string comparison,
    not the numeric key the construction used."""
    check(sorted(sampled) == sorted(ordered), "ordered list is not a permutation of the sampled numbers")
    check(len(ordered) == len(set(ordered)), "sampled numbers must be distinct")
    for a, b in zip(ordered, ordered[1:]):
        text_a, text_b = str(a), str(b)
        if len(text_a) != len(text_b):
            check(len(text_a) < len(text_b), f"{a} has more digits than {b} but is placed before it")
        else:
            check(text_a < text_b, f"{a} does not precede {b} as equal-length decimal strings")


def _format_grouped(n: int) -> str:
    """Comma thousands-grouping (e.g. 12345 -> "12,345"), the format
    ``grade_ordered_list`` explicitly tolerates for a numeric item. No
    existing large-number formatting helper was found elsewhere in
    ``mathgen/generators`` (checked ``curriculum_expansion.py`` and
    ``acara_gap_build.py``) to reuse instead."""
    return f"{n:,}"


@generator(
    "large_number_ordering",
    level=3,
    presets={
        "easy": dict(digit_lengths=(5,)),
        "medium": dict(digit_lengths=(6,)),
        "hard": dict(digit_lengths=(5, 6, 7)),
    },
)
def large_number_ordering(rng, digit_lengths=(5, 6, 7), count=4):
    """Order a set of distinct natural numbers greater than 10,000,
    ascending, smallest first."""
    if count < 2:
        raise ValueError("count must be at least 2")
    digit_lengths = tuple(digit_lengths)
    unknown = [length for length in digit_lengths if length not in _LENGTH_RANGES]
    if not digit_lengths or unknown:
        raise ValueError(f"digit_lengths must be a non-empty subset of {sorted(_LENGTH_RANGES)}; got {digit_lengths!r}")

    if len(digit_lengths) == 1:
        lo, hi = _LENGTH_RANGES[digit_lengths[0]]
        if hi - lo + 1 < count:
            raise ValueError("digit-length range is too small for count distinct numbers")
        numbers = list(rng.sample(range(lo, hi + 1), count))
    else:
        # hard: force at least two distinct digit lengths to actually appear,
        # rather than hoping a uniform per-item choice happens to vary --
        # a naive independent rng.choice per item has a real (if small)
        # chance of landing on the same length every time.
        chosen_lengths = [rng.choice(digit_lengths) for _ in range(count)]
        if len(set(chosen_lengths)) == 1:
            other_lengths = [length for length in digit_lengths if length != chosen_lengths[0]]
            chosen_lengths[rng.randrange(count)] = rng.choice(other_lengths)
        numbers = []
        seen: set[int] = set()
        for length in chosen_lengths:
            lo, hi = _LENGTH_RANGES[length]
            n = rng.randint(lo, hi)
            while n in seen:
                n = rng.randint(lo, hi)
            seen.add(n)
            numbers.append(n)

    ordered = sorted(numbers, key=lambda n: (len(str(n)), n))
    _verify_large_number_order(numbers, ordered)

    display_order = list(numbers)
    rng.shuffle(display_order)

    # Displayed with "; " rather than ", " between numbers so a learner
    # copying the question's punctuation into their typed answer does not
    # walk into `_split_ordered_items`'s documented comma-grouping trap
    # (comma is both the item separator and the thousands-grouping
    # character; a comma-separated display with no space after each
    # separator, e.g. six-digit numbers at medium difficulty, would each
    # render as an exact 3-digit group and could be swallowed into one
    # merged numeral if a learner drops the spaces when retyping it). The
    # typed *answer* itself is still expected comma/newline/arrow-separated,
    # per the grader's own contract -- only the question's display changes.
    display_text = "; ".join(_format_grouped(n) for n in display_order)
    answer_text = ", ".join(_format_grouped(n) for n in ordered)

    # Misconception distractors:
    #  - reversed-order: the numbers ordered largest to smallest.
    #  - digit-count-only: grouped by digit count but never compared by value
    #    within a length group (equal-length numbers left in as-shown order).
    #  - first-digit-only: sorted purely by each numeral's leading digit,
    #    ignoring the rest of the number and its overall digit count.
    #  - text-sort-ignores-length: the numerals sorted as plain decimal-digit
    #    strings without regard to differing digit counts, so a shorter
    #    number with a larger leading digit wrongly outranks a longer number
    #    with a smaller leading digit (e.g. "99,999" placed after "100,000").
    #  - adjacent-swap: correctly ordered overall, but the two middle values
    #    are swapped, as if two neighbouring magnitudes did not need
    #    comparing once "roughly" placed. Kept last in priority order but is
    #    the one candidate proven distinct from both the answer and
    #    reversed_order for every count-4 case: it differs from the answer at
    #    the swapped middle positions (the two middle values are always
    #    distinct), and it differs from reversed_order at position 0 (its
    #    first item is still the smallest value, never the largest, since
    #    only the two middle positions moved) -- this is what keeps the
    #    distractor floor at 2 (confirmed by a 9,000-seed scan; see the
    #    text-sort-ignores-length degenerate into duplicates of each other,
    #    the answer, or reversed_order (which happens whenever every sampled
    #    number shares one digit length, e.g. easy/medium: text-sort then
    #    coincides exactly with numeric order, since equal-length decimal
    #    strings compare the same way their values do).
    reversed_order, adjacent_swap = reversed_and_adjacent_swap(ordered)
    digit_count_only = sorted(display_order, key=lambda n: len(str(n)))
    first_digit_only = sorted(display_order, key=lambda n: str(n)[0])
    text_sort_ignores_length = sorted(display_order, key=str)
    candidates = [
        ", ".join(_format_grouped(n) for n in reversed_order),
        ", ".join(_format_grouped(n) for n in digit_count_only),
        ", ".join(_format_grouped(n) for n in first_digit_only),
        ", ".join(_format_grouped(n) for n in text_sort_ignores_length),
        ", ".join(_format_grouped(n) for n in adjacent_swap),
    ]

    return dict(
        text=(f"Order these numbers from smallest to largest: {display_text}."),
        latex=rf"\text{{Order: }} {display_text}",
        answer=list(ordered),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            "Count the digits in each number -- a number with more digits is always "
            "larger, no matter what its digits are.",
            "Within numbers that have the same digit count, compare digit by digit from the left.",
            f"Ordered smallest to largest: {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


# -- day_time_sequencing ----------------------------------------------------

_DAY_PARTS = ["morning", "afternoon", "evening", "night"]
_DAY_PART_INDEX = {label: i for i, label in enumerate(_DAY_PARTS)}
_DAY_PART_EVENTS = {
    "morning": [
        "wake up and eat breakfast",
        "get dressed for school",
        "brush your teeth after waking up",
    ],
    "afternoon": [
        "eat lunch",
        "play outside after school",
        "do afternoon sport training",
    ],
    "evening": [
        "eat dinner with your family",
        "watch the sun go down",
        "do homework after school",
    ],
    "night": [
        "put on pyjamas and go to sleep",
        "read a bedtime story",
        "turn off the lights to sleep",
    ],
}

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_WEEKDAY_INDEX = {label: i for i, label in enumerate(_WEEKDAYS)}
_WEEKDAY_EVENTS = {
    "Monday": ["a music lesson", "the school assembly", "the start of the school week"],
    "Tuesday": ["a library visit", "sports training", "an art class"],
    "Wednesday": ["swimming lessons", "a school excursion", "a science class"],
    "Thursday": ["a computer lab session", "gardening club", "a class news day"],
    "Friday": ["free-dress day", "an end-of-week quiz", "an awards assembly"],
    "Saturday": ["a weekend sport game", "a family outing", "a sleep-in morning"],
    "Sunday": ["a family lunch", "a quiet rest day", "getting ready for the school week"],
}


def _verify_chronological_order(labels: list[str], canonical_index: dict[str, int]) -> None:
    """Independent check: verify every pair (not just adjacent ones) strictly
    increases against the fixed canonical index, by direct pairwise
    comparison -- never by re-running whatever sort or shuffle-reversal built
    the sequence in the first place."""
    check(len(labels) == len(set(labels)), "sequenced labels must be distinct")
    for label in labels:
        check(label in canonical_index, f"{label!r} is not a recognised label")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(
                canonical_index[labels[i]] < canonical_index[labels[j]],
                f"{labels[i]} does not chronologically precede {labels[j]}",
            )


@generator(
    "day_time_sequencing",
    level=1,
    presets={
        "easy": dict(window="day_parts"),
        "medium": dict(window="weekdays", run_lengths=(4,)),
        "hard": dict(window="weekdays", run_lengths=(5, 6)),
    },
)
def day_time_sequencing(rng, window="day_parts", run_lengths=(4,)):
    """Order a shuffled window of familiar day-parts or days of the week into
    their correct chronological sequence. Each label is shown paired with a
    short familiar event; the graded answer is the label sequence."""
    if window == "day_parts":
        labels = list(_DAY_PARTS)
        canonical_index = _DAY_PART_INDEX
        events = _DAY_PART_EVENTS
        cycle_name = "parts of the day"
    elif window == "weekdays":
        run_lengths = tuple(run_lengths)
        if not run_lengths or any(length < 2 or length > len(_WEEKDAYS) for length in run_lengths):
            raise ValueError(
                f"run_lengths must be non-empty values between 2 and {len(_WEEKDAYS)}; got {run_lengths!r}"
            )
        length = rng.choice(run_lengths)
        max_start = len(_WEEKDAYS) - length
        start = rng.randint(0, max_start)  # never lets the run reach past Sunday
        labels = _WEEKDAYS[start : start + length]
        canonical_index = _WEEKDAY_INDEX
        events = _WEEKDAY_EVENTS
        cycle_name = "days of the week"
    else:
        raise ValueError(f"unknown window {window!r}; expected 'day_parts' or 'weekdays'")

    correct_order = sorted(labels, key=lambda label: canonical_index[label])
    _verify_chronological_order(correct_order, canonical_index)

    display_items = [(label, rng.choice(events[label])) for label in labels]
    rng.shuffle(display_items)

    lines = "; ".join(f"{label} -- {event}" for label, event in display_items)
    answer_text = ", ".join(correct_order)

    # Misconception distractors:
    #  - reversed-order: latest first instead of earliest first.
    #  - alphabetical-order: labels sorted alphabetically instead of
    #    chronologically.
    #  - adjacent-swap: swaps two adjacent periods (e.g. afternoon and
    #    evening) as if their order did not matter.
    #  - keeps-shuffled-order: writes the periods back in the order they
    #    were shown, without reordering them at all.
    reversed_order, adjacent_swap = reversed_and_adjacent_swap(correct_order)
    alphabetical_order = sorted(correct_order, key=str.lower)
    keeps_shuffled_order = [label for label, _event in display_items]
    candidates = [
        ", ".join(reversed_order),
        ", ".join(alphabetical_order),
        ", ".join(adjacent_swap),
        ", ".join(keeps_shuffled_order),
    ]

    return dict(
        text=(
            f"Here are some {cycle_name}, each shown with something that happens then, "
            f"listed in a random order: {lines}. Write the {cycle_name} in the order "
            "they actually happen, earliest first."
        ),
        latex=rf"\text{{Order: {lines}}}",
        answer=list(correct_order),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"The {cycle_name} always happen in a fixed order: "
            f"{', '.join(labels if window == 'day_parts' else _WEEKDAYS)}.",
            f"Reading off just the ones shown, in that fixed order, gives: {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )
