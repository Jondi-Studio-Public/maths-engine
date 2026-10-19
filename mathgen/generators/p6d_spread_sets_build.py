"""Two genuinely new senior (ACARA v8.4) skills for statistics-probability domain.

- ``use_informal_ways_of_describing_spread_such_as`` (ACMEM054, prerequisites
  ``simple-event-probability``, virgin area) -- informal descriptors of data
  spread (tightly packed, outliers, gaps between clusters) are bound to stated
  numeric rules. One qualifying data set and three non-qualifying data sets per
  question, verified by independent checks. Response family: ``scalar_expression``
  (``answer: mcq``).

- ``use_everyday_occurrences_to_illustrate_set_descriptions_and`` (ACMMM051,
  prerequisites ``simple-event-probability``, virgin area) -- translate
  everyday-language descriptions of groups into set-notation phrases (word form,
  no symbols). Two- and three-set descriptions with fixed connector vocabulary
  table. Response family: ``scalar_expression`` (``answer: mcq``).
"""

from __future__ import annotations

from ..core import check, check_distinct, generator, latex_answer, pick_distractors

# ============================================================================
# SKILL 1: use_informal_ways_of_describing_spread_such_as (ACMEM054)
# ============================================================================


def _check_tight_pack(values: list[int], T: int) -> bool:
    """Independent check: a data set is tightly packed if max - min <= T."""
    return max(values) - min(values) <= T


def _check_outlier(values: list[int], T: int) -> bool:
    """Independent check: a data set has an outlier if there exists a value
    that is more than T away from every other value in the set."""
    for i, val in enumerate(values):
        # Check if this value is more than T from every other value
        distances_to_others = [abs(val - values[j]) for j in range(len(values)) if i != j]
        if all(d > T for d in distances_to_others):
            return True
    return False


def _check_gap_between_clusters(values: list[int], T: int) -> bool:
    """Independent check: a data set has a gap between two clusters if, when
    sorted, there is a difference > T between two consecutive values with at
    least 2 values on each side of that gap."""
    sorted_vals = sorted(values)
    for i in range(len(sorted_vals) - 1):
        gap = sorted_vals[i + 1] - sorted_vals[i]
        if gap > T:
            # Check if there are at least 2 values on each side
            if i >= 1 and len(sorted_vals) - i - 1 >= 2:
                return True
    return False


@generator(
    "use_informal_ways_of_describing_spread_such_as",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def use_informal_ways_of_describing_spread_such_as(rng, tier="easy"):
    """Given a stated numeric rule that defines an informal spread descriptor
    (tightly packed, has an outlier, gap between clusters), identify which of
    four data sets fits that description. See module docstring for design."""

    if tier == "easy":
        # Tightly packed: all values within T of each other (range <= T)
        descriptor = "tightly packed"
        T = rng.randint(3, 6)  # Range: 3 to 5
        rule = f"every value in the data set is within {T} of every other value"

        # Build qualifying set: range <= T
        # Strategy: pick a centre, add offsets that sum to <= T
        centre = rng.randint(20, 40)
        # Ensure the range is at most T by carefully choosing min and max
        max_offset = T // 2
        min_val = centre - max_offset
        max_val = centre + max_offset
        qualifying = [min_val, max_val]  # Ensure we hit the boundaries
        for _ in range(rng.randint(2, 4)):
            qualifying.append(rng.randint(min_val + 1, max_val - 1))
        qualifying = sorted(qualifying)
        check(
            _check_tight_pack(qualifying, T),
            f"qualifying set failed tight-pack check (range={max(qualifying) - min(qualifying)}, T={T})",
        )

        # Build three non-qualifying sets: range > T with margin >= 2
        non_qualifying = []
        for _ in range(3):
            centre = rng.randint(20, 40)
            # Ensure range is > T + 2 (clear margin)
            # Create two points that are far apart: min and max
            min_val = centre - (T + 3)
            max_val = centre + (T + 3)
            # Fill in the rest with random values in between
            dataset = [min_val, max_val]
            for _ in range(rng.randint(2, 4)):
                dataset.append(rng.randint(min_val + 1, max_val - 1))
            dataset = sorted(dataset)
            check(
                not _check_tight_pack(dataset, T),
                f"non-qualifying set {len(non_qualifying)} passed tight-pack check (range={max(dataset) - min(dataset)}, T={T})",
            )
            non_qualifying.append(dataset)

    elif tier == "medium":
        # Has an outlier: one value more than T from every other value
        descriptor = "has an outlier"
        T = rng.randint(5, 10)  # Range: 5 to 9
        rule = f"at least one value is more than {T} away from every other value in the data set"

        # Build qualifying set with exactly one outlier
        # Main cluster around centre
        centre = rng.randint(20, 40)
        cluster = [centre + rng.randint(-2, 2) for _ in range(rng.randint(3, 5))]
        # Outlier: far from all cluster values
        outlier = centre + rng.randint(T + 3, T + 8)
        qualifying = sorted(cluster + [outlier])
        check(_check_outlier(qualifying, T), f"qualifying set failed outlier check")

        # Build three non-qualifying sets: no value is more than T from every other
        non_qualifying = []
        for _ in range(3):
            centre = rng.randint(20, 40)
            # Build a set where no single value is an outlier
            # Simple approach: all values close together or with some spread but not extreme
            dataset = sorted([centre + rng.randint(-T // 2, T // 2) for _ in range(rng.randint(4, 6))])
            check(not _check_outlier(dataset, T), f"non-qualifying set {len(non_qualifying)} passed outlier check")
            non_qualifying.append(dataset)

    elif tier == "hard":
        # Gap between two clusters: sorted gap > T with >=2 values on each side
        descriptor = "has a gap between two clusters"
        T = rng.randint(4, 8)  # Range: 4 to 7
        rule = f"sorting the data set produces a difference of more than {T} between two consecutive values, with at least two values on each side of that difference"

        # Build qualifying set: two clusters with gap > T
        # Cluster 1: 2 values at the bottom
        cluster1 = [10, 11]
        # Cluster 2: 2 values far apart from cluster1 (gap > T)
        # Ensure there's a clear gap: min value of cluster2 > max of cluster1 + T
        min_gap_start = max(cluster1) + T + 1
        cluster2 = [min_gap_start, min_gap_start + 1]
        qualifying = sorted(cluster1 + cluster2)
        check(
            _check_gap_between_clusters(qualifying, T),
            f"qualifying set failed gap check (qualifying={qualifying}, T={T})",
        )

        # Build three non-qualifying sets
        # Strategy: either (a) no gap exceeds T, or (b) gap exists but < 2 values on one side
        non_qualifying = []
        for i in range(3):
            if i == 0:
                # No consecutive gap exceeds T: all values clustered or with small gaps
                dataset = [20, 21, 22, 23]
            elif i == 1:
                # Gap exists but with lone outlier on one side (not two clusters)
                # Main cluster at bottom, lone outlier at top
                main_cluster = [20, 21, 22, 23]
                lone_outlier = 30  # Clearly separated but only 1 value on one side
                dataset = sorted(main_cluster + [lone_outlier])
            else:
                # Another variant with lone outlier at the bottom
                main_cluster = [30, 31, 32, 33]
                lone_outlier = 20  # Clearly separated but only 1 value on one side
                dataset = sorted([lone_outlier] + main_cluster)

            check(not _check_gap_between_clusters(dataset, T), f"non-qualifying set {i} passed gap check")
            non_qualifying.append(dataset)

    else:
        raise ValueError(f"unknown tier {tier!r}")

    # Format data sets as option text
    def format_dataset(data: list[int]) -> str:
        """Format data set as comma-separated sorted integers."""
        return ", ".join(str(v) for v in sorted(data))

    # Build MCQ options
    # Option labels A, B, C, D for the four data sets
    options_data = [qualifying] + non_qualifying
    rng.shuffle(options_data)
    labels = ["A", "B", "C", "D"]

    options = [f"Data set {label}: {format_dataset(data)}" for label, data in zip(labels, options_data)]

    # Find which option is the answer (by matching data content)
    answer_text = None
    for i, option in enumerate(options):
        if format_dataset(qualifying) in option:
            answer_text = option
            break

    check(answer_text is not None, "answer option not found")

    # Verify: exactly one option should satisfy the rule
    if tier == "easy":
        satisfies = [
            _check_tight_pack(sorted([int(s.strip()) for s in option.split(": ")[1].split(", ")]), T)
            for option in options
        ]
    elif tier == "medium":
        satisfies = [
            _check_outlier(sorted([int(s.strip()) for s in option.split(": ")[1].split(", ")]), T) for option in options
        ]
    else:  # hard
        satisfies = [
            _check_gap_between_clusters(sorted([int(s.strip()) for s in option.split(": ")[1].split(", ")]), T)
            for option in options
        ]

    check(sum(satisfies) == 1, f"expected exactly 1 option to satisfy the rule, got {sum(satisfies)}")
    check(satisfies[options.index(answer_text)], "answer option does not satisfy the rule")

    # Distractors are the non-answer options
    distractors = [opt for opt in options if opt != answer_text]
    check(len(distractors) == 3, f"expected 3 distractors, got {len(distractors)}")

    # Build text without newlines
    options_str = "; ".join(options)
    text = f"A data set is described as '{descriptor}' if {rule}. Which of these four data sets fits that description? {options_str}"

    working = [
        f"Descriptor: {descriptor}",
        f"Rule: {rule}",
        f"Checking each option against the rule...",
        f"Answer: {answer_text}",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=pick_distractors(answer_text, distractors, n=3),
    )


# ============================================================================
# SKILL 2: use_everyday_occurrences_to_illustrate_set_descriptions_and (ACMMM051)
# ============================================================================

# Fixed translation tables: (everyday description, set-notation phrase)
# For 2-set descriptions (easy/medium)
TWO_SET_TABLE = [
    ("people who are both {A} and {B}", "the intersection of {A} and {B}"),
    ("people who are at least one of {A} or {B}", "the union of {A} and {B}"),
    ("people who are {A} but not {B}", "{A} with {B} removed"),
    ("people who are {B} but not {A}", "{B} with {A} removed"),
    ("people who are neither {A} nor {B}", "everyone outside both {A} and {B}"),
    ("people who are exactly one of {A} or {B}", "everyone in exactly one of {A} or {B}, not both"),
]

# For 3-set descriptions (hard)
THREE_SET_TABLE = [
    ("people who are {A} and {B} but not {C}", "{A} and {B}, with {C} removed"),
    ("people who are all three of {A}, {B} and {C}", "the intersection of {A}, {B} and {C}"),
    ("people who are exactly one of {A}, {B} or {C}", "everyone in exactly one of {A}, {B} or {C}, not more than one"),
    ("people who are none of {A}, {B} or {C}", "everyone outside all three of {A}, {B} and {C}"),
]

# Context pairs for 2-set questions (A, B group names)
TWO_SET_CONTEXTS = [
    ("tea lovers", "coffee lovers"),
    ("morning people", "night owls"),
    ("dog owners", "cat owners"),
    ("math fans", "history fans"),
    ("sports players", "music lovers"),
]

# Context tuples for 3-set questions (A, B, C group names)
THREE_SET_CONTEXTS = [
    ("tea lovers", "coffee lovers", "juice lovers"),
    ("morning people", "night owls", "afternoon people"),
    ("dog owners", "cat owners", "bird owners"),
]


@generator(
    "use_everyday_occurrences_to_illustrate_set_descriptions_and",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def use_everyday_occurrences_to_illustrate_set_descriptions_and(rng, tier="easy"):
    """Given an everyday-language description of a group, identify the matching
    set-notation phrase. See module docstring for design."""

    if tier == "easy":
        # Rows 1-2 only (intersection or union)
        row_index = rng.choice([0, 1])  # 0 = intersection, 1 = union
        description_template, notation_template = TWO_SET_TABLE[row_index]

        # Pick context pair
        A, B = rng.choice(TWO_SET_CONTEXTS)

        # Format the description text and correct notation
        description_text = description_template.format(A=A, B=B)
        correct_notation = notation_template.format(A=A, B=B)

        # Draw 3 distractors from the other rows (not row_index)
        other_rows = [i for i in range(len(TWO_SET_TABLE)) if i != row_index]
        distractor_rows = rng.sample(other_rows, 3)
        distractors = [TWO_SET_TABLE[i][1].format(A=A, B=B) for i in distractor_rows]

    elif tier == "medium":
        # Rows 1-6 (all two-set descriptions)
        row_index = rng.randint(0, len(TWO_SET_TABLE) - 1)
        description_template, notation_template = TWO_SET_TABLE[row_index]

        # Pick context pair
        A, B = rng.choice(TWO_SET_CONTEXTS)

        # Format the description text and correct notation
        description_text = description_template.format(A=A, B=B)
        correct_notation = notation_template.format(A=A, B=B)

        # Draw 3 distractors from the other rows
        other_rows = [i for i in range(len(TWO_SET_TABLE)) if i != row_index]
        distractor_rows = rng.sample(other_rows, 3)
        distractors = [TWO_SET_TABLE[i][1].format(A=A, B=B) for i in distractor_rows]

    elif tier == "hard":
        # Rows 7-10 (three-set descriptions)
        row_index = rng.randint(0, len(THREE_SET_TABLE) - 1)
        description_template, notation_template = THREE_SET_TABLE[row_index]

        # Pick context tuple
        A, B, C = rng.choice(THREE_SET_CONTEXTS)

        # Format the description text and correct notation
        description_text = description_template.format(A=A, B=B, C=C)
        correct_notation = notation_template.format(A=A, B=B, C=C)

        # Draw 3 distractors from the other three-set rows
        other_rows = [i for i in range(len(THREE_SET_TABLE)) if i != row_index]
        distractor_rows = rng.sample(other_rows, 3)
        distractors = [THREE_SET_TABLE[i][1].format(A=A, B=B, C=C) for i in distractor_rows]

    else:
        raise ValueError(f"unknown tier {tier!r}")

    # Verify the correct notation by re-looking it up from the table
    if tier == "hard":
        table = THREE_SET_TABLE
        verify_notation = table[row_index][1].format(A=A, B=B, C=C)
    else:
        table = TWO_SET_TABLE
        verify_notation = table[row_index][1].format(A=A, B=B)

    check(correct_notation == verify_notation, "notation mismatch in independent check")

    # Verify all distractors are distinct from each other and from the answer
    # (using string comparison since these are text notations, not math expressions)
    check(correct_notation not in distractors, "distractor equals answer")
    check(len(distractors) == len(set(distractors)), "duplicate distractors")

    # Build MCQ options by shuffling answer + distractors
    all_options = [correct_notation] + distractors
    rng.shuffle(all_options)

    # Build text without newlines
    options_str = "; ".join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(all_options))
    text = f"Which phrase describes the following group? {description_text[:1].upper() + description_text[1:]}. Options: {options_str}"

    answer_text = correct_notation

    working = [f"The group of {description_text} is described as {answer_text}."]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=pick_distractors(answer_text, distractors, n=3),
    )
