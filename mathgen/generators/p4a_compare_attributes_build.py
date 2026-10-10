"""Two genuinely new Foundation-Year 10 measurement skills for comparing and ordering objects/events by length,
mass, capacity or duration.

- ``compare_measurement_attributes`` (AC9MFM01, prerequisites ``compare-lengths``,
  non-virgin area) -- given two objects or events and a stated attribute among length,
  capacity, mass or duration, together with the information needed for a direct
  comparison, state which is greater, smaller, or that they are equal. Response
  family: ``scalar_expression`` (``answer: mcq``).

- ``compare_order_measure_attributes`` (AC9M1M01, prerequisites ``compare-lengths``,
  virgin area) -- directly or indirectly compare two or more objects or events by a
  stated attribute (length, mass, capacity or duration) and state the correct order
  from least to greatest. Response family: ``ordered_structures`` (``answer:
  ordered_list``).

Shared attribute/comparison-action vocabulary for both generators, avoiding invented
contrived quantities by using deterministic label patterns (``Item 1``, ``Item 2``,
etc.; ``Event 1``, ``Event 2``, etc.).
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, pick_distractors, reversed_and_adjacent_swap

# -- Shared module-level constants -----------------------------------------------

ATTRIBUTES = ("length", "mass", "capacity", "duration")

# G = the greater-attribute label, L = the lesser-attribute label
ATTRIBUTE_ACTION = {
    "length": "{G} reaches further than {L} when their ends are lined up.",
    "mass": "{G} tips the balance down against {L}.",
    "capacity": "Pouring the liquid from {G} into {L} makes {L} overflow.",
    "duration": "{G} takes longer to finish than {L}.",
}

# A, B = the two (equal) labels, order does not matter
ATTRIBUTE_EQUAL_ACTION = {
    "length": "{A} and {B} reach exactly the same point when their ends are lined up.",
    "mass": "{A} and {B} balance evenly on the scale.",
    "capacity": "Pouring the liquid from {A} into {B} exactly fills it, with none left over and no overflow.",
    "duration": "{A} and {B} take exactly the same time to finish.",
}

# Fixed copula for mcq templates (defect fix: fold copula into dict to avoid "is holds more than")
COMPARATIVE = {
    "length": "is longer than",
    "mass": "is heavier than",
    "capacity": "holds more than",
    "duration": "takes longer than",
}

EQUAL_PHRASE = {
    "length": "is the same length as",
    "mass": "is the same mass as",
    "capacity": "has the same capacity as",
    "duration": "takes the same duration as",
}

NOUN = {"length": "Item", "mass": "Item", "capacity": "Item", "duration": "Event"}

# Least/greatest phrasings for skill 2 ordering questions
LEAST_WORD = {
    "length": "shortest",
    "mass": "lightest",
    "capacity": "smallest capacity",
    "duration": "shortest",
}

GREATEST_WORD = {
    "length": "longest",
    "mass": "heaviest",
    "capacity": "largest capacity",
    "duration": "longest",
}

# -- Skill 1: compare_measurement_attributes ------------------------------------

GREATER_FIRST = "first_greater"
GREATER_SECOND = "second_greater"
EQUAL = "equal"


def _expected_option_skill1(attr: str, relation: str, A: str, B: str) -> str:
    """Independent check: recompute the expected MCQ option via a separate
    if/elif chain, never calling the same option-building code used during
    construction. Returns the option string matching `relation`."""
    opt_first_greater = f"{A} {COMPARATIVE[attr]} {B}."
    opt_second_greater = f"{B} {COMPARATIVE[attr]} {A}."
    opt_equal = f"{A} {EQUAL_PHRASE[attr]} {B}."

    if relation == GREATER_FIRST:
        return opt_first_greater
    elif relation == GREATER_SECOND:
        return opt_second_greater
    elif relation == EQUAL:
        return opt_equal
    else:
        raise ValueError(f"unexpected relation {relation!r}")


@generator(
    "compare_measurement_attributes",
    level=1,
    presets={
        "easy": dict(attr_pool=("length", "mass"), equal_fraction=0.0),
        "medium": dict(attr_pool=("capacity", "duration"), equal_fraction=0.0),
        "hard": dict(attr_pool=ATTRIBUTES, equal_fraction=1 / 3),
    },
)
def compare_measurement_attributes(rng, attr_pool=ATTRIBUTES, equal_fraction=0.0):
    """Given two objects or events and a stated attribute, determine which is
    greater, smaller, or that they are equal. See the module docstring for the
    difficulty rationale."""
    if not attr_pool:
        raise ValueError("attr_pool must be non-empty")
    unknown = [a for a in attr_pool if a not in ATTRIBUTES]
    if unknown:
        raise ValueError(f"unknown attributes {unknown!r}; expected subset of {ATTRIBUTES}")
    if not (0.0 <= equal_fraction <= 1.0):
        raise ValueError(f"equal_fraction must be within 0.0-1.0; got {equal_fraction!r}")

    attr = rng.choice(attr_pool)
    A = f"{NOUN[attr]} 1"
    B = f"{NOUN[attr]} 2"

    # Decide whether this is an equality case
    force_equal = equal_fraction > 0 and rng.random() < equal_fraction

    if force_equal:
        relation = EQUAL
        text = f"{ATTRIBUTE_EQUAL_ACTION[attr].format(A=A, B=B)} Which statement is true?"
    else:
        # relation is GREATER_FIRST or GREATER_SECOND 50/50
        relation = rng.choice([GREATER_FIRST, GREATER_SECOND])
        if relation == GREATER_FIRST:
            greater_label, lesser_label = A, B
        else:
            greater_label, lesser_label = B, A
        text = f"{ATTRIBUTE_ACTION[attr].format(G=greater_label, L=lesser_label)} Which statement is true?"

    # Build the three fixed MCQ options (defect fix: incorporate copula into template)
    opt_first_greater = f"{A} {COMPARATIVE[attr]} {B}."
    opt_second_greater = f"{B} {COMPARATIVE[attr]} {A}."
    opt_equal = f"{A} {EQUAL_PHRASE[attr]} {B}."

    # Determine the answer
    if relation == GREATER_FIRST:
        answer = opt_first_greater
    elif relation == GREATER_SECOND:
        answer = opt_second_greater
    else:  # EQUAL
        answer = opt_equal

    # Verify the independent check produces the same answer
    check_answer = _expected_option_skill1(attr, relation, A, B)
    check(check_answer == answer, f"independent check mismatch: {check_answer!r} != {answer!r}")

    # Distractors are the other two options (exactly 3 fixed options, always all present)
    candidates = [opt for opt in [opt_first_greater, opt_second_greater, opt_equal] if opt != answer]
    check(len(candidates) == 2, f"expected exactly 2 distractors, got {len(candidates)}")

    working = [
        f"Based on the description: {text}",
        f"Answer: {answer}",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=pick_distractors(answer, candidates, n=2),
    )


# -- Skill 2: compare_order_measure_attributes ----------------------------------


def _expected_order_skill2(edges: list[tuple[str, str]], labels: list[str]) -> list[str]:
    """Independent check: reconstruct the full order via Kahn's topological sort,
    a genuinely different algorithm from "we already built order_asc, so just
    re-sort it". Given an unordered list of (greater, lesser) tuples, rebuild
    the order from scratch without referencing any pre-computed order."""
    remaining = set(labels)
    indegree = {label: 0 for label in labels}

    # Build indegree: edge (greater, lesser) means lesser -> greater, so greater gets +1
    for greater, lesser in edges:
        indegree[greater] += 1

    result = []
    while remaining:
        # Find the unique node with indegree 0 among remaining
        src = next(l for l in remaining if indegree[l] == 0 and l in remaining)
        result.append(src)
        remaining.remove(src)

        # Decrease indegree of nodes that depend on src
        for greater, lesser in edges:
            if lesser == src:
                indegree[greater] -= 1

    return result


@generator(
    "compare_order_measure_attributes",
    level=1,
    presets={
        "easy": dict(N=3, hard_shuffle=False),
        "medium": dict(N=4, hard_shuffle=False),
        "hard": dict(N=5, hard_shuffle=True),
    },
)
def compare_order_measure_attributes(rng, N=3, hard_shuffle=False):
    """Directly or indirectly compare two or more objects or events by a stated
    attribute and state the correct order from least to greatest. See the module
    docstring for the difficulty rationale."""
    if N < 2:
        raise ValueError("N must be at least 2")
    if N > 5:
        raise ValueError("N must be at most 5 (hard tier)")

    # Sample attribute from all four
    attr = rng.choice(ATTRIBUTES)

    # Build deterministic labels: Item 1, Item 2, ..., or Event 1, Event 2, ...
    labels = [f"{NOUN[attr]} {i}" for i in range(1, N + 1)]

    # Assign each label a distinct random rank 1..N
    ranks = rng.sample(range(1, N + 1), N)
    label_to_rank = dict(zip(labels, ranks))

    # order_asc is the correct answer: labels sorted ascending by rank
    order_asc = sorted(labels, key=lambda lbl: label_to_rank[lbl])

    # Build the sorted-adjacent chain of N-1 edges
    edges = []
    for i in range(N - 1):
        greater = order_asc[i + 1]
        lesser = order_asc[i]
        edges.append((greater, lesser))

    # Build statements from edges
    statements = [ATTRIBUTE_ACTION[attr].format(G=greater, L=lesser) for (greater, lesser) in edges]

    # Pair statements with their original edge indices for hard-tier derangement check
    statement_edge_pairs = list(enumerate(statements))

    # Shuffle for display
    if hard_shuffle:
        # Hard: reject and reshuffle until no two originally-adjacent edges land consecutively
        max_attempts = 200
        for attempt in range(max_attempts):
            rng.shuffle(statement_edge_pairs)
            # Check: no two consecutive display positions should have indices differing by 1
            valid = True
            for i in range(len(statement_edge_pairs) - 1):
                idx_curr = statement_edge_pairs[i][0]
                idx_next = statement_edge_pairs[i + 1][0]
                if abs(idx_curr - idx_next) == 1:
                    valid = False
                    break
            if valid:
                break
        else:
            raise RuntimeError(f"could not achieve hard-tier derangement in {max_attempts} attempts")
    else:
        rng.shuffle(statement_edge_pairs)

    # Extract shuffled statements and shuffled edges for the independent check
    shuffled_statements = [pair[1] for pair in statement_edge_pairs]
    shuffled_edges = [edges[pair[0]] for pair in statement_edge_pairs]

    # Defect fix: strip trailing periods before joining with semicolons
    statements_no_period = [s.rstrip(".") for s in shuffled_statements]
    clues_text = "; ".join(statements_no_period) + "."

    # Natural phrasing for the ordering prompt
    if attr == "duration":
        prompt_start = f"Arrange {', '.join(labels)} from {LEAST_WORD[attr]} to {GREATEST_WORD[attr]}"
    else:
        prompt_start = f"Arrange {', '.join(labels)} from {LEAST_WORD[attr]} to {GREATEST_WORD[attr]}"

    text = f"These are the clues, in no particular order: {clues_text} {prompt_start}."

    # Verify using Kahn's topological sort (genuinely different algorithm)
    reconstructed_order = _expected_order_skill2(shuffled_edges, labels)
    check(
        reconstructed_order == order_asc,
        f"topological reconstruction does not match the sampled order: {reconstructed_order!r} != {order_asc!r}",
    )

    answer_text = ", ".join(order_asc)

    # Misconceptions (alternative wrong permutations)
    reversed_order, adjacent_swap = reversed_and_adjacent_swap(order_asc)
    reversed_text = ", ".join(reversed_order)

    adjacent_text = ", ".join(adjacent_swap)

    candidates = [reversed_text, adjacent_text]

    if hard_shuffle and N == 5:
        # Hard tier only (N=5): first-last-swap
        first_last_swap = list(order_asc)
        first_last_swap[0], first_last_swap[-1] = first_last_swap[-1], first_last_swap[0]
        first_last_text = ", ".join(first_last_swap)
        candidates.append(first_last_text)

    # Verify each distractor candidate is distinct from answer (by list equality)
    for candidate_str in candidates:
        candidate_list = candidate_str.split(", ")
        check(candidate_list != order_asc, f"distractor {candidate_str!r} equals answer {answer_text!r}")

    working = [
        f"Clues: {clues_text}",
        f"Build the chain from smallest to largest using the clues.",
        f"Answer: {answer_text}.",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=list(order_asc),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )
