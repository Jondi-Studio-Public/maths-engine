"""Outcomes/sample-space language + likelihood comparison, both senior ``acara-senior-v8-4``, domain
``statistics-probability``, ``answer: mcq``.

Two generators:

- ``outcomes_sample_classify`` (``senior-review-the-concepts-and-language-of-outcomes-sample``,
  ACMMM049) -- given a finite equally-likely sample space S = {1, 2, ..., n} and a set X
  (its members listed explicitly), classify X using the fixed four-option MCQ: "a single
  outcome", "the sample space", "an event (a set of outcomes)", or "not a valid subset of
  the sample space".

- ``probability_likelihood_compare`` (``senior-review-probability-as-a-measure-of-the-likelihood``,
  ACMMM052) -- given two named events A and B with stated probabilities P(A) and P(B),
  determine which is more likely (or whether they are equally likely) using the fixed
  three-option MCQ: "event A is more likely", "event B is more likely", or "A and B
  are equally likely".

Both use fixed, semantically mutually exclusive MCQ option strings.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, latex_answer, pick_distractors

# =============================================================================
# outcomes_sample_classify (Skill A)
# =============================================================================

OPTION_SINGLE = "a single outcome"
OPTION_SPACE = "the sample space"
OPTION_EVENT = "an event (a set of outcomes)"
OPTION_INVALID = "not a valid subset of the sample space"
OPTIONS_A = (OPTION_SINGLE, OPTION_SPACE, OPTION_EVENT, OPTION_INVALID)


def classify(n: int, x_list: list[int]) -> str:
    """Independent check: classify a set X (as a literal list) given sample space
    size n. This function recomputes the category from-scratch, structurally, off
    the presented list -- a genuinely different route from construction."""
    if any(v < 1 or v > n for v in x_list):
        return OPTION_INVALID
    distinct = sorted(set(x_list))
    if distinct == list(range(1, n + 1)):
        return OPTION_SPACE
    if len(distinct) == 1:
        return OPTION_SINGLE
    return OPTION_EVENT


@generator(
    "outcomes_sample_classify",
    level=1,
    presets={
        "easy": dict(difficulty_tier="easy"),
        "medium": dict(difficulty_tier="medium"),
        "hard": dict(difficulty_tier="hard"),
    },
)
def outcomes_sample_classify(rng, difficulty_tier="easy"):
    """Classify a given set X as a single outcome, the sample space, an event,
    or an invalid subset of a finite equally-likely sample space S = {1, 2, ..., n}."""
    n = rng.randint(4, 8)

    if difficulty_tier == "easy":
        # Easy: rng.choice between OPTION_SINGLE or OPTION_SPACE
        if rng.choice([True, False]):
            # OPTION_SINGLE: x_list = [rng.randint(1, n)]
            x_list = [rng.randint(1, n)]
            intended_label = OPTION_SINGLE
        else:
            # OPTION_SPACE: x_list = list(range(1, n + 1))
            x_list = list(range(1, n + 1))
            intended_label = OPTION_SPACE

    elif difficulty_tier == "medium":
        # Medium: always OPTION_EVENT, either even or greater-than rule
        # Ensure result is a proper, nonempty subset
        intended_label = OPTION_EVENT
        use_even = rng.choice([True, False])

        if use_even:
            # Even rule - always works for n in [4, 8]
            x_list = [v for v in range(1, n + 1) if v % 2 == 0]
        else:
            # Greater than k - resample if result is empty, single element, or equals whole range
            max_k_attempts = 20
            for k_attempt in range(max_k_attempts):
                k = rng.randint(1, n - 1)
                x_list = [v for v in range(1, n + 1) if v > k]
                # Proper subset: at least 2 elements and not the whole range
                if len(x_list) >= 2 and x_list != list(range(1, n + 1)):
                    break
            else:
                # Fallback: use even rule
                x_list = [v for v in range(1, n + 1) if v % 2 == 0]

    else:  # hard
        # Hard: rng.choice (equal weight) between three branches
        branch = rng.choice(["event_via_rule", "space_via_trap_rule", "invalid_list"])

        if branch == "event_via_rule":
            # Identical to medium's rule branch
            intended_label = OPTION_EVENT
            use_even = rng.choice([True, False])

            if use_even:
                x_list = [v for v in range(1, n + 1) if v % 2 == 0]
            else:
                # Greater than k
                max_k_attempts = 20
                for k_attempt in range(max_k_attempts):
                    k = rng.randint(1, n - 1)
                    x_list = [v for v in range(1, n + 1) if v > k]
                    # Proper subset: at least 2 elements and not the whole range
                    if len(x_list) >= 2 and x_list != list(range(1, n + 1)):
                        break
                else:
                    x_list = [v for v in range(1, n + 1) if v % 2 == 0]

        elif branch == "space_via_trap_rule":
            # State rule "greater than 0", which evaluates to entire space
            x_list = list(range(1, n + 1))
            intended_label = OPTION_SPACE

        else:  # invalid_list
            # Build proper subset (2 to n-1 members), then append out-of-range value
            subset_size = rng.randint(2, min(n - 1, n - 1))
            subset = rng.sample(range(1, n + 1), subset_size)
            out_of_range = n + rng.randint(1, 3)
            x_list = subset + [out_of_range]
            x_list.sort()
            intended_label = OPTION_INVALID

    check(classify(n, x_list) == intended_label, "classification mismatch")

    # Format x_list for display
    x_display = "{" + ", ".join(str(v) for v in sorted(set(x_list))) + "}"

    if difficulty_tier == "medium" or (difficulty_tier == "hard" and branch == "event_via_rule"):
        if all(v % 2 == 0 for v in x_list):
            rule_text = f"Consider X, the outcomes in S that are even: X = {x_display}."
        else:
            k = min(v - 1 for v in x_list)
            rule_text = f"Consider X, the outcomes in S greater than {k}: X = {x_display}."
    elif difficulty_tier == "hard" and branch == "space_via_trap_rule":
        rule_text = f"Consider X, the outcomes in S greater than 0: X = {x_display}."
    else:
        rule_text = ""

    text = f"The sample space S = {{1, 2, ..., {n}}}. Set X = {x_display}. Which of the following best describes X?"
    if rule_text:
        text = rule_text + " " + text

    # Distractors: pick_distractors returns exactly the three options other than the label
    candidates = [o for o in OPTIONS_A if o != intended_label]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=intended_label,
        answer_text=intended_label,
        answer_latex=latex_answer(intended_label, plain=True),
        working=[],
        distractors=pick_distractors(intended_label, candidates, n=3),
    )


# =============================================================================
# probability_likelihood_compare (Skill B)
# =============================================================================

OPTION_A_MORE = "event A is more likely"
OPTION_B_MORE = "event B is more likely"
OPTION_EQUAL = "A and B are equally likely"
OPTIONS_B = (OPTION_A_MORE, OPTION_B_MORE, OPTION_EQUAL)


def classify_likelihood(num_a: int, den_a: int, num_b: int, den_b: int) -> str:
    """Independent check: compare probabilities using cross-multiplication on
    raw integers, never relying on Python's Fraction comparison."""
    lhs = num_a * den_b
    rhs = num_b * den_a
    if lhs > rhs:
        return OPTION_A_MORE
    if lhs < rhs:
        return OPTION_B_MORE
    return OPTION_EQUAL


@generator(
    "probability_likelihood_compare",
    level=1,
    presets={
        "easy": dict(difficulty_tier="easy"),
        "medium": dict(difficulty_tier="medium"),
        "hard": dict(difficulty_tier="hard"),
    },
)
def probability_likelihood_compare(rng, difficulty_tier="easy"):
    """Compare the likelihood of two events given their probabilities as fractions
    or counts, determining which is more likely or if they are equally likely."""

    if difficulty_tier == "easy":
        # Easy: shared denominator, distinct numerators (never equal)
        d = rng.choice((4, 5, 6, 8, 10))
        num_a, num_b = rng.sample(range(1, d), 2)
        den_a, den_b = d, d

    elif difficulty_tier == "medium":
        # Medium: two different denominators, allow equal naturally
        d_a, d_b = rng.sample((5, 6, 8, 9, 10, 12), 2)
        num_a = rng.randint(1, d_a - 1)
        num_b = rng.randint(1, d_b - 1)
        den_a, den_b = d_a, d_b

    else:  # hard
        # Hard: event A computed from count, event B given as fraction
        y = rng.randint(6, 20)
        x = rng.randint(1, y - 1)
        num_a, den_a = x, y

        den_b = rng.choice((4, 5, 6, 8, 10, 12))
        num_b = rng.randint(1, den_b - 1)

    intended_label = classify_likelihood(num_a, den_a, num_b, den_b)
    check(intended_label in OPTIONS_B, "classification must return one of the three options")

    # Build question text
    if difficulty_tier == "hard":
        times_word = "time" if x == 1 else "times"
        text = (
            f"Event A occurs {x} {times_word} out of {y} equally likely trials. "
            f"Event B has probability {num_b}/{den_b}. Which of the following is true?"
        )
    else:
        text = (
            f"Event A has probability {num_a}/{den_a}. Event B has probability {num_b}/{den_b}. "
            f"Which of the following is true?"
        )

    working = []
    if difficulty_tier == "hard":
        times_word = "time" if x == 1 else "times"
        working.append(f"Event A: P(A) = {x}/{y} (occurs {x} {times_word} out of {y})")
    working.append(f"Event B: P(B) = {num_b}/{den_b}")

    if difficulty_tier == "hard":
        working.append(f"Compare {x}/{y} and {num_b}/{den_b} using cross-multiplication:")
        lhs = x * den_b
        rhs = num_b * y
        working.append(f"{x} × {den_b} = {lhs}, {num_b} × {y} = {rhs}")
        if lhs > rhs:
            working.append("A is more likely.")
        elif lhs < rhs:
            working.append("B is more likely.")
        else:
            working.append("They are equally likely.")
    else:
        lhs = num_a * den_b
        rhs = num_b * den_a
        working.append(f"Compare using cross-multiplication: {num_a} × {den_b} = {lhs}, {num_b} × {den_a} = {rhs}")
        if lhs > rhs:
            working.append("A is more likely.")
        elif lhs < rhs:
            working.append("B is more likely.")
        else:
            working.append("They are equally likely.")

    # Distractors: pick the other two options from the three fixed options
    candidates = [o for o in OPTIONS_B if o != intended_label]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=intended_label,
        answer_text=intended_label,
        answer_latex=latex_answer(intended_label, plain=True),
        working=working,
        distractors=pick_distractors(intended_label, candidates, n=2),
    )
