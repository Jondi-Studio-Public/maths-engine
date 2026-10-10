"""Senior causality-vs-correlation + interpolation-vs-extrapolation skills. Both skills are binary
classifications: causality vs correlation, and interpolation vs extrapolation. Each has exactly four fixed MCQ
options: two real, contract-named categories, plus two structurally-always-wrong padding options.

- ``causality_correlation_classify`` (senior-distinguish-between-causality-and-correlation-through-examples,
  ACMEM147, Essential Mathematics) -- Given a study description with one marker (random assignment
  for causation, or a named confounding variable for correlation), classify whether the evidence
  supports causation, correlation only, or neither. The "both" and "neither" options are always wrong
  by design: vignettes state exactly one marker (never both), and always state their marker explicitly.

- ``interpolation_extrapolation_classify`` (senior-distinguish-between-interpolation-and-extrapolation-when-using-the,
  ACMGM062, General Mathematics) -- Given a fitted-line model and a prediction point, classify whether
  the prediction lies within the data range (interpolation) or outside it (extrapolation). The "both"
  and "cannot determine" options are always wrong: a point is either inside or outside the inclusive
  data range (strict binary), and both the range and prediction point are always stated.
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, pick_distractors

# ============================================================================
# Skill A: causality_correlation_classify
# ============================================================================

# Fixed MCQ options for Skill A (verbatim, same order every seed)
OPTION_CAUSATION = "Causation is supported (a controlled experiment with random assignment)"
OPTION_CORRELATION = "Correlation only (an observational study; a named third variable may explain the association)"
OPTION_BOTH = "Both causation and correlation are supported equally"
OPTION_NEITHER = "Neither can be determined from the information given"
OPTIONS_A = (OPTION_CAUSATION, OPTION_CORRELATION, OPTION_BOTH, OPTION_NEITHER)

# Pairs a researcher can plausibly randomly assign to participants --
# used ONLY by the causation (random-assignment) branch.
ASSIGNABLE_PAIRS = (
    ("hours of study", "exam score"),
    ("hours of sleep", "reaction time"),
    ("hours of exercise per week", "resting heart rate"),
    ("dosage of a new medication", "recovery time in days"),
    ("amount of daily practice", "typing speed"),
    ("class size", "average test score"),
)

# Pairs that can only be observed, never randomly assigned -- used ONLY by the
# correlation (observational) branch. Each pair carries the third variable that
# actually explains its association; the confound is not separable from the pair
# and there is no general pool to fall back on. Drawing the confound
# independently is how 63 of 240 observational questions came to name a third
# variable that did not explain the association.
#
# Every pair here must be genuinely spurious: the association is real but
# neither variable causes the other. "outdoor temperature ~ icy-pole sales" was
# removed because temperature does cause icy-pole sales, so classifying it
# "correlation only" teaches the opposite of the lesson.
OBSERVATIONAL_PAIRS = (
    ("the number of fire trucks at a fire", "the amount of damage caused", "the size of the fire"),
    ("ice-cream sales", "the number of people who go swimming", "temperature"),
    ("shoe size", "reading ability", "the child's age"),
    ("household heater use", "the number of colds reported", "the season"),
)

# Fixed r values for flavour text (never load-bearing for classification)
R_VALUES = (0.6, 0.7, 0.8, 0.9)

# Marker templates (must use exact wording for independent check)
CAUSATION_MARKER = "Researchers randomly assigned participants to different levels of {X}, then measured {Y}."
CORRELATION_MARKER = "Researchers observed {X} and {Y} in a sample without controlling other factors. A third variable, {Z}, was also associated with both {X} and {Y}."

# Branch labels for internal logic
CAUSATION_BRANCH = "causation"
CORRELATION_BRANCH = "correlation"


def _independent_check_skill_a(text: str, question_type: str, label: str) -> None:
    """Independent check: recompute the label from rendered text only, by keyword
    scanning the addressed segment. For hard tier, segment is split on "Study 2:"
    or "Study 1:" markers. Raises VerificationError if the re-derived label does
    not match construction's intended label.
    """
    if question_type in ("easy", "medium"):
        # Single vignette: scan the whole text
        segment = text
    else:  # hard
        # Two vignettes: extract the addressed study segment
        # Split on "Study 2:" to find where Study 2 begins, if asked about Study 2
        if "Study 2:" in text:
            parts = text.split("Study 2:")
            if len(parts) == 2 and "Which of these is correct about Study 2" in text:
                segment = "Study 2:" + parts[1]
            elif "Which of these is correct about Study 1" in text:
                segment = parts[0]  # Study 1 is before "Study 2:"
            else:
                segment = text
        else:
            segment = text

    # Keyword scan: exactly one marker should be present in the addressed segment
    has_random = "randomly assigned" in segment
    has_confound = "third variable" in segment

    # XOR: exactly one must be true (never both, never neither)
    check(
        (has_random and not has_confound) or (has_confound and not has_random),
        f"segment must have exactly one marker (random or third variable); "
        f"has_random={has_random}, has_confound={has_confound}",
    )

    if has_random:
        derived_label = CAUSATION_BRANCH
    else:  # has_confound
        derived_label = CORRELATION_BRANCH

    check(
        derived_label == label, f"independent check mismatch: derived {derived_label!r} != construction label {label!r}"
    )


@generator(
    "causality_correlation_classify",
    level=11,
    presets={
        "easy": dict(question_type="easy"),
        "medium": dict(question_type="medium"),
        "hard": dict(question_type="hard"),
    },
)
def causality_correlation_classify(rng, question_type="easy"):
    """Classify a vignette (or two vignettes for hard) by whether the evidence
    supports causation (random assignment), correlation only (named confound),
    or neither.
    """
    if question_type not in ("easy", "medium", "hard"):
        raise ValueError(f"question_type must be 'easy', 'medium', or 'hard'; got {question_type!r}")

    if question_type == "easy":
        # Single vignette, causation marker, one variable pair, flavour r sentence
        label = CAUSATION_BRANCH
        X, Y = rng.choice(ASSIGNABLE_PAIRS)
        marker = CAUSATION_MARKER.format(X=X, Y=Y)
        r = rng.choice(R_VALUES)
        text = f"{marker} The correlation coefficient was approximately {r}. Which of these is correct?"

    elif question_type == "medium":
        # Single vignette, correlation marker, different variable pair, flavour r sentence
        label = CORRELATION_BRANCH
        X, Y, Z = rng.choice(OBSERVATIONAL_PAIRS)
        marker = CORRELATION_MARKER.format(X=X, Y=Y, Z=Z)
        r = rng.choice(R_VALUES)
        text = f"{marker} The observed correlation coefficient was approximately {r}. Which of these is correct?"

    else:  # hard
        # Two vignettes, one with causation marker, one with correlation marker
        # Randomise which study gets which marker
        causation_first = rng.choice([True, False])

        # Draw one pair from each pool to ensure no observational pair is in causation slot
        Xa, Ya = rng.choice(ASSIGNABLE_PAIRS)
        Xo, Yo, Z = rng.choice(OBSERVATIONAL_PAIRS)

        r1 = rng.choice(R_VALUES)
        r2 = rng.choice(R_VALUES)

        if causation_first:
            study1_marker = CAUSATION_MARKER.format(X=Xa, Y=Ya)
            study2_marker = CORRELATION_MARKER.format(X=Xo, Y=Yo, Z=Z)
            study1_text = f"{study1_marker} The correlation coefficient was approximately {r1}."
            study2_text = f"{study2_marker} The observed correlation coefficient was approximately {r2}."
        else:
            study1_marker = CORRELATION_MARKER.format(X=Xo, Y=Yo, Z=Z)
            study2_marker = CAUSATION_MARKER.format(X=Xa, Y=Ya)
            study1_text = f"{study1_marker} The observed correlation coefficient was approximately {r1}."
            study2_text = f"{study2_marker} The correlation coefficient was approximately {r2}."

        # Randomise which study is asked about
        ask_study_1 = rng.choice([True, False])

        # Joined with spaces, not newlines: the front end renders question text
        # into a <span> with no white-space rule and no newline-to-<br> step, so
        # a newline collapses and the learner never sees the break the author
        # intended. Each part
        # ends with a full stop, so the joined prose reads as the learner gets it.
        full_text = f"Study 1: {study1_text} Study 2: {study2_text} "

        if ask_study_1:
            label = CAUSATION_BRANCH if causation_first else CORRELATION_BRANCH
            full_text += "Which of these is correct about Study 1?"
        else:
            label = CORRELATION_BRANCH if causation_first else CAUSATION_BRANCH
            full_text += "Which of these is correct about Study 2?"

        text = full_text

    # Determine the correct answer from the label
    if label == CAUSATION_BRANCH:
        answer = OPTION_CAUSATION
    else:  # CORRELATION_BRANCH
        answer = OPTION_CORRELATION

    # Independent check: re-derive the label from rendered text
    _independent_check_skill_a(text, question_type, label)

    # Verify that OPTION_BOTH and OPTION_NEITHER are never the intended answer
    check(
        answer not in (OPTION_BOTH, OPTION_NEITHER), f"padding options must never be the correct answer; got {answer!r}"
    )

    # Distractors: pick the other two padding options, always wrong
    # Since answer is one of (CAUSATION, CORRELATION), distractors are (BOTH, NEITHER)
    distractors = [OPTION_BOTH, OPTION_NEITHER]

    working = [
        f"Which marker is present? {'Randomly assigned' if label == CAUSATION_BRANCH else 'Third variable'} indicates {label}."
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Skill B: interpolation_extrapolation_classify
# ============================================================================

# Fixed MCQ options for Skill B (verbatim, same order every seed)
OPTION_INTERPOLATION = "Interpolation (the prediction lies within the range of the collected data)"
OPTION_EXTRAPOLATION = "Extrapolation (the prediction lies outside the range of the collected data)"
OPTION_BOTH_B = "Both interpolation and extrapolation apply"
OPTION_NEITHER_B = "Cannot be determined from the information given"
OPTIONS_B = (OPTION_INTERPOLATION, OPTION_EXTRAPOLATION, OPTION_BOTH_B, OPTION_NEITHER_B)

# Branch labels for internal logic
INTERPOLATION_BRANCH = "interpolation"
EXTRAPOLATION_BRANCH = "extrapolation"


def _independent_check_skill_b(x_values: list, x0: float, label: str) -> None:
    """Independent check: recompute xmin and xmax from the table's own x-values
    and classify x0 against the inclusive range [xmin, xmax]. Raises VerificationError
    if the re-derived label does not match construction's intended label.
    """
    xmin = min(x_values)
    xmax = max(x_values)

    # Inclusive boundary: x0 at xmin or xmax is interpolation
    if xmin <= x0 <= xmax:
        derived_label = INTERPOLATION_BRANCH
    else:
        derived_label = EXTRAPOLATION_BRANCH

    check(
        derived_label == label,
        f"independent check mismatch: derived {derived_label!r} != construction label {label!r} "
        f"(xmin={xmin}, xmax={xmax}, x0={x0})",
    )


@generator(
    "interpolation_extrapolation_classify",
    level=11,
    presets={
        "easy": dict(question_type="easy"),
        "medium": dict(question_type="medium"),
        "hard": dict(question_type="hard"),
    },
)
def interpolation_extrapolation_classify(rng, question_type="easy"):
    """Classify a prediction as interpolation (within data range) or extrapolation
    (outside data range).
    """
    if question_type not in ("easy", "medium", "hard"):
        raise ValueError(f"question_type must be 'easy', 'medium', or 'hard'; got {question_type!r}")

    if question_type == "easy":
        # Explicit x-range, prediction clearly inside or clearly outside
        xmin, xmax = rng.randint(1, 10), rng.randint(20, 30)

        # Randomise inside vs outside (50/50)
        is_inside = rng.choice([True, False])

        if is_inside:
            label = INTERPOLATION_BRANCH
            # x0 well inside: not at boundary
            x0 = rng.randint(xmin + 2, xmax - 2)
        else:
            label = EXTRAPOLATION_BRANCH
            # x0 well outside: either below xmin or above xmax
            if rng.choice([True, False]):
                x0 = rng.randint(xmax + 5, xmax + 15)  # well above
            else:
                x0 = rng.randint(xmin - 15, xmin - 5)  # well below

        text = f"A fitted line models data collected for x from {xmin} to {xmax}. A prediction is made at x = {x0}. Which of these is correct?"

    elif question_type == "medium":
        # Explicit x-range, prediction at a boundary (x0 == xmin or x0 == xmax)
        # Always interpolation by contract
        label = INTERPOLATION_BRANCH
        xmin, xmax = rng.randint(1, 10), rng.randint(20, 30)

        # x0 is exactly at one boundary
        x0 = rng.choice([xmin, xmax])

        text = f"A fitted line models data collected for x from {xmin} to {xmax}. A prediction is made at x = {x0}. Which of these is correct?"

    else:  # hard
        # No explicit range; give a table of 5-6 data points
        # Learner must scan the table to find min/max, then classify x0
        table_size = rng.randint(5, 6)

        # Generate x values: at least two distinct values
        x_values = sorted(list(set(rng.randint(1, 50) for _ in range(table_size))))
        if len(x_values) < 2:
            x_values = [rng.randint(1, 20), rng.randint(21, 50)]

        # Shuffle the table rows so x is not in sorted order
        table_rows = [(x, rng.randint(1, 100)) for x in x_values]
        rng.shuffle(table_rows)

        xmin = min(x_values)
        xmax = max(x_values)

        # Randomise x0: inside, outside, or exactly at boundary
        # All three should be testable
        x0_position = rng.choice(["inside", "outside", "boundary"])

        if x0_position == "inside":
            label = INTERPOLATION_BRANCH
            x0 = rng.randint(xmin + 1, xmax - 1)
        elif x0_position == "boundary":
            label = INTERPOLATION_BRANCH
            x0 = rng.choice([xmin, xmax])
        else:  # outside
            label = EXTRAPOLATION_BRANCH
            if rng.choice([True, False]):
                x0 = rng.randint(xmax + 1, xmax + 20)
            else:
                x0 = rng.randint(xmin - 20, xmin - 1)

        # Build table text as a single line with comma-separated ordered pairs
        table_str = ", ".join(f"({x_val}, {y_val})" for x_val, y_val in table_rows)
        text = f"Data table: {table_str}. A fitted line is used to make a prediction at x = {x0}. Which of these is correct?"

    # Determine the correct answer from the label
    if label == INTERPOLATION_BRANCH:
        answer = OPTION_INTERPOLATION
    else:  # EXTRAPOLATION_BRANCH
        answer = OPTION_EXTRAPOLATION

    # Independent check: recompute from first principles
    if question_type == "easy":
        x_check = [xmin, xmax]
    elif question_type == "medium":
        x_check = [xmin, xmax]
    else:  # hard
        x_check = x_values

    _independent_check_skill_b(x_check, x0, label)

    # Verify that OPTION_BOTH_B and OPTION_NEITHER_B are never the intended answer
    check(
        answer not in (OPTION_BOTH_B, OPTION_NEITHER_B),
        f"padding options must never be the correct answer; got {answer!r}",
    )

    # Distractors: pick the other two padding options, always wrong
    distractors = [OPTION_BOTH_B, OPTION_NEITHER_B]

    # Build working text
    if question_type == "hard":
        data_range = f"x = {xmin} to x = {xmax}"
    else:
        data_range = f"x = {xmin} to x = {xmax}"

    working = [
        f"The data spans {data_range}. The prediction is at x = {x0}, which is {'within' if label == INTERPOLATION_BRANCH else 'outside'} this range, so this is {label}."
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
    )
