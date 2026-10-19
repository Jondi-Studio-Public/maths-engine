"""Generates `angle_turn_right_reference` and `shape_features_classification` questions: express a turn as a
number of right angles, and classify a shape from its features.

Angle-turn-as-right-angles + shape classification.

Two genuinely new Foundation-Year 10 geometry/measurement skills:

- ``angle_turn_right_reference`` (AC9M2M05, AC9M3M05) -- express a stated
  everyday turn (quarter, half, three-quarter or full) as a number of right
  angles. Response family: ``scalar_expression`` (``answer: number``).

- ``shape_features_classification`` (AC9MFSP01, AC9M1SP01, AC9M2SP01, AC9M3SP01)
  -- state a named familiar shape's number of sides, or whether it is curved or
  straight-sided. Response family: ``scalar_expression`` (``answer: mcq``).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, latex_answer, pick_distractors

# -- Skill 1: angle_turn_right_reference -----------

TURN_VALUES = {
    "quarter turn": 1,
    "half turn": 2,
    "three-quarter turn": 3,
    "full turn": 4,
}


_SUBJECTS = ("A shape", "A spinner", "A turntable", "A robot", "A windmill sail", "A toy car")
# The closing question, by what is asked: one turn, two turns in total, or the second of two turns.
_ASKS = (
    {
        "single": "How many right angles is this turn equal to?",
        "total": "In total, how many right angles has it turned through?",
        "second": "How many right angles did the second turn add?",
    },
    {
        "single": "How many right angles is that?",
        "total": "How many right angles is that altogether?",
        "second": "How many right angles was the second turn?",
    },
    {
        "single": "What is the turn in right angles?",
        "total": "What is the total turn in right angles?",
        "second": "What was the second turn in right angles?",
    },
)


@generator(
    "angle_turn_right_reference",
    level=1,
    presets={
        "easy": dict(num_turns=1),
        "medium": dict(num_turns=2),
        "hard": dict(num_turns=2, inverse=True),
    },
)
def angle_turn_right_reference(rng, num_turns=1, inverse=False, varied=False):
    """Express a stated everyday turn as a number of right angles.

    - easy: sample one turn, report its right-angle count directly.
    - medium: sample two turns independently, sum their right-angle counts.
    - hard: inverse -- sample label1 and a total T such that T - value(label1)
      is a valid turn value in {1,2,3,4}, then ask for the second turn's
      contribution to T.

    ``varied=True`` tells the same question about a spinner, a dancer or a robot (and words the last
    sentence three ways) so a tier holds dozens of different prompts; the numbers and the working are
    unchanged. It draws from ``rng`` only after the original draws, so ``varied=False`` is byte-identical.
    """
    turn_labels = list(TURN_VALUES.keys())

    if inverse:
        # Hard tier inverse: sample label1 and label2, compute total T,
        # then phrase around label1 and T, asking for the second turn.
        label1 = rng.choice(turn_labels)
        label2 = rng.choice(turn_labels)
        value1 = TURN_VALUES[label1]
        value2 = TURN_VALUES[label2]
        T = value1 + value2
        answer = value2
        text = (
            f"A shape makes a {label1}. It then keeps turning until it has "
            f"turned through {T} right angles in total. How many right angles "
            f"did the second turn add?"
        )
    elif num_turns == 2:
        # Medium tier: two independent turns, sum them
        label1 = rng.choice(turn_labels)
        label2 = rng.choice(turn_labels)
        value1 = TURN_VALUES[label1]
        value2 = TURN_VALUES[label2]
        answer = value1 + value2
        text = (
            f"A shape makes a {label1}, then makes another {label2}. "
            f"In total, how many right angles has it turned through?"
        )
    else:
        # Easy tier: single turn
        label = rng.choice(turn_labels)
        answer = TURN_VALUES[label]
        text = f"A shape makes a {label}. How many right angles is this turn equal to?"

    if varied:
        # Same numbers and answer, told about a different mover and asked a different way.
        subject = rng.choice(_SUBJECTS)
        ask = rng.choice(_ASKS)
        lead = text.rpartition(". ")[0].replace("A shape", subject, 1) + ". "
        tail = "total" if num_turns == 2 and not inverse else "second" if inverse else "single"
        text = lead + ask[tail]

    # Helper: extract Fraction for a turn label (check three-quarter first to avoid
    # matching "quarter" inside "three-quarter")
    def turn_to_fraction(label):
        if "three-quarter" in label:
            return Fraction(3, 4)
        elif "quarter" in label:
            return Fraction(1, 4)
        elif "half" in label:
            return Fraction(1, 2)
        else:
            return Fraction(1, 1)

    # Independent check: express each turn as a Fraction of a full turn,
    # multiply by 4, verify against dict values, then re-derive the answer
    # via fraction arithmetic.
    if inverse:
        turn1_frac = turn_to_fraction(label1)
        check(value1 == int(turn1_frac * 4), f"turn {label1}: dict {value1} != fraction {int(turn1_frac * 4)}")
        # Total T was computed as value1 + value2, so verify via fractions
        turn2_frac = turn_to_fraction(label2)
        total_frac = (turn1_frac + turn2_frac) * 4
        check(T == int(total_frac), f"total via fractions: {int(total_frac)} != dict-computed {T}")
        check(
            answer == int(total_frac - turn1_frac * 4),
            f"answer via fractions: {int(total_frac - turn1_frac * 4)} != {answer}",
        )
    elif num_turns == 2:
        turn1_frac = turn_to_fraction(label1)
        turn2_frac = turn_to_fraction(label2)
        check(value1 == int(turn1_frac * 4), f"turn {label1}: dict {value1} != fraction {int(turn1_frac * 4)}")
        check(value2 == int(turn2_frac * 4), f"turn {label2}: dict {value2} != fraction {int(turn2_frac * 4)}")
        total_frac = (turn1_frac + turn2_frac) * 4
        check(answer == int(total_frac), f"total via fractions: {int(total_frac)} != dict-computed {answer}")
    else:
        # Easy: single turn via fraction
        turn_frac = turn_to_fraction(label)
        check(answer == int(turn_frac * 4), f"turn {label}: dict {answer} != fraction {int(turn_frac * 4)}")

    # Build distractors: always include off-by-one-under and off-by-one-over
    off_by_one_under = answer - 1
    off_by_one_over = answer + 1

    # Confuses-half-and-quarter distractor (only if applicable/exercisable)
    # Helper: swap quarter<->half, return the swapped value (1 or 2)
    def swap_quarter_half(label):
        if "quarter" in label and "three-quarter" not in label:
            return 2  # quarter (1) -> half (2)
        elif "half" in label:
            return 1  # half (2) -> quarter (1)
        return None

    confuses_distractor = None
    if inverse:
        swapped = swap_quarter_half(label1)
        if swapped is not None:
            confuses_distractor = T - swapped
    elif num_turns == 2:
        swapped1 = swap_quarter_half(label1)
        swapped2 = swap_quarter_half(label2)
        if swapped1 is not None or swapped2 is not None:
            confuses_distractor = (swapped1 if swapped1 is not None else value1) + (
                swapped2 if swapped2 is not None else value2
            )
    else:
        # Easy: single turn
        confuses_distractor = swap_quarter_half(label)

    # Forgets-to-subtract distractor (hard tier only)
    forgets_distractor = None
    if inverse:
        forgets_distractor = T

    # Build candidate list
    candidates = [str(off_by_one_under), str(off_by_one_over)]
    if confuses_distractor is not None:
        candidates.append(str(confuses_distractor))
    if forgets_distractor is not None:
        candidates.append(str(forgets_distractor))

    # Verify backbone pair is always distinct
    check(
        off_by_one_under != answer and off_by_one_over != answer and off_by_one_under != off_by_one_over,
        f"backbone pair not distinct: answer={answer}, under={off_by_one_under}, over={off_by_one_over}",
    )

    working = [
        f"Turn: {label if num_turns == 1 else f'{label1}, {label2}' if not inverse else f'{label1} + unknown'}",
        f"Right angles: {answer}",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(str(answer), candidates, n=3),
    )


# -- Skill 2: shape_features_classification --------

# Fixed shape vocabulary
SHAPES_CURVED = ["circle", "oval"]
SHAPES_STRAIGHT_ALL = ["triangle", "square", "pentagon", "hexagon", "octagon"]
SHAPES_STRAIGHT_UNIQUE = ["triangle", "pentagon", "hexagon", "octagon"]

SHAPE_SIDES = {
    "circle": 0,
    "oval": 0,
    "triangle": 3,
    "square": 4,
    "pentagon": 5,
    "hexagon": 6,
    "octagon": 8,
}


def _article(shape: str) -> str:
    """ "a" or "an", by the shape name's own leading sound -- "an oval",
    "an octagon", but "a circle"/"a triangle"/etc. Every name in this
    module's fixed shape vocabulary is a plain noun with no silent leading
    letter, so a leading-vowel check is exact here (not a general English
    a/an rule)."""
    return "an" if shape[0] in "aeiou" else "a"


@generator(
    "shape_features_classification",
    level=1,
    presets={
        "easy": dict(mode="curved_straight"),
        "medium": dict(mode="forward_sides"),
        "hard": dict(mode="inverse_sides"),
    },
)
def shape_features_classification(rng, mode="forward_sides"):
    """Classify a named shape by feature (curved/straight or side count).

    - easy: binary curved-vs-straight for one shape from {circle, oval, triangle, square}
    - medium: forward side-count lookup for one shape from {triangle, square, pentagon, hexagon, octagon}
    - hard: inverse -- given a count from {3, 5, 6, 8}, name the shape from {triangle, pentagon, hexagon, octagon}
    """

    if mode == "which_curved_straight":
        # Easy, four options: pick the one shape that has a curved side, or the one with only straight sides.
        curved = ["circle", "oval", "semicircle"]
        straight = ["triangle", "square", "pentagon", "hexagon", "octagon"]
        want_curved = rng.random() < 0.5
        answer = rng.choice(curved if want_curved else straight)
        wrong = rng.sample(straight if want_curved else curved, 3)
        # Second route: a shape has a curved side exactly when it is not a polygon with a side count.
        polygon_sides = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "octagon": 8}
        check(
            (answer in polygon_sides) != want_curved and all((w in polygon_sides) == want_curved for w in wrong),
            "curved/straight key or distractor wrong",
        )
        text = rng.choice(
            (
                "Which of these shapes has a curved side?",
                "Which one of these shapes has a curved edge?",
                "Which of these shapes is not made only of straight lines?",
            )
            if want_curved
            else (
                "Which of these shapes has only straight sides?",
                "Which one of these shapes has no curved edge?",
                "Which of these shapes is made only of straight lines?",
            )
        )
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer,
            answer_text=answer,
            answer_latex=latex_answer(answer, plain=True),
            working=[f"{answer.capitalize()} {'has a curved side' if want_curved else 'has only straight sides'}."],
            distractors=pick_distractors(answer, wrong, n=3),
        )

    if mode == "curved_straight":
        # Easy: curved-vs-straight binary
        shape = rng.choice(SHAPES_CURVED + ["triangle", "square"])
        is_curved = shape in SHAPES_CURVED
        answer = "curved" if is_curved else "straight-sided"
        text = f"Is {_article(shape)} {shape} curved or straight-sided?"
        distractors_list = ["straight-sided", "curved"]
    elif mode == "inverse_sides":
        # Hard: given count from unique pool, name the shape
        count = rng.choice([3, 5, 6, 8])
        shape = [s for s in SHAPES_STRAIGHT_UNIQUE if SHAPE_SIDES[s] == count][0]
        answer = shape
        text = f"Which of these shapes has {count} sides?"
        # Distractors: the other three shapes from unique pool
        distractors_list = [s for s in SHAPES_STRAIGHT_UNIQUE if s != shape]
    else:
        # Medium: forward side-count lookup
        shape = rng.choice(SHAPES_STRAIGHT_ALL)
        count = SHAPE_SIDES[shape]
        answer = str(count)
        text = f"How many sides does {_article(shape)} {shape} have?"
        # Distractors: the other four shapes' side counts in this same pool
        other_counts = [str(SHAPE_SIDES[s]) for s in SHAPES_STRAIGHT_ALL if s != shape]
        distractors_list = other_counts

    # Independent check via second dict, never referencing the main one
    shape_to_sides_check = {
        "triangle": 3,
        "square": 4,
        "pentagon": 5,
        "hexagon": 6,
        "octagon": 8,
    }
    curved_set_check = {"circle", "oval"}

    if mode == "curved_straight":
        # Verify curved/straight classification via second source
        is_curved_check = shape in curved_set_check
        answer_check = "curved" if is_curved_check else "straight-sided"
        check(answer == answer_check, f"curved/straight mismatch: {answer} != {answer_check}")
    elif mode == "inverse_sides":
        # Verify the answer shape has the correct side count
        check(shape_to_sides_check.get(answer) == count, f"inverse: shape {answer} does not have {count} sides")
        # Verify each distractor is a different shape from the unique pool
        unique_set = {"triangle", "pentagon", "hexagon", "octagon"}
        check(all(d in unique_set for d in distractors_list), f"distractors {distractors_list} not all in unique pool")
    else:
        # Medium: forward lookup
        check(shape_to_sides_check.get(shape) == int(answer), f"forward: shape {shape} does not have {answer} sides")
        # Verify each distractor count is a real count from SHAPES_STRAIGHT_ALL
        for distractor in distractors_list:
            distractor_int = int(distractor)
            check(
                any(SHAPE_SIDES[s] == distractor_int for s in SHAPES_STRAIGHT_ALL),
                f"distractor {distractor} not a real side count in the pool",
            )

    if mode == "curved_straight":
        working = [
            f"{_article(shape).capitalize()} {shape} has {'a curved edge' if answer == 'curved' else 'straight sides'}, so it is {answer}."
        ]
    elif mode == "inverse_sides":
        working = [f"The shape with {count} sides is {_article(answer)} {answer}."]
    else:
        working = [f"{_article(shape).capitalize()} {shape} has {answer} sides."]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=pick_distractors(answer, distractors_list, n=3),
    )
