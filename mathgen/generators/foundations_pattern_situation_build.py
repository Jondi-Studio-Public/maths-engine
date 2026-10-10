"""Early-foundations pair B: two more genuinely new cognitive operations in this domain, both
framework ``acara-f10-v9``.

- ``pattern_repeat_unit`` (AC9M1A02, prerequisite ``continue-patterns``) --
  given a repeating pattern of objects/shapes/symbols displayed for several
  full-plus-partial repetitions, state the smallest repeating unit, in
  order. This is a **different** graded output from the existing
  prerequisite ``continue-patterns`` (``mathgen/generators/early_math.py``,
  ``repeating_pattern``), which predicts a single *next* element via
  ``mcq``; here the whole unit must be stated, as an ``ordered_list``.
  Difficulty is driven by unit length: ``easy`` uses a 2-element unit (the
  classic case a learner can spot at a glance); ``medium`` uses a 3-element
  unit with three genuinely distinct elements, breaking the common
  "everything repeats every 2" assumption; ``hard`` uses a 3-element unit
  shaped ``[X, X, Y]`` (the first two elements repeat consecutively) --
  deliberately chosen so that a learner who assumes period 2 and reads only
  the first two displayed items lands on ``[X, X]``, a plausible-looking but
  wrong answer, not the true 3-element unit. The independent check
  (``_verify_minimal_period``) never reuses the ``unit`` variable the
  generator tiled to build the display: it brute-force scans candidate
  period lengths from 1 up to half the displayed sequence's length, finds
  the smallest length that tiles the whole sequence exactly, and asserts
  that length equals ``len(unit)`` before separately confirming the tiled
  reconstruction matches the display.

- ``practical_situation_sum`` (AC9MFN05, AC9M1N05, prerequisites
  ``add-within-10``, ``subtract-within-10``) -- given a short word problem
  describing a joining or separating scenario with quantities up to 20,
  determine the resulting quantity. Two scenario types exist,
  ``join`` (true operation: addition) and ``separate`` (true operation:
  subtraction) -- the operation is fixed entirely by this label, never
  derived from which words happen to appear in the rendered text. Each
  scenario type has three phrasing variants selecting the genuine
  reasoning demand (not just bigger numbers): ``direct`` (easy) uses an
  explicit keyword that directly cues the operation ("gets ... more",
  "gives away"); ``comparison`` (medium) uses a "more than"/"fewer than"
  framing that requires inferring the operation from a relationship between
  two people's quantities, not a keyword; ``trap`` (hard) is a genuine
  keyword trap, well documented in maths-education literature on word
  problems -- the join-trap scenario is worded with "gave away"/"left"
  (surface cues for subtraction) but is actually a start-unknown problem
  solved by addition; the separate-trap scenario is worded with "got some
  more"/"in total" (surface cues for addition) but is actually a
  missing-addend problem solved by subtraction. The independent check
  (``_verify_scenario_answer``) re-derives the answer from the two sampled
  quantities and the scenario-type label alone, via a freshly summed loop
  (``join``) or a fresh sort-and-difference (``separate``) -- never by
  re-reading whatever expression built the displayed answer.
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, pick_distractors

# -- pattern_repeat_unit -----------------------------------------------

# A small pool of shape-name words standing in for "objects, shapes, symbols
# or numbers". Deliberately typeable tokens, not emoji/glyph symbols: this
# skill's answer kind is `ordered_list` (a typed answer, matching
# `order_to_twenty`'s own precedent of typeable numerals and its explicit
# "plain typed answer grades correct" convention), unlike the prerequisite
# `continue-patterns`'s tap-based `mcq`, where an emoji/glyph choice among
# fixed options is fine because nothing has to be typed. `grade_ordered_list`
# compares string items case-insensitively, so "Triangle, Square" grades the
# same as "triangle, square". Also deliberately not `early_math.DOTS` or
# `repeating_pattern`'s own emoji pool, so this generator's output is never
# confused with either sibling generator's own vocabulary.
_ELEMENTS = ("triangle", "square", "circle", "star", "diamond")

_VALID_SHAPES = ("distinct", "trap")


def _build_unit(rng, unit_length: int, shape: str) -> list[str]:
    if shape == "distinct":
        return list(rng.sample(_ELEMENTS, unit_length))
    if shape == "trap":
        if unit_length != 3:
            raise ValueError("'trap' shape requires unit_length == 3 (first two elements repeat consecutively)")
        x, y = rng.sample(_ELEMENTS, 2)
        return [x, x, y]
    raise ValueError(f"unknown shape {shape!r}; expected one of {_VALID_SHAPES}")


def _true_minimal_period(sequence: list[str]) -> int:
    """Brute-force scan candidate period lengths from 1 up to half the
    sequence length, returning the smallest length that tiles the whole
    sequence exactly. Deliberately does not touch whatever `unit` variable
    the caller used to build `sequence` -- this re-derives the period from
    the rendered sequence alone."""
    n = len(sequence)
    for period in range(1, n // 2 + 1):
        if all(sequence[i] == sequence[i % period] for i in range(n)):
            return period
    return n


def _verify_minimal_period(sequence: list[str], unit: list[str]) -> None:
    """Independent check: the sequence's true minimal period (found without
    ever reading `unit`) must equal `len(unit)`, and tiling `unit` itself
    must exactly reconstruct the displayed sequence."""
    true_period = _true_minimal_period(sequence)
    check(true_period == len(unit), f"sequence's true minimal period is {true_period}, expected {len(unit)}")
    reconstructed = [unit[i % len(unit)] for i in range(len(sequence))]
    check(reconstructed == sequence, "tiling the stated unit does not reconstruct the displayed sequence")


@generator(
    "pattern_repeat_unit",
    level=1,
    presets={
        "easy": dict(unit_length=2, shape="distinct", full_repeats=(3, 4)),
        "medium": dict(unit_length=3, shape="distinct", full_repeats=(2, 3)),
        "hard": dict(unit_length=3, shape="trap", full_repeats=(2, 3)),
    },
)
def pattern_repeat_unit(rng, unit_length=2, shape="distinct", full_repeats=(2, 3)):
    """State the smallest repeating unit of a displayed pattern, in order.
    See the module docstring for the unit-length/trap-shape difficulty
    rationale."""
    if unit_length < 2:
        raise ValueError("unit_length must be at least 2")
    if unit_length > len(_ELEMENTS):
        raise ValueError(f"unit_length must be at most {len(_ELEMENTS)} (only that many distinct elements available)")
    if shape not in _VALID_SHAPES:
        raise ValueError(f"unknown shape {shape!r}; expected one of {_VALID_SHAPES}")

    unit = _build_unit(rng, unit_length, shape)

    if isinstance(full_repeats, (tuple, list)):
        if not full_repeats:
            raise ValueError("full_repeats sequence must not be empty")
        num_full = rng.randint(full_repeats[0], full_repeats[1])
    else:
        num_full = full_repeats
    if num_full < 2:
        raise ValueError("need at least 2 full repetitions to show a genuine repeating pattern")

    # A genuine partial final repetition -- never a whole extra repeat, and
    # never zero (the display must show more than whole repeats alone).
    partial_len = rng.randint(1, unit_length - 1)
    sequence = list(unit) * num_full + list(unit)[:partial_len]

    _verify_minimal_period(sequence, unit)

    display_text = ", ".join(sequence)
    answer_text = ", ".join(unit)

    working = [f"The pattern repeats every {unit_length} items."]
    if shape == "trap":
        working.append(
            "Do not assume the unit is only 2 items long -- check further along the "
            "pattern before deciding where it starts repeating again."
        )
    working.append(f"One full repeat, in order: {answer_text}.")

    # Misconception distractors:
    #  - assumes-period-two: reads only the first two displayed items as the
    #    unit, regardless of the pattern's true unit length. Most exposed at
    #    hard, where this equals [X, X] -- a plausible-looking but wrong
    #    answer for the true 3-element unit [X, X, Y].
    #  - wrong-offset-rotation: states a genuine rotation of the true unit,
    #    starting from the wrong item in the cycle.
    #  - unit-too-long: states an exact multiple of the true unit (here,
    #    double it) rather than the smallest repeating chunk.
    period_two_guess = sequence[0:2]
    rotation = unit[1:] + unit[:1]
    too_long = unit * 2
    candidates = [
        ", ".join(period_two_guess),
        ", ".join(rotation),
        ", ".join(too_long),
    ]

    return dict(
        text=f"This pattern repeats. What is the smallest repeating unit, in order? {display_text}, ...",
        latex=f"plain:{display_text}",
        answer=list(unit),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


# -- practical_situation_sum ---------------------------------------------

_ACTORS = (
    ("Mia", "She"),
    ("Sam", "He"),
    ("Liam", "He"),
    ("Ava", "She"),
    ("Noah", "He"),
    ("Zoe", "She"),
    ("Ben", "He"),
    ("Tia", "She"),
)
_ITEMS = ("stickers", "marbles", "apples", "pencils", "shells", "blocks")
_SCENARIO_TYPES = ("join", "separate")
_VARIANTS = ("direct", "comparison", "trap")


def _item_word(n: int, item: str) -> str:
    """Agree an ``_ITEMS`` noun (always stored plural) with a count: "1
    marble" but "2 marbles". Every item in ``_ITEMS`` is a regular plural, so
    dropping the trailing "s" always gives the singular."""
    return item[:-1] if n == 1 else item


def _sample_join_operands(rng) -> tuple[int, int]:
    """Both operands and their sum stay within 0-20 by construction: `a` is
    drawn from 1-15, and `b` is capped so `a + b <= 20`."""
    a = rng.randint(1, 15)
    b = rng.randint(1, 20 - a)
    return a, b


def _sample_separate_operands(rng) -> tuple[int, int]:
    """`larger` (the starting/total quantity) and `smaller` (the quantity
    removed/being compared) stay within 0-20 by construction, with
    `smaller < larger` so the difference is always non-negative."""
    larger = rng.randint(2, 20)
    smaller = rng.randint(1, larger - 1)
    return larger, smaller


def _verify_scenario_answer(scenario_type: str, x: int, y: int, answer: int) -> None:
    """Independent check: recompute the answer from the two sampled
    quantities and the scenario-type label alone, via a route distinct from
    whatever expression built the displayed answer -- a freshly summed loop
    for `join`, a fresh sort-and-subtract for `separate`. Never re-reads the
    same `a + b`/`a - b` subexpression that constructed `answer`."""
    if scenario_type == "join":
        total = 0
        for quantity in (x, y):
            total += quantity
        check(total == answer, f"re-summed total {total} does not match stated answer {answer}")
    elif scenario_type == "separate":
        ordered = sorted((x, y))
        difference = ordered[1] - ordered[0]
        check(difference == answer, f"re-derived difference {difference} does not match stated answer {answer}")
    else:
        raise ValueError(f"unknown scenario_type {scenario_type!r}; expected one of {_SCENARIO_TYPES}")


def _join_direct_text(name, pronoun, item, a, b):
    return (
        f"{name} has {a} {_item_word(a, item)}. {pronoun} gets {b} more {_item_word(b, item)}. "
        f"How many {item} does {name} have now?"
    )


def _join_comparison_text(name_a, name_b, item, a, b):
    return (
        f"{name_a} has {a} {_item_word(a, item)}. {name_b} has {b} more {_item_word(b, item)} than {name_a}. "
        f"How many {item} does {name_b} have?"
    )


def _join_trap_text(name, pronoun, item, a, b):
    return (
        f"{name} had some {item}. {pronoun} gave away {b} {_item_word(b, item)} and now has {a} {_item_word(a, item)} left. "
        f"How many {item} did {name} have to start with?"
    )


def _separate_direct_text(name, pronoun, item, larger, smaller):
    return (
        f"{name} has {larger} {_item_word(larger, item)}. {pronoun} gives away {smaller} {_item_word(smaller, item)}. "
        f"How many {item} does {name} have left?"
    )


def _separate_comparison_text(name_a, name_b, item, larger, smaller):
    return (
        f"{name_a} has {larger} {_item_word(larger, item)}. {name_b} has {smaller} fewer {_item_word(smaller, item)} than {name_a}. "
        f"How many {item} does {name_b} have?"
    )


def _separate_trap_text(name, pronoun, item, larger, smaller):
    return (
        f"{name} had {smaller} {_item_word(smaller, item)}. {pronoun} got some more and now has {larger} {_item_word(larger, item)} in total. "
        f"How many {item} did {name} get?"
    )


@generator(
    "practical_situation_sum",
    level=1,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="comparison"),
        "hard": dict(variant="trap"),
    },
)
def practical_situation_sum(rng, variant="direct"):
    """Determine the resulting quantity for a joining or separating word
    problem with quantities up to 20. See the module docstring for which
    operation each scenario type/variant genuinely requires."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {_VARIANTS}")

    scenario_type = rng.choice(_SCENARIO_TYPES)
    item = rng.choice(_ITEMS)

    if scenario_type == "join":
        a, b = _sample_join_operands(rng)
        answer = a + b
    else:
        a, b = _sample_separate_operands(rng)
        answer = a - b  # `a` is always the larger operand for `separate`.

    if not (0 <= a <= 20 and 0 <= b <= 20 and 0 <= answer <= 20):
        raise ValueError(f"sampled operands/result out of the 0-20 bound: a={a}, b={b}, answer={answer}")

    _verify_scenario_answer(scenario_type, a, b, answer)

    if variant == "comparison":
        (name_a, _), (name_b, _) = rng.sample(_ACTORS, 2)
        name = pronoun = None
    else:
        name, pronoun = rng.choice(_ACTORS)
        name_a = name_b = None

    if scenario_type == "join":
        if variant == "direct":
            text = _join_direct_text(name, pronoun, item, a, b)
            working = [f"Start with {a}, then add the {b} more: {a} + {b} = {answer}."]
        elif variant == "comparison":
            text = _join_comparison_text(name_a, name_b, item, a, b)
            working = [
                f"{name_b} has {b} more {_item_word(b, item)} than {name_a}'s {a}, so add: {a} + {b} = {answer}."
            ]
        else:  # trap
            text = _join_trap_text(name, pronoun, item, a, b)
            working = [
                "The words 'gave away' and 'left' sound like subtraction, but the "
                "starting amount is what's unknown here.",
                f"Add back what was given away: {a} + {b} = {answer}.",
            ]
    else:
        if variant == "direct":
            text = _separate_direct_text(name, pronoun, item, a, b)
            working = [f"Start with {a}, then take away {b}: {a} - {b} = {answer}."]
        elif variant == "comparison":
            text = _separate_comparison_text(name_a, name_b, item, a, b)
            working = [
                f"{name_b} has {b} fewer {_item_word(b, item)} than {name_a}'s {a}, so subtract: {a} - {b} = {answer}."
            ]
        else:  # trap
            text = _separate_trap_text(name, pronoun, item, a, b)
            working = [
                "The words 'got some more' and 'in total' sound like addition, but "
                "you need the difference between the two amounts here.",
                f"Subtract the starting amount from the new total: {a} - {b} = {answer}.",
            ]

    # Misconception distractors:
    #  - opposite-operation: applies the opposite operation to the scenario.
    #  - off-by-one-slip: a plain arithmetic slip of one.
    #  - ignores-one-quantity: answers with just one of the two stated
    #    quantities, forgetting the other entirely.
    opposite = abs(a - b) if scenario_type == "join" else a + b
    off_by_one = answer + 1 if answer < 20 else answer - 1
    ignores_quantity = a
    candidates = [str(opposite), str(off_by_one), str(ignores_quantity)]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(str(answer), candidates),
    )
