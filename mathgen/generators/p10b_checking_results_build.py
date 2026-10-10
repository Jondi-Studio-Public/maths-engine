"""Generates `reasonableness_verdict` and `locate_calculation_error` questions: judging whether a stated
calculation result is reasonable, and finding the step where it went wrong.

Checking a result cluster. senior-ascertain-the-reasonableness-of-answers-to-arithmetic-calculations
(ACMEM003) + senior-check-results-of-calculations-for-accuracy (ACMEM006).

Both skills are ``answer: mcq`` classifications over a small, fixed option set --
never a graded numeric band. They judge the SAME kind of object (a stated
calculation and its stated result) but ask a genuinely different question:

- ``reasonableness_verdict`` (ACMEM003) -- the answer is a VERDICT on a single
  stated result, never a recomputation. Every error, when present, is GROSS:
  an order of magnitude (x10 / x0.1) or a sign flip -- the kind of thing caught
  by eyeballing magnitude alone, without reworking the arithmetic.
- ``locate_calculation_error`` (ACMEM006) -- the answer IDENTIFIES A STEP in a
  3-step worked calculation, never a value. Every error, when present, is
  LOCAL: a small dropped-carry (+10) or times-table (+2) slip that a
  magnitude-only glance cannot find -- the learner must recompute each step
  from its own stated inputs.

Both draw their answer class FIRST, uniformly, then construct the calculation to realize it
(construct-to-target).
"""

from __future__ import annotations

from fractions import Fraction

from sympy import sympify

from ..core import check, check_equivalent, generator, latex_answer

# ============================================================================
# Skill A: reasonableness_verdict (ACMEM003)
# ============================================================================

# Fixed MCQ options (verbatim, same order every seed).
OPTION_REASONABLE = "Reasonable"
OPTION_TOO_LARGE = "Not reasonable -- the stated answer is too large"
OPTION_TOO_SMALL = "Not reasonable -- the stated answer is too small"
OPTION_WRONG_SIGN = "Not reasonable -- the stated answer has the wrong sign"
OPTIONS_A = (OPTION_REASONABLE, OPTION_TOO_LARGE, OPTION_TOO_SMALL, OPTION_WRONG_SIGN)

# ratio (stated / true) -> the option it maps to. Keyed on exact Fraction/int
# value, never trusted from the construction's own class label -- the
# independent check below recomputes this ratio from the constructed numbers
# and looks it up here a second time.
_RATIO_TO_OPTION = {
    1: OPTION_REASONABLE,
    10: OPTION_TOO_LARGE,
    Fraction(1, 10): OPTION_TOO_SMALL,
    -1: OPTION_WRONG_SIGN,
}


def _draw_true_and_expr_easy(rng):
    """Two operands, both multiples of 10, addition only -- ``true`` is
    always an exact multiple of 10 so ``true / 10`` is always an integer."""
    a = rng.randint(2, 99) * 10
    b = rng.randint(2, 99) * 10
    expr = f"{a} + {b}"
    true = a + b
    return expr, true


def _draw_true_and_expr_medium(rng):
    """Multiplication, one operand forced to a multiple of 10 -- multiplicative
    order-of-magnitude reasoning, genuinely different from additive (easy)."""
    a = rng.randint(2, 99) * 10
    b = rng.randint(2, 999)
    expr = f"{a} × {b}"
    true = a * b
    return expr, true


def _draw_true_and_expr_hard(rng):
    """Three operands, two operations respecting order-of-operations (multiply
    before add) -- both the multiplied and the added operand are multiples of
    10, so the compounded result is still an exact multiple of 10."""
    a = rng.randint(2, 99) * 10
    b = rng.randint(2, 99)
    c = rng.randint(2, 99) * 10
    if rng.choice([True, False]):
        expr = f"{a} × {b} + {c}"
        true = a * b + c
    else:
        expr = f"{c} + {a} × {b}"
        true = c + a * b
    return expr, true


_EXPR_BUILDERS = {
    "easy": _draw_true_and_expr_easy,
    "medium": _draw_true_and_expr_medium,
    "hard": _draw_true_and_expr_hard,
}


def _independent_check_a(expr: str, true: int, stated: int, answer: str) -> None:
    """Independent route 1: re-evaluate the expression STRING with sympy
    (never reusing the int arithmetic construction used), and confirm it
    agrees with the ``true`` value construction computed.

    Independent route 2: recompute ``ratio = stated / true`` and look the
    CORRECT option up in ``_RATIO_TO_OPTION``, never trusting the class label
    construction drew.
    """
    true_from_text = int(sympify(expr.replace("×", "*")))
    check_equivalent(true_from_text, true, f"expression {expr!r} does not evaluate to {true}")
    check(true != 0, "true result must never be zero (ratio would be undefined)")
    ratio = Fraction(stated, true)
    check(
        ratio in _RATIO_TO_OPTION,
        f"ratio {ratio} (stated={stated}, true={true}) is not one of the four sanctioned reasonableness ratios",
    )
    derived_answer = _RATIO_TO_OPTION[ratio]
    check(
        derived_answer == answer,
        f"independent check mismatch: ratio {ratio} maps to {derived_answer!r}, construction chose {answer!r}",
    )


@generator(
    "reasonableness_verdict",
    level=11,
    presets={
        "easy": dict(question_type="easy"),
        "medium": dict(question_type="medium"),
        "hard": dict(question_type="hard"),
    },
)
def reasonableness_verdict(rng, question_type="easy"):
    """Given a calculation and a stated (proposed) result, judge whether the
    stated result is reasonable, or -- if not -- whether it is too large, too
    small, or has the wrong sign. The stated result is presented as a
    completed fact; the learner is never shown working and never asked to
    recompute, only to judge plausibility of magnitude and sign."""
    if question_type not in _EXPR_BUILDERS:
        raise ValueError(f"question_type must be 'easy', 'medium', or 'hard'; got {question_type!r}")

    # Draw the verdict class FIRST (construct-to-target), then build the
    # calculation to realize it.
    answer = rng.choice(OPTIONS_A)

    expr, true = _EXPR_BUILDERS[question_type](rng)
    check(true > 0, "constructed true result must be positive")
    check(true % 10 == 0, f"constructed true result {true} must be an exact multiple of 10")

    if answer == OPTION_REASONABLE:
        stated = true
    elif answer == OPTION_TOO_LARGE:
        stated = true * 10
    elif answer == OPTION_TOO_SMALL:
        stated = true // 10
    else:  # OPTION_WRONG_SIGN
        stated = -true

    _independent_check_a(expr, true, stated, answer)

    text = (
        f"A calculation states: {expr} = {stated}. Is this reasonable? "
        "Select the option that best describes this stated answer."
    )

    distractors = [option for option in OPTIONS_A if option != answer]
    check(len(distractors) == 3, "reasonableness_verdict must offer exactly 3 distractors")

    working = [f"The true result of {expr} is {true}; the stated result is {stated}."]

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
# Skill B: locate_calculation_error (ACMEM006)
# ============================================================================

OPTION_STEP1 = "The error is in Step 1"
OPTION_STEP2 = "The error is in Step 2"
OPTION_STEP3 = "The error is in Step 3"
OPTION_NO_ERROR = "There is no error -- every step is correct"
OPTIONS_B = (OPTION_STEP1, OPTION_STEP2, OPTION_STEP3, OPTION_NO_ERROR)
_STEP_OPTION = {1: OPTION_STEP1, 2: OPTION_STEP2, 3: OPTION_STEP3, None: OPTION_NO_ERROR}

# Floors: the minimum TRUE result at any step, chosen so the local slip below
# is always comfortably under 30% of the true result (10/40 = 25% at the
# additive floor; 2/40 = 5% at the same floor for a x step).
_FLOOR = {"easy": 40, "medium": 40, "hard": 300}
_PLUS_MINUS_SLIP = 10
_TIMES_SLIP = 2


def _slip_for(op: str) -> int:
    return _TIMES_SLIP if op == "×" else _PLUS_MINUS_SLIP


def _draw_ops(rng, question_type):
    """[op1, op2, op3] for the three steps.

    easy: +/- only. medium/hard: a mix of +/- and x, with at least one x step
    -- the genuine escalation is recognising which slip MODEL applies to
    which operation, not a wider number range.
    """
    if question_type == "easy":
        return [rng.choice(["+", "-"]) for _ in range(3)]
    ops = [rng.choice(["+", "-", "×"]) for _ in range(3)]
    if "×" not in ops:
        ops[rng.randrange(3)] = "×"
    return ops


#: The '-' step's second operand range is drawn INDEPENDENTLY of ``first``
#: (never as a function of how close ``first`` already is to the floor), so
#: it is always a believable two-digit-or-larger dropped-carry candidate --
#: never the tiny/zero second operand a first-dependent cap can be forced
#: into ("83 - 0 = 93", "113 - 1 = 122" have no borrow to drop). `first1` (the
#: fresh operand entering Step 1) is drawn large enough that even the WORST
#: case -- every one of the 3 steps drawn as '-' (easy only; medium/hard
#: always force at least one '×', capping the longest possible run of
#: consecutive '-' steps at 2) each taking the top of this range -- still
#: leaves ``true >= _FLOOR[question_type]`` at every step. '+' and '×' only
#: ever increase the running value, so a consecutive '-' run starting anywhere
#: other than Step 1 begins from an even larger value than this bound assumes.
_MINUS_SECOND_RANGE = {"easy": (10, 40), "medium": (10, 40), "hard": (50, 150)}
_MINUS_RUN = {"easy": 3, "medium": 2, "hard": 2}


def _gen_second_and_true(rng, first: int, op: str, question_type: str) -> tuple[int, int]:
    """Draw the step's second operand and compute its TRUE result from
    ``first`` (which may be a prior step's own STATED, possibly wrong,
    result). Guarantees ``true >= _FLOOR[question_type]`` by construction:
    - '+': second is drawn from a fixed range, so true = first + second >=
      second >= the floor outright.
    - '-': second is drawn from a fixed range, independent of ``first`` --
      see `first1`'s own sizing (below) for why the floor still holds even
      after the longest possible run of consecutive '-' steps.
    - '×': second is drawn from a fixed range >= 4, so true = first * second
      >= floor * 4, comfortably clear.
    """
    floor = _FLOOR[question_type]
    if op == "+":
        second = rng.randint(150, 400) if question_type == "hard" else rng.randint(30, 90)
        true = first + second
    elif op == "-":
        low, high = _MINUS_SECOND_RANGE[question_type]
        second = rng.randint(low, high)
        true = first - second
    else:  # "×"
        second = rng.randint(8, 20) if question_type == "hard" else rng.randint(4, 12)
        true = first * second
    check(second >= 10 or op != "-", f"step second operand must be >= 10 for a believable dropped-carry (op {op!r})")
    check(true >= floor, f"step true result {true} fell below the floor {floor} for op {op!r}")
    return second, true


def _independent_check_b(steps: list[dict], drawn_error_step: int | None) -> None:
    """Independent route: recompute EVERY step's result from its own two
    stated operands and operator via `Fraction` arithmetic (never trusting
    the ``true`` value construction already attached to the step), and
    assert exactly one step mismatches its own STATED result for an error
    class -- and that step's index is the drawn class -- and zero mismatch
    for 'no error'."""
    mismatches = []
    for step in steps:
        a, b, op, stated = step["first"], step["second"], step["op"], step["stated"]
        if op == "+":
            recomputed = Fraction(a) + Fraction(b)
        elif op == "-":
            recomputed = Fraction(a) - Fraction(b)
        else:
            recomputed = Fraction(a) * Fraction(b)
        check_equivalent(recomputed, Fraction(step["true"]), f"step {step['index']} true result mismatch")
        if recomputed != stated:
            mismatches.append(step["index"])
    if drawn_error_step is None:
        check(not mismatches, f"'no error' class but step(s) {mismatches} mismatch their own recomputation")
    else:
        check(
            mismatches == [drawn_error_step], f"expected exactly step {drawn_error_step} to mismatch; got {mismatches}"
        )


@generator(
    "locate_calculation_error",
    level=11,
    presets={
        "easy": dict(question_type="easy"),
        "medium": dict(question_type="medium"),
        "hard": dict(question_type="hard"),
    },
)
def locate_calculation_error(rng, question_type="easy"):
    """Given a 3-step worked calculation (each step showing its own two
    operands, operator and stated result, each step's inputs including the
    prior step's own stated result), identify which step (if any) contains
    an arithmetic error. Every step's own arithmetic is internally correct
    given its own stated inputs, except at the one step (if any) where a
    local slip was introduced -- a dropped-carry (+10, additive steps) or a
    times-table slip (+2, multiplicative steps)."""
    if question_type not in _FLOOR:
        raise ValueError(f"question_type must be 'easy', 'medium', or 'hard'; got {question_type!r}")

    # Draw the class FIRST: which step (1/2/3) is wrong, or None (no error).
    drawn_error_step = rng.choice([1, 2, 3, None])
    answer = _STEP_OPTION[drawn_error_step]

    ops = _draw_ops(rng, question_type)
    # Sized so the WORST case -- the longest possible run of consecutive '-'
    # steps, each taking the top of `_MINUS_SECOND_RANGE` -- still clears the
    # floor: easy allows a run of 3 x 40 = 120 (floor 40, so >= 160 suffices);
    # hard allows a run of 2 x 150 = 300 (floor 300, so >= 600 suffices).
    first1 = rng.randint(650, 1200) if question_type == "hard" else rng.randint(200, 450)

    steps = []
    first = first1
    for index, op in enumerate(ops, start=1):
        second, true = _gen_second_and_true(rng, first, op, question_type)
        if drawn_error_step == index:
            stated = true + _slip_for(op)
        else:
            stated = true
        steps.append(dict(index=index, first=first, second=second, op=op, true=true, stated=stated))
        first = stated  # next step's input is THIS step's stated (possibly wrong) result

    _independent_check_b(steps, drawn_error_step)

    step_texts = [f"Step {s['index']}: {s['first']} {s['op']} {s['second']} = {s['stated']}." for s in steps]
    text = " ".join(step_texts) + " Which of these is correct?"

    distractors = [option for option in OPTIONS_B if option != answer]
    check(len(distractors) == 3, "locate_calculation_error must offer exactly 3 distractors")

    if drawn_error_step is None:
        working = ["Every step's stated result equals its own true result from its own stated inputs."]
    else:
        wrong_step = steps[drawn_error_step - 1]
        working = [
            f"Step {drawn_error_step}'s own inputs give a true result of {wrong_step['true']}, "
            f"not the stated {wrong_step['stated']}."
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
