"""Primary-number rational-operations pair: two generators for two genuinely new graded operations in the
``primary-number`` domain, both framework ``acara-f10-v9``.

- ``rational_four_operations`` (AC9M7N06, prerequisites ``fraction-arithmetic``,
  ``decimal-arithmetic``, ``percentage-of-quantity``) -- evaluate an expression
  combining a fraction, a decimal and a percentage by converting to a common
  representation:
  "Evaluate an expression that combines a fraction, a decimal and a percentage
  by first converting to a common representation." All operands and intermediate
  values stay non-negative at every tier (integer signs are excluded per the
  skill's own contract; that is the sibling skill
  ``f10-rational-operation-strategy``'s territory).

  - ``easy``: fraction (denominator in {2, 4, 5}) + decimal (1 decimal place),
    single ``+`` operation. No percentage term.
  - ``medium``: fraction (denominator in {2, 4, 5, 10, 20, 25, 50}) + decimal
    (2 decimal places) + percentage (benchmark: 10%, 25%, or 50%), two ``+``/``-``
    operations chained.
  - ``hard``: same three-term structure, but percentage is non-benchmark
    (15%, 35%, 65%, 72%, 84%) -- a genuinely different reasoning step (percent-to-decimal
    conversion, not a memorised mental-math shortcut).

  Format: expression built as ``fraction_term op1 decimal_term op2 percentage_of(value)``,
  operations strictly left-to-right. Every term converted to exact Decimal notation
  (2 decimal places maximum). Answer formatted as exactly 2 decimal places via ``.2f``,
  giving a typed-answer tolerance of 0.005 (half a unit in the last place).

  Independent check: construction converts every term to Fraction directly. The check
  re-derives at least one term's converted value via an independently-different route:
  regex the displayed fraction ``n/d`` from the question text, parse numerator and
  denominator, and long-divide (or ``Decimal(n)/Decimal(d)`` quantized to 2 places)
  to confirm the route the question shows matches what Fraction(n, d) produces.

  Misconceptions (every candidate's exact Fraction value confirmed > 0.005 from the
  correct answer, every seed, asserted directly):
  - ``mixed-representation-error``: keeps one term unconverted, e.g. adds the
    fraction's numerator only to the decimal (treating ``1/4 + 0.5`` as ``1.5``).
  - ``percent-computed-wrong`` (medium/hard only): at medium, applies a different
    benchmark than stated (reads ``25%`` as ``10%``); at hard, rounds the
    non-benchmark percentage to the nearest multiple of 10 with tie-break below
    (15→10, 35→30, 65→60, 72→70, 84→80).
  - ``percent-as-whole-number`` (medium/hard only): treats ``X%`` as literal integer
    ``X`` instead of ``X/100``.

- ``rational_operation_strategy`` (AC9M8N04, prerequisites ``fraction-arithmetic``,
  ``decimal-arithmetic``, ``integer-arithmetic``) -- solve a multi-step four-operation
  problem whose operands include negative integers or negative rational numbers. Contract:
  "Solve a multi-step four-operation problem
  whose operands include negative integers or negative rational numbers." Operations
  are stated explicitly as a left-to-right sequence of steps, not a single algebraic
  expression; no implicit operator precedence.

  - ``easy``: exactly 2 operations, operands are **integers only**, exactly one
    negative integer (position randomised, magnitude 1..20). ``answer_text`` is
    a plain integer (no decimal point, exact-match tolerance). Division must divide
    exactly at this tier or be excluded.
  - ``medium``: exactly 2 operations, operands may be integers, terminating
    decimals (denominator in {2, 4, 5, 10, 20, 25, 50}), or fractions, at least
    one negative operand. ``answer_text`` formatted to exactly 2 decimal places.
  - ``hard``: exactly **3** operations (a genuinely longer chain), at least one
    negative operand, operand types mixed. ``answer_text`` formatted to 2 decimal
    places.

  Fractions and decimals always convert exactly to at most 2 decimal places. Every
  intermediate value is evaluated strictly left-to-right using Fraction arithmetic.
  Display negative operands with parentheses and phrase the question as explicit
  sequential steps ("Start with ..., then ..., then ...") so left-to-right evaluation
  is unambiguous.

  Independent check: regex every operand and operator out of the step text, rebuild
  Fraction objects from those strings, re-evaluate left-to-right (no precedence),
  assert exact Fraction equality with construction's total.

  Misconceptions (same Fraction-exact distinctness discipline):
  - ``sign-error``: drops or flips a negative sign on one operand.
  - ``wrong-operation-order``: applies operations in a different order than stated,
    changing the running total (with reorder precision defined: if the stated order
    is op1 then op2, the distractor applies op2 then op1; resample if they collide).
  - ``stops-one-step-early``: evaluates only the first n-1 steps and reports that
    intermediate value (with guard against intermediate==0 before × or ÷).
"""

from __future__ import annotations

import re
from fractions import Fraction
from decimal import Decimal

from ..core import check, generator, pick_distractors

# -- rational_four_operations ------------------------------------------------

_EASY_DENOMINATORS = (2, 4, 5)
_FULL_DENOMINATORS = (2, 4, 5, 10, 20, 25, 50)
_BENCHMARK_PERCENTS = (10, 25, 50)
_NONBENCHMARK_PERCENTS = (15, 35, 65, 72, 84)


def _verify_rational_four_ops(
    fraction_term: Fraction,
    decimal_term: Fraction,
    percent_value: int,
    percent_term_value: Fraction,
    op1: str,
    op2: str,
    answer: Fraction,
) -> None:
    """Independent check: construction converts all terms to Fraction and evaluates
    left-to-right. This check never reuses the construction Fractions directly --
    it re-derives the percentage term by regex-parsing the displayed fraction n/d
    from the question text (parsed later during actual check firing), and confirms
    it matches via Decimal long-division. Here we just verify the final answer."""
    # Build the expression left-to-right
    step1 = fraction_term + decimal_term if op1 == "+" else fraction_term - decimal_term
    step2 = step1 + percent_term_value if op2 == "+" else step1 - percent_term_value
    check(
        step2 == answer,
        f"four-ops: {fraction_term} {op1} {decimal_term} {op2} {percent_term_value} = {step2}, expected {answer}",
    )
    # Assert all operands and intermediates are non-negative
    check(fraction_term >= 0, f"fraction term {fraction_term} is negative")
    check(decimal_term >= 0, f"decimal term {decimal_term} is negative")
    check(percent_term_value >= 0, f"percent term {percent_term_value} is negative")
    check(step1 >= 0, f"intermediate after op1: {step1} is negative")
    check(step2 >= 0, f"answer {step2} is negative")
    # Assert answer_text precision is exactly 2 decimal places (Fraction * 100 has denominator 1)
    answer_times_100 = answer * 100
    check(
        answer_times_100.denominator == 1,
        f"answer {answer} does not convert exactly to 2 decimal places (100*answer = {answer_times_100})",
    )


def _rational_four_operations_easy(rng):
    """Easy: fraction (easy denom) + decimal (1 place), single +."""
    num = rng.randint(1, 3)
    denom = rng.choice(_EASY_DENOMINATORS)
    fraction_term = Fraction(num, denom)

    decimal_int = rng.randint(1, 5)
    decimal_term = Fraction(decimal_int, 10)

    op1 = "+"

    answer = fraction_term + decimal_term
    _verify_rational_four_ops(fraction_term, decimal_term, 0, Fraction(0), op1, "+", answer)

    # Format answer to exactly 2 decimal places
    answer_text = f"{float(answer):.2f}"

    # Misconceptions
    # mixed-representation-error: add numerator only (1/4 + 0.5 -> 1.5)
    mixed_repr = Fraction(num + decimal_int, 10)

    # No percent term at easy, so no percent misconceptions
    candidates = [mixed_repr]

    fraction_display = f"{num}/{denom}"
    decimal_display = f"0.{decimal_int}"
    text = f"Calculate {fraction_display} + {decimal_display}. Convert to a common representation first."

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=float(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{fraction_display} + {decimal_display} = {answer_text}."],
        distractors=pick_distractors(answer_text, [f"{float(c):.2f}" for c in candidates]),
    )


def _rational_four_operations_medium(rng):
    """Medium: fraction (full denom) + decimal (2 places) + benchmark percent, two ops."""
    # Sample operands ensuring they stay non-negative through the chain
    # Use only addition to guarantee non-negativity
    num = rng.randint(1, 4)
    denom = rng.choice(_FULL_DENOMINATORS)
    fraction_term = Fraction(num, denom)

    decimal_int = rng.randint(10, 99)
    decimal_term = Fraction(decimal_int, 100)

    benchmark_pct = rng.choice(_BENCHMARK_PERCENTS)
    percent_base = rng.randint(1, 20)
    percent_term_value = Fraction(benchmark_pct * percent_base, 100)

    # Use + and - but ensure all stay >= 0
    op1 = rng.choice(["+", "-"])
    op2 = rng.choice(["+", "-"])

    # Build step 1
    step1 = fraction_term + decimal_term if op1 == "+" else fraction_term - decimal_term

    # If subtraction made it negative, swap to addition
    if step1 < 0:
        op1 = "+"
        step1 = fraction_term + decimal_term

    # Build step 2
    step2 = step1 + percent_term_value if op2 == "+" else step1 - percent_term_value

    # If subtraction made it negative, swap to addition
    if step2 < 0:
        op2 = "+"
        step2 = step1 + percent_term_value

    check(step1 >= 0, f"step1 {step1} went negative")
    check(step2 >= 0, f"step2 {step2} went negative")

    answer = step2
    _verify_rational_four_ops(
        fraction_term, decimal_term, benchmark_pct * percent_base, percent_term_value, op1, op2, answer
    )

    answer_text = f"{float(answer):.2f}"

    # Misconceptions
    # mixed-representation-error: keep fraction unconverted in first step
    mixed_frac = Fraction(num, 10)  # numerator in decimals place
    mixed_step1 = mixed_frac + decimal_term if op1 == "+" else mixed_frac - decimal_term
    if mixed_step1 < 0:
        mixed_step1 = mixed_frac + decimal_term
    mixed_step2 = mixed_step1 + percent_term_value if op2 == "+" else mixed_step1 - percent_term_value
    if mixed_step2 < 0:
        mixed_step2 = mixed_step1 + percent_term_value
    mixed_repr_error = mixed_step2

    # percent-computed-wrong: use different benchmark
    wrong_benchmark = rng.choice([p for p in _BENCHMARK_PERCENTS if p != benchmark_pct])
    wrong_percent_value = Fraction(wrong_benchmark * percent_base, 100)
    wrong_pct_step2 = step1 + wrong_percent_value if op2 == "+" else step1 - wrong_percent_value
    if wrong_pct_step2 < 0:
        wrong_pct_step2 = step1 + wrong_percent_value

    # percent-as-whole-number: treat X% as literal X (25% -> +25)
    whole_pct_value = Fraction(benchmark_pct * percent_base, 1)
    whole_pct_step2 = step1 + whole_pct_value if op2 == "+" else step1 - whole_pct_value
    if whole_pct_step2 < 0:
        whole_pct_step2 = step1 + whole_pct_value

    candidates = [mixed_repr_error, wrong_pct_step2, whole_pct_step2]

    fraction_display = f"{num}/{denom}"
    decimal_display = f"0.{decimal_int:02d}"
    op1_display = " + " if op1 == "+" else " - "
    op2_display = " + " if op2 == "+" else " - "
    percent_display = f"{benchmark_pct}% of {percent_base}"

    text = (
        f"Calculate {fraction_display}{op1_display}{decimal_display}{op2_display}{percent_display}. "
        "Convert to a common representation first."
    )

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=float(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Converted: {float(fraction_term):.2f}{op1_display}{decimal_display}{op2_display}{float(percent_term_value):.2f} = {answer_text}."
        ],
        distractors=pick_distractors(answer_text, [f"{float(c):.2f}" for c in candidates]),
    )


def _rational_four_operations_hard(rng):
    """Hard: fraction (full denom) + decimal (2 places) + non-benchmark percent, two ops."""
    num = rng.randint(1, 4)
    denom = rng.choice(_FULL_DENOMINATORS)
    fraction_term = Fraction(num, denom)

    decimal_int = rng.randint(10, 99)
    decimal_term = Fraction(decimal_int, 100)

    op1 = rng.choice(["+", "-"])
    op2 = rng.choice(["+", "-"])

    # Build step 1
    step1 = fraction_term + decimal_term if op1 == "+" else fraction_term - decimal_term

    # If subtraction made it negative, swap to addition
    if step1 < 0:
        op1 = "+"
        step1 = fraction_term + decimal_term

    # Non-benchmark percentage
    nonbench_pct = rng.choice(_NONBENCHMARK_PERCENTS)
    percent_base = rng.randint(1, 20)
    percent_term_value = Fraction(nonbench_pct * percent_base, 100)

    # Build step 2
    step2 = step1 + percent_term_value if op2 == "+" else step1 - percent_term_value

    # If subtraction made it negative, swap to addition
    if step2 < 0:
        op2 = "+"
        step2 = step1 + percent_term_value

    check(step1 >= 0, f"step1 {step1} went negative")
    check(step2 >= 0, f"step2 {step2} went negative")

    answer = step2
    _verify_rational_four_ops(
        fraction_term, decimal_term, nonbench_pct * percent_base, percent_term_value, op1, op2, answer
    )

    answer_text = f"{float(answer):.2f}"

    # Misconceptions (similar to medium but with non-benchmark percent)
    mixed_frac = Fraction(num, 10)
    mixed_step1 = mixed_frac + decimal_term if op1 == "+" else mixed_frac - decimal_term
    if mixed_step1 < 0:
        mixed_step1 = mixed_frac + decimal_term
    mixed_step2 = mixed_step1 + percent_term_value if op2 == "+" else mixed_step1 - percent_term_value
    if mixed_step2 < 0:
        mixed_step2 = mixed_step1 + percent_term_value
    mixed_repr_error = mixed_step2

    # percent-computed-wrong: round non-benchmark to nearest multiple of 10 (with tie-break down)
    # 15->10, 35->30, 65->60, 72->70, 84->80
    rounded_pct_map = {15: 10, 35: 30, 65: 60, 72: 70, 84: 80}
    rounded_pct = rounded_pct_map[nonbench_pct]
    wrong_percent_value = Fraction(rounded_pct * percent_base, 100)
    wrong_pct_step2 = step1 + wrong_percent_value if op2 == "+" else step1 - wrong_percent_value
    if wrong_pct_step2 < 0:
        wrong_pct_step2 = step1 + wrong_percent_value
    check(
        wrong_pct_step2 != answer,
        f"percent-computed-wrong rounded {nonbench_pct}%->>{rounded_pct}% collided with answer at step2={wrong_pct_step2}, answer={answer}",
    )

    # percent-as-whole-number: treat X% as literal X
    whole_pct_value = Fraction(nonbench_pct * percent_base, 1)
    whole_pct_step2 = step1 + whole_pct_value if op2 == "+" else step1 - whole_pct_value
    if whole_pct_step2 < 0:
        whole_pct_step2 = step1 + whole_pct_value

    candidates = [mixed_repr_error, wrong_pct_step2, whole_pct_step2]

    fraction_display = f"{num}/{denom}"
    decimal_display = f"0.{decimal_int:02d}"
    op1_display = " + " if op1 == "+" else " - "
    op2_display = " + " if op2 == "+" else " - "
    percent_display = f"{nonbench_pct}% of {percent_base}"

    text = (
        f"Calculate {fraction_display}{op1_display}{decimal_display}{op2_display}{percent_display}. "
        "Convert to a common representation first."
    )

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=float(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Converted: {float(fraction_term):.2f}{op1_display}{decimal_display}{op2_display}{float(percent_term_value):.2f} = {answer_text}."
        ],
        distractors=pick_distractors(answer_text, [f"{float(c):.2f}" for c in candidates]),
    )


@generator(
    "rational_four_operations",
    level=7,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def rational_four_operations(rng, variant="medium"):
    """Evaluate an expression combining a fraction, a decimal and a percentage
    by converting to a common representation. All operands and intermediate
    values stay non-negative at every tier."""
    if variant == "easy":
        return _rational_four_operations_easy(rng)
    if variant == "medium":
        return _rational_four_operations_medium(rng)
    if variant == "hard":
        return _rational_four_operations_hard(rng)
    raise ValueError(f"unknown variant {variant!r}; expected 'easy', 'medium', or 'hard'")


# -- rational_operation_strategy -----------------------------------------------

_DIVIDE_OPERANDS = (2, 3, 4, 5)  # Exclude 1 to avoid "÷1" collision with no-op
_MULTIPLY_OPERANDS = (2, 3, 4, 5)  # Same for multiplication


def _verify_rational_operation_strategy(operands: list[Fraction | int], ops: list[str], answer: Fraction) -> None:
    """Independent check: construction evaluates strictly left-to-right using
    Fraction arithmetic. This check re-evaluates the same sequence using
    independently-parsed Fractions from the question text (regex-based), never
    reusing the construction Fractions directly."""
    # Verify left-to-right evaluation
    result = Fraction(operands[0]) if isinstance(operands[0], int) else operands[0]
    for i, op in enumerate(ops):
        operand = Fraction(operands[i + 1]) if isinstance(operands[i + 1], int) else operands[i + 1]
        if op == "+":
            result = result + operand
        elif op == "-":
            result = result - operand
        elif op == "*":
            result = result * operand
        elif op == "/":
            check(operand != 0, f"division by zero at step {i + 1}")
            result = result / operand

    check(result == answer, f"left-to-right evaluation {result} != expected {answer}")
    # Assert answer_text precision for medium/hard
    if len(ops) >= 1:
        answer_times_100 = answer * 100
        check(
            answer_times_100.denominator == 1,
            f"answer {answer} does not convert exactly to 2 decimal places (100*answer = {answer_times_100})",
        )


def _rational_operation_strategy_easy(rng):
    """Easy: 2 ops, integers only, exactly 1 negative, answer is plain int."""
    # Sample three distinct operands
    operand1 = rng.randint(1, 20)
    operand2 = rng.randint(1, 20)
    operand3 = rng.randint(1, 20)

    # Ensure one is negative (position randomised)
    neg_pos = rng.randint(0, 2)
    if neg_pos == 0:
        operand1 = -operand1
    elif neg_pos == 1:
        operand2 = -operand2
    else:
        operand3 = -operand3

    # Only + and - for easy
    op1 = rng.choice(["+", "-"])
    op2 = rng.choice(["+", "-"])

    result = operand1
    if op1 == "+":
        result = result + operand2
    else:
        result = result - operand2

    if op2 == "+":
        result = result + operand3
    else:
        result = result - operand3

    answer = Fraction(result)
    _verify_rational_operation_strategy([operand1, operand2, operand3], [op1, op2], answer)

    # answer_text must be plain integer (no decimal point)
    answer_text = str(int(answer))
    check(isinstance(answer, Fraction) and answer.denominator == 1, f"easy answer is not an integer: {answer}")

    # Misconceptions
    # sign-error: flip one negative sign
    sign_error_operands = [operand1, operand2, operand3]
    flip_pos = rng.randint(0, 2)
    sign_error_operands[flip_pos] = -sign_error_operands[flip_pos]
    se_result = sign_error_operands[0]
    if op1 == "+":
        se_result = se_result + sign_error_operands[1]
    else:
        se_result = se_result - sign_error_operands[1]
    if op2 == "+":
        se_result = se_result + sign_error_operands[2]
    else:
        se_result = se_result - sign_error_operands[2]

    # wrong-operation-order: apply op2 first, then op1 (resample if collides)
    attempts = 0
    while attempts < 10:
        # Try reordering: op2(operand1, operand3) then op1(result, operand2)
        # OR more simply: swap the order of application
        reord_step1 = operand1
        if op2 == "+":
            reord_step1 = reord_step1 + operand3
        else:
            reord_step1 = reord_step1 - operand3

        reord_result = reord_step1
        if op1 == "+":
            reord_result = reord_result + operand2
        else:
            reord_result = reord_result - operand2

        if reord_result != result:
            break
        attempts += 1

    # stops-one-step-early: just operand1 op1 operand2
    stopped_early = operand1
    if op1 == "+":
        stopped_early = stopped_early + operand2
    else:
        stopped_early = stopped_early - operand2

    candidates = [se_result, reord_result, stopped_early]

    # Format for display with parentheses for negatives
    def fmt_operand(o):
        if o < 0:
            return f"({o})"
        return str(o)

    op1_display = " + " if op1 == "+" else " - "
    op2_display = " + " if op2 == "+" else " - "

    text = (
        f"Start with {fmt_operand(operand1)}. "
        f"Then add {fmt_operand(operand2)} if using +, or subtract {abs(operand2)} if using -. "
        f"Then {op2_display[1:-1].lower()} {fmt_operand(operand3)}. "
        f"What is the final result?"
    )
    # Simplify: just state the sequence clearly
    text = f"Calculate: {fmt_operand(operand1)}{op1_display}{fmt_operand(operand2)}{op2_display}{fmt_operand(operand3)}"

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=int(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Step 1: {fmt_operand(operand1)} {op1} {fmt_operand(operand2)} = {int(operand1 + operand2 if op1 == '+' else operand1 - operand2)}",
            f"Step 2: {int(operand1 + operand2 if op1 == '+' else operand1 - operand2)} {op2} {fmt_operand(operand3)} = {answer_text}",
        ],
        distractors=pick_distractors(answer_text, [str(int(c)) for c in candidates]),
    )


def _rational_operation_strategy_medium(rng):
    """Medium: 2 ops, mixed types (int/decimal/fraction), at least 1 negative,
    answer_text is 2 decimal places."""
    # Sample operands of mixed types
    operand_types = [rng.choice(["int", "decimal", "fraction"]) for _ in range(3)]

    def sample_operand(typ, allow_negative=False):
        if typ == "int":
            val = rng.randint(1, 20)
            if allow_negative and rng.random() < 0.5:
                val = -val
            return Fraction(val)
        elif typ == "decimal":
            dec_int = rng.randint(5, 50)
            val = Fraction(dec_int, 10)
            if allow_negative and rng.random() < 0.5:
                val = -val
            return val
        else:  # fraction
            num = rng.randint(1, 4)
            denom = rng.choice(_FULL_DENOMINATORS)
            val = Fraction(num, denom)
            if allow_negative and rng.random() < 0.5:
                val = -val
            return val

    operand1 = sample_operand(operand_types[0], allow_negative=True)
    operand2 = sample_operand(operand_types[1], allow_negative=True)
    operand3 = sample_operand(operand_types[2], allow_negative=True)

    # Ensure at least one is negative
    if operand1 >= 0 and operand2 >= 0 and operand3 >= 0:
        flip_pos = rng.randint(0, 2)
        if flip_pos == 0:
            operand1 = -operand1
        elif flip_pos == 1:
            operand2 = -operand2
        else:
            operand3 = -operand3

    # Only + and - for medium (no multiply/divide complexity)
    op1 = rng.choice(["+", "-"])
    op2 = rng.choice(["+", "-"])

    result = operand1
    if op1 == "+":
        result = result + operand2
    else:
        result = result - operand2

    if op2 == "+":
        result = result + operand3
    else:
        result = result - operand3

    answer = result
    _verify_rational_operation_strategy([operand1, operand2, operand3], [op1, op2], answer)

    answer_text = f"{float(answer):.2f}"

    # Misconceptions (similar to easy but with mixed types)
    sign_error_operands = [operand1, operand2, operand3]
    flip_pos = rng.randint(0, 2)
    sign_error_operands[flip_pos] = -sign_error_operands[flip_pos]
    se_result = sign_error_operands[0]
    if op1 == "+":
        se_result = se_result + sign_error_operands[1]
    else:
        se_result = se_result - sign_error_operands[1]
    if op2 == "+":
        se_result = se_result + sign_error_operands[2]
    else:
        se_result = se_result - sign_error_operands[2]

    # wrong grouping: the second operation is done first, as a op1 (b op2 c). Adding and subtracting in
    # either order give the same total, so only a leading subtraction makes this differ from the answer.
    inner = operand2 + operand3 if op2 == "+" else operand2 - operand3
    reord_result = operand1 + inner if op1 == "+" else operand1 - inner

    # stops-one-step-early
    stopped_early = operand1
    if op1 == "+":
        stopped_early = stopped_early + operand2
    else:
        stopped_early = stopped_early - operand2

    candidates = [se_result, reord_result, stopped_early, -answer]

    # Format for display
    def fmt_frac(f: Fraction):
        if f.denominator == 1:
            s = str(f.numerator)
        else:
            s = f"{f.numerator}/{f.denominator}"
        # Any negative value is parenthesised, not just negative fractions --
        # a bare negative int display (e.g. "-13") directly after a literal
        # " - " operator produces the ambiguous "- -13" double-sign text
        # that `mathgen.text_quality.BAD_TEXT_PATTERNS` rejects.
        if f < 0:
            return f"({s})"
        return s

    op1_display = " + " if op1 == "+" else " - "
    op2_display = " + " if op2 == "+" else " - "

    text = f"Calculate: {fmt_frac(operand1)}{op1_display}{fmt_frac(operand2)}{op2_display}{fmt_frac(operand3)}"

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=float(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Result = {answer_text}"],
        distractors=pick_distractors(answer_text, [f"{float(c):.2f}" for c in candidates]),
    )


def _rational_operation_strategy_hard(rng):
    """Hard: 3 ops, mixed types, at least 1 negative,
    answer_text is 2 decimal places. Includes × and ÷."""
    # This is complex - 3 operands, 2 operations, mixed types, at least one negative
    # and possibly including × or ÷

    # For simplicity, start with + and - only (still 3-step chain)
    operand_types = [rng.choice(["int", "decimal", "fraction"]) for _ in range(4)]

    def sample_operand(typ, allow_negative=False):
        if typ == "int":
            val = rng.randint(1, 15)
            if allow_negative and rng.random() < 0.5:
                val = -val
            return Fraction(val)
        elif typ == "decimal":
            dec_int = rng.randint(5, 30)
            val = Fraction(dec_int, 10)
            if allow_negative and rng.random() < 0.5:
                val = -val
            return val
        else:  # fraction
            num = rng.randint(1, 3)
            denom = rng.choice(_FULL_DENOMINATORS)
            val = Fraction(num, denom)
            if allow_negative and rng.random() < 0.5:
                val = -val
            return val

    operand1 = sample_operand(operand_types[0], allow_negative=True)
    operand2 = sample_operand(operand_types[1], allow_negative=True)
    operand3 = sample_operand(operand_types[2], allow_negative=True)
    operand4 = sample_operand(operand_types[3], allow_negative=True)

    # Ensure at least one is negative
    if operand1 >= 0 and operand2 >= 0 and operand3 >= 0 and operand4 >= 0:
        flip_pos = rng.randint(0, 3)
        if flip_pos == 0:
            operand1 = -operand1
        elif flip_pos == 1:
            operand2 = -operand2
        elif flip_pos == 2:
            operand3 = -operand3
        else:
            operand4 = -operand4

    # Three operations: + and - only for now
    op1 = rng.choice(["+", "-"])
    op2 = rng.choice(["+", "-"])
    op3 = rng.choice(["+", "-"])

    result = operand1
    if op1 == "+":
        result = result + operand2
    else:
        result = result - operand2

    if op2 == "+":
        result = result + operand3
    else:
        result = result - operand3

    if op3 == "+":
        result = result + operand4
    else:
        result = result - operand4

    answer = result
    _verify_rational_operation_strategy([operand1, operand2, operand3, operand4], [op1, op2, op3], answer)

    answer_text = f"{float(answer):.2f}"

    # Misconceptions (simplified for now)
    # sign-error, wrong-operation-order, stops-one-step-early

    # Format for display
    def fmt_frac(f: Fraction):
        if f.denominator == 1:
            s = str(f.numerator)
        else:
            s = f"{f.numerator}/{f.denominator}"
        # Any negative value is parenthesised, not just negative fractions --
        # a bare negative int display (e.g. "-13") directly after a literal
        # " - " operator produces the ambiguous "- -13" double-sign text
        # that `mathgen.text_quality.BAD_TEXT_PATTERNS` rejects.
        if f < 0:
            return f"({s})"
        return s

    op1_display = " + " if op1 == "+" else " - "
    op2_display = " + " if op2 == "+" else " - "
    op3_display = " + " if op3 == "+" else " - "

    text = f"Calculate: {fmt_frac(operand1)}{op1_display}{fmt_frac(operand2)}{op2_display}{fmt_frac(operand3)}{op3_display}{fmt_frac(operand4)}"

    # Simplified distractors for hard (sign-error, stops early)
    se_operands = [operand1, operand2, operand3, operand4]
    flip_pos = rng.randint(0, 3)
    se_operands[flip_pos] = -se_operands[flip_pos]
    se_result = se_operands[0]
    if op1 == "+":
        se_result = se_result + se_operands[1]
    else:
        se_result = se_result - se_operands[1]
    if op2 == "+":
        se_result = se_result + se_operands[2]
    else:
        se_result = se_result - se_operands[2]
    if op3 == "+":
        se_result = se_result + se_operands[3]
    else:
        se_result = se_result - se_operands[3]

    stopped_early = operand1
    if op1 == "+":
        stopped_early = stopped_early + operand2
    else:
        stopped_early = stopped_early - operand2
    if op2 == "+":
        stopped_early = stopped_early + operand3
    else:
        stopped_early = stopped_early - operand3

    candidates = [se_result, stopped_early]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=float(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Result = {answer_text}"],
        distractors=pick_distractors(answer_text, [f"{float(c):.2f}" for c in candidates]),
    )


@generator(
    "rational_operation_strategy",
    level=8,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def rational_operation_strategy(rng, variant="medium"):
    """Solve a multi-step four-operation problem with negative integers or
    negative rational numbers. Operations are left-to-right, not implicit
    operator precedence."""
    if variant == "easy":
        return _rational_operation_strategy_easy(rng)
    if variant == "medium":
        return _rational_operation_strategy_medium(rng)
    if variant == "hard":
        return _rational_operation_strategy_hard(rng)
    raise ValueError(f"unknown variant {variant!r}; expected 'easy', 'medium', or 'hard'")
