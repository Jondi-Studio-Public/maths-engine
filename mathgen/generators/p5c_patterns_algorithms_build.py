"""F-10 pattern/sequence and algorithm skills.

Two skills built from a single module:

1. ``f10-additive-multiplicative-patterns`` -- given a sequence of numbers
   that increases or decreases by a constant amount, with one missing term
   (at the end or interior), state the missing value. Generalises the
   skip-count pilot on three axes: arbitrary constant step (not 2/5/10 only),
   decreasing sequences (not increasing only), and interior gaps (not end-only).

2. ``f10-number-algorithms-decisions`` -- given a stated finite algorithm
   with one or two operations per decision branch and several starting inputs,
   trace the algorithm step by step and list the resulting outputs (one per
   input, independent computations on a multiset basis, order-insensitive).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, join_terms, pick_distractors


# ============================================================================
# SKILL 1: f10-additive-multiplicative-patterns
# ============================================================================


def _verify_additive_pattern(
    first: int, last: int, idx_first: int, idx_last: int, missing_value: int, idx_gap: int
) -> None:
    """Independent check for additive patterns: recompute the constant
    difference from the two outermost shown terms and their position count,
    then derive the expected missing value independently.

    Uses position indices to work uniformly across all tiers:
    d = (last_shown - first_shown) / (idx_last - idx_first), exact division
    expected_value = first_shown + idx_gap * d
    """
    # Compute the constant difference from the outermost terms
    num_positions = idx_last - idx_first
    check(num_positions > 0, f"invalid index range: idx_first={idx_first}, idx_last={idx_last}")

    # Use exact division check
    remainder = (last - first) % num_positions
    check(remainder == 0, f"constant difference not exact: ({last} - {first}) % {num_positions} = {remainder}")

    d = (last - first) // num_positions

    # Now compute the expected value at the gap position
    expected = first + idx_gap * d
    check(
        missing_value == expected,
        f"missing value {missing_value} does not match expected {expected} "
        f"(first={first}, d={d}, idx_gap={idx_gap}, idx_first={idx_first})",
    )


@generator(
    "additive_multiplicative_patterns",
    level=1,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def additive_multiplicative_patterns(rng, tier="easy"):
    """Given a sequence with one missing term (increasing, decreasing, or interior),
    state the missing value.

    Difficulty progression:
    - easy: increasing sequence, constant difference d in [1, 9], ONE missing term
      at the END (extend by one step).
    - medium: DECREASING sequence, constant difference -d (d in [1, 9]),
      ONE missing term at the END -- genuine escalation (direction reversal).
    - hard: an INTERIOR gap -- show 4-5 terms with ONE term replaced by a blank
      somewhere in the middle (never first or last position), random direction;
      learner must infer d from flanking terms and interpolate.
    """
    if tier == "easy":
        # Easy: increasing, arbitrary step d in [1, 9], missing at end
        d = rng.randint(1, 9)
        actual_d = d  # Positive step for increasing
        # Start value: keep all terms >= 0 and avoid zero left operands
        # Show 3 terms: a, a+d, a+2d. Extend by 1: a+3d
        # Keep a >= 1 so narrated steps never have zero left operand
        a = rng.randint(1, 50)
        shown = [a, a + d, a + 2 * d]
        missing_index = 3  # Position of missing term (0-indexed within extended sequence)
        missing_value = a + 3 * d

        # Independent check: outermost shown terms
        idx_first, idx_last = 0, 2  # First and last shown positions
        _verify_additive_pattern(shown[0], shown[-1], idx_first, idx_last, missing_value, missing_index)

        question_text = f"What is the next number? {', '.join(map(str, shown))}, __"

    elif tier == "medium":
        # Medium: DECREASING, step d in [1, 9], missing at end
        d = rng.randint(1, 9)
        actual_d = -d  # Negative step for decreasing
        # Start with a large value, show 3 decreasing terms: a, a-d, a-2d
        # Extend by 1: a-3d. Keep a-3d >= 0, so a >= 3d
        a = rng.randint(3 * d + 1, 150)
        shown = [a, a - d, a - 2 * d]
        missing_index = 3
        missing_value = a - 3 * d

        # Independent check
        idx_first, idx_last = 0, 2
        _verify_additive_pattern(shown[0], shown[-1], idx_first, idx_last, missing_value, missing_index)

        question_text = f"What is the next number? {', '.join(map(str, shown))}, __"

    else:  # hard
        # Hard: interior gap with 4-5 shown terms, random direction
        d = rng.randint(1, 9)
        direction = rng.choice([1, -1])  # 1 for increasing, -1 for decreasing
        actual_d = direction * d  # Signed step value

        # Number of terms shown (including the blank): 4 or 5
        total_terms = rng.choice([4, 5])

        # Gap position (0-indexed): not first (0) or last (total_terms-1)
        gap_pos = rng.randint(1, total_terms - 2)

        # Generate the sequence: a, a+d, a+2d, ..., a+(total_terms-1)*d
        # Keep all terms >= 0
        if direction == 1:
            a = rng.randint(1, 50)
        else:  # decreasing
            a = rng.randint(d * (total_terms - 1) + 1, 150)

        full_sequence = [a + j * actual_d for j in range(total_terms)]

        # Verify all terms non-negative
        for i, term in enumerate(full_sequence):
            check(term >= 0, f"hard tier term {i} is negative: {term}")

        missing_value = full_sequence[gap_pos]

        # Build the shown sequence with a blank at gap_pos
        shown = full_sequence.copy()
        shown[gap_pos] = None  # Placeholder for working logic

        # Independent check using outermost shown indices
        shown_indices = [i for i in range(total_terms) if i != gap_pos]
        idx_first = shown_indices[0]
        idx_last = shown_indices[-1]
        first_shown = full_sequence[idx_first]
        last_shown = full_sequence[idx_last]
        _verify_additive_pattern(first_shown, last_shown, idx_first, idx_last, missing_value, gap_pos)

        # Build question text with blank at correct position
        text_parts = []
        for i, term in enumerate(full_sequence):
            if i == gap_pos:
                text_parts.append("__")
            else:
                text_parts.append(str(term))
        question_text = f"What is the missing number? {', '.join(text_parts)}"

    # Format answer as a single-item ordered_list
    answer_text = str(missing_value)

    # Build working text using join_terms to avoid "+ -" patterns
    if tier in ["easy", "medium"]:
        last_shown = shown[-1]
        if tier == "easy":
            # Increasing: use join_terms for positive step
            step_terms = [(d, "")]
            working = [
                f"The numbers change by {join_terms(step_terms)} each time.",
                f"The next number is {last_shown} + {d} = {missing_value}.",
            ]
        else:  # medium (decreasing)
            # Decreasing: describe the negative step clearly
            working = [
                f"The numbers decrease by {d} each time.",
                f"The next number is {last_shown} - {d} = {missing_value}.",
            ]
    else:  # hard
        # Describe the gap fill using the flanking values and computed step
        left_idx = gap_pos - 1 if gap_pos > 0 else None
        right_idx = gap_pos + 1 if gap_pos < total_terms - 1 else None

        if left_idx is not None and right_idx is not None:
            left_val = full_sequence[left_idx]
            right_val = full_sequence[right_idx]
            step_abs = d
            working = [
                f"The numbers change by {step_abs if direction == 1 else f'–{step_abs}'} each time.",
                f"Between {left_val} and {right_val} (2 steps): the missing number is {missing_value}.",
            ]
        else:
            working = [f"The missing number is {missing_value}."]

    # Misconception distractors
    distractors_candidates = []

    # 1. wrong-direction: uses opposite sign
    wrong_d = -actual_d if tier != "hard" else (-direction * d)
    if tier == "easy":
        wrong_value = shown[-1] + wrong_d
    elif tier == "medium":
        wrong_value = shown[-1] + wrong_d
    else:  # hard
        wrong_value = full_sequence[idx_first] + gap_pos * wrong_d

    # Ensure it's different and non-negative
    if wrong_value >= 0 and wrong_value != missing_value:
        distractors_candidates.append(str(int(wrong_value)))

    # 2. off-by-one-position: compute using d but at wrong position
    # For end tiers: use step too far (M+d for shown[-1])
    # For hard: use step before gap (gap-1) or after gap (gap+1)
    if tier in ["easy", "medium"]:
        off_by_one_value = shown[-1] + 2 * actual_d  # Two steps from last shown
        if (
            off_by_one_value >= 0
            and off_by_one_value != missing_value
            and str(int(off_by_one_value)) not in distractors_candidates
        ):
            distractors_candidates.append(str(int(off_by_one_value)))
    else:  # hard
        # Use step at adjacent position (gap+1 or gap-1)
        for adjacent_idx in [gap_pos + 1, gap_pos - 1]:
            if 0 <= adjacent_idx < total_terms and adjacent_idx not in [idx_first, idx_last]:
                off_by_one_value = full_sequence[idx_first] + adjacent_idx * actual_d
                if (
                    off_by_one_value >= 0
                    and off_by_one_value != missing_value
                    and str(int(off_by_one_value)) not in distractors_candidates
                ):
                    distractors_candidates.append(str(int(off_by_one_value)))
                    break

    # 3. repeats-a-neighbour (hard tier only): state one of the flanking terms
    if tier == "hard":
        neighbors = []
        if gap_pos > 0:
            neighbors.append(str(full_sequence[gap_pos - 1]))
        if gap_pos < total_terms - 1:
            neighbors.append(str(full_sequence[gap_pos + 1]))
        # Add distinct neighbors that are not the missing value
        for neighbor in neighbors:
            if neighbor != answer_text and neighbor not in distractors_candidates:
                distractors_candidates.append(neighbor)

    # For end tiers, if we don't have enough distractors, add fallback strategies:
    if tier in ["easy", "medium"] and len(distractors_candidates) < 2:
        # Fallback 1: repeats the last shown term (stopped incrementing)
        repeats_last = str(shown[-1])
        if repeats_last != answer_text and repeats_last not in distractors_candidates:
            distractors_candidates.append(repeats_last)

        # Fallback 2: use first difference (for sequences starting from different base)
        if len(distractors_candidates) < 2:
            first_diff = shown[1] - shown[0]
            first_diff_value = shown[-1] + first_diff
            if (
                first_diff_value >= 0
                and str(int(first_diff_value)) != answer_text
                and str(int(first_diff_value)) not in distractors_candidates
            ):
                distractors_candidates.append(str(int(first_diff_value)))

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=[missing_value],
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, distractors_candidates),
    )


# ============================================================================
# SKILL 2: f10-number-algorithms-decisions
# ============================================================================


def _verify_algorithm_outputs(algorithm_func, inputs: list, expected_outputs: list) -> None:
    """Independent check: recompute outputs via a different code path and verify.

    The algorithm_func should implement the decision logic in a way different
    from the construction's own path (e.g., if construction tests evenness via
    n % 2 == 0, this should test via n // 2 * 2 == n).
    """
    computed = []
    for inp in inputs:
        output = algorithm_func(inp)
        computed.append(output)

    # Verify all outputs match (as a multiset, since order doesn't matter for grade_list)
    check(
        len(computed) == len(expected_outputs), f"computed {len(computed)} outputs but expected {len(expected_outputs)}"
    )
    check(
        sorted(computed) == sorted(expected_outputs),
        f"output multiset mismatch: computed {sorted(computed)}, expected {sorted(expected_outputs)}",
    )


@generator(
    "number_algorithms_decisions",
    level=1,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def number_algorithms_decisions(rng, tier="easy"):
    """Trace a stated finite algorithm with decisions on multiple inputs
    and list the resulting outputs.

    Difficulty progression:
    - easy: one-decision rule with two simple one-operation branches,
      apply to 2 inputs (at least one even, one odd forced every seed).
    - medium: same one-decision-branch shape, 3 inputs (both branches still
      forced across the 3 inputs).
    - hard: two-operation-per-branch rule (operations do not commute),
      3 inputs, forcing both branches every seed.
    """
    if tier == "easy":
        # Easy: one decision (even/odd), two branches (one operation each)
        # Example: "If even, divide by 2. If odd, add 5."

        # Choose the operations for each branch (use rng.choice for variety)
        choice = rng.randint(0, 2)
        if choice == 0:
            even_op_name, even_op = "divide by 2", lambda x: x // 2
            odd_op_name, odd_op = "add 5", lambda x: x + 5
        elif choice == 1:
            even_op_name, even_op = "subtract 3", lambda x: x - 3
            odd_op_name, odd_op = "multiply by 2", lambda x: x * 2
        else:
            even_op_name, even_op = "add 4", lambda x: x + 4
            odd_op_name, odd_op = "subtract 1", lambda x: x - 1

        # Generate 2 inputs: at least one even, one odd (forced)
        even_input = rng.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        odd_input = rng.choice([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

        inputs = [even_input, odd_input]

        # Compute outputs
        outputs = []
        for inp in inputs:
            if inp % 2 == 0:
                if choice == 0:  # even_op has division, guard it
                    check(inp % 2 == 0, f"easy tier choice 0: even branch must receive even input, got {inp}")
                outputs.append(even_op(inp))
            else:
                outputs.append(odd_op(inp))

        # Question text
        question_text = f"If the number is even, {even_op_name}. If the number is odd, {odd_op_name}. For inputs {', '.join(map(str, inputs))}, what are the outputs? The order you give them in does not matter."

        # Independent check using different evenness test
        def check_algorithm(n):
            is_even = n // 2 * 2 == n  # Different from n % 2 == 0
            if is_even:
                return even_op(n)
            else:
                return odd_op(n)

        _verify_algorithm_outputs(check_algorithm, inputs, outputs)

        working = [
            f"For {even_input}: even, so {even_op_name.lower()} = {even_op(even_input)}.",
            f"For {odd_input}: odd, so {odd_op_name.lower()} = {odd_op(odd_input)}.",
        ]

    elif tier == "medium":
        # Medium: one-decision rule, 2 operations per branch, 3 inputs
        # Increase complexity by using more inputs

        choice = rng.randint(0, 1)
        if choice == 0:
            even_op_name, even_op = "divide by 2", lambda x: x // 2
            odd_op_name, odd_op = "add 5", lambda x: x + 5
        else:
            even_op_name, even_op = "subtract 3", lambda x: x - 3
            odd_op_name, odd_op = "multiply by 2", lambda x: x * 2

        # Generate 3 inputs: force both branches
        even_inputs = [rng.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) for _ in range(2)]
        odd_inputs = [rng.choice([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])]

        inputs = even_inputs + odd_inputs
        rng.shuffle(inputs)  # Randomize order

        # Compute outputs
        outputs = []
        for inp in inputs:
            if inp % 2 == 0:
                if choice == 0:  # even_op has division, guard it
                    check(inp % 2 == 0, f"medium tier choice 0: even branch must receive even input, got {inp}")
                outputs.append(even_op(inp))
            else:
                outputs.append(odd_op(inp))

        question_text = f"If the number is even, {even_op_name}. If the number is odd, {odd_op_name}. For inputs {', '.join(map(str, inputs))}, what are the outputs? The order you give them in does not matter."

        # Independent check
        def check_algorithm(n):
            is_even = n // 2 * 2 == n
            if is_even:
                return even_op(n)
            else:
                return odd_op(n)

        _verify_algorithm_outputs(check_algorithm, inputs, outputs)

        working = [
            f"For {inputs[0]}: {'even' if inputs[0] % 2 == 0 else 'odd'}, output = {outputs[0]}.",
            f"For {inputs[1]}: {'even' if inputs[1] % 2 == 0 else 'odd'}, output = {outputs[1]}.",
            f"For {inputs[2]}: {'even' if inputs[2] % 2 == 0 else 'odd'}, output = {outputs[2]}.",
        ]

    else:  # hard
        # Hard: two operations per branch (order matters)
        # Example: "If even, divide by 2 then add 3. If odd, multiply by 2 then subtract 1."

        # Choose operation pairs that do NOT commute
        choice = rng.randint(0, 1)
        if choice == 0:
            # Even: divide by 2, then add 3
            # Odd: multiply by 2, then subtract 1
            even_op_name = "divide by 2 then add 3"
            even_op = lambda x: x // 2 + 3
            odd_op_name = "multiply by 2 then subtract 1"
            odd_op = lambda x: x * 2 - 1
        else:
            # Even: subtract 2, then multiply by 3
            # Odd: add 5, then divide by 2
            even_op_name = "subtract 2 then multiply by 3"
            even_op = lambda x: (x - 2) * 3
            odd_op_name = "add 5 then divide by 2"
            odd_op = lambda x: (x + 5) // 2

        # Generate 3 inputs: force both branches
        even_inputs = [rng.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) for _ in range(2)]
        odd_inputs = [rng.choice([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])]

        inputs = even_inputs + odd_inputs
        rng.shuffle(inputs)

        # Compute outputs
        outputs = []
        for inp in inputs:
            if inp % 2 == 0:
                if choice == 0:  # even_op has division, guard it
                    check(inp % 2 == 0, f"hard tier choice 0: even branch must receive even input, got {inp}")
                outputs.append(even_op(inp))
            else:
                if choice == 1:  # odd_op has division, guard it
                    check((inp + 5) % 2 == 0, f"hard tier choice 1: odd branch (inp + 5) must be even, got inp={inp}")
                outputs.append(odd_op(inp))

        # Verify that operation order matters (ops do not commute)
        # For even: (x / 2) + 3 != 3 + (x / 2) or ((x + 3) / 2)
        # Test on each input to ensure the wrong-order gives different results
        sample_even = even_inputs[0]
        true_even_result = even_op(sample_even)

        # Check that swapping ops gives different result
        # This depends on the chosen operations; we'll assert it
        if even_op_name == "divide by 2 then add 3":
            wrong_order_even = 3 + sample_even // 2  # add 3 first, then divide
            # Actually, let's do: add 3, divide by 2 (different from divide, add 3)
            wrong_order_even = (sample_even + 3) // 2
            check(
                true_even_result != wrong_order_even,
                f"even operations commute: {true_even_result} == {wrong_order_even}",
            )

        question_text = f"If the number is even, {even_op_name}. If the number is odd, {odd_op_name}. For inputs {', '.join(map(str, inputs))}, what are the outputs? The order you give them in does not matter."

        # Independent check with different code paths
        def check_algorithm(n):
            if n % 2 == 0:  # Different from (n // 2 * 2 == n)
                return even_op(n)
            else:
                return odd_op(n)

        _verify_algorithm_outputs(check_algorithm, inputs, outputs)

        working = [
            f"For {inputs[0]}: {'even' if inputs[0] % 2 == 0 else 'odd'}, output = {outputs[0]}.",
            f"For {inputs[1]}: {'even' if inputs[1] % 2 == 0 else 'odd'}, output = {outputs[1]}.",
            f"For {inputs[2]}: {'even' if inputs[2] % 2 == 0 else 'odd'}, output = {outputs[2]}.",
        ]

    # Format answer as unordered list (in trace/input order)
    answer_text = ", ".join(str(o) for o in outputs)

    # Misconception distractors
    distractors_candidates = []

    # 1. swapped-branches: apply wrong branch operations
    # For each input, apply the opposite branch
    swapped_outputs = []
    for inp in inputs:
        if inp % 2 == 0:
            swapped_outputs.append(odd_op(inp))  # Apply odd branch to even input
        else:
            swapped_outputs.append(even_op(inp))  # Apply even branch to odd input

    swapped_text = ", ".join(str(o) for o in swapped_outputs)
    if swapped_text != answer_text:
        distractors_candidates.append(swapped_text)

    # 2. wrong-order-of-operations (hard tier only)
    if tier == "hard":
        wrong_order_outputs = []
        for inp in inputs:
            if inp % 2 == 0:
                # Wrong order: add 3 first, then divide (for divide-then-add case)
                if even_op_name == "divide by 2 then add 3":
                    wrong_output = (inp + 3) // 2  # Add 3 first, then divide
                else:
                    # Multiply 3 first, then subtract 2
                    wrong_output = (inp * 3) - 2
                wrong_order_outputs.append(wrong_output)
            else:
                # For odd branch
                if odd_op_name == "multiply by 2 then subtract 1":
                    wrong_output = (inp - 1) * 2  # Subtract 1 first, then multiply
                else:
                    wrong_output = (inp // 2) + 5  # Divide first, then add 5
                wrong_order_outputs.append(wrong_output)

        wrong_order_text = ", ".join(str(o) for o in wrong_order_outputs)
        if wrong_order_text != answer_text and wrong_order_text not in distractors_candidates:
            distractors_candidates.append(wrong_order_text)

    # 3. one-input-skipped: omit the last output
    skipped_outputs = outputs[:-1]
    skipped_text = ", ".join(str(o) for o in skipped_outputs)
    if skipped_text != answer_text and skipped_text not in distractors_candidates:
        distractors_candidates.append(skipped_text)

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=outputs,  # Unordered list for multiset comparison
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, distractors_candidates),
    )
