"""Primary-number pilot pair: two more generators for two genuinely new graded operations in the
``primary-number`` domain, both framework ``acara-f10-v9``.

- ``order_to_1000`` (AC9M2N01, prerequisites ``place-value-to-100``,
  ``order-numbers-to-10``) -- order a stated set of distinct whole numbers up
  to 1000, ascending, by numeral. Only the ordering-by-numeral
  operation is graded -- representing/naming/materials/number-line are not.
  Difficulty is driven by item count, closeness of values, and two genuine
  place-value traps: a shared-hundreds-digit pair (forces digit-by-digit
  tens/ones comparison, not just a leading-digit read) at ``medium`` and
  ``hard``, and a mix of 2- and 3-digit numbers (forces digit-*count*
  comparison, the same principle ``large_number_ordering``'s own hard tier
  used in ``ordered_sequence_build.py``, applied at this smaller
  range) at ``hard`` only. The sampled population is whole numbers 10-999
  (two- and three-digit): the curriculum's wording is "to at least 1000", which
  names the range this skill covers, not a requirement to literally sample the
  number 1000 itself; numbers beyond 1000 are excluded.

- ``odd_even_properties`` (AC9M4N02, prerequisite ``multiplication-facts``)
  -- given the *stated* parities of two operands (no concrete numbers shown
  at all) and a stated operation (addition, subtraction or multiplication),
  predict whether the result is odd or even. Classifying an actual number's parity is not
  graded (the parities are always supplied by the question) and an open
  "justify" clause is not graded either -- only the
  odd/even prediction itself. The correct label is looked up from a fixed
  parity table; verified independently by instantiating several concrete
  representative integers of each stated parity and checking their *real*
  arithmetic result's actual parity (``% 2``) matches the looked-up label,
  the same "rule-based construction, concrete-instance-based independent
  check" shape as ``foundations.order_of_operations``'s own manual
  precedence evaluator. Difficulty is driven by which operation(s) are in
  play and, at ``hard``, by chaining two operations in sequence (a genuine
  extra reasoning step -- there are no numbers shown at any difficulty, only
  parity labels and operation names).
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, pick_distractors, reversed_and_adjacent_swap

# -- order_to_1000 -----------------------------------------------------------

_TWO_DIGIT_RANGE = (10, 99)
_THREE_DIGIT_RANGE = (100, 999)
_LENGTH_RANGES = {2: _TWO_DIGIT_RANGE, 3: _THREE_DIGIT_RANGE}


def _hundreds_digit(n: int) -> int:
    return n // 100


def _min_gap(numbers: list[int]) -> int:
    ordered = sorted(numbers)
    return min(b - a for a, b in zip(ordered, ordered[1:]))


def _count_same_hundreds_pairs(numbers: list[int]) -> int:
    """Count pairs of *3-digit* numbers sharing the same hundreds digit --
    the place-value trap that forces tens/ones-level, digit-by-digit
    comparison rather than a leading-digit-only read."""
    three_digit = [n for n in numbers if n >= 100]
    return sum(
        1
        for i in range(len(three_digit))
        for j in range(i + 1, len(three_digit))
        if _hundreds_digit(three_digit[i]) == _hundreds_digit(three_digit[j])
    )


def _sample_order_to_1000_numbers(
    rng,
    count: int,
    *,
    digit_lengths: tuple[int, ...],
    min_gap: int | None,
    min_same_hundreds_pairs: int,
    max_attempts: int = 8000,
) -> list[int]:
    """Rejection-sample `count` distinct whole numbers from the requested
    digit-length pools (10-99 and/or 100-999), satisfying the requested
    difficulty shape."""
    if count < 2:
        raise ValueError("count must be at least 2")
    pool: list[int] = []
    for length in digit_lengths:
        lo, hi = _LENGTH_RANGES[length]
        pool.extend(range(lo, hi + 1))
    if count > len(pool):
        raise ValueError(f"cannot sample {count} distinct numbers from a pool of only {len(pool)}")

    for _ in range(max_attempts):
        if len(digit_lengths) > 1:
            # Hard tier: force at least one number from each requested
            # digit length to actually appear -- a naive independent choice
            # per item has a real (if small) chance of landing on the same
            # length every time, silently degenerating into the single-
            # length case.
            chosen_lengths = [rng.choice(digit_lengths) for _ in range(count)]
            if len(set(chosen_lengths)) < len(digit_lengths):
                missing = [length for length in digit_lengths if length not in chosen_lengths]
                for i, length in enumerate(missing):
                    chosen_lengths[i] = length
            numbers: list[int] = []
            seen: set[int] = set()
            ok = True
            for length in chosen_lengths:
                lo, hi = _LENGTH_RANGES[length]
                attempt = 0
                n = rng.randint(lo, hi)
                while n in seen:
                    n = rng.randint(lo, hi)
                    attempt += 1
                    if attempt > 500:
                        ok = False
                        break
                if not ok:
                    break
                seen.add(n)
                numbers.append(n)
            if not ok:
                continue
        else:
            numbers = list(rng.sample(pool, count))

        if min_gap is not None and _min_gap(numbers) < min_gap:
            continue
        if _count_same_hundreds_pairs(numbers) < min_same_hundreds_pairs:
            continue
        return numbers

    raise RuntimeError(
        "could not sample numbers satisfying the requested difficulty shape "
        f"(count={count}, digit_lengths={digit_lengths}, min_gap={min_gap}, "
        f"min_same_hundreds_pairs={min_same_hundreds_pairs}) in {max_attempts} attempts"
    )


def _verify_order_to_1000(sampled: list[int], ordered: list[int]) -> None:
    """Independent check, same discipline as ``ordered_sequence_build.
    _verify_large_number_order``: construction sorts numerically
    (``sorted(numbers)``); this check never re-runs that sort. It walks
    every pair (i, j) with i < j -- not just adjacent pairs -- and asserts
    ascending order via digit-count comparison first, then plain decimal
    string comparison for equal-length numbers, independent of whichever
    numeric key built ``ordered``."""
    check(sorted(sampled) == sorted(ordered), "ordered list is not a permutation of the sampled numbers")
    check(len(ordered) == len(set(ordered)), "sampled numbers must be distinct")
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            text_a, text_b = str(a), str(b)
            if len(text_a) != len(text_b):
                check(len(text_a) < len(text_b), f"{a} does not precede {b} by digit count")
            else:
                check(text_a < text_b, f"{a} does not precede {b} as equal-length decimal strings")


@generator(
    "order_to_1000",
    level=2,
    presets={
        "easy": dict(count=(3, 4), digit_lengths=(3,), min_gap=100, min_same_hundreds_pairs=0),
        "medium": dict(count=4, digit_lengths=(3,), min_gap=None, min_same_hundreds_pairs=1),
        "hard": dict(count=5, digit_lengths=(2, 3), min_gap=None, min_same_hundreds_pairs=1),
    },
)
def order_to_1000(rng, count=4, digit_lengths=(3,), min_gap=None, min_same_hundreds_pairs=0):
    """Order a stated set of distinct whole numbers up to 1000 (numerals
    only, no words), ascending. See the module docstring for the difficulty
    rationale."""
    if isinstance(count, (tuple, list)):
        if not count:
            raise ValueError("count sequence must not be empty")
        count = rng.choice(tuple(count))
    if count < 2:
        raise ValueError("count must be at least 2")
    digit_lengths = tuple(digit_lengths)
    unknown = [length for length in digit_lengths if length not in _LENGTH_RANGES]
    if not digit_lengths or unknown:
        raise ValueError(f"digit_lengths must be a non-empty subset of {sorted(_LENGTH_RANGES)}; got {digit_lengths!r}")

    numbers = _sample_order_to_1000_numbers(
        rng,
        count,
        digit_lengths=digit_lengths,
        min_gap=min_gap,
        min_same_hundreds_pairs=min_same_hundreds_pairs,
    )
    ordered = sorted(numbers)
    _verify_order_to_1000(numbers, ordered)

    display_order = list(numbers)
    rng.shuffle(display_order)
    display_text = ", ".join(str(n) for n in display_order)
    answer_text = ", ".join(str(n) for n in ordered)

    mixed_lengths = len(digit_lengths) > 1
    working = (
        [
            "If two numbers have a different number of digits, the one with more digits is always larger.",
            "Within numbers that have the same number of digits, compare digit by digit from the left "
            "(hundreds, then tens, then ones).",
            f"In order: {answer_text}.",
        ]
        if mixed_lengths
        else [
            "Compare the numbers digit by digit from the left (hundreds, then tens, then ones).",
            f"In order: {answer_text}.",
        ]
    )

    # Misconception distractors:
    #  - reversed-order: largest to smallest instead of smallest to largest.
    #  - adjacent-swap: correctly ordered overall, but the two middle values
    #    are swapped, as if two neighbouring values did not need comparing.
    #  - leading-digit-only (mixed-digit-length questions only): sorted by
    #    each numeral's leading digit alone, ignoring overall digit count --
    #    e.g. wrongly ranking "99" above "100" because "9" > "1".
    reversed_order, adjacent_swap = reversed_and_adjacent_swap(ordered)
    candidates = [
        ", ".join(str(n) for n in reversed_order),
        ", ".join(str(n) for n in adjacent_swap),
    ]
    if mixed_lengths:
        leading_digit_only = sorted(display_order, key=lambda n: str(n)[0])
        candidates.append(", ".join(str(n) for n in leading_digit_only))

    return dict(
        text=f"Arrange these numbers from smallest to largest: {display_text}.",
        latex=rf"\text{{Order: }} {display_text}",
        answer=list(ordered),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


# -- odd_even_properties ------------------------------------------------

# The standard parity table for +, - and *. Derived, not guessed:
#  addition/subtraction share one table (the parity of a sum and of a
#  difference of the same two operands is always identical -- adding or
#  subtracting an even number never changes parity, adding or subtracting an
#  odd number always flips it);
#  multiplication is odd only when *both* factors are odd (an even factor
#  always makes the product even).
_PARITY_TABLE: dict[str, dict[tuple[str, str], str]] = {
    "+": {
        ("odd", "odd"): "even",
        ("odd", "even"): "odd",
        ("even", "odd"): "odd",
        ("even", "even"): "even",
    },
    "-": {
        ("odd", "odd"): "even",
        ("odd", "even"): "odd",
        ("even", "odd"): "odd",
        ("even", "even"): "even",
    },
    "*": {
        ("odd", "odd"): "odd",
        ("odd", "even"): "even",
        ("even", "odd"): "even",
        ("even", "even"): "even",
    },
}

_OP_PHRASE = {"+": "plus", "-": "minus", "*": "multiplied by"}
_OTHER_LABEL = {"odd": "even", "even": "odd"}

# Concrete representative integers for each stated parity, used only by the
# independent check below -- never by the construction/lookup path.
_ODD_SAMPLES = (1, 3, 5, 7, 9)
_EVEN_SAMPLES = (0, 2, 4, 6, 8)
_CHECK_INSTANCES = 3  # "a handful"


def _lookup_parity(op: str, p1: str, p2: str) -> str:
    return _PARITY_TABLE[op][(p1, p2)]


def _sample_distinct_for_parity(parity: str, rng, n: int = _CHECK_INSTANCES) -> list[int]:
    """Sample `n` *distinct* concrete integers of the stated parity --
    ``rng.sample`` (no replacement), so every call genuinely tries several
    different concrete instances, not the same lucky pair repeated."""
    pool = _ODD_SAMPLES if parity == "odd" else _EVEN_SAMPLES
    return rng.sample(pool, n)


def _apply_real(op: str, a: int, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"unknown operation {op!r}; expected one of '+', '-', '*'")


def _real_parity(n: int) -> str:
    return "odd" if n % 2 else "even"


def _verify_single_op(op: str, p1: str, p2: str, predicted: str, rng) -> list[tuple[int, int, int]]:
    """Independent check, genuinely different from the table lookup used to
    construct ``predicted``: sample several concrete representative
    integers of each stated parity, compute the real arithmetic result, and
    assert its actual parity (`% 2`) matches the predicted label -- for
    several different concrete instances, not just one, so a table
    transcription error would be caught rather than coincidentally
    validated by a single lucky pair."""
    xs = _sample_distinct_for_parity(p1, rng)
    ys = _sample_distinct_for_parity(p2, rng)
    tried = []
    for x, y in zip(xs, ys):
        result = _apply_real(op, x, y)
        check(
            _real_parity(result) == predicted,
            f"{x} {op} {y} = {result}, which is {_real_parity(result)}, expected {predicted}",
        )
        tried.append((x, y, result))
    return tried


def _verify_chained_op(
    op1: str,
    p1: str,
    p2: str,
    op2: str,
    p3: str,
    predicted: str,
    rng,
) -> list[tuple[int, int, int, int, int]]:
    """Same discipline as `_verify_single_op`, extended to the chained
    (hard-tier) case: sample concrete integers for all three stated
    parities, compute the real two-step arithmetic result, and assert its
    actual parity matches the predicted label -- for several different
    concrete instances."""
    xs = _sample_distinct_for_parity(p1, rng)
    ys = _sample_distinct_for_parity(p2, rng)
    zs = _sample_distinct_for_parity(p3, rng)
    tried = []
    for x, y, z in zip(xs, ys, zs):
        intermediate = _apply_real(op1, x, y)
        final = _apply_real(op2, intermediate, z)
        check(
            _real_parity(final) == predicted,
            f"({x} {op1} {y}) {op2} {z} = {final}, which is {_real_parity(final)}, expected {predicted}",
        )
        tried.append((x, y, z, intermediate, final))
    return tried


@generator(
    "odd_even_properties",
    level=4,
    presets={
        "easy": dict(ops=("+",), chained=False),
        "medium": dict(ops=("-", "*"), chained=False),
        "hard": dict(ops=("+", "-", "*"), chained=True),
    },
)
def odd_even_properties(rng, ops=("+",), chained=False):
    """Given the stated parities of two (or, chained, three) operands and
    stated operation(s), predict whether the result is odd or even. No
    concrete numbers are ever shown to the learner -- only parity labels and
    operation names. See the module docstring for the difficulty
    rationale."""
    ops = tuple(ops)
    if not ops:
        raise ValueError("ops must be non-empty")
    unknown_ops = [op for op in ops if op not in _PARITY_TABLE]
    if unknown_ops:
        raise ValueError(f"unknown operations {unknown_ops!r}; expected a subset of {sorted(_PARITY_TABLE)}")

    parities = ("odd", "even")

    if not chained:
        op = rng.choice(ops)
        p1 = rng.choice(parities)
        p2 = rng.choice(parities)
        predicted = _lookup_parity(op, p1, p2)
        _verify_single_op(op, p1, p2, predicted, rng)

        text = f"An {p1} number {_OP_PHRASE[op]} an {p2} number -- is the result odd or even?"
        working = [
            f"{p1.capitalize()} {_OP_PHRASE[op]} {p2} always gives an {predicted} result.",
        ]
    else:
        if len(ops) < 2:
            raise ValueError("chained mode needs at least two candidate operations to pick from")
        op1, op2 = rng.choice(ops), rng.choice(ops)
        p1, p2, p3 = rng.choice(parities), rng.choice(parities), rng.choice(parities)
        intermediate = _lookup_parity(op1, p1, p2)
        predicted = _lookup_parity(op2, intermediate, p3)
        _verify_chained_op(op1, p1, p2, op2, p3, predicted, rng)

        text = (
            f"An {p1} number {_OP_PHRASE[op1]} an {p2} number, then that answer "
            f"{_OP_PHRASE[op2]} an {p3} number -- is the final result odd or even?"
        )
        working = [
            f"First step: {p1} {_OP_PHRASE[op1]} {p2} gives an {intermediate} result.",
            f"Second step: that {intermediate} result {_OP_PHRASE[op2]} {p3} gives an {predicted} result.",
        ]

    other_label = _OTHER_LABEL[predicted]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=predicted,
        answer_text=predicted,
        answer_latex=latex_answer(predicted, plain=True),
        working=working,
        # A genuine 2-option classification (odd vs. even): the sole
        # distractor is the other label -- there is no third option, so a
        # single correct "the other label" candidate is both necessary and
        # sufficient here (mathgen/grading/typed.py::mcq_options builds its option
        # list from [answer_text] + distractors).
        distractors=pick_distractors(predicted, [other_label], n=1),
    )
