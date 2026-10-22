"""Primary-number estimation cluster:
two generators for whole-number and percentage-based estimation skills, both
framework acara-f10-v9.

- ``estimate_counts_and_check_reasonableness`` (AC9M3N05, AC9M4N07, AC9M5N08,
  AC9M6N08, prerequisites ``round-decimals``, ``estimate-financial-total``) --
  "Estimate the result of a whole-number addition, subtraction or multiplication
  by rounding each operand to a stated place and computing." Every question
  states the rounding place(s) explicitly.

  **Rounding implementation**: uses half-up rounding (not Python's banker's
  rounding) via pure integer arithmetic. For each tier, the independent check
  uses string-digit inspection to verify the half-up rule independently from
  the arithmetic construction.

  Three tiers, escalating genuine reasoning steps:

  - ``easy``: round to the nearest **10**; two 2-digit operands (10..99); op
    in {+, -}; for -, if both rounded operands would give a negative result,
    swap them so result is always non-negative.
  - ``medium``: round to the nearest **100**; two 3-digit operands (100..999);
    op in {+, -, *} -- multiplying two already-rounded values is a qualitatively
    new combining step.
  - ``hard``: the two operands are rounded to **two different, explicitly
    stated places** (e.g. "round the first number to the nearest hundred and
    the second to the nearest ten, then add them") -- tracking two different
    rounding rules at once is the new reasoning step; op in {+, -, *};
    3-digit operands.

  Misconceptions, every value re-verified at every seed to differ in value
  from the correct answer; report measured collision rate per tier honestly:

  - ``rounds-down-always``: floors every operand to the stated place instead of
    applying half-up.
  - ``uses-unrounded-operands``: computes op on the raw, unrounded operands.
  - ``wrong-place-value``: rounds to one place coarser than instructed.
  - (hard tier only) ``same-place-for-both``: applies only the first stated
    place to both operands, ignoring the second operand's own different place.

- ``rational_percentage_estimation`` (AC9M6N08, prerequisites
  ``percentage-of-quantity``, ``round-decimals``) --
  "Estimate a percentage of a quantity using a
  stated benchmark percentage (10%, 25%, 50%)."

  **Honesty constraint**: this skill's every question has a stated benchmark
  percentage of the stated quantity that is **never** an integer, so the answer
  is a genuine approximation, never the exact value the existing skill
  percentage-of-quantity already grades. Assert this directly on a seed sweep:
  benchmark * quantity is never a multiple of 100.

  **Rounding implementation**: for benchmark 50%, the exact value 50% of Q is
  Q / 2 -- for this to be non-integer at all, Q must be odd, which makes the
  fractional part always exactly 0.5, i.e. every 50%-benchmark seed sits
  exactly on the rounding midpoint. There is no way to avoid this; handle it
  correctly by construction (round half up). Use exact Fraction arithmetic,
  e.g. floor(exact_value + Fraction(1, 2)) where exact_value is a Fraction.

  Three tiers, escalating genuine reasoning steps:

  - ``easy``: benchmark 10%; two-digit quantity Q (10..99) not a multiple of 10.
  - ``medium``: benchmark 25%; quantity Q (10..199) not a multiple of 4.
  - ``hard``: benchmark chosen at random from {10%, 25%, 50%}; three-digit
    quantity Q (100..999) satisfying the non-integer constraint for whichever
    benchmark is chosen -- the learner must first identify which mental-math
    strategy applies.

  Misconceptions, report measured collision/coverage rate per tier honestly:

  - ``wrong-benchmark-divisor``: applies a structurally wrong mental-math
    divisor (10% treated as divide-by-100; 25% treated as divide-by-2;
    50% treated as divide-by-4).
  - ``always-floor``: never rounds up even at/past the midpoint (floor
    exact_value); collides with correct whenever fractional part is < 0.5;
    on 50% branch this is never (fractional part always 0.5).
  - ``always-ceiling``: always rounds up regardless of fractional part
    (floor exact_value + 1); collides whenever fractional part is >= 0.5;
    on 50% branch always collides (half-up and ceiling agree at 0.5).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors


# -- estimate_counts_and_check_reasonableness --------------------------------


def round_half_up_to_place(value: int, place: int) -> int:
    """Round a whole number half-up to the given place using pure integer
    arithmetic. place=10 means nearest 10, place=100 means nearest 100, etc.

    Example: round_half_up_to_place(25, 10) == 30 (round UP at midpoint, not
    banker's rounding)."""
    return ((value + place // 2) // place) * place


def verify_round_half_up_string_inspection(value: int, place: int, expected: int) -> None:
    """Independent check: verify the rounded value via string-digit inspection,
    a genuinely different route from the arithmetic construction. Inspect the
    digit immediately after the rounding place and decide up/down independently.
    For place=10, look at ones digit; for place=100, look at tens digit; etc."""
    value_str = str(value)
    place_str = str(place)

    # drop_count = number of zeros in place string
    # For place=10, drop_count=1 (look at ones digit, rightmost position)
    # For place=100, drop_count=2 (look at tens digit, second from right)
    drop_count = len(place_str) - 1

    # Truncate to the place
    truncated = (value // place) * place

    # Get the digit at the rounding boundary
    # For place=10, we want value_str[-1] (ones)
    # For place=100, we want value_str[-2] (tens)
    if len(value_str) >= drop_count + 1:
        # Extract the digit at position drop_count from the right
        digit_to_check = int(value_str[-drop_count]) if drop_count > 0 else int(value_str[-1])
    else:
        # Not enough digits; the digit below the place is 0
        digit_to_check = 0

    # Decide whether to round up
    if digit_to_check >= 5:
        expected_rounded = truncated + place
    else:
        expected_rounded = truncated

    check(
        expected_rounded == expected,
        f"rounding {value} to place {place} via digit inspection: "
        f"digit to check is {digit_to_check}, truncated is {truncated}, "
        f"expected {expected_rounded} but got {expected}",
    )


def _sample_whole(rng, lo: int, hi: int) -> int:
    """Sample a whole number in [lo, hi]."""
    return rng.randint(lo, hi)


def _whole_estimate_easy(rng) -> dict:
    """Easy tier: round to nearest 10, two 2-digit operands."""
    place = 10
    a = _sample_whole(rng, 10, 99)
    b = _sample_whole(rng, 10, 99)

    round_a = round_half_up_to_place(a, place)
    round_b = round_half_up_to_place(b, place)
    verify_round_half_up_string_inspection(a, place, round_a)
    verify_round_half_up_string_inspection(b, place, round_b)

    op = rng.choice(["+", "-"])
    if op == "+":
        exact = a + b
        estimate = round_a + round_b
        rounds_down_always = ((a // place) * place) + ((b // place) * place)
        uses_unrounded = a + b
        wrong_place = round_half_up_to_place(a, 100) + round_half_up_to_place(b, 100)
        op_text = "adding"
    else:  # op == "-"
        # Ensure non-negative result by swapping if necessary
        if round_a < round_b:
            a, b = b, a
            round_a, round_b = round_b, round_a
        exact = a - b
        estimate = round_a - round_b
        rounds_down_always = ((a // place) * place) - ((b // place) * place)
        uses_unrounded = a - b
        wrong_place = round_half_up_to_place(a, 100) - round_half_up_to_place(b, 100)
        op_text = "subtracting"

    text = f"Estimate {a} {_display_op(op)} {b} by rounding each number to the nearest 10, then {op_text}."

    candidates = [rounds_down_always, uses_unrounded, wrong_place]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=estimate,
        answer_text=str(estimate),
        answer_latex=str(estimate),
        working=[
            f"{a} rounds to {round_a} and {b} rounds to {round_b}.",
            f"{round_a} {_display_op(op)} {round_b} = {estimate}.",
        ],
        distractors=pick_distractors(str(estimate), [str(c) for c in candidates]),
    )


def _whole_estimate_medium(rng) -> dict:
    """Medium tier: round to nearest 100, two 3-digit operands, op in {+, -, *}."""
    place = 100
    a = _sample_whole(rng, 100, 999)
    b = _sample_whole(rng, 100, 999)

    round_a = round_half_up_to_place(a, place)
    round_b = round_half_up_to_place(b, place)
    verify_round_half_up_string_inspection(a, place, round_a)
    verify_round_half_up_string_inspection(b, place, round_b)

    op = rng.choice(["+", "-", "*"])

    if op == "+":
        exact = a + b
        estimate = round_a + round_b
        rounds_down_always = ((a // place) * place) + ((b // place) * place)
        uses_unrounded = a + b
        wrong_place = round_half_up_to_place(a, 1000) + round_half_up_to_place(b, 1000)
        op_text = "adding"
    elif op == "-":
        # Ensure non-negative result by swapping if necessary
        if round_a < round_b:
            a, b = b, a
            round_a, round_b = round_b, round_a
        exact = a - b
        estimate = round_a - round_b
        rounds_down_always = ((a // place) * place) - ((b // place) * place)
        uses_unrounded = a - b
        wrong_place = round_half_up_to_place(a, 1000) - round_half_up_to_place(b, 1000)
        op_text = "subtracting"
    else:  # op == "*"
        exact = a * b
        estimate = round_a * round_b
        rounds_down_always = ((a // place) * place) * ((b // place) * place)
        uses_unrounded = a * b
        wrong_place = round_half_up_to_place(a, 1000) * round_half_up_to_place(b, 1000)
        op_text = "multiplying"

    text = f"Estimate {a} {_display_op(op)} {b} by rounding each number to the nearest 100, then {op_text}."

    candidates = [rounds_down_always, uses_unrounded, wrong_place]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=estimate,
        answer_text=str(estimate),
        answer_latex=str(estimate),
        working=[
            f"{a} rounds to {round_a} and {b} rounds to {round_b}.",
            f"{round_a} {_display_op(op)} {round_b} = {estimate}.",
        ],
        distractors=pick_distractors(str(estimate), [str(c) for c in candidates]),
    )


def _display_op(op: str) -> str:
    """The sign a learner reads: the multiplication sign, never an asterisk."""
    return "\u00d7" if op == "*" else op


def _whole_estimate_hard(rng) -> dict:
    """Hard tier: two different rounding places explicitly stated, 3-digit operands."""
    a = _sample_whole(rng, 100, 999)
    b = _sample_whole(rng, 100, 999)

    # Randomly choose which operand gets which place
    places_order = rng.choice(["100_10", "10_100"])
    if places_order == "100_10":
        place_a, place_b = 100, 10
        place_a_text, place_b_text = "hundred", "ten"
    else:
        place_a, place_b = 10, 100
        place_a_text, place_b_text = "ten", "hundred"

    round_a = round_half_up_to_place(a, place_a)
    round_b = round_half_up_to_place(b, place_b)
    verify_round_half_up_string_inspection(a, place_a, round_a)
    verify_round_half_up_string_inspection(b, place_b, round_b)

    op = rng.choice(["+", "-", "*"])

    if op == "+":
        exact = a + b
        estimate = round_a + round_b
        rounds_down_always = ((a // place_a) * place_a) + ((b // place_b) * place_b)
        uses_unrounded = a + b
        same_place_for_both = round_half_up_to_place(a, place_a) + round_half_up_to_place(b, place_a)
        # Coarser places for wrong-place-value
        coarser_place_a = place_a * 10
        coarser_place_b = place_b * 10
        wrong_place_value = round_half_up_to_place(a, coarser_place_a) + round_half_up_to_place(b, coarser_place_b)
        op_text = "adding"
        op_sym = "+"
    elif op == "-":
        # Ensure non-negative result by swapping if necessary
        if round_a < round_b:
            a, b = b, a
            round_a, round_b = round_b, round_a
            place_a, place_b = place_b, place_a
            place_a_text, place_b_text = place_b_text, place_a_text
        exact = a - b
        estimate = round_a - round_b
        rounds_down_always = ((a // place_a) * place_a) - ((b // place_b) * place_b)
        uses_unrounded = a - b
        same_place_for_both = round_half_up_to_place(a, place_a) - round_half_up_to_place(b, place_a)
        # Coarser places for wrong-place-value
        coarser_place_a = place_a * 10
        coarser_place_b = place_b * 10
        wrong_place_value = round_half_up_to_place(a, coarser_place_a) - round_half_up_to_place(b, coarser_place_b)
        op_text = "subtracting"
        op_sym = "-"
    else:  # op == "*"
        exact = a * b
        estimate = round_a * round_b
        rounds_down_always = ((a // place_a) * place_a) * ((b // place_b) * place_b)
        uses_unrounded = a * b
        same_place_for_both = round_half_up_to_place(a, place_a) * round_half_up_to_place(b, place_a)
        # Coarser places for wrong-place-value
        coarser_place_a = place_a * 10
        coarser_place_b = place_b * 10
        wrong_place_value = round_half_up_to_place(a, coarser_place_a) * round_half_up_to_place(b, coarser_place_b)
        op_text = "multiplying"
        op_sym = "*"

    text = (
        f"Estimate {a} {_display_op(op_sym)} {b} by rounding the first number to the nearest {place_a_text} "
        f"and the second to the nearest {place_b_text}, then {op_text}."
    )

    candidates = [rounds_down_always, uses_unrounded, wrong_place_value, same_place_for_both]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=estimate,
        answer_text=str(estimate),
        answer_latex=str(estimate),
        working=[
            f"{a} rounds to {round_a} (nearest {place_a_text}) and {b} rounds to {round_b} (nearest {place_b_text}).",
            f"{round_a} {_display_op(op_sym)} {round_b} = {estimate}.",
        ],
        distractors=pick_distractors(str(estimate), [str(c) for c in candidates]),
    )


@generator(
    "estimate_counts_and_check_reasonableness",
    level=10,
    presets={
        "easy": {"tier": "easy"},
        "medium": {"tier": "medium"},
        "hard": {"tier": "hard"},
    },
)
def estimate_counts_and_check_reasonableness(rng, tier="easy"):
    """Estimate a whole-number addition, subtraction or multiplication by
    rounding each operand to a stated place and computing. See the module
    docstring for the full tier/misconception design."""
    if tier == "easy":
        return _whole_estimate_easy(rng)
    elif tier == "medium":
        return _whole_estimate_medium(rng)
    elif tier == "hard":
        return _whole_estimate_hard(rng)
    else:
        raise ValueError(f"unknown tier {tier!r}")


# -- rational_percentage_estimation ------------------------------------------


def _get_benchmark_quantities(benchmark: int, rng) -> tuple[int, int]:
    """For a given benchmark percentage, get a valid quantity that makes the
    result non-integer. Returns (benchmark, quantity)."""
    if benchmark == 10:
        # 10% of Q is Q/10, non-integer when Q not multiple of 10
        while True:
            q = rng.randint(10, 99)
            if q % 10 != 0 and benchmark * q % 100 != 0:
                return benchmark, q
    elif benchmark == 25:
        # 25% of Q is Q/4, non-integer when Q not multiple of 4
        while True:
            q = rng.randint(10, 199)
            if q % 4 != 0 and benchmark * q % 100 != 0:
                return benchmark, q
    elif benchmark == 50:
        # 50% of Q is Q/2, non-integer when Q is odd
        while True:
            q = rng.randint(100, 999)
            if q % 2 != 0 and benchmark * q % 100 != 0:
                return benchmark, q
    else:
        raise ValueError(f"unknown benchmark {benchmark}")


def _decimal_text(value: Fraction) -> str:
    """Exact decimal text of a Fraction whose denominator divides 100."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _percentage_estimate_easy(rng) -> dict:
    """Easy tier: benchmark 10%, two-digit quantity."""
    benchmark = 10
    # Q must not be multiple of 10 to ensure 10% of Q is non-integer
    while True:
        quantity = rng.randint(10, 99)
        if quantity % 10 != 0:
            break

    # Verify the result is actually non-integer
    check(benchmark * quantity % 100 != 0, f"10% of {quantity} is an integer")

    # Exact value is (benchmark * quantity) / 100
    exact_frac = Fraction(benchmark * quantity, 100)
    # Round half up: floor(exact_frac + 0.5)
    estimate = int(exact_frac + Fraction(1, 2))

    # Independent check via divmod
    whole, remainder = divmod(benchmark * quantity, 100)
    expected_estimate = whole + (1 if remainder * 2 >= 100 else 0)
    check(expected_estimate == estimate, f"percentage estimate mismatch for {benchmark}% of {quantity}")

    # Misconceptions
    # wrong_benchmark_divisor: treats 10% as divide-by-100 instead of divide-by-10
    wrong_benchmark_divisor = quantity // 100
    always_floor = int(exact_frac)
    always_ceiling = int(exact_frac) + (1 if exact_frac.denominator != 1 else 0)

    text = f"Estimate 10% of {quantity}, rounded to the nearest whole number (round a half up)."

    candidates = [wrong_benchmark_divisor, always_floor, always_ceiling]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=estimate,
        answer_text=str(estimate),
        answer_latex=str(estimate),
        working=[f"{benchmark}% of {quantity} = {_decimal_text(exact_frac)}, which rounds to {estimate}."],
        distractors=pick_distractors(str(estimate), [str(c) for c in candidates]),
    )


def _percentage_estimate_medium(rng) -> dict:
    """Medium tier: benchmark 25%, quantity not multiple of 4."""
    benchmark = 25
    # Q must not be multiple of 4 to ensure 25% of Q is non-integer
    while True:
        quantity = rng.randint(10, 199)
        if quantity % 4 != 0 and benchmark * quantity % 100 != 0:
            break

    exact_frac = Fraction(benchmark * quantity, 100)
    estimate = int(exact_frac + Fraction(1, 2))

    # Independent check
    whole, remainder = divmod(benchmark * quantity, 100)
    expected_estimate = whole + (1 if remainder * 2 >= 100 else 0)
    check(expected_estimate == estimate, f"percentage estimate mismatch for {benchmark}% of {quantity}")

    # Misconceptions
    # wrong_benchmark_divisor: treats 25% as divide-by-2 instead of divide-by-4
    wrong_benchmark_divisor = quantity // 2
    always_floor = int(exact_frac)
    always_ceiling = int(exact_frac) + (1 if exact_frac.denominator != 1 else 0)

    text = f"Estimate 25% of {quantity}, rounded to the nearest whole number (round a half up)."

    candidates = [wrong_benchmark_divisor, always_floor, always_ceiling]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=estimate,
        answer_text=str(estimate),
        answer_latex=str(estimate),
        working=[f"{benchmark}% of {quantity} = {_decimal_text(exact_frac)}, which rounds to {estimate}."],
        distractors=pick_distractors(str(estimate), [str(c) for c in candidates]),
    )


def _percentage_estimate_hard(rng) -> dict:
    """Hard tier: random benchmark from {10%, 25%, 50%}, three-digit quantity."""
    benchmark = rng.choice([10, 25, 50])

    if benchmark == 10:
        while True:
            quantity = rng.randint(100, 999)
            if quantity % 10 != 0 and benchmark * quantity % 100 != 0:
                break
    elif benchmark == 25:
        while True:
            quantity = rng.randint(100, 999)
            if quantity % 4 != 0 and benchmark * quantity % 100 != 0:
                break
    else:  # benchmark == 50
        while True:
            quantity = rng.randint(100, 999)
            if quantity % 2 != 0 and benchmark * quantity % 100 != 0:
                break

    exact_frac = Fraction(benchmark * quantity, 100)
    estimate = int(exact_frac + Fraction(1, 2))

    # Independent check
    whole, remainder = divmod(benchmark * quantity, 100)
    expected_estimate = whole + (1 if remainder * 2 >= 100 else 0)
    check(expected_estimate == estimate, f"percentage estimate mismatch for {benchmark}% of {quantity}")

    # Misconceptions depend on benchmark
    if benchmark == 10:
        wrong_benchmark_divisor = quantity // 100
    elif benchmark == 25:
        wrong_benchmark_divisor = quantity // 2
    else:  # benchmark == 50
        wrong_benchmark_divisor = quantity // 4

    always_floor = int(exact_frac)
    always_ceiling = int(exact_frac) + (1 if exact_frac.denominator != 1 else 0)

    text = f"Estimate {benchmark}% of {quantity}, rounded to the nearest whole number (round a half up)."

    candidates = [wrong_benchmark_divisor, always_floor, always_ceiling]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=estimate,
        answer_text=str(estimate),
        answer_latex=str(estimate),
        working=[f"{benchmark}% of {quantity} = {_decimal_text(exact_frac)}, which rounds to {estimate}."],
        distractors=pick_distractors(str(estimate), [str(c) for c in candidates]),
    )


@generator(
    "rational_percentage_estimation",
    level=10,
    presets={
        "easy": {"tier": "easy"},
        "medium": {"tier": "medium"},
        "hard": {"tier": "hard"},
    },
)
def rational_percentage_estimation(rng, tier="easy"):
    """Estimate a percentage of a quantity using a stated benchmark percentage
    (10%, 25%, 50%), rounded to the nearest whole number. See the module
    docstring for the full tier/misconception design."""
    if tier == "easy":
        return _percentage_estimate_easy(rng)
    elif tier == "medium":
        return _percentage_estimate_medium(rng)
    elif tier == "hard":
        return _percentage_estimate_hard(rng)
    else:
        raise ValueError(f"unknown tier {tier!r}")
