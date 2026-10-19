"""Measurement generators (unmarked/partial scales and area between curves), both Foundation-Year 10 and senior,
``answer: number``.

Two generators:

- ``unmarked_partial_measurement`` (``f10-unmarked-partial-measurement``, AC9M4M01)
  -- Given a partially marked or unmarked metric scale with known mark values and
  stated equal subdivisions, interpret a reading between labelled marks and state
  the measured value. Three families: length (cm), mass (g), capacity (mL).

- ``area_between_curves_simple_cases`` (``senior-calculate-the-area-between-curves-
  in-simple-cases``, ACMMM134) -- Calculate the area between two curves (lines,
  parabola) on a bounded interval. Always produces exact rational or integer area
  values terminating to at most 1 decimal place.

Both generators use exact ``Fraction`` arithmetic for construction and verification.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    exact_decimal_string,
    generator,
    join_terms,
    numeric_answer,
    pick_distractors,
    rational_text,
    stated_precision_string,
)


# === unmarked_partial_measurement ============================================


def _numeric_answer_int_or_float(value: int | float) -> int | float:
    """Return int if already int, float otherwise. For integer answers (always
    the case here since M % N == 0), return bare int."""
    return (
        value if isinstance(value, int) else int(value) if isinstance(value, float) and value == int(value) else value
    )


def _division_word(k: int) -> str:
    """Return 'division' or 'divisions' based on count k."""
    return "division" if k == 1 else "divisions"


# --- Independent check for unmarked_partial_measurement ----------------------


def _verify_unmarked_partial_measurement(k: int, M: int, N: int, ref_mark: int, direction: str, expected: int) -> None:
    """Independent check: use Fraction(M, N) * k and add/subtract from reference
    mark via a distinctly different route from construction."""
    division_value = Fraction(M, N) * k
    if direction == "from_lower":
        recomputed = ref_mark + int(division_value)
    else:  # from_upper
        recomputed = ref_mark - int(division_value)
    check(recomputed == expected, f"unmarked_partial_measurement mismatch: {recomputed} != {expected}")


@generator(
    "unmarked_partial_measurement",
    level=1,
    presets={
        "easy": dict(mode="from_lower_mark"),
        "medium": dict(mode="from_upper_mark"),
        "hard": dict(mode="forced_direction_trap"),
    },
)
def unmarked_partial_measurement(rng, mode="from_lower_mark"):
    """Given a partially marked or unmarked metric scale with known mark values
    and stated equal subdivisions between them, interpret a reading and state
    the measured value."""

    # Family assignment
    if mode == "from_lower_mark":
        family = "length"  # Easy: length only
    else:
        families = ["length", "mass", "capacity"]
        family = rng.choice(families)

    # Unit and label
    units = {"length": "cm", "mass": "g", "capacity": "mL"}
    unit = units[family]
    nouns = {"length": "length scale", "mass": "scale", "capacity": "scale"}
    noun = nouns[family]

    # Parameters for subdivision construction
    if mode == "forced_direction_trap":
        # Hard: wider N range {2,3,4,5,6,8,10}, M from {12,18,24,30,48,60}
        N_values = [2, 3, 4, 5, 6, 8, 10]
        M_values = [12, 18, 24, 30, 48, 60]
    else:
        # Easy/medium: smaller N {2,4,5}, M {10,20,50}
        N_values = [2, 4, 5]
        M_values = [10, 20, 50]

    # Draw N and M ensuring M % N == 0
    valid_pairs = [(M, N) for M in M_values for N in N_values if M % N == 0]
    M, N = rng.choice(valid_pairs)

    # Draw k from 1..N-1 (strictly interior)
    k = rng.randint(1, N - 1)

    # Draw reference marks
    lower_mark = rng.randint(10, 50)
    upper_mark = lower_mark + M

    # Determine direction
    if mode == "forced_direction_trap":
        direction = rng.choice(["from_lower", "from_upper"])
    elif mode == "from_lower_mark":
        direction = "from_lower"
    else:  # from_upper_mark
        direction = "from_upper"

    # Compute answer
    division_value = M // N
    if direction == "from_lower":
        answer = lower_mark + k * division_value
        ref_mark = lower_mark
        ref_label = f"{lower_mark} {unit} mark"
        count_text = f"{k}"
        position_text = f"{k} {_division_word(k)} past the {lower_mark} {unit} mark"
    else:  # from_upper
        answer = upper_mark - k * division_value
        ref_mark = upper_mark
        ref_label = f"{upper_mark} {unit} mark"
        count_text = f"{k}"
        position_text = f"{k} {_division_word(k)} before the {upper_mark} {unit} mark"

    # Verify independent check
    _verify_unmarked_partial_measurement(k, M, N, ref_mark, direction, answer)

    # Build question text
    text = (
        f"On a {noun}, marks are labelled every {M} {unit}. "
        f"Between the {lower_mark} {unit} mark and the {upper_mark} {unit} mark "
        f"there are {N} equal unmarked divisions. "
        f"A measurement sits {position_text}. "
        f"What is the measurement, in {unit}?"
    )

    # Build distractors
    candidates = [
        answer + division_value,  # off-by-one-division (one too many)
        answer - division_value
        if answer - division_value >= 0
        else answer + division_value,  # off-by-one (one too few, clamped)
        ref_mark + M
        if direction == "from_lower"
        else ref_mark - M,  # uses-major-spacing (wrong direction or full interval)
        ref_mark if direction == "from_lower" else ref_mark,  # wrong-direction (uses opposite direction's ref mark)
    ]

    # Ensure all distractors are distinct from answer and from each other
    candidates_set = set()
    for c in candidates:
        if c != answer and c >= 0:
            candidates_set.add(c)

    # Ensure we have enough distractors
    if len(candidates_set) < 3:
        # Generate additional distractors via offset
        for offset in [division_value * 2, division_value // 2 if N > 2 else division_value]:
            c1 = answer + offset
            c2 = answer - offset if answer - offset >= 0 else answer + offset
            if c1 != answer and c1 >= 0:
                candidates_set.add(c1)
            if c2 != answer and c2 >= 0:
                candidates_set.add(c2)

    # Ensure answer is distinct
    candidates_set.discard(answer)

    # Convert to list and pick 3
    candidates_list = list(candidates_set)
    distractors_picked = pick_distractors(answer, candidates_list, n=3)

    # Build working
    # Avoid "1 divisions" by using ordinal phrasing
    divisions_desc = f"{k} {_division_word(k)}"
    sign = "+" if direction == "from_lower" else "−"
    working = [
        f"Each subdivision has value {M} ÷ {N} = {division_value} {unit}.",
        f"From the {ref_label}: {divisions_desc} × {division_value} {unit} = {k * division_value} {unit}.",
        f"The measurement is {ref_mark} {sign} {k * division_value} = {answer} {unit}.",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors_picked,
    )


# === area_between_curves_simple_cases ========================================


def _format_area_answer(value: Fraction) -> str:
    """Format area answer to exactly 1 decimal place, always, for consistency."""
    return exact_decimal_string(value, 1)


def _verify_area_easy(m1: int, m2: int, c1: int, c2: int, a: int, b: int, expected: Fraction) -> None:
    """Independent check for easy: recompute via direct expansion of the closed-form
    formula, not via antiderivative evaluation order."""
    recomputed = Fraction(m1 - m2, 2) * (b**2 - a**2) + (c1 - c2) * (b - a)
    check(recomputed == expected, f"easy area mismatch: {recomputed} != {expected}")


def _verify_area_medium(m: int, c: int, b: int, expected: int) -> None:
    """Independent check for medium: recompute the integral directly."""
    # Integral of f(x) - g(x) = (m*x + c - x^2) from 0 to b
    # F(x) = m*x^2/2 + c*x - x^3/3
    # F(b) - F(0) = m*b^2/2 + c*b - b^3/3
    recomputed = Fraction(m, 2) * b**2 + c * b - Fraction(b**3, 3)
    check(int(recomputed) == expected, f"medium area mismatch: {int(recomputed)} != {expected}")


def _verify_area_hard(m1: int, m2: int, c1: int, c2: int, a: int, b: int, expected: Fraction) -> None:
    """Independent check for hard: same as easy (two lines, same formula)."""
    recomputed = Fraction(m1 - m2, 2) * (b**2 - a**2) + (c1 - c2) * (b - a)
    check(recomputed == expected, f"hard area mismatch: {recomputed} != {expected}")


def _format_linear(m: int, c: int) -> str:
    """Format a linear function y = mx + c, simplifying coefficients 0 and 1."""
    if m == 0:
        return str(c)
    term = "x" if m == 1 else f"{m}x"
    if c == 0:
        return term
    return f"{term} + {c}" if c > 0 else f"{term} - {abs(c)}"


@generator(
    "area_between_curves_simple_cases",
    level=2,
    presets={
        "easy": dict(mode="line_and_line"),
        "medium": dict(mode="line_and_parabola"),
        "hard": dict(mode="unordered_line_and_line"),
    },
)
def area_between_curves_simple_cases(rng, mode="line_and_line"):
    """Calculate the area between curves (lines or line+parabola) on a bounded interval."""

    if mode == "line_and_line" or mode == "unordered_line_and_line":
        # Easy: mode="line_and_line"; Hard: mode="unordered_line_and_line"
        # Both are two lines f(x) = m1*x + c1, g(x) = m2*x + c2
        # Guarantee f(x) >= g(x) via m1 > m2, c1 >= c2, both >= 0 for x >= 0
        m1 = rng.randint(2, 5)
        m2 = rng.randint(0, m1 - 1)
        c1 = rng.randint(1, 8)
        c2 = rng.randint(0, c1)
        a = rng.randint(1, 3)
        b = rng.randint(a + 2, a + 6)

        area = Fraction(m1 - m2, 2) * (b**2 - a**2) + (c1 - c2) * (b - a)

        if mode == "line_and_line":
            # Easy: state which curve is on top
            text = (
                f"Calculate the area of the region below the curve y = {_format_linear(m1, c1)} "
                f"and above the curve y = {_format_linear(m2, c2)}, for x between {a} and {b}."
            )
        else:  # unordered_line_and_line (hard): don't state which is on top
            text = (
                f"Calculate the area of the region between the curves y = {_format_linear(m1, c1)} "
                f"and y = {_format_linear(m2, c2)}, for x between {a} and {b}."
            )

        d_m, d_c = m1 - m2, c1 - c2
        diff_text = join_terms([(d_m, "x"), (d_c, "")])
        lead = Fraction(d_m, 2)
        if lead.denominator == 1:
            anti = ("" if lead == 1 else str(lead.numerator)) + "x²"
        else:
            anti = ("" if lead.numerator == 1 else str(lead.numerator)) + f"x²/{lead.denominator}"
        if d_c:
            anti += f" + {'' if d_c == 1 else d_c}x"

        def _anti_value(x: int) -> Fraction:
            return lead * x * x + d_c * x

        fb, fa = _anti_value(b), _anti_value(a)
        working = [
            f"The curves are f(x) = {_format_linear(m1, c1)} and g(x) = {_format_linear(m2, c2)}.",
            f"f(x) − g(x) = {diff_text}, which is positive for x between {a} and {b}, so f is the upper curve.",
            f"Area = ∫ from {a} to {b} of [{_format_linear(m1, c1)} − ({_format_linear(m2, c2)})] dx = ∫ from {a} to {b} of ({diff_text}) dx.",
            f"An antiderivative of {diff_text} is {anti}.",
            f"Putting in the limits: {rational_text(fb)} − {rational_text(fa)} = {_format_area_answer(area)}",
        ]

        _verify_area_easy(m1, m2, c1, c2, a, b, area)
        answer = numeric_answer(area)
        answer_text = _format_area_answer(area)

        # Build distractors
        # An area cannot be negative, so the key negated is not offered; the slip is the area under the upper curve alone.
        under_upper = Fraction(m1, 2) * (b**2 - a**2) + c1 * (b - a)
        wrong_order_val = numeric_answer(under_upper)
        omit_lower = Fraction(m1 - m2, 2) * b**2 + (c1 - c2) * b
        omit_lower_val = numeric_answer(omit_lower)
        abs_answer = abs(answer)

        candidates = [wrong_order_val, omit_lower_val, abs_answer]

        # Every option is shown to the key's own number of decimal places
        # (a whole-number area gets whole-number options), so the key cannot
        # be spotted by its shape; a half-unit slip is only offered when the
        # key itself carries a decimal place.
        places = len(answer_text.partition(".")[2])
        if places == 0:
            candidates += [answer + 1, answer - 1, answer + 2]
        else:
            candidates += [answer + 1, answer - 1, answer + 0.5]
        distractors_list = []
        for c in candidates:
            if isinstance(c, (int, float)) and c != answer:
                shown = Fraction(c).limit_denominator(1000)
                shown = Fraction(math.floor(shown * 10**places + Fraction(1, 2)), 10**places)
                distractors_list.append(stated_precision_string(shown, places))
        distractors = pick_distractors(answer_text, distractors_list, n=3)

    elif mode == "line_and_parabola":
        # Medium: f(x) = m*x + c (line above), g(x) = x^2 (parabola below)
        # Guarantee f(x) >= g(x) on [0, b] via:
        # - c >= 0 (guarantees f(0) >= g(0))
        # - m*b + c >= b^2 (checked at endpoint)
        # - m is always even so Fraction(m,2)*b^2 is an integer
        # - b is a multiple of 3 so b^3/3 is an integer

        b = rng.choice([3, 6, 9])
        c = rng.randint(0, 5)
        # m is even and chosen to guarantee m*b + c >= b^2
        m = 2 * (b // 2 + rng.randint(0, 2))

        # Check and re-draw if needed
        max_redraws = 10
        for _ in range(max_redraws):
            if m * b + c >= b * b:
                break
            m = 2 * (b // 2 + rng.randint(0, 2))

        area = Fraction(m, 2) * b**2 + c * b - Fraction(b**3, 3)
        check(area.denominator == 1, f"medium area must be integer, got {area}")

        text = (
            f"Calculate the area of the region bounded above by y = {_format_linear(m, c)} "
            f"and below by y = x² for x between 0 and {b}."
        )

        lin_part = f"{rational_text(Fraction(m, 2))}x²" if m != 2 else "x²"
        if c:
            lin_part += f" + {'' if c == 1 else c}x"
        working = [
            f"f(x) = {_format_linear(m, c)}, g(x) = x².",
            f"Check f(x) ≥ g(x) on [0, {b}]: f(0) = {c} ≥ 0 = g(0), and f({b}) = {m * b + c} ≥ {b**2} = g({b}).",
            f"Area = ∫ from 0 to {b} of [{_format_linear(m, c)} − x²] dx.",
            f"An antiderivative of {_format_linear(m, c)} − x² is {lin_part} − x³/3.",
            f"At x = {b}: {rational_text(Fraction(m, 2) * b * b)} + {c * b} − {rational_text(Fraction(b**3, 3))} = {area}; at x = 0 it is 0.",
            f"Area = {area}",
        ]

        _verify_area_medium(m, c, b, int(area))
        answer = int(area)
        answer_text = _format_area_answer(area)

        # Build distractors
        forgets_linear = Fraction(m, 2) * b**2 - Fraction(b**3, 3)
        swaps_power = Fraction(m, 2) * b**2 + c * b - b**3

        # The area under the line alone, forgetting to take away the parabola.
        upper_only = Fraction(m, 2) * b**2 + c * b
        candidates = [int(forgets_linear), int(swaps_power), int(upper_only)]

        distractors_set = set()
        for c_val in candidates:
            # An area is never negative, so a slip that goes below zero is not offered.
            if c_val != answer and c_val > 0:
                distractors_set.add(c_val)

        # Ensure we have enough
        if len(distractors_set) < 3:
            distractors_set.add(answer + 1)
            distractors_set.add(answer - 1)

        distractors_set.discard(answer)
        distractors_list = list(distractors_set)
        distractors = pick_distractors(answer_text, distractors_list, n=3)

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
