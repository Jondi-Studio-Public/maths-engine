"""Two new Foundation-Year 10 measurement skills.

- ``metric_measure_instrument_estimate`` (``f10-metric-measure-instrument-estimate``,
  AC9M3M01, AC9M3M02, AC9M5M01, domain measurement) -- given a description of a
  metric instrument's labelled markings for length, mass or capacity, where a
  stated value falls among them (using a stated familiar-item benchmark if needed
  for estimation), choose the suitable metric unit and state the measured value.

- ``measurement_error_interpretation`` (``f10-measurement-error-interpretation``,
  AC9M9M04, AC9M10M04, domain measurement) -- given a measured value and its true
  or accepted value, calculate the absolute, relative, or percentage error.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors


def _plural_unit(count: int, unit_singular: str = "unit") -> str:
    """Pluralize correctly for a count: 1 unit, 2 units, etc."""
    if count == 1:
        return unit_singular
    else:
        return f"{unit_singular}s"


# ============================================================================
# Skill 1: metric_measure_instrument_estimate
# ============================================================================

INSTRUMENTS = {
    "length": ("ruler", "cm"),
    "mass": ("scale", "g"),
    "capacity": ("jug", "mL"),
}


def _ordinal(k: int) -> str:
    """Convert an integer k (1-10) to its ordinal form (1st, 2nd, 3rd, etc.)."""
    if k == 1:
        return "1st"
    elif k == 2:
        return "2nd"
    elif k == 3:
        return "3rd"
    else:
        return f"{k}th"


def _verify_easy_reading(k: int, M: int, expected: int) -> None:
    """Independent check for easy tier: verify answer using a different route."""
    computed = k * M
    check(computed == expected, f"easy reading: k={k}, M={M}, expected={expected}, got {computed}")


def _verify_medium_reading(k: int, D: int, M: int, n: int, expected: int) -> None:
    """Independent check for medium tier: verify via total minor divisions."""
    total_minor_divisions = k * D + n
    minor_division_value = M // D
    computed = total_minor_divisions * minor_division_value
    check(computed == expected, f"medium reading mismatch: expected {expected}, got {computed}")


def _verify_before_mark(k: int, D: int, M: int, n: int, expected: int) -> None:
    """Independent check for before_mark (hard tier): verify via total minor divisions before mark."""
    total_minor = (k + 1) * D - n
    minor_division_value = M // D
    computed = total_minor * minor_division_value
    check(computed == expected, f"before_mark mismatch: expected {expected}, got {computed}")


def _verify_benchmark_estimate(T: int, B: int, expected: int) -> None:
    """Independent check for benchmark_estimate: verify via repeated subtraction."""
    remaining = T
    steps = 0
    while remaining > 0:
        remaining -= B
        steps += 1
    check(steps == expected, f"benchmark_estimate: repeated subtraction gave {steps}, expected {expected}")
    check(remaining == 0, f"benchmark_estimate: remaining={remaining} after division")


@generator(
    "metric_measure_instrument_estimate",
    level=10,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def metric_measure_instrument_estimate(rng, tier="easy"):
    """Estimate and read measured values from metric instruments."""
    # Sample instrument kind uniformly
    kind = rng.choice(list(INSTRUMENTS.keys()))
    instrument_name, unit = INSTRUMENTS[kind]

    if tier == "easy":
        # Easy: reading exactly at a labelled mark
        M = rng.choice([2, 5, 10])
        k = rng.randint(1, 10)
        answer = k * M

        _verify_easy_reading(k, M, answer)

        text = f"A {instrument_name} is marked every {M} {unit}. The pointer sits exactly on the {_ordinal(k)} mark from zero. What value does it show, in {unit}?"

        # Misconceptions
        # 1. off_by_one_mark: reads adjacent mark
        prev_mark = max(0, (k - 1) * M)
        next_mark = (k + 1) * M
        off_by_one_candidates = [prev_mark, next_mark]

        # 2. wrong_major_spacing: misreads the mark spacing (e.g., uses M+1 or M-1)
        wrong_spacing_candidates = [
            (k * (M + 1)) if M < 10 else (k * (M - 1)),
            (k * (M - 1)) if M > 2 else (k * (M + 1)),
        ]

        candidates = off_by_one_candidates + wrong_spacing_candidates

        answer_text = str(answer)
        distractors = pick_distractors(answer_text, [str(c) for c in candidates])

        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"The pointer is at the {_ordinal(k)} mark, so the value is {k} × {M} = {answer} {unit}."],
            distractors=distractors,
        )

    elif tier == "medium":
        # Medium: reading with minor divisions (addition)
        M = rng.choice([10, 20])
        D = rng.choice([2, 4, 5])
        while M % D != 0:
            D = rng.choice([2, 4, 5])
        k = rng.randint(1, 5)
        n = rng.randint(1, D - 1)
        minor_div_value = M // D
        answer = k * M + n * minor_div_value

        _verify_medium_reading(k, D, M, n, answer)

        text = f"A {instrument_name} has major marks every {M} {unit}. Each major interval is split into {D} small divisions, each worth {minor_div_value} {unit}. The pointer sits at the {_ordinal(k)} major mark, plus {n} small division{'s' if n > 1 else ''}. What value does it show, in {unit}?"

        # Misconceptions
        # 1. reads_previous_mark_only: ignores minor divisions
        reads_prev = k * M
        # 2. confuses_major_minor: uses M instead of M//D as minor value
        confuses_mm = k * M + n * M
        # 3. off_by_one_mark: adjacent major mark + minor divisions
        off_by_one = (k + 1) * M + n * minor_div_value if k < 5 else (k - 1) * M + n * minor_div_value

        candidates = [reads_prev, confuses_mm, off_by_one]

        answer_text = str(answer)
        distractors = pick_distractors(answer_text, [str(c) for c in candidates])

        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"At {_ordinal(k)} major mark: {k} × {M} = {k * M}. Plus {n} small division{'s' if n > 1 else ''}: {n} × {minor_div_value} = {n * minor_div_value}. Total: {k * M} + {n * minor_div_value} = {answer} {unit}."
            ],
            distractors=distractors,
        )

    elif tier == "hard":
        # Hard: forced trap between before_mark and benchmark_estimate
        trap_type = rng.choice(["before_mark", "benchmark_estimate"])

        if trap_type == "before_mark":
            # Before mark (subtraction): pointer is n divisions before the (k+1)-th mark
            M = rng.choice([10, 20])
            D = rng.choice([2, 4, 5])
            while M % D != 0:
                D = rng.choice([2, 4, 5])
            k = rng.randint(1, 4)  # k such that (k+1) is a valid mark
            n = rng.randint(1, D - 1)
            minor_div_value = M // D
            answer = (k + 1) * M - n * minor_div_value

            _verify_before_mark(k, D, M, n, answer)

            text = f"A {instrument_name} has major marks every {M} {unit}. Each major interval is split into {D} small divisions, each worth {minor_div_value} {unit}. The pointer sits {n} small division{'s' if n > 1 else ''} before the {_ordinal(k + 1)} major mark. What value does it show, in {unit}?"

            # Misconceptions
            # 1. wrong_direction: adds instead of subtracts
            wrong_dir = (k + 1) * M + n * minor_div_value
            # 2. reads_next_mark_only: reads the next mark, ignoring divisions
            reads_next = (k + 1) * M
            # 3. confuses_major_minor: uses M instead of M//D
            confuses_mm = (k + 1) * M - n * M

            candidates = [wrong_dir, reads_next, confuses_mm]
            working_line = (
                f"The {_ordinal(k + 1)} major mark is {(k + 1) * M} {unit}. The pointer is {n} small "
                f"division{'s' if n > 1 else ''} before it: {(k + 1) * M} - {n} × {minor_div_value} = {answer} {unit}."
            )

        else:  # benchmark_estimate
            # Benchmark estimate: familiar item weight and total count
            B = rng.randint(1, 10)  # benchmark weight in units
            count = rng.randint(2, 12)  # how many items
            T = B * count  # total (exact multiple)
            answer = count

            _verify_benchmark_estimate(T, B, answer)

            # Pick appropriate benchmark and verb based on kind
            if kind == "mass":
                familiar_item = "small weight"
                verb = "balance"
            elif kind == "length":
                familiar_item = "pencil"
                verb = "measure out"
            else:  # capacity
                familiar_item = "small cup"
                verb = "fill"

            text = (
                f"A {familiar_item} is about {B} {unit}. About how many would it take to {verb} a total of {T} {unit}?"
            )

            # Misconceptions
            # 1. off_by_one_count: count ± 1
            candidates = [count - 1, count + 1, count + 2 if count < 11 else count - 2]
            working_line = f"{T} {unit} ÷ {B} {unit} = {count}, so it would take about {count}."

        answer_text = str(answer)
        distractors = pick_distractors(answer_text, [str(c) for c in candidates])

        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[working_line],
            distractors=distractors,
        )

    else:
        raise ValueError(f"unknown tier: {tier!r}")


# ============================================================================
# Skill 2: measurement_error_interpretation
# ============================================================================


def _verify_absolute_error(measured: int, T: int, expected: int) -> None:
    """Independent check for absolute error: use sort-then-subtract."""
    hi, lo = max(measured, T), min(measured, T)
    computed = hi - lo
    check(computed == expected, f"absolute error mismatch: expected {expected}, got {computed}")


def _verify_relative_error(measured: int, T: int, expected: Fraction) -> None:
    """Independent check for relative error: recompute as abs(measured/T - 1)."""
    frac_measured_over_true = Fraction(measured, T)
    computed = abs(frac_measured_over_true - 1)
    check(computed == expected, f"relative error mismatch: expected {expected}, got {computed}")


def _verify_percentage_error(relative_frac: Fraction, expected: int) -> None:
    """Independent check for percentage error: multiply relative fraction by 100."""
    computed = relative_frac * 100
    check(computed.denominator == 1, f"percentage must be whole number, got {computed}")
    check(int(computed) == expected, f"percentage mismatch: expected {expected}, got {computed}")


def _sample_e_with_no_trailing_zero(rng, T: int, resample_limit: int = 200) -> int:
    """Sample e from 1..T-1, rejecting if (100*e)//T ends in 0.

    This prevents trailing-zero 2-decimal displays like 0.10, 0.20, 0.50.
    Only used for relative and percentage error tiers.
    """
    for _ in range(resample_limit):
        e = rng.randint(1, T - 1)
        if ((100 * e) // T) % 10 != 0:
            return e
    # Fallback: return the last sampled e (should rarely happen)
    return e


@generator(
    "measurement_error_interpretation",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def measurement_error_interpretation(rng, tier="easy"):
    """Interpret and calculate measurement errors."""
    if tier == "easy":
        # Easy: absolute error
        T = rng.choice([10, 20, 25, 50, 100])
        e = rng.randint(1, T - 1)
        # Sample sign
        if rng.choice([True, False]):
            measured = T + e
        else:
            measured = T - e
        answer = e

        _verify_absolute_error(measured, T, answer)

        text = f"A recipe calls for {T} g of flour. A cook actually uses {measured} g. What is the absolute error, in grams?"

        # Misconceptions
        # 1. uses_signed_difference: reports signed difference instead of absolute
        signed_diff = measured - T
        # 2. confuses_measured_and_true: uses measured - e instead of true - e (if applicable)
        wrong_comparison = abs(measured - e) if e != measured else measured
        # 3. off by something random
        other_wrong = e + 1 if e < T - 1 else e - 1

        candidates = [signed_diff, wrong_comparison, other_wrong]

        answer_text = str(answer)
        distractors = pick_distractors(answer_text, [str(c) for c in candidates])

        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"Absolute error = |measured - true| = |{measured} - {T}| = {answer} g."],
            distractors=distractors,
        )

    elif tier == "medium":
        # Medium: relative error (division, 2 decimal places)
        T = rng.choice([20, 25, 50, 100])
        e = _sample_e_with_no_trailing_zero(rng, T)
        # Verify that the relative error is exact to 2 decimal places
        check((100 * e) % T == 0, f"relative error not exact: 100*{e} mod {T} = {(100 * e) % T}")

        # Sample sign
        if rng.choice([True, False]):
            measured = T + e
        else:
            measured = T - e

        relative = Fraction(e, T)
        answer_float = float(relative)
        answer_text = f"{answer_float:.2f}"

        _verify_relative_error(measured, T, relative)

        text = f"A measurement is {measured} cm, but the true value is {T} cm. What is the relative error as a decimal, to 2 decimal places?"

        # Misconceptions
        # 1. swapped_numerator_denominator: divides by measured instead of true
        swapped = Fraction(e, measured)
        swapped_text = f"{float(swapped):.2f}"
        # 2. uses absolute (no division): just the error amount itself
        absolute_text = f"{float(e):.2f}"
        # 3. omits percent scaling but reports as decimal (confuses with percentage): e/T*100 but displayed as decimal
        percent_as_decimal = Fraction(100 * e, T)
        percent_as_decimal_text = f"{float(percent_as_decimal):.2f}"
        # 4. off by a small amount (different sampling)
        if e < T - 1:
            perturbed = Fraction(e + 1, T)
        else:
            perturbed = Fraction(e - 1, T)
        perturbed_text = f"{float(perturbed):.2f}"

        candidates = [swapped_text, absolute_text, percent_as_decimal_text, perturbed_text]

        distractors = pick_distractors(answer_text, candidates)

        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer_float,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"Relative error = error / true value = {e} / {T} = {answer_text}."],
            distractors=distractors,
        )

    elif tier == "hard":
        # Hard: classify then compute (one of three types)
        request_type = rng.choice(["absolute", "relative", "percentage"])

        if request_type == "absolute":
            T = rng.choice([10, 20, 25, 50, 100])
            e = rng.randint(1, T - 1)
            if rng.choice([True, False]):
                measured = T + e
            else:
                measured = T - e
            answer = e
            answer_text = str(answer)

            _verify_absolute_error(measured, T, answer)

            measured_unit = _plural_unit(measured, "unit")
            true_unit = _plural_unit(T, "unit")
            text = f"A measurement is {measured} {measured_unit}, and the true value is {T} {true_unit}. Express the error as a plain difference: the larger value minus the smaller, so the answer is not negative."
            request_phrase = "as a plain difference"

            # Generate diverse misconceptions
            candidates = []
            # 1. off by one
            if e < T - 1:
                candidates.append(e + 1)
            if e > 1:
                candidates.append(e - 1)
            # 2. signed difference
            candidates.append(measured - T)
            # 3. wrong measurement used
            candidates.append(measured)
            # 4. true value used
            candidates.append(T)
            # Ensure we have at least 2 distinct non-answer values
            candidates = [str(c) for c in candidates if c != answer]
            working_line = f"The plain difference is {max(measured, T)} - {min(measured, T)} = {answer}."

        elif request_type == "relative":
            T = rng.choice([20, 25, 50, 100])
            e = _sample_e_with_no_trailing_zero(rng, T)
            check((100 * e) % T == 0, f"relative error not exact: 100*{e} mod {T} = {(100 * e) % T}")

            if rng.choice([True, False]):
                measured = T + e
            else:
                measured = T - e

            relative = Fraction(e, T)
            answer_float = float(relative)
            answer_text = f"{answer_float:.2f}"

            _verify_relative_error(measured, T, relative)

            measured_unit = _plural_unit(measured, "unit")
            true_unit = _plural_unit(T, "unit")
            text = f"A measurement is {measured} {measured_unit}, and the true value is {T} {true_unit}. Express the error as a decimal fraction of the true value, to 2 decimal places."
            request_phrase = "as a decimal fraction of the true value, to 2 decimal places"

            # Generate diverse misconceptions
            cand_list = []
            # 1. swapped_numerator_denominator
            swapped = Fraction(e, measured)
            cand_list.append(f"{float(swapped):.2f}")
            # 2. absolute (no division)
            cand_list.append(f"{float(e):.2f}")
            # 3. percentage as decimal
            percent_as_decimal = Fraction(100 * e, T)
            cand_list.append(f"{float(percent_as_decimal):.2f}")
            # 4. perturbed
            if e < T - 1:
                perturbed = Fraction(e + 1, T)
            else:
                perturbed = Fraction(e - 1, T)
            cand_list.append(f"{float(perturbed):.2f}")

            candidates = cand_list
            answer = answer_float
            working_line = f"Error = |{measured} - {T}| = {e}, and {e} ÷ {T} = {answer_text}."

        else:  # percentage
            T = rng.choice([20, 25, 50, 100])
            e = _sample_e_with_no_trailing_zero(rng, T)
            check((100 * e) % T == 0, f"percentage error not exact: 100*{e} mod {T} = {(100 * e) % T}")

            if rng.choice([True, False]):
                measured = T + e
            else:
                measured = T - e

            pct = 100 * e // T
            answer = pct
            answer_text = str(pct)

            # Verify percentage is whole
            check(100 * e % T == 0, f"percentage calculation failed: 100*{e} not divisible by {T}")

            _verify_percentage_error(Fraction(e, T), pct)

            measured_unit = _plural_unit(measured, "unit")
            true_unit = _plural_unit(T, "unit")
            text = f"A measurement is {measured} {measured_unit}, and the true value is {T} {true_unit}. Express the error as a whole number of percent."
            request_phrase = "as a whole number of percent"

            # Generate diverse misconceptions
            cand_list = []
            # 1. off by one
            if pct > 1:
                cand_list.append(pct - 1)
            if pct < 100:
                cand_list.append(pct + 1)
            # 2. divides by the measured value instead of the true value
            # (kept whole-number so every option shares the answer's shape)
            cand_list.append((200 * e + measured) // (2 * measured))
            # 3. absolute error as percentage
            cand_list.append(e)
            # 4. the complement: reports how close the reading was, not how far off
            cand_list.append(100 - pct)

            candidates = [str(c) for c in cand_list if str(c) != answer_text]
            working_line = f"Error = |{measured} - {T}| = {e}, and {e} ÷ {T} × 100 = {pct}%."

        # Ensure candidates is a list of strings
        if not isinstance(candidates, list):
            candidates = [str(c) for c in candidates]
        distractors = pick_distractors(answer_text, candidates)

        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[working_line],
            distractors=distractors,
        )

    else:
        raise ValueError(f"unknown tier: {tier!r}")
