"""Generates `use_units_of_time_conversions_between_units_fractional` questions: convert a duration between
minutes, decimal hours and fractional hours.

Senior ACARA v8.4, domain measurement.

Converts between units of time (minutes, decimal hours, fractional hours).
BOUNDARY: This skill is CONVERSION BETWEEN UNITS of time (hours/minutes/
seconds, fractional and decimal representations of a stated duration).
It never computes elapsed time between two clock times or calendar dates.
No question states two times-of-day or two dates and asks for the gap
between them; every question states a duration directly (already in one unit
or compound form) and asks for it re-expressed in another unit.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check_equivalent,
    check_distinct,
    exact_decimal_string,
    generator,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)


def _ensure_decimal_format(text: str, min_places: int = 1) -> str:
    """Ensure a formatted number has at least min_places decimal places.
    If text is an integer like '7', returns '7.0' (if min_places >= 1)."""
    if "." not in text:
        return text + ".0" if min_places >= 1 else text
    return text


def _shaped_distractors(answer_text: str, candidates: list[str]) -> list[str]:
    """For a decimal key, show every option at the key's own number of decimal
    places ("1.80" beside "0.65", not "1.8"), keeping the misconceptions that
    land on that grid and topping up with power-of-ten slips of the key, so the
    option shapes never give the key away. A whole-number key is unchanged."""
    if "." not in answer_text:
        return pick_distractors(answer_text, candidates)
    answer = Fraction(answer_text)
    places = len(answer_text.partition(".")[2])
    scale = 10**places
    pool = [Fraction(c) for c in candidates] + [answer * 10, answer * 100, answer + 1, answer + 2]
    return pick_distractors(
        answer_text,
        [stated_precision_string(v, places) for v in pool if v > 0 and (v * scale).denominator == 1],
    )


# Hard tier: fraction pool and verification
FRACTION_POOL = [
    (1, 2, 30),  # 1/2 hour = 30 minutes
    (1, 3, 20),  # 1/3 hour = 20 minutes
    (2, 3, 40),  # 2/3 hour = 40 minutes
    (1, 4, 15),  # 1/4 hour = 15 minutes
    (3, 4, 45),  # 3/4 hour = 45 minutes
    (1, 5, 12),  # 1/5 hour = 12 minutes
    (2, 5, 24),  # 2/5 hour = 24 minutes
    (3, 5, 36),  # 3/5 hour = 36 minutes
    (4, 5, 48),  # 4/5 hour = 48 minutes
    (1, 6, 10),  # 1/6 hour = 10 minutes
    (5, 6, 50),  # 5/6 hour = 50 minutes
    (1, 10, 6),  # 1/10 hour = 6 minutes
    (1, 12, 5),  # 1/12 hour = 5 minutes
    (1, 15, 4),  # 1/15 hour = 4 minutes
    (1, 20, 3),  # 1/20 hour = 3 minutes
]


# Verify FRACTION_POOL at module load time
def _verify_fraction_pool():
    for num, den, minutes_equiv in FRACTION_POOL:
        expected = Fraction(num, den) * 60
        check_equivalent(expected, minutes_equiv, f"fraction pool entry ({num}/{den}) mismatch")


_verify_fraction_pool()


def _build_easy(rng) -> dict:
    """Easy tier: single-step conversion between whole minutes and decimal
    hours (mode="minutes_hours_direct")."""
    direction = rng.choice(["minutes_to_hours", "hours_to_minutes"])

    if direction == "minutes_to_hours":
        # k in 1..19, minutes = 3*k (3..57, always multiple of 3)
        # guarantees minutes/60 terminates within 2 decimal places
        k = rng.randint(1, 19)
        minutes = 3 * k
        answer = Fraction(minutes, 60)
        answer_text = exact_decimal_string(answer, max_places=3)
        text = f"Convert {minutes} minutes to hours. Give your answer as a decimal."
        working = [
            "There are 60 minutes in 1 hour, so divide by 60.",
            f"{minutes} ÷ 60 = {answer_text} hours.",
        ]

        # Misconceptions (as numeric values for check_distinct)
        # Only use distractors that are in decimal format to match answer shape
        # divides_by_10, divides_by_100
        wrong_by_10 = Fraction(minutes, 10)
        wrong_by_100 = Fraction(minutes, 100)

        check_distinct(answer, wrong_by_10, wrong_by_100, msg="easy minutes_to_hours distractor equals answer")

        # Check tolerance: ensure distractors differ by more than half a unit in the last decimal place
        # For 2 dp: tolerance = 0.005; for 1 dp: tolerance = 0.05
        places = len(answer_text.split(".")[1]) if "." in answer_text else 0
        if places > 0:
            tol = Fraction(1, 2 * (10**places))
            # If wrong_by_100 is too close, resample
            if abs(wrong_by_100 - answer) <= tol:
                return _build_easy(rng)

        # Format candidates for display (all decimal)
        candidates = [
            _ensure_decimal_format(exact_decimal_string(wrong_by_10, max_places=3)),  # divides_by_10
            _ensure_decimal_format(exact_decimal_string(wrong_by_100, max_places=3)),  # divides_by_100
        ]

        # Verify round-trip: parse formatted answer back to Fraction
        check_equivalent(
            Fraction(answer_text), Fraction(minutes, 60), f"minutes_to_hours round-trip failed for {minutes} minutes"
        )

    else:  # hours_to_minutes
        n = rng.randint(0, 4)
        j = rng.choice([0, 25, 50, 75])
        # Resample if n==0 and j==0 (never ask about "0 hours")
        if n == 0 and j == 0:
            return _build_easy(rng)

        hours_value = n + Fraction(j, 100)
        hours_display = exact_decimal_string(hours_value, max_places=3)
        answer = 60 * n + (3 * j) // 5  # exact integer

        hour_word = "hour" if hours_value == 1 else "hours"
        text = f"Convert {hours_display} {hour_word} to minutes."
        answer_text = str(answer)
        working = [
            "There are 60 minutes in 1 hour, so multiply by 60.",
            f"{hours_display} × 60 = {answer} minutes.",
        ]

        # Misconceptions as numeric values
        wrong_reads_digits = 100 * n + j
        wrong_no_frac = 60 * n if j != 0 else None
        wrong_by_10 = answer // 10

        candidates_values = [wrong_reads_digits, wrong_by_10]
        if wrong_no_frac is not None:
            candidates_values.insert(1, wrong_no_frac)

        check_distinct(answer, *candidates_values, msg="easy hours_to_minutes distractor equals answer")

        # Format for display
        candidates = [
            str(wrong_reads_digits),  # reads_the_decimal_digits_as_minutes
        ]
        if wrong_no_frac is not None:
            candidates.append(str(wrong_no_frac))  # ignores_the_fractional_part

        candidates.append(str(wrong_by_10))  # off_by_a_factor_of_ten

        # Verify via multiplication: hours_value * 60 == answer
        check_equivalent(hours_value * 60, answer, f"hours_to_minutes check failed for {hours_display} hours")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_medium(rng) -> dict:
    """Medium tier: compound duration (hours-and-minutes, or decimal-hours)
    converted to the other representation (mode="hours_minutes_compound")."""
    sub_mode = rng.choice(["hm_to_decimal", "decimal_to_minutes"])

    if sub_mode == "hm_to_decimal":
        # hours and minutes -> decimal hours
        hours = rng.randint(1, 5)
        minutes = 3 * rng.randint(1, 19)  # 3..57, multiple of 3
        answer = hours + Fraction(minutes, 60)
        answer_text = exact_decimal_string(answer, max_places=3)

        text = f"A duration is {hours} {'hour' if hours == 1 else 'hours'} and {minutes} minutes. Write this as a decimal number of hours."
        working = [
            f"{minutes} minutes = {minutes} ÷ 60 = {exact_decimal_string(Fraction(minutes, 60), max_places=3)} hours.",
            f"{hours} + {exact_decimal_string(Fraction(minutes, 60), max_places=3)} = {answer_text} hours.",
        ]

        # Misconceptions as numeric values - only use divides_by_10 to avoid tolerance issues
        wrong_by_10 = hours + Fraction(minutes, 10)
        wrong_drop_hours = Fraction(minutes, 60)

        check_distinct(answer, wrong_by_10, wrong_drop_hours, msg="medium hm_to_decimal distractor equals answer")

        # Format for display with consistent decimal formatting
        candidates = [
            _ensure_decimal_format(exact_decimal_string(wrong_by_10, max_places=3)),  # divides_by_10
            _ensure_decimal_format(exact_decimal_string(wrong_drop_hours, max_places=3)),  # drops_the_hours
        ]

        # Verify via total-minutes-over-60
        check_equivalent(
            Fraction(hours * 60 + minutes, 60), answer, f"hm_to_decimal check failed for {hours}h{minutes}m"
        )

    else:  # decimal_to_minutes
        # decimal hours (richer pool: k in 1..19, richer than easy's j in {0,25,50,75})
        n = rng.randint(1, 5)
        k = rng.randint(1, 19)
        hours_value = n + Fraction(k, 20)
        hours_display = exact_decimal_string(hours_value, max_places=3)
        answer = 60 * n + 3 * k  # exact integer

        text = f"A duration is {hours_display} hours (decimal). Convert this to total minutes."
        answer_text = str(answer)
        working = [
            "There are 60 minutes in 1 hour, so multiply by 60.",
            f"{hours_display} × 60 = {answer} minutes.",
        ]

        # Misconceptions as numeric values
        wrong_reads_digits = 100 * n + 5 * k
        wrong_no_frac = 60 * n
        wrong_by_10 = answer // 10

        check_distinct(
            answer,
            wrong_reads_digits,
            wrong_no_frac,
            wrong_by_10,
            msg="medium decimal_to_minutes distractor equals answer",
        )

        # Format for display
        candidates = [
            str(wrong_reads_digits),  # reads_the_decimal_digits_as_minutes
            str(wrong_no_frac),  # ignores_the_fractional_part
            str(wrong_by_10),  # off_by_a_factor_of_ten
        ]

        # Verify via multiplication
        check_equivalent(hours_value * 60, answer, f"decimal_to_minutes check failed for {hours_display} hours")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_hard(rng) -> dict:
    """Hard tier: convert a FRACTION of an hour (not decimal) to minutes,
    optionally with a whole-hour part (mode="fraction_of_hour_to_minutes")."""
    num, den, minutes_equiv = rng.choice(FRACTION_POOL)
    n = rng.randint(0, 4)

    if n == 0:
        text = f"Convert {num}/{den} of an hour to minutes."
    else:
        text = f"Convert {n} and {num}/{den} hours to minutes."

    answer = n * 60 + minutes_equiv
    answer_text = str(answer)
    working = [
        f"{num}/{den} of an hour = {num}/{den} × 60 = {minutes_equiv} minutes.",
        *(
            [
                f"{n} {'hour' if n == 1 else 'hours'} = {n * 60} minutes, so the total is {n * 60} + {minutes_equiv} = {answer} minutes."
            ]
            if n
            else []
        ),
    ]

    # Misconceptions as numeric values
    wrong_by_100_not_60 = n * 60 + (num * 100) // den
    wrong_ignore_frac = n * 60
    wrong_add_instead = n * 60 + num + den

    check_distinct(
        answer, wrong_by_100_not_60, wrong_ignore_frac, wrong_add_instead, msg="hard tier distractor equals answer"
    )

    # Format for display
    candidates = [
        str(wrong_by_100_not_60),  # multiplies_fraction_by_100_not_60
        str(wrong_ignore_frac),  # ignores_fraction_entirely (always distinct: minutes_equiv >= 3)
        str(wrong_add_instead),  # adds_instead_of_converts
    ]

    # Verify independently
    check_equivalent(Fraction(num, den) * 60, minutes_equiv, f"hard tier fraction pool check failed for {num}/{den}")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "use_units_of_time_conversions_between_units_fractional",
    level=11,
    presets={
        "easy": dict(mode="minutes_hours_direct"),
        "medium": dict(mode="hours_minutes_compound"),
        "hard": dict(mode="fraction_of_hour_to_minutes"),
    },
)
def use_units_of_time_conversions_between_units_fractional(
    rng,
    mode="minutes_hours_direct",
):
    """Convert between units of time (minutes/decimal hours/fractional hours).
    BOUNDARY: converts a stated duration between units, never elapsed time
    between two clock times or calendar dates."""
    if mode == "minutes_hours_direct":
        return _build_easy(rng)
    if mode == "hours_minutes_compound":
        return _build_medium(rng)
    if mode == "fraction_of_hour_to_minutes":
        return _build_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")
