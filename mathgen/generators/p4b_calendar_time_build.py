"""Calendar day-count and time-unit conversion (f10-describe-calendar-duration,
f10-time-units-and-nearest-minute).

- f10-describe-calendar-duration: Determine the number of days between two
  stated calendar dates. Easy: same month. Medium: same year, one month boundary.
  Hard: two+ month boundaries, half the time crossing a year (both non-leap).
  All questions phrased "How many days after {date1} is {date2}?" to avoid
  inclusive/exclusive ambiguity.

- f10-time-units-and-nearest-minute: Convert a stated duration between days,
  hours, minutes and seconds. Easy: adjacent units only. Medium: non-adjacent
  units (two factors). Hard: compound duration (two adjacent units together),
  convert to the smaller unit. All answers divide evenly (no remainder).
"""

from __future__ import annotations

from datetime import date, timedelta
from fractions import Fraction

from ..core import check, generator, pick_distractors


# ============================================================================
# Skill 1: f10-describe-calendar-duration
# ============================================================================

MONTH_LENGTHS_NONLEAP = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _day_word(n: int) -> str:
    """Agree a day count with its noun: "1 day" but "2 days"."""
    return "day" if n == 1 else "days"


def _verify_calendar_duration(date1: date, date2: date, answer_days: int) -> None:
    """Independent check via sum of month lengths and day-of-year ordinals.
    Never re-calls (date2 - date1).days directly."""

    def day_of_year_ordinal(d: date) -> int:
        """Compute day-of-year as sum of month lengths + day-of-month."""
        total = sum(MONTH_LENGTHS_NONLEAP[: d.month - 1]) + d.day
        return total

    # If date2 is in an earlier year than date1, this is an error in construction
    if date2.year < date1.year:
        raise ValueError(f"date2 {date2} is before date1 {date1}")

    if date2.year == date1.year:
        # Same year: just difference in ordinals
        ordinal1 = day_of_year_ordinal(date1)
        ordinal2 = day_of_year_ordinal(date2)
        expected = ordinal2 - ordinal1
    else:
        # Across year boundary: ordinal2 + 365 * (num_full_years) - ordinal1
        ordinal1 = day_of_year_ordinal(date1)
        ordinal2 = day_of_year_ordinal(date2)
        num_full_years = date2.year - date1.year - 1
        expected = (365 - ordinal1) + 365 * num_full_years + ordinal2

    check(expected == answer_days, f"day-count check via ordinals: {expected} != {answer_days}")


@generator(
    "describe_calendar_duration",
    level=2,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def describe_calendar_duration(rng, tier="easy"):
    """Determine the number of days between two stated calendar dates."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")

    # Use 2023 (non-leap year) for easy and medium; hard can cross into 2024
    # But avoid 29 February: never land on it in any tier.

    year_crossing = False  # Track if this is a year-crossing hard case

    if tier == "easy":
        # Same month: sample a month 1-12, then two distinct days within it
        month = rng.randint(1, 12)
        day1 = rng.randint(1, MONTH_LENGTHS_NONLEAP[month - 1] - 1)
        day2 = rng.randint(day1 + 1, MONTH_LENGTHS_NONLEAP[month - 1])
        date1 = date(2023, month, day1)
        date2 = date(2023, month, day2)

    elif tier == "medium":
        # Same year, crossing one month boundary.
        # Pick month m, day near end of m; date2 in month m+1.
        month = rng.randint(1, 11)  # 1-11, so month+1 <= 12
        max_day_in_m = MONTH_LENGTHS_NONLEAP[month - 1]
        day1 = rng.randint(max(1, max_day_in_m - 5), max_day_in_m)
        max_day_in_next = MONTH_LENGTHS_NONLEAP[month]
        day2 = rng.randint(1, max_day_in_next)
        date1 = date(2023, month, day1)
        date2 = date(2023, month + 1, day2)

    else:  # hard
        # Two or more month boundaries, and half the time a year boundary.
        # If crossing year, date1 in Nov/Dec 2023, date2 in Jan/Feb 2024.
        # Avoid 29 Feb: in the Dec-2023-to-Feb-2024 sub-branch, 2024 IS a leap year,
        # but we use day2 = rng.randint(1, 28), so the end date never reaches 29 Feb.

        if rng.choice((True, False)):
            # Cross a year boundary.
            year_crossing = True
            if rng.choice((True, False)):
                # Start in Nov 2023, end in Jan 2024
                month1 = 11
                day1 = rng.randint(1, MONTH_LENGTHS_NONLEAP[10])  # Nov = month 11, index 10
                date1 = date(2023, month1, day1)
                # End in Jan 2024
                day2 = rng.randint(1, 31)
                date2 = date(2024, 1, day2)
            else:
                # Start in Dec 2023, end in Feb 2024
                # Avoid 29 Feb: sample day2 in 1..28
                month1 = 12
                day1 = rng.randint(1, MONTH_LENGTHS_NONLEAP[11])  # Dec = month 12, index 11
                date1 = date(2023, month1, day1)
                day2 = rng.randint(1, 28)  # Feb has 28 days in 2024 (non-leap)
                date2 = date(2024, 2, day2)
        else:
            # Same year, multiple month boundaries (2023).
            # Pick month m in 1..10, day2 in month m+2 or later.
            month1 = rng.randint(1, 10)
            day1 = rng.randint(1, MONTH_LENGTHS_NONLEAP[month1 - 1])
            month2 = rng.randint(month1 + 2, 12)
            day2 = rng.randint(1, MONTH_LENGTHS_NONLEAP[month2 - 1])
            date1 = date(2023, month1, day1)
            date2 = date(2023, month2, day2)

    # Compute correct answer and verify no 29 Feb
    answer_days = (date2 - date1).days
    check(answer_days > 0, f"dates not distinct: {date1} -> {date2}")
    check(date1.month != 2 or date1.day != 29, f"date1 is 29 Feb: {date1}")
    check(date2.month != 2 or date2.day != 29, f"date2 is 29 Feb: {date2}")

    # Format dates as "5 March", "19 March", etc.
    MONTH_NAMES = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    date1_str = f"{date1.day} {MONTH_NAMES[date1.month]}"
    date2_str = f"{date2.day} {MONTH_NAMES[date2.month]}"

    # Set prompt and working text based on tier and year-crossing status
    if year_crossing:
        # Hard tier, year-crossing: state both years explicitly
        text = f"How many days after {date1_str} 2023 is {date2_str} 2024?"
        working = [f"From {date1_str} 2023 to {date2_str} 2024 is {answer_days} {_day_word(answer_days)}."]
    else:
        # Easy, medium, and hard same-year: state 2023 consistently
        text = f"In 2023, how many days after {date1_str} is {date2_str}?"
        working = [f"In 2023, from {date1_str} to {date2_str} is {answer_days} {_day_word(answer_days)}."]

    answer_text = str(answer_days)

    # Independent check via different route
    _verify_calendar_duration(date1, date2, answer_days)

    # Distractors
    off_by_one_inclusive = answer_days + 1
    off_by_one_exclusive = answer_days - 1

    # ignores-month-boundary: day2.day - day1.day
    # (only valid distractor if month boundary crossed)
    ignores_mb = date2.day - date1.day

    # ignores-year-boundary: like ignores-month-boundary but for years
    # (compute as if both in same year)
    ignores_yb = None
    if date2.year > date1.year:
        # Hard tier, year boundary crossed
        # Compute ordinals as if both in same year
        ordinal1 = sum(MONTH_LENGTHS_NONLEAP[: date1.month - 1]) + date1.day
        ordinal2 = sum(MONTH_LENGTHS_NONLEAP[: date2.month - 1]) + date2.day
        ignores_yb = ordinal2 - ordinal1

    candidates = [off_by_one_inclusive, off_by_one_exclusive]

    # Add ignores-month-boundary only if month boundary crossed
    if date1.month != date2.month:
        # Ensure it's distinct from answer
        if ignores_mb != answer_days:
            candidates.append(ignores_mb)

    # Add ignores-year-boundary only for hard tier if year boundary crossed
    if tier == "hard" and ignores_yb is not None:
        if ignores_yb != answer_days and ignores_yb not in candidates:
            candidates.append(ignores_yb)

    # Ensure all candidates are distinct integers
    candidates_str = [str(c) for c in candidates]
    distractors = pick_distractors(answer_text, candidates_str)

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer_days,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Skill 2: f10-time-units-and-nearest-minute
# ============================================================================


def _verify_time_conversion(value: int, unit_from: str, unit_to: str, answer: int) -> None:
    """Independent check via explicit Fraction conversion factors."""
    # Define all conversion factors as Fractions
    factors = {
        ("seconds", "minutes"): Fraction(1, 60),
        ("minutes", "seconds"): Fraction(60, 1),
        ("minutes", "hours"): Fraction(1, 60),
        ("hours", "minutes"): Fraction(60, 1),
        ("hours", "days"): Fraction(1, 24),
        ("days", "hours"): Fraction(24, 1),
        # Multi-step conversions
        ("seconds", "hours"): Fraction(1, 3600),
        ("hours", "seconds"): Fraction(3600, 1),
        ("seconds", "days"): Fraction(1, 86400),
        ("days", "seconds"): Fraction(86400, 1),
        ("minutes", "days"): Fraction(1, 1440),
        ("days", "minutes"): Fraction(1440, 1),
    }

    if (unit_from, unit_to) not in factors:
        raise ValueError(f"unknown conversion {unit_from} -> {unit_to}")

    factor = factors[(unit_from, unit_to)]
    result_frac = Fraction(value, 1) * factor

    # Result must be a whole number
    check(
        result_frac.denominator == 1, f"conversion not whole: {value} {unit_from} -> {unit_to} = {float(result_frac)}"
    )
    check(
        int(result_frac) == answer,
        f"conversion mismatch: {value} {unit_from} -> {unit_to} = {int(result_frac)}, not {answer}",
    )


@generator(
    "time_units_and_nearest_minute",
    level=3,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def time_units_and_nearest_minute(rng, tier="easy"):
    """Convert a stated duration between days, hours, minutes and seconds."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")

    if tier == "easy":
        # Adjacent units, single factor.
        # Either seconds<->minutes or minutes<->hours.
        if rng.choice((True, False)):
            # seconds <-> minutes
            if rng.choice((True, False)):
                # seconds to minutes: sample value as multiple of 60
                value = rng.randint(1, 50) * 60
                unit_from, unit_to = "seconds", "minutes"
                answer = value // 60
            else:
                # minutes to seconds: sample value freely
                value = rng.randint(1, 50)
                unit_from, unit_to = "minutes", "seconds"
                answer = value * 60
        else:
            # minutes <-> hours
            if rng.choice((True, False)):
                # minutes to hours: sample as multiple of 60
                value = rng.randint(1, 50) * 60
                unit_from, unit_to = "minutes", "hours"
                answer = value // 60
            else:
                # hours to minutes: sample freely
                value = rng.randint(1, 50)
                unit_from, unit_to = "hours", "minutes"
                answer = value * 60

    elif tier == "medium":
        # Non-adjacent units, two factors.
        # Either hours<->seconds or days<->minutes.
        if rng.choice((True, False)):
            # hours <-> seconds
            if rng.choice((True, False)):
                # hours to seconds: sample as multiple of 3600
                value = rng.randint(1, 24)
                unit_from, unit_to = "hours", "seconds"
                answer = value * 3600
            else:
                # seconds to hours: sample as multiple of 3600
                value = rng.randint(1, 24) * 3600
                unit_from, unit_to = "seconds", "hours"
                answer = value // 3600
        else:
            # days <-> minutes
            if rng.choice((True, False)):
                # days to minutes: sample as multiple of 1440
                value = rng.randint(1, 30)
                unit_from, unit_to = "days", "minutes"
                answer = value * 1440
            else:
                # minutes to days: sample as multiple of 1440
                value = rng.randint(1, 30) * 1440
                unit_from, unit_to = "minutes", "days"
                answer = value // 1440

    else:  # hard
        # Compound duration: two adjacent units together.
        # Sample two adjacent units, then convert to the smaller unit.
        # Always convert to the smaller (finer) unit.
        mins = secs = hours = mins_val = days = 0  # Initialize all variables
        hard_variant = None  # Track which variant we use
        hard_working = ""

        if rng.choice((True, False)):
            if rng.choice((True, False)):
                # minutes and seconds: convert to seconds
                hard_variant = "mins_secs"
                mins = rng.randint(1, 20)
                secs = rng.randint(1, 59)
                answer = mins * 60 + secs
                mins_str = "minute" if mins == 1 else "minutes"
                secs_str = "second" if secs == 1 else "seconds"
                text = f"Convert {mins} {mins_str} and {secs} {secs_str} to seconds."
                hard_working = f"{mins} × 60 = {mins * 60}, {mins * 60} + {secs} = {answer}."
            else:
                # hours and minutes: convert to minutes
                hard_variant = "hours_mins"
                hours = rng.randint(1, 20)
                mins_val = rng.randint(1, 59)
                answer = hours * 60 + mins_val
                hours_str = "hour" if hours == 1 else "hours"
                mins_str = "minute" if mins_val == 1 else "minutes"
                text = f"Convert {hours} {hours_str} and {mins_val} {mins_str} to minutes."
                hard_working = f"{hours} × 60 = {hours * 60}, {hours * 60} + {mins_val} = {answer}."
        else:
            # days and hours: convert to hours
            hard_variant = "days_hours"
            days = rng.randint(1, 20)
            hours = rng.randint(1, 23)
            answer = days * 24 + hours
            days_str = "day" if days == 1 else "days"
            hours_str = "hour" if hours == 1 else "hours"
            text = f"Convert {days} {days_str} and {hours} {hours_str} to hours."
            hard_working = f"{days} × 24 = {days * 24}, {days * 24} + {hours} = {answer}."

    # For easy and medium, construct text differently
    if tier in ("easy", "medium"):
        # Fix singular/plural
        unit_from_str = unit_from if value != 1 else unit_from.rstrip("s")
        unit_to_str = unit_to if answer != 1 else unit_to.rstrip("s")
        text = f"Convert {value} {unit_from_str} to {unit_to_str}."

    answer_text = str(answer)

    if tier in ("easy", "medium"):
        working = [f"Convert {value} {unit_from_str} to {unit_to_str}: {answer_text} {unit_to_str}."]
    else:
        # For hard, text was already constructed above
        working = [f"{text} {hard_working}"]

    # Verify the conversion
    if tier in ("easy", "medium"):
        _verify_time_conversion(value, unit_from, unit_to, answer)

    # Distractors
    candidates = []

    # wrong-factor: use 100 instead of 60 (or 10 instead of 24)
    if tier == "easy":
        if "minute" in unit_from or "minute" in unit_to:
            # If converting to minutes, wrong-factor might be value*100/60 or value (forgot)
            # If converting to seconds, wrong-factor might be value/100*60 = value*0.6
            if unit_from == "seconds" or unit_to == "seconds":
                wrong_factor_val = max(1, int(value * 10 / 60))
            else:
                wrong_factor_val = max(1, int(value * 100 / 60))
        else:
            # Hours/days
            wrong_factor_val = max(1, int(value * 10 / 24))
        if wrong_factor_val != answer:
            candidates.append(wrong_factor_val)

        # forgot-to-convert: report the original value unchanged (relabeled)
        forgot_val = value
        if forgot_val != answer:
            candidates.append(forgot_val)

        # Ensure at least 2 candidates by adding offsets
        while len(set(candidates) - {answer}) < 2:
            alt_val = answer + len(candidates) + 1
            if alt_val not in candidates:
                candidates.append(alt_val)

    elif tier == "medium":
        # For hours<->seconds, wrong factor might be 100 or something else
        if "hour" in unit_from or "hour" in unit_to:
            if "second" in unit_from or "second" in unit_to:
                # Should multiply/divide by 3600, instead use 100 * 60 = 6000
                wrong_factor_val = max(1, int(value * 6000 / 3600))
            else:
                # hours <-> minutes, should be 60, use 100
                wrong_factor_val = max(1, int(value * 100 / 60))
        else:
            # days <-> minutes, should be 1440, use 1000
            wrong_factor_val = max(1, int(value * 1000 / 1440))
        if wrong_factor_val != answer:
            candidates.append(wrong_factor_val)

        # forgot-to-convert: report the original value unchanged (relabeled)
        forgot_val = value
        if forgot_val != answer:
            candidates.append(forgot_val)

        # Ensure at least 2 candidates by adding offsets
        while len(set(candidates) - {answer}) < 2:
            alt_val = answer + len(candidates) + 1
            if alt_val not in candidates:
                candidates.append(alt_val)

    else:  # hard
        # For compound, we have a more complex structure
        # Let's define wrong-factor differently: use 100 for 60, use 10 for 24
        if hard_variant == "mins_secs":
            # 60 seconds per minute, use 100
            wrong_factor_val = mins * 100 + secs
        elif hard_variant == "hours_mins":
            # 60 minutes per hour, use 100
            wrong_factor_val = hours * 100 + mins_val
        else:  # days_hours
            # 24 hours per day, use 10
            wrong_factor_val = days * 10 + hours

        if wrong_factor_val != answer:
            candidates.append(wrong_factor_val)

        # forgot-to-convert: report the sum of the two numerals
        if hard_variant == "mins_secs":
            forgot_val = mins + secs
        elif hard_variant == "hours_mins":
            forgot_val = hours + mins_val
        else:
            forgot_val = days + hours
        if forgot_val != answer:
            candidates.append(forgot_val)

        # drops-one-part: report only the larger unit's contribution
        if hard_variant == "mins_secs":
            drops_val = mins * 60  # ignore the secs part
        elif hard_variant == "hours_mins":
            drops_val = hours * 60  # ignore the mins part
        else:
            drops_val = days * 24  # ignore the hours part
        if drops_val != answer:
            candidates.append(drops_val)

    candidates_str = [str(c) for c in candidates]
    distractors = pick_distractors(answer_text, candidates_str)

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
