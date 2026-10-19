"""Travelling east and west, time-zone difference (senior-solve-problems-relating-to-travelling-east-and-west,
ACMEM167).

Given a departure clock time at place A, a travel direction ("east"/"west"),
a flight duration, place B's arrival clock time and an explicit calendar-day
label ("the same day" / "the day before" / "the day after"), the learner
returns the MAGNITUDE of the time-zone difference between A and B, in hours.
The direction is given, not derived; only the magnitude is graded.

Construction (deterministic) and the independent check both work in whole
minutes-since-midnight. `dep_A`, every flight duration and every table
offset are multiples of 30 minutes by design, so every quantity this module
ever renders -- the true answer and every misconception distractor -- is a
whole or half number of hours, which is what keeps "the answer and every
distractor formatted alike" (see `_format_hours`).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_distinct, generator, latex_answer, pick_distractors, stated_precision_string


# ============================================================================
# The real place-pair table. `offset` is the REAL zone-offset magnitude, in
# hours, between A and B; `direction` is the travel direction from A to B.
# `duration_range` is a plausible (min, max) flight-duration span in minutes
# for that route, always a multiple of 30 -- these are what the construction
# draws from, at every difficulty.
# ============================================================================

ROWS = [
    dict(
        a="Sydney", b="Perth", direction="west", offset=Fraction(2, 1), note="AEST vs AWST", duration_range=(240, 330)
    ),
    dict(a="Perth", b="Tokyo", direction="east", offset=Fraction(1, 1), note="AWST vs JST", duration_range=(540, 660)),
    dict(
        a="Sydney",
        b="Auckland",
        direction="east",
        offset=Fraction(2, 1),
        note="AEST/AEDT vs NZST/NZDT, offset unchanged either season",
        duration_range=(180, 240),
    ),
    dict(
        a="Adelaide",
        b="Perth",
        direction="west",
        offset=Fraction(3, 2),
        note="ACST(+9:30) vs AWST(+8) -- the half-hour offset case",
        duration_range=(120, 180),
    ),
    dict(
        a="Adelaide",
        b="Mumbai",
        direction="west",
        offset=Fraction(4, 1),
        note="ACST(+9:30) vs IST(+5:30)",
        duration_range=(600, 720),
    ),
    dict(
        a="Perth", b="London", direction="west", offset=Fraction(8, 1), note="AWST vs GMT", duration_range=(900, 1080)
    ),
    dict(
        a="Sydney",
        b="London",
        direction="west",
        offset=Fraction(9, 1),
        note="AEST vs GMT (approx, stated as the scenario's given fact)",
        duration_range=(1200, 1320),
    ),
    dict(
        a="Sydney",
        b="Los Angeles",
        direction="west",
        offset=Fraction(17, 1),
        note="large hard-tier case",
        duration_range=(780, 900),
    ),
]

EASY_ROWS = ROWS[0:3]
MEDIUM_ROWS = ROWS[3:7]
HARD_ROWS = ROWS  # the full 8-row pool

DAY_LABELS = {-1: "the day before", 0: "the same day", 1: "the day after"}
LABEL_TO_OFFSET = {v: k for k, v in DAY_LABELS.items()}
#: phrase used inline in the question text, always naming the departure day
#: explicitly so "the day after" can only be read as "the day after the day
#: they departed <A>" -- never ambiguous about which day it is after.
DAY_PHRASE = {
    -1: "on the day before they departed",
    0: "on the same day they departed",
    1: "on the day after they departed",
}


def _offset_minutes(row: dict) -> int:
    return int(row["offset"] * 60)


def _clock(m: int) -> str:
    """Format minutes-since-midnight (any int) as a 24-hour "HH:MM" string,
    wrapping into a single day."""
    m %= 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def _duration_text(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    parts = []
    if h:
        parts.append(f"{h} hour" + ("" if h == 1 else "s"))
    if m:
        parts.append(f"{m} minute" + ("" if m == 1 else "s"))
    return " ".join(parts) if parts else "0 minutes"


def _format_hours(value: Fraction) -> tuple:
    """Render a magnitude in hours the way the learner must type it: a bare
    integer when whole (never a trailing ".0"), one decimal place only for a
    genuine half-hour. Returns (answer, answer_text)."""
    if value.denominator == 1:
        v = int(value)
        return v, str(v)
    v = float(value)
    return v, f"{v:.1f}"


def _draw_duration(rng, row: dict) -> int:
    lo, hi = row["duration_range"]
    return rng.randrange(lo, hi + 30, 30)


def _zero_window(row: dict, duration: int) -> tuple:
    """The [dep_lo0, dep_hi0] range of `dep_A` (inclusive) for which this
    row's flight, of the given duration, lands with day_offset_B == 0."""
    offset_min = _offset_minutes(row)
    if row["direction"] == "west":
        lo, hi = offset_min, 1440 + offset_min - 1
    else:
        lo, hi = 0, 1440 - offset_min - 1
    dep_lo0 = max(0, lo - duration)
    dep_hi0 = min(1439, hi - duration)
    return dep_lo0, dep_hi0


def _construct_no_rollover(rng, row: dict) -> tuple:
    """easy/medium: draw (dep_A, duration) so day_offset_B == 0 always."""
    duration = _draw_duration(rng, row)
    dep_lo0, dep_hi0 = _zero_window(row, duration)
    check(dep_lo0 <= dep_hi0, f"no zero-rollover dep_A window for {row['a']}-{row['b']} duration={duration}")
    dep_A = rng.randrange(dep_lo0, dep_hi0 + 1, 30)
    return dep_A, duration


def _construct_forced_rollover(rng, row: dict) -> tuple:
    """hard: draw (dep_A, duration) so day_offset_B is forced to -1 or +1 (never 0)."""
    duration = _draw_duration(rng, row)
    dep_lo0, dep_hi0 = _zero_window(row, duration)
    feasible = []
    if dep_lo0 > 0:
        feasible.append(-1)
    if dep_hi0 < 1439:
        feasible.append(1)
    check(feasible, f"no rollover feasible for {row['a']}-{row['b']} duration={duration}")
    target = rng.choice(feasible)
    if target == -1:
        dep_A = rng.randrange(0, dep_lo0, 30)
    else:
        dep_A = rng.randrange(dep_hi0 + 1, 1440, 30)
    return dep_A, duration


def _compute(row: dict, dep_A: int, duration: int) -> tuple:
    offset_min = _offset_minutes(row)
    signed_offset = -offset_min if row["direction"] == "west" else offset_min
    local_equiv = dep_A + duration
    arr_B_total = local_equiv + signed_offset
    arr_B_clock = arr_B_total % 1440
    day_offset_B = arr_B_total // 1440
    check(day_offset_B in (-1, 0, 1), f"day_offset_B {day_offset_B} out of range")
    return signed_offset, local_equiv, arr_B_total, arr_B_clock, day_offset_B


def _verify_offset_via_inverse(
    dep_A: int, duration: int, arr_B_clock: int, day_label: str, direction: str, offset_hours: Fraction
) -> None:
    """Independent check: inverse derivation from visible data ONLY --
    `dep_A`, `duration`, `arr_B_clock` and `day_label` are exactly what a
    learner is given; `signed_offset_minutes`/`offset_hours` never feed the
    reconstruction, only the final comparison (mirrors
    `_verify_duration_via_inverse` in p4a_circle_duration_build.py, with
    which quantity is known vs derived inverted)."""
    day_off = LABEL_TO_OFFSET[day_label]
    arr_B_total_reconstructed = day_off * 1440 + arr_B_clock
    recomputed_signed_offset = arr_B_total_reconstructed - (dep_A + duration)
    expected_magnitude = round(offset_hours * 60)
    check(
        abs(recomputed_signed_offset) == expected_magnitude,
        f"inverse magnitude {recomputed_signed_offset} != {expected_magnitude}",
    )
    expected_sign = -1 if direction == "west" else 1
    actual_sign = 1 if recomputed_signed_offset > 0 else -1
    check(actual_sign == expected_sign, f"inverse sign {actual_sign} != expected {expected_sign}")


@generator(
    "travel_east_west_zone_difference",
    level=1,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def travel_east_west_zone_difference(rng, mode="easy"):
    """Given a travel journey's departure/arrival clock times, direction,
    flight duration and calendar-day label, find the magnitude of the
    time-zone difference between the two places."""
    if mode not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "easy":
        row = rng.choice(EASY_ROWS)
        dep_A, duration = _construct_no_rollover(rng, row)
    elif mode == "medium":
        row = rng.choice(MEDIUM_ROWS)
        dep_A, duration = _construct_no_rollover(rng, row)
    else:  # hard
        row = rng.choice(HARD_ROWS)
        dep_A, duration = _construct_forced_rollover(rng, row)

    A, B, direction = row["a"], row["b"], row["direction"]
    signed_offset, local_equiv, arr_B_total, arr_B_clock, day_offset_B = _compute(row, dep_A, duration)
    day_label = DAY_LABELS[day_offset_B]
    day_phrase = DAY_PHRASE[day_offset_B]
    ahead_behind = "ahead of" if direction == "east" else "behind"

    answer, answer_text = _format_hours(row["offset"])

    _verify_offset_via_inverse(dep_A, duration, arr_B_clock, day_label, direction, row["offset"])

    dep_clock = _clock(dep_A)
    arr_clock = _clock(arr_B_clock)
    duration_text = _duration_text(duration)

    local_equiv_days, local_equiv_clock = divmod(local_equiv, 1440)

    text = (
        f"A traveller departs {A} at {dep_clock}, travelling {direction}, on a flight lasting "
        f"{duration_text}. They land in {B} at {arr_clock} local time, {day_phrase} {A}. "
        f"How many hours {ahead_behind} {A} is {B}'s time zone?"
    )
    answer_hour_word = "hour" if answer_text in ("1", "-1") else "hours"
    working = [
        f"{duration_text} after {dep_clock} (with no zone change) would be {_clock(local_equiv_clock)}"
        + (f", {local_equiv_days} day{'s' if local_equiv_days != 1 else ''} later" if local_equiv_days else "")
        + f" in {A}'s time zone.",
        f"{B} is stated to receive the flight at {arr_clock} local time, {day_phrase} {A}.",
        f"Reconstructing the difference: {abs(round(row['offset'] * 60))} minutes "
        f"= {answer_text} {answer_hour_word} -> {B} is {answer_text} {answer_hour_word} {ahead_behind} {A}.",
    ]

    # -- misconception distractors -------------------------------------
    candidates: list[str] = []

    # omits-day-adjustment: only offered at hard tier, where day_offset_B is guaranteed nonzero and
    # so this collides with nothing; at easy/medium day_offset_B == 0 always, which would make this
    # identical to the correct answer, so it is honestly withheld there.
    if mode == "hard":
        naive_signed = arr_B_clock - local_equiv
        naive_val, naive_txt = _format_hours(Fraction(abs(naive_signed), 60))
        check_distinct(answer, naive_val, msg="omits-day-adjustment collided with the answer")
        candidates.append(naive_txt)

    # reports-duration-not-offset: always exercisable.
    duration_val, duration_txt = _format_hours(Fraction(duration, 60))
    check_distinct(answer, duration_val, msg="reports-duration-not-offset collided with the answer")
    candidates.append(duration_txt)

    # day-term-sign-flip: only meaningfully distinct at hard tier, same
    # reasoning as omits-day-adjustment above.
    if mode == "hard":
        wrong_reconstructed = (-day_offset_B) * 1440 + arr_B_clock
        wrong_signed = wrong_reconstructed - local_equiv
        wrong_val, wrong_txt = _format_hours(Fraction(abs(wrong_signed), 60))
        check_distinct(answer, wrong_val, msg="day-term-sign-flip collided with the answer")
        candidates.append(wrong_txt)

    # off-by-rounding: only the Adelaide/Perth row has a non-whole-hour
    # answer (1.5h) to round away -- exercisable at medium AND hard tier,
    # whichever tier happens to draw that row; every other row's answer is
    # already a whole number, so rounding it is a no-op, not a misconception.
    if row["offset"].denominator != 1:
        rounded_val = round(row["offset"])
        rounded_val, rounded_txt = int(rounded_val), str(int(rounded_val))
        check_distinct(answer, rounded_val, msg="off-by-rounding collided with the answer")
        candidates.append(rounded_txt)

    # The four named ones structurally cannot supply >=2 distinct options at easy tier or at most
    # medium-tier rows (omits-day-adjustment and day-term-sign-flip both collide with the answer
    # whenever day_offset_B == 0, i.e. every easy/medium seed; off-by-rounding only exists for the
    # one non-whole row). A flat +1 hour miscount is always distinct (a nonzero constant shift) and
    # always preserves the answer's own whole/half-hour shape, so it also closes the "answer is the
    # only option of its shape" gap for the Adelaide/Perth row without disturbing any tier where the
    # named misconceptions already supply enough options.
    off_by_one_val, off_by_one_txt = _format_hours(row["offset"] + 1)
    check_distinct(answer, off_by_one_val, msg="off-by-one-hour collided with the answer")
    decimal_answer = row["offset"].denominator != 1
    has_decimal_candidate = any("." in c for c in candidates)
    if decimal_answer and not has_decimal_candidate:
        candidates.insert(0, off_by_one_txt)
    elif len(candidates) < 2:
        candidates.append(off_by_one_txt)

    # Shape the options like the key: a half-hour key shows every option to one
    # decimal place ("3.0" beside "1.5"), a whole-hour key keeps only whole-hour
    # options (a lone "3.5" or "9.5" would give the key away by its decimal
    # point), topped up with +/-1 and +2 hour miscounts.
    places = 0 if answer_text.lstrip("-").isdigit() else 1
    offset = Fraction(row["offset"])
    pool = [Fraction(c) for c in candidates] + [offset + 1, offset - 1, offset + 2]
    distractors = pick_distractors(
        answer_text,
        [stated_precision_string(v, places) for v in pool if v > 0 and (v * (10**places)).denominator == 1],
        n=4,
    )

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )
