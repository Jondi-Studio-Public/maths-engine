"""Timetable/itinerary duration lookup (f10-timetable-itinerary-duration).

Skill: Given a timetable listing named events/journeys and their start/end times,
determine the duration of a named event/journey.

Difficulty tiers:
- easy: 2 rows, asked row's times within same hour (no hour boundary crossing)
- medium: 3 rows, asked row's times cross one hour boundary (minute borrowing needed)
- hard: 4 rows, asked row's times cross noon (12am/12pm) boundary
"""

from __future__ import annotations

from ..core import check, generator, pick_distractors


# Fixed vocabulary of event/journey names (deterministic, no repeated labels per seed, letters only)
EVENT_LABELS = ["Bus A", "Bus B", "Train C", "Train D", "Ferry E", "Ferry F"]


def _minute_word(n):
    """Agree a minute count with its noun: "1 minute" but "2 minutes"."""
    return "minute" if n == 1 else "minutes"


def _clock12(total_minutes):
    """Convert minutes-since-midnight (0-1439) to 12-hour clock string like '9:05am'."""
    total_minutes %= 24 * 60
    hour24, minute = divmod(total_minutes, 60)
    suffix = "am" if hour24 < 12 else "pm"
    hour = hour24 % 12 or 12
    return f"{hour}:{minute:02d}{suffix}"


def _minutes_since_midnight_construction(hour, minute, am_pm):
    """Construction path: convert 12-hour time to minutes-since-midnight.

    Uses: 12am->0, 1am-11am->H*60+M, 12pm->720, 1pm-11pm->(H+12)*60+M
    """
    if am_pm == "am":
        if hour == 12:
            return 0 + minute
        else:
            return hour * 60 + minute
    else:  # pm
        if hour == 12:
            return 720 + minute
        else:
            return (hour + 12) * 60 + minute


def _minutes_since_midnight_verify(hour, minute, am_pm):
    """Independent verification path: convert via 24-hour normalization first.

    Never re-calls the construction method. Uses:
    Normalize to 24-hour: hour24 = hour if am/pm=="am" else hour+12, but special case 12
    Then: hour24*60+minute
    """
    if am_pm == "am":
        # 12am is hour 0, 1am-11am are hours 1-11
        if hour == 12:
            hour24 = 0
        else:
            hour24 = hour
    else:  # pm
        # 12pm is hour 12, 1pm-11pm are hours 13-23
        if hour == 12:
            hour24 = 12
        else:
            hour24 = hour + 12

    return hour24 * 60 + minute


def _verify_timetable_duration(
    timetable_rows, asked_label, start_h, start_m, start_ap, end_h, end_m, end_ap, answer_duration
):
    """Independent verification: locate row by label, convert times via separate path,
    assert durations match and both time conversions agree with construction."""

    # Re-find the row by label (never trust stored index)
    found_row = None
    for label, sh, sm, sap, eh, em, eap in timetable_rows:
        if label == asked_label:
            found_row = (label, sh, sm, sap, eh, em, eap)
            break

    check(found_row is not None, f"asked label {asked_label!r} not found in timetable")

    label, row_sh, row_sm, row_sap, row_eh, row_em, row_eap = found_row

    # Verify the row matches what we were given
    check(row_sh == start_h and row_sm == start_m and row_sap == start_ap, f"start time mismatch in re-found row")
    check(row_eh == end_h and row_em == end_m and row_eap == end_ap, f"end time mismatch in re-found row")

    # Convert using the independent verification method
    start_min_verify = _minutes_since_midnight_verify(start_h, start_m, start_ap)
    end_min_verify = _minutes_since_midnight_verify(end_h, end_m, end_ap)

    # Also verify these match the construction method
    start_min_construct = _minutes_since_midnight_construction(start_h, start_m, start_ap)
    end_min_construct = _minutes_since_midnight_construction(end_h, end_m, end_ap)

    check(
        start_min_verify == start_min_construct,
        f"start time conversion mismatch: verify={start_min_verify}, construct={start_min_construct}",
    )
    check(
        end_min_verify == end_min_construct,
        f"end time conversion mismatch: verify={end_min_verify}, construct={end_min_construct}",
    )

    # Compute duration via the independent method
    expected_duration = end_min_verify - start_min_verify

    check(
        expected_duration == answer_duration, f"duration mismatch: verify={expected_duration}, answer={answer_duration}"
    )


@generator(
    "timetable_itinerary_duration",
    level=2,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def timetable_itinerary_duration(rng, tier="easy"):
    """Generate a timetable question asking for the duration of a named row."""

    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")

    if tier == "easy":
        num_rows = 2
        # Asked row's times must be within same hour
        # Generate: asked hour (6am-6pm), start minute, end minute in same hour

        asked_hour = rng.randint(6, 18)  # 6am to 6pm
        asked_start_min = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45])  # 5-45 to leave room for end_min
        asked_end_min = rng.randint(asked_start_min + 5, 59)  # must be > start, stays in hour

        asked_start_h, asked_start_m, asked_start_ap = asked_hour, asked_start_min, "am" if asked_hour < 12 else "pm"
        asked_end_h, asked_end_m, asked_end_ap = asked_hour, asked_end_min, "am" if asked_hour < 12 else "pm"

        duration_asked = asked_end_min - asked_start_min

        # Generate another row with different duration
        # Reroll until the other row's duration differs from asked
        other_hour = rng.choice([h for h in range(6, 19) if h != asked_hour])
        while True:
            other_start_min = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45])  # 5-45 to leave room for end_min
            other_end_min = rng.randint(other_start_min + 5, 59)
            duration_other = other_end_min - other_start_min
            if duration_other != duration_asked:
                break

        other_start_h, other_start_m, other_start_ap = other_hour, other_start_min, "am" if other_hour < 12 else "pm"
        other_end_h, other_end_m, other_end_ap = other_hour, other_end_min, "am" if other_hour < 12 else "pm"

        # Randomly choose which row is asked about
        asked_row_index = rng.randint(0, 1)

        if asked_row_index == 0:
            rows = [
                (asked_start_h, asked_start_m, asked_start_ap, asked_end_h, asked_end_m, asked_end_ap, duration_asked),
                (other_start_h, other_start_m, other_start_ap, other_end_h, other_end_m, other_end_ap, duration_other),
            ]
        else:
            rows = [
                (other_start_h, other_start_m, other_start_ap, other_end_h, other_end_m, other_end_ap, duration_other),
                (asked_start_h, asked_start_m, asked_start_ap, asked_end_h, asked_end_m, asked_end_ap, duration_asked),
            ]

        answer_duration = duration_asked

    elif tier == "medium":
        num_rows = 3
        # Asked row's times must cross one hour boundary (stay within 6am-11am to avoid noon boundary)

        asked_hour = rng.randint(6, 11)  # 6am to 11am (so hour+1 gives 7am-12pm, all valid morning hours)
        asked_start_min = rng.randint(50, 59)  # 50-59 to cross boundary
        asked_end_min = rng.randint(0, 30)  # 0-30 in next hour
        asked_end_hour = asked_hour + 1  # 7-12 (7am-12pm)

        asked_start_ap = "am"  # Always am in this range
        asked_end_ap = "am" if asked_end_hour < 12 else "pm"  # pm only if asked_end_hour is 12 (noon)

        # Duration with hour crossing: (60 - start_min) + end_min
        duration_asked = (60 - asked_start_min) + asked_end_min

        # Generate 2 more rows (not crossing hour boundary)
        other_rows = []
        for i in range(2):
            while True:
                other_hour = rng.randint(6, 17)  # Limit to 6-17 too
                other_start_min = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45])
                other_end_min = rng.randint(other_start_min + 5, 59)
                duration_other = other_end_min - other_start_min
                if duration_other != duration_asked:
                    break
            other_ap = "am" if other_hour < 12 else "pm"
            other_rows.append(
                (other_hour, other_start_min, other_ap, other_hour, other_end_min, other_ap, duration_other)
            )

        # Build timetable rows with asked row at random position
        rows = [None] * 3
        asked_row_index = rng.randint(0, 2)
        rows[asked_row_index] = (
            asked_hour,
            asked_start_min,
            asked_start_ap,
            asked_end_hour,
            asked_end_min,
            asked_end_ap,
            duration_asked,
        )

        j = 0
        for i in range(3):
            if i != asked_row_index:
                rows[i] = other_rows[j]
                j += 1

        answer_duration = duration_asked

    else:  # hard
        num_rows = 4
        # Asked row's times cross noon (12am/12pm) boundary

        # Start in morning (before noon), end in afternoon (after noon)
        asked_start_hour = rng.randint(10, 11)  # 10-11am
        asked_start_min = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        asked_start_ap = "am"

        asked_end_hour = rng.randint(1, 5)  # 1-5pm
        asked_end_min = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        asked_end_ap = "pm"

        # Duration: use construction method
        start_min_total = _minutes_since_midnight_construction(asked_start_hour, asked_start_min, asked_start_ap)
        end_min_total = _minutes_since_midnight_construction(asked_end_hour, asked_end_min, asked_end_ap)
        duration_asked = end_min_total - start_min_total

        # Generate 3 more rows (not crossing noon)
        other_rows = []
        for i in range(3):
            while True:
                other_hour = rng.randint(6, 17)  # Limit to 6-17 to stay on same day
                other_start_min = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45])
                other_end_min = rng.randint(other_start_min + 5, 59)
                duration_other = other_end_min - other_start_min
                if duration_other != duration_asked:
                    break
            other_ap = "am" if other_hour < 12 else "pm"
            other_rows.append(
                (other_hour, other_start_min, other_ap, other_hour, other_end_min, other_ap, duration_other)
            )

        # Build timetable rows with asked row at random position
        rows = [None] * 4
        asked_row_index = rng.randint(0, 3)
        rows[asked_row_index] = (
            asked_start_hour,
            asked_start_min,
            asked_start_ap,
            asked_end_hour,
            asked_end_min,
            asked_end_ap,
            duration_asked,
        )

        j = 0
        for i in range(4):
            if i != asked_row_index:
                rows[i] = other_rows[j]
                j += 1

        answer_duration = duration_asked

    # Assign labels (sample without replacement)
    selected_labels = rng.sample(EVENT_LABELS, num_rows)

    # Build timetable rows with labels
    timetable_rows = []
    for idx, (hour_s, min_s, ap_s, hour_e, min_e, ap_e, dur) in enumerate(rows):
        label = selected_labels[idx]
        timetable_rows.append((label, hour_s, min_s, ap_s, hour_e, min_e, ap_e, dur))

    # Find the asked row (it's at asked_row_index)
    asked_label, asked_start_h, asked_start_m, asked_start_ap, asked_end_h, asked_end_m, asked_end_ap, _ = (
        timetable_rows[asked_row_index]
    )

    # Build question text
    timetable_text = ""
    for label, sh, sm, sap, eh, em, eap, _ in timetable_rows:
        start_time = _clock12(_minutes_since_midnight_construction(sh, sm, sap))
        end_time = _clock12(_minutes_since_midnight_construction(eh, em, eap))
        timetable_text += f"{label} departs {start_time} and arrives {end_time}. "

    question_text = f"{timetable_text}How many minutes does {asked_label}'s journey take?"
    answer_text = str(answer_duration)

    # Verify the answer
    timetable_for_verify = [(label, sh, sm, sap, eh, em, eap) for label, sh, sm, sap, eh, em, eap, _ in timetable_rows]
    # Verify via independent check
    _verify_timetable_duration(
        timetable_for_verify,
        asked_label,
        asked_start_h,
        asked_start_m,
        asked_start_ap,
        asked_end_h,
        asked_end_m,
        asked_end_ap,
        answer_duration,
    )

    # Distractors
    candidates = []

    # Backbone distractor 1: wrong-row (adjacent row in listed order)
    # The adjacent row is either asked_row_index-1 or asked_row_index+1
    if asked_row_index > 0:
        adjacent_idx = asked_row_index - 1
    elif asked_row_index < num_rows - 1:
        adjacent_idx = asked_row_index + 1
    else:
        adjacent_idx = 0  # wrap if asked is last

    wrong_row_duration = timetable_rows[adjacent_idx][7]  # index 7 is the duration
    candidates.append(str(wrong_row_duration))

    # Backbone distractor 2: swapped-times (negative of correct duration)
    swapped_duration = -answer_duration
    candidates.append(str(swapped_duration))

    # Optional: minute-only-subtraction (only when hour boundary is crossed)
    minute_only_distractor = abs(asked_end_m - asked_start_m)
    if asked_start_ap != asked_end_ap or asked_start_h != asked_end_h:
        # Hour boundary crossed
        if minute_only_distractor not in [answer_duration, wrong_row_duration, swapped_duration]:
            candidates.append(str(minute_only_distractor))

    # Optional: am-pm-blind (hard tier only, when noon boundary is crossed)
    if tier == "hard":
        # Compute duration without the +12 adjustment for pm times
        # Naive: treat pm hour as just the number without +12
        if asked_start_ap == "am":
            naive_start_min = asked_start_h * 60 + asked_start_m
        else:
            naive_start_min = asked_start_h * 60 + asked_start_m  # wrong: no +12

        if asked_end_ap == "am":
            naive_end_min = asked_end_h * 60 + asked_end_m
        else:
            naive_end_min = asked_end_h * 60 + asked_end_m  # wrong: no +12

        naive_duration = naive_end_min - naive_start_min
        if naive_duration not in [answer_duration, wrong_row_duration, swapped_duration]:
            candidates.append(str(naive_duration))

    # Use pick_distractors to select unique distractors
    distractors = pick_distractors(answer_text, candidates)

    asked_start_text = _clock12(_minutes_since_midnight_construction(asked_start_h, asked_start_m, asked_start_ap))
    asked_end_text = _clock12(_minutes_since_midnight_construction(asked_end_h, asked_end_m, asked_end_ap))
    working = [
        f"From {asked_start_text} to {asked_end_text} is {answer_duration} {_minute_word(answer_duration)}.",
        f"{asked_label}'s journey takes {answer_duration} {_minute_word(answer_duration)}.",
    ]

    return dict(
        text=question_text,
        latex=rf"\text{{{question_text}}}",
        answer=answer_duration,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
