"""Circle measures and timezone duration (f10-circle-measures, f10-circle-and-prism-duration-rates).

- f10-circle-measures: Calculate a circle's circumference or area using π ≈ 3.14
  given its radius or diameter. Easy tier: circumference with radius. Medium tier:
  area with radius. Hard tier: either circumference or area with diameter.
  Rounding to 1 decimal place via half-up via Fraction arithmetic.

- f10-circle-and-prism-duration-rates: Determine total flight duration across
  time zones using 24-hour clock notation. Learner given departure time, arrival
  time in a different zone, zone offset, and connecting layover; compute total
  elapsed flight time in minutes.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors, round_half_up


# ============================================================================
# Skill 1: f10-circle-measures
# ============================================================================

PI = Fraction(314, 100)


def _verify_circle_measures(r: int, measure_type: str, answer_frac: Fraction) -> None:
    """Independent check: recompute raw value via repeated addition
    (different route from multiplication), then round and verify.
    measure_type is 'circumference' or 'area'.
    answer_frac is the Fraction value (e.g., Fraction(816, 10) for 81.6)."""
    if measure_type == "circumference":
        raw_total = sum(PI for _ in range(2 * r))
    elif measure_type == "area":
        raw_total = sum(PI * r for _ in range(r))
    else:
        raise ValueError(f"unknown measure_type {measure_type!r}")

    # Re-derive the rounded value from the raw value
    check(raw_total >= 0, "circle duration must not be negative")
    rounded_frac = round_half_up(raw_total, 1)
    check(rounded_frac == answer_frac, f"repeated-addition check {float(rounded_frac)} != {float(answer_frac)}")


@generator(
    "circle_measures",
    level=1,
    presets={
        "easy": dict(mode="circumference_radius"),
        "medium": dict(mode="area_radius"),
        "hard": dict(mode="diameter_mixed"),
    },
)
def circle_measures(rng, mode="circumference_radius"):
    """Calculate the circumference or area of a circle using π ≈ 3.14."""
    if mode not in ("circumference_radius", "area_radius", "diameter_mixed"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "circumference_radius":
        # Easy: circumference, radius given directly 3..15
        r = rng.randint(3, 15)
        raw = PI * 2 * r
        check(raw >= 0, "circle duration must not be negative")
        rounded = round_half_up(raw, 1)
        answer = float(rounded)

        text = f"A circle has a radius of {r} cm. Using π ≈ 3.14, what is its circumference, in centimetres, to 1 decimal place?"
        answer_text = f"{float(rounded):.1f}"
        working = [
            "Circumference = 2 × π × radius",
            f"Circumference = 2 × 3.14 × {r} = {float(raw):.2f}",
            f"To 1 decimal place: {answer_text} cm",
        ]
        _verify_circle_measures(r, "circumference", rounded)

        # Distractors:
        # - confuses_formula: area instead = π * r * r
        # - omits_pi_factor: just 2 * r (no pi)
        confuses = PI * r * r
        check(confuses >= 0, "circle duration must not be negative")
        confuses_rounded = round_half_up(confuses, 1)
        confuses_ans = int(confuses_rounded * 10)

        omits_pi = 2 * r
        check(omits_pi >= 0, "circle duration must not be negative")
        omits_pi_rounded = round_half_up(Fraction(omits_pi, 1), 1)
        omits_pi_ans = int(omits_pi_rounded * 10)

        candidates = [str(confuses_ans / 10), str(omits_pi_ans / 10)]
        distractors = pick_distractors(answer_text, candidates)

    elif mode == "area_radius":
        # Medium: area, radius given directly 3..15
        r = rng.randint(3, 15)
        raw = PI * r * r
        check(raw >= 0, "circle duration must not be negative")
        rounded = round_half_up(raw, 1)
        answer = float(rounded)

        text = f"A circle has a radius of {r} cm. Using π ≈ 3.14, what is its area, in square centimetres, to 1 decimal place?"
        answer_text = f"{float(rounded):.1f}"
        working = [
            "Area = π × radius²",
            f"Area = 3.14 × {r}² = 3.14 × {r * r} = {float(raw):.2f}",
            f"To 1 decimal place: {answer_text} cm²",
        ]
        _verify_circle_measures(r, "area", rounded)

        # Distractors:
        # - confuses_formula: circumference instead = 2 * π * r
        # - omits_pi_factor: just r * r (no pi)
        confuses = PI * 2 * r
        check(confuses >= 0, "circle duration must not be negative")
        confuses_rounded = round_half_up(confuses, 1)
        confuses_ans = int(confuses_rounded * 10)

        omits_pi = r * r
        check(omits_pi >= 0, "circle duration must not be negative")
        omits_pi_rounded = round_half_up(Fraction(omits_pi, 1), 1)
        omits_pi_ans = int(omits_pi_rounded * 10)

        candidates = [str(confuses_ans / 10), str(omits_pi_ans / 10)]
        distractors = pick_distractors(answer_text, candidates)

    else:  # mode == "diameter_mixed"
        # Hard: diameter given 6..30 (even), randomly circumference or area
        d = rng.randint(3, 15) * 2
        check(d >= 6 and d <= 30 and d % 2 == 0, f"diameter {d} not even or out of range")
        r = d // 2
        check(r >= 3, f"radius {r} < 3")

        measure = rng.choice(["circumference", "area"])

        if measure == "circumference":
            raw = PI * 2 * r
            sq_or_plain = "centimetres"
        else:  # area
            raw = PI * r * r
            sq_or_plain = "square centimetres"

        check(raw >= 0, "circle duration must not be negative")
        rounded = round_half_up(raw, 1)
        answer = float(rounded)

        text = f"A circle has a diameter of {d} cm. Using π ≈ 3.14, what is its {measure}, in {sq_or_plain}, to 1 decimal place?"
        answer_text = f"{float(rounded):.1f}"
        if measure == "circumference":
            calc = f"Circumference = 2 × 3.14 × {r} = {float(raw):.2f}"
        else:
            calc = f"Area = 3.14 × {r}² = 3.14 × {r * r} = {float(raw):.2f}"
        working = [
            f"{measure.capitalize()} = {('2 × π × radius' if measure == 'circumference' else 'π × radius²')}",
            f"Radius = {d} ÷ 2 = {r} cm",
            calc,
            f"To 1 decimal place: {answer_text} {sq_or_plain}",
        ]
        _verify_circle_measures(r, measure, rounded)

        # Distractors:
        # - confuses_formula: the other formula
        # - omits_pi_factor: the formula without pi
        # - uses_diameter_as_radius: use d as r instead of r = d/2
        if measure == "circumference":
            confuses = PI * r * r  # area
            omits_pi = 2 * r
            wrong_r = PI * 2 * d  # diameter as radius
        else:  # area
            confuses = PI * 2 * r  # circumference
            omits_pi = r * r
            wrong_r = PI * d * d  # diameter as radius

        check(confuses >= 0, "circle duration must not be negative")
        confuses_rounded = round_half_up(confuses, 1)
        confuses_ans = int(confuses_rounded * 10)

        check(omits_pi >= 0, "circle duration must not be negative")
        omits_pi_rounded = round_half_up(Fraction(omits_pi, 1), 1)
        omits_pi_ans = int(omits_pi_rounded * 10)

        check(wrong_r >= 0, "circle duration must not be negative")
        wrong_r_rounded = round_half_up(wrong_r, 1)
        wrong_r_ans = int(wrong_r_rounded * 10)

        candidates = [str(confuses_ans / 10), str(omits_pi_ans / 10), str(wrong_r_ans / 10)]
        distractors = pick_distractors(answer_text, candidates)

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Skill 2: f10-circle-and-prism-duration-rates
# ============================================================================


def _hm(minutes: int) -> str:
    """Minutes as hours and minutes, e.g. 270 -> '4 h 30 min'."""
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h} h {m} min"
    return f"{h} h" if h else f"{m} min"


def _clock(m: int) -> str:
    """Format minutes-since-midnight (0..1439) as "HH:MM" 24-hour string."""
    hours = m // 60
    mins = m % 60
    return f"{hours:02d}:{mins:02d}"


def _verify_duration_via_inverse(
    dep_A: int,
    arr_B: int,
    offset_minutes: int,
    layover: int,
    wrapped: bool,
    answer: int,
) -> None:
    """Independent check: inverse derivation from visible data.
    Given departure (Zone A), arrival (Zone B), offset, layover, and wrapped flag,
    re-derive flight_duration using the learner-facing formula."""
    wrap_term = 1440 if wrapped else 0
    arr_in_A_frame_clock = (arr_B + offset_minutes) % 1440
    true_arrival_A_frame = arr_in_A_frame_clock + wrap_term
    flight_duration = true_arrival_A_frame - dep_A
    recomputed = flight_duration + layover
    check(recomputed == answer, f"inverse check {recomputed} != {answer}")


@generator(
    "circle_and_prism_duration_rates",
    level=1,
    presets={
        "easy": dict(mode="no_layover"),
        "medium": dict(mode="with_layover"),
        "hard": dict(mode="midnight_crossing"),
    },
)
def circle_and_prism_duration_rates(rng, mode="no_layover"):
    """Determine total flight duration across time zones given clock times and
    zone offsets."""
    if mode not in ("no_layover", "with_layover", "midnight_crossing"):
        raise ValueError(f"unknown mode {mode!r}")

    # Common setup: offset is 1, 2, or 3 hours (zone B is behind A)
    offset_hours = rng.choice([1, 2, 3])
    offset_minutes = offset_hours * 60
    hour_word = f"{offset_hours} hour" + ("" if offset_hours == 1 else "s")

    if mode == "no_layover":
        # Easy: single flight leg, no layover, same-day arrival
        layover = 0
        flight_duration = rng.randint(4, 20) * 15  # 60..300 min
        dep_A = rng.randint(12, 60) * 15  # 03:00..15:00
        arr_in_A_frame_clock = dep_A + flight_duration
        wrapped = False
        check(arr_in_A_frame_clock < 1440, f"arrival without wrapping {arr_in_A_frame_clock} >= 1440")
        check(arr_in_A_frame_clock >= offset_minutes, f"arrival time {arr_in_A_frame_clock} < offset {offset_minutes}")
        arr_B = arr_in_A_frame_clock - offset_minutes

        text = f"A flight departs Zone A at {_clock(dep_A)}, arrives Zone B at {_clock(arr_B)}, on the same day. Zone B is {hour_word} behind Zone A. What is the flight duration, in minutes?"
        answer = flight_duration
        answer_text = str(answer)
        working = [
            f"Zone B is {hour_word} behind Zone A, so {_clock(arr_B)} in Zone B is {_clock(arr_in_A_frame_clock)} in Zone A's time.",
            f"Flight duration = {_clock(arr_in_A_frame_clock)} − {_clock(dep_A)} = {_hm(answer)} = {answer} minutes.",
        ]
        _verify_duration_via_inverse(dep_A, arr_B, offset_minutes, layover, wrapped, answer)

        # Distractors:
        # - ignores_offset: arr_B - dep_A (no offset)
        # - wrong_offset_direction: subtract offset twice instead of adding
        ignores_offset_val = (arr_B - dep_A) + layover
        wrong_offset_val = (arr_B - offset_minutes - dep_A) + layover

        candidates = [str(ignores_offset_val), str(wrong_offset_val)]
        distractors = pick_distractors(answer_text, candidates)

    elif mode == "with_layover":
        # Medium: same as easy, plus a layover
        layover = rng.randint(1, 6) * 15  # 15..90 min
        flight_duration = rng.randint(4, 20) * 15  # 60..300 min
        dep_A = rng.randint(12, 60) * 15  # 03:00..15:00
        arr_in_A_frame_clock = dep_A + flight_duration
        wrapped = False
        check(arr_in_A_frame_clock < 1440, f"arrival without wrapping {arr_in_A_frame_clock} >= 1440")
        check(arr_in_A_frame_clock >= offset_minutes, f"arrival time {arr_in_A_frame_clock} < offset {offset_minutes}")
        arr_B = arr_in_A_frame_clock - offset_minutes

        text = (
            f"A flight departs Zone A at {_clock(dep_A)} and lands in Zone B at {_clock(arr_B)} on the same day "
            f"(both times are local). Zone B is {hour_word} behind Zone A. After landing, the passenger waits a "
            f"{layover}-minute layover. What is the total time from departure to the end of the layover, in minutes?"
        )
        answer = flight_duration + layover
        answer_text = str(answer)
        working = [
            f"Zone B is {hour_word} behind Zone A, so {_clock(arr_B)} in Zone B is {_clock(arr_in_A_frame_clock)} in Zone A's time.",
            f"Flight duration = {_clock(arr_in_A_frame_clock)} − {_clock(dep_A)} = {_hm(flight_duration)} = {flight_duration} minutes.",
            f"Total time = {flight_duration} + {layover} = {answer} minutes.",
        ]
        _verify_duration_via_inverse(dep_A, arr_B, offset_minutes, layover, wrapped, answer)

        # Distractors:
        # - ignores_offset: arr_B - dep_A + layover (no offset)
        # - wrong_offset_direction: subtract offset twice
        # - omits_layover: just flight_duration
        ignores_offset_val = (arr_B - dep_A) + layover
        wrong_offset_val = (arr_B - offset_minutes - dep_A) + layover
        omits_layover_val = arr_B - dep_A + 1440 - offset_minutes

        candidates = [str(ignores_offset_val), str(wrong_offset_val), str(flight_duration)]
        distractors = pick_distractors(answer_text, candidates)

    else:  # mode == "midnight_crossing"
        # Hard: forced midnight crossing (arrival next day)
        dep_A = rng.randint(80, 95) * 15  # 20:00..23:45
        flight_duration = rng.randint(28, 40) * 15  # 420..600 min (7-10 hours)
        layover = rng.randint(1, 6) * 15  # 15..90 min
        arr_in_A_frame_total = dep_A + flight_duration
        check(
            arr_in_A_frame_total >= 1620 and arr_in_A_frame_total <= 2025,
            f"arrival {arr_in_A_frame_total} not in expected range for hard tier",
        )
        wrapped = True
        arr_in_A_frame_clock = arr_in_A_frame_total - 1440
        check(
            arr_in_A_frame_clock >= 180 and arr_in_A_frame_clock <= 585,
            f"unwrapped arrival {arr_in_A_frame_clock} not in expected range",
        )
        check(arr_in_A_frame_clock >= offset_minutes, f"arrival time {arr_in_A_frame_clock} < offset {offset_minutes}")
        arr_B = arr_in_A_frame_clock - offset_minutes

        text = (
            f"A flight departs Zone A at {_clock(dep_A)} on Day 1 and lands in Zone B at {_clock(arr_B)} the next day "
            f"(both times are local). Zone B is {hour_word} behind Zone A. After landing, the passenger waits a "
            f"{layover}-minute layover. What is the total time from departure to the end of the layover, in minutes?"
        )
        answer = flight_duration + layover
        answer_text = str(answer)
        until_midnight = 1440 - dep_A
        working = [
            f"Zone B is {hour_word} behind Zone A, so {_clock(arr_B)} in Zone B is {_clock(arr_in_A_frame_clock)} "
            f"on Day 2 in Zone A's time.",
            f"From {_clock(dep_A)} to midnight is {_hm(until_midnight)}, and from midnight to "
            f"{_clock(arr_in_A_frame_clock)} is {_hm(arr_in_A_frame_clock)}.",
            f"Flight duration = {until_midnight} + {arr_in_A_frame_clock} = {flight_duration} minutes.",
            f"Total time = {flight_duration} + {layover} = {answer} minutes.",
        ]
        _verify_duration_via_inverse(dep_A, arr_B, offset_minutes, layover, wrapped, answer)

        # Distractors:
        # - ignores_offset: (arr_B - dep_A) + wrap + layover
        # - wrong_offset_direction: subtract offset twice
        # - omits_layover: just flight_duration
        # - forgets_midnight_wrap: abs((arr_B + offset) - dep_A) + layover (magnitude of wrong negative)
        wrap_term = 1440
        ignores_offset_val = (arr_B - dep_A) + wrap_term + layover
        wrong_offset_val = (arr_B - offset_minutes - dep_A) + wrap_term + layover
        naive_signed = (arr_B + offset_minutes) - dep_A
        # naive_signed is negative here; the "fix" is abs + layover
        forgets_wrap_val = abs(naive_signed) + layover

        candidates = [str(ignores_offset_val), str(wrong_offset_val), str(flight_duration), str(forgets_wrap_val)]
        distractors = pick_distractors(answer_text, candidates)

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
