"""Transport-time comparison and effective annual rate, two senior ``acara-senior-v8-4`` skills, both ``answer:
ordered_list`` (an ordered PAIR of computed numbers, each graded to its own stated precision -- see
``mathgen.grading.grade_ordered_list``'s per-item numeric comparison, and the worked example this module
follows, ``mathgen/generators/p3c_recursion_build.py``).

Two generators:

- ``p19i_transport_time_compare`` (``senior-compare-the-time-taken-to-
  travel-a-specific``, ACMEM082): given one stated distance and two named
  transport modes, each with its own stated average speed, compute the
  travel time for each mode (time = distance / speed). Never a single
  "winner" label -- the deliverable is always the ordered pair (mode 1's
  time, mode 2's time). Distinguished from the shipping neighbour
  ``f10-timetable-itinerary-duration`` (reads a duration from a stated
  timetable; never computes distance/speed).

- ``p19i_effective_annual_rate`` (``senior-calculate-the-effective-annual-
  rate-of-interest-and``, ACMGM095): given one stated nominal annual rate
  and two named compounding frequencies (from daily n=365, monthly n=12,
  quarterly n=4, six-monthly n=2), compute EAR = (1 + r/n)^n - 1 at each
  frequency. Distinguished from the two shipping neighbours: ``simple-
  interest`` (flat, non-compounding -- never confused with this compounding
  formula) and ``senior-investigate-the-effect-of-the-interest-rate-and``
  (computes a repayment TERM/period, never an effective rate).

Both generators use exact ``Fraction`` arithmetic throughout and round only once, at final display.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    numeric_answer,
    pairwise_distinct,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
)


# ============================================================================
# p19i_transport_time_compare
# ============================================================================

_MIN_TRIP_HOURS = Fraction(3, 10)

# (name, speed_lo, speed_hi [km/h], plausible max single-trip distance [km])
_MODES = [
    ("walking", 4, 6, 20),
    ("cycling", 15, 25, 60),
    ("car", 40, 100, 400),
    ("train", 60, 160, 500),
    ("boat", 10, 40, 150),
]


def _is_exact_at(value: Fraction, places: int) -> bool:
    return (value * (10**places)).denominator == 1


def _min_distance(m1, m2) -> int:
    """Smallest distance (multiple of 5 km, never under 10) at which even the
    faster mode's top speed still takes at least 0.3 h (18 minutes)."""
    fastest = max(m1[2], m2[2])
    return max(10, 5 * math.ceil(_MIN_TRIP_HOURS * fastest / 5))


def _draw_distance(rng, dist_max: int, dist_min: int = 10) -> int:
    return rng.randint(dist_min // 5, dist_max // 5) * 5


def _both_whole_hours(r1: Fraction, r2: Fraction) -> bool:
    """True when both rounded times are whole hours -- the one case where
    'rounds to the wrong precision' (nearest whole hour) cannot be
    distinguished from the correctly-rounded 1-d.p. answer."""
    return r1.denominator == 1 and r2.denominator == 1


def _trip_long_enough(t1: Fraction, t2: Fraction) -> bool:
    """True when BOTH computed times (hours) are at least 18 minutes, so the
    faster mode is never an implausibly short trip."""
    return min(t1, t2) >= _MIN_TRIP_HOURS


def _degenerate_pair(r1, r2) -> bool:
    """True when the two rounded values are equal -- the swap distractor
    (and, for exam, the wrong-precision distractor) becomes a no-op then,
    since swapping two equal values reproduces the answer."""
    return r1 == r2


def _inverted_collides(speed1: int, speed2: int, distance: int, r1: Fraction, r2: Fraction) -> bool:
    """True when the DISPLAYED 'inverted' distractor (round(speed1/distance,
    1), round(speed2/distance, 1)) coincides with the displayed answer or
    its swap -- checked on the rounded, displayed values (not the exact
    fractions), because a rounding coincidence collides even when the exact
    values are not literally reciprocal."""
    inv1 = round_half_up(Fraction(speed1, distance), 1)
    inv2 = round_half_up(Fraction(speed2, distance), 1)
    return (inv1, inv2) in ((r1, r2), (r2, r1))


def _divisors_in_range(n: int, lo: int, hi: int) -> list[int]:
    lo = max(lo, 1)
    return [d for d in range(lo, hi + 1) if n % d == 0]


def _draw_kmh_speed_pair(rng, m1, m2, dist_max, *, exact: bool):
    dist_min = _min_distance(m1, m2)
    """Draw a distance and two plain km/h speeds and return
    (distance, speed1, speed2, t1, t2) where t1/t2 (hours, Fraction) are
    exact-at-1dp iff `exact` (constructed directly from the divisors of
    distance x 10, never guessed by rejection sampling), and where the
    rounded pair is never BOTH whole hours (so the 'wrong precision'
    distractor built from this pair is always genuinely exercisable)."""
    for _ in range(200):
        distance = _draw_distance(rng, dist_max, dist_min)
        if exact:
            speeds1 = _divisors_in_range(distance * 10, m1[1], m1[2])
            speeds2 = _divisors_in_range(distance * 10, m2[1], m2[2])
            if not speeds1 or not speeds2:
                continue
            speed1 = rng.choice(speeds1)
            speed2 = rng.choice(speeds2)
            t1 = Fraction(distance, speed1)
            t2 = Fraction(distance, speed2)
            if (
                _trip_long_enough(t1, t2)
                and not _both_whole_hours(t1, t2)
                and not _degenerate_pair(t1, t2)
                and not _inverted_collides(speed1, speed2, distance, t1, t2)
            ):
                return distance, speed1, speed2, t1, t2
        else:
            speed1 = rng.randint(m1[1], m1[2])
            speed2 = rng.randint(m2[1], m2[2])
            t1 = Fraction(distance, speed1)
            t2 = Fraction(distance, speed2)
            ok = _is_exact_at(t1, 1) and _is_exact_at(t2, 1)
            r1, r2 = round_half_up(t1, 1), round_half_up(t2, 1)
            if (
                not ok
                and _trip_long_enough(t1, t2)
                and rounds_unambiguously(float(t1), 1)
                and rounds_unambiguously(float(t2), 1)
                and not _both_whole_hours(r1, r2)
                and not _degenerate_pair(r1, r2)
                and not _inverted_collides(speed1, speed2, distance, r1, r2)
            ):
                return distance, speed1, speed2, t1, t2
    check(False, "could not draw a speed pair with the required exactness")


# Walking's own km/h range (4-6) is too narrow to hold more than one whole
# m/s value or one-decimal km/min value, which would collapse hard/exam's
# speed draw to a single fixed number -- so walking never takes the
# unit-converted slot at hard/exam (it still appears freely at easy/medium,
# and may still take the plain-km/h slot at hard).
_MODES_ALT_UNIT_ELIGIBLE = [m for m in _MODES if m[0] != "walking"]


def _mps_speed_for(m, rng):
    """An integer m/s speed whose exact km/h equivalent (x 3.6) lands in
    mode `m`'s own plausible km/h range."""
    lo_mps = math.ceil(m[1] / 3.6)
    hi_mps = math.floor(m[2] / 3.6)
    check(lo_mps <= hi_mps, f"no whole m/s value lands inside {m[0]}'s km/h range")
    return rng.randint(lo_mps, hi_mps)


def _kmpmin_speed_for(m, rng):
    """A one-decimal km/min speed whose exact km/h equivalent (x 60) lands in
    mode `m`'s own plausible km/h range."""
    lo_tenths = math.ceil((m[1] / 60) * 10)
    hi_tenths = math.floor((m[2] / 60) * 10)
    check(lo_tenths <= hi_tenths, f"no one-decimal km/min value lands inside {m[0]}'s km/h range")
    return Fraction(rng.randint(lo_tenths, hi_tenths), 10)


def round_down_1dp(value: Fraction) -> Fraction:
    """Truncate (round toward zero) to 1 d.p. -- the 'chopped, not rounded' slip."""
    return Fraction(math.floor(value * 10), 10)


def _fmt_hours(value: Fraction) -> str:
    return stated_precision_string(value, 1)


def _fmt_minutes(value: Fraction) -> str:
    return stated_precision_string(value, 0)


def _answer_pair_hours(t1: Fraction, t2: Fraction):
    r1 = round_half_up(t1, 1)
    r2 = round_half_up(t2, 1)
    return [numeric_answer(r1), numeric_answer(r2)], f"{_fmt_hours(r1)}, {_fmt_hours(r2)}"


def _answer_pair_minutes(m1: Fraction, m2: Fraction):
    r1 = round_half_up(m1, 0)
    r2 = round_half_up(m2, 0)
    return [numeric_answer(r1), numeric_answer(r2)], f"{_fmt_minutes(r1)}, {_fmt_minutes(r2)}"


@generator(
    "p19i_transport_time_compare",
    level=12,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p19i_transport_time_compare(rng, variant="easy"):
    """Given one stated distance and two named transport modes, each with its
    own stated average speed, compute the travel time for each mode (time =
    distance / speed). Answer is the ORDERED pair (mode 1's time, mode 2's
    time)."""
    for _ in range(100):
        result = _try_build_transport(rng, variant)
        if result is not None:
            return result
    check(False, "could not draw a transport question whose distractors are all plausible")


# Every option (answer and distractors) must be a believable trip.
_PLAUSIBLE_HOURS = (Fraction(3, 10), Fraction(24))
_PLAUSIBLE_MINUTES = (Fraction(18), Fraction(1440))


def _pair_in_band(text: str, band) -> bool:
    return all(band[0] <= Fraction(part) <= band[1] for part in text.split(", "))


def _try_build_transport(rng, variant):
    """One attempt; None (retryable) when fewer than three plausible,
    distinct distractors exist for the drawn numbers."""
    for _ in range(200):
        if variant == "exam":
            # Both modes need an alt-unit speed (see `_MODES_ALT_UNIT_ELIGIBLE`).
            m1, m2 = rng.sample(_MODES_ALT_UNIT_ELIGIBLE, 2)
        elif variant == "hard":
            # Only mode 2 needs an alt-unit (m/s) speed.
            m1 = rng.choice(_MODES)
            m2 = rng.choice([m for m in _MODES_ALT_UNIT_ELIGIBLE if m != m1])
        else:
            m1, m2 = rng.sample(_MODES, 2)
        # Reject mode pairs whose shared distance range cannot keep the
        # faster trip at 18 minutes or more (e.g. walking vs train).
        if min(m1[3], m2[3]) >= _min_distance(m1, m2) + 5:
            break
    else:
        check(False, "could not pick a feasible mode pair")
    dist_max = min(m1[3], m2[3])
    check(dist_max >= 10, "could not find two modes with a shared plausible distance range")
    name1, name2 = m1[0], m2[0]

    if variant == "easy":
        distance, speed1, speed2, t1, t2 = _draw_kmh_speed_pair(rng, m1, m2, dist_max, exact=True)
        check(
            speed1 * t1 == distance and speed2 * t2 == distance,
            "time computation inconsistent with distance = speed x time",
        )
        question_text = (
            f"A trip covers a distance of {distance} km. By {name1}, the average speed is "
            f"{speed1} km/h. By {name2}, the average speed is {speed2} km/h. Find the time "
            f"taken by {name1} and the time taken by {name2}, each in hours, to 1 decimal place."
        )
        answer, answer_text = _answer_pair_hours(t1, t2)
        r1, r2 = round_half_up(t1, 1), round_half_up(t2, 1)

    elif variant == "medium":
        distance, speed1, speed2, t1, t2 = _draw_kmh_speed_pair(rng, m1, m2, dist_max, exact=False)
        check(
            speed1 * t1 == distance and speed2 * t2 == distance,
            "time computation inconsistent with distance = speed x time",
        )
        question_text = (
            f"A trip covers a distance of {distance} km. By {name1}, the average speed is "
            f"{speed1} km/h. By {name2}, the average speed is {speed2} km/h. Find the time "
            f"taken by {name1} and the time taken by {name2}, each in hours, to 1 decimal place."
        )
        answer, answer_text = _answer_pair_hours(t1, t2)
        r1, r2 = round_half_up(t1, 1), round_half_up(t2, 1)

    elif variant == "hard":
        for _ in range(200):
            distance = _draw_distance(rng, dist_max, _min_distance(m1, m2))
            speed1 = rng.randint(m1[1], m1[2])
            speed2_mps = _mps_speed_for(m2, rng)
            speed2_kmh = Fraction(36, 10) * speed2_mps
            t1 = Fraction(distance, speed1)
            t2 = Fraction(distance, speed2_kmh)
            r1_try, r2_try = round_half_up(t1, 1), round_half_up(t2, 1)
            forgot_try = round_half_up(Fraction(distance, speed2_mps), 1)
            if (
                _trip_long_enough(t1, t2)
                and (not _is_exact_at(t1, 1) or not _is_exact_at(t2, 1))
                and rounds_unambiguously(float(t1), 1)
                and rounds_unambiguously(float(t2), 1)
                and not _both_whole_hours(r1_try, r2_try)
                and not _degenerate_pair(r1_try, r2_try)
                and forgot_try != r2_try
            ):
                break
        else:
            check(False, "could not draw a hard-tier speed pair")
        check(speed2_mps * Fraction(36, 10) == speed2_kmh, "m/s -> km/h conversion inconsistent")
        check(
            speed1 * t1 == distance and speed2_kmh * t2 == distance,
            "time computation inconsistent with distance = speed x time",
        )
        question_text = (
            f"A trip covers a distance of {distance} km. By {name1}, the average speed is "
            f"{speed1} km/h. By {name2}, the average speed is {speed2_mps} m/s. Find the time "
            f"taken by {name1} and the time taken by {name2}, each in hours, to 1 decimal place."
        )
        answer, answer_text = _answer_pair_hours(t1, t2)
        r1, r2 = round_half_up(t1, 1), round_half_up(t2, 1)
        forgot_conversion_t2 = Fraction(distance, speed2_mps)

    elif variant == "exam":
        for _ in range(200):
            distance = _draw_distance(rng, dist_max, _min_distance(m1, m2))
            speed1_mps = _mps_speed_for(m1, rng)
            speed2_kmpmin = _kmpmin_speed_for(m2, rng)
            speed1_kmh = Fraction(36, 10) * speed1_mps
            speed2_kmh = Fraction(60) * speed2_kmpmin
            t1_hours = Fraction(distance, speed1_kmh)
            t2_hours = Fraction(distance, speed2_kmh)
            min1 = t1_hours * 60
            min2 = t2_hours * 60
            r1_try, r2_try = round_half_up(min1, 0), round_half_up(min2, 0)
            forgot1_try = round_half_up(Fraction(distance, speed1_mps) * 60, 0)
            forgot2_try = round_half_up(Fraction(distance, speed2_kmpmin) * 60, 0)
            if (
                _trip_long_enough(t1_hours, t2_hours)
                and (not _is_exact_at(min1, 0) or not _is_exact_at(min2, 0))
                and rounds_unambiguously(float(min1), 0)
                and rounds_unambiguously(float(min2), 0)
                and not (r1_try % 10 == 0 and r2_try % 10 == 0)
                and not _degenerate_pair(r1_try, r2_try)
                and (forgot1_try != r1_try or forgot2_try != r2_try)
            ):
                break
        else:
            check(False, "could not draw an exam-tier speed pair")
        check(speed1_mps * Fraction(36, 10) == speed1_kmh, "m/s -> km/h conversion inconsistent")
        check(speed2_kmpmin * 60 == speed2_kmh, "km/min -> km/h conversion inconsistent")
        check(
            speed1_kmh * t1_hours == distance and speed2_kmh * t2_hours == distance,
            "time computation inconsistent with distance = speed x time",
        )
        speed2_str = stated_precision_string(speed2_kmpmin, 1)
        question_text = (
            f"A trip covers a distance of {distance} km. By {name1}, the average speed is "
            f"{speed1_mps} m/s. By {name2}, the average speed is {speed2_str} km/min. Find the "
            f"time taken by {name1} and the time taken by {name2}, each in minutes, to the "
            f"nearest minute."
        )
        answer, answer_text = _answer_pair_minutes(min1, min2)
        r1, r2 = round_half_up(min1, 0), round_half_up(min2, 0)
        forgot_conversion_min1 = round_half_up(Fraction(distance, speed1_mps) * 60, 0)
        forgot_conversion_min2 = round_half_up(Fraction(distance, speed2_kmpmin) * 60, 0)

    else:
        raise ValueError(f"unknown variant {variant!r}")

    # -- misconception distractors ------------------------------------------
    # Every candidate is a real slip; any whose value falls outside the
    # plausible-trip band (or that collides with the answer / another
    # candidate) is dropped, and the question is redrawn if fewer than three
    # remain.
    a1, a2 = answer_text.split(", ")
    # 1. swap: mode1's time reported for mode2 and vice versa.
    swap_text = f"{a2}, {a1}"

    if variant in ("easy", "medium"):
        band, fmt = _PLAUSIBLE_HOURS, _fmt_hours
        # 2. inverted: computes speed / distance instead of distance / speed.
        inv1 = round_half_up(Fraction(speed1, distance), 1)
        inv2 = round_half_up(Fraction(speed2, distance), 1)
        # 3. wrong precision: rounds to the nearest whole hour instead of 1 d.p.
        wp = (round_half_up(r1, 0), round_half_up(r2, 0))
        # 4. truncates instead of rounding; 5/6. one speed used for both trips.
        tr = (round_down_1dp(t1), round_down_1dp(t2))
        candidates = [
            swap_text,
            f"{fmt(inv1)}, {fmt(inv2)}",
            f"{fmt(wp[0])}, {fmt(wp[1])}",
            f"{fmt(tr[0])}, {fmt(tr[1])}",
            f"{a1}, {a1}",
            f"{a2}, {a2}",
        ]
    elif variant == "hard":
        band, fmt = _PLAUSIBLE_HOURS, _fmt_hours
        # 2. forgot the unit conversion: uses the raw m/s number as km/h.
        forgot_t2 = round_half_up(forgot_conversion_t2, 1)
        # 3. wrong precision: rounds to the nearest whole hour instead of 1 d.p.
        wp = (round_half_up(r1, 0), round_half_up(r2, 0))
        # 4. truncates instead of rounding; 5/6. one trip's time used for both.
        tr = (round_down_1dp(t1), round_down_1dp(t2))
        candidates = [
            swap_text,
            f"{fmt(r1)}, {fmt(forgot_t2)}",
            f"{fmt(wp[0])}, {fmt(wp[1])}",
            f"{fmt(tr[0])}, {fmt(tr[1])}",
            f"{a1}, {a1}",
            f"{a2}, {a2}",
        ]
    else:  # exam
        band, fmt = _PLAUSIBLE_MINUTES, _fmt_minutes
        # 2. forgot the m/s -> km/h conversion for mode 1 only.
        # 3. wrong precision: rounds to the nearest 10 minutes.
        wp = (round_half_up(Fraction(int(r1), 10), 0) * 10, round_half_up(Fraction(int(r2), 10), 0) * 10)
        # 4. 100 minutes to the hour: converts hours to minutes with x100.
        h100 = (round_half_up(t1_hours * 100, 0), round_half_up(t2_hours * 100, 0))
        # 5/6. one trip's time used for both.
        candidates = [
            swap_text,
            f"{fmt(wp[0])}, {fmt(wp[1])}",
            f"{fmt(h100[0])}, {fmt(h100[1])}",
            f"{fmt(forgot_conversion_min1)}, {a2}",
            f"{a1}, {a1}",
            f"{a2}, {a2}",
        ]

    valid = []
    for c in candidates:
        if c == answer_text or c in valid or not _pair_in_band(c, band):
            continue
        valid.append(c)
    if swap_text not in valid or len(valid) < 3 or not _pair_in_band(answer_text, band):
        return None
    if not pairwise_distinct(answer_text, *valid):
        return None
    # swap always shown; two more from the rest, in priority order.
    distractors = pick_distractors(answer_text, valid, n=3)
    if swap_text not in distractors:
        distractors = [swap_text] + [d for d in distractors if d != swap_text][:2]

    working = [
        f"Distance = {distance} km.",
        f"Time by {name1} = distance / speed.",
        f"Time by {name2} = distance / speed.",
        f"Answer: {answer_text}",
    ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# ============================================================================
# p19i_effective_annual_rate
# ============================================================================

# (name, n)
_FREQUENCIES = [("daily", 365), ("monthly", 12), ("quarterly", 4), ("six-monthly", 2)]
_FREQ_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]

_EASY_RATES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
_HARD_RATES = [Fraction(x, 10) for x in range(20, 151, 5)]  # 2.0 .. 15.0, step 0.5

# exam tier: clean 2-d.p. per-period percentages for each eligible
# "frequency 1", chosen so per_period x n1 (the recovered nominal annual
# rate) lands in roughly the same 1-15% plausible range every other tier
# uses -- e.g. 0.50% every month recovers a 6% nominal annual rate.
_EXAM_PER_PERIOD_PCT = {
    "monthly": [Fraction(x, 100) for x in range(10, 121, 5)],  # 0.10 .. 1.20, nominal 1.2 .. 14.4
    "quarterly": [Fraction(x, 100) for x in range(50, 351, 25)],  # 0.50 .. 3.50, nominal 2.0 .. 14.0
    "six-monthly": [Fraction(x, 100) for x in range(100, 751, 50)],  # 1.00 .. 7.50, nominal 2.0 .. 15.0
}


def _ear(r: Fraction, n: int) -> Fraction:
    """Closed-form EAR (as a proportion, not a percentage): (1 + r/n)^n - 1."""
    return (1 + r / n) ** n - 1


def _ear_iterative(r: Fraction, n: int) -> Fraction:
    """Independent recomputation: accumulate the per-period growth factor by
    repeated multiplication instead of exponentiation -- a genuinely
    different code path from the closed form above."""
    factor = Fraction(1)
    step = 1 + r / n
    for _ in range(n):
        factor *= step
    return factor - 1


def _ear_from_period_rate(p: Fraction, n: int) -> Fraction:
    """EAR of a rate that is ALREADY the per-period rate `p` (not a nominal
    annual rate to be divided by `n`), compounded `n` times per year:
    (1 + p)^n - 1. Distinct from ``_ear``, which expects a nominal annual
    rate and divides by `n` itself -- calling ``_ear(p, n)`` with an
    already-per-period `p` divides it by `n` a second time."""
    return (1 + p) ** n - 1


def _ear_from_period_rate_iterative(p: Fraction, n: int) -> Fraction:
    """Independent recomputation of ``_ear_from_period_rate`` by repeated
    multiplication instead of exponentiation."""
    factor = Fraction(1)
    step = 1 + p
    for _ in range(n):
        factor *= step
    return factor - 1


# Hard tier states each frequency DESCRIPTIVELY (an interval), so the learner
# must first work out n (12 / months-per-interval, or 365 for a day) before
# applying the formula.
_FREQ_DESCRIPTION = {
    "daily": "every day",
    "monthly": "every month",
    "quarterly": "every 3 months",
    "six-monthly": "every 6 months",
}

# Wrong compounding counts, each a real learner error:
# daily -> 30 (days-in-a-month mistaken for the yearly count), monthly -> 13
# (off-by-one miscount), quarterly -> 3 (the "3" of "every 3 months" or an
# off-by-one taken as n), six-monthly -> 6 (the "6" of "six-monthly"/"every 6
# months" taken as the yearly count instead of 2). Daily uses 30 rather than
# 364 because 364 and 365 round to the same 2-d.p. EAR at every stated rate.
_WRONG_N = {"daily": 30, "monthly": 13, "quarterly": 3, "six-monthly": 6}

# Exam frequency 1 compounds a PER-PERIOD rate `n` times, so a wrong count swings the answer far
# harder than for a nominal rate (six-monthly's "6" would be over 3x the key).
_WRONG_N_PERIOD = {"monthly": 13, "quarterly": 3, "six-monthly": 1}

# Scenario nouns widen prompt variety; the rate sentences the
# oracle test parses are unchanged.
_SCENARIOS = ["savings account", "term deposit", "investment", "bank account", "fixed deposit", "loan"]


# Easy tier used to fix the frequency pair at exactly monthly-vs-quarterly (100% of 200 rows),
# giving only 13 x 2 = 26 possible answer pairs -- memorisable without any understanding of
# compounding. Widen it to the three non-daily pairs (monthly/quarterly, monthly/six-monthly,
# quarterly/six-monthly) so the tier still holds its "mechanical minimum" character (no daily n=365,
# the largest and least mechanical frequency, at easy) while multiplying the achievable answer-pair
# count. Pooled across all four tiers (medium/hard already vary across all six pairs, per
# `test_skill_b_medium_hard_frequencies_vary_across_all_six_pairs`), every one of the six unordered
# pairs, including the two involving daily, is still represented.
_EASY_FREQ_PAIRS = [(1, 2), (1, 3), (2, 3)]


def _pick_two_frequencies(rng, *, vary: bool):
    if not vary:
        idx = rng.choice(_EASY_FREQ_PAIRS)
    else:
        idx = rng.choice(_FREQ_PAIRS)
    order = list(idx)
    rng.shuffle(order)
    return _FREQUENCIES[order[0]], _FREQUENCIES[order[1]]


@generator(
    "p19i_effective_annual_rate",
    level=12,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p19i_effective_annual_rate(rng, variant="easy"):
    """Given a nominal annual interest rate and two named compounding
    frequencies (from daily n=365, monthly n=12, quarterly n=4, six-monthly
    n=2), compute EAR = (1 + r/n)^n - 1 at each frequency. Answer is the
    ORDERED pair (EAR at frequency 1, EAR at frequency 2), each a percentage
    to 2 decimal places."""
    for _ in range(100):
        result = _try_build_ear(rng, variant)
        if result is not None:
            return result
    check(False, "could not draw an EAR question whose distractors are all distinct")


def _try_build_ear(rng, variant):
    """One attempt at building the full question dict. Returns None (a
    retryable failure, never raised) when a misconception distractor would
    coincide with the answer, the swap, or another distractor -- this can
    happen by pure numeric coincidence (e.g. two frequencies whose EAR
    rounds to the same 2-d.p. value at a low rate)."""
    if variant == "easy":
        (name1, n1), (name2, n2) = _pick_two_frequencies(rng, vary=False)
        r = Fraction(rng.choice(_EASY_RATES), 100)
        scenario = rng.choice(_SCENARIOS)
        ear1, ear2 = _ear(r, n1), _ear(r, n2)
        check_equivalent(ear1, _ear_iterative(r, n1), "EAR closed form disagrees with iterative recomputation")
        check_equivalent(ear2, _ear_iterative(r, n2), "EAR closed form disagrees with iterative recomputation")
        r_pct = r * 100
        question_text = (
            f"A nominal annual interest rate of {stated_precision_string(r_pct, 0)}% is applied to a "
            f"{scenario}. Calculate the effective annual rate of interest (EAR) when compounded {name1} and "
            f"when compounded {name2}, each as a percentage to 2 decimal places."
        )

    elif variant == "medium":
        (name1, n1), (name2, n2) = _pick_two_frequencies(rng, vary=True)
        r = Fraction(rng.choice(_EASY_RATES), 100)
        scenario = rng.choice(_SCENARIOS)
        ear1, ear2 = _ear(r, n1), _ear(r, n2)
        check_equivalent(ear1, _ear_iterative(r, n1), "EAR closed form disagrees with iterative recomputation")
        check_equivalent(ear2, _ear_iterative(r, n2), "EAR closed form disagrees with iterative recomputation")
        r_pct = r * 100
        question_text = (
            f"A nominal annual interest rate of {stated_precision_string(r_pct, 0)}% is applied to a "
            f"{scenario}. Calculate the effective annual rate of interest (EAR) when compounded {name1} and "
            f"when compounded {name2}, each as a percentage to 2 decimal places."
        )

    elif variant == "hard":
        (name1, n1), (name2, n2) = _pick_two_frequencies(rng, vary=True)
        r = rng.choice(_HARD_RATES) / 100
        ear1, ear2 = _ear(r, n1), _ear(r, n2)
        check_equivalent(ear1, _ear_iterative(r, n1), "EAR closed form disagrees with iterative recomputation")
        check_equivalent(ear2, _ear_iterative(r, n2), "EAR closed form disagrees with iterative recomputation")
        r_pct = r * 100
        question_text = (
            f"Two accounts each pay a nominal annual interest rate of {stated_precision_string(r_pct, 1)}%. "
            f"Interest is added {_FREQ_DESCRIPTION[name1]} in the first account and "
            f"{_FREQ_DESCRIPTION[name2]} in the second. Calculate the effective annual rate of "
            f"interest (EAR) for each account, as a percentage to 2 decimal places."
        )

    elif variant == "exam":
        # Frequency 1 is the one whose PER-PERIOD rate is stated directly;
        # frequency 2 is a different, named frequency using the recovered
        # nominal annual rate. Frequency 1 is never "daily": even a tiny
        # clean 2-d.p. daily rate recovers an implausible nominal annual
        # rate (365x it) -- see the per-frequency tables below, each chosen
        # so the recovered nominal lands in the same 1-15%-ish plausible
        # range every other tier uses.
        idx1 = rng.choice([1, 2, 3])  # monthly, quarterly, six-monthly
        idx2 = rng.choice([i for i in range(4) if i != idx1])
        name1, n1 = _FREQUENCIES[idx1]
        name2, n2 = _FREQUENCIES[idx2]
        per_period_pct = rng.choice(_EXAM_PER_PERIOD_PCT[name1])
        p = per_period_pct / 100
        r_nominal = p * n1  # recovered nominal annual rate (proportion)
        # `p` is already the PER-PERIOD rate, compounded at its own frequency
        # n1: EAR = (1 + p)^n1 - 1. `_ear(p, n1)` would divide `p` by n1 a
        # second time -- use `_ear_from_period_rate`
        # instead, which does not divide `p` again.
        ear1 = _ear_from_period_rate(p, n1)
        ear2 = _ear(r_nominal, n2)
        check_equivalent(
            ear1, _ear_from_period_rate_iterative(p, n1), "EAR closed form disagrees with iterative recomputation"
        )
        check_equivalent(ear2, _ear_iterative(r_nominal, n2), "EAR closed form disagrees with iterative recomputation")
        check(r_nominal == p * n1, "nominal annual rate recovery is inconsistent")
        period_noun = {"daily": "day", "monthly": "month", "quarterly": "quarter", "six-monthly": "six months"}[name1]
        scenario = rng.choice(_SCENARIOS)
        question_text = (
            f"A {scenario} is opened. "
            f"A rate of {stated_precision_string(per_period_pct, 2)}% is applied every "
            f"{period_noun} ({name1} compounding). Calculate the effective annual rate of interest (EAR) at this "
            f"same {name1} frequency, and the EAR the equivalent nominal annual rate would give if "
            f"instead compounded {name2}, each as a percentage to 2 decimal places."
        )

    else:
        raise ValueError(f"unknown variant {variant!r}")

    r1 = round_half_up(ear1 * 100, 2)
    r2 = round_half_up(ear2 * 100, 2)
    answer = [numeric_answer(r1), numeric_answer(r2)]
    answer_text = f"{stated_precision_string(r1, 2)}, {stated_precision_string(r2, 2)}"

    # -- misconception distractors ------------------------------------------
    # 1. swap: EAR values reported for the wrong frequency.
    swap_text = f"{stated_precision_string(r2, 2)}, {stated_precision_string(r1, 2)}"
    # 2. wrong n: a realistic miscount of the compounding periods, see _WRONG_N.
    _wrong_n = _WRONG_N
    fmt = lambda v: stated_precision_string(v, 2)  # noqa: E731
    if variant == "exam":
        # Same (correct) method as the answer, wrong number of compounding
        # periods: frequency 1 uses the per-period-rate formula (like ear1
        # above), frequency 2 uses the nominal-rate formula (like ear2).
        wn1 = round_half_up(_ear_from_period_rate(p, _WRONG_N_PERIOD[name1]) * 100, 2)
        wn2 = round_half_up(_ear(r_nominal, _wrong_n[name2]) * 100, 2)
    else:
        wn1 = round_half_up(_ear(r, _wrong_n[name1]) * 100, 2)
        wn2 = round_half_up(_ear(r, _wrong_n[name2]) * 100, 2)
    candidates = [swap_text, f"{fmt(wn1)}, {fmt(wn2)}"]
    if variant == "exam":
        # 3. simple interest: the recovered nominal rate quoted as the EAR at
        #    both frequencies (no compounding).
        nominal_pct = round_half_up(r_nominal * 100, 2)
        candidates.append(f"{fmt(nominal_pct)}, {fmt(nominal_pct)}")
        # 4/5. one frequency's EAR reported for both.
        candidates.append(f"{fmt(r1)}, {fmt(r1)}")
        candidates.append(f"{fmt(r2)}, {fmt(r2)}")
    else:
        # 3. simple (non-compounding) interest instead of the compounding formula.
        candidates.append(f"{fmt(r_pct)}, {fmt(r_pct)}")
        # 4/5. one frequency's EAR reported for both.
        candidates.append(f"{fmt(r1)}, {fmt(r1)}")
        candidates.append(f"{fmt(r2)}, {fmt(r2)}")

    # Every option stays within 0.5x-1.5x of the key (per position).
    k1, k2 = r1, r2

    def _bounded(text):
        v1, v2 = (Fraction(x) for x in text.split(", "))
        return k1 / 2 <= v1 <= k1 * Fraction(3, 2) and k2 / 2 <= v2 <= k2 * Fraction(3, 2)

    valid = []
    for c in candidates:
        if c != answer_text and c not in valid and _bounded(c):
            valid.append(c)
    if swap_text not in valid or len(valid) < 3 or not pairwise_distinct(answer_text, *valid):
        return None  # retryable: caller redraws
    distractors = pick_distractors(answer_text, valid, n=3)
    if swap_text not in distractors:
        distractors = [swap_text] + [d for d in distractors if d != swap_text][:2]

    working = [
        "EAR = (1 + r/n)^n - 1, where r is the nominal annual rate (as a proportion) and n the "
        "number of compounding periods per year.",
        f"Answer: {answer_text}",
    ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
