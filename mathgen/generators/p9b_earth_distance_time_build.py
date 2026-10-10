"""Earth distance along a meridian and longitude-based solar-time difference
(senior-find-distances-between-two-places-on-earth-on / ACMEM160,
senior-find-time-differences-between-two-places-on-earth / ACMEM165).

Both skills grade a plain number (nautical miles / kilometres / hours / degrees of longitude) -- never a clock
time or a calendar day.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_tolerance,
)


# ============================================================================
# Shared rounding/formatting helpers
# ============================================================================


def _fixed_str(value: Fraction, dp: int) -> str:
    """Render an exact Fraction that terminates within `dp` decimal places
    as a fixed-point string ("20.9", "-58.4"), never scientific notation and
    never re-rounded (the caller rounds first via `round_half_up`)."""
    scaled = value * (10**dp)
    check(scaled.denominator == 1, f"{value} does not terminate at {dp} dp")
    n = scaled.numerator
    sign = "-" if n < 0 else ""
    n = abs(n)
    if dp == 0:
        return f"{sign}{n}"
    digits = str(n).zfill(dp + 1)
    whole, frac = digits[:-dp], digits[-dp:]
    return f"{sign}{whole}.{frac}"


def _trim_str(value: Fraction, dp: int = 1) -> str:
    """`_fixed_str` with a trailing ".0" collapsed to a bare integer -- for
    a value display (a degree measure, a working-line coefficient) where an
    exact whole number must never be shown with a spurious decimal point."""
    return str(int(value)) if value.denominator == 1 else _fixed_str(value, dp)


def _hours_phrase(dt: Fraction) -> str:
    """ "1 hour" / "3 hours" / "1.5 hours" -- singular only for an exact 1,
    and never a trailing ".0" on a whole number of hours (a plural after a count
    of one and a trailing zero on a stated quantity are both defects)."""
    word = "hour" if dt == 1 else "hours"
    return f"{_trim_str(dt)} {word}"


# ============================================================================
# Skill 1: senior-find-distances-between-two-places-on-earth-on (ACMEM160)
# ============================================================================

NM_PER_KM = Fraction(1852, 1000)  # 1 nautical mile = 1.852 km, exactly
STATUTE_MILE_PER_KM = Fraction(1609, 1000)  # the mile-to-km factor (misconception only)

# Each row: (place_a, deg_a, min_a, hemi_a, place_b, deg_b, min_b, hemi_b, lon_display)
# Latitudes are stored as (degrees, arcminutes) so that the 1-d.p. decimal
# form (deg + min/60) and the degrees+minutes form (deg, min) are both exact
# for every row by construction (min is always a multiple of 6, since
# 0.1 degree = 6 arcminutes exactly).
SAME_HEMISPHERE = [
    ("Melbourne", 37, 48, "S", "Cairns", 16, 54, "S", "145°E"),
    ("Hobart", 42, 54, "S", "Cairns", 16, 54, "S", "146°E"),
    ("Sydney", 33, 54, "S", "Brisbane", 27, 30, "S", "152°E"),
    ("London", 51, 30, "N", "Accra", 5, 36, "N", "0°"),
    # Widened from 4 to 11 rows so
    # `easy`/`None` clear the shared _MIN_DISTINCT=10 floor without pinning.
    # "Paris/Accra" longitude is drafted as a range ("~1-2°E"); 2°E is used
    # as the single displayed value (within 3 degrees of both real
    # longitudes: Paris 2.3°E, Accra 0.2°W).
    ("Paris", 48, 54, "N", "Accra", 5, 36, "N", "2°E"),
    ("New York", 40, 42, "N", "Bogotá", 4, 42, "N", "74°W"),
    ("Perth", 31, 54, "S", "Denpasar (Bali)", 8, 42, "S", "115.5°E"),
    ("Rome", 41, 54, "N", "Tripoli", 32, 54, "N", "13°E"),
    ("Berlin", 52, 30, "N", "Rome", 41, 54, "N", "13°E"),
    ("Berlin", 52, 30, "N", "Tripoli", 32, 54, "N", "13°E"),
    ("Toronto", 43, 42, "N", "Panama City", 9, 0, "N", "79.5°W"),
]

OPPOSITE_HEMISPHERE = [
    ("Adelaide", 34, 54, "S", "Tokyo", 35, 42, "N", "139°E"),
    ("Perth", 31, 54, "S", "Beijing", 39, 54, "N", "116°E"),
    ("Cairo", 30, 0, "N", "Durban", 29, 54, "S", "31°E"),
    ("Lima", 12, 0, "S", "New York", 40, 42, "N", "76°W"),
    # Widened from 4 to 10 rows, same reason as SAME_HEMISPHERE above.
    ("Lima", 12, 0, "S", "Bogotá", 4, 42, "N", "75.5°W"),
    ("Toronto", 43, 42, "N", "Quito", 0, 12, "S", "79°W"),
    ("Panama City", 9, 0, "N", "Quito", 0, 12, "S", "79°W"),
    ("Ho Chi Minh City", 10, 48, "N", "Jakarta", 6, 12, "S", "106.8°E"),
    ("Singapore", 1, 18, "N", "Jakarta", 6, 12, "S", "105°E"),
    ("Maputo", 25, 54, "S", "Cairo", 30, 0, "N", "32°E"),
]


def _arcmin_total(deg: int, minute: int) -> int:
    """Magnitude of one latitude reading, in arcminutes from the equator."""
    return deg * 60 + minute


def _decimal_degrees(deg: int, minute: int) -> Fraction:
    return Fraction(deg) + Fraction(minute, 60)


def _decimal_lat_str(deg: int, minute: int, hemi: str) -> str:
    check(minute % 6 == 0, f"minute {minute} not a multiple of 6 (not exact at 1 d.p.)")
    return f"{deg}.{minute // 6}°{hemi}"


def _dm_lat_str(deg: int, minute: int, hemi: str) -> str:
    return f"{deg}°{minute:02d}′{hemi}"


def _verify_nm_via_decimal_degrees(
    deg_a: int,
    min_a: int,
    deg_b: int,
    min_b: int,
    same_hemi: bool,
    nm: int,
) -> None:
    """Independent check: construction works directly in arcminutes
    (`_arcmin_total`, an integer add/subtract); this re-derives the same
    total via a genuinely different route -- decimal degrees
    (deg + minutes/60) combined and then multiplied by 60 again -- and
    asserts the two totals agree exactly."""
    dec_a = _decimal_degrees(deg_a, min_a)
    dec_b = _decimal_degrees(deg_b, min_b)
    delta_deg = abs(dec_a - dec_b) if same_hemi else dec_a + dec_b
    nm_check = delta_deg * 60
    check(nm_check == nm, f"decimal-degree route {nm_check} != arcminute route {nm}")


def _verify_km_from_nm(nm: int, km: int) -> None:
    """Independent check for the hard tier: re-derive the km value via the
    same exact 1.852 factor from the nm figure, independently of how
    construction rounded, and confirm the rounding agrees."""
    raw = Fraction(nm) * NM_PER_KM
    check(raw >= 0, "nautical-mile distance must not be negative")
    check(round_half_up(raw, 0) == km, f"km re-derivation {round_half_up(raw, 0)} != {km}")


@generator(
    "earth_meridian_distance",
    level=12,
    presets={
        "easy": dict(mode="same_hemisphere"),
        "medium": dict(mode="opposite_hemisphere"),
        "hard": dict(mode="combined_km"),
    },
)
def earth_meridian_distance(rng, mode="same_hemisphere"):
    """Distance between two places on Earth on the same longitude, along a
    meridian, using the exact definition of the nautical mile (1 minute of
    latitude = 1 nautical mile)."""
    if mode not in ("same_hemisphere", "opposite_hemisphere", "combined_km"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "same_hemisphere":
        row = rng.choice(SAME_HEMISPHERE)
    elif mode == "opposite_hemisphere":
        row = rng.choice(OPPOSITE_HEMISPHERE)
    else:
        row = rng.choice(SAME_HEMISPHERE + OPPOSITE_HEMISPHERE)

    place_a, deg_a, min_a, hemi_a, place_b, deg_b, min_b, hemi_b, lon_display = row
    same_hemi = hemi_a == hemi_b
    arcmin_a = _arcmin_total(deg_a, min_a)
    arcmin_b = _arcmin_total(deg_b, min_b)
    delta_arcmin = abs(arcmin_a - arcmin_b) if same_hemi else arcmin_a + arcmin_b
    nm = delta_arcmin
    _verify_nm_via_decimal_degrees(deg_a, min_a, deg_b, min_b, same_hemi, nm)

    # Misconception values common to every tier (nm-shaped; see below for the
    # hard tier's extra km-conversion-specific ones).
    wrong_hemi_delta = (arcmin_a + arcmin_b) if same_hemi else abs(arcmin_a - arcmin_b)
    forgot_conv_delta_deg = Fraction(delta_arcmin, 60)
    check(forgot_conv_delta_deg >= 0, "degrees of latitude difference must not be negative")
    forgot_conv_val = int(round_half_up(forgot_conv_delta_deg, 0))
    check(delta_arcmin % 2 == 0, f"delta_arcmin {delta_arcmin} is not even")
    halved_val = delta_arcmin // 2

    if mode != "combined_km":
        lat_a_display = _decimal_lat_str(deg_a, min_a, hemi_a)
        lat_b_display = _decimal_lat_str(deg_b, min_b, hemi_b)
        text = (
            f"{place_a} and {place_b} are both located at approximately {lon_display} longitude. "
            f"{place_a} is at latitude {lat_a_display}, and {place_b} is at latitude {lat_b_display}. "
            f"Along a meridian, 1 minute of latitude is exactly 1 nautical mile. What is the "
            f"distance between {place_a} and {place_b} along the meridian, in nautical miles?"
        )
        answer = nm
        answer_text = str(nm)
        working = [
            f"{'Same' if same_hemi else 'Opposite'} hemisphere rule: Δ = {delta_arcmin} arcminutes of latitude",
            f"1 arcminute of latitude = 1 nautical mile, so distance = {nm} nm",
        ]

        check_distinct(answer, wrong_hemi_delta, forgot_conv_val, halved_val)
        candidates = [str(wrong_hemi_delta), str(forgot_conv_val), str(halved_val)]
        distractors = pick_distractors(answer_text, candidates)
    else:
        lat_a_display = _dm_lat_str(deg_a, min_a, hemi_a)
        lat_b_display = _dm_lat_str(deg_b, min_b, hemi_b)
        raw_km = Fraction(nm) * NM_PER_KM
        check(raw_km >= 0, "nautical-mile distance must not be negative")
        km = int(round_half_up(raw_km, 0))
        _verify_km_from_nm(nm, km)

        text = (
            f"{place_a} and {place_b} are both located at approximately {lon_display} longitude. "
            f"{place_a} is at latitude {lat_a_display}, and {place_b} is at latitude {lat_b_display}. "
            f"Along a meridian, 1 minute of latitude is exactly 1 nautical mile, and 1 nautical mile "
            f"= 1.852 km exactly. What is the distance between {place_a} and {place_b} along the "
            f"meridian, in kilometres, to the nearest kilometre?"
        )
        answer = km
        answer_text = str(km)
        working = [
            f"{'Same' if same_hemi else 'Opposite'} hemisphere rule: "
            f"Δ = {delta_arcmin} arcminutes of latitude = {nm} nm",
            f"{nm} nm × 1.852 = {float(raw_km):.3f} km",
            f"To the nearest kilometre: {km} km",
        ]

        # Hard-tier-only misconceptions.
        forgot_unit_val = nm
        statute_raw = Fraction(nm) * STATUTE_MILE_PER_KM
        check(statute_raw >= 0, "nautical-mile distance must not be negative")
        statute_val = int(round_half_up(statute_raw, 0))

        wrong_hemi_km_raw = Fraction(wrong_hemi_delta) * NM_PER_KM
        check(wrong_hemi_km_raw >= 0, "nautical-mile distance must not be negative")
        wrong_hemi_km = int(round_half_up(wrong_hemi_km_raw, 0))
        forgot_conv_km_raw = Fraction(forgot_conv_val) * NM_PER_KM
        check(forgot_conv_km_raw >= 0, "nautical-mile distance must not be negative")
        forgot_conv_km = int(round_half_up(forgot_conv_km_raw, 0))
        halved_km_raw = Fraction(halved_val) * NM_PER_KM
        check(halved_km_raw >= 0, "nautical-mile distance must not be negative")
        halved_km = int(round_half_up(halved_km_raw, 0))

        check_distinct(answer, forgot_unit_val, statute_val, wrong_hemi_km, forgot_conv_km, halved_km)
        candidates = [
            str(forgot_unit_val),
            str(statute_val),
            str(wrong_hemi_km),
            str(forgot_conv_km),
            str(halved_km),
        ]

        # `misreads-minutes-as-hundredths` is only a genuinely different
        # value when the two places' arcminutes differ. For a SAME-hemisphere
        # pair whose arcminutes happen to be equal (Hobart/Cairns,
        # New York/Bogotá, Rome/Tripoli all carry equal minutes on both
        # sides -- 54'/54', 42'/42', 54'/54' respectively; a fact about this
        # table, not a formula bug), the misread's minute term cancels in the
        # subtraction exactly the same way the correct one does (deg_a -
        # deg_b, whatever the minutes are divided by), so the "distractor" is
        # provably identical to the answer, not merely coincidentally so this
        # seed. Skip it there rather than assert an identity that is always
        # false.
        if not (same_hemi and min_a == min_b):
            dec_a_wrong = deg_a + Fraction(min_a, 100)
            dec_b_wrong = deg_b + Fraction(min_b, 100)
            delta_wrong = abs(dec_a_wrong - dec_b_wrong) if same_hemi else dec_a_wrong + dec_b_wrong
            nm_wrong = delta_wrong * 60
            misread_km_raw = nm_wrong * NM_PER_KM
            check(misread_km_raw >= 0, "nautical-mile distance must not be negative")
            misread_val = int(round_half_up(misread_km_raw, 0))
            check_distinct(answer, misread_val)
            candidates.append(str(misread_val))

        distractors = pick_distractors(answer_text, candidates)

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Skill 2: senior-find-time-differences-between-two-places-on-earth (ACMEM165)
# ============================================================================

HOURS_PER_DEGREE = Fraction(1, 15)  # idealised solar-time rule: 15 deg = 1 h

# Longitude, 1 d.p., E positive / W negative.
LONGITUDES: dict[str, Fraction] = {
    "Sydney": Fraction(1512, 10),
    "Auckland": Fraction(1748, 10),
    "Perth": Fraction(1159, 10),
    "Tokyo": Fraction(1397, 10),
    "Cape Town": Fraction(184, 10),
    "Moscow": Fraction(376, 10),
    "London": Fraction(-1, 10),
    "Cairo": Fraction(312, 10),
    "Lima": Fraction(-770, 10),
    "Accra": Fraction(-2, 10),
    "Beijing": Fraction(1164, 10),
    "Buenos Aires": Fraction(-584, 10),
    # Widened from 12 to 36 places so SAME_SIDE_PAIRS/OPPOSITE_SIDE_PAIRS below can grow past the
    # shared _MIN_DISTINCT=10 floor without pinning.
    "Melbourne": Fraction(1450, 10),
    "Jakarta": Fraction(1068, 10),
    "Manila": Fraction(1210, 10),
    "Singapore": Fraction(1038, 10),
    "Paris": Fraction(23, 10),
    "Nairobi": Fraction(368, 10),
    "Rome": Fraction(125, 10),
    "Berlin": Fraction(134, 10),
    "Athens": Fraction(237, 10),
    "New York": Fraction(-740, 10),
    "Los Angeles": Fraction(-1182, 10),
    "Toronto": Fraction(-794, 10),
    "Vancouver": Fraction(-1231, 10),
    "Bogotá": Fraction(-741, 10),
    "Santiago": Fraction(-707, 10),
    "Mexico City": Fraction(-991, 10),
    "Panama City": Fraction(-795, 10),
    "Quito": Fraction(-785, 10),
    "Dubai": Fraction(553, 10),
    "Mumbai": Fraction(729, 10),
    "Johannesburg": Fraction(280, 10),
    "Istanbul": Fraction(290, 10),
    "Warsaw": Fraction(210, 10),
    "Stockholm": Fraction(181, 10),
}

SAME_SIDE_PAIRS = [
    ("Sydney", "Auckland"),
    ("Perth", "Tokyo"),
    ("Cape Town", "Moscow"),
    # Widened from 3 to 15 pairs, same reason as LONGITUDES above.
    ("Melbourne", "Auckland"),
    ("Jakarta", "Manila"),
    ("Singapore", "Jakarta"),
    ("Paris", "Moscow"),
    ("Rome", "Nairobi"),
    ("Berlin", "Athens"),
    ("New York", "Los Angeles"),
    ("Toronto", "Vancouver"),
    ("Bogotá", "Santiago"),
    ("Mexico City", "Panama City"),
    ("Quito", "Lima"),
    ("Dubai", "Mumbai"),
]
OPPOSITE_SIDE_PAIRS = [
    ("London", "Cairo"),
    ("Cairo", "Lima"),
    ("Tokyo", "Accra"),
    ("Beijing", "Buenos Aires"),
    ("Auckland", "London"),
    # Widened from 5 to 12 pairs, same reason as LONGITUDES above.
    ("Nairobi", "New York"),
    ("Johannesburg", "Santiago"),
    ("Dubai", "Toronto"),
    ("Mumbai", "Mexico City"),
    ("Istanbul", "Bogotá"),
    ("Warsaw", "Quito"),
    ("Stockholm", "Panama City"),
]
ALL_PLACES = list(LONGITUDES)

# Delta-t candidates for the inverse (hard) tier. A first draft allowed
# "a clean multiple of 0.25 hours so that Delta t x 15 is a clean number of degrees". An ODD
# multiple of 0.25 (0.25, 0.75, 1.25...) x 15 ends in.25 or.75 -- TWO decimal places -- which, added
# to a 1-d.p. longitude, forces a 2-d.p. result and breaks the "1 decimal place, no
# rounding dispute" requirement. Multiples of 0.5 are still multiples of 0.25 (0.5 = 2 x 0.25) and
# are exactly the subset for which Delta t x 15 is always an integer or an exact half-degree (0 or 1
# d.p.), so the sum with a 1-d.p. longitude always terminates at 1 d.p. exactly.
DELTA_T_CHOICES = [Fraction(n, 2) for n in range(1, 9)]  # 0.5 .. 4.0 h


def _same_side(lon_a: Fraction, lon_b: Fraction) -> bool:
    return (lon_a >= 0) == (lon_b >= 0)


def _delta_lon(lon_a: Fraction, lon_b: Fraction) -> Fraction:
    return abs(lon_a - lon_b) if _same_side(lon_a, lon_b) else abs(lon_a) + abs(lon_b)


def _verify_delta_lon(lon_a: Fraction, lon_b: Fraction, delta: Fraction) -> None:
    """Independent check: convert both longitudes to the signed -180..180
    scale (they already are) and compute the minimal circular separation via
    min(|a-b|, 360-|a-b|) -- a route independent of the same/opposite-side
    branching construction uses -- and assert the two totals agree."""
    diff = abs(lon_a - lon_b)
    minimal = min(diff, Fraction(360) - diff)
    check(minimal == delta, f"circular-separation route {minimal} != side-rule route {delta}")


def _lon_display(value: Fraction) -> str:
    hemi = "E" if value >= 0 else "W"
    return f"{_fixed_str(abs(value), 1)}°{hemi}"


def _wrap_valid(value: Fraction) -> bool:
    return Fraction(-180) < value <= Fraction(180)


def _build_inverse(rng):
    place_a = rng.choice(ALL_PLACES)
    lon_a = LONGITUDES[place_a]
    dt = rng.choice(DELTA_T_CHOICES)
    mag = dt * 15  # exact degrees, at most 1 d.p.

    east_lon = lon_a + mag
    west_lon = lon_a - mag
    east_ok = _wrap_valid(east_lon)
    west_ok = _wrap_valid(west_lon)
    check(east_ok or west_ok, f"neither direction keeps {place_a} +/- {mag} in (-180, 180]")
    if east_ok and west_ok:
        direction = rng.choice(["east", "west"])
    else:
        direction = "east" if east_ok else "west"
    lon_b = east_lon if direction == "east" else west_lon
    return place_a, lon_a, dt, mag, direction, lon_b


def _verify_inverse(lon_a: Fraction, lon_b: Fraction, dt: Fraction) -> None:
    """Forward-simulation check: recompute Delta t from the constructed
    lonB the same way the forward (easy/medium) tiers do, and confirm it
    recovers the stated Delta t exactly."""
    delta = _delta_lon(lon_a, lon_b)
    _verify_delta_lon(lon_a, lon_b, delta)
    check(delta == dt * 15, f"forward Δlon {delta} != stated Δt×15 {dt * 15}")


@generator(
    "earth_longitude_time_difference",
    level=12,
    presets={
        "easy": dict(mode="same_side"),
        "medium": dict(mode="opposite_side"),
        "hard": dict(mode="inverse"),
    },
)
def earth_longitude_time_difference(rng, mode="same_side"):
    """Time difference between two places on Earth, using the idealised
    solar-time rule (15 degrees of longitude = 1 hour), never a real-world
    time zone."""
    if mode not in ("same_side", "opposite_side", "inverse"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "inverse":
        place_a, lon_a, dt, mag, direction, lon_b = _build_inverse(rng)
        _verify_inverse(lon_a, lon_b, dt)

        lon_b_wrong_dir = (lon_a - mag) if direction == "east" else (lon_a + mag)
        mag2 = dt * 30
        lon_b_off_double = (lon_a + mag2) if direction == "east" else (lon_a - mag2)

        answer_val = lon_b
        answer_text = _fixed_str(answer_val, 1)
        hours_phrase = _hours_phrase(dt)
        dt_num = _trim_str(dt)
        mag_str = _trim_str(mag)

        text = (
            f"{place_a} is located at longitude {_lon_display(lon_a)}. Using the rule that 15° "
            f"of longitude corresponds to exactly 1 hour of solar time, place B is {hours_phrase} "
            f"different in solar time from {place_a}, and place B is further {direction} of "
            f"{place_a}. A research vessel is at place B. What is the longitude of place B, to 1 "
            f"decimal place? Give your answer as a signed number of degrees: positive for East, "
            f"negative for West."
        )
        working = [
            f"Δlon = Δt × 15 = {dt_num} × 15 = {mag_str}°",
            f"Place B is further {direction} of {place_a}, so lonB = {_lon_display(lon_a)} "
            f"{'+' if direction == 'east' else '-'} {mag_str}° = {answer_text}°",
        ]

        check_distinct(answer_val, lon_b_wrong_dir, lon_b_off_double)
        candidates = [_fixed_str(lon_b_wrong_dir, 1), _fixed_str(lon_b_off_double, 1)]
        distractors = pick_distractors(
            answer_text,
            candidates,
            tolerance=stated_precision_tolerance(1),
        )
        answer = numeric_answer(answer_val)
    else:
        pairs = SAME_SIDE_PAIRS if mode == "same_side" else OPPOSITE_SIDE_PAIRS
        place_a, place_b = rng.choice(pairs)
        lon_a, lon_b = LONGITUDES[place_a], LONGITUDES[place_b]
        delta = _delta_lon(lon_a, lon_b)
        _verify_delta_lon(lon_a, lon_b, delta)

        dt_raw = delta * HOURS_PER_DEGREE
        check(dt_raw >= 0, "time difference in hours must not be negative")
        dt_rounded = round_half_up(dt_raw, 2)
        answer_text = _fixed_str(dt_rounded, 2)
        answer = numeric_answer(dt_rounded)

        same = _same_side(lon_a, lon_b)
        text = (
            f"{place_a} is located at longitude {_lon_display(lon_a)}, and {place_b} is located at "
            f"longitude {_lon_display(lon_b)}. Using the rule that 15° of longitude corresponds "
            f"to exactly 1 hour of solar time, what is the time difference between {place_a} and "
            f"{place_b}, in hours, to 2 decimal places?"
        )
        working = [
            f"Δlon = {_fixed_str(delta, 1)}° ({'same side' if same else 'opposite sides'} rule)",
            f"Δt = {_fixed_str(delta, 1)} ÷ 15 = {answer_text} h",
        ]

        # `wrong-side-rule`: add magnitudes when same-side (should subtract),
        # or subtract magnitudes when opposite-side (should add). Built from
        # MAGNITUDES, not signed longitudes -- for two opposite-signed
        # longitudes, |a - b| == |a| + |b| identically, so a signed
        # subtraction is mathematically the SAME value as the correct
        # opposite-side rule, not a distinct misconception.
        mag_a, mag_b = abs(lon_a), abs(lon_b)
        alt_delta = (mag_a + mag_b) if same else abs(mag_a - mag_b)
        wrong_side_raw = alt_delta * HOURS_PER_DEGREE
        check(wrong_side_raw >= 0, "time difference in hours must not be negative")
        wrong_side_val = round_half_up(wrong_side_raw, 2)
        check(delta >= 0, "degrees of longitude difference must not be negative")
        forgot_divide_val = round_half_up(delta, 2)
        inverted_raw = Fraction(15) / delta
        check(inverted_raw >= 0, "time difference in hours must not be negative")
        inverted_val = round_half_up(inverted_raw, 2)

        check_distinct(dt_rounded, wrong_side_val, forgot_divide_val, inverted_val)
        candidates = [
            _fixed_str(wrong_side_val, 2),
            _fixed_str(forgot_divide_val, 2),
            _fixed_str(inverted_val, 2),
        ]
        distractors = pick_distractors(
            answer_text,
            candidates,
            tolerance=stated_precision_tolerance(2),
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
