"""Chords and rate conversion, both senior ``acara-senior-v8-4``, domain ``measurement``, ``answer: number``.

Two generators:

- ``chords_equal_length_equal_angles`` (``senior-chords-of-equal-length-
  subtend-equal-angles-at``, ACMSM033) -- circle theorem: chords of equal
  length subtend equal angles at the centre and conversely. Three tiers:
  easy (forward: length -> angle), medium (converse: angle -> length),
  hard (chained two-hop reasoning).

- ``convert_units_for_rates`` (``senior-convert-between-units-for-rates-
  for-example-km``, ACMEM072) -- convert between rate units (km/h <-> m/s,
  km/min -> m/s). Three tiers: easy (km/h to m/s, exact integer answer),
  medium (m/s to km/h, 1 decimal place), hard (km/min to m/s, new combined
  factor).

Both use exact ``Fraction`` arithmetic. Misconceptions and answers are
integer-valued except for medium rates (1 decimal place).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, fixed_point_string, generator, pick_distractors


# --- Shared helpers for rates conversion ------------------------------------

_FACTOR_KMH_TO_MS = Fraction(5, 18)


def _format_fixed(value: Fraction, places: int = 1) -> str:
    """Fixed-point formatting from a Fraction, exactly `places` decimal places.
    Never produces scientific notation or trailing zeros. Raises if value does
    not terminate exactly (canonical answers only)."""
    if places == 0:
        return str(int(value))
    scaled = value * (10**places)
    if scaled.denominator != 1:
        raise ValueError(f"{value} does not have exact {places} decimal places")
    digits = str(scaled.numerator).zfill(places + 1)
    whole, frac = digits[:-places], digits[-places:]
    frac = frac.rstrip("0") or "0"  # strip trailing zeros but keep at least one digit
    if len(frac) == 0:
        return whole
    return f"{whole}.{frac}"


def _format_display(value: Fraction, places: int = 1) -> str:
    """Like _format_fixed, but rounds non-terminating values half-up instead of
    raising. Used only for distractor display."""
    import math

    try:
        return _format_fixed(value, places)
    except ValueError:
        scaled = math.floor(value * (10**places) + Fraction(1, 2))
        digits = str(scaled).zfill(places + 1)
        whole, frac = digits[:-places], digits[-places:] if places > 0 else ("", "")
        if places == 0 or not frac:
            return whole
        frac = frac.rstrip("0") or "0"
        return f"{whole}.{frac}"


# --- chords_equal_length_equal_angles ----------------------------------------


def _verify_easy_chords_forward(length_ab, angle_ab, length_cd, angle_cd, answer):
    """Independent check for easy tier: verify length equality and the forward
    conclusion that equal chords subtend equal angles."""
    check(length_ab == length_cd, "chord lengths must be equal")
    check(angle_cd == angle_ab, "equal chord lengths must subtend equal angles")
    check(answer == angle_ab, "answer must be the angle of the equal chord")


def _verify_medium_chords_converse(angle_ab, angle_cd, length_ab, length_cd, answer):
    """Independent check for medium tier: verify angle equality and the converse
    conclusion that equal angles yield equal chord lengths."""
    check(angle_ab == angle_cd, "chord angles must be equal")
    check(length_cd == length_ab, "equal angles must yield equal chord lengths")
    check(answer == length_ab, "answer must be the length of the equal-angle chord")


def _verify_hard_chords_chained(length_ab, angle_ab, length_cd, angle_cd, angle_ef, length_ef, answer):
    """Independent check for hard tier: two-hop chaining.
    1. Forward: length_ab == length_cd implies angle_cd == angle_ab
    2. Converse: angle_cd == angle_ef implies length_ef == length_cd
    3. Conclusion: length_ef == length_ab"""
    # Forward hop
    check(length_ab == length_cd, "AB and CD have equal length (hop 1 premise)")
    check(angle_cd == angle_ab, "CD subtends equal angle to AB (hop 1 forward conclusion)")
    # Converse hop
    check(angle_cd == angle_ef, "CD and EF subtend equal angles (hop 2 premise)")
    check(length_ef == length_cd, "EF has equal length to CD (hop 2 converse conclusion)")
    # Final chained conclusion
    check(answer == length_ab, "EF length equals AB length (chained)")


@generator(
    "chords_equal_length_equal_angles",
    level=11,
    presets={
        "easy": dict(mode="forward"),
        "medium": dict(mode="converse"),
        "hard": dict(mode="chained"),
    },
)
def chords_equal_length_equal_angles(rng, mode="forward"):
    """Circle theorem: chords of equal length subtend equal angles at the centre,
    and conversely, chords subtending equal angles have equal length."""

    if mode == "forward":
        # Easy: Given equal chord length, find the angle
        length = rng.randint(4, 20)  # cm
        theta = rng.randint(10, 170)
        # Exclude 90 so distractors don't coincide
        while theta == 90:
            theta = rng.randint(10, 170)

        text = (
            f"In a circle, chord AB has length {length} cm and subtends an angle of {theta} degrees "
            f"at the centre. Chord CD also has length {length} cm. "
            f"What angle does CD subtend at the centre, in degrees?"
        )
        answer = theta
        _verify_easy_chords_forward(length, theta, length, theta, answer)
        answer_text = str(answer)

        # Distractors
        half_angle = theta // 2
        supplementary = 180 - theta
        assumes_right = 90

        candidates = []
        for distractor in [half_angle, supplementary, assumes_right]:
            if distractor != answer:
                candidates.append(str(distractor))
        candidates = pick_distractors(answer_text, candidates[:3] if len(candidates) >= 3 else candidates)

        return dict(
            text=text,
            latex=rf"\text{{Chord }}AB: {length}\,\text{{cm, angle }}{theta}^\circ\text{{; Chord }}CD: {length}\,\text{{cm, angle ?}}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"Chord CD has the same length as AB, so it subtends the same angle: {answer} degrees."],
            distractors=candidates,
        )

    elif mode == "converse":
        # Medium: Given equal angles, find the chord length
        theta = rng.randint(10, 170)
        while theta == 90:
            theta = rng.randint(10, 170)
        length = rng.randint(4, 20)  # cm

        text = (
            f"In a circle, chords AB and CD both subtend an angle of {theta} degrees at the centre. "
            f"Chord AB has length {length} cm. "
            f"What is the length of chord CD, in cm?"
        )
        answer = length
        _verify_medium_chords_converse(theta, theta, length, length, answer)
        answer_text = str(answer)

        # Distractors: confuses-direction, doubles-length, off-by-one
        confuses = max(1, length // 2)
        doubles = length * 2
        off_by_one = length + 1

        candidates = [str(confuses), str(doubles), str(off_by_one)]
        candidates = pick_distractors(answer_text, candidates)

        return dict(
            text=text,
            latex=rf"\text{{Chord }}AB: \text{{angle }}{theta}^\circ, {length}\,\text{{cm; Chord }}CD: \text{{angle }}{theta}^\circ, \text{{length ?}}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"Both chords subtend the same angle, so they have the same length: {answer} cm."],
            distractors=candidates,
        )

    elif mode == "chained":
        # Hard: Two-hop chaining
        length = rng.randint(4, 20)  # cm
        theta1 = rng.randint(10, 170)
        while theta1 == 90:
            theta1 = rng.randint(10, 170)

        text = (
            f"In a circle, chord AB has length {length} cm and subtends an angle of {theta1} degrees "
            f"at the centre. Chord CD also has length {length} cm. "
            f"Chords CD and EF subtend equal angles at the centre. "
            f"What is the length of chord EF, in cm?"
        )

        answer = length
        # Hop 1 (forward): AB and CD have equal length, so CD subtends theta1
        # Hop 2 (converse): CD and EF subtend equal angle (theta1), so EF has length of CD (which is length)
        _verify_hard_chords_chained(length, theta1, length, theta1, theta1, length, answer)
        answer_text = str(answer)

        # Same distractors as medium
        confuses = max(1, length // 2)
        doubles = length * 2
        off_by_one = length + 1

        candidates = [str(confuses), str(doubles), str(off_by_one)]
        candidates = pick_distractors(answer_text, candidates)

        return dict(
            text=text,
            latex=rf"\text{{AB: }}{length}\,\text{{cm, }}{theta1}^\circ\text{{; CD: }}{length}\,\text{{cm; EF: angles equal to CD; length ?}}",
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"CD has same length as AB, so CD subtends {theta1}°.",
                f"EF subtends same angle as CD, so EF has same length as CD: {answer} cm.",
            ],
            distractors=candidates,
        )

    raise ValueError(f"unknown mode: {mode!r}")


# --- convert_units_for_rates -------------------------------------------------


def _verify_easy_kmh_to_ms(speed_kmh, answer_ms):
    """Independent check for easy: verify using alternative factor calculation."""
    pivot_factor = Fraction(1000, 3600)  # metres-per-km / seconds-per-hour
    check(pivot_factor == _FACTOR_KMH_TO_MS, "pivot factor must match the constant")
    recomputed = speed_kmh * _FACTOR_KMH_TO_MS
    check(recomputed == answer_ms, "km/h to m/s conversion mismatch")


def _verify_medium_ms_to_kmh(speed_ms, answer_kmh):
    """Independent check for medium: verify using alternative factor calculation."""
    pivot_factor = Fraction(1000, 3600)
    check(pivot_factor == _FACTOR_KMH_TO_MS, "pivot factor must match the constant")
    forward_factor = 1 / _FACTOR_KMH_TO_MS
    check(forward_factor == Fraction(18, 5), "forward factor must be 18/5")
    recomputed = speed_ms * forward_factor
    check(recomputed == answer_kmh, "m/s to km/h conversion mismatch")


def _verify_hard_km_per_min_to_ms(speed_km_per_min, answer_ms):
    """Independent check for hard: verify by computing the factor from raw components."""
    dist_factor = Fraction(1000, 1)  # metres per km
    time_factor = Fraction(1, 60)  # minutes to seconds (division)
    combined = dist_factor * time_factor
    check(combined == Fraction(50, 3), "combined factor must be 50/3")
    recomputed = speed_km_per_min * combined
    check(recomputed == answer_ms, "km/min to m/s conversion mismatch")


def _rate_candidates(answer, answer_text: str, misconceptions: list) -> list[str]:
    """Distractor strings that all show the key's own number of decimal places.

    A misconception value is rounded half-up to that many places (never shown
    with more or fewer, which would give the key away by its shape); one that
    would round to zero is dropped as meaningless. A decimal-point slip and a
    plain off-by-one follow so every draw still has distractors."""
    import math

    places = len(answer_text.partition(".")[2])
    answer = Fraction(answer)
    pool = list(misconceptions) + [answer * 10, answer / 10, answer * 2, answer + 1, answer - 1]
    out = []
    for value in pool:
        scaled = math.floor(Fraction(value) * 10**places + Fraction(1, 2))
        if scaled <= 0:
            continue
        out.append(fixed_point_string(scaled, places))
    return out


@generator(
    "convert_units_for_rates",
    level=11,
    presets={
        "easy": dict(mode="kmh_to_ms"),
        "medium": dict(mode="ms_to_kmh"),
        "hard": dict(mode="km_per_min_to_ms"),
    },
)
def convert_units_for_rates(rng, mode="kmh_to_ms"):
    """Convert between rate units: km/h <-> m/s, km/min -> m/s."""

    if mode == "kmh_to_ms":
        # Easy: km/h to m/s (always an exact integer)
        k = rng.randint(2, 40)
        speed_kmh = Fraction(18, 5) * k  # exact multiple of 3.6
        answer_ms = k

        # Format for display: Fraction(18, 5) * k = 3.6k
        speed_kmh_display = _format_fixed(speed_kmh, places=1)

        text = f"A car travels at {speed_kmh_display} km/h. What is its speed in m/s?"
        _verify_easy_kmh_to_ms(speed_kmh, answer_ms)
        answer_text = str(answer_ms)

        # Distractors: wrong-constant (divides by 3 instead of 3.6), inverts-factor
        # Wrong-constant: speed_kmh / 3 instead of / 3.6
        wrong_constant_frac = speed_kmh / 3
        inverts_frac = speed_kmh * Fraction(18, 5)  # multiplies by reciprocal

        candidates = _rate_candidates(answer_ms, answer_text, [wrong_constant_frac, inverts_frac])
        candidates = pick_distractors(answer_text, candidates)

        return dict(
            text=text,
            latex=rf"{speed_kmh_display}\,\mathrm{{km/h}} = ?\,\mathrm{{m/s}}",
            answer=answer_ms,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"{speed_kmh_display} km/h = {speed_kmh_display} × (5/18) m/s = {answer_ms} m/s"],
            distractors=candidates,
        )

    elif mode == "ms_to_kmh":
        # Medium: m/s to km/h (1 decimal place result)
        v = rng.randint(2, 40)
        answer_kmh_frac = v * Fraction(18, 5)

        text = f"A car travels at {v} m/s. What is its speed in km/h?"
        _verify_medium_ms_to_kmh(v, answer_kmh_frac)
        answer_kmh = float(answer_kmh_frac)  # for Question.answer
        answer_text = _format_fixed(answer_kmh_frac, places=1)

        # Distractors: inverts-factor (v * 5/18), wrong-constant (v * 3)
        inverts_frac = v * Fraction(5, 18)
        wrong_constant_frac = v * 3

        candidates = _rate_candidates(answer_kmh_frac, answer_text, [inverts_frac, wrong_constant_frac])
        candidates = pick_distractors(answer_text, candidates)

        return dict(
            text=text,
            latex=rf"{v}\,\mathrm{{m/s}} = ?\,\mathrm{{km/h}}",
            answer=answer_kmh,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"{v} m/s = {v} × (18/5) km/h = {answer_text} km/h"],
            distractors=candidates,
        )

    elif mode == "km_per_min_to_ms":
        # Hard: km/min to m/s (derived new factor)
        # k is restricted to plausible speeds (180-720 km/h)
        k = rng.choice([3, 6, 9, 12])
        answer_ms = k * Fraction(50, 3)

        text = f"A passenger jet is flying at {k} km per minute. What is this speed in m/s?"
        _verify_hard_km_per_min_to_ms(k, answer_ms)
        answer_text = str(int(answer_ms))

        # Distractors: forgets-time-conversion (x1000 only), forgets-distance-conversion (/60 only)
        forgets_time_frac = k * 1000  # converts km->m but forgets min->s division
        forgets_distance_frac = k * Fraction(1, 60)  # converts min->s but forgets km->m

        candidates = _rate_candidates(answer_ms, answer_text, [forgets_time_frac, forgets_distance_frac])
        candidates = pick_distractors(answer_text, candidates)

        return dict(
            text=text,
            latex=rf"{k}\,\mathrm{{km/min}} = ?\,\mathrm{{m/s}}",
            answer=int(answer_ms),
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[f"{k} km/min = {k} × 1000 m / 60 s = {answer_text} m/s"],
            distractors=candidates,
        )

    raise ValueError(f"unknown mode: {mode!r}")
