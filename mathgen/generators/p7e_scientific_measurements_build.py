"""Builds `f10-scientific-notation-measurements` (`scientific_notation_measurements`, AC9M9M02): express
measurements in scientific notation, `answer: number`.

One generator, ``scientific_notation_measurements`` (AC9M9M02), applied
arithmetic on quantities that are ALREADY given in scientific notation --
distinct from the skill's own prerequisite ``scientific-notation`` (which
converts a single decimal INTO standard form) and from
``f10-powers-ten-multiply-divide`` (a bare power-of-ten shift, no scientific
notation, no applied context).

Three variants/tiers, each with its own curated draw table so every seed's
arithmetic lands on an exact whole number (``grade_number``'s literal-extraction
regex has no exponent syntax, so a typed scientific-notation answer is not
gradable without a bespoke parser):

- ``multiply_rate_time`` (easy): distance = speed x time. Speed's
  coefficient carries one decimal digit; time's coefficient is a whole
  number 1-9; the combined exponent is constrained to >= 2 so the one
  stated decimal digit is always absorbed into a whole-number metre count.
- ``divide_distance_speed`` (medium): time = distance / speed. Coefficients
  drawn from a curated (speed_c, quotient, distance_c) table so the
  coefficient quotient is always an exact integer; the exponent
  subtraction is always >= 0 so the result is a whole number of hours.
- ``compare_magnitude`` (hard): "how many times as massive as the lighter
  object is the heavier object", both given in scientific notation. The
  object with the SMALLER coefficient is deliberately the LARGER value
  (its exponent gap more than compensates), so the learner must judge
  magnitude before dividing -- the part of the skill's descriptor
  ("interpret the magnitude") that can be asked as a computable
  ratio. Coefficients are drawn from a curated table where the ratio is a
  terminating decimal that the exponent gap always renders as a whole
  number.

Every tier's construction and its independent verification route are kept
deliberately different: construction combines coefficients and exponents
separately (the "scientific notation law"); the check expands each
quantity to a plain ``Fraction`` first and then multiplies/divides the
absolute values directly.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_distinct, generator, pick_distractors


# --- multiply_rate_time (easy) -------------------------------------------------

# Coefficients with exactly one non-zero decimal digit, in [1.1, 9.9] --
# stored as tenths (11..99, multiples of 10 excluded) so `Fraction(m, 10)`
# is exact and `sc + tc`/`sc * tc` always have at most one stated decimal
# place (a whole-number `tc` cannot add extra decimal places).
_EASY_SC_TENTHS = [m for m in range(11, 100) if m % 10 != 0]
_EASY_TC = range(1, 10)
# Two (sc, tc) pairs make "adds the coefficients instead of multiplying"
# land on the SAME value as the real answer ((sc-1)(tc-1)=1): sc=1.5,tc=3
# and sc=1.2,tc=6. Excluded from the draw space so `check_distinct` below
# can run unconditionally on every seed rather than needing a try/except.
_EASY_COEF_PAIRS = [(m, t) for m in _EASY_SC_TENTHS for t in _EASY_TC if (m, t) not in {(15, 3), (12, 6)}]
# Combined exponent (speed_e + time_e) in [2, 5]: >=2 leaves headroom to
# absorb the one stated decimal place at exponent (combined - 1) as well
# as at (combined); <=5 keeps the metre answer to a readable few digits.
_EASY_EXPONENT_PAIRS = [(se, te) for se in range(-3, 6) for te in range(-2, 7) if 2 <= se + te <= 5]


def _verify_multiply(sc: Fraction, se: int, tc: int, te: int, expected: Fraction) -> None:
    """Independent route: expand speed and time to absolute values first,
    then multiply -- construction instead multiplies coefficients and
    scales by the COMBINED exponent."""
    speed_val = sc * Fraction(10) ** se
    time_val = Fraction(tc) * Fraction(10) ** te
    check(speed_val * time_val == expected, "multiply_rate_time: speed x time mismatch")


def _build_multiply_rate_time(rng) -> dict:
    se, te = rng.choice(_EASY_EXPONENT_PAIRS)
    m, tc = rng.choice(_EASY_COEF_PAIRS)
    sc = Fraction(m, 10)
    ee = se + te

    distance = sc * tc * Fraction(10) ** ee
    _verify_multiply(sc, se, tc, te, distance)
    check(distance.denominator == 1, "multiply_rate_time: distance is not a whole number")
    answer = int(distance)
    answer_text = str(answer)

    # Misconceptions: adds coefficients instead of multiplying
    # (wrong-power-combination); shifts the decimal point one place too
    # big/small (off-by-one-order-of-magnitude).
    adds_coeffs = (sc + tc) * Fraction(10) ** ee
    shifted_big = distance * 10
    shifted_small = sc * tc * Fraction(10) ** (ee - 1)
    check(adds_coeffs.denominator == 1, "multiply_rate_time: adds_coeffs distractor not whole")
    check(shifted_small.denominator == 1, "multiply_rate_time: shifted_small distractor not whole")
    check_distinct(
        distance, adds_coeffs, shifted_big, shifted_small, msg="multiply_rate_time: a distractor equals the answer"
    )

    sc_display = f"{m / 10:g}"
    text = (
        f"An object moves at a constant speed of {sc_display} × 10^{se} m/s. "
        f"Find the distance it travels in {tc} × 10^{te} s. "
        f"Give your answer in metres, as a whole number."
    )
    candidates = [str(int(adds_coeffs)), str(int(shifted_big)), str(int(shifted_small))]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates),
    )


# --- divide_distance_speed (medium) --------------------------------------------

# (speed_c, quotient, distance_c) with distance_c = speed_c * quotient <= 9
# (both coefficients stay valid scientific notation, 1 <= c < 10),
# speed_c >= 2 (so "use the distance's own coefficient" distractor never
# equals `quotient`, which would happen at speed_c == 1) and
# quotient != speed_c (so "use the speed's own coefficient" distractor
# never equals `quotient` either).
_MEDIUM_TABLE = [(2, 3, 6), (2, 4, 8), (3, 2, 6), (4, 2, 8)]
_MEDIUM_SPEED_E = (1, 2, 3, 4)  # >=1 so "adds instead of subtracts" never collides (see below)
_MEDIUM_DIFF = (0, 1, 2, 3, 4, 5)


def _verify_divide(dc: int, distance_e: int, sc: int, speed_e: int, expected: Fraction) -> None:
    """Independent route: expand distance and speed to absolute values
    first, then divide -- construction instead divides the coefficients
    and scales by the SUBTRACTED exponent."""
    distance_val = Fraction(dc) * Fraction(10) ** distance_e
    speed_val = Fraction(sc) * Fraction(10) ** speed_e
    check(distance_val / speed_val == expected, "divide_distance_speed: distance / speed mismatch")


def _build_divide_distance_speed(rng) -> dict:
    sc, k, dc = rng.choice(_MEDIUM_TABLE)
    speed_e = rng.choice(_MEDIUM_SPEED_E)
    diff = rng.choice(_MEDIUM_DIFF)
    distance_e = speed_e + diff

    time = Fraction(k) * Fraction(10) ** diff
    _verify_divide(dc, distance_e, sc, speed_e, time)
    check(time.denominator == 1, "divide_distance_speed: time is not a whole number")
    answer = int(time)
    answer_text = str(answer)

    # Misconceptions: adds the exponents instead of subtracting them
    # (wrong-power-combination) -- never equal to the answer since
    # speed_e >= 1 makes (distance_e + speed_e) != (distance_e - speed_e);
    # uses one quantity's own coefficient directly instead of dividing the
    # two coefficients (wrong-power-combination, the other half) -- never
    # equal to the answer given the table's speed_c >= 2 / quotient !=
    # speed_c constraints above.
    add_exponents = Fraction(k) * Fraction(10) ** (distance_e + speed_e)
    use_distance_coefficient = Fraction(dc) * Fraction(10) ** diff
    use_speed_coefficient = Fraction(sc) * Fraction(10) ** diff
    check_distinct(
        time,
        add_exponents,
        use_distance_coefficient,
        use_speed_coefficient,
        msg="divide_distance_speed: a distractor equals the answer",
    )

    text = (
        f"A journey covers a distance of {dc} × 10^{distance_e} km. "
        f"It is travelled at a constant speed of {sc} × 10^{speed_e} km/h. "
        f"How many hours does the journey take? Give your answer as a whole number."
    )
    candidates = [str(int(add_exponents)), str(int(use_distance_coefficient)), str(int(use_speed_coefficient))]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates),
    )


# --- compare_magnitude (hard) ---------------------------------------------------

# (dc, sc, m) with sc = m * dc, m in {2, 4, 5, 8} so dc/sc is a terminating
# decimal. dc's quantity (a SMALLER coefficient) is the LARGER value once
# its bigger exponent is applied -- see _HARD_MIN_DIFF.
_HARD_TABLE = [
    (1, 2, 2),
    (2, 4, 2),
    (3, 6, 2),
    (4, 8, 2),
    (1, 4, 4),
    (2, 8, 4),
    (1, 5, 5),
    (1, 8, 8),
]
# Minimum exponent gap so `m` exactly divides `10 ** diff` (m=2,5 -> 10^1;
# m=4 -> 10^2; m=8 -> 10^3).
_HARD_MIN_DIFF = {2: 1, 4: 2, 5: 1, 8: 3}
_HARD_SMALL_E = (1, 2, 3, 4)
_HARD_DIFF_OFFSETS = (0, 1, 2)


def _verify_compare(dc: int, big_e: int, sc: int, small_e: int, expected: Fraction) -> None:
    """Independent route: expand both quantities to absolute values first,
    then divide -- construction instead divides the coefficients (as
    ``1/m``) and scales by the exponent gap."""
    big_val = Fraction(dc) * Fraction(10) ** big_e
    small_val = Fraction(sc) * Fraction(10) ** small_e
    check(big_val / small_val == expected, "compare_magnitude: ratio mismatch")


def _build_compare_magnitude(rng) -> dict:
    dc, sc, m = rng.choice(_HARD_TABLE)
    diff = _HARD_MIN_DIFF[m] + rng.choice(_HARD_DIFF_OFFSETS)
    small_e = rng.choice(_HARD_SMALL_E)
    big_e = small_e + diff

    ratio = Fraction(10) ** diff / m
    _verify_compare(dc, big_e, sc, small_e, ratio)
    check(ratio.denominator == 1, "compare_magnitude: ratio is not a whole number")
    answer = int(ratio)
    answer_text = str(answer)

    # Misconceptions: inverts-comparison-ratio -- divides the larger
    # coefficient by the smaller one directly (m * 10^diff), the reverse of
    # which coefficient goes on top; order-of-magnitude-only -- ignores the
    # coefficients entirely (10^diff); off-by-one-order-of-magnitude --
    # the correct ratio, misplaced one power of ten too big. None of the
    # three can ever equal `ratio` for m in {2, 4, 5, 8} (m != 1 and
    # m**2 != 10), so `check_distinct` runs unconditionally.
    invert_ratio = Fraction(m) * Fraction(10) ** diff
    order_of_magnitude_only = Fraction(10) ** diff
    off_by_one = ratio * 10
    check_distinct(
        ratio,
        invert_ratio,
        order_of_magnitude_only,
        off_by_one,
        msg="compare_magnitude: a distractor equals the answer",
    )

    big_desc = f"a mass of {dc} × 10^{big_e} kg"
    small_desc = f"a mass of {sc} × 10^{small_e} kg"
    if rng.choice((True, False)):
        first_label, first_desc = "Object P", big_desc
        second_label, second_desc = "Object Q", small_desc
    else:
        first_label, first_desc = "Object P", small_desc
        second_label, second_desc = "Object Q", big_desc
    text = (
        f"{first_label} has {first_desc}. {second_label} has {second_desc}. "
        f"How many times as massive as the lighter object is the heavier object? "
        f"Give your answer as a whole number."
    )
    # Distractors on both sides of the key, so the smallest option is not always right:
    # a power of ten too small (one order below the exponent gap, or the ratio / 10 when that
    # is a whole number) and the three larger misconceptions above.
    below = [10 ** (diff - 1)]
    if (ratio / 10).denominator == 1:
        below.append(int(ratio / 10))
    above = [int(invert_ratio), int(order_of_magnitude_only), int(off_by_one)]
    n_below = rng.randint(1, len(below))
    chosen = rng.sample(below, n_below) + rng.sample(above, 3 - n_below)
    check(all(c != answer for c in chosen), "compare_magnitude: a distractor equals the answer")
    candidates = [str(c) for c in chosen]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "scientific_notation_measurements",
    level=8,
    presets={
        "easy": dict(variant="multiply_rate_time"),
        "medium": dict(variant="divide_distance_speed"),
        "hard": dict(variant="compare_magnitude"),
    },
)
def scientific_notation_measurements(rng, variant="multiply_rate_time"):
    """Applied measurement/time-scale/interval arithmetic on quantities
    already expressed in scientific notation (AC9M9M02)."""
    if variant == "multiply_rate_time":
        return _build_multiply_rate_time(rng)
    if variant == "divide_distance_speed":
        return _build_divide_distance_speed(rng)
    if variant == "compare_magnitude":
        return _build_compare_magnitude(rng)
    raise ValueError(f"unknown variant: {variant!r}")
