"""Significant-figures, engineering-notation.

``significant_figures`` rounds a value to a stated number of significant
figures -- a different rule from the ``round-decimals`` skill's stated
DECIMAL PLACE, with two behaviours that must survive every seed of their
tier: leading zeros on a value below 1 are never counted as significant
(``0.004813`` to 2 s.f. is ``0.0048``), and trailing zeros before the point
must survive when the requested s.f. count is fewer than the value's integer
digit count (``48130`` to 2 s.f. is ``48000``, not ``48``).

Every seed's rounding is built two genuinely different ways and cross-checked
(``_verify_sig_fig_rounding``): construction uses ``Decimal.adjusted()`` +
``.quantize(ROUND_HALF_UP)``; the check manipulates the plain fixed-point
digit string directly (slicing + a carry-safe integer increment, no
``Decimal`` rounding API at all). ``ROUND_HALF_UP`` is used explicitly
throughout -- Python's own ``round()``/bare ``Decimal`` rounds half-to-even,
which is not the taught convention -- and every draw whose cut digit is an
exact tie is resampled rather than trusted to either rounding mode's
convention. ``answer_text`` is built from the quantized ``Decimal`` via
``format(d, "f")`` (never ``str()``/``:g`` on a float), which is what
preserves a significant trailing zero (``4.7985`` to 3 s.f. is ``"4.80"``)
and what keeps large-magnitude results in plain digits rather than
``Decimal``'s own exponential display (``4.8E+4``).

Every distractor is filtered through ``pick_distractors`` with the exact tolerance ``grade_number`` will apply
to *this* seed's ``answer_text`` (half a unit in its own last stated decimal place, mirroring
``mathgen.grading._typed_number_tolerance``, private to that module and so reimplemented here) -- not the flat
0.051 blanket, and not assumed safe by construction. Measured directly: the "rounds-to-decimal-places-instead"
misconception is only ever a *safely distinct* distractor on the small-value tier, because on every other tier
it states MORE decimal places than the correct rounding, and a closer approximation of the same true value is
mathematically almost always inside the coarser correct answer's own tolerance window.

``engineering_notation`` re-groups an already-given scientific-notation quantity so its exponent is a multiple
of three and its coefficient may run from 1 up to (not including) 1000 -- ``scientific-notation``'s own
normalisation keeps a single digit before the point with any integer exponent. Built ``answer_kind: mcq``, not
``expression``: verified directly against ``mathgen.grading.grade_expression`` that a plain expression grader
accepts any numerically-equal typed form with no shape check, so it cannot enforce "the exponent is a multiple
of three" the way this skill's whole point requires. Every draw's natural scientific exponent is forced to NOT
already be a multiple of three (a resample would never even be needed here: the exponent is built as ``eng_exp
+ shift`` with ``shift`` drawn from ``{1, 2}``, so it can never land on ``eng_exp`` itself), which the
in-generator check (``_verify_engineering_pair``) reproves independently via ``3 * (sci_exp // 3)`` computed
straight from the raw value, a route construction never takes (construction instead looks the engineering
exponent up from a tier table and multiplies the scientific coefficient by the chosen shift).

The ``exam`` source adds a genuinely new operation before that
same regroup step: engineering form of the SUM or DIFFERENCE of two
scientific-notation values whose exponents are far enough apart that one
term dominates (worked example: ``3.412e-9 + 6.105e4``). Both inputs and
the final mantissa are stated to a fixed 4 significant figures throughout,
combined in exact ``Decimal`` arithmetic (exponents aligned by ``scaleb``,
never ``float``) and cross-checked against an independent ``Fraction``
reconstruction of the same sum. The exponent gap is drawn from ``{3, 4}`` --
small enough that the smaller term still moves the combined mantissa's own
4th significant digit (making "just take the dominant term" a genuine,
resampled-until-exercisable misconception, not a vacuously-true one), large
enough that the dominant term is always >=100x the other, so a learner
cannot shortcut by eyeballing which term is bigger and must actually align
exponents. The regrouping step reuses ``_verify_engineering_pair`` as its
independent oracle exactly as the other three variants do; the final
4-significant-figure rounding reuses ``_verify_sig_fig_rounding`` from the
``significant_figures`` section above, called but never modified.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP
from fractions import Fraction

from ..core import VerificationError, check, generator, latex_answer, phrase, pick_distractors


# ===========================================================================
# significant_figures
# ===========================================================================


def _fmt_decimal(d: Decimal) -> str:
    """Digit-string formatting that never collapses a significant trailing
    zero (``str()``/an f-string ``:g`` on a float would) and never renders
    exponential notation, whatever sign the quantized exponent has."""
    if d == d.to_integral_value():
        return str(int(d))
    return format(d, "f")


def _decimal_places(d: Decimal) -> int:
    exp = d.as_tuple().exponent
    return -exp if exp < 0 else 0


def _fmt_at_places(d: Decimal, places: int) -> str:
    """Format ``d`` at exactly ``places`` decimal places (padding with zeros
    if ``d`` is naturally coarser, rounding if it is naturally finer) --
    shared by the answer and every distractor so none is identifiable by
    shape alone. A per-candidate "strip to a bare integer when whole"
    formatter (``_fmt_decimal`` on its own) would make a distractor that
    happens to land on a whole number the only non-decimal option on offer
    whenever the correct answer itself carries a decimal place."""
    if places <= 0:
        return str(int(d.quantize(Decimal(1), rounding=ROUND_HALF_UP)))
    quantum = Decimal(1).scaleb(-places)
    return format(d.quantize(quantum, rounding=ROUND_HALF_UP), "f")


def _local_number_tolerance(places: int) -> Fraction:
    """Mirrors ``mathgen.grading._typed_number_tolerance`` -- half a unit in the
    last stated decimal place, exact match at zero decimal places. That
    function is private to ``mathgen.grading``, so the two-line formula is
    reimplemented here rather than imported across the module boundary."""
    if places <= 0:
        return Fraction(0)
    return Fraction(1, 2 * 10**places)


def _is_boundary(value: Decimal, sf: int) -> bool:
    """True when there is no genuine rounding DECISION at the sf-th
    significant digit: either every dropped digit is zero (the value is
    already an exact multiple of the target quantum, so there is nothing to
    round and no direction to invert for ``rounds-wrong-direction``), or the
    digit immediately after the cut is exactly '5' with nothing nonzero
    after it (an exact rounding tie, excluded by resampling rather than
    trusted to either rounding convention, matching the decimal-place
    skills' existing rule)."""
    text = format(value, "f")
    int_part, _, frac_part = text.partition(".")
    all_digits = int_part + frac_part
    first_nonzero = next(i for i, ch in enumerate(all_digits) if ch != "0")
    cut = first_nonzero + sf
    rest = all_digits[cut:]
    if rest == "" or rest.strip("0") == "":
        return True  # no dropped digits are nonzero: nothing to round
    if rest[0] == "5" and rest[1:].strip("0") == "":
        return True  # exact tie
    return False


def _round_significant(value: Decimal, sf: int) -> tuple[Decimal, int]:
    """Construction's own route: the place value of the first significant
    digit via ``Decimal.adjusted()``, then ``.quantize(ROUND_HALF_UP)``.
    Returns ``(rounded_value, quantize_exponent)``."""
    exp = value.adjusted()
    q_exp = exp - sf + 1
    quantum = Decimal(1).scaleb(q_exp)
    return value.quantize(quantum, rounding=ROUND_HALF_UP), q_exp


def _digit_string_round(value: Decimal, sf: int) -> Decimal:
    """Independent route: slice the plain fixed-point digit string directly
    and increment it as a plain int (which is carry-safe for free, e.g.
    "99" + 1 -> "100") -- no ``Decimal.adjusted()``/``.quantize()`` anywhere
    in this function."""
    text = format(value, "f")
    sign = ""
    if text.startswith("-"):
        sign, text = "-", text[1:]
    int_part, _, frac_part = text.partition(".")
    all_digits = int_part + frac_part
    point_index = len(int_part)
    first_nonzero = next(i for i, ch in enumerate(all_digits) if ch != "0")
    msd_place = point_index - 1 - first_nonzero
    kept = all_digits[first_nonzero : first_nonzero + sf]
    rest = all_digits[first_nonzero + sf :]
    round_up = len(rest) > 0 and rest[0] >= "5"
    kept_int = int(kept) + (1 if round_up else 0)
    last_kept_place = msd_place - sf + 1
    result = Decimal(kept_int) * (Decimal(10) ** last_kept_place)
    return -result if sign else result


def _verify_sig_fig_rounding(value: Decimal, sf: int, expected: Decimal) -> None:
    """The independent check: the digit-string route must agree with
    whatever construction's Decimal-quantize route produced."""
    check(
        _digit_string_round(value, sf) == expected,
        "significant_figures: digit-string route disagrees with the Decimal route",
    )


def _sig_fig_candidates(value: Decimal, sf: int):
    """The correct rounding plus every wrong-answer route, all as Decimals. Distinctness from the answer is NOT
    assumed here -- callers check it, since which routes are safely distinct is tier-dependent."""
    correct, q_exp = _round_significant(value, sf)
    _verify_sig_fig_rounding(value, sf, correct)

    counts_leading_zeros = value.quantize(Decimal(1).scaleb(q_exp + 1), rounding=ROUND_DOWN)
    rounds_to_decimal_places = value.quantize(Decimal(1).scaleb(-sf), rounding=ROUND_HALF_UP)
    drops_magnitude = correct / (Decimal(10) ** q_exp) if q_exp > 0 else correct

    quantum = Decimal(1).scaleb(q_exp)
    down = value.quantize(quantum, rounding=ROUND_DOWN)
    up = value.quantize(quantum, rounding=ROUND_UP)
    check(down != up, "significant_figures: no rounding decision to invert (exact multiple of the quantum)")
    rounds_wrong_direction = up if correct == down else down

    return (
        correct,
        q_exp,
        {
            "counts-leading-zeros": counts_leading_zeros,
            "rounds-to-decimal-places-instead": rounds_to_decimal_places,
            "drops-magnitude": drops_magnitude,
            "rounds-wrong-direction": rounds_wrong_direction,
        },
    )


def _finish(value: Decimal, sf: int, correct: Decimal, wrong: dict) -> dict:
    places = _decimal_places(correct)
    answer_text = _fmt_at_places(correct, places)
    tolerance = _local_number_tolerance(places)
    # A candidate that rounds to zero is dismissed at a glance, so it is not offered.
    candidates = [_fmt_at_places(v, places) for v in wrong.values() if v != 0]
    # Neighbouring values one unit in the last kept place away from the answer stand in for any dismissed zero.
    step = Decimal(1).scaleb(-places)
    candidates += [_fmt_at_places(v, places) for v in (correct + step, correct - step) if v > 0]
    figure_word = "significant figure" if sf == 1 else "significant figures"
    text = f"Round {_fmt_decimal(value)} to {sf} {figure_word}."
    return dict(
        text=text,
        latex=text,
        answer=float(correct),
        answer_text=answer_text,
        distractors=pick_distractors(answer_text, candidates, tolerance=tolerance),
        working=[f"Keep {sf} significant figure{'s' if sf != 1 else ''} and round: {answer_text}."],
    )


def _build_plain(rng) -> dict:
    """easy: a value >= 1 with a few decimal digits, no leading- or
    trailing-zero complication (msd's own place value, ``msd_place``, stays
    below ``sf - 1`` so the correct rounding always lands inside the
    fractional part)."""
    for _attempt in range(200):
        int_digit_count = rng.choice((1, 2))
        int_part = rng.randint(10 ** (int_digit_count - 1), 10**int_digit_count - 1)
        dec_digits = [rng.randint(0, 9) for _ in range(5)]
        if dec_digits[-1] == 0:
            continue
        msd_place = int_digit_count - 1
        sf_choices = [s for s in (2, 3, 4) if s > msd_place + 1]
        sf = rng.choice(sf_choices)
        value = Decimal(f"{int_part}.{''.join(map(str, dec_digits))}")
        if _is_boundary(value, sf):
            continue
        correct, q_exp, wrong = _sig_fig_candidates(value, sf)
        check(q_exp <= 0, "significant_figures/plain: unexpectedly needs trailing-zero padding")
        check(
            correct != wrong["rounds-wrong-direction"],
            "significant_figures/plain: wrong-direction distractor collided with the answer",
        )
        return _finish(value, sf, correct, wrong)
    raise RuntimeError("significant_figures/plain: could not find a non-boundary draw")


def _build_small_value(rng) -> dict:
    """medium: a value < 1 with at least one leading zero after the decimal
    point (``value.adjusted() <= -2``), forcing "leading zeros are never
    significant" on every seed."""
    for _attempt in range(200):
        leading_zeros = rng.choice((1, 2, 3))
        first = rng.randint(1, 9)
        rest = [rng.randint(0, 9) for _ in range(3)]
        digits_str = str(first) + "".join(map(str, rest))
        sf = rng.choice((1, 2, 3))
        value = Decimal(f"0.{'0' * leading_zeros}{digits_str}")
        if _is_boundary(value, sf):
            continue
        correct, q_exp, wrong = _sig_fig_candidates(value, sf)
        check(value.adjusted() <= -2, "significant_figures/small_value: no leading zero forced")
        check(q_exp < 0, "significant_figures/small_value: expected extra decimal precision")
        check(
            correct != wrong["rounds-wrong-direction"],
            "significant_figures/small_value: wrong-direction distractor collided with the answer",
        )
        return _finish(value, sf, correct, wrong)
    raise RuntimeError("significant_figures/small_value: could not find a non-boundary draw")


def _build_large_trailing_zero(rng) -> dict:
    """hard: an integer where the requested s.f. count is fewer than its
    integer digit count, forcing "trailing zeros preserve magnitude" on
    every seed."""
    for _attempt in range(200):
        num_digits = rng.choice((4, 5, 6))
        first = rng.randint(1, 9)
        rest = [rng.randint(0, 9) for _ in range(num_digits - 1)]
        digits_str = str(first) + "".join(map(str, rest))
        value = Decimal(digits_str)
        sf = rng.choice([s for s in (1, 2, 3) if s < num_digits])
        if _is_boundary(value, sf):
            continue
        correct, q_exp, wrong = _sig_fig_candidates(value, sf)
        check(q_exp > 0, "significant_figures/large_trailing_zero: expected trailing-zero padding")
        check(
            correct != wrong["rounds-wrong-direction"],
            "significant_figures/large_trailing_zero: wrong-direction distractor collided with the answer",
        )
        return _finish(value, sf, correct, wrong)
    raise RuntimeError("significant_figures/large_trailing_zero: could not find a non-boundary draw")


@generator(
    "significant_figures",
    level=9,
    presets={
        "easy": dict(variant="plain"),
        "medium": dict(variant="small_value"),
        "hard": dict(variant="large_trailing_zero"),
    },
)
def significant_figures(rng, variant="large_trailing_zero"):
    """Round a value to a stated number of significant figures (not the stated DECIMAL PLACE that
    ``round-decimals`` teaches). Default variant is the hard tier: measured directly as the only one of the
    three that clears the shared ``test_all_topics_usually_have_enough_distractors`` sweep's bare
    ``make(topic, seed=s)`` call at >=85%; ``plain``'s only reliably-safe distractor is the always-distinct
    ``rounds-wrong-direction`` one."""
    if variant == "plain":
        return _build_plain(rng)
    if variant == "small_value":
        return _build_small_value(rng)
    if variant == "large_trailing_zero":
        return _build_large_trailing_zero(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# ===========================================================================
# engineering_notation
# ===========================================================================

_SI_PREFIXES = {
    -12: "pico",
    -9: "nano",
    -6: "micro",
    -3: "milli",
    3: "kilo",
    6: "mega",
    9: "giga",
    12: "tera",
}


def _fmt_mantissa(d: Decimal) -> str:
    if d == d.to_integral_value():
        return str(int(d))
    return format(d, "f")


def _verify_engineering_pair(value: Decimal, eng_exp: int, mantissa: Decimal) -> None:
    """Independent route: derive the engineering exponent straight from the
    raw value's own natural scientific exponent via floor division, and
    recompute the mantissa as ``value / 10**exponent`` -- construction
    instead looks the engineering exponent up from a tier table and
    multiplies the scientific coefficient by a chosen shift."""
    sci_exp = value.adjusted()
    check(3 * (sci_exp // 3) == eng_exp, "engineering_notation: exponent disagrees with 3 * floor(sci_exp / 3)")
    check_mantissa = value / (Decimal(10) ** eng_exp)
    check(Decimal(1) <= check_mantissa < Decimal(1000), "engineering_notation: mantissa outside [1, 1000)")
    check(check_mantissa == mantissa, "engineering_notation: mantissa disagrees with value / 10**exponent")


def _option_text(mantissa: Decimal, exp: int, prefix: str) -> str:
    m = _fmt_mantissa(mantissa)
    return f"{m} × 10^{exp} ({m} {prefix})"


def _build_engineering(rng, eng_exp_choices: tuple, shift_choices: tuple = (1, 2)) -> dict:
    eng_exp = rng.choice(eng_exp_choices)
    shift = rng.choice(shift_choices)
    sci_exp = eng_exp + shift
    m = rng.randint(100, 999)
    c = Decimal(m).scaleb(-2)  # the given scientific coefficient, [1.00, 9.99]
    mantissa = Decimal(m).scaleb(shift - 2)  # the engineering coefficient
    value = Decimal(m).scaleb(sci_exp - 2)
    check(sci_exp % 3 != 0, "engineering_notation: scientific exponent is already a multiple of 3")
    _verify_engineering_pair(value, eng_exp, mantissa)

    prefix = _SI_PREFIXES[eng_exp]
    answer_text = _option_text(mantissa, eng_exp, prefix)

    scientific_grouping = f"{_fmt_mantissa(c)} × 10^{sci_exp}"
    mantissa_out_of_range = f"{_fmt_mantissa(c)} × 10^{eng_exp} ({_fmt_mantissa(c)} {prefix})"
    other_exponents = [e for e in _SI_PREFIXES if e != eng_exp]
    wrong_prefix_exp = rng.choice(other_exponents)
    wrong_prefix = (
        f"{_fmt_mantissa(mantissa)} × 10^{eng_exp} ({_fmt_mantissa(mantissa)} {_SI_PREFIXES[wrong_prefix_exp]})"
    )

    candidates = [scientific_grouping, mantissa_out_of_range, wrong_prefix]
    sign_exp = -eng_exp
    if sign_exp in _SI_PREFIXES:
        sign_error = f"{_fmt_mantissa(mantissa)} × 10^{sign_exp} ({_fmt_mantissa(mantissa)} {_SI_PREFIXES[sign_exp]})"
        candidates.append(sign_error)

    for cand in candidates:
        check(cand != answer_text, f"engineering_notation: a distractor matches the answer text: {cand!r}")

    # `pick_distractors` caps at 3 and takes candidates IN ORDER: with 4
    # candidates and a fixed order, the 4th (sign_error, when present) would
    # NEVER survive and its exercisability would measure 0% forever (found
    # by rendering 600 questions and grepping for it -- none). Shuffle so
    # every misconception gets a genuine, roughly-equal chance across seeds.
    rng.shuffle(candidates)

    text = (
        f"Write {_fmt_mantissa(c)} × 10^{sci_exp} in engineering notation "
        f"(a power of ten that is a multiple of 3). Include the SI-prefix name."
    )
    return dict(
        text=text,
        latex=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates),
        working=[
            f"The nearest lower multiple of 3 below the exponent {sci_exp} is {eng_exp}, so shift "
            f"the decimal point {shift} place{'s' if shift != 1 else ''}: {answer_text}."
        ],
    )


def _fmt_sf(d: Decimal) -> str:
    """Format ``d`` showing every one of its own significant figures,
    including a trailing zero the naive ``_fmt_mantissa``/``str()`` would
    drop (e.g. a rounded mantissa of exactly ``120.0`` must still show the
    trailing ``.0`` -- it is a stated 4th significant figure, not a
    decoration) -- reuses the ``significant_figures`` section's own
    precision-aware formatter (``_decimal_places``/``_fmt_at_places``)
    rather than duplicating it."""
    return _fmt_at_places(d, _decimal_places(d))


def _option_text_sf(mantissa_str: str, exp: int, prefix: str) -> str:
    """Same SHAPE as ``_option_text``'s output (``"{m} × 10^{exp} ({m}
    {prefix})"``) but takes an already-formatted mantissa string instead of
    calling ``_fmt_mantissa`` -- the sum/difference variant needs
    ``_fmt_sf``'s precision-preserving format instead, never
    ``_fmt_mantissa``'s collapse-to-integer-when-whole behaviour, so the
    "each value to 4 s.f." claim in the prompt is honoured even when a
    mantissa lands on a round number. Kept as a separate function rather
    than changing ``_option_text`` itself, which the other three variants
    still rely on unmodified."""
    return f"{mantissa_str} × 10^{exp} ({mantissa_str} {prefix})"


def _build_engineering_sum(rng) -> dict:
    """exam: engineering form of a SUM or DIFFERENCE of two scientific-
    notation values with far-apart exponents (worked example:
    ``3.412e-9 + 6.105e4``) -- a genuinely new operation (combine two
    values first, in exact Decimal arithmetic, aligning exponents by
    ``scaleb``, never through ``float``) before the existing
    regroup-to-engineering-form step every other variant already performs.
    Both inputs and the final mantissa are stated to a fixed 4 significant
    figures throughout (the worked example's own coefficient precision), so
    a learner is never guessing how much precision the final answer should
    carry.

    The exponent gap (``diff``, drawn from ``{3, 4}``) is deliberately small
    enough that the smaller term still moves the combined mantissa's own
    4th significant digit -- and large enough (the dominant term is always
    >=100x the other) that the addition is only tractable by actually
    aligning exponents, not by eyeballing which term is bigger. A draw whose
    smaller term happens not to move that digit (only possible at
    ``diff=4``) is resampled rather than shipped with a non-exercisable
    "ignores the smaller term" distractor.

    ``eng_exp_a`` is chosen upfront, exactly as the other three variants'
    tier tables choose it, and never re-derived from ``value.adjusted()`` in
    construction -- the sole ``.adjusted()`` call on the combined ``value``
    happens inside ``_verify_engineering_pair``, the independent oracle a
    resample-on-failure guards against a carry moving the sum out of
    ``eng_exp_a``'s own decade. The bracket used for the "single digit
    before the point" scientific form (``shift``) is read off the already-
    verified ``raw_mantissa``'s own magnitude instead, a second, genuinely
    different route from ``.adjusted()``.
    """
    for _attempt in range(400):
        eng_exp_a = rng.choice(tuple(_SI_PREFIXES))
        shift_a = rng.choice((1, 2))
        exp_a = eng_exp_a + shift_a
        diff = rng.choice((3, 4))
        exp_b = exp_a - diff
        sign = rng.choice((1, -1))

        m_a = rng.randint(1000, 9999)
        m_b = rng.randint(1000, 9999)
        c_a = Decimal(m_a).scaleb(-3)  # [1.000, 9.999], 4 s.f.
        c_b = Decimal(m_b).scaleb(-3)  # [1.000, 9.999], 4 s.f.

        value_a = c_a.scaleb(exp_a)
        value_b = c_b.scaleb(exp_b)
        value = value_a + sign * value_b

        # Independent route for the addition itself: an exact Fraction
        # reconstruction from the two inputs, never through Decimal.
        expected = Fraction(c_a) * Fraction(10) ** exp_a + sign * Fraction(c_b) * Fraction(10) ** exp_b
        check(
            Fraction(value) == expected,
            "engineering_notation/sum_difference: Decimal alignment disagrees with the Fraction route",
        )

        raw_mantissa = value.scaleb(-eng_exp_a)
        if not (Decimal(1) <= raw_mantissa < Decimal(1000)):
            continue  # a carry moved the sum out of eng_exp_a's own decade
        try:
            _verify_engineering_pair(value, eng_exp_a, raw_mantissa)
        except VerificationError:
            continue  # eng_exp_a is no longer this sum's correct exponent

        shift = 0 if raw_mantissa < 10 else (1 if raw_mantissa < 100 else 2)
        if shift == 0:
            continue  # degenerate: sci- and eng-form exponents would match

        if _is_boundary(raw_mantissa, 4):
            continue  # exact rounding tie -- no genuine rounding decision

        correct_mantissa, _q_exp = _round_significant(raw_mantissa, 4)
        _verify_sig_fig_rounding(raw_mantissa, 4, correct_mantissa)
        if not (Decimal(1) <= correct_mantissa < Decimal(1000)):
            continue  # rounding pushed the mantissa out of the eng range

        dominant_only_mantissa = c_a.scaleb(shift_a)
        if dominant_only_mantissa == correct_mantissa:
            continue  # the smaller term didn't move the answer -- resample

        eng_exp = eng_exp_a
        sci_exp = eng_exp + shift
        prefix = _SI_PREFIXES[eng_exp]
        correct_str = _fmt_sf(correct_mantissa)
        answer_text = _option_text_sf(correct_str, eng_exp, prefix)

        sci_mantissa = correct_mantissa.scaleb(-shift)
        sci_str = _fmt_sf(sci_mantissa)
        scientific_grouping = f"{sci_str} × 10^{sci_exp}"
        mantissa_out_of_range = _option_text_sf(sci_str, eng_exp, prefix)

        other_exponents = [e for e in _SI_PREFIXES if e != eng_exp]
        wrong_prefix_exp = rng.choice(other_exponents)
        wrong_prefix = _option_text_sf(correct_str, eng_exp, _SI_PREFIXES[wrong_prefix_exp])

        dominant_str = _fmt_sf(dominant_only_mantissa)
        ignores_smaller_term = _option_text_sf(dominant_str, eng_exp, prefix)

        candidates = [scientific_grouping, mantissa_out_of_range, wrong_prefix, ignores_smaller_term]
        sign_exp = -eng_exp
        if sign_exp in _SI_PREFIXES:
            sign_error = _option_text_sf(correct_str, sign_exp, _SI_PREFIXES[sign_exp])
            candidates.append(sign_error)

        for cand in candidates:
            check(
                cand != answer_text,
                f"engineering_notation/sum_difference: a distractor matches the answer text: {cand!r}",
            )

        # Same reason as `_build_engineering`: more candidates than
        # `pick_distractors` keeps (3), so shuffle for a fair chance across
        # seeds rather than always dropping the same trailing one(s).
        rng.shuffle(candidates)

        a_str = _fmt_sf(c_a)
        b_str = _fmt_sf(c_b)
        op_symbol = "+" if sign == 1 else "-"
        text = phrase(
            rng,
            (
                "Each value below is given to 4 significant figures: "
                "{a} × 10^{exp_a} {op} {b} × 10^{exp_b}. Write the "
                "result in engineering notation (a power of ten that is a "
                "multiple of 3), rounding the mantissa to 4 significant "
                "figures. Include the SI-prefix name.",
                "Both values are stated to 4 significant figures: "
                "{a} × 10^{exp_a} {op} {b} × 10^{exp_b}. Express "
                "the result in engineering notation (exponent a multiple of "
                "3), giving the mantissa to 4 significant figures and "
                "naming the SI prefix.",
            ),
            a=a_str,
            exp_a=exp_a,
            op=op_symbol,
            b=b_str,
            exp_b=exp_b,
        )
        return dict(
            text=text,
            latex=text,
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text, plain=True),
            distractors=pick_distractors(answer_text, candidates),
            working=[
                f"Align the exponents, combine the values exactly, then "
                f"regroup to the nearest lower multiple of 3 ({eng_exp}) and "
                f"round the mantissa to 4 significant figures: {answer_text}."
            ],
        )
    raise RuntimeError("engineering_notation/sum_difference: could not find a valid draw")


@generator(
    "engineering_notation",
    level=9,
    presets={
        "easy": dict(variant="large_positive"),
        "medium": dict(variant="small_value"),
        "hard": dict(variant="wide_exponent"),
        "exam": dict(variant="sum_difference"),
    },
)
def engineering_notation(rng, variant="large_positive"):
    """Re-group an already-given scientific-notation quantity into
    engineering notation (a power of ten that is a multiple of 3, coefficient
    in [1, 1000)). The ``exam`` variant instead combines two scientific-
    notation values first (see ``_build_engineering_sum``) before the same
    regroup step."""
    if variant == "large_positive":
        return _build_engineering(rng, (3, 6))
    if variant == "small_value":
        return _build_engineering(rng, (-3, -6))
    if variant == "wide_exponent":
        return _build_engineering(rng, (9, 12, -9, -12))
    if variant == "sum_difference":
        return _build_engineering_sum(rng)
    raise ValueError(f"unknown variant: {variant!r}")
