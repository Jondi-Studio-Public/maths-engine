"""Degree<->radian conversion, decimal-answer half of an exam-style item.

- ``t4_degrees_to_radians_decimal`` (skill `angle-degrees-to-radians-decimal`)
  -- ``radians = total_degrees * pi / 180``.
- ``t4_radians_to_degrees_decimal`` (skill `angle-radians-to-degrees-decimal`)
  -- ``degrees = R * 180 / pi``.

Both are graded ``answer: number`` on the EXISTING typed-numeric-precision path
(`mathgen.grading._typed_number_target`/`_typed_number_tolerance`): the comparison tolerance is half a unit in
whatever decimal place the generator's own ``answer_text`` states, not the loose 0.051 spoken-transcript
tolerance.

``answer`` is always a plain ``float`` and ``answer_text`` is
``f"{value:.{places}f}"`` applied to a value already rounded with
``Decimal``/``ROUND_HALF_UP`` (never bare Python ``round()``, which rounds
half-to-even and can disagree with what a learner's calculator shows). Every
draw that would land within 0.05 of an exact ``x.5`` rounding boundary at the
stated ``places`` is resampled (``_rounds_unambiguously``, the pattern from
``p5d_interval_traffic_build.py``, duplicated here
rather than centralised -- already duplicated three times across
the codebase).

**Sign choice:** a "random (+/-)" sign for both ``medium`` tiers would be wrong, because
each tier's own ``dropped_sign``/``forgot-sign`` distractor formula (``abs(D)*pi/180`` / ``abs(R)*180/pi``) is
only a genuinely different value from the answer when the input is negative -- for a positive input, "drop the
negative sign" is a no-op and the distractor is IDENTICAL to the answer, which would raise `VerificationError`
on ~half of all seeds. Both `medium` tiers draw their signed value always negative instead: still a genuine
signed, non-whole escalation over `easy`'s positive whole number, and the only sign for which "drops the
negative sign" is a coherent misconception.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

from ..core import (
    check,
    format_pi_multiple,
    generator,
    pick_distractors,
    rounds_unambiguously,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _round_half_up(value: float, places: int) -> float:
    """Round ``value`` to ``places`` decimals via ``Decimal``/``ROUND_HALF_UP``
    (never bare Python ``round()``, which is round-half-to-even and can
    disagree with a learner's calculator). Normalises ``-0.0`` to ``0.0`` so
    a distractor that rounds to zero never renders as the ugly, and
    misleading-looking, ``"-0.00"``."""
    quantum = Decimal(1).scaleb(-places)
    rounded = float(Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP))
    return 0.0 if rounded == 0 else rounded


def _fmt(value: float, places: int) -> str:
    return f"{value:.{places}f}"


def _tolerance(places: int) -> Fraction:
    """Mirrors ``mathgen.grading._typed_number_tolerance``: half a unit in the
    last stated decimal place. Reimplemented here (that function is private
    to ``mathgen.grading``) exactly as ``p8a_notation_build.py``'s
    ``_local_number_tolerance`` already does."""
    return Fraction(1, 2 * 10**places)


def _check_distractor_gap(answer_text: str, candidate_text: str, places: int, where: str) -> None:
    """The check that actually matches what the real grader will do: compare
    the two ROUNDED, DISPLAYED values (exactly what `grade_number` compares,
    via `_typed_number_target`) rather than trust an abstract margin computed
    on the unrounded construction values. Two unrounded values can be more
    than half a unit apart and still round to the same displayed value, so
    this is asserted on the post-rounding text every time, not derived from
    the pre-rounding gap."""
    gap = abs(Fraction(candidate_text) - Fraction(answer_text))
    check(
        gap > _tolerance(places),
        f"{where}: distractor {candidate_text!r} is within the grader's tolerance "
        f"of the answer {answer_text!r} (gap={gap}, tolerance={_tolerance(places)})",
    )


def _degree_word(n) -> str:
    """``"degree"`` for a magnitude of exactly 1, ``"degrees"`` otherwise --
    matches ``p8d_quadrilateral_theorems_build.py``'s own ``_deg_word``
    helper, needed here because `easy`'s ``D`` ranges over ``[1, 359]`` and
    so can genuinely be 1."""
    return "degree" if abs(n) == 1 else "degrees"


# ---------------------------------------------------------------------------
# Topic 1: t4_degrees_to_radians_decimal
# ---------------------------------------------------------------------------


def _build_deg_to_rad_whole(rng) -> dict:
    """easy: D whole degrees, D in [1, 359], 2dp."""
    places = 2
    for _attempt in range(30):
        D = rng.randint(1, 359)
        radians = D * math.pi / 180
        if rounds_unambiguously(radians, places):
            break
    else:
        raise RuntimeError("t4_degrees_to_radians_decimal/forward_whole: no unambiguous draw in 30 attempts")

    check(abs(radians - math.radians(D)) < 1e-9, f"forward_whole: own arithmetic disagrees with math.radians for D={D}")

    rounded = _round_half_up(radians, places)
    answer_text = _fmt(rounded, places)
    text = f"Convert {D} {_degree_word(D)} to radians. Give your answer to 2 decimal places."

    forgot_pi = D / 180
    check(abs(radians - forgot_pi) > 0.005, f"forward_whole: forgot-pi margin too small for D={D}")
    forgot_pi_text = _fmt(_round_half_up(forgot_pi, places), places)
    _check_distractor_gap(answer_text, forgot_pi_text, places, f"forward_whole D={D} forgot_pi")

    doubled = 2 * radians
    check(abs(doubled - radians) > 0.005, f"forward_whole: doubled margin too small for D={D}")
    doubled_text = _fmt(_round_half_up(doubled, places), places)
    _check_distractor_gap(answer_text, doubled_text, places, f"forward_whole D={D} doubled")

    candidates = [forgot_pi_text, doubled_text]
    distractors = pick_distractors(answer_text, candidates, tolerance=_tolerance(places))

    return dict(
        text=text,
        latex=text,
        answer=rounded,
        answer_text=answer_text,
        distractors=distractors,
        working=[f"radians = {D} times pi divided by 180 = {answer_text}."],
    )


def _build_deg_to_rad_signed(rng) -> dict:
    """medium: D a signed (always negative -- see module docstring),
    non-whole one-decimal value, 3dp."""
    places = 3
    for _attempt in range(30):
        whole = rng.randint(10, 350)
        tenths = rng.randint(1, 9)
        D = -(whole + tenths / 10)
        radians = D * math.pi / 180
        if rounds_unambiguously(radians, places):
            break
    else:
        raise RuntimeError("t4_degrees_to_radians_decimal/forward_signed: no unambiguous draw in 30 attempts")

    check(
        abs(radians - math.radians(D)) < 1e-9, f"forward_signed: own arithmetic disagrees with math.radians for D={D}"
    )

    rounded = _round_half_up(radians, places)
    answer_text = _fmt(rounded, places)
    text = f"Convert {D:.1f} degrees to radians. Give your answer to 3 decimal places."

    dropped_sign = abs(D) * math.pi / 180
    check(abs(dropped_sign - radians) > 0.0005, f"forward_signed: dropped-sign margin too small for D={D}")
    dropped_sign_text = _fmt(_round_half_up(dropped_sign, places), places)
    _check_distractor_gap(answer_text, dropped_sign_text, places, f"forward_signed D={D} dropped_sign")

    forgot_pi = D / 180
    check(abs(radians - forgot_pi) > 0.0005, f"forward_signed: forgot-pi margin too small for D={D}")
    forgot_pi_text = _fmt(_round_half_up(forgot_pi, places), places)
    _check_distractor_gap(answer_text, forgot_pi_text, places, f"forward_signed D={D} forgot_pi")

    candidates = [dropped_sign_text, forgot_pi_text]
    distractors = pick_distractors(answer_text, candidates, tolerance=_tolerance(places))

    return dict(
        text=text,
        latex=text,
        answer=rounded,
        answer_text=answer_text,
        distractors=distractors,
        working=[f"radians = {D:.1f} times pi divided by 180 = {answer_text}."],
    )


def _build_deg_to_rad_dms(rng) -> dict:
    """hard: D degrees, M minutes (M never 0), 4dp."""
    places = 4
    for _attempt in range(30):
        D = rng.randint(1, 359)
        M = rng.randint(1, 59)
        total_degrees = D + Fraction(M, 60)
        radians = float(total_degrees) * math.pi / 180
        if rounds_unambiguously(radians, places):
            break
    else:
        raise RuntimeError("t4_degrees_to_radians_decimal/dms_forward: no unambiguous draw in 30 attempts")

    check(
        abs(radians - math.radians(float(total_degrees))) < 1e-9,
        f"dms_forward: own arithmetic disagrees with math.radians for D={D} M={M}",
    )

    rounded = _round_half_up(radians, places)
    answer_text = _fmt(rounded, places)
    text = f"Convert {D}\u00b0{M}\u2032 to radians. Give your answer to 4 decimal places."
    check("\u00b0" in text and "\u2032" in text, "dms_forward: degree/prime symbols dropped from the question text")

    forgot_minutes = D * math.pi / 180
    check(abs(radians - forgot_minutes) > 0.00005, f"dms_forward: forgot-minutes margin too small for D={D} M={M}")
    forgot_minutes_text = _fmt(_round_half_up(forgot_minutes, places), places)
    _check_distractor_gap(answer_text, forgot_minutes_text, places, f"dms_forward D={D} M={M} forgot_minutes")

    minutes_as_decimal_total = D + Fraction(M, 100)
    minutes_as_decimal = float(minutes_as_decimal_total) * math.pi / 180
    check(
        abs(radians - minutes_as_decimal) > 0.00005, f"dms_forward: minutes-as-decimal margin too small for D={D} M={M}"
    )
    minutes_as_decimal_text = _fmt(_round_half_up(minutes_as_decimal, places), places)
    _check_distractor_gap(answer_text, minutes_as_decimal_text, places, f"dms_forward D={D} M={M} minutes_as_decimal")

    forgot_pi = float(total_degrees) / 180
    check(abs(radians - forgot_pi) > 0.00005, f"dms_forward: forgot-pi margin too small for D={D} M={M}")
    forgot_pi_text = _fmt(_round_half_up(forgot_pi, places), places)
    _check_distractor_gap(answer_text, forgot_pi_text, places, f"dms_forward D={D} M={M} forgot_pi")

    candidates = [forgot_minutes_text, minutes_as_decimal_text, forgot_pi_text]
    distractors = pick_distractors(answer_text, candidates, tolerance=_tolerance(places))

    return dict(
        text=text,
        latex=text,
        answer=rounded,
        answer_text=answer_text,
        distractors=distractors,
        working=[f"radians = ({D} + {M}/60) times pi divided by 180 = {answer_text}."],
    )


@generator(
    "t4_degrees_to_radians_decimal",
    level=12,
    presets={
        "easy": dict(variant="forward_whole"),
        "medium": dict(variant="forward_signed"),
        "hard": dict(variant="dms_forward"),
    },
)
def t4_degrees_to_radians_decimal(rng, variant="forward_whole"):
    """Convert degrees (whole, signed-decimal, or DMS) to radians, decimal answer."""
    if variant == "forward_whole":
        return _build_deg_to_rad_whole(rng)
    if variant == "forward_signed":
        return _build_deg_to_rad_signed(rng)
    if variant == "dms_forward":
        return _build_deg_to_rad_dms(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# ---------------------------------------------------------------------------
# Topic 2: t4_radians_to_degrees_decimal
# ---------------------------------------------------------------------------


def _build_rad_to_deg_basic(rng) -> dict:
    """easy: R a 2dp decimal in [0.50, 6.00], 1dp answer."""
    places = 1
    for _attempt in range(30):
        n = rng.randint(50, 600)
        R = n / 100
        degrees = R * 180 / math.pi
        if rounds_unambiguously(degrees, places):
            break
    else:
        raise RuntimeError("t4_radians_to_degrees_decimal/forward_basic: no unambiguous draw in 30 attempts")

    check(abs(degrees - math.degrees(R)) < 1e-9, f"forward_basic: own arithmetic disagrees with math.degrees for R={R}")

    rounded = _round_half_up(degrees, places)
    answer_text = _fmt(rounded, places)
    text = f"Convert {R:.2f} rad to degrees. Give your answer to 1 decimal place."

    inverted_factor = R * math.pi / 180
    check(abs(degrees - inverted_factor) > 0.05, f"forward_basic: inverted-factor margin too small for R={R}")
    inverted_factor_text = _fmt(_round_half_up(inverted_factor, places), places)
    _check_distractor_gap(answer_text, inverted_factor_text, places, f"forward_basic R={R} inverted_factor")

    halved_base = R * 90 / math.pi
    check(abs(degrees - halved_base) > 0.05, f"forward_basic: halved-base margin too small for R={R}")
    halved_base_text = _fmt(_round_half_up(halved_base, places), places)
    _check_distractor_gap(answer_text, halved_base_text, places, f"forward_basic R={R} halved_base")

    candidates = [inverted_factor_text, halved_base_text]
    distractors = pick_distractors(answer_text, candidates, tolerance=_tolerance(places))

    return dict(
        text=text,
        latex=text,
        answer=rounded,
        answer_text=answer_text,
        distractors=distractors,
        working=[f"degrees = {R:.2f} times 180 divided by pi = {answer_text}."],
    )


def _build_rad_to_deg_signed(rng) -> dict:
    """medium: R a signed (always negative -- see module docstring) 2dp
    magnitude in [0.10, 6.00], 2dp answer."""
    places = 2
    for _attempt in range(30):
        n = rng.randint(10, 600)
        R = -(n / 100)
        degrees = R * 180 / math.pi
        if rounds_unambiguously(degrees, places):
            break
    else:
        raise RuntimeError("t4_radians_to_degrees_decimal/forward_signed: no unambiguous draw in 30 attempts")

    check(
        abs(degrees - math.degrees(R)) < 1e-9, f"forward_signed: own arithmetic disagrees with math.degrees for R={R}"
    )

    rounded = _round_half_up(degrees, places)
    answer_text = _fmt(rounded, places)
    text = f"Convert {R:.2f} rad to degrees. Give your answer to 2 decimal places."

    dropped_sign = abs(R) * 180 / math.pi
    check(abs(dropped_sign - degrees) > 0.005, f"forward_signed: dropped-sign margin too small for R={R}")
    dropped_sign_text = _fmt(_round_half_up(dropped_sign, places), places)
    _check_distractor_gap(answer_text, dropped_sign_text, places, f"forward_signed R={R} dropped_sign")

    inverted_factor = R * math.pi / 180
    check(abs(degrees - inverted_factor) > 0.005, f"forward_signed: inverted-factor margin too small for R={R}")
    inverted_factor_text = _fmt(_round_half_up(inverted_factor, places), places)
    _check_distractor_gap(answer_text, inverted_factor_text, places, f"forward_signed R={R} inverted_factor")

    candidates = [dropped_sign_text, inverted_factor_text]
    distractors = pick_distractors(answer_text, candidates, tolerance=_tolerance(places))

    return dict(
        text=text,
        latex=text,
        answer=rounded,
        answer_text=answer_text,
        distractors=distractors,
        working=[f"degrees = {R:.2f} times 180 divided by pi = {answer_text}."],
    )


_PI_FRACTION_DENOMINATORS = (7, 11, 13, 14)


def _build_rad_to_deg_pi_fraction(rng) -> dict:
    """hard: R = k*pi/m exactly, m not a divisor of 180, 2dp answer."""
    places = 2
    for _attempt in range(30):
        m = rng.choice(_PI_FRACTION_DENOMINATORS)
        k = rng.randint(1, 3 * m)
        if math.gcd(k, m) != 1:
            continue
        deg_fraction = Fraction(k, m) * 180
        degrees = float(deg_fraction)
        if rounds_unambiguously(degrees, places):
            break
    else:
        raise RuntimeError("t4_radians_to_degrees_decimal/pi_fraction_input: no unambiguous draw in 30 attempts")

    check(
        abs(degrees - math.degrees(k * math.pi / m)) < 1e-9,
        f"pi_fraction_input: own arithmetic disagrees with math.degrees for k={k} m={m}",
    )

    rounded = _round_half_up(degrees, places)
    answer_text = _fmt(rounded, places)
    pi_str = format_pi_multiple(Fraction(k, m))
    text = f"Convert {pi_str} rad to degrees. Give your answer to 2 decimal places."

    forgot_multiply = float(Fraction(k, m))
    check(
        abs(degrees - forgot_multiply) > 0.005, f"pi_fraction_input: forgot-multiply margin too small for k={k} m={m}"
    )
    forgot_multiply_text = _fmt(_round_half_up(forgot_multiply, places), places)
    _check_distractor_gap(answer_text, forgot_multiply_text, places, f"pi_fraction_input k={k} m={m} forgot_multiply")

    halved_base = float(Fraction(k, m) * 90)
    check(abs(degrees - halved_base) > 0.005, f"pi_fraction_input: halved-base margin too small for k={k} m={m}")
    halved_base_text = _fmt(_round_half_up(halved_base, places), places)
    _check_distractor_gap(answer_text, halved_base_text, places, f"pi_fraction_input k={k} m={m} halved_base")

    candidates = [forgot_multiply_text, halved_base_text]
    distractors = pick_distractors(answer_text, candidates, tolerance=_tolerance(places))

    return dict(
        text=text,
        latex=text,
        answer=rounded,
        answer_text=answer_text,
        distractors=distractors,
        working=[f"degrees = ({k}/{m}) times 180 = {answer_text}."],
    )


@generator(
    "t4_radians_to_degrees_decimal",
    level=12,
    presets={
        "easy": dict(variant="forward_basic"),
        "medium": dict(variant="forward_signed"),
        "hard": dict(variant="pi_fraction_input"),
    },
)
def t4_radians_to_degrees_decimal(rng, variant="forward_basic"):
    """Convert radians (basic decimal, signed decimal, or exact pi-fraction
    input) to degrees, decimal answer."""
    if variant == "forward_basic":
        return _build_rad_to_deg_basic(rng)
    if variant == "forward_signed":
        return _build_rad_to_deg_signed(rng)
    if variant == "pi_fraction_input":
        return _build_rad_to_deg_pi_fraction(rng)
    raise ValueError(f"unknown variant: {variant!r}")
