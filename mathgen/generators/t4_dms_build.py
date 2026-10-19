"""Generates `t4_radians_to_dms` questions: convert a radian value to degrees and minutes, rounded to the nearest
minute.

Radians -> degrees/minutes (DMS) pair.

- ``t4_radians_to_dms`` (``angle-radians-to-dms``, domain trigonometry,
  answer: pair) -- given a radian value, convert to degrees and minutes,
  rounded to the nearest minute.

This is the reverse-direction half of "DMS both directions". This topic covers only the
reverse direction (radians -> degrees + minutes), deliberately split into its own skill
because one skill declares exactly one grader kind and ``grade_pair``'s tolerance/parsing machinery is
genuinely different from ``grade_number``'s stated-precision tolerance.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, format_pi_multiple, generator

_MAX_ATTEMPTS = 30


def _rounds_unambiguously_minutes(minutes_exact: float, margin: float = 0.05) -> bool:
    """True when ``minutes_exact`` sits far enough from a .5 boundary that
    rounding it to the nearest whole minute is determined.

    Same family as ``mathgen/generators/p5d_interval_traffic_build.py:_rounds_unambiguously`` -- duplicated
    locally. A `.5`-boundary minute is a genuine ambiguity a learner's calculator could round either way, not
    merely a cosmetic risk, because ``minutes_rounded`` is the value the learner types.
    """
    return abs((minutes_exact % 1) - 0.5) >= margin


def _degree_decimal_and_minutes(R: float) -> tuple[float, int, float]:
    degree_decimal = R * 180 / math.pi
    D = math.floor(degree_decimal)
    minutes_exact = (degree_decimal - D) * 60
    return degree_decimal, D, minutes_exact


def _finalize_carry(D: int, minutes_exact: float) -> tuple[int, int]:
    minutes_rounded = round(minutes_exact)
    if minutes_rounded == 60:
        D += 1
        minutes_rounded = 0
    check(0 <= minutes_rounded < 60, f"minutes out of range after carry: {minutes_rounded}")
    return D, minutes_rounded


def _independent_check(R: float, D: int, minutes_rounded: int) -> None:
    """Re-derive D/minutes via ``math.degrees`` -- a different code path from
    ``R * 180 / math.pi`` -- and require an EXACT match, not just a close
    one, since the whole point is catching a formatting slip in one route
    that the other route would not share."""
    degree_decimal_check = math.degrees(R)
    D_check = math.floor(degree_decimal_check)
    minutes_exact_check = (degree_decimal_check - D_check) * 60
    D_check, minutes_rounded_check = _finalize_carry(D_check, minutes_exact_check)
    check(
        D_check == D and minutes_rounded_check == minutes_rounded,
        f"independent check mismatch: constructed ({D}, {minutes_rounded}) vs "
        f"math.degrees route ({D_check}, {minutes_rounded_check})",
    )


def _format_dms(D: int, minutes: int) -> str:
    return f"{D}°{minutes:02d}′"


def _build_distractors(D: int, minutes_rounded: int, degree_decimal: float, D_orig: int) -> list[str]:
    """Component-wise integer differences. Every candidate below is checked unconditionally: since the answer
    components are always integers and ``_PAIR_TOLERANCE`` is 0.051, a difference of 1 in either component is
    already >19x the tolerance, so an exact-inequality check on the (degrees, minutes) tuple is sufficient.
    Dedupe is manual (on the tuple, before formatting) since ``pick_distractors`` is built for numeric-token
    strings and a degree/minute-symbol string is opaque to it.
    """
    answer_tuple = (D, minutes_rounded)
    accepted: list[tuple[int, int]] = []

    # dropped_minutes: forgot to convert the fractional degree to minutes.
    dropped_minutes = (D, 0)
    if dropped_minutes != answer_tuple:
        check(dropped_minutes != answer_tuple, "dropped_minutes collides with answer")
        accepted.append(dropped_minutes)

    # minutes_as_hundredths: misreads the fractional degree as hundredths. Anchored on D_orig (the
    # pre-carry floor of degree_decimal), NOT the final (possibly carried) D: degree_decimal -
    # D_orig is always in [0, 1) by construction (D_orig = floor(degree_decimal)), so wrong_minutes
    # is always in [0, 100). Anchoring on the post-carry D instead makes (degree_decimal - D)
    # slightly NEGATIVE whenever the carry has fired (D_final = D_orig + 1 > degree_decimal),
    # producing a malformed negative "minute" (e.g. "148°-1′") -- found by the test
    # suite at the seed 24/easy carry-firing draw.
    wrong_minutes = round((degree_decimal - D_orig) * 100)
    wm_degrees = D_orig
    if wrong_minutes >= 100:
        wrong_minutes -= 100
        wm_degrees += 1
    minutes_as_hundredths = (wm_degrees, wrong_minutes)
    # A "minutes" figure of 60 or more cannot be a real answer and gives the distractor away, so it is
    # not offered (the complement below takes its place).
    if wrong_minutes < 60 and minutes_as_hundredths != answer_tuple and minutes_as_hundredths not in accepted:
        check(minutes_as_hundredths != answer_tuple, "minutes_as_hundredths collides with answer")
        accepted.append(minutes_as_hundredths)

    # swapped: swaps the two components (D % 60 keeps the "minutes" slot
    # looking plausible even when D >= 60).
    swapped = (minutes_rounded, D % 60)
    if swapped != answer_tuple and swapped not in accepted:
        check(swapped != answer_tuple, "swapped collides with answer")
        accepted.append(swapped)

    # complement: measured the minutes back from the next whole degree instead of forward from this one.
    if 0 < minutes_rounded < 60:
        complement = (D, 60 - minutes_rounded)
        if complement != answer_tuple and complement not in accepted:
            accepted.append(complement)

    return [_format_dms(d, m) for d, m in accepted[:3]]


_CONTEXT_TEMPLATES = [
    "A support strut is tilted {R} rad from the vertical. Express this angle in degrees and minutes.",
    "A robotic arm rotates through {R} rad. Express this rotation in degrees and minutes.",
    "A satellite dish is angled {R} rad above the horizon. Express this angle in degrees and minutes.",
]

# Denominators for the "exam" tier's pi-fraction input. This is a
# genuinely different mathematical object from the existing `hard`
# (`fraction_input`) tier's `k/m` draw: `fraction_input` produces a bare
# rational number of radians (e.g. "9/4 rad" literally means 2.25 rad, with
# no pi symbol anywhere in the question), while this tier produces a true
# multiple of pi (e.g. "5pi/13 rad" means (5/13)*pi rad).
#
# Deliberately NOT the classic "special angle" denominators {2,3,4,6,12}:
# every one of those divides 180 exactly, so k*180/m is ALWAYS a whole
# number of degrees for any k -- the DMS rounding-to-minutes machinery this
# skill exists to teach (`_finalize_carry`, `_rounds_unambiguously_minutes`,
# the `dropped_minutes`/`minutes_as_hundredths` distractors) never fires,
# because "minutes" is always 0. (Denominator 8 is only marginally better:
# 180/8 = 22.5, so k odd gives exactly x.5 degrees every time -- a fixed 30
# minutes, not a varied one.) Verified empirically while designing this
# generator: seeds 0-19 against {3,4,6,8,12} produced `minutes_rounded == 0` on
# 16/20 draws and only a single valid distractor (`swapped`) on those, since
# `dropped_minutes`/`minutes_as_hundredths` both collapse onto the answer
# when minutes is 0.
#
# None of the denominators below divides 180 (180 = 2^2*3^2*5), so
# `k*180/m` is never a whole number for any `k` coprime to `m` -- every draw
# has a genuine, varied fractional-degree part to convert and round, and the
# family is also fully disjoint from `fraction_input`'s {2..9} pool.
_EXAM_PI_DENOMINATORS = [11, 13, 14, 16, 17, 19]


@generator(
    "t4_radians_to_dms",
    level=12,
    presets={
        "easy": dict(variant="basic"),
        "medium": dict(variant="context"),
        "hard": dict(variant="fraction_input"),
        "exam": dict(variant="exam_pi_fraction"),
    },
)
def t4_radians_to_dms(rng, variant="basic"):
    if variant == "basic":
        R = None
        for _attempt in range(_MAX_ATTEMPTS):
            candidate_R = rng.randint(50, 3000) / 1000
            degree_decimal, D, minutes_exact = _degree_decimal_and_minutes(candidate_R)
            if _rounds_unambiguously_minutes(minutes_exact):
                R = candidate_R
                break
        else:
            raise RuntimeError(
                f"t4_radians_to_dms/basic: no draw avoided the .5-minute rounding boundary in {_MAX_ATTEMPTS} attempts"
            )
        D_orig = D
        D, minutes_rounded = _finalize_carry(D, minutes_exact)
        question_text = f"Convert {R:.3f} rad to degrees and minutes. Give your answer to the nearest minute."

    elif variant == "context":
        R = None
        for _attempt in range(_MAX_ATTEMPTS):
            candidate_R = rng.randint(50, 6200) / 1000
            degree_decimal, D, minutes_exact = _degree_decimal_and_minutes(candidate_R)
            if _rounds_unambiguously_minutes(minutes_exact):
                R = candidate_R
                break
        else:
            raise RuntimeError(
                "t4_radians_to_dms/context: no draw avoided the .5-minute "
                f"rounding boundary in {_MAX_ATTEMPTS} attempts"
            )
        D_orig = D
        D, minutes_rounded = _finalize_carry(D, minutes_exact)
        template = rng.choice(_CONTEXT_TEMPLATES)
        question_text = template.format(R=f"{R:.3f}")

    elif variant == "fraction_input":
        R = None
        k = m = None
        for _attempt in range(_MAX_ATTEMPTS):
            candidate_m = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
            k_max = min(6 * candidate_m, math.floor(2 * math.pi * candidate_m))
            candidate_k = rng.randint(1, k_max)
            if math.gcd(candidate_k, candidate_m) != 1:
                continue
            candidate_R = candidate_k / candidate_m
            degree_decimal, D, minutes_exact = _degree_decimal_and_minutes(candidate_R)
            if _rounds_unambiguously_minutes(minutes_exact):
                R, k, m = candidate_R, candidate_k, candidate_m
                break
        else:
            raise RuntimeError(
                "t4_radians_to_dms/fraction_input: no draw gave a lowest-terms "
                f"fraction that avoided the .5-minute rounding boundary in "
                f"{_MAX_ATTEMPTS} attempts"
            )
        D_orig = D
        D, minutes_rounded = _finalize_carry(D, minutes_exact)
        question_text = f"Convert {k}/{m} rad to degrees and minutes. Give your answer to the nearest minute."

    elif variant == "exam_pi_fraction":
        # Genuinely different input shape from `fraction_input`:
        # that tier's `k/m` is a bare rational number of radians (no pi
        # anywhere in the question); this tier's `R` is a true multiple of
        # pi (e.g. "5pi/6 rad"), the standard way an exact radian angle is
        # stated in senior/tertiary trigonometry, drawn from the classic
        # unit-circle denominator family and wrapped in one of the existing
        # `_CONTEXT_TEMPLATES` real-world sentences (reused unmodified --
        # `context` (medium) draws from the same list via `rng.choice`, so
        # appending a new entry would shift its own choice of template at
        # unrelated seeds and break the render-baseline replay; calling
        # `rng.choice` on the SAME list here, without changing its contents,
        # is what "reuse" safely means).
        R = None
        k = m = None
        for _attempt in range(_MAX_ATTEMPTS):
            candidate_m = rng.choice(_EXAM_PI_DENOMINATORS)
            k_max = 2 * candidate_m - 1  # keeps R < 2*pi (a full turn)
            candidate_k = rng.randint(1, k_max)
            if math.gcd(candidate_k, candidate_m) != 1:
                continue
            candidate_R = candidate_k * math.pi / candidate_m
            degree_decimal, D, minutes_exact = _degree_decimal_and_minutes(candidate_R)
            if _rounds_unambiguously_minutes(minutes_exact):
                R, k, m = candidate_R, candidate_k, candidate_m
                break
        else:
            raise RuntimeError(
                "t4_radians_to_dms/exam_pi_fraction: no draw gave a lowest-terms "
                f"pi-fraction that avoided the .5-minute rounding boundary in "
                f"{_MAX_ATTEMPTS} attempts"
            )
        D_orig = D
        D, minutes_rounded = _finalize_carry(D, minutes_exact)
        pi_fraction_label = format_pi_multiple(Fraction(k, m))
        template = rng.choice(_CONTEXT_TEMPLATES)
        question_text = template.format(R=pi_fraction_label)

    else:
        raise ValueError(f"unknown variant {variant!r}")

    _independent_check(R, D, minutes_rounded)

    answer = {"degrees": D, "minutes": minutes_rounded}
    answer_text = _format_dms(D, minutes_rounded)
    distractors = _build_distractors(D, minutes_rounded, degree_decimal, D_orig)

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
    )
