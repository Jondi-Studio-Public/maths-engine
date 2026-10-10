"""Degree <-> radian conversion in exact pi form.

Four difficulty tiers on one topic, `t4_degrees_radians_pi_form`:
- easy   (`variant="forward_common"`): degrees -> exact pi-form radians, degree
  drawn from the pool whose reduced denominator is in {1,2,3,4,6}.
- medium (`variant="forward_fine"`): same direction, degree drawn from the
  pool whose reduced denominator is 12 (including the +360 coterminal half).
- hard   (`variant="reverse"`): exact pi-form radians -> plain integer
  degrees, degree drawn from the union of both pools above.
- exam   (`variant="forward_fine_exam"`): same direction as easy/medium, but
  degree is a half-integer (37.5, 38.5, ...) whose reduced denominator is
  24, 40 or 72 (finer than medium's fixed 12), framed as a surveying
  word problem so its `prompt_shape()` is structurally distinct from
  easy/medium's shared sentence, not just a decimal swapped into the same
  words (see `FINE_DEGREES_EXAM`'s docstring comment
  for the masking proof).

ASCII "pi" display ("2pi/3", "pi", "pi/6") via `core.format_pi_multiple`, plus
a digit-adjacent-to-pi star-insertion step for the grader-facing string
("2pi/3" -> "2*pi/3").
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    format_pi_multiple,
    generator,
    latex_pi_multiple,
    phrase,
    pick_distractors,
)

# Denominator in {1,2,3,4,6} after reduction, degrees in (0, 360].
COMMON_DEGREES: tuple[int, ...] = (
    30,
    45,
    60,
    90,
    120,
    135,
    150,
    180,
    210,
    225,
    240,
    270,
    300,
    315,
    330,
    360,
)
# Denominator 12 after reduction, degrees in (0, 360], plus their +360 coterminal partners. The
# coterminal half is required: without it this pool is only 8 distinct question texts.
_FINE_BASE: tuple[int, ...] = (15, 75, 105, 165, 195, 255, 285, 345)
FINE_DEGREES: tuple[int, ...] = _FINE_BASE + tuple(d + 360 for d in _FINE_BASE)
# The union of both pools, for the reverse (hard) direction.
REVERSE_DEGREES: tuple[int, ...] = COMMON_DEGREES + FINE_DEGREES

# Exam tier: half-integer degrees (37.5, 38.5...) in (0, 360]. A half-integer degree's doubled value
# (2*deg) is always odd, so `Fraction(2*deg, 360)`'s denominator can only ever be 360 divided by a
# divisor of 360's odd part (45 = 3^2*5) -- the reachable denominators are exactly {8, 24, 40, 72,
# 120, 360}, never anything in between. Of those six: 8 is COARSER than the existing forward_fine
# (medium) tier's fixed denominator 12 and is excluded (an exam tier must be finer, not sometimes
# coarser); 24, 40 and 72 are the pool the "denominators up to 72" language describes and
# are taken in full (96 values); 120 and 360 are finer still (360 alone has 192 members) and are
# represented by a single explicit addition, 38.5 (-> 77pi/360),
# rather than the whole bin, so the pool stays a deliberately curated set and not every value that
# happens to clear a filter.
FINE_DEGREES_EXAM: tuple[float, ...] = tuple(
    sorted({half / 2 for half in range(1, 720, 2) if Fraction(half, 360).denominator in (24, 40, 72)} | {38.5})
)


def _grader_pi_string(display: str) -> str:
    """Insert the '*' the grader needs between a digit and 'pi' ("2pi/3" ->
    "2*pi/3"); a string with no digit directly before "pi" (bare "pi",
    "pi/6") is already sympifiable and returned unchanged. One uniform
    substitution, no denominator-based branching -- deg=360 formats to
    "2pi" (denominator 1) and still needs the same fix as "2pi/3"
    (denominator 3), so denominator alone does not decide whether a fix is
    needed."""
    if "pi" not in display:
        return display
    return re.sub(r"(\d)pi", r"\1*pi", display)


@generator(
    "t4_degrees_radians_pi_form",
    level=12,
    presets={
        "easy": dict(variant="forward_common"),
        "medium": dict(variant="forward_fine"),
        "hard": dict(variant="reverse"),
        "exam": dict(variant="forward_fine_exam"),
    },
)
def t4_degrees_radians_pi_form(rng, variant="forward_common"):
    """Convert between a degree measure and its exact radian value in terms
    of pi."""

    if variant in ("forward_common", "forward_fine"):
        pool = COMMON_DEGREES if variant == "forward_common" else FINE_DEGREES
        deg = rng.choice(pool)
        coeff = Fraction(deg, 180)
        answer_expr = coeff * sympy.pi
        answer_text = format_pi_multiple(coeff)
        answer = _grader_pi_string(answer_text)
        answer_latex = latex_pi_multiple(coeff)

        text = f"Convert {deg} degrees to radians. Give your answer as an exact value in terms of pi."

        # Independent check: parse the actual grader-facing string back into
        # a sympy expression and confirm it reduces to deg/180 * pi computed
        # fresh from `deg` -- a route that exercises `_format_pi_multiple`'s
        # own formatting, which reusing `answer_expr` (built from the same
        # `coeff` the display was formatted from) could not catch.
        check_equivalent(
            sympy.sympify(answer),
            sympy.Rational(deg, 180) * sympy.pi,
            "displayed pi-form does not reduce back to deg/180 * pi",
        )

        used_90 = Fraction(deg, 90) * sympy.pi  # halved the base: used 90, not 180
        dropped_pi = sympy.Rational(coeff.numerator, coeff.denominator)  # forgets the pi factor
        used_deg_as_coeff = deg * sympy.pi  # used the raw degree as the pi-coefficient
        check_distinct(
            answer_expr,
            used_90,
            dropped_pi,
            used_deg_as_coeff,
            msg="a forward-direction distractor equals the answer value",
        )

        distractor_candidates = [
            format_pi_multiple(Fraction(deg, 90)),
            str(sympy.Rational(coeff.numerator, coeff.denominator)),
            format_pi_multiple(Fraction(deg, 1)),
        ]
        distractors = pick_distractors(answer_text, distractor_candidates)

    elif variant == "forward_fine_exam":
        deg = rng.choice(FINE_DEGREES_EXAM)
        coeff = Fraction(deg) / 180
        answer_expr = coeff * sympy.pi
        answer_text = format_pi_multiple(coeff)
        answer = _grader_pi_string(answer_text)
        answer_latex = latex_pi_multiple(coeff)

        # Three word-problem framings via `phrase`, not one fixed sentence: `prompt_shape` masks
        # every digit run (including the decimal point) to one token, so "Convert 37 degrees..." and
        # "Convert 37.5 degrees..." mask IDENTICALLY -- reusing forward_common/ forward_fine's
        # sentence here would make `measure_tiers.py` see the same shape as a lower tier and call
        # this range-only even though the reduction (denominator up to 72+, instead of <=12) is a
        # real escalation.
        text = phrase(
            rng,
            [
                "A surveyor measures an angle of {deg} degrees, correct to "
                "the nearest half degree. Convert this angle to its exact "
                "radian value in terms of pi.",
                "An angle of {deg} degrees is recorded to the nearest half "
                "degree on a site plan. Express this angle in radians as "
                "an exact multiple of pi.",
                "A technician reads an angle of {deg} degrees off a "
                "protractor, accurate to the nearest half degree. Write "
                "this angle in radians as an exact multiple of pi.",
            ],
            deg=deg,
        )

        # Independent check: parse the actual grader-facing string back into
        # a sympy expression and confirm it reduces to deg/180 * pi computed
        # fresh from `deg` -- same discipline as forward_common/forward_fine.
        check_equivalent(
            sympy.sympify(answer),
            sympy.Rational(deg, 180) * sympy.pi,
            "displayed pi-form does not reduce back to deg/180 * pi",
        )

        # halved the base: used 90, not 180.
        used_90 = (Fraction(deg) / 90) * sympy.pi
        # forgets the pi factor.
        dropped_pi = sympy.Rational(coeff.numerator, coeff.denominator)
        # used the raw degree as the pi-coefficient. `sympy.Rational(deg)`, not `deg * sympy.pi`, to
        # keep this an exact rational multiple of pi (deg is a float here) rather than an
        # approximate `Float` -- pi stays exact in every intermediate, distractors
        # included.
        used_deg_as_coeff = sympy.Rational(deg) * sympy.pi
        check_distinct(
            answer_expr,
            used_90,
            dropped_pi,
            used_deg_as_coeff,
            msg="a forward-direction distractor equals the answer value",
        )

        distractor_candidates = [
            format_pi_multiple(Fraction(deg) / 90),
            str(sympy.Rational(coeff.numerator, coeff.denominator)),
            format_pi_multiple(Fraction(deg)),
        ]
        distractors = pick_distractors(answer_text, distractor_candidates)

    elif variant == "reverse":
        deg = rng.choice(REVERSE_DEGREES)
        coeff = Fraction(deg, 180)
        pi_str = format_pi_multiple(coeff)
        answer_text = str(deg)
        answer = answer_text
        answer_latex = answer_text

        text = f"The angle {pi_str} rad is equivalent to how many degrees?"

        # Independent check: parse the pi-form STRING the question displays
        # back into a coefficient (dividing the sympified grader string by
        # pi) and multiply by 180 -- this exercises `_format_pi_multiple`'s
        # own round trip, which a fresh `coeff * 180` (reusing the same
        # `coeff` the string was formatted from) cannot see if the formatter
        # itself has a bug.
        recovered_coeff = sympy.sympify(_grader_pi_string(pi_str)) / sympy.pi
        check_equivalent(
            recovered_coeff * 180, deg, "reverse-direction answer does not match the parsed pi-form question"
        )

        # forgot_multiply: reports the bare numerator of the pi-coefficient as
        # if it were already the degree answer. coeff.numerator can only equal
        # deg when deg == 0 (impossible here, deg > 0 for every pool value), so
        # no denominator-based guard is needed; pick_distractors below drops any
        # candidate that coincides with the answer regardless.
        forgot_multiply = coeff.numerator
        # halved: used 90 instead of 180 as the base, reversed -- half the
        # correct answer.
        halved = deg // 2 if deg % 2 == 0 else (deg - 1) // 2
        # flipped_km: flips numerator and denominator of the pi-coefficient
        # before multiplying by 180. coeff.numerator >= 1 for every value in
        # scope (deg > 0), so this is always defined.
        flipped_km = int(Fraction(coeff.denominator, coeff.numerator) * 180)

        distractor_candidates = [str(forgot_multiply), str(halved), str(flipped_km)]
        distractors = pick_distractors(answer_text, distractor_candidates)
        check_distinct(
            sympy.Integer(deg),
            *(sympy.Integer(int(d)) for d in distractors),
            msg="a reverse-direction distractor equals the answer value",
        )

    else:
        raise ValueError(f"unknown variant: {variant}")

    # Every learner-facing field is ASCII "pi", no bare '*' -- the grader-facing
    # `answer` field is the only place a '*' is allowed to appear.
    check("*" not in text, "bare * found in text")
    check("*" not in answer_text, "bare * found in answer_text")
    for distractor in distractors:
        check("*" not in distractor, f"bare * found in distractor: {distractor}")
    check("π" not in text and "π" not in answer_text, "unicode pi found where ASCII 'pi' is required")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
        distractors=distractors,
    )
