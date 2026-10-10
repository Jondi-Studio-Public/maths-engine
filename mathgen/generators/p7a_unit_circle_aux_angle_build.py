"""Unit-circle tan evaluation and the auxiliary-angle (R cos/sin) conversion.

- ``unit_circle_tan_evaluate``
  (``senior-understand-the-unit-circle-definition-of-cos-theta-acmmm034``,
  ACMMM034, domain trigonometry, answer: mcq) -- evaluate tan(theta) =
  sin(theta)/cos(theta) from a stated terminal point on the unit circle, or
  state that it is undefined where cos(theta) = 0.
- ``aux_angle_r_cos_conversion``
  (``senior-convert-sums-mathrm-a-cos-mathrm-x-mathrm``, ACMSM048, domain
  trigonometry, answer: pair) -- convert a cos(x) + b sin(x) to R cos(x -
  alpha) or R sin(x + alpha), given integer a and b drawn from a
  Pythagorean-triple pool so R = sqrt(a^2 + b^2) is always an integer.

Both topics share one pool of primitive Pythagorean triples (Euclid's
formula, hypotenuse <= 100) so every coordinate/leg pair is fully specified
in the question text -- no memorised unit-circle table is needed.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import Rational

from ..core import (
    GRADER_TOLERANCE,
    check,
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    paren_negative,
    pick_distractors,
)

# Primitive Pythagorean triples (leg, leg, hyp), hyp <= 100, generated once via
# Euclid's formula (m, n coprime, opposite parity, m > n > 0). Bigger than the
# 5-triple table in p5b_trig_ratio_build.py on purpose -- reusing that exact
# table would correlate this skill's distractor pool with p5b's for no reason, and 16 triples x 2 leg-orientations gives a genuine
# >=20-distinct-question pool at every tier.
TRIPLES: tuple[tuple[int, int, int], ...] = (
    (3, 4, 5),
    (5, 12, 13),
    (7, 24, 25),
    (8, 15, 17),
    (9, 40, 41),
    (11, 60, 61),
    (12, 35, 37),
    (13, 84, 85),
    (16, 63, 65),
    (20, 21, 29),
    (28, 45, 53),
    (33, 56, 65),
    (36, 77, 85),
    (39, 80, 89),
    (48, 55, 73),
    (65, 72, 97),
)


# ---------------------------------------------------------------------------
# Skill A: unit_circle_tan_evaluate
# ---------------------------------------------------------------------------

# (sign_x, sign_y) for each quadrant, matching the unit-circle convention.
QUADRANT_SIGNS = {1: (1, 1), 2: (-1, 1), 3: (-1, -1), 4: (1, -1)}

# The four quadrantal terminal points: (angle_degrees, numerator_x,
# numerator_y, hyp=1). x = 0 -> tan undefined; y = 0 -> tan = 0 (defined).
QUADRANTAL_POINTS = ((0, 1, 0, 1), (90, 0, 1, 1), (180, -1, 0, 1), (270, 0, -1, 1))

_INSTRUCTION = (
    'Using tan(theta) = sin(theta)/cos(theta), evaluate tan(theta), or select "undefined" if it does not exist.'
)


def _frac_str(numerator: int, denominator: int) -> str:
    return str(Fraction(numerator, denominator))


def _rat(fr: Fraction):
    return Rational(fr.numerator, fr.denominator)


def _draw_signed_point(rng, quadrant: int) -> tuple[int, int, int]:
    """A (numerator_x, numerator_y, hyp) point in the given quadrant, drawn
    from the Pythagorean-triple pool with a random leg orientation."""
    leg1, leg2, hyp = rng.choice(TRIPLES)
    x_leg, y_leg = (leg1, leg2) if rng.random() < 0.5 else (leg2, leg1)
    sx, sy = QUADRANT_SIGNS[quadrant]
    return sx * x_leg, sy * y_leg, hyp


def _defined_question(rng, quadrant: int, easy: bool) -> dict:
    nx, ny, hyp = _draw_signed_point(rng, quadrant)

    # Independent check: the point is genuinely on the unit circle (a
    # different route from "just divide the two coordinates"), and the
    # sin/cos ratio simplification that produces tan is itself verified.
    check(nx * nx + ny * ny == hyp * hyp, "point is not on the unit circle")
    tan_value = Fraction(ny, nx)
    check_equivalent(
        _rat(tan_value),
        _rat(Fraction(ny, hyp)) / _rat(Fraction(nx, hyp)),
        "tan(theta) != sin(theta)/cos(theta)",
    )
    answer_text = str(tan_value)

    point_str = f"({_frac_str(nx, hyp)}, {_frac_str(ny, hyp)})"
    text = f"Theta's terminal point on the unit circle is {point_str}. {_INSTRUCTION}"

    swap = Fraction(nx, ny)  # treats x as sin and y as cos
    y_alone = Fraction(ny, hyp)  # forgot to divide by cos(theta)
    sign_flip = Fraction(-ny, nx)  # correct magnitude, wrong sign
    # swap/y_alone/sign_flip are always distinct from tan_value by construction
    # (proved once, not per-question): swap needs nx != ny (true legs are
    # always distinct integers); y_alone needs hyp != nx (true, hyp > nx
    # strictly); sign_flip needs tan_value != 0 (true, neither coordinate is
    # zero in this branch). check_distinct proves it anyway, every question.
    check_distinct(_rat(tan_value), _rat(swap), _rat(y_alone), _rat(sign_flip))
    candidates = [str(swap), str(y_alone)]
    if easy:
        candidates.append(str(sign_flip))
    else:
        # sign dropped (abs value) -- the new medium/hard failure mode. This
        # one is NOT always distinct from the answer (it equals tan_value
        # exactly whenever tan is already positive, e.g. quadrant III), so it
        # is deliberately left out of check_distinct above and instead relies
        # on pick_distractors to drop it when it collides, falling back to
        # sign_flip (appended next) to keep exactly 3 distractors either way.
        candidates.append(str(Fraction(abs(ny), abs(nx))))
        candidates.append(str(sign_flip))

    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "unit_circle_tan_evaluate: fewer than 3 distractors survived")

    working = [
        f"cos(theta) = {_frac_str(nx, hyp)}, sin(theta) = {_frac_str(ny, hyp)}",
        f"tan(theta) = sin(theta)/cos(theta) = {answer_text}",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _quadrantal_question(rng) -> dict:
    angle_deg, nx, ny, hyp = rng.choice(QUADRANTAL_POINTS)
    point_str = f"({nx}, {ny})"
    text = f"Theta = {angle_deg} degrees, with terminal point {point_str} on the unit circle. {_INSTRUCTION}"

    if nx == 0:
        # Classification outcome: no symbolic equivalence to prove -- the
        # in-generator check is simply the gate.
        check(nx == 0, "quadrantal question: cos(theta) is not zero")
        answer_text = "undefined"
    else:
        check(nx != 0, "quadrantal question: cos(theta) is zero")
        tan_value = Fraction(ny, nx)
        check_equivalent(_rat(tan_value), _rat(Fraction(0, 1)), "tan(theta) != 0 at this quadrantal point")
        answer_text = "0"

    pool = ["0", "undefined", "1", "-1"]
    candidates = [c for c in pool if c != answer_text]
    numeric_candidates = [Fraction(c) for c in candidates if c != "undefined"]
    if answer_text != "undefined":
        check_distinct(_rat(Fraction(answer_text)), *[_rat(c) for c in numeric_candidates])
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "unit_circle_tan_evaluate: fewer than 3 distractors survived (quadrantal)")

    if nx == 0:
        working = [f"cos(theta) = {nx}, so tan(theta) = sin(theta)/cos(theta) is undefined."]
    else:
        working = [f"sin(theta) = {ny}, cos(theta) = {nx}, so tan(theta) = {ny}/{nx} = 0."]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "unit_circle_tan_evaluate",
    level=13,
    presets={
        "easy": dict(variant="quadrant_one"),
        "medium": dict(variant="signed_quadrant"),
        "hard": dict(variant="mixed_with_quadrantal"),
    },
)
def unit_circle_tan_evaluate(rng, variant="quadrant_one"):
    """Evaluate tan(theta) = sin(theta)/cos(theta) from a stated terminal
    point on the unit circle, or classify it as undefined. ACMMM034's whole
    reason to exist over its sibling ACMMM029 (cos/sin only, not built) is
    the tan-specific content: the ratio itself, the sign by quadrant, and the
    undefined-where-cos-is-zero edge case."""
    if variant == "quadrant_one":
        return _defined_question(rng, quadrant=1, easy=True)
    if variant == "signed_quadrant":
        quadrant = rng.choice([2, 3, 4])
        return _defined_question(rng, quadrant=quadrant, easy=False)
    if variant == "mixed_with_quadrantal":
        # ~20% quadrantal draw keeps the undefined/zero edge case a real,
        # reliably-hit slice of the hard tier (at least ~10%).
        if rng.random() < 0.2:
            return _quadrantal_question(rng)
        quadrant = rng.choice([2, 3, 4])
        return _defined_question(rng, quadrant=quadrant, easy=False)
    raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Skill B: aux_angle_r_cos_conversion
# ---------------------------------------------------------------------------

_ALPHA_INTERVAL = "using -180 < alpha <= 180"


def _fmt_pair(r_value: int, alpha_value: float) -> str:
    return f"R = {r_value}, alpha = {alpha_value:.1f}"


def _draw_ab(rng, sign_patterns: list[tuple[int, int]]) -> tuple[int, int, int]:
    leg1, leg2, hyp = rng.choice(TRIPLES)
    leg_a, leg_b = (leg1, leg2) if rng.random() < 0.5 else (leg2, leg1)
    sa, sb = rng.choice(sign_patterns)
    return sa * leg_a, sb * leg_b, hyp


def _aux_angle_question(rng, sign_patterns: list[tuple[int, int]], form: str) -> dict:
    a, b, r_value = _draw_ab(rng, sign_patterns)

    if form == "cos":
        alpha = math.degrees(math.atan2(b, a))
        naive_alpha = math.degrees(math.atan(b / a))  # no quadrant correction
        swap_alpha = math.degrees(math.atan2(a, b))  # sin-form formula, wrong form
        form_text = "R cos(x - alpha)"
    else:
        alpha = math.degrees(math.atan2(a, b))
        naive_alpha = math.degrees(math.atan(a / b))
        swap_alpha = math.degrees(math.atan2(b, a))  # cos-form formula, wrong form
        form_text = "R sin(x + alpha)"

    # Independent check: a genuinely different route from the atan2 formula --
    # confirm R*cos(alpha)/R*sin(alpha) (or the sin-form equivalents)
    # reconstruct a and b directly, a numeric rounded-real check per the
    # protocol's "deliberately not checked" carve-out (not a symbolic
    # shortcut). Checked against the UNROUNDED alpha -- rounding to 1 d.p.
    # can shift R*cos(alpha) by up to R*(0.05 degrees in radians), which
    # exceeds a tight tolerance for the larger R values in this pool (e.g.
    # R=97), so this is a check on the construction formula, not on the
    # display rounding.
    alpha_rad = math.radians(alpha)
    if form == "cos":
        check(abs(r_value * math.cos(alpha_rad) - a) < 1e-6, "R*cos(alpha) does not reconstruct a")
        check(abs(r_value * math.sin(alpha_rad) - b) < 1e-6, "R*sin(alpha) does not reconstruct b")
    else:
        check(abs(r_value * math.sin(alpha_rad) - a) < 1e-6, "R*sin(alpha) does not reconstruct a")
        check(abs(r_value * math.cos(alpha_rad) - b) < 1e-6, "R*cos(alpha) does not reconstruct b")

    alpha = round(alpha, 1) + 0.0  # +0.0 collapses a "-0.0" rounding to "0.0"
    answer_text = _fmt_pair(r_value, alpha)
    answer = {"R": r_value, "alpha": alpha}

    text = (
        f"Convert a cos(x) + b sin(x) to the form {form_text}, for a = {a} and b = {b}. "
        f"Give R as an integer and alpha in degrees to 1 d.p., {_ALPHA_INTERVAL}."
    )

    r_sum_wrong = abs(a + b)
    r_diff_wrong = abs(abs(a) - abs(b))
    raw_candidates = [
        (r_value, naive_alpha),  # no quadrant correction (the central trap)
        (r_sum_wrong, alpha),  # R computed as a + b
        (r_diff_wrong, alpha),  # R computed as |a| - |b|
        (r_value, swap_alpha),  # a and b swapped in the alpha formula
        (r_value, -alpha),  # sign-flipped alpha
    ]
    candidate_strings = [_fmt_pair(r, round(al, 1) + 0.0) for r, al in raw_candidates]
    distractors = pick_distractors(answer_text, candidate_strings, n=3, tolerance=GRADER_TOLERANCE)
    check(len(distractors) == 3, "aux_angle_r_cos_conversion: fewer than 3 distractors survived")

    working = [
        f"R = sqrt({paren_negative(a)}^2 + {paren_negative(b)}^2) = {r_value}",
        f"alpha = {'atan2(b, a)' if form == 'cos' else 'atan2(a, b)'} = {answer_text.split(', ')[1]}",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


@generator(
    "aux_angle_r_cos_conversion",
    level=14,
    presets={
        "easy": dict(variant="cos_form_q1"),
        "medium": dict(variant="cos_form_mixed_sign"),
        "hard": dict(variant="sin_form_mixed_sign"),
    },
)
def aux_angle_r_cos_conversion(rng, variant="cos_form_q1"):
    """Convert a cos(x) + b sin(x) to R cos(x - alpha) (or R sin(x + alpha)),
    given integer a, b (not both zero) drawn from a Pythagorean-triple pool
    so R = sqrt(a^2 + b^2) is always an integer. alpha is stated to 1 decimal
    place on a fixed interval (-180 < alpha <= 180), since it is otherwise
    ambiguous which representative angle is wanted."""
    if variant == "cos_form_q1":
        return _aux_angle_question(rng, [(1, 1)], form="cos")
    if variant == "cos_form_mixed_sign":
        return _aux_angle_question(rng, [(-1, 1), (1, -1)], form="cos")
    if variant == "sin_form_mixed_sign":
        return _aux_angle_question(rng, [(-1, 1), (1, -1), (-1, -1)], form="sin")
    raise ValueError(f"unknown variant: {variant}")
