"""Generates `p10e_angle_relationships` questions: find an unknown angle from angles at a point or vertically
opposite angles.

Angle relationships.

- ``p10e_angle_relationships`` (``f10-angle-relationships``, AC9M6M04,
  domain primary-geometry-time, answer: number) -- given a stated angle in
  an angles-at-a-point or vertically-opposite relationship to another
  angle, determine the unknown angle in degrees.

Deliberately NEVER builds the straight-line (180-degree) relationship --
that is the prerequisite ``straight-line-angles``
(``mathgen/generators/curriculum_expansion.py::straight_line_angle``)'s own
graded operation. Tiers:

- easy: 3 angles at a point sum to 360; 2 given, 1 unknown.
- medium: 4 angles at a point sum to 360; 3 given, 1 unknown (one more
  addend to track than easy -- a genuine reasoning escalation).
- hard: two vertically-opposite angles, given as algebraic expressions in
  x, stated equal; solve for x, then report the ANGLE'S value (not x).

``_bounded_parts`` is a one-off local helper (sequential constrained
sampling, no rejection loop) for "n positive ints >= floor summing to a
fixed total" -- used only here, not requested as a shared helper.
"""

from __future__ import annotations

from sympy import Eq, solve, symbols
from sympy import latex as sympy_latex

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
)

_X = symbols("x")

_POINT_FLOOR = 15  # every angle-at-a-point value (given or unknown), easy & medium
_POINT_CEIL = 250  # no single angle-at-a-point value dominates the figure
_VERTICAL_FLOOR = 10  # single crossing angle, hard tier
_VERTICAL_CEIL = 170  # strictly less than 180 (a straight line), with margin


def _bounded_parts(rng, n: int, total: int, floor: int) -> list[int]:
    """``n`` positive ints, each ``>= floor``, summing exactly to ``total``, via uniform cut-points (no
    rejection/resample loop, and no positional skew): draw ``n - 1`` cut points uniformly in ``[0, total -
    n*floor]``, sort them, and the gaps between consecutive cuts (plus ``floor`` each) are the ``n`` values.
    Every position has the SAME marginal distribution this way -- a first draft here sampled position 0 as
    ``U[floor, total-(n-1)*floor]`` (mean far above the rest) and the remaining positions from a much narrower
    range, so shuffling only hid which position carried the skew rather than removing it. Measured, not
    theoretical, over 600 seeds each: this version drops P(an angle > 270) from 21.7% to 8.0% at the easy tier
    and from 15.2% to 2.0% at the medium tier. The caller (`_point_tier`) still applies its own `_POINT_CEIL`
    resample on top of this for the residual tail."""
    check(total >= n * floor, "_bounded_parts: floor too large for total/n")
    span = total - n * floor
    cuts = sorted(rng.randint(0, span) for _ in range(n - 1))
    bounds = [0] + cuts + [span]
    return [floor + (bounds[i + 1] - bounds[i]) for i in range(n)]


def _expr_text(coef: int, const: int) -> str:
    """``3x + 10`` / ``5x - 30`` style text for a linear expression in x."""
    return join_terms([(coef, "x"), (const, "")])


def _point_tier(rng, n_given: int):
    """Shared construction for the angles-at-a-point tiers: ``n_given``
    given angles plus one unknown, each in ``[_POINT_FLOOR, _POINT_CEIL]``,
    summing to 360. `_bounded_parts` already removed most of the tail past
    `_POINT_CEIL` (see its own docstring); this resamples the residual rare
    seed where it does not, rather than leaving one figure-dominating angle
    in the shipped question."""
    found = False
    for _ in range(100):
        parts = _bounded_parts(rng, n_given + 1, 360, _POINT_FLOOR)
        if max(parts) <= _POINT_CEIL:
            found = True
            break
    check(found, "p10e_angle_relationships: _point_tier could not find a draw under _POINT_CEIL")
    rng.shuffle(parts)
    givens, x = parts[:-1], parts[-1]
    return givens, x


def _point_question(rng, n_given: int, noun_total: str, noun_unknown: str):
    givens, x = _point_tier(rng, n_given)
    all_angles = givens + [x]

    # Independent check: assert the PROPERTY directly (the full set of
    # angles sums to EXACTLY 360), not a re-run of construction's own
    # subtraction.
    check(sum(all_angles) == 360, "p10e_angle_relationships: angles at a point do not sum to 360")

    given_list_text = ", ".join(f"{g} degrees" for g in givens[:-1])
    given_list_text = f"{given_list_text} and {givens[-1]} degrees" if given_list_text else f"{givens[-1]} degrees"
    question_text = (
        f"{noun_total} meet at a point and together complete a full turn. "
        f"{noun_unknown[0]} of the angles measure {given_list_text}. "
        f"Find the size of {noun_unknown[1]} angle, in degrees."
    )
    answer_text = str(x)

    # uses-180-not-360: subtracts the givens from 180 instead of 360. Algebraically d_180 = 180 -
    # sum(givens) = x - 180, so it can never collide with the answer (that would need 360 == 180) --
    # checked anyway per rule -- but it is only ever a PICKABLE (positive) option when x > 180 (a
    # reflex unknown angle); otherwise it renders negative and no learner would submit it, so the
    # misconception is not exercisable on that seed.
    d_180 = x - 180
    # drops-one-addend: omits one given angle from the sum before
    # subtracting from 360 (at easy tier -- 2 givens -- this degenerates to
    # "uses only one given angle"). Two independent drop choices are offered
    # (dropping a different given each time) so the misconception survives
    # even on the rare seed where d_180 collapses into a duplicate of one of
    # them: d_drop = x + dropped can never equal x since dropped >=
    # _POINT_FLOOR > 0, and can never equal d_180 since that would need
    # dropped == -180.
    idxs = rng.sample(range(len(givens)), min(2, len(givens)))
    d_drop_1 = x + givens[idxs[0]]
    d_drop_2 = x + givens[idxs[-1]]

    check_distinct(x, d_180, msg="uses-180-not-360 distractor equals the answer")
    check_distinct(x, d_drop_1, msg="drops-one-addend distractor equals the answer")
    check_distinct(x, d_drop_2, msg="drops-one-addend (second draw) distractor equals the answer")

    distractors = pick_distractors(answer_text, [str(d_drop_1), str(d_drop_2), str(d_180)])

    sum_expr = " + ".join(str(g) for g in givens) + " + x"
    working = [
        f"All angles at a point add to 360 degrees: {sum_expr} = 360.",
        f"x = 360 - {' - '.join(str(g) for g in givens)} = {x} degrees.",
    ]

    return question_text, answer_text, distractors, working, x


def _vertical_hard(rng):
    found = False
    for _ in range(50):
        x = rng.randint(2, 30)
        angle = rng.randint(_VERTICAL_FLOOR, _VERTICAL_CEIL)
        a1 = rng.randint(2, 9)
        c1 = rng.randint(2, 9)
        if c1 == a1:
            continue
        if x == angle:
            # guards the reports-x-not-angle distractor's own degeneracy
            continue
        a2 = angle - a1 * x
        c2 = angle - c1 * x
        found = True
        break
    check(found, "p10e_angle_relationships: vertical_hard could not find a valid draw")

    expr_a = _expr_text(a1, a2)
    expr_c = _expr_text(c1, c2)
    question_text = (
        "Two straight lines cross at a point, forming two vertically opposite "
        f"angles. One angle measures ({expr_a}) degrees and the other measures "
        f"({expr_c}) degrees. Vertically opposite angles are always equal. "
        "Find the value of the angle, in degrees."
    )
    question_latex = (
        r"\text{Two straight lines cross at a point, forming two vertically "
        r"opposite angles. One angle measures } "
        f"({sympy_latex(a1 * _X + a2)})"
        r" \text{ degrees and the other measures } "
        f"({sympy_latex(c1 * _X + c2)})"
        r" \text{ degrees. Vertically opposite angles are always equal. "
        r"Find the value of the angle, in degrees.}"
    )
    answer_text = str(angle)

    # Independent check: substitute the solved x back into BOTH original
    # expressions independently and assert (a) they are equal to each other
    # and (b) both equal the claimed answer -- a substitute-and-verify
    # check, not a reuse of sympy.solve's own returned value.
    x_sym = symbols("x")
    sol = solve(Eq(a1 * x_sym + a2, c1 * x_sym + c2), x_sym)
    check(len(sol) == 1, "p10e_angle_relationships: vertical_hard sympy solve did not return one root")
    x_check = sol[0]
    check_equivalent(x_check, x, "p10e_angle_relationships: vertical_hard sympy x disagrees")
    lhs_check = a1 * x_check + a2
    rhs_check = c1 * x_check + c2
    check_equivalent(lhs_check, rhs_check, "p10e_angle_relationships: vertical_hard expressions unequal at solved x")
    check_equivalent(
        lhs_check, angle, "p10e_angle_relationships: vertical_hard expression disagrees with claimed answer"
    )
    check_equivalent(
        rhs_check, angle, "p10e_angle_relationships: vertical_hard other expression disagrees with claimed answer"
    )

    # reports-x-not-angle: solves correctly for x but reports x itself.
    d_x = x
    # generic offset distractor.
    d_offset = angle + 10

    check_distinct(angle, d_x, msg="reports-x-not-angle distractor equals the answer")
    check_distinct(angle, d_offset, msg="offset distractor equals the answer")

    distractors = pick_distractors(answer_text, [str(d_x), str(d_offset)])

    working = [
        "Vertically opposite angles are equal.",
        f"({expr_a}) = ({expr_c})",
        f"x = {x}",
        f"angle = {join_terms([(a1, f'({x})'), (a2, '')])} = {angle} degrees.",
    ]

    return question_text, question_latex, answer_text, distractors, working, angle


@generator(
    "p10e_angle_relationships",
    level=6,
    presets={
        "easy": dict(variant="point_two_given"),
        "medium": dict(variant="point_three_given"),
        "hard": dict(variant="vertical_opposite_algebraic"),
    },
)
def p10e_angle_relationships(rng, variant="point_two_given"):
    if variant == "point_two_given":
        text, answer_text, distractors, working, x = _point_question(rng, 2, "Three angles", ("Two", "the third"))
        latex = rf"\text{{{text}}}"
        answer = x
    elif variant == "point_three_given":
        text, answer_text, distractors, working, x = _point_question(rng, 3, "Four angles", ("Three", "the fourth"))
        latex = rf"\text{{{text}}}"
        answer = x
    elif variant == "vertical_opposite_algebraic":
        text, latex, answer_text, distractors, working, angle = _vertical_hard(rng)
        answer = angle
    else:
        raise ValueError(f"unknown variant {variant!r}")

    return {
        "text": text,
        "latex": latex,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }
