"""Senior coordinate-geometry skill ACMMM001: determine the coordinates of the
midpoint of two points, scalar (single named coordinate) only.

This generator asks for exactly ONE coordinate (x or y) of the midpoint,
never the full ordered pair. In contrast, the existing F-10 skill
`midpoint_two_points` (mathgen/generators/coordinate_geometry_build.py::
"Calculate the exact midpoint of two supplied Cartesian points.") computes
the FULL coordinate pair with answer kind `pair`.

Three difficulty tiers:
- easy: sum of same-axis coordinates is even (integer midpoint)
- medium: sum is odd (half-integer midpoint)
- hard: reverse-solve for the OTHER endpoint's coordinate given one endpoint
  and the midpoint's coordinate (genuinely different reasoning, not merely
  a relabelled forward question)

Answer type: `number` (not `pair`); single numeric answer formatted with
Python's `:g` format to avoid spurious trailing zeros.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, join_terms, pick_distractors, paren_negative


def _fmt(value, places: int = 0) -> str:
    """Render a candidate in the same style as ``answer_text``.

    The answer used ``f"{float(v):g}"`` while distractors used ``str()`` on a
    Fraction, so a half-integer answer read "1.5" among options like "-11/2"
    and was the only decimal on offer. A learner could pick it without doing
    the arithmetic in a third of all questions.
    """
    if places:
        # A half-integer key reads "1.5", so every option carries the same one
        # decimal place ("3.0", not "3"): the point must not single out the key.
        return f"{float(value):.{places}f}"
    return f"{float(value):g}"


def _verify_forward(p1, p2, answer):
    """Independent check for forward tiers (easy/medium): the answer is
    equidistant from both endpoints on the number line, a route that never
    reuses the averaging formula used to construct it."""
    answer_frac = Fraction(answer)
    p1_frac = Fraction(p1)
    p2_frac = Fraction(p2)
    check(abs(answer_frac - p1_frac) == abs(answer_frac - p2_frac), "midpoint is not equidistant from both endpoints")


def _verify_reverse(x1, x2, answer):
    """Independent check for hard tier (reverse operation): recompute the
    midpoint from x1 and the constructed x2, independently of the 2*m - x1
    formula used to solve for x2, and confirm it equals the stated answer m."""
    x1_frac = Fraction(x1)
    x2_frac = Fraction(x2)
    answer_frac = Fraction(answer)
    computed_midpoint = Fraction(x1_frac + x2_frac, 2)
    check(computed_midpoint == answer_frac, "recomputed midpoint does not equal stated answer")


@generator(
    "midpoint_coordinate_scalar",
    level=8,
    presets={
        "easy": dict(coord_lo=0, coord_hi=6, parity_mode="even"),
        "medium": dict(coord_lo=-8, coord_hi=8, parity_mode="odd"),
        "hard": dict(coord_lo=-14, coord_hi=14, parity_mode=None),
    },
)
def midpoint_coordinate_scalar(rng, coord_lo=-8, coord_hi=8, parity_mode="even"):
    """Determine one named coordinate of the midpoint of two points; or
    (hard tier) solve for the OTHER endpoint's coordinate given one endpoint
    and the midpoint's coordinate."""
    if coord_hi <= coord_lo:
        raise ValueError("coord_hi must be greater than coord_lo")

    # Forward tiers (easy/medium): two points, ask for midpoint coordinate
    if parity_mode in ("even", "odd"):
        # Sample two coordinates whose sum has the target parity, with collision guards
        while True:
            p1 = rng.randint(coord_lo, coord_hi)
            p2 = rng.randint(coord_lo, coord_hi)
            # Ensure distinct points (kills used-one-endpoint misconception)
            if p1 == p2:
                continue
            # Enforce parity: sum is even for easy, odd for medium
            parity_ok = (parity_mode == "even" and (p1 + p2) % 2 == 0) or (parity_mode == "odd" and (p1 + p2) % 2 == 1)
            if not parity_ok:
                continue

            # Collision guards:
            # - p1 + p2 ≠ 0 (forgot-to-halve = answer)
            # - p2 ≠ 0 (subtracted = answer)
            # - p1 ≠ −3·p2 (forgot-to-halve = subtracted)
            if p1 + p2 == 0:
                continue
            if p2 == 0:
                continue
            if p1 == -3 * p2:
                continue
            # All guards passed
            break

        # Compute the midpoint coordinate
        answer_frac = Fraction(p1 + p2, 2)
        answer = float(answer_frac)  # JSON-serialisable
        answer_text = f"{answer:g}"  # ":g" avoids spurious ".0"

        # Verify using the independent equidistance route
        _verify_forward(p1, p2, answer_frac)

        places = 1 if "." in answer_text else 0

        # Generate misconception distractors
        distractor_sum = p1 + p2  # forgot-to-halve
        distractor_one = p1  # used-one-endpoint-only
        distractor_diff = Fraction(p1 - p2, 2)  # subtracted-instead-of-averaged

        candidates = [
            _fmt(distractor_sum, places),
            _fmt(distractor_one, places),
            _fmt(distractor_diff, places),
        ]

        # Pick axis (x or y) at random
        axis = rng.choice(["x", "y"])
        if axis == "x":
            x1, y1 = p1, rng.randint(coord_lo, coord_hi)
            x2, y2 = p2, rng.randint(coord_lo, coord_hi)
            while (x1, y1) == (x2, y2):
                y2 = rng.randint(coord_lo, coord_hi)
        else:
            x1 = rng.randint(coord_lo, coord_hi)
            y1 = p1
            x2 = rng.randint(coord_lo, coord_hi)
            y2 = p2
            while (x1, y1) == (x2, y2):
                x2 = rng.randint(coord_lo, coord_hi)

        axis_label = "x" if axis == "x" else "y"
        return dict(
            text=(f"Find the {axis_label}-coordinate of the midpoint of the points ({x1}, {y1}) and ({x2}, {y2})."),
            latex=(
                rf"\text{{Find the {axis_label}-coordinate of the midpoint of}} "
                rf"({x1},{y1}) \text{{ and }}({x2},{y2})"
            ),
            answer=answer,
            answer_text=answer_text,
            answer_latex=(
                rf"\frac{{{join_terms([(p1, ''), (p2, '')])}}}{{2}} = {answer_text}"
                if answer_frac.denominator == 1
                else rf"\frac{{{join_terms([(p1, ''), (p2, '')])}}}{{2}} = \frac{{{answer_frac.numerator}}}{{{answer_frac.denominator}}}"
            ),
            working=[
                f"Midpoint {axis_label}-coordinate = ({join_terms([(p1, ''), (p2, '')])})/2",
                f"Midpoint {axis_label}-coordinate = {answer_frac.numerator}/{answer_frac.denominator}"
                if answer_frac.denominator != 1
                else f"Midpoint {axis_label}-coordinate = {answer_frac.numerator}",
                f"Midpoint {axis_label}-coordinate = {answer_text}",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    else:
        # Hard tier: reverse operation
        # Given x1 and the midpoint coordinate m, solve for x2 = 2*m - x1
        while True:
            x1 = rng.randint(coord_lo, coord_hi)
            # Midpoint coordinate m (a half-integer or integer)
            m_int_part = rng.randint(coord_lo, coord_hi)
            use_half = rng.random() < 0.5
            if use_half:
                m = Fraction(m_int_part * 2 + 1, 2)
            else:
                m = Fraction(m_int_part)

            # Solve for x2: midpoint = (x1 + x2) / 2, so x2 = 2*m - x1
            x2 = 2 * m - x1
            x2_int = int(x2)  # Should be an integer since x1 is int and m is integer or half-int

            # Ensure distinct points
            if x1 == x2_int:
                continue

            # Collision guards:
            # - x1 ≠ x2 (same-endpoint-copy = answer) -- already checked above
            # - x1 ≠ −x2 (forgot-to-double = answer)
            # - x2 ≠ 3·x1 (forgot-to-double = same-endpoint-copy)
            if x1 == -x2_int:
                continue
            if x2_int == 3 * x1:
                continue
            # All guards passed
            break

        answer = float(x2)
        answer_text = f"{answer:g}"

        # Verify using the independent recomputation route
        _verify_reverse(x1, x2_int, m)

        # Generate hard-tier misconception distractors
        distractor_same = x1  # same-endpoint-copy: states x1 instead of solving
        distractor_half_diff = m - x1  # forgot-to-double: states m - x1 instead of 2*m - x1

        # The key is a whole number, so every option is: a half-integer slip
        # (m - x1 with a half-integer m) would give the key away by its point.
        distractor_added = 2 * m + x1  # added-instead-of-subtracted
        distractor_doubled_diff = 2 * (m - x1)  # doubled the wrong difference
        candidates = [
            _fmt(v)
            for v in (distractor_same, distractor_half_diff, distractor_added, distractor_doubled_diff)
            if Fraction(v).denominator == 1
        ]

        # Pick axis at random
        axis = rng.choice(["x", "y"])
        if axis == "x":
            # x1 and midpoint's x-coordinate are known; generate y values
            y1 = rng.randint(coord_lo, coord_hi)
            y2 = rng.randint(coord_lo, coord_hi)
            while (x1, y1) == (x2_int, y2):
                y2 = rng.randint(coord_lo, coord_hi)
            axis_label = "x"
            p1_coord, p2_coord = x1, x2_int
            text_m = f"{m:g}"  # Format m with :g to avoid ".0"
        else:
            # y1 and midpoint's y-coordinate are known; generate x values
            x1_for_point = rng.randint(coord_lo, coord_hi)
            x2_for_point = rng.randint(coord_lo, coord_hi)
            while (x1_for_point, x1) == (x2_for_point, x2_int):
                x2_for_point = rng.randint(coord_lo, coord_hi)
            y1, y2 = x1, x2_int  # y is the coordinate we're solving for
            axis_label = "y"
            p1_coord, p2_coord = x1, x2_int
            text_m = f"{m:g}"
            x1 = x1_for_point
            x2_int = x2_for_point

        return dict(
            text=(
                f"One endpoint is ({x1}, {y1}) and the {axis_label}-coordinate "
                f"of the midpoint is {text_m}. Find the {axis_label}-coordinate of "
                f"the other endpoint."
            ),
            latex=(
                rf"\text{{Endpoint: }}({x1},{y1}), "
                rf"\text{{ midpoint's {axis_label}-coordinate: }}{text_m}, "
                rf"\text{{ find other's {axis_label}-coordinate}}"
            ),
            answer=answer,
            answer_text=answer_text,
            answer_latex=(rf"{p2_coord}" if isinstance(p2_coord, int) else rf"\frac{{{int(p2_coord * 2)}}}{{{2}}}"),
            working=[
                f"Midpoint {axis_label}-coordinate = ({p1_coord} + {axis_label}_2)/2",
                f"{text_m} = ({p1_coord} + {axis_label}_2)/2",
                f"{text_m} × 2 = {p1_coord} + {axis_label}_2",
                f"{float(2 * m):g} = {p1_coord} + {axis_label}_2",
                f"{axis_label}_2 = {float(2 * m):g} - {paren_negative(p1_coord)}",
                f"{axis_label}_2 = {answer_text}",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )
