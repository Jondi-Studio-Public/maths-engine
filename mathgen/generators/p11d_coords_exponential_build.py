"""Two genuinely new Foundation-Year 10 skills.

- ``p11d_three_dimensional_coordinates`` (AC9M8SP03, prerequisite
  ``f10-coordinate-movement``, virgin area) -- given two (or, hard tier,
  three) stated points in a three-dimensional coordinate system, determine
  the displacement (change in each coordinate) from one point to the other.
  A direct 3-D extension of ``p4b_coordinate_prism_build.py``'s own
  ``coordinate_movement`` generator. Response family: ``ordered_structures``
  (``answer: pair``, a 3-label dict -- proven to grade correctly at length 3

- ``p11d_exponential_common_base_solve`` (AC9M10A03, prerequisites
  ``f10-exponent-laws-integer-variable``/``function-notation``, virgin area)
  -- given a bounded exponential equation solvable by expressing both sides
  as powers of a common prime base, solve for x. No logarithms; a single
  solution, never up to two roots. Response family: ``scalar_expression``
  (``answer: fraction`` -- ``grade_fraction``'s ``{"p", "q"}`` dict branch,
  the same shape ``foundations_algebra.py``'s ``gradient`` generator uses,
  since some tiers' solutions are non-integer rationals).
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Rational as SymRational
from sympy import latex as sympy_latex

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    latex_answer,
    paren_negative,
    pick_distractors,
    rational_text,
)

# -- Skill 1: p11d_three_dimensional_coordinates (pair answer) --------------

# Opening/closing phrasing rotation for the easy/medium (single-movement)
# template. Every variant keeps the literal "from (x1, y1, z1) to (x2, y2,
# z2)" fragment `tests/test_p11d_coords_exponential_oracle.py`'s
# `_SIMPLE_MOVE_RE` searches for -- only the sentence AROUND it varies, so
# the kernel ("displacement between two stated points") is unchanged.
_SIMPLE_MOVE_TEMPLATES = (
    "A point moves from ({start}) to ({end}). What is the change in each coordinate?",
    "A point travels from ({start}) to ({end}). Find the change in each coordinate (dx, dy, dz).",
    "An object moves from ({start}) to ({end}). Determine the change in each coordinate.",
)

# Opening/closing phrasing rotation for the hard (three-point) template.
# Every variant keeps the literal "A(x, y, z), moves to B(x, y, z), then
# moves again to C(x, y, z)" fragment `_HARD_MOVE_RE` searches for.
_HARD_MOVE_TEMPLATES = (
    "A point starts at A({a}), moves to B({b}), then moves again to C({c}). "
    "What is the total change in each coordinate from A to C?",
    "An object starts at A({a}), moves to B({b}), then moves again to C({c}). "
    "Find the total change in each coordinate from A to C.",
    "A particle starts at A({a}), moves to B({b}), then moves again to C({c}). "
    "Determine the total change in each coordinate from A to C.",
)


def _draw_point(rng, lo: int, hi: int) -> tuple[int, int, int]:
    return (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))


def _verify_coordinate_change_3d(start, end, delta) -> None:
    """Independent check via addition (invert the subtraction operation):
    reconstruct the claimed end point by adding the claimed (dx, dy, dz)
    back onto the start point and assert it equals the given end point
    exactly, in every one of the three coordinates -- the direct 3-D
    generalisation of ``p4b_coordinate_prism_build.py``'s own
    ``_verify_coordinate_change``."""
    reconstructed = tuple(s + d for s, d in zip(start, delta))
    check(
        reconstructed == end,
        f"reconstructed endpoint {reconstructed} does not match given endpoint {end}",
    )


@generator(
    "p11d_three_dimensional_coordinates",
    level=8,
    presets={
        "easy": dict(coord_lo=1, coord_hi=10),
        "medium": dict(coord_lo=-10, coord_hi=10),
        "hard": dict(coord_lo=-10, coord_hi=10, sequence_length=3),
    },
)
def three_dimensional_coordinates(rng, coord_lo=-10, coord_hi=10, sequence_length=2):
    """Given two (or, hard tier, three named) points in a three-dimensional
    coordinate system, state the displacement (change in each coordinate)
    from the first to the last."""
    if coord_hi <= coord_lo:
        raise ValueError("coord_hi must be greater than coord_lo")
    if sequence_length not in (2, 3):
        raise ValueError("sequence_length must be 2 (easy/medium) or 3 (hard)")

    if sequence_length == 2:
        # Phrasing drawn FIRST so it consumes RNG state deterministically
        # regardless of how many resample attempts the point draw below
        # takes -- a surface rotation only, no change to the kernel.
        template = rng.choice(_SIMPLE_MOVE_TEMPLATES)

        # Easy/medium: single movement. Resample (plain-tuple pre-filter,
        # cheap) until the answer and every misconception candidate are
        # pairwise distinct as VALUES, then confirm with check_pairwise_distinct
        # unconditionally -- the same two-stage pattern
        # p7e_exponent_laws_build.py uses for its own collision-free draws.
        for _ in range(50):
            start = _draw_point(rng, coord_lo, coord_hi)
            end = _draw_point(rng, coord_lo, coord_hi)
            if start == end:
                continue
            delta = tuple(e - s for s, e in zip(start, end))
            swapped = (delta[2], delta[0], delta[1])  # cyclic shift
            wrong_direction = tuple(-d for d in delta)  # negates all three
            omits_subtraction = end  # reports raw end coords
            if len({delta, swapped, wrong_direction, omits_subtraction}) == 4:
                break
        else:
            raise RuntimeError("could not draw a collision-free easy/medium coordinate set")

        _verify_coordinate_change_3d(start, end, delta)
        check_pairwise_distinct(
            delta,
            swapped,
            wrong_direction,
            omits_subtraction,
            msg="an easy/medium-tier distractor collides with the answer or another option",
        )

        dx, dy, dz = delta
        start_text = f"{start[0]}, {start[1]}, {start[2]}"
        end_text = f"{end[0]}, {end[1]}, {end[2]}"
        text = template.format(start=start_text, end=end_text)
        working = [
            f"Change in x: dx = (end x) - (start x) = {paren_negative(end[0])} - {paren_negative(start[0])} = {dx}",
            f"Change in y: dy = (end y) - (start y) = {paren_negative(end[1])} - {paren_negative(start[1])} = {dy}",
            f"Change in z: dz = (end z) - (start z) = {paren_negative(end[2])} - {paren_negative(start[2])} = {dz}",
        ]
        candidates = [
            f"dx = {swapped[0]}, dy = {swapped[1]}, dz = {swapped[2]}",
            f"dx = {wrong_direction[0]}, dy = {wrong_direction[1]}, dz = {wrong_direction[2]}",
            f"dx = {omits_subtraction[0]}, dy = {omits_subtraction[1]}, dz = {omits_subtraction[2]}",
        ]

    else:  # sequence_length == 3, hard tier
        # Phrasing drawn FIRST, same reasoning as the sequence_length == 2
        # branch above.
        template = rng.choice(_HARD_MOVE_TEMPLATES)

        for _ in range(50):
            a_pt = _draw_point(rng, coord_lo, coord_hi)
            b_pt = _draw_point(rng, coord_lo, coord_hi)
            if a_pt == b_pt:
                continue
            c_pt = _draw_point(rng, coord_lo, coord_hi)
            if c_pt == b_pt or c_pt == a_pt:
                continue
            delta = tuple(c - a for a, c in zip(a_pt, c_pt))
            swapped = (delta[2], delta[0], delta[1])
            wrong_direction = tuple(-d for d in delta)
            ignores_first_leg = tuple(c - b for b, c in zip(b_pt, c_pt))
            if len({delta, swapped, wrong_direction, ignores_first_leg}) == 4:
                break
        else:
            raise RuntimeError("could not draw a collision-free hard coordinate set")

        _verify_coordinate_change_3d(a_pt, c_pt, delta)
        check_pairwise_distinct(
            delta,
            swapped,
            wrong_direction,
            ignores_first_leg,
            msg="a hard-tier distractor collides with the answer or another option",
        )

        dx, dy, dz = delta
        a_text = f"{a_pt[0]}, {a_pt[1]}, {a_pt[2]}"
        b_text = f"{b_pt[0]}, {b_pt[1]}, {b_pt[2]}"
        c_text = f"{c_pt[0]}, {c_pt[1]}, {c_pt[2]}"
        text = template.format(a=a_text, b=b_text, c=c_text)
        working = [
            f"Change in x: dx = x_C - x_A = {paren_negative(c_pt[0])} - {paren_negative(a_pt[0])} = {dx}",
            f"Change in y: dy = y_C - y_A = {paren_negative(c_pt[1])} - {paren_negative(a_pt[1])} = {dy}",
            f"Change in z: dz = z_C - z_A = {paren_negative(c_pt[2])} - {paren_negative(a_pt[2])} = {dz}",
        ]
        candidates = [
            f"dx = {swapped[0]}, dy = {swapped[1]}, dz = {swapped[2]}",
            f"dx = {wrong_direction[0]}, dy = {wrong_direction[1]}, dz = {wrong_direction[2]}",
            f"dx = {ignores_first_leg[0]}, dy = {ignores_first_leg[1]}, dz = {ignores_first_leg[2]}",
        ]

    answer_text = f"dx = {dx}, dy = {dy}, dz = {dz}"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"dx": dx, "dy": dy, "dz": dz},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


# -- Skill 2: p11d_exponential_common_base_solve (fraction answer) ----------

PRIMES = (2, 3, 5)

# Easy/hard tier n-ranges, one per prime.
_SIMPLE_N_RANGE = {2: range(1, 13), 3: range(1, 8), 5: range(1, 6)}  # pool 12+7+5=24
_MEDIUM_M = (2, 3)
_MEDIUM_N_RANGE = range(-8, 9)  # -8..8, zero excluded below
_HARD_COEF_EXPONENT = 2  # a_coef = p**2 -- p**1 would make the
# forgets-the-coefficient distractor (n + 1) collide
# with the adds-the-exponents distractor (n + m, m=1)
# on every seed; p**2 keeps them one apart always.


def _latex_fraction(value: Fraction) -> str:
    return sympy_latex(SymRational(value.numerator, value.denominator))


# Phrasing rotation for the single-equation (easy/medium) form, each drawing
# the same base/value pair into a different sentence. Every variant keeps
# the exact "{base}^x = {value}" fragment the oracle's relaxed
# `_EASY_RE`/`_MEDIUM_RE` search for -- only the surrounding sentence varies.
_SIMPLE_EQUATION_TEMPLATES = (
    ("Solve {a}^x = {b} for x.", r"\text{{Solve }} {a}^{{x}} = {b} \text{{ for }} x."),
    ("Find x if {a}^x = {b}.", r"\text{{Find }} x \text{{ if }} {a}^{{x}} = {b}."),
    (
        "For what value of x does {a}^x = {b}?",
        r"\text{{For what value of }} x \text{{ does }} {a}^{{x}} = {b}?",
    ),
)

# Phrasing rotation for the compound (hard-tier) coefficient form, keeping
# the exact "{a_coef} × {p}^x = {c_value}" fragment `_HARD_RE` searches for.
_COMPOUND_EQUATION_TEMPLATES = (
    (
        "Solve {a_coef} × {p}^x = {c_value} for x.",
        r"\text{{Solve }} {a_coef} \times {p}^{{x}} = {c_value} \text{{ for }} x.",
    ),
    (
        "Find x if {a_coef} × {p}^x = {c_value}.",
        r"\text{{Find }} x \text{{ if }} {a_coef} \times {p}^{{x}} = {c_value}.",
    ),
    (
        "For what value of x does {a_coef} × {p}^x = {c_value}?",
        r"\text{{For what value of }} x \text{{ does }} {a_coef} \times {p}^{{x}} = {c_value}?",
    ),
)


def _verify_simple(a_base: int, b_value: Fraction, x_value: Fraction) -> None:
    """Independent check: substitute the DISPLAYED base and target back into
    the equation via sympy exponentiation -- a genuinely different route from
    the n/m division used to derive ``x_value``."""
    lhs = SymRational(a_base) ** SymRational(x_value.numerator, x_value.denominator)
    rhs = SymRational(b_value.numerator, b_value.denominator)
    check(lhs == rhs, f"{a_base}^{x_value} != {b_value}")


def _verify_compound(a_coef: int, b_base: int, c_value: int, x_value: Fraction) -> None:
    lhs = SymRational(a_coef) * SymRational(b_base) ** SymRational(x_value.numerator, x_value.denominator)
    check(lhs == SymRational(c_value), f"{a_coef} * {b_base}^{x_value} != {c_value}")


def _build_easy(rng) -> dict:
    """m = 1 (the stated base IS the prime itself): x = n, a positive integer."""
    # Phrasing drawn FIRST so it consumes RNG state deterministically
    # regardless of how many resample attempts the value draw below takes.
    text_template, latex_template = rng.choice(_SIMPLE_EQUATION_TEMPLATES)

    for _ in range(50):
        p = rng.choice(PRIMES)
        n = rng.choice(_SIMPLE_N_RANGE[p])
        b_value = p**n
        x_value = Fraction(n, 1)
        wrong_add = Fraction(n + 1)  # adds-the-exponents (m = 1)
        wrong_target = Fraction(b_value)  # reports the target value itself, not x
        if len({x_value, wrong_add, wrong_target}) == 3:
            break
    else:
        raise RuntimeError("could not draw a collision-free easy exponential set")

    _verify_simple(p, Fraction(b_value), x_value)
    check_pairwise_distinct(
        x_value,
        wrong_add,
        wrong_target,
        msg="an easy-tier distractor collides with the answer or another option",
    )

    answer_text = rational_text(x_value)
    text = text_template.format(a=p, b=b_value)
    latex_text = latex_template.format(a=p, b=b_value)
    candidates = [rational_text(wrong_add), rational_text(wrong_target)]

    return dict(
        text=text,
        latex=latex_text,
        answer={"p": x_value.numerator, "q": x_value.denominator},
        answer_text=answer_text,
        answer_latex=_latex_fraction(x_value),
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_medium(rng) -> dict:
    """m in {2, 3} (the stated base is itself a composite power of the
    common prime): n is drawn so the result never collapses to an
    easy-tier-reachable positive integer (n > 0 and m | n excluded on every
    seed) -- the genuine escalation is recognising a = p**m before matching
    exponents, on top of a possibly negative/fractional x."""
    # Phrasing drawn FIRST, same reasoning as _build_easy above.
    text_template, latex_template = rng.choice(_SIMPLE_EQUATION_TEMPLATES)

    for _ in range(50):
        m = rng.choice(_MEDIUM_M)
        n = rng.choice([v for v in _MEDIUM_N_RANGE if v != 0])
        if n > 0 and n % m == 0:
            continue
        p = rng.choice(PRIMES)
        a_base = p**m
        b_value = Fraction(p) ** n
        x_value = Fraction(n, m)

        wrong_add = Fraction(n + m)  # adds-the-exponents
        wrong_target = b_value  # reports the target value itself, not x
        # off-by-one-multiple: (n + m) / m = x_value + 1. gcd(n, m) == gcd(n + m, m) always, so this
        # shares x_value's own reduced denominator exactly -- always the SAME option "shape"
        # (fraction vs integer) as the answer. Not one of the three named misconceptions; added so a
        # fractional answer is never the only fraction-shaped option among otherwise-integer
        # distractors, which wrong_add/wrong_target alone cannot guarantee since both are plain
        # integers whenever n > 0.
        wrong_off_by_one = Fraction(n + m, m)
        pool = [x_value, wrong_add, wrong_target, wrong_off_by_one]
        wrong_drop_sign = None
        if n < 0:
            wrong_drop_sign = Fraction(abs(n), m)  # drops-the-negative-sign
            pool.append(wrong_drop_sign)

        if len(set(pool)) == len(pool):
            break
    else:
        raise RuntimeError("could not draw a collision-free medium exponential set")

    _verify_simple(a_base, b_value, x_value)
    check_pairwise_distinct(*pool, msg="a medium-tier distractor collides with the answer or another option")

    answer_text = rational_text(x_value)
    b_text = rational_text(b_value)
    text = text_template.format(a=a_base, b=b_text)
    latex_text = latex_template.format(a=a_base, b=_latex_fraction(b_value))
    candidate_values = [wrong_add, wrong_target, wrong_off_by_one] + (
        [wrong_drop_sign] if wrong_drop_sign is not None else []
    )
    candidates = [rational_text(v) for v in candidate_values]

    return dict(
        text=text,
        latex=latex_text,
        answer={"p": x_value.numerator, "q": x_value.denominator},
        answer_text=answer_text,
        answer_latex=_latex_fraction(x_value),
        distractors=pick_distractors(answer_text, candidates, n=len(candidates)),
    )


def _build_hard(rng) -> dict:
    """Compound form a_coef * p^x = c, m = 1 (isolating a_coef is itself the
    new step this tier adds): a_coef = p**2 so it is a genuine coefficient
    to divide through, not the base itself."""
    # Phrasing drawn FIRST, same reasoning as _build_easy above. This tier
    # has no resample loop, but the draw order still matters for
    # `test_determinism_across_seeds` and for keeping RNG consumption
    # consistent with the other two tiers.
    text_template, latex_template = rng.choice(_COMPOUND_EQUATION_TEMPLATES)
    p = rng.choice(PRIMES)
    n = rng.choice(_SIMPLE_N_RANGE[p])
    a_coef = p**_HARD_COEF_EXPONENT
    c_value = a_coef * p**n
    x_value = Fraction(n, 1)

    wrong_add = Fraction(n + 1)  # adds-the-exponents (m = 1)
    wrong_forgets_coef = Fraction(n + _HARD_COEF_EXPONENT)  # solves p^x = c directly

    check_pairwise_distinct(
        x_value,
        wrong_add,
        wrong_forgets_coef,
        msg="a hard-tier distractor collides with the answer or another option",
    )
    _verify_compound(a_coef, p, c_value, x_value)

    answer_text = rational_text(x_value)
    text = text_template.format(a_coef=a_coef, p=p, c_value=c_value)
    latex_text = latex_template.format(a_coef=a_coef, p=p, c_value=c_value)
    candidates = [rational_text(wrong_add), rational_text(wrong_forgets_coef)]

    return dict(
        text=text,
        latex=latex_text,
        answer={"p": x_value.numerator, "q": x_value.denominator},
        answer_text=answer_text,
        answer_latex=_latex_fraction(x_value),
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "p11d_exponential_common_base_solve",
    level=10,
    presets={
        "easy": dict(variant="simple_positive"),
        "medium": dict(variant="composite_base"),
        "hard": dict(variant="compound_coefficient"),
    },
)
def exponential_common_base_solve(rng, variant="simple_positive"):
    """Solve a bounded exponential equation (a^x = b, or a_coef * b^x = c)
    whose sides can both be expressed as powers of a common prime, for x.
    Common-base matching only -- no logarithms, always a single solution."""
    if variant == "simple_positive":
        return _build_easy(rng)
    if variant == "composite_base":
        return _build_medium(rng)
    if variant == "compound_coefficient":
        return _build_hard(rng)
    raise ValueError(f"unknown variant: {variant!r}")
