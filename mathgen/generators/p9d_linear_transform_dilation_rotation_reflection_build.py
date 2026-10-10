"""Basic linear transformations: dilations, rotations about the origin, and
reflections in a line through the origin -- each shown by its own 2x2 matrix.

Skill: senior-define-and-use-basic-linear-transformations-dilations-of
(ACMSM055). Scope: this skill grades the *image point*
produced by applying a stated transformation to a supplied point. Every
question's `working` shows the transformation's own 2x2 matrix explicitly,
but the matrix itself is never a separately typed/graded answer -- `answer`
is always a coordinate pair (`xy_solution`).

Three tiers, one per transformation family:
- easy (dilation): (x, y) -> (lambda1 x, lambda2 y), lambda1 != lambda2,
  each drawn from {-4,-3,-2,2,3,4}.
- medium (rotation): 90/180/270 degrees about the origin.
- hard (reflection): in the x-axis, y-axis, y = x or y = -x.

CONSTRUCTION route (every tier): apply the family's own well-known matrix via
`_apply_matrix`. INDEPENDENT CHECK (every tier): the family's direct
coordinate rule, compared with a plain tuple `check(...)` -- not
`check_equivalent`/`check_distinct`, which route through `symbolically_equal`
and do not parse an "x = 3, y = -5"-labelled string; these are plain integer
tuples, not sympy expressions.
"""

from __future__ import annotations

from ..core import check, fmt_signed, generator, pick_distractors, xy_solution


def _apply_matrix(M, x, y):
    """Generic 2x2 matrix-vector product. M is ((a,b),(c,d))."""
    (a, b), (c, d) = M
    return a * x + b * y, c * x + d * y


def _matrix_text(M):
    (a, b), (c, d) = M
    return f"[[{a}, {b}], [{c}, {d}]]"


def _sample_point(rng, lo=-9, hi=9):
    """A point (x, y) in [lo, hi]^2, resampled until it is not the origin --
    every tier's transformation is trivial or degenerate there for at least
    one family."""
    while True:
        x, y = rng.randint(lo, hi), rng.randint(lo, hi)
        if (x, y) != (0, 0):
            return x, y


# ---------------------------------------------------------------------------
# Dilation
# ---------------------------------------------------------------------------

_DILATION_FACTORS = (-4, -3, -2, 2, 3, 4)


def _dilation(rng):
    lambda1 = rng.choice(_DILATION_FACTORS)
    lambda2 = rng.choice(_DILATION_FACTORS)
    while lambda2 == lambda1:
        lambda2 = rng.choice(_DILATION_FACTORS)
    x, y = _sample_point(rng)

    M = ((lambda1, 0), (0, lambda2))
    ix, iy = _apply_matrix(M, x, y)
    # Independent check: the direct per-axis scalar rule.
    direct = (lambda1 * x, lambda2 * y)
    check((ix, iy) == direct, "dilation construction does not match the direct rule")

    answer_tuple = (ix, iy)
    # Misconceptions: swapped which lambda scales which axis; used the same factor on both axes;
    # flipped the sign of the first factor only. Each is filtered against the true answer.
    raw_candidates = [
        (lambda2 * x, lambda1 * y),  # swapped-lambda
        (lambda1 * x, lambda1 * y),  # same-factor
        (-lambda1 * x, lambda2 * y),  # sign-flip-first
    ]
    candidates = [f"x = {a}, y = {b}" for a, b in raw_candidates if (a, b) != answer_tuple]

    lam1_term = fmt_signed(lambda1, "x")
    lam2_term = fmt_signed(lambda2, "y")
    matrix_text = _matrix_text(M)
    text = f"Apply a dilation (x, y) -> ({lam1_term}, {lam2_term}) to the point ({x}, {y}). Find the image point."

    result = xy_solution(ix, iy)
    result["distractors"] = pick_distractors(result["answer_text"], candidates)
    result["text"] = text
    result["latex"] = rf"\text{{{text}}}"
    result["working"] = [
        f"The dilation matrix is {matrix_text}.",
        f"Image = ({lambda1} × {x}, {lambda2} × {y}) = ({ix}, {iy}).",
    ]
    return result


# ---------------------------------------------------------------------------
# Rotation about the origin
# ---------------------------------------------------------------------------

_ROTATION_MATRICES = {
    90: ((0, -1), (1, 0)),
    180: ((-1, 0), (0, -1)),
    270: ((0, 1), (-1, 0)),
}
_ROTATION_WRONG_ANGLE = {90: 270, 180: 90, 270: 90}


def _rotation_direct(theta, x, y):
    if theta == 90:
        return (-y, x)
    if theta == 180:
        return (-x, -y)
    return (y, -x)  # theta == 270


def _rotation(rng):
    theta = rng.choice((90, 180, 270))
    x, y = _sample_point(rng)

    M = _ROTATION_MATRICES[theta]
    ix, iy = _apply_matrix(M, x, y)
    direct = _rotation_direct(theta, x, y)
    check((ix, iy) == direct, "rotation construction does not match the direct rule")

    answer_tuple = (ix, iy)
    wrong_theta = _ROTATION_WRONG_ANGLE[theta]
    # Misconceptions: forgot to transform at all; swapped the coordinates
    # instead of rotating; applied a different angle's rule.
    raw_candidates = [
        (x, y),  # identity
        (y, x),  # swapped-coords
        _rotation_direct(wrong_theta, x, y),  # wrong-rotation
    ]
    candidates = [f"x = {a}, y = {b}" for a, b in raw_candidates if (a, b) != answer_tuple]

    matrix_text = _matrix_text(M)
    # "counterclockwise" is stated explicitly: without it, at theta=90 or 270 a learner using the
    # opposite (clockwise) convention computes exactly the "wrong-rotation" distractor and would be
    # marked wrong for a genuinely defensible reading (at theta=180 direction is irrelevant --
    # CW-180 == CCW-180 -- so this specific risk is confined to 90/270, but stating the direction
    # removes the ambiguity for all three uniformly).
    text = f"Rotate the point ({x}, {y}) by {theta} degrees counterclockwise about the origin. Find the image point."

    result = xy_solution(ix, iy)
    result["distractors"] = pick_distractors(result["answer_text"], candidates)
    result["text"] = text
    result["latex"] = rf"\text{{{text}}}"
    result["working"] = [
        f"The rotation matrix for {theta} degrees is {matrix_text}.",
        f"Image = ({ix}, {iy}).",
    ]
    return result


# ---------------------------------------------------------------------------
# Reflection in a line through the origin
# ---------------------------------------------------------------------------

_REFLECTION_MATRICES = {
    "x-axis": ((1, 0), (0, -1)),
    "y-axis": ((-1, 0), (0, 1)),
    "y=x": ((0, 1), (1, 0)),
    "y=-x": ((0, -1), (-1, 0)),
}
_REFLECTION_WRONG_LINE = {"x-axis": "y=x", "y-axis": "y=-x", "y=x": "x-axis", "y=-x": "y-axis"}
_REFLECTION_LINE_PHRASE = {
    "x-axis": "the x-axis",
    "y-axis": "the y-axis",
    "y=x": "the line y = x",
    "y=-x": "the line y = -x",
}


def _reflection_direct(line, x, y):
    if line == "x-axis":
        return (x, -y)
    if line == "y-axis":
        return (-x, y)
    if line == "y=x":
        return (y, x)
    return (-y, -x)  # line == "y=-x"


def _reflection_degenerate(line, x, y):
    """Whether (x, y) sits ON the line -- a no-op reflection, pedagogically
    trivial and the reason the "identity" distractor below could never
    differ from the answer."""
    if line == "x-axis":
        return y == 0
    if line == "y-axis":
        return x == 0
    if line == "y=x":
        return x == y
    return x == -y  # line == "y=-x"


def _reflection(rng):
    line = rng.choice(("x-axis", "y-axis", "y=x", "y=-x"))
    x, y = _sample_point(rng)
    while _reflection_degenerate(line, x, y):
        x, y = _sample_point(rng)

    M = _REFLECTION_MATRICES[line]
    ix, iy = _apply_matrix(M, x, y)
    direct = _reflection_direct(line, x, y)
    check((ix, iy) == direct, "reflection construction does not match the direct rule")

    answer_tuple = (ix, iy)
    wrong_line = _REFLECTION_WRONG_LINE[line]
    # Misconceptions: forgot to transform at all; reflected in the wrong
    # (paired) line; treated it as a 180 degree rotation instead of a
    # reflection.
    raw_candidates = [
        (x, y),  # identity
        _reflection_direct(wrong_line, x, y),  # wrong-line
        (-x, -y),  # point-reflection
    ]
    candidates = [f"x = {a}, y = {b}" for a, b in raw_candidates if (a, b) != answer_tuple]

    matrix_text = _matrix_text(M)
    line_phrase = _REFLECTION_LINE_PHRASE[line]
    text = f"Reflect the point ({x}, {y}) in {line_phrase}. Find the image point."

    result = xy_solution(ix, iy)
    result["distractors"] = pick_distractors(result["answer_text"], candidates)
    result["text"] = text
    result["latex"] = rf"\text{{{text}}}"
    result["working"] = [
        f"The reflection matrix for {line_phrase} is {matrix_text}.",
        f"Image = ({ix}, {iy}).",
    ]
    return result


@generator(
    "linear_transform_dilation_rotation_reflection",
    level=13,
    presets={
        "easy": dict(variant="dilation"),
        "medium": dict(variant="rotation"),
        "hard": dict(variant="reflection"),
    },
)
def linear_transform_dilation_rotation_reflection(rng, variant="dilation"):
    if variant == "dilation":
        return _dilation(rng)
    if variant == "rotation":
        return _rotation(rng)
    if variant == "reflection":
        return _reflection(rng)
    raise ValueError(f"Unknown variant: {variant}")
