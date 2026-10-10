"""Generates `p14c_coordinate_transformation` questions (`ucv1-coordinate-transformation`): transform a point
between coordinate systems, answered as a pair.

Given a stated point ``(x0, y0)`` (integers), a stated linear part ``M`` (a
reflection in the x-axis/y-axis/``y=x``/``y=-x``, a rotation by 90/180/270
degrees about the origin, or a dilation ``(x, y) -> (a*x, b*y)``) and a
stated NONZERO translation ``(tx, ty)`` (integers), this skill grades the
image point of the AFFINE map:

    (x1, y1) = M * (x0, y0) + (tx, ty)

Every tier's ``working`` states the linear part's own 2x2 matrix explicitly,
satisfying the descriptor's "the representations of these transformations by
2x2 matrices" as shown reasoning, but (as in the shipping
``senior-define-and-use-basic-linear-transformations-dilations-of``
skill) the matrix itself is never a separately typed/graded answer --
``answer`` is always the resulting coordinate pair.

Boundary against the three shipping senior linear-transformation skills
(``senior-define-and-use-basic-linear-transformations-dilations-of``,
``...composition-of-linear-transformations-and``, ``...inverses-of-linear-transformations-and``): none of the
three, at any tier, ever applies a translation -- every one of their renders is a PURE linear map through the
origin. This skill's every tier includes a nonzero translation component by construction
(``_sample_translation`` excludes ``(0, 0)``), which makes every rendered question here structurally
distinguishable from all three neighbours regardless of which linear part (rotation/reflection/dilation) also
appears..

Difficulty progression:
- easy:   linear part is a REFLECTION (x-axis, y-axis, y=x or y=-x) plus a
          stated translation.
- medium: linear part is a ROTATION (90/180/270 degrees about the origin)
          plus a stated translation.
- hard:   linear part is a general DILATION (x, y) -> (a*x, b*y), a and b
          any nonzero small integers (not just +-1), plus a stated
          translation.
- exam:   INVERT -- given the affine map and the IMAGE point, find the
          ORIGINAL point: subtract the translation FIRST, then apply the
          inverse of the linear part -- the reverse order from the forward
          map, and a genuinely different procedure from every lower tier and
          from the shipping inverses-of-linear-transformations skill (which
          only ever inverts a pure linear map with no translation to
          subtract).

CONSTRUCTION route (every tier): build the linear part's own well-known
matrix and apply it via ``_apply_matrix``, then add the translation.
INDEPENDENT CHECK: the family's direct coordinate rule (structurally
different code, not a re-derivation of the same matrix-multiply expression)
for the forward tiers; the family's own inverse matrix, applied backward
from the image point, for the exam tier.

The linear part ``M`` is NEVER the identity on any tier (reflections and
rotations by 90/180/270 are never the identity by construction; the dilation
draw explicitly excludes ``a == b == 1``) -- this matters for
``translates-before-linear``, which computes ``M*((x0,y0)+(tx,ty))`` instead
of ``M*(x0,y0)+(tx,ty)``: for ``M = I`` the two orders always coincide, and
even for some non-identity ``M`` (e.g. a reflection whose axis happens to
contain the translation vector) they can still coincide by chance, so every
draw is verified with ``check_pairwise_distinct``/``check_distinct`` before
the candidate is offered, and silently dropped (never crashed on) when it
collides on that particular draw.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    check_pairwise_distinct,
    fmt_signed,
    generator,
    phrase,
    rational_text,
    xy_solution,
)

_MAX_ATTEMPTS = 500
_IDENTITY = ((1, 0), (0, 1))

# Three genuinely different phrasings of "apply the linear part, then
# translate the result" -- every one keeps the nonzero-translation clause,
# so no wording collapses this into a pure-linear (no-translation) prompt
# that would collide with the shipping senior transformation skills (see
# module docstring). `mathgen.text_quality.prompt_shape` masks digits (and
# slot-bank words) only, so these need to differ in actual wording, not
# just numerals -- unlike the rotation `phrase_text`, which for a fixed
# family only ever varies by the masked-out degree/coefficient and so
# cannot supply the variety on its own (this was the medium tier's defect:
# rotation's three degrees all collapse to one shape once digits are
# masked).
_FORWARD_TEMPLATES = (
    "Start at the point ({x0}, {y0}). First {phrase_text}, then translate "
    "the result by ({tx}, {ty}). Find the image point.",
    "Consider the point ({x0}, {y0}). First {phrase_text}, then translate "
    "the result by ({tx}, {ty}) to reach the image point. What are its "
    "coordinates?",
    "You are given the point ({x0}, {y0}). First {phrase_text}, then "
    "translate the result by ({tx}, {ty}) to obtain the image point. State "
    "the image point.",
)

# ---------------------------------------------------------------------------
# Linear parts: matrix, direct coordinate rule, and (for exam) inverse matrix
# ---------------------------------------------------------------------------

_REFLECTION_MATRICES = {
    "x-axis": ((1, 0), (0, -1)),
    "y-axis": ((-1, 0), (0, 1)),
    "y=x": ((0, 1), (1, 0)),
    "y=-x": ((0, -1), (-1, 0)),
}
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


_ROTATION_MATRICES = {
    90: ((0, -1), (1, 0)),
    180: ((-1, 0), (0, -1)),
    270: ((0, 1), (-1, 0)),
}


def _rotation_direct(theta, x, y):
    if theta == 90:
        return (-y, x)
    if theta == 180:
        return (-x, -y)
    return (y, -x)  # theta == 270


_DILATION_POOL = tuple(v for v in range(-4, 5) if v != 0)  # {-4,...,-1,1,...,4}


def _apply_matrix(M, x, y):
    """Generic 2x2 matrix-vector product. Works for int or Fraction entries."""
    (a, b), (c, d) = M
    return a * x + b * y, c * x + d * y


def _matrix_text(M):
    (a, b), (c, d) = M
    return f"[[{a}, {b}], [{c}, {d}]]"


def _draw_linear(rng, family):
    """Returns (M, direct(x, y) -> (ix, iy), phrase_text) for the family."""
    if family == "reflection":
        line = rng.choice(("x-axis", "y-axis", "y=x", "y=-x"))
        M = _REFLECTION_MATRICES[line]
        direct = lambda x, y: _reflection_direct(line, x, y)
        phrase_text = f"reflect it in {_REFLECTION_LINE_PHRASE[line]}"
        return M, direct, phrase_text
    if family == "rotation":
        theta = rng.choice((90, 180, 270))
        M = _ROTATION_MATRICES[theta]
        direct = lambda x, y: _rotation_direct(theta, x, y)
        phrase_text = f"rotate it {theta} degrees counterclockwise about the origin"
        return M, direct, phrase_text
    # family == "dilation"
    a = rng.choice(_DILATION_POOL)
    b = rng.choice(_DILATION_POOL)
    while a == 1 and b == 1:  # would make M the identity
        a = rng.choice(_DILATION_POOL)
        b = rng.choice(_DILATION_POOL)
    M = ((a, 0), (0, b))
    direct = lambda x, y: (a * x, b * y)
    a_term = fmt_signed(a, "x")
    b_term = fmt_signed(b, "y")
    phrase_text = f"apply the dilation (x, y) -> ({a_term}, {b_term})"
    return M, direct, phrase_text


def _inverse_matrix(family, M):
    """M's own inverse, exact (Fraction where needed). Reflections are
    involutions (M^-1 = M); rotation matrices are orthogonal (M^-1 = M^T);
    a dilation's inverse is the reciprocal diagonal."""
    (a, b), (c, d) = M
    if family == "reflection":
        return M
    if family == "rotation":
        return ((a, c), (b, d))
    return ((Fraction(1, a), 0), (0, Fraction(1, d)))  # dilation


def _sample_point(rng, lo=-9, hi=9):
    while True:
        x, y = rng.randint(lo, hi), rng.randint(lo, hi)
        if (x, y) != (0, 0):
            return x, y


def _sample_translation(rng, lo=-9, hi=9):
    while True:
        tx, ty = rng.randint(lo, hi), rng.randint(lo, hi)
        if (tx, ty) != (0, 0):
            return tx, ty


def _offer(answer, value, offered, label):
    """Append `value` under `label` iff it genuinely differs from `answer`
    and from every candidate already offered -- see the module docstring's
    note on `translates-before-linear` colliding for some non-identity M."""
    if value == answer:
        return
    if value in offered.values():
        return
    check_distinct(answer, value, msg=f"{label} collided with the answer")
    for other_label, other in offered.items():
        check_pairwise_distinct(value, other, msg=f"{label} collided with {other_label}")
    offered[label] = value


def _fmt_pair(x, y):
    return f"x = {rational_text(x)}, y = {rational_text(y)}"


# ---------------------------------------------------------------------------
# Forward tiers: easy (reflection), medium (rotation), hard (dilation)
# ---------------------------------------------------------------------------


def _build_forward(rng, family):
    for _attempt in range(_MAX_ATTEMPTS):
        x0, y0 = _sample_point(rng)
        tx, ty = _sample_translation(rng)
        M, direct, phrase_text = _draw_linear(rng, family)
        check(M != _IDENTITY, "linear part must never be the identity")

        lx, ly = _apply_matrix(M, x0, y0)
        dlx, dly = direct(x0, y0)
        check((lx, ly) == (dlx, dly), "matrix route and direct route disagree on the linear part")

        x1, y1 = lx + tx, ly + ty
        answer = (x1, y1)

        offered: dict[str, tuple] = {}
        _offer(answer, (lx, ly), offered, "omits-translation")
        blx, bly = _apply_matrix(M, x0 + tx, y0 + ty)  # translates-before-linear
        _offer(answer, (blx, bly), offered, "translates-before-linear")
        _offer(answer, (lx - tx, ly - ty), offered, "wrong-sign-translation")

        if not offered:
            continue

        result = xy_solution(x1, y1)
        distractor_strs = [_fmt_pair(a, b) for a, b in offered.values()]
        result["distractors"] = [d for d in distractor_strs if d != result["answer_text"]]
        if not result["distractors"]:
            continue

        text = phrase(rng, _FORWARD_TEMPLATES, x0=x0, y0=y0, phrase_text=phrase_text, tx=tx, ty=ty)
        matrix_text = _matrix_text(M)
        result["text"] = text
        result["latex"] = rf"\text{{{text}}}"
        result["working"] = [
            f"The linear part's matrix is {matrix_text}.",
            f"Linear step: ({x0}, {y0}) -> ({lx}, {ly}).",
            f"Translate: ({lx}, {ly}) + ({tx}, {ty}) = ({x1}, {y1}).",
        ]
        return result
    raise RuntimeError(f"could not draw a usable {family} coordinate-transformation question")


def _build_easy(rng):
    return _build_forward(rng, "reflection")


def _build_medium(rng):
    return _build_forward(rng, "rotation")


def _build_hard(rng):
    return _build_forward(rng, "dilation")


# ---------------------------------------------------------------------------
# Exam tier: invert -- given the map and the image, find the original point
# ---------------------------------------------------------------------------


def _build_exam(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        family = rng.choice(("reflection", "rotation", "dilation"))
        x0, y0 = _sample_point(rng)
        tx, ty = _sample_translation(rng)
        M, direct, phrase_text = _draw_linear(rng, family)
        check(M != _IDENTITY, "linear part must never be the identity")
        if family == "dilation":
            a, d = M[0][0], M[1][1]
            check(a != 0 and d != 0, "dilation scale factors must be nonzero to be invertible")

        lx, ly = _apply_matrix(M, x0, y0)
        dlx, dly = direct(x0, y0)
        check((lx, ly) == (dlx, dly), "matrix route and direct route disagree on the linear part")
        x1, y1 = lx + tx, ly + ty  # the GIVEN image point

        # Independent check: invert via the family's own inverse matrix,
        # applied to the image point AFTER subtracting the translation --
        # the reverse order from construction, never re-using the forward
        # matrix-multiply expression.
        M_inv = _inverse_matrix(family, M)
        rec_x, rec_y = _apply_matrix(M_inv, x1 - tx, y1 - ty)
        check((rec_x, rec_y) == (x0, y0), "inversion route did not recover the original point")

        answer = (x0, y0)
        offered: dict[str, tuple] = {}
        inv_only = _apply_matrix(M_inv, x1, y1)  # forgets to subtract translation first
        _offer(answer, inv_only, offered, "inverts-linear-part-only")
        after_x, after_y = _apply_matrix(M_inv, x1, y1)
        sub_after = (after_x - tx, after_y - ty)  # subtracts raw translation AFTER inverting
        _offer(answer, sub_after, offered, "subtracts-translation-after-inverting")

        if not offered:
            continue

        result = xy_solution(x0, y0)
        distractor_strs = [_fmt_pair(a, b) for a, b in offered.values()]
        result["distractors"] = [d for d in distractor_strs if d != result["answer_text"]]
        if not result["distractors"]:
            continue

        matrix_text = _matrix_text(M)
        text = (
            f"A transformation is defined by: {phrase_text.capitalize()}, then translate the "
            f"result by ({tx}, {ty}). Applied to some starting point, this transformation "
            f"produces the image point ({x1}, {y1}). Find the original starting point."
        )
        result["text"] = text
        result["latex"] = rf"\text{{{text}}}"
        result["working"] = [
            f"The linear part's matrix is {matrix_text}, with inverse {_matrix_text(M_inv)}.",
            f"Undo the translation first: ({x1}, {y1}) - ({tx}, {ty}) = ({x1 - tx}, {y1 - ty}).",
            f"Apply the inverse linear map: ({x0}, {y0}).",
        ]
        return result
    raise RuntimeError("could not draw a usable exam-tier coordinate-transformation question")


@generator(
    "p14c_coordinate_transformation",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def coordinate_transformation(rng, tier="easy"):
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
