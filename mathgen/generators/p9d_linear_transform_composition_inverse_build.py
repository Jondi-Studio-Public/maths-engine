"""Level 13: composition of linear transformations (matrix products) and
inverses of linear transformations (matrix inverse).

Skill A -- senior-define-and-use-composition-of-linear-transformations-and (ACMSM057)
    topic: linear_transform_composition
    Grades the image point of a point under two STATED transformations applied
    in a stated order. Construction ALWAYS goes via the matrix PRODUCT
    (_matmul then _apply_matrix once); the independent check ALWAYS goes via
    applying each transformation's own direct coordinate rule SEQUENTIALLY, in
    that same order -- so the "and the corresponding matrix products" clause
    of the observable contract is exercised by construction, and the plain
    "composition of linear transformations" clause is exercised by the check.
    Never a separately typed/graded matrix answer.

Skill B -- senior-define-and-use-inverses-of-linear-transformations-and (ACMSM058)
    topic: linear_transform_inverse
    Grades recovery of an ORIGINAL point from its stated image under a stated
    transformation -- i.e. applying the transformation's inverse. Never a
    separately typed/graded inverse-matrix answer. Every tier's answer is
    reached by a different route from its own independent check (see each
    tier function's own docstring for which pairing and why).
"""

from __future__ import annotations

from ..core import check, generator, pick_distractors, xy_solution


# --------------------------------------------------------------------------
# Shared small helpers (module-local; see this file's own docstring for why
# these are not imported from the sibling module).
# --------------------------------------------------------------------------


def _apply_matrix(M, x, y):
    """Generic 2x2 matrix-vector product. M is ((a,b),(c,d))."""
    (a, b), (c, d) = M
    return a * x + b * y, c * x + d * y


def _matmul(M2, M1):
    """Generic 2x2-by-2x2 matrix product, M2 @ M1 (apply M1 first, then M2)."""
    (a2, b2), (c2, d2) = M2
    (a1, b1), (c1, d1) = M1
    return ((a2 * a1 + b2 * c1, a2 * b1 + b2 * d1), (c2 * a1 + d2 * c1, c2 * b1 + d2 * d1))


_DILATION_FACTORS = (-4, -3, -2, 2, 3, 4)
_ROTATION_ANGLES = (90, 180, 270)
_REFLECTION_LINES = ("x-axis", "y-axis", "y=x", "y=-x")
# Pairing used by the "wrong-line" distractor, matching the intended map.
_REFLECTION_PAIR = {"x-axis": "y=x", "y-axis": "y=-x", "y=x": "x-axis", "y=-x": "y-axis"}
_LINE_DESC = {
    "x-axis": "the x-axis",
    "y-axis": "the y-axis",
    "y=x": "the line y = x",
    "y=-x": "the line y = -x",
}


def _dilation_matrix(l1, l2):
    return (l1, 0), (0, l2)


def _rotation_matrix(theta):
    return {
        90: ((0, -1), (1, 0)),
        180: ((-1, 0), (0, -1)),
        270: ((0, 1), (-1, 0)),
    }[theta]


def _rotation_direct(theta, x, y):
    if theta == 90:
        return -y, x
    if theta == 180:
        return -x, -y
    if theta == 270:
        return y, -x
    raise ValueError(f"unsupported rotation angle {theta!r}")


def _reflection_matrix(line):
    return {
        "x-axis": ((1, 0), (0, -1)),
        "y-axis": ((-1, 0), (0, 1)),
        "y=x": ((0, 1), (1, 0)),
        "y=-x": ((0, -1), (-1, 0)),
    }[line]


def _reflection_direct(line, x, y):
    if line == "x-axis":
        return x, -y
    if line == "y-axis":
        return -x, y
    if line == "y=x":
        return y, x
    if line == "y=-x":
        return -y, -x
    raise ValueError(f"unsupported reflection line {line!r}")


def _reflection_degenerate(line, x, y):
    if line == "x-axis":
        return y == 0
    if line == "y-axis":
        return x == 0
    if line == "y=x":
        return x == y
    if line == "y=-x":
        return x == -y
    raise ValueError(f"unsupported reflection line {line!r}")


def _sample_dilation(rng):
    """Two distinct factors from the shared dilation table."""
    while True:
        l1, l2 = rng.choice(_DILATION_FACTORS), rng.choice(_DILATION_FACTORS)
        if l1 != l2:
            return l1, l2


def _sample_point(rng, *, exclude_line=None):
    """(x, y), each independently -9..9, resampled off the origin and (if
    ``exclude_line`` is given) off that reflection line's own degeneracy."""
    while True:
        x, y = rng.randint(-9, 9), rng.randint(-9, 9)
        if (x, y) == (0, 0):
            continue
        if exclude_line is not None and _reflection_degenerate(exclude_line, x, y):
            continue
        return x, y


def _sample_nonzero_point(rng, *, exclude_line=None):
    """(a, b), each independently drawn from -9..-1 union 1..9 (never 0), the
    tier-wide rule for Skill B -- keeps every distractor there cleanly
    distinct. Resampled off ``exclude_line``'s own degeneracy when given."""
    choices = [v for v in range(-9, 10) if v != 0]
    while True:
        a, b = rng.choice(choices), rng.choice(choices)
        if exclude_line is not None and _reflection_degenerate(exclude_line, a, b):
            continue
        return a, b


# --------------------------------------------------------------------------
# Skill A: composition of linear transformations (matrix products)
# --------------------------------------------------------------------------


def _verify_composition(
    result, M1, M2, x, y, msg="sequential direct-rule application disagrees with the matrix-product construction"
):
    """Independent check for every Skill A tier: apply M1 then M2 directly to
    (x, y) and confirm it matches the matrix-product construction. Plain
    tuple equality (not check_equivalent/check_distinct) -- these are bare
    integer tuples, not sympy expressions, as is usual for this shape of
    comparison."""
    p1 = _apply_matrix(M1, x, y)
    p2 = _apply_matrix(M2, *p1)
    check(p2 == result, msg)


def _finish_composition(ans, text, working, raw_candidates):
    result = xy_solution(*ans)
    candidates = [f"x = {cand[0]}, y = {cand[1]}" for cand in raw_candidates if cand != ans]
    result["distractors"] = pick_distractors(result["answer_text"], candidates)
    result["text"] = text
    result["latex"] = rf"\text{{{text}}}"
    result["working"] = working
    return result


def _composition_easy(rng):
    """Two dilations, applied in the FIXED stated order D1 then D2. Both
    dilations are per-axis scalars, so this tier's own two operations always
    commute with each other (D2 . D1 == D1 . D2) -- the reasoning act here is
    chaining two independent scalar maps and recognising that the matrix
    product of two diagonal matrices is itself diagonal, not resolving a
    stated order against a non-commuting pair (that first appears at medium).

    Measured on this tree: `only-first`/`only-second` coincide with each
    other (never with the answer) on about 0.3% of seeds, when the sampled
    point has `x == 0` or `y == 0` -- filtered per-candidate as required, but
    that alone can drop the pool to a single distractor, so the whole draw is
    resampled (never just the offending candidate) whenever the resulting
    pool is thinner than 2, the same discipline `matrices.py`'s own
    `determinant3` already uses to avoid a starved distractor pool."""
    while True:
        l1a, l2a = _sample_dilation(rng)
        l1b, l2b = _sample_dilation(rng)
        x, y = _sample_point(rng)
        M1, M2 = _dilation_matrix(l1a, l2a), _dilation_matrix(l1b, l2b)
        M = _matmul(M2, M1)
        ans = _apply_matrix(M, x, y)
        _verify_composition(ans, M1, M2, x, y)

        only_second = _apply_matrix(M2, x, y)
        only_first = _apply_matrix(M1, x, y)
        added_not_composed = ((l1a + l1b) * x, (l2a + l2b) * y)

        desc1 = f"the dilation (x, y) -> ({l1a}x, {l2a}y)"
        desc2 = f"the dilation (x, y) -> ({l1b}x, {l2b}y)"
        text = (
            f"First apply {desc1} to the point ({x}, {y}). "
            f"Then apply {desc2} to the result. Find the final image point."
        )
        working = [
            f"M1 = [[{l1a}, 0], [0, {l2a}]], M2 = [[{l1b}, 0], [0, {l2b}]].",
            f"Matrix product M = M2 M1 = [[{M[0][0]}, {M[0][1]}], [{M[1][0]}, {M[1][1]}]].",
            f"M (x, y) = ({ans[0]}, {ans[1]}); the sequential route agrees.",
        ]
        result = _finish_composition(ans, text, working, [only_second, only_first, added_not_composed])
        if len(result["distractors"]) >= 2:
            return result


def _composition_medium(rng):
    """A dilation and a rotation, order randomised and STATED in the question
    text.

    **The claim that "a dilation and a rotation about the origin here never commute" is wrong
    for theta == 180**: R(180) = -I is a scalar matrix, and a scalar matrix commutes with EVERY matrix (not
    only with another diagonal one) -- so whenever theta == 180 the stated order and the reversed order give
    the SAME image point, and `reversed-order` is (correctly) filtered out as a distractor before it is ever
    offered. Measured directly on this tree: reversed-order collides with the answer on ~41% of 200 sampled
    seeds. Checked directly (not merely inferred from the rate): EVERY collision is theta == 180 -- for a
    diagonal D = diag(l1, l2) with l1 != l2 and R(90)/R(270), D and R commute on a nonzero point only if (l1 -
    l2)*x = (l1 - l2)*y = 0, which the excluded origin already rules out, so theta in {90, 270} never
    collides; the ~41% (vs. the 1/3 chance of drawing theta == 180) is ordinary sampling variance, not a
    second collision source. This still never ships a wrong "correct" answer (the collision is always caught
    by the per-candidate filter before `pick_distractors` ever sees it) -- but on its own it was never
    observed to starve the pool below 2 distractors (only-first/only-second never collided with each other in
    2000 sampled seeds); the resample-if-thin guard below is kept anyway, defensively, matching every other
    tier in this module."""
    while True:
        l1, l2 = _sample_dilation(rng)
        theta = rng.choice(_ROTATION_ANGLES)
        x, y = _sample_point(rng)
        Md, Mr = _dilation_matrix(l1, l2), _rotation_matrix(theta)
        dilation_desc = f"the dilation (x, y) -> ({l1}x, {l2}y)"
        rotation_desc = f"a rotation of {theta} degrees about the origin"

        dilation_first = rng.random() < 0.5
        if dilation_first:
            M1, M2 = Md, Mr
            desc1, desc2 = dilation_desc, rotation_desc
        else:
            M1, M2 = Mr, Md
            desc1, desc2 = rotation_desc, dilation_desc

        M = _matmul(M2, M1)
        ans = _apply_matrix(M, x, y)
        _verify_composition(ans, M1, M2, x, y)

        Mrev = _matmul(M1, M2)
        reversed_order = _apply_matrix(Mrev, x, y)
        only_first = _apply_matrix(M1, x, y)
        only_second = _apply_matrix(M2, x, y)

        text = (
            f"First apply {desc1} to the point ({x}, {y}). "
            f"Then apply {desc2} to the result. Find the final image point."
        )
        working = [
            f"M1 = {list(map(list, M1))}, M2 = {list(map(list, M2))} (in the stated order).",
            f"Matrix product M = M2 M1 = {list(map(list, M))}.",
            f"M (x, y) = ({ans[0]}, {ans[1]}); the sequential route agrees.",
        ]
        result = _finish_composition(ans, text, working, [reversed_order, only_first, only_second])
        if len(result["distractors"]) >= 2:
            return result


def _composition_hard(rng):
    """A rotation and a reflection (in a line through the origin), order
    randomised and STATED. The point is resampled off the reflection line's
    own degeneracy. This tier's reasoning act is selecting the correct one of
    THREE rotation matrices and the correct one of FOUR reflection matrices
    (a bigger, more error-prone combinatorial selection than either easy's or
    medium's two-way choice) and still resolving the stated, non-commuting
    order correctly.

    As at medium, theta == 180 makes R(180) = -I commute with every
    reflection matrix too (checked directly: every measured collision here
    is also theta == 180 -- R(90)/R(270) never collide with any reflection
    on a point excluded from that line), so `reversed-order` collides with
    the answer on roughly a third of seeds (measured ~37%, ordinary sampling
    variance around the 1/3 chance of drawing theta == 180) -- always caught
    by the per-candidate filter. Unlike medium, this can starve the pool here: when
    `reversed-order` is filtered AND `only-first`/`only-second` also happen
    to coincide (both measured, on ~2.4% of seeds), only one distractor is
    left. The resample-if-thin guard below (same discipline as easy's own)
    is load-bearing for this tier, not merely defensive."""
    while True:
        theta = rng.choice(_ROTATION_ANGLES)
        line = rng.choice(_REFLECTION_LINES)
        x, y = _sample_point(rng, exclude_line=line)
        Mr, Mf = _rotation_matrix(theta), _reflection_matrix(line)
        rotation_desc = f"a rotation of {theta} degrees about the origin"
        reflection_desc = f"a reflection in {_LINE_DESC[line]}"

        rotation_first = rng.random() < 0.5
        if rotation_first:
            M1, M2 = Mr, Mf
            desc1, desc2 = rotation_desc, reflection_desc
        else:
            M1, M2 = Mf, Mr
            desc1, desc2 = reflection_desc, rotation_desc

        M = _matmul(M2, M1)
        ans = _apply_matrix(M, x, y)
        _verify_composition(ans, M1, M2, x, y)

        Mrev = _matmul(M1, M2)
        reversed_order = _apply_matrix(Mrev, x, y)
        only_first = _apply_matrix(M1, x, y)
        only_second = _apply_matrix(M2, x, y)

        text = (
            f"First apply {desc1} to the point ({x}, {y}). "
            f"Then apply {desc2} to the result. Find the final image point."
        )
        working = [
            f"M1 = {list(map(list, M1))}, M2 = {list(map(list, M2))} (in the stated order).",
            f"Matrix product M = M2 M1 = {list(map(list, M))}.",
            f"M (x, y) = ({ans[0]}, {ans[1]}); the sequential route agrees.",
        ]
        result = _finish_composition(ans, text, working, [reversed_order, only_first, only_second])
        if len(result["distractors"]) >= 2:
            return result


@generator(
    "linear_transform_composition",
    level=13,
    presets={
        "easy": dict(tier="dilation_dilation"),
        "medium": dict(tier="dilation_rotation"),
        "hard": dict(tier="rotation_reflection"),
    },
)
def linear_transform_composition(rng, tier="dilation_dilation"):
    if tier == "dilation_dilation":
        return _composition_easy(rng)
    if tier == "dilation_rotation":
        return _composition_medium(rng)
    if tier == "rotation_reflection":
        return _composition_hard(rng)
    raise ValueError(f"unknown tier {tier!r}")


# --------------------------------------------------------------------------
# Skill B: inverses of linear transformations (matrix inverse)
# --------------------------------------------------------------------------


def _verify_dilation_inverse(
    a_ans,
    b_ans,
    l1,
    l2,
    p,
    q,
    msg="forward re-application of the dilation to the division-constructed answer disagrees with the stated image",
):
    check((l1 * a_ans, l2 * b_ans) == (p, q), msg)


def _verify_rotation_inverse(
    a_ans,
    b_ans,
    theta,
    p,
    q,
    msg="forward rotation of the inverse-rule-constructed answer disagrees with the stated image",
):
    check(_rotation_direct(theta, a_ans, b_ans) == (p, q), msg)


def _verify_reflection_inverse(
    a_ans, b_ans, line, p, q, msg="the generic matrix-vector product disagrees with the direct-rule-constructed answer"
):
    check(_apply_matrix(_reflection_matrix(line), p, q) == (a_ans, b_ans), msg)


def _finish_inverse(a_ans, b_ans, text, working, candidates):
    result = xy_solution(a_ans, b_ans)
    filtered = []
    for cand in candidates:
        if cand != (a_ans, b_ans):
            filtered.append(f"x = {cand[0]}, y = {cand[1]}")
    result["distractors"] = pick_distractors(result["answer_text"], filtered)
    result["text"] = text
    result["latex"] = rf"\text{{{text}}}"
    result["working"] = working
    return result


def _inverse_easy(rng):
    """Dilation inverse. Construction: DIVIDE each image coordinate by its own scale factor (the inverse of a
    diagonal matrix is diagonal with reciprocal entries). Independent check: forward-MULTIPLY the computed
    answer by the same factors and confirm it reproduces the stated image -- division to construct,
    multiplication to verify, a genuinely different route. Measured at 0/2000 thin (<2 distractor) draws; the
    resample-if-thin guard is kept anyway, defensively, matching every other tier in this module."""
    while True:
        l1, l2 = _sample_dilation(rng)
        a, b = _sample_nonzero_point(rng)
        p, q = l1 * a, l2 * b
        a_ans, b_ans = p // l1, q // l2
        _verify_dilation_inverse(a_ans, b_ans, l1, l2, p, q)

        forward_not_inverse = (l1 * p, l2 * q)
        if p % l2 == 0 and q % l1 == 0:
            swapped_lambda = (p // l2, q // l1)
        else:
            swapped_lambda = (l2 * p, l1 * q)
        sign_flip = (-a_ans, b_ans)

        text = f"A dilation (x, y) -> ({l1}x, {l2}y) maps a point to the image ({p}, {q}). Find the original point."
        working = [
            f"Forward rule: (x, y) -> ({l1}x, {l2}y).",
            f"Invert by dividing each coordinate by its own scale factor: "
            f"({p} / {l1}, {q} / {l2}) = ({a_ans}, {b_ans}).",
            f"Check: ({l1} x {a_ans}, {l2} x {b_ans}) = ({p}, {q}); the forward map agrees.",
        ]
        result = _finish_inverse(a_ans, b_ans, text, working, [forward_not_inverse, swapped_lambda, sign_flip])
        if len(result["distractors"]) >= 2:
            return result


def _inverse_medium(rng):
    """Rotation inverse. theta is drawn ONLY from {90, 270} -- never 180,
    since a 180 degree rotation is its own inverse and would make
    "forward-not-inverse" coincidentally correct. Construction: apply the
    INVERSE rotation (360 - theta) directly. Independent check: forward-
    rotate the computed answer by theta and confirm it reproduces the stated
    image -- inverse rule to construct, forward rule to verify.

    Measured on this tree: whenever the recovered answer happens to satisfy `a_ans == -b_ans` (about 10% of
    seeds, since `swapped_coords` and `sign_flip_both` are then numerically identical -- `(b, a) == (-a, -b)`
    exactly when `a == -b`), the candidate pool thins to 1 after filtering. The resample-if-thin guard below
    is load-bearing for this tier, not merely defensive."""
    while True:
        theta = rng.choice((90, 270))
        a, b = _sample_nonzero_point(rng)
        p, q = _rotation_direct(theta, a, b)
        inv_theta = 360 - theta
        a_ans, b_ans = _rotation_direct(inv_theta, p, q)
        _verify_rotation_inverse(a_ans, b_ans, theta, p, q)

        forward_not_inverse = _rotation_direct(theta, p, q)
        swapped_coords = (b_ans, a_ans)
        sign_flip_both = (-a_ans, -b_ans)

        text = (
            f"A rotation of {theta} degrees about the origin maps a point to "
            f"the image ({p}, {q}). Find the original point."
        )
        working = [
            f"Forward rule: rotate by {theta} degrees about the origin.",
            f"Invert by rotating by 360 - {theta} = {inv_theta} degrees: ({p}, {q}) -> ({a_ans}, {b_ans}).",
            f"Check: rotating ({a_ans}, {b_ans}) by {theta} degrees gives ({p}, {q}); the forward rule agrees.",
        ]
        result = _finish_inverse(a_ans, b_ans, text, working, [forward_not_inverse, swapped_coords, sign_flip_both])
        if len(result["distractors"]) >= 2:
            return result


def _inverse_hard(rng):
    """Reflection inverse (self-inverse). Construction: apply the DIRECT
    coordinate rule to the stated image (a reflection undoes itself).
    Independent check: apply the GENERIC matrix-vector product
    (_apply_matrix with the same line's matrix) to the stated image and
    confirm it reproduces the same answer -- direct formula to construct,
    generic matrix route to check, the OPPOSITE pairing from this skill's own
    easy/medium tiers (which both paired "formula to construct, forward-apply
    to check"); neither leg here is a forward/inverse round trip, so the
    direct-rule-vs-matrix split is what supplies genuine independence.
    Measured at 0/2000 thin (<2 distractor) draws; the resample-if-thin guard
    is kept anyway, defensively, matching every other tier in this module."""
    while True:
        line = rng.choice(_REFLECTION_LINES)
        a, b = _sample_nonzero_point(rng, exclude_line=line)
        p, q = _reflection_direct(line, a, b)
        a_ans, b_ans = _reflection_direct(line, p, q)
        _verify_reflection_inverse(a_ans, b_ans, line, p, q)

        point_reflection = (-p, -q)
        wrong_line = _reflection_direct(_REFLECTION_PAIR[line], p, q)
        identity = (p, q)

        text = f"A reflection in {_LINE_DESC[line]} maps a point to the image ({p}, {q}). Find the original point."
        working = [
            f"A reflection in {_LINE_DESC[line]} is its own inverse.",
            f"Reflecting the image ({p}, {q}) in the same line gives ({a_ans}, {b_ans}).",
            f"Check via the reflection's own matrix: M ({p}, {q}) = ({a_ans}, {b_ans}); the direct rule agrees.",
        ]
        result = _finish_inverse(a_ans, b_ans, text, working, [point_reflection, wrong_line, identity])
        if len(result["distractors"]) >= 2:
            return result


@generator(
    "linear_transform_inverse",
    level=13,
    presets={
        "easy": dict(tier="dilation"),
        "medium": dict(tier="rotation"),
        "hard": dict(tier="reflection"),
    },
)
def linear_transform_inverse(rng, tier="dilation"):
    if tier == "dilation":
        return _inverse_easy(rng)
    if tier == "rotation":
        return _inverse_medium(rng)
    if tier == "reflection":
        return _inverse_hard(rng)
    raise ValueError(f"unknown tier {tier!r}")
