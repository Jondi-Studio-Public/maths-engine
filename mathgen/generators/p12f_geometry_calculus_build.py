"""Gram-Schmidt orthonormalisation (2D) + 2x2 Hessian evaluation.

Two skills, one module:

- ``p12f_gram_schmidt_2d`` (``ucv1-gram-schmidt-2d``, ``answer: ordered_list``)
  -- one Gram-Schmidt step on two independent integer 2D vectors, reporting
  the resulting orthonormal pair as an exact-Fraction 4-tuple
  (e1_x, e1_y, e2_x, e2_y).
- ``p12f_hessian_2x2`` (``ucv1-hessian-2x2``, ``answer: ordered_list``) --
  the 2x2 Hessian of a polynomial map R^2 -> R at a point, reported as the
  row-major integer 4-tuple (fxx, fxy, fyx, fyy).

Both use ``ordered_list`` rather than a plain ``list`` answer:
``grade_list`` (mathgen/grading/typed.py) is order-insensitive multiset comparison and
would false-accept a positionally-wrong matrix/vector answer for a shape
with a structurally repeated component (a Hessian's fxy at two positions);
``grade_ordered_list`` is positional, exact-length, and compares every
numeric item by EXACT Fraction/int equality -- so every value returned here
is a plain Python ``int`` or ``fractions.Fraction``, never a sympy type and
never a float.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

import sympy

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
    rational_text,
)

_MAX_ATTEMPTS = 500

# ---------------------------------------------------------------------------
# Skill 1: ucv1-gram-schmidt-2d
# ---------------------------------------------------------------------------

# (wide) triples, easy's answer is FULLY determined by u alone (n's sign is
# fixed positive at easy, so e2 never flips direction) -- a learner who
# memorised 2 answers passed ~50% of easy draws, and exam (4 triples, n's
# sign also fixed by the exam's own "positive e2_x" criterion) similarly sat
# near 25% per answer. Widened to 12 (easy) / 16 (wide) primitive triples,
# every one still clearing the |a-b|/c >= 0.06 gate (excludes near-isosceles
# triples such as (20, 21, 29), whose legs sit only 0.034 apart), computed
# directly below. Component ORDER is also now varied at draw time (see
# `_p12f_gram_schmidt_forward`/`_p12f_gram_schmidt_exam`): u=(a,b) and
# u=(b,a) are genuinely different vectors with different (e1, e2), so this
# doubles the achievable answer pool on top of the wider triple set, without
# touching the sign-based distinction (easy/medium stay non-negative; hard
# stays "exactly one negative component") the existing tests already pin.
#   (3,4,5)->0.2  (5,12,13)->0.538  (8,15,17)->0.412  (7,24,25)->0.68
#   (9,40,41)->0.756  (12,35,37)->0.622  (11,60,61)->0.803  (28,45,53)->0.321
#   (33,56,65)->0.354  (16,63,65)->0.723  (13,84,85)->0.835  (36,77,85)->0.482
#   (39,80,89)->0.46  (65,72,97)->0.072  (20,99,101)->0.782  (48,55,73)->0.096
_EASY_TRIPLES = (
    (3, 4, 5),
    (5, 12, 13),
    (8, 15, 17),
    (7, 24, 25),
    (9, 40, 41),
    (12, 35, 37),
    (11, 60, 61),
    (28, 45, 53),
    (33, 56, 65),
    (16, 63, 65),
    (13, 84, 85),
    (36, 77, 85),
)
_WIDE_TRIPLES = _EASY_TRIPLES + (
    (39, 80, 89),
    (65, 72, 97),
    (20, 99, 101),
    (48, 55, 73),
)
_SIGNED_POOL = tuple(v for v in range(-4, 5) if v != 0)


def _u_perp(ux, uy):
    return -uy, ux


def _e1e2(ux, uy, c, n):
    """e1 = u/|u|; e2 = sign(n) * u_perp/|u| -- exact, denominator c.

    Algebraically exact because the Gram-Schmidt orthogonal component of
    v = m*u + n*u_perp against u is precisely n*u_perp (the m*u part is
    entirely parallel to u and vanishes under projection).
    """
    e1 = (Fraction(ux, c), Fraction(uy, c))
    px, py = _u_perp(ux, uy)
    sign = 1 if n > 0 else -1
    e2 = (Fraction(sign * px, c), Fraction(sign * py, c))
    return e1, e2


def _independent_e2(ux, uy, vx, vy):
    """Textbook projection route, deliberately NOT the n*u_perp shortcut
    `_e1e2` uses: w2 = v - (u.v/u.u)*u, computed with exact Fraction
    arithmetic throughout; e2 = w2/|w2|, with sqrt used only at the very end
    to confirm |w2|^2 is an exact rational square (it must be (n*c)^2 by
    construction) -- raises rather than silently coercing if it is not."""
    dot_uv = ux * vx + uy * vy
    dot_uu = ux * ux + uy * uy
    proj = Fraction(dot_uv, dot_uu)
    w2x = Fraction(vx) - proj * ux
    w2y = Fraction(vy) - proj * uy
    mag_sq = w2x * w2x + w2y * w2y
    root_num = isqrt(mag_sq.numerator)
    root_den = isqrt(mag_sq.denominator)
    check(
        root_num * root_num == mag_sq.numerator and root_den * root_den == mag_sq.denominator,
        "gram-schmidt: |w2|^2 is not an exact rational square",
    )
    mag = Fraction(root_num, root_den)
    check(mag != 0, "gram-schmidt: orthogonal component vanished")
    return w2x / mag, w2y / mag


def _fmt4(values) -> str:
    return ", ".join(rational_text(v) for v in values)


def _raw_v_distractor(ux, uy, m, n):
    """uses-raw-v-for-e2: v/|v| instead of the orthogonalised-then-normalised
    w2/|w2| -- skips the projection subtraction entirely. Only EXACT (and
    therefore only offered) when m^2+n^2 is itself a perfect square, so
    |v| = c*sqrt(m^2+n^2) is an integer; otherwise this misconception would
    ship an irrational decimal next to three exact fractions, a shape tell,
    so it is skipped rather than approximated."""
    vx = m * ux - n * uy
    vy = m * uy + n * ux
    mag_sq = vx * vx + vy * vy
    root = isqrt(mag_sq)
    if root == 0 or root * root != mag_sq:
        return None
    return (Fraction(vx, root), Fraction(vy, root))


def _offer(answer, offered, candidates, candidate, label):
    """Append `candidate` iff it is a genuinely different 4-tuple from the
    answer and from every candidate already offered -- checked exactly
    (Fraction equality), unconditionally."""
    if candidate == answer or candidate in offered:
        return
    check_distinct(answer, candidate, msg=f"{label} collided with the answer")
    check_pairwise_distinct(*offered, candidate, msg=f"{label} collided with another option")
    offered.append(candidate)
    candidates.append(candidate)


def _misconceptions(ux, uy, c, m, n, e1, e2):
    """forgets-to-normalise-e2, uses-raw-v-for-e2 (conditional, see above)
    and wrong-rotation-direction-for-u_perp -- every candidate 4-tuple
    (e1_x, e1_y, e2_x', e2_y') offered only when distinct, exact-Fraction."""
    answer = (*e1, *e2)
    offered = [answer]
    candidates = []

    px, py = _u_perp(ux, uy)
    unnorm = (Fraction(n * px), Fraction(n * py))
    _offer(answer, offered, candidates, (*e1, *unnorm), "forgets-to-normalise-e2")

    raw = _raw_v_distractor(ux, uy, m, n)
    if raw is not None:
        _offer(answer, offered, candidates, (*e1, *raw), "uses-raw-v-for-e2")

    sign = 1 if n > 0 else -1
    wrong_rot = (Fraction(sign * uy, c), Fraction(sign * -ux, c))
    _offer(answer, offered, candidates, (*e1, *wrong_rot), "wrong-rotation-direction-for-u_perp")

    return offered, candidates


_EASY_PATTERNS = [
    "For u = ({ux}, {uy}) and v = ({vx}, {vy}), apply one Gram-Schmidt step: "
    "orthonormalise u, then orthogonalise and normalise v against u. Give e1_x, e1_y, "
    "e2_x, e2_y in that order, as exact fractions.",
    "Apply one step of the Gram-Schmidt process to u = ({ux}, {uy}) and v = ({vx}, {vy}). "
    "Give e1_x, e1_y, e2_x, e2_y in that order, as exact fractions.",
    "Given u = ({ux}, {uy}) and v = ({vx}, {vy}), find the orthonormal pair produced by "
    "one Gram-Schmidt step. Give e1_x, e1_y, e2_x, e2_y in that order, as exact fractions.",
]

_HARD_PATTERNS = [
    "u = ({ux}, {uy}) has a negative component -- track its sign carefully. Apply one "
    "Gram-Schmidt step to u and v = ({vx}, {vy}). Give e1_x, e1_y, e2_x, e2_y in that order, "
    "as exact fractions.",
    "Orthonormalise u = ({ux}, {uy}) (note the sign of one component) and then "
    "v = ({vx}, {vy}) against it, using one Gram-Schmidt step. Give e1_x, e1_y, e2_x, e2_y "
    "in that order, as exact fractions.",
    "One component of u = ({ux}, {uy}) is negative. Apply one Gram-Schmidt step to u and "
    "v = ({vx}, {vy}). Give e1_x, e1_y, e2_x, e2_y in that order, as exact fractions.",
]

_EXAM_PATTERNS = [
    "For u = ({ux}, {uy}), two candidate vectors are proposed: v_A = ({vax}, {vay}) and "
    "v_B = ({vbx}, {vby}). Exactly one of v_A, v_B produces, after one Gram-Schmidt step "
    "against u, an e2 with a positive x-component. Using that vector, give e1_x, e1_y, e2_x, "
    "e2_y in that order, as exact fractions.",
    "u = ({ux}, {uy}) is fixed. Only one of v_A = ({vax}, {vay}) or v_B = ({vbx}, {vby}) "
    "gives an orthonormal e2 with positive x-component after one Gram-Schmidt step. Find "
    "that pair and give e1_x, e1_y, e2_x, e2_y in that order, as exact fractions.",
    "Given u = ({ux}, {uy}) and two candidates v_A = ({vax}, {vay}), v_B = ({vbx}, {vby}), "
    "verify which candidate produces an e2 with positive x-component under one Gram-Schmidt "
    "step, then give e1_x, e1_y, e2_x, e2_y in that order, as exact fractions, for that candidate.",
]


def _p12f_gram_schmidt_forward(rng, *, tier):
    triples = _EASY_TRIPLES if tier == "easy" else _WIDE_TRIPLES
    for _attempt in range(_MAX_ATTEMPTS):
        a, b, c = rng.choice(triples)
        if rng.random() < 0.5:
            a, b = b, a  # component order swap -- u=(a,b) and u=(b,a) give
            # genuinely different (e1, e2), doubling the achievable answer
            # pool per triple without touching either leg's sign.
        if tier == "hard":
            ux, uy = (-a, b) if rng.random() < 0.5 else (a, -b)
        else:
            ux, uy = a, b

        if tier == "easy":
            m = rng.randint(1, 3)
            n = rng.randint(1, 3)
        else:
            m = rng.choice(_SIGNED_POOL)
            n = rng.choice(_SIGNED_POOL)

        vx = m * ux - n * uy
        vy = m * uy + n * ux

        e1, e2 = _e1e2(ux, uy, c, n)

        # Orthonormality, exact:
        check(e1[0] ** 2 + e1[1] ** 2 == 1, "gram-schmidt: e1 is not a unit vector")
        check(e2[0] ** 2 + e2[1] ** 2 == 1, "gram-schmidt: e2 is not a unit vector")
        check(e1[0] * e2[0] + e1[1] * e2[1] == 0, "gram-schmidt: e1, e2 are not orthogonal")

        # Independent check, genuinely different route from construction:
        # textbook projection applied directly to the presented v, never
        # reusing the n*u_perp shortcut.
        e2_check = _independent_e2(ux, uy, vx, vy)
        check_equivalent(e2, e2_check, "gram-schmidt: independent e2 check failed")

        answer = (*e1, *e2)
        answer_text = _fmt4(answer)
        offered, candidates = _misconceptions(ux, uy, c, m, n, e1, e2)

        swapped = (*e2, *e1)
        _offer(answer, offered, candidates, swapped, "swapped-e1-e2")

        distractors = pick_distractors(answer_text, [_fmt4(cand) for cand in candidates])
        if len(distractors) == 0:
            continue

        patterns = _HARD_PATTERNS if tier == "hard" else _EASY_PATTERNS
        text = phrase(rng, patterns, ux=ux, uy=uy, vx=vx, vy=vy)
        working = [
            f"e1 = u/|u| = ({ux}/{c}, {uy}/{c}).",
            f"n = {n}, so e2 = sign(n) x u_perp/|u| = ({e2[0]}, {e2[1]}).",
        ]
        return dict(
            text=text,
            latex=text,
            answer=[*answer],
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError(f"gram-schmidt {tier}: could not draw a usable question")


def _p12f_gram_schmidt_exam(rng):
    """Verification-then-compute: two candidate v vectors are offered for a
    fixed u; exactly one produces an e2 with a positive x-component (since
    uy > 0 always here, that is always the candidate built with n < 0). The
    learner must identify the consistent candidate before reporting the
    orthonormal pair -- a genuinely different decision from every forward
    tier, not a bigger version of the same problem."""
    for _attempt in range(_MAX_ATTEMPTS):
        a, b, c = rng.choice(_WIDE_TRIPLES)
        if rng.random() < 0.5:
            a, b = b, a  # component order swap (see _p12f_gram_schmidt_forward);
            # uy = b stays positive either way, so the exam tier's own
            # "uy > 0 -> n < 0 gives the positive-e2_x candidate" logic is
            # unaffected by which leg is assigned to x vs y.
        ux, uy = a, b
        m = rng.choice(_SIGNED_POOL)
        n_mag = rng.randint(1, 4)
        n_true, n_wrong = -n_mag, n_mag  # e2_x = sign(n)*(-uy)/c, uy > 0 -> n<0 gives e2_x>0

        e1, e2_true = _e1e2(ux, uy, c, n_true)
        _, e2_wrong = _e1e2(ux, uy, c, n_wrong)
        check(e2_true[0] > 0, "exam: consistent candidate should have positive e2 x-component")
        check(e2_wrong[0] < 0, "exam: inconsistent candidate should have negative e2 x-component")

        v_true = (m * ux - n_true * uy, m * uy + n_true * ux)
        v_wrong = (m * ux - n_wrong * uy, m * uy + n_wrong * ux)
        if v_true == v_wrong:
            continue

        # Independent check: recompute e2_true via the textbook projection
        # formula directly on v_true (never the n*u_perp shortcut), and
        # confirm the inconsistent candidate's own projection really fails
        # the stated criterion -- the fact the learner must actually verify.
        e2_true_check = _independent_e2(ux, uy, *v_true)
        check_equivalent(e2_true, e2_true_check, "exam: independent e2 check failed")
        e2_wrong_check = _independent_e2(ux, uy, *v_wrong)
        check(
            e2_wrong_check[0] < 0, "exam: inconsistent candidate's own projection unexpectedly satisfies the criterion"
        )

        answer = (*e1, *e2_true)
        answer_text = _fmt4(answer)
        offered, candidates = _misconceptions(ux, uy, c, m, n_true, e1, e2_true)

        # uses-wrong-candidate: the (e1, e2) pair produced by the vector that
        # does NOT satisfy the stated criterion.
        wrong_pair = (*e1, *e2_wrong)
        _offer(answer, offered, candidates, wrong_pair, "uses-wrong-candidate")

        distractors = pick_distractors(answer_text, [_fmt4(cand) for cand in candidates])
        if len(distractors) == 0:
            continue

        va, vb = (v_true, v_wrong) if rng.random() < 0.5 else (v_wrong, v_true)
        text = phrase(rng, _EXAM_PATTERNS, ux=ux, uy=uy, vax=va[0], vay=va[1], vbx=vb[0], vby=vb[1])
        working = [
            f"e2's x-component is sign(n) x (-{uy})/{c}; since {uy} > 0, only n < 0 gives a "
            f"positive x-component, so the vector built with n = {n_true} is consistent.",
            f"e1 = ({e1[0]}, {e1[1]}), e2 = ({e2_true[0]}, {e2_true[1]}).",
        ]
        return dict(
            text=text,
            latex=text,
            answer=[*answer],
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("gram-schmidt exam: could not draw a usable question")


@generator(
    "p12f_gram_schmidt_2d",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12f_gram_schmidt_2d(rng, tier="easy"):
    if tier == "exam":
        return _p12f_gram_schmidt_exam(rng)
    if tier in ("easy", "medium", "hard"):
        return _p12f_gram_schmidt_forward(rng, tier=tier)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Skill 2: ucv1-hessian-2x2
# ---------------------------------------------------------------------------

_x, _y = sympy.symbols("x y")


def _poly_display(expr) -> str:
    return str(sympy.expand(expr)).replace("**", "^").replace("*", "")


def _hessian_sympy(A, B, C, D, E, x0, y0):
    """Construction's route: sympy differentiates the polynomial twice and
    substitutes the point."""
    f = A * _x**2 + B * _x * _y + C * _y**2 + D * _x**3 + E * _x**2 * _y
    fxx = int(sympy.diff(f, _x, _x).subs({_x: x0, _y: y0}))
    fxy = int(sympy.diff(f, _x, _y).subs({_x: x0, _y: y0}))
    fyx = int(sympy.diff(f, _y, _x).subs({_x: x0, _y: y0}))
    fyy = int(sympy.diff(f, _y, _y).subs({_x: x0, _y: y0}))
    return f, fxx, fxy, fyx, fyy


def _hessian_closed_form(A, B, C, D, E, x0, y0):
    """The independent check's route: hand-derived closed-form coefficient
    arithmetic, distinct from sympy.diff. For
    f = A x^2 + B xy + C y^2 + D x^3 + E x^2 y:
        f_xx = 2A + 6D x0 + 2E y0, f_xy = f_yx = B + 2E x0, f_yy = 2C."""
    fxx = 2 * A + 6 * D * x0 + 2 * E * y0
    fxy = B + 2 * E * x0
    fyy = 2 * C
    return fxx, fxy, fxy, fyy


def _first_partials(A, B, C, D, E, x0, y0):
    """f_x, f_y at the point -- closed form, used only to build the
    first-derivative-instead-of-second distractor."""
    fx = 2 * A * x0 + B * y0 + 3 * D * x0**2 + 2 * E * x0 * y0
    fy = B * x0 + 2 * C * y0 + E * x0**2
    return fx, fy


_EASY_HESS_PATTERNS = [
    "For f(x, y) = {f}, evaluate the Hessian matrix at ({x0}, {y0}). Give fxx, fxy, fyx, fyy, "
    "reading across each row of the matrix, in that order.",
    "Find the Hessian of f(x, y) = {f} at the point ({x0}, {y0}). Give fxx, fxy, fyx, fyy, "
    "reading across each row of the matrix, in that order.",
    "The function f(x, y) = {f} has a Hessian matrix. Evaluate it at ({x0}, {y0}), giving "
    "fxx, fxy, fyx, fyy, reading across each row of the matrix, in that order.",
]

_HARD_HESS_PATTERNS = [
    "For f(x, y) = {f}, evaluate the Hessian at the point ({x0}, {y0}) -- note the negative "
    "coordinate. Give fxx, fxy, fyx, fyy, reading across each row of the matrix, in that order.",
    "The Hessian of f(x, y) = {f} depends on the point; evaluate it at ({x0}, {y0}), where one "
    "coordinate is negative. Give fxx, fxy, fyx, fyy, reading across each row of the matrix, "
    "in that order.",
    "f(x, y) = {f}. Substitute the point ({x0}, {y0}) -- watch the sign -- into the Hessian. "
    "Give fxx, fxy, fyx, fyy, reading across each row of the matrix, in that order.",
]

_EXAM_HESS_PATTERNS = [
    "f(x, y) = {f_display}, where the coefficient E of the x^2 y term is unknown. At the point "
    "({x0}, {y0}), f_xy = {fxy_given}. Recover E, then give fxx, fxy, fyx, fyy at that point, "
    "reading across each row of the matrix, in that order.",
    "For f(x, y) = {f_display} with unknown coefficient E on the x^2 y term, f_xy at "
    "({x0}, {y0}) equals {fxy_given}. Find E first, then give fxx, fxy, fyx, fyy at that point, "
    "reading across each row of the matrix, in that order.",
    "The coefficient E of the x^2 y term in f(x, y) = {f_display} is unknown, but f_xy = "
    "{fxy_given} at ({x0}, {y0}). Determine E, then give fxx, fxy, fyx, fyy at that point, "
    "reading across each row of the matrix, in that order.",
]


def _hessian_misconceptions(A, B, C, D, E, x0, y0, answer):
    offered = [answer]
    candidates = []
    fxx, fxy, fyx, fyy = answer

    diag_swap = (fyy, fxy, fyx, fxx)
    _offer(answer, offered, candidates, diag_swap, "diagonal-swap")

    if fxy != 0:
        sign_err = (fxx, -fxy, -fyx, fyy)
        _offer(answer, offered, candidates, sign_err, "off-diagonal-sign-error")

    fx, fy = _first_partials(A, B, C, D, E, x0, y0)
    first_deriv = (fx, fy, fx, fy)
    _offer(answer, offered, candidates, first_deriv, "first-derivative-instead-of-second")

    if D != 0 or E != 0:
        ignored = (2 * A, B, B, 2 * C)
        _offer(answer, offered, candidates, ignored, "point-ignored")

    return offered, candidates


def _hessian_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        A = rng.randint(-5, 5) or 1
        C = rng.randint(-5, 5) or 2
        B = rng.randint(-5, 5)
        if A == C:
            continue
        x0, y0 = rng.randint(-4, 4), rng.randint(-4, 4)

        _f, fxx, fxy, fyx, fyy = _hessian_sympy(A, B, C, 0, 0, x0, y0)
        check(fxy == fyx, "hessian: mixed partials are not equal")
        closed = _hessian_closed_form(A, B, C, 0, 0, x0, y0)
        check_equivalent((fxx, fxy, fyx, fyy), closed, "hessian easy: independent check failed")

        answer = (fxx, fxy, fyx, fyy)
        answer_text = _fmt4(answer)
        offered, candidates = _hessian_misconceptions(A, B, C, 0, 0, x0, y0, answer)
        distractors = pick_distractors(answer_text, [_fmt4(c) for c in candidates])
        if len(distractors) == 0:
            continue

        f_expr = A * _x**2 + B * _x * _y + C * _y**2
        f_display = _poly_display(f_expr)
        text = phrase(rng, _EASY_HESS_PATTERNS, f=f_display, x0=x0, y0=y0)
        working = [
            f"f_xx = {fxx}, f_xy = f_yx = {fxy}, f_yy = {fyy} (constant here; the point "
            f"does not change the Hessian at this tier)."
        ]
        return dict(
            text=text,
            latex=text,
            answer=[*answer],
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("hessian easy: could not draw a usable question")


def _hessian_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        A = rng.randint(-5, 5) or 1
        C = rng.randint(-5, 5) or 2
        B = rng.randint(-5, 5)
        D = rng.choice((-3, -2, -1, 1, 2, 3))
        x0 = rng.choice((-3, -2, -1, 1, 2, 3))
        y0 = rng.randint(-4, 4)

        _f, fxx, fxy, fyx, fyy = _hessian_sympy(A, B, C, D, 0, x0, y0)
        check(fxy == fyx, "hessian: mixed partials are not equal")
        closed = _hessian_closed_form(A, B, C, D, 0, x0, y0)
        check_equivalent((fxx, fxy, fyx, fyy), closed, "hessian medium: independent check failed")
        if fxx == fyy:
            continue

        answer = (fxx, fxy, fyx, fyy)
        answer_text = _fmt4(answer)
        offered, candidates = _hessian_misconceptions(A, B, C, D, 0, x0, y0, answer)
        distractors = pick_distractors(answer_text, [_fmt4(c) for c in candidates])
        if len(distractors) == 0:
            continue

        f_expr = A * _x**2 + B * _x * _y + C * _y**2 + D * _x**3
        f_display = _poly_display(f_expr)
        text = phrase(rng, _EASY_HESS_PATTERNS, f=f_display, x0=x0, y0=y0)
        working = [f"f_xx = 2({A}) + 6({D})({x0}) = {fxx}. f_xy = f_yx = {fxy}. f_yy = {fyy}."]
        return dict(
            text=text,
            latex=text,
            answer=[*answer],
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("hessian medium: could not draw a usable question")


def _hessian_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        A = rng.randint(-5, 5) or 1
        C = rng.randint(-5, 5) or 2
        B = rng.randint(-5, 5)
        D = rng.choice((-3, -2, -1, 1, 2, 3))
        E = rng.choice((-3, -2, -1, 1, 2, 3))
        x0 = rng.choice((-3, -2, -1))  # sign trap: always negative
        y0 = rng.choice((-3, -2, -1, 1, 2, 3))

        _f, fxx, fxy, fyx, fyy = _hessian_sympy(A, B, C, D, E, x0, y0)
        check(fxy == fyx, "hessian: mixed partials are not equal")
        closed = _hessian_closed_form(A, B, C, D, E, x0, y0)
        check_equivalent((fxx, fxy, fyx, fyy), closed, "hessian hard: independent check failed")
        if fxx == fyy:
            continue

        answer = (fxx, fxy, fyx, fyy)
        answer_text = _fmt4(answer)
        offered, candidates = _hessian_misconceptions(A, B, C, D, E, x0, y0, answer)
        distractors = pick_distractors(answer_text, [_fmt4(c) for c in candidates])
        if len(distractors) == 0:
            continue

        f_expr = A * _x**2 + B * _x * _y + C * _y**2 + D * _x**3 + E * _x**2 * _y
        f_display = _poly_display(f_expr)
        text = phrase(rng, _HARD_HESS_PATTERNS, f=f_display, x0=x0, y0=y0)
        working = [
            f"f_xx = 2({A}) + 6({D})({x0}) + 2({E})({y0}) = {fxx}. "
            f"f_xy = f_yx = ({B}) + 2({E})({x0}) = {fxy}. f_yy = {fyy}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=[*answer],
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("hessian hard: could not draw a usable question")


def _hessian_exam(rng):
    """Inverse/coefficient-recovery shape: E is unknown in f's own formula;
    f_xy at the point is stated instead (it depends on E alone: B + 2E x0).
    The learner must solve for E before the Hessian can be completed --
    a genuinely different decision from every forward tier, and the answer
    stays the same 4-tuple shape."""
    for _attempt in range(_MAX_ATTEMPTS):
        A = rng.randint(-5, 5) or 1
        C = rng.randint(-5, 5) or 2
        B = rng.randint(-5, 5)
        D = rng.choice((-3, -2, -1, 1, 2, 3))
        E = rng.choice((-3, -2, -1, 1, 2, 3))
        x0 = rng.choice((-3, -2, -1, 1, 2, 3))
        y0 = rng.choice((-3, -2, -1, 1, 2, 3))

        _f, fxx, fxy, fyx, fyy = _hessian_sympy(A, B, C, D, E, x0, y0)
        check(fxy == fyx, "hessian: mixed partials are not equal")
        closed = _hessian_closed_form(A, B, C, D, E, x0, y0)
        check_equivalent((fxx, fxy, fyx, fyy), closed, "hessian exam: independent check failed")
        if fxx == fyy:
            continue

        fxy_given = fxy  # B + 2E x0, the fact stated to the learner
        # Independent check: recover E from the stated f_xy value via the
        # closed-form relation alone (never touching the true E directly),
        # then confirm the recovered E reproduces the FULL closed-form
        # Hessian -- the two-symbolic-routes pattern applied to the inverse
        # step itself.
        e_recovered = Fraction(fxy_given - B, 2 * x0)
        check(e_recovered.denominator == 1, "hessian exam: recovered E is not an exact integer")
        check(int(e_recovered) == E, "hessian exam: recovered E does not match construction's E")
        recovered_fxx = 2 * A + 6 * D * x0 + 2 * int(e_recovered) * y0
        check(recovered_fxx == fxx, "hessian exam: Hessian rebuilt from recovered E does not match")

        answer = (fxx, fxy, fyx, fyy)
        answer_text = _fmt4(answer)
        offered, candidates = _hessian_misconceptions(A, B, C, D, E, x0, y0, answer)
        # uses-wrong-sign-on-recovered-E: solves the linear equation with the
        # opposite sign convention (E = (fxy_given - B) / (-2 x0)), reachable
        # only when it lands on an exact integer.
        alt = Fraction(fxy_given - B, -2 * x0)
        if alt.denominator == 1:
            alt_fxx = 2 * A + 6 * D * x0 + 2 * int(alt) * y0
            _offer(answer, offered, candidates, (alt_fxx, fxy, fyx, fyy), "wrong-sign-recovered-E")

        distractors = pick_distractors(answer_text, [_fmt4(c) for c in candidates])
        if len(distractors) == 0:
            continue

        f_expr_known = A * _x**2 + B * _x * _y + C * _y**2 + D * _x**3
        f_display = _poly_display(f_expr_known) + " + E x^2 y"
        text = phrase(rng, _EXAM_HESS_PATTERNS, f_display=f_display, x0=x0, y0=y0, fxy_given=fxy_given)
        working = [
            f"f_xy = B + 2E x0, so {fxy_given} = ({B}) + 2E({x0}), giving E = {int(e_recovered)}.",
            f"f_xx = 2({A}) + 6({D})({x0}) + 2({int(e_recovered)})({y0}) = {fxx}. f_yy = {fyy}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=[*answer],
            answer_text=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("hessian exam: could not draw a usable question")


@generator(
    "p12f_hessian_2x2",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12f_hessian_2x2(rng, tier="easy"):
    if tier == "easy":
        return _hessian_easy(rng)
    if tier == "medium":
        return _hessian_medium(rng)
    if tier == "hard":
        return _hessian_hard(rng)
    if tier == "exam":
        return _hessian_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
