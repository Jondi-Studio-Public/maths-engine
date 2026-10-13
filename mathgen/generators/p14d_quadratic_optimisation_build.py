"""Generates critical-point and Hessian-sign questions for quadratic functions of two variables.

Both skills share ONE quadratic family: the critical-point topic uses `f(x, y) = a*x^2 + b*y^2 + c*x*y + d*x +
e*y`; the Hessian-sign topic uses the same family with the linear terms dropped (`d = e = 0`, irrelevant to a
constant Hessian). Neither claims global optimality, constrained optimisation or nonsmooth functions -- see
each skill's stated exclusions.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    draw_until,
    generator,
    join_terms,
    pick_distractors,
    rational_text,
)

_X, _Y = sympy.symbols("x y")


def _poly_display(expr) -> str:
    """`x**2` -> `x^2`, `3*x` -> `3x` -- the same convention
    `mathgen.generators.university_foundations._poly_display` uses. Kept
    local to this module: it is a one-off string transform, not a shared
    numeric helper."""
    return str(expr).replace("**", "^").replace("*", "")


def _draw_independent_linear_pair(rng):
    """Draw two linear-form coefficient pairs `(p1, q1)`, `(p2, q2)` with
    `p1*q2 - p2*q1 != 0` (linearly independent), for the
    ucv1-unconstrained-critical-point exam tier's sum-of-two-squares
    disguise: `f = (p1*x + q1*y + r1)^2 + (p2*x + q2*y + r2)^2`. Independence
    is what makes the target point the UNIQUE critical point -- see the
    module-level construction comment in `p14d_critical_point`."""
    for _ in range(30):
        p1 = rng.randint(-3, 3)
        q1 = rng.randint(-3, 3)
        p2 = rng.randint(-3, 3)
        q2 = rng.randint(-3, 3)
        if p1 == 0 and q1 == 0:
            continue
        if p2 == 0 and q2 == 0:
            continue
        if p1 * q2 - p2 * q1 != 0:
            return p1, q1, p2, q2
    raise ValueError("could not draw linearly independent (p, q) pairs")


def _split_cross_term(c_true, rng):
    """Split the TRUE Hessian off-diagonal sum `c_true` into two DIFFERENT,
    both-NONZERO integers `A12, A21` with `A12 + A21 == c_true`, for the
    ucv1-convexity-hessian-sign exam tier's non-symmetric-matrix disguise:
    `f = A11*x^2 + A12*x*y + A21*y*x + A22*y^2` with `A12 != A21`, so the
    prompt visibly comes from an unsymmetrised matrix rather than an
    already-combined `c*x*y` term. Both nonzero so both the `x*y` and `y*x`
    terms are actually rendered (a zero coefficient drops the term)."""
    offsets = list(range(1, 6))
    rng.shuffle(offsets)
    for k in offsets:
        for sign in (1, -1):
            a21 = sign * k
            a12 = c_true - a21
            if a12 != a21 and a12 != 0 and a21 != 0:
                return a12, a21
    # c_true == 0 with every small split colliding is vanishingly rare; force
    # a valid split with a slightly larger offset.
    for k in range(6, 40):
        a21 = k
        a12 = c_true - a21
        if a12 != a21 and a12 != 0 and a21 != 0:
            return a12, a21
    raise ValueError(f"could not split cross term {c_true!r} into two distinct nonzero halves")


# ---------------------------------------------------------------------------
# ucv1-unconstrained-critical-point
# ---------------------------------------------------------------------------


def _critical_point_payload(x0, y0):
    x0, y0 = Fraction(x0), Fraction(y0)
    answer_text = f"x = {rational_text(x0)}, y = {rational_text(y0)}"
    return {"x": x0, "y": y0}, answer_text


@generator(
    "p14d_critical_point",
    level=18,
    presets={
        "easy": dict(coeff_hi=4, point_hi=5, coupled=False, frac=False, disguise=False),
        "medium": dict(coeff_hi=4, point_hi=6, coupled=True, frac=False, disguise=False),
        "hard": dict(coeff_hi=5, point_hi=6, coupled=True, frac=True, disguise=False),
        "exam": dict(coeff_hi=4, point_hi=6, coupled=True, frac=False, disguise=True),
    },
)
def p14d_critical_point(rng, coeff_hi=4, point_hi=5, coupled=False, frac=False, disguise=False):
    """Find the critical point of f(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y.

    Constructed backwards from the target point (x0, y0): a, b, c are drawn
    with a nonsingular Hessian (4ab - c^2 != 0, retried on a singular draw),
    then d, e are solved so the gradient vanishes exactly at (x0, y0).

    The independent check solves the linear system 2a*x + c*y = -d,
    c*x + 2b*y = -e via Cramer's rule from a, b, c, d, e ALONE -- never from
    x0, y0 -- the genuinely different route: construction goes forward from
    the point, the check solves the system the construction produced.

    x0 == y0, x0 == 0 and y0 == 0 are excluded by the draw so the three
    distractors (coordinate swap, negate-x, negate-y) are always distinct
    from the answer and from each other (checked unconditionally below).

    The exam tier (`disguise=True`) replaces the random-Hessian draw with a
    SUM-OF-TWO-SQUARES construction: pick two linearly independent linear
    forms L1 = p1*x + q1*y + r1, L2 = p2*x + q2*y + r2 with
    L1(x0, y0) = L2(x0, y0) = 0 by construction (r1, r2 solved from x0, y0),
    so f = L1^2 + L2^2 has gradient 2*L1*grad(L1) + 2*L2*grad(L2), which
    vanishes at (x0, y0); independence of (p1, q1), (p2, q2) rules out any
    OTHER solution of grad(f) = 0 (L1 = L2 = 0 is then the only way to zero
    both terms). The standard-form a, b, c, d, e are then the binomial
    expansion of L1^2 + L2^2, extracted from `sympy.expand` and cross-checked
    against a second, hand-built expansion from p1, q1, p2, q2, r1, r2
    directly -- so a slip in either route is caught. The learner must expand
    the squared form (or apply the chain rule) before differentiating: no
    lower tier contains a squared linear expression.
    """
    if point_hi < 1:
        raise ValueError(
            "p14d_critical_point needs point_hi >= 1: the target point needs two different nonzero coordinates"
        )

    def draw_instance(rng):
        """One attempt at the whole construction: a tuple, or None when it fails a rule and must be redrawn."""
        if frac:
            denom = rng.choice([2, 3])
            num_hi = point_hi * denom

            def _draw_fraction():
                n = draw_until(
                    rng,
                    lambda r: r.randint(-num_hi, num_hi),
                    lambda num: num % denom != 0,
                    what="a numerator that is not a multiple of the denominator",
                )
                return Fraction(n, denom)

            x0, y0 = _draw_fraction(), _draw_fraction()
        else:
            x0 = rng.randint(-point_hi, point_hi)
            y0 = rng.randint(-point_hi, point_hi)
        if x0 == y0 or x0 == 0 or y0 == 0:
            return None

        if disguise:
            p1, q1, p2, q2 = _draw_independent_linear_pair(rng)
            r1 = -(p1 * x0 + q1 * y0)
            r2 = -(p2 * x0 + q2 * y0)
            l1_expr = p1 * _X + q1 * _Y + r1
            l2_expr = p2 * _X + q2 * _Y + r2
            f_squared = l1_expr**2 + l2_expr**2
            expanded = sympy.expand(f_squared)
            poly = sympy.Poly(expanded, _X, _Y)
            a = int(poly.coeff_monomial(_X**2))
            b = int(poly.coeff_monomial(_Y**2))
            c = int(poly.coeff_monomial(_X * _Y))
            d = int(poly.coeff_monomial(_X))
            e = int(poly.coeff_monomial(_Y))

            # Binomial-expansion formula, built directly from p1, q1, p2, q2,
            # r1, r2 rather than trusting sympy.expand alone. The constant
            # term (r1^2 + r2^2) is irrelevant to the gradient/critical point
            # but must be included for a genuine full-expression check.
            a_hat = p1**2 + p2**2
            b_hat = q1**2 + q2**2
            c_hat = 2 * (p1 * q1 + p2 * q2)
            d_hat = 2 * (p1 * r1 + p2 * r2)
            e_hat = 2 * (q1 * r1 + q2 * r2)
            const_hat = r1**2 + r2**2
            hand_expr = a_hat * _X**2 + b_hat * _Y**2 + c_hat * _X * _Y + d_hat * _X + e_hat * _Y + const_hat
            check_equivalent(
                expanded, hand_expr, "sum-of-squares expansion disagrees with the hand-built binomial expansion"
            )
            check(
                a == a_hat and b == b_hat and c == c_hat and d == d_hat and e == e_hat,
                "sympy.expand-derived coefficients disagree with the hand-built ones",
            )

            if c == 0:
                # Linear independence alone does not rule out orthogonal
                # linear forms (a zero cross term): retry so the exam tier
                # stays genuinely coupled, like every other coupled tier.
                return None
            det = 4 * a * b - c * c
            if det == 0:
                return None
            return x0, y0, a, b, c, d, e, f_squared

        a = rng.randint(1, coeff_hi) * rng.choice([1, -1])
        b = rng.randint(1, coeff_hi) * rng.choice([1, -1])
        c = (rng.randint(1, coeff_hi) * rng.choice([1, -1])) if coupled else 0
        det = 4 * a * b - c * c
        if det == 0:
            return None
        d = -(2 * a * x0 + c * y0)
        e = -(2 * b * y0 + c * x0)
        return x0, y0, a, b, c, d, e, None

    x0, y0, a, b, c, d, e, f_squared = draw_until(
        rng,
        draw_instance,
        lambda drawn: drawn is not None,
        what="a target point and a nonsingular Hessian",
    )

    a_f, b_f, c_f, d_f, e_f = (Fraction(v) for v in (a, b, c, d, e))
    det_f = 4 * a_f * b_f - c_f * c_f
    x_solved = (-2 * b_f * d_f + c_f * e_f) / det_f
    y_solved = (-2 * a_f * e_f + c_f * d_f) / det_f
    check(x_solved == Fraction(x0), "Cramer's-rule critical point x mismatch")
    check(y_solved == Fraction(y0), "Cramer's-rule critical point y mismatch")

    f_expr = a * _X**2 + b * _Y**2 + c * _X * _Y + d * _X + e * _Y
    if disguise:
        # f_expr omits the constant term (r1^2 + r2^2, irrelevant to the
        # gradient) by the family's convention; the x^2/y^2/xy/x/y
        # coefficients already agree by construction (checked above via
        # both the sympy.expand route and the hand-built binomial route).
        f_display = _poly_display(f_squared)
        f_latex_source = f_squared
    else:
        f_display = _poly_display(f_expr)
        f_latex_source = f_expr

    answer, answer_text = _critical_point_payload(x0, y0)
    instruction = "Find the critical point (x, y) where the gradient equals (0, 0)."
    text = f"f(x, y) = {f_display}. {instruction}"
    latex = rf"f(x,y)={sympy.latex(f_latex_source)}.\quad\text{{{instruction}}}"

    _, swap = _critical_point_payload(y0, x0)
    _, neg_x = _critical_point_payload(-x0, y0)
    _, neg_y = _critical_point_payload(x0, -y0)
    check_pairwise_distinct(
        (x0, y0),
        (y0, x0),
        (-x0, y0),
        (x0, -y0),
        msg="critical-point distractor collides with the answer or another distractor",
    )
    distractors = pick_distractors(answer_text, [swap, neg_x, neg_y])

    grad_x = join_terms([(2 * a, "x"), (c, "y"), (d, "")])
    grad_y = join_terms([(c, "x"), (2 * b, "y"), (e, "")])
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"∂f/∂x = {grad_x} = 0",
            f"∂f/∂y = {grad_y} = 0",
            f"Solving simultaneously gives {answer_text}.",
        ],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# ucv1-convexity-hessian-sign
# ---------------------------------------------------------------------------

_CLASSES = ("convex", "concave", "indefinite")


def _near_miss_c(ab4, rng, want_positive_det):
    """A `c` whose square lands close to `ab4` on the wanted side, so the determinant's sign cannot be eyeballed
    from the relative sizes of a*b and c (hard-tier disguise): interleaves a just-positive-det pool with a
    just-negative-det pool of similar magnitude."""
    base = math.isqrt(abs(ab4)) if ab4 > 0 else 1
    offsets = list(range(-5, 6))
    rng.shuffle(offsets)
    for offset in offsets:
        c = base + offset
        if c == 0:
            continue
        det = ab4 - c * c
        if want_positive_det and det > 0:
            return c
        if not want_positive_det and det < 0:
            return c
    return (base + (6 if not want_positive_det else -6)) or 1


def _draw_hessian_coeffs(rng, coeff_hi, target, tier):
    if tier == "easy":
        c = 0
        if target == "convex":
            return rng.randint(1, coeff_hi), rng.randint(1, coeff_hi), c
        if target == "concave":
            return -rng.randint(1, coeff_hi), -rng.randint(1, coeff_hi), c
        a = rng.randint(1, coeff_hi)
        b = -rng.randint(1, coeff_hi)
        if rng.random() < 0.5:
            a, b = -a, -b
        return a, b, c
    if tier in ("medium", "exam"):
        if target in ("convex", "concave"):
            sign = 1 if target == "convex" else -1
            a = sign * rng.randint(1, coeff_hi)
            b = sign * rng.randint(1, coeff_hi)
            c = rng.randint(-coeff_hi, coeff_hi)
            return a, b, c
        # indefinite: mostly a same-sign diagonal (both entries positive OR
        # both negative) with a large cross term, defeating "both diagonal
        # entries positive/negative means definite" -- the always-uses-
        # diagonal-sign misconception; sometimes a trivially opposite-sign
        # diagonal, for variety.
        if rng.random() < 0.7:
            sign = rng.choice([1, -1])
            half = max(2, coeff_hi // 2)
            a = sign * rng.randint(1, half)
            b = sign * rng.randint(1, half)
            c = rng.randint(coeff_hi, coeff_hi * 3) * rng.choice([1, -1])
        else:
            a = rng.randint(1, coeff_hi)
            b = -rng.randint(1, coeff_hi)
            if rng.random() < 0.5:
                a, b = -a, -b
            c = rng.randint(-coeff_hi, coeff_hi)
        return a, b, c
    # hard: near-miss disguise, det's sign cannot be eyeballed from a*b vs c.
    if target == "indefinite":
        sign = rng.choice([1, -1])
    else:
        sign = 1 if target == "convex" else -1
    a = sign * rng.randint(2, coeff_hi + 3)
    b = sign * rng.randint(2, coeff_hi + 3)
    ab4 = 4 * a * b
    c = _near_miss_c(ab4, rng, want_positive_det=(target != "indefinite"))
    return a, b, c


def _classify(a, b, c):
    det = 4 * a * b - c * c
    if det > 0:
        return "convex" if a > 0 else "concave"
    return "indefinite"


@generator(
    "p14d_convexity_hessian_sign",
    level=18,
    presets={
        "easy": dict(coeff_hi=5, tier="easy"),
        "medium": dict(coeff_hi=5, tier="medium"),
        "hard": dict(coeff_hi=4, tier="hard"),
        "exam": dict(coeff_hi=5, tier="exam", disguise=True),
    },
)
def p14d_convexity_hessian_sign(rng, coeff_hi=5, tier="medium", disguise=False):
    """Classify f(x, y) = a*x^2 + b*y^2 + c*x*y by its CONSTANT Hessian
    [[2a, c], [c, 2b]]'s eigenvalue signs: convex (positive definite),
    concave (negative definite) or indefinite.

    Construct-to-target: the class is drawn FIRST, then a, b, c are drawn to
    realise it, so the three classes are drawn evenly over many seeds rather
    than biased toward whichever class is more likely under a free draw. The
    semidefinite edge case (det(H) == 0, a zero eigenvalue) is retried, never
    rendered.

    The exam tier (`disguise=True`) presents the same a, b, c as an explicit
    NON-SYMMETRIC 2x2 matrix A = [[A11, A12], [A21, A22]] with A12 != A21:
    f(x, y) = A11*x^2 + A12*x*y + A21*y*x + A22*y^2, the x*y and y*x terms
    written SEPARATELY (never pre-combined). A11 = a, A22 = b; A12, A21 are
    a non-symmetric split of the true off-diagonal sum c = A12 + A21 (the
    TRUE Hessian symmetrises the two cross terms by summing them, not by
    picking one). This is the `ignores-one-cross-term` misconception's trap:
    reading only A12 (or only A21) as if it were the whole Hessian
    off-diagonal entry.
    """
    target = rng.choice(_CLASSES)
    for _ in range(300):
        a, b, c = _draw_hessian_coeffs(rng, coeff_hi, target, tier)
        det = 4 * a * b - c * c
        if det == 0:
            continue
        if _classify(a, b, c) == target:
            break
    else:
        raise ValueError(f"could not draw coefficients realising class {target!r}")

    hessian = sympy.Matrix([[2 * a, c], [c, 2 * b]])
    signs: list[int] = []
    for eigenvalue, multiplicity in hessian.eigenvals().items():
        value = complex(sympy.N(eigenvalue))
        check(abs(value.imag) < 1e-9, "Hessian eigenvalue unexpectedly complex")
        signs.extend([1 if value.real > 0 else -1] * multiplicity)
    check(len(signs) == 2, "Hessian did not yield exactly two eigenvalues")
    if all(s > 0 for s in signs):
        eigen_class = "convex"
    elif all(s < 0 for s in signs):
        eigen_class = "concave"
    else:
        eigen_class = "indefinite"
    check(eigen_class == target, "eigenvalue-sign classification disagrees with the det/trace shortcut")

    f_expr = a * _X**2 + b * _Y**2 + c * _X * _Y
    if disguise:
        a11, a22 = a, b
        a12, a21 = _split_cross_term(c, rng)
        check(a12 != a21, "exam-tier cross-term split must be non-symmetric")
        check(a12 + a21 == c, "exam-tier cross-term split does not recombine to the true off-diagonal sum")

        # Independent check: the TRUE symmetric Hessian rebuilt directly from
        # A11, A12, A21, A22 (never via a, b, c) must classify the same way.
        matrix_hessian = sympy.Matrix([[2 * a11, a12 + a21], [a12 + a21, 2 * a22]])
        matrix_signs: list[int] = []
        for eigenvalue, multiplicity in matrix_hessian.eigenvals().items():
            value = complex(sympy.N(eigenvalue))
            check(abs(value.imag) < 1e-9, "non-symmetric-matrix Hessian eigenvalue unexpectedly complex")
            matrix_signs.extend([1 if value.real > 0 else -1] * multiplicity)
        check(len(matrix_signs) == 2, "non-symmetric-matrix Hessian did not yield exactly two eigenvalues")
        if all(s > 0 for s in matrix_signs):
            matrix_class = "convex"
        elif all(s < 0 for s in matrix_signs):
            matrix_class = "concave"
        else:
            matrix_class = "indefinite"
        check(matrix_class == target, "matrix-derived classification disagrees with the target class")

        f_display = join_terms([(a11, "x^2"), (a12, "xy"), (a21, "yx"), (a22, "y^2")])
        f_latex = join_terms([(a11, "x^{2}"), (a12, "xy"), (a21, "yx"), (a22, "y^{2}")])
    else:
        f_display = _poly_display(f_expr)
        f_latex = sympy.latex(f_expr)

    instruction = "Classify f as convex, concave or indefinite from the sign of its Hessian's eigenvalues."
    text = f"f(x, y) = {f_display}. {instruction}"
    latex = rf"f(x,y)={f_latex}.\quad\text{{{instruction}}}"

    distractors = pick_distractors(target, [label for label in _CLASSES if label != target], n=2)
    check(len(distractors) == 2, "convexity mcq must always offer exactly two distractors")

    return dict(
        text=text,
        latex=latex,
        answer=target,
        answer_text=target,
        answer_latex=rf"\text{{{target}}}",
        working=[f"H = [[{2 * a}, {c}], [{c}, {2 * b}]]; det(H) = {4 * a * b - c * c}; classification: {target}."],
        distractors=distractors,
    )
