"""Zero_product_property + polynomial_long_division."""

from __future__ import annotations

import sympy
from sympy import symbols

from ..core import (
    GRADER_TOLERANCE,
    check,
    check_distinct,
    check_equivalent,
    draw_until,
    generator,
    pick_distractors,
    rational_text,
)

x = symbols("x")

_TOL = float(GRADER_TOLERANCE)


def _poly_display(expr) -> str:
    """Learner-facing polynomial text: no Python ``**``/``*`` (banned by
    text_quality's double_star/star_operator patterns for build topics --
    the grader's own `_parse_expression` turns a `^` back into `**` before
    parsing, so this is safe on the grading side too)."""
    text = str(sympy.expand(expr)).replace("**", "^")
    return text.replace("*", "")


# ---------------------------------------------------------------------------
# zero_product_property


def _factor_text(a: int, b: int) -> str:
    coef = "x" if a == 1 else ("-x" if a == -1 else f"{a}x")
    sign = "+" if b >= 0 else "-"
    return f"({coef} {sign} {abs(b)})"


def _roots_join(roots) -> str:
    parts = [f"x = {rational_text(r)}" for r in roots]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} or {parts[1]}"
    return ", ".join(parts[:-1]) + f", or {parts[-1]}"


def _roots_join_latex(roots) -> str:
    """The `_roots_join` shape as real LaTeX for `answer_latex`: "or"/", "
    become `\\text{ or }`/`,\\; ` so the front-end renderer renders the prose as text and
    each root through KaTeX, rather than as bare italic letters."""
    parts = [rf"x = {sympy.latex(r)}" for r in roots]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return rf"{parts[0]} \text{{ or }} {parts[1]}"
    return r",\; ".join(parts[:-1]) + rf", \text{{ or }} {parts[-1]}"


def _max_positional_gap(sorted_a, sorted_b) -> float:
    """Max |a_i - b_i| once both lists are sorted ascending -- the reading of
    "differs ... in at least one position". A gap
    above the grader's own list tolerance here means `grade_list` cannot
    confuse the two lists via any matching, not merely the sorted one."""
    return max(abs(float(a) - float(b)) for a, b in zip(sorted_a, sorted_b))


@generator(
    "zero_product_property",
    level=9,
    presets={
        "easy": dict(kind="easy"),
        "medium": dict(kind="medium"),
        "hard": dict(kind="hard"),
    },
)
def zero_product_property(rng, kind="medium", coef_hi=5, const_hi=9):
    """Solve a product of linear factors already set to zero (null factor law).

    Backwards construction: draw (coefficient, constant) pairs for each
    factor, derive each root as -constant/coefficient, then re-derive the
    whole root set a second, independent way: expand the full symbolic
    product and call sympy.solve on it -- a route that never touches the
    per-factor arithmetic used to pick the roots.
    """
    # Constants are nonzero integers in -const_hi..const_hi, so const_hi = 0 leaves none to draw. The easy
    # kind has two monic factors (x + b1)(x + b2) whose roots must differ and must not be opposites, so it
    # needs more than the two values +-1 that const_hi = 1 allows.
    min_const_hi = 2 if kind == "easy" else 1
    if const_hi < min_const_hi:
        raise ValueError(
            f"zero_product_property needs const_hi >= {min_const_hi} for kind={kind!r}: "
            "each factor's constant must be a nonzero integer and the roots must be distinct and not opposites"
        )
    if kind == "easy":
        monic_flags = [True, True]
    elif kind == "medium":
        monic_flags = [False, True]
    else:  # hard
        non_monic = rng.randint(1, 3)
        monic_flags = [False] * non_monic + [True] * (3 - non_monic)

    for _attempt in range(500):
        flags = list(monic_flags)
        rng.shuffle(flags)
        factors = []
        for monic in flags:
            a = 1 if monic else rng.randint(2, coef_hi)
            b = draw_until(
                rng,
                lambda r: r.randint(-const_hi, const_hi),
                lambda constant: constant != 0,
                what="a nonzero factor constant",
            )
            factors.append((a, b))
        roots = [sympy.Rational(-b, a) for a, b in factors]

        if len(set(roots)) != len(roots):
            continue
        if any(roots[i] == -roots[j] for i in range(len(roots)) for j in range(len(roots)) if i != j):
            continue
        ordered_roots = sorted(roots, key=float)
        values = [float(r) for r in ordered_roots]
        if not all(later - earlier > _TOL for earlier, later in zip(values, values[1:])):
            continue

        # Candidate misconception root lists, computed here (before the draw
        # is accepted) so every one of them can be checked for the same
        # positional separation the correct list itself must have -- a
        # candidate list that collides with the correct one within the
        # grader's own tolerance is not a valid distractor at any tier.
        flipped_all = sorted((-r for r in roots), key=float)
        one_flipped = list(roots)
        one_flipped[0] = -one_flipped[0]
        one_flipped = sorted(one_flipped, key=float)
        non_monic_present = any(a != 1 for a, _ in factors)
        forgot = None
        if non_monic_present:
            forgot = sorted((sympy.Rational(-b) for a, b in factors), key=float)

        candidate_lists = [flipped_all, one_flipped]
        if forgot is not None:
            candidate_lists.append(forgot)
        if all(_max_positional_gap(ordered_roots, cand) > _TOL for cand in candidate_lists):
            break
    else:
        check(False, "could not draw roots (and distractors) with sufficient separation")

    product_expr = sympy.Integer(1)
    for a, b in factors:
        product_expr *= a * x + b
    product_expr = sympy.expand(product_expr)

    # Independent route: solve the full symbolic product with sympy.solve,
    # never the per-factor -b/a arithmetic used to draw the roots above.
    solved = sorted(sympy.solve(sympy.Eq(product_expr, 0), x), key=float)
    check(len(solved) == len(ordered_roots), "sympy.solve found a different number of roots")
    for expected, actual in zip(ordered_roots, solved):
        check_equivalent(actual, expected, "root disagrees with sympy.solve on the expanded product")

    factor_text = "".join(_factor_text(a, b) for a, b in factors)
    answer_text = _roots_join(ordered_roots)
    answer_latex = _roots_join_latex(ordered_roots)

    for original, flipped in zip(roots, [-r for r in roots]):
        check_distinct(original, flipped, msg="sign-flip-both candidate equals its own root")
    check_distinct(roots[0], -roots[0], msg="sign-flip-one candidate equals its own root")

    candidates = [_roots_join(flipped_all), _roots_join(one_flipped)]
    if forgot is not None:
        for a, b in factors:
            if a != 1:
                check_distinct(
                    sympy.Rational(-b, a), sympy.Rational(-b), msg="forgot-to-divide candidate equals its own root"
                )
        candidates.append(_roots_join(forgot))

    return dict(
        text=f"Solve {factor_text} = 0",
        latex=rf"\text{{Solve }} {factor_text} = 0",
        answer=list(ordered_roots),
        answer_text=answer_text,
        answer_latex=answer_latex,
        working=[
            "Null factor law: a product is zero only if one of its factors is zero.",
            f"{factor_text} = 0 -> {answer_text}",
        ],
        distractors=pick_distractors(answer_text, candidates, unordered=True, tolerance=GRADER_TOLERANCE),
    )


# ---------------------------------------------------------------------------
# polynomial_long_division


@generator(
    "polynomial_long_division",
    level=11,
    presets={
        "easy": dict(kind="easy"),
        "medium": dict(kind="medium"),
        "hard": dict(kind="hard"),
    },
)
def polynomial_long_division(rng, kind="medium", lead_hi=3):
    """Divide a polynomial by a NON-MONIC linear or quadratic divisor.

    Differs from the shipped `integer_polynomial_quotient_remainder` on both
    axes: that skill's divisor is always monic
    (`divisor_coefficients + [1]`) and its answer is chosen from four mcq
    options; here every tier's divisor has leading coefficient > 1 and the
    answer is typed as one rational expression, `quotient + remainder/divisor`.
    """
    if kind == "easy":
        divisor_degree, quotient_degree = 1, 1
    elif kind == "medium":
        divisor_degree, quotient_degree = 1, 2
    else:
        divisor_degree = 2
        quotient_degree = rng.choice([1, 2])

    for _attempt in range(200):
        divisor_lead = rng.randint(2, lead_hi)
        divisor_rest = [rng.randint(-5, 5) for _ in range(divisor_degree)]
        divisor = sum(c * x**p for p, c in enumerate(divisor_rest)) + divisor_lead * x**divisor_degree

        quotient_coeffs = [rng.randint(-6, 6) for _ in range(quotient_degree + 1)]
        if quotient_coeffs[-1] == 0 or quotient_coeffs[0] == 0:
            continue
        quotient = sum(c * x**p for p, c in enumerate(quotient_coeffs))

        remainder_coeffs = [rng.randint(-7, 7) for _ in range(divisor_degree)]
        if all(c == 0 for c in remainder_coeffs):
            continue
        remainder = sum(c * x**p for p, c in enumerate(remainder_coeffs))
        break
    else:
        check(False, "could not draw a valid division")

    dividend = sympy.expand(divisor * quotient + remainder)

    # Independent route: exact division over QQ (needed since the divisor is
    # non-monic), never the forward multiply-out used to build the dividend.
    q_check, r_check = sympy.div(sympy.Poly(dividend, x, domain=sympy.QQ), sympy.Poly(divisor, x, domain=sympy.QQ))
    check_equivalent(q_check.as_expr(), quotient, "quotient disagrees with sympy.div")
    check_equivalent(r_check.as_expr(), remainder, "remainder disagrees with sympy.div")
    check(sympy.degree(sympy.Poly(remainder, x)) < divisor_degree, "remainder degree is not smaller than the divisor's")

    canonical_expr = quotient + remainder / divisor
    answer_internal = str(canonical_expr)

    quotient_disp = _poly_display(quotient)
    remainder_disp = _poly_display(remainder)
    divisor_disp = _poly_display(divisor)
    dividend_disp = _poly_display(dividend)
    answer_text = f"{quotient_disp} + ({remainder_disp})/({divisor_disp})"

    # Misconceptions -------------------------------------------------------
    # sign-error-on-remainder: flips the remainder's sign.
    sign_error_expr = quotient - remainder / divisor
    check_distinct(canonical_expr, sign_error_expr, msg="sign-error-on-remainder equals the canonical")
    sign_error_text = f"{quotient_disp} - ({remainder_disp})/({divisor_disp})"

    # remainder-not-over-divisor: adds the remainder instead of dividing it
    # in -- forgets the "/divisor" step of the required format entirely.
    # (Not "stopped one division step early": that error re-absorbs the
    # dropped term into a bigger remainder over the SAME divisor, and
    # Q' + R'/D collapses back to the identical rational value as Q + R/D
    # for any consistent division step, so it can never be a value-distinct
    # distractor in this single-expression format -- the same identity that
    # makes the disclosed dividend/divisor loophole below unfixable without
    # a new grader capability.)
    not_over_divisor_expr = quotient + remainder
    check_distinct(canonical_expr, not_over_divisor_expr, msg="remainder-not-over-divisor equals the canonical")
    not_over_divisor_text = f"{quotient_disp} + ({remainder_disp})"

    # dropped-remainder: states the quotient alone.
    check_distinct(canonical_expr, quotient, msg="dropped-remainder candidate equals the canonical")
    dropped_text = quotient_disp

    return dict(
        text=(
            f"Divide f(x) = {dividend_disp} by g(x) = {divisor_disp}. "
            "Give your answer as a single expression in the form "
            "quotient + remainder/divisor."
        ),
        latex=(
            rf"f(x)={sympy.latex(dividend)},\quad g(x)={sympy.latex(divisor)}."
            r"\ \text{Give your answer as a single expression in the form "
            r"quotient + remainder/divisor.}"
        ),
        answer=answer_internal,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Divide f(x) by g(x): quotient {quotient_disp}, remainder {remainder_disp}.",
            f"Answer: {answer_text}",
        ],
        distractors=pick_distractors(answer_text, [sign_error_text, not_over_divisor_text, dropped_text]),
    )
