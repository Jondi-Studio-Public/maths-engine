"""Generates `s2b3_least_squares` questions (easy, medium, hard) and the exam chain `exd_x_s2b3_least_squares`.

``uc2-least-squares-normal-equations`` (``linear-algebra``, answer ``matrix``): the least-squares solution of an
inconsistent system through the normal equations A^T A x = A^T b, and the least-squares line or parabola through data.

* easy: the least-squares line y = c + mx through three data points set in an Australian context; the answer is the
  vector (c, m).
* medium: an inconsistent system A x = b with A a 3x2 or 4x2 integer matrix: either the least-squares solution or the
  residual vector b - A x.
* hard: the least-squares parabola y = a + bx + cx^2 through four or five points (three unknowns, a 3x3 normal system);
  the answer is the vector (a, b, c).
* exam (``exd_x_s2b3_least_squares``): a least-squares line through four data points. (a) det(A^T A), (b) the slope
  (follows through from (a)), (c) the intercept (follows through from (b)), (d) the exact length of the residual vector.

Every answer is worked out in exact fractions with the normal equations, and re-derived with SymPy as the pseudo-inverse
applied to b (a different route) with A^T r = 0 checked on the residual.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import VerificationError, attempts, check, generator, phrase
from ._s2b3_kit import (
    dot,
    etext,
    expression_answer,
    expression_distractors,
    fraction_answer,
    fraction_distractors,
    frac,
    mat_text,
    matrix_distractors,
    mat_vec,
    need,
    number_answer,
    sizes_ok,
    to_sym,
    vec_answer,
    vec_text,
    vsub,
)
from ..answers import Distractor, exact_value
from ._ucv1_linalg import inverse, mmul, rank, transpose


def _fm(rows):
    return [[Fraction(v) for v in row] for row in rows]


def _normal_solve(a, b):
    """x = (A^T A)^-1 A^T b in exact fractions."""
    at = transpose(_fm(a))
    return mat_vec(inverse(mmul(at, _fm(a))), mat_vec(at, b))


def _residual(a, b, x):
    return vsub(b, mat_vec(_fm(a), x))


def _verify(a, b, x):
    """SymPy: the pseudo-inverse gives the same x, and the residual is orthogonal to every column of A."""
    sa = to_sym(_fm(a))
    check([frac(e) for e in sa.pinv() * to_sym(list(b))] == list(x), "SymPy's least-squares solution differs")
    r = _residual(a, b, x)
    check(all(dot(col, r) == 0 for col in transpose(_fm(a))), "the residual is not orthogonal to the columns of A")
    check(any(e != 0 for e in r), "the system is consistent")


def _first_rows(a, b):
    """The exact solution of the first n equations only (a square system), or None when it is singular."""
    n = len(a[0])
    try:
        return mat_vec(inverse(_fm(a[:n])), b[:n])
    except Exception:  # noqa: BLE001 - a singular square block simply offers no such distractor
        return None


def _x_errors(a, b, x, *, line: bool = False):
    at = transpose(_fm(a))
    out = [
        (_first_rows(a, b), "ls-solves-only-the-first-equations"),
        (mat_vec(at, b), "ls-reports-the-right-hand-side-of-the-normal-equations"),
        (list(reversed(x)), "ls-coefficients-in-the-wrong-order"),
    ]
    if line:
        out.append(([sum(b, Fraction(0)) / len(b), Fraction(0)], "ls-line-through-the-mean-only"))
    return out


# -- contexts for the line fits ------------------------------------------------------------------------------------

# (introduction, x quantity, y quantity, x start choices, y intercept range, slope choices, noise)
_CONTEXTS = (
    (
        "A solar farm near Port Augusta records its output on days with different amounts of full sun.",
        "hours of full sun",
        "energy produced (MWh)",
        (2, 3, 4),
        (3, 9),
        (2, 3, 4),
        2,
    ),
    (
        "A school in Toowoomba tracks its enrolment from year to year.",
        "number of years since 2019",
        "number of students enrolled",
        (0, 1),
        (380, 520),
        (7, 9, 12, 15),
        6,
    ),
    (
        "A Queensland cattle station weighs a calf at regular check-ups.",
        "age in weeks",
        "mass in kilograms",
        (1, 2, 3),
        (35, 48),
        (6, 7, 8, 9),
        3,
    ),
    (
        "A diver at Ningaloo Reef measures the water temperature at different depths.",
        "depth in metres",
        "temperature in °C",
        (1, 2, 3),
        (26, 29),
        (-1, -2),
        1,
    ),
    (
        "A cyclist times repeats of the same climb in the Adelaide Hills at different stages of a ride.",
        "distance climbed in kilometres",
        "time taken in minutes",
        (1, 2, 3),
        (4, 9),
        (3, 4, 5),
        2,
    ),
)


def _draw_line_data(rng, n, ctx):
    intro, xname, yname, starts, (lo, hi), slopes, noise = ctx
    for _ in attempts(4000, "line data with a readable least-squares line"):
        x0 = rng.choice(starts)
        xs = [x0]
        for _i in range(n - 1):
            xs.append(xs[-1] + rng.choice((1, 1, 2)))
        base, slope = rng.randint(lo, hi), rng.choice(slopes)
        ys = [base + slope * x + rng.randint(-noise, noise) for x in xs]
        a = [[1, x] for x in xs]
        x = _normal_solve(a, ys)
        if x[1] == 0 or not sizes_ok(x, 2000, 24) or all(e == 0 for e in _residual(a, ys, x)):
            continue
        return xs, ys, a, x
    raise AssertionError  # pragma: no cover


def _points(xs, ys):
    return ", ".join(f"({x}, {y})" for x, y in zip(xs, ys))


_EASY_STEMS = (
    "{intro} With x the {xname} and y the {yname}, the readings were {points}. Find the least-squares line "
    "y = c + mx through these points, giving the vector (c, m).",
    "{intro} It records x, the {xname}, against y, the {yname}: {points}. The line y = c + mx fits these points best "
    "in the least-squares sense. Find (c, m).",
)


def _easy(rng):
    ctx = rng.choice(_CONTEXTS)
    xs, ys, a, x = _draw_line_data(rng, 3, ctx)
    _verify(a, ys, x)
    cands = matrix_distractors(_x_errors(a, ys, x, line=True), x, vector=True)
    return dict(
        text=phrase(rng, _EASY_STEMS, intro=ctx[0], xname=ctx[1], yname=ctx[2], points=_points(xs, ys)),
        distractors=need(cands),
        working=[
            f"With A having rows (1, x) and b the y values, the normal equations are AᵀA x = Aᵀb: {mat_text(mmul(transpose(_fm(a)), _fm(a)))} x = {vec_text(mat_vec(transpose(_fm(a)), ys))}.",
            f"Solving gives (c, m) = {vec_text(x)}.",
        ],
        **vec_answer(x),
    )


# -- medium ---------------------------------------------------------------------------------------------------------

_SOLUTION_STEMS = (
    "The system Ax = b with A = {a} and b = {b} has no exact solution. Find its least-squares solution x̂.",
    "Find the least-squares solution of Ax = b, where A = {a} and b = {b}.",
)
_RESIDUAL_STEMS = (
    "Let A = {a} and b = {b}. Find the residual vector r = b - Ax̂ for the least-squares solution x̂ of Ax = b.",
    "The system Ax = b, with A = {a} and b = {b}, is inconsistent. Find the residual vector b - Ax̂ at its "
    "least-squares solution x̂.",
)


def _draw_system(rng, rows, cols, den=24):
    for _ in attempts(4000, "an inconsistent system with a readable least-squares solution"):
        a = [[rng.randint(-3, 3) for _ in range(cols)] for _ in range(rows)]
        b = [rng.randint(-5, 5) for _ in range(rows)]
        if rank(_fm(a)) != cols or any(all(row[c] == 0 for row in a) for c in range(cols)):
            continue
        if sum(1 for row in a for v in row if v == 0) > 2:
            continue
        x = _normal_solve(a, b)
        r = _residual(a, b, x)
        if all(e == 0 for e in r) or any(e == 0 for e in x) or not sizes_ok(x, 40, den) or not sizes_ok(r, 40, den):
            continue
        return a, b, x, r
    raise AssertionError  # pragma: no cover


def _medium(rng):
    """One medium question; a system whose slips leave fewer than two distinct wrong columns is redrawn."""
    for _ in attempts(60, "a system that leaves two usable distractors"):
        try:
            return _medium_once(rng)
        except VerificationError as err:
            if "distractors" not in str(err):
                raise


def _medium_once(rng):
    rows, cols = rng.choice(((3, 2), (4, 2), (4, 2)))
    a, b, x, r = _draw_system(rng, rows, cols)
    _verify(a, b, x)
    if rng.random() < 0.5:
        cands = matrix_distractors(_x_errors(a, b, x), x, vector=True)
        return dict(
            text=phrase(rng, _SOLUTION_STEMS, a=mat_text(a), b=vec_text(b)),
            distractors=need(cands),
            working=[
                f"AᵀA = {mat_text(mmul(transpose(_fm(a)), _fm(a)))} and Aᵀb = {vec_text(mat_vec(transpose(_fm(a)), b))}.",
                f"Solving AᵀA x̂ = Aᵀb gives x̂ = {vec_text(x)}.",
            ],
            **vec_answer(x),
        )
    fitted = mat_vec(_fm(a), x)
    wrong = _first_rows(a, b)
    cands = matrix_distractors(
        [
            (fitted, "ls-gives-the-fitted-values"),
            ([-e for e in r], "ls-residual-sign-reversed"),
            (None if wrong is None else _residual(a, b, wrong), "ls-solves-only-the-first-equations"),
            (
                vsub(b, mat_vec(_fm(a), mat_vec(transpose(_fm(a)), b))),
                "ls-reports-the-right-hand-side-of-the-normal-equations",
            ),
        ],
        r,
        vector=True,
    )
    return dict(
        text=phrase(rng, _RESIDUAL_STEMS, a=mat_text(a), b=vec_text(b)),
        distractors=need(cands),
        working=[
            f"The normal equations give x̂ = {vec_text(x)}.",
            f"Then r = b - Ax̂ = {vec_text(r)}, which is orthogonal to the columns of A.",
        ],
        **vec_answer(r),
    )


# -- hard: a parabola ------------------------------------------------------------------------------------------------

_X_SETS = ((-1, 0, 1, 2), (-2, -1, 0, 1, 2), (0, 1, 2, 3), (-2, -1, 1, 2), (-1, 0, 1, 2, 3), (0, 1, 2, 3, 4))
_PARABOLA_STEMS = (
    "Find the least-squares parabola y = a + bx + cx² through the points {points}, giving the vector (a, b, c).",
    "The points {points} are fitted by a parabola y = a + bx + cx² in the least-squares sense. Find (a, b, c).",
)


def _hard(rng):
    for _ in attempts(4000, "parabola data with a readable least-squares fit"):
        xs = list(rng.choice(_X_SETS))
        ys = [rng.randint(-4, 8) for _ in xs]
        a = [[1, x, x * x] for x in xs]
        x = _normal_solve(a, ys)
        r = _residual(a, ys, x)
        if x[2] == 0 or all(e == 0 for e in r) or not sizes_ok(x, 60, 40):
            continue
        break
    _verify(a, ys, x)
    cands = matrix_distractors(_x_errors(a, ys, x), x, vector=True)
    return dict(
        text=phrase(rng, _PARABOLA_STEMS, points=_points(xs, ys)),
        distractors=need(cands),
        working=[
            f"A has rows (1, x, x²). AᵀA = {mat_text(mmul(transpose(_fm(a)), _fm(a)))}, Aᵀb = {vec_text(mat_vec(transpose(_fm(a)), ys))}.",
            f"Solving the 3 by 3 normal system gives (a, b, c) = {vec_text(x)}.",
        ],
        **vec_answer(x),
    )


@generator(
    "s2b3_least_squares",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2b3_least_squares(rng, tier="easy"):
    """Least squares by the normal equations; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _int_distractors(true, candidates):
    out, seen = [], {true}
    for value, tag in candidates:
        if value in seen or value <= 0:
            continue
        seen.add(value)
        out.append(Distractor(str(value), tag, exact_value(value)))
    return out[:3]


@generator("exd_x_s2b3_least_squares", level=18, presets={"exam": dict(tier="exam")})
def exd_x_s2b3_least_squares(rng, tier="exam"):
    """A least-squares line through four points: det(A^T A), the slope, the intercept and the residual length."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(4000, "four data points with a readable line and an irrational residual"):
        xs, ys, a, x = _draw_line_data(rng, 4, ctx)
        r = _residual(a, ys, x)
        sse = dot(r, r)
        if sse <= 0 or sse.denominator > 40:
            continue
        break
    _verify(a, ys, x)
    c, m = x
    n = 4
    sx, sy = sum(xs), sum(ys)
    sxx, sxy, syy = sum(v * v for v in xs), sum(u * v for u, v in zip(xs, ys)), sum(v * v for v in ys)
    det = n * sxx - sx * sx
    check(sympy.Matrix(a).T * sympy.Matrix(a) == sympy.Matrix([[n, sx], [sx, sxx]]), "AᵀA differs from the sums")
    check(sympy.Matrix([[n, sx], [sx, sxx]]).det() == det, "SymPy's determinant differs")
    check(
        m == Fraction(n * sxy - sx * sy, det) and c == Fraction(sy, n) - m * Fraction(sx, n), "slope/intercept formulas"
    )
    check(sse == syy - m * sxy - c * sy, "the residual identity |r|^2 = b.b - x.A^T b fails")
    root = sympy.sqrt(sympy.Rational(sse.numerator, sse.denominator))
    mean = Fraction(sy, n)

    a_dist = _int_distractors(
        det,
        [
            (n * sxx + sx * sx, "ls-determinant-sign-slip"),
            (sxx - sx * sx, "ls-determinant-forgot-n"),
            (n * sxx, "ls-determinant-dropped-second-term"),
            (sx * sx, "ls-determinant-dropped-first-term"),
        ],
    )
    b_dist = fraction_distractors(
        m,
        [
            (Fraction(sxy, sxx), "ls-slope-without-centring"),
            (Fraction(n * sxy - sx * sy, n * sxx), "ls-slope-wrong-denominator"),
            (Fraction(sx * sy - n * sxy, det), "ls-slope-numerator-reversed"),
            (Fraction(sxy - sx * sy, det), "ls-slope-forgot-n"),
        ],
    )
    c_dist = fraction_distractors(
        c,
        [
            (mean, "ls-intercept-is-the-mean-of-y"),
            ((Fraction(sy) + m * sx) / n, "sign-error-moving-term"),
            (Fraction(sy) - m * sx, "ls-intercept-forgot-divide-by-n"),
            (Fraction(ys[0]) - m * xs[0], "ls-intercept-from-one-point"),
        ],
    )
    sse_rat = sympy.Rational(sse.numerator, sse.denominator)
    l1_frac = sum((abs(e) for e in r), Fraction(0))
    l1 = sympy.Rational(l1_frac.numerator, l1_frac.denominator)
    d_dist = expression_distractors(
        root,
        [
            (sse_rat, "ls-forgot-the-square-root"),
            (l1, "ls-sum-of-residuals-not-length"),
            (sympy.sqrt(sympy.Rational(syy) - sympy.Rational(sy * sy, n)), "ls-residual-from-the-mean-line"),
            (sympy.sqrt(sympy.Rational(syy)), "ls-length-of-b-not-the-residual"),
        ],
    )
    stem = (
        f"{ctx[0]} With x the {ctx[1]} and y the {ctx[2]}, the readings were {_points(xs, ys)}. A least-squares line y = c + mx is fitted. "
        "Let A be the 4 by 2 matrix with rows (1, x) and b the vector of y values."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find det(AᵀA).",
                marks=2,
                sensible_range=(0, None),
                working=[f"AᵀA = {mat_text([[n, sx], [sx, sxx]])}, so det(AᵀA) = {n}({sxx}) - ({sx})² = {det}."],
                distractors=need(a_dist),
                **number_answer(det),
            ),
            dict(
                prompt="Hence find the slope m of the least-squares line.",
                marks=2,
                follow_through=f"({n * sxy} - {sx * sy})/part_a",
                working=[f"Aᵀb = {vec_text([sy, sxy])}. m = ({n}({sxy}) - ({sx})({sy}))/{det}."],
                distractors=need(b_dist),
                **fraction_answer(m),
            ),
            dict(
                prompt="Find the intercept c of the line.",
                marks=2,
                follow_through=f"({sy} - part_b*{sx})/{n}",
                working=[f"From the first normal equation, {n}c + {sx}m = {sy}, so c = ({sy} - {sx}m)/{n}."],
                distractors=need(c_dist),
                **fraction_answer(c),
            ),
            dict(
                prompt="Find the exact length of the residual vector r = b - Ax̂.",
                marks=2,
                working=[f"|r|² = bᵀb - x̂ᵀAᵀb = {syy} - m({sxy}) - c({sy}) = {sse}, so |r| = {etext(root)}."],
                distractors=need(d_dist),
                **expression_answer(root),
            ),
        ],
    )
