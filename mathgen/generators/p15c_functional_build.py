"""Generates `ucv1-diagonal-operator-norm` and `ucv1-function-space-approximation` questions: the operator norm
of a diagonal map and function-space approximation.

``ucv1-diagonal-operator-norm`` (number): the Euclidean (2-norm) operator norm of a 2-by-2 DIAGONAL linear map
is the largest absolute diagonal entry. Boundary: escalate by what has to be worked out FIRST, not by bigger
numbers -- easy states the diagonal directly, medium gives the map by its action on (x, y), hard composes two
stated diagonal maps (so the "multiply the two operator norms" shortcut is a genuine, and generically WRONG,
trap -- submultiplicativity is not equality unless both maps peak on the same coordinate), and exam takes an
integer power of a map with FRACTIONAL entries (so a sign trap on an odd power is possible). The independent
check builds the actual 2x2 diagonal `sympy.Matrix` and reads its singular values -- a genuinely different
computational route from the "take the max of the two entries" construction.

``ucv1-function-space-approximation`` (pair): least-squares coefficients of a function sampled on a fixed
TWO-FUNCTION basis that is never {1, x}. Bases used here: {1, x^2} (easy), {x, x^2} (medium), {x^2, x^3}
(hard/exam). No basis of sin/cos at rational-multiple-of-pi angles can fix this while staying exact, so
hard/exam now use {x^2, x^3} instead, with `_draw_columns` explicitly verifying the off-diagonal
normal-equations term sum(col1*col2) is nonzero (not just the sub-determinant `_fit`'s own residual
construction needs) -- a genuinely coupled 2x2 solve is now guaranteed by construction, not generic luck. The
prompt states the coefficient ORDER (first basis function's coefficient first) and that both are exact
integers. Construction fixes the target (c1, c2) and adds noise orthogonal to both basis columns over 3 sample
points (the same "genuinely noisy but exact" idiom as `least_squares_line`, generalised past a 2-point
residual pattern to an arbitrary basis via a direct 2x2 solve); the independent check re-forms and re-solves
the NORMAL EQUATIONS directly from the sampled data, never from the construction's own (c1, c2, e) variables.
Exam's own escalation: the data is not handed over as plain y-values or as a "two readings" arithmetic
disguise, but as a stated FUNCTION f (f(x) = |x|) sampled at the given points PLUS a stated residual, so the
learner must evaluate f at each point before they can even start the fit -- a genuine extra step hard does not
require.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import (
    paren_negative,
    check,
    check_distinct,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
    rational_text,
)


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


def _pair_payload(c1, c2) -> tuple[dict, str]:
    c1, c2 = Fraction(c1), Fraction(c2)
    return ({"c1": c1, "c2": c2}, f"c1 = {rational_text(c1)}, c2 = {rational_text(c2)}")


# ---------------------------------------------------------------------------
# ucv1-diagonal-operator-norm


def _nonzero(rng, hi):
    v = rng.randint(-hi, hi)
    while v == 0:
        v = rng.randint(-hi, hi)
    return v


def _op_norm_via_singular_values(a, b) -> Fraction:
    m = sympy.Matrix([[sympy.Rational(a), 0], [0, sympy.Rational(b)]])
    svals = m.singular_values()
    top = max(sympy.Rational(v) for v in svals)
    return Fraction(top.p, top.q)


def _coef_var_text(coef, var: str) -> str:
    """`fmt_signed` (mathgen.core) handles the integer/unit cases but formats a non-integer Fraction coefficient
    as f"{coef}{var}" -- the same broken shape -- so this stays local rather than risk a wrong hand-roll of
    the integer cases fmt_signed already owns; only the fraction branch below is genuinely new."""
    coef = Fraction(coef)
    if coef == 1:
        return var
    if coef == -1:
        return f"-{var}"
    if coef.denominator == 1:
        return f"{coef.numerator}{var}"
    num, den = coef.numerator, coef.denominator
    if num == 1:
        return f"{var}/{den}"
    if num == -1:
        return f"-{var}/{den}"
    return f"{num}{var}/{den}"


def _build_diag_easy(rng, hi=9):
    answer_val = rng.randint(2, hi)
    sign1 = rng.choice([1, -1])
    other = rng.randint(1, answer_val - 1) if answer_val > 1 else 0
    sign2 = rng.choice([1, -1])
    d_max = sign1 * answer_val
    d_other = sign2 * other if other else sign2 * answer_val  # avoid a 0 entry
    if rng.choice([True, False]):
        d1, d2 = d_max, d_other
    else:
        d1, d2 = d_other, d_max
    check(max(abs(d1), abs(d2)) == answer_val, "p15c_diagonal_operator_norm: easy target construction mismatch")
    check(
        _op_norm_via_singular_values(d1, d2) == Fraction(answer_val),
        "p15c_diagonal_operator_norm: easy singular-value mismatch",
    )
    answer_text = str(answer_val)
    candidates = [
        str(min(abs(d1), abs(d2))),
        str(abs(d1 + d2)),
        str(abs(d1 * d2)),
        str(abs(d1) + abs(d2)),
        str(abs(d1 - d2)),
    ]
    text = phrase(
        rng,
        [
            "A linear map T: R^2 -> R^2 is given by the diagonal matrix diag({d1}, {d2}). "
            "Find its Euclidean (2-norm) operator norm.",
            "Let T: R^2 -> R^2 be the diagonal linear map diag({d1}, {d2}). What is its "
            "Euclidean (2-norm) operator norm?",
            "The diagonal matrix diag({d1}, {d2}) defines a linear map T: R^2 -> R^2. "
            "Compute its Euclidean (2-norm) operator norm.",
        ],
        d1=d1,
        d2=d2,
    )
    return (
        answer_val,
        answer_text,
        candidates,
        text,
        [
            f"The operator norm of a diagonal matrix is the largest |entry|: max(|{d1}|, |{d2}|).",
            f"Operator norm = {answer_text}.",
        ],
    )


def _build_diag_medium(rng, hi=12):
    # Same construct-to-target discipline as easy.
    answer_val = rng.randint(2, hi)
    sign1 = rng.choice([1, -1])
    other = rng.randint(1, answer_val - 1) if answer_val > 1 else 0
    sign2 = rng.choice([1, -1])
    d_max = sign1 * answer_val
    d_other = sign2 * other if other else sign2 * answer_val
    if rng.choice([True, False]):
        a, b = d_max, d_other
    else:
        a, b = d_other, d_max
    check(max(abs(a), abs(b)) == answer_val, "p15c_diagonal_operator_norm: medium target construction mismatch")
    check(
        _op_norm_via_singular_values(a, b) == Fraction(answer_val),
        "p15c_diagonal_operator_norm: medium singular-value mismatch",
    )
    answer_text = str(answer_val)
    candidates = [
        str(min(abs(a), abs(b))),
        str(abs(a + b)),
        str(abs(a) * abs(b)),
        str(abs(a) + abs(b)),
        str(abs(a - b)),
    ]
    text = phrase(
        rng,
        [
            "A linear map T: R^2 -> R^2 acts by T(x, y) = ({a}, {b}). Find its Euclidean (2-norm) operator norm.",
            "Let T: R^2 -> R^2 be defined by T(x, y) = ({a}, {b}). What is its Euclidean (2-norm) operator norm?",
            "A linear map T: R^2 -> R^2 is defined by T(x, y) = ({a}, {b}). Compute its "
            "Euclidean (2-norm) operator norm.",
        ],
        a=f"{_coef_var_text(a, 'x')}",
        b=f"{_coef_var_text(b, 'y')}",
    )
    return (
        answer_val,
        answer_text,
        candidates,
        text,
        [
            f"T is diagonal with entries {a} and {b} (read off the coefficients of x and y).",
            f"Operator norm = max(|{a}|, |{b}|) = {answer_text}.",
        ],
    )


def _build_diag_hard(rng, hi=9):
    for _ in range(200):
        s1, s2 = _nonzero(rng, hi), _nonzero(rng, hi)
        t1, t2 = _nonzero(rng, hi), _nonzero(rng, hi)
        c1, c2 = s1 * t1, s2 * t2
        answer_val = max(abs(c1), abs(c2))
        norm_s, norm_t = max(abs(s1), abs(s2)), max(abs(t1), abs(t2))
        multiply_norms = norm_s * norm_t
        if multiply_norms == answer_val:
            continue  # trap must be a genuinely different value
        break
    else:
        raise RuntimeError("p15c_diagonal_operator_norm: hard trap never diverged")
    check(
        _op_norm_via_singular_values(c1, c2) == Fraction(answer_val),
        "p15c_diagonal_operator_norm: hard singular-value mismatch",
    )
    check_distinct(answer_val, multiply_norms, msg="p15c_diagonal_operator_norm: multiply-norms trap equals the answer")
    answer_text = str(answer_val)
    swapped = max(abs(s1 * t2), abs(s2 * t1))
    candidates = [
        str(multiply_norms),
        str(min(abs(c1), abs(c2))),
        str(swapped),
        str(abs(c1) + abs(c2)),
        str(abs(s1) * abs(t1)),
    ]
    t1_text, t2_text = _coef_var_text(t1, "x"), _coef_var_text(t2, "y")
    s1_text, s2_text = _coef_var_text(s1, "x"), _coef_var_text(s2, "y")
    text = phrase(
        rng,
        [
            "Two diagonal linear maps act on R^2: T(x, y) = ({t1}, {t2}) and S(x, y) = "
            "({s1}, {s2}). Consider the composition S after T (apply T first, then S). "
            "Find the Euclidean (2-norm) operator norm of the composed map.",
            "Two diagonal linear maps on R^2 are given by T(x, y) = ({t1}, {t2}) and "
            "S(x, y) = ({s1}, {s2}). Compute the Euclidean (2-norm) operator norm of the "
            "composition of S with T (T applied first).",
            "Let T(x, y) = ({t1}, {t2}) and S(x, y) = ({s1}, {s2}) be diagonal linear "
            "maps on R^2. What is the Euclidean (2-norm) operator norm of the composition "
            "S after T (T first, then S)?",
        ],
        t1=t1_text,
        t2=t2_text,
        s1=s1_text,
        s2=s2_text,
    )
    return (
        answer_val,
        answer_text,
        candidates,
        text,
        [
            f"The composed map is diagonal with entries {s1} × {paren_negative(t1)} = {c1} and {s2} × {paren_negative(t2)} = {c2} "
            "(multiply the matching diagonal entries, never the two operator norms).",
            f"Operator norm = max(|{c1}|, |{c2}|) = {answer_text}.",
        ],
    )


_EXAM_BASES = sorted({Fraction(p, q) for p in range(1, 6) for q in (2, 3, 4, 5)})
_EXAM_POWERS = [2, 3, 4]


def _build_diag_exam(rng):
    # Construct-to-target: draw k and the LARGER base value D first (the
    # entry that will end up strictly largest in absolute value after the
    # power, since |x| < |y| implies |x|^k < |y|^k for any k >= 1 -- so
    # fixing |d1| > |d2| in BASE terms is enough to guarantee it survives
    # exponentiation), then draw d2 strictly smaller in absolute value.
    # This replaces the old free draw (which collapsed to only 6 distinct
    # answers, '27/8' alone 32% of 200 draws) with a target chosen from a
    # wide, explicit bank.
    for _ in range(200):
        k = rng.choice(_EXAM_POWERS)
        d_big = rng.choice(_EXAM_BASES)
        smaller_bases = [v for v in _EXAM_BASES if v < d_big]
        if not smaller_bases:
            continue
        d_small = rng.choice(smaller_bases)
        sign1, sign2 = rng.choice([1, -1]), rng.choice([1, -1])
        if rng.choice([True, False]):
            d1, d2 = sign1 * d_big, sign2 * d_small
        else:
            d1, d2 = sign2 * d_small, sign1 * d_big
        p1, p2 = d1**k, d2**k
        answer_val = max(abs(p1), abs(p2))
        check(answer_val == d_big**k, "p15c_diagonal_operator_norm: exam target construction mismatch")
        raw_max = max(p1, p2)  # forgets to take absolute value before comparing
        no_power = max(abs(d1), abs(d2))  # forgets to apply the power at all
        scaled = max(abs(k * d1), abs(k * d2))  # multiplies by k instead of exponentiating
        values = [answer_val, raw_max, no_power, scaled]
        if len(set(values)) == 4:
            break
    else:
        raise RuntimeError("p15c_diagonal_operator_norm: exam traps never diverged")
    m = sympy.Matrix(
        [[sympy.Rational(p1.numerator, p1.denominator), 0], [0, sympy.Rational(p2.numerator, p2.denominator)]]
    )
    top = max(sympy.Rational(v) for v in m.singular_values())
    check(Fraction(top.p, top.q) == answer_val, "p15c_diagonal_operator_norm: exam singular-value mismatch")
    check_pairwise_distinct(
        answer_val, raw_max, no_power, scaled, msg="p15c_diagonal_operator_norm: exam candidate collides"
    )
    answer, answer_text = _fraction_payload(answer_val)
    candidates = [rational_text(raw_max), rational_text(no_power), rational_text(scaled)]
    d1_text, d2_text = _coef_var_text(d1, "x"), _coef_var_text(d2, "y")
    text = phrase(
        rng,
        [
            "A diagonal linear map is T(x, y) = ({d1}, {d2}). Find the Euclidean (2-norm) "
            "operator norm of T^{k} (T applied {k} times), as a fraction.",
            "Let T(x, y) = ({d1}, {d2}) be a diagonal linear map. Compute the Euclidean "
            "(2-norm) operator norm of T^{k} (T composed with itself {k} times), as a fraction.",
            "T(x, y) = ({d1}, {d2}) defines a diagonal linear map. What is the Euclidean "
            "(2-norm) operator norm of T^{k}, applying T {k} times, as a fraction?",
        ],
        d1=d1_text,
        d2=d2_text,
        k=k,
    )
    working = [
        f"T^{k} is diagonal with entries ({rational_text(d1)})^{k} = {rational_text(p1)} "
        f"and ({rational_text(d2)})^{k} = {rational_text(p2)}.",
        f"Operator norm = max(|{rational_text(p1)}|, |{rational_text(p2)}|) = {answer_text}.",
    ]
    return answer, answer_text, candidates, text, working


@generator(
    "p15c_diagonal_operator_norm",
    level=17,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p15c_diagonal_operator_norm(rng, variant="easy"):
    for _attempt in range(50):
        if variant == "exam":
            answer, answer_text, candidates, text, working = _build_diag_exam(rng)
        else:
            builder = {"easy": _build_diag_easy, "medium": _build_diag_medium, "hard": _build_diag_hard}[variant]
            answer_val, answer_text, candidates, text, working = builder(rng)
            answer = answer_val
        if len(pick_distractors(answer_text, candidates)) == 3:
            break
    else:
        raise RuntimeError("p15c_diagonal_operator_norm: could not draw 3 distinct distractors")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text if variant == "exam" else f"plain:{answer_text}",
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


# ---------------------------------------------------------------------------
# ucv1-function-space-approximation


def _solve2(a11, a12, b1, a21, a22, b2):
    det = a11 * a22 - a12 * a21
    if det == 0:
        return None
    e1 = Fraction(b1 * a22 - a12 * b2, 1) / det
    e2 = Fraction(a11 * b2 - a21 * b1, 1) / det
    return e1, e2


def _normal_equations_solve(points, col1, col2, y):
    """Independent route: form and solve the 2x2 NORMAL EQUATIONS directly
    from the sampled (col1, col2, y) data -- never from the construction's
    own target/residual variables."""
    s11 = sum(c * c for c in col1)
    s12 = sum(c1 * c2 for c1, c2 in zip(col1, col2))
    s22 = sum(c * c for c in col2)
    sy1 = sum(c * yy for c, yy in zip(col1, y))
    sy2 = sum(c * yy for c, yy in zip(col2, y))
    solved = _solve2(s11, s12, sy1, s12, s22, sy2)
    return solved


def _fit(rng, col1, col2, c1_hi=4, c2_hi=4):
    for _ in range(50):
        c1 = _nonzero(rng, c1_hi)
        c2 = _nonzero(rng, c2_hi)
        if c1 == c2:
            continue
        e0 = rng.choice([-2, -1, 1, 2])
        solved = _solve2(col1[1], col1[2], -col1[0] * e0, col2[1], col2[2], -col2[0] * e0)
        if solved is None:
            continue
        e1, e2 = solved
        e = [Fraction(e0), e1, e2]
        y = [Fraction(c1) * col1[i] + Fraction(c2) * col2[i] + e[i] for i in range(3)]
        check(
            sum(col1[i] * e[i] for i in range(3)) == 0,
            "p15c_function_space_approximation: residual not orthogonal to basis 1",
        )
        check(
            sum(col2[i] * e[i] for i in range(3)) == 0,
            "p15c_function_space_approximation: residual not orthogonal to basis 2",
        )
        fitted = _normal_equations_solve([0, 1, 2], col1, col2, y)
        if fitted is None:
            continue
        f1, f2 = fitted
        check(
            f1 == c1 and f2 == c2,
            "p15c_function_space_approximation: normal-equation solve disagrees with construction",
        )
        return c1, c2, y
    raise RuntimeError("p15c_function_space_approximation: could not construct a fit")


def _points_poly(rng, hi, exclude_zero):
    pts = []
    while len(pts) < 3:
        x = rng.randint(-hi, hi)
        if exclude_zero and x == 0:
            continue
        if x not in pts:
            pts.append(x)
    return pts


def _draw_columns(rng, basis: str):
    """Draw sample points whose basis columns (a) have a nonzero (1, 2) sub-determinant -- the specific 2x2 solve
    `_fit` needs for its residual construction -- and (b) have a nonzero OFF-DIAGONAL normal-equations term
    sum(col1*col2), so the full 3-point least-squares system is genuinely coupled. Retries the point draw
    (never the c1/c2 draw) until both conditions hold."""
    for _ in range(200):
        if basis == "one_xsq":
            points = _points_poly(rng, 4, exclude_zero=False)
            col1 = [Fraction(1) for _ in points]
            col2 = [Fraction(x * x) for x in points]
        elif basis == "x_xsq":
            points = _points_poly(rng, 5, exclude_zero=True)
            col1 = [Fraction(x) for x in points]
            col2 = [Fraction(x * x) for x in points]
        else:  # sq_cube: {x^2, x^3}
            points = _points_poly(rng, 4, exclude_zero=True)
            col1 = [Fraction(x * x) for x in points]
            col2 = [Fraction(x * x * x) for x in points]
        sub_det_ok = col1[1] * col2[2] - col1[2] * col2[1] != 0
        off_diag_ok = sum(c1 * c2 for c1, c2 in zip(col1, col2)) != 0
        if sub_det_ok and off_diag_ok:
            return points, col1, col2
    raise RuntimeError(f"p15c_function_space_approximation: no usable sample points for {basis!r}")


def _build_approx(rng, basis: str, evaluate_fn: bool):
    basis_text = {"one_xsq": "{1, x^2}", "x_xsq": "{x, x^2}", "sq_cube": "{x^2, x^3}"}[basis]

    if evaluate_fn:
        # The data must BE the stated function, exactly, for a function genuinely outside the
        # basis's span: f(x) = x^4 is a degree-4 polynomial, so no combination of {x^2, x^3}
        # reproduces it exactly, and the resulting least-squares fit is a genuine approximation --
        # never an exact reconstruction of a pre-chosen (c1, c2) target the way every other tier's
        # `_fit` builds one. `grade_pair`'s labelled/positional comparisons both apply the
        # spoken-transcript +/-0.051 tolerance PER COMPONENT (mathgen/grading/typed.py
        # `_PAIR_TOLERANCE`); every other tier's `_fit` construct-to-target always drew integer
        # coefficients far apart, so this never mattered before. A genuine normal-equations solve on
        # f(x) = x^4 data can land arbitrarily close to zero or to itself under a swap/sign flip, so
        # every candidate is checked against that tolerance here, not assumed safe by construction.
        _MARGIN = 0.06  # > 2x the 0.051 tolerance, so a sign flip is always outside it
        for _ in range(200):
            points, col1, col2 = _draw_columns(rng, basis)
            y = [Fraction(p) ** 4 for p in points]
            fitted = _normal_equations_solve(None, col1, col2, y)
            if fitted is None:
                continue
            c1, c2 = fitted
            if abs(float(c1)) > _MARGIN and abs(float(c2)) > _MARGIN and abs(float(c1) - float(c2)) > _MARGIN:
                break
        else:
            raise RuntimeError("p15c_function_space_approximation: no usable exam draw")
        answer, answer_text = _pair_payload(c1, c2)
        sample_lines = ", ".join(f"x = {p}" for p in points)
        y_text = (
            "; ".join(f"y{i + 1} = f({p}) = {rational_text(val)}" for i, (p, val) in enumerate(zip(points, y)))
            + " (f(x) = x^4)"
        )
    else:
        points, col1, col2 = _draw_columns(rng, basis)
        c1, c2, y = _fit(rng, col1, col2)
        answer, answer_text = _pair_payload(c1, c2)
        sample_lines = ", ".join(f"x = {p}" for p in points)
        y_text = "; ".join(f"y{i + 1} = {rational_text(val)}" for i, val in enumerate(y))

    basis1 = basis_text.split(",")[0].strip("{} ")
    basis2 = basis_text.split(",")[1].strip("{} ")
    term1 = "c1" if basis1 == "1" else f"c1 {basis1}"
    term2 = f"c2 {basis2}"
    if evaluate_fn:
        text = phrase(
            rng,
            (
                "A function is sampled on the basis {basis_text} at {sample_lines}, giving "
                "{y_text}. Find the least-squares coefficients c1, c2 in the fit "
                "y = {term1} + {term2}, exactly (c1 first, then c2).",
                "On the basis {basis_text}, a function is sampled at {sample_lines}, giving "
                "{y_text}. Find the exact least-squares coefficients c1, c2 (c1 first, then "
                "c2) for the fit y = {term1} + {term2}.",
                "Using the basis {basis_text}, a function is sampled at {sample_lines}, giving "
                "{y_text}. Compute the exact least-squares coefficients c1, c2 for the fit "
                "y = {term1} + {term2} (c1 first, then c2).",
            ),
            basis_text=basis_text,
            sample_lines=sample_lines,
            y_text=y_text,
            term1=term1,
            term2=term2,
        )
    else:
        text = (
            f"A function is sampled on the basis {basis_text} at {sample_lines}, giving "
            f"{y_text}. Find the least-squares coefficients c1, c2 in the fit "
            f"y = {term1} + {term2}, "
            "exactly (c1 first, then c2)."
        )

    swap_val = (c2, c1)
    sign1_val = (-c1, c2)
    sign2_val = (c1, -c2)
    check_pairwise_distinct(
        (c1, c2), swap_val, sign1_val, sign2_val, msg="p15c_function_space_approximation: a candidate collides"
    )
    candidates = [
        f"c1 = {rational_text(swap_val[0])}, c2 = {rational_text(swap_val[1])}",
        f"c1 = {rational_text(sign1_val[0])}, c2 = {rational_text(sign1_val[1])}",
        f"c1 = {rational_text(sign2_val[0])}, c2 = {rational_text(sign2_val[1])}",
    ]

    working = [
        f"Basis columns at the sample points: col1={[str(v) for v in col1]}, col2={[str(v) for v in col2]}.",
        f"Normal equations solved for (c1, c2): {answer_text}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates, unordered=False),
    )


@generator(
    "p15c_function_space_approximation",
    level=17,
    presets={
        "easy": dict(basis="one_xsq", evaluate_fn=False),
        "medium": dict(basis="x_xsq", evaluate_fn=False),
        "hard": dict(basis="sq_cube", evaluate_fn=False),
        "exam": dict(basis="sq_cube", evaluate_fn=True),
    },
)
def p15c_function_space_approximation(rng, basis="one_xsq", evaluate_fn=False):
    return _build_approx(rng, basis, evaluate_fn)
