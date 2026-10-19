"""Generates `uc2-lp-duality-dual-formulation` questions (easy, medium, hard) and the exam chain
`exd_x_r3l_lp_duality`.

``uc2-lp-duality-dual-formulation`` (``optimisation``, answer ``matrix``). The dual of a linear programme is written as
an augmented matrix in one fixed layout, so each answer is unique: one row for each dual constraint (the coefficients
of y1, ..., ym, then the right-hand side), followed by the row of the dual objective's coefficients and a final 0.
For a primal with data A (m x n), b and c the matrix is [[A^T | c], [b | 0]]; the primal's objective decides only
whether the dual is a minimisation (primal max, <=) or a maximisation (primal min, >=).

* easy: the dual of a primal in standard form (max with <= constraints, or min with >= constraints).
* medium: a maximisation with a mix of <= and >= constraints; the stem says to rewrite each >= constraint as <= (multiply
  by -1) first, so the dual variables are all non-negative and the matrix carries the sign changes.
* hard: the primal is max c.x, A x <= b, x >= 0 with three constraints and two variables, and an optimal x* is given
  with both components positive and exactly one constraint slack. By complementary slackness the dual variable of the
  slack constraint is 0 and the other two solve A_B^T y = c; the answer is the column vector y*.
* exam (``exd_x_r3l_lp_duality``): (a) the dual as an augmented matrix, (b) the slack of each primal constraint at x*,
  (c) the optimal dual value y_k* of a binding constraint, (d) the new optimum when b_k rises by a stated amount
  (follows through from (c)).

The hard and exam instances are built backwards: choose x* and a dual y* >= 0 with one zero, set c = A^T y*, and set
b_i = A_i x* on the binding rows and A_i x* + slack on the other. Optimality is then certain, and is re-checked by
enumerating every corner point of the feasible region and by strong duality (b.y* = c.x*).
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, pick_distractors
from ._r3l_kit import corner_points, linear_text, num, objective, sub
from ._ucv1_linalg import fm, mat_str, matrix_distractors, rows_value, transpose


def _names(prefix, k):
    return [sub(prefix, i + 1) for i in range(k)]


def _constraints_text(a, b, senses, xs):
    sym = {"<=": "≤", ">=": "≥"}
    return "; ".join(f"{linear_text(row, xs)} {sym[s]} {r}" for row, r, s in zip(a, b, senses))


def _dual_matrix(a, b, c):
    """[[A^T | c], [b | 0]] as Fractions."""
    at = transpose(fm(a))
    rows = [at[j] + [Fraction(c[j])] for j in range(len(c))]
    rows.append([Fraction(v) for v in b] + [Fraction(0)])
    return rows


def _layout(m) -> str:
    ys = ", ".join(_names("y", m))
    return (
        f"Write the dual as an augmented matrix: one row for each dual constraint (the coefficients of {ys}, then the "
        f"right-hand side), followed by one row holding the coefficients of the dual objective and a final 0. "
        "The dual variable yᵢ belongs to constraint i."
    )


def _dual_distractors(a, b, c, answer, n_wanted=3):
    m, n = len(a), len(c)
    at = transpose(fm(a))
    good = _dual_matrix(a, b, c)
    cands = []
    # rows of A written as the dual constraints (A not transposed): m rows of length n + 1, then b padded
    wrong1 = [list(map(Fraction, a[i])) + [Fraction(b[i])] for i in range(m)] + [
        [Fraction(v) for v in c] + [Fraction(0)]
    ]
    cands.append((wrong1, "r3l-dual-not-transposed"))
    neg_obj = [list(r) for r in good]
    neg_obj[-1] = [-v for v in good[-1]]
    cands.append((neg_obj, "r3l-dual-objective-row-negated"))
    neg_rhs = [list(r) for r in good]
    for i in range(n):
        neg_rhs[i][-1] = -good[i][-1]
    cands.append((neg_rhs, "r3l-dual-inequality-flip-sign"))
    swapped = [list(r) for r in good]
    for i in range(n):
        swapped[i][-1] = Fraction(c[i])
    last = [Fraction(c[i]) if i < n else Fraction(0) for i in range(m)] + [Fraction(0)]
    swapped[-1] = last
    cands.append((swapped, "r3l-dual-uses-primal-objective"))
    return matrix_distractors(cands, rows_value(answer))[:n_wanted]


def _sensible(rng, m, n):
    a = [[rng.randint(0, 6) for _ in range(n)] for _ in range(m)]
    b = [rng.randint(6, 40) for _ in range(m)]
    c = [rng.randint(1, 9) for _ in range(n)]
    return a, b, c


def _draw_standard(rng, m, n):
    for _ in attempts(500, "a dual with distinct entries"):
        a, b, c = _sensible(rng, m, n)
        if any(sum(r) == 0 for r in a) or any(all(a[i][j] == 0 for i in range(m)) for j in range(n)):
            continue
        if len({tuple(r) for r in a}) < m or sum(1 for r in a for v in r if v == 0) > 1:
            continue
        if len(set(b)) < m:
            continue
        return a, b, c
    raise AssertionError  # pragma: no cover


def _easy(rng):
    m, n = rng.choice([(2, 2), (3, 2), (2, 3), (3, 3)])
    a, b, c = _draw_standard(rng, m, n)
    xs = _names("x", n)
    if rng.random() < 0.5:
        primal = f"Maximise z = {linear_text(c, xs)} subject to {_constraints_text(a, b, ['<='] * m, xs)}, and {', '.join(xs)} ≥ 0."
        kind = "min"
    else:
        primal = f"Minimise w = {linear_text(c, xs)} subject to {_constraints_text(a, b, ['>='] * m, xs)}, and {', '.join(xs)} ≥ 0."
        kind = "max"
    dual = _dual_matrix(a, b, c)
    distractors = _dual_distractors(a, b, c, dual)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=f"{primal} {_layout(m)}",
        answer_kind="matrix",
        answer=rows_value(dual),
        answer_text=mat_str(dual),
        answer_latex="plain:" + mat_str(dual),
        working=[
            f"The dual is a {kind}imisation with one variable for each of the {m} primal constraints and one constraint for each of the {n} primal variables.",
            "The coefficients of dual constraint j are column j of A, its right-hand side is c_j, and the dual objective uses b.",
            f"Hence {mat_str(dual)}.",
        ],
        distractors=distractors,
    )


def _medium(rng):
    m, n = rng.choice([(3, 2), (3, 3), (3, 2)])
    a, b, c = _draw_standard(rng, m, n)
    ge = sorted(rng.sample(range(m), rng.choice([1, 2])))
    senses = ["<=" if i not in ge else ">=" for i in range(m)]
    xs = _names("x", n)
    # the stem as the learner reads it; the standard-form data have the >= rows negated
    a_std = [[-v if i in ge else v for v in a[i]] for i in range(m)]
    b_std = [-v if i in ge else v for i, v in enumerate(b)]
    # the ">=" right-hand sides are small, so the negated values are modest
    primal = (
        f"Maximise z = {linear_text(c, xs)} subject to {_constraints_text(a, b, senses, xs)}, and {', '.join(xs)} ≥ 0."
    )
    dual = _dual_matrix(a_std, b_std, c)
    cands = []
    plain = _dual_matrix(a, b, c)  # forgot to negate anything
    cands.append((plain, "r3l-dual-ge-row-not-negated"))
    half = _dual_matrix(a_std, b, c)  # negated the coefficients but not the right-hand side
    cands.append((half, "r3l-dual-ge-rhs-not-negated"))
    half2 = _dual_matrix(a, b_std, c)  # negated only the right-hand side
    cands.append((half2, "r3l-dual-ge-coefficients-not-negated"))
    neg_obj = [list(r) for r in dual]
    neg_obj[-1] = [-v for v in dual[-1]]
    cands.append((neg_obj, "r3l-dual-objective-row-negated"))
    distractors = matrix_distractors(cands, rows_value(dual))[:3]
    check(len(distractors) >= 2, "too few distractors")
    rewritten = _constraints_text(a_std, b_std, ["<="] * m, xs)
    return dict(
        text=(
            f"{primal} First rewrite each ≥ constraint as a ≤ constraint by multiplying it by −1, so that every dual "
            f"variable is non-negative. {_layout(m)}"
        ),
        answer_kind="matrix",
        answer=rows_value(dual),
        answer_text=mat_str(dual),
        answer_latex="plain:" + mat_str(dual),
        working=[
            f"With every constraint as ≤ the primal is: {rewritten}.",
            f"The dual's constraint rows come from the columns of this system, so {mat_str(dual)}.",
        ],
        distractors=distractors,
    )


# -- complementary slackness instances ----------------------------------------------------------------------------------


def _backwards_instance(rng, m=3, n=2):
    """(A, b, c, x*, y*, slack index s, binding rows) with A x* <= b optimal, built from x* and y*."""
    for _ in attempts(2000, "a complementary-slackness instance"):
        a = [[rng.randint(1, 6) for _ in range(n)] for _ in range(m)]
        if len({Fraction(r[0], r[1]) for r in a}) < m:
            continue
        s = rng.randrange(m)
        binding = [i for i in range(m) if i != s]
        det = a[binding[0]][0] * a[binding[1]][1] - a[binding[0]][1] * a[binding[1]][0]
        if det == 0:
            continue
        xstar = [rng.randint(1, 6) for _ in range(n)]
        ystar = [0] * m
        for i in binding:
            ystar[i] = rng.randint(1, 4)
        c = [sum(a[i][j] * ystar[i] for i in range(m)) for j in range(n)]
        slack = [0] * m
        slack[s] = rng.randint(1, 6)
        b = [sum(a[i][j] * xstar[j] for j in range(n)) + slack[i] for i in range(m)]
        if max(b) > 70 or max(c) > 40:
            continue
        # independent checks: x* is the best corner point; strong duality
        cons = [(a[i][0], a[i][1], b[i], "<=") for i in range(m)]
        pts = corner_points(cons)
        best = max(pts, key=lambda p: objective(c, p))
        if best != tuple(Fraction(v) for v in xstar):
            continue
        if sum(b[i] * ystar[i] for i in range(m)) != sum(c[j] * xstar[j] for j in range(n)):
            continue
        if sum(1 for p in pts if objective(c, p) == objective(c, best)) != 1:
            continue
        return a, b, c, xstar, ystar, s, binding, slack
    raise AssertionError  # pragma: no cover


def _solve2(rows, rhs):
    (p, q), (r, s) = rows
    det = Fraction(p * s - q * r)
    return [Fraction(rhs[0] * s - q * rhs[1]) / det, Fraction(p * rhs[1] - rhs[0] * r) / det]


def _hard(rng):
    a, b, c, xstar, ystar, s, binding, slack = _backwards_instance(rng)
    m, n = 3, 2
    xs = _names("x", n)
    ys = _names("y", m)
    # SymPy-free cross-check by solving the 2x2 system for the binding rows directly
    atb = [[a[binding[0]][0], a[binding[1]][0]], [a[binding[0]][1], a[binding[1]][1]]]
    sol = _solve2(atb, c)
    check(
        [int(v) if v.denominator == 1 else v for v in sol] == [ystar[binding[0]], ystar[binding[1]]],
        "dual solve differs",
    )
    answer = [[Fraction(v)] for v in ystar]
    cands = []
    for pair in ((0, 1), (0, 2), (1, 2)):
        if set(pair) == set(binding):
            continue
        rows = [[a[pair[0]][0], a[pair[1]][0]], [a[pair[0]][1], a[pair[1]][1]]]
        det = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
        if det == 0:
            continue
        y = _solve2(rows, c)
        full = [Fraction(0)] * m
        full[pair[0]], full[pair[1]] = y
        cands.append(([[v] for v in full], "r3l-dual-wrong-binding-set"))
    # solving A_B y = c (not transposed)
    rows_nt = [list(map(Fraction, a[binding[0]])), list(map(Fraction, a[binding[1]]))]
    det = rows_nt[0][0] * rows_nt[1][1] - rows_nt[0][1] * rows_nt[1][0]
    if det != 0:
        y = _solve2(rows_nt, c)
        full = [Fraction(0)] * m
        full[binding[0]], full[binding[1]] = y
        cands.append(([[v] for v in full], "r3l-dual-system-not-transposed"))
    cands.append(([[Fraction(v)] for v in xstar] + [[Fraction(0)]], "r3l-dual-primal-solution-reused"))
    cands.append(([[Fraction(slack[i] if i == s else 0)] for i in range(m)], "r3l-dual-slack-as-multiplier"))
    distractors = matrix_distractors(cands, rows_value(answer))[:3]
    check(len(distractors) >= 2, "too few distractors")
    text = (
        f"Maximise z = {linear_text(c, xs)} subject to {_constraints_text(a, b, ['<='] * m, xs)}, and {', '.join(xs)} ≥ 0. "
        f"The optimal solution is x* = ({xstar[0]}, {xstar[1]}). Use complementary slackness to find the optimal solution "
        f"y* = ({', '.join(ys)}) of the dual, and give it as a column vector."
    )
    return dict(
        text=text,
        answer_kind="matrix",
        answer=rows_value(answer),
        answer_text=mat_str(answer),
        answer_latex="plain:" + mat_str(answer),
        working=[
            f"At x* constraint {s + 1} has slack {slack[s]}, so {ys[s]} = 0 (its dual variable is zero by complementary slackness).",
            "Both x₁* and x₂* are positive, so both dual constraints hold with equality: Aᵀy = c.",
            f"Solving the two equations in the remaining variables gives y* = {mat_str(answer)}.",
        ],
        distractors=distractors,
    )


@generator(
    "r3l_lp_duality",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3l_lp_duality(rng, tier="easy"):
    """The dual of a linear programme as a matrix, or its optimal solution by complementary slackness."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------------


def _reoptimise(a, b, c, k, delta):
    cons = [(a[i][0], a[i][1], b[i] + (delta if i == k else 0), "<=") for i in range(len(a))]
    return max(objective(c, p) for p in corner_points(cons))


@generator("exd_x_r3l_lp_duality", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r3l_lp_duality(rng, tier="exam"):
    """Exam chain: the dual matrix, constraint slacks, a dual value, and the new optimum (follows through)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "an instance with a valid unit increase"):
        a, b, c, xstar, ystar, s, binding, slack = _backwards_instance(rng)
        k = rng.choice(binding)
        delta = rng.randint(1, 2)
        z0 = sum(c[j] * xstar[j] for j in range(2))
        new = _reoptimise(a, b, c, k, delta)
        if new == z0 + ystar[k] * delta:
            break
    else:  # pragma: no cover
        raise AssertionError
    m, n = 3, 2
    xs = _names("x", n)
    dual = _dual_matrix(a, b, c)
    a_dist = _dual_distractors(a, b, c, dual, 3)
    slacks = [b[i] - sum(a[i][j] * xstar[j] for j in range(n)) for i in range(m)]
    check(slacks == slack, "slack differs")
    s_text = ", ".join(map(str, slacks))

    def vec(values, tag):
        return Distractor(", ".join(map(str, values)), tag, list(values))

    b_cands = [
        vec([b[i] for i in range(m)], "r3l-dual-constraint-rhs-as-slack"),
        vec([sum(a[i][j] * xstar[j] for j in range(n)) for i in range(m)], "r3l-dual-lhs-as-slack"),
        vec([slacks[(i + 1) % m] for i in range(m)], "r3l-dual-slack-wrong-row"),
        vec([-v for v in slacks], "r3l-dual-slack-sign"),
    ]
    b_dist = pick_distractors(s_text, b_cands, 3)
    yk = ystar[k]
    c_cands = [Distractor(str(v), "r3l-dual-wrong-binding-set", v) for v in ystar if v != yk]
    c_cands += [
        Distractor(str(z0), "r3l-dual-primal-solution-reused", z0),
        Distractor(str(xstar[0]), "r3l-dual-primal-solution-reused", xstar[0]),
    ]
    c_dist = pick_distractors(str(yk), c_cands, 3)
    d_cands = [
        Distractor(str(z0 + delta), "r3l-dual-price-ignored", z0 + delta),
        Distractor(str(yk * delta), "r3l-dual-increase-not-added", yk * delta),
        Distractor(str(z0 - yk * delta), "r3l-dual-price-sign", z0 - yk * delta),
    ]
    d_dist = pick_distractors(str(new), d_cands, 3)
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    ys = _names("y", m)
    text = (
        f"Consider the linear programme: maximise z = {linear_text(c, xs)} subject to {_constraints_text(a, b, ['<='] * m, xs)}, "
        f"and {', '.join(xs)} ≥ 0. Its optimal solution is x* = ({xstar[0]}, {xstar[1]})."
    )
    return dict(
        text=text,
        parts=[
            dict(
                prompt=_layout(m).replace("Write the dual", "Write the dual of the linear programme"),
                answer_kind="matrix",
                answer=rows_value(dual),
                answer_text=mat_str(dual),
                answer_latex="plain:" + mat_str(dual),
                marks=3,
                working=[f"The dual's rows come from the columns of A: {mat_str(dual)}."],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the slack in each of the three constraints at x*, in order.",
                answer_kind="ordered_list",
                answer=slacks,
                answer_text=s_text,
                marks=2,
                working=[f"Constraint i has slack bᵢ − (Ax*)ᵢ = {s_text}."],
                distractors=b_dist,
            ),
            dict(
                prompt=f"Use complementary slackness to find the optimal value of the dual variable {ys[k]}.",
                answer_kind="number",
                answer=yk,
                answer_text=str(yk),
                marks=3,
                sensible_range=(0, None),
                working=[
                    f"Constraint {s + 1} has positive slack, so {ys[s]} = 0; x* has both components positive, so Aᵀy = c holds with equality.",
                    f"Solving gives {ys[k]} = {yk}.",
                ],
                distractors=c_dist,
            ),
            dict(
                prompt=f"The right-hand side of constraint {k + 1} is increased by {delta}. Find the new optimal value of z, assuming the same constraints stay binding.",
                answer_kind="number",
                answer=new,
                answer_text=str(new),
                marks=2,
                follow_through=f"{z0} + {delta}*part_c",
                sensible_range=(0, None),
                working=[f"z rises by {ys[k]}·{delta} = {yk * delta}, from {z0} to {new}."],
                distractors=d_dist,
            ),
        ],
    )
