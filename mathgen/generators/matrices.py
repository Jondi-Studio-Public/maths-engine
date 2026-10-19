"""Level 13: determinants, inverses, matrix-form linear systems (2x2 and 3x3)."""

from sympy import Matrix, Rational, eye, latex

from ..core import generator, check, paren_negative, pick_distractors, rational_text, xy_solution


def _mat_text(M):
    return "[" + "; ".join(" ".join(str(M[i, j]) for j in range(M.cols)) for i in range(M.rows)) + "]"


def _unimodular(rng, n, shears=4, hi=3):
    """Product of integer shear matrices: det = ±1, so the inverse is
    guaranteed to be an integer matrix (backwards construction)."""
    M = eye(n)
    for _ in range(shears):
        i, j = rng.sample(range(n), 2)
        E = eye(n)
        E[i, j] = rng.randint(-hi, hi) or 1
        M = M * E
    if rng.random() < 0.5:
        S = eye(n)
        S[0, 0] = -1
        M = M * S
    return M


def _det3_working(M, ans):
    """Cofactor expansion along the first row, with every 2 x 2 minor written out."""
    pn = paren_negative

    def minor(r, c):
        (a, b), (d, e) = [[M[i, j] for j in range(3) if j != c] for i in range(3) if i != r]
        return f"({pn(a)} × {pn(e)} - {pn(b)} × {pn(d)})", int(a * e - b * d)

    terms = [minor(0, c) for c in range(3)]
    line = f"det A = {pn(M[0, 0])} × {terms[0][0]} - {pn(M[0, 1])} × {terms[1][0]} + {pn(M[0, 2])} × {terms[2][0]}"
    return [
        "Expand along the first row: det A = a(ei - fh) - b(di - fg) + c(dh - eg).",
        line,
        f"= {pn(M[0, 0])} × {pn(terms[0][1])} - {pn(M[0, 1])} × {pn(terms[1][1])} + {pn(M[0, 2])} × {pn(terms[2][1])}",
        f"= {ans}",
    ]


def _system_working(A, b_vec, x0, y0):
    """Solve A x = b with the 2 x 2 inverse: x = (1/det) [[d, -b], [-c, a]] b."""
    a, b, c, d = (int(A[0, 0]), int(A[0, 1]), int(A[1, 0]), int(A[1, 1]))
    det = a * d - b * c
    p, q = int(b_vec[0]), int(b_vec[1])
    pn = paren_negative
    return [
        f"det A = {pn(a)} × {pn(d)} - {pn(b)} × {pn(c)} = {det}.",
        f"x = (1/{pn(det)}) × ({pn(d)} × {pn(p)} - {pn(b)} × {pn(q)}) = {d * p - b * q}/{pn(det)} = {x0}.",
        f"y = (1/{pn(det)}) × ({pn(a)} × {pn(q)} - {pn(c)} × {pn(p)}) = {a * q - c * p}/{pn(det)} = {y0}.",
    ]


@generator(
    "matrices",
    level=13,
    presets={
        "easy": dict(kinds=("determinant", "system"), hi=4),
        "medium": dict(kinds=("determinant", "inverse", "system"), hi=6),
        "hard": dict(kinds=("determinant", "inverse", "system", "determinant3", "inverse3"), hi=6),
    },
)
def matrices(rng, kinds=("determinant", "inverse", "system", "determinant3", "inverse3"), hi=6):
    kind = rng.choice(list(kinds))
    if kind == "determinant":
        M = Matrix(2, 2, [rng.randint(-hi, hi) for _ in range(4)])
        ans = int(M.det())
        check(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0] == ans)
        # Misconceptions: ad+bc sign slip; negated; forgot the bc term;
        # multiplied down the columns instead of the diagonals.
        ad, bc = M[0, 0] * M[1, 1], M[0, 1] * M[1, 0]
        cols = M[0, 0] * M[1, 0] - M[0, 1] * M[1, 1]
        cands = [str(ad + bc), str(-ans), str(ad), str(cols)]
        # Fallbacks, used only when the slips above all land on the answer
        # (e.g. a zero determinant): off-by-one and off-by-two arithmetic.
        cands += [str(ans + 1), str(ans - 1), str(ans + 2)]
        return dict(
            text=f"Find the determinant of A = {_mat_text(M)}",
            latex=rf"\det {latex(M)}",
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            working=[f"det = ({M[0, 0]})({M[1, 1]}) - ({M[0, 1]})({M[1, 0]}) = {ans}"],
            distractors=pick_distractors(str(ans), cands),
        )
    if kind == "determinant_product":
        # Given two 2x2 matrices A, B, find det(AB). Requires either multiplying A and B first then
        # taking the determinant, or applying det(AB) = det(A) * det(B) -- a genuine extra step /
        # property no lower tier of this skill needs. Computed via the product route, independently
        # checked against the det(A)*det(B) route.
        A = Matrix(2, 2, [rng.randint(-hi, hi) for _ in range(4)])
        B = Matrix(2, 2, [rng.randint(-hi, hi) for _ in range(4)])
        ans = int((A * B).det())
        detA, detB = int(A.det()), int(B.det())
        check(ans == detA * detB)
        # Misconceptions: summed the two determinants instead of multiplying
        # them; took det(A) alone (ignoring B); took det(B) alone (ignoring
        # A); added the matrices then took the determinant instead of
        # multiplying them first.
        summed = detA + detB
        added = int((A + B).det())
        cands = [str(summed), str(detA), str(detB), str(added)]
        return dict(
            text=f"A = {_mat_text(A)}, B = {_mat_text(B)}. Find det(AB).",
            latex=rf"\det(AB),\quad A={latex(A)},\ B={latex(B)}",
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            working=[f"det(AB) = det(A) × det(B) = ({detA}) × ({detB}) = {ans}"],
            distractors=pick_distractors(str(ans), cands),
        )
    if kind == "determinant3":
        while True:
            M = Matrix(3, 3, [rng.randint(-3, 3) for _ in range(9)])
            # A zero row or column makes a trivial question (and starves the
            # distractor pool) — regenerate.
            if all(any(M[i, j] for j in range(3)) for i in range(3)) and all(
                any(M[i, j] for i in range(3)) for j in range(3)
            ):
                break
        ans = int(M.det())
        # Independent check: cofactor expansion along the first row.
        cof = (
            M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
            - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
            + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
        )
        check(cof == ans)
        # Misconceptions: forgot to negate the middle cofactor (all-plus
        # expansion); negated overall; only summed the "down" diagonals.
        allplus = (
            M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
            + M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
            + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
        )
        down_only = M[0, 0] * M[1, 1] * M[2, 2] + M[0, 1] * M[1, 2] * M[2, 0] + M[0, 2] * M[1, 0] * M[2, 1]
        up_only = M[0, 2] * M[1, 1] * M[2, 0] + M[0, 0] * M[1, 2] * M[2, 1] + M[0, 1] * M[1, 0] * M[2, 2]
        diag = M[0, 0] * M[1, 1] * M[2, 2]
        return dict(
            text=f"Find the determinant of A = {_mat_text(M)}",
            latex=rf"\det {latex(M)}",
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            working=_det3_working(M, ans),
            distractors=pick_distractors(str(ans), [str(allplus), str(-ans), str(down_only), str(up_only), str(diag)]),
        )
    if kind == "determinant3_product":
        # Given two 3x3 matrices A, B, find det(AB). Requires either multiplying A and B first then
        # taking the determinant, or applying det(AB) = det(A) * det(B) -- a genuine extra step /
        # property no lower tier of this skill needs. Same zero-row/ zero-column reroll guard as the
        # existing "determinant3" branch, applied independently to each of A and B.
        def _draw3x3():
            while True:
                D = Matrix(3, 3, [rng.randint(-3, 3) for _ in range(9)])
                if all(any(D[i, j] for j in range(3)) for i in range(3)) and all(
                    any(D[i, j] for i in range(3)) for j in range(3)
                ):
                    return D

        A = _draw3x3()
        B = _draw3x3()
        ans = int((A * B).det())
        detA, detB = int(A.det()), int(B.det())
        check(ans == detA * detB)

        # Third, independent confirmation: cofactor expansion along the
        # first row (the existing "determinant3" branch's own independent
        # route), applied to each of A and B individually.
        def _cofactor3(D):
            return (
                D[0, 0] * (D[1, 1] * D[2, 2] - D[1, 2] * D[2, 1])
                - D[0, 1] * (D[1, 0] * D[2, 2] - D[1, 2] * D[2, 0])
                + D[0, 2] * (D[1, 0] * D[2, 1] - D[1, 1] * D[2, 0])
            )

        check(_cofactor3(A) == detA)
        check(_cofactor3(B) == detB)
        # Misconceptions: summed the two determinants instead of multiplying
        # them; took det(A) alone (ignoring B); took det(B) alone (ignoring
        # A); added the matrices then took the determinant instead of
        # multiplying them first.
        summed = detA + detB
        added = int((A + B).det())
        cands = [str(summed), str(detA), str(detB), str(added)]
        return dict(
            text=f"A = {_mat_text(A)}, B = {_mat_text(B)}. Find det(AB).",
            latex=rf"\det(AB),\quad A={latex(A)},\ B={latex(B)}",
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            working=[f"det(AB) = det(A) × det(B) = ({detA}) × ({detB}) = {ans}"],
            distractors=pick_distractors(str(ans), cands),
        )
    if kind == "inverse":
        M = _unimodular(rng, 2, shears=2)
        Minv = M.inv()
        check(M * Minv == eye(2))
        ans = [[int(Minv[i, j]) for j in range(2)] for i in range(2)]
        # Misconceptions on (1/det)[[d,-b],[-c,a]]: forgot to swap a and d;
        # forgot to negate b and c; forgot to divide by det (only differs when
        # det = -1; collisions are filtered).
        det = M.det()
        a_, b_, c_, d_ = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        cands = [
            _mat_text(Matrix([[a_, -b_], [-c_, d_]]) / det),
            _mat_text(Matrix([[d_, b_], [c_, a_]]) / det),
            _mat_text(Matrix([[d_, -b_], [-c_, a_]])),
            _mat_text(-Minv),  # global sign error
            _mat_text(M),
        ]  # "it's its own inverse"
        return dict(
            text=f"Find the inverse of A = {_mat_text(M)}",
            latex=rf"{latex(M)}^{{-1}}",
            answer=ans,
            answer_text=f"A⁻¹ = {_mat_text(Minv)}",
            answer_latex=latex(Minv),
            distractors=pick_distractors(f"A⁻¹ = {_mat_text(Minv)}", [f"A⁻¹ = {c}" for c in cands]),
        )
    if kind == "inverse3":
        M = _unimodular(rng, 3, shears=3, hi=2)
        Minv = M.inv()
        check(M * Minv == eye(3))
        ans = [[int(Minv[i, j]) for j in range(3)] for i in range(3)]
        # Misconceptions: forgot to transpose the cofactor matrix; sign error
        # overall; assumed the matrix is its own inverse.
        cands = [_mat_text(Minv.T), _mat_text(-Minv), _mat_text(M)]
        return dict(
            text=f"Find the inverse of A = {_mat_text(M)}",
            latex=rf"{latex(M)}^{{-1}}",
            answer=ans,
            answer_text=f"A⁻¹ = {_mat_text(Minv)}",
            answer_latex=latex(Minv),
            distractors=pick_distractors(f"A⁻¹ = {_mat_text(Minv)}", [f"A⁻¹ = {c}" for c in cands]),
        )
    if kind == "system_context":
        # A short two-item purchase scenario the student must
        # translate into Ax = b before solving -- same backwards
        # construction as the "system" branch below (pick x0, y0, pick A
        # with det(A) != 0 via the same reroll loop, compute b_vec = A * v,
        # answer via xy_solution(x0, y0)). The DIFFERENCE is presentation
        # only: instead of printing the raw matrix and vector, A's rows and
        # b_vec's entries are phrased as a two-item purchase scenario, so A's
        # entries are drawn from a small POSITIVE integer range (1-6, a
        # plausible "how many of item 1/2" count) rather than the generic
        # -hi..hi range the raw-matrix branch uses below -- a negative or
        # zero quantity of apples makes no scenario sense. x0, y0 (the two
        # item prices) are likewise drawn positive (1-9) for the same
        # real-world-scenario reason, a deliberate deviation from the raw
        # branch's -5..5 range.
        x0, y0 = rng.randint(1, 9), rng.randint(1, 9)
        while True:
            A = Matrix(2, 2, [rng.randint(1, 6) for _ in range(4)])
            if A.det() != 0:
                break
        v = Matrix([x0, y0])
        b_vec = A * v
        check(A.solve(b_vec) == v)
        # Confirm the scenario's stated purchase totals really are b_vec's
        # entries (direct by construction -- a plain assertion).
        check(A[0, 0] * x0 + A[0, 1] * y0 == b_vec[0])
        check(A[1, 0] * x0 + A[1, 1] * y0 == b_vec[1])

        def _qty(n, singular, plural):
            return f"1 {singular}" if n == 1 else f"{n} {plural}"

        text = (
            f"{_qty(A[0, 0], 'apple', 'apples')} and {_qty(A[0, 1], 'melon', 'melons')} "
            f"cost ${b_vec[0]} in total; "
            f"{_qty(A[1, 0], 'apple', 'apples')} and {_qty(A[1, 1], 'melon', 'melons')} "
            f"cost ${b_vec[1]} in total. Find the price of one apple and one melon."
        )
        base = xy_solution(x0, y0)
        ans_text = base["answer_text"]
        # Misconceptions specific to translating a word scenario into a
        # system, distinct from the sign-error/swapped-pair ideas
        # xy_solution already offers (both still included below, reused
        # verbatim): (1) reversed the two items' roles when setting up the
        # system -- solved A^T x = b instead of Ax = b, a genuine "which
        # number goes where" slip; (2) used only ONE of the two purchase
        # statements -- solved a single equation, treating the other item's
        # price as already known (assumed 0).
        wrong_T = A.T.solve(b_vec)
        roles_reversed = f"x = {rational_text(wrong_T[0])}, y = {rational_text(wrong_T[1])}"
        single_eq_x = rational_text(Rational(int(b_vec[0]), int(A[0, 0])))
        single_eq = f"x = {single_eq_x}, y = 0"
        swapped = f"x = {y0}, y = {x0}"
        sign_x = f"x = {-x0}, y = {y0}"
        sign_y = f"x = {x0}, y = {-y0}"
        cands = [swapped, roles_reversed, single_eq, sign_x, sign_y]
        return dict(
            text=text,
            latex=rf"{latex(A)} \mathbf{{x}} = {latex(b_vec)}",
            answer=base["answer"],
            answer_text=ans_text,
            answer_latex=base["answer_latex"],
            working=_system_working(A, b_vec, x0, y0),
            distractors=pick_distractors(ans_text, cands, n=4),
        )
    # 2x2 system in matrix form, backwards from a chosen solution.
    x0, y0 = rng.randint(-5, 5), rng.randint(-5, 5)
    while True:
        A = Matrix(2, 2, [rng.randint(-hi, hi) for _ in range(4)])
        if A.det() != 0:
            break
    v = Matrix([x0, y0])
    b_vec = A * v
    check(A.solve(b_vec) == v)
    solved = xy_solution(x0, y0)
    if len(solved["distractors"]) < 3:
        # x0 = y0 = 0 (or a pair the swap/sign slips cannot move) starves the
        # standard slips; top up with off-by-one values.
        extras = [f"x = {x0 + 1}, y = {y0}", f"x = {x0}, y = {y0 + 1}", f"x = {x0 - 1}, y = {y0 - 1}"]
        solved["distractors"] = pick_distractors(solved["answer_text"], solved["distractors"] + extras)
    return dict(
        text=f"Solve A x = b where A = {_mat_text(A)} and b = ({b_vec[0]}, {b_vec[1]})",
        latex=rf"{latex(A)} \mathbf{{x}} = {latex(b_vec)}",
        working=_system_working(A, b_vec, x0, y0),
        **solved,
    )
