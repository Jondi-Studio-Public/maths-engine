"""Fit a two-feature linear regression (``ucv1-multivariate-regression``, ``statistical-learning``, topic
``p14e_multivariate_regression``, ``answer: ordered_list``).

**Kernel** (the skill's stated contract): given a
two-feature model ``y = b0 + b1·x1 + b2·x2``, find ``[b0, b1, b2]``, in that
stated order, solving the normal equations ``(X^T X) b = X^T y`` for the
supplied design -- either raw ``(x1, x2, y)`` data, or the Gram matrix
``X^T X``/``X^T y`` directly.

**Grader kind.** A planning draft said ``answer_kind: list``, but that was only a
label. ``grade_list`` (``mathgen/grading/typed.py``, line 638) is
order-insensitive -- "the set of values must match exactly" -- which would
silently accept a transposed ``[b0, b2, b1]``, a genuinely different (wrong)
answer. This skill uses ``answer: ordered_list`` (``grade_ordered_list``,
line 870, order-sensitive, positional) instead, and every prompt states
"Give your answer in the order [b0, b1, b2]." exactly once.

**Boundary against the prerequisite** ``least-squares-line``
(``mathgen/generators/university_stat_learning.py``): that skill fits a
ONE-feature line ``y = mx + b`` from raw data, answering an unordered pair
``{m, b}`` (``answer: pair``). This skill fits a TWO-feature model
``y = b0 + b1·x1 + b2·x2``, answers a length-3 ORDERED list, and (hard/exam
tiers) never shows raw data at all -- the learner solves a genuine 3x3 (or
2x2, exam) linear system as algebra, a decision `least-squares-line` never
asks for.

**Construct-to-target.** ``b0, b1, b2`` are picked as small integers FIRST.
Easy uses the fixed n=4 orthogonal design ``x1 = (-3, -1, 1, 3)``,
``x2 = (1, -1, -1, 1)`` (verified: pairwise-orthogonal with the intercept
column, so ``X^T X`` is exactly ``diag(4, 20, 4)``) with a residual along the
n=4 cubic contrast ``c = (-1, 3, -3, 1)`` (verified orthogonal to every
design column), so the fit recovers ``b0, b1, b2`` exactly regardless of the
residual scale ``k``. Medium uses a fixed non-orthogonal n=4 design
``x1 = (0, 1, 2, 3)``, ``x2 = (0, 1, 1, 2)`` with its own (computed, not
guessed) residual direction ``(1, -1, -1, 1)``, verified orthogonal to
``1``, ``x1`` and ``x2`` for THIS design. Hard picks a diagonally dominant
integer-symmetric 3x3 Gram matrix directly and computes ``X^T y = S @ b``.
Exam presents the intercept-decoupled block form (``x1``, ``x2`` centred, so
the top row/column of ``S`` is ``[n, 0, 0]``) plus the sample means, so the
learner solves a 2x2 sub-system for ``b1, b2`` and then derives
``b0 = ybar - b1×x1bar - b2×x2bar`` -- a genuinely new final combination
step no lower tier needs.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
)

_MAX_DRAW_ATTEMPTS = 500

_ORDER = "Give your answer in the order [b0, b1, b2]."

# --- easy: fixed n=4 orthogonal design -------------------------------------
_X1_EASY = (-3, -1, 1, 3)
_X2_EASY = (1, -1, -1, 1)
_C_EASY = (-1, 3, -3, 1)  # n=4 cubic contrast; orthogonal to 1, x1, x2 above

# --- medium: fixed n=4 non-orthogonal design -------------------------------
_X1_MED = (0, 1, 2, 3)
_X2_MED = (0, 1, 1, 2)
_C_MED = (1, -1, -1, 1)  # orthogonal to 1, x1, x2 for THIS design (verified)

_EASY_TEMPLATES = (
    "Fit the two-feature linear model y = b0 + b1·x1 + b2·x2 to these four "
    "data points (x1, x2, y): ({p0}), ({p1}), ({p2}), ({p3}). {order}",
    "For the two-feature design with data points (x1, x2, y) = ({p0}), "
    "({p1}), ({p2}), ({p3}), find the fitted coefficients of "
    "y = b0 + b1·x1 + b2·x2. {order}",
    "A two-feature linear regression y = b0 + b1·x1 + b2·x2 is fitted to "
    "(x1, x2, y) = ({p0}), ({p1}), ({p2}), ({p3}). {order}",
)

_MEDIUM_TEMPLATES = (
    "Fit the two-feature linear model y = b0 + b1·x1 + b2·x2 to these four "
    "data points (x1, x2, y): ({p0}), ({p1}), ({p2}), ({p3}), where x1 and "
    "x2 are not independent of each other. {order}",
    "For the two-feature design with data points (x1, x2, y) = ({p0}), "
    "({p1}), ({p2}), ({p3}) (x1 and x2 are correlated here), find the "
    "fitted coefficients of y = b0 + b1·x1 + b2·x2. {order}",
    "A two-feature linear regression y = b0 + b1·x1 + b2·x2 is fitted to "
    "(x1, x2, y) = ({p0}), ({p1}), ({p2}), ({p3}); x1 and x2 are not "
    "orthogonal for this design. {order}",
)

_HARD_TEMPLATES = (
    "A two-feature linear model y = b0 + b1·x1 + b2·x2 has Gram matrix "
    "XtX = [[{s00}, {s01}, {s02}], [{s01}, {s11}, {s12}], "
    "[{s02}, {s12}, {s22}]] and Xty = [{t0}, {t1}, {t2}]. Solve "
    "XtX × b = Xty for b = [b0, b1, b2]. {order}",
    "No raw data is given here -- only the Gram matrix "
    "XtX = [[{s00}, {s01}, {s02}], [{s01}, {s11}, {s12}], "
    "[{s02}, {s12}, {s22}]] and the vector Xty = [{t0}, {t1}, {t2}] for a "
    "two-feature model y = b0 + b1·x1 + b2·x2. Solve for b = [b0, b1, b2]. "
    "{order}",
    "For a two-feature regression, XtX = [[{s00}, {s01}, {s02}], "
    "[{s01}, {s11}, {s12}], [{s02}, {s12}, {s22}]] and "
    "Xty = [{t0}, {t1}, {t2}]. Find b = [b0, b1, b2] solving XtX × b = Xty. "
    "{order}",
)

_EXAM_TEMPLATES = (
    "For a two-feature model, x1 and x2 have already been centred at their "
    "means, so the Gram matrix decouples: XtX = [[{n}, 0, 0], "
    "[0, {a}, {b}], [0, {b}, {c}]]. The lower two entries of the matching "
    "Xty vector are [{u1}, {u2}]. The sample means are x1bar = {x1bar}, "
    "x2bar = {x2bar}, ybar = {ybar}. First solve the 2x2 sub-system for "
    "b1, b2, then find b0 = ybar - b1×x1bar - b2×x2bar. {order}",
    "x1 and x2 are centred, so the intercept row/column of XtX is "
    "[{n}, 0, 0]; the full matrix is XtX = [[{n}, 0, 0], [0, {a}, {b}], "
    "[0, {b}, {c}]]. The bottom two entries of Xty are [{u1}, {u2}]. Given "
    "x1bar = {x1bar}, x2bar = {x2bar}, ybar = {ybar}, find b = [b0, b1, b2] "
    "(solve the 2x2 block for b1, b2 first, then combine with the means "
    "for b0). {order}",
    "A two-feature model's Gram matrix, with x1 and x2 centred, is "
    "XtX = [[{n}, 0, 0], [0, {a}, {b}], [0, {b}, {c}]], with Xty's lower "
    "two entries [{u1}, {u2}]. The sample means are x1bar = {x1bar}, "
    "x2bar = {x2bar}, ybar = {ybar}. Solve for b = [b0, b1, b2], deriving "
    "b0 from the means after solving for b1, b2. {order}",
)


def _solve3(rows, rhs) -> list[Fraction]:
    sol = sympy.Matrix(rows).solve(sympy.Matrix(rhs))
    return [Fraction(int(sol[i].p), int(sol[i].q)) for i in range(3)]


def _solve2(rows, rhs) -> list[Fraction]:
    sol = sympy.Matrix(rows).solve(sympy.Matrix(rhs))
    return [Fraction(int(sol[i].p), int(sol[i].q)) for i in range(2)]


def _round_component(value) -> int:
    """Round a (possibly non-integer) misconception value to the nearest
    integer, ties-to-even via `Fraction.__round__`, so every distractor
    renders in the same exact-integer shape as the canonical answer (never
    a fraction sticking out among an all-integer option set)."""
    return int(round(Fraction(value)))


def _fmt(triple) -> str:
    """Canonical/distractor rendering, e.g. ``[5, -2, 3]``. Deliberately a
    bare bracketed list -- not ``"b0 = 5, b1 = -2, b2 = 3"`` -- because
    `grade_ordered_list`/`_split_ordered_items` splits a typed answer on
    top-level commas and parses each item as a bare number
    (`_parse_numeric_item`); a "b0 = " label glued onto each item would make
    every split item fail to parse as a Fraction and reject the canonical
    answer outright. The prompt still names each position ("in the order
    [b0, b1, b2]"), so the labels live in the QUESTION, not the graded
    answer string."""
    b0, b1, b2 = triple
    return f"[{b0}, {b1}, {b2}]"


def _build_easy(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        b0 = rng.randint(-6, 6)
        b1 = rng.randint(-6, 6)
        if b1 == 0:
            continue
        b2 = rng.randint(-6, 6)
        if b2 == 0 or b2 == b1:
            continue
        k = rng.choice((-3, -2, -1, 1, 2, 3))

        y = [b0 + b1 * x1i + b2 * x2i + k * ci for x1i, x2i, ci in zip(_X1_EASY, _X2_EASY, _C_EASY)]

        # Independent check: recompute the Gram matrix and cross-product
        # vector from the actual rendered data (not from b0/b1/b2/k),
        # then solve the normal equations the other way.
        s_computed = [
            [len(_X1_EASY), sum(_X1_EASY), sum(_X2_EASY)],
            [sum(_X1_EASY), sum(x * x for x in _X1_EASY), sum(a * b for a, b in zip(_X1_EASY, _X2_EASY))],
            [sum(_X2_EASY), sum(a * b for a, b in zip(_X1_EASY, _X2_EASY)), sum(x * x for x in _X2_EASY)],
        ]
        check(
            s_computed == [[4, 0, 0], [0, 20, 0], [0, 0, 4]],
            "easy-tier orthogonal Gram matrix is not diagonal as expected",
        )
        xty = [sum(y), sum(a * b for a, b in zip(_X1_EASY, y)), sum(a * b for a, b in zip(_X2_EASY, y))]
        solved = _solve3(s_computed, xty)
        check_equivalent(tuple(solved), (b0, b1, b2), "easy-tier normal-equations solve mismatch")

        answer = (b0, b1, b2)
        candidates = {
            "wrong-order": (b0, b2, b1),
            "ignores-one-feature": (b0, b1, 0),
            "sign-flip-b1": (b0, -b1, b2),
            "sign-flip-b2": (b0, b1, -b2),
        }
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"easy-tier collision: b0={b0}, b1={b1}, b2={b2}, k={k}",
            )
        except VerificationError:
            continue

        points = [f"{x1i}, {x2i}, {yi}" for x1i, x2i, yi in zip(_X1_EASY, _X2_EASY, y)]
        text = phrase(rng, _EASY_TEMPLATES, p0=points[0], p1=points[1], p2=points[2], p3=points[3], order=_ORDER)
        answer_text = _fmt(answer)
        distractor_strs = [_fmt(v) for v in candidates.values()]
        return dict(
            text=text,
            answer=[b0, b1, b2],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                "With the intercept and both features pairwise orthogonal, "
                "XtX = diag(4, 20, 4), so each coefficient solves "
                "independently.",
                f"b0 = mean(y) = {b0}. b1 = sum(x1×y)/20 = {b1}. b2 = sum(x2×y)/4 = {b2}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=4),
        )
    raise VerificationError("could not draw a collision-free easy-tier question")


def _build_medium(rng) -> dict:
    s_fixed = [
        [len(_X1_MED), sum(_X1_MED), sum(_X2_MED)],
        [sum(_X1_MED), sum(x * x for x in _X1_MED), sum(a * b for a, b in zip(_X1_MED, _X2_MED))],
        [sum(_X2_MED), sum(a * b for a, b in zip(_X1_MED, _X2_MED)), sum(x * x for x in _X2_MED)],
    ]
    check(s_fixed == [[4, 6, 4], [6, 14, 9], [4, 9, 6]], "medium-tier non-orthogonal Gram matrix is not as expected")
    # One-feature (x1-only, with intercept) submatrix, constant for this
    # fixed design -- used by the ignores-one-feature misconception below.
    sub_x1 = [[s_fixed[0][0], s_fixed[0][1]], [s_fixed[1][0], s_fixed[1][1]]]

    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        b0 = rng.randint(-8, 8)
        b1 = rng.randint(-8, 8)
        if b1 == 0:
            continue
        b2 = rng.randint(-8, 8)
        if b2 == 0 or b2 == b1:
            continue
        k = rng.choice((-3, -2, -1, 1, 2, 3))

        y = [b0 + b1 * x1i + b2 * x2i + k * ci for x1i, x2i, ci in zip(_X1_MED, _X2_MED, _C_MED)]

        xty = [sum(y), sum(a * b for a, b in zip(_X1_MED, y)), sum(a * b for a, b in zip(_X2_MED, y))]
        solved = _solve3(s_fixed, xty)
        check_equivalent(tuple(solved), (b0, b1, b2), "medium-tier normal-equations solve mismatch")

        # ignores-one-feature: fit y = b0' + b1'*x1 only (drop x2 entirely,
        # a genuinely different vector since x1/x2 are correlated here).
        sub_rhs = [xty[0], xty[1]]
        b0_ig, b1_ig = _solve2(sub_x1, sub_rhs)
        ignores_one = (_round_component(b0_ig), _round_component(b1_ig), 0)

        # off-diagonal-ignored: solve as if XtX were diagonal.
        off_diag = (
            _round_component(Fraction(xty[0], s_fixed[0][0])),
            _round_component(Fraction(xty[1], s_fixed[1][1])),
            _round_component(Fraction(xty[2], s_fixed[2][2])),
        )

        answer = (b0, b1, b2)
        candidates = {
            "wrong-order": (b0, b2, b1),
            "ignores-one-feature": ignores_one,
            "off-diagonal-ignored": off_diag,
            "sign-flip-b1": (b0, -b1, b2),
        }
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"medium-tier collision: b0={b0}, b1={b1}, b2={b2}, k={k}",
            )
        except VerificationError:
            continue

        points = [f"{x1i}, {x2i}, {yi}" for x1i, x2i, yi in zip(_X1_MED, _X2_MED, y)]
        text = phrase(rng, _MEDIUM_TEMPLATES, p0=points[0], p1=points[1], p2=points[2], p3=points[3], order=_ORDER)
        answer_text = _fmt(answer)
        distractor_strs = [_fmt(v) for v in candidates.values()]
        return dict(
            text=text,
            answer=[b0, b1, b2],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"XtX = [[4, 6, 4], [6, 14, 9], [4, 9, 6]] (x1 and x2 are "
                f"not orthogonal here). Xty = [{xty[0]}, {xty[1]}, {xty[2]}].",
                f"Solving XtX × b = Xty (a genuine 3-variable system, not independent divisions) gives {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=4),
        )
    raise VerificationError("could not draw a collision-free medium-tier question")


def _draw_dominant_3x3(rng, *, diag_lo=4, diag_hi=9, off_hi=3):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        d0 = rng.randint(diag_lo, diag_hi)
        d1 = rng.randint(diag_lo, diag_hi)
        d2 = rng.randint(diag_lo, diag_hi)
        s01 = rng.randint(-off_hi, off_hi)
        s02 = rng.randint(-off_hi, off_hi)
        s12 = rng.randint(-off_hi, off_hi)
        if abs(s01) + abs(s02) >= d0:
            continue
        if abs(s01) + abs(s12) >= d1:
            continue
        if abs(s02) + abs(s12) >= d2:
            continue
        return [[d0, s01, s02], [s01, d1, s12], [s02, s12, d2]]
    raise VerificationError("could not draw a diagonally dominant 3x3 Gram matrix")


def _build_hard(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        s_rows = _draw_dominant_3x3(rng)
        b0 = rng.randint(-6, 6)
        b1 = rng.randint(-6, 6)
        if b1 == 0:
            continue
        b2 = rng.randint(-6, 6)
        if b2 == 0 or b2 == b1:
            continue

        xty = [sum(s_rows[i][j] * (b0, b1, b2)[j] for j in range(3)) for i in range(3)]

        # Independent route: solve XtX × b = Xty back out via sympy, a
        # genuinely different path from the forward matrix-vector product
        # used to construct Xty above.
        solved = _solve3(s_rows, xty)
        check_equivalent(tuple(solved), (b0, b1, b2), "hard-tier Gram-matrix solve mismatch")

        sub = [[s_rows[0][0], s_rows[0][1]], [s_rows[1][0], s_rows[1][1]]]
        sub_rhs = [xty[0], xty[1]]
        b0_ig, b1_ig = _solve2(sub, sub_rhs)
        ignores_one = (_round_component(b0_ig), _round_component(b1_ig), 0)

        off_diag = (
            _round_component(Fraction(xty[0], s_rows[0][0])),
            _round_component(Fraction(xty[1], s_rows[1][1])),
            _round_component(Fraction(xty[2], s_rows[2][2])),
        )

        answer = (b0, b1, b2)
        candidates = {
            "wrong-order": (b0, b2, b1),
            "ignores-one-feature": ignores_one,
            "off-diagonal-ignored": off_diag,
            "sign-flip-b1": (b0, -b1, b2),
            "sign-flip-b2": (b0, b1, -b2),
        }
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"hard-tier collision: S={s_rows}, b0={b0}, b1={b1}, b2={b2}",
            )
        except VerificationError:
            continue

        text = phrase(
            rng,
            _HARD_TEMPLATES,
            s00=s_rows[0][0],
            s01=s_rows[0][1],
            s02=s_rows[0][2],
            s11=s_rows[1][1],
            s12=s_rows[1][2],
            s22=s_rows[2][2],
            t0=xty[0],
            t1=xty[1],
            t2=xty[2],
            order=_ORDER,
        )
        answer_text = _fmt(answer)
        distractor_strs = [_fmt(v) for v in candidates.values()]
        return dict(
            text=text,
            answer=[b0, b1, b2],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"XtX = {s_rows}, Xty = {xty}.",
                f"Solving XtX × b = Xty as pure algebra (no data context) gives {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free hard-tier question")


def _build_exam(rng) -> dict:
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        n = rng.randint(4, 9)
        a = rng.randint(4, 9)
        c = rng.randint(4, 9)
        b_off = rng.randint(-3, 3)
        if abs(b_off) >= min(a, c):
            continue
        m_rows = [[a, b_off], [b_off, c]]

        b0 = rng.randint(-6, 6)
        b1 = rng.randint(-6, 6)
        if b1 == 0:
            continue
        b2 = rng.randint(-6, 6)
        if b2 == 0 or b2 == b1:
            continue
        x1bar = rng.randint(-5, 5)
        if x1bar == 0:
            continue
        x2bar = rng.randint(-5, 5)
        if x2bar == 0 or x2bar == x1bar:
            continue

        u1 = a * b1 + b_off * b2
        u2 = b_off * b1 + c * b2
        ybar = b0 + b1 * x1bar + b2 * x2bar

        # Independent check: reconstruct the FULL 3x3 system from the
        # rendered numbers (n, a, b_off, c, u1, u2, ybar) -- since the
        # intercept-decoupled row gives n*beta0' = n*ybar identically --
        # solve it the other way, then re-derive b0 from the SOLVED b1, b2
        # and the given means, never from the construction variables.
        s_full = [[n, 0, 0], [0, a, b_off], [0, b_off, c]]
        xty_full = [n * ybar, u1, u2]
        solved = _solve3(s_full, xty_full)
        check_equivalent(solved[0], Fraction(ybar), "exam-tier decoupled intercept row does not give ybar")
        check_equivalent((solved[1], solved[2]), (b1, b2), "exam-tier 2x2 sub-solve mismatch")
        b0_recovered = Fraction(ybar) - solved[1] * x1bar - solved[2] * x2bar
        check_equivalent(b0_recovered, Fraction(b0), "exam-tier recovered intercept does not match construction")

        answer = (b0, b1, b2)

        # off-diagonal-ignored: solve the 2x2 block as if diagonal, but
        # still apply the (correct) mean-combination step.
        b1_od = _round_component(Fraction(u1, a))
        b2_od = _round_component(Fraction(u2, c))
        off_diag = (
            _round_component(Fraction(ybar) - b1_od * x1bar - b2_od * x2bar),
            b1_od,
            b2_od,
        )

        # ignores-one-feature: drop x2 (b2 -> 0), solve the 1x1 subsystem
        # a*b1' = u1 for b1', then combine with only the x1 mean.
        b1_if = _round_component(Fraction(u1, a))
        b0_if = _round_component(Fraction(ybar) - b1_if * x1bar)
        ignores_one = (b0_if, b1_if, 0)

        # omits-mean-adjustment: the tier-specific misconception this
        # escalation exists to catch -- reports b0 = ybar directly, skipping
        # the derived-intercept combination step entirely.
        omits_mean = (ybar, b1, b2)

        candidates = {
            "wrong-order": (b0, b2, b1),
            "off-diagonal-ignored": off_diag,
            "ignores-one-feature": ignores_one,
            "omits-mean-adjustment": omits_mean,
            "sign-flip-b1": (b0, -b1, b2),
        }
        try:
            check_pairwise_distinct(
                answer,
                *candidates.values(),
                msg=f"exam-tier collision: n={n}, a={a}, b_off={b_off}, c={c}, b0={b0}, b1={b1}, b2={b2}",
            )
        except VerificationError:
            continue

        text = phrase(
            rng,
            _EXAM_TEMPLATES,
            n=n,
            a=a,
            b=b_off,
            c=c,
            u1=u1,
            u2=u2,
            x1bar=x1bar,
            x2bar=x2bar,
            ybar=ybar,
            order=_ORDER,
        )
        answer_text = _fmt(answer)
        distractor_strs = [_fmt(v) for v in candidates.values()]
        return dict(
            text=text,
            answer=[b0, b1, b2],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"XtX's bottom-right block is [[{a}, {b_off}], [{b_off}, {c}]]; "
                f"solving it against [{u1}, {u2}] gives b1 = {b1}, b2 = {b2}.",
                f"b0 = ybar - b1×x1bar - b2×x2bar = {ybar} - ({b1})({x1bar}) - ({b2})({x2bar}) = {b0}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free exam-tier question")


@generator(
    "p14e_multivariate_regression",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def multivariate_regression(rng, tier="easy"):
    """Fit a two-feature linear regression y = b0 + b1·x1 + b2·x2 -- find
    [b0, b1, b2], in that order, solving (X^T X) b = X^T y for the supplied
    design.

    easy: fixed n=4 orthogonal raw-data design -- three independent
    per-column divisions.
    medium: fixed n=4 non-orthogonal raw-data design -- a genuine 3-variable
    linear system.
    hard: Gram matrix (XtX, Xty) given directly, no raw data at all.
    exam: block-structured Gram matrix (x1, x2 centred) plus sample means --
    solve a 2x2 sub-system, then derive b0 as a genuinely new final step.
    """
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
