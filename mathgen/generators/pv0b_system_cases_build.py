"""Generates `pv0b_system_cases` questions: classify a 3x3 linear system as having one, no or infinitely many
solutions.

Classify a 3x3 linear system.

senior-examine-the-three-cases-for-solutions-of-systems (ACMSM110).

``pv0b_system_cases`` is an ``answer: mcq`` classification over exactly three
keys -- ``"exactly one solution"``, ``"no solution"``, ``"infinitely many
solutions"`` -- for a system of three linear equations in three variables
(or, at the easy tier, its row-reduced echelon form). The class is drawn
FIRST, uniformly over the three keys, then the system is constructed to
realize it (construct-to-target). The skill classifies; it never asks for
the solution itself (that is ``matrix-solve-system``/
``solve-simultaneous-equations``).

Every construction's classification is re-derived independently via
``sympy.Matrix.rank()`` on the actual constructed rows (never trusted from
the construction bookkeeping) before the question is returned -- see
``_rank_class``.
"""

from __future__ import annotations

from sympy import Eq, Matrix, solve, symbols

from ..core import check, generator, join_terms, latex_answer

# ============================================================================
# Answer space
# ============================================================================

KEY_UNIQUE = "exactly one solution"
KEY_NONE = "no solution"
KEY_INFINITE = "infinitely many solutions"
_CLASSES = (KEY_UNIQUE, KEY_NONE, KEY_INFINITE)

MAXABS = 12


def _bounded(*values) -> bool:
    return all(-MAXABS <= v <= MAXABS for v in values)


def _distractors_for(answer: str) -> list[str]:
    distractors = [c for c in _CLASSES if c != answer]
    check(len(distractors) == 2, "system_cases must offer exactly 2 distractors")
    check(answer not in distractors, "the answer must not also appear as a distractor")
    return distractors


def _rank_class(rows) -> str:
    """Independent-of-construction classification: rank(A) vs rank([A|b]).

    ``rows`` is a list of (a, b, c, d) integer tuples. Never trusts which
    branch the caller thinks it built -- recomputed from the final numbers.
    """
    A = Matrix([list(r[:3]) for r in rows])
    Aug = Matrix([list(r) for r in rows])
    rank_a = A.rank()
    rank_aug = Aug.rank()
    if rank_a == 3:
        return KEY_UNIQUE
    if rank_a == rank_aug:
        return KEY_INFINITE
    return KEY_NONE


def _independent(a1, b1, c1, a2, b2, c2) -> bool:
    """True unless (a2, b2, c2) is a scalar multiple of (a1, b1, c1)."""
    return not (a1 * b2 - a2 * b1 == 0 and a1 * c2 - a2 * c1 == 0 and b1 * c2 - b2 * c1 == 0)


def _draw_pair(rng, lo, hi, dlo, dhi):
    """Two independent equations (a, b, c, d), coefficients never all zero."""
    while True:
        a1, b1, c1 = rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)
        if (a1, b1, c1) == (0, 0, 0):
            continue
        a2, b2, c2 = rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)
        if (a2, b2, c2) == (0, 0, 0):
            continue
        if _independent(a1, b1, c1, a2, b2, c2):
            break
    d1, d2 = rng.randint(dlo, dhi), rng.randint(dlo, dhi)
    return (a1, b1, c1, d1), (a2, b2, c2, d2)


def _combo(eq1, eq2, p, q):
    a1, b1, c1, d1 = eq1
    a2, b2, c2, d2 = eq2
    return p * a1 + q * a2, p * b1 + q * b2, p * c1 + q * c2, p * d1 + q * d2


def _eq_text(a, b, c, d) -> str:
    lhs = join_terms([(a, "x"), (b, "y"), (c, "z")])
    check(lhs != "0", "a printed equation must not have an all-zero left side")
    return f"{lhs} = {d}"


def _stack_latex(*lines: str) -> str:
    """Typeset stacked equations in a real LaTeX environment (never a bare
    top-level ``\\\\``), matching the shipping pattern of wrapping a joined
    system in an environment (``mathgen.generators.algebra``'s
    ``simultaneous`` uses ``\\begin{cases}...\\end{cases}``; an array with a
    single left-aligned column reads the same way for three rows)."""
    body = r" \\ ".join(lines)
    return r"\begin{array}{l} " + body + r" \end{array}"


# ============================================================================
# Easy: row-reduced (echelon) augmented matrix, last row decisive.
# ============================================================================

_EASY_INTROS = (
    "The row-reduced (echelon) form of a system in x, y and z is given below. Classify its solution set.",
    "A system of three linear equations in x, y and z has been reduced to echelon "
    "form, shown below. Which of the three cases applies: exactly one solution, "
    "no solution, or infinitely many solutions?",
    "Below is the echelon form of a linear system in x, y and z. State whether "
    "the system has exactly one solution, no solution, or infinitely many "
    "solutions.",
)


def _nonzero(rng, lo, hi):
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _build_easy(rng):
    cls = rng.choice(_CLASSES)
    a1, b1, c1 = _nonzero(rng, -6, 6), rng.randint(-6, 6), rng.randint(-6, 6)
    d1 = rng.randint(-9, 9)
    b2, c2 = _nonzero(rng, -6, 6), rng.randint(-6, 6)
    d2 = rng.randint(-9, 9)
    if cls == KEY_UNIQUE:
        c3, d3 = _nonzero(rng, -6, 6), rng.randint(-9, 9)
    elif cls == KEY_NONE:
        c3, d3 = 0, _nonzero(rng, -9, 9)
    else:
        c3, d3 = 0, 0

    rows = [(a1, b1, c1, d1), (0, b2, c2, d2), (0, 0, c3, d3)]
    derived = _rank_class(rows)
    check(derived == cls, f"echelon rows {rows} classify as {derived!r}, expected {cls!r}")

    intro = rng.choice(_EASY_INTROS)
    row_strs = [f"[{a} {b} {c} | {d}]" for a, b, c, d in rows]
    text = f"{intro} " + "  ".join(f"Row {i + 1}: {s}" for i, s in enumerate(row_strs))

    matrix_body = " \\\\ ".join(" & ".join(str(v) for v in row) for row in rows)
    # Spaced column spec: KaTeX ignores the spaces, and an unspaced "ccc" reads as an
    # English letter run to tests/test_prompt_latex_mangled.py's detector.
    matrix_latex = r"\left[\begin{array}{c c c|c} " + matrix_body + r" \end{array}\right]"
    latex = r"\text{" + intro + "} " + matrix_latex

    working = [f"The bottom row is [0 0 {c3} | {d3}]; that alone decides the class."]
    answer = cls
    distractors = _distractors_for(answer)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Medium: the full system, with the dependency VISIBLE.
# ============================================================================

_MEDIUM_INTROS = (
    "Classify the solution set of the following system of equations in x, y and z.",
    "For the system of three linear equations below, decide whether it has "
    "exactly one solution, no solution, or infinitely many solutions.",
    "Consider the following system in x, y and z. Which of the three cases "
    "applies: a unique solution, no solution, or infinitely many solutions?",
)


def _draw_medium(rng):
    cls = rng.choice(_CLASSES)
    for _ in range(500):
        eq1, eq2 = _draw_pair(rng, -6, 6, -9, 9)
        template = rng.choice(("multiple", "sumdiff"))
        if template == "multiple":
            m = rng.choice((-3, -2, 2, 3))
            base = eq1 if rng.choice((True, False)) else eq2
            ba, bb, bc, bd = base
            a3, b3, c3 = m * ba, m * bb, m * bc
            consistent_d3 = m * bd
        else:
            sign = rng.choice((1, -1))
            a3, b3, c3, consistent_d3 = _combo(eq1, eq2, 1, sign)

        if cls == KEY_INFINITE:
            d3 = consistent_d3
        elif cls == KEY_NONE:
            e = rng.choice([v for v in range(-4, 5) if v != 0])
            d3 = consistent_d3 + e
        else:  # KEY_UNIQUE -- perturb ONE coefficient of the near-giveaway triple
            idx = rng.randrange(3)
            delta = rng.choice((-1, 1))
            coeffs = [a3, b3, c3]
            coeffs[idx] += delta
            a3, b3, c3 = coeffs
            d3 = rng.randint(-9, 9)

        if not _bounded(a3, b3, c3, d3):
            continue
        rows = [eq1, eq2, (a3, b3, c3, d3)]
        if any(row[:3] == (0, 0, 0) for row in rows):
            continue
        if len({row[:3] for row in rows}) < 3:
            continue
        derived = _rank_class(rows)
        if derived != cls:
            continue
        return rows, cls
    raise RuntimeError("could not construct a medium-tier system in 500 attempts")


def _build_medium(rng):
    rows, cls = _draw_medium(rng)
    eqs = [_eq_text(*row) for row in rows]
    check(len(set(eqs)) == 3, "two equations in the system are printed identically")
    intro = rng.choice(_MEDIUM_INTROS)
    text = f"{intro} {eqs[0]}; {eqs[1]}; {eqs[2]}."
    latex = r"\text{" + intro + "} " + _stack_latex(*eqs)

    working = ["Comparing the rank of the coefficient matrix to the rank of the augmented matrix decides the class."]
    answer = cls
    distractors = _distractors_for(answer)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Hard: the full system, no visible dependency (non-obvious combination).
# ============================================================================

_HARD_INTROS = (
    "A system of three linear equations in x, y and z is given below. Determine its solution set.",
    "The following system of three equations in x, y and z may have one "
    "solution, no solution, or infinitely many. Decide which.",
    "Examine the system below and classify its solution set.",
)


def _draw_hard(rng):
    cls = rng.choice(_CLASSES)
    for _ in range(500):
        eq1, eq2 = _draw_pair(rng, -4, 4, -9, 9)
        if cls == KEY_UNIQUE:
            a3, b3, c3 = rng.randint(-6, 6), rng.randint(-6, 6), rng.randint(-6, 6)
            if (a3, b3, c3) == (0, 0, 0):
                continue
            d3 = rng.randint(-12, 12)
        else:
            p = rng.choice((-3, -2, 2, 3))
            q = rng.choice((-3, -2, 2, 3))
            a3, b3, c3, consistent_d3 = _combo(eq1, eq2, p, q)
            if cls == KEY_INFINITE:
                d3 = consistent_d3
            else:
                e = rng.choice([v for v in range(-5, 6) if v != 0])
                d3 = consistent_d3 + e

        if not _bounded(a3, b3, c3, d3):
            continue
        rows = [eq1, eq2, (a3, b3, c3, d3)]
        if any(row[:3] == (0, 0, 0) for row in rows):
            continue
        if len({row[:3] for row in rows}) < 3:
            continue
        derived = _rank_class(rows)
        if derived != cls:
            continue
        return rows, cls
    raise RuntimeError("could not construct a hard-tier system in 500 attempts")


def _build_hard(rng):
    rows, cls = _draw_hard(rng)
    eqs = [_eq_text(*row) for row in rows]
    check(len(set(eqs)) == 3, "two equations in the system are printed identically")
    intro = rng.choice(_HARD_INTROS)
    text = f"{intro} {eqs[0]}; {eqs[1]}; {eqs[2]}."
    latex = r"\text{" + intro + "} " + _stack_latex(*eqs)

    working = ["Full elimination is needed here; no row is a visible multiple of another."]
    answer = cls
    distractors = _distractors_for(answer)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Exam: a real parameter k, and a stated CONDITION on k.
# ============================================================================

_EXAM_INTROS = (
    "A system of three linear equations in x, y and z contains a real parameter k.",
    "The system below, in x, y and z, depends on a real parameter k.",
    "Consider the following system in x, y and z, where k is a real parameter.",
)

_k = symbols("k")


def _condition_and_closing(kind: str, value: int | None = None) -> tuple[str, str]:
    """Condition sentence and closing instruction for the exam tier, matched
    to the condition form so the closing never contradicts it: a strict inequality genuinely ranges over infinitely many
    values ("for every value of k satisfying this condition"); a single
    stated value is not a range ("for this value of k"); "for every value of
    k" already says so, so the closing does not repeat it. Uses ``≠``,
    never Python's ``!=``."""
    if kind == "neq":
        condition = f"Here, k is a real constant and k ≠ {value}."
        closing = "Classify the system's solution set for every value of k satisfying this condition."
    elif kind == "eq":
        condition = f"Here, k = {value}."
        closing = "Classify the system's solution set for this value of k."
    else:  # "forall"
        condition = "This holds for every value of k."
        closing = "Classify the system's solution set."
    return condition, closing


def _k_expr(offset: int) -> str:
    if offset == 0:
        return "k"
    if offset > 0:
        return f"(k + {offset})"
    return f"(k - {abs(offset)})"


def _const_expr(k_coef: int, offset: int) -> str:
    if k_coef == 0:
        return str(offset)
    if k_coef == 1:
        if offset == 0:
            return "k"
        return f"k + {offset}" if offset > 0 else f"k - {abs(offset)}"
    # k_coef == -1
    if offset == 0:
        return "-k"
    return f"-k + {offset}" if offset > 0 else f"-k - {abs(offset)}"


def _sample_ks_for(kind: str, c: int | None) -> list[int]:
    """>= 5 concrete k values satisfying ``kind``'s condition."""
    if kind == "neq":
        pool = [v for v in range(-8, 9) if v != c]
        return pool[:5]
    if kind == "eq":
        return [c] * 5
    return [-4, -1, 0, 2, 5]  # "forall"


def _verify_over_ks(rows_at, ks, expected_class: str) -> None:
    """Independent-of-bookkeeping check: substitute k at several concrete admissible
    values and re-derive the class from the actual numeric matrix at each
    substitution, for ANY exam construction (k in a coefficient or in a
    constant)."""
    for kv in ks:
        rows = rows_at(kv)
        derived = _rank_class(rows)
        check(
            derived == expected_class,
            f"at k={kv} the system classifies as {derived!r}, expected {expected_class!r}: rows={rows}",
        )


def _draw_exam_unique(rng):
    """L-position (k in a coefficient), 'k != c' -- det(A) is linear in k with
    its one root at c, so k != c keeps det(A) nonzero (recipe U1). Also
    returns the LU multipliers/combination coefficients (``comboA``,
    ``comboB``) so the L/'k = c' recipes can reuse the same
    factorisation and decide none-vs-infinite at the singular value itself."""
    for _ in range(1000):
        u1 = _nonzero(rng, -4, 4)
        u2 = _nonzero(rng, -4, 4)
        p, q, r = rng.randint(-3, 3), rng.randint(-3, 3), rng.randint(-3, 3)
        l21, l31, l32 = (rng.choice((-2, -1, 1, 2)) for _ in range(3))
        c = rng.randint(-6, 6)

        a1, b1, c1 = u1, p, q
        a2 = l21 * u1
        b2 = l21 * p + u2
        c2 = l21 * q + r
        a3 = l31 * u1
        b3 = l31 * p + l32 * u2
        k_offset = l31 * q + l32 * r - c

        if a3 == 0 and b3 == 0:
            continue
        if (a1, b1, c1) == (0, 0, 0) or (a2, b2, c2) == (0, 0, 0):
            continue
        d1, d2, d3 = rng.randint(-9, 9), rng.randint(-9, 9), rng.randint(-9, 9)
        if not _bounded(a1, b1, c1, a2, b2, c2, a3, b3, k_offset, d1, d2, d3):
            continue

        A_k = Matrix([[a1, b1, c1], [a2, b2, c2], [a3, b3, _k + k_offset]])
        det_k = A_k.det()
        roots = solve(Eq(det_k, 0), _k)
        if len(roots) != 1 or roots[0] != c:
            continue

        return dict(
            a1=a1,
            b1=b1,
            c1=c1,
            d1=d1,
            a2=a2,
            b2=b2,
            c2=c2,
            d2=d2,
            a3=a3,
            b3=b3,
            k_offset=k_offset,
            d3=d3,
            c=c,
            # LU bookkeeping: at k = c, row3(A) = comboA*row1(A) +
            # comboB*row2(A) -- see the L/'k = c' recipes below.
            l21=l21,
            l31=l31,
            l32=l32,
            comboA=l31 - l32 * l21,
            comboB=l32,
        )
    raise RuntimeError("could not construct an exam-tier unique system in 1000 attempts")


def _build_exam_unique_L(rng):
    """Recipe U1: L position, 'k != c' -- unchanged."""
    draw = _draw_exam_unique(rng)
    eq1 = _eq_text(draw["a1"], draw["b1"], draw["c1"], draw["d1"])
    eq2 = _eq_text(draw["a2"], draw["b2"], draw["c2"], draw["d2"])
    z3 = _k_expr(draw["k_offset"])
    lhs3 = join_terms([(draw["a3"], "x"), (draw["b3"], "y")])
    lhs3 = f"{lhs3} + {z3}z" if lhs3 != "0" else f"{z3}z"
    eq3 = f"{lhs3} = {draw['d3']}"
    c = draw["c"]
    condition, closing = _condition_and_closing("neq", c)

    intro = rng.choice(_EXAM_INTROS)
    text = f"{intro} {eq1}; {eq2}; {eq3}. {condition} {closing}"
    latex = (
        r"\text{"
        + intro
        + "} "
        + _stack_latex(eq1, eq2, eq3)
        + r" \text{"
        + condition
        + "} "
        + r"\text{"
        + closing
        + "}"
    )

    working = [f"det(A) is linear in k with root k = {c}; the condition k ≠ {c} makes it nonzero."]
    answer = KEY_UNIQUE
    distractors = _distractors_for(answer)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
    )


def _draw_exam_dependent(rng, mode: str):
    """R position (k only ever in a constant, never a coefficient): eq3 = eq1
    + eq2 on the left, always. ``mode`` fixes which of eq1/eq2 carries k and
    whether the constants are made to agree."""
    for _ in range(500):
        eq1, eq2 = _draw_pair(rng, -5, 5, -9, 9)
        a1, b1, c1, _ = eq1
        a2, b2, c2, _ = eq2
        a3, b3, c3 = a1 + a2, b1 + b2, c1 + c2
        if not _bounded(a3, b3, c3):
            continue
        if (a3, b3, c3) == (0, 0, 0):
            continue

        r1 = rng.randint(-9, 9)
        r2 = rng.randint(-9, 9)
        if "forall" in mode:  # "infinite_forall" or "none_forall"
            k1_coef, k2_coef = 1, -1
            d3_fixed = r1 + r2
            c_value = None
            if mode == "none_forall":
                e = rng.choice([v for v in range(-4, 5) if v != 0])
                d3_fixed += e
        else:
            k1_coef, k2_coef = 1, 0
            c_value = rng.randint(-6, 6)
            d3_fixed = c_value + r1 + r2
        if not _bounded(r1, r2, d3_fixed):
            continue

        return dict(
            a1=a1,
            b1=b1,
            c1=c1,
            k1_coef=k1_coef,
            r1=r1,
            a2=a2,
            b2=b2,
            c2=c2,
            k2_coef=k2_coef,
            r2=r2,
            a3=a3,
            b3=b3,
            c3=c3,
            d3=d3_fixed,
            c_value=c_value,
        )
    raise RuntimeError(f"could not construct an exam-tier {mode!r} system in 500 attempts")


def _verify_exam_dependent(draw, mode, expected_class):
    """Independent-of-bookkeeping check: substitute k at several concrete
    values satisfying the condition and re-derive the class from the rank of
    the actual numeric matrix at each substitution."""
    if "forall" in mode:
        sample_ks = [-4, -1, 0, 2, 5]
    elif draw["c_value"] is not None and mode == "none_at_c":
        sample_ks = [v for v in range(-6, 7) if v != draw["c_value"]][:5]
    else:  # infinite_at_c
        sample_ks = [draw["c_value"]] * 5  # the single admissible value

    def rows_at(kv):
        d1 = draw["k1_coef"] * kv + draw["r1"]
        d2 = draw["k2_coef"] * kv + draw["r2"]
        return [
            (draw["a1"], draw["b1"], draw["c1"], d1),
            (draw["a2"], draw["b2"], draw["c2"], d2),
            (draw["a3"], draw["b3"], draw["c3"], draw["d3"]),
        ]

    _verify_over_ks(rows_at, sample_ks, expected_class)


def _render_exam_dependent(draw):
    """Shared renderer for every ``_draw_exam_dependent`` recipe (R position:
    k only ever in a constant, never a coefficient)."""
    eq1 = f"{join_terms([(draw['a1'], 'x'), (draw['b1'], 'y'), (draw['c1'], 'z')])} = " + _const_expr(
        draw["k1_coef"], draw["r1"]
    )
    eq2 = f"{join_terms([(draw['a2'], 'x'), (draw['b2'], 'y'), (draw['c2'], 'z')])} = " + _const_expr(
        draw["k2_coef"], draw["r2"]
    )
    eq3 = _eq_text(draw["a3"], draw["b3"], draw["c3"], draw["d3"])
    return eq1, eq2, eq3


def _finish_exam_question(eq1, eq2, eq3, condition, closing, intro, working, answer):
    text = f"{intro} {eq1}; {eq2}; {eq3}. {condition} {closing}"
    latex = (
        r"\text{"
        + intro
        + "} "
        + _stack_latex(eq1, eq2, eq3)
        + r" \text{"
        + condition
        + "} "
        + r"\text{"
        + closing
        + "}"
    )
    distractors = _distractors_for(answer)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
    )


def _build_exam_none_R(rng):
    """Recipe N1 (unchanged): R position, 'k != c' -- no solution."""
    draw = _draw_exam_dependent(rng, "none_at_c")
    _verify_exam_dependent(draw, "none_at_c", KEY_NONE)
    eq1, eq2, eq3 = _render_exam_dependent(draw)
    condition, closing = _condition_and_closing("neq", draw["c_value"])
    intro = rng.choice(_EXAM_INTROS)
    working = [
        "Eliminating x and y from equations 1 and 2 forces the left side of "
        f"equation 3, but the constants disagree unless k = {draw['c_value']}, which "
        "is excluded."
    ]
    return _finish_exam_question(eq1, eq2, eq3, condition, closing, intro, working, KEY_NONE)


def _build_exam_infinite_at_c(rng):
    """Recipe I1 (unchanged): R position, 'k = c' -- infinitely many."""
    draw = _draw_exam_dependent(rng, "infinite_at_c")
    _verify_exam_dependent(draw, "infinite_at_c", KEY_INFINITE)
    eq1, eq2, eq3 = _render_exam_dependent(draw)
    condition, closing = _condition_and_closing("eq", draw["c_value"])
    intro = rng.choice(_EXAM_INTROS)
    working = [
        "Equation 3's left side is always equation 1's plus equation 2's, and at this "
        "value of k the constants also match, so the third equation is redundant."
    ]
    return _finish_exam_question(eq1, eq2, eq3, condition, closing, intro, working, KEY_INFINITE)


def _build_exam_infinite_forall(rng):
    """Recipe I2 (unchanged): R position, 'for every k' -- infinitely many."""
    draw = _draw_exam_dependent(rng, "infinite_forall")
    _verify_exam_dependent(draw, "infinite_forall", KEY_INFINITE)
    eq1, eq2, eq3 = _render_exam_dependent(draw)
    condition, closing = _condition_and_closing("forall")
    intro = rng.choice(_EXAM_INTROS)
    working = [
        "Equation 3's left side is always equation 1's plus equation 2's, and the "
        "constants match for every value of k, so the third equation is always redundant."
    ]
    return _finish_exam_question(eq1, eq2, eq3, condition, closing, intro, working, KEY_INFINITE)


def _build_exam_none_forall(rng):
    """Recipe N4: R position, 'for every k' -- no solution.
    Same dependency as I2, but the constants are offset by a fixed nonzero
    amount that no value of k can undo."""
    draw = _draw_exam_dependent(rng, "none_forall")
    _verify_exam_dependent(draw, "none_forall", KEY_NONE)
    eq1, eq2, eq3 = _render_exam_dependent(draw)
    condition, closing = _condition_and_closing("forall")
    intro = rng.choice(_EXAM_INTROS)
    working = [
        "Equation 3's left side is always equation 1's plus equation 2's, but the "
        "constants are off by a fixed amount that no value of k can close, so the "
        "system is inconsistent for every value of k."
    ]
    return _finish_exam_question(eq1, eq2, eq3, condition, closing, intro, working, KEY_NONE)


# R position, nonsingular integer coefficient matrix -- unique regardless of k, for every condition
# form (recipes U2/U3/U4).


def _draw_nonsingular_A(rng):
    """A 3x3 integer coefficient matrix with rank 3, |entries| <= MAXABS.
    k never appears in it -- only ever on the right-hand side -- so the class
    is unique for literally any k, whatever the stated condition."""
    for _ in range(500):
        rows = [tuple(rng.randint(-4, 4) for _ in range(3)) for _ in range(3)]
        if any(r == (0, 0, 0) for r in rows):
            continue
        if not _bounded(*[v for row in rows for v in row]):
            continue
        if Matrix([list(r) for r in rows]).rank() == 3:
            return rows
    raise RuntimeError("could not construct a nonsingular 3x3 integer matrix in 500 attempts")


def _build_exam_unique_R(rng, cond_kind: str):
    """Recipes U2 ('k != c'), U3 ('k = c'), U4 ('for every k'): the
    coefficient matrix is nonsingular and k-free, so k on the right never
    changes the class -- a real exam trap for a learner who has memorised
    'k on the right, condition holds' as meaning something other than
    unique."""
    (a1, b1, c1), (a2, b2, c2), (a3, b3, c3) = _draw_nonsingular_A(rng)
    r1 = rng.randint(-9, 9)
    r2 = rng.randint(-9, 9)
    d3 = rng.randint(-9, 9)
    if cond_kind == "forall":
        k1_coef, k2_coef = 1, -1
        c_value = None
    else:
        k1_coef, k2_coef = 1, 0
        c_value = rng.randint(-6, 6)

    def rows_at(kv):
        d1 = k1_coef * kv + r1
        d2 = k2_coef * kv + r2
        return [(a1, b1, c1, d1), (a2, b2, c2, d2), (a3, b3, c3, d3)]

    ks = _sample_ks_for(cond_kind, c_value)
    _verify_over_ks(rows_at, ks, KEY_UNIQUE)

    condition, closing = _condition_and_closing(cond_kind, c_value)
    eq1 = f"{join_terms([(a1, 'x'), (b1, 'y'), (c1, 'z')])} = " + _const_expr(k1_coef, r1)
    eq2 = f"{join_terms([(a2, 'x'), (b2, 'y'), (c2, 'z')])} = " + _const_expr(k2_coef, r2)
    eq3 = _eq_text(a3, b3, c3, d3)
    intro = rng.choice(_EXAM_INTROS)
    working = [
        "The coefficient matrix does not involve k at all and has nonzero "
        "determinant, so the system has exactly one solution for every "
        "value of k satisfying the stated condition."
    ]
    return _finish_exam_question(eq1, eq2, eq3, condition, closing, intro, working, KEY_UNIQUE)


# L position (k in eq3's z-coefficient), a dependent eq1/eq2 pair whose OWN consistency decides the
# class -- k's coefficient sits elsewhere and never affects the rank (recipes N2/I3).


def _draw_exam_dup(rng, want_none: bool):
    for _ in range(500):
        a1 = rng.randint(-4, 4)
        b1 = rng.randint(-4, 4)
        c1 = rng.randint(-4, 4)
        if (a1, b1) == (0, 0):  # exclude the z-only degenerate direction
            continue
        d1 = rng.randint(-3, 3)
        m = rng.choice((-3, -2, 2, 3))
        a2, b2, c2 = m * a1, m * b1, m * c1
        if not _bounded(a2, b2, c2):
            continue

        if want_none:
            # eq1, eq2 already disagree as a 2-equation subsystem -- no
            # solution regardless of anything eq3 says (a 2-equation
            # contradiction can never be rescued by a third equation).
            e = rng.choice([v for v in range(-4, 5) if v != 0])
            d2 = m * d1 + e
            a3 = rng.randint(-4, 4)
            b3 = rng.randint(-4, 4)
            c = rng.randint(-6, 6)
            offset = rng.randint(-6, 6)
        else:
            # eq1, eq2 are the SAME plane (consistent); eq3's (x, y)
            # coefficients are proportional to eq1's, but its z-coefficient
            # is k-dependent and matches the proportion ONLY at k = c, so
            # for every k != c, eq3 is a genuinely independent second plane
            # -- and two independent planes in 3D always share a line, so
            # the system is infinite for every admissible k, regardless of
            # d3 (a two-plane intersection is never inconsistent).
            d2 = m * d1
            n = rng.choice([v for v in (-2, -1, 2, 3) if v != m])
            a3, b3 = n * a1, n * b1
            if not _bounded(a3, b3):
                continue
            c = rng.randint(-6, 6)
            offset = n * c1 - c

        d3 = rng.randint(-9, 9)
        if not _bounded(offset, d3, c):
            continue

        return dict(a1=a1, b1=b1, c1=c1, d1=d1, a2=a2, b2=b2, c2=c2, d2=d2, a3=a3, b3=b3, offset=offset, d3=d3, c=c)
    raise RuntimeError("could not construct an exam-tier duplicate-plane system in 500 attempts")


def _render_exam_dup(draw):
    eq1 = _eq_text(draw["a1"], draw["b1"], draw["c1"], draw["d1"])
    eq2 = _eq_text(draw["a2"], draw["b2"], draw["c2"], draw["d2"])
    z3 = _k_expr(draw["offset"])
    lhs3 = join_terms([(draw["a3"], "x"), (draw["b3"], "y")])
    lhs3 = f"{lhs3} + {z3}z" if lhs3 != "0" else f"{z3}z"
    eq3 = f"{lhs3} = {draw['d3']}"
    return eq1, eq2, eq3


def _build_exam_dup_none(rng):
    """Recipe N2: L position, 'k != c' -- no solution.
    Equations 1 and 2 already contradict each other; equation 3's
    k-coefficient is irrelevant to the class (the singularity is elsewhere)."""
    draw = _draw_exam_dup(rng, want_none=True)

    def rows_at(kv):
        return [
            (draw["a1"], draw["b1"], draw["c1"], draw["d1"]),
            (draw["a2"], draw["b2"], draw["c2"], draw["d2"]),
            (draw["a3"], draw["b3"], kv + draw["offset"], draw["d3"]),
        ]

    c = draw["c"]
    ks = _sample_ks_for("neq", c)
    _verify_over_ks(rows_at, ks, KEY_NONE)

    eq1, eq2, eq3 = _render_exam_dup(draw)
    condition, closing = _condition_and_closing("neq", c)
    intro = rng.choice(_EXAM_INTROS)
    working = [
        "Equations 1 and 2 alone already disagree (their left sides are "
        "proportional but their constants are not), so the whole system has "
        "no solution whatever equation 3's k-coefficient is."
    ]
    return _finish_exam_question(eq1, eq2, eq3, condition, closing, intro, working, KEY_NONE)


def _build_exam_dup_infinite(rng):
    """Recipe I3: L position, 'k != c' -- infinitely many.
    Equations 1 and 2 are the same plane; equation 3 is a genuinely
    independent second plane for every admissible k (it only lines up with
    equation 1 at the excluded value k = c), and two independent planes in
    3D always meet in a line."""
    draw = _draw_exam_dup(rng, want_none=False)

    def rows_at(kv):
        return [
            (draw["a1"], draw["b1"], draw["c1"], draw["d1"]),
            (draw["a2"], draw["b2"], draw["c2"], draw["d2"]),
            (draw["a3"], draw["b3"], kv + draw["offset"], draw["d3"]),
        ]

    c = draw["c"]
    ks = _sample_ks_for("neq", c)
    _verify_over_ks(rows_at, ks, KEY_INFINITE)

    eq1, eq2, eq3 = _render_exam_dup(draw)
    condition, closing = _condition_and_closing("neq", c)
    intro = rng.choice(_EXAM_INTROS)
    working = [
        "Equations 1 and 2 describe the same plane; equation 3 is only "
        f"parallel to that plane at the excluded value k = {c}, so for every "
        "admissible k it is a genuinely independent second plane, and two "
        "planes in 3D always share a line."
    ]
    return _finish_exam_question(eq1, eq2, eq3, condition, closing, intro, working, KEY_INFINITE)


# L position, 'k = c' -- the singular value itself. Reuses the U1 LU factorisation; whether the
# third row's implied constant matches the combination of the first two decides none vs infinite
# (recipes N3/I4).


def _build_exam_lu_at_c(rng, want_none: bool):
    for _ in range(200):
        draw = _draw_exam_unique(rng)
        comboA, comboB, c = draw["comboA"], draw["comboB"], draw["c"]
        d1 = rng.randint(-3, 3)
        d2 = rng.randint(-3, 3)
        d3_consistent = comboA * d1 + comboB * d2
        if want_none:
            e = rng.choice([v for v in range(-4, 5) if v != 0])
            d3 = d3_consistent + e
        else:
            d3 = d3_consistent
        if not _bounded(d1, d2, d3):
            continue

        def rows_at(kv, draw=draw, d1=d1, d2=d2, d3=d3):
            return [
                (draw["a1"], draw["b1"], draw["c1"], d1),
                (draw["a2"], draw["b2"], draw["c2"], d2),
                (draw["a3"], draw["b3"], kv + draw["k_offset"], d3),
            ]

        expected = KEY_NONE if want_none else KEY_INFINITE
        ks = _sample_ks_for("eq", c)
        _verify_over_ks(rows_at, ks, expected)

        eq1 = _eq_text(draw["a1"], draw["b1"], draw["c1"], d1)
        eq2 = _eq_text(draw["a2"], draw["b2"], draw["c2"], d2)
        z3 = _k_expr(draw["k_offset"])
        lhs3 = join_terms([(draw["a3"], "x"), (draw["b3"], "y")])
        lhs3 = f"{lhs3} + {z3}z" if lhs3 != "0" else f"{z3}z"
        eq3 = f"{lhs3} = {d3}"
        condition, closing = _condition_and_closing("eq", c)
        intro = rng.choice(_EXAM_INTROS)
        if want_none:
            working = [
                f"At k = {c} the third row becomes a fixed combination of the "
                "first two, but the constants disagree with that same "
                "combination, so the system is inconsistent -- no solution."
            ]
        else:
            working = [
                f"At k = {c} the third row becomes a fixed combination of the "
                "first two, and the constants agree with that same "
                "combination, so the third equation is redundant -- "
                "infinitely many solutions."
            ]
        return _finish_exam_question(eq1, eq2, eq3, condition, closing, intro, working, expected)
    raise RuntimeError("could not construct an exam-tier L/'k = c' system in 200 attempts")


def _build_exam(rng):
    cls = rng.choice(_CLASSES)
    if cls == KEY_UNIQUE:
        recipe = rng.choice(("L_neq", "R_neq", "R_eq", "R_forall"))
        if recipe == "L_neq":
            return _build_exam_unique_L(rng)
        return _build_exam_unique_R(rng, {"R_neq": "neq", "R_eq": "eq", "R_forall": "forall"}[recipe])
    if cls == KEY_NONE:
        recipe = rng.choice(("R_neq", "L_neq", "L_eq", "R_forall"))
        if recipe == "R_neq":
            return _build_exam_none_R(rng)
        if recipe == "L_neq":
            return _build_exam_dup_none(rng)
        if recipe == "L_eq":
            return _build_exam_lu_at_c(rng, want_none=True)
        return _build_exam_none_forall(rng)
    # KEY_INFINITE
    recipe = rng.choice(("R_eq", "R_forall", "L_neq", "L_eq"))
    if recipe == "R_eq":
        return _build_exam_infinite_at_c(rng)
    if recipe == "R_forall":
        return _build_exam_infinite_forall(rng)
    if recipe == "L_neq":
        return _build_exam_dup_infinite(rng)
    return _build_exam_lu_at_c(rng, want_none=False)


# ============================================================================
# Entry point
# ============================================================================


@generator(
    "pv0b_system_cases",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def pv0b_system_cases(rng, tier="easy"):
    """Classify a system of three linear equations in three variables (or,
    at the easy tier, its row-reduced echelon form) as having exactly one
    solution, no solution, or infinitely many solutions. The class is drawn
    first, uniformly, then the system is constructed to realize it. Never
    asks for the solution itself."""
    if tier not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"tier must be one of easy/medium/hard/exam; got {tier!r}")
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    return _build_exam(rng)
