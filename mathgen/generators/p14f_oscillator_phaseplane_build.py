"""Generates `ucv1-linear-oscillator-classification` and `ucv1-phase-plane-linear-classification` questions:
classifying linear oscillators and linear phase-plane systems.

`ucv1-linear-oscillator-classification` and `ucv1-phase-plane-linear-classification`.

Two `answer: mcq` classification skills, both linear-ODE-adjacent:

- `p14f_linear_oscillator_classification`: classify a second-order linear
  oscillator mx'' + bx' + kx = 0 as underdamped / critically damped /
  overdamped, from the discriminant b**2 - 4*m*k.
- `p14f_phase_plane_linear_classification`: classify the origin equilibrium
  of a 2x2 linear system x' = A x into one of six honest planar classes
  (stable/unstable node, saddle, stable/unstable spiral, centre), from A's
  trace and determinant.

Both are construct-to-target: the target class is drawn FIRST and the
coefficients/matrix are built to realise it exactly, then re-verified via a
fresh sympy computation independent of the construction path.
"""

from __future__ import annotations

import sympy
from sympy import Matrix, Rational

from mathgen.core import (
    check,
    generator,
    latex_answer,
)

# ===========================================================================
# Skill A: p14f_linear_oscillator_classification
# ===========================================================================

_OSC_CLASSES = ["underdamped", "critically damped", "overdamped"]

# Per-pair discriminant-reading error, documented (not constructed -- the two
# non-target classes are always the two distractors by construction of a
# 3-way answer space): underdamped<->overdamped is a sign-flip on the whole
# discriminant (4mk - b^2 instead of b^2 - 4mk); either real class vs
# critically damped is treating a nonzero discriminant as exactly zero (or
# vice versa), missing the exact equality b^2 = 4mk that defines the boundary
# -- see the skill YAML's `misconceptions` block for the exact wording.


def _osc_classify(m, b, k) -> str:
    """Classify from a fresh sympy computation of the discriminant, never
    reusing the Python-int construction path directly (Rationals so an
    equality compare is exact)."""
    D = Rational(b) ** 2 - 4 * Rational(m) * Rational(k)
    if D < 0:
        return "underdamped"
    if D == 0:
        return "critically damped"
    return "overdamped"


def _osc_build_easy(rng):
    target = rng.choice(_OSC_CLASSES)
    m = 1
    if target == "critically damped":
        q = rng.randint(1, 5)
        k, b = q * q, 2 * q
    elif target == "underdamped":
        k = rng.randint(2, 9)
        # b in [0, floor(sqrt(4k)) - 1] guarantees b^2 < 4k (b=0 allowed).
        b_max = int((4 * k) ** 0.5)
        while b_max * b_max >= 4 * k:
            b_max -= 1
        b = rng.randint(0, max(b_max, 0))
    else:  # overdamped
        k = rng.randint(1, 6)
        b_min = int((4 * k) ** 0.5) + 1
        while b_min * b_min <= 4 * k:
            b_min += 1
        b = rng.randint(b_min, b_min + 6)
    return m, b, k, target


def _osc_build_medium(rng):
    target = rng.choice(_OSC_CLASSES)
    m = rng.randint(2, 6)
    if target == "critically damped":
        q = rng.randint(1, 4)
        k, b = m * q * q, 2 * m * q
    elif target == "underdamped":
        k = rng.randint(2, 9)
        threshold_sq = 4 * m * k
        b_max = int(threshold_sq**0.5)
        while b_max * b_max >= threshold_sq:
            b_max -= 1
        b = rng.randint(0, max(b_max, 0))
    else:
        k = rng.randint(1, 6)
        threshold_sq = 4 * m * k
        b_min = int(threshold_sq**0.5) + 1
        while b_min * b_min <= threshold_sq:
            b_min += 1
        b = rng.randint(b_min, b_min + 6)
    return m, b, k, target


def _osc_build_balanced(rng):
    """The shared 3-way uniform target draw used by both hard and exam:
    draw the target class first, then construct m, b, k to realise it
    exactly. b = 0 is excluded (unlike easy/medium, where it is the
    undamped limiting case of underdamped): a zero damping term renders as
    an ugly literal "0x'"/"0 g/s" once the equation is rearranged or its
    coefficients are given in non-SI units, which is a wording defect, not
    a mathematical one. This is done by drawing b from [1, b_max] directly
    for the underdamped branch -- never by resampling the whole
    (target, m, k, b) tuple on a b == 0 draw, which would silently bias the
    3-way class balance away from underdamped (measured: resampling the
    whole tuple pushed 'overdamped' to 40% of 400 hard-tier draws, a real
    defect this direct construction fixes instead of masking)."""
    target = rng.choice(_OSC_CLASSES)
    m = rng.randint(2, 6)
    if target == "critically damped":
        q = rng.randint(1, 4)
        k, b = m * q * q, 2 * m * q
    elif target == "underdamped":
        k = rng.randint(2, 9)
        threshold_sq = 4 * m * k
        b_max = int(threshold_sq**0.5)
        while b_max * b_max >= threshold_sq:
            b_max -= 1
        b_max = max(b_max, 1)
        b = rng.randint(1, b_max)
    else:
        k = rng.randint(1, 6)
        threshold_sq = 4 * m * k
        b_min = int(threshold_sq**0.5) + 1
        while b_min * b_min <= threshold_sq:
            b_min += 1
        b = rng.randint(b_min, b_min + 6)
    return m, b, k, target


def _osc_build_hard(rng):
    """Same 3-way uniform target draw as medium (via `_osc_build_balanced`),
    but rendered in a rearranged/non-standard form so the student must
    recognise standard form before reading off m, b, k (the genuine kind
    change over medium)."""
    m, b, k, target = _osc_build_balanced(rng)
    variant = rng.choice(["moved_rhs", "shuffled_order", "moved_k_term"])
    return m, b, k, target, variant


_OSC_EXAM_TEMPLATES = (
    "An oscillator has mass m = {m_g} g, damping constant b = {b_gs} g/s "
    "and stiffness k = {k} N/m. Convert to SI units (kg, kg/s) and "
    "classify its damping regime.",
    "A mass-spring-damper system has m = {m_g} g, b = {b_gs} g/s and "
    "k = {k} N/m. Working in SI units throughout, classify the damping "
    "regime of the resulting oscillator.",
    "An oscillator's mass is {m_g} g, its damping constant is {b_gs} g/s "
    "and its stiffness is {k} N/m. Convert the mass and damping constant "
    "to SI units first, then classify the damping regime.",
)


def _osc_render_standard(m, b, k) -> str:
    m_term = "x''" if m == 1 else f"{m}x''"
    b_term = " + x'" if b == 1 else (f" + {b}x'" if b else "")
    k_term = " + x = 0" if k == 1 else f" + {k}x = 0"
    return f"{m_term}{b_term}{k_term}"


def _osc_render_hard(m, b, k, variant) -> str:
    m_term = "x''" if m == 1 else f"{m}x''"
    b_term = "x'" if b == 1 else f"{b}x'"
    k_term = "x" if k == 1 else f"{k}x"
    if variant == "moved_rhs":
        return f"{m_term} = -{b_term} - {k_term}"
    if variant == "shuffled_order":
        return f"{k_term} + {m_term} + {b_term} = 0"
    # moved_k_term: the k*x term is moved to the other side, e.g.
    # "mx'' + bx' = -kx" -- a genuine rearrangement step (unlike a
    # parenthesised-but-unmoved grouping, which needs no algebra at all).
    return f"{m_term} + {b_term} = -{k_term}"


@generator(
    "p14f_linear_oscillator_classification",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p14f_linear_oscillator_classification(rng, tier="easy"):
    if tier == "easy":
        m, b, k, target = _osc_build_easy(rng)
        recomputed = _osc_classify(m, b, k)
        check(recomputed == target, "easy oscillator classification mismatch")
        eq_text = _osc_render_standard(m, b, k)
        text = f"An oscillator obeys {eq_text}. Classify its damping regime."
        answer_text = target
        working = [
            f"Discriminant D = b^2 - 4mk = {b}^2 - 4({m})({k}) = {b * b - 4 * m * k}.",
            f"D {'< 0' if b * b - 4 * m * k < 0 else ('= 0' if b * b - 4 * m * k == 0 else '> 0')} -> {target}.",
        ]
    elif tier == "medium":
        m, b, k, target = _osc_build_medium(rng)
        recomputed = _osc_classify(m, b, k)
        check(recomputed == target, "medium oscillator classification mismatch")
        eq_text = _osc_render_standard(m, b, k)
        text = f"An oscillator obeys {eq_text}. Classify its damping regime."
        answer_text = target
        working = [
            f"Discriminant D = b^2 - 4mk = {b}^2 - 4({m})({k}) = {b * b - 4 * m * k}.",
            f"D {'< 0' if b * b - 4 * m * k < 0 else ('= 0' if b * b - 4 * m * k == 0 else '> 0')} -> {target}.",
        ]
    elif tier == "hard":
        m, b, k, target, variant = _osc_build_hard(rng)
        recomputed = _osc_classify(m, b, k)
        check(recomputed == target, "hard oscillator classification mismatch")
        eq_text = _osc_render_hard(m, b, k, variant)
        text = (
            f"An oscillator obeys {eq_text} (rearrange to standard form "
            f"mx'' + bx' + kx = 0 first). Classify its damping regime."
        )
        answer_text = target
        working = [
            f"Standard form: {_osc_render_standard(m, b, k)}.",
            f"Discriminant D = b^2 - 4mk = {b}^2 - 4({m})({k}) = {b * b - 4 * m * k}.",
            f"D {'< 0' if b * b - 4 * m * k < 0 else ('= 0' if b * b - 4 * m * k == 0 else '> 0')} -> {target}.",
        ]
    else:  # exam
        # Replaced with the same 3-way uniform target draw as hard (`_osc_build_balanced`,
        # critically damped genuinely drawn and reachable), but the coefficients are given in NON-SI
        # units -- mass in grams, damping in grams/second -- so the student must convert to kg and
        # kg/s before applying the SAME b^2 vs 4mk test every other tier uses. That conversion is
        # the genuine reasoning step hard's rearrangement does not have; k is left in SI (N/m) so
        # only two conversions are needed, not three.
        m, b, k, target = _osc_build_balanced(rng)
        recomputed = _osc_classify(m, b, k)
        check(recomputed == target, "exam oscillator classification mismatch")
        m_g = m * 1000
        b_gs = b * 1000
        template = rng.choice(_OSC_EXAM_TEMPLATES)
        text = template.format(m_g=m_g, b_gs=b_gs, k=k)
        answer_text = target
        D = b * b - 4 * m * k
        working = [
            f"Convert to SI: m = {m_g} g = {m} kg; b = {b_gs} g/s = {b} kg/s.",
            f"Discriminant D = b^2 - 4mk = {b}^2 - 4({m})({k}) = {D}.",
            f"D {'< 0' if D < 0 else ('= 0' if D == 0 else '> 0')} -> {target}.",
        ]

    # Independent check: fresh sympy Rational computation of the discriminant
    # sign, never reusing the Python-int comparison used above to construct
    # the draw.
    m_check = m
    b_check = b
    k_check = k
    D_sym = sympy.Integer(b_check) ** 2 - 4 * sympy.Integer(m_check) * sympy.Integer(k_check)
    if D_sym < 0:
        sym_class = "underdamped"
    elif D_sym == 0:
        sym_class = "critically damped"
    else:
        sym_class = "overdamped"
    check(sym_class == answer_text, "sympy discriminant-sign re-derivation mismatch")

    distractors = [c for c in _OSC_CLASSES if c != answer_text]
    check(len(distractors) == 2, "oscillator answer space must offer exactly 2 distractors")
    for d in distractors:
        check(d != answer_text, "oscillator distractor collides with the answer")

    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


# ===========================================================================
# Skill B: p14f_phase_plane_linear_classification
# ===========================================================================

_PP_CLASSES = ["stable node", "unstable node", "saddle", "stable spiral", "unstable spiral", "centre"]

_PP_STABILITY_FLIP = {
    "stable node": "unstable node",
    "unstable node": "stable node",
    "stable spiral": "unstable spiral",
    "unstable spiral": "stable spiral",
}
_PP_COMPLEX_CONFUSION = {
    "stable node": "stable spiral",
    "unstable node": "unstable spiral",
    "stable spiral": "stable node",
    "unstable spiral": "unstable node",
}


def _pp_classify_from_TD(T, D) -> str:
    """Classify from trace T and determinant D. Callers must exclude D == 0 and (T**2 == 4*D and T != 0) by
    construction before calling this."""
    check(D != 0, "phase-plane degenerate: zero eigenvalue")
    if D < 0:
        return "saddle"
    disc = T * T - 4 * D
    check(not (disc == 0 and T != 0), "phase-plane degenerate: repeated real eigenvalue")
    if disc > 0:
        return "stable node" if T < 0 else "unstable node"
    if T == 0:
        return "centre"
    return "stable spiral" if T < 0 else "unstable spiral"


def _pp_matrix_from_eigs(rng, l1, l2):
    """Integer A with eigenvalues l1, l2, via A = P diag(l1,l2) P^-1 for a
    unimodular integer P (det(P) = 1), so P^-1 is also integer."""
    P = rng.choice(
        [Matrix([[1, 1], [0, 1]]), Matrix([[1, 0], [1, 1]]), Matrix([[2, 1], [1, 1]]), Matrix([[1, 2], [1, 1]])]
    )
    D = Matrix([[l1, 0], [0, l2]])
    A = P * D * P.inv()
    A = A.applyfunc(lambda v: sympy.nsimplify(v))
    check(all(v.is_Integer for v in A), "unimodular construction did not stay integer")
    return [[int(A[0, 0]), int(A[0, 1])], [int(A[1, 0]), int(A[1, 1])]]


def _pp_matrix_diagonal(l1, l2):
    return [[l1, 0], [0, l2]]


def _pp_matrix_companion(T, D):
    return [[0, 1], [-D, T]]


def _pp_TD(A) -> tuple[int, int]:
    return A[0][0] + A[1][1], A[0][0] * A[1][1] - A[0][1] * A[1][0]


def _pp_render(A) -> str:
    a, b = A[0]
    c, d = A[1]

    def term(coef, var):
        if coef == 0:
            return None
        sign = "+" if coef > 0 else "-"
        mag = abs(coef)
        piece = var if mag == 1 else f"{mag}{var}"
        return sign, piece

    def line(coef1, var1, coef2, var2):
        parts = [term(coef1, var1), term(coef2, var2)]
        parts = [p for p in parts if p is not None]
        if not parts:
            return "0"
        out = ""
        for i, (sign, piece) in enumerate(parts):
            if i == 0:
                out += piece if sign == "+" else f"-{piece}"
            else:
                out += f" {sign} {piece}"
        return out

    return f"x' = {line(a, 'x', b, 'y')}, y' = {line(c, 'x', d, 'y')}"


def _pp_word_terms(coef_x: int, coef_y: int) -> str:
    """Sign-aware prose for one row of the exam word context: '3 times x plus
    5 times y', '3 times x minus 5 times y', or a single term when the other
    coefficient is zero -- never a literal '0 times x' (a giveaway wording
    defect, and the companion-matrix construction always sets the x-row's
    x-coefficient to 0 for a complex-eigenvalue draw)."""

    def piece(mag, var):
        return var if mag == 1 else f"{mag} times {var}"

    parts = []
    for coef, var in ((coef_x, "x"), (coef_y, "y")):
        if coef == 0:
            continue
        parts.append((abs(coef), var, coef < 0))
    if not parts:
        return "always zero"
    mag0, var0, neg0 = parts[0]
    out = piece(mag0, var0) if not neg0 else f"minus {piece(mag0, var0)}"
    for mag, var, neg in parts[1:]:
        out += f" {'minus' if neg else 'plus'} {piece(mag, var)}"
    return out


def _pp_build_real(rng, node_or_saddle_target=None, diagonal=False):
    """Draw distinct nonzero integer eigenvalues realising a node/saddle
    target, and build A (diagonal on easy, unimodular-conjugated otherwise)."""
    if node_or_saddle_target is None:
        target = rng.choice(["stable node", "unstable node", "saddle"])
    else:
        target = node_or_saddle_target
    if target == "saddle":
        l1 = rng.choice([v for v in range(-6, 0) if v != 0])
        l2 = rng.choice([v for v in range(1, 7)])
    else:
        sign = -1 if target == "stable node" else 1
        vals = [v for v in range(1, 7)]
        a, b = rng.sample(vals, 2)
        l1, l2 = sign * a, sign * b
    check(l1 != l2, "node/saddle construction collapsed to a repeated eigenvalue")
    A = _pp_matrix_diagonal(l1, l2) if diagonal else _pp_matrix_from_eigs(rng, l1, l2)
    return A, target


def _pp_build_complex(rng, target=None):
    if target is None:
        target = rng.choice(["stable spiral", "unstable spiral", "centre"])
    if target == "centre":
        T = 0
        D = rng.randint(1, 12)
    else:
        T = rng.choice([v for v in range(1, 6)]) * (-1 if target == "stable spiral" else 1)
        # Need T^2 < 4D -> D > T^2/4.
        D_min = T * T // 4 + 1
        D = rng.randint(D_min, D_min + 8)
    check(T * T < 4 * D, "complex construction failed T^2 < 4D")
    A = _pp_matrix_companion(T, D)
    return A, target


def _pp_distractors(rng, target):
    """Exactly 3 distractors: stability-flip sibling, real/complex-confusion
    sibling, then fill from remaining classes -- skipping slots that don't
    apply (saddle has neither; centre has neither dict entry either, but gets
    a dedicated policy below instead of falling through to the generic fill).

    `centre` is the one class the two dicts above have no entry for (it has
    no stability distinction to flip, and it IS the T == 0 boundary the
    real/complex confusion sits on) -- naively falling through to "remaining
    classes in `_PP_CLASSES` order" would offer {stable node, unstable node,
    saddle} every time and never a spiral, even though "the student misjudges
    the T == 0 boundary and picks a spiral" is exactly the confusion a real
    learner makes there. So `centre` gets both spirals first instead."""
    picked = []
    remaining = [c for c in _PP_CLASSES if c != target]
    if target == "centre":
        picked.extend(["stable spiral", "unstable spiral"])
    if target in _PP_STABILITY_FLIP:
        picked.append(_PP_STABILITY_FLIP[target])
    if target in _PP_COMPLEX_CONFUSION:
        picked.append(_PP_COMPLEX_CONFUSION[target])
    for c in remaining:
        if len(picked) >= 3:
            break
        if c not in picked:
            picked.append(c)
    picked = picked[:3]
    check(len(picked) == 3, "phase-plane distractor construction did not reach 3")
    check(target not in picked, "phase-plane distractor collides with the answer")
    return picked


@generator(
    "p14f_phase_plane_linear_classification",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p14f_phase_plane_linear_classification(rng, tier="easy"):
    if tier == "easy":
        A, target = _pp_build_real(rng, diagonal=True)
        text = f"For the linear system {_pp_render(A)}, classify the equilibrium at the origin."
    elif tier == "medium":
        A, target = _pp_build_real(rng, diagonal=False)
        text = f"For the linear system {_pp_render(A)}, classify the equilibrium at the origin."
    elif tier == "hard":
        target_kind = rng.choice(["real", "complex"])
        if target_kind == "real":
            A, target = _pp_build_real(rng, diagonal=False)
        else:
            A, target = _pp_build_complex(rng)
        text = f"For the linear system {_pp_render(A)}, classify the equilibrium at the origin."
    else:  # exam
        target_kind = rng.choice(["real", "complex"])
        if target_kind == "real":
            A, target = _pp_build_real(rng, diagonal=False)
        else:
            A, target = _pp_build_complex(rng)
        a, b = A[0]
        c, d = A[1]
        text = (
            f"A coupled system is described by: the rate of change of x is "
            f"{_pp_word_terms(a, b)}; the rate of change of y is "
            f"{_pp_word_terms(c, d)}. Identify the system's matrix and "
            f"classify the equilibrium at the origin."
        )

    T, D = _pp_TD(A)
    check(D != 0, "phase-plane draw is degenerate (D == 0)")
    check(not (T * T == 4 * D and T != 0), "phase-plane draw is a degenerate repeated real eigenvalue")
    recomputed = _pp_classify_from_TD(T, D)
    check(recomputed == target, "phase-plane classification mismatch")

    # Independent check: fresh sympy Matrix(A).eigenvals() computation, never
    # reusing the trace/determinant construction path.
    M = Matrix(A)
    eigs = list(M.eigenvals().keys())
    check(len(eigs) in (1, 2), "unexpected eigenvalue count")
    if len(eigs) == 1 or all(e.is_real for e in eigs):
        vals = sorted(float(e) for e in (eigs * 2 if len(eigs) == 1 else eigs))
        check(len(vals) == 2, "expected two real eigenvalues")
        l1, l2 = vals
        check(l1 != 0 and l2 != 0, "sympy eigenvalues include a zero (degenerate)")
        check(l1 != l2, "sympy eigenvalues are repeated (degenerate)")
        if l1 * l2 < 0:
            sym_class = "saddle"
        elif l1 < 0 and l2 < 0:
            sym_class = "stable node"
        else:
            sym_class = "unstable node"
    else:
        re_part = float(sympy.re(eigs[0]))
        if re_part == 0:
            sym_class = "centre"
        else:
            sym_class = "stable spiral" if re_part < 0 else "unstable spiral"
    check(sym_class == target, "sympy eigenvalue re-derivation mismatch")

    distractors = _pp_distractors(rng, target)

    working = [
        f"trace T = {T}, determinant D = {D}.",
        f"D {'< 0' if D < 0 else '> 0'}" + (f", T^2 - 4D = {T * T - 4 * D}" if D > 0 else "") + f" -> {target}.",
    ]

    return dict(
        text=text,
        answer=target,
        answer_text=target,
        answer_latex=latex_answer(target, plain=True),
        working=working,
        distractors=distractors,
    )
