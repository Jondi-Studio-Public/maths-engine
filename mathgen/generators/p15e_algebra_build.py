"""Generates `p15e_quotient_ring_operation` questions: add or multiply two polynomials modulo a monic polynomial.

Quotient-ring polynomial arithmetic.

``p15e_quotient_ring_operation`` (``ucv1-quotient-ring-operation``, ``answer:
expression``) -- reduce the sum or product of two supplied polynomials
modulo a supplied monic polynomial m(x): compute a(x) op b(x) in
Z[x]/(m(x)) (or Z_p[x]/(m(x)) at the exam tier), returning the unique
representative of degree < deg(m). Construction reduces via the recurrence
x^d = -(low-degree part of m) -- the substitution rule a student applies by
hand; the independent check re-derives the SAME remainder via SymPy's own
polynomial division (``sympy.rem``/``sympy.div``), a genuinely different
algorithm.

Brand new topic/module; no existing generator is edited.
"""

from __future__ import annotations

import sympy

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    join_terms,
    phrase,
    pick_distractors,
)

_MAX_ATTEMPTS = 500
_X = sympy.symbols("x")


def _fmt(expr) -> str:
    """Canonical display for `answer_text`/a distractor: no Python `**`
    (forbidden on every learner-facing field by `mathgen.text_quality`'s
    `double_star` check) -- rewritten to `^`, which `_parse_expression`
    converts straight back."""
    return str(sympy.expand(expr)).replace("**", "^")


def _text_fmt(expr) -> str:
    """Display for embedding a polynomial INSIDE the prompt `text`, where
    `star_operator` also applies: a bare `*` is rewritten to `·`."""
    return str(sympy.expand(expr)).replace("**", "^").replace("*", "·")


def _poly_from_asc(coeffs) -> sympy.Expr:
    return sympy.expand(sum(c * _X**i for i, c in enumerate(coeffs)))


def _asc_coeffs(expr) -> list:
    """Ascending-order integer coefficient list for `expr` (ZZ[x]), degree
    read directly from `sympy.Poly` -- never assumed from how many terms
    were passed in, since a leading term can cancel."""
    expr = sympy.expand(expr)
    if expr == 0:
        return [0]
    poly = sympy.Poly(expr, _X)
    deg = poly.degree()
    return [int(poly.coeff_monomial(_X**i)) for i in range(deg + 1)]


def _reduce(raw, m_asc, *, wrong_sign=False, drop_constant_term=False):
    """Reduce `raw` (a sympy expr) modulo the monic polynomial whose
    ascending coefficients are `m_asc` (length d+1, m_asc[-1] == 1), via the
    substitution recurrence x^d = -(m_asc[0] + ... + m_asc[d-1] x^{d-1}) --
    the SAME rule used both for the correct construction and, with a flag
    flipped, for the sign-error/drops-a-term misconceptions below (so those
    distractors are genuinely "the same method, one step wrong", not an
    unrelated wrong value).
    """
    d = len(m_asc) - 1
    coeffs = _asc_coeffs(raw)
    sign = 1 if wrong_sign else -1
    while len(coeffs) - 1 >= d:
        top_deg = len(coeffs) - 1
        top = coeffs[-1]
        shift = top_deg - d
        for i in range(d):
            if drop_constant_term and i == 0:
                continue
            coeffs[shift + i] += sign * top * m_asc[i]
        coeffs.pop()
    while len(coeffs) < d:
        coeffs.append(0)
    return _poly_from_asc(coeffs)


def _independent_reduce(raw, m_expr):
    """Independent check: SymPy's own division algorithm (`sympy.div`), a
    library routine never touched by the hand-substitution recurrence
    `_reduce` implements."""
    _, remainder = sympy.div(sympy.expand(raw), m_expr, _X)
    return sympy.expand(remainder)


def _mod_p_asc(expr, p) -> list:
    return [c % p for c in _asc_coeffs(expr)]


def _mod_p_expr(expr, p):
    return _poly_from_asc(_mod_p_asc(expr, p))


def _reverse_coeffs(expr, length) -> sympy.Expr:
    coeffs = _asc_coeffs(expr)
    coeffs = coeffs + [0] * (length - len(coeffs))
    return _poly_from_asc(list(reversed(coeffs[:length])))


_AB_POOL = tuple(range(-7, 8))
_M_POOL = tuple(range(-6, 7))
_PRIME_POOL = (5, 7, 11)


def _mono(degree: int) -> str:
    """x^d display for one monomial: "" (constant), "x", "x^2", ..."""
    if degree == 0:
        return ""
    if degree == 1:
        return "x"
    return f"x^{degree}"


def _poly_text(coeffs_asc, name: str) -> str:
    """ "{name}(x) = ..." for a polynomial given by ascending coefficients,
    via `join_terms` so a zero coefficient is dropped and a negative one
    reads as "- 3" rather than "+ (-3)" (avoiding displays like "x^2 + (0)x + (-2)" and "6x + (-3)")."""
    d = len(coeffs_asc) - 1
    terms = [(coeffs_asc[i], _mono(i)) for i in range(d, -1, -1)]
    return f"{name}(x) = {join_terms(terms)}"


def _m_text(m_asc) -> str:
    """ "m(x) = ..." for the MONIC modulus: same helper, leading coefficient
    forced to 1 (m_asc's own top entry is always 1, by construction)."""
    return _poly_text(m_asc, "m")


def _offer(answer, value, offered, formatted, label):
    value = sympy.expand(value)
    if any(sympy.expand(value - v) == 0 for v in offered) or sympy.expand(value - answer) == 0:
        return
    check_pairwise_distinct(answer, *offered, value, msg=f"{label} collided")
    offered.append(value)
    formatted.append(_fmt(value))


_EASY_PATTERNS = [
    "In the quotient ring Z[x]/(m(x)), {m_text}, let {a_text} and "
    "{b_text}. Compute a(x) + b(x), reduced to degree less than 2.",
    "Working in Z[x]/(m(x)) with {m_text}, add {a_text} and "
    "{b_text}, giving your answer as a representative of degree "
    "less than 2.",
    "Given {m_text}, and {a_text}, {b_text} in Z[x]/(m(x)), find a(x) + b(x) reduced modulo m(x).",
]

_MEDIUM_PATTERNS = [
    "In the quotient ring Z[x]/(m(x)), {m_text}, let {a_text} and "
    "{b_text}. Compute a(x)b(x), reduced modulo m(x) to degree "
    "less than 2.",
    "Working in Z[x]/(m(x)) with {m_text}, multiply {a_text} by {b_text} and reduce the product modulo m(x).",
    "Given {m_text}, and {a_text}, {b_text} in Z[x]/(m(x)), find a(x)b(x) reduced modulo m(x).",
]

_HARD_PATTERNS = [
    "In the quotient ring Z[x]/(m(x)), {m_text}, let {a_text} "
    "and {b_text}. Compute a(x)b(x), "
    "reduced modulo m(x) to degree less than 3.",
    "Working in Z[x]/(m(x)) with {m_text}, multiply {a_text} by {b_text} and reduce the product modulo m(x).",
    "Given {m_text}, and {a_text}, {b_text} in Z[x]/(m(x)), find a(x)b(x) reduced modulo m(x).",
]

_EXAM_PATTERNS = [
    "In the quotient ring Z_{p}[x]/(m(x)), {m_text}, let {a_text} "
    "and {b_text}, with every coefficient read modulo "
    "{p}. Compute a(x)b(x), reduced modulo m(x), and give every coefficient "
    "of your answer as the least non-negative residue modulo {p}.",
    "Working in Z_{p}[x]/(m(x)) with {m_text}, multiply {a_text} "
    "by {b_text} (coefficients modulo {p}) and reduce "
    "the product modulo m(x), giving every coefficient as a residue in "
    "{{0, ..., {p_minus_1}}}.",
    "Given {m_text} over Z_{p}, and {a_text}, {b_text} in "
    "Z_{p}[x]/(m(x)), find a(x)b(x) reduced modulo m(x) with "
    "coefficients as least non-negative residues modulo {p}.",
]


def _p15e_qr_easy(rng):
    for _ in range(_MAX_ATTEMPTS):
        m0, m1 = rng.sample(_M_POOL, 2)
        a0, a1, b0, b1 = (rng.choice(_AB_POOL) for _ in range(4))
        if a1 == 0 or b1 == 0:
            continue
        m_asc = [m0, m1, 1]
        a_expr = _poly_from_asc([a0, a1])
        b_expr = _poly_from_asc([b0, b1])
        raw = sympy.expand(a_expr + b_expr)
        answer = _reduce(raw, m_asc)
        check(
            sympy.expand(answer - _independent_reduce(raw, _poly_from_asc(m_asc))) == 0,
            "easy: independent division check failed",
        )

        offered, formatted = [], []
        _offer(answer, sympy.expand(a_expr - b_expr), offered, formatted, "subtracts-instead-of-adds")
        _offer(answer, _poly_from_asc([a0 + b1, a1 + b0]), offered, formatted, "mismatched-degree-addition")
        _offer(answer, a_expr, offered, formatted, "drops-second-polynomial")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        text = phrase(
            rng,
            _EASY_PATTERNS,
            m_text=_m_text(m_asc),
            a_text=_poly_text([a0, a1], "a"),
            b_text=_poly_text([b0, b1], "b"),
        )
        working = [
            f"a(x) + b(x) = {join_terms([(a1 + b1, 'x'), (a0 + b0, '')])} = {_text_fmt(answer)}, "
            "already of degree less than 2 -- no reduction step is needed."
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p15e_qr_medium(rng):
    for _ in range(_MAX_ATTEMPTS):
        m0, m1 = rng.sample(_M_POOL, 2)
        a0, a1, b0, b1 = (rng.choice(_AB_POOL) for _ in range(4))
        if a1 == 0 or b1 == 0:
            continue
        m_asc = [m0, m1, 1]
        m_expr = _poly_from_asc(m_asc)
        a_expr = _poly_from_asc([a0, a1])
        b_expr = _poly_from_asc([b0, b1])
        raw = sympy.expand(a_expr * b_expr)
        answer = _reduce(raw, m_asc)
        check(sympy.expand(answer - _independent_reduce(raw, m_expr)) == 0, "medium: independent division check failed")

        offered, formatted = [], []
        _offer(answer, raw, offered, formatted, "forgets-to-reduce")
        _offer(answer, _reduce(raw, m_asc, wrong_sign=True), offered, formatted, "sign-error-in-reduction")
        _offer(answer, _reduce(raw, m_asc, drop_constant_term=True), offered, formatted, "drops-a-term-of-modulus")
        _offer(answer, _reverse_coeffs(answer, 2), offered, formatted, "coefficient-transposition")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        text = phrase(
            rng,
            _MEDIUM_PATTERNS,
            m_text=_m_text(m_asc),
            a_text=_poly_text([a0, a1], "a"),
            b_text=_poly_text([b0, b1], "b"),
        )
        working = [
            f"a(x)b(x) = {_text_fmt(raw)} before reduction.",
            f"x^2 = {join_terms([(-m1, 'x'), (-m0, '')])} (mod m(x)), so a(x)b(x) reduces to {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p15e_qr_hard(rng):
    for _ in range(_MAX_ATTEMPTS):
        m0, m1, m2 = rng.sample(_M_POOL, 3)
        a0, a1, a2, b0, b1, b2 = (rng.choice(_AB_POOL) for _ in range(6))
        if a2 == 0 or b2 == 0:
            continue
        m_asc = [m0, m1, m2, 1]
        m_expr = _poly_from_asc(m_asc)
        a_expr = _poly_from_asc([a0, a1, a2])
        b_expr = _poly_from_asc([b0, b1, b2])
        raw = sympy.expand(a_expr * b_expr)
        answer = _reduce(raw, m_asc)
        check(sympy.expand(answer - _independent_reduce(raw, m_expr)) == 0, "hard: independent division check failed")

        offered, formatted = [], []
        _offer(answer, raw, offered, formatted, "forgets-to-reduce")
        _offer(answer, _reduce(raw, m_asc, wrong_sign=True), offered, formatted, "sign-error-in-reduction")
        _offer(answer, _reduce(raw, m_asc, drop_constant_term=True), offered, formatted, "drops-a-term-of-modulus")
        _offer(answer, _reverse_coeffs(answer, 3), offered, formatted, "coefficient-transposition")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        text = phrase(
            rng,
            _HARD_PATTERNS,
            m_text=_m_text(m_asc),
            a_text=_poly_text([a0, a1, a2], "a"),
            b_text=_poly_text([b0, b1, b2], "b"),
        )
        working = [
            f"a(x)b(x) = {_text_fmt(raw)} before reduction (degree up to 4).",
            f"Repeatedly substitute x^3 = {join_terms([(-m2, 'x^2'), (-m1, 'x'), (-m0, '')])} until the "
            f"degree drops below 3: a(x)b(x) reduces to {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("hard: could not draw a usable question")


def _p15e_qr_exam(rng):
    for _ in range(_MAX_ATTEMPTS):
        p = rng.choice(_PRIME_POOL)
        m0, m1 = rng.sample(range(0, p), 2)
        a0 = rng.randint(0, p - 1)
        a1 = rng.randint(1, p - 1)
        b0 = rng.randint(0, p - 1)
        b1 = rng.randint(1, p - 1)
        m_asc = [m0, m1, 1]
        m_expr = _poly_from_asc(m_asc)
        a_expr = _poly_from_asc([a0, a1])
        b_expr = _poly_from_asc([b0, b1])
        raw = sympy.expand(a_expr * b_expr)
        reduced_z = _reduce(raw, m_asc)
        answer = _mod_p_expr(reduced_z, p)
        check(
            sympy.expand(reduced_z - _independent_reduce(raw, m_expr)) == 0, "exam: independent division check failed"
        )
        check(all(0 <= c < p for c in _asc_coeffs(answer)), "exam: answer coefficient out of residue range")

        offered, formatted = [], []
        _offer(answer, reduced_z, offered, formatted, "forgets-mod-p-reduction")
        _offer(answer, _mod_p_expr(raw, p), offered, formatted, "forgets-to-reduce-mod-m")
        _offer(
            answer, _mod_p_expr(_reduce(raw, m_asc, wrong_sign=True), p), offered, formatted, "sign-error-in-reduction"
        )
        _offer(answer, _reverse_coeffs(answer, 2), offered, formatted, "coefficient-transposition")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        text = phrase(
            rng,
            _EXAM_PATTERNS,
            m_text=_m_text(m_asc),
            a_text=_poly_text([a0, a1], "a"),
            b_text=_poly_text([b0, b1], "b"),
            p=p,
            p_minus_1=p - 1,
        )
        working = [
            f"a(x)b(x) = {_text_fmt(raw)} over Z, reducing to {_text_fmt(reduced_z)} modulo m(x).",
            f"Reading each coefficient modulo {p} gives {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p15e_quotient_ring_operation",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p15e_quotient_ring_operation(rng, tier="medium"):
    if tier == "easy":
        return _p15e_qr_easy(rng)
    if tier == "medium":
        return _p15e_qr_medium(rng)
    if tier == "hard":
        return _p15e_qr_hard(rng)
    if tier == "exam":
        return _p15e_qr_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
