"""Independent oracle for senior-factorise-cubic-polynomials-in-cases-where-a-linear.

Re-derives the (linear)(quadratic) factorisation from `question.text` alone by
applying the rational root theorem (brute-force integer divisor search) and
synthetic division -- both written from scratch here, with no reference to
how the generator picked p, q, r, a. Never imports
`mathgen.generators.p7c_polynomial_solve_build`.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make

TOPIC = "factorise_cubic_linear_factor"

_QUESTION_RE = re.compile(r"Factorise (.+) completely as a product")
_ANSWER_RE = re.compile(r"^(-?\d+)?\((.+?)\)\((.+?)\)$")


def _parse_signed_terms(expr_text: str, symbols: list[str]) -> dict:
    """``{symbol: coefficient}`` for a join_terms-style signed sum of terms.

    ``symbols`` lists the recognised variable markers in descending-degree
    order (e.g. ``["x^3", "x^2", "x", ""]``); a term with none of them is the
    constant. Coefficient magnitude 1 is implicit when a symbol is present
    (``"x^2"`` means coefficient 1); a bare number is always the constant.
    """
    s = expr_text.strip()
    if not s.startswith(("+", "-")):
        s = "+" + s
    coeffs = {sym: 0 for sym in symbols}
    for raw_term in re.findall(r"[+-]\s*[^+-]+", s):
        term = raw_term.strip()
        sign = -1 if term.startswith("-") else 1
        body = term[1:].strip()
        matched_symbol = ""
        for sym in symbols:
            if sym and body.endswith(sym):
                matched_symbol = sym
                body = body[: -len(sym)].strip()
                break
        coef_str = body
        coef = int(coef_str) if coef_str else 1
        coeffs[matched_symbol] = sign * coef
    return coeffs


def _cubic_coeffs(text: str):
    m = _QUESTION_RE.search(text)
    if not m:
        raise ValueError(f"could not find cubic in: {text!r}")
    terms = _parse_signed_terms(m.group(1), ["x^3", "x^2", "x", ""])
    return terms["x^3"], terms["x^2"], terms["x"], terms[""]


def _parse_factored_answer(answer_text: str):
    """(a, p, q, r) from "a(x - p)(x^2 + qx + r)" / "(x - p)(x^2 + qx + r)"."""
    m = _ANSWER_RE.match(answer_text.strip())
    if not m:
        raise ValueError(f"could not parse answer_text: {answer_text!r}")
    a_str, linear_body, quad_body = m.groups()
    a = int(a_str) if a_str else 1
    lin_terms = _parse_signed_terms(linear_body, ["x", ""])
    assert lin_terms["x"] == 1, f"linear factor's x-coefficient must be 1: {linear_body!r}"
    p = -lin_terms[""]
    quad_terms = _parse_signed_terms(quad_body, ["x^2", "x", ""])
    assert quad_terms["x^2"] == 1, f"quadratic factor must be monic: {quad_body!r}"
    return a, p, quad_terms["x"], quad_terms[""]


def _synthetic_divide(coeffs, root):
    """Divide [A, B, C, D] (highest degree first) by (x - root)."""
    out = [coeffs[0]]
    for c in coeffs[1:]:
        out.append(c + root * out[-1])
    remainder = out.pop()
    return out, remainder


def _rational_roots(A, B, C, D):
    """Every integer root of Ax^3+Bx^2+Cx+D found by brute-force divisor search
    over the constant term (D=0 handled separately) -- the rational root
    theorem, applied from scratch, not read from the generator."""
    roots = []
    if D == 0:
        roots.append(0)
        candidates = range(-40, 41)
    else:
        candidates = range(-abs(D), abs(D) + 1)
    for candidate in candidates:
        if candidate == 0 and D != 0:
            continue
        value = A * candidate**3 + B * candidate**2 + C * candidate + D
        if value == 0 and candidate not in roots:
            roots.append(candidate)
    return roots


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_factorisation_matches_answer_text(difficulty):
    mismatches = []
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        A, B, C, D = _cubic_coeffs(q.text)
        a_ans, p_ans, q_ans, r_ans = _parse_factored_answer(q.answer_text)

        roots = _rational_roots(A, B, C, D)
        if p_ans not in roots:
            mismatches.append((seed, q.text, q.answer_text, "p not an integer root", roots))
            continue

        quotient, remainder = _synthetic_divide([A, B, C, D], p_ans)
        if remainder != 0:
            mismatches.append((seed, q.text, q.answer_text, "remainder != 0", quotient, remainder))
            continue

        a2, b2, c2 = quotient  # a2*x^2 + b2*x + c2
        if a2 != a_ans or b2 != a_ans * q_ans or c2 != a_ans * r_ans:
            mismatches.append((seed, q.text, q.answer_text, "quotient mismatch", quotient, (a_ans, q_ans, r_ans)))

    assert not mismatches, f"oracle mismatches: {mismatches}"


def test_oracle_covers_all_tiers():
    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(5):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            _cubic_coeffs(q.text)
            _parse_factored_answer(q.answer_text)


def test_hard_tier_quadratic_factor_is_irreducible_over_the_integers():
    """'Completely' means the quadratic factor must not split further (e.g. 2x^3+8x^2+8x = 2x(x+2)^2)."""
    import sympy
    from mathgen import make

    xs = sympy.Symbol("x")
    for seed in range(400):
        q = make("factorise_cubic_linear_factor", seed=seed, tier="hard")
        factors = sympy.factor_list(sympy.sympify(q.answer))[1]
        assert any(sympy.degree(f, xs) == 2 for f, _ in factors), (seed, q.answer_text)
