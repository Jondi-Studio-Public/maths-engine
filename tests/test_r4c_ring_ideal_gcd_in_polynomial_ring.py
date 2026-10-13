"""Tests `uc2-ring-ideal-gcd-in-polynomial-ring`: generators of ideals are recomputed with SymPy from the polynomials
written in the question text, over Q or over Z_p."""

from __future__ import annotations

import re

import sympy
from sympy import Poly, QQ, symbols

import mathgen.generators.r4c_ring_ideal_gcd_in_polynomial_ring_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4c_support as sup

TOPIC = "r4c_ring_ideal_gcd_in_polynomial_ring"
SKILL_ID = "uc2-ring-ideal-gcd-in-polynomial-ring"
x = symbols("x")
_TERM = re.compile(r"([+-]?)\s*(\d*)\s*(x(?:\^(\d+))?)?")


def parse(text):
    """'4x^3 - 2x + 7' -> a SymPy expression in x."""
    total = 0
    for sign, coef, var, power in re.findall(r"([+-]?)\s*(\d*)(x(?:\^(\d+))?)?(?=\s|$)", text.strip()):
        if not coef and not var:
            continue
        c = int(coef) if coef else 1
        c = -c if sign == "-" else c
        total += c * (x ** (int(power) if power else 1) if var else 1)
    return total


def polynomials(text, count=2):
    found = re.findall(r"(?:f|g) = ([^,]*?)(?= and | with|\.|, |$)", text)
    return [parse(s) for s in found[:count]]


def _modulus(text):
    return int(re.search(r"Z_(\d+)", text).group(1))


def _to_text(poly, p):
    return Poly(poly, x, modulus=p)


def _canonical(poly, p):
    """Coefficients reduced into 0..p-1 as an integer-coefficient SymPy expression."""
    coeffs = Poly(poly, x, modulus=p).all_coeffs()
    degree = len(coeffs) - 1
    return sum((int(c) % p) * x ** (degree - i) for i, c in enumerate(coeffs))


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "expression")


def test_parser_reads_signed_terms():
    assert sympy.expand(parse("2x^3 - x^2 + 5x - 7") - (2 * x**3 - x**2 + 5 * x - 7)) == 0
    assert sympy.expand(parse("x^2 + 3") - (x**2 + 3)) == 0


def test_easy_gcd_over_q():
    for seed in range(80):
        question = sup.q(TOPIC, "easy", seed)
        f, g = polynomials(question.text)
        expected = Poly(f, x, domain=QQ).gcd(Poly(g, x, domain=QQ)).monic()
        assert sympy.expand(sympy.sympify(question.answer) - expected.as_expr()) == 0, (seed, question.text)
        assert 1 <= expected.degree() <= 2


def test_medium_gcd_over_zp():
    for seed in range(80):
        question = sup.q(TOPIC, "medium", seed)
        p = _modulus(question.text)
        f, g = polynomials(question.text)
        expected = Poly(f, x, modulus=p).gcd(Poly(g, x, modulus=p)).monic()
        assert sympy.expand(sympy.sympify(question.answer) - _canonical(expected.as_expr(), p)) == 0, (
            seed,
            question.text,
        )


def test_hard_lcm_over_zp():
    for seed in range(80):
        question = sup.q(TOPIC, "hard", seed)
        p = _modulus(question.text)
        f, g = polynomials(question.text)
        pf, pg = Poly(f, x, modulus=p), Poly(g, x, modulus=p)
        expected = (pf * pg).quo(pf.gcd(pg)).monic()
        assert sympy.expand(sympy.sympify(question.answer) - _canonical(expected.as_expr(), p)) == 0, (
            seed,
            question.text,
        )
        assert expected.degree() == pf.degree() + pg.degree() - pf.gcd(pg).degree()


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "expression")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 15, "medium": 30, "hard": 40})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(
        SKILL_ID, TOPIC, "abcd", (2, 3, 1, 1), ("expression", "expression", "number", "number"), (0, 0, 0, 1)
    )


def test_exam_keys_are_recomputed_from_the_text():
    for seed in range(60):
        question = sup.exam(TOPIC, seed)
        p = _modulus(question.stem)
        f, g = polynomials(question.stem)
        pf, pg = Poly(f, x, modulus=p), Poly(g, x, modulus=p)
        a, b, c, d = question.parts
        remainder = pf.rem(pg)
        assert sympy.expand(sympy.sympify(a.answer) - _canonical(remainder.as_expr(), p)) == 0, (seed, question.stem)
        gcd = pf.gcd(pg).monic()
        assert sympy.expand(sympy.sympify(b.answer) - _canonical(gcd.as_expr(), p)) == 0
        assert int(c.answer) == gcd.degree()
        assert int(d.answer) == p ** gcd.degree()


def test_exam_follow_through_for_the_ring_size():
    s = sup.skill(SKILL_ID)
    checked = 0
    for seed in range(40):
        question = sup.exam(TOPIC, seed)
        p = _modulus(question.stem)
        wrong = int(question.parts[2].answer) + 1
        sent = {part.label: part.answer_text for part in question.parts} | {"c": str(wrong), "d": str(p**wrong)}
        result = grade(s, question, sent)
        assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True, (seed, sent)
        checked += 1
    assert checked == 40
