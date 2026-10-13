"""Multi-part exam tier of `senior-integrate-using-the-trigonometric-identities-mathrm-s-mathrm`: rewrite with an
identity, integrate, evaluate a definite integral.

The oracle reads f out of the rendered stem and the upper limit out of part (c), then checks (a) by comparing the
rewritten form with f at sample points, (b) by differentiating, and (c) by SymPy's own definite integral.
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "senior-integrate-using-the-trigonometric-identities-mathrm-s-mathrm"
TOPIC = "exd_x_r3e_trig_identity_integral"
x, C = sympy.symbols("x C")
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def parse(text: str):
    return parse_expr(text.replace("^", "**"), local_dict={"x": x, "C": C, "pi": sympy.pi}, transformations=TRANSFORMS)


def upper_limit(q):
    return sympy.pi / int(re.search(r"x = π/(\d+)\.$", q.parts[2].prompt).group(1))


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abc", ["fraction", "expression", "expression"], [1, 2, 2])
    sup.assert_answer_space(TOPIC)


def test_all_three_families_appear():
    seen = set()
    for seed in range(60):
        s = sup.exam_question(TOPIC, seed).stem
        seen.add("sin2" if "sin^2" in s else "cos2" if "cos^2" in s else "sincos")
    assert seen == {"sin2", "cos2", "sincos"}


def test_key_matches_independent_derivation_from_the_rendered_stem():
    for seed in range(120):
        q = sup.exam_question(TOPIC, seed)
        f = parse(re.search(r"f\(x\) = (.+)\.$", q.stem).group(1))
        L = upper_limit(q)
        a_part, b_part, c_part = q.parts
        a = sympy.Rational(a_part.answer_text)
        # (a): the rewritten form agrees with f at sample points
        k2 = int(re.search(r"\((\d+)x\)", a_part.prompt).group(1))
        form = {"- a cos": a - a * sympy.cos(k2 * x), "+ a cos": a + a * sympy.cos(k2 * x)}
        rewritten = next((v for key, v in form.items() if key in a_part.prompt), a * sympy.sin(k2 * x))
        for probe in (sympy.Rational(1, 3), sympy.Rational(5, 7)):
            assert abs(float((rewritten - f).subs(x, probe))) < 1e-12, (seed, q.stem)
        # (b): F' = f, with + C present
        F = parse(b_part.answer_text)
        assert sympy.simplify(sympy.diff(F, x) - f) == 0, (seed, b_part.answer_text)
        assert "+ C" in b_part.answer_text
        # (c): the exact value agrees with SymPy's definite integral
        exact = sympy.integrate(f, (x, 0, L))
        assert sympy.simplify(parse(c_part.answer_text) - exact) == 0, (seed, c_part.answer_text, exact)


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC, seeds=range(60))


def test_wrong_constant_carries_into_the_integral_and_the_value():
    skill = sup.skill_object(SKILL)
    checked = 0
    for seed in range(40):
        q = sup.exam_question(TOPIC, seed)
        L = upper_limit(q)
        key_b = parse(q.parts[1].answer_text)
        a = sympy.Rational(q.parts[0].answer_text)
        wa = a + sympy.Rational(3, 2)
        # the key's x-coefficient and trigonometric term are both a times something, so scaling by wa/a is the
        # integral a learner gets from the wrong constant
        wb = sympy.expand((key_b - C) * (wa / a)) + C
        wc = sympy.simplify(wb.subs(x, L) - wb.subs(x, 0))
        text_b = sympy.sstr(wb)
        got = grade(skill, q, sup.reply(q, a=str(wa), b=text_b, c=sympy.sstr(wc)))
        assert got.parts["a"].marks == 0
        assert got.parts["b"].marks == 2 and got.parts["c"].marks == 2, (seed, text_b, sympy.sstr(wc), got)
        checked += 1
    assert checked >= 30


def test_a_wrong_value_earns_nothing():
    skill = sup.skill_object(SKILL)
    for seed in range(30):
        q = sup.exam_question(TOPIC, seed)
        exact = parse(q.parts[2].answer_text)
        got = grade(skill, q, sup.reply(q, c=sympy.sstr(exact + 1)))
        assert got.parts["c"].marks == 0
