"""Multi-part exam tier of `senior-find-and-use-the-derivative-of-the-inverse`: f'(x), the gradient at the point
where the inside function is 1, and where the tangent there meets the x-axis.

The oracle reads the function back out of the rendered stem and works the answers with SymPy and a numerical
difference quotient; it never imports the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "senior-find-and-use-the-derivative-of-the-inverse"
TOPIC = "exd_x_r3e_inverse_trig_tangent"
x = sympy.Symbol("x")
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def parse(text: str):
    return parse_expr(
        text.replace("^", "**").replace("arctan", "atan"), local_dict={"x": x}, transformations=TRANSFORMS
    )


def oracle(q):
    f = parse(re.search(r"f\(x\) = (.+)\.$", q.stem).group(1))
    x0 = int(re.search(r"where x = (-?\d+)\.", q.parts[1].prompt).group(1))
    fp = sympy.diff(f, x)
    slope = sympy.nsimplify(fp.subs(x, x0))
    value = sympy.N(f.subs(x, x0), 30)
    intercept = x0 - value / sympy.N(slope, 30)
    return f, fp, x0, sympy.Rational(slope), intercept


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abc", ["expression", "fraction", "number"], [2, 1, 2])
    sup.assert_answer_space(TOPIC)


def test_key_matches_independent_derivation_from_the_rendered_stem():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        f, fp, x0, slope, intercept = oracle(q)
        a, b, c = q.parts
        assert sympy.simplify(parse(a.answer_text) - fp) == 0, (seed, q.stem, a.answer_text)
        assert Fraction(b.answer_text) == Fraction(int(slope.p), int(slope.q)), (seed, b.answer_text)
        assert abs(sympy.Float(c.answer_text, 10) - intercept) <= sympy.Rational(1, 200) + sympy.Rational(1, 10**9), (
            seed,
            c.answer_text,
            intercept,
        )
        # numerical tangent: the gradient from a difference quotient, the x-intercept from the tangent line
        fl = sympy.lambdify(x, f, "math")
        h = 1e-6
        m_num = (fl(x0 + h) - fl(x0 - h)) / (2 * h)
        assert abs(m_num - float(slope)) < 1e-5
        assert abs((x0 - fl(x0) / m_num) - float(c.answer_text)) < 0.006 + 1e-4


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC, seeds=range(80))


def test_the_wrong_first_part_carries_into_the_gradient_and_the_intercept():
    skill = sup.skill_object(SKILL)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        f, fp, x0, slope, intercept = oracle(q)
        wrong_a = q.parts[0].distractors[0]
        slope_w = sympy.nsimplify(parse(wrong_a).subs(x, x0))
        if not slope_w.is_Rational or slope_w == 0 or slope_w == slope:
            continue
        value = sympy.N(f.subs(x, x0), 30)
        target = x0 - value / sympy.N(slope_w, 30)
        if abs(target) > 10**6:
            continue
        text_c = f"{float(sympy.N(target, 30)):.2f}"
        replies = sup.reply(q, a=wrong_a, b=str(slope_w), c=text_c)
        got = grade(skill, q, replies)
        assert got.parts["a"].marks == 0
        assert got.parts["b"].marks == q.parts[1].marks and got.parts["b"].follow_through is True, (seed, replies)
        assert got.parts["c"].marks == q.parts[2].marks, (seed, replies)
        checked += 1
    assert checked >= 25


def test_a_wrong_gradient_carries_into_the_intercept():
    skill = sup.skill_object(SKILL)
    for seed in range(40):
        q = sup.exam_question(TOPIC, seed)
        f, fp, x0, slope, intercept = oracle(q)
        slope_w = slope + sympy.Rational(3, 2)
        if slope_w == 0:
            continue
        target = x0 - sympy.N(f.subs(x, x0), 30) / sympy.N(slope_w, 30)
        replies = sup.reply(q, b=str(slope_w), c=f"{float(target):.2f}")
        got = grade(skill, q, replies)
        assert got.parts["b"].marks == 0
        assert got.parts["c"].marks == q.parts[2].marks and got.parts["c"].follow_through is True, (seed, replies)


def test_a_wrong_intercept_does_not_earn_the_marks():
    skill = sup.skill_object(SKILL)
    for seed in range(40):
        q = sup.exam_question(TOPIC, seed)
        wrong = f"{float(q.parts[2].answer_text) + 1:.2f}"
        assert grade(skill, q, sup.reply(q, c=wrong)).parts["c"].marks == 0
