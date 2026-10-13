"""Multi-part exam tier of `ucv1-product-measure-rectangle`: mu(A), nu(B), the measure of A x B, and the measure of
the part of X x Y outside it.

The oracle reads the point masses and the subsets out of the rendered stem and sums the masses over the points of the
product space directly, without using the product rule the question tests.
"""

from __future__ import annotations

import re
from fractions import Fraction
from itertools import product

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "ucv1-product-measure-rectangle"
TOPIC = "exd_x_r3e_product_measure_complement"


def oracle(q):
    mu = [Fraction(v) for v in re.findall(r"mu\(x\d\) = ([\d/]+)", q.stem)]
    nu = [Fraction(v) for v in re.findall(r"nu\(y\d\) = ([\d/]+)", q.stem)]
    a_set = {int(i) - 1 for i in re.findall(r"x(\d)", re.search(r"A = \{(.*?)\}", q.stem).group(1))}
    b_set = {int(i) - 1 for i in re.findall(r"y(\d)", re.search(r"B = \{(.*?)\}", q.stem).group(1))}
    cells = list(product(range(3), range(3)))
    mu_a = sum(mu[i] for i in a_set)
    nu_b = sum(nu[j] for j in b_set)
    rect = sum(mu[i] * nu[j] for i, j in cells if i in a_set and j in b_set)
    outside = sum(mu[i] * nu[j] for i, j in cells if not (i in a_set and j in b_set))
    return mu_a, nu_b, rect, outside


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abcd", ["fraction"] * 4, [1, 1, 2, 2])
    sup.assert_answer_space(TOPIC)


def test_key_matches_independent_derivation_from_the_rendered_stem():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        want = oracle(q)
        got = tuple(Fraction(p.answer_text) for p in q.parts)
        assert got == want, (seed, q.stem, q.answer_text)


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC)


def test_wrong_set_measures_carry_into_the_rectangle_and_its_complement():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        mu_a, nu_b, rect, outside = oracle(q)
        total = rect + outside
        wa = mu_a + Fraction(1, 3)
        wc = wa * nu_b
        wd = total - wc
        if wd <= 0 or wc in (wa, nu_b) or wd in (wa, nu_b, wc):
            continue
        got = grade(skill, q, sup.reply(q, a=str(wa), c=str(wc), d=str(wd)))
        assert got.parts["a"].marks == 0
        assert got.parts["c"].marks == 2 and got.parts["c"].follow_through is True, (seed, got)
        assert got.parts["d"].marks == 2, (seed, got)


def test_a_wrong_rectangle_carries_but_a_wrong_complement_does_not():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        mu_a, nu_b, rect, outside = oracle(q)
        total = rect + outside
        wc = rect + Fraction(1, 7)
        if total - wc in (wc, mu_a, nu_b):
            continue
        got = grade(skill, q, sup.reply(q, c=str(wc), d=str(total - wc)))
        assert got.parts["c"].marks == 0
        assert got.parts["d"].marks == 2 and got.parts["d"].follow_through is True, (seed, got)
        assert grade(skill, q, sup.reply(q, d=str(outside + 1))).parts["d"].marks == 0
