"""Multi-part exam tier of `complex-real-part`: Re(z), Im(z) and Re(z^2) for a quotient z = u/v.

The oracle parses u and v out of the rendered stem and works the quotient with SymPy's complex arithmetic.
"""

from __future__ import annotations

import re

import sympy

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "complex-real-part"
TOPIC = "exd_x_r3e_complex_quotient"


def parse_complex(text: str):
    return sympy.sympify(re.sub(r"(\d)i", r"\1*I", text).replace("i", "I").replace(" ", ""))


def oracle(q):
    m = re.search(r"z = \((.+?)\)/\((.+?)\)", q.stem)
    u, v = parse_complex(m.group(1)), parse_complex(m.group(2))
    z = sympy.expand(u / v, complex=True)
    return z, sympy.re(z), sympy.im(z), sympy.re(sympy.expand(z**2))


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abc", ["number", "number", "number"], [2, 1, 2])
    sup.assert_answer_space(TOPIC)


def test_key_matches_independent_derivation_from_the_rendered_stem():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        z, re_z, im_z, re_sq = oracle(q)
        assert [int(p.answer_text) for p in q.parts] == [re_z, im_z, re_sq], (seed, q.stem, q.answer_text)
        assert re_z != 0 and im_z != 0


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC)


def test_wrong_real_and_imaginary_parts_carry_into_the_square():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        _, re_z, im_z, _ = oracle(q)
        wa, wb = int(re_z) + 3, int(im_z) - 2
        if 0 in (wa, wb, wa * wa - wb * wb):
            continue
        replies = sup.reply(q, a=str(wa), b=str(wb), c=str(wa * wa - wb * wb))
        got = grade(skill, q, replies)
        assert got.parts["a"].marks == 0 and got.parts["b"].marks == 0
        assert got.parts["c"].marks == 2 and got.parts["c"].follow_through is True, (seed, replies)


def test_one_wrong_part_carries_and_a_wrong_square_does_not():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        _, re_z, im_z, re_sq = oracle(q)
        wa = int(re_z) + 2
        if wa == 0 or wa * wa == int(im_z) ** 2:
            continue
        got = grade(skill, q, sup.reply(q, a=str(wa), c=str(wa * wa - int(im_z) ** 2)))
        assert got.parts["a"].marks == 0 and got.parts["c"].marks == 2
        got = grade(skill, q, sup.reply(q, c=str(int(re_sq) + 1)))
        assert got.parts["c"].marks == 0 and got.parts["a"].marks == 2
