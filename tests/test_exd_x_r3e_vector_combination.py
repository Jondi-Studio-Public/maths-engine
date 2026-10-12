"""Multi-part exam tier of `senior-define-and-use-multiplication-by-a-scalar-of`: w = m p - n q, then the scalar k
that removes the i-component of kp + w, then the j-coefficient of that vector.

The oracle parses p, q, m and n out of the rendered stem and works every answer from them.
"""

from __future__ import annotations

import re

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "senior-define-and-use-multiplication-by-a-scalar-of"
TOPIC = "exd_x_r3e_vector_combination"


def comps(text: str) -> tuple[int, int]:
    """(i, j) coefficients of a vector written like '-6i - 4j' or 'i + 2j'."""
    i = re.search(r"(?:^|\s)([+-]?\s?\d*)i", " " + text)
    j = re.search(r"([+-]?\s?\d*)j", text)

    def val(m):
        s = m.group(1).replace(" ", "")
        return {"": 1, "+": 1, "-": -1}.get(s, None) or int(s)

    return val(i), val(j)


def oracle(q):
    m = re.search(r"p = (.+?) and q = (.+?)\. The vector w = (\d+)p - (\d+)q\.", q.stem)
    p, qq = comps(m.group(1)), comps(m.group(2))
    mm, nn = int(m.group(3)), int(m.group(4))
    w = (mm * p[0] - nn * qq[0], mm * p[1] - nn * qq[1])
    k = -w[0] // p[0]
    assert k * p[0] + w[0] == 0
    return p, qq, w, k, k * p[1] + w[1]


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abcd", ["number"] * 4, [1, 1, 2, 1])
    sup.assert_answer_space(TOPIC)


def test_key_matches_independent_derivation_from_the_rendered_stem():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        p, qq, w, k, last = oracle(q)
        assert [int(x.answer_text) for x in q.parts] == [w[0], w[1], k, last], (seed, q.stem, q.answer_text)


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC)


def test_wrong_w_carries_into_the_scalar_and_the_last_coefficient():
    skill = sup.skill_object(SKILL)
    checked = 0
    for seed in range(80):
        q = sup.exam_question(TOPIC, seed)
        p, qq, w, k, last = oracle(q)
        wa, wb = w[0] + 2 * p[0], w[1] + 3
        if wa == 0 or wa % p[0]:
            continue
        kw = -wa // p[0]
        if kw == 0:
            continue
        got = grade(skill, q, sup.reply(q, a=str(wa), b=str(wb), c=str(kw), d=str(kw * p[1] + wb)))
        assert got.parts["a"].marks == 0 and got.parts["b"].marks == 0
        assert got.parts["c"].marks == 2 and got.parts["d"].marks == 1, (seed, got)
        checked += 1
    assert checked >= 30


def test_wrong_scalar_carries_into_the_last_coefficient_only():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        p, qq, w, k, last = oracle(q)
        kw = k + 2
        if kw == 0 or kw * p[1] + w[1] == 0:
            continue
        got = grade(skill, q, sup.reply(q, c=str(kw), d=str(kw * p[1] + w[1])))
        assert got.parts["c"].marks == 0
        assert got.parts["d"].marks == 1 and got.parts["d"].follow_through is True, (seed, got)
        got = grade(skill, q, sup.reply(q, d=str(last + 1)))
        assert got.parts["d"].marks == 0
