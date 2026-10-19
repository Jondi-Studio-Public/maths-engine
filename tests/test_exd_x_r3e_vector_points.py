"""Multi-part exam tier of `senior-use-ordered-pair-notation-and-column-vector-notation`: B from A and AB, C from B
and CB, the vector AC, and the midpoint of AC, all along the one axis the question names.

The oracle parses A and both column vectors out of the rendered stem and works every answer from them.
"""

from __future__ import annotations

import re

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "senior-use-ordered-pair-notation-and-column-vector-notation"
TOPIC = "exd_x_r3e_vector_points"


def oracle(q):
    nums = [int(n) for n in re.findall(r"-?\d+", q.stem)]
    A, AB, CB = nums[0:2], nums[2:4], nums[4:6]
    t = 0 if "x-coordinate" in q.parts[0].prompt else 1
    B = [A[i] + AB[i] for i in (0, 1)]
    C = [B[i] - CB[i] for i in (0, 1)]
    AC = [C[i] - A[i] for i in (0, 1)]
    M = [A[i] + AC[i] / 2 for i in (0, 1)]
    return t, A, AB, CB, B, C, AC, M


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abcd", ["number"] * 4, [1, 2, 1, 2])
    sup.assert_answer_space(TOPIC)


def test_both_axes_are_asked():
    axes = {"x" if "x-coordinate" in sup.exam_question(TOPIC, s).parts[0].prompt else "y" for s in range(40)}
    assert axes == {"x", "y"}


def test_key_matches_independent_derivation_from_the_rendered_stem():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        t, A, AB, CB, B, C, AC, M = oracle(q)
        assert [int(p.answer_text) for p in q.parts] == [B[t], C[t], AC[t], M[t]], (seed, q.text, q.answer_text)
        assert M[t] == int(M[t])


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC)


def test_wrong_b_carries_through_c_d_and_the_vector():
    skill = sup.skill_object(SKILL)
    checked = 0
    for seed in range(80):
        q = sup.exam_question(TOPIC, seed)
        t, A, AB, CB, B, C, AC, M = oracle(q)
        wa = B[t] + 4  # an arithmetic slip in (a)
        wb = wa - CB[t]
        wc = wb - A[t]
        if wc % 2 or wa == B[t] or wb == C[t] or 0 in (wa, wb, wc):
            continue
        wd = A[t] + wc // 2
        if wd in (wa, wb, wc, 0) or wc in (wa, wb):
            continue
        got = grade(skill, q, sup.reply(q, a=str(wa), b=str(wb), c=str(wc), d=str(wd)))
        assert got.parts["a"].marks == 0
        assert (got.parts["b"].marks, got.parts["c"].marks, got.parts["d"].marks) == (2, 1, 2), (seed, got)
        checked += 1
    assert checked >= 20


def test_a_wrong_vector_carries_into_the_midpoint_but_a_wrong_midpoint_earns_nothing():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        t, A, AB, CB, B, C, AC, M = oracle(q)
        wc = AC[t] + 2
        if wc == 0 or A[t] + wc // 2 in (0, wc):
            continue
        got = grade(skill, q, sup.reply(q, c=str(wc), d=str(A[t] + wc // 2)))
        assert got.parts["c"].marks == 0 and got.parts["d"].marks == 2 and got.parts["d"].follow_through is True
        assert grade(skill, q, sup.reply(q, d=str(int(M[t]) + 1))).parts["d"].marks == 0
