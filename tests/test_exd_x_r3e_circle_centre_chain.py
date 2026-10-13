"""Multi-part exam tier of `senior-the-angle-at-the-centre-subtended-by-an`: angle ABC, AOC, OAC and BAC for three
points on a circle with the centre inside the triangle.

The oracle reads the two given base angles out of the rendered stem and places the points on the unit circle with
coordinates, so the angles are measured, not derived by the generator's own rule.
"""

from __future__ import annotations

import math
import re

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "senior-the-angle-at-the-centre-subtended-by-an"
TOPIC = "exd_x_r3e_circle_centre_chain"


def angle(p, q, r):
    v1, v2 = (p[0] - q[0], p[1] - q[1]), (r[0] - q[0], r[1] - q[1])
    return math.degrees(math.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))))


def oracle(q):
    given_a, given_c = (int(v) for v in re.findall(r"(\d+) degrees", q.stem))
    # B at angle 0; OA = OB so the central angle AOB is 180 - 2 * OAB, likewise for C on the other side.
    A = (math.cos(math.radians(180 - 2 * given_a)), math.sin(math.radians(180 - 2 * given_a)))
    C = (math.cos(-math.radians(180 - 2 * given_c)), math.sin(-math.radians(180 - 2 * given_c)))
    O, B = (0.0, 0.0), (1.0, 0.0)
    assert abs(angle(O, A, B) - given_a) < 1e-9 and abs(angle(O, C, B) - given_c) < 1e-9
    return given_a, given_c, [angle(A, B, C), angle(A, O, C), angle(O, A, C), angle(B, A, C)]


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abcd", ["number"] * 4, [2, 1, 1, 1])
    sup.assert_answer_space(TOPIC)


def test_both_wordings_of_the_second_angle_appear():
    seen = {("OCB" if "OCB" in sup.exam_question(TOPIC, s).stem else "OBC") for s in range(40)}
    assert seen == {"OCB", "OBC"}


def test_key_matches_measured_angles_from_the_rendered_stem():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        a, c, measured = oracle(q)
        assert a + c < 90
        for part, got in zip(q.parts, measured):
            assert abs(float(part.answer_text) - got) < 1e-6, (seed, q.stem, part.label, part.answer_text, got)


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC)


def test_a_wrong_first_angle_carries_through_every_later_part():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        a, c, _ = oracle(q)
        wa = a + c + 6
        wb = 2 * wa
        wc = (180 - wb) / 2
        if wc <= 0 or wc != int(wc):
            continue
        wd = a + int(wc)
        got = grade(skill, q, sup.reply(q, a=str(wa), b=str(wb), c=str(int(wc)), d=str(wd)))
        assert got.parts["a"].marks == 0
        assert (got.parts["b"].marks, got.parts["c"].marks, got.parts["d"].marks) == (1, 1, 1), (seed, got)


def test_a_wrong_centre_angle_does_not_earn_the_marks():
    skill = sup.skill_object(SKILL)
    for seed in range(40):
        q = sup.exam_question(TOPIC, seed)
        a, c, measured = oracle(q)
        got = grade(skill, q, sup.reply(q, b=str(int(measured[1]) + 4)))
        assert got.parts["b"].marks == 0
