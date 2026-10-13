"""Multi-part exam tier of `senior-determine-the-optimum-assignment-s-by-inspection-for`: the reduction totals, the
minimum cost they give, and the extra cost of assigning in order.

The oracle parses the cost table out of the rendered stem and brute-forces all 24 assignments.
"""

from __future__ import annotations

import re
from itertools import permutations

import exd_x_r3e_support as sup
from mathgen.engine import grade

SKILL = "senior-determine-the-optimum-assignment-s-by-inspection-for"
TOPIC = "exd_x_r3e_assignment_reduction"


def table(q):
    costs = [int(v) for v in re.findall(r"\$(\d+)", q.stem)]
    assert len(costs) == 16
    return [costs[4 * i : 4 * i + 4] for i in range(4)]


def oracle(q):
    cost = table(q)
    row_min = [min(r) for r in cost]
    reduced = [[c - m for c in r] for r, m in zip(cost, row_min)]
    col_min = [min(reduced[i][j] for i in range(4)) for j in range(4)]
    opt = min(sum(cost[i][p[i]] for i in range(4)) for p in permutations(range(4)))
    diag = sum(cost[i][i] for i in range(4))
    return sum(row_min), sum(col_min), opt, diag - opt, diag


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_clean_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)
    sup.assert_text_has_no_working_words(TOPIC)


def test_shape_and_variety():
    sup.assert_shapes(SKILL, TOPIC)
    sup.assert_chain_shape(TOPIC, "abcd", ["number"] * 4, [1, 1, 2, 1])
    sup.assert_answer_space(TOPIC)


def test_key_matches_independent_brute_force_from_the_rendered_stem():
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        R, C, opt, extra, diag = oracle(q)
        assert [int(p.answer_text) for p in q.parts] == [R, C, opt, extra], (seed, q.answer_text)
        assert R + C == opt and extra > 0


def test_key_and_distractors_are_graded_part_by_part():
    sup.assert_full_marks_and_distractors(SKILL, TOPIC)


def test_wrong_row_total_carries_into_the_minimum_and_the_extra_cost():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        R, C, opt, extra, diag = oracle(q)
        wa = R + 15
        if diag - (wa + C) <= 0:
            continue
        got = grade(skill, q, sup.reply(q, a=str(wa), c=str(wa + C), d=str(diag - (wa + C))))
        assert got.parts["a"].marks == 0
        assert got.parts["c"].marks == 2 and got.parts["c"].follow_through is True, (seed, got)
        assert got.parts["d"].marks == 1, (seed, got)


def test_a_wrong_minimum_carries_into_the_extra_cost_only():
    skill = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        R, C, opt, extra, diag = oracle(q)
        wc = opt + 20
        if diag - wc <= 0:
            continue
        got = grade(skill, q, sup.reply(q, c=str(wc), d=str(diag - wc)))
        assert got.parts["c"].marks == 0
        assert got.parts["d"].marks == 1 and got.parts["d"].follow_through is True, (seed, got)
        assert grade(skill, q, sup.reply(q, d=str(extra + 5))).parts["d"].marks == 0
