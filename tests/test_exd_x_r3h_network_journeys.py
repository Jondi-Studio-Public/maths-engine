"""Tests for the exam tier `exd_x_r3h_network_journeys` (senior-use-matrices-for-storing-and-displaying-information-that)."""

from __future__ import annotations

import re

import sympy

from mathgen.engine import grade
from tests import exd_x_r3h_support as sup
from tests.exd_x_r3h_common_tests import make_tests

TOPIC = "exd_x_r3h_network_journeys"
SKILL = "senior-use-matrices-for-storing-and-displaying-information-that"


def read(q):
    table = re.search(r"column: (.*?)\. A journey", q.stem).group(1)
    rows = [row.split(" | ") for row in table.split("; ")]
    names = rows[0][1:]
    matrix = sympy.Matrix([[int(v) for v in row[1:]] for row in rows[1:]])
    assert [row[0] for row in rows[1:]] == names
    start = re.search(r"leaving (.+?)\.$", q.parts[0].prompt).group(1)
    end = re.search(r"from .+? to (.+?) using exactly", q.parts[1].prompt).group(1)
    return matrix, names.index(start), names.index(end)


def oracle(q):
    matrix, p, e = read(q)
    squared = matrix**2
    return {
        "a": str(sum(matrix.row(p))),
        "b": str(squared[p, e]),
        "c": str(matrix[p, e] + squared[p, e]),
    }


(
    test_source_wired_and_shapes,
    test_keys_rederived_from_the_rendered_prompt,
    test_answer_space,
    test_key_accepted_and_every_distractor_rejected,
    test_follow_through_from_a_wrong_first_answer,
) = make_tests(TOPIC, SKILL, oracle)


def test_explicit_follow_through_into_at_most_two_services():
    skill = sup.skill(SKILL)
    for seed in range(60):
        q = sup.draw(TOPIC, seed)
        matrix, p, e = read(q)
        mine = (matrix**2)[p, e] + 2
        reply = {"a": q.parts[0].answer_text, "b": str(mine), "c": str(mine + matrix[p, e])}
        result = grade(skill, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
        assert result.parts["c"].follow_through is True


def test_the_matrix_has_a_zero_diagonal_and_names_are_distinct():
    for seed in range(100):
        q = sup.draw(TOPIC, seed)
        matrix, _, _ = read(q)
        assert all(matrix[i, i] == 0 for i in range(matrix.rows))
        table = re.search(r"column: (.*?)\. A journey", q.stem).group(1)
        header = table.split("; ")[0].split(" | ")[1:]
        assert len(set(header)) == len(header)
