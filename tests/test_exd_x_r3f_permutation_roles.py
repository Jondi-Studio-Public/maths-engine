"""Exam chain of `permutations`: ways to fill roles, none from a subgroup, at least one, and the probability."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from exd_x_r3f_support import (
    assert_all_wrong_earns_nothing,
    assert_answer_space,
    assert_clean_and_deterministic,
    assert_follow_through,
    assert_full_marks_and_distractors,
    assert_shapes,
    assert_wired,
    exam_question,
)

TOPIC = "exd_x_r3f_permutation_roles"
SKILL_ID = "permutations"


def read_question(q):
    n, s = (int(v) for v in re.search(r"has (\d+) members, (\d+) of whom", q.stem).groups())
    roles = re.search(r"The roles of (.*) are each filled", q.stem).group(1)
    r = len(re.split(r", | and ", roles))
    return n, s, r


def oracle(q):
    n, s, r = read_question(q)
    # Count arrangements of r distinct members into r ordered roles, by brute-force multiplication per role.
    total = 1
    for i in range(r):
        total *= n - i
    none = 1
    for i in range(r):
        none *= (n - s) - i
    return total, none, total - none, Fraction(total - none, total)


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "fraction"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    for q in questions:
        total, none, at_least, probability = oracle(q)
        n, s, r = read_question(q)
        assert total == math.perm(n, r)
        assert [int(q.parts[i].answer_text) for i in range(3)] == [total, none, at_least], q.text
        assert Fraction(q.parts[3].answer_text) == probability, q.text


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC, min_options=2)


def test_answer_space():
    assert_answer_space(TOPIC)


def test_all_three_role_counts_and_both_contexts_occur(questions):
    assert {read_question(q)[2] for q in questions} == {3, 4, 5}
    assert len({q.stem.split(" has ")[0] for q in questions}) == 3


def _wrong_total_replies(q):
    total, none, _al, _p = oracle(q)
    larger = [int(d) for d in q.parts[0].distractors if int(d) > none]
    if not larger:
        return None
    total_wrong = larger[0]
    at_least = total_wrong - none
    return {
        "a": str(total_wrong),
        "b": str(none),
        "c": str(at_least),
        "d": str(Fraction(at_least, total_wrong)),
    }


def test_follow_through_from_a_wrong_total():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_total_replies, ["a"], ["c", "d"])


def _wrong_complement_replies(q):
    total, none, _al, _p = oracle(q)
    none_wrong = int(q.parts[1].distractors[0])
    at_least = total - none_wrong
    if at_least <= 0:
        return None
    return {
        "a": str(total),
        "b": str(none_wrong),
        "c": str(at_least),
        "d": str(Fraction(at_least, total)),
    }


def test_follow_through_from_a_wrong_complement():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_complement_replies, ["b"], ["c", "d"])


def test_unreduced_probability_is_accepted():
    from exd_x_r3f_support import skill
    from mathgen.engine import grade

    q = exam_question(TOPIC, 11)
    total, none, at_least, _p = oracle(q)
    replies = {"a": str(total), "b": str(none), "c": str(at_least), "d": f"{at_least}/{total}"}
    assert grade(skill(SKILL_ID), q, replies).correct


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "991", "b": "992", "c": "993", "d": "994/995"})
