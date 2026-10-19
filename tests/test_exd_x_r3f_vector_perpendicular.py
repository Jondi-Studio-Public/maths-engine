"""Exam chain of `senior-examine-properties-of-parallel-and-perpendicular-vectors-and`: dot products, k, |p|^2."""

from __future__ import annotations

import re

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

TOPIC = "exd_x_r3f_vector_perpendicular"
SKILL_ID = "senior-examine-properties-of-parallel-and-perpendicular-vectors-and"


def read_vectors(stem: str):
    nums = [int(n) for n in re.findall(r"-?\d+", re.sub(r"2u", "", stem))]
    return (nums[0], nums[1]), (nums[2], nums[3])


def oracle(q):
    u, v = read_vectors(q.stem)
    w = (2 * u[0] - v[0], 2 * u[1] - v[1])
    dot = lambda p, r: p[0] * r[0] + p[1] * r[1]  # noqa: E731
    uw, vw = dot(u, w), dot(v, w)
    k, remainder = divmod(-uw, vw)
    assert remainder == 0
    p = (u[0] + k * v[0], u[1] + k * v[1])
    assert dot(p, w) == 0
    return uw, vw, k, dot(p, p)


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    for q in questions:
        assert tuple(int(p.answer_text) for p in q.parts) == oracle(q), q.text


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    assert_answer_space(TOPIC)


def test_k_is_a_nonzero_whole_number(questions):
    for q in questions:
        assert int(q.parts[2].answer_text) != 0


def _wrong_dot_replies(q):
    """A wrong u . w in (a), then k and |p|^2 worked out correctly from it."""
    u, v = read_vectors(q.stem)
    w = (2 * u[0] - v[0], 2 * u[1] - v[1])
    uw_wrong = int(q.parts[0].distractors[0])
    vw = v[0] * w[0] + v[1] * w[1]
    if uw_wrong % vw:
        return None
    k = -uw_wrong // vw
    if k == 0:
        return None
    p = (u[0] + k * v[0], u[1] + k * v[1])
    reply_d = str(p[0] ** 2 + p[1] ** 2)
    if reply_d == q.parts[3].answer_text:
        return None  # a different k that happens to give the same |p|^2 is simply right
    return {"a": str(uw_wrong), "b": str(vw), "c": str(k), "d": reply_d}


def test_follow_through_from_a_wrong_dot_product():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_dot_replies, ["a"], ["c", "d"], seeds=range(300))


def _wrong_k_replies(q):
    u, v = read_vectors(q.stem)
    uw, vw, k, _p = oracle(q)
    k_wrong = int(q.parts[2].distractors[0])
    if k_wrong == 0:
        return None
    p = (u[0] + k_wrong * v[0], u[1] + k_wrong * v[1])
    reply_d = str(p[0] ** 2 + p[1] ** 2)
    if reply_d == q.parts[3].answer_text:
        return None
    return {"a": str(uw), "b": str(vw), "c": str(k_wrong), "d": reply_d}


def test_follow_through_from_a_wrong_k():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_k_replies, ["c"], ["d"])


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "991", "b": "992", "c": "993", "d": "994"})
