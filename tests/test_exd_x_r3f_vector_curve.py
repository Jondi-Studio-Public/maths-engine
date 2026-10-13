"""Exam chain of `senior-use-vector-equations-of-curves-in-two-or`: t at P, b, K in (x-p)(y-c)=K, y at a new x."""

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

TOPIC = "exd_x_r3f_vector_curve"
SKILL_ID = "senior-use-vector-equations-of-curves-in-two-or"


def read_question(q):
    m = re.search(r"\((-?\d+)t(?: ([+-]) (\d+))?, b/t(?: ([+-]) (\d+))?\)", q.stem)
    a = int(m.group(1))
    p = 0 if m.group(2) is None else (int(m.group(3)) if m.group(2) == "+" else -int(m.group(3)))
    c = 0 if m.group(4) is None else (int(m.group(5)) if m.group(4) == "+" else -int(m.group(5)))
    x0, y0 = (int(n) for n in re.search(r"P\((-?\d+), (-?\d+)\)", q.stem).groups())
    x1 = int(re.search(r"when x = (-?\d+)", q.parts[3].prompt).group(1))
    return a, p, c, x0, y0, x1


def oracle(q):
    a, p, c, x0, y0, x1 = read_question(q)
    t0, rem = divmod(x0 - p, a)
    assert rem == 0 and t0 != 0
    b, rem = y0 - c, 0
    b = (y0 - c) * t0
    k = a * b
    y1, rem = divmod(k, x1 - p)
    assert rem == 0
    return t0, b, k, y1 + c


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    for q in questions:
        assert tuple(int(p.answer_text) for p in q.parts) == oracle(q), q.text


def test_relation_in_part_c_matches_the_curve(questions):
    for q in questions[:100]:
        a, p, c, *_ = read_question(q)
        m = re.search(r"\(x([^)]*)\)\(y([^)]*)\) = K", q.parts[2].prompt)
        shift = lambda s: (
            0
            if not s.strip()
            else (int(s.replace(" ", "").replace("+", "")) if "+" in s else -int(s.replace(" ", "").replace("-", "")))
        )  # noqa: E731
        assert (-shift(m.group(1)), -shift(m.group(2))) == (p, c)


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    assert_answer_space(TOPIC)


def _wrong_t_replies(q):
    a, p, c, x0, y0, x1 = read_question(q)
    t_wrong = int(q.parts[0].distractors[0])
    if t_wrong == 0:
        return None
    b = (y0 - c) * t_wrong
    k = a * b
    if (x1 - p) == 0 or k % (x1 - p):
        return None
    return {"a": str(t_wrong), "b": str(b), "c": str(k), "d": str(c + k // (x1 - p))}


def test_follow_through_from_a_wrong_parameter():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_t_replies, ["a"], ["b", "c", "d"], seeds=range(300))


def _wrong_k_replies(q):
    a, p, c, x0, y0, x1 = read_question(q)
    t0, b, k, y1 = oracle(q)
    k_wrong = int(q.parts[2].distractors[0])
    if k_wrong == 0 or k_wrong % (x1 - p):
        return None
    return {"a": str(t0), "b": str(b), "c": str(k_wrong), "d": str(c + k_wrong // (x1 - p))}


def test_follow_through_from_a_wrong_constant():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_k_replies, ["c"], ["d"], seeds=range(300))


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "991", "b": "992", "c": "993", "d": "994"})
