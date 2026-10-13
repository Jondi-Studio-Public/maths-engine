"""Exam chain of `senior-the-diagonals-of-a-parallelogram-meet-at-right`: x, the side, the other diagonal, the area."""

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

TOPIC = "exd_x_r3f_rhombus_chain"
SKILL_ID = "senior-the-diagonals-of-a-parallelogram-meet-at-right"


def _side_expression(text: str):
    m = re.fullmatch(r"(\d+)x ([+-]) (\d+)", text)
    return int(m.group(1)), (int(m.group(3)) if m.group(2) == "+" else -int(m.group(3)))


def read_question(q):
    ab, bc = re.findall(r"=? ?\((\d+x [+-] \d+)\) cm", q.stem)[:2]
    ac = int(re.findall(r"(\d+) cm", q.stem)[-1])
    return _side_expression(ab), _side_expression(bc), ac


def oracle(q):
    (a1, b1), (c1, d1), ac = read_question(q)
    x, rem = divmod(d1 - b1, a1 - c1)
    assert rem == 0
    side = a1 * x + b1
    assert side == c1 * x + d1 > 0
    half_bd_squared = side**2 - (ac / 2) ** 2
    bd = 2 * math.sqrt(half_bd_squared)
    area = ac * bd / 2
    return x, side, round(bd, 1), round(area, 1)


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    for q in questions:
        x, side, bd, area = oracle(q)
        got = [p.answer_text for p in q.parts]
        assert got == [str(x), str(side), f"{bd:.1f}", f"{area:.1f}"], q.text


def test_decimal_parts_carry_one_place_and_integers_none(questions):
    for q in questions[:100]:
        assert "." not in q.parts[0].answer_text and "." not in q.parts[1].answer_text
        for part in q.parts[2:]:
            assert re.fullmatch(r"\d+\.\d", part.answer_text)
            assert all(re.fullmatch(r"\d+\.\d", d) for d in part.distractors)


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC, min_options=2)


def test_answer_space():
    assert_answer_space(TOPIC)


def _from_side(side, ac):
    """Replies for (c) and (d) worked out correctly from a learner's own side length, or None if it is too short."""
    if side <= ac / 2:
        return None
    bd = 2 * math.sqrt(side**2 - (ac / 2) ** 2)
    area = ac * float(f"{bd:.1f}") / 2
    return {"c": f"{bd:.1f}", "d": f"{area:.1f}"}


def _wrong_x_replies(q):
    (a1, b1), _second, ac = read_question(q)
    for distractor in q.parts[0].distractors:
        x_wrong = int(distractor)
        side = a1 * x_wrong + b1
        rest = _from_side(side, ac)
        if rest and x_wrong != 0 and side > 0:
            return {"a": str(x_wrong), "b": str(side), **rest}
    return None


def test_follow_through_from_a_wrong_x():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_x_replies, ["a"], ["b"], seeds=range(300))


def test_follow_through_into_the_diagonal_and_area_from_a_wrong_side():
    def replies(q):
        _first, _second, ac = read_question(q)
        x, *_ = oracle(q)
        for distractor in q.parts[1].distractors:
            rest = _from_side(int(distractor), ac)
            if rest:
                return {"a": str(x), "b": distractor, **rest}
        return None

    assert_follow_through(SKILL_ID, TOPIC, replies, ["b"], ["c", "d"], seeds=range(300))


def test_integer_side_gives_whole_diagonal_and_area(questions):
    for q in questions:
        bd, area = Fraction(q.parts[2].answer_text), Fraction(q.parts[3].answer_text)
        assert bd.denominator == 1 and area.denominator == 1


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "991", "b": "992", "c": "993.3", "d": "994.4"})
