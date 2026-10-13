"""Exam chain of `senior-consider-complex-numbers-as-points-in-a-plane`: the fourth vertex, then the area."""

from __future__ import annotations

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
    skill,
)
from mathgen.engine import grade

TOPIC = "exd_x_r3f_complex_parallelogram"
SKILL_ID = "senior-consider-complex-numbers-as-points-in-a-plane"
_COMPLEX = re.compile(r"(-?\d+)(?: ([+-]) (\d*)i)?|(-?\d*)i")


def read_vertices(stem: str):
    """The three complex numbers in the stem, in the order A, B, C, as (x, y) integer pairs."""
    body = re.sub(r"\b[A-D]\b", "", stem)
    tokens = re.findall(r"(?<![\w^])(-?\d*i|-?\d+(?: [+-] \d*i)?)(?=[ ,.]|$)", body)
    points = []
    for token in tokens:
        m = re.fullmatch(r"(-?\d+)(?: ([+-]) (\d*)i)?", token)
        if m:
            re_part = int(m.group(1))
            if m.group(2):
                mag = int(m.group(3)) if m.group(3) else 1
                im_part = mag if m.group(2) == "+" else -mag
            else:
                im_part = 0
        else:
            raw = token[:-1]
            re_part, im_part = 0, (-1 if raw == "-" else int(raw) if raw else 1)
        points.append((re_part, im_part))
    return points


def oracle(q):
    a, b, c = read_vertices(q.stem)
    z = complex(*a) - complex(*b) + complex(*c)  # ABCD in order: D = A - B + C
    d = (int(z.real), int(z.imag))
    ab, ad = (b[0] - a[0], b[1] - a[1]), (d[0] - a[0], d[1] - a[1])
    return d, abs(ab[0] * ad[1] - ab[1] * ad[0])


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    for q in questions:
        assert len(read_vertices(q.stem)) == 3, q.stem
        (dx, dy), area = oracle(q)
        assert (int(q.parts[0].answer_text), int(q.parts[1].answer_text), int(q.parts[2].answer_text)) == (
            dx,
            dy,
            area,
        ), q.text
        assert dx != 0 and dy != 0 and area > 0


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    assert_answer_space(TOPIC)


def _area(a, b, d):
    return abs((b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]))


def _wrong_x_replies(q):
    a, b, _c = read_vertices(q.stem)
    (_dx, dy), _ = oracle(q)
    wrong_x = int(q.parts[0].distractors[0])
    return {"a": str(wrong_x), "b": str(dy), "c": str(_area(a, b, (wrong_x, dy)))}


def _wrong_both_replies(q):
    """A pair of wrong coordinates, taken from the offered distractors, whose area differs from the true one."""
    a, b, _c = read_vertices(q.stem)
    _d, area = oracle(q)
    for wx in q.parts[0].distractors:
        for wy in q.parts[1].distractors:
            wrong = (int(wx), int(wy))
            new_area = _area(a, b, wrong)
            if 0 not in wrong and new_area not in (
                0,
                area,
                *wrong,
            ):  # repeating an earlier reply is not following through
                return {"a": str(wrong[0]), "b": str(wrong[1]), "c": str(new_area)}
    return None


def test_follow_through_from_wrong_x():
    def usable(q):
        replies = _wrong_x_replies(q)
        nonzero = int(replies["a"]) != 0 and int(replies["c"]) not in (
            0,
            oracle(q)[1],
            int(replies["a"]),
            int(replies["b"]),
        )
        return replies if nonzero else None

    assert_follow_through(SKILL_ID, TOPIC, usable, ["a"], ["c"])


def test_follow_through_from_both_coordinates_wrong():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_both_replies, ["a", "b"], ["c"])


def test_area_is_not_given_for_the_triangle():
    """Half the parallelogram's area (the triangle ABC) is a distractor, never accepted."""
    sk = skill(SKILL_ID)
    for seed in range(60):
        q = exam_question(TOPIC, seed)
        (_d, area) = oracle(q)
        replies = {p.label: p.answer_text for p in q.parts}
        replies["c"] = str(Fraction(area, 2).numerator) if area % 2 == 0 else str(area + 1)
        assert grade(sk, q, replies).parts["c"].marks == 0


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "99", "b": "98", "c": "97"})
