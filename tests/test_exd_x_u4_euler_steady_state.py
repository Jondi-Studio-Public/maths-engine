"""Exam source of `explicit-euler-linear-ode`: two Euler steps for dC/dt = r - kC, then the gap to the steady state."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from mathgen.generators._u4_kit import survives_nsimplify
from tests.exd_x_u4_support import (
    check_answer_space,
    check_distractors_rejected,
    check_full_marks,
    check_no_text_defects,
    check_tagged,
    exam,
    frac,
    shapes,
    skill,
)

TOPIC = "exd_x_u4_euler_steady_state"
SKILL_ID = "explicit-euler-linear-ode"


def _setting(text: str) -> tuple[int, int, int, int]:
    r, k = (int(v) for v in re.search(r"dC/dt = (\d+) − (\d+)C", text).groups())
    c0 = int(re.search(r"Initially C = (\d+)", text).group(1))
    n = int(re.search(r"h = 1/(\d+)", text).group(1))
    return r, k, c0, n


def _oracle(text: str) -> dict[str, Fraction]:
    r, k, c0, n = _setting(text)
    value = Fraction(c0)
    estimates = []
    for _ in range(2):
        value = value + Fraction(1, n) * (r - k * value)
        estimates.append(value)
    steady = Fraction(r, k)  # dC/dt = 0
    return {"a": estimates[0], "b": estimates[1], "c": steady - estimates[1]}


def test_prompt_variety_and_new_shape():
    exam_shapes = shapes(TOPIC, 60, difficulty="exam")
    hard = shapes("explicit_euler_linear_ode", 60, difficulty="hard", steps=2)
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(TOPIC, seed)
        assert {p.label: frac(p.answer_text) for p in q.parts} == _oracle(q.text), (seed, q.text)


def test_structure():
    for seed in range(60):
        q = exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors_rejected(SKILL_ID, TOPIC)
    check_tagged(TOPIC)


def test_follow_through_from_wrong_first_step():
    sk = skill(SKILL_ID)
    tested = 0
    for seed in range(100):
        q = exam(TOPIC, seed)
        r, k, _, n = _setting(q.text)
        a = frac(q.parts[0].answer_text) + Fraction(1, 3)  # a wrong, positive C1
        b = a + Fraction(1, n) * (r - k * a)
        c = Fraction(r, k) - b
        if len({a, b, c}) < 3 or c <= 0 or not survives_nsimplify(a, b, c):
            continue
        tested += 1
        result = grade(sk, q, {"a": str(a), "b": str(b), "c": str(c)})
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks
            assert result.parts[label].follow_through is True
        assert not result.correct
    assert tested >= 40


def test_all_wrong_earns_zero():
    sk = skill(SKILL_ID)
    for seed in range(40):
        assert grade(sk, exam(TOPIC, seed), {"a": "999", "b": "998", "c": "997"}).marks == 0


def test_deterministic():
    assert exam(TOPIC, 7).to_json() == exam(TOPIC, 7).to_json()
