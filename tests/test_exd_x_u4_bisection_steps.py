"""Exam source of `bisection-interval-update`: f at a midpoint, the bracket after two steps, and the steps needed."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen.engine import grade
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

TOPIC = "exd_x_u4_bisection_steps"
SKILL_ID = "bisection-interval-update"
_POWERS = {"²": 2, "³": 3}


def _setting(text: str):
    increasing = re.search(r"f\(x\) = x([²³]) − (\d+)", text)
    if increasing:
        power, c, sign = _POWERS[increasing.group(1)], int(increasing.group(2)), 1
    else:
        reversed_ = re.search(r"f\(x\) = (\d+) − x([²³])", text)
        power, c, sign = _POWERS[reversed_.group(2)], int(reversed_.group(1)), -1
    left, right = (int(v) for v in re.search(r"bracket \[(\d+), (\d+)\]", text).groups())
    tol = Fraction(re.search(r"width less than ([\d.]+)\?", text).group(1))
    return (lambda x: sign * (x**power - c)), left, right, tol


def _oracle(text: str):
    f, left, right, tol = _setting(text)
    lo, hi = Fraction(left), Fraction(right)
    assert f(lo) * f(hi) < 0
    value = f((lo + hi) / 2)
    for _ in range(2):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:  # the sign change is in the left half
            hi = mid
        else:
            lo = mid
    n = math.ceil(math.log2((right - left) / tol))
    assert Fraction(right - left, 2**n) < tol <= Fraction(right - left, 2 ** (n - 1))
    return value, (lo, hi), n


def test_prompt_variety_and_new_shape():
    exam_shapes = shapes(TOPIC, 60, difficulty="exam")
    hard = shapes("bisection_interval_update", 60, difficulty="hard", reversed_sign=True)
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(TOPIC, seed)
        value, (lo, hi), n = _oracle(q.text)
        a, b, c = q.parts
        assert frac(a.answer_text) == value, (seed, q.text)
        assert (Fraction(b.answer["left"]), Fraction(b.answer["right"])) == (lo, hi), (seed, q.text)
        assert int(c.answer_text) == n, (seed, q.text)


def test_both_function_senses_and_powers_appear():
    texts = [exam(TOPIC, s).text for s in range(60)]
    assert any("x² −" in t for t in texts) and any("x³ −" in t for t in texts)
    assert any(re.search(r"f\(x\) = \d+ − x", t) for t in texts)


def test_structure():
    for seed in range(60):
        q = exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.answer_kind for p in q.parts] == ["fraction", "pair", "number"]
        assert all(p.follow_through is None for p in q.parts)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors_rejected(SKILL_ID, TOPIC)
    check_tagged(TOPIC)


def test_partial_marks_for_the_parts_that_are_right():
    sk = skill(SKILL_ID)
    for seed in range(60):
        q = exam(TOPIC, seed)
        reply = {"a": q.parts[0].answer_text, "b": q.parts[1].distractors[0], "c": q.parts[2].answer_text}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 1 and result.parts["b"].marks == 0 and result.parts["c"].marks == 2
        assert result.marks == 3 and not result.correct


def test_all_wrong_earns_zero():
    sk = skill(SKILL_ID)
    for seed in range(40):
        reply = {"a": "99999", "b": "left = 99, right = 100", "c": "77"}
        assert grade(sk, exam(TOPIC, seed), reply).marks == 0


def test_deterministic():
    assert exam(TOPIC, 7).to_json() == exam(TOPIC, 7).to_json()
