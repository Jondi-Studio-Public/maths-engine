"""Exam chain for `senior-use-trigonometric-functions-and-their-derivatives-to-solve`: maximum rate, first time at a
stated rate, value then, and the rate of the rate."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest

from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "senior-use-trigonometric-functions-and-their-derivatives-to-solve"
TOPIC = "exd_x_r2j_trig_rate_chain"
SEEDS = range(300)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def _dp(x: float, places: int) -> str:
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def read(q):
    a, b, c = re.search(r"h\(t\) = (\d+)sin\((\d*)t\) \+ (\d+)", q.stem).groups()
    k = int(re.search(r"rate of change of h is (-?\d+) ", q.parts[1].prompt).group(1))
    return int(a), int(b or 1), int(c), k


def oracle(q) -> dict[str, str]:
    """Bisect h'(t) = k on [0, pi/b] (h' is decreasing there), then evaluate h and h'' at the root."""
    a, b, c, k = read(q)
    lo, hi = 0.0, math.pi / b
    for _ in range(200):
        mid = (lo + hi) / 2
        if a * b * math.cos(b * mid) > k:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    return {
        "a": str(a * b),
        "b": _dp(t, 3),
        "c": _dp(a * math.sin(b * t) + c, 2),
        "d": _dp(-a * b * b * math.sin(b * t), 1),
    }


def test_wiring_shapes_defects_determinism():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == kit.canon(q), q.text


def test_grading():
    kit.check_grading(SKILL)


def test_follow_through_from_wrong_time(questions):
    checked = 0
    for q in questions[:120]:
        a, b, c, k = read(q)
        t_wrong = float(kit.canon(q)["b"]) + 0.137
        h_wrong = _dp(a * math.sin(b * t_wrong) + c, 2)
        acc_wrong = _dp(-a * b * b * math.sin(b * t_wrong), 1)
        reply = {**kit.canon(q), "b": f"{t_wrong:.3f}", "c": h_wrong, "d": acc_wrong}
        if h_wrong == kit.canon(q)["c"] or acc_wrong == kit.canon(q)["d"]:
            continue
        result = grade(kit.skill(SKILL), q, reply)
        assert result.parts["b"].marks == 0
        for label in "cd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, reply)
            assert result.parts[label].follow_through is True
        checked += 1
    assert checked >= 60
