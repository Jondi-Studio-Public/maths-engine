"""Exam chain for `senior-identify-contexts-suitable-for-modelling-by-trigonometric-functions`: value, first time at a
threshold, time above it per cycle, and its share of the cycle."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "senior-identify-contexts-suitable-for-modelling-by-trigonometric-functions"
TOPIC = "exd_x_r2j_sinusoid_context_chain"
SEEDS = range(300)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def _half_up(x: float, places: int) -> str:
    scaled = Fraction(repr(x)) * 10**places
    return f"{(scaled + Fraction(1, 2)) // 1 / 10**places:.{places}f}"


def read(q):
    stem = q.stem
    a = float(re.search(r"amplitude (?:is )?(\d+) ", stem).group(1))
    d = float(re.search(r"(?:mean of|mean value is|mean) (\d+) ", stem).group(1))
    period = float(
        re.search(r"period of (\d+) ", stem).group(1)
        if "period of" in stem
        else re.search(r"period (\d+) ", stem).group(1)
    )
    t1 = float(re.search(r"after ([\d.]+) ", q.parts[0].prompt).group(1))
    h = float(re.search(r"reaches ([\d.]+) ", q.parts[1].prompt).group(1))
    return a, d, period, t1, h


def oracle(q) -> dict[str, str]:
    """Start at the minimum: bisect d - a cos(2 pi t / T) = h on [0, T/2], where the height is increasing."""
    a, d, period, t1, h = read(q)

    def height(t):
        return d - a * math.cos(2 * math.pi * t / period)

    lo, hi = 0.0, period / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if height(mid) < h:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    above = period - 2 * t
    # the symmetric second crossing, found independently from the other side of the cycle
    lo, hi = period / 2, period
    for _ in range(200):
        mid = (lo + hi) / 2
        if height(mid) > h:
            lo = mid
        else:
            hi = mid
    assert abs(((lo + hi) / 2 - t) - above) < 1e-9
    return {
        "a": _half_up(height(t1), 1),
        "b": _half_up(t, 3),
        "c": _half_up(above, 2),
        "d": _half_up(100 * above / period, 0).split(".")[0],
    }


def test_wiring_shapes_defects_determinism():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == kit.canon(q), q.text


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_grading():
    kit.check_grading(SKILL)


def test_follow_through_from_wrong_first_time(questions):
    checked = 0
    for q in questions[:100]:
        a, d, period, t1, h = read(q)
        c = kit.canon(q)
        t_wrong = float(c["b"]) + 0.2
        above_wrong = period - 2 * t_wrong
        if not 0 < above_wrong < period:
            continue
        reply = {
            "b": f"{t_wrong:.3f}",
            "c": _half_up(above_wrong, 2),
            "d": f"{(Fraction(_half_up(above_wrong, 2)) / int(period) * 100 + Fraction(1, 2)) // 1}",
        }
        result = grade(kit.skill(SKILL), q, {**c, **reply})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["d"].marks == q.parts[3].marks, (
            q.text,
            reply,
        )
        checked += 1
    assert checked >= 60
