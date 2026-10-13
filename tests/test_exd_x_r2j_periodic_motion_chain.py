"""Exam chain for `senior-model-periodic-motion-using-sine-and-cosine-functions`: amplitude and period from a
description of rotating motion, then the height at a time and the first time at a height."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "senior-model-periodic-motion-using-sine-and-cosine-functions"
TOPIC = "exd_x_r2j_periodic_motion_chain"
SEEDS = range(300)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def _half_up(x: float, places: int) -> str:
    scaled = Fraction(repr(x)) * 10**places
    return f"{(scaled + Fraction(1, 2)) // 1 / 10**places:.{places}f}"


def read(q):
    stem = q.stem
    high = float(re.search(r"(?:highest point is|rising to) ([\d.]+) m", stem).group(1))
    low = float(re.search(r"(?:lowest point is|falling to) ([\d.]+) m", stem).group(1))
    n = int(re.search(r"(?:completes|makes) (\d+) revolutions", stem).group(1))
    m = int(re.search(r"revolutions in (\d+) ", stem).group(1))
    t0 = float(re.search(r"after ([\d.]+) ", q.parts[2].prompt).group(1))
    h1 = float(re.search(r"reaches ([\d.]+) m", q.parts[3].prompt).group(1))
    return high, low, n, m, t0, h1


def oracle(q) -> dict[str, str]:
    """Fit h(t) = D - A cos(2 pi t / T) to the description, then bisect for the first time at the stated height."""
    high, low, n, m, t0, h1 = read(q)
    amp, mean, period = (high - low) / 2, (high + low) / 2, m / n

    def height(t):
        return mean - amp * math.cos(2 * math.pi * t / period)

    assert abs(height(0) - low) < 1e-9 and abs(height(period / 2) - high) < 1e-9
    lo, hi = 0.0, period / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if height(mid) < h1:
            lo = mid
        else:
            hi = mid
    amp_text = f"{amp:g}"
    return {"a": amp_text, "b": _half_up(period, 2), "c": _half_up(height(t0), 1), "d": _half_up((lo + hi) / 2, 2)}


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


def test_follow_through_from_wrong_amplitude_and_period(questions):
    checked = 0
    for q in questions[:100]:
        high, low, n, m, t0, h1 = read(q)
        a_wrong = high - low  # the range, not the amplitude
        t_wrong = m / n + 1
        c_wrong = low + a_wrong - a_wrong * math.cos(2 * math.pi * t0 / t_wrong)
        r = (low + a_wrong - h1) / a_wrong
        if abs(r) >= 1 or c_wrong <= 0:
            continue
        d_wrong = t_wrong * math.acos(r) / (2 * math.pi)
        c = kit.canon(q)
        reply = {"a": f"{a_wrong:g}", "b": f"{t_wrong:.2f}", "c": _half_up(c_wrong, 1), "d": _half_up(d_wrong, 2)}
        result = grade(kit.skill(SKILL), q, {**c, **reply})
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["d"].marks == q.parts[3].marks, (
            q.text,
            reply,
        )
        checked += 1
    assert checked >= 60
