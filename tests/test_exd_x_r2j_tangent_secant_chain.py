"""Exam chain for `senior-when-a-secant-meeting-the-circle-at-a`: AM, the chord, its distance from the centre, an angle."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest

from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "senior-when-a-secant-meeting-the-circle-at-a"
TOPIC = "exd_x_r2j_tangent_secant_chain"
SEEDS = range(300)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def read(q):
    r = int(re.search(r"radius (\d+) cm", q.stem).group(1))
    t = int(re.search(r"TM (?:is|=) (\d+) cm", q.stem).group(1))
    m = int(re.search(r"BM (?:is|=) (\d+) cm", q.stem).group(1))
    return r, t, m


def oracle(q) -> dict[str, str]:
    """Rebuild the figure in coordinates: O at the origin, M on the x-axis, and check the secant by intersection."""
    r, t, m = read(q)
    om = math.hypot(r, t)  # TM is a tangent length
    am = t * t / m
    ab = am - m
    half = ab / 2
    h = math.sqrt(r * r - half * half)
    # the secant leaves M at angle phi to MO; the chord's midpoint is (m + half) along it, h from O
    phi = math.atan2(h, m + half)
    # independent check: the point (m + half) along the secant is at distance h from O, and B is on the circle
    mx = om
    bx, by = mx - m * math.cos(phi), m * math.sin(phi)
    assert abs(math.hypot(bx, by) - r) < 1e-6
    ax, ay = mx - am * math.cos(phi), am * math.sin(phi)
    assert abs(math.hypot(ax, ay) - r) < 1e-6
    deg = Decimal(repr(math.degrees(phi))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return {"a": f"{am:.0f}", "b": f"{ab:.0f}", "c": f"{h:.0f}", "d": str(deg)}


def test_wiring_shapes_defects_determinism():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == kit.canon(q), q.text


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_grading():
    kit.check_grading(SKILL)


def test_follow_through_from_wrong_secant(questions):
    for q in questions[:100]:
        r, t, m = read(q)
        c = kit.canon(q)
        am_wrong = int(c["a"]) + 4
        ab_wrong = am_wrong - m
        if ab_wrong / 2 >= r or ab_wrong <= 0:
            continue
        h_wrong = math.sqrt(r * r - (ab_wrong / 2) ** 2)
        reply = {
            "a": str(am_wrong),
            "b": str(ab_wrong),
            "c": f"{h_wrong:.2f}" if h_wrong % 1 else str(int(h_wrong)),
            "d": f"{math.degrees(math.atan2(h_wrong, m + ab_wrong / 2)):.1f}",
        }
        result = grade(kit.skill(SKILL), q, reply)
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == q.parts[1].marks, (q.text, reply)
