"""Tests the position-time-graph-velocity skill: keys re-derived from the drawn segments, figure, grading, exam."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import figures
from mathgen.engine import grade
from tests import sr2b_support as sup

SKILL = "position-time-graph-velocity"
TOPIC = "sr2b_position_time"
EXAM = "sr2b_exam_position_time"


def _knots(q):
    segs = q.figure["segments"]
    assert all(a["to"] == b["from"] for a, b in zip(segs, segs[1:])), "the segments are not joined"
    return [tuple(segs[0]["from"])] + [tuple(s["to"]) for s in segs]


def _position(knots, t):
    for (t0, p0), (t1, p1) in zip(knots, knots[1:]):
        if t0 <= t <= t1:
            return Fraction(p0) + Fraction(p1 - p0, t1 - t0) * (t - t0)
    raise AssertionError(t)


def _distance(knots, lo, hi):
    cuts = sorted({lo, hi} | {t for t, _ in knots if lo < t < hi})
    return sum(abs(_position(knots, b) - _position(knots, a)) for a, b in zip(cuts, cuts[1:]))


def _first_return(knots):
    start = knots[0][1]
    for (t0, p0), (t1, p1) in zip(knots, knots[1:]):
        if t0 == knots[0][0]:
            continue
        if p0 != start and (p0 - start) * (p1 - start) <= 0 and p1 != p0:
            return Fraction(t0) + Fraction(start - p0, p1 - p0) * (t1 - t0)
        if p1 == start and p0 != p1:
            return Fraction(t1)
    return None


def _window(stem, knots):
    found = re.search(r"from (?:time )?(\d+) to (?:time )?(\d+)", stem)
    return (int(found.group(1)), int(found.group(2))) if found else (knots[0][0], knots[-1][0])


def _derive(q):
    knots = _knots(q)
    lo, hi = _window(q.text, knots)
    if "average velocity" in q.text:
        return (_position(knots, hi) - _position(knots, lo)) / (hi - lo)
    if "average speed" in q.text:
        return _distance(knots, lo, hi) / (hi - lo)
    assert "starting position" in q.text or "get back" in q.text, q.text
    return _first_return(knots)


def _near(text, value):
    return abs(Fraction(text) - value) <= Fraction(1, 20) or abs(Fraction(text) - value) <= abs(value) / 100


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "number",
        ["senior-interpret-distance-versus-time-graphs", "rate-of-change"],
    )
    data = sup.skill_yaml(SKILL)
    assert data["capability_gates"] == ["figure_stimulus"]
    assert data["needs_figure"] is True


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_the_drawn_graph(tier):
    for seed in range(150):
        q = sup.tier_question(TOPIC, tier, seed)
        assert _near(q.answer_text, _derive(q)), (seed, q.text, q.answer_text, _derive(q))
        assert q.figure["kind"] == "cartesian"


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8)


def test_figure_does_not_show_an_answer():
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.tier_question(TOPIC, tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        figures.assert_no_leak(q.figure, [x for p in q.parts for x in (p.answer_text, *p.distractors)])


def test_graph_has_a_negative_position_and_a_doubling_back():
    seen_negative = seen_back = 0
    for seed in range(60):
        knots = _knots(sup.tier_question(TOPIC, "medium", seed))
        ps = [p for _, p in knots]
        seen_negative += min(ps) < 0
        seen_back += _distance(knots, knots[0][0], knots[-1][0]) > abs(ps[-1] - ps[0])
    assert seen_negative >= 5 and seen_back >= 40


def test_tiers_differ():
    asked = {t: " ".join(sup.tier_question(TOPIC, t, s).text for s in range(60)) for t in sup.TIERS}
    assert "average velocity" in asked["easy"] and "starting position" not in asked["easy"]
    assert "average speed" in asked["medium"]
    assert "first" in asked["hard"]
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number"] * 4, [1, 2, 2, 2], [False, False, False, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.1)


def test_exam_keys_match_the_drawn_graph():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        knots = _knots(q)
        (t0, p0), (t1, p1) = knots[-2], knots[-1]
        total = knots[-1][0] - knots[0][0]
        dist = _distance(knots, knots[0][0], knots[-1][0])
        keys = [Fraction(p1 - p0, t1 - t0), _first_return(knots), dist, dist / total]
        for p, key in zip(q.parts, keys):
            assert _near(p.answer_text, key), (seed, p.label, p.answer_text, key)


def test_exam_average_speed_follows_through_from_a_wrong_distance():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        total = int(re.search(r"t = (\d+) ", q.text.split("lasts from")[1].split("to")[1]).group(1))
        wrong_distance = int(canon["c"]) + total
        carried = Fraction(wrong_distance, total)
        shown = f"{float(carried):g}"
        replies = {**canon, "c": str(wrong_distance), "d": shown}
        assert grade(sk, q, replies).parts["d"].marks == q.parts[3].marks, (seed, shown)
