"""Tests `f10-scale-drawing-floor-plans`: topic `fsp_scale_plan` (easy, medium, hard) and the exam chain `fsp_scale_plan_exam`.

The oracle reads the marked plan lengths off the figure's own labels and the scale off the question text, then works
the real size out in exact fractions (a second decomposition for the L-shape).
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.fsp_scale_plans_build as sp
from mathgen import figures, make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILL_ID = "f10-scale-drawing-floor-plans"
TIERS = ("easy", "medium", "hard")
SEEDS = range(80)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make("fsp_scale_plan", seed=seed, difficulty=tier)


def _exam(seed):
    return make("fsp_scale_plan_exam", seed=seed, difficulty="exam")


def _scale(text):
    m = re.search(r"1 : (\d+)", text) or re.search(r"stands for (\d+) cm", text)
    return int(m.group(1))


def _labels(fig):
    """{(from, to): plan length in cm} for every marked side."""
    out = {}
    for s in fig["segments"]:
        if s.get("label"):
            out[(s["from"], s["to"])] = out[(s["to"], s["from"])] = Fraction(s["label"].split()[0])
    return out


def _fmt(value, dp):
    return f"{float(value):.{dp}f}"


def test_skill_wiring():
    assert _SKILL.answer == "number" and _SKILL.status == "active"
    assert _SKILL.needs_figure is True and _SKILL.capability_gates == ["figure_stimulus"]
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        ("fsp_scale_plan", "easy"),
        ("fsp_scale_plan", "medium"),
        ("fsp_scale_plan", "hard"),
        ("fsp_scale_plan_exam", "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert {m.id for m in _SKILL.misconceptions} <= set(CATALOGUE)


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_a_to_scale_diagram_and_the_reader_agrees(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.figure["kind"] == "diagram" and not q.figure["not_to_scale"]
        assert not figures.validate(q.figure)
        assert float(figures.get_reader("fsp_scale_plan")(q)) == pytest.approx(float(q.answer_text)), (tier, seed)
        assert not find_text_defects(q, include_working=True)


def test_easy_key_is_plan_length_times_scale():
    walls = Counter()
    for seed in SEEDS:
        q = _q("easy", seed)
        k = _scale(q.text)
        wall = re.search(r"wall ([A-F])([A-F])", q.text)
        marked = _labels(q.figure)
        plan = marked[(wall.group(1), wall.group(2))]
        assert Fraction(q.answer_text) == plan * k / 100, (seed, q.text)
        walls[wall.group(0)] += 1
    assert len(walls) >= 3


def test_medium_key_scales_both_lengths():
    for seed in SEEDS:
        q = _q("medium", seed)
        k = _scale(q.text)
        room = re.search(r"room ([A-F]{4})", q.text).group(1)
        marked = _labels(q.figure)
        width = marked.get(("A", "B")) if room in ("ABCD", "ABEF") else marked[("B", "C")]
        height = marked.get(("B", "C")) if room == "ABCD" else marked[("A", "F")]
        assert Fraction(q.answer_text) == (width * k / 100) * (height * k / 100), (seed, q.text)
        # the plan area times the ratio once is the commonest error and is never the key
        assert Fraction(q.answer_text) != width * height * k / 10000


def test_hard_l_shape_area_and_perimeter():
    asked = Counter()
    for seed in range(160):
        q = _q("hard", seed)
        k = _scale(q.text)
        m = _labels(q.figure)
        a, b = m[("A", "B")], m[("B", "C")]
        c, d = m[("E", "F")], m[("F", "A")]
        f = Fraction(k, 100)
        if "floor area" in q.text:
            asked["area"] += 1
            # the L is a bottom rectangle a by b plus an upper arm c wide up to d high
            assert Fraction(q.answer_text) == (a * b + c * (d - b)) * f * f, (seed, q.text)
        else:
            asked["perimeter"] += 1
            sides = [a, b, a - c, d - b, c, d]
            assert Fraction(q.answer_text) == sum(sides) * f, (seed, q.text)
        # two sides are unmarked
        assert len(m) == 8  # four sides, each stored both ways
    assert min(asked.values()) >= 40


def test_tiers_differ_in_shape():
    easy = [_q("easy", s) for s in SEEDS]
    medium = [_q("medium", s) for s in SEEDS]
    hard = [_q("hard", s) for s in SEEDS]
    assert all("wall" in q.text and "area" not in q.text for q in easy)
    assert all("area" in q.text and "L-shaped" not in q.text for q in medium)
    assert all("L-shaped" in q.text for q in hard)
    assert len({len(q.figure["points"]) for q in easy}) >= 2
    assert {len(q.figure["points"]) for q in hard} == {6}


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed)
        assert len(q.distractors) == 3
        dp = len(q.answer_text.partition(".")[2])
        for text in q.distractors:
            assert len(text.partition(".")[2]) == dp, (tier, seed, text)
            result = grade(_SKILL, q, text)
            assert not result.correct, (tier, seed, text)
            assert result.misconception in CATALOGUE


@pytest.mark.parametrize("tier", TIERS)
def test_prompts_vary(tier):
    assert len({_q(tier, s).text for s in range(60)}) >= 20


def test_figure_does_not_leak_the_answer():
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text])


# -- the exam chain -------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through for p in q.parts] == [None, None, "part_a*part_b", "part_c/0.25"]


def test_exam_answers_recomputed_from_the_plan():
    scales = Counter()
    for seed in range(120):
        q = _exam(seed)
        k = _scale(q.stem)
        scales[k] += 1
        m = _labels(q.figure)
        long_m, short_m = m[("A", "B")] * k / 100, m[("B", "C")] * k / 100
        area = long_m * short_m
        assert [Fraction(p.answer_text) for p in q.parts] == [long_m, short_m, area, area * 4], (seed, q.text)
        assert long_m > short_m
    assert set(scales) == {50, 100, 200}


def test_exam_full_marks_and_distractors_rejected():
    for seed in range(60):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert 2 <= len(p.distractors) <= 3
            dp = len(p.answer_text.partition(".")[2])
            for d in p.distractors:
                assert len(d.partition(".")[2]) == dp
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_exam_follow_through_from_a_wrong_wall():
    for seed in range(60):
        q = _exam(seed)
        short = Fraction(q.parts[1].answer_text)
        wrong = Fraction(q.parts[0].distractors[0])
        canon = {p.label: p.answer_text for p in q.parts}
        area = wrong * short
        dp = len(canon["c"].partition(".")[2])
        reply = {**canon, "a": q.parts[0].distractors[0], "c": _fmt(area, dp), "d": str(int(area * 4))}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        if area * 4 == int(area * 4) and result.parts["c"].marks:
            assert result.parts["c"].follow_through


def test_exam_prompts_vary():
    assert len({_exam(s).text for s in range(60)}) >= 20
