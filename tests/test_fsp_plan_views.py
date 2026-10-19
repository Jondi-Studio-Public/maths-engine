"""Tests `f10-plan-views-cube-stacks`: topic `fsp_plan_views` (easy, medium, hard) and the exam chain `fsp_plan_views_exam`.

The oracle rebuilds the grid of stacks from the labels drawn on the plan, then recomputes the total and both elevations,
the elevations a second way (layer by layer).
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.fsp_plans_build as plans
from mathgen import figures, make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILL_ID = "f10-plan-views-cube-stacks"
TIERS = ("easy", "medium", "hard")
SEEDS = range(80)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make("fsp_plan_views", seed=seed, difficulty=tier)


def _exam(seed):
    return make("fsp_plan_views_exam", seed=seed, difficulty="exam")


def test_skill_wiring():
    assert _SKILL.answer == "number" and _SKILL.status == "active"
    assert _SKILL.needs_figure is True and _SKILL.capability_gates == ["figure_stimulus"]
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        ("fsp_plan_views", "easy"),
        ("fsp_plan_views", "medium"),
        ("fsp_plan_views", "hard"),
        ("fsp_plan_views_exam", "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert {m.id for m in _SKILL.misconceptions} <= set(CATALOGUE)


def test_elevation_counts_agree_two_ways():
    grid = [[1, 3, 0], [2, 0, 4], [0, 2, 2]]
    assert plans.front_squares(grid) == plans._layer_view_squares(grid, "front") == 2 + 3 + 4
    assert plans.right_squares(grid) == plans._layer_view_squares(grid, "right") == 3 + 4 + 2


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_a_plan_grid_and_the_reader_agrees(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        grid = plans.read_grid(q.figure)
        assert q.figure["kind"] == "cartesian" and not figures.validate(q.figure)
        assert len(q.figure["points"]) == len(grid) * len(grid[0])
        assert figures.get_reader("fsp_plan_views")(q) == q.answer_text, (tier, seed, q.text)
        assert not find_text_defects(q, include_working=True)


def test_easy_key_is_the_total():
    for seed in SEEDS:
        q = _q("easy", seed)
        grid = plans.read_grid(q.figure)
        assert int(q.answer_text) == sum(map(sum, grid))


def test_medium_key_is_the_elevation_not_the_total():
    views = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        grid = plans.read_grid(q.figure)
        view = "front" if "front elevation" in q.text else "right"
        views[view] += 1
        assert int(q.answer_text) == plans._layer_view_squares(grid, view)
        assert int(q.answer_text) < sum(map(sum, grid))
        assert str(sum(map(sum, grid))) in q.distractors
    assert min(views.values()) >= 60


def test_hard_unknown_stack_is_fixed_by_the_elevation():
    for seed in SEEDS:
        q = _q("hard", seed)
        grid = plans.read_grid(q.figure)
        shown = int(re.search(r"(\d+) squares", q.text).group(1))
        view = "right" if "right-side elevation" in q.text else "front"
        unknown = [(y, x) for y, row in enumerate(grid) for x, v in enumerate(row) if v is None]
        assert len(unknown) == 1
        y, x = unknown[0]
        key = int(q.answer_text)
        grid[y][x] = key
        assert plans._layer_view_squares(grid, view) == shown
        # no other number would give that elevation
        for other in range(0, 12):
            if other != key:
                grid[y][x] = other
                assert plans._layer_view_squares(grid, view) != shown, (seed, other)
        assert key not in [v for row in plans.read_grid(q.figure) for v in row if v is not None]


def test_tiers_differ_in_shape():
    assert all("?" not in [p["label"] for p in _q(t, s).figure["points"]] for t in ("easy", "medium") for s in SEEDS)
    assert all("?" in [p["label"] for p in _q("hard", s).figure["points"]] for s in SEEDS)
    assert all("elevation" not in _q("easy", s).text for s in SEEDS)
    assert all("elevation" in _q("medium", s).text for s in SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed)
        assert len(q.distractors) == 3
        for text in q.distractors:
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
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]


def test_exam_answers_recomputed_from_the_plan():
    for seed in range(120):
        q = _exam(seed)
        grid = plans.read_grid(q.figure)
        edge = int(re.search(r"(\d+) cm", q.text).group(1))
        total = sum(map(sum, grid))
        assert [int(p.answer_text) for p in q.parts] == [
            total,
            plans._layer_view_squares(grid, "front"),
            plans._layer_view_squares(grid, "right"),
            edge**3 * total,
        ], (seed, q.text)


def test_exam_full_marks_and_distractors_rejected():
    for seed in range(60):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert 2 <= len(p.distractors) <= 3
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_exam_follow_through_from_a_wrong_total():
    for seed in range(60):
        q = _exam(seed)
        edge = int(re.search(r"(\d+) cm", q.text).group(1))
        wrong = int(q.parts[0].distractors[0])
        canon = {p.label: p.answer_text for p in q.parts}
        reply = {**canon, "a": str(wrong), "d": str(edge**3 * wrong)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["d"].marks == q.parts[3].marks and result.parts["d"].follow_through


def test_exam_prompts_vary():
    assert len({_exam(s).text for s in range(60)}) >= 20
