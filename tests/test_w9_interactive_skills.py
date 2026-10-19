"""Tests the six w9 interactive-graph skills end to end: wiring, every tier's key accepted and every distractor
rejected (through the real graders), exam chains, the figure canvas agreeing with the answer key, no answer leaked in
the figure, determinism, pool variety and generator speed."""

from __future__ import annotations

import statistics
import time

import pytest

import mathgen.generators.w9_line_draw_build  # noqa: F401  (registers the topics)
import mathgen.generators.w9_numberline_inequality_build  # noqa: F401
import mathgen.generators.w9_parabola_draw_build  # noqa: F401
import mathgen.generators.w9_plot_points_build  # noqa: F401
import mathgen.generators.w9_shade_region_build  # noqa: F401
import mathgen.generators.w9_transform_shape_build  # noqa: F401
from mathgen import figures
from mathgen.engine import grade
from mathgen.figure_kinds.canvas import TOOLS
from mathgen.misconceptions import CATALOGUE
from tests import w7b_support as sup

# skill id, topic, answer kind, exam part kinds, exam part marks
SKILLS = [
    (
        "plot-a-point-from-coordinates",
        "w9_plot_points",
        "points",
        ["points", "points", "points", "number"],
        [2, 1, 2, 2],
    ),
    ("line-draw-from-gradient-and-point", "w9_line_draw", "line", ["line", "number", "line", "number"], [2, 2, 2, 2]),
    (
        "inequality-on-number-line",
        "w9_numberline_inequality",
        "number_line_set",
        ["inequality", "number_line_set", "number_line_set", "number"],
        [2, 1, 2, 1],
    ),
    (
        "shade-linear-inequality-region",
        "w9_shade_region",
        "half_planes",
        ["number", "half_planes", "number", "half_planes"],
        [1, 2, 1, 3],
    ),
    (
        "transform-a-shape-on-grid",
        "w9_transform_shape",
        "polygon",
        ["polygon", "polygon", "polygon", "number"],
        [2, 2, 3, 2],
    ),
    (
        "parabola-from-vertex-and-point",
        "w9_parabola_draw",
        "curve",
        ["number", "number", "curve", "number"],
        [1, 1, 3, 1],
    ),
]
IDS = [s[0] for s in SKILLS]


def _case(request):
    return request.param


@pytest.fixture(params=SKILLS, ids=IDS)
def case(request):
    sid, topic, kind, kinds, marks = request.param
    return sup.skill_of(sid), topic, kind, kinds, marks


def test_wiring(case):
    skill, topic, kind, _, _ = case
    sup.check_wiring(skill, topic, "exd_x_" + topic, kind)
    assert skill.needs_figure and skill.capability_gates == ["figure_stimulus"]
    assert not skill.placement_eligible
    assert skill.misconceptions and all(m.id in CATALOGUE for m in skill.misconceptions)
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(case, tier):
    skill, topic, *_ = case
    sup.check_single_tier(skill, topic, tier, seeds=60)


def test_exam_chain(case):
    skill, topic, _, kinds, marks = case
    sup.check_exam_chain(skill, "exd_x_" + topic, marks, kinds, seeds=60)


def test_pools_and_shapes(case):
    _, topic, *_ = case
    sup.check_pools(topic)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_answers_vary(case, tier):
    _, topic, *_ = case
    assert len(sup.answer_spread(topic, tier, 60)) >= 12, tier


def _figures(q):
    return [q.figure] if q.figure else []


def test_figure_canvas_agrees_with_the_answer(case):
    """Each single question carries a figure whose canvas asks for the answer's own kind; the canvas ranges are the
    ranges the grader enforces; and the figure never shows the answer."""
    skill, topic, kind, *_ = case
    for tier in sup.TIERS:
        for seed in range(40):
            q = sup.single(topic, tier, seed)
            assert q.answer_kind == kind
            fig = q.figure
            assert fig, (tier, seed)
            assert not figures.validate(fig), (tier, seed, figures.validate(fig))
            cv = fig["canvas"]
            assert cv["reply"] == kind and all(TOOLS[t] == kind for t in cv["tools"])
            if isinstance(q.answer, dict):
                ans = q.answer
                if "x_range" in ans:
                    assert list(ans["x_range"]) == list(fig["x_range"]), (tier, seed)
                    assert list(ans["y_range"]) == list(fig["y_range"]), (tier, seed)
                if "range" in ans:
                    assert list(ans["range"]) == list(fig["range"]), (tier, seed)
                if "snap" in ans:
                    assert float(ans["snap"]) == float(cv["snap"]), (tier, seed)
            figures.assert_no_leak(fig, [q.answer_text])


def test_exam_figures_do_not_leak(case):
    _, topic, *_ = case
    for seed in range(40):
        q = sup.exam("exd_x_" + topic, seed)
        if q.figure:
            assert not figures.validate(q.figure), seed
            figures.assert_no_leak(q.figure, [p.answer_text for p in q.parts if p.answer_kind not in ("number",)])


def test_wrong_kind_replies_are_rejected(case):
    """A reply written for another answer kind never earns the mark."""
    skill, topic, kind, *_ = case
    other = {"points": "{}", "line": "[]", "x": "", "garbage": '{"pieces": 7}'}
    for seed in range(10):
        q = sup.single(topic, "medium", seed)
        for reply in other.values():
            assert not grade(skill, q, reply).correct


def test_generator_speed(case):
    _, topic, *_ = case
    for tier in (*sup.TIERS, "exam"):
        name = ("exd_x_" + topic) if tier == "exam" else topic
        times = []
        for seed in range(25):
            t0 = time.perf_counter()
            sup.exam(name, seed) if tier == "exam" else sup.single(name, tier, seed)
            times.append(time.perf_counter() - t0)
        assert statistics.median(times) < 0.25, (topic, tier)
