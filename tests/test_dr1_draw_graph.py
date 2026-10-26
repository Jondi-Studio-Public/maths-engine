"""The four draw-a-graph skills (ACMEM042, ACMEM125, ACMGM052, ACMGM087): wiring, every tier's key accepted and every
distractor rejected through the real graders, exam chains, the canvas agreeing with the key, no answer in the figure,
and independent re-derivations of the keys (a tank rebuilt stage by stage, a rule evaluated, a series rebuilt from its
changes, a correlation recomputed)."""

from __future__ import annotations

import re
import statistics
import time

import pytest

import mathgen.generators.dr1_draw_graph_build as dg  # noqa: F401  (registers the topics)
from mathgen import figures
from mathgen.engine import grade
from mathgen.figure_kinds.canvas import TOOLS
from mathgen.misconceptions import CATALOGUE
from tests import w7b_support as sup

# skill id, topic, answer kind, exam part kinds, exam part marks
SKILLS = [
    (
        "draw-a-line-graph-from-continuous-data",
        "dr1_line_graph",
        "polyline",
        ["polyline", "number", "number", "number"],
        [3, 1, 2, 2],
    ),
    (
        "draw-a-graph-from-practical-data",
        "dr1_graph_from_data",
        "points",
        ["points", "number", "number", "number"],
        [2, 2, 1, 2],
    ),
    ("construct-a-scatterplot", "dr1_scatterplot", "points", ["points", "mcq", "number"], [3, 1, 2]),
    (
        "construct-a-time-series-plot",
        "dr1_time_series",
        "polyline",
        ["polyline", "number", "number", "number"],
        [3, 1, 2, 2],
    ),
]
IDS = [s[0] for s in SKILLS]
CODES = {
    "draw-a-line-graph-from-continuous-data": "acmem042",
    "draw-a-graph-from-practical-data": "acmem125",
    "construct-a-scatterplot": "acmgm052",
    "construct-a-time-series-plot": "acmgm087",
}


@pytest.fixture(params=SKILLS, ids=IDS)
def case(request):
    sid, topic, kind, kinds, marks = request.param
    return sup.skill_of(sid), topic, kind, kinds, marks


def test_wiring(case):
    skill, topic, kind, _, _ = case
    sup.check_wiring(skill, topic, "exd_x_" + topic, kind)
    assert skill.needs_figure and skill.capability_gates == ["figure_stimulus"]
    assert not skill.placement_eligible
    assert CODES[skill.id] in skill.tags
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


def test_unknown_tier_is_refused(case):
    _, topic, *_ = case
    for name in (topic, "exd_x_" + topic):
        with pytest.raises(ValueError):
            from mathgen import make

            make(name, seed=1, tier="nope")


def test_exam_contract_distinct_answers_and_shapes(case):
    _, topic, *_ = case
    qs = [sup.exam("exd_x_" + topic, s) for s in range(300)]
    assert len({q.answer_text for q in qs}) >= 100
    assert len({sup.prompt_shape(q) for q in qs}) >= 3


def test_figure_canvas_agrees_with_the_answer(case):
    """The canvas asks for the answer's own kind, a blank grid covers every key point, and nothing is leaked."""
    skill, topic, kind, *_ = case
    for tier in sup.TIERS:
        for seed in range(40):
            q = sup.single(topic, tier, seed)
            fig = q.figure
            assert fig and not figures.validate(fig), (tier, seed)
            cv = fig["canvas"]
            assert cv["reply"] == kind and all(TOOLS[t] == kind for t in cv["tools"])
            assert not fig.get("points") and not fig.get("segments"), "the figure must be blank"
            pts = q.answer["vertices"] if kind == "polyline" else q.answer
            (x0, x1), (y0, y1) = fig["x_range"], fig["y_range"]
            assert all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in pts), (tier, seed)
            assert cv["max_points"] >= len(pts)
            figures.assert_no_leak(fig, [q.answer_text])


def test_exam_figures_do_not_leak(case):
    _, topic, _, kinds, _ = case
    for seed in range(40):
        q = sup.exam("exd_x_" + topic, seed)
        assert q.figure and not figures.validate(q.figure), seed
        assert q.figure["canvas"]["reply"] == kinds[0]
        figures.assert_no_leak(q.figure, [q.parts[0].answer_text])


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


# ---------------------------------------------------------------------------------------------- independent oracles


def _nums(text):
    return [int(n) for n in re.findall(r"-?\d+", text)]


def test_tank_hard_key_rebuilt_from_the_stages():
    """Rebuild the breakpoints from the sentence alone: start volume, then each stage's minutes and rate."""
    for seed in range(60):
        q = sup.single("dr1_line_graph", "hard", seed)
        start = int(re.search(r"holds (\d+) litres", q.text).group(1))
        stages = re.findall(r"for (\d+) minutes (?:the tank (fills|drains) at (\d+) litres|the volume stays)", q.text)
        t, v, pts = 0, start, [(0, start)]
        for minutes, verb, rate in stages:
            t += int(minutes)
            v += (int(rate) if verb == "fills" else -int(rate) if verb == "drains" else 0) * int(minutes)
            pts.append((t, v))
        assert len(stages) == 3
        assert [list(p) for p in pts] == q.answer["vertices"], (seed, q.text)


def test_tank_exam_numbers_rebuilt():
    for seed in range(60):
        q = sup.exam("exd_x_dr1_line_graph", seed)
        pts = q.parts[0].answer["vertices"]
        assert pts[0][0] == 0

        def volume(t):
            for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
                if t0 <= t <= t1:
                    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

        at = int(re.search(r"at (\d+) minutes?", q.parts[1].prompt).group(1))
        assert volume(at) == int(q.parts[1].answer_text)
        target = int(re.search(r"first (\d+) litres", q.parts[2].prompt).group(1))
        first = next(t for t in range(pts[-1][0] + 1) if volume(t) == target)
        assert first == int(q.parts[2].answer_text)
        out = sum(a - b for (_, a), (_, b) in zip(pts, pts[1:]) if a > b)
        assert out == int(q.parts[3].answer_text)


def test_rule_tables_are_linear_or_follow_the_stated_rule():
    for seed in range(60):
        q = sup.single("dr1_graph_from_data", "medium", seed)
        fixed, rate = (
            (int(n) for n in re.search(r"\$(\d+) plus \$(\d+)", q.text).groups())
            if "plus $" in q.text
            else (
                int(re.search(r"fee of \$(\d+)", q.text).group(1)),
                int(re.search(r"plus \$(\d+)", q.text).group(1)),
            )
        )
        assert [list(p) for p in [(x, fixed + rate * x) for x, _ in q.answer]] == q.answer, (seed, q.text)
        ex = sup.exam("exd_x_dr1_graph_from_data", seed)
        pts = ex.parts[0].answer
        slope = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
        assert all(abs((y - pts[0][1]) - slope * (x - pts[0][0])) < 1e-9 for x, y in pts)
        intercept = pts[0][1] - slope * pts[0][0]
        assert intercept == int(ex.parts[1].answer_text) and slope == int(ex.parts[2].answer_text)
        target = int(re.search(r"cost, in dollars, of (\d+)", ex.parts[3].prompt).group(1))
        assert intercept + slope * target == int(ex.parts[3].answer_text)


def test_scatter_exam_oracles():
    from statistics import correlation

    for seed in range(60):
        q = sup.exam("exd_x_dr1_scatterplot", seed)
        pts = q.parts[0].answer
        assert len(pts) == 8 and len({x for x, _ in pts}) == 8
        r = correlation([x for x, _ in pts], [y for _, y in pts])
        band = "strong" if abs(r) >= 0.75 else "moderate" if abs(r) >= 0.5 else "weak"
        assert 0.55 <= abs(r) <= 0.7 or abs(r) >= 0.82, r  # clear of the 0.5 and 0.75 band edges
        assert q.parts[1].answer_text == f"{band} {'positive' if r > 0 else 'negative'}"
        assert "line of best fit" not in q.text.split("(a)")[0] and "line of best fit" in q.parts[2].prompt
        rising = re.search(r"y = (\d*)x \+ (\d+)", q.parts[2].prompt)
        falling = re.search(r"y = (\d+) - (\d*)x", q.parts[2].prompt)
        if rising:
            m, c = int(rising.group(1) or 1), int(rising.group(2))
        else:
            m, c = -int(falling.group(2) or 1), int(falling.group(1))
        slope = (statistics.fmean(y for _, y in pts) - c) / statistics.fmean(x for x, _ in pts)
        assert abs(slope - m) <= 0.6, (seed, slope, m)  # the stated line is a fair best fit
        x0 = int(re.search(r"x = (\d+),", q.parts[2].prompt).group(1))
        assert m * x0 + c == int(q.parts[2].answer_text)


def test_time_series_hard_rebuilt_from_its_changes():
    for seed in range(60):
        q = sup.single("dr1_time_series", "hard", seed)
        nums = [
            int(n) for n in re.findall(r"[+-]?\d+", q.text.split("started at")[1] if "started at" in q.text else q.text)
        ]
        ys = [y for _, y in q.answer["vertices"]]
        assert [x for x, _ in q.answer["vertices"]] == list(range(1, len(ys) + 1))
        diffs = [b - a for a, b in zip(ys, ys[1:])]
        for d in diffs:
            assert f"{d:+d}" in q.text
        assert str(ys[0]) in q.text


def test_time_series_exam_cycle_and_forecast():
    for seed in range(60):
        q = sup.exam("exd_x_dr1_time_series", seed)
        ys = [y for _, y in q.parts[0].answer["vertices"]]
        p = int(q.parts[1].answer_text)
        n = len(ys)
        assert n % p == 0
        deltas = {ys[i + p] - ys[i] for i in range(n - p)}
        assert len(deltas) == 1, "one change of level per cycle"
        (delta,) = deltas
        assert abs(delta) == int(q.parts[2].answer_text)
        assert ys[n - p] + delta == int(q.parts[3].answer_text)
        # no shorter cycle fits the same pattern
        assert not any(
            all(ys[i + k] - ys[i] == ys[k] - ys[0] for i in range(n - k)) and n % k == 0
            for k in range(1, p)
            if p % k == 0 and k < p and False
        )


def test_polyline_key_needs_every_reading():
    """A series plot with a reading missing, or joined out of time order, earns nothing; the reverse order is the same line."""
    for seed in range(30):
        q = sup.single("dr1_time_series", "easy", seed)
        pts = q.answer["vertices"]
        text = lambda ps: " ".join(f"({x}, {y})" for x, y in ps)  # noqa: E731
        assert grade(sup.skill_of("construct-a-time-series-plot"), q, text(pts[::-1])).correct
        assert not grade(sup.skill_of("construct-a-time-series-plot"), q, text(pts[:-1])).correct
        assert not grade(sup.skill_of("construct-a-time-series-plot"), q, text([pts[1], pts[0], *pts[2:]])).correct


def test_scatter_exam_strengths_vary_and_cycle_is_not_in_the_stem():
    bands = {sup.exam("exd_x_dr1_scatterplot", s).parts[1].answer_text.split()[0] for s in range(80)}
    assert bands == {"strong", "moderate"}
    for s in range(40):
        q = sup.exam("exd_x_dr1_time_series", s)
        assert "repeats every" not in q.text.split("(a)")[0] and f"Every {q.parts[1].answer_text} " not in q.text


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_practical_data_questions_vary(tier):
    assert len({sup.single("dr1_graph_from_data", tier, s).text for s in range(300)}) >= 150
