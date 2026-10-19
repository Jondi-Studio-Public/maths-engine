"""The statistical display skills: picture graph, column graph, dot plot, histogram, box plot (read and compare),
pie chart, plus the figure reader of each converted skill (line graph, histogram shape, line of best fit by eye).

Every question is generated through its skill's own sources. Checked for every tier: the figure is valid and does
not show the answer, the registered reader re-derives the key from the figure alone, every distractor is graded
wrong by the real grader, and the prompt never states the value that is read."""

import collections
import functools
import re

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.grading import grade
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape
from scripts.measure_tiers import classify_skill

NUMBER_SKILLS = {
    "f2-read-a-picture-graph": ("stat1_picture_graph", "picture_graph", "AC9M2ST02"),
    "f3-read-a-column-graph": ("stat2_column_graph", "bar_chart", "AC9M3ST02"),
    "f6-read-a-dot-plot": ("stat3_dot_plot", "dot_plot", "AC9M7ST02"),
    "f7-read-a-histogram": ("stat4_histogram", "histogram", "AC9M7ST02"),
    "f10-read-a-box-plot": ("stat5_box_plot_read", "box_plot", "AC9M10ST02"),
    "f6-read-a-pie-chart": ("stat6_pie_chart", "pie_chart", "AC9M6ST01"),
}
MCQ_SKILLS = {"f10-compare-box-plots": ("stat5_box_plot_compare", "box_plot", "AC9M10ST02")}
CONVERTED = {
    "f10-line-graph-change-over-time": ("p18a_line_graph_claims", "cartesian"),
    "senior-compare-the-characteristics-of-the-shape-of-histograms": ("p19b_histogram_shape", "histogram"),
    "senior-find-the-line-of-best-fit-by-eye": ("p8c_line_best_fit_by_eye", "cartesian"),
}
ALL = {**NUMBER_SKILLS, **MCQ_SKILLS}
TIERS = ["easy", "medium", "hard"]
# Skills that also carry a multi-part exam chain as a fourth source (stages [exam], its own exd_x_* topic).
EXAM_CHAINED = {"f7-read-a-histogram", "f10-read-a-box-plot", "f10-compare-box-plots"}
SEEDS = range(60)


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


def _gen(skill_id, tier, seed):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", [*ALL, *CONVERTED])
def test_skill_is_active_behind_the_figure_gate(skill_id):
    skill = _skills()[skill_id]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.mental is None
    assert skill.domain == "statistics-probability" and skill.misconceptions


@pytest.mark.parametrize("skill_id,info", ALL.items())
def test_new_skill_shape(skill_id, info):
    skill = _skills()[skill_id]
    topic, _kind, code = info
    assert skill.answer == ("mcq" if skill_id in MCQ_SKILLS else "number")
    core = [s for s in skill.sources if s.difficulty != "exam"]
    exam = [s for s in skill.sources if s.difficulty == "exam"]
    assert [s.difficulty for s in core] == TIERS
    assert all(s.topic == topic for s in core)
    assert [list(s.stages) for s in core] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
    if skill_id in EXAM_CHAINED:
        assert len(exam) == 1 and list(exam[0].stages) == ["exam"] and exam[0].topic.startswith("exd_x_")
    else:
        assert exam == []
    assert code in skill.tags and "acara-f10-v9" in skill.tags


@pytest.mark.parametrize("skill_id", ALL)
def test_tiers_escalate_in_prompt_shape(skill_id):
    verdict, adds = classify_skill(_skills()[skill_id], 60)
    expected = [False, True, True, True] if skill_id in EXAM_CHAINED else [False, True, True]
    assert verdict == "escalates" and adds == expected, adds


# ------------------------------------------------------------------ figures, readers, graders


@pytest.mark.parametrize("skill_id,info", ALL.items())
@pytest.mark.parametrize("tier", TIERS)
def test_figure_valid_unleaked_and_read_back(skill_id, info, tier):
    topic, kind, _code = info
    reader = figures.get_reader(topic)
    assert reader is not None
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert q.figure is not None and q.figure["kind"] == kind, (tier, seed)
        assert figures.validate(q.figure) == [], (tier, seed)
        assert reader(q) == q.answer_text, (tier, seed, q.text)
        if skill_id not in MCQ_SKILLS:
            figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        # the alt text names the object, never a number
        assert not re.search(r"\d", q.figure["alt"]), q.figure["alt"]


@pytest.mark.parametrize("skill_id", NUMBER_SKILLS)
@pytest.mark.parametrize("tier", TIERS)
def test_number_key_graded_right_and_every_distractor_wrong(skill_id, tier):
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert grade("number", q, q.answer_text).correct, (tier, seed)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3 and q.answer_text not in q.distractors
        for d in q.distractors:
            assert not grade("number", q, d).correct, (tier, seed, d, q.answer_text)
        assert not grade("number", q, "").correct


@pytest.mark.parametrize("tier", TIERS)
def test_compare_box_plots_mcq_key_and_distractors(tier):
    for seed in SEEDS:
        q = _gen("f10-compare-box-plots", tier, seed)
        options = mcq_options(q)
        assert len(options) == (4 if tier == "hard" else 3) and len(set(options)) == len(options)
        for idx, opt in enumerate(options):
            assert grade_mcq(q, idx).correct == (opt == q.answer_text)


@pytest.mark.parametrize("skill_id", ALL)
def test_deterministic_and_unknown_tier_raises(skill_id):
    topic = ALL[skill_id][0]
    for tier in TIERS:
        for seed in range(15):
            assert _gen(skill_id, tier, seed).to_json() == _gen(skill_id, tier, seed).to_json()
    with pytest.raises(ValueError):
        make(topic, seed=1, tier="bogus")


@pytest.mark.parametrize("skill_id,info", ALL.items())
@pytest.mark.parametrize("tier", TIERS)
def test_variety_and_no_stated_answer(skill_id, info, tier):
    topic = info[0]
    qs = [_gen(skill_id, tier, s) for s in SEEDS]
    assert len({prompt_shape(q) for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 20
    if skill_id in NUMBER_SKILLS:
        # the prompt may state a total or a count it asks the learner to build on, never the reading itself
        counts = collections.Counter(q.answer_text for q in qs)
        assert max(counts.values()) / len(qs) < 0.3, counts.most_common(3)
    assert all("  " not in q.text for q in qs)


# ------------------------------------------------------------------ the readers do not peek at the key


@pytest.mark.parametrize("skill_id,info", NUMBER_SKILLS.items())
def test_reader_ignores_the_stored_answer(skill_id, info):
    q = _gen(skill_id, "hard", 3)
    key = q.answer_text
    q.answer_text, q.answer, q.distractors = "-999", -999, []
    assert figures.get_reader(info[0])(q) == key


# ------------------------------------------------------------------ per-skill properties


def test_picture_graph_key_matters_and_halves_only_with_even_keys():
    halves_seen = False
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen("f2-read-a-picture-graph", tier, seed)
            each = q.figure["key"]["each"]
            symbols = [r["symbols"] for r in q.figure["rows"]]
            half = any(s != int(s) for s in symbols)
            halves_seen = halves_seen or half
            assert not half or each % 2 == 0
            if tier == "easy":
                assert each in (1, 2) and not half
            else:
                assert each in (2, 5, 10)
    assert halves_seen


def test_picture_graph_easy_offers_the_key_as_one_error():
    hits = 0
    for seed in SEEDS:
        q = _gen("f2-read-a-picture-graph", "easy", seed)
        if q.figure["key"]["each"] == 2:
            asked = next(r for r in q.figure["rows"] if r["label"] in q.text)
            hits += str(int(asked["symbols"])) in q.distractors
    assert hits > 10


def test_picture_graph_hard_reaches_every_variant():
    seen = set()
    for seed in range(120):
        t = _gen("f2-read-a-picture-graph", "hard", seed).text
        seen.add(
            "total"
            if "total" in t.lower() or "altogether" in t
            else "extreme"
            if "most popular" in t or "biggest" in t
            else "inverse"
        )
    assert seen == {"total", "extreme", "inverse"}


def test_column_graph_orientations_series_and_half_lines():
    orient, grouped, minor = set(), 0, 0
    for tier in TIERS:
        for seed in SEEDS:
            fig = _gen("f3-read-a-column-graph", tier, seed).figure
            orient.add(fig["orientation"])
            axis = fig["value_axis"]
            if tier == "easy":
                assert axis["step"] in (1, 2, 5) and len(fig["series"]) == 1 and not axis["minor"]
                assert all(v % axis["step"] == 0 for v in fig["series"][0]["values"])
            elif tier == "medium":
                assert axis["step"] in (2, 5, 10) and len(fig["series"]) == 1 and not axis["minor"]
            else:
                grouped += len(fig["series"]) == 2
                minor += axis["minor"]
                if axis["minor"]:
                    assert any(v % axis["step"] for s in fig["series"] for v in s["values"])
    assert orient == {"vertical", "horizontal"} and grouped > 10 and minor > 10


def test_dot_plot_hard_needs_the_whole_data_set():
    saw_half_median = saw_mean = False
    for seed in range(120):
        q = _gen("f6-read-a-dot-plot", "hard", seed)
        if "median" in q.text and "." in q.answer_text:
            saw_half_median = True
        if "mean" in q.text:
            saw_mean = True
            assert "." not in q.answer_text or q.answer_text.endswith(".5")
    assert saw_half_median and saw_mean


def test_dot_plot_medium_mode_is_the_unique_tallest_stack():
    for seed in range(120):
        q = _gen("f6-read-a-dot-plot", "medium", seed)
        if "mode" in q.text.lower():
            counts = sorted((d["count"] for d in q.figure["dots"]), reverse=True)
            assert counts[0] > counts[1]


def test_histogram_hard_percentages_are_whole_and_bars_use_half_lines():
    pct = 0
    for seed in range(120):
        q = _gen("f7-read-a-histogram", "hard", seed)
        panel = q.figure["panels"][0]
        assert panel["frequency_axis"]["minor"] is True
        assert sum(1 for f in panel["frequencies"] if f % panel["frequency_axis"]["step"]) >= 3
        if "percentage" in q.text:
            pct += 1
            total = sum(panel["frequencies"])
            assert total in (20, 25, 40, 50)
            assert "." not in q.answer_text
    assert pct > 5


def test_histogram_prompt_names_boundaries_but_no_frequency():
    for tier in TIERS:
        for seed in range(30):
            q = _gen("f7-read-a-histogram", tier, seed)
            assert (
                "frequency" not in q.text.lower()
                or "highest frequency" in q.text.lower()
                or "lowest frequency" in q.text.lower()
            )
            assert len(q.figure["panels"]) == 1


def test_box_plot_hard_draws_outliers_when_the_question_is_about_them():
    saw = 0
    for seed in range(120):
        q = _gen("f10-read-a-box-plot", "hard", seed)
        plot = q.figure["plots"][0]
        if "outlier" in q.text or "including" in q.text:
            assert plot["outliers"]
            saw += 1
    assert saw > 10


def test_box_plot_easy_reads_the_named_landmark_and_medium_may_use_half_ticks():
    minor = 0
    for seed in range(120):
        q = _gen("f10-read-a-box-plot", "medium", seed)
        minor += q.figure["minor"]
    assert minor > 5


def test_compare_box_plots_medium_plots_disagree_on_the_other_measure_half_the_time():
    disagree = total = 0
    for seed in range(200):
        q = _gen("f10-compare-box-plots", "medium", seed)
        a, b = q.figure["plots"]
        iqr = (a["q3"] - a["q1"]) - (b["q3"] - b["q1"])
        rng = (a["max"] - a["min"]) - (b["max"] - b["min"])
        assert iqr != 0 and rng != 0
        total += 1
        disagree += (iqr > 0) != (rng > 0)
    assert 0.3 < disagree / total < 0.7


def test_compare_box_plots_hard_has_exactly_one_true_statement_of_distinct_kinds():
    for seed in range(60):
        q = _gen("f10-compare-box-plots", "hard", seed)
        assert len({re.sub(r"Class [AB]|Year \d+|Group [XY]|Team \w+", "_", o) for o in mcq_options(q)}) >= 3


def test_pie_chart_sectors_are_countable_slices():
    parts_seen = set()
    for tier in TIERS:
        for seed in SEEDS:
            fig = _gen("f6-read-a-pie-chart", tier, seed).figure
            assert sum(s["parts"] for s in fig["sectors"]) == fig["parts"]
            parts_seen.add(fig["parts"])
            q = _gen("f6-read-a-pie-chart", tier, seed)
            # a stated count is a whole number of slices' worth of people
            total = [int(n) for n in re.findall(r"\b\d+\b", q.text)]
            assert len(total) == 1
    assert parts_seen >= {4, 5, 8, 10}


# ------------------------------------------------------------------ the converted skills draw their data


# The histogram skill's exam tier is the r2k frequency-table chain, which draws no figure, and
# its easy tier names the shape in words, so only its medium and hard sources draw.
_DRAWN_SOURCES = {"senior-compare-the-characteristics-of-the-shape-of-histograms": 2}


def test_converted_skills_register_readers_and_draw_figures():
    for skill_id, (topic, kind) in CONVERTED.items():
        assert figures.get_reader(topic) is not None
        skill = _skills()[skill_id]
        drawn = 0
        for src in skill.sources:
            q = make(src.topic, seed=1, difficulty=src.difficulty, **(src.params or {}))
            if q.figure is not None:
                assert q.figure["kind"] == kind
                assert figures.get_reader(topic)(q) == q.answer_text
                drawn += 1
        assert drawn >= _DRAWN_SOURCES.get(skill_id, 3), (skill_id, drawn)


def test_line_graph_values_sit_on_gridlines_and_are_not_listed():
    for tier in ("easy", "medium", "hard", "exam"):
        for seed in range(40):
            q = make("p18a_line_graph_claims", seed=seed, difficulty=tier)
            fig = q.figure
            assert fig["equal_scale"] is False and fig["y_range"][0] == 0
            for p in fig["points"]:
                assert p["at"][1] % fig["y_grid"] == 0
            body = q.text.split("A claim is made")[0]
            assert not re.search(r"\d", body), body


def test_histogram_shape_hard_reflects_and_exam_draws_two_panels():
    for seed in range(30):
        q = make("p19b_histogram_shape", seed=seed, difficulty="exam")
        assert len(q.figure["panels"]) == 2
        q = make("p19b_histogram_shape", seed=seed, difficulty="medium")
        assert len(q.figure["panels"]) == 1 and not re.search(r"\d+-\d+:", q.text)
    assert make("p19b_histogram_shape", seed=1, difficulty="easy").figure is None


def test_best_fit_by_eye_draws_the_points_and_lists_none():
    for seed in range(30):
        q = make("p8c_line_best_fit_by_eye", seed=seed, difficulty="medium")
        assert len(q.figure["points"]) == 5 and q.figure["equal_scale"] is False
        assert not re.search(r"\(-?\d+, -?\d+\)", q.text)
