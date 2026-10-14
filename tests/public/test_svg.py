"""The SVG renderer draws every figure kind, deterministically, and escapes all text."""

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from mathgen import figures, svg
from mathgen.engine import load_skills, make_for_skill
from mathgen.figure_kinds import data, early, measure, pattern, space
from mathgen.figure_kinds.measure import SCALE_VARIANTS

SVG_NS = "{http://www.w3.org/2000/svg}"

# Live skills whose questions carry a figure, chosen to cover every kind and variant that any
# skill produces today. A skill that is renamed is skipped; the coverage test below still fails
# if a kind is left with no figure to render.
FIGURE_SKILLS = [
    "f10-base-ten-blocks-number",
    "f10-clock-half-quarter-hour",
    "f10-read-time-to-the-minute",
    "f10-multiply-divide-one-digit-arrays",
    "f10-subitise-name-to-5",
    "count-to-10",
    "f10-angle-relationships",
    "f10-parallel-line-angle-theorems",
    "triangle-area-base-height",
    "f10-fraction-compare-models",
    "f10-fraction-shaded-name",
    "f10-hundred-chart-patterns",
    "f10-money-change-from-note",
    "f10-money-compare-collections",
    "f10-measure-angle-protractor",
    "f10-read-measurement-scale",
    "f10-compare-scale-readings",
    "f10-probability-tree-diagrams",
    "f10-measure-uniform-informal-units",
    "f10-venn-diagram-counts",
    "senior-determine-the-slope-and-intercepts-of-a-straight",
    "senior-interpret-the-derivative-as-the-slope-or-gradient",
    "f3-read-a-column-graph",
    "f2-read-a-picture-graph",
    "f6-read-a-dot-plot",
    "f7-read-a-histogram",
    "f10-read-a-box-plot",
    "f10-compare-box-plots",
    "f6-read-a-pie-chart",
]
SEEDS = (1, 2, 3, 4)


def _live_figures() -> list[dict]:
    skills = {s.id: s for s in load_skills()[1]}
    found = []
    for sid in FIGURE_SKILLS:
        skill = skills.get(sid)
        if skill is None:
            continue
        for index in range(len(skill.sources)):
            for seed in SEEDS:
                fig = make_for_skill(skill, seed, source=index).figure
                if fig:
                    found.append(fig)
    return found


def _hand_built() -> list[dict]:
    """Figures no live skill draws today, and variants a live skill might not hit."""
    frame = [[0, 1, 1, 2, 0, 1, 0, 0, 2, 1]]
    return [
        figures.number_line(
            range_=(-5, 5),
            alt="A number line",
            marks=[{"at": 2, "label": "P"}, {"at": -3, "open": True}],
            rays=[{"from": 1, "direction": "right", "open": True}],
            intervals=[{"from": -4, "to": -2, "open_end": True}],
        ),
        figures.number_line(range_=(0, 1), alt="Tenths", tick=0.1),
        figures.cartesian(
            x_range=(-2, 2),
            y_range=(-1, 1),
            alt="A sine wave",
            x_tick_format="pi",
            x_grid=0.5,
            y_grid=0.5,
            equal_scale=False,
            title="y = sin x",
            lines=[{"through": [[-2, -1], [2, 1]], "label": "L"}],
            segments=[{"from": [-1, 0], "to": [1, 0], "open_start": True, "label": "s"}],
            points=[{"at": [1, 1], "label": "P"}, {"at": [0, 0], "open": True}],
            curves=[{"pieces": [[[-2, 0], [0, 0.5], [2, -0.5]]], "label": "f"}],
        ),
        early.counters(layout="ten_frame", alt="Ten frames", frames=frame + frame, shape="star"),
        early.counters(layout="array", alt="An array", rows=2, cols=3, shape="square", colours=[[1, 2, 1], [2, 1, 2]]),
        early.counters(layout="groups", alt="Groups", groups=[3, 0, 5, 9], shape="diamond"),
        early.counters(layout="dice", alt="Dice", faces=[{"n": n, "colour": 1 + n % 2} for n in (1, 2, 6)]),
        pattern.pattern_strip(
            alt="Pattern rows",
            numbered=True,
            rows=[
                {
                    "label": "A",
                    "cells": [
                        pattern.shape_cell(s, f)
                        for s, f in (("heart", "striped"), ("star", "open"), ("circle", "solid"))
                    ]
                    + [pattern.blank_cell(), pattern.mark_cell()],
                    "bracket": [0, 2],
                },
                {
                    "label": "B",
                    "cells": [pattern.action_cell("clap"), pattern.action_cell("stamp"), pattern.action_cell("jump")]
                    + [pattern.text_cell(7), pattern.text_cell("M"), pattern.stack_cell(10)],
                },
            ],
        ),
    ]


@pytest.fixture(scope="module")
def all_figures() -> list[dict]:
    return _live_figures() + _hand_built()


def test_every_registered_kind_has_a_renderer():
    assert set(figures.kinds()) == set(svg.rendered_kinds())


def test_every_kind_is_exercised(all_figures):
    seen = {f["kind"] for f in all_figures}
    assert seen == set(figures.kinds()), f"no figure to render for {set(figures.kinds()) - seen}"


def test_every_scale_variant_and_counter_layout_is_exercised(all_figures):
    assert {f["variant"] for f in all_figures if f["kind"] == "scale"} == set(SCALE_VARIANTS)
    assert {f["layout"] for f in all_figures if f["kind"] == "counters"} == {
        "ten_frame",
        "array",
        "groups",
        "dice",
        "scatter",
    }


def test_figures_render_to_valid_svg_with_the_alt_text_as_title(all_figures):
    assert all_figures
    for fig in all_figures:
        text = svg.figure_svg(fig)
        root = ET.fromstring(text)
        assert root.tag == SVG_NS + "svg", fig["kind"]
        assert root.get("role") == "img"
        titles = [e for e in root if e.tag == SVG_NS + "title"]
        assert len(titles) == 1 and titles[0].text == fig["alt"], fig["kind"]
        assert root.get("aria-labelledby") == titles[0].get("id")
        # no scripts, external references or fonts
        assert "<script" not in text and "href" not in text and "@import" not in text


def test_rendering_is_deterministic(all_figures):
    for fig in all_figures:
        assert svg.figure_svg(fig) == svg.figure_svg(dict(fig)), fig["kind"]


def test_width_scales_the_displayed_size_only(all_figures):
    fig = all_figures[0]
    small, big = ET.fromstring(svg.figure_svg(fig, width=200)), ET.fromstring(svg.figure_svg(fig, width=800))
    assert small.get("viewBox") == big.get("viewBox")
    assert (small.get("width"), big.get("width")) == ("200", "800")
    assert int(big.get("height")) == pytest.approx(4 * int(small.get("height")), abs=3)


def test_unknown_kind_and_invalid_spec_raise_value_error():
    with pytest.raises(ValueError):
        svg.figure_svg({"kind": "hologram", "alt": "x"})
    with pytest.raises(ValueError):
        svg.figure_svg({"kind": "base_ten", "alt": "", "hundreds": 1, "tens": 0, "ones": 0})
    with pytest.raises(ValueError):
        svg.figure_svg("not a spec")  # type: ignore[arg-type]


def test_a_clock_with_no_numerals_writes_no_time():
    clock = measure.clock(hour=3, minute=15, alt="A clock face", numerals="none")
    root = ET.fromstring(svg.figure_svg(clock))
    assert [t.text for t in root.iter(SVG_NS + "text")] == []


def test_hostile_text_is_escaped():
    evil = '<script>alert("x")</script> & \'quote\' "dq"'
    fig = figures.cartesian(
        x_range=(0, 4),
        y_range=(0, 4),
        alt=evil,
        x_label=evil,
        title=evil,
        points=[{"at": [1, 1], "label": evil}],
    )
    text = svg.figure_svg(fig)
    assert "<script" not in text
    root = ET.fromstring(text)
    texts = [t.text for t in root.iter(SVG_NS + "text")]
    assert texts.count(evil) == 3  # title, x label, point label
    assert next(root.iter(SVG_NS + "title")).text == evil


def test_hostile_text_in_other_kinds_is_escaped():
    evil = "<b>&amp;</b>"
    money = early.money(collections=[["$5"], ["20c"]], alt=evil, labels=["<", "&"])
    venn = space.venn(
        alt=evil,
        sets=[{"id": "A", "name": "<A>"}, {"id": "B", "name": "&B"}],
        regions={"A": "1", "B": "2", "AB": "3", "none": "?"},
    )
    for fig in (money, venn):
        root = ET.fromstring(svg.figure_svg(fig))
        assert next(root.iter(SVG_NS + "title")).text == evil


def test_control_characters_cannot_break_the_xml():
    fig = figures.number_line(range_=(0, 4), alt="a\x00b\x08c", marks=[{"at": 1, "label": "x\x0by"}])
    root = ET.fromstring(svg.figure_svg(fig))
    assert next(root.iter(SVG_NS + "title")).text == "abc"


def _ids_of(text: str) -> set[str]:
    return {el.get("id") for el in ET.fromstring(text).iter() if el.get("id")}


def test_element_ids_do_not_encode_spec_fields_that_are_not_drawn():
    # with no numerals the face shows hands only; two different times differ in the drawing, but a spec field that
    # is not drawn at all must change nothing, ids included
    a = measure.clock(hour=3, minute=15, alt="A clock face", numerals="none")
    b = {**a, "hidden_note": "anything"}
    assert svg.figure_svg(a) == svg.figure_svg(b)
    ids = _ids_of(svg.figure_svg(a))
    assert ids and all(i.startswith("f") for i in ids)
    # the prefix is a hash of the finished drawing, not of the spec
    prefix = next(iter(ids)).split("-")[0]
    text = svg.figure_svg(a)
    expected = "f" + hashlib.sha1(text.replace(prefix, "fZZidprefixZZ").encode()).hexdigest()[:8]
    assert prefix == expected


def test_ids_are_unique_between_different_drawings_and_the_placeholder_never_leaks(all_figures):
    prefixes = set()
    for fig in all_figures:
        text = svg.figure_svg(fig)
        assert "ZZidprefixZZ" not in text
        prefixes.add(next(iter(_ids_of(text))).split("-")[0])
    assert len(prefixes) > 1


# ---------------------------------------------------------------- data displays
#
# bar_chart, picture_graph, dot_plot, histogram, box_plot and pie_chart are read-the-chart figures: the picture
# must show every value to scale, and must never write a value the learner is being asked for.

DATA_KINDS = ("bar_chart", "picture_graph", "dot_plot", "histogram", "box_plot", "pie_chart")
DATA_SKILLS = {
    "bar_chart": "f3-read-a-column-graph",
    "picture_graph": "f2-read-a-picture-graph",
    "dot_plot": "f6-read-a-dot-plot",
    "histogram": "f7-read-a-histogram",
    "box_plot": "f10-compare-box-plots",
    "pie_chart": "f6-read-a-pie-chart",
}
MORE_SEEDS = tuple(range(1, 9))


def _skill_figures(skill_id: str, kind: str) -> list[dict]:
    skill = {s.id: s for s in load_skills()[1]}[skill_id]
    found = []
    for index in range(len(skill.sources)):
        for seed in MORE_SEEDS:
            fig = make_for_skill(skill, seed, source=index).figure
            if fig and fig["kind"] == kind:
                found.append(fig)
    return found


@pytest.fixture(scope="module")
def live_data() -> dict[str, list[dict]]:
    skills = dict(DATA_SKILLS)
    out = {kind: _skill_figures(sid, kind) for kind, sid in skills.items()}
    out["box_plot"] += _skill_figures("f10-read-a-box-plot", "box_plot")
    return out


def _texts(root) -> list[str]:
    return [t.text for t in root.iter(SVG_NS + "text")]


def _tag(root, name):
    return list(root.iter(SVG_NS + name))


def _num(v) -> str:
    return svg._num_text(v)


def _axis_numerals(lo, hi, step) -> set[str]:
    n = round((hi - lo) / step)
    return {_num(lo + i * step) for i in range(n + 1)}


def _allowed_text(fig: dict) -> set[str]:
    """Every string a figure of this kind may draw: names the spec carries, titles, and axis numerals."""
    allowed = {fig.get(k) for k in ("title", "x_label", "y_label") if fig.get(k)}
    kind = fig["kind"]
    if kind == "bar_chart":
        allowed |= set(fig["categories"])
        if len(fig["series"]) == 2:
            allowed |= {s["name"] for s in fig["series"]}
        ax = fig["value_axis"]
        allowed |= _axis_numerals(0, ax["max"], ax["step"])
    elif kind == "picture_graph":
        allowed |= {r["label"] for r in fig["rows"]} | {data._key_text(fig)}
    elif kind in ("dot_plot", "box_plot"):
        allowed |= _axis_numerals(*fig["range"], fig["step"])
        allowed |= {p["label"] for p in fig.get("plots", ()) if p["label"]}
    elif kind == "histogram":
        for p in fig["panels"]:
            allowed |= {_num(e) for e in p["edges"]} | ({p["label"]} if p["label"] else set())
            ax = p["frequency_axis"]
            allowed |= _axis_numerals(0, ax["max"], ax["step"])
    elif kind == "pie_chart":
        allowed |= {s["label"] for s in fig["sectors"]}
    return allowed


@pytest.mark.parametrize("kind", DATA_KINDS)
def test_data_displays_render_real_specs_to_well_formed_deterministic_svg(kind, live_data):
    figs = live_data[kind]
    assert len(figs) >= 8, f"{DATA_SKILLS[kind]} gave too few {kind} figures"
    for fig in figs:
        text = svg.figure_svg(fig)
        root = ET.fromstring(text)
        assert next(root.iter(SVG_NS + "title")).text == fig["alt"]
        assert text == svg.figure_svg(json.loads(json.dumps(fig)))
        assert "ZZidprefixZZ" not in text and "<script" not in text and "href" not in text


@pytest.mark.parametrize("kind", DATA_KINDS)
def test_data_displays_write_only_names_and_axis_numerals_never_a_value(kind, live_data):
    for fig in live_data[kind]:
        shown = set(_texts(ET.fromstring(svg.figure_svg(fig))))
        extra = shown - _allowed_text(fig)
        assert not extra, f"{kind} drew text the spec does not carry: {extra}"


def _fill_rects(root, fill):
    return [r for r in _tag(root, "rect") if r.get("fill") == fill and r.get("stroke") == "currentColor"]


def test_bar_chart_bars_are_drawn_to_their_values():
    for orientation in ("vertical", "horizontal"):
        fig = data.bar_chart(
            categories=["Mon", "Tue", "Wed", "Thu"],
            series=[{"name": "", "values": [35, 0, 15, 50]}],
            value_max=50,
            value_step=10,
            minor_lines=True,
            orientation=orientation,
            alt="a column graph",
            x_label="Day",
            y_label="Sold",
        )
        root = ET.fromstring(svg.figure_svg(fig))
        bars = _fill_rects(root, svg.DARK)
        assert len(bars) == 3  # the zero bar has no area, so it is not drawn
        size = "height" if orientation == "vertical" else "width"
        lengths = [float(b.get(size)) for b in bars]
        assert lengths[0] / lengths[1] == pytest.approx(35 / 15, rel=1e-2)
        assert lengths[2] / lengths[1] == pytest.approx(50 / 15, rel=1e-2)
        # the tallest bar reaches the top gridline; the value axis has a numeral at every step
        assert {"0", "10", "20", "30", "40", "50"} <= set(_texts(root))


def test_bar_chart_two_series_has_a_legend_and_two_fills_and_long_names_fit():
    names = ["A very long category", "Another long one!!", "Third long category", "Fourth long category"]
    fig = data.bar_chart(
        categories=names,
        series=[{"name": "Girls", "values": [5, 10, 15, 20]}, {"name": "Boys", "values": [20, 15, 10, 5]}],
        value_max=20,
        value_step=5,
        alt="a bar graph",
    )
    root = ET.fromstring(svg.figure_svg(fig))
    assert (
        len(_fill_rects(root, svg.DARK)) == 4 + 1 and len(_fill_rects(root, svg.LIGHT)) == 4 + 1
    )  # +1 legend swatch each
    assert {"Girls", "Boys", *names} <= set(_texts(root))
    # a crowded category axis is tilted, not truncated or overlapped
    assert any("rotate(" in (t.get("transform") or "") for t in _tag(root, "text"))


def test_histogram_bars_touch_and_match_frequencies_and_two_panels_are_stacked():
    fig = data.histogram(
        panels=[
            data.histogram_panel(
                edges=[0, 10, 20, 30, 40], frequencies=[4, 0, 8, 2], freq_max=8, freq_step=2, label="Boys"
            ),
            data.histogram_panel(
                edges=[0, 10, 20, 30, 40],
                frequencies=[1, 3, 5, 7],
                freq_max=8,
                freq_step=1,
                minor_lines=True,
                label="Girls",
            ),
        ],
        alt="two histograms",
        x_label="Time",
    )
    root = ET.fromstring(svg.figure_svg(fig))
    bars = _fill_rects(root, svg.MID)
    assert len(bars) == 3 + 4  # the empty class has no bar
    first = bars[:3]
    assert float(first[1].get("height")) / float(first[2].get("height")) == pytest.approx(4, rel=1e-2)  # 8 against 2
    second = bars[3:]
    assert float(second[3].get("height")) / float(second[0].get("height")) == pytest.approx(7, rel=1e-2)
    # touching: the bar after a bar starts exactly where it ended
    assert float(second[0].get("x")) + float(second[0].get("width")) == pytest.approx(
        float(second[1].get("x")), abs=0.02
    )
    assert float(second[0].get("y")) > float(first[0].get("y"))  # panel one is above panel two
    assert {"Boys", "Girls"} <= set(_texts(root))


def test_dot_plot_stacks_one_dot_per_observation_above_its_tick():
    fig = data.dot_plot(low=2, high=8, step=1, dots={3: 3, 4: 6, 7: 1}, alt="a dot plot", x_label="Goals")
    root = ET.fromstring(svg.figure_svg(fig))
    dots = [c for c in _tag(root, "circle")]
    assert len(dots) == 10
    columns: dict[str, int] = {}
    for c in dots:
        columns[c.get("cx")] = columns.get(c.get("cx"), 0) + 1
    assert sorted(columns.values()) == [1, 3, 6]
    xs = sorted(float(x) for x in columns)
    assert (xs[1] - xs[0]) / (xs[2] - xs[1]) == pytest.approx(1 / 3, rel=1e-2)  # values 3, 4, 7
    assert set(_texts(root)) == {"2", "3", "4", "5", "6", "7", "8", "Goals"}  # no counts


def test_box_plot_five_number_summary_sits_on_the_scaled_axis():
    fig = data.box_plot(
        low=0,
        high=60,
        step=5,
        minor_ticks=True,
        alt="a box plot",
        plots=[{"label": "", "min": 20, "q1": 27.5, "median": 30, "q3": 32.5, "max": 40, "outliers": [0]}],
    )
    root = ET.fromstring(svg.figure_svg(fig))
    grid = sorted(
        float(ln.get("x1"))
        for ln in _tag(root, "line")
        if ln.get("stroke") == svg.GRID and ln.get("x1") == ln.get("x2")
    )
    x0, x1 = grid[0], grid[-1]

    def X(v):
        return x0 + v / 60 * (x1 - x0)

    box = next(r for r in _tag(root, "rect") if r.get("fill") == svg.LIGHT)
    assert float(box.get("x")) == pytest.approx(X(27.5), abs=0.02)
    assert float(box.get("x")) + float(box.get("width")) == pytest.approx(X(32.5), abs=0.02)
    median = next(ln for ln in _tag(root, "line") if ln.get("stroke-width") == "3")
    assert float(median.get("x1")) == pytest.approx(X(30), abs=0.02)
    whiskers = {
        float(ln.get("x1"))
        for ln in _tag(root, "line")
        if ln.get("stroke-width") == "1.8" and ln.get("x1") == ln.get("x2")
    }
    assert any(abs(w - X(20)) < 0.02 for w in whiskers) and any(abs(w - X(40)) < 0.02 for w in whiskers)
    crosses = [
        ln
        for ln in _tag(root, "line")
        if ln.get("stroke-width") == "1.8" and ln.get("y1") != ln.get("y2") and ln.get("x1") != ln.get("x2")
    ]
    assert len(crosses) == 2  # the outlier is one cross, centred on its value
    assert all((float(c.get("x1")) + float(c.get("x2"))) / 2 == pytest.approx(X(0), abs=0.02) for c in crosses)
    assert set(_texts(root)) == {str(v) for v in range(0, 61, 5)}


def test_picture_graph_draws_whole_and_left_half_symbols_with_the_key():
    for symbol in data.SYMBOLS:
        fig = data.picture_graph(
            categories=["Cat", "Dog"], symbols=[3, 2.5], each=2, symbol=symbol, alt="a picture graph", unit="votes"
        )
        text = svg.figure_svg(fig)
        root = ET.fromstring(text)
        assert "Each symbol = 2 votes" in _texts(root)
        # 3 + 2 whole symbols and one clipped half, plus the key's symbol and the half's cut-edge line
        clipped = [g for g in _tag(root, "g") if (g.get("clip-path") or "").startswith("url(#")]
        assert len(clipped) == 2
        ids = {e.get("id") for e in _tag(root, "clipPath")}
        assert all(g.get("clip-path")[5:-1] in ids for g in clipped)
        assert set(_texts(root)) == {"Cat", "Dog", "Each symbol = 2 votes"}


def test_pie_chart_has_a_wedge_boundary_for_every_part_and_labels_beside_the_sectors():
    fig = data.pie_chart(
        parts=8,
        sectors=[
            {"label": "Bus", "parts": 4},
            {"label": "Train", "parts": 1},
            {"label": "A long sector label", "parts": 3},
        ],
        alt="a pie chart",
    )
    root = ET.fromstring(svg.figure_svg(fig))
    assert len(_tag(root, "path")) == 3
    assert len(_tag(root, "line")) == 8 - 3  # thin radii only where a boundary is not already a sector edge
    assert _texts(root) == ["Bus", "Train", "A long sector label"]  # no counts, fractions or percentages
    assert float(root.get("viewBox").split()[2]) == 400
    for t in _tag(root, "text"):
        assert 0 <= float(t.get("x")) <= 400


@pytest.mark.parametrize("kind", DATA_KINDS)
def test_data_display_hostile_text_is_escaped(kind):
    evil = "<script>alert(\"x\")</script> & 'q'"
    short = "<b>&amp;</b>"
    if kind == "bar_chart":
        fig = data.bar_chart(
            categories=[short, "b"],
            series=[{"name": short, "values": [1, 2]}, {"name": "n2", "values": [2, 1]}],
            value_max=2,
            value_step=1,
            alt=evil,
            x_label=evil,
            y_label=evil,
            title=evil,
        )
        wanted = [short, evil]
    elif kind == "picture_graph":
        fig = data.picture_graph(
            categories=[short, "b"], symbols=[1, 2], each=1, symbol="star", alt=evil, unit=short, title=evil
        )
        wanted = [short, evil]
    elif kind == "dot_plot":
        fig = data.dot_plot(low=0, high=2, step=1, dots={1: 2}, alt=evil, x_label=evil, title=evil)
        wanted = [evil]
    elif kind == "histogram":
        panel = data.histogram_panel(edges=[0, 1, 2], frequencies=[1, 2], freq_max=2, freq_step=1, label=short)
        fig = data.histogram(panels=[panel], alt=evil, x_label=evil, y_label=evil, title=evil)
        wanted = [short, evil]
    elif kind == "box_plot":
        plot = {"label": short, "min": 1, "q1": 2, "median": 3, "q3": 4, "max": 5, "outliers": []}
        fig = data.box_plot(low=0, high=6, step=1, plots=[plot], alt=evil, x_label=evil, title=evil)
        wanted = [short, evil]
    else:
        fig = data.pie_chart(
            parts=2, sectors=[{"label": short, "parts": 1}, {"label": evil[:20], "parts": 1}], alt=evil
        )
        wanted = [short, evil[:20]]
    text = svg.figure_svg(fig)
    assert "<script" not in text and "<b>" not in text
    root = ET.fromstring(text)
    assert next(root.iter(SVG_NS + "title")).text == evil
    for w in wanted:
        assert w in _texts(root)


def test_data_displays_validate_through_figures_check():
    for fig in (
        {"kind": "pie_chart", "alt": "a pie chart", "parts": 4, "sectors": [{"label": "A", "parts": 1}]},
        {"kind": "dot_plot", "alt": "", "range": [0, 2], "step": 1, "dots": [{"at": 1, "count": 1}]},
        {"kind": "bar_chart", "alt": "x", "orientation": "diagonal", "categories": ["a"], "series": []},
    ):
        with pytest.raises(ValueError):
            svg.figure_svg(fig)
