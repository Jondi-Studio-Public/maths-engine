"""The SVG renderer draws every figure kind, deterministically, and escapes all text."""

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from mathgen import figures, svg
from mathgen.engine import load_skills, make_for_skill
from mathgen.figure_kinds import (
    calendar as calendar_kinds,
    compare,
    data,
    early,
    grid,
    measure,
    pattern,
    scene,
    space,
    tally,
)
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
    "sort-and-count-objects",
    "tally-tables",
    "compare-tally-and-picture-displays",
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


def _compare_samples() -> list[dict]:
    """One of every compare_picture layout, with every object style, unit style and container shape."""
    styles = ("pencil", "ribbon", "snake", "rope", "train")
    objs = [{"label": st, "style": st, "start": i, "length": 6 - i % 3} for i, st in enumerate(styles)]
    clips = [{"label": "pencil", "style": "bar", "start": 0, "length": 3, "units": 3}]
    book = {"label": "book", "mass": 5, "size": 3}
    apple = {"label": "apple", "mass": 2, "size": 1}
    shoe = {"label": "shoe", "mass": 9, "size": 2}
    pans = compare.pan
    return [
        compare.compare_lengths(objects=objs, lined_up=False, alt="Five objects"),
        compare.compare_lengths(objects=[{**o, "start": 0} for o in objs[:2]], lined_up=True, alt="Two lined up"),
        compare.compare_lengths(objects=clips, lined_up=True, alt="Clips", unit={"name": "block", "style": "block"}),
        compare.compare_lengths(
            objects=[{**clips[0], "start": 2}], lined_up=True, alt="Clips", unit={"name": "clip", "style": "paper_clip"}
        ),
        compare.compare_balance(
            balances=[compare.balance(left=pans(objects=[book]), right=pans(blocks=3))], alt="A balance and blocks"
        ),
        compare.compare_balance(
            balances=[
                compare.balance(
                    left=pans(objects=[apple, book], blocks=0), right=pans(objects=[shoe, shoe | {"label": "ball"}])
                )
            ],
            alt="Two objects a side",
        ),
        compare.compare_balance(
            balances=[
                compare.balance(left=pans(objects=[book]), right=pans(objects=[apple])),
                compare.balance(left=pans(objects=[apple]), right=pans(objects=[shoe])),
            ],
            alt="A chain of two balances",
        ),
        compare.compare_containers(
            containers=[
                {"label": "jug", "shape": "jug", "form": "tall_thin", "capacity": 8, "fill": 8, "cups": 8},
                {"label": "bowl", "shape": "bowl", "form": "short_wide", "capacity": 8, "fill": 5, "cups": 5},
                {"label": "bottle", "shape": "bottle", "form": "normal", "capacity": 4, "fill": 0},
                {"label": "cup", "shape": "cup", "form": "normal", "capacity": 2, "fill": 2},
            ],
            alt="Four containers",
        ),
    ]


def _grid_specimens() -> dict[str, dict]:
    """A square map with everything, an isometric stack with a canvas, and a shape drawn on its own grid."""
    return {
        "map": grid.grid_drawing(
            alt="A map on a square grid.",
            title="Map of Kiama",
            grid={"type": "square", "width": 8, "height": 6},
            cells=[
                {"cells": [[0, 0], [1, 0], [1, 1]], "style": "shaded"},
                {"cells": [[5, 4], [6, 4]], "style": "hatched"},
                {"cells": [[7, 0], [7, 1]], "style": "dotted"},
            ],
            polygons=[{"vertices": [[1, 2], [4, 2], [4, 4]], "style": "dashed", "label": "Oval"}],
            lines=[
                {"points": [[0, 3], [3, 3], [3, 6]], "style": "road"},
                {"points": [[0, 5], [4, 5], [8, 4]], "style": "river"},
                {"points": [[4, 0], [4, 2]], "style": "rail"},
                {"points": [[5, 0], [8, 0]], "style": "wall"},
                {"points": [[5, 2], [8, 2]], "style": "fence"},
                {"points": [[0, 1.5], [3, 1.5]], "style": "mirror"},
            ],
            symbols=[{"at": [1.5, 3.5], "symbol": "school"}, {"at": [6.5, 3.5], "symbol": "post_office"}],
            legend=True,
            column_labels=list("ABCDEFGH"),
            row_labels=["1", "2", "3", "4", "5", "6"],
            labels=[{"at": [6, 3], "text": "Beach"}],
            path=[[0, 4], [1, 4], [1, 5]],
            scale={"text": "1 cm : 2 m", "bar_cells": 2},
            north=True,
        ),
        "stack": grid.grid_drawing(
            alt="A stack of cubes on isometric dot paper.",
            grid={"type": "isometric", "width": 3, "height": 3, "layers": 3},
            cubes=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 2, 0], [1, 2, 1], [2, 0, 0], [2, 1, 0]],
            canvas={"width": 5, "height": 4},
        ),
        "shape": grid.grid_drawing(
            alt="A shape on a square grid.",
            grid={"type": "square", "width": 6, "height": 6},
            cells=[{"cells": [[1, 1], [2, 1], [2, 2]], "style": "shaded"}],
            canvas="grid",
        ),
    }


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
        scene.shape_scene(
            alt="Shapes",
            layout="grid",
            cols=3,
            items=[
                scene.scene_item("square", size="large", style="shaded", turn=45, label="A"),
                scene.scene_item("semicircle", style="dotted"),
                scene.scene_item("cone", size="small"),
            ],
        ),
        scene.shape_scene(alt="A house", object="house"),
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
        tally.tally(
            alt="Tally with counts and pictures",
            rows=[("dogs", 14), ("cats", 0), ("fish", 20)],
            show_counts=True,
            pictures={"symbol": "heart", "each": 2, "counts": [7, 0, 10]},
            title="Pets",
        ),
        tally.tally(
            alt="Objects and a tally",
            items=[(name, 0.14 * i, 0.33 * (i % 3)) for i, name in enumerate(tally.ICONS)],
            rows=[("apples", 3), ("cats", 5)],
        ),
        calendar_kinds.calendar_month(
            year=2026,
            month=10,
            alt="A calendar",
            marks=[
                {"day": 9, "style": "star", "label": "Party"},
                {"day": 16, "style": "ring", "label": "Swim"},
                {"day": 21, "style": "shade"},
            ],
        ),
        calendar_kinds.calendar_month(year=2024, month=2, alt="A leap February", week_start="sun", show_year=True),
        calendar_kinds.day_strip(cards=["wake_up", "breakfast", "school", "play", "dinner", "bed"], alt="A day"),
        calendar_kinds.day_strip(cards=["sunrise", "bath", "moon", "lunch"], alt="Scrambled", numbered=True),
        calendar_kinds.day_strip(cards=["bed"], alt="One card", numbered=True),
        *_compare_samples(),
        *_grid_specimens().values(),
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


# ---------------------------------------------------------------- grid drawings


def _grid_root(fig):
    return ET.fromstring(svg.figure_svg(fig))


def _assert_inside_the_drawing(root) -> None:
    """Every x coordinate outside a scaled symbol (whose coordinates are local to it) lies in the 400 units."""

    def walk(el):
        if el.get("transform") or el.tag == SVG_NS + "pattern":
            return
        for key in ("x", "x1", "x2", "cx"):
            if el.get(key) is not None:
                assert -1 <= float(el.get(key)) <= 401, (key, el.tag, el.get(key))
        for child in el:
            walk(child)

    walk(root)


def _grid_texts(fig) -> list[str]:
    return [t.text for t in _grid_root(fig).iter(SVG_NS + "text")]


@pytest.mark.parametrize("name", ["map", "stack", "shape"])
def test_grid_specimens_fit_the_drawing_area_and_draw_only_the_specs_own_text(name):
    fig = _grid_specimens()[name]
    root = _grid_root(fig)
    assert float(root.get("viewBox").split()[2]) == 400
    _assert_inside_the_drawing(root)
    shown = set(_grid_texts(fig))
    assert shown <= set(figures.labels(fig)) | {fig.get("title")}
    assert set(figures.labels(fig)[1:]) <= shown | {""}
    assert "<script" not in svg.figure_svg(fig) and "ZZidprefixZZ" not in svg.figure_svg(fig)


def _drawn_extent(root) -> tuple[float, float, float, float]:
    """(min x, min y, max x, max y) of everything drawn, with symbol tiles (local coordinates inside a
    ``translate scale`` group) taken as 11 local units either side of their centre, and text boxes estimated."""
    import re

    xs: list[float] = []
    ys: list[float] = []

    def walk(el, dx, dy):
        tf = el.get("transform") or ""
        t = re.fullmatch(r"translate\(([-\d.]+) ([-\d.]+)\)(?: scale\(([-\d.]+)\))?", tf)
        if t:
            ox, oy = dx + float(t.group(1)), dy + float(t.group(2))
            if t.group(3):  # a symbol: a tile of +-11 local units
                k = float(t.group(3))
                xs.extend((ox - 11 * k, ox + 11 * k))
                ys.extend((oy - 11 * k, oy + 11 * k))
                return
            dx, dy = ox, oy
        elif tf:
            return  # a rotated text or pattern: not part of the check
        if el.tag == SVG_NS + "pattern":
            return
        get = lambda k: float(el.get(k)) if el.get(k) is not None else None  # noqa: E731
        if el.tag == SVG_NS + "rect" and get("width") is not None and get("x") is not None:
            xs.extend((dx + get("x"), dx + get("x") + get("width")))
            ys.extend((dy + get("y"), dy + get("y") + get("height")))
        elif el.tag == SVG_NS + "line":
            xs.extend((dx + get("x1"), dx + get("x2")))
            ys.extend((dy + get("y1"), dy + get("y2")))
        elif el.tag == SVG_NS + "circle":
            xs.extend((dx + get("cx") - get("r"), dx + get("cx") + get("r")))
            ys.extend((dy + get("cy") - get("r"), dy + get("cy") + get("r")))
        elif el.tag == SVG_NS + "polygon" or el.tag == SVG_NS + "polyline":
            pts = [tuple(map(float, q.split(","))) for q in el.get("points").split()]
            xs.extend(dx + a for a, _ in pts)
            ys.extend(dy + b for _, b in pts)
        elif el.tag == SVG_NS + "text":
            size = float(el.get("font-size", 12))
            half = len(el.text or "") * 6.2 * size / 12 / 2
            x = float(el.get("x"))
            anchor = el.get("text-anchor", "middle")
            lo, hi = {"start": (x, x + 2 * half), "end": (x - 2 * half, x)}.get(anchor, (x - half, x + half))
            xs.extend((dx + lo, dx + hi))
            ys.extend((dy + float(el.get("y")) - size / 2, dy + float(el.get("y")) + size / 2))
        for child in el:
            walk(child, dx, dy)

    walk(root, 0, 0)
    return min(xs), min(ys), max(xs), max(ys)


def _assert_drawn_inside_the_viewbox(fig) -> None:
    root = _grid_root(fig)
    _, _, vw, vh = (float(v) for v in root.get("viewBox").split())
    x0, y0, x1, y1 = _drawn_extent(root)
    assert x0 >= -0.01 and y0 >= -0.01, (x0, y0)
    assert x1 <= vw + 0.01 and y1 <= vh + 0.01, (x1, y1, vw, vh)


def test_row_labels_with_a_wide_canvas_stay_inside_the_viewbox():
    for size, canvas in (((10, 8), (20, 4)), ((10, 8), (20, 20)), ((8, 6), (20, 2))):
        fig = grid.grid_drawing(
            alt="g",
            grid={"type": "square", "width": size[0], "height": size[1]},
            row_labels=[str(i) for i in range(1, size[1] + 1)],
            canvas={"width": canvas[0], "height": canvas[1]},
        )
        _assert_drawn_inside_the_viewbox(fig)


def test_long_row_labels_on_a_wide_grid_stay_inside_the_viewbox():
    fig = grid.grid_drawing(
        alt="g",
        grid={"type": "square", "width": 30, "height": 5},
        row_labels=[("R" * 23) + str(i) for i in range(5)],
        canvas={"width": 20, "height": 3},
    )
    _assert_drawn_inside_the_viewbox(fig)


@pytest.mark.parametrize("north", [False, True])
def test_symbols_and_labels_on_the_grid_boundary_are_not_clipped(north):
    for w, h in ((10, 10), (30, 30), (4, 3)):
        fig = grid.grid_drawing(
            alt="g",
            grid={"type": "square", "width": w, "height": h},
            symbols=[
                {"at": [0, 0], "symbol": "house"},
                {"at": [w, h], "symbol": "school"},
                {"at": [0, h], "symbol": "tree"},
                {"at": [w, 0], "symbol": "hospital"},
            ],
            labels=[{"at": [0, h / 2], "text": "Park lands"}, {"at": [w, h / 2], "text": "Estuary"}],
            north=north,
        )
        _assert_drawn_inside_the_viewbox(fig)


def test_grid_origin_is_bottom_left_so_row_zero_is_drawn_at_the_bottom():
    # 6 by 4 grid, no labels: the grid starts at (10, 10) and a cell is 36 units, so the rows are drawn
    # top to bottom at y = 10, 46, 82, 118 and the columns left to right at x = 10, 46, ..., 190.
    fig = grid.grid_drawing(
        alt="g",
        grid={"type": "square", "width": 6, "height": 4},
        cells=[{"cells": [[0, 0]], "style": "shaded"}, {"cells": [[5, 3]], "style": "hatched"}],
        polygons=[{"vertices": [[0, 0], [0, 4], [2, 0]], "style": "solid"}],
        labels=[{"at": [0, 0], "text": "O"}],
    )
    root = _grid_root(fig)
    squares = {
        (float(r.get("x")), float(r.get("y"))): r.get("fill")
        for r in _tag(root, "rect")
        if r.get("width") == "36" and r.get("height") == "36"
    }
    assert squares[(10.0, 118.0)] == svg.MID  # cell [0, 0]: left column, bottom row
    assert squares[(190.0, 10.0)].startswith("url(#")  # cell [5, 3]: right column, top row
    triangle = next(p for p in _tag(root, "polygon") if p.get("fill") in (None, "none") and "10,10" in p.get("points"))
    assert [tuple(map(float, q.split(","))) for q in triangle.get("points").split()] == [
        (10, 154),
        (10, 10),
        (82, 154),
    ]  # the lattice point [0, 0] is the bottom-left corner, [0, 4] the top-left
    note = next(t for t in _tag(root, "text") if t.text == "O")
    assert (float(note.get("x")), float(note.get("y"))) == (10, 154)


def test_grid_cell_styles_differ_by_more_than_tone():
    fig = _grid_specimens()["map"]
    text = svg.figure_svg(fig)
    root = ET.fromstring(text)
    patterns = _tag(root, "pattern")
    assert len(patterns) == 2
    inner = {next(p.iter(SVG_NS + "line"), None) is not None for p in patterns}
    assert inner == {True, False}  # one pattern is lines, the other dots
    fills = {
        r.get("fill") for r in _tag(root, "rect") if r.get("stroke") == "none" and r.get("width") == r.get("height")
    }
    assert svg.MID in fills and sum(f.startswith("url(#") for f in fills) == 2
    ids = {p.get("id") for p in patterns}
    assert all(f"url(#{i})" in fills for i in ids)
    # an unused style draws no pattern
    assert not _tag(_grid_root(_grid_specimens()["shape"]), "pattern")


def test_grid_cubes_are_three_greys_painted_back_to_front():
    fig = _grid_specimens()["stack"]
    root = _grid_root(fig)
    faces = [p for p in _tag(root, "polygon") if p.get("fill") in (svg.LIGHT, svg.MID, svg.DARK)]
    assert len(faces) == 3 * len(fig["cubes"])
    assert {p.get("fill") for p in faces} == {svg.LIGHT, svg.MID, svg.DARK}
    cubes = [g for g in _tag(root, "g") if len(g) == 3 and all(p.tag == SVG_NS + "polygon" for p in g)]
    assert len(cubes) == len(fig["cubes"])
    assert [p.get("fill") for p in cubes[0]] == [svg.DARK, svg.MID, svg.LIGHT]

    # Cubes in front (larger x, smaller y, larger z) come after the cubes behind them: check on a two-cube pair.
    pair = grid.grid_drawing(
        alt="Two cubes.",
        grid={"type": "isometric", "width": 2, "height": 2, "layers": 1},
        cubes=[[1, 0, 0], [0, 1, 0]],
    )
    order = [c[1].get("points") for c in _tag(_grid_root(pair), "g") if len(c) == 3]
    back = grid.grid_drawing(alt="One cube.", grid=pair["grid"], cubes=[[0, 1, 0]])
    back_points = next(c[1].get("points") for c in _tag(_grid_root(back), "g") if len(c) == 3)
    assert order[0] == back_points


def test_grid_canvas_is_below_a_wide_figure_and_beside_a_tall_one():
    def height(fig):
        return int(_grid_root(fig).get("viewBox").split()[3])

    wide = grid.grid_drawing(alt="Grid.", grid={"type": "square", "width": 8, "height": 4})
    wide_c = grid.grid_drawing(
        alt="Grid.", grid={"type": "square", "width": 8, "height": 4}, canvas={"width": 5, "height": 5}
    )
    assert height(wide_c) > height(wide) + 40
    tall = grid.grid_drawing(alt="Grid.", grid={"type": "square", "width": 4, "height": 8})
    tall_c = grid.grid_drawing(
        alt="Grid.", grid={"type": "square", "width": 4, "height": 8}, canvas={"width": 5, "height": 5}
    )
    assert height(tall_c) == height(tall)
    shape = _grid_specimens()["shape"]
    assert svg.figure_svg(shape) == svg.figure_svg({k: v for k, v in shape.items() if k != "canvas"})


@pytest.mark.parametrize("symbol", sorted(grid.SYMBOLS))
def test_every_grid_symbol_and_the_largest_grids_render(symbol):
    fig = grid.grid_drawing(
        alt="A map.",
        grid={"type": "square", "width": 30, "height": 30},
        symbols=[{"at": [1.5, 1.5], "symbol": symbol}],
        legend=True,
        column_labels=[str(i) for i in range(30)],
        row_labels=[str(i) for i in range(30)],
        canvas={"width": 20, "height": 20},
    )
    root = _grid_root(fig)
    assert grid.SYMBOLS[symbol] in _grid_texts(fig)
    _assert_inside_the_drawing(root)
    full = grid.grid_drawing(
        alt="Cubes.",
        grid={"type": "isometric", "width": 8, "height": 8, "layers": 8},
        cubes=[[x, y, 0] for x in range(8) for y in range(8)],
        canvas={"width": 20, "height": 20},
        north=True,
    )
    assert float(_grid_root(full).get("viewBox").split()[2]) == 400


def test_grid_hostile_text_is_escaped():
    evil = "<b>&amp;</b>"
    fig = grid.grid_drawing(
        alt=evil,
        grid={"type": "square", "width": 2, "height": 2},
        labels=[{"at": [1, 1], "text": evil}],
        scale={"text": evil},
        column_labels=[evil, "b"],
    )
    text = svg.figure_svg(fig)
    assert "<b>" not in text
    assert evil in _grid_texts(fig)
