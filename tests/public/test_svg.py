"""The SVG renderer draws every figure kind, deterministically, and escapes all text."""

import xml.etree.ElementTree as ET

import pytest

from mathgen import figures, svg
from mathgen.engine import load_skills, make_for_skill
from mathgen.figure_kinds import early, measure, space
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
