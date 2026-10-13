"""The statistical display figure kinds (`mathgen/figure_kinds/data.py`): what each builder returns, and what each
validator refuses because a learner could not read it honestly."""

import copy

import pytest

from mathgen import figures
from mathgen.figure_kinds import data


def _bar(**over):
    kw = dict(
        categories=["Ava", "Ben", "Chloe"],
        series=[{"name": "", "values": [4, 6, 10]}],
        value_max=10,
        value_step=2,
        alt="a column graph",
        x_label="Student",
        y_label="Books",
    )
    kw.update(over)
    return data.bar_chart(**kw)


def test_every_kind_is_registered():
    assert {"bar_chart", "picture_graph", "dot_plot", "histogram", "box_plot", "pie_chart"} <= set(figures.kinds())


# ------------------------------------------------------------------ bar chart


def test_bar_chart_builds_and_reports_its_text():
    fig = _bar()
    assert fig["kind"] == "bar_chart" and fig["orientation"] == "vertical"
    assert fig["value_axis"] == {"min": 0, "max": 10, "step": 2, "minor": False}
    shown = figures.labels(fig)
    assert "Ava" in shown and "Student" in shown and "Books" in shown


def test_bar_chart_two_series_need_a_legend_and_report_it():
    fig = _bar(series=[{"name": "Girls", "values": [4, 6, 10]}, {"name": "Boys", "values": [2, 8, 6]}])
    assert "Girls" in figures.labels(fig) and "Boys" in figures.labels(fig)
    with pytest.raises(ValueError, match="legend"):
        _bar(series=[{"name": "", "values": [4, 6, 10]}, {"name": "", "values": [2, 8, 6]}])


@pytest.mark.parametrize(
    "over,message",
    [
        (dict(series=[{"name": "", "values": [4, 7, 10]}]), "gridline"),  # 7 is between lines on a step-2 axis
        (dict(series=[{"name": "", "values": [4, 6, 12]}]), "above the axis maximum"),
        (dict(categories=["Ava", "Ava", "Chloe"]), "distinct"),
        (dict(categories=["Ava", "Ben"]), "one value per category"),
        (dict(value_max=9), "whole number of steps"),
        (dict(value_max=200, value_step=2), "gridlines"),
        (dict(orientation="diagonal"), "orientation"),
    ],
)
def test_bar_chart_refuses_what_cannot_be_read(over, message):
    with pytest.raises(ValueError, match=message):
        _bar(**over)


def test_bar_chart_half_lines_allow_half_step_values_only_when_declared():
    values = [{"name": "", "values": [5, 6, 10]}]
    with pytest.raises(ValueError, match="gridline"):
        _bar(series=values)
    assert _bar(series=values, minor_lines=True)["value_axis"]["minor"] is True


def test_bar_chart_axis_must_start_at_zero():
    spec = _bar()
    spec["value_axis"]["min"] = 2
    assert any("start" in p or "zero" in p for p in figures.validate(spec))


# ------------------------------------------------------------------ picture graph


def _pic(**over):
    kw = dict(categories=["Dog", "Cat", "Bird"], symbols=[3, 2.5, 1], each=2, symbol="star", alt="a picture graph")
    kw.update(over)
    return data.picture_graph(**kw)


def test_picture_graph_key_text_is_reported():
    fig = _pic(unit="students")
    assert "Each symbol = 2 students" in figures.labels(fig)
    assert "Dog" in figures.labels(fig)


def test_picture_graph_half_symbol_needs_an_even_key():
    with pytest.raises(ValueError, match="half symbol"):
        _pic(each=5)
    assert _pic(each=10)["key"]["each"] == 10


@pytest.mark.parametrize(
    "over,message",
    [
        (dict(symbols=[3, 2.3, 1]), "whole or half"),
        (dict(symbols=[13, 2, 1]), "more than 12"),
        (dict(symbol="dragon"), "symbol"),
        (dict(each=0), "key.each"),
        (dict(categories=["Dog", "Dog", "Bird"]), "distinct"),
    ],
)
def test_picture_graph_refuses_what_cannot_be_read(over, message):
    with pytest.raises(ValueError, match=message):
        _pic(**over)


# ------------------------------------------------------------------ dot plot


def test_dot_plot_builds_sorted_and_validates_ticks():
    fig = data.dot_plot(low=0, high=10, step=2, dots={6: 3, 2: 1}, alt="a dot plot", x_label="Goals")
    assert [d["at"] for d in fig["dots"]] == [2, 6]
    for dots, message in (({3: 1}, "tick"), ({12: 1}, "tick"), ({2: 13}, "from 1 to 12"), ({}, "non-empty")):
        with pytest.raises(ValueError, match=message):
            data.dot_plot(low=0, high=10, step=2, dots=dots, alt="a dot plot")
    with pytest.raises(ValueError, match="40 ticks"):
        data.dot_plot(low=0, high=100, step=1, dots={1: 1}, alt="a dot plot")


# ------------------------------------------------------------------ histogram


def _panel(**over):
    kw = dict(edges=[0, 10, 20, 30], frequencies=[4, 8, 2], freq_max=10, freq_step=2)
    kw.update(over)
    return data.histogram_panel(**kw)


def test_histogram_one_and_two_panels():
    one = data.histogram(panels=[_panel()], alt="a histogram", x_label="Mark")
    assert one["panels"][0]["frequency_axis"]["step"] == 2
    two = data.histogram(panels=[_panel(label="Group A"), _panel(label="Group B")], alt="two histograms")
    assert "Group A" in figures.labels(two)
    with pytest.raises(ValueError, match="distinct"):
        data.histogram(panels=[_panel(label="A"), _panel(label="A")], alt="two histograms")
    with pytest.raises(ValueError, match="one or two"):
        data.histogram(panels=[], alt="a histogram")


@pytest.mark.parametrize(
    "over,message",
    [
        (dict(edges=[0, 10, 21, 30]), "equal class width"),
        (dict(frequencies=[4, 8]), "one whole number per class"),
        (dict(frequencies=[4, 7, 2]), "gridline"),
        (dict(frequencies=[4, 12, 2]), "above the axis maximum"),
        (dict(edges=[0, 10]), "3 to 13"),
    ],
)
def test_histogram_refuses_what_cannot_be_read(over, message):
    with pytest.raises(ValueError, match=message):
        data.histogram(panels=[_panel(**over)], alt="a histogram")


# ------------------------------------------------------------------ box plot


def _box(**over):
    plot = dict(min=10, q1=20, median=30, q3=40, max=50, label="", outliers=[])
    plot.update(over)
    return data.box_plot(low=0, high=100, step=10, plots=[plot], alt="a box plot", x_label="Mark")


def test_box_plot_builds_and_accepts_an_outlier_beyond_the_fence():
    assert _box()["plots"][0]["median"] == 30
    # IQR 20, upper fence 70: 80 is an outlier, 60 is not
    assert _box(outliers=[80])["plots"][0]["outliers"] == [80]
    with pytest.raises(ValueError, match="not beyond"):
        _box(outliers=[60])


@pytest.mark.parametrize(
    "over,message",
    [
        (dict(median=35), "off the axis or between ticks"),
        (dict(q1=40, q3=40), "in order"),
        (dict(max=80), "fences"),  # a whisker end past the 1.5 x IQR fence is an outlier, not a whisker
        (dict(min=30, q1=20), "in order"),
        (dict(outliers=[80, 90, 100, 100, 100]), "outliers"),
    ],
)
def test_box_plot_refuses_what_cannot_be_read(over, message):
    with pytest.raises(ValueError, match=message):
        _box(**over)


def test_box_plot_two_plots_need_labels():
    plots = [
        dict(min=10, q1=20, median=30, q3=40, max=50, label="Class A"),
        dict(min=0, q1=20, median=40, q3=60, max=80, label="Class B"),
    ]
    fig = data.box_plot(low=0, high=100, step=10, plots=plots, alt="two box plots")
    assert "Class A" in figures.labels(fig)
    plots[1]["label"] = "Class A"
    with pytest.raises(ValueError, match="distinct"):
        data.box_plot(low=0, high=100, step=10, plots=plots, alt="two box plots")


def test_box_plot_half_ticks_only_when_declared():
    with pytest.raises(ValueError, match="between ticks"):
        _box(median=35)
    plot = dict(min=10, q1=20, median=35, q3=40, max=50, label="")
    assert data.box_plot(low=0, high=100, step=10, plots=[plot], alt="a box plot", minor_ticks=True)["minor"] is True


# ------------------------------------------------------------------ pie chart


def test_pie_chart_slices_add_to_the_circle():
    fig = data.pie_chart(
        parts=8,
        sectors=[{"label": "A", "parts": 4}, {"label": "B", "parts": 3}, {"label": "C", "parts": 1}],
        alt="a pie chart",
    )
    assert "B" in figures.labels(fig)
    for parts, sectors, message in (
        (8, [{"label": "A", "parts": 4}, {"label": "B", "parts": 3}], "sum"),
        (7, [{"label": "A", "parts": 4}, {"label": "B", "parts": 3}], "parts must be one of"),
        (8, [{"label": "A", "parts": 8}], "2 to 6"),
        (8, [{"label": "A", "parts": 4}, {"label": "A", "parts": 4}], "distinct"),
        (8, [{"label": "A", "parts": 0}, {"label": "B", "parts": 8}], "whole number"),
    ):
        with pytest.raises(ValueError, match=message):
            data.pie_chart(parts=parts, sectors=sectors, alt="a pie chart")


def test_alt_text_is_required_by_the_shared_validator():
    spec = copy.deepcopy(_bar())
    spec["alt"] = " "
    assert any("alt" in p for p in figures.validate(spec))
