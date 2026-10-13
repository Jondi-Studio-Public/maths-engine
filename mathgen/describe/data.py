"""Descriptions for the data figure kinds.

Each describer states what the chart draws (axis range and step, labels, bar heights, symbol counts, dot
counts, class frequencies, box plot positions, wedge counts) and works nothing out from it.
"""

from __future__ import annotations

from mathgen.describe import count, join, num, register_describer, sentence


def _quote(text: str) -> str:
    return f"'{text}'"


def _axis_phrase(axis: dict) -> str:
    text = f"from 0 to {num(axis['max'])} in steps of {num(axis['step'])}"
    if axis.get("minor"):
        text += ", with a lighter line halfway between each pair of numbered lines"
    return text


# ---------------------------------------------------------------- bar chart


def _describe_bar_chart(spec: dict) -> str:
    cats, series, axis = spec["categories"], spec["series"], spec["value_axis"]
    vertical = spec["orientation"] == "vertical"
    two = len(series) == 2
    parts = [
        sentence(
            f"A bar chart with {count(len(cats), 'category', 'categories')}, drawn with "
            f"{'vertical columns' if vertical else 'horizontal bars'}"
            + (", each category having a pair of bars" if two else "")
        )
    ]
    if vertical:
        value_side, cat_side, value_label, cat_label = (
            "vertical",
            "horizontal",
            spec.get("y_label"),
            spec.get("x_label"),
        )
    else:
        value_side, cat_side, value_label, cat_label = (
            "horizontal",
            "vertical",
            spec.get("x_label"),
            spec.get("y_label"),
        )
    parts.append(sentence(f"The {value_side} axis is the value axis and is numbered {_axis_phrase(axis)}"))
    if value_label:
        parts.append(sentence(f"The {value_side} axis is labelled {_quote(value_label)}"))
    if cat_label:
        parts.append(sentence(f"The {cat_side} axis is labelled {_quote(cat_label)}"))
    if two:
        first, second = ("left", "right") if vertical else ("top", "bottom")
        parts.append(
            sentence(
                f"A legend names the two kinds of bar: {_quote(series[0]['name'])} is the {first} bar of each pair "
                f"and {_quote(series[1]['name'])} is the {second} bar"
            )
        )
    order = "left to right" if vertical else "top to bottom"
    parts.append(f"Each bar's {'height' if vertical else 'length'} is the value it reaches on the value axis.")
    items = []
    for i, c in enumerate(cats):
        if two:
            reach = " and ".join(f"{_quote(s['name'])} reaches {num(s['values'][i])}" for s in series)
        else:
            reach = f"reaches {num(series[0]['values'][i])}"
        items.append(f"{_quote(c)}: {reach}")
    parts.append(sentence(f"Categories from {order}: " + "; ".join(items)))
    return " ".join(parts)


# ---------------------------------------------------------------- picture graph


def _symbols_phrase(n, symbol: str) -> str:
    whole = int(n)
    half = n - whole > 1e-9
    plural = "diamonds" if symbol == "diamond" else symbol + "s"
    if not half:
        return "no symbols" if whole == 0 else count(whole, symbol)
    if whole == 0:
        return f"a half {symbol}"
    return f"{whole} and a half {plural}"


def _describe_picture_graph(spec: dict) -> str:
    symbol, key, rows = spec["symbol"], spec["key"], spec["rows"]
    key_text = f"Each symbol = {key['each']}" + (f" {key['unit']}" if key.get("unit") else "")
    parts = [
        sentence(f"A picture graph with {count(len(rows), 'row')}, each row made of {symbol} symbols"),
        sentence(f"The key reads {_quote(key_text)}, beside one whole {symbol}"),
        "A half symbol is the left half of a symbol.",
    ]
    items = [f"{_quote(r['label'])} has {_symbols_phrase(r['symbols'], symbol)}" for r in rows]
    parts.append(sentence("Rows from top to bottom: " + "; ".join(items)))
    return " ".join(parts)


# ---------------------------------------------------------------- dot plot


def _describe_dot_plot(spec: dict) -> str:
    low, high = spec["range"]
    parts = [
        sentence(
            f"A dot plot on a horizontal number line from {num(low)} to {num(high)}, with a tick and a numeral "
            f"every {num(spec['step'])}"
        )
    ]
    if spec.get("x_label"):
        parts.append(sentence(f"The axis is labelled {_quote(spec['x_label'])}"))
    parts.append("Each dot stands on the axis in a single column above its tick.")
    dots = [f"{num(d['count'])} above {num(d['at'])}" for d in spec["dots"]]
    parts.append(f"Dots: {join(dots)}.")
    parts.append("No other tick has a dot above it.")
    return " ".join(parts)


# ---------------------------------------------------------------- histogram


def _describe_histogram(spec: dict) -> str:
    panels = spec["panels"]
    two = len(panels) == 2
    parts = [
        sentence(
            "A histogram of equal-width classes whose bars touch"
            + (", in two panels one above the other, for comparing two sets of data" if two else "")
        )
    ]
    if spec.get("x_label"):
        parts.append(sentence(f"The horizontal axis is labelled {_quote(spec['x_label'])}"))
    if spec.get("y_label"):
        parts.append(sentence(f"The vertical axis is labelled {_quote(spec['y_label'])}"))
    ordinals = ("top", "bottom")
    for pi, panel in enumerate(panels):
        edges = panel["edges"]
        head = f"The {ordinals[pi]} panel is labelled {_quote(panel['label'])}." if two else ""
        axis = panel["frequency_axis"]
        axis_text = (
            f"The horizontal axis is numbered at every class boundary from {num(edges[0])} to {num(edges[-1])}. "
            f"The vertical frequency axis is numbered {_axis_phrase(axis)}."
        )
        if not two and panel.get("label"):
            head = f"The panel is labelled {_quote(panel['label'])}."
        classes = []
        for k, f in enumerate(panel["frequencies"]):
            classes.append(f"from {num(edges[k])} up to but not including {num(edges[k + 1])}, frequency {num(f)}")
        parts.append(" ".join(x for x in (head, axis_text) if x))
        parts.append(sentence("Classes left to right, with the height of each bar: " + "; ".join(classes)))
    return " ".join(parts)


# ---------------------------------------------------------------- box plot


def _plot_text(p: dict) -> str:
    text = (
        f"the left whisker ends at {num(p['min'])}, the box runs from {num(p['q1'])} to {num(p['q3'])} "
        f"with a line at {num(p['median'])}, and the right whisker ends at {num(p['max'])}"
    )
    out = p.get("outliers") or []
    if out:
        text += f". Crosses mark separate points at {join([num(v) for v in sorted(out)])}"
    return text


def _describe_box_plot(spec: dict) -> str:
    low, high = spec["range"]
    plots = spec["plots"]
    two = len(plots) == 2
    parts = [
        sentence(
            f"{'Two box-and-whisker plots, one above the other,' if two else 'A box-and-whisker plot'} on a "
            f"horizontal number line from {num(low)} to {num(high)}, with a tick and a numeral every "
            f"{num(spec['step'])}"
            + (" and a shorter unlabelled tick halfway between each pair" if spec.get("minor") else "")
        )
    ]
    if spec.get("x_label"):
        parts.append(sentence(f"The axis is labelled {_quote(spec['x_label'])}"))
    ordinals = ("The top plot", "The bottom plot")
    for i, p in enumerate(plots):
        who = ordinals[i] if two else "The plot"
        if p.get("label"):
            who += f", labelled {_quote(p['label'])},"
        parts.append(sentence(f"{who}: {_plot_text(p)}".replace(",:", ":")))
    return " ".join(parts)


# ---------------------------------------------------------------- pie chart


def _describe_pie_chart(spec: dict) -> str:
    parts_n, sectors = spec["parts"], spec["sectors"]
    items = [f"{_quote(s['label'])} covers {count(s['parts'], 'wedge')}" for s in sectors]
    return " ".join(
        [
            sentence(
                f"A pie chart: a circle cut into {parts_n} equal wedges, with a thin line at every wedge boundary"
            ),
            sentence(
                f"The circle is shaded in {count(len(sectors), 'sector')}, each a group of neighbouring wedges "
                "with its label beside it"
            ),
            sentence("Going clockwise from 12 o'clock: " + "; ".join(items)),
        ]
    )


register_describer("bar_chart", _describe_bar_chart)
register_describer("picture_graph", _describe_picture_graph)
register_describer("dot_plot", _describe_dot_plot)
register_describer("histogram", _describe_histogram)
register_describer("box_plot", _describe_box_plot)
register_describer("pie_chart", _describe_pie_chart)
