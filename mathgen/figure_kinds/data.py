"""Statistical display figure kinds. See the package docstring.

Kinds: ``bar_chart``, ``picture_graph``, ``dot_plot``, ``histogram``, ``box_plot`` and
``pie_chart``. A scatter plot or a line graph is a ``cartesian`` figure (``equal_scale=False``),
not a kind here.

Shared rules, enforced by the validators so a learner can always read a value honestly:

- A value axis starts at 0 where it carries counts or amounts (no truncated bars), has a
  ``step`` that divides its maximum, and draws at most 25 gridlines.
- A value that a question can ask for sits ON a gridline, or on the clearly lighter "half
  line" drawn midway between two gridlines when the axis says ``"minor": true``. A value
  between lines is rejected.
- Every figure's ``alt`` names the object only ("a column graph"), never a value.
- The front end draws exactly what the spec carries and evaluates nothing.
"""

from __future__ import annotations

from typing import Iterable

from mathgen.figures import Number, _clean, _is_num, _range_ok, check, register_kind

MAX_GRIDLINES = 25
SYMBOLS = ("circle", "square", "triangle", "star", "heart", "diamond")
PIE_PARTS = (2, 3, 4, 5, 6, 8, 10, 12)
MAX_LABEL = 22


# ---------------------------------------------------------------- shared checks


def _on_grid(value: Number, origin: Number, step: Number, minor: bool) -> bool:
    """``value`` is ``origin`` plus a whole number of steps (or of half steps when ``minor``)."""
    unit = step / 2 if minor else step
    q = (value - origin) / unit
    return abs(q - round(q)) < 1e-9


def _label_ok(text) -> bool:
    return isinstance(text, str) and bool(text.strip()) and len(text) <= MAX_LABEL


def _value_axis(axis, *, what: str) -> tuple[list[str], bool]:
    """Problems with a ``{"min", "max", "step", "minor"}`` axis dict, and whether it is usable."""
    if not isinstance(axis, dict):
        return [f"{what} must be a dict with min, max, step, minor"], False
    problems = []
    lo, hi, step = axis.get("min"), axis.get("max"), axis.get("step")
    if lo != 0:
        problems.append(f"{what}.min must be 0 (a bar starts at zero)")
    if not _is_num(hi) or not _is_num(step) or step <= 0 or hi <= 0:
        problems.append(f"{what}.max and {what}.step must be positive numbers")
        return problems, False
    if not _on_grid(hi, 0, step, False):
        problems.append(f"{what}.max must be a whole number of steps")
    if hi / step > MAX_GRIDLINES:
        problems.append(f"{what} has more than {MAX_GRIDLINES} gridlines; widen the step")
    if not isinstance(axis.get("minor"), bool):
        problems.append(f"{what}.minor must be a bool")
    return problems, not problems


def _values_on_axis(values, axis, what: str) -> list[str]:
    problems = []
    for i, v in enumerate(values):
        if not _is_num(v) or v < 0:
            problems.append(f"{what}[{i}] must be a number >= 0")
        elif v > axis["max"]:
            problems.append(f"{what}[{i}] = {v} is above the axis maximum")
        elif not _on_grid(v, 0, axis["step"], axis["minor"]):
            problems.append(f"{what}[{i}] = {v} is not on a gridline or half line")
    return problems


def _axis_dict(maximum: Number, step: Number, minor: bool) -> dict:
    return {"min": 0, "max": _clean(maximum), "step": _clean(step), "minor": bool(minor)}


# ---------------------------------------------------------------- bar chart


def bar_chart(
    *,
    categories: Iterable[str],
    series: Iterable[dict],
    value_max: Number,
    value_step: Number,
    alt: str,
    orientation: str = "vertical",
    minor_lines: bool = False,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
) -> dict:
    """A bar (column) graph with a gridded value axis; one series, or two for a comparison.

    - ``orientation``: ``"vertical"`` draws columns standing on the horizontal axis, the first
      category at the left, the value axis up the left edge with 0 at the bottom.
      ``"horizontal"`` draws bars growing rightwards from the vertical axis, the first
      category at the TOP, the value axis along the bottom with 0 at the left.
    - ``categories``: 1 to 10 distinct short names, one per bar (or bar pair), in drawn order.
    - ``series``: one or two ``{"name": str, "values": [num, ...]}``, one value per category.
      With two series each bar pair is drawn side by side, series 0 first (left or top), in
      two clearly different fills, and a legend lists the two ``name`` values; a single series
      has no legend (its ``name`` may be ``""``). At most six categories when there are two.
    - ``value_axis``: ``{"min": 0, "max", "step", "minor"}``. A gridline and a tick label at
      every ``step`` from 0 to ``max``. With ``minor`` true a lighter, unlabelled line is drawn
      midway between gridlines, and values may sit on those half lines. Every value is a
      multiple of ``step`` (of ``step / 2`` when ``minor``) and at most ``max``.
    - ``x_label`` / ``y_label``: titles of the horizontal and vertical axes (so for a
      horizontal chart ``x_label`` titles the value axis); ``title`` is optional.
    """
    spec = {
        "kind": "bar_chart",
        "alt": alt,
        "orientation": orientation,
        "categories": list(categories),
        "series": [{"name": s.get("name", ""), "values": [_clean(v) for v in s["values"]]} for s in series],
        "value_axis": _axis_dict(value_max, value_step, minor_lines),
        "x_label": x_label,
        "y_label": y_label,
    }
    if title:
        spec["title"] = title
    return check(spec)


def _validate_bar_chart(spec: dict) -> list[str]:
    problems = []
    if spec.get("orientation") not in ("vertical", "horizontal"):
        problems.append("orientation must be 'vertical' or 'horizontal'")
    cats = spec.get("categories")
    series = spec.get("series")
    if not isinstance(cats, list) or not 1 <= len(cats) <= 10:
        return problems + ["categories must be a list of 1 to 10 names"]
    if not all(_label_ok(c) for c in cats) or len(set(cats)) != len(cats):
        problems.append(f"categories must be distinct, non-empty names of at most {MAX_LABEL} characters")
    if not isinstance(series, list) or not 1 <= len(series) <= 2:
        return problems + ["series must be a list of one or two series"]
    if len(series) == 2:
        if len(cats) > 6:
            problems.append("two series allow at most six categories")
        names = [s.get("name") for s in series]
        if not all(_label_ok(n) for n in names) or names[0] == names[1]:
            problems.append("two series need distinct, non-empty names for the legend")
    axis_problems, ok = _value_axis(spec.get("value_axis"), what="value_axis")
    problems += axis_problems
    for si, s in enumerate(series):
        vals = s.get("values")
        if not isinstance(vals, list) or len(vals) != len(cats):
            problems.append(f"series[{si}].values must have one value per category")
        elif ok:
            problems += _values_on_axis(vals, spec["value_axis"], f"series[{si}].values")
    return problems


def _bar_texts(spec: dict) -> list[str]:
    out = list(spec.get("categories", []))
    if len(spec.get("series", [])) == 2:
        out += [s.get("name", "") for s in spec["series"]]
    return out


# ---------------------------------------------------------------- picture graph


def picture_graph(
    *,
    categories: Iterable[str],
    symbols: Iterable[Number],
    each: int,
    symbol: str,
    alt: str,
    unit: str = "",
    title: str = "",
) -> dict:
    """A picture graph (pictograph): one row of repeated symbols per category.

    - ``symbol``: one of ``circle``, ``square``, ``triangle``, ``star``, ``heart``,
      ``diamond``; the same symbol is drawn in every row.
    - ``key``: ``{"each": k, "unit": str}``. The key is drawn once as the text
      ``Each symbol = k`` (followed by ``unit`` when it is not empty, e.g. ``Each symbol = 2
      students``) beside one whole symbol. ``k`` is a whole number >= 1.
    - ``rows``: ``{"label": str, "symbols": n}`` in drawn order, the first row at the TOP, the
      label at the left. ``n`` counts whole symbols and may end in ``.5``, drawn as the LEFT
      half of the symbol; a half is only allowed when ``k`` is even, so a half symbol is a
      whole number of the thing counted. A row holds at most 12 symbols. A row of ``n``
      symbols stands for ``n * k``.
    """
    spec = {
        "kind": "picture_graph",
        "alt": alt,
        "symbol": symbol,
        "key": {"each": each, "unit": unit},
        "rows": [{"label": c, "symbols": _clean(n)} for c, n in zip(categories, symbols, strict=True)],
    }
    if title:
        spec["title"] = title
    return check(spec)


def _key_text(spec: dict) -> str:
    key = spec.get("key", {})
    unit = key.get("unit", "")
    return f"Each symbol = {key.get('each')}" + (f" {unit}" if unit else "")


def _validate_picture_graph(spec: dict) -> list[str]:
    problems = []
    if spec.get("symbol") not in SYMBOLS:
        problems.append(f"symbol must be one of {', '.join(SYMBOLS)}")
    key = spec.get("key")
    each = key.get("each") if isinstance(key, dict) else None
    if not (isinstance(each, int) and not isinstance(each, bool) and each >= 1):
        return problems + ["key.each must be a whole number >= 1"]
    if not isinstance(key.get("unit", ""), str):
        problems.append("key.unit must be a string")
    rows = spec.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10:
        return problems + ["rows must be a list of 1 to 10 rows"]
    labels = [r.get("label") for r in rows]
    if not all(_label_ok(lb) for lb in labels) or len(set(labels)) != len(labels):
        problems.append("row labels must be distinct, non-empty short names")
    for i, r in enumerate(rows):
        n = r.get("symbols")
        if not _is_num(n) or n < 0 or abs(n * 2 - round(n * 2)) > 1e-9:
            problems.append(f"rows[{i}].symbols must be a whole or half number >= 0")
            continue
        if n > 12:
            problems.append(f"rows[{i}] has more than 12 symbols")
        if abs(n - round(n)) > 1e-9 and each % 2:
            problems.append(f"rows[{i}] has a half symbol but each symbol stands for the odd number {each}")
    return problems


def _picture_texts(spec: dict) -> list[str]:
    return [r.get("label", "") for r in spec.get("rows", [])] + [_key_text(spec)]


# ---------------------------------------------------------------- dot plot


def dot_plot(
    *,
    low: Number,
    high: Number,
    step: Number,
    dots: dict | Iterable,
    alt: str,
    x_label: str = "",
    title: str = "",
) -> dict:
    """A dot plot (line plot): a number axis with a stack of dots above each value.

    - ``range``: ``[low, high]``, the ends of the horizontal axis. A tick and a numeral at
      every ``step`` from ``low`` (so ``high - low`` is a whole number of steps, at most 40).
    - ``dots``: ``{"at": value, "count": n}`` for each value that has data, in increasing
      order of ``at``. Each ``at`` is a tick (``low`` plus a whole number of steps). The ``n``
      dots are drawn as a single column of equal circles standing on the axis directly above
      the tick, one per observation. ``n`` is 1 to 12.
    - ``x_label``: what the axis measures; ``title`` is optional.
    """
    items = sorted(dots.items()) if isinstance(dots, dict) else sorted((d["at"], d["count"]) for d in dots)
    spec = {
        "kind": "dot_plot",
        "alt": alt,
        "range": [_clean(low), _clean(high)],
        "step": _clean(step),
        "dots": [{"at": _clean(v), "count": n} for v, n in items],
        "x_label": x_label,
    }
    if title:
        spec["title"] = title
    return check(spec)


def _validate_dot_plot(spec: dict) -> list[str]:
    r, step = spec.get("range"), spec.get("step")
    if not _range_ok(r):
        return ["range must be [low, high] with low < high"]
    if not _is_num(step) or step <= 0:
        return ["step must be a positive number"]
    problems = []
    if not _on_grid(r[1], r[0], step, False):
        problems.append("range must span a whole number of steps")
    if (r[1] - r[0]) / step > 40:
        problems.append("more than 40 ticks; widen the step")
    dots = spec.get("dots")
    if not isinstance(dots, list) or not dots:
        return problems + ["dots must be a non-empty list"]
    last = None
    for i, d in enumerate(dots):
        at, n = d.get("at"), d.get("count")
        if not _is_num(at) or not r[0] <= at <= r[1] or not _on_grid(at, r[0], step, False):
            problems.append(f"dots[{i}].at must be a tick on the axis")
            continue
        if last is not None and at <= last:
            problems.append("dots must be in strictly increasing order of at")
        last = at
        if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 12):
            problems.append(f"dots[{i}].count must be a whole number from 1 to 12")
    return problems


# ---------------------------------------------------------------- histogram


def histogram_panel(
    *,
    edges: Iterable[Number],
    frequencies: Iterable[int],
    freq_max: Number,
    freq_step: Number,
    minor_lines: bool = False,
    label: str = "",
) -> dict:
    """One panel for :func:`histogram`: see that function for what each key draws."""
    return {
        "label": label,
        "edges": [_clean(e) for e in edges],
        "frequencies": [int(f) for f in frequencies],
        "frequency_axis": _axis_dict(freq_max, freq_step, minor_lines),
    }


def histogram(
    *,
    panels: Iterable[dict],
    alt: str,
    x_label: str = "",
    y_label: str = "Frequency",
    title: str = "",
) -> dict:
    """A histogram of a numerical variable in equal-width classes; two panels compare two data sets.

    - ``panels``: one or two panels (build each with :func:`histogram_panel`), the first drawn at
      the TOP. Two panels are stacked, each with its own axes, and its ``label`` written above it
      (a single panel's ``label`` may be ``""``). Each panel is
      ``{"label", "edges", "frequencies", "frequency_axis"}``:
    - ``edges``: the class boundaries in increasing order, one class width apart (3 to 13 numbers,
      so 2 to 12 classes). Class ``k`` is drawn between ``edges[k]`` and ``edges[k+1]`` and a data
      value equal to ``edges[k+1]`` belongs to class ``k + 1`` (lower end included, upper end
      excluded). The horizontal axis is a continuous number line with a tick and a numeral at
      every edge.
    - ``frequencies``: whole numbers, one per class; a bar of that height stands on each class
      interval. Adjacent bars TOUCH (no gaps); a bar of height 0 is simply empty.
    - ``frequency_axis``: ``{"min": 0, "max", "step", "minor"}``, drawn up the panel's left edge
      with a gridline and numeral at every ``step`` (and unlabelled half lines when ``minor``).
      Every frequency sits on a gridline, or on a half line when ``minor``.
    - ``x_label`` / ``y_label``: axis titles shared by the panels; ``title`` is optional.
    """
    spec = {
        "kind": "histogram",
        "alt": alt,
        "panels": [dict(p) for p in panels],
        "x_label": x_label,
        "y_label": y_label,
    }
    if title:
        spec["title"] = title
    return check(spec)


def _validate_histogram(spec: dict) -> list[str]:
    panels = spec.get("panels")
    if not isinstance(panels, list) or not 1 <= len(panels) <= 2:
        return ["panels must be a list of one or two panels"]
    problems = []
    if len(panels) == 2:
        names = [p.get("label") for p in panels]
        if not all(_label_ok(n) for n in names) or names[0] == names[1]:
            problems.append("two panels need distinct, non-empty labels")
    for pi, panel in enumerate(panels):
        edges, freqs = panel.get("edges"), panel.get("frequencies")
        where = f"panels[{pi}]"
        if not isinstance(edges, list) or not 3 <= len(edges) <= 13 or not all(_is_num(e) for e in edges):
            problems.append(f"{where}.edges must be 3 to 13 numbers")
            continue
        width = edges[1] - edges[0]
        if width <= 0 or any(abs((b - a) - width) > 1e-9 for a, b in zip(edges, edges[1:])):
            problems.append(f"{where}.edges must increase by one equal class width")
        if not isinstance(freqs, list) or len(freqs) != len(edges) - 1:
            problems.append(f"{where}.frequencies must have one whole number per class")
            continue
        if not all(isinstance(f, int) and not isinstance(f, bool) and f >= 0 for f in freqs):
            problems.append(f"{where}.frequencies must be whole numbers >= 0")
            continue
        axis_problems, ok = _value_axis(panel.get("frequency_axis"), what=f"{where}.frequency_axis")
        problems += axis_problems
        if ok:
            problems += _values_on_axis(freqs, panel["frequency_axis"], f"{where}.frequencies")
    return problems


def _histogram_texts(spec: dict) -> list[str]:
    return [p.get("label", "") for p in spec.get("panels", [])]


# ---------------------------------------------------------------- box plot


def box_plot(
    *,
    low: Number,
    high: Number,
    step: Number,
    plots: Iterable[dict],
    alt: str,
    minor_ticks: bool = False,
    x_label: str = "",
    title: str = "",
) -> dict:
    """One or two box-and-whisker plots (parallel when two) on a shared horizontal axis.

    - ``range`` ``[low, high]`` and ``step``: a tick and numeral at every ``step`` from
      ``low`` (a whole number of steps, at most 40). With ``minor_ticks`` true an unlabelled
      short tick is also drawn midway between ticks and values may sit on those.
    - ``plots``: one or two ``{"label": str, "min", "q1", "median", "q3", "max",
      "outliers": [...]}``, the first drawn at the TOP. ``label`` is drawn at the left (it
      may be ``""`` for a single plot). ``min`` and ``max`` are the whisker ends, i.e. the
      smallest and largest values that are NOT outliers. The box runs from ``q1`` to
      ``q3`` with a line at ``median``; whiskers run from ``min`` to ``q1`` and ``q3`` to
      ``max``. Each outlier is drawn as a separate point (a cross) on the plot's centre line.
    - Every number lies on a tick (or a half tick when ``minor_ticks``); ``min <= q1 <=
      median <= q3 <= max`` with ``q1 < q3``. An outlier lies beyond the usual fences
      (more than ``1.5 * IQR`` below ``q1`` or above ``q3``); the whisker ends lie inside
      them. This keeps the "outlier" a learner sees consistent with the 1.5 x IQR rule.
    - ``x_label``: what the axis measures; ``title`` is optional.
    """
    out = []
    for p in plots:
        item = {k: _clean(p[k]) for k in ("min", "q1", "median", "q3", "max")}
        item["label"] = p.get("label", "")
        item["outliers"] = [_clean(v) for v in p.get("outliers", ())]
        out.append(item)
    spec = {
        "kind": "box_plot",
        "alt": alt,
        "range": [_clean(low), _clean(high)],
        "step": _clean(step),
        "minor": bool(minor_ticks),
        "plots": out,
        "x_label": x_label,
    }
    if title:
        spec["title"] = title
    return check(spec)


def _validate_box_plot(spec: dict) -> list[str]:
    r, step = spec.get("range"), spec.get("step")
    if not _range_ok(r):
        return ["range must be [low, high] with low < high"]
    if not _is_num(step) or step <= 0:
        return ["step must be a positive number"]
    problems = []
    if not _on_grid(r[1], r[0], step, False):
        problems.append("range must span a whole number of steps")
    if (r[1] - r[0]) / step > 40:
        problems.append("more than 40 ticks; widen the step")
    minor = bool(spec.get("minor"))
    plots = spec.get("plots")
    if not isinstance(plots, list) or not 1 <= len(plots) <= 2:
        return problems + ["plots must be a list of one or two plots"]
    if len(plots) == 2:
        names = [p.get("label") for p in plots]
        if not all(_label_ok(n) for n in names) or names[0] == names[1]:
            problems.append("two plots need distinct, non-empty labels")
    for i, p in enumerate(plots):
        keys = ("min", "q1", "median", "q3", "max")
        nums = [p.get(k) for k in keys]
        if not all(_is_num(v) for v in nums):
            problems.append(f"plots[{i}] needs numeric min, q1, median, q3, max")
            continue
        if any(a > b for a, b in zip(nums, nums[1:])) or p["q1"] >= p["q3"]:
            problems.append(f"plots[{i}] five numbers must be in order with q1 < q3")
            continue
        outliers = p.get("outliers", [])
        if not isinstance(outliers, list) or not all(_is_num(v) for v in outliers):
            problems.append(f"plots[{i}].outliers must be a list of numbers")
            continue
        for v in nums + outliers:
            if not r[0] <= v <= r[1] or not _on_grid(v, r[0], step, minor):
                problems.append(f"plots[{i}] value {v} is off the axis or between ticks")
        iqr = p["q3"] - p["q1"]
        lo_fence, hi_fence = p["q1"] - 1.5 * iqr, p["q3"] + 1.5 * iqr
        if p["min"] < lo_fence or p["max"] > hi_fence:
            problems.append(f"plots[{i}] whisker ends must lie inside the 1.5 x IQR fences")
        for v in outliers:
            if lo_fence <= v <= hi_fence:
                problems.append(f"plots[{i}] outlier {v} is not beyond the 1.5 x IQR fences")
        if len(outliers) != len(set(outliers)) or len(outliers) > 4:
            problems.append(f"plots[{i}] outliers must be distinct, at most four")
    return problems


def _box_texts(spec: dict) -> list[str]:
    return [p.get("label", "") for p in spec.get("plots", [])]


# ---------------------------------------------------------------- pie chart


def pie_chart(*, parts: int, sectors: Iterable[dict], alt: str, title: str = "") -> dict:
    """A pie chart whose circle is cut into ``parts`` equal slices, so every size is countable.

    - ``parts``: 2, 3, 4, 5, 6, 8, 10 or 12. The circle is drawn divided into this many equal
      wedges, with a thin radius drawn at every wedge boundary, even inside a sector.
    - ``sectors``: 2 to 6 of ``{"label": str, "parts": k}`` in drawn order, the first starting
      at 12 o'clock and the rest following clockwise; each is ``k`` consecutive wedges (k >= 1)
      in its own fill, with its ``label`` written beside it. The ``k`` values sum to ``parts``.
      Sector sizes are therefore ``k / parts`` of the circle, countable without measuring.
    """
    spec = {
        "kind": "pie_chart",
        "alt": alt,
        "parts": parts,
        "sectors": [{"label": s["label"], "parts": s["parts"]} for s in sectors],
    }
    if title:
        spec["title"] = title
    return check(spec)


def _validate_pie_chart(spec: dict) -> list[str]:
    problems = []
    parts, sectors = spec.get("parts"), spec.get("sectors")
    if parts not in PIE_PARTS:
        return [f"parts must be one of {PIE_PARTS}"]
    if not isinstance(sectors, list) or not 2 <= len(sectors) <= 6:
        return ["sectors must be a list of 2 to 6"]
    labels = [s.get("label") for s in sectors]
    if not all(_label_ok(lb) for lb in labels) or len(set(labels)) != len(labels):
        problems.append("sector labels must be distinct, non-empty short names")
    ks = [s.get("parts") for s in sectors]
    if not all(isinstance(k, int) and not isinstance(k, bool) and k >= 1 for k in ks):
        return problems + ["each sector needs a whole number of parts >= 1"]
    if sum(ks) != parts:
        problems.append("sector parts must sum to parts")
    return problems


def _pie_texts(spec: dict) -> list[str]:
    return [s.get("label", "") for s in spec.get("sectors", [])]


register_kind("bar_chart", _validate_bar_chart, texts=_bar_texts)
register_kind("picture_graph", _validate_picture_graph, texts=_picture_texts)
register_kind("dot_plot", _validate_dot_plot)
register_kind("histogram", _validate_histogram, texts=_histogram_texts)
register_kind("box_plot", _validate_box_plot, texts=_box_texts)
register_kind("pie_chart", _validate_pie_chart, texts=_pie_texts)
