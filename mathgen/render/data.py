"""SVG renderers for the statistical displays: bar_chart, picture_graph, dot_plot, histogram, box_plot and
pie_chart.

Every mark is the spec's own number run through a linear map; nothing is evaluated and no value the spec does
not label is printed. The geometry, wrapping and spacing follow the reference web renderer
(``DataFigures.tsx``) unit for unit, on a fixed 400-unit-wide drawing.

Wrapped text: a label that wraps is one ``<text>`` of several ``<tspan>`` lines. Every line but the last ends in
a space, so the element's text reads as the whole label ("Year 7 maths"), which is what a screen reader and the
label check see; the anchor is shifted by the width of that space so the glyphs sit where they would without it.
"""

from __future__ import annotations

import math

from mathgen.render._svg import W, Ctx, el, esc, fmt, g, points_attr, text, text_width

FS_TICK = 14
FS_LABEL = 15
LABEL_GAP = 12
MARGIN_R = 14
NICE_STRIDES = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
LINE_H = 16
SPACE_W = 0.28  # a space, in ems, for the anchor shift of wrapped lines

CSS = (
    ".mgf .mgf-dt-grid line{stroke:var(--_border-strong);stroke-width:1}"
    ".mgf .mgf-dt-grid .mgf-dt-minor{stroke-opacity:.4}"
    ".mgf .mgf-dt-grid .mgf-dt-half{stroke-dasharray:3 3;stroke-opacity:.7}"
    ".mgf .mgf-dt-bar{stroke:var(--_surface);stroke-width:1}"
    ".mgf .mgf-dt-hist-bar{fill-opacity:.8;stroke:var(--_tick);stroke-width:1.2}"
    ".mgf .mgf-dt-axis-minor text{font-weight:500}"
    ".mgf .mgf-dt-symbols g>*{fill:var(--_accent);stroke:var(--_accent);stroke-width:1px;stroke-linejoin:round;"
    "vector-effect:non-scaling-stroke}"
    ".mgf .mgf-dt-dot{fill:var(--_accent);stroke:var(--_tick);stroke-width:1}"
    ".mgf .mgf-dt-box{--bx:var(--_accent)}.mgf .mgf-dt-box-1{--bx:var(--_c1)}"
    ".mgf .mgf-dt-box-body{fill:var(--bx);fill-opacity:.28;stroke:var(--bx);stroke-width:2.2}"
    ".mgf .mgf-dt-whisker,.mgf .mgf-dt-median,.mgf .mgf-dt-outlier"
    "{stroke:var(--bx);stroke-width:2.2;fill:none;stroke-linecap:round}"
    ".mgf .mgf-dt-median{stroke-width:3.6}"
    ".mgf .mgf-dt-pie-sector{stroke:none}"
    ".mgf .mgf-dt-pie-radii line{stroke:var(--_surface);stroke-width:1.2;stroke-opacity:.9}"
    ".mgf .mgf-dt-pie-bounds line{stroke:var(--_surface);stroke-width:3}"
    ".mgf .mgf-dt-pie-rim{fill:none;stroke:var(--_tick);stroke-width:1}"
    ".mgf .mgf-dt-pie-leaders polyline{fill:none;stroke:var(--_tick);stroke-width:1}"
)


# ---------------------------------------------------------------- helpers


def _tw(s: str, fs: float = FS_TICK) -> float:
    return text_width(s, fs)


def _round6(v: float) -> float:
    return round(v, 6)


def _numeral(v: float) -> str:
    """A tick value: float noise rounded away, a proper minus sign (U+2212)."""
    r = _round6(float(v))
    s = f"{r:.6f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    return s.replace("-", "−")


def _ticks(lo: float, step: float, hi: float) -> list[float]:
    n = round((hi - lo) / step)
    return [_round6(lo + i * step) for i in range(n + 1)]


def _label_stride(spacing: float, w: float, gap: float = LABEL_GAP) -> int:
    """Every ``stride``-th tick is labelled: the smallest nice stride that leaves a clear gap."""
    need = math.ceil((w + gap) / max(spacing, 1e-6))
    return next((n for n in NICE_STRIDES if n >= need), need)


def _wrap(s: str, max_w: float, fs: float = FS_TICK) -> list[str]:
    """Greedy word wrap to ``max_w`` drawn units; a single over-long word stays whole."""
    words = s.split()
    if not words:
        return []
    lines: list[str] = []
    cur = ""
    for word in words:
        nxt = f"{cur} {word}" if cur else word
        if cur and _tw(nxt, fs) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = nxt
    lines.append(cur)
    return lines


def _widest(lines: list[list[str]], fs: float = FS_TICK) -> float:
    return max([0.0] + [_tw(ln, fs) for group in lines for ln in group])


def _lines(lines: list[str], x: float, y: float, anchor: str, lh: float = LINE_H, fs: float = FS_TICK) -> str:
    """One ``<text>`` of one or more lines (see the module docstring for the trailing-space convention)."""
    if len(lines) == 1:
        return text(x, y, lines[0], anchor=anchor)
    shift = {"start": 0.0, "middle": 0.5, "end": 1.0}[anchor] * SPACE_W * fs
    spans = []
    for i, ln in enumerate(lines):
        last = i == len(lines) - 1
        spans.append(
            el(
                "tspan",
                {"x": x + (0 if last else shift), "dy": 0 if i == 0 else lh},
                esc(ln if last else ln + " "),
            )
        )
    return el("text", {"x": x, "y": y, "text_anchor": anchor}, *spans)


def _line(x1: float, y1: float, x2: float, y2: float, **attrs) -> str:
    return el("line", {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **attrs})


def _frame(spec: dict, build) -> tuple[float, float, str]:
    """The common frame: the optional title, then ``build(top)`` -> (markup, bottom y)."""
    title_lines = _wrap(spec["title"], W - 20, FS_LABEL) if spec.get("title") else []
    title_h = len(title_lines) * 18 + 8 if title_lines else 0
    node, bottom = build(8 + title_h)
    height = math.ceil(bottom + 8)
    title = (
        g({"class_": "mgf-axis-label mgf-title"}, _lines(title_lines, W / 2, 22, "middle", 18, FS_LABEL))
        if title_lines
        else ""
    )
    return W, height, title + node


def _legend(names: list[str], y: float) -> str:
    sw = 14
    widths = [sw + 6 + _tw(n) for n in names]
    total = sum(widths) + 20 * (len(names) - 1)
    one_row = total <= W - 8
    out = []
    for i, n in enumerate(names):
        start = (W - total) / 2 + sum(w + 20 for w in widths[:i]) if one_row else 6
        row_y = y if one_row else y + i * 22
        out.append(
            el("rect", {"class_": f"mgf-fill-{i}", "x": start, "y": row_y - 12, "width": sw, "height": sw, "rx": 2})
        )
        out.append(text(start + sw + 6, row_y, n, anchor="start"))
    return g({"class_": "mgf-tick-label"}, *out)


def _legend_h(names: list[str]) -> float:
    total = sum(14 + 6 + _tw(s) for s in names) + 20 * (len(names) - 1)
    return 26 if total <= W - 8 else len(names) * 22 + 4


def _axis_label(s: str, x: float, y: float, anchor: str = "middle", extra: str = "") -> str:
    return g({"class_": f"mgf-axis-label{extra}"}, text(x, y, s, anchor=anchor))


# ---------------------------------------------------------------- bar chart


def _render_bar_chart(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    vertical = spec["orientation"] == "vertical"
    series = spec["series"]
    two = len(series) == 2
    cats = spec["categories"]
    n_cat = len(cats)
    axis = spec["value_axis"]
    ticks = _ticks(0, axis["step"], axis["max"])
    tick_texts = [_numeral(v) for v in ticks]
    tick_w = max(_tw(t) for t in tick_texts)
    names = [s["name"] for s in series]

    def build(top: float):
        y = top
        legend = ""
        if two:
            legend = _legend(names, y + 14)
            y += _legend_h(names)
        y_title = ""
        if spec.get("y_label"):
            y_title = _axis_label(spec["y_label"], 4, y + 14, "start")
            y += 22
        plot_top = y + 8

        if vertical:
            ph = 230
            plot_bottom = plot_top + ph
            plot_l = max(28, tick_w + 14)
            plot_r = W - max(MARGIN_R, 6)
            slot = (plot_r - plot_l) / n_cat

            def py(v):
                return plot_bottom - (v / axis["max"]) * ph

            stride = _label_stride(ph * axis["step"] / axis["max"], FS_TICK * 0.9)
            cat_lines = [_wrap(c, slot - 4) for c in cats]
            rotated = _widest(cat_lines) > slot - 2
            if rotated:
                label_zone = max(_tw(c) for c in cats) * 0.9 + 12
            else:
                label_zone = max(len(ln) for ln in cat_lines) * LINE_H + 8
            cat_bottom = plot_bottom + label_zone
            x_label_y = cat_bottom + 16
            group_w = slot * (0.8 if two else 0.66)
            bw = group_w / len(series)
            grid = [
                _line(plot_l, py(v), plot_r, py(v), class_=None if i % stride == 0 else "mgf-dt-minor")
                for i, v in enumerate(ticks)
            ]
            if axis["minor"]:
                grid += [
                    _line(plot_l, py(v + axis["step"] / 2), plot_r, py(v + axis["step"] / 2), class_="mgf-dt-half")
                    for v in ticks[:-1]
                ]
            bars = [
                el(
                    "rect",
                    {
                        "class_": f"mgf-dt-bar mgf-fill-{si}",
                        "x": plot_l + slot * i + (slot - group_w) / 2 + si * bw,
                        "y": py(v),
                        "width": bw,
                        "height": plot_bottom - py(v),
                    },
                )
                for si, s in enumerate(series)
                for i, v in enumerate(s["values"])
            ]
            tick_labels = [
                text(plot_l - 6, py(v) + 5, tick_texts[i], anchor="end") for i, v in enumerate(ticks) if i % stride == 0
            ]
            for i, c in enumerate(cats):
                cx = plot_l + slot * i + slot / 2
                if rotated:
                    tick_labels.append(
                        text(
                            0,
                            0,
                            c,
                            anchor="end",
                            transform=f"translate({fmt(cx + 5)},{fmt(plot_bottom + 8)}) rotate(-90)",
                        )
                    )
                else:
                    tick_labels.append(_lines(cat_lines[i], cx, plot_bottom + 18, "middle"))
            node = (
                legend
                + y_title
                + g({"class_": "mgf-dt-grid"}, *grid)
                + g(None, *bars)
                + g(
                    {"class_": "mgf-axis"},
                    _line(plot_l, plot_bottom, plot_r, plot_bottom),
                    _line(plot_l, py(axis["max"]), plot_l, plot_bottom),
                )
                + g({"class_": "mgf-tick-label"}, *tick_labels)
                + (_axis_label(spec["x_label"], (plot_l + plot_r) / 2, x_label_y) if spec.get("x_label") else "")
            )
            return node, (x_label_y + 4 if spec.get("x_label") else cat_bottom)

        # Horizontal: categories down the left, first at the TOP.
        cat_lines = [_wrap(c, 128) for c in cats]
        plot_l = min(190, max(_widest(cat_lines) + 16, 28))
        last_half = _tw(tick_texts[-1]) / 2
        plot_r = W - max(MARGIN_R, last_half + 3)
        row_h = max(44 if two else 30, max(len(ln) for ln in cat_lines) * LINE_H + 8)
        ph = row_h * n_cat
        plot_bottom = plot_top + ph

        def px(v):
            return plot_l + (v / axis["max"]) * (plot_r - plot_l)

        stride = _label_stride((plot_r - plot_l) * axis["step"] / axis["max"], _widest([tick_texts]))
        group_h = row_h * (0.8 if two else 0.62)
        bh = group_h / len(series)
        x_label_y = plot_bottom + 18 + 22
        grid = [
            _line(px(v), plot_top, px(v), plot_bottom, class_=None if i % stride == 0 else "mgf-dt-minor")
            for i, v in enumerate(ticks)
        ]
        if axis["minor"]:
            grid += [
                _line(px(v + axis["step"] / 2), plot_top, px(v + axis["step"] / 2), plot_bottom, class_="mgf-dt-half")
                for v in ticks[:-1]
            ]
        bars = [
            el(
                "rect",
                {
                    "class_": f"mgf-dt-bar mgf-fill-{si}",
                    "x": plot_l,
                    "y": plot_top + row_h * i + (row_h - group_h) / 2 + si * bh,
                    "width": px(v) - plot_l,
                    "height": bh,
                },
            )
            for si, s in enumerate(series)
            for i, v in enumerate(s["values"])
        ]
        tick_labels = [
            text(px(v), plot_bottom + 18, tick_texts[i], anchor="middle")
            for i, v in enumerate(ticks)
            if i % stride == 0
        ]
        for i, lines in enumerate(cat_lines):
            cy = plot_top + row_h * i + row_h / 2
            tick_labels.append(_lines(lines, plot_l - 8, cy - (len(lines) - 1) * LINE_H / 2 + 5, "end"))
        # The category axis is the vertical one: its title sits top left.
        h_title = (
            _axis_label(spec["y_label"], 4, top + (_legend_h(names) if two else 0) + 14, "start")
            if spec.get("y_label")
            else ""
        )
        node = (
            legend
            + h_title
            + g({"class_": "mgf-dt-grid"}, *grid)
            + g(None, *bars)
            + g(
                {"class_": "mgf-axis"},
                _line(plot_l, plot_bottom, plot_r, plot_bottom),
                _line(plot_l, plot_top, plot_l, plot_bottom),
            )
            + g({"class_": "mgf-tick-label"}, *tick_labels)
            + (_axis_label(spec["x_label"], (plot_l + plot_r) / 2, x_label_y) if spec.get("x_label") else "")
        )
        return node, (x_label_y + 4 if spec.get("x_label") else plot_bottom + 26)

    return _frame(spec, build)


# ------------------------------------------------------------ picture graph


def _symbol_shape(symbol: str) -> str:
    """Outline of each symbol in a unit box, scaled by the symbol size."""
    if symbol == "square":
        return el("rect", {"x": 0.06, "y": 0.06, "width": 0.88, "height": 0.88, "rx": 0.06})
    if symbol == "triangle":
        return el("polygon", {"points": "0.5,0.05 0.96,0.93 0.04,0.93"})
    if symbol == "diamond":
        return el("polygon", {"points": "0.5,0.02 0.96,0.5 0.5,0.98 0.04,0.5"})
    if symbol == "star":
        pts = []
        for k in range(10):
            r = 0.5 if k % 2 == 0 else 0.21
            a = math.pi * k / 5
            pts.append(f"{0.5 + r * math.sin(a):.4f},{0.54 - r * math.cos(a):.4f}")
        return el("polygon", {"points": " ".join(pts)})
    if symbol == "heart":
        return el(
            "path",
            {
                "d": "M0.5 0.93 C0.1 0.62 0.02 0.42 0.02 0.29 C0.02 0.13 0.14 0.05 0.27 0.05 C0.38 0.05 0.46 0.12 0.5 0.22 "
                "C0.54 0.12 0.62 0.05 0.73 0.05 C0.86 0.05 0.98 0.13 0.98 0.29 C0.98 0.42 0.9 0.62 0.5 0.93 Z"
            },
        )
    return el("circle", {"cx": 0.5, "cy": 0.5, "r": 0.46})


def _scaled(x: float, y: float, size: float) -> str:
    return f"translate({fmt(x)},{fmt(y)}) scale({size:.4f})"


def _render_picture_graph(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    key = spec["key"]
    key_text = f"Each symbol = {key['each']}" + (f" {key['unit']}" if key.get("unit") else "")
    rows = spec["rows"]
    symbol = spec["symbol"]
    half_id = ctx.uid("half")
    label_lines = [_wrap(r["label"], 120) for r in rows]
    plot_l = min(170, _widest(label_lines) + 18)
    avail = W - plot_l - MARGIN_R
    max_n = max([1] + [math.ceil(r["symbols"]) for r in rows])
    size = min(34, avail / max_n / 1.14)
    gap = size * 0.14
    row_h = max(size + 14, max(len(ln) for ln in label_lines) * LINE_H + 8)

    def build(top: float):
        y0 = top + 4
        bottom_rows = y0 + row_h * len(rows)
        key_y = bottom_rows + 30
        key_size = 22
        key_w = key_size + 8 + _tw(key_text, FS_LABEL)
        key_x = max(6, (W - key_w) / 2)
        grid = [
            _line(6, y0 + row_h * (i + 1), W - 6, y0 + row_h * (i + 1), class_="mgf-dt-minor") for i in range(len(rows))
        ]
        grid.append(_line(6, y0, W - 6, y0, class_="mgf-dt-minor"))
        labels = []
        marks = []
        for i, r in enumerate(rows):
            cy = y0 + row_h * i + row_h / 2
            lines = label_lines[i]
            labels.append(_lines(lines, plot_l - 10, cy - (len(lines) - 1) * LINE_H / 2 + 5, "end"))
            whole = math.floor(r["symbols"] + 1e-9)
            half = r["symbols"] - whole > 0.25
            shapes = [
                el("g", {"transform": _scaled(plot_l + k * (size + gap), cy - size / 2, size)}, _symbol_shape(symbol))
                for k in range(whole)
            ]
            if half:
                shapes.append(
                    el(
                        "g",
                        {
                            "transform": _scaled(plot_l + whole * (size + gap), cy - size / 2, size),
                            "clip_path": f"url(#{half_id})",
                        },
                        _symbol_shape(symbol),
                    )
                )
            marks.append(g({"class_": "mgf-dt-symbols"}, *shapes))
        node = (
            el(
                "defs",
                None,
                el("clipPath", {"id": half_id}, el("rect", {"x": 0, "y": -0.1, "width": 0.5, "height": 1.2})),
            )
            + g({"class_": "mgf-dt-grid"}, *grid)
            + g({"class_": "mgf-tick-label"}, *labels)
            + "".join(marks)
            + g(
                {"class_": "mgf-dt-symbols"},
                el("g", {"transform": _scaled(key_x, key_y - key_size / 2 - 4, key_size)}, _symbol_shape(symbol)),
            )
            + _axis_label(key_text, key_x + key_size + 8, key_y + 2, "start")
        )
        return node, key_y + 14

    return _frame(spec, build)


# ----------------------------------------------------------------- dot plot


def _render_dot_plot(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    lo, hi = spec["range"]
    ticks = _ticks(lo, spec["step"], hi)
    texts = [_numeral(v) for v in ticks]
    pad_l = max(16, _tw(texts[0]) / 2 + 4)
    pad_r = max(16, _tw(texts[-1]) / 2 + 4)
    plot_l = pad_l
    plot_r = W - pad_r
    spacing = (plot_r - plot_l) / (len(ticks) - 1)

    def px(v):
        return plot_l + ((v - lo) / (hi - lo)) * (plot_r - plot_l)

    r = min(8, spacing * 0.42)
    pitch = 2 * r + 1.5
    max_count = max(d["count"] for d in spec["dots"])
    stride = _label_stride(spacing, _widest([texts]))

    def build(top: float):
        stack_top = top + 6
        axis_y = stack_top + max_count * pitch + 6
        x_label_y = axis_y + 22 + 22
        grid = [
            _line(px(v), stack_top, px(v), axis_y, class_=None if i % stride == 0 else "mgf-dt-minor")
            for i, v in enumerate(ticks)
        ]
        dots = [
            el("circle", {"class_": "mgf-dt-dot", "cx": px(d["at"]), "cy": axis_y - 1 - r - k * pitch, "r": r})
            for d in spec["dots"]
            for k in range(d["count"])
        ]
        node = (
            g({"class_": "mgf-dt-grid"}, *grid)
            + g({"class_": "mgf-axis"}, _line(plot_l - 8, axis_y, plot_r + 8, axis_y))
            + g({"class_": "mgf-tick"}, *[_line(px(v), axis_y, px(v), axis_y + 6) for v in ticks])
            + g(None, *dots)
            + g(
                {"class_": "mgf-tick-label"},
                *[text(px(v), axis_y + 22, texts[i]) for i, v in enumerate(ticks) if i % stride == 0],
            )
            + (_axis_label(spec["x_label"], (plot_l + plot_r) / 2, x_label_y) if spec.get("x_label") else "")
        )
        return node, (x_label_y + 4 if spec.get("x_label") else axis_y + 30)

    return _frame(spec, build)


# ---------------------------------------------------------------- histogram


def _render_histogram(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    panels = spec["panels"]
    two = len(panels) == 2
    ph = 150 if two else 200

    def build(top: float):
        y = top
        nodes = []
        for pi, panel in enumerate(panels):
            axis = panel["frequency_axis"]
            y_ticks = _ticks(0, axis["step"], axis["max"])
            y_texts = [_numeral(v) for v in y_ticks]
            edges = panel["edges"]
            edge_texts = [_numeral(e) for e in edges]
            plot_l = max(28, max(_tw(t) for t in y_texts) + 14)
            plot_r = W - max(MARGIN_R, _tw(edge_texts[-1]) / 2 + 3)
            e0, e1 = edges[0], edges[-1]

            def px(v, plot_l=plot_l, plot_r=plot_r, e0=e0, e1=e1):
                return plot_l + ((v - e0) / (e1 - e0)) * (plot_r - plot_l)

            header = []
            if panel.get("label"):
                header.append(_axis_label(panel["label"], 4, y + 14, "start"))
                y += 22
            if spec.get("y_label"):
                header.append(_axis_label(spec["y_label"], 4, y + 14, "start", " mgf-dt-axis-minor"))
                y += 22
            plot_top = y + 8
            plot_bottom = plot_top + ph

            def py(v, plot_bottom=plot_bottom, axis=axis):
                return plot_bottom - (v / axis["max"]) * ph

            y_stride = _label_stride(ph * axis["step"] / axis["max"], FS_TICK * 0.9)
            x_stride = _label_stride((plot_r - plot_l) / (len(edges) - 1), _widest([edge_texts]), 6)
            grid = [
                _line(plot_l, py(v), plot_r, py(v), class_=None if i % y_stride == 0 else "mgf-dt-minor")
                for i, v in enumerate(y_ticks)
            ]
            if axis["minor"]:
                grid += [
                    _line(plot_l, py(v + axis["step"] / 2), plot_r, py(v + axis["step"] / 2), class_="mgf-dt-half")
                    for v in y_ticks[:-1]
                ]
            bars = [
                el(
                    "rect",
                    {
                        "class_": f"mgf-dt-bar mgf-dt-hist-bar mgf-fill-{pi}",
                        "x": px(edges[k]),
                        "y": py(f),
                        "width": px(edges[k + 1]) - px(edges[k]),
                        "height": plot_bottom - py(f),
                    },
                )
                for k, f in enumerate(panel["frequencies"])
            ]
            tick_labels = [
                text(plot_l - 6, py(v) + 5, y_texts[i], anchor="end")
                for i, v in enumerate(y_ticks)
                if i % y_stride == 0
            ]
            tick_labels += [
                text(px(e), plot_bottom + 22, edge_texts[i]) for i, e in enumerate(edges) if i % x_stride == 0
            ]
            last = pi == len(panels) - 1
            nodes.append(
                g(
                    None,
                    *header,
                    g({"class_": "mgf-dt-grid"}, *grid),
                    g(None, *bars),
                    g(
                        {"class_": "mgf-axis"},
                        _line(plot_l, plot_bottom, plot_r, plot_bottom),
                        _line(plot_l, py(axis["max"]), plot_l, plot_bottom),
                    ),
                    g({"class_": "mgf-tick"}, *[_line(px(e), plot_bottom, px(e), plot_bottom + 6) for e in edges]),
                    g({"class_": "mgf-tick-label"}, *tick_labels),
                    _axis_label(spec["x_label"], (plot_l + plot_r) / 2, plot_bottom + 22 + 22)
                    if spec.get("x_label") and last
                    else "",
                )
            )
            y = plot_bottom + 28 + (24 if spec.get("x_label") and last else 0) + (16 if pi == 0 and two else 0)
        return "".join(nodes), y - 6

    return _frame(spec, build)


# ----------------------------------------------------------------- box plot


def _render_box_plot(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    lo, hi = spec["range"]
    step = spec["step"]
    plots = spec["plots"]
    ticks = _ticks(lo, step, hi)
    texts = [_numeral(v) for v in ticks]
    label_lines = [_wrap(p["label"], 120) for p in plots]
    any_label = any(p["label"] for p in plots)
    plot_l = max(min(170, _widest(label_lines) + 18) if any_label else 0, _tw(texts[0]) / 2 + 6)
    plot_r = W - max(MARGIN_R, _tw(texts[-1]) / 2 + 4)

    def px(v):
        return plot_l + ((v - lo) / (hi - lo)) * (plot_r - plot_l)

    spacing = (plot_r - plot_l) / (len(ticks) - 1)
    stride = _label_stride(spacing, _widest([texts]))
    band = 62
    minor = bool(spec.get("minor"))

    def build(top: float):
        bands_top = top + 6
        axis_y = bands_top + band * len(plots)
        x_label_y = axis_y + 22 + 22
        grid = [
            _line(px(v), bands_top, px(v), axis_y, class_=None if i % stride == 0 else "mgf-dt-minor")
            for i, v in enumerate(ticks)
        ]
        if minor:
            grid += [
                _line(px(v + step / 2), bands_top, px(v + step / 2), axis_y, class_="mgf-dt-half") for v in ticks[:-1]
            ]
        boxes = []
        for i, p in enumerate(plots):
            cy = bands_top + band * i + band / 2
            parts = [
                _line(px(p["min"]), cy, px(p["q1"]), cy, class_="mgf-dt-whisker"),
                _line(px(p["q3"]), cy, px(p["max"]), cy, class_="mgf-dt-whisker"),
                _line(px(p["min"]), cy - 9, px(p["min"]), cy + 9, class_="mgf-dt-whisker"),
                _line(px(p["max"]), cy - 9, px(p["max"]), cy + 9, class_="mgf-dt-whisker"),
                el(
                    "rect",
                    {
                        "class_": "mgf-dt-box-body",
                        "x": px(p["q1"]),
                        "y": cy - 17,
                        "width": px(p["q3"]) - px(p["q1"]),
                        "height": 34,
                    },
                ),
                _line(px(p["median"]), cy - 17, px(p["median"]), cy + 17, class_="mgf-dt-median"),
            ]
            for o in p.get("outliers", ()):
                x = px(o)
                d = (
                    f"M{fmt(x - 5)},{fmt(cy - 5)} L{fmt(x + 5)},{fmt(cy + 5)} "
                    f"M{fmt(x - 5)},{fmt(cy + 5)} L{fmt(x + 5)},{fmt(cy - 5)}"
                )
                parts.append(el("path", {"class_": "mgf-dt-outlier", "d": d}))
            boxes.append(g({"class_": f"mgf-dt-box mgf-dt-box-{i}"}, *parts))
        ticks_g = [_line(px(v), axis_y, px(v), axis_y + 6) for v in ticks]
        if minor:
            ticks_g += [_line(px(v + step / 2), axis_y, px(v + step / 2), axis_y + 3.5) for v in ticks[:-1]]
        tick_labels = [text(px(v), axis_y + 22, texts[i]) for i, v in enumerate(ticks) if i % stride == 0]
        for i, lines in enumerate(label_lines):
            if lines:
                cy = bands_top + band * i + band / 2
                tick_labels.append(_lines(lines, plot_l - 10, cy - (len(lines) - 1) * LINE_H / 2 + 5, "end"))
        node = (
            g({"class_": "mgf-dt-grid"}, *grid)
            + "".join(boxes)
            + g({"class_": "mgf-axis"}, _line(plot_l - 6, axis_y, plot_r + 6, axis_y))
            + g({"class_": "mgf-tick"}, *ticks_g)
            + g({"class_": "mgf-tick-label"}, *tick_labels)
            + (_axis_label(spec["x_label"], (plot_l + plot_r) / 2, x_label_y) if spec.get("x_label") else "")
        )
        return node, (x_label_y + 4 if spec.get("x_label") else axis_y + 30)

    return _frame(spec, build)


# ---------------------------------------------------------------- pie chart


def _render_pie_chart(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    total = spec["parts"]
    cx = W / 2

    def polar(turn: float, rad: float) -> tuple[float, float]:
        return cx + rad * math.sin(turn * 2 * math.pi), rad * -math.cos(turn * 2 * math.pi)

    # Pick the radius so the widest wrapped label still fits beside the circle.
    radius = 112.0
    wrapped: list[list[str]] = []
    for _ in range(3):
        wrapped = [_wrap(s["label"], W / 2 - radius - 24) for s in spec["sectors"]]
        radius = max(56.0, min(112.0, W / 2 - 22 - _widest(wrapped)))
    R = radius

    sectors = []
    before = 0
    for i, s in enumerate(spec["sectors"]):
        a0, a1 = before / total, (before + s["parts"]) / total
        sectors.append({"a0": a0, "a1": a1, "mid": (a0 + a1) / 2, "i": i})
        before += s["parts"]

    # Labels in a local frame with the circle centre at y = 0.
    labels = []
    for s in sectors:
        lines = wrapped[s["i"]]
        sx = math.sin(s["mid"] * 2 * math.pi)
        sy = -math.cos(s["mid"] * 2 * math.pi)
        anchor = "start" if sx > 0.2 else "end" if sx < -0.2 else "middle"
        ex, ey = polar(s["mid"], R + 1)
        bx, by = polar(s["mid"], R + 9)
        bw = _widest([lines])
        bh = len(lines) * LINE_H
        ax = bx + (4 if anchor == "start" else -4 if anchor == "end" else 0)
        centre_y = (by + (-bh / 2 - 2 if sy < 0 else bh / 2 + 2)) if anchor == "middle" else by
        x0 = ax if anchor == "start" else ax - bw if anchor == "end" else ax - bw / 2
        labels.append(
            {
                "lines": lines,
                "ax": ax,
                "ay": centre_y - bh / 2 + 12,
                "anchor": anchor,
                "x0": x0,
                "x1": x0 + bw,
                "y0": centre_y - bh / 2,
                "y1": centre_y + bh / 2,
                "edge": (ex, ey),
                "elbow": (bx, by),
                "idx": s["i"],
            }
        )
    # Nudge apart labels that would collide (adjacent thin sectors).
    move = [0.0] * len(labels)
    for _ in range(12):
        moved = False
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                A, B = labels[a], labels[b]
                ay0, by0 = A["y0"] + move[a], B["y0"] + move[b]
                h_over = A["x0"] < B["x1"] + 4 and B["x0"] < A["x1"] + 4
                v_over = ay0 < B["y1"] + move[b] + 2 and by0 < A["y1"] + move[a] + 2
                if h_over and v_over:
                    a_centre = ay0 + (A["y1"] - A["y0"]) / 2
                    b_centre = by0 + (B["y1"] - B["y0"]) / 2
                    up, down = (a, b) if a_centre <= b_centre else (b, a)
                    need = labels[up]["y1"] + move[up] + 3 - (labels[down]["y0"] + move[down])
                    move[down] += need / 2
                    move[up] -= need / 2
                    moved = True
        if not moved:
            break
    placed = [
        {
            **lb,
            "ay": lb["ay"] + move[i],
            "y0": lb["y0"] + move[i],
            "y1": lb["y1"] + move[i],
            "shifted": abs(move[i]) > 3,
        }
        for i, lb in enumerate(labels)
    ]
    min_y = min([-R] + [lb["y0"] for lb in placed])
    max_y = max([R] + [lb["y1"] for lb in placed])

    def build(top: float):
        oy = top + 6 - min_y
        paths = []
        for s in sectors:
            x0, y0 = polar(s["a0"], R)
            x1, y1 = polar(s["a1"], R)
            large = 1 if s["a1"] - s["a0"] > 0.5 else 0
            d = f"M{fmt(cx)},0 L{fmt(x0)},{fmt(y0)} A{fmt(R)},{fmt(R)} 0 {large} 1 {fmt(x1)},{fmt(y1)} Z"
            paths.append(el("path", {"class_": f"mgf-dt-pie-sector mgf-fill-{s['i']}", "d": d}))
        radii = [_line(cx, 0, *polar(j / total, R)) for j in range(total)]
        bounds = [_line(cx, 0, *polar(s["a0"], R)) for s in sectors]
        leaders = []
        for lb in placed:
            if lb["shifted"]:
                dx = -2 if lb["anchor"] == "start" else 2 if lb["anchor"] == "end" else 0
                pts = [lb["edge"], lb["elbow"], (lb["ax"] + dx, (lb["y0"] + lb["y1"]) / 2)]
            else:
                pts = [lb["edge"], lb["elbow"]]
            leaders.append(el("polyline", {"points": points_attr(pts)}))
        node = el(
            "g",
            {"transform": f"translate(0,{fmt(oy)})"},
            *paths,
            g({"class_": "mgf-dt-pie-radii"}, *radii),
            g({"class_": "mgf-dt-pie-bounds"}, *bounds),
            el("circle", {"class_": "mgf-dt-pie-rim", "cx": cx, "cy": 0, "r": R}),
            g({"class_": "mgf-dt-pie-leaders"}, *leaders),
            g({"class_": "mgf-tick-label"}, *[_lines(lb["lines"], lb["ax"], lb["ay"], lb["anchor"]) for lb in placed]),
        )
        return node, oy + max_y + 4

    return _frame(spec, build)


from mathgen.render import register_renderer  # noqa: E402

register_renderer("bar_chart", _render_bar_chart)
register_renderer("picture_graph", _render_picture_graph)
register_renderer("dot_plot", _render_dot_plot)
register_renderer("histogram", _render_histogram)
register_renderer("box_plot", _render_box_plot)
register_renderer("pie_chart", _render_pie_chart)
