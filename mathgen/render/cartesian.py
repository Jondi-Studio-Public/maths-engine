"""SVG renderers for the ``cartesian`` and ``number_line`` figure kinds.

No maths is evaluated beyond the linear map from the spec's coordinates to the drawing, and clipping an
infinite line to the plane. Tick labels are laid out first; curve, line, segment and point labels then
dodge them and each other using rough text boxes.
"""

from __future__ import annotations

import math

from mathgen.render._svg import W, Ctx, el, g, points_attr, text

PAD = 34
# Extra room below a plane so the x tick labels and the x-axis title both sit outside the grid.
PAD_B = 46
FS_TICK = 14
FS_LABEL = 15
# Clear space kept between neighbouring tick labels.
LABEL_GAP = 12
NL_H = 96
# Height of a plane whose axes are different quantities (equal_scale false).
UNEQUAL_SCALE_H = W * 0.75
NICE_STRIDES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
# Denominators a trig grid step reduces to, tried smallest first.
PI_DENOMINATORS = [1, 2, 3, 4, 6, 8, 12]
MINUS = "−"

CSS = ""

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1


def _round(v: float) -> int:
    """Round half up, as the reference does (Python's round() goes to even)."""
    return math.floor(v + 0.5)


def _nice_ticks(lo: float, hi: float, step: float) -> list[float]:
    out = []
    v = math.ceil(lo / step) * step
    while v <= hi + step * 1e-9:
        out.append(_round(v / step) * step)
        v += step
    return out


def _fmt_tick(v: float) -> str:
    r = _round(v * 1e6) / 1e6
    s = f"{r:.6f}".rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s.replace("-", MINUS)


def _fmt_pi_tick(v: float) -> str:
    if abs(v) < 1e-9:
        return "0"
    sign = MINUS if v < 0 else ""
    av = abs(v)
    for den in PI_DENOMINATORS:
        num = av * den
        rounded = _round(num)
        if rounded > 0 and abs(num - rounded) < 1e-4:
            d0 = math.gcd(rounded, den)
            n, d = rounded // d0, den // d0
            pi_part = "π" if n == 1 else f"{n}π"
            return sign + (pi_part if d == 1 else f"{pi_part}/{d}")
    return f"{sign}{_fmt_tick(av)}π"


def _label_stride(spacing: float, w: float) -> int:
    need = math.ceil((w + LABEL_GAP) / max(spacing, 1e-6))
    return next((n for n in NICE_STRIDES if n >= need), need)


def _edge_anchor(x: float, s: str, lo: float, hi: float) -> str:
    hw = len(s) * FS_TICK * 0.62 / 2
    if x - hw < lo:
        return "start"
    if x + hw > hi:
        return "end"
    return "middle"


def _text_rect(x: float, y: float, anchor: str, s: str, fs: float) -> Rect:
    w = len(s) * fs * 0.62
    x0 = x if anchor == "start" else x - w if anchor == "end" else x - w / 2
    return (x0, y - fs * 0.82, x0 + w, y + fs * 0.25)


def _overlaps(a: Rect, b: Rect, gap: float = 1.5) -> bool:
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def _inside(r: Rect, b: Rect) -> bool:
    return r[0] >= b[0] and r[2] <= b[2] and r[1] >= b[1] and r[3] <= b[3]


def _place_label(cands, s: str, fs: float, bounds: Rect, taken: list[Rect]):
    """First candidate inside ``bounds`` clear of ``taken``; else the first inside; else the first."""
    rects = [_text_rect(x, y, a, s, fs) for x, y, a in cands]
    k = next((i for i, r in enumerate(rects) if _inside(r, bounds) and not any(_overlaps(r, t) for t in taken)), -1)
    if k < 0:
        k = next((i for i, r in enumerate(rects) if _inside(r, bounds)), 0)
    taken.append(rects[k])
    return cands[k]


def _dot(cx: float, cy: float, open_: bool) -> str:
    cls = "mgf-dot mgf-dot-open" if open_ else "mgf-dot mgf-dot-closed"
    return el("circle", {"cx": cx, "cy": cy, "r": 5, "class_": cls})


def _clip_line(through, x_range, y_range):
    """Extend the line through two points to the plane's edges; None if it misses the plane."""
    (x1, y1), (x2, y2) = through
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return None
    t_lo, t_hi = -math.inf, math.inf
    for p, q in (
        (-dx, x1 - x_range[0]),
        (dx, x_range[1] - x1),
        (-dy, y1 - y_range[0]),
        (dy, y_range[1] - y1),
    ):
        if p == 0:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                t_lo = max(t_lo, r)
            else:
                t_hi = min(t_hi, r)
    if t_lo > t_hi:
        return None
    return [(x1 + t_lo * dx, y1 + t_lo * dy), (x1 + t_hi * dx, y1 + t_hi * dy)]


def _scales(x_range, y_range, equal_scale: bool, pad_l: float):
    (x0, x1), (y0, y1) = x_range, y_range
    if equal_scale:
        unit_x = unit_y = (W - pad_l - PAD) / max(x1 - x0, y1 - y0)
        h = (y1 - y0) * unit_y + PAD + PAD_B
        w = (x1 - x0) * unit_x + pad_l + PAD
    else:
        unit_x = (W - pad_l - PAD) / (x1 - x0)
        h = UNEQUAL_SCALE_H
        unit_y = (h - PAD - PAD_B) / (y1 - y0)
        w = W

    def px(x):
        return pad_l + (x - x0) * unit_x

    def py(y):
        return h - PAD_B - (y - y0) * unit_y

    return px, py, w, h


def _plane_left_pad(spec: dict) -> float:
    """Left padding wide enough for the y tick labels when they sit in the left margin."""
    if not spec["tick_labels"] or spec["x_range"][0] < 0:
        return PAD
    ticks = _nice_ticks(spec["y_range"][0], spec["y_range"][1], spec["y_grid"])
    widest = max([1] + [len(_fmt_tick(v)) for v in ticks])
    return max(PAD, math.ceil(widest * FS_TICK * 0.62 + 6 + 6))


def _render_cartesian(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    x0, x1 = spec["x_range"]
    y0, y1 = spec["y_range"]
    x_grid, y_grid = spec["x_grid"], spec["y_grid"]
    tick_labels = spec["tick_labels"]
    px, py, w, h = _scales(spec["x_range"], spec["y_range"], spec["equal_scale"], _plane_left_pad(spec))
    x_ticks = _nice_ticks(x0, x1, x_grid)
    y_ticks = _nice_ticks(y0, y1, y_grid)
    fmt_x = _fmt_pi_tick if spec["x_tick_format"] == "pi" else _fmt_tick
    show_x = y0 <= 0 <= y1
    show_y = x0 <= 0 <= x1
    both = show_x and show_y
    plane: Rect = (px(x0), py(y1), px(x1), py(y0))
    view: Rect = (2, 2, w - 2, h - 2)
    points, segments = spec["points"], spec["segments"]
    lines, curves = spec["lines"], spec["curves"]
    x_label, y_label = spec["x_label"], spec["y_label"]

    # Tick labels first; everything else dodges them.
    taken: list[Rect] = []
    x_axis_y = py(0 if show_x else y0)
    x_row_y = x_axis_y + 16
    x_stride = _label_stride(
        abs(px(x0 + x_grid) - px(x0)),
        max([1] + [len(fmt_x(v)) for v in x_ticks]) * FS_TICK * 0.62,
    )
    y_stride = _label_stride(abs(py(y0 + y_grid) - py(y0)), FS_TICK * 0.9)
    x_labels: list[tuple[str, float, float, str]] = []
    y_labels: list[tuple[str, float, float, str]] = []
    origin_label = None

    # Everything drawn that a tick label should keep clear of, as pixel polylines.
    strokes: list[list[tuple[float, float]]] = []
    for c in curves:
        for piece in c["pieces"]:
            strokes.append([(px(x), py(y)) for x, y in piece])
    clipped_lines = []
    for ln in lines:
        cl = _clip_line(ln["through"], spec["x_range"], spec["y_range"])
        clipped_lines.append(cl)
        if cl:
            strokes.append([(px(x), py(y)) for x, y in cl])
    for sg in segments:
        strokes.append([(px(sg["from"][0]), py(sg["from"][1])), (px(sg["to"][0]), py(sg["to"][1]))])
    dot_rects: list[Rect] = [
        (px(p["at"][0]) - 7, py(p["at"][1]) - 7, px(p["at"][0]) + 7, py(p["at"][1]) + 7) for p in points
    ]

    def struck(r: Rect) -> bool:
        if any(_overlaps(r, d, 0) for d in dot_rects):
            return True
        pad = 1
        for pl in strokes:
            for i in range(len(pl) - 1):
                ax, ay = pl[i]
                bx, by = pl[i + 1]
                n = max(1, math.ceil(math.hypot(bx - ax, by - ay) / 3))
                for k in range(n + 1):
                    x = ax + (bx - ax) * k / n
                    y = ay + (by - ay) * k / n
                    if r[0] - pad <= x <= r[2] + pad and r[1] - pad <= y <= r[3] + pad:
                        return True
        return False

    def settle(cands, s: str, gap: float = 3):
        """First candidate clear of the curves and placed labels; else clear of labels; else the first."""
        rs = [_text_rect(x, y, a, s, FS_TICK) for x, y, a in cands]
        k = next((i for i, r in enumerate(rs) if not struck(r) and not any(_overlaps(r, o, gap) for o in taken)), -1)
        if k < 0:
            k = next((i for i, r in enumerate(rs) if not any(_overlaps(r, o, gap) for o in taken)), 0)
        taken.append(rs[k])
        return cands[k]

    if tick_labels:
        for v in x_ticks:
            if (both and v == 0) or _round(v / x_grid) % x_stride != 0:
                continue
            t = fmt_x(v)
            a = _edge_anchor(px(v), t, view[0], view[2])
            c = settle([(px(v), x_row_y, a), (px(v), x_row_y + 5, a)], t)
            x_labels.append((t, c[0], c[1], c[2]))
        y_x = px(0 if show_y else x0) - 6
        for v in y_ticks:
            if (both and v == 0) or _round(v / y_grid) % y_stride != 0:
                continue
            t = _fmt_tick(v)
            base = py(v) + 5
            if any(_overlaps(_text_rect(y_x, base, "end", t, FS_TICK), o, 1.5) for o in taken):
                continue
            c = settle([(y_x, base, "end"), (y_x - 5, base, "end")], t, 1.5)
            y_labels.append((t, c[0], c[1], c[2]))
        if both:
            # The origin "0" sits left of the y axis, or right of it when "-1" crowds it.
            origin_label = settle([(px(0) - 6, py(0) + 16, "end"), (px(0) + 6, py(0) + 16, "start")], "0", 10)

    # Axis titles: below the tick labels on a bottom-edge axis; else past the right end of a mid-graph axis.
    x_title_below = (not show_x) or y0 == 0 or py(0) >= plane[3] - 1
    room_right = px(x1) + 11 + len(x_label) * FS_LABEL * 0.62 <= w - 2
    if x_title_below:
        x_title = (w - 4, x_axis_y + 16 + FS_LABEL + 2, "end")
    elif room_right:
        # A pi axis has a wide end label right under the axis end, so its title rides above the line.
        x_title = (px(x1) + 11, x_axis_y + (-4 if spec["x_tick_format"] == "pi" else 5), "start")
    else:
        x_title = (w - PAD, x_axis_y - 6, "end")
    y_title_w = len(y_label) * FS_LABEL * 0.64
    y_title = (max(2, min(px(0 if show_y else x0) + 6, w - 2 - y_title_w)), PAD - 9, "start")
    taken.append(_text_rect(x_title[0], x_title[1], x_title[2], x_label, FS_LABEL))
    taken.append(_text_rect(y_title[0], y_title[1], y_title[2], y_label, FS_LABEL))

    # Curve, line, segment and point labels, placed in that order.
    multi = len(curves) > 1
    curve_labels = []
    for i, c in enumerate(curves):
        if not c.get("label"):
            continue
        pts = [(x, y) for piece in c["pieces"] for x, y in piece if x0 <= x <= x1 and y0 <= y <= y1]
        if not pts:
            continue
        # Spread different curves' labels along their length so they do not share a crossing.
        f = 0.25 + 0.5 * ((i * 0.37) % 1) + 0.1
        cx, cy = pts[min(len(pts) - 1, math.floor(len(pts) * min(f, 0.9)))]
        X, Y = px(cx), py(cy)
        p = _place_label(
            [(X + 7, Y - 7, "start"), (X - 7, Y - 7, "end"), (X + 7, Y + 17, "start"), (X - 7, Y + 17, "end")],
            c["label"],
            FS_LABEL,
            plane,
            taken,
        )
        curve_labels.append((c["label"], p, i))
    line_labels = []
    for ln, cl in zip(lines, clipped_lines):
        if not (ln.get("label") and cl):
            continue
        X, Y = px(cl[1][0]), py(cl[1][1])
        p = _place_label(
            [(X - 5, Y - 7, "end"), (X - 5, Y + 17, "end"), (X + 5, Y - 7, "start")],
            ln["label"],
            FS_LABEL,
            plane,
            taken,
        )
        line_labels.append((ln["label"], p))
    seg_labels = []
    for s in segments:
        if not s.get("label"):
            continue
        X = (px(s["from"][0]) + px(s["to"][0])) / 2
        Y = (py(s["from"][1]) + py(s["to"][1])) / 2
        p = _place_label([(X, Y - 9, "middle"), (X, Y + 20, "middle")], s["label"], FS_LABEL, plane, taken)
        seg_labels.append((s["label"], p))
    # Dots are obstacles for point labels too.
    for p in points:
        X, Y = px(p["at"][0]), py(p["at"][1])
        taken.append((X - 6, Y - 6, X + 6, Y + 6))
    point_labels = []
    for pt in points:
        if not pt.get("label"):
            continue
        X, Y = px(pt["at"][0]), py(pt["at"][1])
        p = _place_label(
            [
                (X + 8, Y - 8, "start"),
                (X - 8, Y - 8, "end"),
                (X + 8, Y + 19, "start"),
                (X - 8, Y + 19, "end"),
                (X, Y - 11, "middle"),
            ],
            pt["label"],
            FS_LABEL,
            plane,
            taken,
        )
        point_labels.append((pt["label"], p))

    def grid_class(v, grid, stride):
        return "mgf-grid-minor" if tick_labels and stride > 1 and _round(v / grid) % stride != 0 else None

    clip_id = ctx.uid("clip")
    defs = el(
        "defs",
        None,
        el(
            "clipPath",
            {"id": clip_id},
            el("rect", {"x": px(x0), "y": py(y1), "width": px(x1) - px(x0), "height": py(y0) - py(y1)}),
        ),
    )
    grid = g(
        {"class_": "mgf-grid"},
        *[
            el(
                "line",
                {"class_": grid_class(v, x_grid, x_stride), "x1": px(v), "x2": px(v), "y1": py(y0), "y2": py(y1)},
            )
            for v in x_ticks
        ],
        *[
            el(
                "line",
                {"class_": grid_class(v, y_grid, y_stride), "x1": px(x0), "x2": px(x1), "y1": py(v), "y2": py(v)},
            )
            for v in y_ticks
        ],
    )
    axes = g(
        {"class_": "mgf-axis"},
        el("line", {"x1": px(x0), "x2": px(x1), "y1": py(0), "y2": py(0)}) if show_x else "",
        el("line", {"x1": px(0), "x2": px(0), "y1": py(y0), "y2": py(y1)}) if show_y else "",
    )
    drawn = []
    for cl in clipped_lines:
        if cl:
            (ax, ay), (bx, by) = cl
            drawn.append(
                g({"class_": "mgf-line"}, el("line", {"x1": px(ax), "y1": py(ay), "x2": px(bx), "y2": py(by)}))
            )
    for i, c in enumerate(curves):
        for piece in c["pieces"]:
            cls = f"mgf-curve mgf-curve-{i % 3}" if multi else "mgf-curve"
            drawn.append(el("polyline", {"class_": cls, "points": points_attr((px(x), py(y)) for x, y in piece)}))
    for s in segments:
        parts = [
            el("line", {"x1": px(s["from"][0]), "y1": py(s["from"][1]), "x2": px(s["to"][0]), "y2": py(s["to"][1])})
        ]
        # An endpoint dot only where the spec says open or closed (step graphs).
        if s.get("open_start") is not None:
            parts.append(_dot(px(s["from"][0]), py(s["from"][1]), bool(s["open_start"])))
        if s.get("open_end") is not None:
            parts.append(_dot(px(s["to"][0]), py(s["to"][1]), bool(s["open_end"])))
        drawn.append(g({"class_": "mgf-segment"}, *parts))
    for p in points:
        drawn.append(g({"class_": "mgf-point"}, _dot(px(p["at"][0]), py(p["at"][1]), bool(p.get("open")))))
    plot = g({"clip_path": f"url(#{clip_id})"}, *drawn)

    # All text last, on a halo, so no curve strikes through it.
    def texts(items, **extra):
        return [text(x, y, t, anchor=a, **extra) for t, x, y, a in items]

    tick_group = ""
    if tick_labels:
        items = x_labels + y_labels
        if origin_label:
            items = items + [("0", origin_label[0], origin_label[1], origin_label[2])]
        tick_group = g({"class_": "mgf-tick-label"}, *texts(items))
    axis_titles = g(
        {"class_": "mgf-axis-label"},
        text(x_title[0], x_title[1], x_label, anchor=x_title[2]),
        text(y_title[0], y_title[1], y_label, anchor=y_title[2]),
    )
    data = [
        text(p[0], p[1], t, anchor=p[2], class_=f"mgf-curve-text-{i % 3}" if multi else "mgf-curve-text-0")
        for t, p, i in curve_labels
    ]
    data += [text(p[0], p[1], t, anchor=p[2]) for t, p in line_labels + seg_labels + point_labels]
    data_group = g({"class_": "mgf-data-label"}, *data)
    return w, h, defs + grid + axes + plot + tick_group + axis_titles + data_group


def _render_number_line(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    r0, r1 = spec["range"]
    tick = spec["tick"]
    y = 36
    arrow_margin = 14
    scale = (W - 2 * PAD) / (r1 - r0)

    def px(x):
        return PAD + (x - r0) * scale

    line_start = PAD - arrow_margin / 2
    line_end = W - PAD + arrow_margin / 2
    ticks = _nice_ticks(r0, r1, tick)
    stride = _label_stride(scale * tick, max([1] + [len(_fmt_tick(v)) for v in ticks]) * FS_TICK * 0.62)
    arrow_id = ctx.uid("arrow")
    arrow_ref = f"url(#{arrow_id})"

    defs = el(
        "defs",
        None,
        el(
            "marker",
            {"id": arrow_id, "markerWidth": 8, "markerHeight": 8, "refX": 4, "refY": 4, "orient": "auto"},
            el("path", {"d": "M0,0 L8,4 L0,8 Z", "class_": "mgf-arrowhead"}),
        ),
    )
    rule = el(
        "line",
        {
            "x1": line_start,
            "x2": line_end,
            "y1": y,
            "y2": y,
            "class_": "mgf-axis mgf-number-line-rule",
            "marker-start": arrow_ref,
            "marker-end": arrow_ref,
        },
    )
    tick_marks = g(
        {"class_": "mgf-tick"}, *[el("line", {"x1": px(v), "x2": px(v), "y1": y - 6, "y2": y + 6}) for v in ticks]
    )
    labels = ""
    if spec["tick_labels"]:
        labels = g(
            {"class_": "mgf-tick-label"},
            *[
                text(px(v), y + 26, _fmt_tick(v), anchor=_edge_anchor(px(v), _fmt_tick(v), 2, W - 2))
                for v in ticks
                if _round(v / tick) % stride == 0
            ],
        )
    intervals = [
        g(
            {"class_": "mgf-interval"},
            el("line", {"x1": px(iv["from"]), "x2": px(iv["to"]), "y1": y, "y2": y}),
            _dot(px(iv["from"]), y, bool(iv.get("open_start"))),
            _dot(px(iv["to"]), y, bool(iv.get("open_end"))),
        )
        for iv in spec["intervals"]
    ]
    rays = [
        g(
            {"class_": "mgf-ray"},
            el(
                "line",
                {
                    "x1": px(ray["from"]),
                    "x2": line_end if ray["direction"] == "right" else line_start,
                    "y1": y,
                    "y2": y,
                },
            ),
            _dot(px(ray["from"]), y, bool(ray.get("open"))),
        )
        for ray in spec["rays"]
    ]
    marks = []
    for m in spec["marks"]:
        parts = [_dot(px(m["at"]), y, bool(m.get("open")))]
        if m.get("label"):
            parts.append(text(px(m["at"]), y - 14, m["label"], anchor=_edge_anchor(px(m["at"]), m["label"], 2, W - 2)))
        marks.append(g({"class_": "mgf-mark"}, *parts))
    return W, NL_H, defs + rule + tick_marks + labels + "".join(intervals + rays + marks)


from mathgen.render import register_renderer  # noqa: E402

register_renderer("cartesian", _render_cartesian)
register_renderer("number_line", _render_number_line)
