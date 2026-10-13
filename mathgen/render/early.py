"""SVG renderers for the early-years kinds: money, hundred_chart, fraction_model, counters, base_ten."""

from __future__ import annotations

import math

from mathgen.render import register_renderer
from mathgen.render._svg import W, Ctx, el, esc, fmt, text

INK = "var(--_text)"
LINE = "var(--_border-strong)"
# Counter colour 1 is a filled accent counter; colour 2 is a hollow counter with a thick ring.
C1 = "var(--_accent)"
C2 = "var(--_c1)"


def _round(v: float) -> int:
    """Round half up, as JavaScript's Math.round does."""
    return math.floor(v + 0.5)


def _plain(x, y, s, size, weight, *, anchor="middle", class_="mgf-plain", **kw):
    return text(x, y, s, anchor=anchor, class_=class_, font_size=size, font_weight=weight, **kw)


# ------------------------------------------------------------------ money

# Generic, suggestive sizes only (diameter in viewBox units): gold $1 larger than gold $2.
COIN_D = {"5c": 48, "10c": 54, "20c": 62, "50c": 68, "$1": 60, "$2": 52}
NOTE_W = 112
NOTE_H = 60
NOTE_FILL = {"$5": "#f4b6d2", "$10": "#9cc3f2", "$20": "#f2a39a", "$50": "#f6dc7a", "$100": "#a8d9a0"}
MONEY_INK = "#1f2937"
GAP = 10
ROW_H = 76


def _money_item(item: dict, x: float, cy: float) -> str:
    denom = item["denom"]
    if item["type"] == "note":
        return el(
            "g",
            None,
            el(
                "rect",
                {
                    "x": x,
                    "y": cy - NOTE_H / 2,
                    "width": NOTE_W,
                    "height": NOTE_H,
                    "rx": 6,
                    "fill": NOTE_FILL.get(denom, "#e5e7eb"),
                    "stroke": MONEY_INK,
                    "stroke-width": 2,
                },
            ),
            el(
                "rect",
                {
                    "x": x + 5,
                    "y": cy - NOTE_H / 2 + 5,
                    "width": NOTE_W - 10,
                    "height": NOTE_H - 10,
                    "rx": 3,
                    "fill": "none",
                    "stroke": MONEY_INK,
                    "stroke-opacity": 0.4,
                    "stroke-width": 1.2,
                },
            ),
            _plain(x + NOTE_W / 2, cy + 9, denom, 27, 800, class_="mgf-plain mgf-money-text"),
        )
    d = COIN_D.get(denom, 48)
    gold = denom in ("$1", "$2")
    return el(
        "g",
        None,
        el(
            "circle",
            {
                "cx": x + d / 2,
                "cy": cy,
                "r": d / 2 - 1,
                "fill": "#e6bd45" if gold else "#d3d8de",
                "stroke": MONEY_INK,
                "stroke-width": 2,
            },
        ),
        el(
            "circle",
            {
                "cx": x + d / 2,
                "cy": cy,
                "r": d / 2 - 5,
                "fill": "none",
                "stroke": MONEY_INK,
                "stroke-opacity": 0.35,
                "stroke-width": 1.2,
            },
        ),
        _plain(x + d / 2, cy + 7, denom, 19 if len(denom) > 2 else 21, 800, class_="mgf-plain mgf-money-text"),
    )


def _render_money(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    pad = 12
    avail = W - 2 * pad - 16
    y = 6.0
    panels = []
    groups = spec["groups"]
    multi = len(groups) > 1
    for g in groups:
        rows: list[list[tuple[dict, float]]] = [[]]
        used = 0.0
        for item in g["items"]:
            w = NOTE_W if item["type"] == "note" else COIN_D.get(item["denom"], 48)
            if used > 0 and used + GAP + w > avail:
                rows.append([])
                used = 0.0
            rows[-1].append((item, w))
            used += (GAP if used > 0 else 0) + w
        head = 30 if g["label"] else 0
        h = head + len(rows) * ROW_H + 8
        top = y
        parts = []
        if multi:
            parts.append(
                el(
                    "rect",
                    {
                        "x": pad,
                        "y": top,
                        "width": W - 2 * pad,
                        "height": h,
                        "rx": 10,
                        "fill": "var(--_surface-2)",
                        "stroke": "var(--_border)",
                    },
                )
            )
        if g["label"]:
            parts.append(_plain(pad + 14, top + 24, g["label"], 20, 800, anchor="start"))
        for ri, row in enumerate(rows):
            x = pad + 8
            cy = top + head + ri * ROW_H + ROW_H / 2
            for item, w in row:
                parts.append(_money_item(item, x, cy))
                x += w + GAP
        panels.append(el("g", None, *parts))
        y += h + 10
    return W, y, "".join(panels)


# ------------------------------------------------------------ hundred chart


def _render_hundred_chart(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    cell = 37
    x0 = (W - 10 * cell) / 2
    y0 = 8
    hidden, shaded, marked = set(spec["hidden"]), set(spec["shaded"]), set(spec["marked"])
    cells, rings = [], []
    for r in range(10):
        for c in range(10):
            n = spec["start"] + r * 10 + c
            x = x0 + c * cell
            y = y0 + r * cell
            is_hidden = n in hidden
            if n in shaded:
                fill = "color-mix(in srgb, var(--_accent) 38%, var(--_surface))"
            elif is_hidden:
                fill = "var(--_surface)"
            else:
                fill = "var(--_surface-2)"
            parts = [
                el(
                    "rect",
                    {"x": x, "y": y, "width": cell, "height": cell, "fill": fill, "stroke": LINE, "stroke-width": 1.2},
                )
            ]
            if not is_hidden:
                parts.append(_plain(x + cell / 2, y + cell / 2 + 5.5, str(n), 14 if n >= 100 else 16, 600))
            cells.append(el("g", None, *parts))
            if n in marked:
                rings.append(
                    el(
                        "rect",
                        {
                            "x": x + 2,
                            "y": y + 2,
                            "width": cell - 4,
                            "height": cell - 4,
                            "rx": 4,
                            "fill": "none",
                            "stroke": C2,
                            "stroke-width": 4,
                        },
                    )
                )
    return W, y0 + 10 * cell + 8, "".join(cells) + "".join(rings)


# ----------------------------------------------------------- fraction model


def _polar(cx, cy, r, turns):
    return cx + r * math.sin(turns * 2 * math.pi), cy - r * math.cos(turns * 2 * math.pi)


def _whole(m: dict, x, y, w, h, hatch_id: str) -> str:
    shaded = set(m["shaded"])

    def fill_of(i):
        return f"url(#{hatch_id})" if i in shaded else "var(--_surface)"

    inner = []
    base = {"stroke": INK, "stroke-width": 2}
    if m["shape"] == "bar":
        pw = w / m["parts"]
        for i in range(m["parts"]):
            inner.append(el("rect", {"x": x + i * pw, "y": y, "width": pw, "height": h, "fill": fill_of(i), **base}))
    elif m["shape"] == "rectangle":
        rows = m.get("rows") or 1
        cols = m.get("cols") or m["parts"]
        cw, ch = w / cols, h / rows
        for i in range(m["parts"]):
            inner.append(
                el(
                    "rect",
                    {
                        "x": x + (i % cols) * cw,
                        "y": y + (i // cols) * ch,
                        "width": cw,
                        "height": ch,
                        "fill": fill_of(i),
                        **base,
                    },
                )
            )
    else:
        r = min(w, h) / 2
        cx, cy = x + w / 2, y + h / 2
        for i in range(m["parts"]):
            ax, ay = _polar(cx, cy, r, i / m["parts"])
            bx, by = _polar(cx, cy, r, (i + 1) / m["parts"])
            d = f"M{fmt(cx)},{fmt(cy)} L{fmt(ax)},{fmt(ay)} A{fmt(r)},{fmt(r)} 0 0 1 {fmt(bx)},{fmt(by)} Z"
            inner.append(el("path", {"d": d, "fill": fill_of(i), "stroke-linejoin": "round", **base}))
    return el("g", None, *inner)


def _render_fraction_model(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    hatch_id = ctx.uid("hatch")
    models = spec["models"]
    n = len(models)
    shape = models[0]["shape"]
    out = []

    def label(m, lx, ly, anchor="middle"):
        return _plain(lx, ly, m["label"], 22, 800, anchor=anchor) if m["label"] else ""

    if shape == "bar":
        bw, bh = 316, 54
        x = 62 if models[0]["label"] else (W - bw) / 2
        for i, m in enumerate(models):
            y = 10 + i * (bh + 22)
            out.append(el("g", None, label(m, 30, y + bh / 2 + 8), _whole(m, x, y, bw, bh, hatch_id)))
        height = 10 + n * (bh + 22) - 12
    else:
        # Equal outlines: circle diameter / rectangle box identical across models.
        per_row = 1 if n == 1 else 3 if n == 3 else 2
        rows_n = math.ceil(n / per_row)
        cell_w = (W - 24) / per_row
        size = 220 if n == 1 else 168 if n == 2 else 110 if n == 3 else 150
        if shape == "rectangle":
            bw, bh = (108, 78) if n == 3 else (min(size * 1.2, cell_w - 24), min(size * 1.2, cell_w - 24) * 2 / 3)
        else:
            bw = bh = size
        head_h = 30 if models[0]["label"] else 6
        row_h = head_h + bh + 18
        for i, m in enumerate(models):
            col, row = i % per_row, i // per_row
            cx = 12 + cell_w * col + cell_w / 2
            top = 6 + row * row_h
            out.append(el("g", None, label(m, cx, top + 22), _whole(m, cx - bw / 2, top + head_h, bw, bh, hatch_id)))
        height = 6 + rows_n * row_h - 6
    defs = el(
        "defs",
        None,
        el(
            "pattern",
            {
                "id": hatch_id,
                "width": 9,
                "height": 9,
                "patternUnits": "userSpaceOnUse",
                "patternTransform": "rotate(45)",
            },
            el("rect", {"width": 9, "height": 9, "fill": "color-mix(in srgb, var(--_accent) 30%, var(--_surface))"}),
            el("line", {"x1": 0, "y1": 0, "x2": 0, "y2": 9, "stroke": "var(--_accent)", "stroke-width": 4}),
        ),
    )
    return W, height, defs + "".join(out)


# ------------------------------------------------------------------ counters


def _star_path(cx, cy, big_r) -> str:
    pts = []
    for i in range(10):
        rad = big_r if i % 2 == 0 else big_r * 0.45
        a = i * math.pi / 5
        pts.append(f"{fmt(cx + rad * math.sin(a))},{fmt(cy - rad * math.cos(a))}")
    return "M" + "L".join(pts) + "Z"


def _counter(shape: str, cx, cy, r, colour: int) -> str:
    """One counter of ``shape`` at (cx, cy) with nominal radius r."""
    ring = colour == 2
    sw = max(3.5, r * 0.3) if ring else 2
    style = {
        "fill": "var(--_surface)" if ring else C1,
        "stroke": C2 if ring else C1,
        "stroke-width": sw,
        "stroke-linejoin": "round",
    }
    k = sw / 2 if ring else 0
    if shape == "square":
        s = r * 0.9 - k
        return el("rect", {"x": cx - s, "y": cy - s, "width": 2 * s, "height": 2 * s, "rx": r * 0.15, **style})
    if shape == "diamond":
        s = r * 1.15 - k
        d = f"M{fmt(cx)},{fmt(cy - s)}L{fmt(cx + s)},{fmt(cy)}L{fmt(cx)},{fmt(cy + s)}L{fmt(cx - s)},{fmt(cy)}Z"
        return el("path", {"d": d, **style})
    if shape == "star":
        return el("path", {"d": _star_path(cx, cy + r * 0.05, r * 1.2 - k), **style})
    return el("circle", {"cx": cx, "cy": cy, "r": r - k, **style})


DIE_DOTS = {
    1: [(1, 1)],
    2: [(0, 0), (2, 2)],
    3: [(0, 0), (1, 1), (2, 2)],
    4: [(0, 0), (2, 0), (0, 2), (2, 2)],
    5: [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
    6: [(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)],
}


def _render_counters(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    shape = spec["shape"]
    layout = spec["layout"]
    out = []
    if layout == "ten_frame":
        frames = spec["frames"]
        box = 36 if len(frames) > 1 else 56
        fw = box * 5
        gap = 24
        total = len(frames) * fw + (len(frames) - 1) * gap
        x0 = (W - total) / 2
        for fi, f in enumerate(frames):
            fx = x0 + fi * (fw + gap)
            out.append(
                el(
                    "rect",
                    {
                        "x": fx,
                        "y": 10,
                        "width": fw,
                        "height": box * 2,
                        "fill": "var(--_surface-2)",
                        "stroke": INK,
                        "stroke-width": 2.5,
                    },
                )
            )
            for i in range(1, 5):
                out.append(
                    el(
                        "line",
                        {
                            "x1": fx + i * box,
                            "x2": fx + i * box,
                            "y1": 10,
                            "y2": 10 + 2 * box,
                            "stroke": INK,
                            "stroke-width": 1.8,
                        },
                    )
                )
            out.append(
                el(
                    "line",
                    {"x1": fx, "x2": fx + fw, "y1": 10 + box, "y2": 10 + box, "stroke": INK, "stroke-width": 1.8},
                )
            )
            for i, v in enumerate(f):
                if v:
                    out.append(
                        _counter(shape, fx + (i % 5) * box + box / 2, 10 + (i // 5) * box + box / 2, box * 0.34, v)
                    )
        height = 20 + 2 * box
    elif layout == "array":
        rows, cols = spec["rows"], spec["cols"]
        colours = spec.get("colours")
        pitch = min(56, 340 / cols)
        x0 = (W - cols * pitch) / 2
        for r in range(rows):
            for c in range(cols):
                colour = colours[r][c] if colours else 1
                out.append(
                    _counter(shape, x0 + c * pitch + pitch / 2, 10 + r * pitch + pitch / 2, pitch * 0.33, colour)
                )
        height = 20 + rows * pitch
    elif layout == "groups":
        groups = spec["groups"]
        n = len(groups)
        per_row = n if n <= 3 else 2 if n == 4 else 3
        d_ring = 112
        pitch_x = min(128, (W - 16) / per_row)
        rows_n = math.ceil(n / per_row)
        row_h = d_ring + 14
        for gi, cnt in enumerate(groups):
            in_row = min(per_row, n - (gi // per_row) * per_row)
            col = gi % per_row
            cx = W / 2 + (col - (in_row - 1) / 2) * pitch_x
            cy = 10 + (gi // per_row) * row_h + d_ring / 2
            out.append(
                el(
                    "circle",
                    {
                        "cx": cx,
                        "cy": cy,
                        "r": d_ring / 2 - 1,
                        "fill": "var(--_surface-2)",
                        "stroke": INK,
                        "stroke-width": 2.5,
                    },
                )
            )
            per_line = cnt if cnt <= 3 else 2 if cnt == 4 else 3 if cnt == 9 else 3 if cnt <= 6 else 4
            lines = []
            left = cnt
            while left > 0:
                lines.append(min(per_line, left))
                left -= per_line
            p = 26
            for li, ln in enumerate(lines):
                for k in range(ln):
                    out.append(_counter(shape, cx + (k - (ln - 1) / 2) * p, cy + (li - (len(lines) - 1) / 2) * p, 9, 1))
        height = 10 + rows_n * row_h
    elif layout == "dice":
        faces = spec["faces"]
        s = 100
        gap = 22
        total = len(faces) * s + (len(faces) - 1) * gap
        x0 = (W - total) / 2
        for i, f in enumerate(faces):
            fx = x0 + i * (s + gap)
            out.append(
                el(
                    "rect",
                    {
                        "x": fx,
                        "y": 10,
                        "width": s,
                        "height": s,
                        "rx": 16,
                        "fill": "var(--_surface)",
                        "stroke": INK,
                        "stroke-width": 3,
                    },
                )
            )
            for c, r in DIE_DOTS[f["n"]]:
                d = s * 0.26
                out.append(
                    _counter("circle", fx + s / 2 + (c - 1) * d, 10 + s / 2 + (r - 1) * d, s * 0.09, f["colour"])
                )
        height = 20 + s
    else:
        bx, by, bw, bh = 20, 10, W - 40, 220
        out.append(
            el(
                "rect",
                {
                    "x": bx,
                    "y": by,
                    "width": bw,
                    "height": bh,
                    "rx": 12,
                    "fill": "var(--_surface-2)",
                    "stroke": "var(--_border)",
                    "stroke-width": 1.5,
                },
            )
        )
        ix, iy = 26, 24
        for d in spec["dots"]:
            out.append(
                _counter(
                    shape, bx + ix + d["at"][0] * (bw - 2 * ix), by + iy + d["at"][1] * (bh - 2 * iy), 14, d["colour"]
                )
            )
        height = by + bh + 10
    return W, height, "".join(out)


# ------------------------------------------------------------------ base ten


def _render_base_ten(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    nh, nt, no = spec["hundreds"], spec["tens"], spec["ones"]
    margin = 14
    avail = W - 2 * margin
    block = "color-mix(in srgb, var(--_accent) 45%, var(--_surface))"

    # Regions at unit size u. A flat is 10u square, a rod 10u tall by u wide, a cube u square.
    def layout(u):
        g = max(2, _round(u * 0.4))
        regions = []
        if nh:
            cols = nh if nh <= 3 else min(3, math.ceil(nh / 2))
            rows = math.ceil(nh / cols)
            regions.append(("h", cols, nh, cols * 10 * u + (cols - 1) * g, rows * 10 * u + (rows - 1) * g))
        if nt:
            cols = nt if nt <= 10 else math.ceil(nt / 2)
            rows = math.ceil(nt / cols)
            regions.append(("t", cols, nt, cols * u + (cols - 1) * g, rows * 10 * u + (rows - 1) * g))
        if no:
            cols = no if no <= 5 else 5
            rows = math.ceil(no / cols)
            regions.append(("o", cols, no, cols * u + (cols - 1) * g, rows * u + (rows - 1) * g))
        sep = 22
        one_line = sum(r[3] for r in regions) + sep * (len(regions) - 1)
        return g, regions, sep, one_line

    # Largest unit whose regions each fit the width and whose wrapped stack stays a sensible height.
    def flow_height(t):
        _, regions, sep, _ = layout(t)
        if any(r[3] > avail for r in regions):
            return math.inf
        x = h = line_h = 0
        for r in regions:
            if x > 0 and x + r[3] > avail:
                h += line_h + 18
                x = line_h = 0
            x += r[3] + sep
            line_h = max(line_h, r[4])
        return h + line_h

    u = 6
    for t in range(14, 5, -1):
        if flow_height(t) <= 270:
            u = t
            break
    g, regions, sep, one_line = layout(u)
    wrap = one_line > avail
    out = []
    x = margin
    y = 10
    line_h = 0
    if not wrap:
        x = margin + (avail - one_line) / 2
    ink = {"stroke": INK}
    for kind, cols, cnt, rw, rh in regions:
        if wrap and x > margin and x + rw > margin + avail:
            x = margin
            y += line_h + 18
            line_h = 0
        for i in range(cnt):
            c, r = i % cols, i // cols
            if kind == "h":
                bx, by = x + c * (10 * u + g), y + r * (10 * u + g)
                d = "".join(
                    f"M{fmt(bx + k * u)},{fmt(by)}v{fmt(10 * u)}M{fmt(bx)},{fmt(by + k * u)}h{fmt(10 * u)}"
                    for k in range(1, 10)
                )
                out.append(
                    el(
                        "g",
                        None,
                        el(
                            "rect",
                            {
                                "x": bx,
                                "y": by,
                                "width": 10 * u,
                                "height": 10 * u,
                                "fill": block,
                                "stroke-width": 1.6,
                                **ink,
                            },
                        ),
                        el("path", {"d": d, "stroke-opacity": 0.45, "stroke-width": 0.8, "fill": "none", **ink}),
                    )
                )
            elif kind == "t":
                bx, by = x + c * (u + g), y + r * (10 * u + g)
                d = "".join(f"M{fmt(bx)},{fmt(by + k * u)}h{fmt(u)}" for k in range(1, 10))
                out.append(
                    el(
                        "g",
                        None,
                        el(
                            "rect",
                            {"x": bx, "y": by, "width": u, "height": 10 * u, "fill": block, "stroke-width": 1.4, **ink},
                        ),
                        el("path", {"d": d, "stroke-opacity": 0.55, "stroke-width": 0.8, "fill": "none", **ink}),
                    )
                )
            else:
                out.append(
                    el(
                        "rect",
                        {
                            "x": x + c * (u + g),
                            "y": y + r * (u + g),
                            "width": u,
                            "height": u,
                            "fill": block,
                            "stroke-width": 1.4,
                            **ink,
                        },
                    )
                )
        x += rw + sep
        line_h = max(line_h, rh)
    return W, y + line_h + 12, "".join(out)


register_renderer("money", _render_money)
register_renderer("hundred_chart", _render_hundred_chart)
register_renderer("fraction_model", _render_fraction_model)
register_renderer("counters", _render_counters)
register_renderer("base_ten", _render_base_ten)
