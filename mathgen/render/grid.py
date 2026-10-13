"""SVG renderer for ``grid_drawing`` (spec in `mathgen.figure_kinds.grid`).

Square grids draw, back to front: the cell groups, the grid lines, the lattice polygons, the lines, the path, the
symbols, the labels, then the grid references, the scale, the north arrow and the legend. Isometric grids draw
the dot lattice and the cubes, painted back to front so that a face which is hidden is never seen.

Every cell style differs from the others by more than colour: shaded cells are solid, hatched cells carry
diagonal lines, dotted cells carry dots. Every line style has its own shape: a road is a thick double line, a
river is wavy, a railway has cross ticks, a wall is thick, a fence is dashed with posts, a mirror line is
dash-dot. Symbols are plain pictograms built from basic shapes.

Nothing is evaluated here beyond mapping grid coordinates onto the drawing. Text drawn is the spec's own
(labels, references, scale, legend names); the north arrow's "N" is drawn as lines, not text.
"""

from __future__ import annotations

import math

from mathgen.figure_kinds.grid import SYMBOLS
from mathgen.render import register_renderer
from mathgen.render._svg import Ctx, el, fmt, g, points_attr, text
from mathgen.render.space import tw

__all__ = ["CSS", "canvas_below"]

PAD = 10
FS = 13
COS30 = math.sqrt(3) / 2

Pt = tuple[float, float]

CSS = (
    ".mgf .mgf-gd-bg{fill:var(--_surface);stroke:none}"
    ".mgf .mgf-gd-frame{fill:none;stroke:var(--_border-strong);stroke-width:1.6}"
    ".mgf .mgf-gd-line{stroke:var(--_border-strong);stroke-width:1}"
    ".mgf .mgf-gd-shaded{fill:var(--_accent);fill-opacity:.6;stroke:none}"
    ".mgf .mgf-gd-hatch-line{stroke:var(--_c1);stroke-width:1.8}"
    ".mgf .mgf-gd-dots-dot{fill:var(--_c2)}"
    ".mgf .mgf-gd-poly{fill:var(--_accent);fill-opacity:.12;stroke:var(--_text);stroke-width:2.6;stroke-linejoin:round}"
    ".mgf .mgf-gd-poly-dashed{stroke-dasharray:8 5}"
    ".mgf .mgf-gd-label text{fill:var(--_text);font-size:13px;font-weight:700}"
    ".mgf .mgf-gd-ref text{fill:var(--_tick);font-size:13px;font-weight:600}"
    ".mgf .mgf-gd-road-edge{fill:none;stroke:var(--_text);stroke-width:10;stroke-linejoin:round}"
    ".mgf .mgf-gd-road-core{fill:none;stroke:var(--_surface);stroke-width:5.5;stroke-linejoin:round}"
    ".mgf .mgf-gd-river{fill:none;stroke:var(--_accent);stroke-width:4;stroke-linecap:round;stroke-linejoin:round}"
    ".mgf .mgf-gd-rail{fill:none;stroke:var(--_text);stroke-width:2.6;stroke-linejoin:round}"
    ".mgf .mgf-gd-rail-tick{stroke:var(--_text);stroke-width:2}"
    ".mgf .mgf-gd-wall{fill:none;stroke:var(--_text);stroke-width:6.5;stroke-linecap:square;stroke-linejoin:miter}"
    ".mgf .mgf-gd-fence{fill:none;stroke:var(--_text);stroke-width:2;stroke-dasharray:6 4;stroke-linejoin:round}"
    ".mgf .mgf-gd-post{fill:var(--_text)}"
    ".mgf .mgf-gd-mirror{fill:none;stroke:var(--_c3);stroke-width:2.6;stroke-dasharray:11 3 2 3}"
    ".mgf .mgf-gd-path{fill:none;stroke:var(--_c1);stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round}"
    ".mgf .mgf-gd-path-head{fill:var(--_c1);stroke:none}"
    ".mgf .mgf-gd-path-start{fill:var(--_surface);stroke:var(--_c1);stroke-width:2.6}"
    ".mgf .mgf-gd-tile{fill:var(--_surface);stroke:var(--_border-strong);stroke-width:.8}"
    ".mgf .mgf-gd-sym *{fill:none;stroke:var(--_text);stroke-width:1.4;stroke-linejoin:round;stroke-linecap:round}"
    ".mgf .mgf-gd-sym .mgf-gd-tile{fill:var(--_surface);stroke:var(--_border-strong);stroke-width:.8}"
    ".mgf .mgf-gd-sym .mgf-gd-thick{stroke-width:2.6}"
    ".mgf .mgf-gd-sym .mgf-gd-fa{fill:var(--_accent)}"
    ".mgf .mgf-gd-sym .mgf-gd-f1{fill:var(--_c1)}"
    ".mgf .mgf-gd-sym .mgf-gd-f2{fill:var(--_c2)}"
    ".mgf .mgf-gd-sym .mgf-gd-fs{fill:var(--_surface)}"
    ".mgf .mgf-gd-sym .mgf-gd-ft{fill:var(--_text)}"
    ".mgf .mgf-gd-legend{fill:var(--_surface);stroke:var(--_border-strong);stroke-width:1.2}"
    ".mgf .mgf-gd-legend-name text{fill:var(--_text);font-size:13px;font-weight:600}"
    ".mgf .mgf-gd-scale-bar{stroke:var(--_text);stroke-width:2.6;fill:none}"
    ".mgf .mgf-gd-scale text{fill:var(--_text);font-size:13px;font-weight:600}"
    ".mgf .mgf-gd-north{fill:none;stroke:var(--_text);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
    ".mgf .mgf-gd-north-head{fill:var(--_text)}"
    ".mgf .mgf-gd-dot{fill:var(--_tick)}"
    ".mgf .mgf-gd-top{fill:color-mix(in srgb,var(--_accent) 50%,white);stroke:var(--_text);stroke-width:1.2;"
    "stroke-linejoin:round}"
    ".mgf .mgf-gd-left{fill:var(--_accent);stroke:var(--_text);stroke-width:1.2;stroke-linejoin:round}"
    ".mgf .mgf-gd-right{fill:color-mix(in srgb,var(--_accent) 62%,black);stroke:var(--_text);stroke-width:1.2;"
    "stroke-linejoin:round}"
)


# ---------------------------------------------------------------- the canvas rule shared with the describer


def canvas_below(spec: dict) -> bool:
    """True when the blank answer grid is drawn below the figure, false when it is drawn to its right: below
    when the figure is wider than it is tall."""
    grid = spec["grid"]
    if grid["type"] == "square":
        return grid["width"] > grid["height"]
    return 0.366 * (grid["width"] + grid["height"]) > grid["layers"]


# ---------------------------------------------------------------- polyline helpers


def _walk(pts: list[Pt], step: float):
    """Points every ``step`` along the polyline, each with the unit direction of its segment."""
    out = []
    carry = step / 2
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        d = carry
        while d <= length:
            out.append((x0 + ux * d, y0 + uy * d, ux, uy))
            d += step
        carry = d - length
    return out


def _d(pts: list[Pt]) -> str:
    return "M" + "L".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)


def _wavy(pts: list[Pt]) -> list[Pt]:
    out: list[Pt] = []
    s = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        n = max(1, int(length / 2))
        for i in range(n + 1):
            d = length * i / n
            off = 3.2 * math.sin(2 * math.pi * (s + d) / 18)
            out.append((x0 + ux * d - uy * off, y0 + uy * d + ux * off))
        s += length
    return out


def _line_el(style: str, pts: list[Pt]) -> str:
    d = _d(pts)
    if style == "road":
        return el("path", {"class_": "mgf-gd-road-edge", "d": d}) + el("path", {"class_": "mgf-gd-road-core", "d": d})
    if style == "river":
        return el("path", {"class_": "mgf-gd-river", "d": _d(_wavy(pts))})
    if style == "rail":
        ticks = [
            el(
                "line",
                {"class_": "mgf-gd-rail-tick", "x1": x - uy * 5, "y1": y + ux * 5, "x2": x + uy * 5, "y2": y - ux * 5},
            )
            for x, y, ux, uy in _walk(pts, 9)
        ]
        return el("path", {"class_": "mgf-gd-rail", "d": d}) + "".join(ticks)
    if style == "wall":
        return el("path", {"class_": "mgf-gd-wall", "d": d})
    if style == "fence":
        posts = [el("circle", {"class_": "mgf-gd-post", "cx": x, "cy": y, "r": 2.6}) for x, y, _, _ in _walk(pts, 20)]
        for x, y in pts:
            posts.append(el("circle", {"class_": "mgf-gd-post", "cx": x, "cy": y, "r": 2.6}))
        return el("path", {"class_": "mgf-gd-fence", "d": d}) + "".join(posts)
    return el("path", {"class_": "mgf-gd-mirror", "d": d})


# ---------------------------------------------------------------- symbols (unit frame -10..10, y down)


def _r(x, y, w, h, c=""):
    return el("rect", {"class_": c or None, "x": x, "y": y, "width": w, "height": h})


def _p(pts, c=""):
    return el("polygon", {"class_": c or None, "points": points_attr(pts)})


def _c(x, y, r, c=""):
    return el("circle", {"class_": c or None, "cx": x, "cy": y, "r": r})


def _l(x1, y1, x2, y2, c=""):
    return el("line", {"class_": c or None, "x1": x1, "y1": y1, "x2": x2, "y2": y2})


def _a(d, c=""):
    return el("path", {"class_": c or None, "d": d})


_PLUS = [(-2.5, -7), (2.5, -7), (2.5, -2.5), (7, -2.5), (7, 2.5), (2.5, 2.5), (2.5, 7), (-2.5, 7), (-2.5, 2.5)]
_PLUS += [(-7, 2.5), (-7, -2.5), (-2.5, -2.5)]
_PLANE = [(0, -10), (1.5, -6), (1.5, -1), (9, 4), (9, 6), (1.5, 3), (1.5, 7), (4, 9), (4, 10), (0, 9)]
_PLANE += [(-4, 10), (-4, 9), (-1.5, 7), (-1.5, 3), (-9, 6), (-9, 4), (-1.5, -1), (-1.5, -6)]

_SHAPES = {
    "house": lambda: [
        _r(-6, 0, 12, 8, "mgf-gd-fs"),
        _p([(-8, 0), (0, -8), (8, 0)], "mgf-gd-f1"),
        _r(-1.5, 3, 3, 5, "mgf-gd-ft"),
    ],
    "school": lambda: [
        _r(-9, -1, 18, 9, "mgf-gd-fs"),
        _r(-3, -6, 6, 5, "mgf-gd-fs"),
        _l(0, -6, 0, -10),
        _p([(0, -10), (5, -8.5), (0, -7)], "mgf-gd-f1"),
        _r(-1.5, 3, 3, 5, "mgf-gd-ft"),
        _r(-7, 2, 3, 3, "mgf-gd-fa"),
        _r(4, 2, 3, 3, "mgf-gd-fa"),
    ],
    "church": lambda: [
        _r(-5, -1, 10, 10, "mgf-gd-fs"),
        _p([(-6, -1), (0, -5), (6, -1)], "mgf-gd-f1"),
        _l(0, -5, 0, -10),
        _l(-2, -8, 2, -8),
        _r(-1.5, 4, 3, 5, "mgf-gd-ft"),
    ],
    "hospital": lambda: [_r(-9, -9, 18, 18, "mgf-gd-fs"), _p(_PLUS, "mgf-gd-f1")],
    "tree": lambda: [_r(-1.5, 3, 3, 6, "mgf-gd-ft"), _c(0, -2, 7, "mgf-gd-f2")],
    "bridge": lambda: [
        _r(-10, -4, 20, 3, "mgf-gd-fs"),
        _a("M-7,-1Q0,11 7,-1"),
        _l(-9, -1, -9, 8),
        _l(9, -1, 9, 8),
    ],
    "campground": lambda: [
        _p([(-9, 8), (0, -8), (9, 8)], "mgf-gd-f1"),
        _p([(-3, 8), (0, 0), (3, 8)], "mgf-gd-ft"),
        _l(-10, 8, 10, 8),
    ],
    "lookout": lambda: [
        _l(-5, 9, -3, 0),
        _l(5, 9, 3, 0),
        _l(-4, 5, 4, 5),
        _r(-6, -8, 12, 7, "mgf-gd-fs"),
        _p([(-7, -8), (0, -10), (7, -8)], "mgf-gd-f1"),
    ],
    "post_office": lambda: [_r(-9, -6, 18, 12, "mgf-gd-fs"), _a("M-9,-6L0,1L9,-6")],
    "railway_station": lambda: [
        _r(-8, -8, 16, 14, "mgf-gd-f1"),
        _r(-5, -5, 10, 6, "mgf-gd-fs"),
        _c(-4, 8, 2.2, "mgf-gd-ft"),
        _c(4, 8, 2.2, "mgf-gd-ft"),
    ],
    "airport": lambda: [_p(_PLANE, "mgf-gd-ft")],
    "lighthouse": lambda: [
        _p([(-4, 9), (-2.5, -3), (2.5, -3), (4, 9)], "mgf-gd-fs"),
        _p([(-3.2, 2), (3.2, 2), (3.6, 5.5), (-3.6, 5.5)], "mgf-gd-f1"),
        _r(-3, -7, 6, 4, "mgf-gd-fa"),
        _p([(-3.5, -7), (0, -10), (3.5, -7)], "mgf-gd-ft"),
        _l(-4, -5, -9, -6),
        _l(4, -5, 9, -6),
    ],
    "picnic_area": lambda: [
        _r(-9, -1, 18, 3, "mgf-gd-f1"),
        _l(-6, 2, -8, 9),
        _l(6, 2, 8, 9),
        _l(-7, 6, 7, 6),
    ],
    "toilets": lambda: [
        _c(-4.5, -6, 2, "mgf-gd-ft"),
        _r(-7, -3, 5, 8, "mgf-gd-f1"),
        _l(-5.5, 5, -5.5, 10),
        _c(4.5, -6, 2, "mgf-gd-ft"),
        _p([(4.5, -3.5), (8, 5), (1, 5)], "mgf-gd-f2"),
        _l(0, -9, 0, 10),
    ],
    "parking": lambda: [
        _r(-9, -9, 18, 18, "mgf-gd-fs"),
        _a("M-3,6L-3,-6L1.5,-6A3.5,3.5 0 0 1 1.5,1L-3,1", "mgf-gd-thick"),
    ],
    "information": lambda: [
        _c(0, 0, 9, "mgf-gd-fs"),
        _c(0, -4.5, 1.4, "mgf-gd-ft"),
        _r(-1.3, -1, 2.6, 6.5, "mgf-gd-ft"),
    ],
    "door": lambda: [
        _l(-10, 8, -6, 8, "mgf-gd-thick"),
        _l(6, 8, 10, 8, "mgf-gd-thick"),
        _l(-6, 8, -6, -4),
        _a("M-6,-4A12,12 0 0 1 6,8"),
    ],
    "window": lambda: [
        _l(-10, 0, -7, 0, "mgf-gd-thick"),
        _l(7, 0, 10, 0, "mgf-gd-thick"),
        _r(-7, -2.5, 14, 5, "mgf-gd-fs"),
        _l(-7, 0, 7, 0),
    ],
}


def _symbol(name: str, cx: float, cy: float, r: float) -> str:
    k = r / 10
    tile = el("rect", {"class_": "mgf-gd-tile", "x": -11, "y": -11, "width": 22, "height": 22, "rx": 3})
    return el(
        "g",
        {
            "class_": f"mgf-gd-sym mgf-gd-sym-{name.replace('_', '-')}",
            "transform": f"translate({fmt(cx)} {fmt(cy)}) scale({fmt(k)})",
        },
        tile,
        *_SHAPES[name](),
    )


# ---------------------------------------------------------------- square grid


def _centroid(pts: list[Pt]) -> Pt:
    area2 = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area2) < 1e-9:
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return (cx / (3 * area2), cy / (3 * area2))


def _blank_grid(w: int, h: int, s: float, x0: float, y0: float) -> str:
    parts = [el("rect", {"class_": "mgf-gd-bg", "x": x0, "y": y0, "width": w * s, "height": h * s})]
    for i in range(1, w):
        parts.append(
            el("line", {"class_": "mgf-gd-line", "x1": x0 + i * s, "y1": y0, "x2": x0 + i * s, "y2": y0 + h * s})
        )
    for j in range(1, h):
        parts.append(
            el("line", {"class_": "mgf-gd-line", "x1": x0, "y1": y0 + j * s, "x2": x0 + w * s, "y2": y0 + j * s})
        )
    parts.append(el("rect", {"class_": "mgf-gd-frame", "x": x0, "y": y0, "width": w * s, "height": h * s}))
    return el("g", {"class_": "mgf-gd-canvas"}, *parts)


def _scale_el(scale: dict, s: float, x: float, y: float) -> tuple[str, float]:
    """The scale bar and text with its left edge at ``x`` and its middle at ``y``; also the width used."""
    parts = []
    x_text = x
    if "bar_cells" in scale:
        n = scale["bar_cells"] * s
        parts.append(
            el(
                "path",
                {
                    "class_": "mgf-gd-scale-bar",
                    "d": f"M{fmt(x)},{fmt(y - 5)}L{fmt(x)},{fmt(y + 5)}M{fmt(x)},{fmt(y)}L{fmt(x + n)},{fmt(y)}M{fmt(x + n)},{fmt(y - 5)}L{fmt(x + n)},{fmt(y + 5)}",
                },
            )
        )
        x_text = x + n + 10
    parts.append(g({"class_": "mgf-gd-scale"}, text(x_text, y + 4.5, scale["text"], anchor="start")))
    return "".join(parts), (x_text - x) + tw(scale["text"], FS)


def _north_el(cx: float, top: float) -> str:
    """A north arrow whose tip is at (cx, top): an arrow up with an N drawn as lines above it."""
    n = f"M{fmt(cx - 4)},{fmt(top + 10)}L{fmt(cx - 4)},{fmt(top)}L{fmt(cx + 4)},{fmt(top + 10)}L{fmt(cx + 4)},{fmt(top)}"
    tip = top + 16
    head = _p([(cx, tip), (cx - 5, tip + 11), (cx + 5, tip + 11)], "mgf-gd-north-head")
    shaft = _l(cx, tip + 11, cx, tip + 36)
    return el(
        "g",
        {"class_": "mgf-gd-north-arrow"},
        el("path", {"class_": "mgf-gd-north", "d": n}),
        el("g", {"class_": "mgf-gd-north"}, shaft),
        head,
    )


def _arrow_head(x: float, y: float, ux: float, uy: float, size: float = 7) -> str:
    bx, by = x - ux * size, y - uy * size
    return el(
        "polygon",
        {
            "class_": "mgf-gd-path-head",
            "points": points_attr(
                [(x, y), (bx - uy * size * 0.6, by + ux * size * 0.6), (bx + uy * size * 0.6, by - ux * size * 0.6)]
            ),
        },
    )


def _legend(names: list[str], x: float, y: float) -> tuple[str, float, float]:
    cols = 1 if len(names) <= 5 else 2
    rows = math.ceil(len(names) / cols)
    row_h = 28
    col_w = max(tw(SYMBOLS[n], FS) for n in names) + 44
    width = col_w * cols + 12
    height = rows * row_h + 10
    parts = [el("rect", {"class_": "mgf-gd-legend", "x": x, "y": y, "width": width, "height": height, "rx": 4})]
    for i, name in enumerate(names):
        col, row = divmod(i, rows)
        cx = x + 8 + col * col_w
        cy = y + 5 + row * row_h + row_h / 2
        parts.append(_symbol(name, cx + 11, cy, 9))
        parts.append(g({"class_": "mgf-gd-legend-name"}, text(cx + 28, cy + 4.5, SYMBOLS[name], anchor="start")))
    return el("g", {"class_": "mgf-gd-legend-box"}, *parts), width, height


def _render_square(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    grid = spec["grid"]
    w, h = grid["width"], grid["height"]
    s = int(max(10, min(36, 360 // max(w, h))))
    GW, GH = w * s, h * s
    rows = spec.get("row_labels", [])
    cols = spec.get("column_labels", [])
    title = spec.get("title", "")
    left = max((tw(t, FS) for t in rows), default=-6) + 12 if rows else 0
    gx0 = PAD + left
    gy0 = PAD + (26 if title else 0)

    def px(x: float) -> float:
        return gx0 + x * s

    def py(y: float) -> float:
        return gy0 + (h - y) * s

    parts: list[str] = []
    defs: list[str] = []
    parts.append(el("rect", {"class_": "mgf-gd-bg", "x": gx0, "y": gy0, "width": GW, "height": GH}))
    # Cells.
    pattern_ids: dict[str, str] = {}
    for group in spec.get("cells", []):
        style = group["style"]
        if style != "shaded" and style not in pattern_ids:
            pid = ctx.uid(style)
            pattern_ids[style] = pid
            if style == "hatched":
                inner = el("line", {"class_": "mgf-gd-hatch-line", "x1": 0, "y1": 0, "x2": 0, "y2": 7})
                defs.append(
                    el(
                        "pattern",
                        {
                            "id": pid,
                            "patternUnits": "userSpaceOnUse",
                            "width": 7,
                            "height": 7,
                            "patternTransform": "rotate(45)",
                        },
                        inner,
                    )
                )
            else:
                inner = el("circle", {"class_": "mgf-gd-dots-dot", "cx": 4, "cy": 4, "r": 1.7})
                defs.append(
                    el("pattern", {"id": pid, "patternUnits": "userSpaceOnUse", "width": 8, "height": 8}, inner)
                )
        rects = []
        for x, y in group["cells"]:
            attrs = {"x": px(x), "y": py(y + 1), "width": s, "height": s}
            if style == "shaded":
                attrs["class_"] = "mgf-gd-shaded"
            else:
                attrs["fill"] = f"url(#{pattern_ids[style]})"
            rects.append(el("rect", attrs))
        parts.append(el("g", {"class_": f"mgf-gd-cells mgf-gd-cells-{style}"}, *rects))
    # Grid lines.
    lines = []
    for i in range(1, w):
        lines.append(el("line", {"class_": "mgf-gd-line", "x1": px(i), "y1": gy0, "x2": px(i), "y2": gy0 + GH}))
    for j in range(1, h):
        lines.append(
            el("line", {"class_": "mgf-gd-line", "x1": gx0, "y1": gy0 + j * s, "x2": gx0 + GW, "y2": gy0 + j * s})
        )
    parts.append(el("g", {"class_": "mgf-gd-lines"}, *lines))
    parts.append(el("rect", {"class_": "mgf-gd-frame", "x": gx0, "y": gy0, "width": GW, "height": GH}))
    # Polygons.
    for poly in spec.get("polygons", []):
        pts = [(px(x), py(y)) for x, y in poly["vertices"]]
        cls = "mgf-gd-poly" + (" mgf-gd-poly-dashed" if poly.get("style") == "dashed" else "")
        piece = el("polygon", {"class_": cls, "points": points_attr(pts)})
        if poly.get("label"):
            lx, ly = _centroid(pts)
            piece += g({"class_": "mgf-gd-label"}, text(lx, ly + 4.5, poly["label"]))
        parts.append(el("g", {"class_": "mgf-gd-polygon"}, piece))
    # Lines.
    for line in spec.get("lines", []):
        pts = [(px(x), py(y)) for x, y in line["points"]]
        parts.append(
            el("g", {"class_": f"mgf-gd-feature mgf-gd-feature-{line['style']}"}, _line_el(line["style"], pts))
        )
    # Path.
    if spec.get("path"):
        cells = spec["path"]["cells"]
        centres = [(px(x + 0.5), py(y + 0.5)) for x, y in cells]
        bits = []
        for (x0, y0), (x1, y1) in zip(centres, centres[1:]):
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy)
            ux, uy = dx / ln, dy / ln
            start = 0.0 if (x0, y0) != centres[0] else min(7.0, s * 0.3)
            ax, ay = x0 + ux * start, y0 + uy * start
            ex, ey = x1 - ux * s * 0.2, y1 - uy * s * 0.2
            bits.append(el("line", {"class_": "mgf-gd-path", "x1": ax, "y1": ay, "x2": ex - ux * 5, "y2": ey - uy * 5}))
            bits.append(_arrow_head(ex, ey, ux, uy))
        bits.append(
            el(
                "circle",
                {"class_": "mgf-gd-path-start", "cx": centres[0][0], "cy": centres[0][1], "r": min(7.0, s * 0.3)},
            )
        )
        parts.append(el("g", {"class_": "mgf-gd-route"}, *bits))
    # Symbols.
    r_sym = max(8.0, min(14.0, s * 0.42))
    used: list[str] = []
    for sym in spec.get("symbols", []):
        parts.append(_symbol(sym["symbol"], px(sym["at"][0]), py(sym["at"][1]), r_sym))
        if sym["symbol"] not in used:
            used.append(sym["symbol"])
    # Labels at points.
    for lab in spec.get("labels", []):
        parts.append(g({"class_": "mgf-gd-label"}, text(px(lab["at"][0]), py(lab["at"][1]) + 4.5, lab["text"])))
    # Grid references.
    y_below = gy0 + GH
    if cols:
        parts.append(
            g(
                {"class_": "mgf-gd-ref mgf-gd-columns"},
                *[text(px(i + 0.5), y_below + 16, t) for i, t in enumerate(cols)],
            )
        )
        y_below += 22
    if rows:
        parts.append(
            g(
                {"class_": "mgf-gd-ref mgf-gd-rows"},
                *[text(gx0 - 7, py(j + 0.5) + 4.5, t, anchor="end") for j, t in enumerate(rows)],
            )
        )
    width = gx0 + GW + PAD
    # North arrow, to the right of the grid.
    if spec.get("north"):
        parts.append(_north_el(gx0 + GW + 22, gy0 + 2))
        width += 34
    # Scale.
    y_below += 8
    if spec.get("scale"):
        markup, used_w = _scale_el(spec["scale"], s, gx0, y_below + 8)
        parts.append(markup)
        width = max(width, gx0 + used_w + PAD)
        y_below += 28
    # Legend.
    if spec.get("legend") and used:
        markup, lw, lh = _legend(used, gx0, y_below + 2)
        parts.append(markup)
        width = max(width, gx0 + lw + PAD)
        y_below += lh + 6
    height = y_below + PAD
    if title:
        parts.insert(0, g({"class_": "mgf-axis-label mgf-title"}, text(PAD, PAD + 14, title, anchor="start")))
    if defs:
        parts.insert(0, el("defs", None, *defs))
    return _with_canvas(spec, width, height, gx0, gy0, s, "".join(parts))


def _with_canvas(spec: dict, width: float, height: float, gx0: float, gy0: float, s: float, body: str):
    canvas = spec.get("canvas")
    if not isinstance(canvas, dict):
        return width, height, body
    cw, ch = canvas["width"], canvas["height"]
    sc = max(10.0, min(float(s), 24.0, 300.0 / cw, 300.0 / ch))
    gap = 22
    if canvas_below(spec):
        x0, y0 = gx0, height + gap - PAD
        width = max(width, x0 + cw * sc + PAD)
        height = y0 + ch * sc + PAD
    else:
        x0, y0 = width + gap - PAD, gy0
        width = x0 + cw * sc + PAD
        height = max(height, y0 + ch * sc + PAD)
    return width, height, body + _blank_grid(cw, ch, sc, x0, y0)


# ---------------------------------------------------------------- isometric grid


def _render_isometric(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    grid = spec["grid"]
    w, h, layers = grid["width"], grid["height"], grid["layers"]
    s = min(34.0, 340 / ((w + h) * COS30), 300 / (layers + (w + h) * 0.5))
    title = spec.get("title", "")
    ox = PAD
    oy = PAD + (26 if title else 0) + h * 0.5 * s + layers * s

    def P(x: float, y: float, z: float) -> Pt:
        return (ox + (x + y) * COS30 * s, oy + (x - y) * 0.5 * s - z * s)

    parts: list[str] = []
    seen: set[tuple[int, int]] = set()
    dots = []
    for x in range(w + 1):
        for y in range(h + 1):
            for z in range(layers + 1):
                i, j = x + y, 2 * z + y - x
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                px_, py_ = P(x, y, z)
                dots.append(el("circle", {"class_": "mgf-gd-dot", "cx": px_, "cy": py_, "r": 1.8}))
    parts.append(el("g", {"class_": "mgf-gd-lattice"}, *dots))
    # Back to front: a cube in front is nearer the viewer, who is at the front right and above, so it has a
    # larger x, a smaller y and a larger z. Sorting by x - y + z paints every cube after the cubes it hides.
    cubes = sorted((tuple(c) for c in spec["cubes"]), key=lambda c: (c[0] - c[1] + c[2], c[2], c[0]))
    drawn = []
    for x, y, z in cubes:
        top = [P(x, y, z + 1), P(x + 1, y, z + 1), P(x + 1, y + 1, z + 1), P(x, y + 1, z + 1)]
        left = [P(x, y, z), P(x + 1, y, z), P(x + 1, y, z + 1), P(x, y, z + 1)]
        right = [P(x + 1, y, z), P(x + 1, y + 1, z), P(x + 1, y + 1, z + 1), P(x + 1, y, z + 1)]
        drawn.append(
            el(
                "g",
                {"class_": "mgf-gd-cube", "data-cube": f"{x},{y},{z}"},
                el("polygon", {"class_": "mgf-gd-left", "points": points_attr(left)}),
                el("polygon", {"class_": "mgf-gd-right", "points": points_attr(right)}),
                el("polygon", {"class_": "mgf-gd-top", "points": points_attr(top)}),
            )
        )
    parts.append(el("g", {"class_": "mgf-gd-cubes"}, *drawn))
    for lab in spec.get("labels", []):
        lx, ly = P(lab["at"][0], lab["at"][1], 0)
        parts.append(g({"class_": "mgf-gd-label"}, text(lx, ly + 4.5, lab["text"])))
    width = ox + (w + h) * COS30 * s + PAD
    height = oy + w * 0.5 * s + PAD
    if spec.get("north"):
        parts.append(_north_el(width + 12, PAD + (26 if title else 0)))
        width += 34
    if spec.get("scale"):
        markup, used_w = _scale_el(spec["scale"], s, ox, height + 8)
        parts.append(markup)
        width = max(width, ox + used_w + PAD)
        height += 28
    if title:
        parts.insert(0, g({"class_": "mgf-axis-label mgf-title"}, text(PAD, PAD + 14, title, anchor="start")))
    return _with_canvas(spec, width, height, ox, PAD + (26 if title else 0), 24.0, "".join(parts))


def _render_grid_drawing(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    if spec["grid"]["type"] == "isometric":
        return _render_isometric(spec, ctx)
    return _render_square(spec, ctx)


register_renderer("grid_drawing", _render_grid_drawing)
