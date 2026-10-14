"""Render a question's figure spec as a standalone SVG string, with no dependencies.

``figure_svg(spec)`` takes the JSON-able dict a generator puts in ``Question.figure`` (see
:mod:`mathgen.figures`) and returns an ``<svg>`` document you can embed in HTML, print, or
rasterise. The output is deterministic (the same spec gives a byte-identical string), draws
black ink on white, loads nothing external, and escapes every piece of text.

What is drawn is exactly what a learner is meant to see. Fields that exist only so a program
can check or re-derive the answer are never drawn: a clock's ``hour`` and ``minute`` (the hands
are drawn from the angles), a money item's ``cents``, the protractor's numeric ``angle``, and
the like. A blank the learner fills in (a tree branch ``"?"``, a Venn region ``"?"``) is drawn
as a blank.

The data displays (bar chart, picture graph, dot plot, histogram, box plot, pie chart) are read-the-chart figures:
every value is drawn to scale, and the only numerals are the ticks on a labelled axis. No bar, row, dot stack, box or
sector is ever labelled with its own value, count, fraction or percentage, because a question asks the learner for it.

The picture has a fixed 400-unit wide drawing area; ``width`` only sets the size the SVG is
displayed at (its height follows the aspect ratio).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction
from typing import Any, Callable
from xml.sax.saxutils import escape, quoteattr

from mathgen import figures
from mathgen.figure_kinds.data import _key_text
from mathgen.figure_kinds.tally import picture_key_text

__all__ = ["figure_svg"]

CANVAS = 400
INK = "currentColor"
PAPER = "#ffffff"
GRID = "#d4d4d4"
LIGHT = "#ececec"
MID = "#b0b0b0"
DARK = "#6b6b6b"

_XML_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")

# A rendered body: SVG markup for the drawing (in a 400-wide space) and its height.
Body = tuple[str, float]


# ---------------------------------------------------------------- markup helpers


def _n(v: float) -> str:
    """A compact number: rounded to 2 places, no trailing zeros, never ``-0``."""
    r = round(float(v), 2)
    if r == 0:
        return "0"
    return f"{r:.2f}".rstrip("0").rstrip(".")


def _attr_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _n(v)
    return str(v)


def _el(tag: str, content: str | None = None, **attrs: Any) -> str:
    """One element. Trailing underscores are dropped and the rest become hyphens (``stroke_width``)."""
    parts = [tag]
    for key, val in attrs.items():
        if val is None:
            continue
        parts.append(f"{key.rstrip('_').replace('_', '-')}={quoteattr(_attr_value(val))}")
    head = " ".join(parts)
    if content is None:
        return f"<{head}/>"
    return f"<{head}>{content}</{tag}>"


def _g(*children: str, **attrs: Any) -> str:
    return _el("g", "".join(children), **attrs)


def _clean_text(s: Any) -> str:
    return _XML_ILLEGAL.sub("", str(s))


def _text(x: float, y: float, s: Any, *, size: float = 12, anchor: str = "middle", weight: str | None = None) -> str:
    """Text centred vertically on ``y``. The string is XML-escaped here and nowhere else."""
    return _el(
        "text",
        escape(_clean_text(s)),
        x=x,
        y=y,
        dy=".35em",
        font_size=size,
        text_anchor=anchor,
        font_weight=weight,
        fill=INK,
        stroke="none",
    )


def _line(x1: float, y1: float, x2: float, y2: float, **kw: Any) -> str:
    return _el("line", x1=x1, y1=y1, x2=x2, y2=y2, **kw)


def _rect(x: float, y: float, w: float, h: float, **kw: Any) -> str:
    return _el("rect", x=x, y=y, width=w, height=h, **kw)


def _circle(cx: float, cy: float, r: float, **kw: Any) -> str:
    return _el("circle", cx=cx, cy=cy, r=r, **kw)


def _poly(points: list[tuple[float, float]], closed: bool = False, **kw: Any) -> str:
    pts = " ".join(f"{_n(x)},{_n(y)}" for x, y in points)
    return _el("polygon" if closed else "polyline", points=pts, **kw)


def _path(d: str, **kw: Any) -> str:
    return _el("path", d=d, **kw)


def _pt(cx: float, cy: float, r: float, deg_ccw: float) -> tuple[float, float]:
    """A point on a circle at ``deg_ccw`` degrees anticlockwise from the +x axis, as seen on screen."""
    t = math.radians(deg_ccw)
    return cx + r * math.cos(t), cy - r * math.sin(t)


def _clock_pt(cx: float, cy: float, r: float, deg_cw: float) -> tuple[float, float]:
    """A point at ``deg_cw`` degrees clockwise from 12 o'clock."""
    t = math.radians(deg_cw)
    return cx + r * math.sin(t), cy - r * math.cos(t)


def _arc(cx: float, cy: float, r: float, start_ccw: float, end_ccw: float, *, clockwise: bool = False) -> str:
    """An arc from one angle to another (anticlockwise degrees from +x), travelling the stated way round."""
    delta = (end_ccw - start_ccw) % 360
    if clockwise:
        delta = 360 - delta if delta else 0
    x1, y1 = _pt(cx, cy, r, start_ccw)
    x2, y2 = _pt(cx, cy, r, end_ccw)
    large = 1 if delta > 180 else 0
    sweep = 1 if clockwise else 0
    return f"M{_n(x1)} {_n(y1)}A{_n(r)} {_n(r)} 0 {large} {sweep} {_n(x2)} {_n(y2)}"


def _frac(v: Any) -> Fraction:
    return Fraction(str(round(float(v), 6))).limit_denominator(10**6)


def _star(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    pts = []
    for i in range(10):
        rad = r if i % 2 == 0 else r * 0.45
        pts.append(_clock_pt(cx, cy, rad, i * 36))
    return pts


class _Ids:
    """Element ids that are unique per figure (so several can share one page) yet deterministic.

    Renderers write ids with a placeholder prefix; `figure_svg` swaps in a hash of the finished drawing, so an id
    depends only on what the figure shows and never on spec fields it keeps out of sight (a clock's hour and
    minute when it has no numerals, an angle that is not drawn).
    """

    PLACEHOLDER = "fZZidprefixZZ"

    def __init__(self) -> None:
        self.prefix = self.PLACEHOLDER

    @classmethod
    def finish(cls, svg: str) -> str:
        digest = "f" + hashlib.sha1(svg.encode("utf-8")).hexdigest()[:8]
        return svg.replace(cls.PLACEHOLDER, digest)

    def __call__(self, name: str) -> str:
        return f"{self.prefix}-{name}"


# ---------------------------------------------------------------- cartesian

_PI = "π"


def _pi_text(v: float) -> str:
    f = Fraction(v).limit_denominator(24)
    if f == 0:
        return "0"
    sign = "−" if f < 0 else ""
    p, q = abs(f.numerator), f.denominator
    top = _PI if p == 1 else f"{p}{_PI}"
    return f"{sign}{top}" if q == 1 else f"{sign}{top}/{q}"


def _num_text(v: float) -> str:
    s = _n(v)
    return s.replace("-", "−") if s.startswith("-") else s


def _clip_line(p0, p1, box) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip the infinite line through p0 and p1 to ``box`` = (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    lo, hi = -1e9, 1e9
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]), (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
    if lo > hi:
        return None
    return (p0[0] + lo * dx, p0[1] + lo * dy), (p0[0] + hi * dx, p0[1] + hi * dy)


def _cartesian(s: dict, ids: _Ids) -> Body:
    x0, x1 = s["x_range"]
    y0, y1 = s["y_range"]
    left, top, bottom = 46, 14, 40
    max_w, max_h = 336.0, 300.0
    dx, dy = x1 - x0, y1 - y0
    if s["equal_scale"]:
        k = min(max_w / dx, max_h / dy)
        pw, ph = dx * k, dy * k
    else:
        pw, ph = max_w, max_h
    sx, sy = pw / dx, ph / dy

    def X(x: float) -> float:
        return left + (x - x0) * sx

    def Y(y: float) -> float:
        return top + ph - (y - y0) * sy

    out = [_rect(left, top, pw, ph, stroke=MID, stroke_width=1)]
    xg, yg = s["x_grid"], s["y_grid"]
    nx0, nx1 = math.ceil(x0 / xg - 1e-9), math.floor(x1 / xg + 1e-9)
    ny0, ny1 = math.ceil(y0 / yg - 1e-9), math.floor(y1 / yg + 1e-9)
    for n in range(nx0, nx1 + 1):
        out.append(_line(X(n * xg), top, X(n * xg), top + ph, stroke=GRID, stroke_width=0.7))
    for n in range(ny0, ny1 + 1):
        out.append(_line(left, Y(n * yg), left + pw, Y(n * yg), stroke=GRID, stroke_width=0.7))

    ax = X(0) if x0 <= 0 <= x1 else left
    ay = Y(0) if y0 <= 0 <= y1 else top + ph
    out.append(_line(left, ay, left + pw, ay, stroke=INK, stroke_width=1.5))
    out.append(_line(ax, top, ax, top + ph, stroke=INK, stroke_width=1.5))

    if s.get("tick_labels", True):
        kx = max(1, math.ceil(30 / (xg * sx)))
        ky = max(1, math.ceil(18 / (yg * sy)))
        fmt_x = _pi_text if s["x_tick_format"] == "pi" else _num_text
        for n in range(nx0, nx1 + 1):
            v = n * xg
            out.append(_line(X(v), ay - 3, X(v), ay + 3, stroke=INK, stroke_width=1))
            if n % kx == 0 and not (abs(v) < 1e-9 and y0 <= 0 <= y1 and x0 <= 0 <= x1):
                out.append(_text(X(v), ay + 14, fmt_x(v), size=11))
        for n in range(ny0, ny1 + 1):
            v = n * yg
            out.append(_line(ax - 3, Y(v), ax + 3, Y(v), stroke=INK, stroke_width=1))
            if n % ky == 0 and not (abs(v) < 1e-9 and y0 <= 0 <= y1 and x0 <= 0 <= x1):
                out.append(_text(ax - 7, Y(v), _num_text(v), size=11, anchor="end"))
        if y0 <= 0 <= y1 and x0 <= 0 <= x1:
            out.append(_text(ax - 7, ay + 12, "0", size=11, anchor="end"))
    if s.get("x_label"):
        out.append(_text(left + pw - 2, ay - 11, s["x_label"], size=12, anchor="end", weight="bold"))
    if s.get("y_label"):
        out.append(_text(ax + 8, top + 9, s["y_label"], size=12, anchor="start", weight="bold"))

    clip = ids("plot")
    inner = []
    for c in s.get("curves", ()):
        for piece in c["pieces"]:
            inner.append(_poly([(X(a), Y(b)) for a, b in piece], stroke=INK, stroke_width=2, stroke_linejoin="round"))
    box = (x0, x1, y0, y1)
    labels = []
    for ln in s.get("lines", ()):
        seg = _clip_line(ln["through"][0], ln["through"][1], box)
        if seg is None:
            continue
        (ax_, ay_), (bx_, by_) = seg
        inner.append(_line(X(ax_), Y(ay_), X(bx_), Y(by_), stroke=INK, stroke_width=2))
        if ln.get("label"):
            labels.append(_text(X(bx_) - 4, Y(by_) - 9, ln["label"], size=12, anchor="end"))
    for sg in s.get("segments", ()):
        (a, b), (c, d) = sg["from"], sg["to"]
        inner.append(_line(X(a), Y(b), X(c), Y(d), stroke=INK, stroke_width=2))
        for flag, (px, py) in (("open_start", (a, b)), ("open_end", (c, d))):
            if sg.get(flag):
                inner.append(_circle(X(px), Y(py), 4, fill=PAPER, stroke=INK, stroke_width=1.8))
        if sg.get("label"):
            labels.append(_text((X(a) + X(c)) / 2, (Y(b) + Y(d)) / 2 - 10, sg["label"], size=12))
    for c in s.get("curves", ()):
        if c.get("label") and c["pieces"]:
            ex, ey = c["pieces"][-1][-1]
            labels.append(_text(min(X(ex) + 4, left + pw - 4), Y(ey) - 9, c["label"], size=12, anchor="end"))
    for p in s.get("points", ()):
        px, py = X(p["at"][0]), Y(p["at"][1])
        if p.get("open"):
            inner.append(_circle(px, py, 4.2, fill=PAPER, stroke=INK, stroke_width=1.8))
        else:
            inner.append(_circle(px, py, 4.2, fill=INK, stroke="none"))
        if p.get("label"):
            right = px > left + pw - 50
            labels.append(
                _text(px + (-7 if right else 7), py - 9, p["label"], size=12, anchor="end" if right else "start")
            )
    out.append(
        _el("clipPath", _rect(left - 4, top - 4, pw + 8, ph + 8), id=clip)
        + _g(*inner, clip_path=f"url(#{clip})")
        + _g(*labels)
    )
    return "".join(out), top + ph + bottom


# ---------------------------------------------------------------- number line


def _number_line(s: dict, ids: _Ids) -> Body:
    lo, hi = s["range"]
    left, right, y = 30.0, 370.0, 62.0

    def X(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (right - left)

    out = [_line(left - 12, y, right + 12, y, stroke=INK, stroke_width=1.5)]
    out.append(_poly([(left - 12, y), (left - 6, y - 3.5), (left - 6, y + 3.5)], True, fill=INK, stroke="none"))
    out.append(_poly([(right + 12, y), (right + 6, y - 3.5), (right + 6, y + 3.5)], True, fill=INK, stroke="none"))
    tick = s["tick"]
    n0, n1 = math.ceil(lo / tick - 1e-9), math.floor(hi / tick + 1e-9)
    step = max(1, math.ceil(26 / (tick * (right - left) / (hi - lo))))
    for n in range(n0, n1 + 1):
        v = n * tick
        out.append(_line(X(v), y - 6, X(v), y + 6, stroke=INK, stroke_width=1.2))
        if s.get("tick_labels", True) and n % step == 0:
            out.append(_text(X(v), y + 20, _num_text(v), size=12))
    for iv in s.get("intervals", ()):
        out.append(_line(X(iv["from"]), y, X(iv["to"]), y, stroke=INK, stroke_width=5, stroke_linecap="butt"))
        for flag, v in (("open_start", iv["from"]), ("open_end", iv["to"])):
            hollow = iv.get(flag)
            out.append(_circle(X(v), y, 5.5, fill=PAPER if hollow else INK, stroke=INK, stroke_width=2))
    for r in s.get("rays", ()):
        end = left - 8 if r["direction"] == "left" else right + 8
        out.append(_line(X(r["from"]), y, end, y, stroke=INK, stroke_width=5, stroke_linecap="butt"))
        sgn = -1 if r["direction"] == "left" else 1
        out.append(_poly([(end + sgn * 9, y), (end, y - 7), (end, y + 7)], True, fill=INK, stroke="none"))
        out.append(_circle(X(r["from"]), y, 5.5, fill=PAPER if r.get("open") else INK, stroke=INK, stroke_width=2))
    for m in s.get("marks", ()):
        out.append(_circle(X(m["at"]), y, 5.5, fill=PAPER if m.get("open") else INK, stroke=INK, stroke_width=2))
        if m.get("label"):
            out.append(_text(X(m["at"]), y - 20, m["label"], size=13, weight="bold"))
    return "".join(out), 100


# ---------------------------------------------------------------- money

_COIN_RADIUS = {"5c": 18, "10c": 22, "20c": 26, "50c": 29, "$1": 24, "$2": 20}


def _money(s: dict, ids: _Ids) -> Body:
    out = []
    y = 12.0
    for g in s["groups"]:
        if g["label"]:
            out.append(_text(20, y + 8, g["label"], size=15, anchor="start", weight="bold"))
            y += 24
        x, row_h, row_top = 20.0, 62.0, y
        for it in g["items"]:
            coin = it["type"] == "coin"
            w = 2 * _COIN_RADIUS[it["denom"]] if coin else 78
            if x + w > CANVAS - 20 and x > 20:
                x = 20.0
                row_top += row_h
            cy = row_top + row_h / 2
            if coin:
                r = _COIN_RADIUS[it["denom"]]
                out.append(_circle(x + r, cy, r, fill=LIGHT, stroke=INK, stroke_width=1.8))
                out.append(_circle(x + r, cy, r - 3.5, stroke=DARK, stroke_width=0.8))
                out.append(_text(x + r, cy, it["denom"], size=13, weight="bold"))
            else:
                out.append(_rect(x, cy - 20, w, 40, rx=3, fill=LIGHT, stroke=INK, stroke_width=1.8))
                out.append(_rect(x + 4, cy - 16, w - 8, 32, rx=2, stroke=DARK, stroke_width=0.8))
                out.append(_text(x + w / 2, cy, it["denom"], size=15, weight="bold"))
            x += w + 10
        y = row_top + row_h + 10
    return "".join(out), y + 4


# ---------------------------------------------------------------- hundred chart


def _hundred_chart(s: dict, ids: _Ids) -> Body:
    cell, left, top = 32.0, 40.0, 10.0
    hidden, shaded, marked = set(s["hidden"]), set(s["shaded"]), set(s["marked"])
    out = []
    for r in range(10):
        for c in range(10):
            v = s["start"] + r * 10 + c
            x, y = left + c * cell, top + r * cell
            out.append(_rect(x, y, cell, cell, fill=MID if v in shaded else PAPER, stroke=INK, stroke_width=1))
            if v not in hidden:
                out.append(_text(x + cell / 2, y + cell / 2, v, size=13))
    for v in sorted(marked):
        r, c = divmod(v - s["start"], 10)
        out.append(_rect(left + c * cell + 2, top + r * cell + 2, cell - 4, cell - 4, rx=6, stroke=INK, stroke_width=3))
    return "".join(out), top + 10 * cell + 10


# ---------------------------------------------------------------- fraction models


def _fraction_model(s: dict, ids: _Ids) -> Body:
    models = s["models"]
    n = len(models)
    hatch = ids("hatch")
    defs = _el(
        "defs",
        _el(
            "pattern",
            _rect(0, 0, 6, 6, fill=MID, stroke="none") + _line(0, 0, 0, 6, stroke=INK, stroke_width=1.6),
            id=hatch,
            patternUnits="userSpaceOnUse",
            width=6,
            height=6,
            patternTransform="rotate(45)",
        ),
    )
    fill_on, fill_off = f"url(#{hatch})", PAPER
    out = [defs]
    if models[0]["shape"] == "bar":
        bw, bh = 320.0, 42.0
        y = 16.0
        for m in models:
            shaded = set(m["shaded"])
            if m["label"]:
                out.append(_text(22, y + bh / 2, m["label"], size=16, weight="bold"))
            pw = bw / m["parts"]
            for i in range(m["parts"]):
                out.append(
                    _rect(
                        44 + i * pw, y, pw, bh, fill=fill_on if i in shaded else fill_off, stroke=INK, stroke_width=1.8
                    )
                )
            y += bh + 26
        return "".join(out), y - 10
    size = min(150.0, (CANVAS - 40) / n - 14)
    gap = 14.0
    x = (CANVAS - (n * size + (n - 1) * gap)) / 2
    top = 40.0
    for m in models:
        shaded = set(m["shaded"])
        if m["label"]:
            out.append(_text(x + size / 2, 20, m["label"], size=16, weight="bold"))
        if m["shape"] == "rectangle":
            h = size * 0.7
            cw, ch = size / m["cols"], h / m["rows"]
            for i in range(m["parts"]):
                r, c = divmod(i, m["cols"])
                out.append(
                    _rect(
                        x + c * cw,
                        top + r * ch,
                        cw,
                        ch,
                        fill=fill_on if i in shaded else fill_off,
                        stroke=INK,
                        stroke_width=1.8,
                    )
                )
            height = h
        else:
            cx, cy, rad = x + size / 2, top + size / 2, size / 2
            parts = m["parts"]
            for i in range(parts):
                a0, a1 = 90 - i * 360 / parts, 90 - (i + 1) * 360 / parts  # anticlockwise degrees, clockwise travel
                x0_, y0_ = _pt(cx, cy, rad, a0)
                x1_, y1_ = _pt(cx, cy, rad, a1)
                d = f"M{_n(cx)} {_n(cy)}L{_n(x0_)} {_n(y0_)}A{_n(rad)} {_n(rad)} 0 0 1 {_n(x1_)} {_n(y1_)}Z"
                out.append(
                    _path(
                        d,
                        fill=fill_on if i in shaded else fill_off,
                        stroke=INK,
                        stroke_width=1.8,
                        stroke_linejoin="round",
                    )
                )
            height = size
        x += size + gap
    return "".join(out), top + height + 16


# ---------------------------------------------------------------- counters


def _counter(shape: str, cx: float, cy: float, r: float, colour: int) -> str:
    """Colour 1 is a solid counter; colour 2 is a white counter with a double outline."""
    solid = colour == 1
    kw: dict[str, Any] = {"fill": INK if solid else PAPER, "stroke": INK, "stroke_width": 1.5 if solid else 2.2}
    inner: str | None = None
    if shape == "circle":
        main = _circle(cx, cy, r, **kw)
        inner = None if solid else _circle(cx, cy, r * 0.5, stroke=INK, stroke_width=1.5)
    elif shape == "square":
        main = _rect(cx - r, cy - r, 2 * r, 2 * r, **kw)
        inner = None if solid else _rect(cx - r * 0.5, cy - r * 0.5, r, r, stroke=INK, stroke_width=1.5)
    elif shape == "diamond":
        main = _poly([(cx, cy - r * 1.15), (cx + r * 1.15, cy), (cx, cy + r * 1.15), (cx - r * 1.15, cy)], True, **kw)
        inner = None
        if not solid:
            q = r * 0.55
            inner = _poly([(cx, cy - q), (cx + q, cy), (cx, cy + q), (cx - q, cy)], True, stroke=INK, stroke_width=1.5)
    else:
        main = _poly(_star(cx, cy, r * 1.2), True, stroke_linejoin="round", **kw)
        inner = None if solid else _poly(_star(cx, cy, r * 0.55), True, stroke=INK, stroke_width=1.5)
    return main + (inner or "")


_DIE_DOTS = {
    1: [(0.5, 0.5)],
    2: [(0.27, 0.27), (0.73, 0.73)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.27, 0.27), (0.73, 0.27), (0.27, 0.73), (0.73, 0.73)],
    5: [(0.27, 0.27), (0.73, 0.27), (0.5, 0.5), (0.27, 0.73), (0.73, 0.73)],
    6: [(0.3, 0.2), (0.3, 0.5), (0.3, 0.8), (0.7, 0.2), (0.7, 0.5), (0.7, 0.8)],
}


def _counters(s: dict, ids: _Ids) -> Body:
    layout, shape = s["layout"], s["shape"]
    out = []
    if layout == "ten_frame":
        frames = s["frames"]
        box = 36.0
        gap = 24.0
        total = len(frames) * 5 * box + (len(frames) - 1) * gap
        x0 = (CANVAS - total) / 2
        for fi, frame in enumerate(frames):
            fx = x0 + fi * (5 * box + gap)
            for i, c in enumerate(frame):
                r, col = divmod(i, 5)
                bx, by = fx + col * box, 16 + r * box
                out.append(_rect(bx, by, box, box, stroke=INK, stroke_width=2))
                if c:
                    out.append(_counter(shape, bx + box / 2, by + box / 2, 12, c))
        return "".join(out), 16 + 2 * box + 16
    if layout == "array":
        rows, cols = s["rows"], s["cols"]
        sp = 30.0
        x0 = (CANVAS - cols * sp) / 2 + sp / 2
        colours = s.get("colours")
        for r in range(rows):
            for c in range(cols):
                col = colours[r][c] if colours else 1
                out.append(_counter(shape, x0 + c * sp, 24 + r * sp, 11, col))
        return "".join(out), 24 + (rows - 1) * sp + 24
    if layout == "groups":
        groups = s["groups"]
        cell, per_row = 84.0, 4
        n_rows = math.ceil(len(groups) / per_row)
        for i, count in enumerate(groups):
            r, c = divmod(i, per_row)
            n_in_row = min(per_row, len(groups) - r * per_row)
            cx = (CANVAS - n_in_row * cell) / 2 + c * cell + cell / 2
            cy = 46 + r * 90
            out.append(_circle(cx, cy, 36, stroke=INK, stroke_width=1.8, stroke_dasharray="5 3"))
            cols = 1 if count <= 1 else 2 if count <= 4 else 3
            rows_in = math.ceil(count / cols) if count else 0
            for k in range(count):
                kr, kc = divmod(k, cols)
                in_row = min(cols, count - kr * cols)
                px = cx + (kc - (in_row - 1) / 2) * 17
                py = cy + (kr - (rows_in - 1) / 2) * 17
                out.append(_counter(shape, px, py, 6.5, 1))
        return "".join(out), 46 + (n_rows - 1) * 90 + 50
    if layout == "dice":
        faces = s["faces"]
        side, gap = 78.0, 24.0
        x0 = (CANVAS - (len(faces) * side + (len(faces) - 1) * gap)) / 2
        for i, f in enumerate(faces):
            fx = x0 + i * (side + gap)
            out.append(_rect(fx, 16, side, side, rx=12, fill=PAPER, stroke=INK, stroke_width=2.5))
            for ux, uy in _DIE_DOTS[f["n"]]:
                out.append(_counter("circle", fx + ux * side, 16 + uy * side, 7.5, f["colour"]))
        return "".join(out), 16 + side + 16
    w, h = 320.0, 160.0
    for d in s["dots"]:
        out.append(_counter(shape, 40 + d["at"][0] * w, 14 + d["at"][1] * h, 10.5, d["colour"]))
    return "".join(out), h + 28


# ---------------------------------------------------------------- base ten


def _base_ten(s: dict, ids: _Ids) -> Body:
    flats, tens, ones = s["hundreds"], s["tens"], s["ones"]
    u = 5.0 if flats else 8.0  # with no flats there is room to draw the rods and cubes larger
    out = []
    y = 14.0
    if flats:
        per_row = 6
        for i in range(flats):
            r, c = divmod(i, per_row)
            fx, fy = 20 + c * (10 * u + 8), y + r * (10 * u + 8)
            out.append(_rect(fx, fy, 10 * u, 10 * u, fill=LIGHT, stroke=INK, stroke_width=1.5))
            for k in range(1, 10):
                out.append(_line(fx + k * u, fy, fx + k * u, fy + 10 * u, stroke=DARK, stroke_width=0.5))
                out.append(_line(fx, fy + k * u, fx + 10 * u, fy + k * u, stroke=DARK, stroke_width=0.5))
        y += math.ceil(flats / per_row) * (10 * u + 8) + 8
    x = 20.0
    row_h = 0.0
    if tens:
        for i in range(tens):
            rx_ = x + i * (u + 3)
            out.append(_rect(rx_, y, u, 10 * u, fill=LIGHT, stroke=INK, stroke_width=1.2))
            for k in range(1, 10):
                out.append(_line(rx_, y + k * u, rx_ + u, y + k * u, stroke=DARK, stroke_width=0.5))
        x += tens * (u + 3) + 24
        row_h = 10 * u
    if ones:
        per_col = 10 if ones > 10 else ones
        for i in range(ones):
            c, r = divmod(i, per_col)
            out.append(_rect(x + c * (u + 3), y + r * (u + 3), u, u, fill=LIGHT, stroke=INK, stroke_width=1.2))
        row_h = max(row_h, per_col * (u + 3))
    return "".join(out), y + row_h + 14


# ---------------------------------------------------------------- clock


def _clock_face(cx: float, cy: float, r: float, s: dict) -> list[str]:
    out = [_circle(cx, cy, r, fill=PAPER, stroke=INK, stroke_width=2.5)]
    ticks = s["minute_ticks"]
    for i in range(60):
        major = i % 5 == 0
        if not major and not ticks:
            continue
        a = i * 6
        long_ = 0.12 if major else 0.06
        p1, p2 = _clock_pt(cx, cy, r * (1 - long_), a), _clock_pt(cx, cy, r - 1, a)
        out.append(_line(*p1, *p2, stroke=INK, stroke_width=2 if major else 0.9))
    numerals = s["numerals"]
    for h in range(1, 13):
        if numerals == "none" or (numerals == "quarters" and h % 3):
            continue
        x, y = _clock_pt(cx, cy, r * 0.72, h * 30)
        out.append(_text(x, y, h, size=r * 0.2, weight="bold"))
    hx, hy = _clock_pt(cx, cy, r * 0.5, s["hour_angle"])
    mx, my = _clock_pt(cx, cy, r * 0.88, s["minute_angle"])
    out.append(_line(cx, cy, hx, hy, stroke=INK, stroke_width=5, stroke_linecap="round"))
    out.append(_line(cx, cy, mx, my, stroke=INK, stroke_width=3, stroke_linecap="round"))
    out.append(_circle(cx, cy, 4, fill=INK, stroke="none"))
    return out


def _clock(s: dict, ids: _Ids) -> Body:
    r = 90.0
    cx, cy = CANVAS / 2, 14 + r
    out = _clock_face(cx, cy, r, s)
    h = cy + r + 14
    if s.get("caption"):
        out.append(_text(cx, cy + r + 18, s["caption"], size=15, weight="bold"))
        h += 24
    return "".join(out), h


def _clock_set(s: dict, ids: _Ids) -> Body:
    clocks = s["clocks"]
    n = len(clocks)
    cell = CANVAS / n if n < 4 else CANVAS / 2
    r = min(80.0, cell / 2 - 10)
    per_row = n if n < 4 else 2
    out = []
    height = 0.0
    for i, c in enumerate(clocks):
        row, col = divmod(i, per_row)
        cx = cell * col + cell / 2
        cy = 14 + r + row * (2 * r + 50)
        out.extend(_clock_face(cx, cy, r, c))
        out.append(_text(cx, cy + r + 18, c["caption"], size=15, weight="bold"))
        height = cy + r + 34
    return "".join(out), height


# ---------------------------------------------------------------- protractor


def _protractor(s: dict, ids: _Ids) -> Body:
    cx, cy, R = 200.0, 200.0, 118.0
    rot = s["rotation"]
    local = []
    # frame: baseline, outer arc
    local.append(_line(cx - R, cy, cx + R, cy, stroke=INK, stroke_width=1.5))
    local.append(
        _path(f"M{_n(cx + R)} {_n(cy)}A{_n(R)} {_n(R)} 0 0 0 {_n(cx - R)} {_n(cy)}", stroke=INK, stroke_width=1.5)
    )
    local.append(
        _path(
            f"M{_n(cx + R - 35)} {_n(cy)}A{_n(R - 35)} {_n(R - 35)} 0 0 0 {_n(cx - R + 35)} {_n(cy)}",
            stroke=MID,
            stroke_width=0.8,
        )
    )
    local.append(_line(cx - 7, cy, cx + 7, cy, stroke=INK, stroke_width=1.5))
    local.append(_line(cx, cy - 7, cx, cy + 7, stroke=INK, stroke_width=1.5))
    for t in range(0, 181, 5):
        long_ = t % 10 == 0
        p1, p2 = _pt(cx, cy, R - (11 if long_ else 6), t), _pt(cx, cy, R, t)
        local.append(_line(*p1, *p2, stroke=INK, stroke_width=1 if long_ else 0.6))
        if long_:
            ox, oy = _pt(cx, cy, R - 23, t)
            ix, iy = _pt(cx, cy, R - 47, t)
            local.append(_line(*_pt(cx, cy, R - 35, t), *_pt(cx, cy, R - 39, t), stroke=INK, stroke_width=0.8))
            local.append(_text(ox, oy, t, size=8))
            local.append(_text(ix, iy, 180 - t, size=8))
    # rays, in the protractor's own frame
    base = (s["base_direction"] - rot) % 360
    second = (s["second_direction"] - rot) % 360
    ray = 168.0
    for ang in (base, second):
        end = _pt(cx, cy, ray, ang)
        local.append(_line(cx, cy, *end, stroke=INK, stroke_width=2.5, stroke_linecap="round"))
    delta = (second - base) % 360
    if s["reflex"]:
        local.append(_path(_arc(cx, cy, R + 16, base, second, clockwise=delta < 180), stroke=INK, stroke_width=1.8))
    else:
        local.append(_path(_arc(cx, cy, 34, base, second, clockwise=delta > 180), stroke=INK, stroke_width=1.8))
    out = [_g(*local, transform=f"rotate({_n(-rot)} {_n(cx)} {_n(cy)})")]

    def to_screen(p: tuple[float, float]) -> tuple[float, float]:
        t = math.radians(rot)
        dx, dy = p[0] - cx, cy - p[1]
        return cx + dx * math.cos(t) - dy * math.sin(t), cy - (dx * math.sin(t) + dy * math.cos(t))

    names = s.get("names") or {}
    if names.get("vertex"):
        out.append(_text(cx - 12, cy + 14, names["vertex"], size=14, weight="bold"))
    for key, ang in (("base", base), ("second", second)):
        if names.get(key):
            x, y = to_screen(_pt(cx, cy, ray + 14, ang))
            out.append(_text(x, y, names[key], size=14, weight="bold"))
    return "".join(out), 2 * cy


# ---------------------------------------------------------------- unit strip


def _unit_shape(style: str, x: float, y: float, w: float, h: float) -> str:
    base = {"fill": PAPER, "stroke": INK, "stroke_width": 1.4}
    if style == "paper_clip":
        return _rect(x, y, w, h * 0.7, rx=h * 0.35, **base) + _rect(
            x + w * 0.2, y + h * 0.18, w * 0.6, h * 0.34, rx=h * 0.17, stroke=INK, stroke_width=1
        )
    if style == "cube":
        return (
            _rect(x, y, w, h, **base)
            + _line(x, y + h * 0.3, x + w, y + h * 0.3, stroke=INK, stroke_width=0.8)
            + _line(x + w * 0.3, y + h * 0.3, x + w * 0.3, y + h, stroke=INK, stroke_width=0.8)
        )
    if style == "block":
        return _rect(x, y, w, h, rx=1.5, **{**base, "fill": LIGHT}) + _line(
            x + 2, y + h / 2, x + w - 2, y + h / 2, stroke=DARK, stroke_width=0.8
        )
    if style == "footprint":
        cx = x + w / 2
        toes = "".join(
            _circle(x + w * (0.22 + 0.19 * i), y + h * 0.1, h * 0.09, fill=INK, stroke="none") for i in range(4)
        )
        return _el("ellipse", cx=cx, cy=y + h * 0.62, rx=w * 0.4, ry=h * 0.34, **{**base, "fill": LIGHT}) + toes
    return _circle(x + w / 2, y + h / 2, min(w, h) / 2, **base)


def _unit_strip(s: dict, ids: _Ids) -> Body:
    rows = s["rows"]
    left = 78.0
    extent = max(max(r["length"], *(u["start"] + u["length"] for u in r["units"])) for r in rows)
    u = min(40.0, (CANVAS - left - 24) / extent)
    out = []
    y = 20.0
    h = min(22.0, u * 0.7 + 4)
    for r in rows:
        bar_y = y + 8
        out.append(_text(left - 10, bar_y, r["label"], size=13, anchor="end"))
        out.append(_line(left, bar_y, left + r["length"] * u, bar_y, stroke=INK, stroke_width=2.5))
        for ex in (0, r["length"]):
            out.append(_line(left + ex * u, bar_y - 7, left + ex * u, bar_y + 7, stroke=INK, stroke_width=2))
        for un in r["units"]:
            out.append(_unit_shape(s["unit_style"], left + un["start"] * u, bar_y + 10, un["length"] * u, h))
        y += 26 + h + 10
    out.append(_text(CANVAS / 2, y + 2, f"Unit: {s['unit']}", size=12))
    return "".join(out), y + 18


# ---------------------------------------------------------------- scale


def _scale_ticks(s: dict) -> list[tuple[Fraction, str]]:
    lo, hi, major, minor = (_frac(s[k]) for k in ("min", "max", "major", "minor"))
    ticks = []
    steps = int((hi - lo) / minor)
    per_major = int(major / minor)
    for i in range(steps + 1):
        ticks.append((lo + i * minor, "major" if i % per_major == 0 else "minor"))
    for e in s.get("extra_ticks", ()):
        ticks.append((_frac(e), "extra"))
    return ticks


_TICK_LEN = {"major": 16, "minor": 8, "extra": 4}


def _tri(tip: tuple[float, float], dx: float, dy: float, half: float = 6) -> str:
    """A filled arrowhead with its tip at ``tip``, pointing opposite to the direction (dx, dy) it grows in."""
    px, py = -dy, dx
    return _poly(
        [
            tip,
            (tip[0] + dx * 12 + px * half, tip[1] + dy * 12 + py * half),
            (tip[0] + dx * 12 - px * half, tip[1] + dy * 12 - py * half),
        ],
        True,
        fill=INK,
        stroke="none",
    )


def _scale(s: dict, ids: _Ids) -> Body:
    variant = s["variant"]
    lo, hi = float(_frac(s["min"])), float(_frac(s["max"]))
    span = hi - lo
    ticks = _scale_ticks(s)
    unit = s["unit"]
    out: list[str] = []
    level = s.get("level")

    if variant == "ruler":
        left, right, top, bot = 30.0, 370.0, 70.0, 112.0

        def X(v: float) -> float:
            return left + (float(v) - lo) / span * (right - left)

        out.append(_rect(left, top, right - left, bot - top, fill=LIGHT, stroke=INK, stroke_width=1.8))
        for v, kind in ticks:
            out.append(
                _line(X(v), bot, X(v), bot - _TICK_LEN[kind], stroke=INK, stroke_width=1.4 if kind == "major" else 0.9)
            )
            if kind == "major":
                out.append(_text(X(v), bot - 26, _num_text(float(v)), size=12))
        out.append(_text(right, bot + 12, unit, size=12, anchor="end", weight="bold"))
        obj = s.get("object")
        if obj:
            a, b = X(obj["start"]), X(obj["end"])
            out.append(_rect(a, 26, b - a, 26, fill=PAPER, stroke=INK, stroke_width=1.8))
            if obj.get("label"):
                out.append(_text((a + b) / 2, 39, obj["label"], size=13))
            out.append(_line(a, 52, a, bot, stroke=INK, stroke_width=1, stroke_dasharray="3 2"))
            out.append(_line(b, 52, b, bot, stroke=INK, stroke_width=1, stroke_dasharray="3 2"))
        for m in s.get("marks", ()):
            out.append(_tri((X(m["value"]), bot + 2), 0, 1))
            out.append(_text(X(m["value"]), bot + 26, m["label"], size=14, weight="bold"))
        return "".join(out), 148

    if variant == "thermometer":
        tx, ybot, ytop = 150.0, 300.0, 36.0

        def Y(v: float) -> float:
            return ybot - (float(v) - lo) / span * (ybot - ytop)

        out.append(
            _path(
                f"M{_n(tx - 10)} 312L{_n(tx - 10)} 30A10 10 0 0 1 {_n(tx + 10)} 30L{_n(tx + 10)} 312",
                stroke=INK,
                stroke_width=1.8,
            )
        )
        out.append(_circle(tx, 322, 20, fill=PAPER, stroke=INK, stroke_width=1.8))
        out.append(_circle(tx, 322, 15, fill=DARK, stroke="none"))
        if level is not None:
            out.append(_rect(tx - 5, Y(level), 10, 322 - Y(level), fill=DARK, stroke="none"))
        for v, kind in ticks:
            out.append(
                _line(
                    tx + 10,
                    Y(v),
                    tx + 10 + _TICK_LEN[kind],
                    Y(v),
                    stroke=INK,
                    stroke_width=1.4 if kind == "major" else 0.9,
                )
            )
            if kind == "major":
                out.append(_text(tx + 34, Y(v), _num_text(float(v)), size=12, anchor="start"))
        out.append(_text(tx + 34, 10, unit, size=13, anchor="start", weight="bold"))
        for m in s.get("marks", ()):
            out.append(_tri((tx - 12, Y(m["value"])), -1, 0))
            out.append(_text(tx - 38, Y(m["value"]), m["label"], size=14, weight="bold"))
        return "".join(out), 352

    if variant == "jug":
        left, right, ybot, ytop = 130.0, 210.0, 300.0, 50.0

        def Y(v: float) -> float:
            return ybot - (float(v) - lo) / span * (ybot - ytop)

        if level is not None:
            out.append(_rect(left + 1, Y(level), right - left - 2, 307 - Y(level), fill=LIGHT, stroke="none"))
            out.append(_line(left + 1, Y(level), right - 1, Y(level), stroke=INK, stroke_width=1.5))
        body = (
            f"M{_n(left - 8)} 26L{_n(left)} 36L{_n(left)} 300Q{_n(left)} 308 {_n(left + 8)} 308"
            f"L{_n(right - 8)} 308Q{_n(right)} 308 {_n(right)} 300L{_n(right)} 36L{_n(right + 8)} 26"
        )
        out.append(_path(body, stroke=INK, stroke_width=2, stroke_linejoin="round"))
        for v, kind in ticks:
            out.append(
                _line(
                    right, Y(v), right - _TICK_LEN[kind], Y(v), stroke=INK, stroke_width=1.4 if kind == "major" else 0.9
                )
            )
            if kind == "major":
                out.append(_text(right + 12, Y(v), _num_text(float(v)), size=12, anchor="start"))
        out.append(_text((left + right) / 2, 10, unit, size=13, weight="bold"))
        for m in s.get("marks", ()):
            out.append(_tri((left - 4, Y(m["value"])), -1, 0))
            out.append(_text(left - 30, Y(m["value"]), m["label"], size=14, weight="bold"))
        return "".join(out), 322

    # dial
    cx, cy, r = 200.0, 170.0, 118.0
    start, sweep = s["start_angle"], s["sweep"]

    def angle(v: float) -> float:
        return start + (float(v) - lo) / span * sweep

    out.append(_circle(cx, cy, r, fill=PAPER, stroke=INK, stroke_width=2.5))
    for v, kind in ticks:
        a = angle(v)
        out.append(
            _line(
                *_clock_pt(cx, cy, r - _TICK_LEN[kind], a),
                *_clock_pt(cx, cy, r - 1, a),
                stroke=INK,
                stroke_width=1.6 if kind == "major" else 0.9,
            )
        )
        if kind == "major":
            x, y = _clock_pt(cx, cy, r - 30, a)
            out.append(_text(x, y, _num_text(float(v)), size=12))
    out.append(_text(cx, cy + 52, unit, size=13, weight="bold"))
    if level is not None:
        out.append(
            _line(cx, cy, *_clock_pt(cx, cy, r - 18, angle(level)), stroke=INK, stroke_width=3, stroke_linecap="round")
        )
    out.append(_circle(cx, cy, 6, fill=INK, stroke="none"))
    for m in s.get("marks", ()):
        a = angle(m["value"])
        tip = _clock_pt(cx, cy, r + 2, a)
        dxu, dyu = math.sin(math.radians(a)), -math.cos(math.radians(a))
        out.append(_tri(tip, dxu, dyu))
        out.append(_text(*_clock_pt(cx, cy, r + 30, a), m["label"], size=14, weight="bold"))
    return "".join(out), cy + r + 40


# ---------------------------------------------------------------- diagram


def _diagram(s: dict, ids: _Ids) -> Body:
    fw, fh = s["frame"]
    margin = 34.0
    k = min((CANVAS - 2 * margin) / fw, 290.0 / fh)
    ox, oy = (CANVAS - fw * k) / 2, margin
    pos = {p["id"]: (ox + p["at"][0] * k, oy + (fh - p["at"][1]) * k) for p in s["points"]}
    centroid = (sum(p[0] for p in pos.values()) / len(pos), sum(p[1] for p in pos.values()) / len(pos))

    def unit(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        return ((b[0] - a[0]) / d, (b[1] - a[1]) / d) if d else (0.0, 0.0)

    seg_pairs = {frozenset((sg["from"], sg["to"])) for sg in s.get("segments", ())}
    out = []
    incident: dict[str, list[str]] = {}
    for sg in s.get("segments", ()):
        incident.setdefault(sg["from"], []).append(sg["to"])
        incident.setdefault(sg["to"], []).append(sg["from"])
    for poly in s.get("polygons", ()):
        pts = poly["points"]
        coords = [pos[p] for p in pts]
        if poly.get("fill"):
            out.append(_poly(coords, True, fill=LIGHT, stroke="none"))
        for i, a in enumerate(pts):
            b = pts[(i + 1) % len(pts)]
            incident.setdefault(a, []).append(b)
            incident.setdefault(b, []).append(a)
            if frozenset((a, b)) not in seg_pairs:
                out.append(_line(*pos[a], *pos[b], stroke=INK, stroke_width=2, stroke_linecap="round"))
    marks = []
    for sg in s.get("segments", ()):
        a, b = pos[sg["from"]], pos[sg["to"]]
        out.append(
            _line(
                *a,
                *b,
                stroke=INK,
                stroke_width=2,
                stroke_linecap="round",
                stroke_dasharray="6 4" if sg.get("dashed") else None,
            )
        )
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ux, uy = unit(a, b)
        nx, ny = -uy, ux
        if (mid[0] - centroid[0]) * nx + (mid[1] - centroid[1]) * ny < 0:
            nx, ny = -nx, -ny
        for i in range(sg.get("ticks") or 0):
            off = (i - (sg["ticks"] - 1) / 2) * 5
            cx_, cy_ = mid[0] + ux * off, mid[1] + uy * off
            marks.append(_line(cx_ - nx * 5, cy_ - ny * 5, cx_ + nx * 5, cy_ + ny * 5, stroke=INK, stroke_width=1.6))
        for i in range(sg.get("arrows") or 0):
            off = (i - (sg["arrows"] - 1) / 2) * 8
            cx_, cy_ = mid[0] + ux * off, mid[1] + uy * off
            tip = (cx_ + ux * 4, cy_ + uy * 4)
            marks.append(
                _poly(
                    [
                        (cx_ - ux * 3 + nx * 5, cy_ - uy * 3 + ny * 5),
                        tip,
                        (cx_ - ux * 3 - nx * 5, cy_ - uy * 3 - ny * 5),
                    ],
                    stroke=INK,
                    stroke_width=1.6,
                    stroke_linejoin="round",
                )
            )
        if sg.get("label"):
            marks.append(_text(mid[0] + nx * 13, mid[1] + ny * 13, sg["label"], size=13))
    out.extend(marks)

    for a in s.get("angles", ()):
        at = pos[a["at"]]
        t1 = math.degrees(math.atan2(-(pos[a["from"]][1] - at[1]), pos[a["from"]][0] - at[0])) % 360
        t2 = math.degrees(math.atan2(-(pos[a["to"]][1] - at[1]), pos[a["to"]][0] - at[0])) % 360
        sweep = (t2 - t1) % 360
        if a.get("right"):
            u1, u2 = unit(at, pos[a["from"]]), unit(at, pos[a["to"]])
            q = 12.0
            out.append(
                _poly(
                    [
                        (at[0] + u1[0] * q, at[1] + u1[1] * q),
                        (at[0] + (u1[0] + u2[0]) * q, at[1] + (u1[1] + u2[1]) * q),
                        (at[0] + u2[0] * q, at[1] + u2[1] * q),
                    ],
                    stroke=INK,
                    stroke_width=1.5,
                )
            )
            continue
        ra = 24.0
        out.append(_path(_arc(at[0], at[1], ra, t1, t2), stroke=INK, stroke_width=1.5))
        if a.get("label"):
            lx, ly = _pt(at[0], at[1], ra + 15, t1 + sweep / 2)
            out.append(_text(lx, ly, a["label"], size=13))

    for p in s["points"]:
        if not p.get("label"):
            continue
        here = pos[p["id"]]
        sx_ = sy_ = 0.0
        first = None
        for other in incident.get(p["id"], ()):
            ux, uy = unit(here, pos[other])
            sx_ += ux
            sy_ += uy
            first = first or (ux, uy)
        norm = math.hypot(sx_, sy_)
        if norm > 0.3:
            dx_, dy_ = -sx_ / norm, -sy_ / norm
        elif first:
            dx_, dy_ = -first[1], first[0]
        else:
            dx_, dy_ = 0.0, -1.0
        out.append(_dot_label(here, dx_, dy_, p["label"]))
    height = oy + fh * k + margin
    if s.get("not_to_scale"):
        out.append(_text(CANVAS / 2, height - 6, "Not drawn to scale", size=11))
        height += 10
    return "".join(out), height


def _dot_label(here: tuple[float, float], dx: float, dy: float, label: str) -> str:
    return _text(here[0] + dx * 13, here[1] + dy * 13, label, size=15, weight="bold")


# ---------------------------------------------------------------- tree diagram


def _tree(s: dict, ids: _Ids) -> Body:
    root = s["root"]
    row_h, top = 40.0, 28.0
    leaf_counter = [0]
    depth = [0]

    def lay(node: dict, level: int) -> dict:
        depth[0] = max(depth[0], level)
        kids = [lay(c, level + 1) for c in node.get("children") or ()]
        if kids:
            y = (kids[0]["y"] + kids[-1]["y"]) / 2
        else:
            y = top + leaf_counter[0] * row_h
            leaf_counter[0] += 1
        return {"node": node, "level": level, "y": y, "kids": kids}

    tree = lay(root, 0)
    d = depth[0]
    widths = [0.0] * (d + 1)

    def measure(t: dict) -> None:
        if t["level"]:
            widths[t["level"]] = max(widths[t["level"]], 6.4 * len(_clean_text(t["node"]["label"])) + 4)
        for k_ in t["kids"]:
            measure(k_)

    measure(tree)
    outcome_w = 104.0
    avail = CANVAS - 14 - 26 - outcome_w - sum(widths) - 8 * d
    gap = max(34.0, avail / d)
    xs = [26.0]
    for lvl in range(1, d + 1):
        xs.append(xs[-1] + (widths[lvl - 1] + 8 if lvl > 1 else 0) + gap)
    out = []
    labels = []

    def draw(t: dict) -> None:
        lvl = t["level"]
        x_start = xs[lvl] + (widths[lvl] + 8 if lvl else 0)
        for kid in t["kids"]:
            x_end = xs[lvl + 1]
            out.append(_line(x_start, t["y"], x_end, kid["y"], stroke=INK, stroke_width=1.6))
            mx, my = (x_start + x_end) / 2, (t["y"] + kid["y"]) / 2
            branch = kid["node"]["branch"]
            if branch == "?":
                labels.append(_rect(mx - 11, my - 8, 22, 16, fill=PAPER, stroke=INK, stroke_width=1.4))
            else:
                labels.append(_text(mx, my - 9, branch, size=12))
            labels.append(_text(x_end + 4, kid["y"], kid["node"]["label"], size=13, anchor="start", weight="bold"))
            if not kid["kids"]:
                ox_ = x_end + widths[lvl + 1] + 14
                node = kid["node"]
                p = node.get("outcome_p")
                labels.append(_text(ox_, kid["y"] - (7 if p else 0), node["outcome"], size=12, anchor="start"))
                if p:
                    if p == "?":
                        labels.append(_text(ox_, kid["y"] + 8, "P =", size=11, anchor="start"))
                        labels.append(_rect(ox_ + 24, kid["y"] + 1, 24, 14, fill=PAPER, stroke=INK, stroke_width=1.2))
                    else:
                        labels.append(_text(ox_, kid["y"] + 8, f"P = {p}", size=11, anchor="start"))
            draw(kid)

    draw(tree)
    out.append(_circle(xs[0], tree["y"], 3.5, fill=INK, stroke="none"))
    if root.get("label"):
        labels.append(_text(xs[0], tree["y"] - 14, root["label"], size=12, anchor="middle", weight="bold"))
    return "".join(out) + "".join(labels), top + leaf_counter[0] * row_h - row_h + 30


# ---------------------------------------------------------------- venn

_VENN_TWO = {"A": (112, 135), "B": (288, 135), "AB": (200, 135), "none": (356, 226)}
_VENN_THREE = {
    "A": (118, 100),
    "B": (282, 100),
    "C": (200, 232),
    "AB": (200, 82),
    "AC": (158, 166),
    "BC": (242, 166),
    "ABC": (200, 138),
    "none": (360, 272),
}


def _venn(s: dict, ids: _Ids) -> Body:
    sets = s["sets"]
    two = len(sets) == 2
    out = []
    if two:
        circles = [(150.0, 135.0, 80.0), (250.0, 135.0, 80.0)]
        rect_h, spots = 226.0, _VENN_TWO
        name_at = [(120.0, 40.0), (280.0, 40.0)]
    else:
        circles = [(155.0, 115.0, 70.0), (245.0, 115.0, 70.0), (200.0, 190.0, 70.0)]
        rect_h, spots = 286.0, _VENN_THREE
        name_at = [(112.0, 36.0), (288.0, 36.0), (310.0, 232.0)]
    out.append(_rect(10, 10, 380, rect_h, stroke=INK, stroke_width=2))
    out.append(_text(20, 24, s["universe"], size=14, anchor="start", weight="bold"))
    for (cx, cy, r), meta, (nx, ny) in zip(circles, sets, name_at):
        out.append(_circle(cx, cy, r, stroke=INK, stroke_width=2))
        out.append(_text(nx, ny, meta.get("name") or meta["id"], size=14, weight="bold"))
    position = {meta["id"]: "ABC"[i] for i, meta in enumerate(sets)}
    for key, text in s["regions"].items():
        spot = spots["none"] if key == "none" else spots["".join(sorted(position[ch] for ch in key))]
        out.append(_text(spot[0], spot[1], text, size=13, weight="bold"))
    return "".join(out), rect_h + 22


# ---------------------------------------------------------------- data displays
#
# bar_chart, picture_graph, dot_plot, histogram, box_plot and pie_chart are all "read the value off the
# picture" figures. Each is drawn to its scale and nothing else: no bar, row, stack, sector or box ever
# carries its own value as text. The only numerals are the ticks on a labelled axis (a learner needs those
# to read with); the only words are the ones the spec carries (category, row, series, panel and sector
# names, axis titles, the picture-graph key).

_AX_FONT = 11
_CHAR_W = 6.2  # a generous width for one character at 11-12 units, for fitting labels


def _text_w(s: Any, size: float = _AX_FONT) -> float:
    return len(_clean_text(s)) * _CHAR_W * size / 12 + 2


def _rotated_text(x: float, y: float, s: Any, deg: float, *, size: float = 12, anchor: str = "middle") -> str:
    """``_text`` turned ``deg`` degrees (negative is anticlockwise) about its own anchor point."""
    t = _text(x, y, s, size=size, anchor=anchor)
    return t.replace("<text ", f'<text transform="rotate({_n(deg)} {_n(x)} {_n(y)})" ', 1)


def _grid_marks(axis: dict) -> list[tuple[float, bool]]:
    """``(value, is_minor)`` for every gridline of a ``{"max", "step", "minor"}`` value axis, 0 first."""
    steps = round(axis["max"] / axis["step"])
    marks: list[tuple[float, bool]] = []
    for i in range(steps + 1):
        marks.append((i * axis["step"], False))
        if axis["minor"] and i < steps:
            marks.append(((i + 0.5) * axis["step"], True))
    return marks


def _grid_line(minor: bool) -> dict:
    if minor:
        return {"stroke": "#dcdcdc", "stroke_width": 0.7, "stroke_dasharray": "2 3"}
    return {"stroke": MID, "stroke_width": 0.8}


def _every(spacing: float, need: float) -> int:
    """Label every n-th tick so labels ``need`` units wide never overlap ticks ``spacing`` apart."""
    return max(1, math.ceil(need / max(spacing, 1e-6)))


def _axis_title(x: float, y: float, s: Any, *, vertical: bool = False) -> str:
    return _rotated_text(x, y, s, -90, size=12) if vertical else _text(x, y, s, size=12)


def _bar_chart(s: dict, ids: _Ids) -> Body:
    horizontal = s["orientation"] == "horizontal"
    cats, series, axis = s["categories"], s.get("series"), s["value_axis"]
    vmax, step = axis["max"], axis["step"]
    two = len(series) == 2
    fills = (DARK, LIGHT)
    n = len(cats)
    out: list[str] = []
    top = 10.0
    if two:
        # legend: a swatch and the series name, side by side along the top
        lx = 20.0
        for fill, ser in zip(fills, series):
            out.append(_rect(lx, top, 12, 12, fill=fill, stroke=INK, stroke_width=1.2))
            out.append(_text(lx + 17, top + 6, ser["name"], size=12, anchor="start"))
            lx += 17 + _text_w(ser["name"], 12) + 18
        top += 26
    numw = _text_w(_num_text(vmax))
    cat_title = s.get("y_label") if horizontal else s.get("x_label")
    val_title = s.get("x_label") if horizontal else s.get("y_label")
    marks = _grid_marks(axis)

    if not horizontal:
        left = 14 + (16 if val_title else 0) + numw + 10
        pw, ph = CANVAS - 14 - left, 220.0
        base = top + ph
        slot = pw / n

        def V(v: float) -> float:
            return base - v / vmax * ph

        for v, minor in marks:
            out.append(_line(left, V(v), left + pw, V(v), **_grid_line(minor)))
        group = slot * 0.7
        bw = group / len(series)
        for ci in range(n):
            for si, ser in enumerate(series):
                v = ser["values"][ci]
                if v > 0:
                    x = left + slot * ci + (slot - group) / 2 + bw * si
                    out.append(_rect(x, V(v), bw, base - V(v), fill=fills[si], stroke=INK, stroke_width=1.3))
        out.append(_line(left, top, left, base, stroke=INK, stroke_width=1.5))
        out.append(_line(left, base, left + pw, base, stroke=INK, stroke_width=1.5))
        ky = _every(ph / round(vmax / step), 13)
        for i in range(round(vmax / step) + 1):
            out.append(_line(left - 4, V(i * step), left, V(i * step), stroke=INK, stroke_width=1.2))
            if i % ky == 0:
                out.append(_text(left - 7, V(i * step), _num_text(i * step), size=_AX_FONT, anchor="end"))
        tilt = max(_text_w(c) for c in cats) > slot - 4
        reserve = 20.0
        for ci, c in enumerate(cats):
            cx = left + slot * ci + slot / 2
            if tilt:
                out.append(_rotated_text(cx + 3, base + 11, c, -40, size=_AX_FONT, anchor="end"))
            else:
                out.append(_text(cx, base + 13, c, size=_AX_FONT))
        if tilt:
            reserve = max(_text_w(c) for c in cats) * 0.65 + 16
        y_end = base + reserve
        if cat_title:
            out.append(_axis_title(left + pw / 2, y_end + 8, cat_title))
            y_end += 20
        if val_title:
            out.append(_axis_title(14, top + ph / 2, val_title, vertical=True))
        return "".join(out), y_end + 6

    # horizontal: categories down the left edge (first at the top), the value axis along the bottom
    left = 14 + (16 if cat_title else 0) + max(_text_w(c) for c in cats) + 10
    left = min(left, 190.0)
    pw = CANVAS - left - 26
    slot = min(44.0, 260.0 / n)
    ph = slot * n
    base = top + ph

    def H(v: float) -> float:
        return left + v / vmax * pw

    for v, minor in marks:
        out.append(_line(H(v), top, H(v), base, **_grid_line(minor)))
    group = slot * 0.7
    bh = group / len(series)
    for ci in range(n):
        for si, ser in enumerate(series):
            v = ser["values"][ci]
            if v > 0:
                y = top + slot * ci + (slot - group) / 2 + bh * si
                out.append(_rect(left, y, H(v) - left, bh, fill=fills[si], stroke=INK, stroke_width=1.3))
    out.append(_line(left, top, left, base, stroke=INK, stroke_width=1.5))
    out.append(_line(left, base, left + pw, base, stroke=INK, stroke_width=1.5))
    for ci, c in enumerate(cats):
        out.append(_text(left - 7, top + slot * ci + slot / 2, c, size=_AX_FONT, anchor="end"))
    nsteps = round(vmax / step)
    kx = _every(pw / nsteps, numw + 6)
    for i in range(nsteps + 1):
        out.append(_line(H(i * step), base, H(i * step), base + 4, stroke=INK, stroke_width=1.2))
        if i % kx == 0:
            out.append(_text(H(i * step), base + 15, _num_text(i * step), size=_AX_FONT))
    y_end = base + 26
    if val_title:
        out.append(_axis_title(left + pw / 2, y_end + 6, val_title))
        y_end += 18
    if cat_title:
        out.append(_axis_title(14, top + ph / 2, cat_title, vertical=True))
    return "".join(out), y_end + 6


# ---------------------------------------------------------------- picture graph


def _symbol_shape(symbol: str, x: float, y: float, w: float, h: float, **kw: Any) -> str:
    cx, cy = x + w / 2, y + h / 2
    if symbol == "square":
        return _rect(x + w * 0.06, y + h * 0.06, w * 0.88, h * 0.88, **kw)
    if symbol == "triangle":
        return _poly([(cx, y + h * 0.06), (x + w * 0.03, y + h * 0.94), (x + w * 0.97, y + h * 0.94)], True, **kw)
    if symbol == "diamond":
        return _poly([(cx, y + h * 0.02), (x + w * 0.98, cy), (cx, y + h * 0.98), (x + w * 0.02, cy)], True, **kw)
    if symbol == "star":
        return _poly(_star(cx, cy, min(w, h) / 2 * 1.04), True, **kw)
    if symbol == "heart":
        d = (
            f"M{_n(cx)} {_n(y + h * 0.92)}"
            f"C{_n(x - w * 0.08)} {_n(y + h * 0.55)} {_n(x + w * 0.02)} {_n(y + h * 0.06)} {_n(x + w * 0.27)} {_n(y + h * 0.1)}"
            f"C{_n(x + w * 0.4)} {_n(y + h * 0.12)} {_n(cx)} {_n(y + h * 0.25)} {_n(cx)} {_n(y + h * 0.32)}"
            f"C{_n(cx)} {_n(y + h * 0.25)} {_n(x + w * 0.6)} {_n(y + h * 0.12)} {_n(x + w * 0.73)} {_n(y + h * 0.1)}"
            f"C{_n(x + w * 0.98)} {_n(y + h * 0.06)} {_n(x + w * 1.08)} {_n(y + h * 0.55)} {_n(cx)} {_n(y + h * 0.92)}Z"
        )
        return _path(d, **kw)
    return _circle(cx, cy, min(w, h) * 0.46, **kw)


def _picture_graph(s: dict, ids: _Ids) -> Body:
    rows, symbol = s["rows"], s["symbol"]
    left = min(14 + max(_text_w(r["label"], 13) for r in rows) + 10, 170.0)
    cell = min(26.0, (CANVAS - left - 14) / 12)
    pitch = cell + 4
    row_h = cell + 12
    top = 8.0
    out = [_line(left, top, left, top + row_h * len(rows), stroke=INK, stroke_width=1.5)]
    halves = []
    for i, r in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        out.append(
            _line(left, top + row_h * (i + 1), CANVAS - 10, top + row_h * (i + 1), stroke=GRID, stroke_width=0.8)
        )
        out.append(_text(left - 8, cy, r["label"], size=13, anchor="end"))
        whole = int(r["symbols"])
        x0 = left + 8
        for k in range(whole):
            out.append(
                _symbol_shape(
                    symbol, x0 + k * pitch, cy - cell / 2, cell, cell, fill=DARK, stroke=INK, stroke_width=1.3
                )
            )
        if r["symbols"] - whole >= 0.5:
            x = x0 + whole * pitch
            y = cy - cell / 2
            keep, shape = ids(f"half{i}"), ids(f"whole{i}")
            halves.append(
                _el("clipPath", _rect(x - 2, y - 2, cell / 2 + 2, cell + 4), id=keep)
                + _el("clipPath", _symbol_shape(symbol, x, y, cell, cell), id=shape)
            )
            out.append(
                _g(
                    _symbol_shape(symbol, x, y, cell, cell, fill=DARK, stroke=INK, stroke_width=1.3),
                    clip_path=f"url(#{keep})",
                )
            )
            # close the cut edge so the half reads as a half symbol rather than an open outline
            out.append(
                _g(
                    _line(x + cell / 2, y - 2, x + cell / 2, y + cell + 2, stroke=INK, stroke_width=1.6),
                    clip_path=f"url(#{shape})",
                )
            )
    y = top + row_h * len(rows) + 20
    out.append(_symbol_shape(symbol, left + 8, y - cell / 2, cell, cell, fill=DARK, stroke=INK, stroke_width=1.3))
    out.append(_text(left + 8 + cell + 8, y, _key_text(s), size=13, anchor="start"))
    return "".join(halves) + "".join(out), y + cell / 2 + 10


# ---------------------------------------------------------------- dot plot


def _tick_axis(
    lo: float, hi: float, step: float, left: float, pw: float, base: float, minor: bool
) -> tuple[list[str], float]:
    """A horizontal number axis from ``lo`` to ``hi``: its line, ticks and numerals. Returns the markup and X(v)."""

    def X(v: float) -> float:
        return left + (v - lo) / (hi - lo) * pw

    n = round((hi - lo) / step)
    out = [_line(left - 8, base, left + pw + 8, base, stroke=INK, stroke_width=1.5)]
    widest = max(_text_w(_num_text(lo + i * step)) for i in range(n + 1))
    k = _every(pw / n, widest + 6)
    for i in range(n + 1):
        v = lo + i * step
        out.append(_line(X(v), base - 5, X(v), base + 5, stroke=INK, stroke_width=1.3))
        if minor and i < n:
            out.append(_line(X(v + step / 2), base - 3, X(v + step / 2), base + 3, stroke=INK, stroke_width=1))
        if i % k == 0:
            out.append(_text(X(v), base + 17, _num_text(v), size=_AX_FONT))
    return out, X


def _dot_plot(s: dict, ids: _Ids) -> Body:
    lo, hi = s["range"]
    step = s["step"]
    left, pw = 34.0, CANVAS - 68.0
    n = round((hi - lo) / step)
    spacing = pw / n
    r = max(2.0, min(9.0, spacing * 0.4))
    pitch = 2 * r + 1.5
    tallest = max(d["count"] for d in s["dots"])
    top = 12.0
    base = top + tallest * pitch + 12
    axis, X = _tick_axis(lo, hi, step, left, pw, base, False)
    out = []
    for d in s["dots"]:
        for k in range(d["count"]):
            out.append(_circle(X(d["at"]), base - 2 - r - k * pitch, r, fill=INK, stroke="none"))
    out.extend(axis)
    y = base + 28
    if s.get("x_label"):
        out.append(_text(left + pw / 2, y + 4, s["x_label"], size=12))
        y += 18
    return "".join(out), y + 8


# ---------------------------------------------------------------- histogram


def _histogram(s: dict, ids: _Ids) -> Body:
    panels = s["panels"]
    two = len(panels) == 2
    numw = max(_text_w(_num_text(p["frequency_axis"]["max"])) for p in panels)
    left = 14 + (16 if s.get("y_label") else 0) + numw + 10
    pw = CANVAS - left - 22
    ph = 118.0 if two else 190.0
    out: list[str] = []
    y = 10.0
    stack_top = y
    for p in panels:
        axis, edges, freqs = p["frequency_axis"], p["edges"], p["frequencies"]
        vmax, step = axis["max"], axis["step"]
        if p.get("label"):
            out.append(_text(left, y + 6, p["label"], size=13, anchor="start", weight="bold"))
            y += 20
        base = y + ph
        classes = len(edges) - 1

        def X(v: float) -> float:
            return left + (v - edges[0]) / (edges[-1] - edges[0]) * pw

        def V(v: float) -> float:
            return base - v / vmax * ph

        for v, minor in _grid_marks(axis):
            out.append(_line(left, V(v), left + pw, V(v), **_grid_line(minor)))
        for k, f in enumerate(freqs):
            if f > 0:
                out.append(
                    _rect(
                        X(edges[k]),
                        V(f),
                        X(edges[k + 1]) - X(edges[k]),
                        base - V(f),
                        fill=MID,
                        stroke=INK,
                        stroke_width=1.3,
                    )
                )
        out.append(_line(left, y, left, base, stroke=INK, stroke_width=1.5))
        out.append(_line(left, base, left + pw, base, stroke=INK, stroke_width=1.5))
        nsteps = round(vmax / step)
        ky = _every(ph / nsteps, 13)
        for i in range(nsteps + 1):
            out.append(_line(left - 4, V(i * step), left, V(i * step), stroke=INK, stroke_width=1.2))
            if i % ky == 0:
                out.append(_text(left - 7, V(i * step), _num_text(i * step), size=_AX_FONT, anchor="end"))
        # a numeral at every class boundary (a learner reads a boundary off it); a smaller face before any is dropped
        edge_size = (
            _AX_FONT if _text_w(_num_text(max(edges, key=lambda e: len(_num_text(e))))) + 4 <= pw / classes else 10
        )
        kx = _every(pw / classes, max(_text_w(_num_text(e), edge_size) for e in edges) + 3)
        for i, e in enumerate(edges):
            out.append(_line(X(e), base, X(e), base + 4, stroke=INK, stroke_width=1.2))
            if i % kx == 0:
                out.append(_text(X(e), base + 15, _num_text(e), size=edge_size))
        last_base = base
        y = base + 28
    if s.get("y_label"):
        out.append(_axis_title(14, (stack_top + last_base) / 2, s["y_label"], vertical=True))
    y = last_base + 24
    if s.get("x_label"):
        out.append(_text(left + pw / 2, y + 6, s["x_label"], size=12))
        y += 16
    return "".join(out), y + 6


# ---------------------------------------------------------------- box plot


def _cross(cx: float, cy: float, r: float) -> str:
    return _line(cx - r, cy - r, cx + r, cy + r, stroke=INK, stroke_width=1.8) + _line(
        cx - r, cy + r, cx + r, cy - r, stroke=INK, stroke_width=1.8
    )


def _box_plot(s: dict, ids: _Ids) -> Body:
    lo, hi = s["range"]
    step = s["step"]
    plots = s["plots"]
    left = max(40.0, 14 + max(_text_w(p["label"], 13) for p in plots) + 10)
    left = min(left, 170.0)
    pw = CANVAS - left - 26
    row_h, top = 56.0, 10.0
    base = top + row_h * len(plots) + 4
    axis, X = _tick_axis(lo, hi, step, left, pw, base, bool(s.get("minor")))
    n = round((hi - lo) / step)
    out = [_line(X(lo + i * step), top, X(lo + i * step), base, stroke=GRID, stroke_width=0.7) for i in range(n + 1)]
    for i, p in enumerate(plots):
        cy = top + row_h * i + row_h / 2 + 2
        half = 13.0
        if p["label"]:
            out.append(_text(left - 10, cy, p["label"], size=13, anchor="end", weight="bold"))
        out.append(_line(X(p["min"]), cy, X(p["q1"]), cy, stroke=INK, stroke_width=1.8))
        out.append(_line(X(p["q3"]), cy, X(p["max"]), cy, stroke=INK, stroke_width=1.8))
        for end in (p["min"], p["max"]):
            out.append(_line(X(end), cy - 7, X(end), cy + 7, stroke=INK, stroke_width=1.8))
        out.append(
            _rect(X(p["q1"]), cy - half, X(p["q3"]) - X(p["q1"]), 2 * half, fill=LIGHT, stroke=INK, stroke_width=1.8)
        )
        out.append(_line(X(p["median"]), cy - half, X(p["median"]), cy + half, stroke=INK, stroke_width=3))
        for o in p["outliers"]:
            out.append(_cross(X(o), cy, 4.5))
    out.extend(axis)
    y = base + 34
    if s.get("x_label"):
        out.append(_text(left + pw / 2, y, s["x_label"], size=12))
        y += 14
    return "".join(out), y + 6


# ---------------------------------------------------------------- pie chart

_PIE_FILLS = ("#ffffff", "#b4b4b4", "#e2e2e2", "#808080", "#f4f4f4", "#c8c8c8")


def _pie_layout(sectors: list[dict], parts: int) -> tuple[float, float, float, list[float]]:
    """Pick a font size and radius where every label fits beside the circle; return (size, r, cx, mid angles)."""
    mids, run = [], 0
    for sec in sectors:
        mids.append((run + sec["parts"] / 2) / parts * 360)
        run += sec["parts"]
    for size, r in ((12, 85), (12, 70), (11, 62), (10, 55)):
        w = [len(_clean_text(sec["label"])) * _CHAR_W * size / 12 for sec in sectors]
        need_l = max([wi for wi, m in zip(w, mids) if math.sin(math.radians(m)) < -0.2] + [0])
        need_r = max([wi for wi, m in zip(w, mids) if math.sin(math.radians(m)) > 0.2] + [0])
        mid_half = max([wi / 2 for wi, m in zip(w, mids) if abs(math.sin(math.radians(m))) <= 0.2] + [0])
        lo = max(10 + need_l + 14 + r, 10 + mid_half)
        hi = min(CANVAS - 10 - need_r - 14 - r, CANVAS - 10 - mid_half)
        if lo <= hi or size == 10:
            return size, r, min(max(CANVAS / 2, lo), max(lo, hi)), mids
    raise AssertionError("unreachable")


def _pie_chart(s: dict, ids: _Ids) -> Body:
    parts, sectors = s["parts"], s["sectors"]
    size, r, cx, mids = _pie_layout(sectors, parts)
    cy = r + 30
    out = []
    boundaries = set()
    run = 0
    for si, sec in enumerate(sectors):
        a0, a1 = run / parts * 360, (run + sec["parts"]) / parts * 360
        boundaries.update((run, run + sec["parts"]))
        x0, y0 = _clock_pt(cx, cy, r, a0)
        x1, y1 = _clock_pt(cx, cy, r, a1)
        large = 1 if a1 - a0 > 180 else 0
        d = f"M{_n(cx)} {_n(cy)}L{_n(x0)} {_n(y0)}A{_n(r)} {_n(r)} 0 {large} 1 {_n(x1)} {_n(y1)}Z"
        out.append(_path(d, fill=_PIE_FILLS[si % len(_PIE_FILLS)], stroke=INK, stroke_width=2, stroke_linejoin="round"))
        run += sec["parts"]
    for k in range(parts):  # a thin radius at every wedge boundary that is not already a sector edge
        if k not in boundaries:
            x, y = _clock_pt(cx, cy, r, k / parts * 360)
            out.append(_line(cx, cy, x, y, stroke=DARK, stroke_width=0.9))
    for sec, mid in zip(sectors, mids):
        x, y = _clock_pt(cx, cy, r + 10, mid)
        sn, cs = math.sin(math.radians(mid)), math.cos(math.radians(mid))
        if sn > 0.2:
            out.append(_text(x + 3, y, sec["label"], size=size, anchor="start"))
        elif sn < -0.2:
            out.append(_text(x - 3, y, sec["label"], size=size, anchor="end"))
        else:
            out.append(_text(x, y - 6 if cs > 0 else y + 6, sec["label"], size=size))
    return "".join(out), cy + r + 26


# ---------------------------------------------------------------- shape scene

_SCENE_SCALE = {"small": 0.55, "medium": 0.75, "large": 1.0}


def _scene_paint(style: str, primary: bool = True) -> dict:
    """Plain is hollow, shaded is filled, dotted has a dotted outline: told apart without colour."""
    if style == "shaded":
        return dict(fill=MID if primary else DARK, stroke_width=2, stroke_linejoin="round")
    if style == "dotted":
        return dict(fill=PAPER, stroke_width=2.6, stroke_dasharray="0.1 5.5", stroke_linecap="round")
    return dict(fill=PAPER if primary else LIGHT, stroke_width=2, stroke_linejoin="round")


def _regular(n: int, r: float, start_deg: float = 0) -> list[tuple[float, float]]:
    """A regular polygon's corners about the origin, the first ``start_deg`` clockwise from straight up."""
    return [_clock_pt(0, 0, r, start_deg + k * 360 / n) for k in range(n)]


def _shape_body(shape: str, r: float, style: str) -> str:
    """One shape or solid centred on the origin, ``r`` about its half-size."""
    p, q = _scene_paint(style), _scene_paint(style, primary=False)
    if shape == "circle":
        return _circle(0, 0, r, **p)
    if shape == "oval":
        return _el("ellipse", cx=0, cy=0, rx=r, ry=0.62 * r, **p)
    if shape == "triangle":
        return _poly([(0, -r), (0.95 * r, 0.75 * r), (-0.95 * r, 0.75 * r)], True, **p)
    if shape == "square":
        return _rect(-0.72 * r, -0.72 * r, 1.44 * r, 1.44 * r, **p)
    if shape == "rectangle":
        return _rect(-0.95 * r, -0.55 * r, 1.9 * r, 1.1 * r, **p)
    if shape == "hexagon":
        return _poly(_regular(6, r, 90), True, **p)
    if shape == "pentagon":
        return _poly(_regular(5, r), True, **p)
    if shape == "rhombus":
        return _poly([(0, -r), (0.62 * r, 0), (0, r), (-0.62 * r, 0)], True, **p)
    if shape == "kite":
        return _poly([(0, -0.9 * r), (0.62 * r, -0.2 * r), (0, r), (-0.62 * r, -0.2 * r)], True, **p)
    if shape == "semicircle":
        return _path(f"M{_n(-r)} {_n(0.45 * r)}A{_n(r)} {_n(r)} 0 0 1 {_n(r)} {_n(0.45 * r)}Z", **p)
    if shape in ("cube", "rectangular prism"):
        w, h, d = (1.2, 1.2, 0.4) if shape == "cube" else (1.5, 0.8, 0.4)
        x0, y0 = -0.7 * r if shape == "cube" else -0.85 * r, (-0.3 if shape == "cube" else -0.15) * r
        x1, y1, dd = x0 + w * r, y0 + h * r, d * r
        return (
            _poly([(x0, y0), (x1, y0), (x1 + dd, y0 - dd), (x0 + dd, y0 - dd)], True, **q)
            + _poly([(x1, y0), (x1 + dd, y0 - dd), (x1 + dd, y1 - dd), (x1, y1)], True, **q)
            + _rect(x0, y0, w * r, h * r, **p)
        )
    if shape == "sphere":
        return _circle(0, 0, r, **p) + _path(
            f"M{_n(-r)} 0A{_n(r)} {_n(0.3 * r)} 0 0 0 {_n(r)} 0", **{**q, "fill": "none", "stroke_dasharray": None}
        )
    if shape == "cylinder":
        rx, ry, top, bottom = 0.7 * r, 0.25 * r, -0.65 * r, 0.65 * r
        return _path(
            f"M{_n(-rx)} {_n(top)}V{_n(bottom)}A{_n(rx)} {_n(ry)} 0 0 0 {_n(rx)} {_n(bottom)}V{_n(top)}Z",
            **p,
        ) + _el("ellipse", cx=0, cy=top, rx=rx, ry=ry, **q)
    if shape == "cone":
        rx, ry, base, apex = 0.8 * r, 0.25 * r, 0.65 * r, -0.95 * r
        return _path(
            f"M{_n(-rx)} {_n(base)}L0 {_n(apex)}L{_n(rx)} {_n(base)}A{_n(rx)} {_n(ry)} 0 0 1 {_n(-rx)} {_n(base)}Z", **p
        )
    # square pyramid: a front face, a side face and the base edge
    apex, fl, fr, br = (-0.05 * r, -0.95 * r), (-0.85 * r, 0.6 * r), (0.4 * r, 0.6 * r), (0.9 * r, 0.3 * r)
    return _poly([apex, fr, br], True, **q) + _poly([apex, fl, fr], True, **p)


def _scene_object(name: str, ids: _Ids) -> Body:
    """An everyday object drawn from shapes, in the 400-wide space. Nothing in it is a word."""
    cx = CANVAS / 2
    plain = "plain"
    if name == "house":
        walls = _g(_shape_body("square", 76, plain), transform=f"translate({_n(cx)} 160)")
        roof = _poly([(cx - 100, 108), (cx + 100, 108), (cx, 28)], True, **_scene_paint(plain))
        return walls + roof, 240
    if name == "clock":
        out = [_g(_shape_body("circle", 88, plain), transform=f"translate({_n(cx)} 108)")]
        for k in range(12):
            a, b = _clock_pt(cx, 108, 80, k * 30), _clock_pt(cx, 108, 70, k * 30)
            out.append(_line(a[0], a[1], b[0], b[1], stroke_width=2))
        for deg, length in ((60, 44), (180, 62)):
            x, y = _clock_pt(cx, 108, length, deg)
            out.append(_line(cx, 108, x, y, stroke_width=4, stroke_linecap="round"))
        return "".join(out), 216
    if name == "ball":
        out = [_g(_shape_body("sphere", 88, plain), transform=f"translate({_n(cx)} 108)")]
        out.append(_path(f"M{_n(cx - 62)} {_n(46)}Q{_n(cx - 20)} 108 {_n(cx - 62)} 170", stroke_width=2))
        out.append(_path(f"M{_n(cx + 62)} {_n(46)}Q{_n(cx + 20)} 108 {_n(cx + 62)} 170", stroke_width=2))
        return "".join(out), 216
    if name == "box":
        return _g(_shape_body("rectangular prism", 105, plain), transform=f"translate({_n(cx)} 118)"), 236
    if name == "can":
        return _g(_shape_body("cylinder", 110, plain), transform=f"translate({_n(cx)} 118)"), 236
    # ice cream: a cone point-down, with a scoop sitting on top of it
    cone = _poly([(cx - 52, 120), (cx + 52, 120), (cx, 218)], True, **_scene_paint(plain))
    waffle = "".join(_line(cx + dx, 120, cx + dx * 0.0, 218, stroke_width=1) for dx in (-26, 26)) + _line(
        cx - 30, 150, cx + 30, 150, stroke_width=1
    )
    scoop = _path(f"M{_n(cx - 58)} 120A58 58 0 0 1 {_n(cx + 58)} 120Z", **_scene_paint("shaded"))
    return cone + waffle + scoop, 238


def _shape_scene(s: dict, ids: _Ids) -> Body:
    if s["object"] is not None:
        return _scene_object(s["object"]["name"], ids)
    items = s["items"]
    n = len(items)
    cols = s["cols"] if s["layout"] == "grid" else (n if n <= 6 else math.ceil(n / 2))
    cell = min(84.0, 380.0 / cols)
    label_h = 16.0 if any(it["label"] for it in items) else 0.0
    row_h = cell + label_h
    x0 = (CANVAS - cols * cell) / 2
    out = []
    for i, it in enumerate(items):
        row, col = divmod(i, cols)
        in_row = min(cols, n - row * cols)
        left = (CANVAS - in_row * cell) / 2 if s["layout"] == "row" else x0
        wobble = ((i * 37) % 5 - 2) * cell * 0.04 if s["layout"] == "row" else 0.0
        cx, cy = left + col * cell + cell / 2, 12 + row * row_h + cell / 2 + wobble
        r = cell * 0.40 * _SCENE_SCALE[it["size"]]
        body = _shape_body(it["shape"], r, it["style"])
        rot = f" rotate({it['turn']})" if it["turn"] else ""
        out.append(_g(body, transform=f"translate({_n(cx)} {_n(cy)}){rot}"))
        if it["label"]:
            out.append(_text(cx, 12 + row * row_h + cell + label_h / 2, it["label"], size=13, weight="bold"))
    rows = math.ceil(n / cols)
    return "".join(out), 12 + rows * row_h + 8

# ---------------------------------------------------------------- pattern strip

_STRIP_LEFT = 12.0


def _pattern_shape(cell: dict, cx: float, cy: float, r: float, stripe: str) -> str:
    fill = {"solid": INK, "open": PAPER, "striped": f"url(#{stripe})"}[cell["fill"]]
    return _symbol_shape(cell["shape"], cx - r, cy - r, 2 * r, 2 * r, fill=fill, stroke=INK, stroke_width=2)


def _action_icon(action: str, cx: float, cy: float) -> str:
    ink = {"stroke": INK, "stroke_width": 1.8}
    if action == "clap":
        left = _el(
            "ellipse",
            cx=cx - 5,
            cy=cy,
            rx=4.5,
            ry=10,
            transform=f"rotate(-18 {_n(cx - 5)} {_n(cy)})",
            fill=LIGHT,
            **ink,
        )
        right = _el(
            "ellipse", cx=cx + 5, cy=cy, rx=4.5, ry=10, transform=f"rotate(18 {_n(cx + 5)} {_n(cy)})", fill=LIGHT, **ink
        )
        burst = "".join(
            _line(cx + dx * 4, cy - 15 - dy, cx + dx * 9, cy - 20 - dy, **ink) for dx, dy in ((-1, 0), (0, 2), (1, 0))
        )
        return left + right + burst
    if action == "stamp":
        return (
            _rect(cx - 5, cy - 14, 9, 17, fill=LIGHT, **ink)
            + _rect(cx - 5, cy + 1, 18, 8, rx=3, fill=LIGHT, **ink)
            + _line(cx - 14, cy + 13, cx + 16, cy + 13, **ink)
        )
    return (
        _circle(cx, cy - 13, 3.5, fill=INK, stroke="none")
        + _line(cx, cy - 9, cx, cy + 3, **ink)
        + _line(cx, cy - 6, cx - 8, cy - 14, **ink)
        + _line(cx, cy - 6, cx + 8, cy - 14, **ink)
        + _line(cx, cy + 3, cx - 7, cy + 12, **ink)
        + _line(cx, cy + 3, cx + 7, cy + 12, **ink)
    )


def _stack_counters(count: int, cx: float, top: float, height: float, cw: float) -> str:
    step = min(8.0, height / count)
    r = min(step * 0.42, cw * 0.3)
    return "".join(_circle(cx, top + height - step * (i + 0.5), r, fill=INK, stroke="none") for i in range(count))


def _pattern_strip(s: dict, ids: _Ids) -> Body:
    rows = s["rows"]
    stripe = ids("stripe")
    labelled = any(r.get("label") for r in rows)
    left = _STRIP_LEFT + (34.0 if labelled else 0.0)
    widest = max(len(r["cells"]) for r in rows)
    cw = min(48.0, (CANVAS - left - 12) / widest)
    cells = [c for r in rows for c in r["cells"]]
    has_action = any(c["type"] == "action" for c in cells)
    has_stack = any(c["type"] == "stack" for c in cells)
    box_h = 82.0 if has_stack else (64.0 if has_action else 48.0)
    out = []
    if any(c["type"] == "shape" and c["fill"] == "striped" for c in cells):
        out.append(
            _el(
                "defs",
                _el(
                    "pattern",
                    _rect(0, 0, 5, 5, fill=PAPER, stroke="none") + _line(0, 0, 0, 5, stroke=INK, stroke_width=2),
                    id=stripe,
                    patternUnits="userSpaceOnUse",
                    width=5,
                    height=5,
                    patternTransform="rotate(45)",
                ),
            )
        )
    y = 12.0
    for r in rows:
        if r.get("label"):
            out.append(_text(left - 8, y + box_h / 2, r["label"], size=18, anchor="end", weight="bold"))
        for i, c in enumerate(r["cells"]):
            x = left + i * cw
            cx, cy = x + cw / 2, y + box_h / 2
            t = c["type"]
            if t == "blank":
                out.append(
                    _rect(
                        x + 2,
                        y + 2,
                        cw - 4,
                        box_h - 4,
                        rx=4,
                        fill=PAPER,
                        stroke=INK,
                        stroke_width=1.5,
                        stroke_dasharray="5 4",
                    )
                )
                continue
            out.append(_rect(x + 2, y + 2, cw - 4, box_h - 4, rx=4, fill=PAPER, stroke=GRID, stroke_width=1))
            if t == "mark":
                out.append(_text(cx, cy, "?", size=26, weight="bold"))
            elif t == "text":
                out.append(_text(cx, cy, c["text"], size=min(28.0, cw * 0.7), weight="bold"))
            elif t == "shape":
                out.append(_pattern_shape(c, cx, cy, min(cw * 0.34, 16.0), stripe))
            elif t == "action":
                out.append(_action_icon(c["action"], cx, cy - 8))
                out.append(_text(cx, y + box_h - 11, c["action"], size=min(12.0, cw * 0.3)))
            else:
                out.append(_stack_counters(c["count"], cx, y + 8, box_h - 16, cw))
        if r.get("bracket"):
            a, b = r["bracket"]
            x0, x1, by = left + a * cw + 3, left + (b + 1) * cw - 3, y + box_h + 7
            out.append(_poly([(x0, by - 5), (x0, by), (x1, by), (x1, by - 5)], stroke=INK, stroke_width=2))
            y += 10
        y += box_h
        if s["numbered"]:
            for i in range(len(r["cells"])):
                out.append(_text(left + i * cw + cw / 2, y + 9, i + 1, size=11))
            y += 18
        y += 14

# ---------------------------------------------------------------- tally


def _icon(name: str, cx: float, cy: float) -> str:
    """One small picture, about 28 units across, centred on (cx, cy). Each has its own outline, so no two are told apart by shading alone."""
    kw: dict[str, Any] = {"fill": PAPER, "stroke": INK, "stroke_width": 1.6}
    dot: dict[str, Any] = {"fill": INK, "stroke": "none"}
    out: list[str] = []
    if name == "apple":
        out.append(_circle(cx, cy + 2, 10.5, **kw))
        out.append(_line(cx, cy - 8, cx + 2, cy - 14, stroke=INK, stroke_width=1.8))
        out.append(
            _path(
                f"M{_n(cx + 2)} {_n(cy - 12)}Q{_n(cx + 10)} {_n(cy - 16)} {_n(cx + 9)} {_n(cy - 9)}Z",
                fill=DARK,
                stroke=INK,
                stroke_width=1,
            )
        )
    elif name == "ball":
        out.append(_circle(cx, cy, 11.5, **kw))
        out.append(_path(f"M{_n(cx - 11)} {_n(cy)}H{_n(cx + 11)}", stroke=INK, stroke_width=1.4))
        out.append(
            _path(
                f"M{_n(cx)} {_n(cy - 11.5)}Q{_n(cx + 8)} {_n(cy)} {_n(cx)} {_n(cy + 11.5)}",
                stroke=INK,
                stroke_width=1.4,
            )
        )
    elif name == "cat":
        out.append(_poly([(cx - 10, cy - 1), (cx - 9, cy - 13), (cx - 2, cy - 7)], True, **kw))
        out.append(_poly([(cx + 10, cy - 1), (cx + 9, cy - 13), (cx + 2, cy - 7)], True, **kw))
        out.append(_circle(cx, cy + 2, 10.5, **kw))
        out.append(_circle(cx - 4, cy, 1.4, **dot))
        out.append(_circle(cx + 4, cy, 1.4, **dot))
        out.append(_poly([(cx - 1.5, cy + 3), (cx + 1.5, cy + 3), (cx, cy + 5)], True, **dot))
        for sgn in (-1, 1):
            out.append(_line(cx + sgn * 4, cy + 5, cx + sgn * 13, cy + 3, stroke=INK, stroke_width=1))
            out.append(_line(cx + sgn * 4, cy + 7, cx + sgn * 13, cy + 8, stroke=INK, stroke_width=1))
    elif name == "dog":
        out.append(_el("ellipse", cx=cx - 11, cy=cy + 1, rx=4, ry=8, fill=DARK, stroke=INK, stroke_width=1.4))
        out.append(_el("ellipse", cx=cx + 11, cy=cy + 1, rx=4, ry=8, fill=DARK, stroke=INK, stroke_width=1.4))
        out.append(_el("ellipse", cx=cx, cy=cy, rx=9, ry=10.5, **kw))
        out.append(_circle(cx - 3.5, cy - 2, 1.4, **dot))
        out.append(_circle(cx + 3.5, cy - 2, 1.4, **dot))
        out.append(_el("ellipse", cx=cx, cy=cy + 4, rx=3, ry=2.2, **dot))
    elif name == "fish":
        out.append(_poly([(cx + 6, cy), (cx + 14, cy - 7), (cx + 14, cy + 7)], True, **kw))
        out.append(_el("ellipse", cx=cx - 2, cy=cy, rx=11, ry=7.5, **kw))
        out.append(_circle(cx - 7, cy - 2, 1.5, **dot))
    elif name == "flower":
        out.append(_line(cx, cy + 3, cx, cy + 14, stroke=INK, stroke_width=1.8))
        for k in range(5):
            px, py = _clock_pt(cx, cy - 3, 7.5, k * 72)
            out.append(_circle(px, py, 4.6, **kw))
        out.append(_circle(cx, cy - 3, 3.4, fill=INK, stroke=INK, stroke_width=1))
    elif name == "star":
        out.append(_poly(_star(cx, cy + 1, 14), True, stroke_linejoin="round", **kw))
    else:  # car
        out.append(
            _poly(
                [
                    (cx - 11, cy + 4),
                    (cx - 9, cy - 3),
                    (cx - 4, cy - 8),
                    (cx + 5, cy - 8),
                    (cx + 9, cy - 3),
                    (cx + 13, cy - 2),
                    (cx + 13, cy + 4),
                ],
                True,
                stroke_linejoin="round",
                **kw,
            )
        )
        out.append(_circle(cx - 6, cy + 6, 3.6, fill=INK, stroke=INK, stroke_width=1))
        out.append(_circle(cx + 7, cy + 6, 3.6, fill=INK, stroke=INK, stroke_width=1))
    return "".join(out)


def _tally_marks(x: float, y: float, count: int, gap: float, height: float) -> str:
    """``count`` tally marks from (x, y), the top left; each full gate of five has a diagonal across four upright strokes."""
    out = []
    gate_w = 3 * gap
    pitch = gate_w + 0.9 * gap + 8
    for g in range(count // 5 + (1 if count % 5 else 0)):
        gx = x + g * pitch
        strokes = min(4, count - g * 5)
        for k in range(strokes):
            out.append(
                _line(gx + k * gap, y, gx + k * gap, y + height, stroke=INK, stroke_width=2.2, stroke_linecap="round")
            )
        if count - g * 5 >= 5:
            out.append(
                _line(
                    gx - 4,
                    y + height * 0.82,
                    gx + gate_w + 4,
                    y + height * 0.18,
                    stroke=INK,
                    stroke_width=2.2,
                    stroke_linecap="round",
                )
            )
    return "".join(out)


def _tally(s: dict, ids: _Ids) -> Body:
    rows, items = s["rows"], s["items"]
    out: list[str] = []
    y = 4.0
    if items:
        w, h, x0, y0 = 330.0, 190.0, 35.0, 28.0
        for it in items:
            cx, cy = x0 + it["at"][0] * w, y0 + it["at"][1] * h
            out.append(_icon(it["label"], cx, cy))
            out.append(_text(cx, cy + 22, it["label"], size=11))
        y = y0 + h + 36
        if rows:
            out.append(_line(16, y - 12, CANVAS - 16, y - 12, stroke=GRID, stroke_width=1))
    if rows:
        pics = s.get("pictures")
        left = min(14 + max(_text_w(r["label"], 13) for r in rows) + 10, 120.0)
        gap = 6.5 if pics else 8.0
        pitch = 3 * gap + 0.9 * gap + 8
        widest = max(r["count"] for r in rows)
        tally_w = max(1, -(-widest // 5)) * pitch
        row_h, mark_h = 34.0, 22.0
        x_tally = left + 8
        x_count = x_tally + max(tally_w, 50.0) + 8
        x_pics = (x_count + 40) if s["show_counts"] else x_count
        head = y + 6
        out.append(_text(x_tally, head, "Tally", size=12, anchor="start", weight="bold"))
        if s["show_counts"]:
            out.append(_text(x_count + 14, head, "Count", size=12, weight="bold"))
        if pics:
            out.append(_text(x_pics, head, "Pictures", size=12, anchor="start", weight="bold"))
        top = head + 14
        out.append(_line(14, top, CANVAS - 14, top, stroke=INK, stroke_width=1.3))
        out.append(_line(left, top, left, top + row_h * len(rows), stroke=INK, stroke_width=1.3))
        cell = 15.0
        for i, r in enumerate(rows):
            cy = top + row_h * i + row_h / 2
            out.append(
                _line(14, top + row_h * (i + 1), CANVAS - 14, top + row_h * (i + 1), stroke=GRID, stroke_width=0.8)
            )
            out.append(_text(left - 8, cy, r["label"], size=13, anchor="end"))
            out.append(_tally_marks(x_tally, cy - mark_h / 2, r["count"], gap, mark_h))
            if s["show_counts"]:
                out.append(_text(x_count + 14, cy, r["count"], size=14))
            if pics:
                for k in range(pics["counts"][i]):
                    out.append(
                        _symbol_shape(
                            pics["symbol"],
                            x_pics + k * (cell + 2),
                            cy - cell / 2,
                            cell,
                            cell,
                            fill=DARK,
                            stroke=INK,
                            stroke_width=1.2,
                        )
                    )
        y = top + row_h * len(rows) + 6
        if pics:
            y += 16
            out.append(
                _symbol_shape(pics["symbol"], x_pics, y - cell / 2, cell, cell, fill=DARK, stroke=INK, stroke_width=1.2)
            )
            out.append(_text(x_pics + cell + 8, y, picture_key_text(s), size=13, anchor="start"))
            y += cell / 2
        y += 8
    return "".join(out), y

# ---------------------------------------------------------------- calendar

_CAL_LEFT, _CAL_CELL_W, _CAL_CELL_H = 25.0, 50.0, 40.0


def _star_poly(cx: float, cy: float, r: float) -> str:
    pts = []
    for i in range(10):
        rad = r if i % 2 == 0 else r * 0.45
        pts.append(_pt(cx, cy, rad, 90 + 36 * i))
    return _poly(pts, closed=True, fill=INK, stroke=INK, stroke_width=0.5)


def _cal_label(x: float, y: float, label: str) -> str:
    """A short event name squeezed to fit a square (the text is escaped here)."""
    natural = 4.6 * len(label)
    return _el(
        "text",
        escape(_clean_text(label)),
        x=x,
        y=y,
        dy=".35em",
        font_size=8.5,
        text_anchor="middle",
        font_weight="bold",
        fill=INK,
        stroke="none",
        textLength=_n(min(natural, _CAL_CELL_W - 6)),
        lengthAdjust="spacingAndGlyphs",
    )


def _calendar(s: dict, ids: _Ids) -> Body:
    from mathgen.figure_kinds import calendar as cal

    title = cal.month_name(s) + (f" {s['year']}" if s["show_year"] else "")
    out = [_text(CANVAS / 2, 16, title, size=17, weight="bold")]
    for c, name in enumerate(cal.week_order(s["week_start"])):
        out.append(_text(_CAL_LEFT + (c + 0.5) * _CAL_CELL_W, 40, cal.heading(name), size=11, weight="bold"))
    top = 52.0
    marks = {m["day"]: m for m in s["marks"]}
    for day in range(1, s["days"] + 1):
        r, c = cal.row_of(s, day), cal.column_of(s, day)
        x, y = _CAL_LEFT + c * _CAL_CELL_W, top + r * _CAL_CELL_H
        m = marks.get(day)
        shade = m is not None and m["style"] == "shade"
        out.append(_rect(x, y, _CAL_CELL_W, _CAL_CELL_H, fill=MID if shade else PAPER, stroke=INK, stroke_width=1))
        cx = x + _CAL_CELL_W / 2
        num_y = y + (15 if m and m.get("label") else 20)
        out.append(_text(cx, num_y, day, size=13))
        if m is None:
            continue
        if m["style"] == "ring":
            out.append(_circle(cx, num_y, 11, stroke=INK, stroke_width=2.5))
        elif m["style"] == "star":
            out.append(_star_poly(x + _CAL_CELL_W - 9, y + 9, 6.5))
        if m.get("label"):
            out.append(_cal_label(cx, y + _CAL_CELL_H - 9, m["label"]))
    return "".join(out), top + cal.rows(s) * _CAL_CELL_H + 10


# ---------------------------------------------------------------- day strip


def _icon(name: str, cx: float, cy: float) -> str:
    """A simple monochrome icon about 48 units square, centred on (cx, cy)."""

    def p(x: float, y: float) -> tuple[float, float]:
        return cx + x, cy + y

    def seg(x1: float, y1: float, x2: float, y2: float, w: float = 2.5) -> str:
        return _line(cx + x1, cy + y1, cx + x2, cy + y2, stroke=INK, stroke_width=w, stroke_linecap="round")

    def box(x: float, y: float, w: float, h: float, fill: str = LIGHT, rx: float = 0) -> str:
        return _rect(cx + x, cy + y, w, h, rx=rx, fill=fill, stroke=INK, stroke_width=2)

    def path(cmds: list[tuple], fill: str = "none", w: float = 2) -> str:
        parts = []
        for cmd, *args in cmds:
            if cmd == "A":
                rx, ry, rot, large, sweep, x, y = args
                parts.append(f"A {rx} {ry} {rot} {large} {sweep} {_n(cx + x)} {_n(cy + y)}")
            elif cmd == "Z":
                parts.append("Z")
            else:
                parts.append(cmd + " " + " ".join(f"{_n(cx + a)} {_n(cy + b)}" for a, b in zip(args[::2], args[1::2])))
        return _path(" ".join(parts), fill=fill, stroke=INK, stroke_width=w, stroke_linecap="round")

    if name == "sunrise":
        rays = "".join(
            _line(
                *_pt(cx, cy + 10, 15, a), *_pt(cx, cy + 10, 22, a), stroke=INK, stroke_width=2.5, stroke_linecap="round"
            )
            for a in (20, 55, 90, 125, 160)
        )
        return path([("M", -12, 10), ("A", 12, 12, 0, 0, 1, 12, 10), ("Z",)], fill=LIGHT) + seg(-22, 10, 22, 10) + rays
    if name == "moon":
        return (
            _circle(cx - 3, cy, 17, fill=INK, stroke=INK, stroke_width=1)
            + _circle(cx + 6, cy - 5, 14, fill=PAPER, stroke="none")
            + _star_poly(cx + 17, cy - 14, 5)
            + _star_poly(cx + 20, cy + 6, 3.5)
        )
    if name == "wake_up":
        return (
            _circle(cx - 11, cy - 14, 5, fill=INK, stroke=INK)
            + _circle(cx + 11, cy - 14, 5, fill=INK, stroke=INK)
            + seg(-9, 15, -13, 21)
            + seg(9, 15, 13, 21)
            + _circle(cx, cy + 2, 14, fill=LIGHT, stroke=INK, stroke_width=2.5)
            + seg(0, 2, 0, -8, 2)
            + seg(0, 2, 7, 2, 2)
        )
    if name == "breakfast":
        steam = "".join(path([("M", x, -4), ("Q", x + 4, -9, x, -14)], w=1.8) for x in (-9, 0, 9))
        return (
            steam
            + path([("M", -18, 2), ("A", 18, 18, 0, 0, 0, 18, 2), ("Z",)], fill=LIGHT, w=2.5)
            + seg(-12, 19, 12, 19)
        )
    if name == "school":
        return (
            box(-18, -6, 36, 24)
            + _poly([p(-22, -6), p(0, -22), p(22, -6)], closed=True, fill=MID, stroke=INK, stroke_width=2)
            + _rect(cx - 5, cy + 6, 10, 12, fill=INK, stroke=INK)
            + _rect(cx - 15, cy - 1, 6, 6, fill=PAPER, stroke=INK, stroke_width=1.5)
            + _rect(cx + 9, cy - 1, 6, 6, fill=PAPER, stroke=INK, stroke_width=1.5)
        )
    if name == "lunch":
        return (
            path([("M", -8, -6), ("L", -8, -13), ("L", 8, -13), ("L", 8, -6)], w=2.5)
            + box(-19, -6, 38, 25, rx=3)
            + seg(-19, 5, 19, 5, 2)
            + _circle(cx, cy + 5, 3, fill=INK, stroke=INK)
        )
    if name == "play":
        return (
            _circle(cx, cy, 18, fill=LIGHT, stroke=INK, stroke_width=2.5)
            + seg(-18, 0, 18, 0, 1.8)
            + seg(0, -18, 0, 18, 1.8)
            + path([("M", -12, -13), ("Q", -4, 0, -12, 13)], w=1.8)
            + path([("M", 12, -13), ("Q", 4, 0, 12, 13)], w=1.8)
        )
    if name == "dinner":
        return (
            seg(-21, -14, -21, -5, 2)
            + seg(-17, -14, -17, -5, 2)
            + seg(-21, -5, -17, -5, 2)
            + seg(-19, -5, -19, 18, 2.5)
            + _circle(cx - 1, cy + 2, 12, fill=PAPER, stroke=INK, stroke_width=2.5)
            + _circle(cx - 1, cy + 2, 6.5, fill=LIGHT, stroke=INK, stroke_width=1.5)
            + _poly([p(17, -14), p(22, -8), p(22, 3), p(17, 3)], closed=True, fill=MID, stroke=INK, stroke_width=1.8)
            + seg(17, 3, 17, 18, 2.5)
        )
    if name == "bath":
        return (
            _circle(cx - 6, cy - 9, 5, fill=PAPER, stroke=INK, stroke_width=1.8)
            + _circle(cx + 5, cy - 15, 4, fill=PAPER, stroke=INK, stroke_width=1.8)
            + _circle(cx + 11, cy - 7, 3, fill=PAPER, stroke=INK, stroke_width=1.8)
            + path(
                [
                    ("M", -22, 0),
                    ("L", 22, 0),
                    ("L", 22, 6),
                    ("Q", 22, 17, 11, 17),
                    ("L", -11, 17),
                    ("Q", -22, 17, -22, 6),
                    ("Z",),
                ],
                fill=LIGHT,
                w=2.5,
            )
            + seg(-14, 17, -16, 22)
            + seg(14, 17, 16, 22)
        )
    if name == "bed":
        return (
            _rect(cx - 24, cy - 12, 5, 32, fill=INK, stroke=INK)
            + box(-19, 0, 42, 10)
            + box(-16, -6, 13, 7, fill=PAPER, rx=3)
            + _rect(cx - 1, cy - 5, 24, 6, fill=MID, stroke=INK, stroke_width=1.8)
            + seg(21, 10, 21, 20)
            + seg(-19, 20, 21, 20, 1.5)
        )
    raise ValueError(f"no icon named {name!r}")


_STRIP_CARD_W, _STRIP_GAP, _STRIP_CARD_H = 58.0, 6.0, 82.0


def _day_strip(s: dict, ids: _Ids) -> Body:
    cards = s["cards"]
    total = len(cards) * _STRIP_CARD_W + (len(cards) - 1) * _STRIP_GAP
    left = (CANVAS - total) / 2
    top = 40.0 if s["numbered"] else 10.0
    out = []
    for i, c in enumerate(cards):
        x = left + i * (_STRIP_CARD_W + _STRIP_GAP)
        cx = x + _STRIP_CARD_W / 2
        if s["numbered"]:
            out.append(_circle(cx, 20, 12, fill=PAPER, stroke=INK, stroke_width=2))
            out.append(_text(cx, 20, i + 1, size=14, weight="bold"))
        out.append(_rect(x, top, _STRIP_CARD_W, _STRIP_CARD_H, rx=7, fill=PAPER, stroke=INK, stroke_width=2))
        out.append(_icon(c["icon"], cx, top + 34))
        out.append(_text(cx, top + _STRIP_CARD_H - 11, c["label"], size=10.5, weight="bold"))
    return "".join(out), top + _STRIP_CARD_H + 10

# ---------------------------------------------------------------- compare_picture


def _cp_object(style: str, x0: float, y: float, w: float, h: float) -> str:
    """One object for a length comparison, ``w`` wide and ``h`` tall with its top at ``y``."""
    x1, mid = x0 + w, y + h / 2
    base = dict(fill=PAPER, stroke=INK, stroke_width=1.6)
    if style == "pencil":
        tip = min(10.0, w / 2)
        return (
            _poly([(x0, y), (x1 - tip, y), (x1, mid), (x1 - tip, y + h), (x0, y + h)], True, **base)
            + _line(x0 + 7, y, x0 + 7, y + h, stroke=INK, stroke_width=1.2)
            + _line(x1 - tip, y, x1 - tip, y + h, stroke=INK, stroke_width=1.2)
        )
    if style == "ribbon":
        notch = min(7.0, w / 3)
        return _poly([(x0, y), (x1, y), (x1 - notch, mid), (x1, y + h), (x0, y + h)], True, **base) + _line(
            x0 + 4, mid, x1 - notch - 4, mid, stroke=DARK, stroke_width=0.8, stroke_dasharray="3 2"
        )
    if style == "rope":
        out = [_rect(x0, y, w, h, rx=3, **base)]
        k = 5.0
        while x0 + k + h * 0.5 < x1 - 1:
            out.append(_line(x0 + k, y + h, x0 + k + h * 0.5, y, stroke=DARK, stroke_width=1))
            k += 6
        return "".join(out)
    if style == "snake":
        pts = []
        steps = max(8, int(w / 4))
        for i in range(steps + 1):
            px = x0 + w * 0.97 * i / steps
            pts.append((px, mid + math.sin(i * 1.3) * h * 0.35))
        return (
            _poly(pts, False, stroke=INK, stroke_width=h * 0.5, stroke_linecap="round", stroke_linejoin="round")
            + _poly(pts, False, stroke=PAPER, stroke_width=h * 0.5 - 3, stroke_linecap="round", stroke_linejoin="round")
            + _circle(pts[-1][0], pts[-1][1], h * 0.32, fill=INK, stroke="none")
        )
    if style == "train":
        out = [_rect(x0, y, w, h - 4, rx=2, **base)]
        n = max(1, int(w // 24))
        for i in range(1, n):
            out.append(_line(x0 + w * i / n, y, x0 + w * i / n, y + h - 4, stroke=INK, stroke_width=1.2))
        for cx in (x0 + 8, x1 - 8):
            out.append(_circle(cx, y + h - 3, 3, fill=INK, stroke="none"))
        return "".join(out)
    return _rect(x0, y, w, h, rx=2, **base)


def _cp_unit(style: str, x: float, y: float, w: float) -> str:
    """One informal unit (a paper clip or a block), ``w`` wide, drawn with a little gap either side."""
    x += 1
    w -= 2
    if style == "paper_clip":
        return _rect(x, y + 1, w, 8, rx=4, fill=PAPER, stroke=INK, stroke_width=1.3) + _rect(
            x + 3, y + 3.5, w - 6, 3, rx=1.5, fill="none", stroke=DARK, stroke_width=1
        )
    return _rect(x, y, w, 10, rx=1.5, fill=LIGHT, stroke=INK, stroke_width=1.3)


def _compare_lengths(s: dict) -> Body:
    objects = s["objects"]
    unit = s.get("unit")
    left = 70.0
    extent = max(o["start"] + o["length"] for o in objects)
    u = min(30.0, (CANVAS - left - 14) / max(extent, 4))
    h = 16.0
    pitch = 46.0 if unit else 34.0
    out = []
    y = 14.0
    top = y - 6
    for o in objects:
        x0 = left + o["start"] * u
        out.append(_text(left - 8 if s["lined_up"] else x0 - 8, y + h / 2, o["label"], size=12, anchor="end"))
        out.append(_cp_object(o["style"], x0, y, o["length"] * u, h))
        if unit:
            for i in range(o["units"]):
                out.append(_cp_unit(unit["style"], x0 + i * u, y + h + 4, u))
        y += pitch
    if s["lined_up"]:
        out.append(
            _line(
                left,
                top,
                left,
                y - pitch + h + (24 if unit else 8),
                stroke=DARK,
                stroke_width=1,
                stroke_dasharray="4 3",
            )
        )
    return "".join(out), y + 2


_BAL_SIZE = {1: 14, 2: 20, 3: 28}


def _cp_box(x: float, bottom: float, label: str, size: int) -> tuple[str, float]:
    """An object sitting on a pan with its bottom edge at ``bottom``; (markup, drawn width)."""
    h = _BAL_SIZE[size]
    w = min(66.0, max(30.0, len(label) * 5.6 + 8) + 4 * (size - 1))
    return (
        _rect(x, bottom - h - 4, w, h + 4, rx=3, fill=PAPER, stroke=INK, stroke_width=1.5)
        + _text(x + w / 2, bottom - (h + 4) / 2, label, size=9.5),
        w,
    )


def _cp_pan_load(p: dict, cx: float, bottom: float) -> str:
    """The load on one pan, centred on ``cx``."""
    gap = 6.0
    items: list[tuple[str, int] | None] = [(o["label"], o["size"]) for o in p["objects"]]
    widths = [min(66.0, max(30.0, len(lbl) * 5.6 + 8) + 4 * (sz - 1)) for lbl, sz in items]
    block_cols = min(4, p["blocks"])
    block_w = block_cols * 13.0
    total = sum(widths) + gap * max(0, len(widths) - 1) + (gap if widths and block_cols else 0) + block_w
    x = cx - total / 2
    out = []
    for (lbl, sz), w in zip(items, widths):
        markup, _ = _cp_box(x, bottom, lbl, sz)
        out.append(markup)
        x += w + gap
    for i in range(p["blocks"]):
        col, row = i % 4, i // 4
        out.append(
            _rect(x + col * 13, bottom - 13 * (row + 1) - 1, 12, 12, fill=LIGHT, stroke=INK, stroke_width=1.3)
            + _line(
                x + col * 13 + 2,
                bottom - 13 * (row + 1) + 5,
                x + col * 13 + 10,
                bottom - 13 * (row + 1) + 5,
                stroke=DARK,
                stroke_width=0.8,
            )
        )
    return "".join(out)


def _cp_balance(b: dict, cx: float, half: float, pan_half: float, top: float) -> tuple[str, float]:
    """One pan balance centred on ``cx``; (markup, bottom y). The heavier pan hangs lower."""
    angle = {"left_down": 11.0, "right_down": -11.0, "level": 0.0}[b["tilt"]]
    dy = half * math.sin(math.radians(angle))
    dx = half * math.cos(math.radians(angle))
    pivot = top + 24
    ends = {"left": (cx - dx, pivot + dy), "right": (cx + dx, pivot - dy)}
    string = 46.0
    low = max(e[1] for e in ends.values()) + string
    base_y = low + 18
    out = [
        _line(cx, pivot, cx, base_y, stroke=INK, stroke_width=3),
        _rect(cx - 26, base_y, 52, 7, rx=2, fill=LIGHT, stroke=INK, stroke_width=1.5),
        _line(
            ends["left"][0],
            ends["left"][1],
            ends["right"][0],
            ends["right"][1],
            stroke=INK,
            stroke_width=3.5,
            stroke_linecap="round",
        ),
        _circle(cx, pivot, 4, fill=PAPER, stroke=INK, stroke_width=1.8),
    ]
    for side in ("left", "right"):
        ex, ey = ends[side]
        py = ey + string
        out.append(_line(ex, ey, ex - pan_half, py, stroke=INK, stroke_width=1))
        out.append(_line(ex, ey, ex + pan_half, py, stroke=INK, stroke_width=1))
        out.append(
            _path(
                f"M{_n(ex - pan_half)},{_n(py)} Q{_n(ex)},{_n(py + 11)} {_n(ex + pan_half)},{_n(py)}",
                fill=LIGHT,
                stroke=INK,
                stroke_width=1.8,
            )
        )
        out.append(_cp_pan_load(b[side], ex, py - 1))
    return "".join(out), base_y + 8


def _compare_balance(s: dict) -> Body:
    bals = s["balances"]
    if len(bals) == 1:
        body, bottom = _cp_balance(bals[0], CANVAS / 2, 108, 62, 44)
        return body, bottom + 8
    out, bottom = [], 0.0
    for b, cx in zip(bals, (100, 300)):
        body, bt = _cp_balance(b, cx, 58, 34, 44)
        out.append(body)
        bottom = max(bottom, bt)
    return "".join(out), bottom + 8


def _cp_container(c: dict, cx: float, base: float) -> str:
    from mathgen.figure_kinds.compare import container_size

    w, h = container_size(c["capacity"], c["form"])
    shape = c["shape"]
    flare = {"cup": 1.18, "bowl": 1.5, "jug": 1.0, "bottle": 1.0}[shape]
    wb, wt = w / ((1 + flare) / 2), w / ((1 + flare) / 2) * flare  # bottom and top widths, mean width w
    x0b, x1b = cx - wb / 2, cx + wb / 2
    x0t, x1t = cx - wt / 2, cx + wt / 2
    top = base - h
    frac = c["fill"] / c["capacity"]
    out = []
    if c["fill"]:
        ly = base - h * frac
        wl = wb + (wt - wb) * frac
        out.append(
            _poly(
                [(cx - wb / 2, base), (cx + wb / 2, base), (cx + wl / 2, ly), (cx - wl / 2, ly)],
                True,
                fill=LIGHT,
                stroke="none",
            )
        )
        if c["fill"] < c["capacity"]:
            out.append(_line(cx - wl / 2, ly, cx + wl / 2, ly, stroke=INK, stroke_width=1.6))
    outline = dict(fill="none", stroke=INK, stroke_width=2, stroke_linejoin="round")
    out.append(_poly([(x0t, top), (x0b, base), (x1b, base), (x1t, top)], False, **outline))
    if shape == "jug":
        hx = x1b + (x1t - x1b) / 2
        out.append(
            _path(
                f"M{_n(hx)},{_n(top + h * 0.2)} q{_n(14)},0 {_n(14)},{_n(h * 0.25)} t{_n(-14)},{_n(h * 0.25)}",
                fill="none",
                stroke=INK,
                stroke_width=2,
            )
        )
        out.append(_poly([(x0t, top), (x0t - 7, top - 5), (x0t + 6, top)], False, stroke=INK, stroke_width=2))
    if shape == "bottle":
        nw = min(wt * 0.4, 16)
        out.append(
            _poly(
                [(cx - nw / 2, top - 16), (cx - nw / 2, top), (cx + nw / 2, top), (cx + nw / 2, top - 16)],
                False,
                **outline,
            )
        )
        out.append(_line(cx - nw / 2 - 2, top - 16, cx + nw / 2 + 2, top - 16, stroke=INK, stroke_width=2.5))
    if shape in ("cup", "bowl"):
        out.append(_line(x0t - 2, top, x1t + 2, top, stroke=INK, stroke_width=2.5))
    return "".join(out)


def _compare_containers(s: dict) -> Body:
    from mathgen.figure_kinds.compare import container_size

    cs = s["containers"]
    slot = CANVAS / len(cs)
    tallest = max(container_size(c["capacity"], c["form"])[1] for c in cs)
    base = 22 + tallest + (16 if any(c["shape"] == "bottle" for c in cs) else 0)
    out = []
    bottom = base
    for i, c in enumerate(cs):
        cx = slot * (i + 0.5)
        out.append(_cp_container(c, cx, base))
        out.append(_text(cx, base + 14, c["label"], size=12))
        if c.get("cups"):
            n = c["cups"]
            for k in range(n):
                col, row = k % 4, k // 4
                px = cx - min(n, 4) * 11 / 2 + col * 11
                py = base + 28 + row * 17
                out.append(
                    _poly(
                        [(px, py), (px + 10, py), (px + 8, py + 13), (px + 2, py + 13)],
                        True,
                        fill=LIGHT,
                        stroke=INK,
                        stroke_width=1.2,
                    )
                )
            bottom = max(bottom, base + 28 + ((n - 1) // 4) * 17 + 14)
    return "".join(out), max(bottom, base + 14) + 12


def _compare_picture(s: dict, ids: _Ids) -> Body:
    return {"lengths": _compare_lengths, "balance": _compare_balance, "containers": _compare_containers}[s["layout"]](s)


# ---------------------------------------------------------------- entry point

_RENDERERS: dict[str, Callable[[dict, _Ids], Body]] = {
    "bar_chart": _bar_chart,
    "calendar": _calendar,
    "day_strip": _day_strip,
    "base_ten": _base_ten,
    "box_plot": _box_plot,
    "cartesian": _cartesian,
    "clock": _clock,
    "clock_set": _clock_set,
    "compare_picture": _compare_picture,
    "counters": _counters,
    "diagram": _diagram,
    "dot_plot": _dot_plot,
    "fraction_model": _fraction_model,
    "histogram": _histogram,
    "hundred_chart": _hundred_chart,
    "money": _money,
    "number_line": _number_line,
    "pattern_strip": _pattern_strip,
    "picture_graph": _picture_graph,
    "pie_chart": _pie_chart,
    "protractor": _protractor,
    "scale": _scale,
    "shape_scene": _shape_scene,

    "tally": _tally,
    "tree_diagram": _tree,
    "unit_strip": _unit_strip,
    "venn": _venn,
}


def rendered_kinds() -> tuple[str, ...]:
    """The figure kinds this module can draw."""
    return tuple(sorted(_RENDERERS))


def figure_svg(spec: dict, *, width: int = 400) -> str:
    """Render ``spec`` (a figure from :mod:`mathgen.figures`) as a standalone SVG document.

    The ``<title>`` is the spec's alt text and the root has ``role="img"``. Raises ``ValueError``
    for an unknown kind or an invalid spec. ``width`` is the displayed width in pixels; the
    drawing itself always uses a 400-unit viewBox, so text scales with it.
    """
    if not isinstance(spec, dict) or spec.get("kind") not in _RENDERERS:
        kind = spec.get("kind") if isinstance(spec, dict) else spec
        raise ValueError(f"cannot render figure kind {kind!r} (known: {', '.join(_RENDERERS)})")
    figures.check(spec)
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("width must be a positive integer")
    ids = _Ids()
    body, height = _RENDERERS[spec["kind"]](spec, ids)
    heading = ""
    if spec.get("title"):
        heading = _text(CANVAS / 2, 14, spec["title"], size=15, weight="bold")
        body = _g(body, transform="translate(0 22)")
        height += 22
    height = math.ceil(height)
    title_id = ids("title")
    return _Ids.finish(
        _el(
            "svg",
            _el("title", escape(_clean_text(spec["alt"])), id=title_id)
            + _rect(0, 0, CANVAS, height, fill=PAPER, stroke="none")
            + heading
            + body,
            xmlns="http://www.w3.org/2000/svg",
            viewBox=f"0 0 {CANVAS} {height}",
            width=width,
            height=round(width * height / CANVAS),
            role="img",
            aria_labelledby=title_id,
            font_family="Helvetica, Arial, sans-serif",
            color="#000000",
            fill="none",
            stroke=INK,
            stroke_width=1.5,
        )
    )
