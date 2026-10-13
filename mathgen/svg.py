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


# ---------------------------------------------------------------- entry point

_RENDERERS: dict[str, Callable[[dict, _Ids], Body]] = {
    "base_ten": _base_ten,
    "cartesian": _cartesian,
    "clock": _clock,
    "clock_set": _clock_set,
    "counters": _counters,
    "diagram": _diagram,
    "fraction_model": _fraction_model,
    "hundred_chart": _hundred_chart,
    "money": _money,
    "number_line": _number_line,
    "protractor": _protractor,
    "scale": _scale,
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
