"""Descriptions for the cartesian figure kinds: ``cartesian`` and ``number_line``."""

from __future__ import annotations

import math

from mathgen.describe import count, join, num, register_describer, sentence
from mathgen.render._svg import W
from mathgen.render.cartesian import PAD, _fmt_pi_tick, _label_stride, _nice_ticks, _round

#: A curve is read at no more than this many points.
MAX_CURVE_READINGS = 12
#: A curve point within this fraction of a grid step of a grid line is read as lying on it.
_SNAP = 0.015


def _r2(v) -> float:
    return round(float(v), 2) + 0.0


def _n2(v) -> str:
    return num(_r2(v))


def _pi_text(v) -> str:
    s = _fmt_pi_tick(v)
    if "." in s and len(s.split(".")[1].rstrip("π")) > 2:
        s = _fmt_pi_tick(_r2(v))
    return s.replace("−", "-")


def _xfmt(spec: dict):
    return _pi_text if spec["x_tick_format"] == "pi" else _n2


def _pt(spec: dict, x, y) -> str:
    return f"({_xfmt(spec)(x)}, {_n2(y)})"


def _step(spec: dict, v) -> str:
    """A grid step along x, written as drawn (in multiples of pi on a pi axis)."""
    return _pi_text(v) if spec["x_tick_format"] == "pi" else num(v)


# ---------------------------------------------------------------- curves


def _interp(piece, xv):
    """y of the drawn polyline at x = xv, or None outside the piece."""
    for (xa, ya), (xb, yb) in zip(piece, piece[1:]):
        if xa <= xv <= xb and xb > xa:
            return ya + (yb - ya) * (xv - xa) / (xb - xa)
    return None


def _grid_xs(lo, hi, step):
    out = []
    v = math.ceil(lo / step - 1e-9) * step
    while v <= hi + step * 1e-9:
        out.append(_round(v / step) * step)
        v += step
    return out


def _piece_readings(piece, x_grid, y_grid):
    """(priority, x, y) readings of one drawn piece: priority 0 ends, 1 turning points, 2 grid-line crossings."""
    found = [(0, piece[0][0], piece[0][1]), (0, piece[-1][0], piece[-1][1])]
    for i in range(1, len(piece) - 1):
        a, b, c = piece[i - 1][1], piece[i][1], piece[i + 1][1]
        if (b > a and b > c) or (b < a and b < c):
            found.append((1, piece[i][0], b))
    for xv in _grid_xs(piece[0][0], piece[-1][0], x_grid):
        y = _interp(piece, xv)
        if y is None:
            continue
        snapped = _round(y / y_grid) * y_grid
        if abs(y - snapped) <= _SNAP * y_grid:
            found.append((2, xv, snapped))
    return found


def _fill_readings(piece, x_grid, want):
    """Evenly spread readings at grid lines, so a smooth curve with few lattice points is still readable."""
    xs = _grid_xs(piece[0][0], piece[-1][0], x_grid)
    if not xs or want <= 0:
        return []
    if len(xs) > want:
        xs = [xs[round(k * (len(xs) - 1) / max(want - 1, 1))] for k in range(want)]
    out = []
    for xv in xs:
        y = _interp(piece, xv)
        if y is not None:
            out.append((3, xv, y))
    return out


def _thin(items, keep):
    if len(items) <= keep:
        return items
    if keep <= 1:
        return items[:1]
    return [items[round(k * (len(items) - 1) / (keep - 1))] for k in range(keep)]


def _curve_text(spec: dict, curve: dict, multi: bool, index: int) -> str:
    pieces = curve["pieces"]
    xg, yg = spec["x_grid"], spec["y_grid"]
    name = f"The curve labelled {curve['label']}" if curve.get("label") else None
    if name is None:
        name = f"Curve {index + 1}" if multi else "The curve"
    budget = MAX_CURVE_READINGS
    per_piece = []
    for piece in pieces:
        rs = _piece_readings(piece, xg, yg)
        seen, uniq = set(), []
        for pr, x, y in sorted(rs, key=lambda t: (_r2(t[1]), t[0])):
            key = (_r2(x), _r2(y))
            if key not in seen:
                seen.add(key)
                uniq.append((pr, x, y))
        per_piece.append(uniq)
    total = sum(len(u) for u in per_piece)
    if total > budget:
        share = max(2, budget // len(pieces))
        trimmed = []
        for uniq in per_piece:
            hard = [t for t in uniq if t[0] <= 1]
            soft = [t for t in uniq if t[0] == 2]
            if len(hard) >= share:
                keep = _thin(hard, share)
                if len(keep) >= 2 and hard[0] not in keep:
                    keep[0] = hard[0]
                chosen = keep
            else:
                chosen = hard + _thin(soft, share - len(hard))
            trimmed.append(sorted(chosen, key=lambda t: t[1]))
        per_piece = trimmed
    elif total < 6:
        for k, piece in enumerate(pieces):
            have = {_r2(t[1]) for t in per_piece[k]}
            extra = [
                t for t in _fill_readings(piece, xg, max(2, (6 - total) // len(pieces) + 2)) if _r2(t[1]) not in have
            ]
            per_piece[k] = sorted(per_piece[k] + extra, key=lambda t: t[1])
    parts = []
    if len(pieces) == 1:
        parts.append(f"{name} is one unbroken piece")
    else:
        parts.append(f"{name} is drawn in {count(len(pieces), 'separate piece')}, with a gap between them")
    sents = [sentence(parts[0])]
    for k, piece in enumerate(pieces):
        pts = per_piece[k]
        lead = "It" if len(pieces) == 1 else f"Piece {k + 1}"
        first, last = piece[0], piece[-1]
        text = f"{lead} runs from {_pt(spec, *first)} to {_pt(spec, *last)}"
        mid = [
            t
            for t in pts
            if (_r2(t[1]), _r2(t[2])) not in {(_r2(first[0]), _r2(first[1])), (_r2(last[0]), _r2(last[1]))}
        ]
        if mid:
            text += " and passes through " + join(_pt(spec, t[1], t[2]) for t in mid)
        sents.append(sentence(text))
    return " ".join(sents)


# ---------------------------------------------------------------- cartesian


def _describe_cartesian(spec: dict) -> str:
    x0, x1 = spec["x_range"]
    y0, y1 = spec["y_range"]
    xf = _xfmt(spec)
    xg, yg = spec["x_grid"], spec["y_grid"]
    out = ["A coordinate grid."]
    if spec["x_tick_format"] == "pi":
        out.append(
            f"The horizontal axis runs from {xf(x0)} to {xf(x1)}, marked in multiples of π, with grid lines every {_step(spec, xg)}."
        )
    else:
        out.append(f"The horizontal axis runs from {num(x0)} to {num(x1)}, with grid lines every {num(xg)}.")
    out.append(f"The vertical axis runs from {num(y0)} to {num(y1)}, with grid lines every {num(yg)}.")
    if spec["x_label"]:
        out.append(f"The horizontal axis is labelled '{spec['x_label']}'.")
    if spec["y_label"]:
        out.append(f"The vertical axis is labelled '{spec['y_label']}'.")
    drawn = []
    if y0 <= 0 <= y1:
        drawn.append("the horizontal axis line is at y = 0")
    if x0 <= 0 <= x1:
        drawn.append("the vertical axis line is at x = 0")
    if drawn:
        out.append(sentence(join(drawn).capitalize()))
    if not spec["equal_scale"]:
        out.append("The two axes use different scales.")
    out.append(
        "Numbers are written along the axes." if spec["tick_labels"] else "No numbers are written along the axes."
    )

    pts = sorted(spec["points"], key=lambda p: (p["at"][0], p["at"][1]))
    for want_open, noun in ((False, "closed dot"), (True, "open circle")):
        group = [p for p in pts if bool(p.get("open")) == want_open]
        if group:
            items = [(f"{p['label']} at " if p.get("label") else "") + _pt(spec, *p["at"]) for p in group]
            out.append(sentence(f"{count(len(group), noun).capitalize()}: {join(items)}"))

    for s in sorted(spec["segments"], key=lambda s: (s["from"][0], s["from"][1])):
        text = f"A line segment from {_pt(spec, *s['from'])} to {_pt(spec, *s['to'])}"
        if s.get("label"):
            text += f", labelled {s['label']}"
        ends = []
        if s.get("open_start") is not None:
            ends.append(f"{'an open circle' if s['open_start'] else 'a closed dot'} at its start")
        if s.get("open_end") is not None:
            ends.append(f"{'an open circle' if s['open_end'] else 'a closed dot'} at its end")
        if ends:
            text += ", with " + join(ends)
        out.append(sentence(text))

    for ln in spec["lines"]:
        a, b = ln["through"]
        text = f"A straight line through {_pt(spec, *a)} and {_pt(spec, *b)}, continuing to the edges of the grid"
        if ln.get("label"):
            text += f", labelled {ln['label']}"
        out.append(sentence(text))

    multi = len(spec["curves"]) > 1
    for i, c in enumerate(spec["curves"]):
        out.append(_curve_text(spec, c, multi, i))
    return " ".join(out)


# ---------------------------------------------------------------- number line


def _describe_number_line(spec: dict) -> str:
    r0, r1 = spec["range"]
    tick = spec["tick"]
    out = [f"A number line with an arrowhead at each end, running from {num(r0)} to {num(r1)}."]
    ticks = _nice_ticks(r0, r1, tick)
    out.append(f"Tick marks are every {num(tick)}.")
    if spec["tick_labels"]:
        scale = (W - 2 * PAD) / (r1 - r0)
        stride = _label_stride(scale * tick, max([1] + [len(num(v)) for v in ticks]) * 14 * 0.62)
        shown = [v for v in ticks if _round(v / tick) % stride == 0]
        if shown and stride == 1:
            out.append(f"Every tick mark is numbered, from {num(shown[0])} to {num(shown[-1])}.")
        elif shown:
            out.append(
                f"Numbers are written under every {num(stride)}{_ordinal(stride)} tick mark, "
                f"from {num(shown[0])} to {num(shown[-1])}."
            )
    else:
        out.append("No numbers are written under the tick marks.")

    def circle(open_) -> str:
        return "an open circle" if open_ else "a closed dot"

    for iv in sorted(spec["intervals"], key=lambda iv: iv["from"]):
        out.append(
            sentence(
                f"A shaded stretch of the line from {num(iv['from'])} to {num(iv['to'])}, "
                f"with {circle(iv.get('open_start'))} at {num(iv['from'])} and {circle(iv.get('open_end'))} at {num(iv['to'])}"
            )
        )
    for ray in sorted(spec["rays"], key=lambda r: r["from"]):
        out.append(
            sentence(
                f"A shaded ray starting at {num(ray['from'])} with {circle(ray.get('open'))}, "
                f"going {ray['direction']} to the end of the line"
            )
        )
    for m in sorted(spec["marks"], key=lambda m: m["at"]):
        text = f"{circle(m.get('open')).capitalize()} at {num(m['at'])}"
        if m.get("label"):
            text += f", labelled {m['label']}"
        out.append(sentence(text))
    return " ".join(out)


def _ordinal(n) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


register_describer("cartesian", _describe_cartesian)
register_describer("number_line", _describe_number_line)
