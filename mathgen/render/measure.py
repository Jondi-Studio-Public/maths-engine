"""SVG renderers for the measurement figures: clock, clock_set, protractor, unit_strip and scale.

Every number drawn comes from the spec through a linear map or a rotation; nothing is read off the figure and
no value the spec does not label is printed.
"""

from __future__ import annotations

import math

from mathgen.render._svg import Ctx, el, esc, fmt, text

RAD = math.pi / 180

# Rules this module needs beyond `_svg.STYLE`. Literal colours are the object's own (wood, metal, water) and
# read on both a light and a dark page; the rest follow the theme variables.
CSS = (
    ".mgf .mgf-fm-caption{fill:var(--_text);font-size:20px;font-weight:700}"
    ".mgf .mgf-fm-strip-unit{font-size:15px;font-weight:600;fill:var(--_tick)}"
    ".mgf .mgf-fm-clock-rim{fill:var(--_surface);stroke:var(--_text);stroke-width:3.5}"
    ".mgf .mgf-fm-clock-tick-major{stroke:var(--_text);stroke-width:2.6}"
    ".mgf .mgf-fm-clock-tick-minor{stroke:var(--_tick);stroke-width:1.3}"
    ".mgf .mgf-fm-clock-numeral text{fill:var(--_text);font-size:25px;font-weight:600}"
    ".mgf .mgf-fm-clock-hour{stroke:var(--_text);stroke-width:10;stroke-linecap:round}"
    ".mgf .mgf-fm-clock-minute{stroke:var(--_accent);stroke-width:4.5;stroke-linecap:round}"
    ".mgf .mgf-fm-clock-hub{fill:var(--_text);stroke:var(--_surface);stroke-width:2}"
    ".mgf .mgf-fm-prot-plate{fill:color-mix(in srgb,var(--_accent) 10%,transparent);stroke:var(--_text);"
    "stroke-width:2;stroke-linejoin:round}"
    ".mgf .mgf-fm-prot-ring{fill:none;stroke:var(--_tick);stroke-width:1.2}"
    ".mgf .mgf-fm-prot-base{stroke:var(--_tick);stroke-width:1.2}"
    ".mgf .mgf-fm-prot-tick{stroke:var(--_tick);stroke-width:1.3}"
    ".mgf .mgf-fm-prot-tick-long{stroke:var(--_text);stroke-width:1.8}"
    ".mgf .mgf-fm-prot-label text{fill:var(--_text);font-size:12px;font-weight:600}"
    ".mgf .mgf-fm-prot-ray{stroke:var(--_c1);stroke-width:2.6;stroke-linecap:round}"
    ".mgf .mgf-fm-prot-angle{fill:none;stroke:var(--_c1);stroke-width:2;stroke-dasharray:5 3}"
    ".mgf .mgf-fm-prot-vertex{fill:var(--_c1);stroke:var(--_surface);stroke-width:1.5}"
    ".mgf .mgf-fm-name{fill:var(--_text);font-size:18px;font-weight:700}"
    ".mgf .mgf-fm-strip-label{fill:var(--_text);font-size:16px;font-weight:700}"
    ".mgf .mgf-fm-strip-bar{stroke:var(--_text);stroke-width:3.5;stroke-linecap:butt}"
    ".mgf .mgf-fm-strip-end{stroke:var(--_c1);stroke-width:2}"
    ".mgf .mgf-fm-unit rect,.mgf .mgf-fm-unit ellipse{stroke:var(--_text);stroke-width:1.5;fill-opacity:.88}"
    ".mgf .mgf-fm-unit-clip rect{fill:var(--_surface);stroke:var(--_tick);stroke-width:2}"
    ".mgf .mgf-fm-unit-cube>rect:first-child{fill:#60a5fa}"
    ".mgf .mgf-fm-unit-cube .mgf-fm-unit-face{fill:#3b82f6;stroke:var(--_text)}"
    ".mgf .mgf-fm-unit-block rect{fill:#fbbf24}"
    ".mgf .mgf-fm-unit-foot ellipse{fill:#fb923c}"
    ".mgf .mgf-fm-unit-counter>ellipse:first-child{fill:#f87171}"
    ".mgf .mgf-fm-unit-counter .mgf-fm-unit-face{fill:#ef4444}"
    ".mgf .mgf-fm-scale-tick{stroke:var(--_tick);stroke-width:1.1}"
    ".mgf .mgf-fm-scale-tick-major{stroke:var(--_text);stroke-width:1.9}"
    ".mgf .mgf-fm-scale-label text{fill:var(--_text);font-size:14px;font-weight:600}"
    ".mgf .mgf-fm-scale-unit{fill:var(--_text);font-size:15px;font-weight:700}"
    ".mgf .mgf-fm-scale-name{fill:var(--_text);font-size:15px;font-weight:700}"
    ".mgf .mgf-fm-ruler-body{fill:color-mix(in srgb,var(--_c1) 18%,var(--_surface));stroke:var(--_text);"
    "stroke-width:1.8}"
    ".mgf .mgf-fm-ruler-object{fill:var(--_c1);fill-opacity:.35;stroke:var(--_c1);stroke-width:2}"
    ".mgf .mgf-fm-drop{stroke:var(--_c1);stroke-width:1.5;stroke-dasharray:4 2.5;fill:none}"
    ".mgf .mgf-fm-arrow{fill:var(--_c1);stroke:var(--_surface);stroke-width:1}"
    ".mgf .mgf-fm-mark text{fill:var(--_text);font-size:16px;font-weight:700}"
    ".mgf .mgf-fm-thermo-tube{fill:var(--_surface);stroke:var(--_text);stroke-width:2}"
    ".mgf .mgf-fm-thermo-seam{fill:var(--_surface);stroke:none}"
    ".mgf .mgf-fm-liquid-red{fill:#ef4444}"
    ".mgf .mgf-fm-jug-body{fill:none;stroke:var(--_text);stroke-width:2.4;stroke-linejoin:round;"
    "stroke-linecap:round}"
    ".mgf .mgf-fm-liquid-water{fill:#3b82f6;fill-opacity:.4}"
    ".mgf .mgf-fm-liquid-surface{stroke:#3b82f6;stroke-width:2.2}"
    ".mgf .mgf-fm-dial-face{fill:var(--_surface-2);stroke:var(--_text);stroke-width:3}"
    ".mgf .mgf-fm-needle{fill:var(--_c1);stroke:var(--_surface);stroke-width:1}"
    ".mgf .mgf-fm-needle-hub{fill:var(--_text);stroke:var(--_surface);stroke-width:2}"
)


def _num(v: float) -> str:
    """A tick value: float noise rounded away, a proper minus sign (U+2212)."""
    r = round(float(v), 6)
    s = f"{r:.6f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    return s.replace("-", "−")


def _r1(v: float) -> float:
    return round(v, 2)


def _pt(x: float, y: float) -> str:
    return f"{fmt(x)} {fmt(y)}"


def _text_half_extent(theta: float, label: str, fs: float) -> float:
    """Half the extent of an upright text box along direction ``theta`` (degrees anticlockwise from +x)."""
    w = len(label) * fs * 0.6
    h = fs * 0.72
    return abs(math.cos(theta * RAD)) * w / 2 + abs(math.sin(theta * RAD)) * h / 2


# ---------------------------------------------------------------- clock

CLOCK_R = 120
HOUR_LEN = 62
MINUTE_LEN = 112


def _clock_face(spec: dict) -> str:
    """One face in a local frame of radius 120 centred on (0, 0), 12 at the top."""
    ticks = []
    for i in range(60):
        major = i % 5 == 0
        if not major and not spec["minute_ticks"]:
            continue
        a = i * 6 * RAD
        r0 = CLOCK_R - (15 if major else 7)
        ticks.append(
            el(
                "line",
                {
                    "class_": "mgf-fm-clock-tick-major" if major else "mgf-fm-clock-tick-minor",
                    "x1": _r1(r0 * math.sin(a)),
                    "y1": _r1(-r0 * math.cos(a)),
                    "x2": _r1(CLOCK_R * math.sin(a)),
                    "y2": _r1(-CLOCK_R * math.cos(a)),
                },
            )
        )
    numerals = []
    if spec["numerals"] != "none":
        for n in range(1, 13):
            if spec["numerals"] == "quarters" and n % 3:
                continue
            a = n * 30 * RAD
            numerals.append(text(_r1(84 * math.sin(a)), _r1(-84 * math.cos(a)), str(n), dy="0.36em"))
    return el(
        "g",
        None,
        el("circle", {"class_": "mgf-fm-clock-rim", "r": CLOCK_R}),
        el("g", None, *ticks),
        el(
            "line",
            {
                "class_": "mgf-fm-clock-hour",
                "x1": 0,
                "y1": 10,
                "x2": 0,
                "y2": -HOUR_LEN,
                "transform": f"rotate({fmt(spec['hour_angle'])})",
            },
        ),
        el(
            "line",
            {
                "class_": "mgf-fm-clock-minute",
                "x1": 0,
                "y1": 16,
                "x2": 0,
                "y2": -MINUTE_LEN,
                "transform": f"rotate({fmt(spec['minute_angle'])})",
            },
        ),
        el("circle", {"class_": "mgf-fm-clock-hub", "r": 6.5}),
        # numerals last, so a hand pointing at one never hides it
        el("g", {"class_": "mgf-fm-clock-numeral"}, *numerals),
    )


def _render_clock(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    cap = 34 if spec.get("caption") else 0
    size = 2 * CLOCK_R + 14
    c = size / 2
    body = el("g", {"transform": f"translate({fmt(c)} {fmt(c)})"}, _clock_face(spec))
    if spec.get("caption"):
        body += text(c, size + 22, spec["caption"], class_="mgf-fm-caption")
    return size, size + cap, body


def _render_clock_set(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    # Two faces to a row (three or four wrap); the faces share one size.
    w = 400
    cell_w = w / 2
    s = 0.78
    row_h = 2 * CLOCK_R * s + 46
    clocks = spec["clocks"]
    n = len(clocks)
    rows = math.ceil(n / 2)
    parts = []
    for i, c in enumerate(clocks):
        row = i // 2
        in_row = min(2, n - row * 2)
        x = w / 2 if in_row == 1 else cell_w / 2 + (i % 2) * cell_w
        cy = row * row_h + CLOCK_R * s + 6
        parts.append(el("g", {"transform": f"translate({fmt(x)} {fmt(cy)}) scale({fmt(s)})"}, _clock_face(c)))
        if c.get("caption"):
            parts.append(text(x, cy + CLOCK_R * s + 28, c["caption"], class_="mgf-fm-caption"))
    return w, rows * row_h + 6, "".join(parts)


# ----------------------------------------------------------- protractor

P_R = 160  # plate radius = outer scale ticks
P_RI = 116  # ring the inner scale's ticks hang from
P_MARGIN = 24  # plate below the baseline
P_RAY = 184  # rays are longer than the plate's radius
P_FS = 12


def _render_protractor(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    rot = spec["rotation"]
    phi = rot * RAD

    def loc(x: float, y: float) -> tuple[float, float]:
        """Protractor frame (x right along the baseline, y up) to screen, turned anticlockwise by ``rot``."""
        return (_r1(x * math.cos(phi) - y * math.sin(phi)), _r1(-(x * math.sin(phi) + y * math.cos(phi))))

    def polar(r: float, t: float) -> tuple[float, float]:
        return loc(r * math.cos(t * RAD), r * math.sin(t * RAD))

    def dir_pt(length: float, deg: float) -> tuple[float, float]:
        return (_r1(length * math.cos(deg * RAD)), _r1(-length * math.sin(deg * RAD)))

    def pt(p: tuple[float, float]) -> str:
        return _pt(p[0], p[1])

    box = [math.inf, math.inf, -math.inf, -math.inf]  # x0, y0, x1, y1

    def grow(p: tuple[float, float], pad: float = 0) -> None:
        box[0] = min(box[0], p[0] - pad)
        box[2] = max(box[2], p[0] + pad)
        box[1] = min(box[1], p[1] - pad)
        box[3] = max(box[3], p[1] + pad)

    # Plate: a semicircle plus the strip below the baseline (anticlockwise on screen is sweep-flag 0).
    plate = (
        f"M {pt(loc(P_R, 0))} A {P_R} {P_R} 0 0 0 {pt(loc(-P_R, 0))} "
        f"L {pt(loc(-P_R, -P_MARGIN))} L {pt(loc(P_R, -P_MARGIN))} Z"
    )
    for t in range(0, 181, 10):
        grow(polar(P_R, t), 2)
    grow(loc(-P_R, -P_MARGIN), 2)
    grow(loc(P_R, -P_MARGIN), 2)

    ticks: list[str] = []
    labels: list[str] = []

    def label(t: int, r_end: float, s: str, edge: float) -> None:
        # Numerals run along the radius, as on a real protractor, turned to stay upright on screen.
        w = len(s) * P_FS * 0.6
        r = r_end - 3 - w / 2
        c = loc(r * math.cos(t * RAD), r * math.sin(t * RAD) + edge)
        theta_screen = ((t + rot) % 360 + 360) % 360
        flip = 90 < theta_screen < 270
        turn = 180 - theta_screen if flip else -theta_screen
        labels.append(text(c[0], c[1], s, dy="0.36em", transform=f"rotate({fmt(_r1(turn))} {fmt(c[0])} {fmt(c[1])})"))

    for i in range(37):
        t = i * 5
        tenth = i % 2 == 0
        length = 14 if tenth else 8
        cls = "mgf-fm-prot-tick-long" if tenth else "mgf-fm-prot-tick"
        for ring in (P_R, P_RI):
            a, b = polar(ring, t), polar(ring - length, t)
            ticks.append(el("line", {"class_": cls, "x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1]}))
        if not tenth:
            continue
        edge = 5 if t in (0, 180) else 0  # lift 0 / 180 clear of the baseline
        label(t, P_R - 14, str(t), edge)
        label(t, P_RI - 14, str(180 - t), edge)

    arc_local = f"M {pt(loc(P_RI, 0))} A {P_RI} {P_RI} 0 0 0 {pt(loc(-P_RI, 0))}"

    base = dir_pt(P_RAY, spec["base_direction"])
    second = dir_pt(P_RAY, spec["second_direction"])
    grow(base, 2)
    grow(second, 2)

    # The marked angle runs from the base ray toward the second: anticlockwise for zero_on "outer", else clockwise.
    dir_sign = 1 if spec["zero_on"] == "outer" else -1

    def arc_from(r: float, deg: float, span: float, sign: int) -> str:
        a = dir_pt(r, deg)
        b = dir_pt(r, deg + sign * span)
        return f"M {pt(a)} A {r} {r} 0 {1 if span > 180 else 0} {0 if sign > 0 else 1} {pt(b)}"

    if spec["reflex"]:
        r = P_R + 12
        span = 360 - spec["angle"]
        angle_arc = arc_from(r, spec["base_direction"], span, -dir_sign)
        k = 0
        while k <= span:
            grow(dir_pt(r, spec["base_direction"] - dir_sign * k), 2)
            k += 15
    else:
        angle_arc = arc_from(52, spec["base_direction"], spec["angle"], dir_sign)

    names = spec.get("names") or {}
    name_fs = 18
    name_nodes: list[str] = []
    if names.get("vertex"):
        v = loc(0, -P_MARGIN / 2 - 1)
        name_nodes.append(text(v[0], v[1], names["vertex"], class_="mgf-fm-name", dy="0.36em"))

    def end_name(label_text: str | None, deg: float) -> None:
        if not label_text:
            return
        p = dir_pt(P_RAY + 14, deg)
        grow(p, name_fs * 0.6)
        name_nodes.append(text(p[0], p[1], label_text, class_="mgf-fm-name", dy="0.36em"))

    end_name(names.get("base"), spec["base_direction"])
    end_name(names.get("second"), spec["second_direction"])
    grow((0, 0), 4)
    grow(loc(0, -P_MARGIN), 4)

    pad = 8
    vx = box[0] - pad
    vw = box[2] - box[0] + 2 * pad
    vy = box[1] - pad
    vh = box[3] - box[1] + 2 * pad
    if vw < 340:
        vx -= (340 - vw) / 2
        vw = 340

    left, right = loc(-P_R, 0), loc(P_R, 0)
    body = (
        el("path", {"class_": "mgf-fm-prot-plate", "d": plate})
        + el("path", {"class_": "mgf-fm-prot-ring", "d": arc_local})
        + el("line", {"class_": "mgf-fm-prot-base", "x1": left[0], "y1": left[1], "x2": right[0], "y2": right[1]})
        + el("g", None, *ticks)
        + el("path", {"class_": "mgf-fm-prot-angle", "d": angle_arc})
        + el("line", {"class_": "mgf-fm-prot-ray", "x1": 0, "y1": 0, "x2": base[0], "y2": base[1]})
        + el("line", {"class_": "mgf-fm-prot-ray", "x1": 0, "y1": 0, "x2": second[0], "y2": second[1]})
        # numerals over the rays, so a ray through one never hides its digits
        + el("g", {"class_": "mgf-fm-prot-label"}, *labels)
        + el("circle", {"class_": "mgf-fm-prot-vertex", "r": 3.5})
        + "".join(name_nodes)
    )
    # The root viewBox starts at the origin, so shift the drawing instead of moving the box.
    body = el("g", {"transform": f"translate({fmt(-vx)} {fmt(-vy)})"}, body)
    return _r1(vw), _r1(vh), body


# ----------------------------------------------------------- unit_strip

US_W = 400
US_PAD = 16


def _unit_picture(style: str, x: float, y: float, w: float, h: float) -> str:
    if style == "paper_clip":
        # One wire clip inside its slot: an outer loop and a shorter inner loop.
        return el(
            "g",
            {"class_": "mgf-fm-unit mgf-fm-unit-clip"},
            el("rect", {"x": x + 1, "y": y + 1, "width": w - 2, "height": h - 2, "rx": (h - 2) / 2}),
            el(
                "rect",
                {"x": x + w * 0.22, "y": y + h * 0.26, "width": w * 0.78 - 1.5, "height": h * 0.48, "rx": h * 0.48 / 2},
            ),
        )
    if style == "cube":
        return el(
            "g",
            {"class_": "mgf-fm-unit mgf-fm-unit-cube"},
            el("rect", {"x": x + 0.75, "y": y + 0.75, "width": w - 1.5, "height": h - 1.5, "rx": 2}),
            el(
                "rect",
                {
                    "class_": "mgf-fm-unit-face",
                    "x": x + w * 0.2,
                    "y": y + h * 0.2,
                    "width": w * 0.6,
                    "height": h * 0.6,
                    "rx": 1.5,
                },
            ),
        )
    if style == "block":
        return el(
            "g",
            {"class_": "mgf-fm-unit mgf-fm-unit-block"},
            el("rect", {"x": x + 0.75, "y": y + 0.75, "width": w - 1.5, "height": h - 1.5, "rx": 2}),
        )
    if style == "footprint":
        return el(
            "g",
            {"class_": "mgf-fm-unit mgf-fm-unit-foot"},
            el("ellipse", {"cx": x + w * 0.64, "cy": y + h / 2, "rx": w * 0.36 - 0.75, "ry": h / 2 - 0.75}),
            el("ellipse", {"cx": x + w * 0.18, "cy": y + h / 2, "rx": w * 0.17 - 0.75, "ry": h * 0.3}),
        )
    return el(
        "g",
        {"class_": "mgf-fm-unit mgf-fm-unit-counter"},
        el("ellipse", {"cx": x + w / 2, "cy": y + h / 2, "rx": w / 2 - 0.75, "ry": h / 2 - 0.75}),
        el(
            "ellipse",
            {"class_": "mgf-fm-unit-face", "cx": x + w / 2, "cy": y + h / 2, "rx": w * 0.3, "ry": h * 0.3},
        ),
    )


def _render_unit_strip(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    rows = spec["rows"]
    # Widest extent (object bar or any unit's far edge) fills the drawing, capped so a short object is not huge.
    extent = 1.0
    for r in rows:
        extent = max(extent, r["length"], *(u["start"] + u["length"] for u in r["units"]))
    u = min(34.0, (US_W - 2 * US_PAD) / extent)
    unit_h = max(16.0, min(28.0, u * 0.8))

    def overlap_row(r: dict) -> bool:
        us = r["units"]
        return any(k > 0 and un["start"] < us[k - 1]["start"] + us[k - 1]["length"] - 1e-9 for k, un in enumerate(us))

    # A row whose units overlap draws them staggered a little, so one is clearly over the other.
    row_h = 30 + 16 + 6 + unit_h + 14 + (7 if any(overlap_row(r) for r in rows) else 0)
    x0 = US_PAD
    height = len(rows) * row_h + 40

    def X(v: float) -> float:
        return _r1(x0 + v * u)

    parts = []
    for i, row in enumerate(rows):
        top = 8 + i * row_h
        y_bar = top + 30 + 6
        y_units = y_bar + 8
        over = overlap_row(row)
        # The object's name sits above the left end of its bar.
        parts.append(text(X(0), top + 16, row["label"], anchor="start", class_="mgf-fm-strip-label"))
        parts.append(
            el("line", {"class_": "mgf-fm-strip-bar", "x1": X(0), "y1": y_bar, "x2": X(row["length"]), "y2": y_bar})
        )
        for k, un in enumerate(row["units"]):
            parts.append(
                _unit_picture(
                    spec["unit_style"],
                    X(un["start"]),
                    y_units + (6 if over and k % 2 == 1 else 0),
                    _r1(un["length"] * u),
                    unit_h,
                )
            )
        # End lines run through the unit row, so an overhang shows where the object stops.
        for v in (0, row["length"]):
            parts.append(
                el(
                    "line",
                    {
                        "class_": "mgf-fm-strip-end",
                        "x1": X(v),
                        "y1": y_bar - 9,
                        "x2": X(v),
                        "y2": y_units + unit_h + 4 + (6 if over else 0),
                    },
                )
            )
    # The fixed caption "Unit: " is one of the words the renderers may draw.
    parts.append(text(US_PAD, height - 10, f"Unit: {spec['unit']}", anchor="start", class_="mgf-fm-strip-unit"))
    return US_W, height, "".join(parts)


# ---------------------------------------------------------------- scale

TICK_LEN = {"minor": 8, "major": 16, "extra": 4.5}
SCALE_FS = 14


def _scale_ticks(spec: dict) -> list[tuple[float, str]]:
    """A minor tick at every ``minor`` from ``min``, a major at every ``major``, plus the ``extra_ticks``.
    Counted by integer index so no float drift decides which are majors."""
    n = round((spec["max"] - spec["min"]) / spec["minor"])
    every = max(1, round(spec["major"] / spec["minor"]))
    out = [(round(spec["min"] + i * spec["minor"], 6), "major" if i % every == 0 else "minor") for i in range(n + 1)]
    out.extend((v, "extra") for v in spec.get("extra_ticks", ()))
    return out


def _mark_leads(positions: list[float], gap: float) -> list[int]:
    """Stagger markers closer than ``gap``: the later one gets a longer lead so neither hides the other."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    leads = [0] * len(positions)
    for k in range(1, len(order)):
        if positions[order[k]] - positions[order[k - 1]] < gap:
            leads[order[k]] = 1 if leads[order[k - 1]] == 0 else 0
    return leads


def _tick_cls(kind: str) -> str:
    return "mgf-fm-scale-tick-major" if kind == "major" else "mgf-fm-scale-tick"


def _render_ruler(spec: dict) -> tuple[float, float, str]:
    w = 400
    x0, x1 = 22, w - 22
    lo, hi = spec["min"], spec["max"]

    def X(v: float) -> float:
        return _r1(x0 + (v - lo) / (hi - lo) * (x1 - x0))

    body_h = 40
    obj = spec.get("object")
    top = 70 if obj else 62
    bottom = top + body_h
    height = bottom + 30
    ticks = _scale_ticks(spec)
    marks = spec.get("marks", [])
    leads = _mark_leads([X(m["value"]) for m in marks], 22)

    parts = [
        el("rect", {"class_": "mgf-fm-ruler-body", "x": x0, "y": top, "width": x1 - x0, "height": body_h, "rx": 2})
    ]
    parts.append(
        el(
            "g",
            None,
            *(
                el("line", {"class_": _tick_cls(k), "x1": X(v), "x2": X(v), "y1": bottom, "y2": bottom - TICK_LEN[k]})
                for v, k in ticks
            ),
        )
    )
    parts.append(
        el(
            "g",
            {"class_": "mgf-fm-scale-label"},
            *(text(X(v), bottom + 20, _num(v)) for v, k in ticks if k == "major"),
        )
    )
    parts.append(text(x1 - 6, top + 17, spec["unit"], anchor="end", class_="mgf-fm-scale-unit"))
    if obj:
        kids = [
            el(
                "rect",
                {
                    "class_": "mgf-fm-ruler-object",
                    "x": X(obj["start"]),
                    "y": top - 24,
                    "width": X(obj["end"]) - X(obj["start"]),
                    "height": 18,
                    "rx": 3,
                },
            )
        ]
        kids += [
            el("line", {"class_": "mgf-fm-drop", "x1": X(v), "x2": X(v), "y1": top - 6, "y2": bottom})
            for v in (obj["start"], obj["end"])
        ]
        if obj.get("label"):
            kids.append(text((X(obj["start"]) + X(obj["end"])) / 2, top - 31, obj["label"], class_="mgf-fm-scale-name"))
        parts.append(el("g", None, *kids))
    for m, ld in zip(marks, leads):
        x = X(m["value"])
        lead = ld * 20
        parts.append(
            el(
                "g",
                {"class_": "mgf-fm-mark"},
                el("line", {"class_": "mgf-fm-drop", "x1": x, "x2": x, "y1": top - 6 - lead, "y2": bottom}),
                el(
                    "path",
                    {
                        "class_": "mgf-fm-arrow",
                        "d": f"M {_pt(x, top - 4)} l -6 -10 h 12 Z",
                        "transform": f"translate(0 {fmt(-lead)})",
                    },
                ),
                text(x, top - 20 - lead, m["label"]),
            )
        )
    return w, height, "".join(parts)


def _render_vertical(spec: dict) -> tuple[float, float, str]:
    jug = spec["variant"] == "jug"
    w = 260 if jug else 240
    y_top = 56 if jug else 52
    y_bot = 262 if jug else 244
    lo, hi = spec["min"], spec["max"]

    def Y(v: float) -> float:
        return _r1(y_bot - (v - lo) / (hi - lo) * (y_bot - y_top))

    ticks = _scale_ticks(spec)
    body_l = 94 if jug else 112  # left edge of the tube / jug
    body_r = 194 if jug else 130  # right edge: ticks start here
    height = 288 if jug else 304
    bulb_y = y_bot + 22
    bulb_r = 19
    marks = spec.get("marks", [])
    leads = _mark_leads([Y(m["value"]) for m in marks], 17)
    level = spec.get("level")

    parts = []
    if jug:
        # Body with a spout lip at the top left.
        d = (
            f"M {_pt(body_l - 12, y_top - 36)} L {_pt(body_l, y_top - 30)} L {_pt(body_l, y_bot + 6)} "
            f"Q {_pt(body_l, y_bot + 14)} {_pt(body_l + 8, y_bot + 14)} L {_pt(body_r - 8, y_bot + 14)} "
            f"Q {_pt(body_r, y_bot + 14)} {_pt(body_r, y_bot + 6)} L {_pt(body_r, y_top - 36)}"
        )
        kids = [el("path", {"class_": "mgf-fm-jug-body", "d": d})]
        if level is not None:
            ly = Y(level)
            water = (
                f"M {_pt(body_l + 1.5, ly)} H {fmt(body_r - 1.5)} V {fmt(y_bot + 6)} "
                f"Q {_pt(body_r - 1.5, y_bot + 12.5)} {_pt(body_r - 9, y_bot + 12.5)} H {fmt(body_l + 9)} "
                f"Q {_pt(body_l + 1.5, y_bot + 12.5)} {_pt(body_l + 1.5, y_bot + 6)} Z"
            )
            kids.append(
                el(
                    "g",
                    None,
                    el("path", {"class_": "mgf-fm-liquid-water", "d": water}),
                    el(
                        "line",
                        {"class_": "mgf-fm-liquid-surface", "x1": body_l + 1.5, "x2": body_r - 1.5, "y1": ly, "y2": ly},
                    ),
                )
            )
        parts.append(el("g", None, *kids))
    else:
        mid = (body_l + body_r) / 2
        kids = [
            el(
                "rect",
                {
                    "class_": "mgf-fm-thermo-tube",
                    "x": body_l,
                    "y": y_top - 12,
                    "width": body_r - body_l,
                    "height": bulb_y - y_top + 12,
                    "rx": (body_r - body_l) / 2,
                },
            ),
            el("circle", {"class_": "mgf-fm-thermo-tube", "cx": mid, "cy": bulb_y, "r": bulb_r}),
            el(
                "rect",
                {
                    "class_": "mgf-fm-thermo-seam",
                    "x": body_l + 1.5,
                    "y": bulb_y - bulb_r - 4,
                    "width": body_r - body_l - 3,
                    "height": bulb_r,
                },
            ),
        ]
        if level is not None:
            kids.append(
                el(
                    "g",
                    {"class_": "mgf-fm-liquid-red"},
                    el("circle", {"cx": mid, "cy": bulb_y, "r": bulb_r - 4.5}),
                    el(
                        "rect",
                        {"x": body_l + 4.5, "y": Y(level), "width": body_r - body_l - 9, "height": bulb_y - Y(level)},
                    ),
                )
            )
        parts.append(el("g", None, *kids))
    sign = 1 if jug else -1
    parts.append(
        el(
            "g",
            None,
            *(
                el(
                    "line",
                    {
                        "class_": _tick_cls(k),
                        "x1": body_r,
                        "x2": body_r - sign * TICK_LEN[k],
                        "y1": Y(v),
                        "y2": Y(v),
                    },
                )
                for v, k in ticks
            ),
        )
    )
    parts.append(
        el(
            "g",
            {"class_": "mgf-fm-scale-label"},
            *(text(body_r + 9, Y(v), _num(v), anchor="start", dy="0.36em") for v, k in ticks if k == "major"),
        )
    )
    if jug:
        parts.append(text(body_r + 9, y_top - 24, spec["unit"], anchor="start", class_="mgf-fm-scale-unit"))
    else:
        parts.append(text((body_l + body_r) / 2, 20, spec["unit"], class_="mgf-fm-scale-unit"))
    for m, ld in zip(marks, leads):
        y = Y(m["value"])
        lead = ld * 30
        tip_x = body_l
        parts.append(
            el(
                "g",
                {"class_": "mgf-fm-mark"},
                el("line", {"class_": "mgf-fm-drop", "x1": tip_x - lead, "x2": body_r, "y1": y, "y2": y}),
                el(
                    "path",
                    {
                        "class_": "mgf-fm-arrow",
                        "d": f"M {_pt(tip_x - 1, y)} l -11 -6 v 12 Z",
                        "transform": f"translate({fmt(-lead)} 0)",
                    },
                ),
                text(tip_x - 18 - lead, y, m["label"], anchor="end", dy="0.36em"),
            )
        )
    return w, height, "".join(parts)


def _render_dial(spec: dict) -> tuple[float, float, str]:
    w = 320
    cx, cy, radius = w / 2, 160, 108
    start = spec.get("start_angle", -135)
    sweep = spec.get("sweep", 270)
    lo, hi = spec["min"], spec["max"]

    def theta(v: float) -> float:
        """Clockwise-from-12 angle of a value."""
        return start + sweep * (v - lo) / (hi - lo)

    def at(r: float, deg: float) -> tuple[float, float]:
        return (_r1(cx + r * math.sin(deg * RAD)), _r1(cy - r * math.cos(deg * RAD)))

    def p(pt_: tuple[float, float]) -> str:
        return _pt(pt_[0], pt_[1])

    ticks = _scale_ticks(spec)
    marks = spec.get("marks", [])
    # Letters sit outside the rim; nudge apart along the rim if two are close.
    mark_deg = [theta(m["value"]) for m in marks]
    letter_deg = list(mark_deg)
    if len(marks) == 2 and abs(mark_deg[0] - mark_deg[1]) < 22:
        low = 0 if mark_deg[0] < mark_deg[1] else 1
        letter_deg[low] = mark_deg[low] - 6
        letter_deg[1 - low] = mark_deg[1 - low] + 6
    level = spec.get("level")

    tick_lines = []
    for v, k in ticks:
        a, b = at(radius, theta(v)), at(radius - TICK_LEN[k], theta(v))
        tick_lines.append(el("line", {"class_": _tick_cls(k), "x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1]}))
    labels = []
    for v, k in ticks:
        if k != "major":
            continue
        s = _num(v)
        deg = theta(v)
        pos = at(radius - TICK_LEN["major"] - 4 - _text_half_extent(90 - deg, s, SCALE_FS), deg)
        labels.append(text(pos[0], pos[1], s, dy="0.36em"))

    parts = [
        el("circle", {"class_": "mgf-fm-dial-face", "cx": cx, "cy": cy, "r": radius}),
        el("g", None, *tick_lines),
        el("g", {"class_": "mgf-fm-scale-label"}, *labels),
        text(cx, cy + 52, spec["unit"], class_="mgf-fm-scale-unit"),
    ]
    if level is not None:
        th = theta(level)
        d = f"M {p(at(radius - 8, th))} L {p(at(5, th + 90))} L {p(at(14, th + 180))} L {p(at(5, th - 90))} Z"
        parts.append(
            el(
                "g",
                None,
                el("path", {"class_": "mgf-fm-needle", "d": d}),
                el("circle", {"class_": "mgf-fm-needle-hub", "cx": cx, "cy": cy, "r": 7}),
            )
        )
    for i, m in enumerate(marks):
        tip = at(radius + 1, mark_deg[i])
        b1 = at(radius + 15, mark_deg[i] - 4.2)
        b2 = at(radius + 15, mark_deg[i] + 4.2)
        letter = at(radius + 30, letter_deg[i])
        parts.append(
            el(
                "g",
                {"class_": "mgf-fm-mark"},
                el("path", {"class_": "mgf-fm-arrow", "d": f"M {p(tip)} L {p(b1)} L {p(b2)} Z"}),
                text(letter[0], letter[1], m["label"], dy="0.36em"),
            )
        )
    return w, 2 * cy, "".join(parts)


def _render_scale(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    variant = spec["variant"]
    if variant == "ruler":
        return _render_ruler(spec)
    if variant == "dial":
        return _render_dial(spec)
    return _render_vertical(spec)


from mathgen.render import register_renderer  # noqa: E402

register_renderer("clock", _render_clock)
register_renderer("clock_set", _render_clock_set)
register_renderer("protractor", _render_protractor)
register_renderer("unit_strip", _render_unit_strip)
register_renderer("scale", _render_scale)
