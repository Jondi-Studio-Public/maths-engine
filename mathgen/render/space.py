"""SVG renderers for ``diagram``, ``tree_diagram`` and ``venn`` (specs in `mathgen.figure_kinds.space`).

Nothing is evaluated here beyond one uniform scale from the spec's frame to the drawing; every label is the
spec's own text. Labels are placed by trying a few candidate positions and keeping the first one that clears
the lines and the labels already placed.
"""

from __future__ import annotations

import math

from mathgen.render._svg import W, Ctx, el, esc, fmt, text

FS = 15
FS_SMALL = 13.5

Pt = tuple[float, float]


class Rect:
    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


def tw(s: str, fs: float) -> float:
    """Rough drawn width of bold UI text at font size ``fs``."""
    w = 0.0
    for ch in s:
        if "0" <= ch <= "9":
            w += 0.6
        elif ch == " ":
            w += 0.3
        elif ch in ",.:;'|!il/":
            w += 0.32
        elif ch in "mMW":
            w += 0.92
        elif ch in "°()":
            w += 0.46
        elif "A" <= ch <= "Z":
            w += 0.68
        else:
            w += 0.58
    return w * fs * 1.04 + 2


def rect_c(cx: float, cy: float, w: float, h: float) -> Rect:
    return Rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def rect_overlap(a: Rect, b: Rect, gap: float = 2) -> bool:
    return a.x0 < b.x1 + gap and b.x0 < a.x1 + gap and a.y0 < b.y1 + gap and b.y0 < a.y1 + gap


def seg_hits_rect(a: Pt, b: Pt, r: Rect, pad: float) -> bool:
    """Does the segment a-b touch the rectangle grown by ``pad``? (Liang-Barsky.)"""
    x0, x1, y0, y1 = r.x0 - pad, r.x1 + pad, r.y0 - pad, r.y1 + pad
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    p = (-dx, dx, -dy, dy)
    q = (a[0] - x0, x1 - a[0], a[1] - y0, y1 - a[1])
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return False
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def line_hits(r: Rect, lines: list[list[Pt]], pad: float = 1.5) -> bool:
    return any(seg_hits_rect(pl[i], pl[i + 1], r, pad) for pl in lines for i in range(len(pl) - 1))


def support(nx: float, ny: float, w: float, h: float) -> float:
    """Half-extent of a w x h box along the unit direction (nx, ny)."""
    return abs(nx) * w / 2 + abs(ny) * h / 2


def _fill(colour: str) -> str:
    return f"fill:{colour}"


# ---------------------------------------------------------------- diagram

D_PADX = 52
D_PADY = 44
D_BOX = W - 2 * D_PADX


def _dir(deg: float) -> Pt:
    r = math.radians(deg)
    return (math.cos(r), -math.sin(r))


class _Ang:
    def __init__(self, at: Pt, a: float, sw: float, r: float, ln: float, label, vid: str):
        self.at, self.a, self.sw, self.r, self.len, self.label, self.vid = at, a, sw, r, ln, label, vid


def _render_diagram(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    src = spec["points"]
    xs = [p["at"][0] for p in src]
    ys = [p["at"][1] for p in src]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    # One uniform scale, fitted to the drawn content so a small figure in a big frame is not left floating.
    bw = max(max_x - min_x, (max_y - min_y) * 0.2, 1e-6)
    bh = max(max_y - min_y, (max_x - min_x) * 0.2, 1e-6)
    s = min(D_BOX / bw, D_BOX / bh)
    off_x = (W - 2 * D_PADX - (max_x - min_x) * s) / 2
    note_h = 26 if spec.get("not_to_scale") else 0
    H = (max_y - min_y) * s + 2 * D_PADY + note_h

    def to_px(x, y) -> Pt:
        return (D_PADX + off_x + (x - min_x) * s, D_PADY + (max_y - y) * s)

    pts: dict[str, Pt] = {}
    math_pts: dict[str, Pt] = {}
    for p in src:
        pts[p["id"]] = to_px(*p["at"])
        math_pts[p["id"]] = (p["at"][0], p["at"][1])

    # Edges: polygon sides first, then segments, one line per pair of points.
    edge_map: dict[tuple, dict] = {}
    polygons = spec.get("polygons", [])
    for pi, poly in enumerate(polygons):
        ids = poly["points"]
        for i, pid in enumerate(ids):
            nx = ids[(i + 1) % len(ids)]
            k = tuple(sorted((pid, nx)))
            if k in edge_map:
                edge_map[k]["polys"].append(pi)
            else:
                edge_map[k] = {"a": pid, "b": nx, "seg": None, "polys": [pi]}
    for sg in spec.get("segments", []):
        k = tuple(sorted((sg["from"], sg["to"])))
        if k in edge_map:
            edge_map[k]["seg"] = sg
        else:
            edge_map[k] = {"a": sg["from"], "b": sg["to"], "seg": sg, "polys": []}
    edges = [e for e in edge_map.values() if e["a"] in pts and e["b"] in pts]
    lines: list[list[Pt]] = [[pts[e["a"]], pts[e["b"]]] for e in edges]

    all_centroid = (sum(p[0] for p in pts.values()) / len(src), sum(p[1] for p in pts.values()) / len(src))

    def poly_centroid(pi: int) -> Pt:
        ids = polygons[pi]["points"]
        return (sum(pts[i][0] for i in ids) / len(ids), sum(pts[i][1] for i in ids) / len(ids))

    # Angle arcs: constant radius at a vertex, inside the sweep.
    arcs: list[str] = []
    rights: list[str] = []
    arc_lines: list[list[Pt]] = []
    ang_geo: list[_Ang] = []
    for an in spec.get("angles", []):
        V, F, T = pts.get(an["at"]), pts.get(an["from"]), pts.get(an["to"])
        if not (V and F and T):
            continue
        m0 = math_pts[an["at"]]

        def bearing(to: str) -> float:
            m = math_pts[to]
            return (math.degrees(math.atan2(m[1] - m0[1], m[0] - m0[0])) + 360) % 360

        a = bearing(an["from"])
        sw = (bearing(an["to"]) - a + 360) % 360
        ln = min(math.hypot(F[0] - V[0], F[1] - V[1]), math.hypot(T[0] - V[0], T[1] - V[1]))
        if an.get("right"):
            q = min(15, ln * 0.4)
            u, v = _dir(a), _dir(a + sw)
            p1 = (V[0] + u[0] * q, V[1] + u[1] * q)
            p2 = (p1[0] + v[0] * q, p1[1] + v[1] * q)
            p3 = (V[0] + v[0] * q, V[1] + v[1] * q)
            rights.append(f"M{fmt(p1[0])},{fmt(p1[1])}L{fmt(p2[0])},{fmt(p2[1])}L{fmt(p3[0])},{fmt(p3[1])}")
            arc_lines.append([p1, p2, p3])
            continue
        ang_geo.append(_Ang(V, a, sw, min(20, ln * 0.45), ln, an.get("label"), an["at"]))
    # Neighbouring angles at one vertex alternate between two radii, and an unknown (non-numeric) label takes
    # the outer one, so each arc reads as its own angle rather than joining its neighbours into one circle.
    by_v: dict[str, list[_Ang]] = {}
    for ga in ang_geo:
        by_v.setdefault(ga.vid, []).append(ga)
    for gs in by_v.values():
        if len(gs) < 2:
            continue
        gs.sort(key=lambda t: t.a)
        unk = next((i for i, t in enumerate(gs) if t.label and not _numeric_angle(t.label)), -1)
        shift = (unk + 1) % 2 if unk >= 0 else 0
        for i, t in enumerate(gs):
            t.r = min(t.r, t.len * 0.4) + ((i + shift) % 2) * 8
    for t in ang_geo:
        trim = 1.5 if t.sw > 8 else 0
        u, v = _dir(t.a + trim), _dir(t.a + t.sw - trim)
        p1 = (t.at[0] + u[0] * t.r, t.at[1] + u[1] * t.r)
        p2 = (t.at[0] + v[0] * t.r, t.at[1] + v[1] * t.r)
        arcs.append(
            f"M{fmt(p1[0])},{fmt(p1[1])}A{fmt(t.r)},{fmt(t.r)} 0 {1 if t.sw > 180 else 0} 0 {fmt(p2[0])},{fmt(p2[1])}"
        )
        arc_lines.append(
            [
                (
                    t.at[0] + _dir(t.a + trim + (t.sw - 2 * trim) * k / 12)[0] * t.r,
                    t.at[1] + _dir(t.a + trim + (t.sw - 2 * trim) * k / 12)[1] * t.r,
                )
                for k in range(13)
            ]
        )

    # Tick and arrow marks at the middle of a side.
    marks: list[str] = []
    for e in edges:
        sg = e["seg"]
        if not sg:
            continue
        A, B = pts[sg["from"]], pts[sg["to"]]
        ln = math.hypot(B[0] - A[0], B[1] - A[1])
        u = ((B[0] - A[0]) / ln, (B[1] - A[1]) / ln)
        n = (-u[1], u[0])
        M = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
        for k in range(sg.get("ticks") or 0):
            o = (k - (sg["ticks"] - 1) / 2) * 6
            c = (M[0] + u[0] * o, M[1] + u[1] * o)
            marks.append(
                f"M{fmt(c[0] - n[0] * 7)},{fmt(c[1] - n[1] * 7)}L{fmt(c[0] + n[0] * 7)},{fmt(c[1] + n[1] * 7)}"
            )
        for k in range(sg.get("arrows") or 0):
            o = (k - (sg["arrows"] - 1) / 2) * 9
            c = (M[0] + u[0] * o, M[1] + u[1] * o)
            tip = (c[0] + u[0] * 4, c[1] + u[1] * 4)
            w1 = (c[0] - u[0] * 4 + n[0] * 6, c[1] - u[1] * 4 + n[1] * 6)
            w2 = (c[0] - u[0] * 4 - n[0] * 6, c[1] - u[1] * 4 - n[1] * 6)
            marks.append(f"M{fmt(w1[0])},{fmt(w1[1])}L{fmt(tip[0])},{fmt(tip[1])}L{fmt(w2[0])},{fmt(w2[1])}")

    # Labels. Points, then side labels, then angle labels dodge what is already placed.
    labels: list[tuple[float, float, str, float, str]] = []
    taken: list[Rect] = []
    obstacles = lines + arc_lines

    def clear(r: Rect, pad: float = 1.5) -> bool:
        return not any(rect_overlap(r, t, 2) for t in taken) and not line_hits(r, obstacles, pad)

    # Point names: just outside the figure, opposite the segments leaving the point.
    for p in src:
        if not p.get("label"):
            continue
        P = pts[p["id"]]
        sxv = syv = 0.0
        for e in edges:
            other = e["b"] if e["a"] == p["id"] else e["a"] if e["b"] == p["id"] else None
            if other is None:
                continue
            Q = pts[other]
            d = math.hypot(Q[0] - P[0], Q[1] - P[1])
            sxv += (Q[0] - P[0]) / d
            syv += (Q[1] - P[1]) / d
        th = -math.pi / 2 if math.hypot(sxv, syv) < 0.05 else math.atan2(-syv, -sxv)
        w = tw(p["label"], FS + 1)
        h = (FS + 1) * 0.95
        best = None
        bc = P
        for dd in (0, 25, -25, 50, -50, 80, -80, 115, -115, 180):
            ang = th + math.radians(dd)
            nx, ny = math.cos(ang), math.sin(ang)
            dist = 8 + support(nx, ny, w, h)
            c = (P[0] + nx * dist, P[1] + ny * dist)
            r = rect_c(c[0], c[1], w, h)
            if best is None:
                best, bc = r, c
            if clear(r, 1):
                best, bc = r, c
                break
        taken.append(best)
        labels.append((bc[0], bc[1], p["label"], FS + 1, "point"))

    # Side labels: beside the middle of the side, on the side away from the figure.
    for e in edges:
        sg = e["seg"]
        if not sg or not sg.get("label"):
            continue
        A, B = pts[sg["from"]], pts[sg["to"]]
        ln = math.hypot(B[0] - A[0], B[1] - A[1])
        u = ((B[0] - A[0]) / ln, (B[1] - A[1]) / ln)
        n = (-u[1], u[0])
        cen = poly_centroid(e["polys"][0]) if e["polys"] else all_centroid
        M = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
        if (M[0] - cen[0]) * n[0] + (M[1] - cen[1]) * n[1] < 0:
            n = (-n[0], -n[1])
        w = tw(sg["label"], FS)
        h = FS * 0.95
        extra = 4 if (sg.get("ticks") or sg.get("arrows")) else 0
        best = None
        bc = M
        done = False
        for side in (1, -1):
            for sl in (0, 0.1, -0.1, 0.2, -0.2):
                nn = (n[0] * side, n[1] * side)
                dist = 6 + extra + support(nn[0], nn[1], w, h)
                base = (M[0] + u[0] * sl * ln, M[1] + u[1] * sl * ln)
                c = (base[0] + nn[0] * dist, base[1] + nn[1] * dist)
                r = rect_c(c[0], c[1], w, h)
                if best is None:
                    best, bc = r, c
                if clear(r, 1.5):
                    best, bc = r, c
                    done = True
                    break
            if done:
                break
        taken.append(best)
        labels.append((bc[0], bc[1], sg["label"], FS, "segment"))

    # Angle labels: on the bisector just beyond the arc, as near as the box fits wholly inside the angle.
    for t in ang_geo:
        if not t.label:
            continue
        w = tw(t.label, FS)
        h = FS * 0.95
        mid = t.a + t.sw / 2
        nx, ny = math.cos(math.radians(mid)), -math.sin(math.radians(mid))
        r_min = t.r + support(nx, ny, w, h) + 4
        rays: list[Pt] = []
        if t.sw <= 180:
            a1, a2 = math.radians(t.a), math.radians(t.a + t.sw)
            rays = [(-math.sin(a1), -math.cos(a1)), (math.sin(a2), math.cos(a2))]
        chosen = fallback = None
        rho = r_min
        while rho <= 260:
            c = (t.at[0] + nx * rho, t.at[1] + ny * rho)
            r = rect_c(c[0], c[1], w, h)
            wedge = all(
                (c[0] - t.at[0]) * ry[0] + (c[1] - t.at[1]) * ry[1] >= support(ry[0], ry[1], w, h) + 2 for ry in rays
            )
            free = not any(rect_overlap(r, q, 2) for q in taken)
            if wedge and free and not line_hits(r, obstacles, 1.5):
                chosen = (r, c)
                break
            if wedge and free and fallback is None:
                fallback = (r, c)
            rho += 2
        if chosen is None:
            chosen = fallback
        if chosen is None:
            c = (t.at[0] + nx * r_min, t.at[1] + ny * r_min)
            chosen = (rect_c(c[0], c[1], w, h), c)
        taken.append(chosen[0])
        labels.append((chosen[1][0], chosen[1][1], t.label, FS, "angle"))

    # The view box is the base box, grown to take in anything that spills out.
    vb = Rect(0, 0, W, H)
    for r in taken + [Rect(p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4) for p in pts.values()]:
        vb.x0 = min(vb.x0, r.x0 - 6)
        vb.y0 = min(vb.y0, r.y0 - 6)
        vb.x1 = max(vb.x1, r.x1 + 6)
        vb.y1 = max(vb.y1, r.y1 + 6)

    parts: list[str] = []
    for poly in polygons:
        d = "".join(f"{'L' if i else 'M'}{fmt(pts[i_][0])},{fmt(pts[i_][1])}" for i, i_ in enumerate(poly["points"]))
        if poly.get("fill"):
            parts.append(el("path", {"d": d + "Z", "style": "fill:var(--_accent);fill-opacity:.12;stroke:none"}))
    parts.append(
        el(
            "g",
            {"stroke": "currentColor", "stroke_width": 2.6, "stroke_linecap": "round"},
            *(
                el(
                    "line",
                    {
                        "x1": pts[e["a"]][0],
                        "y1": pts[e["a"]][1],
                        "x2": pts[e["b"]][0],
                        "y2": pts[e["b"]][1],
                        "stroke_dasharray": "7 5" if e["seg"] and e["seg"].get("dashed") else None,
                    },
                )
                for e in edges
            ),
        )
    )
    if marks:
        parts.append(
            el(
                "g",
                {
                    "stroke": "currentColor",
                    "stroke_width": 2,
                    "fill": "none",
                    "stroke_linecap": "round",
                    "stroke_linejoin": "round",
                },
                *(el("path", {"d": d}) for d in marks),
            )
        )
    if arcs or rights:
        parts.append(
            el(
                "g",
                {"style": "stroke:var(--_c1);stroke-width:2.2;fill:none;stroke-linejoin:miter"},
                *(el("path", {"d": d}) for d in arcs + rights),
            )
        )
    for p in src:
        if p.get("label"):
            P = pts[p["id"]]
            parts.append(el("circle", {"cx": P[0], "cy": P[1], "r": 3.4, "fill": "currentColor"}))
    texts = [
        text(
            x,
            y + fs * 0.35,
            s_,
            style=f"{_fill('var(--_c1)' if kind == 'angle' else 'var(--_text)')};font-size:{fmt(fs)}px;"
            f"font-weight:{700 if kind == 'point' else 600}",
        )
        for x, y, s_, fs, kind in labels
    ]
    if spec.get("not_to_scale"):
        texts.append(
            text(
                (vb.x0 + vb.x1) / 2,
                H - 10,
                "Not drawn to scale",
                style="fill:var(--_text-muted);font-size:13.5px;font-weight:500;font-style:italic",
            )
        )
    parts.append(el("g", {"text_anchor": "middle"}, *texts))

    body = el("g", {"transform": f"translate({fmt(-vb.x0)} {fmt(-vb.y0)})"}, *parts)
    return vb.x1 - vb.x0, vb.y1 - vb.y0, body


def _numeric_angle(label: str) -> bool:
    return label.endswith("°") and len(label) > 1 and all(c in "0123456789." for c in label[:-1])


# ---------------------------------------------------------------- tree

T_NODE_FS = 14
T_BR_FS = 14
T_OUT_FS = 13
T_PILL_H = 24
T_BLANK_W = 30
T_BLANK_H = 20


class _TNode:
    def __init__(self, src, label: str, depth: int, branch):
        self.src, self.label, self.depth, self.branch = src, label, depth, branch
        self.kids: list[_TNode] = []
        self.x = self.y = 0.0
        self.w = tw(label, T_NODE_FS) + 14 if label else 8


def _wrap_tokens(s: str, fs: float, limit: float) -> list[str]:
    parts = s.split(", ")
    toks = [f"{t}," if i < len(parts) - 1 else t for i, t in enumerate(parts)]
    out: list[str] = []
    cur = ""
    for t in toks:
        nxt = f"{cur} {t}" if cur else t
        if cur and tw(nxt, fs) > limit:
            out.append(cur)
            cur = t
        else:
            cur = nxt
    if cur:
        out.append(cur)
    return out


def _render_tree_diagram(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    def build(n, depth: int, label: str, branch) -> _TNode:
        node = _TNode(n, label, depth, branch)
        kids = n.get("children") or [] if n else spec["root"]["children"]
        node.kids = [build(c, depth + 1, c["label"], c["branch"]) for c in kids]
        return node

    root = build(None, 0, spec["root"].get("label") or "", None)
    every: list[_TNode] = []
    leaves: list[_TNode] = []

    def walk(n: _TNode):
        every.append(n)
        if n.kids:
            for k in n.kids:
                walk(k)
        else:
            leaves.append(n)

    walk(root)
    max_depth = max(n.depth for n in every)

    # Columns: each gap is as wide as its widest branch label needs.
    col_w: list[float] = []
    gap: list[float] = []
    for d in range(max_depth + 1):
        here = [n for n in every if n.depth == d]
        col_w.append(max(n.w for n in here))
        brs = [n for n in here if n.branch is not None]
        gap.append(
            0 if d == 0 else max(44, *((T_BLANK_W if n.branch == "?" else tw(n.branch, T_BR_FS)) + 28 for n in brs))
        )
    left = 8
    col_x: list[float] = []
    for d in range(max_depth + 1):
        col_x.append(left if d == 0 else col_x[d - 1] + col_w[d - 1] + gap[d])
    out_x = col_x[max_depth] + col_w[max_depth] + 10
    outs = [(lf.src.get("outcome") or "", lf.src.get("outcome_p")) for lf in leaves]

    # The probability of an outcome is drawn as "P = value", or "P =" and an empty box when it is "?".
    def p_width(p) -> float:
        if not p:
            return 0
        return tw(f"P = {'' if p == '?' else p}", T_OUT_FS) + (34 if p == "?" else 0)

    natural_out = max(0, *(tw(o, T_OUT_FS) for o, _ in outs), *(p_width(p) for _, p in outs))
    limit = max(78, 450 - out_x)
    wrap_limit = math.inf if natural_out <= limit else limit
    wrapped = [[o] if wrap_limit == math.inf else _wrap_tokens(o, T_OUT_FS, wrap_limit) for o, _ in outs]
    out_w = max(0, *(tw(t, T_OUT_FS) for ls in wrapped for t in ls), *(p_width(p) for _, p in outs))
    total_w = max(W, out_x + out_w + 8)
    line_counts = [len(ls) + (1 if outs[i][1] else 0) for i, ls in enumerate(wrapped)]
    pitch = max(
        T_PILL_H + 10,
        max(line_counts) * 15 + 10,
        34 if len(leaves) > 4 else 0,
        42 if max(len(n.kids) for n in every) >= 3 else 0,
    )
    top = 14
    for i, lf in enumerate(leaves):
        lf.y = top + pitch / 2 + i * pitch

    def place(n: _TNode):
        n.x = col_x[n.depth]
        if n.kids:
            for k in n.kids:
                place(k)
            n.y = (n.kids[0].y + n.kids[-1].y) / 2

    place(root)
    h = top * 2 + pitch * len(leaves)

    links: list[tuple[Pt, Pt]] = []
    blanks: list[Rect] = []
    texts: list[tuple[float, float, str, float, bool, str]] = []  # x, y, text, size, muted, anchor
    taken: list[Rect] = []
    for n in every:
        if n.label or n.depth > 0:
            taken.append(Rect(n.x, n.y - T_PILL_H / 2, n.x + n.w, n.y + T_PILL_H / 2))
    for n in every:
        for k in n.kids:
            links.append(((n.x + n.w, n.y), (k.x, k.y)))
    # Branch labels sit on their branch; slide along it if a neighbour is in the way.
    for n in every:
        for k in n.kids:
            a, b = (n.x + n.w, n.y), (k.x, k.y)
            blank = k.branch == "?"
            bw = T_BLANK_W if blank else tw(k.branch, T_BR_FS)
            bh = T_BLANK_H if blank else T_BR_FS * 0.95
            chosen = None
            ts = (0.7, 0.6, 0.78, 0.5, 0.4, 0.85) if len(n.kids) >= 3 else (0.6, 0.5, 0.7, 0.4, 0.78, 0.3, 0.85, 0.2)
            for t in ts:
                c = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                r = rect_c(c[0], c[1], bw, bh)
                if chosen is None:
                    chosen = r
                if not any(rect_overlap(r, q, 1.5) for q in taken):
                    chosen = r
                    break
            taken.append(chosen)
            if blank:
                blanks.append(chosen)
            else:
                texts.append(
                    (
                        (chosen.x0 + chosen.x1) / 2,
                        (chosen.y0 + chosen.y1) / 2 + T_BR_FS * 0.35,
                        k.branch,
                        T_BR_FS,
                        False,
                        "middle",
                    )
                )
    for i, lf in enumerate(leaves):
        lines_ = wrapped[i]
        p = outs[i][1]
        count = len(lines_) + (1 if p else 0)
        y0 = lf.y - (count - 1) * 15 / 2
        for j, t in enumerate(lines_):
            texts.append((out_x, y0 + j * 15 + T_OUT_FS * 0.35, t, T_OUT_FS, False, "start"))
        if p:
            y = y0 + len(lines_) * 15 + T_OUT_FS * 0.35
            if p == "?":
                texts.append((out_x, y, "P = ?", T_OUT_FS, True, "start"))
                bx = out_x + tw("P =", T_OUT_FS) + 4
                blanks.append(Rect(bx, y - T_BLANK_H / 2 - 4, bx + T_BLANK_W, y + T_BLANK_H / 2 - 4))
            else:
                texts.append((out_x, y, f"P = {p}", T_OUT_FS, True, "start"))

    parts: list[str] = []
    parts.append(
        el(
            "g",
            {"stroke": "currentColor", "stroke_width": 1.8, "opacity": 0.6},
            *(el("line", {"x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1]}) for a, b in links),
        )
    )
    if not root.label:
        parts.append(el("circle", {"cx": root.x + 4, "cy": root.y, "r": 4, "fill": "currentColor"}))
    for n in every:
        if not (n.depth > 0 or n.label):
            continue
        fill = "var(--_surface-3)" if n.depth == 0 else "var(--_accent-soft)"
        parts.append(
            el(
                "g",
                None,
                el(
                    "rect",
                    {
                        "x": n.x,
                        "y": n.y - T_PILL_H / 2,
                        "width": n.w,
                        "height": T_PILL_H,
                        "rx": T_PILL_H / 2,
                        "style": f"fill:{fill};stroke:var(--_accent);stroke-width:1.6",
                    },
                ),
                text(
                    n.x + n.w / 2,
                    n.y + T_NODE_FS * 0.35,
                    n.label,
                    style=f"fill:var(--_text);stroke:none;font-size:{T_NODE_FS}px;font-weight:700",
                )
                if n.label
                else "",
            )
        )
    for r in blanks:
        parts.append(
            el(
                "rect",
                {
                    "x": r.x0,
                    "y": r.y0,
                    "width": r.x1 - r.x0,
                    "height": r.y1 - r.y0,
                    "rx": 4,
                    "style": "fill:var(--_surface);stroke:var(--_c1);stroke-width:1.8",
                },
            )
        )

    def caption(x, y, s_, **kw):
        # An unknown probability is "P =" followed by the empty answer box.
        return text(x, y, "P =" if s_ == "P = ?" else s_, **kw)

    parts.append(
        el(
            "g",
            None,
            *(
                caption(
                    x,
                    y,
                    s_,
                    anchor=anchor,
                    style=f"fill:{'var(--_text-muted)' if muted else 'var(--_text)'};font-size:{fmt(fs)}px;"
                    f"font-weight:{700 if anchor == 'middle' else 500}",
                )
                for x, y, s_, fs, muted, anchor in texts
            ),
        )
    )
    return total_w, h, "".join(parts)


# ---------------------------------------------------------------- venn

CIRCLE_VARS = ("var(--_accent)", "var(--_c1)", "var(--_c2)")
_CORNERS = ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))


def _render_venn(spec: dict, ctx: Ctx) -> tuple[float, float, str]:
    sets = spec["sets"]
    n = len(sets)
    if n == 2:
        h = 300
        circles = [(143, 160, 98), (257, 160, 98)]
        name_at = [(88, 46), (312, 46)]
    else:
        h = 392
        circles = [(146, 156, 94), (254, 156, 94), (200, 250, 94)]
        name_at = [(78, 48), (322, 48), (200, 372)]
    names = [(name_at[i][0], name_at[i][1], s.get("name") or s["id"]) for i, s in enumerate(sets)]
    uni_x, uni_y, uni_text = 12, 24, spec["universe"]
    # Keep clear of the universe label, the set names and the rectangle edge.
    obstacles = [Rect(uni_x - 2, uni_y - 14, uni_x + tw(uni_text, FS) + 2, uni_y + 6)] + [
        rect_c(x, y, tw(t, FS) + 2, FS + 2) for x, y, t in names
    ]
    fx0, fy0, fx1, fy1 = 6, 6, W - 6, h - 6

    def margin(px: float, py: float, mask: int, floor: float = -math.inf) -> float:
        """Distance from (px, py) to the nearest boundary of its region. It may stop early and return any
        value below ``floor`` once the margin is known to be under it."""
        m = min(px - fx0, fx1 - px, py - fy0, fy1 - py)
        if m < floor:
            return m
        for i, (cx, cy, r) in enumerate(circles):
            d = math.hypot(px - cx, py - cy)
            v = r - d if (mask >> i) & 1 else d - r
            if v < m:
                m = v
                if m < floor:
                    return m
        for o in obstacles:
            if o.x0 - px >= m or px - o.x1 >= m or o.y0 - py >= m or py - o.y1 >= m:
                continue  # farther than the margin so far, so it cannot lower it
            if o.x0 <= px <= o.x1 and o.y0 <= py <= o.y1:
                v = -min(px - o.x0, o.x1 - px, py - o.y0, o.y1 - py)
            else:
                v = math.hypot(max(o.x0 - px, 0, px - o.x1), max(o.y0 - py, 0, py - o.y1))
            if v < m:
                m = v
                if m < floor:
                    return m
        return m

    ids = [s["id"] for s in sets]
    region_defs = [(mask, "".join(ids[i] for i in range(n) if (mask >> i) & 1)) for mask in range(1, 1 << n)]
    region_defs.append((0, "none"))
    regions = []
    for mask, key in region_defs:
        txt = spec["regions"].get(key, "")

        # Bounding box of the circles this region lies inside, to skip points that cannot be in it.
        inside_c = [c for i, c in enumerate(circles) if (mask >> i) & 1]
        bx0 = max((c[0] - c[2] for c in inside_c), default=-math.inf) - 1e-6
        bx1 = min((c[0] + c[2] for c in inside_c), default=math.inf) + 1e-6
        by0 = max((c[1] - c[2] for c in inside_c), default=-math.inf) - 1e-6
        by1 = min((c[1] + c[2] for c in inside_c), default=math.inf) + 1e-6
        wanted = [(cx, cy, r, r * r, bool((mask >> i) & 1)) for i, (cx, cy, r) in enumerate(circles)]

        def in_region(px: float, py: float) -> bool:
            for cx, cy, r, r2, inside in wanted:
                dx, dy = px - cx, py - cy
                d2 = dx * dx + dy * dy
                # Squared distances decide unless the point is on the rim, where hypot keeps the old answer.
                if abs(d2 - r2) > 1e-6:
                    if (d2 < r2) != inside:
                        return False
                elif (math.hypot(dx, dy) < r) != inside:
                    return False
            return True

        # Region centroid, as a tie-break toward the visual middle.
        cxs = cys = 0.0
        cnt = 0
        py = fy0
        while py <= fy1:
            px = fx0
            while px <= fx1:
                if (
                    bx0 <= px <= bx1
                    and by0 <= py <= by1
                    and (mask != 0 or py >= h * 0.6)
                    and margin(px, py, mask, 1e-9) > 0
                ):
                    cxs += px
                    cys += py
                    cnt += 1
                px += 6
            py += 6
        cen = (cxs / cnt, cys / cnt) if cnt else (W / 2, h / 2)
        toks = txt.split(", ")
        layouts: list[list[str]] = []
        if txt == "?" or len(toks) == 1:
            layouts.append([txt])
        else:
            for nl in range(1, min(4, len(toks)) + 1):
                per = math.ceil(len(toks) / nl)
                layouts.append([", ".join(toks[i : i + per]) for i in range(0, len(toks), per)])
        best = None  # (lines, x, y, fs, clr)
        fs = 15 if "," in txt else 18
        while fs >= 11 and not (best and best[4] >= 3):
            for lines_ in layouts:
                bw = max(tw(ln, fs) for ln in lines_)
                bh = len(lines_) * fs * 1.2
                # Candidate spots on a 4-unit grid. The margin changes by at most the distance moved, so
                # blocks of 4 x 4 spots are tried best-sample-first and a block whose sample is too poor to
                # reach the best margin so far is skipped; the result is that of scanning every spot.
                ys = []
                py = max(fy0 + bh / 2, h * 0.6 if mask == 0 else 0)
                while py <= fy1 - bh / 2:
                    ys.append(py)
                    py += 4
                xs = []
                px = fx0 + bw / 2
                while px <= fx1 - bw / 2:
                    xs.append(px)
                    px += 4
                blocks = []
                for bj in range(0, len(ys), 4):
                    for bi in range(0, len(xs), 4):
                        sx_, sy_ = xs[min(bi + 2, len(xs) - 1)], ys[min(bj + 2, len(ys) - 1)]
                        blocks.append((-margin(sx_, sy_, mask), bj, bi))
                blocks.sort()
                top = None
                cands = []  # (grid order, x, y, margin)
                for neg, bj, bi in blocks:
                    if top is not None and -neg + 11.4 < top - 2:
                        break
                    for j in range(bj, min(bj + 4, len(ys))):
                        py = ys[j]
                        for i in range(bi, min(bi + 4, len(xs))):
                            px = xs[i]
                            thr = top - 2 if top is not None else -math.inf
                            if bx0 <= px <= bx1 and by0 <= py <= by1 and in_region(px, py):
                                if margin(px, py, mask, thr) < thr:
                                    continue
                                clr = math.inf
                                for cx_, cy_ in _CORNERS:
                                    clr = min(clr, margin(px + cx_ * bw / 2, py + cy_ * bh / 2, mask, thr))
                                    if clr < -20 or clr < thr:
                                        break
                                cands.append((j * len(xs) + i, px, py, clr))
                                if top is None or clr > top:
                                    top = clr
                if top is None:
                    continue
                near = min(
                    (c for c in cands if c[3] >= top - 2),
                    key=lambda c: (math.hypot(c[1] - cen[0], c[2] - cen[1]), c[0]),
                )
                cand = (lines_, near[1], near[2], fs, near[3])
                if (
                    best is None
                    or (best[4] < 3 and cand[4] > best[4])
                    or (best[3] != fs and cand[4] >= 3 and best[4] < 3)
                ):
                    best = cand
                if best[4] >= 3:
                    break
            fs -= 1
        b = best or ([txt], cen[0], cen[1], 12, 0)
        regions.append((b[0], b[1], b[2], b[3], txt == "?"))

    parts = [
        el(
            "rect",
            {
                "x": 3,
                "y": 3,
                "width": W - 6,
                "height": h - 6,
                "rx": 8,
                "style": "fill:none;stroke:var(--_border-strong);stroke-width:2",
            },
        )
    ]
    for i, (cx, cy, r) in enumerate(circles):
        parts.append(
            el(
                "circle",
                {
                    "cx": cx,
                    "cy": cy,
                    "r": r,
                    "style": f"fill:{CIRCLE_VARS[i]};fill-opacity:.16;stroke:{CIRCLE_VARS[i]};stroke-width:2.2",
                },
            )
        )
    label_parts = [
        text(uni_x, uni_y, uni_text, anchor="start", style=f"fill:var(--_text-muted);font-size:{FS}px;font-weight:700")
    ]
    for x, y, t in names:
        label_parts.append(text(x, y + FS * 0.35, t, style=f"fill:var(--_text);font-size:{FS}px;font-weight:700"))
    for lines_, x, y, fs, blank in regions:
        tspans = [
            el("tspan", {"x": x, "y": y + (j - (len(lines_) - 1) / 2) * fs * 1.2 + fs * 0.35}, esc(line))
            for j, line in enumerate(lines_)
        ]
        label_parts.append(
            el(
                "text",
                {
                    "text_anchor": "middle",
                    "style": f"fill:{'var(--_c1)' if blank else 'var(--_text)'};font-size:{fmt(fs)}px;font-weight:700",
                },
                *tspans,
            )
        )
    parts.append(el("g", None, *label_parts))
    return W, h, "".join(parts)


from mathgen.render import register_renderer  # noqa: E402

register_renderer("diagram", _render_diagram)
register_renderer("tree_diagram", _render_tree_diagram)
register_renderer("venn", _render_venn)
