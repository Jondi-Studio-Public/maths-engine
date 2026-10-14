"""Shared helpers for the drawn-space skills (nets, solids, plans, scale drawings, maps).

Not a generator module (the leading underscore keeps the package loader from importing it as one). It holds:

* ``net_figure``: draws any flat net (a list of polygons) as a ``diagram``, fold lines dashed;
* net builders for the common solids (cube nets from grid cells, cuboid, triangular prism, pyramids);
* two independent ways to decide whether six squares fold into a cube (rolling a cube over the net, and tracking
  which cube face each square lands on), and the corner map that rolling gives;
* ``read_faces``: recovers the faces of a drawn net from the figure spec alone, for the figure readers;
* ``solid_figure``: draws a convex solid as a wireframe with hidden edges dashed.

Every figure uses the existing ``diagram`` kind and nothing else.
"""

from __future__ import annotations

import functools
import math
from collections import defaultdict
from fractions import Fraction

from mathgen.core import check
from mathgen.figure_kinds import space

FRAME_MARGIN = 12.0
MAX_EXTENT = 76.0


# ------------------------------------------------------------------------------------------ geometry


def _key(p) -> tuple[float, float]:
    return (round(float(p[0]), 5) + 0.0, round(float(p[1]), 5) + 0.0)


def transform(faces, rotation: int = 0, mirror: bool = False):
    """The faces turned anticlockwise by ``rotation`` quarter turns (after an optional mirror) and moved so
    the smallest coordinates are zero. Returns ``(faces, move)`` where ``move(p)`` applies the same change to
    any other point."""

    def turn(p):
        x, y = float(p[0]), float(p[1])
        if mirror:
            x = -x
        for _ in range(rotation % 4):
            x, y = -y, x
        return x, y

    moved = [[turn(p) for p in face] for face in faces]
    xs = [p[0] for face in moved for p in face]
    ys = [p[1] for face in moved for p in face]
    dx, dy = min(xs), min(ys)

    def move(p):
        x, y = turn(p)
        return x - dx, y - dy

    return [[(x - dx, y - dy) for x, y in face] for face in moved], move


def net_figure(
    faces,
    *,
    alt: str,
    rotation: int = 0,
    mirror: bool = False,
    corner_labels: dict | None = None,
    side_labels: dict | None = None,
    dimension_lines: list | None = None,
    not_to_scale: bool = False,
    fold_lines: bool = True,
):
    """Draw a flat net as a ``diagram``.

    ``faces`` is a list of polygons, each a list of (x, y) corners in order, in any consistent unit. Two faces
    that share a side share the same pair of corners, and that side is drawn dashed (a fold line) when
    ``fold_lines`` is true. ``corner_labels`` maps an original corner (x, y) to a short label drawn at it.
    ``side_labels`` maps ``(face index, side index)`` to a label for the side from corner ``side`` to
    ``side + 1``. ``dimension_lines`` is a list of ``((x, y), (x, y), label)`` drawn dashed beside the net as an
    overall measurement.

    Returns ``(spec, ids)`` where ``ids`` maps each original corner key to its point id.
    """
    corner_labels = corner_labels or {}
    side_labels = side_labels or {}
    placed, move = transform(faces, rotation, mirror)
    extra = [(move(a), move(b), label) for a, b, label in dimension_lines or []]
    xs = [p[0] for face in placed for p in face] + [p[0] for a, b, _ in extra for p in (a, b)]
    ys = [p[1] for face in placed for p in face] + [p[1] for a, b, _ in extra for p in (a, b)]
    left, bottom = min(xs), min(ys)
    width, height = max(xs) - left, max(ys) - bottom
    scale = MAX_EXTENT / max(width, height)

    def px(p):
        return (FRAME_MARGIN + (p[0] - left) * scale, FRAME_MARGIN + (p[1] - bottom) * scale)

    order = sorted({_key(p) for face in placed for p in face}, key=lambda k: (-round(k[1], 3), k[0]))
    ids = {k: f"p{i}" for i, k in enumerate(order)}
    label_for = {}
    for original, text in corner_labels.items():
        label_for[ids[_key(move(original))]] = text
    points = [
        {"id": pid, "at": list(px(k)), **({"label": label_for[pid]} if pid in label_for else {})}
        for k, pid in ids.items()
    ]

    edges: dict[frozenset, dict] = {}
    for fi, face in enumerate(placed):
        n = len(face)
        for si in range(n):
            a, b = _key(face[si]), _key(face[(si + 1) % n])
            e = edges.setdefault(frozenset((ids[a], ids[b])), {"ends": (ids[a], ids[b]), "faces": 0, "label": None})
            e["faces"] += 1
            if (fi, si) in side_labels:
                e["label"] = side_labels[(fi, si)]
    segments = []
    for e in edges.values():
        seg = {"from": e["ends"][0], "to": e["ends"][1]}
        if e["faces"] > 1 and fold_lines:
            seg["dashed"] = True
        if e["label"]:
            seg["label"] = e["label"]
        segments.append(seg)
    for a, b, label in extra:
        ka, kb = _key(a), _key(b)
        for k in (ka, kb):
            if k not in ids:
                ids[k] = f"p{len(ids)}"
                points.append({"id": ids[k], "at": list(px(k))})
        segments.append({"from": ids[ka], "to": ids[kb], "dashed": True, "label": label})
    frame = (width * scale + 2 * FRAME_MARGIN, height * scale + 2 * FRAME_MARGIN)
    spec = space.diagram(alt=alt, points=points, segments=segments, frame=frame, not_to_scale=not_to_scale)
    id_of_original = {_key(p): ids[_key(move(p))] for face in faces for p in face}
    return spec, id_of_original


# ------------------------------------------------------------------------------------------ polyominoes


@functools.lru_cache(maxsize=None)
def fixed_polyominoes(n: int) -> tuple[frozenset, ...]:
    """Every fixed polyomino of ``n`` cells (rotations and reflections count separately), each a frozenset of
    (column, row) cells moved so its smallest column and row are zero. Sorted, so the order is stable."""

    def norm(cells):
        mx = min(c[0] for c in cells)
        my = min(c[1] for c in cells)
        return frozenset((x - mx, y - my) for x, y in cells)

    level = {frozenset({(0, 0)})}
    for _ in range(n - 1):
        nxt = set()
        for shape in level:
            for x, y in shape:
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c = (x + d[0], y + d[1])
                    if c not in shape:
                        nxt.add(norm(shape | {c}))
        level = nxt
    return tuple(sorted(level, key=lambda s: sorted(s)))


def has_block(cells) -> bool:
    """Does the shape contain a 2 by 2 block (so its squares form a loop and cannot fold as a tree)?"""
    s = set(cells)
    return any({(x + 1, y), (x, y + 1), (x + 1, y + 1)} <= s for x, y in s)


@functools.lru_cache(maxsize=None)
def hexomino_nets() -> tuple[tuple, tuple]:
    """(the fixed hexominoes that fold into a cube, the loop-free ones that do not), each sorted."""
    flat = [s for s in fixed_polyominoes(6) if not has_block(s)]
    return tuple(s for s in flat if folds_into_cube(s)), tuple(s for s in flat if not folds_into_cube(s))


def cell_faces(cells):
    """Unit squares for grid cells, in the same order as ``sorted(cells)``."""
    return [[(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)] for x, y in sorted(cells)]


_STEPS = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}
_OPPOSITE_FACE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}


def _tree_order(cells):
    """(cell, parent, step) in breadth-first order from the lowest-leftmost cell; checks the cells form a tree
    of edge-sharing squares."""
    cells = set(cells)
    start = min(cells, key=lambda c: (c[1], c[0]))
    order = [(start, None, None)]
    seen = {start}
    queue = [start]
    links = 0
    for cur in queue:
        for name, (dx, dy) in _STEPS.items():
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in cells:
                links += 1
                if nxt not in seen:
                    seen.add(nxt)
                    order.append((nxt, cur, name))
                    queue.append(nxt)
    check(len(seen) == len(cells), "the squares are not all joined by shared sides")
    check(links // 2 == len(cells) - 1, "the squares form a loop, not a tree")
    return order


def cube_faces_by_frames(cells) -> dict:
    """Which of the cube's six faces (0 to 5, opposite faces paired 0-1, 2-3, 4-5) each square folds onto.

    Method B (face bookkeeping): a square carries the face under it, the face across its north edge and the
    face across its east edge; stepping to a neighbour moves those three along."""
    state = {}
    for cell, parent, step in _tree_order(cells):
        if parent is None:
            state[cell] = (0, 2, 4)  # face, face across the north edge, face across the east edge
            continue
        face, up, right = state[parent]
        opp = _OPPOSITE_FACE
        if step == "E":
            state[cell] = (right, up, opp[face])
        elif step == "W":
            state[cell] = (opp[right], up, face)
        elif step == "N":
            state[cell] = (up, opp[face], right)
        else:
            state[cell] = (opp[up], face, right)
    return {cell: s[0] for cell, s in state.items()}


def folds_into_cube(cells) -> bool:
    """True when six squares fold into a cube (every cube face is covered exactly once)."""
    faces = cube_faces_by_frames(cells)
    return len(cells) == 6 and len(set(faces.values())) == 6


_ROLL = {
    "E": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "W": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "N": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "S": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
}


def _matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _apply(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def cube_corner_map(cells) -> dict:
    """Roll a cube over the squares (method A) and report which corner of the cube sits on each corner of the
    net. Returns ``{(x, y) net corner: cube corner (a triple of +-1)}``. Raises ``VerificationError`` if two
    squares disagree about a shared corner."""
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    state = {}
    mapping: dict[tuple[int, int], tuple[int, int, int]] = {}
    bodies = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    for cell, parent, step in _tree_order(cells):
        rot = identity if parent is None else _matmul(_ROLL[step], state[parent])
        state[cell] = rot
        for v in bodies:
            w = _apply(rot, v)
            if w[2] == -1:
                corner = (cell[0] + (w[0] + 1) // 2, cell[1] + (w[1] + 1) // 2)
                check(mapping.setdefault(corner, v) == v, "two squares disagree about a corner")
    return mapping


# ------------------------------------------------------------------------------------------ solid nets


def _regular_polygon(n: int, side: float):
    """Corners of a regular n-gon of the given side, counterclockwise, first side along the x axis."""
    r = side / (2 * math.sin(math.pi / n))
    cx, cy = 0.0, 0.0
    pts = []
    for i in range(n):
        a = -math.pi / 2 - math.pi / n + 2 * math.pi * i / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def pyramid_net(n: int, side: float, slant: float):
    """Faces of a net of a pyramid on a regular n-gon base: the base with an isosceles triangle on every side."""
    base = _regular_polygon(n, side)
    faces = [base]
    for i in range(n):
        a, b = base[i], base[(i + 1) % n]
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        # outward normal of side a-b for a counterclockwise polygon
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        nx, ny = dy / length, -dx / length
        h = math.sqrt(slant * slant - (side / 2) ** 2)
        apex = (mx + nx * h, my + ny * h)
        faces.append([a, b, apex])
    return faces


def cuboid_net(l: float, w: float, h: float, *, first_wide: bool, top_cell: int, bottom_cell: int):
    """Faces of a net of a cuboid: a strip of four side faces with a face above and a face below."""
    widths = [l, w, l, w] if first_wide else [w, l, w, l]
    xs = [0.0]
    for width in widths:
        xs.append(xs[-1] + width)
    faces = []
    for i in range(4):
        faces.append([(xs[i], 0.0), (xs[i + 1], 0.0), (xs[i + 1], h), (xs[i], h)])
    other = lambda width: w if width == l else l  # noqa: E731
    t = top_cell
    faces.append([(xs[t], h), (xs[t + 1], h), (xs[t + 1], h + other(widths[t])), (xs[t], h + other(widths[t]))])
    b = bottom_cell
    faces.append([(xs[b], 0.0), (xs[b], -other(widths[b])), (xs[b + 1], -other(widths[b])), (xs[b + 1], 0.0)][::-1])
    return faces, xs, widths


def triangle_apex(base: float, left: float, right: float):
    """The apex of a triangle with the given base on the x axis, left side and right side (apex above)."""
    x = (base * base + left * left - right * right) / (2 * base)
    y = math.sqrt(max(left * left - x * x, 0.0))
    return x, y


def triangular_prism_net(sides, length: float, *, top_rect: int, bottom_rect: int):
    """Faces of a net of a triangular prism: three rectangles in a row, a triangle above one and one below one.

    ``sides`` are the triangle's three sides in the order of the rectangles' widths."""
    a, b, c = sides
    xs = [0.0, a, a + b, a + b + c]
    faces = []
    for i in range(3):
        faces.append([(xs[i], 0.0), (xs[i + 1], 0.0), (xs[i + 1], length), (xs[i], length)])
    order = list(sides)

    def triangle(idx, up):
        base = order[idx]
        left, right = order[(idx + 1) % 3], order[(idx + 2) % 3]
        ax, ay = triangle_apex(base, left, right)
        x0 = xs[idx]
        if up:
            return [(x0, length), (x0 + base, length), (x0 + ax, length + ay)]
        return [(x0, 0.0), (x0 + ax, -ay), (x0 + base, 0.0)]

    faces.append(triangle(top_rect, True))
    faces.append(triangle(bottom_rect, False))
    return faces


# ------------------------------------------------------------------------------------------ reading a net


def _points(fig):
    return {p["id"]: (float(p["at"][0]), float(p["at"][1])) for p in fig["points"]}


def read_faces(fig):
    """The bounded faces of a drawn net, found from the figure spec alone.

    Walks the planar graph of the segments. Each face comes back as a list of the (x, y) corners where its
    outline turns (collinear points dropped), in the figure's own coordinates."""
    pos = _points(fig)
    adj = defaultdict(list)
    for s in fig["segments"]:
        a, b = s["from"], s["to"]
        adj[a].append(b)
        adj[b].append(a)

    def angle(a, b):
        return math.atan2(pos[b][1] - pos[a][1], pos[b][0] - pos[a][0])

    for a in adj:
        adj[a].sort(key=lambda b: angle(a, b))
    seen = set()
    faces = []
    for a in adj:
        for b in adj[a]:
            if (a, b) in seen:
                continue
            loop = []
            u, v = a, b
            while (u, v) not in seen:
                seen.add((u, v))
                loop.append(u)
                around = adj[v]
                i = around.index(u)
                # the next side turns as far left as possible: the previous neighbour clockwise from u
                w = around[(i - 1) % len(around)]
                u, v = v, w
                if len(loop) > 200:
                    raise ValueError("face walk did not close")
            pts = [pos[p] for p in loop]
            area = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1])) / 2
            if area > 1e-6:  # the outer boundary walks the other way round and has negative area
                faces.append(_corners(pts))
    return faces


def _corners(pts):
    out = []
    n = len(pts)
    for i in range(n):
        p, q, r = pts[i - 1], pts[i], pts[(i + 1) % n]
        cross = (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])
        if abs(cross) > 1e-6 * max(1.0, math.dist(p, q) * math.dist(q, r)):
            out.append(q)
    return out


def side_lengths(corners):
    return [math.dist(p, q) for p, q in zip(corners, corners[1:] + corners[:1])]


def is_square(corners) -> bool:
    if len(corners) != 4:
        return False
    s = side_lengths(corners)
    return (
        max(s) - min(s) < 1e-3 * max(s) and abs(math.dist(corners[0], corners[2]) - s[0] * math.sqrt(2)) < 1e-3 * s[0]
    )


# ------------------------------------------------------------------------------------------ solids as wireframes


def prism_solid(n: int, side: float, height: float):
    """(vertices, faces) of a right prism on a regular n-gon base; faces are outward-facing vertex rings."""
    base = _regular_polygon(n, side)
    verts = [(x, y, 0.0) for x, y in base] + [(x, y, height) for x, y in base]
    faces = [list(reversed(range(n))), [n + i for i in range(n)]]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])
    return verts, faces


def cuboid_solid(l: float, w: float, h: float):
    verts = [(0, 0, 0), (l, 0, 0), (l, w, 0), (0, w, 0), (0, 0, h), (l, 0, h), (l, w, h), (0, w, h)]
    verts = [(float(x), float(y), float(z)) for x, y, z in verts]
    faces = [[3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return verts, faces


def pyramid_solid(n: int, side: float, height: float):
    base = _regular_polygon(n, side)
    verts = [(x, y, 0.0) for x, y in base] + [(0.0, 0.0, height)]
    faces = [list(reversed(range(n)))]
    for i in range(n):
        faces.append([i, (i + 1) % n, n])
    return verts, faces


def _normal(verts, ring):
    a, b, c = verts[ring[0]], verts[ring[1]], verts[ring[2]]
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def solid_figure(verts, faces, *, alt: str, yaw_degrees: float, elevation_degrees: float = 28.0):
    """Draw a convex solid as a wireframe: edges behind the solid are dashed.

    The solid is turned by ``yaw_degrees`` about the vertical axis and seen from ``elevation_degrees`` above
    the horizon. An edge is hidden exactly when every face that meets it faces away from the viewer."""
    yaw = math.radians(yaw_degrees)
    el = math.radians(elevation_degrees)
    cy, sy = math.cos(yaw), math.sin(yaw)

    def turn(v):
        return (v[0] * cy - v[1] * sy, v[0] * sy + v[1] * cy, v[2])

    rotated = [turn(v) for v in verts]
    toward_viewer = (0.0, -math.cos(el), math.sin(el))
    facing = []
    for ring in faces:
        nx, ny, nz = _normal(rotated, ring)
        facing.append(nx * toward_viewer[0] + ny * toward_viewer[1] + nz * toward_viewer[2] > 1e-9)
    edge_faces: dict[frozenset, list[int]] = defaultdict(list)
    for fi, ring in enumerate(faces):
        for i in range(len(ring)):
            edge_faces[frozenset((ring[i], ring[(i + 1) % len(ring)]))].append(fi)
    screen = [(v[0], v[2] * math.cos(el) + v[1] * math.sin(el)) for v in rotated]
    xs = [p[0] for p in screen]
    ys = [p[1] for p in screen]
    scale = MAX_EXTENT / max(max(xs) - min(xs), max(ys) - min(ys))
    points = [
        {"id": f"v{i}", "at": [FRAME_MARGIN + (p[0] - min(xs)) * scale, FRAME_MARGIN + (p[1] - min(ys)) * scale]}
        for i, p in enumerate(screen)
    ]
    segments = []
    for edge, owners in sorted(edge_faces.items(), key=lambda kv: sorted(kv[0])):
        a, b = sorted(edge)
        seg = {"from": f"v{a}", "to": f"v{b}"}
        if not any(facing[o] for o in owners):
            seg["dashed"] = True
        segments.append(seg)
    frame = ((max(xs) - min(xs)) * scale + 2 * FRAME_MARGIN, (max(ys) - min(ys)) * scale + 2 * FRAME_MARGIN)
    return space.diagram(alt=alt, points=points, segments=segments, frame=frame)


def _gap_to_segment(p, a, b) -> float:
    ab = math.dist(a, b)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / (ab * ab)))
    return math.dist(p, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))


def readable(fig: dict, gap: float = 2.5) -> bool:
    """Can every point and side of a wireframe be told apart? No two points closer than ``gap`` and no point
    within ``gap`` of a side it is not an end of (a vertex sitting on an edge reads as a corner that is not there)."""
    pos = _points(fig)
    ids = list(pos)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if math.dist(pos[a], pos[b]) < gap:
                return False
    for s in fig["segments"]:
        for pid in ids:
            if pid not in (s["from"], s["to"]) and _gap_to_segment(pos[pid], pos[s["from"]], pos[s["to"]]) < gap:
                return False
    return True


def leaks(fig: dict, answer_text: str) -> bool:
    """Does the answer appear in the figure's visible text? (Generators redraw when it does.)"""
    from mathgen import figures

    try:
        figures.assert_no_leak(fig, [answer_text])
    except AssertionError:
        return True
    return False


def exact(value) -> Fraction:
    return Fraction(value).limit_denominator(10**6)
