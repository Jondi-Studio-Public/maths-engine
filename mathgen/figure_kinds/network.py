"""The ``network`` figure kind: a graph or network of labelled vertices joined by straight edges.

Vertices sit at fixed positions in a 100 by 100 frame (x right, y down). An edge is a straight segment between two
vertices: undirected, or directed (drawn with an arrowhead at its ``to`` end). It can carry a ``label`` (a weight, a
capacity, a flow written ``3/5``), and it can be ``highlight``-ed (drawn thick: a chosen path, a spanning tree, a
cut). A vertex can be highlighted too (drawn double-ringed).

The layout is the generator's, so it is deterministic. :func:`circle_layout`, :func:`grid_layout`,
:func:`two_row_layout` and :func:`best_circle_layout` build the usual ones. The validator keeps the drawing legible:
vertices are not close together, no edge passes through a vertex that is not its end, no two edge labels sit on top of
each other, and ``plane=True`` additionally requires that no two edges cross (a claim that a network is drawn "with
no crossings" is then checked, not trusted). :func:`place_labels` chooses, for every labelled edge, where along the
edge its label sits so labels of crossing edges do not collide.

``alt`` should say what is drawn ("a network of six vertices and eight edges") and never the answer.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import permutations
from typing import Iterable

from mathgen.figures import _is_num, _xy, check, register_kind

FRAME = 100
MIN_VERTEX_GAP = 14.0  # between vertex centres, in frame units
VERTEX_CLEARANCE = 5.0  # an edge passes no closer than this to a vertex that is not its end
MIN_LABEL_GAP = 7.0  # between the centres of two edge labels, and of a label and a vertex
MAX_VERTICES = 12
MAX_EDGES = 30
MAX_LABEL = 8
MAX_SIDE_NAME = 12  # a vertex drawn as a small dot with its name beside it
LABEL_STEPS = (0.5, 0.4, 0.6, 0.32, 0.68, 0.25, 0.75, 0.2, 0.8)


# ---------------------------------------------------------------- geometry


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_cross(p, q, r, s) -> bool:
    """Whether segments pq and rs properly cross (a shared end point does not count)."""
    d1, d2 = _cross(r, s, p), _cross(r, s, q)
    d3, d4 = _cross(p, q, r), _cross(p, q, s)
    return ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and (
        (d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)
    )


def _point_segment_distance(p, a, b) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _point_at(a, b, t: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def crossing_count(positions: dict, edges: Iterable, stop_at: int | None = None) -> int:
    """How many pairs of edges cross. ``edges`` are ``(u, v)`` pairs of keys of ``positions``. With ``stop_at`` the
    count stops as soon as it reaches that number (the exact count is then not needed)."""
    es = [tuple(e) for e in edges]
    total = 0
    for i, (a, b) in enumerate(es):
        for c, d in es[i + 1 :]:
            if len({a, b, c, d}) == 4 and segments_cross(positions[a], positions[b], positions[c], positions[d]):
                total += 1
                if stop_at is not None and total >= stop_at:
                    return total
    return total


# ---------------------------------------------------------------- layouts


def circle_layout(n: int, radius: float | None = None, start_deg: float = 90.0) -> list[list[float]]:
    """``n`` positions evenly round a circle, the first at the top, then clockwise. The default radius is 38,
    or 42 from nine vertices up so an edge between next-but-one neighbours still clears the vertex between."""
    if radius is None:
        radius = 38.0 if n <= 8 else 42.0
    out = []
    for i in range(n):
        ang = math.radians(start_deg - 360.0 * i / n)
        out.append([round(50 + radius * math.cos(ang), 2), round(50 - radius * math.sin(ang), 2)])
    return out


def grid_layout(rows: int, cols: int, margin: float = 14.0) -> list[list[float]]:
    """``rows * cols`` positions in a regular grid, row by row from the top left."""
    xs = [50.0] if cols == 1 else [margin + (FRAME - 2 * margin) * c / (cols - 1) for c in range(cols)]
    ys = [50.0] if rows == 1 else [margin + (FRAME - 2 * margin) * r / (rows - 1) for r in range(rows)]
    return [[round(x, 2), round(y, 2)] for y in ys for x in xs]


def two_row_layout(left: int, right: int, margin: float = 14.0) -> tuple[list[list[float]], list[list[float]]]:
    """Two columns of vertices (a bipartite drawing): ``left`` on x = 24, ``right`` on x = 76, spread top to bottom."""

    def column(n: int, x: float) -> list[list[float]]:
        if n == 1:
            return [[x, 50.0]]
        return [[x, round(margin + (FRAME - 2 * margin) * i / (n - 1), 2)] for i in range(n)]

    return column(left, 24.0), column(right, 76.0)


@lru_cache(maxsize=4096)
def _best_order(n: int, edges: tuple) -> tuple[int, ...]:
    """The cyclic order of vertices round a circle with the fewest crossings (ties: the first found)."""
    slots = circle_layout(n)
    best, best_cost = tuple(range(n)), None
    # Vertex 0 is fixed at the top; a mirror image has the same cost, so only one of each pair is tried.
    for rest in permutations(range(1, n)):
        if n > 2 and rest[0] > rest[-1]:
            continue
        order = (0, *rest)
        pos = {v: slots[i] for i, v in enumerate(order)}
        cost = crossing_count(pos, edges, stop_at=best_cost)
        if best_cost is None or cost < best_cost:
            best, best_cost = order, cost
            if cost == 0:
                break
    return best


def best_circle_layout(n: int, edges: Iterable) -> list[list[float]]:
    """Positions for vertices ``0..n-1`` on a circle, ordered so the fewest edges cross (for ``n`` up to 8; a plain
    ordering beyond that). The result is a function of ``n`` and ``edges`` alone."""
    es = tuple(sorted(tuple(sorted(e)) for e in edges))
    order = _best_order(n, es) if n <= 8 else tuple(range(n))
    slots = circle_layout(n)
    out: list[list[float]] = [[0.0, 0.0]] * n
    for slot, v in enumerate(order):
        out[v] = slots[slot]
    return out


# ---------------------------------------------------------------- the figure


def network(
    *,
    alt: str,
    vertices: Iterable[tuple[str, float, float]],
    edges: Iterable[dict | tuple],
    highlight_vertices: Iterable[str] = (),
    plane: bool = False,
    title: str = "",
) -> dict:
    """A network drawing.

    - ``vertices``: ``(id, x, y)`` triples in the 100 by 100 frame, or ``(id, x, y, side)`` with ``side`` ``"left"`` or
      ``"right"`` to draw a small dot with the name printed beside it (for a named person or place of up to 12
      characters; a plain vertex is a circle holding an id of up to 8). The name is the vertex's id.
    - ``edges``: ``(from, to)`` pairs or dicts ``{"from", "to", "label"?, "directed"?, "highlight"?, "label_t"?}``.
      ``directed`` draws an arrowhead at ``to``; ``highlight`` draws the edge thick; ``label_t`` (0.2 to 0.8, default
      0.5) is how far along the edge, from ``from``, its label sits (see :func:`place_labels`). No loops, and at most
      one edge between a pair of vertices (two directed edges, one each way, are allowed).
    - ``plane=True``: the drawing must have no two edges crossing.
    """
    es = []
    for e in edges:
        d = {"from": e[0], "to": e[1]} if isinstance(e, tuple) else dict(e)
        es.append(d)
    spec: dict = {
        "kind": "network",
        "alt": alt,
        "vertices": [
            {"id": v[0], "at": [round(float(v[1]), 2), round(float(v[2]), 2)], **({"side": v[3]} if len(v) > 3 else {})}
            for v in vertices
        ],
        "edges": es,
        "highlight_vertices": list(highlight_vertices),
        "plane": bool(plane),
    }
    if title:
        spec["title"] = title
    return check(spec)


def place_labels(spec_vertices: dict, edges: list[dict]) -> list[dict]:
    """Set ``label_t`` on every labelled edge so no two labels (nor a label and a vertex) sit within
    :data:`MIN_LABEL_GAP` of each other, trying :data:`LABEL_STEPS` in order, deterministically.
    ``spec_vertices`` maps vertex id to ``[x, y]``. Edges are changed in place and returned."""
    taken: list[tuple[float, float]] = list(map(tuple, spec_vertices.values()))
    for e in edges:
        if not e.get("label"):
            continue
        a, b = spec_vertices[e["from"]], spec_vertices[e["to"]]
        chosen = None
        for t in LABEL_STEPS:
            p = _point_at(a, b, t)
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= MIN_LABEL_GAP for q in taken):
                chosen = t
                break
        if chosen is None:  # no clean spot: take the one with the most room
            chosen = max(
                LABEL_STEPS,
                key=lambda t: min(
                    math.hypot(_point_at(a, b, t)[0] - q[0], _point_at(a, b, t)[1] - q[1]) for q in taken
                ),
            )
        e["label_t"] = chosen
        taken.append(_point_at(a, b, chosen))
    return edges


def label_point(a, b, t: float = 0.5) -> tuple[float, float]:
    return _point_at(a, b, t)


def _validate_network(spec: dict) -> list[str]:
    problems: list[str] = []
    verts = spec.get("vertices")
    if not isinstance(verts, list) or not 1 <= len(verts) <= MAX_VERTICES:
        return [f"vertices must be a list of 1 to {MAX_VERTICES}"]
    pos: dict = {}
    for i, v in enumerate(verts):
        vid = v.get("id") if isinstance(v, dict) else None
        side = v.get("side") if isinstance(v, dict) else None
        if side not in (None, "left", "right"):
            problems.append(f"vertices[{i}].side must be 'left' or 'right'")
        limit = MAX_SIDE_NAME if side else MAX_LABEL
        if not isinstance(vid, str) or not 1 <= len(vid) <= limit:
            problems.append(f"vertices[{i}].id must be a string of 1 to {limit} characters")
            continue
        if vid in pos:
            problems.append(f"vertex id {vid!r} is repeated")
        at = v.get("at")
        if not _xy(at) or not (4 <= at[0] <= FRAME - 4 and 4 <= at[1] <= FRAME - 4):
            problems.append(f"vertices[{i}].at must be [x, y] inside the frame")
            continue
        pos[vid] = tuple(at)
    es = spec.get("edges")
    if not isinstance(es, list) or len(es) > MAX_EDGES:
        problems.append(f"edges must be a list of at most {MAX_EDGES}")
        return problems
    if problems:
        return problems
    ids = list(pos)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if math.dist(pos[a], pos[b]) < MIN_VERTEX_GAP:
                problems.append(f"vertices {a} and {b} are closer than {MIN_VERTEX_GAP}")
    for vid in spec.get("highlight_vertices", ()):
        if vid not in pos:
            problems.append(f"highlight_vertices names unknown vertex {vid!r}")
    seen: set = set()
    pairs = []
    valid = []
    labels: list[tuple[float, float]] = []
    for i, e in enumerate(es):
        u, w = e.get("from"), e.get("to")
        if u not in pos or w not in pos:
            problems.append(f"edges[{i}] joins unknown vertices {u!r}, {w!r}")
            continue
        if u == w:
            problems.append(f"edges[{i}] is a loop")
            continue
        directed = e.get("directed", False)
        if not isinstance(directed, bool) or not isinstance(e.get("highlight", False), bool):
            problems.append(f"edges[{i}] directed and highlight must be bool")
        key = (u, w) if directed else tuple(sorted((u, w)))
        if key in seen:
            problems.append(f"edges[{i}] repeats the edge {u}-{w}")
        seen.add(key)
        label = e.get("label")
        if label is not None and (not isinstance(label, str) or not 1 <= len(label) <= MAX_LABEL):
            problems.append(f"edges[{i}].label must be a string of 1 to {MAX_LABEL} characters")
        t = e.get("label_t", 0.5)
        if not _is_num(t) or not 0.2 <= t <= 0.8:
            problems.append(f"edges[{i}].label_t must be between 0.2 and 0.8")
            continue
        for other, op in pos.items():
            if other not in (u, w) and _point_segment_distance(op, pos[u], pos[w]) < VERTEX_CLEARANCE:
                problems.append(f"edges[{i}] ({u}-{w}) passes through vertex {other}")
        if label:
            p = _point_at(pos[u], pos[w], t)
            for q in labels:
                if math.hypot(p[0] - q[0], p[1] - q[1]) < MIN_LABEL_GAP:
                    problems.append(f"edges[{i}] label overlaps another label")
                    break
            for other, op in pos.items():
                if math.hypot(p[0] - op[0], p[1] - op[1]) < MIN_LABEL_GAP:
                    problems.append(f"edges[{i}] label sits on vertex {other}")
                    break
            labels.append(p)
        pairs.append((u, w))
        valid.append(e)
    undirected = [tuple(sorted(p)) for p, e in zip(pairs, valid) if not e.get("directed")]
    all_pairs = [tuple(sorted(p)) for p in pairs]
    if any(all_pairs.count(p) > 1 for p in undirected):
        problems.append("an undirected edge is joined by a second edge")
    if not isinstance(spec.get("plane"), bool):
        problems.append("plane must be a bool")
    elif spec["plane"] and not problems:
        n = crossing_count(pos, pairs)
        if n:
            problems.append(f"plane=True but {n} pair(s) of edges cross")
    return problems


def _network_texts(spec: dict) -> list[str]:
    out = [v["id"] for v in spec.get("vertices", ())]
    out += [e["label"] for e in spec.get("edges", ()) if e.get("label")]
    return out


register_kind("network", _validate_network, texts=_network_texts)
