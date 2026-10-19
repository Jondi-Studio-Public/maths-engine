"""Shared builders for the ``network`` figure in graph and network questions.

Two layers:

* :func:`graph_figure` turns an ``(n, edges)`` (vertices ``0..n-1`` named ``A``, ``B``, ...) into a validated figure
  with a deterministic layout, for a generator that holds the network itself.
* :func:`with_network_figure` is a decorator for a generator whose question text prints one network (an edge list
  such as ``joins: AB, AC``, weighted pairs such as ``AB 13``, ``A-D, cost $1`` or ``AB=4``, or capacities such as
  ``A→C 6`` and ``Pipe from S to A: capacity 7``). It reads that network back out of the text a learner sees and
  attaches the matching figure, so the picture always agrees with the printed edges. The text is left as it is: a
  question that prints no single network (four listed networks, a matrix, a described situation) gets no figure.

Layouts are fixed by the vertices and edges alone: the circle ordering with the fewest crossings for an undirected
network, and columns by distance from the source for a flow network (source left, sink right).
"""

from __future__ import annotations

import functools
import math
import re
from itertools import permutations, product

from .. import figures
from ..figure_kinds import network as net

LETTERS = "ABCDEFGHIJKL"


def graph_figure(
    n,
    edges,
    *,
    positions=None,
    names=None,
    weights=None,
    directed=False,
    highlight_edges=(),
    highlight_vertices=(),
    plane=False,
    sides=None,
    alt=None,
):
    """A ``network`` figure of vertices ``0..n-1`` and ``edges`` (pairs of vertex numbers).

    ``weights`` maps an edge ``(u, v)`` to its label text; ``highlight_edges`` is a set of edges drawn thick;
    ``names`` overrides the vertex names; ``sides`` maps a vertex number to ``"left"``/``"right"`` to draw it as a dot with its name beside it; ``positions`` is a list of ``[x, y]`` per vertex (default: the circle layout
    with the fewest crossings). ``alt`` names what is drawn and never the answer.
    """
    names = list(names) if names else list(LETTERS[:n])
    pos = positions if positions is not None else net.best_circle_layout(n, edges)
    hot = {tuple(sorted(e)) for e in highlight_edges}
    es = []
    for u, v in edges:
        d = {"from": names[u], "to": names[v]}
        if directed:
            d["directed"] = True
        if weights and (u, v) in weights:
            d["label"] = str(weights[(u, v)])
        elif weights and (v, u) in weights and not directed:
            d["label"] = str(weights[(v, u)])
        if tuple(sorted((u, v))) in hot:
            d["highlight"] = True
        es.append(d)
    sides = sides or {}
    verts = [(names[i], pos[i][0], pos[i][1], *([sides[i]] if i in sides else [])) for i in range(n)]
    net.place_labels({v[0]: (v[1], v[2]) for v in verts}, es)
    if alt is None:
        alt = f"A {'directed ' if directed else ''}network drawn as labelled vertices joined by straight edges."
    return net.network(
        alt=alt,
        vertices=verts,
        edges=es,
        highlight_vertices=[names[i] for i in highlight_vertices],
        plane=plane,
    )


def layered_layout(names, arcs, source="S", sink="T"):
    """Positions for a flow network: ``source`` at the left, ``sink`` at the right, every other vertex in a column by
    its longest distance from the source, columns ordered to give the fewest crossings and no edge through a vertex.

    ``names`` is the list of vertex ids; ``arcs`` the ``(from, to)`` pairs. Returns ``[x, y]`` per name.
    """
    layer = {v: 0 for v in names}
    for _ in names:
        for u, v in arcs:
            if v != sink and layer[v] < layer[u] + 1:
                layer[v] = layer[u] + 1
    top = max([layer[v] for v in names if v not in (source, sink)] + [0]) + 1
    layer[sink] = top
    layer[source] = 0
    cols = [[v for v in names if layer[v] == c] for c in range(top + 1)]
    xs = [14.0 + (net.FRAME - 28.0) * c / top for c in range(top + 1)]

    slots = (50.0, 34.0, 66.0, 18.0, 82.0)

    def spots(c, k):
        if c in (0, top):
            return [(50.0,)] if k == 1 else list(permutations(slots, k))
        return list(permutations(slots, k))

    def place(assignment):
        return {v: (xs[c], y) for c, (col, ys) in enumerate(zip(cols, assignment)) for v, y in zip(col, ys)}

    def cost(pos):
        bad = 0
        for u, v in arcs:
            for w in names:
                if w not in (u, v) and _near(pos[w], pos[u], pos[v]):
                    bad += 100
        edges = [{"from": u, "to": v, "label": "0"} for u, v in arcs]
        net.place_labels(pos, edges)
        pts = [net.label_point(pos[e["from"]], pos[e["to"]], e["label_t"]) for e in edges]
        crowd = sum(
            1
            for i, p in enumerate(pts)
            for q in [*pts[i + 1 :], *pos.values()]
            if math.hypot(p[0] - q[0], p[1] - q[1]) < net.MIN_LABEL_GAP
        )
        return bad + 10 * crowd + net.crossing_count(pos, arcs)

    best, best_cost = None, None
    options = [spots(c, len(col)) for c, col in enumerate(cols)]
    for assignment in product(*options):
        pos = place(assignment)
        c = cost(pos)
        if best_cost is None or c < best_cost:
            best, best_cost = pos, c
            if c == 0:
                break
    return [list(best[v]) for v in names]


def _near(p, a, b) -> bool:
    return net._point_segment_distance(p, a, b) < net.VERTEX_CLEARANCE + 1.5


# ---------------------------------------------------------------- reading a network back out of question text

_NAME = r"[A-Z]"
_PLAIN = re.compile(r"(?:joins:|links are:|edges are|and the \d+ edges) ((?:[A-L]{2}, )*[A-L]{2})\b(?!\s*\d)")
_PAIR_WEIGHTED = re.compile(r"(?:are|is): ((?:[A-L]{2} \d+, )*[A-L]{2} \d+)\.")
_EQ_WEIGHTED = re.compile(r"\b([A-L])([A-L])=(\d+)")
_DASH_COST = re.compile(r"\b([A-L])-([A-L]), (?:cost )?\$(\d+)")
_ARROW_CAP = re.compile(r"\b([A-LST])→([A-LST]) (\d+)")
_PIPE_CAP = re.compile(r"Pipe from ([A-LST]) to ([A-LST]): capacity (\d+)")
_RANGE = re.compile(r"\b([A-L]) to ([A-L])\b")


def _letters_between(text):
    m = _RANGE.search(text)
    return list(LETTERS[LETTERS.index(m.group(1)) : LETTERS.index(m.group(2)) + 1]) if m else []


def network_from_text(text):
    """``(names, [(u, v, weight or None)], directed)`` for the one network ``text`` prints, or ``None``."""
    found = []
    m = _PAIR_WEIGHTED.search(text)
    if m:
        found.append(([(p[0], p[1], p.split()[1]) for p in m.group(1).split(", ")], False))
    elif _PLAIN.search(text):
        found.append(([(p[0], p[1], None) for p in _PLAIN.search(text).group(1).split(", ")], False))
    for rx, directed in ((_EQ_WEIGHTED, False), (_DASH_COST, False), (_ARROW_CAP, True)):
        triples = rx.findall(text)
        if triples:
            found.append(([(a, b, w) for a, b, w in triples], directed))
    pipes = _PIPE_CAP.findall(text)
    if pipes:
        found.append(([(a, b, w) for a, b, w in pipes], True))
    if len(found) != 1:
        return None
    arcs, directed = found[0]
    if len({(min(a, b), max(a, b)) if not directed else (a, b) for a, b, _ in arcs}) != len(arcs):
        return None
    seen = {x for a, b, _ in arcs for x in (a, b)}
    if directed:
        middle = sorted(seen - {"S", "T"})
        names = ["S", *middle, "T"] if "S" in seen else sorted(seen)
    else:
        names = sorted(seen | set(_letters_between(text)))
    return names, arcs, directed


def figure_from_text(text, unweighted=False):
    """The ``network`` figure for the network ``text`` prints, or ``None`` when it prints no single network."""
    read = network_from_text(text)
    if read is None:
        return None
    names, arcs, directed = read
    if unweighted:
        arcs = [(a, b, None) for a, b, _ in arcs]
    if len(names) > net.MAX_VERTICES or len(arcs) > net.MAX_EDGES:
        return None
    index = {v: i for i, v in enumerate(names)}
    edges = [(index[a], index[b]) for a, b, _ in arcs]
    weights = {(index[a], index[b]): w for a, b, w in arcs if w is not None}
    if directed and "S" in names and "T" in names:
        pos = layered_layout(names, [(a, b) for a, b, _ in arcs])
    else:
        pos = net.best_circle_layout(len(names), edges)
    kind = "A flow network" if directed else ("A weighted network" if weights else "A network")
    return graph_figure(
        len(names),
        edges,
        names=names,
        positions=pos,
        weights=weights or None,
        directed=directed,
        alt=f"{kind} drawn as labelled vertices joined by "
        + (
            "arrows, each labelled with its capacity."
            if directed
            else "edges, each labelled with its weight."
            if weights
            else "edges."
        ),
    )


def _answers(d):
    """Every answer text a question asks for (a single answer, or one per part)."""
    out = [d.get("answer_text")] + [p.get("answer_text") for p in d.get("parts") or ()]
    return [str(a) for a in out if a]


def with_network_figure(fn=None, *, unweighted=False):
    """Decorator: attach the figure of the network the returned question's text prints (see the module docstring).

    A figure is kept only if no label or alt text gives away an answer (``figures.assert_no_leak``), which is the
    rule every figure question is swept against. When an answer equals a weight printed on the picture, the network
    is drawn without its weights instead (the text still lists them), so every question that prints a network gets
    a picture. ``unweighted=True`` always draws it that way. The question itself is never redrawn, so the decorator
    changes nothing about the distribution of questions a generator makes."""
    if fn is None:
        return lambda f: with_network_figure(f, unweighted=unweighted)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        d = fn(*args, **kwargs)
        if not (isinstance(d, dict) and d.get("figure") is None and isinstance(d.get("text"), str)):
            return d
        for plain in (True,) if unweighted else (False, True):
            try:
                found = figure_from_text(d["text"], plain)
            except ValueError:  # a layout that cannot be drawn legibly: the text alone is complete, so go without
                return d
            if found is None:
                return d
            try:
                figures.assert_no_leak(found, _answers(d))
            except AssertionError:
                continue
            d["figure"] = found
            return d
        return d

    return wrapper
