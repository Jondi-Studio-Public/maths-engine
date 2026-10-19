"""Generates `uc2-dijkstra-trace` questions (easy, medium, hard) and the exam chain `exd_x_r4j_dijkstra_trace`.

``uc2-dijkstra-trace`` (``discrete-mathematics``, answer ``ordered_list``): the order in which Dijkstra's algorithm
finalises the vertices of a small weighted graph. The tie rule is fixed and stated in every stem: when two unfinalised
vertices have the same smallest tentative distance, the one that comes first alphabetically is finalised first. With that
rule the order is unique, and it equals the vertices sorted by (shortest distance, label).

* easy: an undirected graph on six vertices, all shortest distances different, source A.
* medium: an undirected graph on seven vertices with at least one tie and at least two labels that are lowered after
  they are first set.
* hard: a directed graph on eight vertices, a source other than A, one vertex that cannot be reached (it is left out),
  ties and lowered labels.
* exam (``exd_x_r4j_dijkstra_trace``): on an undirected graph with seven vertices, (a) the finalising order, (b) the
  shortest distance to a named vertex, (c) the unique shortest route to it, (d) the cost at a rate per unit of length,
  which follows through from (b).

The trace is computed by the algorithm itself; the distances are checked against Bellman-Ford and the order against
sorting by (distance, label). The route to the named vertex is checked to be the only shortest one.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._r4j_kit import keep_distinct, say

_TAGS = {
    "tie": "dijkstra-tie-rule-reversed",
    "stale": "dijkstra-labels-not-updated",
    "bfs": "dijkstra-fewest-edges-order",
    "walk": "dijkstra-nearest-neighbour-walk",
    "alpha": "dijkstra-alphabetical-order",
    "fewest": "dijkstra-fewest-edges-path",
    "last_edge": "dijkstra-last-edge-omitted",
    "neighbour": "dijkstra-distance-to-wrong-vertex",
    "reversed": "path-direction-reversed",
    "order_as_path": "dijkstra-order-read-as-path",
    "cost_rate_added": "dijkstra-rate-added-not-multiplied",
    "cost_rate_ignored": "dijkstra-rate-ignored",
    "cost_per_edge": "dijkstra-rate-applied-per-edge",
}

_TIE = "If two unfinalised vertices have the same smallest tentative distance, finalise the one that comes first alphabetically."
_INF = float("inf")


# -- the algorithm and its second route --------------------------------------------------------------------------------


def dijkstra(adj, source, *, tie_last=False, update=True):
    """``(order, dist, pred)``. `adj[u]` is ``{v: w}``. `tie_last` breaks ties by the later label and `update=False`
    never lowers a label once it is set; both are deliberately wrong rules for distractors."""
    dist = {source: 0}
    pred = {source: None}
    done, order = set(), []
    for _ in range(len(adj) + 1):
        open_ = [v for v in dist if v not in done]
        if not open_:
            break
        if tie_last:
            u = min(open_, key=lambda v: (dist[v], [-ord(ch) for ch in v]))
        else:
            u = min(open_, key=lambda v: (dist[v], v))
        done.add(u)
        order.append(u)
        for v, w in adj[u].items():
            if v in done:
                continue
            if v not in dist or (update and dist[u] + w < dist[v]):
                dist[v] = dist[u] + w
                pred[v] = u
    return order, dist, pred


def bellman_ford(adj, source):
    dist = {v: _INF for v in adj}
    dist[source] = 0
    for _ in range(len(adj)):
        for u in adj:
            for v, w in adj[u].items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    return {v: d for v, d in dist.items() if d < _INF}


def _improvements(adj, source):
    """How many times a label that was already set is lowered."""
    dist, done, count = {source: 0}, set(), 0
    for _ in range(len(adj) + 1):
        open_ = [v for v in dist if v not in done]
        if not open_:
            return count
        u = min(open_, key=lambda v: (dist[v], v))
        done.add(u)
        for v, w in adj[u].items():
            if v not in done:
                if v not in dist:
                    dist[v] = dist[u] + w
                elif dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    count += 1
    return count


def _route_count(adj, order, dist, target):
    ways = {order[0]: 1}
    for u in order:
        for v, w in adj[u].items():
            if v in dist and dist[u] + w == dist[v]:
                ways[v] = ways.get(v, 0) + ways.get(u, 0)
    return ways.get(target, 0)


def _path(pred, target):
    out, x = [], target
    while x is not None:
        out.append(x)
        x = pred[x]
    return out[::-1]


def _bfs_order(adj, source):
    seen, order, frontier = {source}, [source], [source]
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        nxt.sort()
        order.extend(nxt)
        frontier = nxt
    return order


def _walk(adj, source):
    """Always move to the nearest unvisited neighbour of the current vertex; when stuck, jump to the first unvisited."""
    order, here = [source], source
    for _ in range(len(adj) + 1):
        options = [(w, v) for v, w in adj[here].items() if v not in order]
        if not options:
            rest = sorted(v for v in adj if v not in order and _reachable_set(adj, source) >= {v})
            if not rest:
                return order
            here = rest[0]
        else:
            here = min(options)[1]
        order.append(here)
    return order


def _reachable_set(adj, source):
    seen, stack = {source}, [source]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


# -- building a graph --------------------------------------------------------------------------------------------------


def _letters(n):
    return [chr(ord("A") + i) for i in range(n)]


def _graph(rng, n, extra, *, directed):
    """A random weighted graph on A.. as ``adj`` and its edge list. Undirected graphs are connected."""
    names = _letters(n)
    adj = {v: {} for v in names}
    edges = []

    def add(u, v):
        if v in adj[u] or (not directed and u in adj[v]):
            return False
        w = rng.randint(1, 9)
        adj[u][v] = w
        if not directed:
            adj[v][u] = w
        edges.append((u, v, w))
        return True

    order = names[:]
    rng.shuffle(order)
    for i in range(1, n):
        a, b = order[i], rng.choice(order[:i])
        if directed and rng.random() < 0.5:
            a, b = b, a
        add(a, b)
    for _ in range(extra):
        for _ in range(20):
            u, v = rng.sample(names, 2)
            if add(u, v):
                break
    edges.sort(key=lambda e: (min(e[0], e[1]) if not directed else e[0], max(e[0], e[1]) if not directed else e[1]))
    return adj, edges


def _edges_text(edges, directed):
    if directed:
        return ", ".join(f"{u}→{v} {w}" for u, v, w in edges)
    return ", ".join(f"{min(u, v)}–{max(u, v)} {w}" for u, v, w in edges)


def _list_text(values):
    return ", ".join(values)


def _ties(dist):
    values = list(dist.values())
    return len(values) - len(set(values))


def _checked(adj, source):
    order, dist, pred = dijkstra(adj, source)
    check(dist == bellman_ford(adj, source), "Dijkstra and Bellman-Ford distances disagree")
    check(order == sorted(dist, key=lambda v: (dist[v], v)), "finalising order is not (distance, label) order")
    return order, dist, pred


def _list_distractors(answer, items):
    return keep_distinct(answer, [(v, tag, _list_text(v)) for v, tag in items if v and len(v) >= 2], key=tuple)


def _single(text, latex, answer, distractors, working):
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        latex=latex,
        answer_kind="ordered_list",
        answer=list(answer),
        answer_text=_list_text(answer),
        working=working,
        distractors=distractors,
    )


def _order_items(adj, source, order):
    reach = [v for v in sorted(adj) if v in set(order)]
    return [
        (dijkstra(adj, source, tie_last=True)[0], _TAGS["tie"]),
        (dijkstra(adj, source, update=False)[0], _TAGS["stale"]),
        (_bfs_order(adj, source), _TAGS["bfs"]),
        (_walk(adj, source), _TAGS["walk"]),
        ([source, *[v for v in reach if v != source]], _TAGS["alpha"]),
    ]


def _trace_working(order, dist):
    return [
        "Finalise the vertex with the smallest tentative distance, then lower the labels of its unfinalised neighbours "
        "wherever the route through it is shorter.",
        "Final distances: " + ", ".join(f"{v} = {dist[v]}" for v in sorted(dist)) + ".",
        f"Sorting by distance, with ties in alphabetical order, gives {_list_text(order)}.",
    ]


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(5000, "a six-vertex graph with distinct distances and a lowered label"):
        adj, edges = _graph(rng, 6, rng.randint(3, 4), directed=False)
        order, dist, _pred = _checked(adj, "A")
        wrong = _order_items(adj, "A", order)
        if _ties(dist) == 0 and _improvements(adj, "A") >= 1 and len(_list_distractors(order, wrong)) >= 3:
            break
    e = _edges_text(edges, False)
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(
            f"A courier network joins towns A to F by roads, with lengths in kilometres: {e}. Run Dijkstra's algorithm from A. "
            f"At each step finalise the unfinalised town with the smallest tentative distance. {_TIE} "
            "List the towns in the order they are finalised, starting with A."
        )
    elif form == 1:
        text, latex = say(
            f"A weighted graph has vertices A to F and edges {e}. Starting at A, apply Dijkstra's algorithm. {_TIE} "
            "Give the order in which the vertices are finalised."
        )
    else:
        text, latex = say(
            f"Trains run between stations A to F, with travel times in minutes: {e}. Apply Dijkstra's algorithm from A, "
            f"permanently labelling the station with the smallest tentative time each step. {_TIE} "
            "List the stations in the order they are permanently labelled."
        )
    return _single(text, latex, order, _list_distractors(order, wrong), _trace_working(order, dist))


# -- medium --------------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(8000, "a seven-vertex graph with a tie and two lowered labels"):
        adj, edges = _graph(rng, 7, rng.randint(4, 5), directed=False)
        order, dist, _pred = _checked(adj, "A")
        wrong = _order_items(adj, "A", order)
        if (
            _ties(dist) >= 1
            and _improvements(adj, "A") >= 2
            and dijkstra(adj, "A", tie_last=True)[0] != order
            and len(_list_distractors(order, wrong)) >= 3
        ):
            break
    e = _edges_text(edges, False)
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(
            f"The road network between towns A to G has these lengths in kilometres: {e}. Use Dijkstra's algorithm from A. "
            f"{_TIE} List the towns in the order they are finalised."
        )
    elif form == 1:
        text, latex = say(
            f"Dijkstra's algorithm is run from A on the graph with vertices A to G and edges {e}. At each step the unfinalised "
            f"vertex with the smallest tentative distance is finalised. {_TIE} In what order are the vertices finalised?"
        )
    else:
        text, latex = say(
            f"A delivery drone can fly between sites A to G along the legs {e} (flight times in minutes). Trace Dijkstra's "
            f"algorithm from A. {_TIE} Write the sites in the order they are finalised."
        )
    return _single(text, latex, order, _list_distractors(order, wrong), _trace_working(order, dist))


# -- hard ----------------------------------------------------------------------------------------------------------


def _hard(rng):
    for _ in attempts(20000, "a directed eight-vertex graph with one unreachable vertex"):
        adj, edges = _graph(rng, 8, rng.randint(5, 7), directed=True)
        source = rng.choice("BCDEF")
        reach = _reachable_set(adj, source)
        if len(reach) != 7:
            continue
        order, dist, _pred = _checked(adj, source)
        wrong = _order_items(adj, source, order)
        if (
            _ties(dist) >= 1
            and _improvements(adj, source) >= 2
            and dijkstra(adj, source, tie_last=True)[0] != order
            and len(_list_distractors(order, wrong)) >= 3
        ):
            break
    missing = next(v for v in adj if v not in reach)
    e = _edges_text(edges, True)
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(
            f"Eight depots A to H are linked by one-way roads, written from → to with the length in kilometres: {e}. "
            f"Run Dijkstra's algorithm from {source}. {_TIE} List the depots in the order they are finalised. "
            "A depot that cannot be reached is never finalised, so leave it out."
        )
    elif form == 1:
        text, latex = say(
            f"A directed graph has vertices A to H and arcs {e} (tail → head, then weight). Apply Dijkstra's algorithm "
            f"from {source}, following arcs in their direction only. {_TIE} Give the order in which the vertices are "
            "finalised, leaving out any vertex that is not reachable."
        )
    else:
        text, latex = say(
            f"Data packets can travel only in the direction shown between servers A to H: {e} (delays in milliseconds). "
            f"Trace Dijkstra's algorithm from server {source}. {_TIE} List the servers in the order they are finalised; "
            "servers the packets cannot reach are left out."
        )
    working = _trace_working(order, dist) + [
        f"Vertex {missing} cannot be reached from {source}, so it is never finalised."
    ]
    return _single(text, latex, order, _list_distractors(order, wrong), working)


@generator(
    "r4j_dijkstra_trace",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4j_dijkstra_trace(rng, tier="easy"):
    """The finalising order of Dijkstra's algorithm; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator("exd_x_r4j_dijkstra_trace", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4j_dijkstra_trace(rng, tier="exam"):
    """Exam chain: the finalising order, a shortest distance, the unique route, then a cost at a rate."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(20000, "a graph with a unique long route to a far vertex"):
        adj, edges = _graph(rng, 7, rng.randint(4, 5), directed=False)
        order, dist, pred = _checked(adj, "A")
        target = order[-1]
        route = _path(pred, target)
        stale_order, stale_dist, stale_pred = dijkstra(adj, "A", update=False)
        fewest = _path(_bfs_pred(adj, "A"), target)
        if (
            len(route) >= 4
            and _route_count(adj, order, dist, target) == 1
            and stale_dist[target] != dist[target]
            and fewest != route
            and len(_list_distractors(order, _order_items(adj, "A", order))) >= 3
        ):
            break
    rate = rng.choice([2, 3, 4, 5])
    e = _edges_text(edges, False)
    form = rng.randrange(3)
    if form == 0:
        stem, stem_tex = say(
            f"A courier company serves towns A to G. The road lengths in kilometres are {e}. Dijkstra's algorithm is run from A. {_TIE}"
        )
    elif form == 1:
        stem, stem_tex = say(
            f"Towns A to G are joined by roads with lengths in kilometres: {e}. A van leaves A. Dijkstra's algorithm is used to "
            f"plan its routes. {_TIE}"
        )
    else:
        stem, stem_tex = say(
            f"A regional freight network has depots A to G and roads {e} (lengths in kilometres). Routes from depot A are found "
            f"with Dijkstra's algorithm. {_TIE}"
        )
    d = dist[target]
    ft_order = order[: order.index(target) + 1]

    def part(prompt, kind, value, text, marks, items, working, follow=None):
        out = dict(
            prompt=prompt,
            answer_kind=kind,
            answer=value,
            answer_text=text,
            marks=marks,
            working=working,
            distractors=items,
        )
        if follow:
            out["follow_through"] = follow
        if kind == "number":
            out["sensible_range"] = (0, None)
        return out

    def nums(answer, pairs):
        return keep_distinct(answer, [(x, tag, str(x)) for x, tag in pairs if x > 0])

    parts = [
        part(
            "List the towns in the order they are finalised.",
            "ordered_list",
            order,
            _list_text(order),
            3,
            _list_distractors(order, _order_items(adj, "A", order)),
            _trace_working(order, dist),
        ),
        part(
            f"What is the shortest distance from A to {target}?",
            "number",
            d,
            str(d),
            2,
            nums(
                d,
                [
                    (sum(adj[u][v] for u, v in zip(fewest, fewest[1:])), _TAGS["fewest"]),
                    (stale_dist[target], _TAGS["stale"]),
                    (dist[pred[target]], _TAGS["last_edge"]),
                    (dist[order[-2]], _TAGS["neighbour"]),
                ],
            ),
            [f"The final label of {target} is {d}."],
        ),
        part(
            f"Write down the shortest route from A to {target}, town by town.",
            "ordered_list",
            route,
            _list_text(route),
            2,
            _list_distractors(
                route,
                [
                    (fewest, _TAGS["fewest"]),
                    (_path(stale_pred, target), _TAGS["stale"]),
                    (route[::-1], _TAGS["reversed"]),
                    (ft_order, _TAGS["order_as_path"]),
                ],
            ),
            [f"Follow the predecessors back from {target}: {_list_text(route)}; this is the only route of length {d}."],
        ),
        part(
            f"The company charges ${rate} for each kilometre travelled. Find the charge for the shortest route to {target}.",
            "number",
            rate * d,
            str(rate * d),
            1,
            nums(
                rate * d,
                [
                    (rate + d, _TAGS["cost_rate_added"]),
                    (d, _TAGS["cost_rate_ignored"]),
                    (rate * (len(route) - 1), _TAGS["cost_per_edge"]),
                    (rate * stale_dist[target], _TAGS["stale"]),
                ],
            ),
            [f"{rate} × {d} = {rate * d} dollars."],
            follow=f"{rate} * part_b",
        ),
    ]
    check(all(len(p["distractors"]) >= 2 for p in parts), "too few distractors in a part")
    return dict(text=stem, latex=stem_tex, parts=parts)


def _bfs_pred(adj, source):
    pred, seen, frontier = {source: None}, {source}, [source]
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    pred[v] = u
                    nxt.append(v)
        frontier = nxt
    return pred
