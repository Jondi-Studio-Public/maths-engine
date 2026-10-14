"""Shared pieces for the graphs-and-networks generators (`snet_*`).

A network here is a simple undirected graph on vertices ``0 .. n-1`` written ``A, B, C, ...`` with its edges as
sorted pairs ``(u, v)``, ``u < v``. Everything is exact integer work on tiny graphs, so each question's answer is
found two ways: the construction that built the graph, and a brute-force or matrix route in this module.

* drawing: ``random_connected`` (a connected simple graph), ``eulerian_graph`` and ``toggle_path`` (graphs with a
  chosen set of odd vertices, built by adding cycles and paths mod 2);
* reading: ``degrees``, ``adjacency``, ``is_connected``, ``classify_route``;
* brute force: ``hamiltonian_cycles``, ``hamiltonian_paths``, ``euler_trail_starts``;
* wording: ``describe_network`` and ``CONTEXTS`` (a town, island, lookout, gallery, suburb or station network);
* distractors: ``int_distractors`` turns ``(value, misconception id)`` pairs into tagged ``Distractor``s.

Networks are given in the question as a list of edges such as ``AB, AC, BD``; none of these skills draws a figure.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations, permutations

from ..answers import Distractor
from ..core import attempts, check, pick_distractors

LETTERS = "ABCDEFGH"

Edge = tuple[int, int]


def norm(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def edge_name(edge: Edge) -> str:
    return LETTERS[edge[0]] + LETTERS[edge[1]]


def edges_text(edges) -> str:
    """``AB, AC, BD``: the edges in alphabetical order."""
    return ", ".join(edge_name(e) for e in sorted(edges))


def route_text(route) -> str:
    """``A-B-C-A``."""
    return "-".join(LETTERS[v] for v in route)


def degrees(n: int, edges) -> list[int]:
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def adjacency(n: int, edges) -> list[list[int]]:
    rows = [[0] * n for _ in range(n)]
    for u, v in edges:
        rows[u][v] = 1
        rows[v][u] = 1
    return rows


def neighbours(n: int, edges) -> list[set[int]]:
    out: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        out[u].add(v)
        out[v].add(u)
    return out


def is_connected(n: int, edges) -> bool:
    """Whether every one of the ``n`` vertices can be reached from vertex 0."""
    nbr = neighbours(n, edges)
    seen = {0}
    queue = deque([0])
    while queue:
        for w in nbr[queue.popleft()]:
            if w not in seen:
                seen.add(w)
                queue.append(w)
    return len(seen) == n


def all_pairs(n: int) -> list[Edge]:
    return list(combinations(range(n), 2))


def random_connected(rng, n: int, m: int) -> list[Edge]:
    """A connected simple graph with ``n`` vertices and ``m`` edges: a random tree, then random extra edges."""
    check(n - 1 <= m <= n * (n - 1) // 2, "no connected simple graph with that many edges")
    order = list(range(n))
    rng.shuffle(order)
    edges = {norm(order[i], order[rng.randrange(i)]) for i in range(1, n)}
    rest = [p for p in all_pairs(n) if p not in edges]
    rng.shuffle(rest)
    edges.update(rest[: m - len(edges)])
    return sorted(edges)


def toggle(edges, extra) -> list[Edge]:
    """The edges with every edge of ``extra`` added if absent and removed if present (adding mod 2)."""
    return sorted(set(edges) ^ {norm(*e) for e in extra})


def cycle_edges(vertices) -> list[Edge]:
    return [norm(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def path_edges(vertices) -> list[Edge]:
    return [norm(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def eulerian_graph(rng, n: int, m_lo: int, m_hi: int) -> list[Edge]:
    """A connected graph on exactly ``n`` vertices, every degree even and non-zero, with ``m_lo..m_hi`` edges.

    Built by adding short cycles mod 2, which keeps every degree even."""
    for _ in attempts(3000, f"a connected even graph on {n} vertices"):
        edges: list[Edge] = []
        for _ in range(rng.randint(2, 4)):
            size = rng.randint(3, min(5, n))
            edges = toggle(edges, cycle_edges(rng.sample(range(n), size)))
        if m_lo <= len(edges) <= m_hi and min(degrees(n, edges)) >= 2 and is_connected(n, edges):
            return edges
    raise AssertionError  # pragma: no cover


def odd_vertices(n: int, edges) -> list[int]:
    return [v for v, d in enumerate(degrees(n, edges)) if d % 2]


def graph_with_odd_vertices(rng, n: int, odd: int, m_lo: int, m_hi: int) -> list[Edge]:
    """A connected graph on ``n`` vertices with exactly ``odd`` odd vertices (0, 2 or 4) and ``m_lo..m_hi`` edges."""
    for _ in attempts(4000, f"a connected graph with {odd} odd vertices"):
        edges = eulerian_graph(rng, n, 4, m_hi)
        for _ in range(odd // 2):
            ends = rng.sample(range(n), 2)
            middle = rng.sample([v for v in range(n) if v not in ends], rng.randint(0, 1))
            edges = toggle(edges, path_edges([ends[0], *middle, ends[1]]))
        if (
            len(odd_vertices(n, edges)) == odd
            and m_lo <= len(edges) <= m_hi
            and min(degrees(n, edges)) >= 1
            and is_connected(n, edges)
        ):
            return edges
    raise AssertionError  # pragma: no cover


# -- brute force ------------------------------------------------------------------------------------------------


def hamiltonian_cycles(n: int, edges) -> list[tuple[int, ...]]:
    """Every Hamiltonian cycle once (a cycle and its reverse are one), each starting at vertex 0."""
    edge_set = set(edges)
    out = []
    for rest in permutations(range(1, n)):
        if rest[0] > rest[-1]:
            continue
        tour = (0, *rest)
        if all(norm(tour[i], tour[(i + 1) % n]) in edge_set for i in range(n)):
            out.append(tour)
    return out


def hamiltonian_paths_from(n: int, edges, start: int) -> list[tuple[int, ...]]:
    """Every Hamiltonian path starting at ``start``."""
    edge_set = set(edges)
    out = []
    others = [v for v in range(n) if v != start]
    for rest in permutations(others):
        tour = (start, *rest)
        if all(norm(tour[i], tour[i + 1]) in edge_set for i in range(n - 1)):
            out.append(tour)
    return out


def euler_trail_exists(n: int, edges) -> tuple[bool, bool]:
    """``(has an Eulerian trail, has an Eulerian circuit)`` found by trying to use every edge exactly once."""
    edge_list = sorted(edges)
    if not edge_list:
        return False, False
    m = len(edge_list)
    nbr: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (u, v) in enumerate(edge_list):
        nbr[u].append((v, k))
        nbr[v].append((u, k))
    used = [False] * m
    trail_found = False
    circuit_found = False

    def walk(at: int, start: int, left: int) -> None:
        nonlocal trail_found, circuit_found
        if left == 0:
            trail_found = True
            if at == start:
                circuit_found = True
            return
        for w, k in nbr[at]:
            if not used[k]:
                used[k] = True
                walk(w, start, left - 1)
                used[k] = False
                if circuit_found:
                    return

    for start in range(n):
        if nbr[start]:
            walk(start, start, m)
        if circuit_found:
            break
    return trail_found, circuit_found


def euler_trail_starts(n: int, edges) -> set[int]:
    """Every vertex an Eulerian trail (open or closed) can start at, found by backtracking."""
    edge_list = sorted(edges)
    m = len(edge_list)
    nbr: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (u, v) in enumerate(edge_list):
        nbr[u].append((v, k))
        nbr[v].append((u, k))
    used = [False] * m

    def walk(at: int, left: int) -> bool:
        if left == 0:
            return True
        for w, k in nbr[at]:
            if not used[k]:
                used[k] = True
                found = walk(w, left - 1)
                used[k] = False
                if found:
                    return True
        return False

    return {s for s in range(n) if nbr[s] and walk(s, m)}


def min_edges_to_add(n: int, edges, goal_odd: int, limit: int = 4) -> int | None:
    """The fewest new edges (each joining two vertices not already joined) that leave the graph connected with exactly
    ``goal_odd`` odd vertices, found by trying every set of new edges in turn."""
    present = set(edges)
    absent = [p for p in all_pairs(n) if p not in present]
    for k in range(limit + 1):
        for extra in combinations(absent, k):
            grown = sorted(present | set(extra))
            if len(odd_vertices(n, grown)) == goal_odd and is_connected(n, grown):
                return k
    return None


# -- routes -----------------------------------------------------------------------------------------------------

#: Route classes, most general first on each side. The open ones end elsewhere; the closed ones end where they start.
OPEN_CLASSES = ("walk", "trail", "path")
CLOSED_CLASSES = ("closed walk", "closed trail", "cycle")
CLASS_OPTION = {
    "walk": "Walk",
    "trail": "Trail",
    "path": "Path",
    "closed walk": "Closed walk",
    "closed trail": "Closed trail (circuit)",
    "cycle": "Cycle",
}


def classify_route(route) -> str:
    """The most specific of walk, trail, path, closed walk, closed trail and cycle for a route (a vertex sequence).

    A route is closed when it ends where it starts. It is a trail when no edge is used twice and a path (or, if closed,
    a cycle) when in addition no vertex is repeated apart from the closing one."""
    route = list(route)
    closed = route[0] == route[-1]
    steps = [norm(route[i], route[i + 1]) for i in range(len(route) - 1)]
    edge_repeat = len(set(steps)) < len(steps)
    inner = route[:-1] if closed else route
    vertex_repeat = len(set(inner)) < len(inner)
    if closed:
        if len(route) <= 1:
            raise ValueError("a closed route needs at least one edge")
        if edge_repeat:
            return "closed walk"
        return "closed trail" if vertex_repeat else "cycle"
    if edge_repeat:
        return "walk"
    return "trail" if vertex_repeat else "path"


def is_walk(route, edges) -> bool:
    edge_set = set(edges)
    return all(norm(route[i], route[i + 1]) in edge_set for i in range(len(route) - 1))


def random_walk(rng, n: int, edges, steps: int):
    nbr = neighbours(n, edges)
    at = rng.randrange(n)
    route = [at]
    for _ in range(steps):
        options = sorted(nbr[at])
        if not options:
            return None
        at = rng.choice(options)
        route.append(at)
    return route


def simple_cycles(n: int, edges) -> list[tuple[int, ...]]:
    """Every simple cycle once, as a vertex tuple starting at its smallest vertex (the direction with the smaller
    second vertex first)."""
    nbr = neighbours(n, edges)
    found: list[tuple[int, ...]] = []

    def extend(path):
        for w in sorted(nbr[path[-1]]):
            if w == path[0] and len(path) >= 3 and path[1] < path[-1]:
                found.append(tuple(path))
            elif w > path[0] and w not in path:
                extend([*path, w])

    for start in range(n):
        extend([start])
    return found


def _rotated(cycle, start, reverse):
    i = cycle.index(start)
    seq = list(cycle[i:] + cycle[:i])
    if reverse:
        seq = [seq[0], *reversed(seq[1:])]
    return seq


def _constructed_route(rng, n: int, edges, wanted: str):
    """A cycle, or a closed trail made of two edge-disjoint cycles that share a vertex, or None if there is none."""
    cycles = simple_cycles(n, edges)
    if wanted == "cycle":
        if not cycles:
            return None
        cyc = rng.choice(cycles)
        route = _rotated(cyc, rng.choice(cyc), rng.random() < 0.5)
        return [*route, route[0]]
    pairs = []
    for c1, c2 in combinations(cycles, 2):
        if not set(cycle_edges_of(c1)) & set(cycle_edges_of(c2)):
            pairs.extend((c1, c2, u) for u in sorted(set(c1) & set(c2)))
    if not pairs:
        return None
    c1, c2, u = rng.choice(pairs)
    first = _rotated(c1, u, rng.random() < 0.5)
    second = _rotated(c2, u, rng.random() < 0.5)
    return [*first, u, *second[1:], u]


def cycle_edges_of(cycle) -> list[Edge]:
    return [norm(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def route_of_class(rng, n: int, edges, wanted: str, lo: int = 2, hi: int = 7):
    """A random route of the wanted class in the graph. Cycles and closed trails are built from the graph's simple
    cycles; the other kinds are found by drawing random walks and keeping the first that fits."""
    if wanted in ("cycle", "closed trail"):
        route = _constructed_route(rng, n, edges, wanted)
        check(route is not None and classify_route(route) == wanted, f"no {wanted} in the network")
        return route
    for _ in attempts(4000, f"a {wanted} in the network"):
        route = random_walk(rng, n, edges, rng.randint(lo, hi))
        if route is not None and classify_route(route) == wanted:
            return route
    raise AssertionError  # pragma: no cover


# -- wording ----------------------------------------------------------------------------------------------------

CONTEXTS = (
    dict(
        vs="towns",
        v="town",
        es="roads",
        e="road",
        travel=True,
        intro="Towns {a} to {z} in a country region are linked by direct roads.",
    ),
    dict(
        vs="islands",
        v="island",
        es="ferry routes",
        e="ferry route",
        travel=True,
        intro="A tour company runs ferries between {n} islands, {a} to {z}.",
    ),
    dict(
        vs="lookouts",
        v="lookout",
        es="walking tracks",
        e="walking track",
        travel=True,
        intro="A national park has {n} lookouts, {a} to {z}, connected by walking tracks.",
    ),
    dict(
        vs="galleries",
        v="gallery",
        es="doorways",
        e="doorway",
        travel=True,
        intro="A museum has {n} galleries, {a} to {z}, joined by doorways.",
    ),
    dict(
        vs="suburbs",
        v="suburb",
        es="bike paths",
        e="bike path",
        travel=True,
        intro="A council has built bike paths between {n} suburbs, {a} to {z}.",
    ),
    dict(
        vs="stations",
        v="station",
        es="rail links",
        e="rail link",
        travel=True,
        intro="A rail network connects {n} stations, {a} to {z}.",
    ),
    dict(
        vs="computers",
        v="computer",
        es="cables",
        e="cable",
        travel=False,
        intro="A school lab has {n} computers, {a} to {z}, joined by network cables.",
    ),
)


def counted(n: int, singular: str, plural: str | None = None) -> str:
    """``1 road``, ``2 roads``: the count and its noun."""
    return f"{n} {singular if n == 1 else plural or singular + 's'}"


def pick_context(rng, travel: bool = False) -> dict:
    pool = [c for c in CONTEXTS if c["travel"]] if travel else list(CONTEXTS)
    return rng.choice(pool)


def describe_network(ctx: dict, n: int, edges, *, listed: str = "") -> str:
    """The sentence pair that introduces a network: what it is, then its edges as pairs of letters."""
    intro = ctx["intro"].format(n=n, a=LETTERS[0], z=LETTERS[n - 1])
    body = listed or edges_text(edges)
    return f"{intro} Each pair of letters names the two {ctx['vs']} one {ctx['e']} joins: {body}."


# -- distractors ------------------------------------------------------------------------------------------------


def int_distractors(answer: int, candidates, n: int = 3, *, lo: int = 0) -> list[Distractor]:
    """Tagged integer distractors from ``(value, misconception id)`` pairs, dropping the answer, repeats, anything
    below ``lo`` and any non-integer value. Order is the order given."""
    seen = {answer}
    pool = []
    for value, tag in candidates:
        if isinstance(value, float):
            if value != int(value):
                continue
            value = int(value)
        if value in seen or value < lo:
            continue
        seen.add(value)
        pool.append(Distractor(str(value), tag, value))
    return pick_distractors(str(answer), pool, n)
