"""Generates `w7a_graph_types` questions (easy, medium, hard) and the exam chain `exd_x_w7a_graph_types`.

``graph-complete-bipartite-connected`` (``financial-discrete``, answer ``number``): the kinds of simple graph, read from a
network described in words or as an edge list. Every answer is a whole number found by counting, so none can be had by
guessing a name.

* easy: the edges of a complete graph from its vertex count; the edges still missing from a network on the way to a
  complete graph; the number of vertices reachable from one vertex when the network is not connected.
* medium: the edges still missing from a stated bipartite network, the larger part of a bipartite network given only as an
  edge list, the edges of the complement of a network known by its degrees, and the degree sum of a complete bipartite graph.
* hard: the most edges a simple graph with a stated number of components can have, the fewest edges that force a simple graph
  to be connected, the most edges of a simple bipartite graph, and the most edges that can be added to a stated connected
  bipartite network while it stays simple and bipartite.
* exam (``exd_x_w7a_graph_types``): one connected bipartite network as an edge list. (a) the degree sum, (b) the larger part,
  (c) the edges of the complete bipartite graph on those two parts, (d) the edges that can still be added.

Every count is found twice. Components come from a union-find and from a breadth-first search; a bipartition from a
breadth-first colouring and from trying all 2^n colourings; an extreme edge count from its closed form and from a search over
every way to split the vertices.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations, product
from math import comb

from ..core import attempts, check, draw_until, generator, phrase
from ._snet_kit import all_pairs, degrees, neighbours, norm, random_connected
from ._w6a_kit import enough, num_dist as _num_dist, num_fields, part, retry_draw
from ._network_fig import with_network_figure

TOPIC = "w7a_graph_types"
EXAM_TOPIC = "exd_x_w7a_graph_types"

AREA = "graph-types"
MISCONCEPTIONS = [
    (
        "graph-complete-double-count",
        "Counts n(n - 1) for the edges of a complete graph, once from each end of every edge.",
        AREA,
    ),
    (
        "graph-complete-degree-as-edges",
        "Gives the degree n - 1 of one vertex of a complete graph as its number of edges.",
        AREA,
    ),
    (
        "graph-complete-includes-loops",
        "Counts a vertex joined to itself as well, giving n(n + 1)/2 edges for a complete graph.",
        AREA,
    ),
    (
        "graph-complete-one-vertex-short",
        "Counts the edges of the complete graph on one vertex fewer than the network has.",
        AREA,
    ),
    (
        "graph-complete-ignores-existing",
        "Gives all the edges of the complete graph where the ones already present had to be taken off.",
        AREA,
    ),
    (
        "graph-complete-gap-from-degree-sum",
        "Takes the degree sum off the edges of the complete graph instead of the number of edges.",
        AREA,
    ),
    ("graph-count-off-by-one", "Uses the right method for the count but is out by one or two.", AREA),
    ("graph-assumes-connected", "Takes every vertex to be reachable, as if the network were connected.", AREA),
    (
        "graph-reachable-excludes-start",
        "Leaves out the starting vertex when counting the vertices that can be reached.",
        AREA,
    ),
    (
        "graph-reachable-other-component",
        "Gives the size of a different piece of the network from the one that holds the start.",
        AREA,
    ),
    (
        "graph-bipartite-uses-complete-graph",
        "Gives the edges missing from a complete graph where a complete bipartite graph was meant.",
        AREA,
    ),
    (
        "graph-bipartite-sum-of-parts",
        "Adds the sizes of the two parts where their product is the number of edges.",
        AREA,
    ),
    (
        "graph-bipartite-assumes-equal-parts",
        "Takes the two parts of a bipartite network to be equal, or as equal as the vertex count allows.",
        AREA,
    ),
    ("graph-bipartite-smaller-part", "Gives the size of the smaller part where the larger was asked for.", AREA),
    (
        "graph-bipartite-whole-vertex-set",
        "Gives the number of vertices of the whole network as the size of one of its two parts.",
        AREA,
    ),
    (
        "graph-complement-same-edges",
        "Gives the number of edges of the network itself as the number of edges of its complement.",
        AREA,
    ),
    (
        "graph-complement-degree-sum-as-edges",
        "Adds the degrees in the complement and gives that sum as its number of edges.",
        AREA,
    ),
    (
        "graph-complement-ignores-handshake",
        "Takes the degree sum as the number of edges before finding the complement's edges.",
        AREA,
    ),
    (
        "graph-complete-bipartite-as-complete",
        "Uses the edges (a + b)(a + b - 1)/2 of a complete graph for a complete bipartite graph.",
        AREA,
    ),
    (
        "graph-bipartite-degree-sum-as-edges",
        "Gives the number of edges where the sum of the degrees (twice as many) was asked for.",
        AREA,
    ),
    (
        "graph-components-ignored",
        "Gives the edges of the complete graph on all the vertices, ignoring that the network has several pieces.",
        AREA,
    ),
    ("graph-components-minus-edges", "Takes one edge off the complete graph for each extra component.", AREA),
    (
        "graph-components-split-evenly",
        "Spreads the vertices over the components as evenly as possible, which gives fewer edges than one large piece.",
        AREA,
    ),
    (
        "graph-components-one-vertex-short",
        "Leaves the c - 1 single vertices out of the count and so gives the edges of the complete graph on n - c vertices.",
        AREA,
    ),
    (
        "graph-connected-needs-tree",
        "Gives n - 1, the edges of a tree, as enough to force a network to be connected.",
        AREA,
    ),
    (
        "graph-connected-needs-all",
        "Gives all the edges of the complete graph as the number needed to force a network to be connected.",
        AREA,
    ),
    (
        "graph-connected-largest-disconnected",
        "Gives the most edges a disconnected network can have, one fewer than forces connection.",
        AREA,
    ),
    (
        "graph-bipartite-max-all-pairs",
        "Gives the edges of the complete graph as the most a bipartite network can have.",
        AREA,
    ),
    (
        "graph-bipartite-max-floored-parts",
        "Takes the two parts of an odd number of vertices to be equal, floor(n/2) each.",
        AREA,
    ),
    ("graph-bipartite-max-unbalanced", "Takes parts that differ by two, which is not the most unequal product.", AREA),
    ("graph-bipartite-max-rounded-up", "Takes both parts of an odd number of vertices to be the larger size.", AREA),
]

LETTERS = "ABCDEFGHIJKL"


def edges_text(edges) -> str:
    """``AB, AC, BD``: the edges in alphabetical order (this module's graphs reach twelve vertices)."""
    return ", ".join(LETTERS[u] + LETTERS[v] for u, v in sorted(edges))


_CONTEXTS = (
    ("teams", "team", "game"),
    ("towns", "town", "road"),
    ("computers", "computer", "cable"),
    ("schools", "school", "fixture"),
)


def num_dist(correct, candidates, n: int = 4):
    """`num_dist`, leaving out options below 2 (a 0 or a 1 is not a believable count of anything here)."""
    return _num_dist(correct, [(v, t) for v, t in candidates if v is not None and v >= 2], n)


# -- reading a graph two ways ------------------------------------------------------------------------------------------


def _components_union_find(n: int, edges) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        parent[find(u)] = find(v)
    groups: dict[int, list[int]] = {}
    for v in range(n):
        groups.setdefault(find(v), []).append(v)
    return sorted(groups.values())


def _components_bfs(n: int, edges) -> list[list[int]]:
    nbr = neighbours(n, edges)
    seen: set[int] = set()
    out = []
    for s in range(n):
        if s in seen:
            continue
        seen.add(s)
        queue, comp = deque([s]), [s]
        while queue:
            for w in sorted(nbr[queue.popleft()]):
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
                    comp.append(w)
        out.append(sorted(comp))
    return sorted(out)


def _components(n: int, edges) -> list[list[int]]:
    a, b = _components_union_find(n, edges), _components_bfs(n, edges)
    check(a == b, "the two component searches disagree")
    return a


def _colouring(n: int, edges):
    """A breadth-first 2-colouring of every component (the first vertex of each gets 0), or None if there is an odd cycle."""
    nbr = neighbours(n, edges)
    colour: dict[int, int] = {}
    for s in range(n):
        if s in colour:
            continue
        colour[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for w in nbr[u]:
                if w not in colour:
                    colour[w] = 1 - colour[u]
                    queue.append(w)
                elif colour[w] == colour[u]:
                    return None
    return colour


def _proper_colourings(n: int, edges) -> list[int]:
    """Every bit mask (bit v = the side of vertex v) that puts the ends of every edge on different sides."""
    return [m for m in range(2**n) if all(((m >> u) & 1) != ((m >> v) & 1) for u, v in edges)]


def _parts_of_connected_bipartite(n: int, edges) -> tuple[int, int]:
    """(larger part, smaller part) of a connected bipartite graph, from the colouring and checked by trying every colouring."""
    colour = _colouring(n, edges)
    check(colour is not None, "the graph is not bipartite")
    ones = sum(colour.values())
    a, b = max(ones, n - ones), min(ones, n - ones)
    masks = _proper_colourings(n, edges)
    check(len(masks) == 2, "a connected bipartite graph has exactly two colourings")
    sizes = {max(bin(m).count("1"), n - bin(m).count("1")) for m in masks}
    check(sizes == {a}, "the two ways of finding the parts disagree")
    return a, b


def _shuffled_labels(rng, n: int):
    order = list(range(n))
    rng.shuffle(order)
    return order


def _random_connected_bipartite(rng, a: int, b: int, m: int):
    """(n, edges, X, Y) for a connected bipartite graph with parts of a and b vertices and m edges, vertices relabelled at random."""
    check(a + b - 1 <= m <= a * b, "no such connected bipartite graph")
    n = a + b
    xs, ys = list(range(a)), list(range(a, n))
    placed_x, placed_y = [xs[0]], [ys[0]]
    edges = {(xs[0], ys[0])}
    rest = xs[1:] + ys[1:]
    rng.shuffle(rest)
    for v in rest:
        if v in xs:
            u = rng.choice(placed_y)
            placed_x.append(v)
        else:
            u = rng.choice(placed_x)
            placed_y.append(v)
        edges.add((min(u, v), max(u, v)))
    cross = [(x, y) for x in xs for y in ys if (x, y) not in edges]
    rng.shuffle(cross)
    edges.update(cross[: m - len(edges)])
    relabel = _shuffled_labels(rng, n)
    new_edges = sorted(norm(relabel[u], relabel[v]) for u, v in edges)
    return n, new_edges, sorted(relabel[x] for x in xs), sorted(relabel[y] for y in ys)


def _names(vertices) -> str:
    return ", ".join(LETTERS[v] for v in sorted(vertices))


def _context(rng):
    return rng.choice(_CONTEXTS)


# -- easy -----------------------------------------------------------------------------------------------------------------


def _easy_complete(rng):
    vs, v, e = _context(rng)
    n = rng.randint(6, 15)
    answer = comb(n, 2)
    check(answer == sum(degrees(n, all_pairs(n))) // 2, "handshake route disagrees")
    cands = [
        (n * (n - 1), "graph-complete-double-count"),
        (n - 1, "graph-complete-degree-as-edges"),
        (n * (n + 1) // 2, "graph-complete-includes-loops"),
        (comb(n - 1, 2), "graph-complete-one-vertex-short"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    link = {
        "game": "plays one game against",
        "road": "is joined by a direct road to",
        "cable": "is joined by a cable to",
        "fixture": "plays one fixture against",
    }[e]
    stem = phrase(
        rng,
        (
            "In a network of {n} {vs} every {v} {link} each of the others, exactly once. The network is drawn as a graph with a "
            "vertex for each {v} and an edge for each {e}. How many edges does the graph have?",
            "A complete graph has {n} vertices, one for each of {n} {vs}: every pair of {vs} is joined by exactly one {e}. "
            "How many edges does it have?",
        ),
        n=n,
        vs=vs,
        v=v,
        e=e,
        link=link,
    )
    working = [
        f"Each of the {n} vertices has degree {n - 1}, so the degrees add to {n} × {n - 1} = {n * (n - 1)}.",
        f"Every edge is counted at both of its ends, so there are {n * (n - 1)} ÷ 2 = {answer} edges.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _random_simple(rng, n: int, m: int):
    pairs = all_pairs(n)
    rng.shuffle(pairs)
    return sorted(pairs[:m])


def _easy_missing(rng):
    n = rng.randint(6, 9)
    total = comb(n, 2)
    m = rng.randint(max(n, total - 14), total - 3)
    edges = _random_simple(rng, n, m)
    answer = total - m
    check(answer == len([p for p in all_pairs(n) if p not in set(edges)]), "missing-edge count disagrees")
    cands = [
        (total, "graph-complete-ignores-existing"),
        (n * (n - 1) - m, "graph-complete-double-count"),
        (total - 2 * m if total - 2 * m > 0 else None, "graph-complete-gap-from-degree-sum"),
        (comb(n - 1, 2) - m if comb(n - 1, 2) - m > 0 else None, "graph-complete-one-vertex-short"),
        (m, "graph-complete-ignores-existing"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    _, v, e = _context(rng)
    vs = {"team": "teams", "town": "towns", "computer": "computers", "school": "schools"}[v]
    stem = (
        f"A simple graph has {n} vertices, A to {LETTERS[n - 1]}, one for each of {n} {vs}. Its {m} edges are "
        f"{edges_text(edges)}, each pair of letters naming the two {vs} one {e} joins. "
        "How many more edges must be added for the graph to become a complete graph?"
    )
    working = [
        f"A complete graph on {n} vertices has {n} × {n - 1} ÷ 2 = {total} edges.",
        f"The graph already has {m}, so {total} - {m} = {answer} are missing.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _graph_with_components(rng, sizes):
    """(n, edges): a graph whose components have exactly these sizes, vertices relabelled at random."""
    n = sum(sizes)
    relabel = _shuffled_labels(rng, n)
    edges, base = [], 0
    for k in sizes:
        if k >= 2:
            m = rng.randint(k - 1, min(comb(k, 2), k + 2))
            edges += [(relabel[base + u], relabel[base + v]) for u, v in random_connected(rng, k, m)]
        base += k
    return n, sorted(norm(u, v) for u, v in edges), relabel


def _easy_reach(rng):
    def draw(r):
        c = r.choice((2, 3))
        return [r.randint(1, 8) for _ in range(c)]

    sizes = draw_until(
        rng,
        draw,
        lambda z: 9 <= sum(z) <= 12 and sum(k >= 2 for k in z) >= 2,
        what="component sizes for a split network",
    )
    n, edges, relabel = _graph_with_components(rng, sizes)
    comps = _components(n, edges)
    check(sorted(len(c) for c in comps) == sorted(sizes), "components do not match the sizes drawn")
    target = draw_until(
        rng,
        lambda r: r.randrange(n),
        lambda s: len(next(c for c in comps if s in c)) >= 2,
        what="a start vertex in a larger piece",
    )
    mine = next(c for c in comps if target in c)
    answer = len(mine)
    others = sorted({len(c) for c in comps if c != mine and len(c) != answer}, reverse=True)
    cands = [(n, "graph-assumes-connected"), (answer - 1, "graph-reachable-excludes-start")]
    cands += [(k, "graph-reachable-other-component") for k in others]
    cands.append((len(comps), "graph-assumes-connected"))
    dist = num_dist(answer, cands)
    enough(dist, 3)
    _, v, e = _context(rng)
    vs = {"team": "teams", "town": "towns", "computer": "computers", "school": "schools"}[v]
    stem = (
        f"A network joins {n} {vs}, {LETTERS[0]} to {LETTERS[n - 1]}. Each pair of letters names the two {vs} one {e} joins: "
        f"{edges_text(edges)}. Starting at {v} {LETTERS[target]} and following {es_of(e)}, how many {vs} can be reached, "
        f"counting {LETTERS[target]} itself?"
    )
    working = [
        f"Follow the edges out from {LETTERS[target]} until nothing new is reached: {_names(mine)}.",
        f"That is {answer} {vs}; the network has {len(comps)} separate pieces, so it is not connected.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def es_of(e: str) -> str:
    return {"game": "games", "road": "roads", "cable": "cables", "fixture": "fixtures"}[e]


# -- medium ---------------------------------------------------------------------------------------------------------------


def _medium_gap(rng):
    a, b = rng.randint(3, 6), rng.randint(3, 6)
    full = a * b
    m = rng.randint(a + b - 1, full - 2)
    n, edges, xs, ys = _random_connected_bipartite(rng, a, b, m)
    present = set(edges)
    answer = len([1 for x in xs for y in ys if norm(x, y) not in present])
    check(answer == full - m, "bipartite gap disagrees")
    total = comb(n, 2)
    cands = [
        (total - m, "graph-bipartite-uses-complete-graph"),
        (full, "graph-complete-ignores-existing"),
        (a + b - m if a + b - m > 0 else None, "graph-bipartite-sum-of-parts"),
        (full - 2 * m if full - 2 * m > 0 else None, "graph-complete-gap-from-degree-sum"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    stem = (
        f"A bipartite graph has the vertices {_names(xs)} in set X and {_names(ys)} in set Y. Its {m} edges are "
        f"{edges_text(edges)}; every one joins a vertex of X to a vertex of Y. How many more edges must be added for it to become "
        "a complete bipartite graph on the same two sets?"
    )
    working = [
        f"A complete bipartite graph joins each of the {a} vertices of X to each of the {b} of Y: {a} × {b} = {full} edges.",
        f"There are {m} already, so {full} - {m} = {answer} are missing.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _medium_larger_part(rng):
    for _ in attempts(200, "unequal parts of a connected bipartite graph"):
        a, b = rng.randint(3, 6), rng.randint(2, 4)
        if a - b >= 1 and a + b >= 7:
            break
    m = rng.randint(a + b - 1, min(a * b, a + b + 3))
    n, edges, xs, ys = _random_connected_bipartite(rng, a, b, m)
    big, small = _parts_of_connected_bipartite(n, edges)
    check((big, small) == (a, b), "the parts found are not the parts built")
    cands = [
        ((n + 1) // 2, "graph-bipartite-assumes-equal-parts"),
        (small, "graph-bipartite-smaller-part"),
        (n, "graph-bipartite-whole-vertex-set"),
        (big + 1, "graph-bipartite-assumes-equal-parts"),
    ]
    dist = num_dist(big, cands)
    enough(dist, 3)
    _, v, e = _context(rng)
    stem = (
        f"A connected graph has the {n} vertices A to {LETTERS[n - 1]} and the {m} edges {edges_text(edges)}. It is bipartite: "
        "its vertices can be split into two sets so that every edge joins a vertex of one set to a vertex of the other. "
        "How many vertices are in the larger of the two sets?"
    )
    working = [
        f"Colour {LETTERS[0]} with the first colour, then each neighbour of a coloured vertex with the other colour, and continue.",
        f"Because the graph is connected the two sets are forced: they have {big} and {small} vertices, so the larger has {big}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(big), answer_kind="number")


def _medium_complement(rng):
    n = rng.randint(6, 9)
    total = comb(n, 2)
    m = rng.randint(n, total - 4)
    edges = _random_simple(rng, n, m)
    degs = degrees(n, edges)
    answer = total - m
    check(sum(n - 1 - d for d in degs) == 2 * answer, "the complement's degrees disagree")
    cands = [
        (m, "graph-complement-same-edges"),
        (total - sum(degs), "graph-complement-ignores-handshake"),
        (n * (n - 1) - m, "graph-complement-degree-sum-as-edges"),
        (total - m + 1, "graph-count-off-by-one"),
    ]
    dist = num_dist(answer, [(v if v > 0 else None, t) for v, t in cands])
    enough(dist, 3)
    listing = ", ".join(f"{LETTERS[i]} has degree {d}" for i, d in enumerate(degs))
    stem = (
        f"A simple graph G has {n} vertices, and {listing}. The complement of G has an edge between two vertices exactly "
        "when G does not. How many edges does the complement of G have?"
    )
    working = [
        f"The degrees of G add to {sum(degs)}, so G has {sum(degs)} ÷ 2 = {m} edges.",
        f"A complete graph on {n} vertices has {total} edges, so the complement has {total} - {m} = {answer}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _medium_kab(rng):
    a, b = rng.sample(range(3, 10), 2)
    answer = 2 * a * b
    check(answer == sum(degrees(a + b, [(x, a + y) for x in range(a) for y in range(b)])), "degree sum disagrees")
    cands = [
        (a * b, "graph-bipartite-degree-sum-as-edges"),
        ((a + b) * (a + b - 1), "graph-complete-bipartite-as-complete"),
        ((a + b) * (a + b - 1) // 2, "graph-complete-bipartite-as-complete"),
        (a * b + a + b, "graph-bipartite-sum-of-parts"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    stem = phrase(
        rng,
        (
            "In a complete bipartite graph one set has {a} vertices and the other has {b}, and every vertex of one set is "
            "joined to every vertex of the other. What is the sum of the degrees of all {n} vertices?",
            "A company has {a} salespeople and {b} clients, and every salesperson has exactly one contract with every client. "
            "Drawn as a graph with an edge for each contract, what is the sum of the degrees of all the vertices?",
        ),
        a=a,
        b=b,
        n=a + b,
    )
    working = [
        f"Each of the {a} vertices in the first set has degree {b}, and each of the {b} in the second has degree {a}.",
        f"The sum is {a} × {b} + {b} × {a} = {answer}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


# -- hard -----------------------------------------------------------------------------------------------------------------


def _best_split(n: int, c: int) -> int:
    """The most edges among simple graphs on n vertices with exactly c components, by trying every split into c pieces."""
    best = 0

    def rec(left, pieces, floor, total):
        nonlocal best
        if pieces == 1:
            if left >= floor:
                best = max(best, total + comb(left, 2))
            return
        for k in range(floor, left // pieces + 1):
            rec(left - k, pieces - 1, k, total + comb(k, 2))

    rec(n, c, 1, 0)
    return best


def _hard_components(rng):
    n, c = rng.randint(8, 14), rng.randint(2, 4)
    answer = comb(n - c + 1, 2)
    check(answer == _best_split(n, c), "closed form and search for the most edges disagree")
    base, extra = divmod(n, c)
    even = extra * comb(base + 1, 2) + (c - extra) * comb(base, 2)
    cands = [
        (comb(n, 2), "graph-components-ignored"),
        (comb(n, 2) - (c - 1), "graph-components-minus-edges"),
        (even, "graph-components-split-evenly"),
        (comb(n - c, 2), "graph-components-one-vertex-short"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    stem = phrase(
        rng,
        (
            "A simple graph has {n} vertices and exactly {c} connected components. What is the largest number of edges it can have?",
            "A network of {n} {vs} is made of exactly {c} separate pieces, with no two {vs} in different pieces joined, and at most "
            "one {e} between any two {vs}. What is the greatest number of {es} the network can have?",
        ),
        n=n,
        c=c,
        vs="computers",
        e="cable",
        es="cables",
    )
    working = [
        f"To have as many edges as possible, make {c - 1} of the components single vertices and put the other {n - c + 1} vertices in one complete graph.",
        f"That graph has {n - c + 1} × {n - c} ÷ 2 = {answer} edges, and no other split of the vertices gives more.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _hard_force_connected(rng):
    n = rng.randint(7, 15)
    most_disconnected = max(comb(k, 2) + comb(n - k, 2) for k in range(1, n))
    answer = most_disconnected + 1
    check(answer == comb(n - 1, 2) + 1, "closed form and search disagree")
    cands = [
        (most_disconnected, "graph-connected-largest-disconnected"),
        (n - 1, "graph-connected-needs-tree"),
        (comb(n, 2), "graph-connected-needs-all"),
        (comb(n - 1, 2) + 2, "graph-count-off-by-one"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    stem = (
        f"A simple graph has {n} vertices. What is the smallest number of edges that makes it certain the graph is connected, "
        "however the edges are placed?"
    )
    working = [
        f"A disconnected graph has at most all the edges among {n - 1} vertices, with the last vertex left alone: {comb(n - 1, 2)} edges.",
        f"One more edge than that, {answer}, cannot be placed without joining every vertex up.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _hard_max_bipartite(rng):
    n = rng.randint(7, 17)
    answer = max(k * (n - k) for k in range(n + 1))
    check(answer == (n * n) // 4, "closed form and search disagree")
    lo = n // 2
    cands = [
        (comb(n, 2), "graph-bipartite-max-all-pairs"),
        (lo * lo, "graph-bipartite-max-floored-parts"),
        ((n + 1) // 2 * ((n + 1) // 2), "graph-bipartite-max-rounded-up"),
        ((n // 2 - 1) * (n - n // 2 + 1), "graph-bipartite-max-unbalanced"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    stem = phrase(
        rng,
        (
            "A simple graph with {n} vertices is bipartite. What is the largest number of edges it can have?",
            "{n} people are split into two groups, and each pair of people, one from each group, may be joined by at most one link; "
            "nobody is linked to anyone in their own group. Over all ways to choose the groups, what is the greatest number of links?",
        ),
        n=n,
    )
    working = [
        f"If the groups have k and {n} - k people there are at most k({n} - k) links, which is largest when the groups are as equal as possible.",
        f"With groups of {n // 2} and {n - n // 2} that is {answer}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _hard_add_bipartite(rng):
    for _ in attempts(200, "unequal parts"):
        a, b = rng.randint(2, 6), rng.randint(3, 6)
        if abs(a - b) >= 2 and a + b >= 7:
            break
    m = rng.randint(a + b - 1, min(a * b - 3, a + b + 3))
    n, edges, xs, ys = _random_connected_bipartite(rng, a, b, m)
    big, small = _parts_of_connected_bipartite(n, edges)
    answer = big * small - m
    present = set(edges)
    addable = 0
    for u, v in all_pairs(n):
        if (u, v) not in present and _colouring(n, [*edges, (u, v)]) is not None:
            addable += 1
    check(addable == answer, "counting the edges that keep the graph bipartite disagrees with the parts")
    balanced = (n * n) // 4
    cands = [
        (comb(n, 2) - m, "graph-bipartite-uses-complete-graph"),
        (big * small, "graph-complete-ignores-existing"),
        (balanced - m, "graph-bipartite-assumes-equal-parts"),
        (n * (n - 1) - m, "graph-complete-double-count"),
    ]
    dist = num_dist(answer, cands)
    enough(dist, 3)
    stem = (
        f"A connected simple graph has the {n} vertices A to {LETTERS[n - 1]} and the {m} edges {edges_text(edges)}. It is bipartite. "
        "What is the largest number of new edges that can be added so that the graph is still simple and still bipartite?"
    )
    working = [
        f"Because the graph is connected, its two sets are forced, with {big} and {small} vertices. A new edge keeps the graph bipartite only if it joins the two sets.",
        f"There are {big} × {small} = {big * small} such edges in all and {m} are already there, so {answer} can be added.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


_EASY = (_easy_complete, _easy_missing, _easy_reach)
_MEDIUM = (_medium_gap, _medium_larger_part, _medium_complement, _medium_kab)
_HARD = (_hard_components, _hard_force_connected, _hard_max_bipartite, _hard_add_bipartite)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_network_figure
def w7a_graph_types(rng, tier="easy"):
    """Complete, connected and bipartite graphs from a description or an edge list; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if not (pool is not None):
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -----------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
@with_network_figure
def exd_x_w7a_graph_types(rng, tier="exam"):
    """Exam chain on a connected bipartite network: degree sum, larger part, the complete bipartite graph, edges still addable."""
    if not (tier == "exam"):
        raise ValueError(f"unknown tier {tier!r}")

    def build(rng):
        for _ in attempts(200, "unequal parts"):
            a, b = rng.randint(3, 10), rng.randint(2, 7)
            if a - b >= 2 and 7 <= a + b <= len(LETTERS) and a * b - 3 >= a + b + 1:
                break
        m = rng.randint(a + b - 1, min(a * b - 3, a + b + 12))
        n, edges, xs, ys = _random_connected_bipartite(rng, a, b, m)
        big, small = _parts_of_connected_bipartite(n, edges)
        check(sum(degrees(n, edges)) == 2 * m, "handshake")
        full = big * small
        gap = full - m
        check(gap > 0, "nothing to add")
        a_dist = num_dist(
            2 * m,
            [
                (m, "net-edges-as-degree-sum"),
                (m + 1, "graph-count-off-by-one"),
                (m // 2 if m % 2 == 0 else None, "net-halved-instead-of-doubled"),
                (2 * m - 1, "graph-count-off-by-one"),
            ],
            3,
        )
        b_dist = num_dist(
            big,
            [
                ((n + 1) // 2, "graph-bipartite-assumes-equal-parts"),
                (small, "graph-bipartite-smaller-part"),
                (n, "graph-bipartite-whole-vertex-set"),
            ],
            3,
        )
        c_dist = num_dist(
            full,
            [
                (comb(n, 2), "graph-bipartite-uses-complete-graph"),
                (n, "graph-bipartite-sum-of-parts"),
                ((n * n) // 4, "graph-bipartite-assumes-equal-parts"),
                (full - m, "graph-complete-ignores-existing"),
            ],
            3,
        )
        d_dist = num_dist(
            gap,
            [
                (comb(n, 2) - m, "graph-bipartite-uses-complete-graph"),
                (full, "graph-complete-ignores-existing"),
                (full - 2 * m if full > 2 * m else None, "graph-complete-gap-from-degree-sum"),
                (gap + 1, "graph-count-off-by-one"),
            ],
            3,
        )
        for d in (a_dist, b_dist, c_dist, d_dist):
            enough(d, 2)
        return n, m, edges, big, small, full, gap, a_dist, b_dist, c_dist, d_dist

    n, m, edges, big, small, full, gap, a_dist, b_dist, c_dist, d_dist = retry_draw(rng, build)
    stem = (
        f"A connected simple graph has the {n} vertices A to {LETTERS[n - 1]} and the {m} edges {edges_text(edges)}. "
        "The graph is bipartite: its vertices split into two sets so that every edge joins the two sets."
    )
    return dict(
        text=stem,
        parts=[
            part(
                "What is the sum of the degrees of all the vertices?",
                "number",
                num_fields(2 * m),
                1,
                [f"Every edge adds 1 to the degree at each of its two ends: 2 × {m} = {2 * m}."],
                a_dist,
                sensible_range=(0, None),
            ),
            part(
                "How many vertices are in the larger of the two sets?",
                "number",
                num_fields(big),
                2,
                [
                    f"Colour the vertices alternately from {LETTERS[0]}; the connected graph forces sets of {big} and {small}."
                ],
                b_dist,
                sensible_range=(0, None),
            ),
            part(
                "How many edges has the complete bipartite graph on these same two sets?",
                "number",
                num_fields(full),
                1,
                [f"{big} × {small} = {full}, taking the smaller set as {n} - part (b)."],
                c_dist,
                follow_through=f"part_b*({n} - part_b)",
                sensible_range=(0, None),
            ),
            part(
                "What is the largest number of new edges that can be added with the graph staying simple and bipartite?",
                "number",
                num_fields(gap),
                2,
                [f"Only edges between the two sets are allowed: {full} - {m} = {gap}."],
                d_dist,
                follow_through=f"part_c - part_a/2",
                sensible_range=(0, None),
            ),
        ],
    )
