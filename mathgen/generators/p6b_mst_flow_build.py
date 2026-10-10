"""Minimal spanning trees and network flow, both senior ``acara-senior-v8-4``, domain ``financial-discrete``,
``answer: number``.

Two generators:

- ``use_minimal_spanning_trees_to_solve_minimal_connector`` (``senior-use-minimal-
  spanning-trees-to-solve-minimal-connector``, ACMGM103) -- minimal spanning tree
  via Kruskal's algorithm. Three tiers: easy (4 towns, 6 edges), medium (5 towns,
  7 edges), hard (6 towns, 9 edges).

- ``solve_small_scale_network_flow_problems_including_the`` (``senior-solve-small-
  scale-network-flow-problems-including-the``, ACMGM109) -- maximum flow via
  Edmonds-Karp and independent check via min-cut brute force. Three tiers: easy
  (single path), medium (two parallel paths), hard (4-node bridge network).

Both use integer arithmetic. Answers are always integers.
"""

from __future__ import annotations

import string
from itertools import combinations

from ..core import check, check_distinct, generator, pick_distractors


# --- Minimal Spanning Tree Generator -----------------------------------------


def _build_graph(rng, n_towns, n_extra_edges, weight_lo, weight_hi):
    """Construct a connected graph with guaranteed spanning path plus extra edges."""
    towns = list(string.ascii_uppercase[:n_towns])
    order = towns[:]
    rng.shuffle(order)
    path_edges = [tuple(sorted((order[i], order[i + 1]))) for i in range(n_towns - 1)]
    all_possible = [tuple(sorted((towns[i], towns[j]))) for i in range(n_towns) for j in range(i + 1, n_towns)]
    used = set(path_edges)
    remaining = [e for e in all_possible if e not in used]
    rng.shuffle(remaining)
    extra_edges = remaining[:n_extra_edges]
    edges = path_edges + extra_edges
    weights = rng.sample(range(weight_lo, weight_hi + 1), len(edges))
    rng.shuffle(edges)  # so the path edges aren't always listed first
    return [(u, v, w) for (u, v), w in zip(edges, weights)]


class UnionFind:
    """Simple union-find with path compression."""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by attaching y's root to x's root."""
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px
            return True
        return False


def _kruskal_mst_weight(towns, edges):
    """Compute MST weight using Kruskal's algorithm."""
    sorted_edges = sorted(edges, key=lambda e: e[2])
    uf = UnionFind(towns)
    mst_weight = 0
    mst_edges = []
    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges.append((u, v, w))
    return mst_weight, mst_edges


def _brute_force_mst_weight(towns, edges):
    """Independent check: brute force over all spanning-tree subsets."""
    best = None
    for subset in combinations(edges, len(towns) - 1):
        # union-find check: does this subset connect all towns with no cycle?
        parent = {t: t for t in towns}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        ok = True
        for u, v, w in subset:
            ru, rv = find(u), find(v)
            if ru == rv:
                ok = False
                break
            parent[ru] = rv
        if not ok:
            continue
        roots = {find(t) for t in towns}
        if len(roots) != 1:
            continue
        total = sum(w for _, _, w in subset)
        if best is None or total < best:
            best = total
    return best


def _verify_mst(towns, edges, mst_weight, mst_edges):
    """Independent check: verify answer against brute-force MST weight."""
    brute_force_weight = _brute_force_mst_weight(towns, edges)
    check(
        mst_weight == brute_force_weight, "MST weight must match the brute-force minimum over all spanning-tree subsets"
    )


@generator(
    "use_minimal_spanning_trees_to_solve_minimal_connector",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def use_minimal_spanning_trees_to_solve_minimal_connector(rng, tier="easy"):
    """Minimal spanning tree problems using Kruskal's algorithm."""

    if tier == "easy":
        n_towns, n_extra_edges = 4, 3
        weight_lo, weight_hi = 1, 20
    elif tier == "medium":
        n_towns, n_extra_edges = 5, 3
        weight_lo, weight_hi = 1, 25
    elif tier == "hard":
        n_towns, n_extra_edges = 6, 4
        weight_lo, weight_hi = 1, 30
    else:
        raise ValueError(f"unknown tier: {tier!r}")

    towns = list(string.ascii_uppercase[:n_towns])
    edges = _build_graph(rng, n_towns, n_extra_edges, weight_lo, weight_hi)

    # Compute MST using Kruskal
    mst_weight, mst_edges = _kruskal_mst_weight(towns, edges)

    # Independent check
    _verify_mst(towns, edges, mst_weight, mst_edges)

    # Build question text
    cable_lines = [f"Cable option: {u}-{v}, cost ${w}." for u, v, w in edges]
    text = (
        "A power company is connecting towns to a single power station by cable. "
        "Every town must be connected to the power station, directly or through other towns. "
        "Each listed cable option may or may not be needed -- you do not have to use every option. "
        + " ".join(cable_lines)
        + " "
        f"What is the minimum total cost, in dollars, to connect all {n_towns} towns "
        "(including the power station) so that every town is connected, directly or through other towns?"
    )

    answer = mst_weight
    answer_text = str(answer)

    # Distractors
    # 1. Sum of all edges (use every edge misconception)
    sum_all = sum(w for _, _, w in edges)
    check_distinct(answer, sum_all, msg="sum of all edges must differ from MST weight")

    # 2. Naive smallest: sum of (n_towns-1) smallest weights without cycle check
    naive_smallest = sum(w for _, _, w in sorted(edges, key=lambda e: e[2])[: n_towns - 1])
    # This may coincide with mst_weight on some seeds; that's fine

    # 3. MST weight minus max edge in MST (leaves one town unconnected)
    max_mst_edge = max(w for _, _, w in mst_edges)
    minus_max = mst_weight - max_mst_edge
    check_distinct(answer, minus_max, msg="MST weight minus max edge must differ from MST weight")

    candidates = [str(sum_all)]
    if naive_smallest != answer:
        candidates.append(str(naive_smallest))
    candidates.append(str(minus_max))
    candidates = pick_distractors(answer_text, candidates)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=[f"Use Kruskal's algorithm to find the minimum spanning tree: ${answer}."],
        distractors=candidates,
    )


# --- Network Flow Generator --------------------------------------------------


def _max_flow_edmonds_karp(nodes, edges, source, sink):
    """Compute maximum flow using Edmonds-Karp (BFS augmenting path)."""
    capacity = {}
    for u, v, c in edges:
        capacity[(u, v)] = capacity.get((u, v), 0) + c
        capacity.setdefault((v, u), 0)
    flow = 0
    while True:
        parent = {source: None}
        queue = [source]
        while queue:
            u = queue.pop(0)
            if u == sink:
                break
            for v in nodes:
                if capacity.get((u, v), 0) > 0 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            break
        path_flow = min(capacity[(parent[v], v)] for v in _path_nodes(parent, sink))
        v = sink
        while parent[v] is not None:
            u = parent[v]
            capacity[(u, v)] -= path_flow
            capacity[(v, u)] += path_flow
            v = u
        flow += path_flow
    return flow


def _path_nodes(parent, sink):
    """Extract nodes on the path from source to sink."""
    v, out = sink, []
    while parent[v] is not None:
        out.append(v)
        v = parent[v]
    return out


def _min_cut_bruteforce(nodes, edges, source, sink):
    """Compute minimum cut capacity by brute force enumeration."""
    free_nodes = [n for n in nodes if n not in (source, sink)]
    best = None
    for mask in range(2 ** len(free_nodes)):
        source_side = {source}
        for i, n in enumerate(free_nodes):
            if mask & (1 << i):
                source_side.add(n)
        cut = sum(c for (u, v, c) in edges if u in source_side and v not in source_side)
        if best is None or cut < best:
            best = cut
    return best


def _verify_flow(nodes, edges, source, sink, max_flow):
    """Independent check: verify max flow equals min cut."""
    min_cut = _min_cut_bruteforce(nodes, edges, source, sink)
    check(max_flow == min_cut, "max flow must equal the minimum cut capacity")


@generator(
    "solve_small_scale_network_flow_problems_including_the",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def solve_small_scale_network_flow_problems_including_the(rng, tier="easy"):
    """Network flow problems using Edmonds-Karp and max-flow-min-cut verification."""

    source, sink = "S", "T"

    if tier == "easy":
        # Single path: S -> A -> T
        nodes = [source, "A", sink]
        cap_sa = rng.randint(4, 20)
        cap_at = rng.randint(4, 20)
        edges = [(source, "A", cap_sa), ("A", sink, cap_at)]
        answer = min(cap_sa, cap_at)

        pipe_text = [
            f"Pipe from {source} to A: capacity {cap_sa} kL per hour.",
            f"Pipe from A to {sink}: capacity {cap_at} kL per hour.",
        ]

    elif tier == "medium":
        # Two parallel paths: S -> A -> T and S -> B -> T
        nodes = [source, "A", "B", sink]
        cap_sa = rng.randint(4, 20)
        cap_at = rng.randint(4, 20)
        cap_sb = rng.randint(4, 20)
        cap_bt = rng.randint(4, 20)
        edges = [(source, "A", cap_sa), ("A", sink, cap_at), (source, "B", cap_sb), ("B", sink, cap_bt)]
        answer = min(cap_sa, cap_at) + min(cap_sb, cap_bt)

        pipe_text = [
            f"Pipe from {source} to A: capacity {cap_sa} kL per hour.",
            f"Pipe from A to {sink}: capacity {cap_at} kL per hour.",
            f"Pipe from {source} to B: capacity {cap_sb} kL per hour.",
            f"Pipe from B to {sink}: capacity {cap_bt} kL per hour.",
        ]

    elif tier == "hard":
        # Bridge network: S -> A -> T, S -> B -> T, A -> B
        nodes = [source, "A", "B", sink]
        cap_sa = rng.randint(4, 20)
        cap_sb = rng.randint(4, 20)
        cap_at = rng.randint(4, 20)
        cap_bt = rng.randint(4, 20)
        cap_ab = rng.randint(4, 20)
        edges = [
            (source, "A", cap_sa),
            (source, "B", cap_sb),
            ("A", sink, cap_at),
            ("B", sink, cap_bt),
            ("A", "B", cap_ab),
        ]

        # Compute via Edmonds-Karp
        answer = _max_flow_edmonds_karp(nodes, edges, source, sink)

        pipe_text = [
            f"Pipe from {source} to A: capacity {cap_sa} kL per hour.",
            f"Pipe from {source} to B: capacity {cap_sb} kL per hour.",
            f"Pipe from A to {sink}: capacity {cap_at} kL per hour.",
            f"Pipe from B to {sink}: capacity {cap_bt} kL per hour.",
            f"Pipe from A to B: capacity {cap_ab} kL per hour.",
        ]

    else:
        raise ValueError(f"unknown tier: {tier!r}")

    # Verify with independent check
    _verify_flow(nodes, edges, source, sink, answer)

    text = (
        "An oil storage tank (source S) is connected to a terminal (sink T) through a network of pipes. "
        "Each pipe has a stated capacity. " + " ".join(pipe_text) + " "
        "What is the maximum flow, in kL per hour, that can be sent from S to T?"
    )

    answer_text = str(answer)

    # Distractors - compute multiple candidate sets to ensure we have at least 1 after filtering
    # Standard distractors:
    sum_s_out = sum(c for u, v, c in edges if u == source)
    sum_t_in = sum(c for u, v, c in edges if v == sink)
    min_capacity = min(c for _, _, c in edges)

    # Fallback distractors (used if standard ones all collide)
    max_capacity = max(c for _, _, c in edges)
    sum_all_edges = sum(c for _, _, c in edges)

    candidates = [str(sum_s_out), str(sum_t_in), str(min_capacity), str(max_capacity), str(sum_all_edges)]
    candidates = pick_distractors(answer_text, candidates)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=[f"Maximum flow from S to T using Edmonds-Karp: {answer} kL per hour."],
        distractors=candidates,
    )
