"""
Independent oracle for MST and network-flow skills.
Derives answers from rendered question text without reading the generator.
"""

import re
from itertools import combinations
import pytest
from mathgen import make


def parse_mst_edges(text):
    """Parse 'Cable option: X-Y, cost $W.' sentences into edge list."""
    pattern = r"Cable option:\s+([A-Z])-([A-Z]),\s+cost\s+\$(\d+)\."
    edges = []
    for match in re.finditer(pattern, text):
        u, v, w = match.groups()
        edges.append((u, v, int(w)))
    return edges


def get_nodes_from_edges(edges):
    """Extract all unique nodes from edge list."""
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    return sorted(nodes)


def prims_mst(edges, nodes):
    """
    Prim's algorithm: start from first node, repeatedly add cheapest edge
    connecting a connected node to an unconnected one.
    """
    if not nodes:
        return 0, []

    # Build adjacency list
    graph = {node: [] for node in nodes}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Start from first node
    connected = {nodes[0]}
    unconnected = set(nodes) - connected
    mst_edges = []
    total_cost = 0

    while unconnected:
        # Find cheapest edge from connected to unconnected
        best_edge = None
        best_cost = float("inf")
        best_next = None

        for u in connected:
            for v, w in graph[u]:
                if v in unconnected and w < best_cost:
                    best_cost = w
                    best_edge = (u, v, w)
                    best_next = v

        if best_edge is None:
            break

        mst_edges.append(best_edge)
        total_cost += best_cost
        connected.add(best_next)
        unconnected.remove(best_next)

    return total_cost, mst_edges


def parse_flow_edges(text):
    """Parse 'Pipe from X to Y: capacity C kL per hour.' into directed edges."""
    pattern = r"Pipe from ([A-Z]) to ([A-Z]):\s+capacity\s+(\d+)\s+kL per hour\."
    edges = []
    for match in re.finditer(pattern, text):
        u, v, c = match.groups()
        edges.append((u, v, int(c)))
    return edges


def min_cut_max_flow(edges):
    """
    Brute-force min-cut enumeration: enumerate all subsets of non-source-non-sink
    nodes as the source-side of a cut (source always included, sink never).
    Compute each cut's capacity and return the minimum.
    """
    # Extract all nodes
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)

    # Identify source (S) and sink (T)
    source = "S"
    sink = "T"

    # Non-terminal nodes
    non_terminals = nodes - {source, sink}
    non_terminals = sorted(non_terminals)

    # Build adjacency: {(u,v): capacity}
    graph = {}
    for u, v, c in edges:
        graph[(u, v)] = c

    min_cap = float("inf")

    # Enumerate all subsets of non-terminals as source_side (excluding source, sink)
    for r in range(len(non_terminals) + 1):
        for subset in combinations(non_terminals, r):
            source_side = set(subset) | {source}
            sink_side = nodes - source_side

            # Compute cut capacity: sum of edges from source_side to sink_side
            cut_cap = 0
            for (u, v), cap in graph.items():
                if u in source_side and v in sink_side:
                    cut_cap += cap

            min_cap = min(min_cap, cut_cap)

    return min_cap


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_mst_oracle(difficulty):
    """Test MST oracle against 60 questions per difficulty tier."""
    topic = "use_minimal_spanning_trees_to_solve_minimal_connector"
    failures = []

    for seed in range(60):
        question = make(topic, seed=seed, difficulty=difficulty)

        # Parse edges
        edges = parse_mst_edges(question.text)
        if not edges:
            failures.append((seed, difficulty, "No edges parsed"))
            continue

        # Get nodes
        nodes = get_nodes_from_edges(edges)

        # Compute MST using Prim's algorithm
        mst_cost, _ = prims_mst(edges, nodes)

        # Extract answer
        expected = int(question.answer_text)

        # Check answer matches
        if mst_cost != expected:
            failures.append((seed, difficulty, f"MST cost {mst_cost} != expected {expected}"))

        # Check answer not in distractors
        for distractor in question.distractors:
            if mst_cost == distractor:
                failures.append((seed, difficulty, f"MST cost {mst_cost} matches distractor"))

    assert not failures, f"MST oracle failures:\n" + "\n".join(str(f) for f in failures)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_flow_oracle(difficulty):
    """Test flow oracle against 60 questions per difficulty tier."""
    topic = "solve_small_scale_network_flow_problems_including_the"
    failures = []

    for seed in range(60):
        question = make(topic, seed=seed, difficulty=difficulty)

        # Parse edges
        edges = parse_flow_edges(question.text)
        if not edges:
            failures.append((seed, difficulty, "No edges parsed"))
            continue

        # Compute max flow using min-cut brute force
        max_flow = min_cut_max_flow(edges)

        # Extract answer
        expected = int(question.answer_text)

        # Check answer matches
        if max_flow != expected:
            failures.append((seed, difficulty, f"Max flow {max_flow} != expected {expected}"))

        # Check answer not in distractors
        for distractor in question.distractors:
            if max_flow == distractor:
                failures.append((seed, difficulty, f"Max flow {max_flow} matches distractor"))

    assert not failures, f"Flow oracle failures:\n" + "\n".join(str(f) for f in failures)
