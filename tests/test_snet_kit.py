"""Tests the shared graph helpers behind the `snet_*` generators on graphs whose answers are known by hand."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from mathgen.generators._snet_kit import (
    classify_route,
    degrees,
    eulerian_graph,
    euler_trail_exists,
    euler_trail_starts,
    graph_with_odd_vertices,
    hamiltonian_cycles,
    hamiltonian_paths_from,
    is_connected,
    min_edges_to_add,
    odd_vertices,
    random_connected,
    simple_cycles,
)

K4 = list(combinations(range(4), 2))
K5 = list(combinations(range(5), 2))


@pytest.mark.parametrize(
    "route, kind",
    [
        ([0, 1, 2], "path"),
        ([0, 1, 2, 0, 3], "trail"),
        ([0, 1, 0], "closed walk"),
        ([0, 1, 2, 1, 3], "walk"),
        ([0, 1, 2, 0], "cycle"),
        ([0, 1, 2, 0, 3, 4, 0], "closed trail"),
        ([0, 1, 2, 0, 1, 3, 0], "closed walk"),
        ([0, 1], "path"),
    ],
)
def test_classify_route(route, kind):
    assert classify_route(route) == kind


def test_euler_search_on_known_graphs():
    assert euler_trail_exists(3, [(0, 1), (1, 2), (0, 2)]) == (True, True)  # a triangle
    assert euler_trail_exists(3, [(0, 1), (1, 2)]) == (True, False)  # a path
    assert euler_trail_exists(4, K4) == (False, False)  # four odd vertices
    assert euler_trail_exists(5, K5) == (True, True)  # every degree 4
    assert euler_trail_exists(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]) == (False, False)  # two pieces
    assert euler_trail_starts(3, [(0, 1), (1, 2)]) == {0, 2}
    assert euler_trail_starts(3, [(0, 1), (1, 2), (0, 2)]) == {0, 1, 2}


def test_hamiltonian_counts():
    assert len(hamiltonian_cycles(4, K4)) == 3
    assert len(hamiltonian_cycles(5, K5)) == 12
    assert len(hamiltonian_cycles(4, [(0, 1), (1, 2), (2, 3)])) == 0
    assert len(hamiltonian_paths_from(4, K4, 0)) == 6
    assert len(hamiltonian_paths_from(4, [(0, 1), (1, 2), (2, 3)], 0)) == 1
    assert len(simple_cycles(4, K4)) == 7  # four triangles and three 4-cycles


def test_min_edges_to_add():
    assert min_edges_to_add(4, [(0, 1), (1, 2), (2, 3)], 0) == 1  # a path needs one edge to close it
    assert min_edges_to_add(4, K4, 2) is None  # K4 already has every edge, so nothing can be added


def test_graph_builders_meet_their_contracts():
    rng = random.Random(7)
    for _ in range(60):
        n = rng.randint(5, 7)
        m = rng.randint(n, n + 4)
        g = random_connected(rng, n, m)
        assert len(g) == m and is_connected(n, g)
        e = eulerian_graph(rng, n, 5, 10)
        assert is_connected(n, e) and not odd_vertices(n, e) and 5 <= len(e) <= 10
        for odd in (0, 2, 4):
            h = graph_with_odd_vertices(rng, n, odd, 6, 10)
            assert len(odd_vertices(n, h)) == odd and is_connected(n, h) and min(degrees(n, h)) >= 1
