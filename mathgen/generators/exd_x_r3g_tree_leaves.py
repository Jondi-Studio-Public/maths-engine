"""Exam tier for "count the edges in a finite tree".

The practice tiers give the number of vertices and ask for n - 1. Here the vertex count is not given: the learner
has to find it from the tree's degree structure, then count edges, then use that count, as a discrete-maths exam
would set it. Two stems share one four-part chain:

(a) the number of leaves, from the degrees of the other vertices (a tree on n vertices has n - 1 edges, so its
    degrees sum to 2(n - 1)), or from the child count of a full m-ary tree;
(b) the number of vertices;
(c) the number of edges;
(d) a connected graph on the same vertices has E edges: how many must be deleted to leave a spanning tree.

Built backwards from the structure, and every number is re-derived by the tests from the printed degrees alone.
"""

from __future__ import annotations

import random

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3g_kit import clashes, number_part


def _classes(counts: dict[int, int]) -> str:
    """'2 vertices of degree 3, 1 vertex of degree 4 and 3 vertices of degree 5' (classes with no vertices omitted)."""
    items = [f"{n} {'vertex' if n == 1 else 'vertices'} of degree {d}" for d, n in counts.items() if n]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


_DEGREE_STEMS = (
    "A tree has {classes}. Every other vertex is a leaf (a vertex of degree 1).",
    "The tree T has {classes}. All other vertices of T have degree 1.",
    "A connected graph with no cycles has {classes}; every remaining vertex has degree 1.",
)
_FULL_STEMS = (
    "In a full {kind} tree every internal vertex has exactly {m} children. This tree has {i} internal vertices.",
    "A rooted tree is full {kind}: each of its {i} internal vertices has exactly {m} children, and every other "
    "vertex is a leaf.",
)
_KINDS = {2: "binary", 3: "ternary", 4: "4-ary"}


@generator("exd_x_r3g_tree_leaves", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3g_tree_leaves(rng: random.Random, tier: str = "exam") -> dict:
    """Leaves, vertices and edges of a tree from its degree structure, then the edges to delete from a larger graph."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a tree whose counts are all distinct from the printed numbers"):
        if rng.random() < 0.55:
            counts = {3: rng.randint(1, 6), 4: rng.randint(0, 5), 5: rng.randint(0, 3)}
            if rng.random() < 0.3:
                counts[5] = 0
            high = sum(counts.values())
            leaves = sum((d - 2) * k for d, k in counts.items()) + 2
            stem = rng.choice(_DEGREE_STEMS).format(classes=_classes(counts))
            wrong_leaves = [
                (leaves - 2, "uses-n-not-n-minus-1"),  # degree sum taken as 2n
                (leaves + 2, "uses-n-plus-1"),  # degree sum taken as 2(n + 1)
                (high + 2, "tree-leaves-counted-from-vertices-only"),
                (high, "tree-leaves-counted-from-vertices-only"),
            ]
            non_leaf = high
            working_a = [
                "The degrees sum to twice the number of edges, and a tree on n vertices has n - 1 edges.",
                f"Leaves = {' + '.join(f'{d - 2}x{k}' for d, k in counts.items() if k)} + 2 = {leaves}.",
            ]
        else:
            m = rng.choice((2, 3, 4))
            i = rng.randint(3, 11)
            leaves = (m - 1) * i + 1
            stem = rng.choice(_FULL_STEMS).format(kind=_KINDS[m], m=m, i=i)
            wrong_leaves = [
                ((m - 1) * i, "tree-leaves-forgets-plus-one"),
                (m * i, "tree-leaves-all-children"),
                (i + 1, "tree-leaves-one-more-than-internal"),
                (m * i + 1, "tree-leaves-counts-all-vertices"),
            ]
            non_leaf = i
            working_a = [
                f"Each internal vertex has {m} children, so there are {m * i} child vertices and the root.",
                f"Leaves = {m * i} + 1 - {i} = {leaves}.",
            ]
        vertices = leaves + non_leaf
        edges = vertices - 1
        extra = rng.randint(2, 6)
        total = edges + extra
        pd = (
            f"Extra edges are now added between vertices of the tree, with no new vertices, until the connected "
            f"graph has {total} edges. How many edges must be deleted to leave a spanning tree of this graph?"
        )
        if clashes((leaves, vertices, edges, extra), stem, pd):
            continue
        break

    check(vertices == edges + 1 and total - edges == extra, "tree counts are inconsistent")
    parts = [
        number_part(
            "How many leaves does the tree have?",
            leaves,
            0,
            3,
            working_a,
            wrong_leaves,
            sensible_range=(0, None),
        ),
        number_part(
            "How many vertices does the tree have?",
            vertices,
            0,
            1,
            [f"Vertices = {non_leaf} non-leaf vertices + {leaves} leaves = {vertices}."],
            [
                (non_leaf, "tree-vertices-omit-the-leaves"),
                (leaves, "tree-vertices-leaves-only"),
                (edges, "tree-vertices-reported-as-edges"),
                (vertices + 1, "off-by-one-count"),
            ],
            follow_through=f"part_a + {non_leaf}",
            sensible_range=(0, None),
        ),
        number_part(
            "How many edges does the tree have?",
            edges,
            0,
            1,
            [f"A tree on {vertices} vertices has {vertices} - 1 = {edges} edges."],
            [
                (vertices, "uses-n-not-n-minus-1"),
                (vertices + 1, "uses-n-plus-1"),
                (2 * edges, "forgets-to-halve-degree-sum"),
                (vertices - 2, "tree-edges-n-minus-two"),
            ],
            follow_through="part_b - 1",
            sensible_range=(0, None),
        ),
        number_part(
            pd,
            extra,
            0,
            1,
            [f"A spanning tree has {edges} edges, so {total} - {edges} = {extra} edges are deleted."],
            [
                (total, "reports-given-edges-directly"),
                (total - vertices, "off-by-one-count"),
                (total - vertices + 2, "off-by-one-count"),
            ],
            follow_through=f"{total} - part_c",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
