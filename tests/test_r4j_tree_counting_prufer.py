"""Tests `uc2-tree-counting-prufer` (topic `r4j_tree_counting_prufer`) and its exam chain.

The oracle reads the tree or the code out of the TEXT and uses SymPy's own Prüfer routines (`Prufer.to_prufer`,
`Prufer.to_tree`) as the second route. The counting formulas are checked by enumerating every code of a small case.
"""

from __future__ import annotations

import itertools
import re

from sympy.combinatorics.prufer import Prufer

import mathgen.generators.r4j_tree_counting_prufer_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r4j_support import check_carried, check_parts, check_single

TOPIC = "r4j_tree_counting_prufer"
EXAM = "exd_x_r4j_tree_counting_prufer"
SKILL = next(s for s in load_skills()[1] if s.id == "uc2-tree-counting-prufer")


def _edges(text):
    return [(int(a), int(b)) for a, b in re.findall(r"(\d+)–(\d+)", text)]


def _code(text):
    return [int(v) for v in re.search(r"\(([\d, ]+)\)", text).group(1).split(",")]


def _tree_from_code(code):
    """(n, sorted edges) by SymPy's decoder, converting between 0-based and 1-based labels."""
    edges = Prufer.to_tree([v - 1 for v in code])
    return len(code) + 2, sorted(tuple(sorted((a + 1, b + 1))) for a, b in edges)


def _neighbours(edges, v):
    return sorted({b if a == v else a for a, b in edges if v in (a, b)})


def _ends(text):
    """The two end vertices of the path asked for, in the order asked."""
    m = re.search(
        r"from vertex (\d+) to vertex (\d+)|from vertex (\d+)\s+to vertex (\d+)|get from vertex (\d+) to vertex (\d+)",
        text,
    )
    return tuple(int(g) for g in m.groups() if g is not None)


def _path(edges, u, w):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    stack = [(u, [u])]
    while stack:
        here, path = stack.pop()
        if here == w:
            return path
        stack.extend((nxt, [*path, nxt]) for nxt in adj[here] if nxt not in path)


def test_skill_metadata():
    assert list(SKILL.year_level) == ["university"] and "university-year-2" in SKILL.tags


def test_easy_matches_sympy_encoding():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="easy")
        edges = _edges(q.text)
        n = len(edges) + 1
        want = [v + 1 for v in Prufer.to_prufer([[a - 1, b - 1] for a, b in edges], n)]
        assert [int(v) for v in q.answer_text.split(",")] == want, (seed, q.text)
        assert len(want) == n - 2
        check_single(q, "ordered_list")


def test_medium_and_hard_match_sympy_decoding():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="medium")
        code = _code(q.text)
        n, edges = _tree_from_code(code)
        v = int(re.search(r"vertex (\d+)", q.text).group(1))
        assert [int(t) for t in q.answer_text.split(",")] == _neighbours(edges, v)
        check_single(q, "ordered_list")
        q = make(TOPIC, seed=seed, difficulty="hard")
        code = _code(q.text)
        n, edges = _tree_from_code(code)
        u, w = _ends(q.text)
        assert [int(t) for t in q.answer_text.split(",")] == _path(edges, u, w), (seed, q.text)
        check_single(q, "ordered_list")


def test_exam_chain_and_counting_formulas():
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        check_parts(q)
        code = _code(q.text)
        n = len(code) + 2
        v = int(re.search(r"vertex (\d+)", q.parts[1].prompt).group(1))
        _, edges = _tree_from_code(code)
        leaves = [t for t in range(1, n + 1) if len(_neighbours(edges, t)) == 1]
        assert [int(t) for t in q.parts[0].answer_text.split(",")] == leaves
        assert int(q.parts[1].answer_text) == len(_neighbours(edges, v))
        leaf_count, not_leaf = int(q.parts[2].answer_text), int(q.parts[3].answer_text)
        assert leaf_count == (n - 1) ** (n - 2) and leaf_count + not_leaf == n ** (n - 2)
        check_carried(q, "c", str(leaf_count + 5), str(n ** (n - 2) - leaf_count - 5))


def test_leaf_count_by_enumeration():
    """Every code on 5 or 6 vertices: the trees in which vertex 2 is a leaf are those whose code omits 2."""
    for n in (5, 6):
        total = leaves = 0
        for code in itertools.product(range(1, n + 1), repeat=n - 2):
            _, edges = _tree_from_code(list(code))
            total += 1
            leaves += len(_neighbours(edges, 2)) == 1
        assert total == n ** (n - 2) and leaves == (n - 1) ** (n - 2)
