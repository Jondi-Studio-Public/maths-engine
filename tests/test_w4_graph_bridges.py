"""Tests `graph-bridges` (topics `w4_graph_bridges` and `w4_graph_bridges_chain`).

The oracles read the network back out of the question's FIGURE (the only place its edges are given) and decide bridges
by removing each edge and testing connectedness with their own search, never from the generator's construction.
"""

from __future__ import annotations

import re
from itertools import combinations

import pytest
from snet_support import check_graded, check_text, skill

from mathgen import figures, make, svg
from mathgen.engine import grade

SK = skill("graph-bridges")
TIERS = ("easy", "medium", "hard")


def network_of(q):
    """(vertex names, edges as frozensets, weights) read from the figure."""
    spec = q.figure
    names = [v["id"] for v in spec["vertices"]]
    edges = [frozenset((e["from"], e["to"])) for e in spec["edges"]]
    weights = {frozenset((e["from"], e["to"])): int(e["label"]) for e in spec["edges"] if e.get("label")}
    return names, edges, weights


def reach(names, edges, start):
    seen, todo = {start}, [start]
    while todo:
        x = todo.pop()
        for e in edges:
            if x in e:
                (y,) = e - {x}
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
    return seen


def bridges(names, edges):
    return [e for e in edges if len(reach(names, [f for f in edges if f != e], names[0])) < len(names)]


def pieces(names, edges):
    left, count = set(names), 0
    while left:
        left -= reach(names, edges, next(iter(sorted(left))))
        count += 1
    return count


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(1, 41):
        check_graded(SK, make("w4_graph_bridges", seed=seed, difficulty=tier))


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_determinism_and_the_figure_is_drawn_not_listed(tier):
    for seed in range(1, 21):
        q = make("w4_graph_bridges", seed=seed, difficulty=tier)
        check_text(q)
        assert figures.validate(q.figure) == [] and svg.figure_svg(q.figure)
        assert not re.search(r"\b[A-H]{2}, [A-H]{2}\b", q.text), "the edges must not be listed in the text"
        names, edges, _ = network_of(q)
        assert len(set(edges)) == len(edges) and reach(names, edges, names[0]) == set(names)  # simple and connected


def test_easy_key_is_the_number_of_bridges_in_the_drawing():
    seen = set()
    for seed in range(1, 61):
        q = make("w4_graph_bridges", seed=seed, difficulty="easy")
        names, edges, _ = network_of(q)
        assert str(len(bridges(names, edges))) == q.answer_text
        seen.add(q.answer_text)
    assert len(seen) >= 4


def test_medium_key_counts_the_piece_holding_a_after_the_named_bridge_goes():
    seen = set()
    for seed in range(1, 61):
        q = make("w4_graph_bridges", seed=seed, difficulty="medium")
        names, edges, _ = network_of(q)
        gone = frozenset(re.search(r"The \w[\w ]* ([A-H]{2}) is removed", q.text).group(1))
        assert gone in edges and gone in bridges(names, edges)
        assert str(len(reach(names, [e for e in edges if e != gone], "A"))) == q.answer_text
        seen.add(q.answer_text)
    assert len(seen) >= 5


def test_hard_key_is_the_best_pair_of_edges_whose_removal_keeps_one_piece():
    seen, trapped = set(), 0
    for seed in range(1, 61):
        q = make("w4_graph_bridges", seed=seed, difficulty="hard")
        names, edges, w = network_of(q)
        ok = [
            (a, b)
            for a, b in combinations(edges, 2)
            if len(reach(names, [e for e in edges if e not in (a, b)], names[0])) == len(names)
        ]
        assert str(max(w[a] + w[b] for a, b in ok)) == q.answer_text
        assert not set(bridges(names, edges)) & {e for pair in ok for e in pair}  # no bridge is ever removed
        loose = sorted((e for e in edges if e not in bridges(names, edges)), key=w.get, reverse=True)[:2]
        if loose not in ok and tuple(loose) not in ok:
            trapped += 1
            assert str(w[loose[0]] + w[loose[1]]) in q.distractors  # the dearest-pair trap is offered
        seen.add(q.answer_text)
    assert len(seen) >= 12 and trapped == 60


def test_figure_alt_does_not_give_the_answer():
    for tier in TIERS:
        for seed in range(1, 21):
            q = make("w4_graph_bridges", seed=seed, difficulty=tier)
            assert q.answer_text not in re.findall(r"\d+|[A-H]{2}", q.figure["alt"])


def test_exam_parts_match_an_independent_derivation():
    seen_a = set()
    for seed in range(1, 41):
        q = make("w4_graph_bridges_chain", seed=seed, difficulty="exam")
        names, edges, _ = network_of(q)
        cut = bridges(names, edges)
        a, b, c, d = (p.answer_text for p in q.parts)
        assert a == str(len(cut)) and b == str(pieces(names, [e for e in edges if e not in cut]))
        new = frozenset(re.search(r"A \w[\w ]* ([A-H]{2}) is to be built", q.text).group(1))
        assert new not in edges
        assert c == str(len(bridges(names, [*edges, new])))
        absent = [frozenset(p) for p in combinations(names, 2) if frozenset(p) not in set(edges)]
        fewest = next(
            k for k in range(0, 5) if any(not bridges(names, [*edges, *extra]) for extra in combinations(absent, k))
        )
        assert d == str(fewest)
        seen_a.add(a)
    assert len(seen_a) >= 2


def test_exam_parts_are_graded_and_part_b_follows_through_from_a_wrong_part_a():
    for seed in range(1, 31):
        q = make("w4_graph_bridges_chain", seed=seed, difficulty="exam")
        check_graded(SK, q)
        canon = {p.label: p.answer_text for p in q.parts}
        wrong_a = int(canon["a"]) + 1
        result = grade(SK, q, {**canon, "a": str(wrong_a), "b": str(wrong_a + 1)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, result)
        assert result.parts["c"].marks == 2 and result.parts["d"].marks == 3
