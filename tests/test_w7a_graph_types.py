"""Tests `graph-complete-bipartite-connected` (topic `w7a_graph_types`) and its exam chain `exd_x_w7a_graph_types`.

The oracles read the edge list back out of the question TEXT and recount with their own union-find, subset search and
brute force over colourings, never with the generator's helpers.
"""

from __future__ import annotations

import re
from itertools import combinations
from math import comb

import pytest

import mathgen.generators.w7a_graph_types_build  # noqa: F401  (registers the topics)
from tests import w7a_support as sup

TOPIC = "w7a_graph_types"
EXAM = "exd_x_w7a_graph_types"
SKILL = sup.skill_of("graph-complete-bipartite-connected")


def _edges(text: str):
    body = re.search(r"(?:\d+ edges (?:are )?|joins: )((?:[A-L]{2}(?:, )?)+)", text).group(1)
    return [(ord(t[0]) - 65, ord(t[1]) - 65) for t in re.findall(r"[A-L]{2}", body)]


def _n_vertices(text: str) -> int:
    m = re.search(r"A to ([A-L])", text)
    return ord(m.group(1)) - 64


def _component_of(n, edges, start):
    seen, todo = {start}, [start]
    while todo:
        u = todo.pop()
        for a, b in edges:
            for x, y in ((a, b), (b, a)):
                if x == u and y not in seen:
                    seen.add(y)
                    todo.append(y)
    return seen


def _proper(n, edges):
    return [m for m in range(2**n) if all((m >> a & 1) != (m >> b & 1) for a, b in edges)]


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "number")
    assert "graph-bipartite-assumes-equal-parts" in {m.id for m in SKILL.misconceptions}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier, seeds=100)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def test_easy_keys_by_independent_counting():
    kinds = set()
    for seed in range(150):
        q = sup.single(TOPIC, "easy", seed)
        if "complete graph has" in q.text or "every" in q.text and "exactly once" in q.text:
            n = int(re.search(r"(\d+) (?:vertices|teams|towns|computers|schools)", q.text).group(1))
            assert len(list(combinations(range(n), 2))) == int(q.answer), (seed, q.text)
            kinds.add("complete")
        elif "How many more edges" in q.text:
            n, edges = _n_vertices(q.text), _edges(q.text)
            assert len(set(combinations(range(n), 2)) - set(edges)) == int(q.answer), (seed, q.text)
            kinds.add("missing")
        else:
            n, edges = _n_vertices(q.text), _edges(q.text)
            start = ord(re.search(r"Starting at \w+ ([A-L])", q.text).group(1)) - 65
            assert len(_component_of(n, edges, start)) == int(q.answer), (seed, q.text)
            kinds.add("reach")
    assert kinds == {"complete", "missing", "reach"}


def test_medium_keys():
    kinds = set()
    for seed in range(200):
        q = sup.single(TOPIC, "medium", seed)
        if "set X" in q.text:
            xs = set(re.search(r"vertices ([A-L, ]+) in set X", q.text).group(1).replace(" ", "").split(","))
            ys = set(re.search(r"X and ([A-L, ]+) in set Y", q.text).group(1).replace(" ", "").split(","))
            edges = {(chr(65 + a), chr(65 + b)) for a, b in _edges(q.text)}
            joined = {frozenset(e) for e in edges}
            assert all((e[0] in xs) != (e[1] in xs) for e in edges)
            missing = sum(frozenset((x, y)) not in joined for x in xs for y in ys)
            assert missing == int(q.answer), (seed, q.text)
            kinds.add("gap")
        elif "larger of the two sets" in q.text:
            n, edges = _n_vertices(q.text), _edges(q.text)
            masks = _proper(n, edges)
            assert len(masks) == 2
            ones = bin(masks[0]).count("1")
            assert max(ones, n - ones) == int(q.answer), (seed, q.text)
            kinds.add("larger")
        elif "complement" in q.text:
            n = int(re.search(r"has (\d+) vertices", q.text).group(1))
            degs = [int(d) for d in re.findall(r"has degree (\d+)", q.text)]
            assert len(degs) == n and sum(degs) % 2 == 0
            assert comb(n, 2) - sum(degs) // 2 == int(q.answer), (seed, q.text)
            kinds.add("complement")
        else:
            m = re.search(r"has (\d+) vertices and the other has (\d+)", q.text) or re.search(
                r"(\d+) salespeople and (\d+) clients", q.text
            )
            a, b = int(m.group(1)), int(m.group(2))
            assert sum(b for _ in range(a)) + sum(a for _ in range(b)) == int(q.answer), (seed, q.text)
            kinds.add("kab")
    assert kinds == {"gap", "larger", "complement", "kab"}


def test_hard_keys():
    kinds = set()
    for seed in range(250):
        q = sup.single(TOPIC, "hard", seed)
        t = q.text
        if "components" in t or "separate pieces" in t:
            n = (
                int(re.search(r"(\d+) (?:vertices|computers)", t).group(1))
                if "vertices" in t
                else int(re.search(r"of (\d+) computers", t).group(1))
            )
            c = int(re.search(r"exactly (\d+)", t).group(1))
            best = 0
            for sizes in _partitions(n, c):
                best = max(best, sum(comb(s, 2) for s in sizes))
            assert best == int(q.answer), (seed, t)
            kinds.add("components")
        elif "certain the graph is connected" in t:
            n = int(re.search(r"has (\d+) vertices", t).group(1))
            most = max(comb(k, 2) + comb(n - k, 2) for k in range(1, n))
            assert most + 1 == int(q.answer), (seed, t)
            kinds.add("force")
        elif "still bipartite" in t:
            n, edges = _n_vertices(t), _edges(t)
            masks = _proper(n, edges)
            assert len(masks) == 2
            side = masks[0]
            cross = sum(1 for u, v in combinations(range(n), 2) if (side >> u & 1) != (side >> v & 1))
            assert cross - len(edges) == int(q.answer), (seed, t)
            kinds.add("add")
        else:
            n = int(re.search(r"(\d+) (?:vertices|people)", t).group(1))
            assert max(k * (n - k) for k in range(n + 1)) == int(q.answer), (seed, t)
            kinds.add("bipartite-max")
    assert kinds == {"components", "force", "add", "bipartite-max"}


def _partitions(n, parts, floor=1):
    if parts == 1:
        if n >= floor:
            yield [n]
        return
    for k in range(floor, n // parts + 1):
        for rest in _partitions(n - k, parts - 1, k):
            yield [k, *rest]


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 2, 1, 2], ["number"] * 4)


def test_exam_keys_by_independent_counting():
    for seed in range(100):
        q = sup.exam(EXAM, seed)
        n, edges = _n_vertices(q.stem), _edges(q.stem)
        a, b, c, d = q.parts
        masks = _proper(n, edges)
        assert len(masks) == 2
        ones = bin(masks[0]).count("1")
        big, small = max(ones, n - ones), min(ones, n - ones)
        assert int(a.answer) == 2 * len(edges)
        assert int(b.answer) == big
        assert int(c.answer) == big * small
        assert int(d.answer) == big * small - len(edges)
        assert c.follow_through == f"part_b*({n} - part_b)"
        assert d.follow_through == "part_c - part_a/2"


def test_exam_follow_through_from_a_wrong_larger_part():
    def wrong(q, canon):
        n = len(q.stem) and _n_vertices(q.stem)
        edges = _edges(q.stem)
        b = int(q.parts[1].answer)
        wb = b - 1
        c = wb * (n - wb)
        if c <= 0 or c == int(q.parts[2].answer):
            return None
        return {"b": str(wb), "c": str(c)}

    assert sup.check_follow_through(SKILL, EXAM, "c", wrong) >= 20


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier)
        assert len(spread) >= 8, (tier, spread)
        assert max(spread.values()) <= 12, (tier, spread.most_common(3))
