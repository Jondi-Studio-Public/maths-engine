"""Tests `uc2-dijkstra-trace` (topic `r4j_dijkstra_trace`) and its exam chain.

The oracle reads the graph out of the TEXT, computes every shortest distance with Floyd-Warshall, and orders the
reachable vertices by (distance, label): the finalising order under the stated tie rule.
"""

from __future__ import annotations

import re

import mathgen.generators.r4j_dijkstra_trace_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r4j_support import check_carried, check_parts, check_single

TOPIC = "r4j_dijkstra_trace"
EXAM = "exd_x_r4j_dijkstra_trace"
SKILL = next(s for s in load_skills()[1] if s.id == "uc2-dijkstra-trace")
INF = float("inf")


def _graph(text):
    arcs = {}
    for a, sep, b, w in re.findall(r"([A-H])([–→])([A-H]) (\d+)", text):
        arcs[(a, b)] = int(w)
        if sep == "–":
            arcs[(b, a)] = int(w)
    return arcs


def _source(text):
    m = re.search(r"(?:from|Starting at|leaves) (?:server |depot )?([A-H])\b", text)
    return m.group(1) if m else "A"


def _floyd(arcs):
    nodes = sorted({v for e in arcs for v in e})
    d = {(a, b): (0 if a == b else arcs.get((a, b), INF)) for a in nodes for b in nodes}
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if d[i, k] + d[k, j] < d[i, j]:
                    d[i, j] = d[i, k] + d[k, j]
    return nodes, d


def _order(text):
    arcs = _graph(text)
    src = _source(text)
    nodes, d = _floyd(arcs)
    return src, d, sorted((v for v in nodes if d[src, v] < INF), key=lambda v: (d[src, v], v))


def test_skill_metadata():
    assert list(SKILL.year_level) == ["university"] and "university-year-2" in SKILL.tags


def test_every_tier_matches_floyd_warshall_order():
    for tier, size in (("easy", 6), ("medium", 7), ("hard", 7)):
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            _src, _d, order = _order(q.text)
            assert [v.strip() for v in q.answer_text.split(",")] == order, (tier, seed, q.text)
            assert len(order) == size
            check_single(q, "ordered_list")


def test_hard_has_unreachable_and_other_sources():
    sources = set()
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="hard")
        sources.add(_source(q.text))
        assert ("leav" in q.text or "left out" in q.text) and len(q.answer) == 7
    assert len(sources) >= 3 and "A" not in sources


def test_exam_chain():
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        check_parts(q)
        arcs = _graph(q.text)
        _src, d, order = _order(q.text)
        assert q.parts[0].answer == order
        target = re.search(r"distance from A to ([A-G])", q.parts[1].prompt).group(1)
        assert int(q.parts[1].answer_text) == d["A", target]
        route = q.parts[2].answer
        assert route[0] == "A" and route[-1] == target
        assert sum(arcs[a, b] for a, b in zip(route, route[1:])) == d["A", target]
        rate = int(re.search(r"\$(\d+) for each", q.parts[3].prompt).group(1))
        assert int(q.parts[3].answer_text) == rate * d["A", target]
        check_carried(q, "b", str(d["A", target] + 2), str(rate * (d["A", target] + 2)))
