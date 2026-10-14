"""Exam-tier checks for `shortest-network-path`: shortest route, shortest route around a closure, extra distance."""

import re

from mathgen import make
from tests.exd_x_sc2_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_key_accepted_distractors_rejected,
    check_no_text_defects,
    check_oracle,
    check_variety,
    check_wired,
    text_of,
)

TOPIC = "exd_x_sc2_network_closure"
SKILL = "shortest-network-path"
NODES = "ABCDEF"


def _network(q):
    text = text_of(q)
    roads = re.search(r"kilometres are: (.*?)\. The driver", text).group(1)
    edges = {}
    for u, v, w in re.findall(r"([A-F])([A-F]) (\d+)", roads):
        edges[(u, v)] = int(w)
    closed = re.search(r"The road ([A-F])([A-F]) is closed", text).groups()
    return edges, closed


def _floyd(edges, skip=None):
    """All-pairs shortest distances: an independent route from the generator's Dijkstra."""
    inf = 10**9
    d = {(a, b): (0 if a == b else inf) for a in NODES for b in NODES}
    for (u, v), w in edges.items():
        if (u, v) == skip:
            continue
        d[(u, v)] = d[(v, u)] = min(d[(u, v)], w)
    for k in NODES:
        for i in NODES:
            for j in NODES:
                if d[(i, k)] + d[(k, j)] < d[(i, j)]:
                    d[(i, j)] = d[(i, k)] + d[(k, j)]
    return d[("A", "F")]


def oracle(q):
    edges, closed = _network(q)
    before = _floyd(edges)
    after = _floyd(edges, skip=tuple(closed))
    return {"a": str(before), "b": str(after), "c": str(after - before)}


def wrong_chain(q):
    edges, closed = _network(q)
    before = _floyd(edges)
    after = _floyd(edges, skip=tuple(closed))
    wrong_a = before + 3
    return {"a": str(wrong_a), "b": str(after), "c": str(after - wrong_a)}


def test_variety_and_new_shapes():
    check_variety(TOPIC, lambda s: make("shortest_path", seed=s))


def test_wired():
    check_wired(SKILL, TOPIC)


def test_oracle_agrees():
    check_oracle(TOPIC, oracle)


def test_answer_space():
    check_answer_space(TOPIC)


def test_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, TOPIC)


def test_follow_through_from_wrong_first_distance():
    """A wrong (a) does not change (b); (c) is the difference of the learner's own (b) and (a)."""
    check_follow_through(SKILL, TOPIC, wrong_chain)


def test_closure_always_lengthens_the_trip():
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert int(q.parts[2].answer_text) > 0


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, TOPIC, {"a": "901", "b": "902", "c": "903"})
