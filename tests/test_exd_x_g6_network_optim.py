"""Tests for the exam-tier minimal-connector and network-flow sources
(``mathgen/generators/exd_x_g6_network_optim.py``)."""

from __future__ import annotations

import re
from itertools import combinations

import pytest

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

MST = "exd_x_g6_connector_constraint"
FLOW = "exd_x_g6_flow_upgrade"
HARD = {
    MST: "use_minimal_spanning_trees_to_solve_minimal_connector",
    FLOW: "solve_small_scale_network_flow_problems_including_the",
}


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


@pytest.mark.parametrize("topic", [MST, FLOW])
def test_variety_and_new_shapes_over_hard_tier(topic):
    exam = {prompt_shape(_exam(topic, s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[topic], seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert len(exam - hard) >= 3


@pytest.mark.parametrize("topic", [MST, FLOW])
def test_no_text_defects_and_deterministic(topic):
    for s in range(60):
        assert find_text_defects(_exam(topic, s)) == []
    for s in (0, 1, 9, 50):
        assert _exam(topic, s).to_json() == _exam(topic, s).to_json()


def test_both_connector_rules_occur():
    texts = [_exam(MST, s).text for s in range(60)]
    assert any("must be laid" in t for t in texts) and any("cannot be laid" in t for t in texts)


def _cheapest(edges, must=None, never=None):
    towns = sorted({t for u, v, _ in edges for t in (u, v)})
    best = None
    for combo in combinations(edges, len(towns) - 1):
        if never and never in combo:
            continue
        if must and must not in combo:
            continue
        comp = {t: {t} for t in towns}
        ok = True
        for u, v, _w in combo:
            if comp[u] is comp[v]:
                ok = False
                break
            merged = comp[u] | comp[v]
            for t in merged:
                comp[t] = merged
        if ok:
            total = sum(w for *_e, w in combo)
            best = total if best is None else min(best, total)
    return best


def test_connector_oracle():
    bad = []
    for s in range(300):
        q = _exam(MST, s)
        edges = [(u, v, int(w)) for u, v, w in re.findall(r"Cable option: ([A-G])-([A-G]), cost \$(\d+)\.", q.text)]
        assert len(edges) == 11
        u, v = re.search(r"The cable ([A-G])-([A-G]) (?:must|cannot) be laid", q.text).groups()
        e = next(x for x in edges if x[:2] == (u, v))
        if "must be laid" in q.text:
            got = _cheapest(edges, must=e)
        else:
            got = _cheapest(edges, never=e)
        free = _cheapest(edges)
        if got != q.answer or q.answer <= free:
            bad.append((s, got, q.answer))
    assert bad == []


def _flow_by_cut(caps):
    nodes = ["A", "B", "C", "D"]
    best = None
    for mask in range(16):
        side = {"S"} | {n for i, n in enumerate(nodes) if mask >> i & 1}
        cut = sum(c for (a, b), c in caps.items() if a in side and b not in side)
        best = cut if best is None else min(best, cut)
    return best


def test_flow_oracle():
    bad = []
    for s in range(300):
        q = _exam(FLOW, s)
        caps = {(a, b): int(c) for a, b, c in re.findall(r"Pipe from ([A-Z]) to ([A-Z]): capacity (\d+) kL", q.text)}
        a, b, u = re.search(
            r"The pipe from ([A-Z]) to ([A-Z]) is replaced by one whose capacity is (\d+) kL", q.text
        ).groups()
        up = dict(caps)
        up[(a, b)] += int(u)
        rise = _flow_by_cut(up) - _flow_by_cut(caps)
        if rise != q.answer or not 0 < rise < int(u):
            bad.append((s, rise, q.answer))
    assert bad == []


@pytest.mark.parametrize("topic", [MST, FLOW])
def test_grader_accepts_key_and_rejects_distractors(topic):
    grade = GRADERS["number"]
    bad = []
    for s in range(150):
        q = _exam(topic, s)
        if not grade(q, q.answer_text).correct or len(q.distractors) < 2:
            bad.append((s, "key"))
        bad += [(s, d) for d in q.distractors if grade(q, d).correct]
        bad += [(s, w) for w in (q.answer - 1, q.answer + 1) if grade(q, str(w)).correct]
    assert bad == []
