"""Exam chain of `senior-use-minimal-spanning-trees-to-solve-minimal-connector`: network, re-join cost, new cost."""

from __future__ import annotations

import re
from itertools import combinations

import pytest

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_connector_reroute"
SKILL = S.load_skill("senior-use-minimal-spanning-trees-to-solve-minimal-connector")


def _edges(q):
    block = re.search(r"are: (.*?) \(a\)", q.text).group(1)
    return [(m[0], m[1], int(m[2])) for m in re.findall(r"([A-G])-([A-G]), \$(\d+)\.", block)]


def _spans(combo):
    reach = {t: {t} for t in "ABCDEFG"}
    for u, v, _ in combo:
        merged = reach[u] | reach[v]
        for t in merged:
            reach[t] = merged
    return len(reach["A"]) == 7


def _cheapest(edges):
    best = None
    for combo in combinations(edges, 6):
        if _spans(combo) and (best is None or sum(e[2] for e in combo) < sum(e[2] for e in best)):
            best = combo
    return best


def oracle(q):
    edges = _edges(q)
    failed = re.search(r"The cable ([A-G])-([A-G]) in this network", q.text).groups()
    base_tree = _cheapest(edges)
    base = sum(e[2] for e in base_tree)
    (cut,) = [e for e in base_tree if (e[0], e[1]) == failed]
    new_tree = _cheapest([e for e in edges if e != cut])
    new_total = sum(e[2] for e in new_tree)
    (joiner,) = [e for e in new_tree if e not in base_tree]
    return {"a": str(base), "b": str(joiner[2]), "c": str(new_total)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions[:120], oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_network_cost(questions):
    def chain(q):
        failed = re.search(r"The cable ([A-G])-([A-G]) in this network", q.text).groups()
        w_failed = next(w for u, v, w in _edges(q) if (u, v) == failed)
        wrong_a = int(q.parts[0].distractors[0])
        wrong_b = int(q.parts[1].distractors[0])
        if wrong_a == wrong_b:
            return None
        return {"a": str(wrong_a), "b": str(wrong_b), "c": str(wrong_a - w_failed + wrong_b)}, {"c"}

    S.check_follow_through(SKILL, questions, chain)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "9999", "b": "8888", "c": "7777"})


def test_deterministic():
    S.check_deterministic(TOPIC)
