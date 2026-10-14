"""Tests for `network-model-from-situation` (ACMGM079): numbers read off the network a situation describes."""

from __future__ import annotations

import re
from itertools import combinations

import pytest

from mathgen import make
from tests.sr2a_support import (
    TIERS,
    check_follow_through,
    check_question_grading,
    check_tagged,
    distinct_texts,
    questions,
    top_share,
)

TOPIC = "sr2a_network_model"
CHAIN = "sr2a_network_model_chain"


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=5, difficulty=tier).to_json() == make(TOPIC, seed=5, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 25
    assert top_share(TOPIC, tier) <= 0.55


def _routes(text):
    return [[stop.strip() for stop in body.split(" - ")] for body in re.findall(r"(?:Route|Run) \d+: ([^.]+)\.", text)]


def _pairs(routes):
    return {frozenset(r[i : i + 2]) for r in routes for i in range(len(r) - 1)}


def _bfs(pairs, a, b):
    frontier, seen, d = {a}, {a}, 0
    while frontier:
        if b in frontier:
            return d
        frontier = {w for e in pairs for v in frontier if v in e for w in e if w != v} - seen
        seen |= frontier
        d += 1
    return None


def test_easy_matches_the_lists():
    for q in questions(TOPIC, "easy", range(1, 61)):
        lists = re.findall(r"([A-Z][a-z]+): ([A-Za-z, ]+)\.", q.text)
        members = {name: set(friends.split(", ")) for name, friends in lists}
        for name, friends in members.items():
            for f in friends:
                assert name in members[f], "a friendship is listed on one side only"
        edges = sum(len(f) for f in members.values()) // 2
        most = max(len(f) for f in members.values())
        assert int(q.answer_text) in (edges, most)
        if "greatest" in q.text:
            assert int(q.answer_text) == most
        else:
            assert int(q.answer_text) == edges


def test_medium_recounts_from_the_routes():
    seen = set()
    for q in questions(TOPIC, "medium", range(1, 81)):
        routes = _routes(q.text)
        pairs = _pairs(routes)
        if "How many different" in q.text or "how many different" in q.text:
            assert int(q.answer_text) == len(pairs)
            assert sum(len(r) - 1 for r in routes) > len(pairs), "no shared link: nothing to merge"
            seen.add("links")
        else:
            a, b = re.search(r"(?:from|between) ([A-Za-z ]+?) to ([A-Za-z ]+?)(?:,|\.|\?| now)", q.text).groups()
            assert int(q.answer_text) == _bfs(pairs, a, b)
            seen.add("fewest")
    assert seen == {"links", "fewest"}


def test_hard_both_variants():
    seen = set()
    for q in questions(TOPIC, "hard", range(1, 81)):
        routes = _routes(q.text)
        if "new" in q.text and "cancelled" not in q.text:
            seen.add("join")
            names = {s for r in routes for s in r}
            comps = []
            for s in names:
                merged = {s}
                for c in [c for c in comps if any(x in c for x in merged)]:
                    comps.remove(c)
                comps.append(merged)
            # union-find by pairs
            groups = [set(r) for r in routes]
            changed = True
            while changed:
                changed = False
                for g1, g2 in combinations(groups, 2):
                    if g1 & g2:
                        g1 |= g2
                        groups.remove(g2)
                        changed = True
                        break
            off = re.search(r"((?:[A-Z][a-z ]+(?:, )?)+) (?:is|are) not on any route", q.text)
            singles = len(off.group(1).split(", ")) if off else 0
            assert int(q.answer_text) == len(groups) + singles - 1
        else:
            seen.add("cancel")
            cancelled = re.search(r"((?:Route|Run) \d+) is cancelled|After ((?:Route|Run) \d+) is cancelled", q.text)
            label = cancelled.group(1) or cancelled.group(2)
            index = int(label.split()[1]) - 1
            kept = [r for i, r in enumerate(routes) if i != index]
            a, b = re.search(r"(?:from|between) ([A-Za-z ]+?) to ([A-Za-z ]+?)(?: now|\?)", q.text).groups()
            assert int(q.answer_text) == _bfs(_pairs(kept), a, b)
            assert _bfs(_pairs(routes), a, b) < int(q.answer_text)
    assert seen == {"join", "cancel"}


def test_exam_chain_grades_and_follows_through():
    honoured = 0
    for q in questions(CHAIN, "exam"):
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        check_question_grading(q)
        honoured += check_follow_through(q)
        routes = _routes(q.text)
        places = {s for r in routes for s in r}
        a, b, c, d = (int(p.answer_text) for p in q.parts)
        assert a == len(places) and b == len(_pairs(routes)) and c == a * (a - 1) // 2 and d == c - b
    assert honoured > 40
