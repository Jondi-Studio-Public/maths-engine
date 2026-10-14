"""Tests for `graph-planar-faces` (ACMGM081): faces of a planar network and the edge-count test."""

from __future__ import annotations

import re
from itertools import combinations

import pytest
import sympy

from mathgen import make
from tests.sr2a_support import (
    SEEDS,
    TIERS,
    check_follow_through,
    check_question_grading,
    check_tagged,
    distinct_texts,
    questions,
    top_share,
)

TOPIC = "sr2a_planar_faces"
CHAIN = "sr2a_planar_faces_chain"


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=7, difficulty=tier).to_json() == make(TOPIC, seed=7, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 25
    assert top_share(TOPIC, tier) <= 0.3


def _edges(text):
    body = text.split("are: ")[1].split(". ")[0]
    return [(LETTER[a], LETTER[b]) for a, b in re.findall(r"([A-I])([A-I])", body)]


LETTER = {c: i for i, c in enumerate("ABCDEFGHI")}


def test_easy_faces_match_the_listed_network():
    """Recount the regions from the edge list alone: v - e + f = 2 for a connected network drawn without crossings."""
    for q in questions(TOPIC, "easy", range(1, 61)):
        edges = _edges(q.text)
        vertices = {v for e in edges for v in e}
        f = sympy.Symbol("f")
        (faces,) = sympy.solve(sympy.Eq(len(vertices) - len(edges) + f, 2), f)
        assert int(q.answer_text) == faces
        # connected: every vertex reached from the first
        seen, todo = {min(vertices)}, [min(vertices)]
        while todo:
            u = todo.pop()
            for a, b in edges:
                for x, y in ((a, b), (b, a)):
                    if x == u and y not in seen:
                        seen.add(y)
                        todo.append(y)
        assert seen == vertices


def test_medium_excess_matches_the_stated_rule():
    for q in questions(TOPIC, "medium", range(1, 61)):
        if "3v - 6" in q.text:
            n = int(re.match(r"(\d+) ", q.text).group(1))
            expected = len(list(combinations(range(n), 2))) - (3 * n - 6)
        else:
            a, b = (int(x) for x in re.search(r"each of (\d+) houses to each of (\d+) depots", q.text).groups())
            assert f"{a * b} cables" in q.text
            expected = a * b - (2 * (a + b) - 4)
        assert int(q.answer_text) == expected >= 1


def test_hard_variants_all_occur_and_follow_euler():
    seen = set()
    for q in questions(TOPIC, "hard", range(1, 81)):
        if "bounded by exactly" in q.text or "enclosed by exactly" in q.text:
            k = int(re.search(r"exactly (\d+)", q.text).group(1))
            m = int(re.findall(r"(\d+) (?:roads|ferry routes|rail links|bike paths)", q.text)[-1])
            faces = 2 * m // k
            assert faces * k == 2 * m
            assert int(q.answer_text) == m - faces + 2
            seen.add("uniform")
        elif "At most how many more" in q.text or "greatest number of further" in q.text:
            v, e = (int(x) for x in re.search(r"of (\d+) \w+ has (\d+) ", q.text).groups())
            assert int(q.answer_text) == 3 * v - 6 - e
            seen.add("add")
        else:
            v, e = (int(x) for x in re.search(r"of (\d+) \w+ has (\d+) ", q.text).groups())
            assert int(q.answer_text) == 2 * v - 4
            assert e < 3 * v - 6
            seen.add("final")
    assert seen == {"uniform", "add", "final"}


def test_exam_chain_grades_and_follows_through():
    honoured = 0
    for q in questions(CHAIN, "exam"):
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "mcq"]
        check_question_grading(q)
        honoured += check_follow_through(q)
        a, b, c = (int(p.answer_text) for p in q.parts[:3])
        n = int(re.search(r"(?:join|shows|Plans show) (\d+)", q.text).group(1))
        assert a == n * (n - 1) // 2 and b == a - (3 * n - 6) and c == 2 * n - 4
    assert honoured > 20
