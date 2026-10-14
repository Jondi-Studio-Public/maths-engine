"""Shared helpers for the graphs-and-networks tests (`test_snet_*.py`).

`network_from_text` reads a network back out of the question TEXT (the edge list printed after "joins:"), so every
oracle re-derives its key from what a learner sees and never from the generator's own construction.
"""

from __future__ import annotations

import re
from itertools import permutations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

LETTERS = "ABCDEFGH"
_EDGE_LIST = re.compile(r"joins: ((?:[A-H]{2}(?:, )?)+)\.")
_RANGE = re.compile(r"\bA to ([A-H])\b")


def skill(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def edges_from_list(listed):
    return sorted(tuple(sorted((LETTERS.index(p[0]), LETTERS.index(p[1])))) for p in listed.split(", "))


def network_from_text(text):
    """(n, sorted edges) for the network a question prints."""
    listed = _EDGE_LIST.search(text).group(1)
    n = LETTERS.index(_RANGE.search(text).group(1)) + 1
    return n, edges_from_list(listed)


def degrees(n, edges):
    d = [0] * n
    for u, v in edges:
        d[u] += 1
        d[v] += 1
    return d


def connected(n, edges):
    seen, todo = {0}, [0]
    while todo:
        x = todo.pop()
        for u, v in edges:
            for a, b in ((u, v), (v, u)):
                if a == x and b not in seen:
                    seen.add(b)
                    todo.append(b)
    return len(seen) == n


def ham_cycles(n, edges):
    es = set(edges)
    return [
        (0, *p)
        for p in permutations(range(1, n))
        if p[0] < p[-1] and all(tuple(sorted(((0, *p)[i], (0, *p)[(i + 1) % n]))) in es for i in range(n))
    ]


def check_graded(sk, q):
    """The canonical answer is full marks; every distractor of every part is wrong and tagged."""
    full = grade(sk, q, q.answer_text)
    assert full.correct, (q.topic, q.seed, q.answer_text)
    if q.parts:
        assert full.marks == full.max_marks == sum(p.marks for p in q.parts)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.seed, p.label)
            for text in p.distractors:
                result = grade(sk, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (q.topic, q.seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE
                assert p.answer_kind == "mcq" or text in p.distractor_values
        return
    assert len(q.distractors) >= 2, (q.topic, q.seed)
    for text in q.distractors:
        assert not grade(sk, q, text).correct, (q.topic, q.seed, text, q.answer_text)
        assert q.option_notes[text] in CATALOGUE
        assert q.answer_kind == "mcq" or text in q.distractor_values


def check_text(q):
    assert find_text_defects(q) == [], (q.topic, q.seed, find_text_defects(q))
    again = make(q.topic, seed=q.seed, difficulty=q.difficulty)
    assert q.to_json() == again.to_json()
