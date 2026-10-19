"""Tests `uc2-chromatic-polynomial-bipartite-hall` (topic `r3l_chromatic_polynomial_hall`) and its exam chains.

The oracle reads the graph or the worker/job lists out of the question TEXT and counts directly: proper colourings
by brute force, the best matching by trying every assignment, and Hall deficiency by enumerating every set of workers.
"""

from __future__ import annotations

import re
from itertools import combinations, product

import sympy

import mathgen.generators.r3l_chromatic_polynomial_hall_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from mathgen.grading import grade
from tests.r3l_support import check_parts, check_single, plain

TOPIC = "r3l_chromatic_polynomial_hall"
EXAM = "exd_x_r3l_chromatic_polynomial_hall"
SKILL_ID = "uc2-chromatic-polynomial-bipartite-hall"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
k = sympy.Symbol("k")


def _graph(text):
    body = plain(text)
    vs = re.search(r"vertices ([A-Z, ]+?) and edges ([A-Z, ]+?)\.", body)
    verts = vs.group(1).replace(" ", "").split(",")
    edges = [tuple(e) for e in vs.group(2).replace(" ", "").split(",")]
    return verts, edges


def _count(verts, edges, colours):
    idx = {v: i for i, v in enumerate(verts)}
    total = 0
    for c in product(range(colours), repeat=len(verts)):
        if all(c[idx[a]] != c[idx[b]] for a, b in edges):
            total += 1
    return total


def _workers(text):
    body = plain(text)
    seg = re.search(r"only do some of them: (.*?)\.\s", body + " ").group(1)
    adj = {}
    for part in seg.split("; "):
        name, jobs = part.split(" can do ")
        adj[name.strip()] = set(j.strip() for j in jobs.split(","))
    return adj


def _best_matching(adj):
    names = list(adj)
    best = 0
    for r in range(len(names), 0, -1):
        for group in combinations(names, r):
            jobs = [sorted(adj[w]) for w in group]
            for pick in product(*jobs):
                if len(set(pick)) == len(pick):
                    return r
    return best


def _deficiency(adj):
    names = list(adj)
    return max(
        len(s) - len(set().union(*(adj[w] for w in s)))
        for r in range(1, len(names) + 1)
        for s in combinations(names, r)
    )


def test_skill_metadata():
    assert list(_SKILL.year_level) == ["university"]
    assert "university-year-2" in _SKILL.tags


def test_easy_colouring_count():
    for seed in range(25):
        q = make(TOPIC, seed=seed, difficulty="easy")
        verts, edges = _graph(q.text)
        colours = int(re.search(r"with (\d+) colours", plain(q.text)).group(1))
        assert int(q.answer_text) == _count(verts, edges, colours)
        check_single(q, "number")


def test_medium_coefficient():
    for seed in range(25):
        q = make(TOPIC, seed=seed, difficulty="medium")
        verts, edges = _graph(q.text)
        power = int(re.search(r"coefficient of k(\d)", q.text.translate(str.maketrans("²³⁴", "234"))).group(1))
        pts = list(range(1, len(verts) + 2))
        poly = sympy.Poly(sympy.interpolate([(c, _count(verts, edges, c)) for c in pts], k), k)
        assert int(q.answer_text) == poly.coeff_monomial(k**power)
        check_single(q, "number")


def test_hard_both_forms():
    seen = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "builder" in q.text:
            seen.add("hall")
            assert int(q.answer_text) == _best_matching(_workers(q.text))
        else:
            seen.add("colour")
            verts, edges = _graph(q.text)
            colours = int(re.search(r"with (\d+) colours", plain(q.text)).group(1))
            assert int(q.answer_text) == _count(verts, edges, colours)
        check_single(q, "number")
    assert seen == {"hall", "colour"}


def test_exam_chains():
    seen = set()
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        check_parts(q)
        if "builder" in q.text:
            seen.add("hall")
            adj = _workers(q.text)
            assert int(q.parts[1].answer_text) == _deficiency(adj)
            assert int(q.parts[2].answer_text) == _best_matching(adj) == len(adj) - _deficiency(adj)
            replies = {p.label: p.answer_text for p in q.parts}
            replies["b"], replies["c"] = str(_deficiency(adj) + 1), str(len(adj) - _deficiency(adj) - 1)
            got = grade(q.parts[0].answer_kind, q, replies).parts["c"]
            assert got.marks == q.parts[2].marks and got.follow_through
        else:
            seen.add("colour")
            verts, edges = _graph(q.text)
            assert int(q.parts[1].answer_text) == _count(
                verts, edges, int(re.search(r"with (\d) colours", q.parts[1].prompt).group(1))
            )
    assert seen == {"hall", "colour"}
