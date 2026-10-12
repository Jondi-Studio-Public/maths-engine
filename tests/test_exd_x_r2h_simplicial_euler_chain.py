"""Tests for the multi-part exam source `exd_x_r2h_simplicial_euler_chain` (skill `ucv1-simplex-homology`).

The oracle reads the maximal faces out of the rendered text, builds the complex, and finds the Betti numbers from the
ranks of the boundary matrices over the rationals.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks
from itertools import combinations

import sympy

TOPIC = "exd_x_r2h_simplicial_euler_chain"
SKILL_ID = "ucv1-simplex-homology"
HARD_TOPIC = "p15b_simplex_homology"
LABELS = list("abcd")
MARKS = [1, 1, 1, 2]
FT_LABELS = ["b", "d"]


def parse(q):
    last = re.search(r"1 to (\d+)|\.\.\., (\d+)", q.text)
    n_vertices = int(next(g for g in last.groups() if g))
    faces = [tuple(sorted(int(v) for v in body.split(","))) for body in re.findall(r"\{([\d, ]+)\}", q.text)]
    return n_vertices, faces


def complex_of(faces):
    simplices = {k: set() for k in (1, 2, 3)}
    for face in faces:
        for k in (1, 2, 3):
            simplices[k].update(combinations(face, k))
    return simplices


def oracle(q):
    n_vertices, faces = parse(q)
    simp = complex_of(faces)
    vertices = sorted({v for (v,) in simp[1]})
    assert vertices == list(range(1, n_vertices + 1))
    edges, triangles = sorted(simp[2]), sorted(simp[3])
    d1 = sympy.zeros(len(vertices), len(edges))
    for col, (u, v) in enumerate(edges):
        d1[vertices.index(u), col] = -1
        d1[vertices.index(v), col] = 1
    d2 = sympy.zeros(len(edges), len(triangles))
    for col, (u, v, w) in enumerate(triangles):
        for sign, edge in ((1, (v, w)), (-1, (u, w)), (1, (u, v))):
            d2[edges.index(edge), col] = sign
    assert d1 * d2 == sympy.zeros(len(vertices), len(triangles)) if triangles else True
    rank1, rank2 = d1.rank(), (d2.rank() if triangles else 0)
    b0 = len(vertices) - rank1
    b1 = len(edges) - rank1 - rank2
    b2 = len(triangles) - rank2
    assert b2 == 0
    chi = len(vertices) - len(edges) + len(triangles)
    assert chi == b0 - b1 + b2
    return {"a": str(len(edges)), "b": str(chi), "c": str(b0), "d": str(b1)}


def test_follow_through_from_wrong_edge_count():
    def wrong_chain(q):
        key = {label: int(p.answer_text) for label, p in zip(LABELS, q.parts)}
        # A carried Euler characteristic that is zero, or repeats the slip, earns no follow-through: avoid those.
        wrong = next(int(d) for d in q.parts[0].distractors if key["b"] + key["a"] - int(d) not in (0, int(d)))
        chi = key["b"] + key["a"] - wrong
        replies = {"a": str(wrong), "b": str(chi), "c": q.parts[2].answer_text, "d": str(key["c"] - chi)}
        return "a", replies, 4

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_follow_through_from_wrong_euler_characteristic():
    def wrong_chain(q):
        key_c = int(q.parts[2].answer_text)
        wrong = int(q.parts[1].distractors[0])
        replies = {
            "a": q.parts[0].answer_text,
            "b": str(wrong),
            "c": q.parts[2].answer_text,
            "d": str(key_c - wrong),
        }
        return "b", replies, 4

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        assert grade(skill, q, {"a": "1009", "b": "211", "c": "307", "d": "401"}).marks == 0


CHECKS = standard_checks(TOPIC, SKILL_ID, oracle, LABELS, MARKS, FT_LABELS, HARD_TOPIC)


def test_wired():
    CHECKS["wired"]()


def test_shape():
    CHECKS["shape"]()


def test_prompt_shapes():
    CHECKS["shapes"]()


def test_oracle_agrees_over_300_seeds():
    CHECKS["oracle_agrees"]()


def test_composed_answer_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()
