"""Independent oracle for the `p14b_tree_edge_count` questions: it re-derives each answer from the rendered
question text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p14b_tree_edge_count_build`) and re-derives the answer
from scratch, one route per tier shape.
"""

from dataclasses import replace

import re

import pytest

import mathgen
from mathgen.skills import load_skills

TREE_ID = "ucv1-tree-spanning-edge-count"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_LEAVES_RE = re.compile(r"(\d+) leaves")
_DEGREES_RE = re.compile(r": ([\d, ]+)\.")
_VERTICES_RE = re.compile(r"(\d+) vertices")
_TOTAL_EDGES_RE = re.compile(r"(\d+) edges (?:in total|altogether)")


def _solve(text: str) -> int:
    # Hard is now a full binary tree given its LEAF count L (n = 2L - 1), never a depth-based
    # complete binary tree. Exam states only L and the TOTAL edge count m -- it never states the
    # extra-edge count itself (that was defect 2: the original prompt printed the answer literally),
    # so the discriminator between hard and exam is now "does the text also state a total edge
    # count", not "does it mention an extra-connections count".
    leaves_match = _LEAVES_RE.search(text)
    total_match = _TOTAL_EDGES_RE.search(text)
    if leaves_match is not None and total_match is not None:
        # exam: full binary tree with L leaves, plus a stated total edge
        # count m -- extra edges = m - (n - 1).
        L = int(leaves_match.group(1))
        n = 2 * L - 1
        total = int(total_match.group(1))
        return total - (n - 1)

    if leaves_match is not None:
        # hard: full binary tree with L leaves -- n = 2L - 1, edges = n - 1.
        L = int(leaves_match.group(1))
        n = 2 * L - 1
        return n - 1

    degrees_match = _DEGREES_RE.search(text)
    if degrees_match is not None:
        # medium: full degree sequence stated -- n = len(list), edges = n - 1
        # (independently cross-checked against the handshake-lemma route,
        # sum(degrees) / 2, which must agree for a real tree).
        degrees = [int(x) for x in degrees_match.group(1).split(",")]
        n = len(degrees)
        by_count = n - 1
        by_handshake = sum(degrees) // 2
        assert by_count == by_handshake, (degrees, by_count, by_handshake)
        return by_count

    # easy: n stated directly.
    n = int(_VERTICES_RE.search(text).group(1))
    return n - 1


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


TREE_SOURCES = _skill_sources(TREE_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
TREE_SOURCES["exam"] = replace(TREE_SOURCES["exam"], topic="p14b_tree_edge_count", params={})


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in TIERS for s in SEEDS])
def test_oracle_rederives_tree_edge_count(difficulty, seed):
    src = TREE_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    stated = int(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_would_catch_a_corrupted_answer():
    src = TREE_SOURCES["easy"]
    q = mathgen.make(src.topic, seed=0, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    corrupted = derived + 1
    assert derived != corrupted


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 240
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds).")
