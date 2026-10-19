"""Tests `graph-adjacency-matrices` (topic `snet_adjacency`) and its exam chain `snet_adjacency_chain`.

The oracles read the matrix (or the edge list) back out of the question TEXT and re-derive each key with SymPy: row sums,
the entries added, and the powers of the matrix.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

import pytest
import sympy
from snet_support import LETTERS, check_graded, check_text, network_from_text, skill

from mathgen import make
from mathgen.engine import grade

SK = skill("graph-adjacency-matrices")
TIERS = ("easy", "medium", "hard")
_MATRIX = re.compile(r"\[\[[0-9, \[\]]*\]\]")


def _q(tier, seed):
    return make("snet_adjacency", seed=seed, difficulty=tier)


def _exam(seed):
    return make("snet_adjacency_chain", seed=seed, difficulty="exam")


def _matrix(text):
    return sympy.Matrix(ast.literal_eval(_MATRIX.search(text).group(0)))


def _vertex(text, word):
    return LETTERS.index(re.search(rf"{word} ([A-H])\b", text).group(1))


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(80):
        check_graded(SK, _q(tier, seed))


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        check_text(_q(tier, seed))


def test_every_stated_matrix_is_an_adjacency_matrix():
    for tier in TIERS:
        for seed in range(100):
            q = _q(tier, seed)
            if _MATRIX.search(q.text):
                m = _matrix(q.text)
                assert m == m.T and all(m[i, i] == 0 for i in range(m.rows)) and set(m) <= {0, 1}


def test_easy_keys_match_the_matrix():
    seen = Counter()
    for seed in range(300):
        q = _q("easy", seed)
        if "joins:" in q.text:
            n, edges = network_from_text(q.text)
            assert q.answer == 2 * len(edges)
            seen["ones"] += 1
        elif "degree" in q.text:
            m = _matrix(q.text)
            x = _vertex(q.text, "vertex")
            assert q.answer == sum(m.row(x)) == sum(m.col(x))
            seen["degree"] += 1
        else:
            m = _matrix(q.text)
            assert q.answer * 2 == sum(m)
            seen["edges"] += 1
    assert set(seen) == {"ones", "degree", "edges"} and min(seen.values()) >= 40, seen


def test_medium_key_is_an_entry_of_the_matrix_squared():
    for seed in range(200):
        q = _q("medium", seed)
        m = _matrix(q.text)
        x, y = _vertex(q.text, "from"), _vertex(q.text, "to")
        assert q.answer == (m**2)[x, y]
        assert 1 <= q.answer <= 4


def test_hard_keys_are_powers_of_the_matrix():
    seen = Counter()
    for seed in range(300):
        q = _q("hard", seed)
        m = _matrix(q.text)
        if "length 3" in q.text:
            x, y = _vertex(q.text, "from"), _vertex(q.text, "to")
            assert q.answer == (m**3)[x, y]
            seen["cubed"] += 1
        else:
            x = LETTERS.index(re.search(r"start at (?:vertex )?([A-H])\b", q.text).group(1))
            assert q.answer == sum((m**2).row(x))
            seen["row"] += 1
    assert set(seen) == {"cubed", "row"} and min(seen.values()) >= 80, seen


def test_tiers_differ_in_shape():
    medium = [_q("medium", s).text for s in range(40)]
    hard = [_q("hard", s).text for s in range(60)]
    assert all("length 2" in t and "length 3" not in t for t in medium)
    assert any("length 3" in t for t in hard) and any("start" in t for t in hard)


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 1, 1]
        assert [p.answer_kind for p in q.parts] == ["matrix", "number", "number", "number"]
        assert [p.follow_through for p in q.parts] == [None, None, None, "part_b - part_c"]
        assert q.parts[1].sensible_range and q.parts[3].sensible_range
        check_text(q)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(200):
        q = _exam(seed)
        n, edges = network_from_text(q.stem)
        m = sympy.zeros(n, n)
        for u, v in edges:
            m[u, v] = m[v, u] = 1
        assert sympy.Matrix(q.parts[0].answer) == m
        x = LETTERS.index(re.search(r"start at ([A-H])\b", q.parts[1].prompt).group(1))
        squared = m**2
        total, closed = sum(squared.row(x)), squared[x, x]
        assert [p.answer for p in q.parts[1:]] == [total, closed, total - closed]
        assert closed == sum(m.row(x))  # a walk of length 2 that returns uses an edge out and back


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        check_graded(SK, _exam(seed))


def test_exam_matrix_reply_forms():
    for seed in range(40):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        rows = q.parts[0].answer
        latex = r"\begin{pmatrix} " + r" \\ ".join(" & ".join(str(v) for v in row) for row in rows) + r" \end{pmatrix}"
        semicolons = "; ".join(" ".join(str(v) for v in row) for row in rows)
        for reply in (latex, "A = " + q.parts[0].answer_text, semicolons):
            assert grade(SK, q, {**canon, "a": reply}).parts["a"].marks == 2, (seed, reply)
        flipped = [row[:] for row in rows]
        flipped[0][1] = flipped[1][0] = 1 - rows[0][1]
        assert grade(SK, q, {**canon, "a": str(flipped)}).parts["a"].marks == 0


def test_exam_follow_through_from_wrong_earlier_parts():
    checked = 0
    for seed in range(100):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        b, c = int(canon["b"]), int(canon["c"])
        wrong_b = b + 1
        if wrong_b - c in (wrong_b, c):
            continue
        result = grade(SK, q, {**canon, "b": str(wrong_b), "d": str(wrong_b - c)})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True, (seed, result)
        checked += 1
    assert checked >= 80


def test_exam_answer_space():
    assert len({_exam(s).text for s in range(300)}) >= 290
    assert len({_exam(s).parts[1].answer for s in range(300)}) >= 5
