"""Tests `ucv1-null-space` (topic `ucv1_null_space`) and its exam chain `exd_x_uc_null_space`.

The oracle reads the matrix (and, for the parameter questions, the vector or the entry with k) back out of the question
TEXT and re-derives the key with SymPy: `Matrix.rank()`, `nullspace()`, the rank of the augmented matrix for each
candidate k, and the roots of det M(k).
"""

from __future__ import annotations

import ast
import re
from collections import Counter

import pytest
import sympy

import mathgen.generators.ucv1_null_space_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_null_space"
EXAM = "exd_x_uc_null_space"
SKILL_ID = "ucv1-null-space"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_MATRIX_RE = re.compile(r"\[\[[^A-Za-zλ]*?\]\]")
_K = sympy.symbols("k")


def _matrix(text):
    return sympy.Matrix(ast.literal_eval(_MATRIX_RE.search(text).group(0)))


def _matrix_with_k(text):
    literal = re.search(r"\[\[[^A-Za-zλ]*?k[^A-Za-zλ]*?\]\]", text).group(0)
    rows = ast.literal_eval(literal.replace("k", "'k'"))
    return sympy.Matrix([[_K if v == "k" else v for v in row] for row in rows])


def _vector_with_k(text):
    literal = re.search(r"\(([^()A-Za-z]*k[^()A-Za-z]*|[^()]*k[^()]*)\)ᵀ", text).group(1)
    return [_K if part.strip() == "k" else int(part) for part in literal.split(",")]


def _k_in_column_space(m, b):
    """All integers k in a wide range for which the augmented matrix has the same rank as m."""
    found = []
    for k in range(-40, 41):
        vec = sympy.Matrix([k if v == _K else v for v in b])
        if m.row_join(vec).rank() == m.rank():
            found.append(k)
    return found


def _q(difficulty, seed):
    return make(TOPIC, seed=seed, difficulty=difficulty)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _flat(answer):
    return [sympy.Integer(r[0]) for r in answer]


def _is_multiple(vec, expected):
    ratios = set()
    for v, e in zip(vec, expected):
        if e == 0:
            if v != 0:
                return False
        else:
            ratios.add(sympy.Rational(v) / e)
    return len(ratios) == 1 and 0 not in ratios


# -- wiring ---------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert _SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    src = _SKILL.sources[-1]
    assert (src.stages, src.params) == (["exam"], {})
    assert "matrix-rank-small" in _SKILL.prerequisites


# -- single-answer tiers ------------------------------------------------------------------------------------------


def test_easy_key_is_the_rank():
    shapes = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        m = _matrix(q.text)
        shapes[(m.rows, m.cols, int(q.answer))] += 1
        assert int(q.answer) == m.rank(), (seed, q.text)
        assert q.answer_kind == "number"
    assert len(shapes) >= 4


def test_easy_ranks_cover_one_two_and_three():
    ranks = Counter(int(_q("easy", seed).answer) for seed in range(120))
    assert set(ranks) == {1, 2, 3}, ranks
    assert ranks[2] >= 30 and ranks[3] >= 15


def test_medium_key_is_the_nullity():
    for seed in range(120):
        q = _q("medium", seed)
        m = _matrix(q.text)
        assert int(q.answer) == len(m.nullspace()) == m.cols - m.rank(), (seed, q.text)
        assert q.answer >= 1


def test_hard_key_is_the_parameter_value():
    seen = Counter()
    for seed in range(160):
        q = _q("hard", seed)
        if re.search(r"ᵀ", q.text):
            m = _matrix(q.text)
            b = _vector_with_k(q.text)
            assert m.rank() == 2
            assert _k_in_column_space(m, b) == [int(q.answer)], (seed, q.text)
            seen["colspace"] += 1
        else:
            m = _matrix_with_k(q.text)
            roots = sympy.solve(m.det(), _K)
            assert roots == [int(q.answer)], (seed, q.text)
            assert m.subs(_K, int(q.answer)).rank() == 2
            seen["det"] += 1
    assert seen["colspace"] >= 40 and seen["det"] >= 20


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(100):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    # The easy rank and medium nullity are small integers, so the variety is in the matrix, not the key.
    texts = [_q(tier, s).text for s in range(60)]
    assert len(set(texts)) >= 55
    answers = [_q(tier, s).answer_text for s in range(60)]
    if tier == "hard":
        assert len(set(answers)) >= 12, Counter(answers).most_common(5)
        assert max(Counter(answers).values()) <= 10
    else:
        assert len(set(answers)) >= 2


# -- exam chain --------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 3]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "matrix", "number"]
        assert [p.follow_through for p in q.parts] == [None, "3 - part_a", None, None]
        assert [p.answer_form for p in q.parts] == [None, None, "scalar_multiple", None]
        assert find_text_defects(q) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_independent_derivation():
    for seed in range(200):
        q = _exam(seed)
        m = _matrix(q.stem)
        assert m.shape == (3, 3)
        rank = m.rank()
        assert int(q.parts[0].answer) == rank == 2
        assert int(q.parts[1].answer) == len(m.nullspace()) == 3 - rank
        null = m.nullspace()[0]
        assert m * sympy.Matrix(_flat(q.parts[2].answer)) == sympy.zeros(3, 1)
        assert _is_multiple(_flat(q.parts[2].answer), list(null))
        b = _vector_with_k(q.parts[3].prompt)
        assert _k_in_column_space(m, b) == [int(q.parts[3].answer)], (seed, q.parts[3].prompt)


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 8
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_null_vector_accepts_any_multiple():
    for seed in range(60):
        q = _exam(seed)
        vec = _flat(q.parts[2].answer)
        canon = {p.label: p.answer_text for p in q.parts}
        for scale in (2, -1, 5):
            reply = "(" + ", ".join(str(scale * v) for v in vec) + ")"
            assert grade(_SKILL, q, {**canon, "c": reply}).parts["c"].marks == 2
        assert grade(_SKILL, q, {**canon, "c": "(0, 0, 0)"}).parts["c"].marks == 0


def test_exam_follow_through_from_a_wrong_rank():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong_rank in (1, 4):
            result = grade(_SKILL, q, {**canon, "a": str(wrong_rank), "b": str(3 - wrong_rank)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, wrong_rank, result)
            checked += 1
    assert checked == 120


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(Counter(answers).values()) <= 15
