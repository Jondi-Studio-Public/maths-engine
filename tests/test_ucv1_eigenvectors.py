"""Tests `ucv1-eigenvectors` (topic `ucv1_eigenvectors`) and its exam chain `exd_x_uc_eigenvectors`.

The oracle reads the matrix and the eigenvalue back out of the question TEXT and re-derives the eigenvector with SymPy's
null space of A - λI, so it shares no code with the generator's own Gauss-Jordan route.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

import pytest
import sympy

import mathgen.generators.ucv1_eigenvectors_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_eigenvectors"
EXAM = "exd_x_uc_eigenvectors"
SKILL_ID = "ucv1-eigenvectors"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_MATRIX_RE = re.compile(r"\[\[[^A-Za-zλ]*?\]\]")


def _matrix_in(text):
    return sympy.Matrix(ast.literal_eval(_MATRIX_RE.search(text).group(0)))


def _stem(q):
    return q.stem if q.parts else q.text


def _stated_eigenvalue(text):
    rest = _MATRIX_RE.sub("A", text)
    return int(re.findall(r"(?:λ = |eigenvalue |\bis )(-?\d+)\b", rest)[-1])


def _eigenvalue_for(q):
    """The eigenvalue the question asks about: stated, or the larger/smaller of a 2×2 matrix's two."""
    a = _matrix_in(q.text)
    if "larger" in q.text or "smaller" in q.text:
        values = sorted(a.eigenvals())
        return values[-1] if "larger" in q.text else values[0]
    return _stated_eigenvalue(q.text)


def _is_multiple(vec, expected):
    """vec is a non-zero multiple of expected (both flat lists of numbers)."""
    ratios = set()
    for v, e in zip(vec, expected):
        if e == 0:
            if v != 0:
                return False
        else:
            ratios.add(sympy.Rational(v) / e)
    return len(ratios) == 1 and 0 not in ratios


def _flat(answer):
    return [sympy.Rational(r[0]["p"], r[0]["q"]) if isinstance(r[0], dict) else sympy.Integer(r[0]) for r in answer]


def _oracle_vector(a, lam):
    ns = (a - lam * sympy.eye(a.rows)).nullspace()
    assert len(ns) == 1
    return list(ns[0])


def _q(difficulty, seed):
    return make(TOPIC, seed=seed, difficulty=difficulty)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


# -- wiring ---------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert _SKILL.answer == "matrix"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert (_SKILL.sources[-1].stages, _SKILL.sources[-1].params) == (["exam"], {})
    assert {"two-by-two-eigenvalues"} <= set(_SKILL.prerequisites)


# -- single-answer tiers ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_eigenvector(tier):
    for seed in range(120):
        q = _q(tier, seed)
        a, lam = _matrix_in(q.text), _eigenvalue_for(q)
        expected = _oracle_vector(a, lam)
        assert q.answer_kind == "matrix" and q.answer_form == "scalar_multiple"
        vec = _flat(q.answer)
        assert len(vec) == a.rows
        assert _is_multiple(vec, expected), (seed, q.text, q.answer_text)
        assert a * sympy.Matrix(vec) == lam * sympy.Matrix(vec)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_in_several_spellings_and_distractors_rejected(tier):
    for seed in range(100):
        q = _q(tier, seed)
        vec = _flat(q.answer)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        doubled = [2 * v for v in vec]
        negative = [-v for v in vec]
        assert grade(_SKILL, q, "(" + ", ".join(str(v) for v in doubled) + ")").correct
        assert grade(_SKILL, q, "[" + "; ".join(str(v) for v in negative) + "]").correct
        assert grade(_SKILL, q, "v = [[" + "], [".join(str(v) for v in vec) + "]]").correct
        assert not grade(_SKILL, q, "(0, " + ", ".join("0" for _ in vec[1:]) + ")").correct
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_not_eigenvectors_of_the_asked_eigenvalue(tier):
    for seed in range(60):
        q = _q(tier, seed)
        a, lam = _matrix_in(q.text), _eigenvalue_for(q)
        for text in q.distractors:
            v = sympy.Matrix(ast.literal_eval(text.replace("(", "[").replace(")", "]")))
            assert a * v != lam * v, (seed, text)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


def test_tier_shapes():
    easy = {_matrix_in(_q("easy", s).text).rows for s in range(40)}
    assert easy == {2}
    medium_text = [_q("medium", s).text for s in range(60)]
    assert any("larger" in t or "smaller" in t for t in medium_text)
    assert any(_matrix_in(t).rows == 3 for t in medium_text)
    hard = [_q("hard", s).text for s in range(60)]
    assert all(_matrix_in(t).rows == 3 for t in hard)
    assert any("one-dimensional" in t for t in hard)
    assert any("one-dimensional" not in t for t in hard)


def test_medium_triangular_answer_is_never_the_unit_vector():
    seen = 0
    for seed in range(80):
        q = _q("medium", seed)
        a = _matrix_in(q.text)
        if a.rows == 3 and a.is_upper and all(a[i, j] != 0 for i, j in ((0, 1), (0, 2), (1, 2))):
            seen += 1
            assert sum(1 for v in _flat(q.answer) if v != 0) >= 2, (seed, q.answer_text)
    assert seen >= 5


def test_hard_defective_has_a_one_dimensional_eigenspace():
    seen = 0
    for seed in range(80):
        q = _q("hard", seed)
        if "one-dimensional" in q.text:
            seen += 1
            a, lam = _matrix_in(q.text), _stated_eigenvalue(q.text)
            assert (a - lam * sympy.eye(3)).rank() == 2
            poly = a.charpoly()
            assert poly.eval(lam) == 0 and poly.diff().eval(lam) == 0  # lam is a repeated root
    assert seen >= 10


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(60)]
    assert len(set(answers)) >= 12, Counter(answers).most_common(5)
    assert max(Counter(answers).values()) <= 18, Counter(answers).most_common(3)


# -- exam chain --------------------------------------------------------------------------------------------------


def _exam_oracle(q):
    a = _matrix_in(q.stem)
    small, large = sorted(a.eigenvals())
    return a, small, large


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "matrix", "matrix"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, False]
        assert [p.answer_form for p in q.parts] == [None, None, "scalar_multiple", "scalar_multiple"]
        assert find_text_defects(q) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_independent_derivation():
    for seed in range(200):
        q = _exam(seed)
        a, small, large = _exam_oracle(q)
        assert int(q.parts[0].answer) == large and int(q.parts[1].answer) == small
        assert _is_multiple(_flat(q.parts[2].answer), _oracle_vector(a, large)), seed
        assert _is_multiple(_flat(q.parts[3].answer), _oracle_vector(a, small)), seed
        assert small != 0 and large != 0


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 7
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_follow_through_from_a_wrong_larger_eigenvalue():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        a, small, large = _exam_oracle(q)
        trace = int(a.trace())
        for wrong in (-large, trace, large + 1):
            # (b) follows the learner's own (a) through the trace
            reply = {"a": str(wrong), "b": str(trace - wrong)}
            if trace - wrong == small:
                continue
            result = grade(_SKILL, q, {**{p.label: p.answer_text for p in q.parts}, **reply})
            assert result.parts["a"].marks == 0
            if wrong != 0:
                assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, wrong, result)
                checked += 1
    assert checked >= 100


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(Counter(answers).values()) <= 15


def test_exam_source_wired_last():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages, src.params) == (EXAM, "exam", ["exam"], {})
