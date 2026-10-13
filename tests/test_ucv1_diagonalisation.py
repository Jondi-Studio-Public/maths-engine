"""Tests `ucv1-diagonalisation` (topic `ucv1_diagonalisation`) and its exam chain `exd_x_uc_diagonalisation`.

The oracle reads P, D or A, and the power, back out of the question TEXT and re-derives each key with SymPy matrix
algebra (`P * D * P.inv()`, `A ** n`, null spaces), sharing nothing with the generator's exact-fraction routines.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

import pytest
import sympy

import mathgen.generators.ucv1_diagonalisation_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_diagonalisation"
EXAM = "exd_x_uc_diagonalisation"
SKILL_ID = "ucv1-diagonalisation"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_MATRIX_RE = re.compile(r"\[\[[^A-Za-zλ]*?\]\]")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _matrices(text):
    return [sympy.Matrix(ast.literal_eval(m)) for m in _MATRIX_RE.findall(text)]


def _value(entry):
    return sympy.Rational(entry["p"], entry["q"]) if isinstance(entry, dict) else sympy.Integer(entry)


def _answer_matrix(answer):
    return sympy.Matrix([[_value(v) for v in row] for row in answer])


def _q(difficulty, seed):
    return make(TOPIC, seed=seed, difficulty=difficulty)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _column_multiples(given: sympy.Matrix, expected: sympy.Matrix) -> bool:
    for c in range(expected.cols):
        g, e = list(given[:, c]), list(expected[:, c])
        ratios = {sympy.Rational(x) / y for x, y in zip(g, e) if y != 0}
        if len(ratios) != 1 or 0 in ratios or any(x != 0 for x, y in zip(g, e) if y == 0):
            return False
    return True


def _eigenvector_matrix(a, values):
    cols = []
    for lam in values:
        ns = (a - lam * sympy.eye(a.rows)).nullspace()
        assert len(ns) == 1
        cols.append(ns[0])
    return sympy.Matrix.hstack(*cols)


# -- wiring ---------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert _SKILL.answer == "matrix"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    src = _SKILL.sources[-1]
    assert (src.stages, src.params) == (["exam"], {})
    assert "two-by-two-eigenvalues" in _SKILL.prerequisites


# -- single-answer tiers ------------------------------------------------------------------------------------------


def test_easy_key_is_p_d_p_inverse():
    for seed in range(120):
        q = _q("easy", seed)
        p, d = _matrices(q.text)
        assert abs(p.det()) == 1 and d.is_diagonal()
        assert _answer_matrix(q.answer) == p * d * p.inv(), (seed, q.text)
        assert q.answer_kind == "matrix" and q.answer_form is None


def test_medium_keys():
    seen = {"D": 0, "P": 0}
    for seed in range(160):
        q = _q("medium", seed)
        mats = _matrices(q.text)
        if len(mats) == 1:  # find D
            seen["D"] += 1
            a = mats[0]
            expected = sympy.diag(*sorted(v for v, m in a.eigenvals().items() for _ in range(m)))
            assert _answer_matrix(q.answer) == expected, (seed, q.text)
            assert q.answer_form is None
        else:  # find P for a stated D
            seen["P"] += 1
            a, d = mats
            given = _answer_matrix(q.answer)
            assert q.answer_form == "column_scalar_multiple"
            assert given.inv() * a * given == d, (seed, q.text, q.answer_text)
            assert _column_multiples(given, _eigenvector_matrix(a, list(d.diagonal())))
    assert seen["D"] >= 30 and seen["P"] >= 30


def test_hard_key_is_the_matrix_power():
    for seed in range(120):
        q = _q("hard", seed)
        (a,) = _matrices(q.text)
        n = int(re.search(r"A([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", q.text).group(1).translate(_SUPERSCRIPTS))
        assert 3 <= n <= 10
        assert _answer_matrix(q.answer) == a**n, (seed, q.text)


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


def test_medium_p_accepts_scaled_columns_and_rejects_swapped_columns():
    checked = 0
    for seed in range(120):
        q = _q("medium", seed)
        if q.answer_form != "column_scalar_multiple":
            continue
        p = _answer_matrix(q.answer)
        scaled = sympy.Matrix.hstack(3 * p[:, 0], -2 * p[:, 1])
        swapped = sympy.Matrix.hstack(p[:, 1], p[:, 0])
        as_text = lambda m: str([list(m.row(i)) for i in range(m.rows)])  # noqa: E731
        assert grade(_SKILL, q, as_text(scaled)).correct
        assert not grade(_SKILL, q, as_text(swapped)).correct
        checked += 1
    assert checked >= 30


def test_hard_distractors_include_the_named_slips():
    tags = Counter()
    for seed in range(80):
        tags.update(_q("hard", seed).option_notes.values())
    assert {"matrix-power-entrywise", "diag-eigenvalues-swapped", "matrix-power-off-by-one"} <= set(tags)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(60)]
    assert len(set(answers)) >= 12, Counter(answers).most_common(5)
    assert max(Counter(answers).values()) <= 12, Counter(answers).most_common(3)


# -- exam chain --------------------------------------------------------------------------------------------------


def _parse_exam(q):
    a = _matrices(q.stem)[0]
    given = int(re.findall(r"(?:eigenvalue|Let A = .*? is) (-?\d+)\.", q.stem)[-1])
    (p_stated,) = _matrices(q.parts[2].prompt)
    row, col, power = re.search(r"row (\d), column (\d) of A([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", q.parts[3].prompt).groups()
    return a, given, p_stated, int(row) - 1, int(col) - 1, int(power.translate(_SUPERSCRIPTS))


def _coefficients(a, given, other, row, col):
    """c1, c2 with (A^n)[row, col] = c1 given^n + c2 other^n, from n = 0 and n = 1 (A is diagonalisable)."""
    c1, c2 = sympy.symbols("c1 c2")
    sol = sympy.solve(
        [c1 + c2 - sympy.eye(2)[row, col], c1 * given + c2 * other - a[row, col]],
        [c1, c2],
    )
    return sol[c1], sol[c2]


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 3]
        assert [p.answer_kind for p in q.parts] == ["number", "matrix", "matrix", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert [p.answer_form for p in q.parts] == [None, "column_scalar_multiple", None, None]
        assert find_text_defects(q) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_independent_derivation():
    for seed in range(200):
        q = _exam(seed)
        a, given, p_stated, row, col, n = _parse_exam(q)
        values = sorted(a.eigenvals())
        other = [v for v in values if v != given] if given in values else None
        assert other and len(other) == 1
        other = other[0]
        assert int(q.parts[0].answer) == other, seed
        p = _answer_matrix(q.parts[1].answer)
        assert p.inv() * a * p == sympy.diag(*values)
        assert _answer_matrix(q.parts[2].answer) == p_stated.inv()
        assert p_stated.inv() * a * p_stated == sympy.diag(*values)
        assert int(q.parts[3].answer) == (a**n)[row, col], seed


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 9
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_follow_through_from_a_wrong_other_eigenvalue():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        a, given, _p, row, col, n = _parse_exam(q)
        other = int(q.parts[0].answer)
        c1, c2 = _coefficients(a, given, other, row, col)
        for wrong in (int(a.trace()), int(a.trace()) + given, -other):
            if (
                wrong in (other, 0) or c1 * given**n + c2 * wrong**n == wrong
            ):  # the engine ignores a reply that just repeats (a)
                continue
            entry = c1 * given**n + c2 * wrong**n
            canon = {p.label: p.answer_text for p in q.parts}
            result = grade(_SKILL, q, {**canon, "a": str(wrong), "d": str(entry)})
            assert result.parts["a"].marks == 0
            if entry != (a**n)[row, col]:
                assert result.parts["d"].marks == 3 and result.parts["d"].follow_through is True, (seed, wrong, result)
                checked += 1
    assert checked >= 100


def test_exam_scaled_columns_in_part_b_and_fractions_in_part_c():
    fractional = 0
    for seed in range(80):
        q = _exam(seed)
        p = _answer_matrix(q.parts[1].answer)
        scaled = sympy.Matrix.hstack(2 * p[:, 0], -3 * p[:, 1])
        canon = {x.label: x.answer_text for x in q.parts}
        text = str([list(scaled.row(i)) for i in range(2)])
        assert grade(_SKILL, q, {**canon, "b": text}).parts["b"].marks == 2
        if any(v.q != 1 for v in _answer_matrix(q.parts[2].answer)):
            fractional += 1
    assert fractional >= 20


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(Counter(answers).values()) <= 15
