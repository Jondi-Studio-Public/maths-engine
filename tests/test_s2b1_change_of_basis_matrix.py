"""Tests `uc2-change-of-basis-matrix` (topic `s2b1_change_of_basis_matrix`) and its exam chain `exd_x_s2b1_change_basis`.

The oracle reads the bases (or the supplied matrices and vector) back out of the question TEXT and re-derives the key
with SymPy: P = C^-1 B, QP for a composition, P^-1 [v]_B for a conversion.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2b1_change_of_basis_matrix_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "s2b1_change_of_basis_matrix"
EXAM = "exd_x_s2b1_change_basis"
SKILL_ID = "uc2-change-of-basis-matrix"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_X = sympy.symbols("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _entry(e):
    return Fraction(e["p"], e["q"]) if isinstance(e, dict) else Fraction(e)


def _rows(answer):
    return [[_entry(e) for e in row] for row in answer]


def _ints(text):
    return [[int(v) for v in g.split(",")] for g in re.findall(r"\(([-\d, ]+)\)", text)]


def _poly_coeffs(text, degree=2):
    expr = parse_expr(text.replace("²", "**2").replace("³", "**3"), transformations=_TRANSFORMS, local_dict={"x": _X})
    poly = sympy.Poly(sympy.expand(expr), _X)
    return [poly.coeff_monomial(_X**d) for d in range(degree + 1)]


def _transition(bcols, ccols):
    return sympy.Matrix(ccols).T.inv() * sympy.Matrix(bcols).T


def _matrices(text):
    return [sympy.Matrix(ast.literal_eval(m)) for m in re.findall(r"\[\[[-\d, \[\]]*?\]\]", text)]


def _oracle(q):
    text = q.text
    if "to D" in text:
        p, qm = _matrices(text)
        return qm * p
    if "for every v" in text and "[v]_B = P [v]_C" in text:
        (p,) = _matrices(text)
        vb = _ints(text.split("[v]_B = ")[-1])[0] if "[v]_B = (" in text else None
        vb = _ints(re.search(r"\[v\]_B = (\([-\d, ]+\))", text).group(1))[0]
        return list(p.LUsolve(sympy.Matrix(vb)))
    if "P₂" in text:
        m = re.search(r"B = \{(.+?)\} and C = \{(.+?)\}", text)
        sets = []
        for group in m.groups():
            parts = re.split(r", (?=[-\dx(])", group)
            sets.append([_poly_coeffs(g) for g in parts])
        return _transition(*sets)
    vecs = _ints(text)
    n = 2 if "ℝ²" in text else 3
    return _transition(vecs[:n], vecs[n : 2 * n])


def _as_fractions(value):
    if isinstance(value, list):
        return [[Fraction(int(v.p), int(v.q))] for v in value]
    return [[Fraction(int(v.p), int(v.q)) for v in value.row(i)] for i in range(value.rows)]


def test_skill_wiring():
    assert _SKILL.answer == "matrix"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert "university-year-2" in _SKILL.tags


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_solution(tier):
    for seed in range(120):
        q = _q(tier, seed)
        assert q.answer_kind == "matrix"
        text = q.text
        if re.search(r"B = \{1, (\(x [+-] \d+\)|x), (\(x [+-] \d+\)|x)²\}", text):
            m = re.search(r"B = \{1, (\(x [+-] \d+\)|x), .*?\} and C = \{1, (\(x [+-] \d+\)|x), ", text)
            shifts = []
            for g in m.groups():
                num = re.search(r"([+-]) (\d+)", g)
                shifts.append(0 if num is None else int(num.group(2)) * (1 if num.group(1) == "-" else -1))
            bcols = [_poly_coeffs(str(sympy.expand((_X - shifts[0]) ** k))) for k in range(3)]
            ccols = [_poly_coeffs(str(sympy.expand((_X - shifts[1]) ** k))) for k in range(3)]
            expected = _transition(bcols, ccols)
        else:
            expected = _oracle(q)
        assert _rows(q.answer) == _as_fractions(expected), (seed, text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(80):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(50):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


def test_tiers_differ_in_shape():
    easy = Counter(len(_q("easy", s).answer) for s in range(40))
    medium = Counter(("P₂" in _q("medium", s).text, len(_q("medium", s).answer)) for s in range(60))
    hard = Counter()
    for s in range(150):
        text = _q("hard", s).text
        hard["compose" if "to D" in text else "shifted" if "P₂" in text else "convert"] += 1
    assert set(easy) == {2}
    assert set(medium) == {(False, 3), (True, 3)}
    assert hard["compose"] >= 25 and hard["convert"] >= 25 and hard["shifted"] >= 25


def test_answer_variety():
    for tier in TIERS:
        texts = {_q(tier, s).answer_text for s in range(80)}
        assert len(texts) >= (20 if tier == "hard" else 60), (tier, len(texts))


# -- exam -------------------------------------------------------------------------------------------------------------


def test_exam_keys_match_independent_derivation():
    for seed in range(100):
        q = _exam(seed)
        vecs = _ints(q.stem)
        b, c = vecs[:2], vecs[2:4]
        p = _transition(b, c)
        assert Fraction(int(q.parts[0].answer)) == Fraction(int(p.det().p), int(p.det().q))
        assert _rows(q.parts[1].answer) == _as_fractions(p)
        vb = _ints(q.parts[2].prompt)[0]
        assert _rows(q.parts[2].answer) == _as_fractions(list(p * sympy.Matrix(vb)))
        entry = p.inv()[0, 1]
        assert _entry(q.parts[3].answer) == Fraction(int(entry.p), int(entry.q))


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 3, 2, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "matrix", "matrix", "fraction"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert find_text_defects(q) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(60):
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


def test_exam_follow_through_from_a_wrong_determinant():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        numerator = sympy.sympify(q.parts[3].follow_through.replace("/part_a", ""))
        wrong = int(q.parts[0].answer) + 3
        if wrong == 0:
            continue
        reply = {**canon, "a": str(wrong), "d": str(numerator / wrong)}
        if reply["d"] in q.parts[3].distractors:
            continue  # a follow-through value that is also a tagged wrong option is graded as that error
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, result)
        checked += 1
    assert checked >= 25


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(200)]
    assert len(set(answers)) >= 190
