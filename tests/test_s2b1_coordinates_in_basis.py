"""Tests `uc2-coordinates-in-basis` (topic `s2b1_coordinates_in_basis`) and its exam chain `exd_x_s2b1_coordinates`.

The oracle reads the basis and the target back out of the question TEXT and re-derives the coordinates with SymPy.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2b1_coordinates_in_basis_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "s2b1_coordinates_in_basis"
EXAM = "exd_x_s2b1_coordinates"
SKILL_ID = "uc2-coordinates-in-basis"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_X, _K = sympy.symbols("x k")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _entry(e):
    return Fraction(e["p"], e["q"]) if isinstance(e, dict) else Fraction(e)


def _coords(answer):
    return [_entry(row[0]) for row in answer]


def _ints(text):
    return [tuple(int(v) for v in g.split(",")) for g in re.findall(r"\(([-\d, ]+)\)", text)]


def _poly(text, symbols=()):
    expr = parse_expr(
        text.replace("²", "**2").replace("³", "**3"), transformations=_TRANSFORMS, local_dict={"x": _X, "k": _K}
    )
    return sympy.Poly(sympy.expand(expr), _X)


def _asc(poly, degree):
    return [poly.coeff_monomial(_X**d) for d in range(degree + 1)]


def _solve(cols, target):
    basis = sympy.Matrix(cols).T
    return list(basis.LUsolve(sympy.Matrix(target)))


def _oracle(tier, q):
    """The coordinates the question's own text implies."""
    text = q.text
    if "Write q" in text or "of q =" in text or "q = " in text and "P₂" in text and "P₃" not in text:
        pass
    if "ℝ⁴" in text:
        vecs = _ints(text)
        return _solve(vecs[:2], vecs[2])
    if "2×2" in text:
        mats = [
            sympy.Matrix(ast.literal_eval(m)).reshape(1, 4).tolist()[0]
            for m in re.findall(r"\[\[[-\d, \[\]]*?\]\]", text)
        ]
        return _solve(mats[:4], mats[4])
    if "ℝ³" in text:
        vecs = _ints(text)
        return _solve(vecs[:3], vecs[3])
    if "P₃" in text:
        target = re.search(r"q = (.+?)(?: relative to B| in terms of)", text).group(1)
        a = re.search(r"\(x ([+-]) (\d+)\)", text)
        shift = int(a.group(2)) * (1 if a.group(1) == "-" else -1)
        basis = [_X * 0 + 1] + [(_X - shift) ** k for k in range(1, 4)]
        polys = [sympy.Poly(b, _X) for b in basis]
        return _solve([_asc(p, 3) for p in polys], _asc(_poly(target), 3))
    m = re.search(r"p₁ = (.+?), p₂ = (.+?) and p₃ = (.+?) form a basis .*? of q = (.+?) relative to B", text)
    if m:
        polys = [_poly(m.group(i)) for i in (1, 2, 3)]
        target = _poly(m.group(4))
    else:
        m = re.search(r"B = \{(.+?), (.+?), (.+?)\}, a basis of P₂\. Write q = (.+?) as c₁", text)
        polys = [_poly(m.group(i)) for i in (1, 2, 3)]
        target = _poly(m.group(4))
    return _solve([_asc(p, 2) for p in polys], _asc(target, 2))


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
        expected = [Fraction(int(c.p), int(c.q)) for c in _oracle(tier, q)]
        assert _coords(q.answer) == expected, (seed, q.text)


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
    easy = {len(_q("easy", s).answer) for s in range(40)}
    medium = {any(_entry(r[0]).denominator > 1 for r in _q("medium", s).answer) for s in range(40)}
    hard = Counter()
    for s in range(120):
        text = _q("hard", s).text
        hard["taylor" if "P₃" in text else "matrix" if "2×2" in text else "plane" if "ℝ⁴" in text else "other"] += 1
    assert easy == {3}
    assert medium == {True}  # medium always has at least one fractional coordinate
    assert hard["taylor"] >= 20 and hard["matrix"] >= 15 and hard["plane"] >= 15 and hard["other"] == 0


def test_easy_coordinates_are_whole_and_not_the_vector_itself():
    for seed in range(120):
        q = _q("easy", seed)
        values = _coords(q.answer)
        assert all(v.denominator == 1 for v in values)
        assert tuple(int(v) for v in values) != _ints(q.text)[3]


def test_answer_variety():
    for tier in TIERS:
        texts = {_q(tier, s).answer_text for s in range(80)}
        assert len(texts) >= 60, (tier, len(texts))


# -- exam -------------------------------------------------------------------------------------------------------------


def _exam_oracle(q):
    text = q.stem
    if "ℝ³" in text:
        m = re.search(r"u₁ = (\(.+?\)), u₂ = (\(.+?\)) and u₃ = (\(.+?\)), where", text)
        vecs = [
            [
                parse_expr(c.replace("k", "*k").replace("-*", "-") if False else c, local_dict={"k": _K})
                for c in v.strip("()").split(", ")
            ]
            for v in m.groups()
        ]
        degree = 2
    else:
        m = re.search(r"p₁ = (.+?), p₂ = (.+?) and p₃ = (.+?), where", text)
        vecs = [_asc(_poly(g), 2) for g in m.groups()]
    matrix = sympy.Matrix(vecs).T
    det = sympy.expand(matrix.det())
    (k0,) = sympy.solve(det, _K)
    mm = int(re.search(r"when k = (-?\d+)", q.parts[1].prompt).group(1))
    at = matrix.subs(_K, mm)
    return k0, int(det.subs(_K, mm)), at


def test_exam_keys_match_independent_derivation():
    for seed in range(100):
        q = _exam(seed)
        k0, det_m, at = _exam_oracle(q)
        assert int(q.parts[0].answer) == k0
        assert int(q.parts[1].answer) == det_m
        target_text = re.search(r"of [vq] = (.+?) relative to it", q.parts[2].prompt).group(1)
        if "ℝ³" in q.stem:
            target = list(_ints(target_text)[0]) if target_text.startswith("(") else None
        else:
            target = _asc(_poly(target_text), 2)
        assert _coords(q.parts[2].answer) == [Fraction(int(c.p), int(c.q)) for c in at.LUsolve(sympy.Matrix(target))]
        coords = [int(c) for c in re.search(r"coordinates \(([-\d, ]+)\)", q.parts[3].prompt).group(1).split(",")]
        comp = sympy.Matrix(coords).T * at.T
        names = ["first", "second", "third", "constant term", "coefficient of x²", "coefficient of x"]
        positions = [0, 1, 2, 0, 2, 1]
        index = next(positions[i] for i, name in enumerate(names) if name in q.parts[3].prompt)
        assert int(q.parts[3].answer) == int(comp[index])


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [3, 1, 3, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "matrix", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, False]
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


def test_exam_follow_through_from_a_wrong_parameter():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        d1 = int(re.match(r"(-?\d+)\*", q.parts[1].follow_through).group(1))
        m = int(re.search(r"when k = (-?\d+)", q.parts[1].prompt).group(1))
        wrong = int(q.parts[0].answer) + 2
        if wrong in (m, 0):
            continue
        reply = {**canon, "a": str(wrong), "b": str(d1 * (m - wrong))}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, result)
        checked += 1
    assert checked >= 40


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(200)]
    assert len(set(answers)) >= 150
    assert max(Counter(answers).values()) <= 5
