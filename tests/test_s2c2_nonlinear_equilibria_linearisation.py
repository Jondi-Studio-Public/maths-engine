"""Tests `uc2-nonlinear-equilibria-linearisation`: the topic `s2c2_nonlinear_equilibria_linearisation` (easy, medium,
hard) and the exam chain `exd_x_s2c2_nonlinear_equilibria_linearisation`.

The oracle reads the vector field out of the question text and uses SymPy to solve for the equilibria, differentiate
the field, evaluate the Jacobian and find eigenvalues and eigenvectors, a route that shares nothing with the
generator's hand-built formulas.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2c2_nonlinear_equilibria_linearisation_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILL_ID = "uc2-nonlinear-equilibria-linearisation"
TOPIC = "s2c2_nonlinear_equilibria_linearisation"
EXAM = "exd_x_s2c2_nonlinear_equilibria_linearisation"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
x, y = sympy.symbols("x y")
_T = standard_transformations + (implicit_multiplication_application,)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _stem(q):
    return q.stem if q.parts else q.text


def _read(expr_text):
    expr_text = re.sub(r"\bsin x\b", "sin(x)", expr_text)
    expr_text = re.sub(r"(?<![A-Za-z])([xy])\(", r"\1*(", expr_text)
    return parse_expr(expr_text, transformations=_T, local_dict={"x": x, "y": y})


def _field(text):
    match = re.search(r"x' = (.+?), y' = (.+?)\. ", text)
    assert match, text
    return sympy.Matrix([_read(match.group(1)), _read(match.group(2))])


def _jacobian(field, point):
    return field.jacobian([x, y]).subs({x: point[0], y: point[1]})


def _coexistence(field):
    sols = sympy.solve(list(field), [x, y], dict=True)
    good = [s for s in sols if s.get(x, 0) > 0 and s.get(y, 0) > 0]
    assert len(good) == 1, (field, sols)
    return good[0][x], good[0][y]


def _matrix_in(text):
    return sympy.Matrix(sympy.sympify(text))


def _stated_point(text):
    match = re.search(r"equilibrium «?\((\d*)π?, 0\)", text) or re.search(r"equilibrium \((\d*)(π)?, 0\)", text)
    return match


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages) == (EXAM, "exam", ["exam"])
    assert [src.difficulty for src in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


def _point_of(q):
    text = q.text
    field = _field(text)
    if "0 < x < 2π" in text:
        return field, (sympy.pi, 0)
    if "x > 0" in text and "y > 0" in text:
        return field, _coexistence(field)
    match = re.search(r"equilibrium \((?:(\d*)π|([\d/]+)), 0\)", text)
    assert match, text
    if match.group(1) is not None:
        return field, (sympy.pi * int(match.group(1) or 1), 0)
    return field, (sympy.Rational(match.group(2)), 0)


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_jacobian_matches_sympy(tier):
    for seed in range(40):
        q = _q(tier, seed)
        field, point = _point_of(q)
        assert _jacobian(field, point) == _matrix_in(q.answer_text), (seed, q.text, q.answer_text)


def test_hard_eigenvector_matches_sympy():
    for seed in range(30):
        q = _q("hard", seed)
        field, point = _point_of(q)
        jac = _jacobian(field, point)
        positive = [lam for lam in jac.eigenvals() if lam > 0]
        assert len(positive) == 1, (seed, q.text)
        vec = (jac - positive[0] * sympy.eye(2)).nullspace()[0]
        got = sympy.Matrix(sympy.sympify(q.answer_text.replace("(", "[", 1).rsplit(")", 1)[0] + "]"))
        assert got[1] == 1 and vec[0] * got[1] - vec[1] * got[0] == 0, (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(30):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


def test_hard_accepts_any_multiple_of_the_eigenvector():
    for seed in range(15):
        q = _q("hard", seed)
        vec = sympy.Matrix(sympy.sympify(q.answer_text.replace("(", "[", 1).rsplit(")", 1)[0] + "]"))
        scaled = vec * vec[0].q if vec[0] != 0 else vec
        assert grade(_SKILL, q, "(" + ", ".join(str(v) for v in scaled) + ")").correct


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_wrong_answers(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert set(q.option_notes) == set(q.distractors)
        assert all(note in CATALOGUE for note in q.option_notes.values())
        assert q.answer_text not in q.distractors
        assert len(set(q.distractors)) == len(q.distractors)


def test_the_tiers_ask_for_different_things():
    for seed in range(40):
        easy, medium, hard = (_q(tier, seed) for tier in TIERS)
        assert "Find the Jacobian matrix of the system at the equilibrium" in easy.text
        assert "x > 0" not in easy.text and "0 < x" not in easy.text
        assert "Jacobian" in medium.text and ("x > 0" in medium.text or "0 < x < 2π" in medium.text)
        assert "eigenvector" in hard.text and "positive eigenvalue" in hard.text


def test_all_three_models_appear_in_each_single_tier():
    for tier in ("easy", "medium"):
        kinds = {
            ("pendulum" if "pendulum" in _q(tier, s).text else "rabbits" if "rabbits" in _q(tier, s).text else "fish")
            for s in range(60)
        }
        assert kinds == {"pendulum", "rabbits", "fish"}


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).text for seed in range(80)]
    assert len(set(answers)) >= 30


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 1]
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "fraction", "mcq"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, False]
        assert find_text_defects(q) == []
        assert len(q.parts[3].distractors) >= 3
        assert all(len(p.distractors) >= 2 for p in q.parts)


def test_exam_keys_match_independent_derivation():
    seen = set()
    for seed in range(40):
        q = _exam(seed)
        field = _field(_stem(q) + " ")
        xs, ys = _coexistence(field)
        a, b, c, d = q.parts
        assert sympy.Rational(a.answer_text) == xs
        assert sympy.Rational(b.answer_text) == ys
        jac = _jacobian(field, (xs, ys))
        assert sympy.Rational(c.answer_text) == jac.det()
        roots = list(jac.eigenvals())
        if any(sympy.im(r) != 0 for r in roots):
            label = "stable spiral" if sympy.re(roots[0]) < 0 else "unstable spiral"
        elif roots[0] * roots[1] < 0:
            label = "saddle"
        else:
            label = "stable node" if roots[0] < 0 else "unstable node"
        assert d.answer_text == label, (seed, q.stem)
        seen.add(label)
    assert {"saddle", "stable node", "stable spiral"} <= seen


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(25):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 7
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_into_the_determinant():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        canon = _canonical(q)
        field = _field(_stem(q) + " ")
        xs, ys = _coexistence(field)
        k = q.parts[2].follow_through.split("*")[-1].strip("()")
        for wrong in q.parts[0].distractors:
            wx = sympy.Rational(wrong)
            if wx == 0 or wx == xs:
                continue
            reply_c = wx * ys * sympy.Rational(k)
            if reply_c == 0 or str(reply_c) in (canon["c"], canon["b"], wrong):
                continue
            result = grade(_SKILL, q, {**canon, "a": wrong, "c": str(reply_c)})
            assert result.parts["a"].marks == 0
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 40


def test_exam_answer_space_and_determinism():
    stems = [_exam(seed).stem for seed in range(100)]
    assert len(set(stems)) >= 80
    assert _exam(7).to_json() == _exam(7).to_json()
