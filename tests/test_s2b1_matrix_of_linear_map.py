"""Tests `uc2-matrix-of-linear-map` (topic `s2b1_matrix_of_linear_map`) and its exam chain `exd_x_s2b1_map_matrix`.

The oracle reads the map's formula and the bases back out of the question TEXT and re-derives the matrix with SymPy:
it applies the formula to each basis element and solves for the coordinates in the target basis.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2b1_matrix_of_linear_map_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "s2b1_matrix_of_linear_map"
EXAM = "exd_x_s2b1_map_matrix"
SKILL_ID = "uc2-matrix-of-linear-map"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_X, _Y = sympy.symbols("x y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _entry(e):
    return Fraction(e["p"], e["q"]) if isinstance(e, dict) else Fraction(e)


def _rows(answer):
    return [[_entry(e) for e in row] for row in answer]


def _frac_rows(matrix):
    return [[Fraction(int(v.p), int(v.q)) for v in matrix.row(i)] for i in range(matrix.rows)]


def _ints(text):
    return [[int(v) for v in g.split(",")] for g in re.findall(r"\(([-\d, ]+)\)", text)]


def _matrices(text):
    out = []
    for literal in re.findall(r"\[\[[-\d, /\[\]]*?\]\]", text):
        rows = ast.literal_eval(re.sub(r"(-?\d+)/(\d+)", r"'\1/\2'", literal))
        out.append(sympy.Matrix([[sympy.Rational(str(v)) for v in row] for row in rows]))
    return out


def _poly_coeffs(text, degree=2):
    expr = parse_expr(text.replace("²", "**2").replace("³", "**3"), transformations=_TRANSFORMS, local_dict={"x": _X})
    poly = sympy.Poly(sympy.expand(expr), _X)
    return [poly.coeff_monomial(_X**d) for d in range(degree + 1)]


def _apply_description(description, poly):
    """Evaluate a formula such as "p'(x) + 2p(x)" for the polynomial `poly`."""
    expr = description.replace("p'(", "dp(")
    expr = re.sub(r"(\d)(?=[xpd])", r"\1*", expr)
    expr = re.sub(r"x (?=dp)", "x*", expr)
    namespace = {"p": lambda t: poly.subs(_X, t), "dp": lambda t: sympy.diff(poly, _X).subs(_X, t), "x": _X}
    return eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307  (test oracle over generator-written text)


def _basis_set(text, label):
    group = re.search(label + r" = \{(.+?)\}(?: |,|\.|$)", text).group(1)
    return re.split(r", (?=[-\dx(])", group)


def _solve(ccols, image):
    return list(sympy.Matrix(ccols).T.LUsolve(sympy.Matrix(image)))


def _matrix_from_images(images, ccols):
    return sympy.Matrix([_solve(ccols, image) for image in images]).T


def _oracle(q):
    text = q.text
    if "ℝ³" in text and "standard" in text:
        (given,) = _matrices(text)
        p = sympy.Matrix(_ints(text)).T
        return p.inv() * given * p if "has matrix A" in text or "matrix A =" in text else p * given * p.inv()
    if "e^(" in text or "eˣ" in text:
        lam = 1 if "eˣ" in text else int(re.search(r"e\^\((-?\d+)x\)", text).group(1))
        shift = re.search(r"f' ([+-]) (\d+)f", text)
        shift = int(shift.group(2)) * (1 if shift.group(1) == "+" else -1)
        t = lam + shift
        return sympy.Matrix([[t, 0, 0], [2, t, 0], [0, 1, t]])
    if "P₃" in text:
        a = re.search(r"\(x ([+-]) (\d+)\)", text)
        shift = int(a.group(2)) * (1 if a.group(1) == "-" else -1)
        images = [_poly_coeffs(str(sympy.diff((_X - shift) ** k, _X))) for k in range(4)]
        return _matrix_from_images(images, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if "→ ℝ²" in text and "P₂" in text:
        formula = re.search(r"T\(p\) = \((.+?)\)\. ", text).group(1)
        bset = [_poly_coeffs(g) for g in _basis_set(text, "B")]
        c = _ints(text.split("basis C = ")[-1] if "basis C = " in text else text.split("C = ")[-1])[:2]
        images = []
        for col in bset:
            poly = sum(sympy.Integer(v) * _X**k for k, v in enumerate(col))
            l1, l2 = eval(
                "(" + re.sub(r"(\d)(?=[xpd])", r"\1*", formula.replace("p'(", "dp(")) + ")",
                {"__builtins__": {}},
                {
                    "p": lambda t, poly=poly: poly.subs(_X, t),
                    "dp": lambda t, poly=poly: sympy.diff(poly, _X).subs(_X, t),
                },
            )
            images.append([l1, l2])
        return _matrix_from_images(images, c)
    if "P₂" in text:
        description = re.search(r"T\(p\)\(x\) = (.+?)\. Find", text).group(1)
        bset = [_poly_coeffs(g) for g in _basis_set(text, "B")]
        images = [
            _poly_coeffs(
                str(
                    sympy.expand(
                        _apply_description(description, sum(sympy.Integer(v) * _X**k for k, v in enumerate(col)))
                    )
                )
            )
            for col in bset
        ]
        return _matrix_from_images(images, bset)
    # easy: plane map
    formula = re.search(r"T\(x, y\) = \((.+?)\)(?:\.|,)", text).group(1)
    first, second = formula.split(", ")
    f1, f2 = (parse_expr(s, transformations=_TRANSFORMS, local_dict={"x": _X, "y": _Y}) for s in (first, second))
    vecs = _ints(text.split("B = ")[-1])[:2] if "B = {" in text else None
    b = vecs
    images = [[f1.subs({_X: v[0], _Y: v[1]}), f2.subs({_X: v[0], _Y: v[1]})] for v in b]
    return _matrix_from_images(images, b)


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
        assert _rows(q.answer) == _frac_rows(_oracle(q)), (seed, q.text)


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
    easy = {(len(_q("easy", s).answer), len(_q("easy", s).answer[0])) for s in range(30)}
    medium = Counter(("P₃" in _q("medium", s).text) for s in range(80))
    hard = Counter()
    for s in range(160):
        text = _q("hard", s).text
        hard[
            "functionals" if "→ ℝ²" in text else "exponential" if ("e^(" in text or "eˣ" in text) else "similarity"
        ] += 1
    assert easy == {(2, 2)}
    assert medium[True] >= 10 and medium[False] >= 40
    assert min(hard.values()) >= 20 and len(hard) == 3


def test_answer_variety():
    for tier in TIERS:
        texts = {_q(tier, s).answer_text for s in range(80)}
        assert len(texts) >= (20 if tier == "hard" else 45), (tier, len(texts))


# -- exam -------------------------------------------------------------------------------------------------------------


def test_exam_keys_match_independent_derivation():
    for seed in range(100):
        q = _exam(seed)
        description = re.search(r"T\(p\)\(x\) = (.+?), where P₂", q.stem).group(1)
        bset = [_poly_coeffs(g) for g in re.search(r"B = \{(.+?)\}, both", q.stem).group(1).split(", ")]
        std = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        a = _matrix_from_images(
            [_poly_coeffs(str(sympy.expand(_apply_description(description, _X**k)))) for k in range(3)], std
        )
        m = _matrix_from_images(
            [
                _poly_coeffs(
                    str(
                        sympy.expand(
                            _apply_description(description, sum(sympy.Integer(v) * _X**k for k, v in enumerate(col)))
                        )
                    )
                )
                for col in bset
            ],
            bset,
        )
        assert int(q.parts[0].answer) == a.trace()
        assert _rows(q.parts[1].answer) == _frac_rows(m)
        assert int(q.parts[2].answer) == m.trace() == a.trace()
        assert int(q.parts[3].answer) == m.det() == a.det()


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [3, 4, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "matrix", "number", "number"]
        assert [p.follow_through for p in q.parts] == [None, None, "part_a", None]
        assert find_text_defects(q) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(60):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 10
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_follow_through_from_a_wrong_trace():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        wrong = int(q.parts[0].answer) + 2
        if wrong == 0 or str(wrong) in q.parts[2].distractors:
            continue
        result = grade(_SKILL, q, {**canon, "a": str(wrong), "c": str(wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, result)
        checked += 1
    assert checked >= 40


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(200)]
    assert len(set(answers)) >= 150
