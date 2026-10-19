"""Tests `uc2-kernel-and-image` (topic `s2b1_kernel_and_image`) and its exam chain `exd_x_s2b1_kernel_image`.

The oracle reads the map's formula back out of the question TEXT, evaluates it on a basis of the domain with SymPy and
takes the rank and nullity of the resulting matrix (or, for the parameter questions, solves for k).
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy

import mathgen.generators.s2b1_kernel_and_image_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "s2b1_kernel_and_image"
EXAM = "exd_x_s2b1_kernel_image"
SKILL_ID = "uc2-kernel-and-image"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_X, _K = sympy.symbols("x k")
_SUBSCRIPT = "₀₁₂₃₄₅₆₇₈₉"


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _python(formula):
    expr = formula.replace("p''(", "ddp(").replace("p'(", "dp(").replace("Xᵀ", "X.T").replace("tr(X)·I", "X.trace()*I")
    expr = re.sub(r"(\d)(?=[xpdXA(])", r"\1*", expr)
    expr = re.sub(r"x (?=dp)", "x*", expr)
    expr = re.sub(r"\)(?=[xpdXA])", ")*", expr)
    expr = re.sub(r"AX", "A*X", expr).replace("XA", "X*A").replace("XB", "X*B")
    return expr.replace("A*X - X*A", "A*X - X*A")


def _matrix_of(formula, domain, extra=None):
    """The matrix of the map `formula` on the monomial basis (polynomial domain) or E11, E12, E21, E22."""
    expr = _python(formula)
    columns = []
    if domain.startswith("P"):
        n = _SUBSCRIPT.index(domain[1])
        for j in range(n + 1):
            poly = _X**j
            env = {
                "p": lambda t, poly=poly: poly.subs(_X, t),
                "dp": lambda t, poly=poly: sympy.diff(poly, _X).subs(_X, t),
                "ddp": lambda t, poly=poly: sympy.diff(poly, _X, 2).subs(_X, t),
                "x": _X,
            }
            value = eval(expr, {"__builtins__": {}}, env)  # noqa: S307  (test oracle over generator-written text)
            if isinstance(value, tuple):
                columns.append([sympy.sympify(v) for v in value])
            else:
                poly_out = sympy.Poly(sympy.expand(value), _X)
                columns.append([poly_out.coeff_monomial(_X**d) for d in range(n + 1)])
        return sympy.Matrix(columns).T
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        e = sympy.zeros(2, 2)
        e[i, j] = 1
        env = {"X": e, "A": extra, "I": sympy.eye(2)}
        value = eval(expr, {"__builtins__": {}}, env)  # noqa: S307
        columns.append([value[0, 0], value[0, 1], value[1, 0], value[1, 1]])
    return sympy.Matrix(columns).T


def _matrices(text):
    out = []
    for literal in re.findall(r"\[\[[-\d, \[\]]*?\]\]", text):
        out.append(sympy.Matrix(ast.literal_eval(literal)))
    return out


def _oracle(tier, q):
    text = q.text
    domain = re.search(r"T: (P[₀-₉]|M₂\(ℝ\)) →", text).group(1)
    desc = re.search(r"(?:with|map) (T\(p\)\(x\) = .+?|T\(p\) = \(.+?\)|T\(X\) = .+?)\. (?:P|M)", text).group(1)
    formula = desc.split(" = ", 1)[1]
    extra = None
    if ", where A =" in formula:
        extra = _matrices(formula)[0]
        formula = formula.split(", where")[0]
    if "where I is" in formula:
        formula = formula.split(", where")[0]
    if formula.startswith("(") and domain.startswith("P"):
        formula = formula  # a tuple of functionals
    m = _matrix_of(formula, domain, extra)
    nullity, rank = len(m.nullspace()), m.rank()
    return nullity if "kernel" in text.split("Find")[-1].split("What")[-1] or "kernel" in text[-60:] else rank


def test_skill_wiring():
    assert _SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert "university-year-2" in _SKILL.tags


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_key_matches_independent_dimension(tier):
    for seed in range(150):
        q = _q(tier, seed)
        assert int(q.answer) == _oracle(tier, q), (seed, q.text)


def test_hard_key_matches_independent_derivation():
    kinds = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        text = q.text
        if "T_k: P₂ → ℝ³" in text or "T_k: P₂" in text or "→ ℝ³ by" in text:
            body = re.search(r"T_k\(p\) = \((.+)\)(?:, where| ?\. For which|\. Find)", text) or re.search(
                r"T_k\(p\) = \((.+)\)", text
            )
            rows = _functional_rows(body.group(1))
            assert sympy.solve(sympy.Matrix(rows).det(), _K) == [int(q.answer)], (seed, text)
            kinds["parameter"] += 1
        elif "AX - kX" in text:
            a = _matrices(text)[0]
            roots = sorted(sympy.solve((a - _K * sympy.eye(2)).det(), _K))
            assert roots[-1] == int(q.answer), (seed, text)
            kinds["eigen"] += 1
        else:
            a, b = _matrices(text)
            m = _matrix_of("A*X - X*B".replace("A*X - X*B", "AX - XB"), "M", None) if False else None
            cols = []
            for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                e = sympy.zeros(2, 2)
                e[i, j] = 1
                v = a * e - e * b
                cols.append([v[0, 0], v[0, 1], v[1, 0], v[1, 1]])
            assert len(sympy.Matrix(cols).T.nullspace()) == int(q.answer), (seed, text)
            kinds["sylvester"] += 1
    assert min(kinds.values()) >= 20 and len(kinds) == 3


def _functional_rows(body):
    """Rows of the matrix of x -> (L1, L2, L3 + k L4) in the basis 1, x, x^2, from the text of the three components."""
    pieces = re.split(r", (?=p)", body)
    rows = []
    for piece in pieces:
        expr = piece.replace("k·(", "k*(").replace("k·", "k*")
        row = []
        for j in range(3):
            poly = _X**j
            env = {
                "p": lambda t, poly=poly: poly.subs(_X, t),
                "dp": lambda t, poly=poly: sympy.diff(poly, _X).subs(_X, t),
                "k": _K,
            }
            row.append(eval(expr.replace("p'(", "dp("), {"__builtins__": {}}, env))  # noqa: S307
        rows.append(row)
    return rows


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
    easy_domains = Counter(re.search(r"T: (P[₀-₉]|M₂\(ℝ\)) →", _q("easy", s).text).group(1) for s in range(80))
    medium_forms = Counter("AX" if "AX" in _q("medium", s).text else "p" for s in range(80))
    hard = Counter()
    for s in range(120):
        text = _q("hard", s).text
        hard["eigen" if "AX - kX" in text else "sylvester" if "AX - XB" in text else "parameter"] += 1
    assert len(easy_domains) >= 3
    assert medium_forms["AX"] >= 10 and medium_forms["p"] >= 20
    assert min(hard.values()) >= 20 and len(hard) == 3


def test_answer_variety():
    for tier in ("easy", "medium"):
        texts = {_q(tier, s).text for s in range(80)}
        assert len(texts) >= 40, (tier, len(texts))
        assert len({_q(tier, s).answer_text for s in range(80)}) >= 3
    hard_answers = Counter(_q("hard", s).answer_text for s in range(120))
    assert len(hard_answers) >= 8


# -- exam -------------------------------------------------------------------------------------------------------------


def test_exam_keys_match_independent_derivation():
    for seed in range(100):
        q = _exam(seed)
        body = re.search(r"T_k\(p\) = \((.+)\), where P₂", q.stem).group(1)
        rows = _functional_rows(body)
        matrix = sympy.Matrix(rows)
        (k0,) = sympy.solve(matrix.det(), _K)
        at = matrix.subs(_K, k0)
        assert int(q.parts[0].answer) == k0
        assert int(q.parts[1].answer) == at.rank() == 2
        assert int(q.parts[2].answer) == len(at.nullspace()) == 1
        kernel = at.nullspace()[0]
        kernel = kernel / kernel[2]
        expected = Fraction(int(kernel[0].p), int(kernel[0].q))
        answer = q.parts[3].answer
        assert (Fraction(answer["p"], answer["q"]) if isinstance(answer, dict) else Fraction(answer)) == expected


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [3, 2, 1, 3]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "fraction"]
        assert [p.follow_through for p in q.parts] == [None, None, "3 - part_b", None]
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


def test_exam_follow_through_from_a_wrong_rank():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong_rank in (1, 4):
            reply = {**canon, "b": str(wrong_rank), "c": str(3 - wrong_rank)}
            if reply["c"] in q.parts[2].distractors:
                continue
            result = grade(_SKILL, q, reply)
            assert result.parts["b"].marks == 0
            assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, wrong_rank, result)
            checked += 1
    assert checked >= 40


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(200)]
    assert len(set(answers)) >= 45
