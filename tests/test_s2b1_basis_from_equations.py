"""Tests `uc2-basis-from-equations` (topic `s2b1_basis_from_equations`) and its exam chain `exd_x_s2b1_basis_equations`.

The oracle reads the equations, conditions or generators back out of the question TEXT and re-derives the basis with
SymPy's `nullspace()` (which returns one vector per free variable, in order: the normal form the stems ask for) or, for
the hard intersections, the reduced row echelon form of an intersection basis.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2b1_basis_from_equations_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "s2b1_basis_from_equations"
EXAM = "exd_x_s2b1_basis_equations"
SKILL_ID = "uc2-basis-from-equations"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_X, _K = sympy.symbols("x k")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SUB = {c: str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _entry(e):
    return Fraction(e["p"], e["q"]) if isinstance(e, dict) else Fraction(e)


def _rows(answer):
    return [[_entry(e) for e in row] for row in answer]


def _frac(value):
    return Fraction(int(value.p), int(value.q))


def _ascii(text):
    return "".join(_SUB.get(c, c) for c in text)


def _vectors(text):
    return [[int(v) for v in g.split(",")] for g in re.findall(r"\(([-\d, ]+)\)", text)]


def _equation_rows(text, n):
    """Rows of the coefficient matrix of every `... = 0` equation in `text` over x1..xn (k left symbolic)."""
    xs = sympy.symbols(f"x1:{n + 1}")
    local = {f"x{i + 1}": xs[i] for i in range(n)} | {"k": _K}
    rows = []
    term = r"[-+]?\s*(?:\d*|\(k [+-] \d+\)|k)x\d"
    for match in re.finditer(rf"({term}(?:\s*{term})*) = 0", _ascii(text)):
        expr = parse_expr(match.group(1).replace("kx", "k*x"), transformations=_TRANSFORMS, local_dict=local)
        rows.append([sympy.expand(expr).coeff(x) for x in xs])
    return rows


def _poly_rows(text):
    body = re.search(r"(?:satisfying|\{p : )(.+?)(?:\}|\. )", text).group(1)
    rows = []
    for cond in body.split(" and "):
        left, right = cond.split(" = ")
        expr = f"({left}) - ({right})".replace("p''(", "ddp(").replace("p'(", "dp(")
        row = []
        for j in range(4):
            poly = _X**j
            env = {
                "p": lambda t, poly=poly: poly.subs(_X, t),
                "dp": lambda t, poly=poly: sympy.diff(poly, _X).subs(_X, t),
                "ddp": lambda t, poly=poly: sympy.diff(poly, _X, 2).subs(_X, t),
            }
            row.append(eval(re.sub(r"(\d)(?=[pd])", r"\1*", expr), {"__builtins__": {}}, env))  # noqa: S307
        rows.append(row)
    return rows


def _matrix_rows(text):
    body = re.search(r"(?:for which|A for which) (.+?)\. (?:Work|Use)", text).group(1)
    bm = re.search(r"B = \[\[(-?\d+), (-?\d+)\], \[(-?\d+), (-?\d+)\]\]", body)
    a, b, c, d = sympy.symbols("a b c d")
    m = sympy.Matrix([[a, b], [c, d]])
    rows = []
    for cond in body.split(" and "):
        if "trace" in cond:
            exprs = [m.trace()]
        elif "symmetric" in cond:
            exprs = [b - c]
        elif "first row" in cond:
            exprs = [a + b]
        elif "second row" in cond:
            exprs = [c + d]
        elif "diagonal entries" in cond:
            exprs = [a - d]
        else:
            bmat = sympy.Matrix([[int(bm.group(1)), int(bm.group(2))], [int(bm.group(3)), int(bm.group(4))]])
            diff = m * bmat - bmat * m
            exprs = list(diff)
        for e in exprs:
            rows.append([sympy.expand(e).coeff(v) for v in (a, b, c, d)])
    return rows


def _rref_rows(vectors):
    reduced, pivots = sympy.Matrix(vectors).rref()
    return [list(reduced.row(i)) for i in range(len(pivots))]


def _intersection_rref(gens_u, gens_w):
    u, w = sympy.Matrix(gens_u).T, sympy.Matrix(gens_w).T
    vectors = [list(u * sol[: u.cols, :]) for sol in u.row_join(-w).nullspace()]
    return _rref_rows(vectors)


def _expected(tier, q):
    text = q.text
    if "span" in text:
        if "U = {x :" in text or "defined by" in text:
            eq = _equation_rows(text.split("W =")[0], 4)
            gens = _vectors(text.split("W = span")[-1].split(". Find")[0])
            null = [list(v) for v in sympy.Matrix(eq).nullspace()]
            return _intersection_rref(gens, null)
        parts = re.findall(r"span\{(.+?)\}", text)
        return _intersection_rref(_vectors(parts[0]), _vectors(parts[1]))
    if "orthogonal" in text or "x · v" in text:
        rows = _vectors(text.split(". Give")[0])
        n = 5
    elif "a₀" in text:
        rows, n = _poly_rows(text), 4
    elif "[[a, b]" in text:
        rows, n = _matrix_rows(text), 4
    else:
        n = 5 if "ℝ⁵" in text else 4
        rows = _equation_rows(text.split(". Find")[0].split(". Give")[0], n)
    return [list(v) for v in sympy.Matrix(rows).nullspace()]


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
        expected = [[_frac(v) for v in row] for row in _expected(tier, q)]
        assert _rows(q.answer) == expected, (seed, q.text)


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


def test_stem_names_the_normal_form():
    for tier in TIERS:
        for seed in range(40):
            text = _q(tier, seed).text
            assert "free variable" in text or "reduced row echelon form" in text, (tier, seed)


def test_tiers_differ_in_shape():
    easy = Counter(len(_q("easy", s).answer) for s in range(40))
    medium = Counter()
    for s in range(120):
        text = _q("medium", s).text
        medium[
            "poly"
            if "a₀" in text
            else "matrix"
            if "[[a, b]" in text
            else "orthogonal"
            if ("orthogonal" in text or "x · v" in text)
            else "dependent"
        ] += 1
    hard = Counter(
        "equation" if ("U = {x" in _q("hard", s).text or "defined by" in _q("hard", s).text) else "spans"
        for s in range(80)
    )
    assert set(easy) == {2}
    assert min(medium.values()) >= 8 and len(medium) == 4
    assert min(hard.values()) >= 15 and len(hard) == 2


def test_answer_variety():
    for tier in TIERS:
        texts = {_q(tier, s).answer_text for s in range(80)}
        assert len(texts) >= 40, (tier, len(texts))


# -- exam -------------------------------------------------------------------------------------------------------------


def _exam_rows(q):
    return sympy.Matrix(_equation_rows(q.stem, 4))


def test_exam_keys_match_independent_derivation():
    for seed in range(100):
        q = _exam(seed)
        matrix = _exam_rows(q)
        ranks = {k: matrix.subs(_K, k).rank() for k in range(-15, 16)}
        low = [k for k, r in ranks.items() if r == 2]
        assert low == [int(q.parts[0].answer)], (seed, q.stem)
        k0 = low[0]
        at = matrix.subs(_K, k0)
        assert int(q.parts[1].answer) == 4 - at.rank() == 2
        assert _rows(q.parts[2].answer) == [[_frac(v) for v in vec] for vec in [list(b) for b in at.nullspace()]]
        assert int(q.parts[3].answer) == 4 - matrix.subs(_K, k0 + 1).rank() == 1


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [3, 2, 3, 1]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "matrix", "number"]
        assert [p.follow_through for p in q.parts] == [None, None, None, "part_b - 1"]
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


def test_exam_follow_through_from_a_wrong_dimension():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong in (3, 4):
            reply = {**canon, "b": str(wrong), "d": str(wrong - 1)}
            if reply["d"] in q.parts[3].distractors:
                continue
            result = grade(_SKILL, q, reply)
            assert result.parts["b"].marks == 0
            assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 20


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(200)]
    assert len(set(answers)) >= 150
