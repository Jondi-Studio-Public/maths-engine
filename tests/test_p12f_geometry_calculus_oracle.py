"""Independent oracle for the `p12f_geometry_calculus` questions: it re-derives each answer from the rendered
question text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p12f_geometry_calculus_build`) and re-derives each
answer from scratch: the Gram-Schmidt route via plain `math`/`fractions`
projection arithmetic, and the Hessian route via `sympy.diff` applied to a
polynomial parsed straight out of the rendered text.
"""

from dataclasses import replace
import math
import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen
from mathgen.skills import load_skills

GRAM_ID = "ucv1-gram-schmidt-2d"
HESS_ID = "ucv1-hessian-2x2"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

# ---------------------------------------------------------------------------
# ucv1-gram-schmidt-2d
# ---------------------------------------------------------------------------

_U_RE = re.compile(r"u = \((-?\d+), (-?\d+)\)")
_V_RE = re.compile(r"[^_]v = \((-?\d+), (-?\d+)\)")
_VA_RE = re.compile(r"v_A = \((-?\d+), (-?\d+)\)")
_VB_RE = re.compile(r"v_B = \((-?\d+), (-?\d+)\)")
_ANSWER4_RE = re.compile(r"(-?\d+(?:/\d+)?), (-?\d+(?:/\d+)?), (-?\d+(?:/\d+)?), (-?\d+(?:/\d+)?)")


def _project(ux, uy, vx, vy):
    """Textbook Gram-Schmidt projection, exact Fraction arithmetic: the same
    route the generator's OWN independent check uses, but re-derived here
    with no import from the generator module at all."""
    dot_uv = ux * vx + uy * vy
    dot_uu = ux * ux + uy * uy
    proj = Fraction(dot_uv, dot_uu)
    w2x = Fraction(vx) - proj * ux
    w2y = Fraction(vy) - proj * uy
    mag_sq = w2x * w2x + w2y * w2y
    root_num = math.isqrt(mag_sq.numerator)
    root_den = math.isqrt(mag_sq.denominator)
    assert root_num * root_num == mag_sq.numerator, mag_sq
    assert root_den * root_den == mag_sq.denominator, mag_sq
    mag = Fraction(root_num, root_den)
    return w2x / mag, w2y / mag


def _e1(ux, uy):
    c_sq = ux * ux + uy * uy
    c = math.isqrt(c_sq)
    assert c * c == c_sq, (ux, uy)
    return Fraction(ux, c), Fraction(uy, c)


def _solve_gram_schmidt(text):
    um = _U_RE.search(text)
    assert um, text
    ux, uy = int(um.group(1)), int(um.group(2))
    e1 = _e1(ux, uy)

    vam, vbm = _VA_RE.search(text), _VB_RE.search(text)
    if vam and vbm:
        # exam tier: two candidates, exactly one gives e2 with positive x.
        va = tuple(int(g) for g in vam.groups())
        vb = tuple(int(g) for g in vbm.groups())
        e2_a = _project(ux, uy, *va)
        e2_b = _project(ux, uy, *vb)
        consistent = [e2 for e2 in (e2_a, e2_b) if e2[0] > 0]
        assert len(consistent) == 1, (text, e2_a, e2_b)
        return (*e1, *consistent[0])

    vm = _V_RE.search(text)
    assert vm, text
    vx, vy = int(vm.group(1)), int(vm.group(2))
    return (*e1, *_project(ux, uy, vx, vy))


def _parse_answer4(answer_text):
    m = _ANSWER4_RE.fullmatch(answer_text)
    assert m, answer_text
    return tuple(Fraction(g) for g in m.groups())


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


GRAM_SOURCES = _skill_sources(GRAM_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
GRAM_SOURCES["exam"] = replace(GRAM_SOURCES["exam"], topic="p12f_gram_schmidt_2d", params={})
HESS_SOURCES = _skill_sources(HESS_ID)


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_gram_schmidt_answer(difficulty, seed):
    src = GRAM_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve_gram_schmidt(q.text)
    stated = _parse_answer4(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


# ---------------------------------------------------------------------------
# ucv1-hessian-2x2
# ---------------------------------------------------------------------------

_F_KNOWN_RE = re.compile(r"f\(x, y\) = (.+?) \+ E x\^2 y")
_F_FULL_RE = re.compile(r"f\(x, y\) = (.+?)(?:,|\.\s|\s+has a Hessian matrix|\s+at the point|\s+depends)")
_POINT_RE = re.compile(r"\((-?\d+), (-?\d+)\)")
_FXY_RE = re.compile(r"f_xy(?:\s*=\s*|\s+at\s*\([^)]*\)\s+equals\s+)(-?\d+)")

_X, _Y = sympy.symbols("x y")
_T = standard_transformations + (convert_xor, implicit_multiplication_application)


def _parse_poly(expr_text):
    return parse_expr(expr_text, local_dict={"x": _X, "y": _Y}, transformations=_T)


def _solve_hessian(text):
    known_m = _F_KNOWN_RE.search(text)
    point_m = _POINT_RE.search(text)
    assert point_m, text
    x0, y0 = int(point_m.group(1)), int(point_m.group(2))

    if known_m:
        # exam tier: E is unknown in f's own formula; recover it from the
        # stated f_xy value via a fresh symbolic solve (not the generator's
        # own closed-form arithmetic), then rebuild the full polynomial.
        known_expr = _parse_poly(known_m.group(1))
        fxy_m = _FXY_RE.search(text)
        assert fxy_m, text
        fxy_given = int(fxy_m.group(1))
        b_known = int(sympy.diff(known_expr, _X, _Y).subs({_X: 0, _Y: 0}))
        e_sym = sympy.Symbol("e_unknown")
        solutions = sympy.solve(sympy.Eq(b_known + 2 * e_sym * x0, fxy_given), e_sym)
        assert len(solutions) == 1, solutions
        e_val = solutions[0]
        assert e_val.is_integer, e_val
        full_expr = known_expr + int(e_val) * _X**2 * _Y
    else:
        full_m = _F_FULL_RE.search(text)
        assert full_m, text
        full_expr = _parse_poly(full_m.group(1))

    fxx = int(sympy.diff(full_expr, _X, _X).subs({_X: x0, _Y: y0}))
    fxy = int(sympy.diff(full_expr, _X, _Y).subs({_X: x0, _Y: y0}))
    fyx = int(sympy.diff(full_expr, _Y, _X).subs({_X: x0, _Y: y0}))
    fyy = int(sympy.diff(full_expr, _Y, _Y).subs({_X: x0, _Y: y0}))
    return (fxx, fxy, fyx, fyy)


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_hessian_answer(difficulty, seed):
    src = HESS_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve_hessian(q.text)
    stated = tuple(int(v) for v in q.answer)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS) * 2  # both skills
    assert len(SEEDS) >= 60
    assert total >= 480
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds x 2 skills).")
