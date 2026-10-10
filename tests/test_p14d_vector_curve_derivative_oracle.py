"""Independent oracle for the `p14d_vector_curve_derivative` questions: it re-derives each answer from the
rendered question text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p14d_vector_curve_derivative_build`) and re-derives the
derivative pair from scratch: parse the two component polynomials straight
out of the rendered text, differentiate with `sympy.diff`, substitute the
stated t0, and compare against the stated answer.
"""

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

SKILL_ID = "ucv1-vector-valued-derivative"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_T = sympy.Symbol("t")
_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)

_FRAC_RE = r"-?\d+(?:/\d+)?"
_RC_PAIR_RE = re.compile(rf"r\(t\) = \((.+?), (.+?)\)")
_XY_RE = re.compile(r"x\(t\) = (.+?) and y\(t\) = (.+?)[,.]")
_T0_RE = re.compile(rf"t = ({_FRAC_RE})\.?\s*$")
_ANSWER_RE = re.compile(rf"\(({_FRAC_RE}), ({_FRAC_RE})\)")


def _to_fraction(text: str) -> Fraction:
    text = text.strip()
    if "/" in text:
        num, den = text.split("/")
        return Fraction(int(num), int(den))
    return Fraction(int(text))


def _parse_poly(expr_text: str):
    return parse_expr(expr_text, local_dict={"t": _T}, transformations=_TRANSFORMS)


def _solve(text: str):
    t0_m = _T0_RE.search(text)
    assert t0_m, text
    t0 = sympy.Rational(_to_fraction(t0_m.group(1)).numerator, _to_fraction(t0_m.group(1)).denominator)

    xy_m = _XY_RE.search(text)
    if xy_m:
        x_text, y_text = xy_m.groups()
        x_expr = _parse_poly(x_text)
        y_expr = _parse_poly(y_text)
    else:
        rc_m = _RC_PAIR_RE.search(text)
        assert rc_m, text
        x_text, y_text = rc_m.groups()
        x_expr = _parse_poly(x_text)
        y_expr = _parse_poly(y_text)

    dx = sympy.diff(x_expr, _T).subs(_T, t0)
    dy = sympy.diff(y_expr, _T).subs(_T, t0)
    return sympy.nsimplify(dx), sympy.nsimplify(dy)


def _parse_answer(answer_text: str):
    m = _ANSWER_RE.fullmatch(answer_text)
    assert m, answer_text
    return tuple(sympy.Rational(Fraction(g).numerator, Fraction(g).denominator) for g in m.groups())


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources(SKILL_ID)


@pytest.mark.parametrize("difficulty,seed", [(t, s) for t in TIERS for s in SEEDS])
def test_oracle_rederives_vector_curve_derivative(difficulty, seed):
    src = SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    stated = _parse_answer(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 240
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds).")
