"""Tests `uc2-nonhomogeneous-recurrence-polynomial-or-exponential-forcing` (topic `r4j_forced_recurrence`) and its exam chain.

The oracle reads the recurrence and the starting terms out of the TEXT with SymPy, iterates it for twenty terms, and checks
that the stated closed form reproduces every one; a closed form is also compared with a perturbed sequence to prove the
check can fail.
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.r4j_forced_recurrence_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r4j_support import check_carried, check_parts, check_single

TOPIC = "r4j_forced_recurrence"
EXAM = "exd_x_r4j_forced_recurrence"
SKILL = next(s for s in load_skills()[1] if s.id == "uc2-nonhomogeneous-recurrence-polynomial-or-exponential-forcing")
n, A1, A2 = sympy.symbols("n A1 A2")
_T = standard_transformations + (implicit_multiplication_application,)


def _read(text):
    """(a0, a1 or None, rhs as a SymPy expression in A1, A2, n)."""
    t = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉₋ₙ−·", "0123456789-n-*"))
    a0 = int(re.search(r"a0 = (-?\d+)", t).group(1))
    m = re.search(r"a1 = (-?\d+)", t)
    rhs = re.search(r"an = (.*?) for n", t).group(1).replace("an-1", "A1").replace("an-2", "A2").replace("^", "**")
    return (
        a0,
        int(m.group(1)) if m else None,
        parse_expr(rhs, local_dict={"n": n, "A1": A1, "A2": A2}, transformations=_T),
    )


def _terms(a0, a1, rhs, count=20):
    if a1 is None:
        out = [sympy.Integer(a0)]
        for k in range(1, count):
            out.append(rhs.subs({A1: out[-1], A2: 0, n: k}))
    else:
        out = [sympy.Integer(a0), sympy.Integer(a1)]
        for k in range(2, count):
            out.append(rhs.subs({A1: out[-1], A2: out[-2], n: k}))
    return out


def _matches(expr_text, terms):
    e = sympy.sympify(expr_text.replace("^", "**"), locals={"n": n})
    return all(sympy.simplify(e.subs(n, i) - t) == 0 for i, t in enumerate(terms))


def test_skill_metadata():
    assert list(SKILL.year_level) == ["university"] and "university-year-2" in SKILL.tags


def test_all_tiers_match_iteration():
    for tier in ("easy", "medium", "hard"):
        for seed in range(30):
            q = make(TOPIC, seed=seed, difficulty=tier)
            a0, a1, rhs = _read(q.text)
            terms = _terms(a0, a1, rhs)
            assert _matches(q.answer_text, terms), (tier, seed, q.text, q.answer_text)
            bumped = terms[:]
            bumped[-1] += 1
            assert not _matches(q.answer_text, bumped)
            check_single(q, "expression")


def test_tiers_use_the_forcing_they_promise():
    for seed in range(40):
        a0, a1, rhs = _read(make(TOPIC, seed=seed, difficulty="easy").text)
        assert a1 is None and rhs.has(n) and not rhs.has(A2)
        a0, a1, rhs = _read(make(TOPIC, seed=seed, difficulty="hard").text)
        assert a1 is not None and rhs.has(A2)


def test_exam_chain():
    for seed in range(25):
        q = make(EXAM, seed=seed, difficulty="exam")
        check_parts(q)
        a0, a1, rhs = _read(q.text)
        terms = _terms(a0, a1, rhs)
        assert _matches(q.parts[2].answer_text, terms)
        index = int(
            re.search(
                r"Hence find a(\d+)", q.parts[3].prompt.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
            ).group(1)
        )
        assert int(q.parts[3].answer_text) == terms[index]
        wrong = sympy.sympify(q.parts[2].answer_text.replace("^", "**"), locals={"n": n}) + 2**n
        check_carried(q, "c", str(wrong).replace("**", "^"), str(wrong.subs(n, index)))
