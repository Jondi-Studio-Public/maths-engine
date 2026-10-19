"""Tests `uc2-linear-recurrence-closed-form` (topic `r3l_linear_recurrence_closed_form`) and its exam chain.

The oracle reads the recurrence out of the question TEXT, iterates it for twenty terms, and checks that the stated
closed form reproduces every term; a closed form is also compared against a different recurrence to prove the check
can fail.
"""

from __future__ import annotations

import re

import sympy

import mathgen.generators.r3l_linear_recurrence_closed_form_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r3l_support import check_follow_through, check_parts, check_single, plain

TOPIC = "r3l_linear_recurrence_closed_form"
EXAM = "exd_x_r3l_linear_recurrence_closed_form"
SKILL_ID = "uc2-linear-recurrence-closed-form"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
n = sympy.Symbol("n")


def _read(text):
    """(a0, a1, p, q, k) for a_n = p a_{n-1} + q a_{n-2} + k."""
    body = plain(text)
    a0, a1 = (int(v) for v in re.search(r"a0 = (-?\d+), a1 = (-?\d+)", body).groups())
    rhs = re.search(r"satisfies an = (.*?) for n", body).group(1).replace(" ", "")

    def coef(lag):
        m = re.search(rf"([+-]?\d*)an-{lag}", rhs)
        if not m:
            return 0
        return {"": 1, "+": 1, "-": -1}.get(m.group(1)) or int(m.group(1))

    p, q = coef(1), coef(2)
    rest = re.sub(r"[+-]?\d*an-[12]", "", rhs)
    return a0, a1, p, q, int(rest) if rest else 0


def _terms(a0, a1, p, q, k, count=20):
    out = [a0, a1]
    while len(out) < count:
        out.append(p * out[-1] + q * out[-2] + k)
    return out


def _matches(expr, terms):
    e = sympy.sympify(expr.replace("^", "**"), locals={"n": n})
    return all(sympy.simplify(e.subs(n, i) - t) == 0 for i, t in enumerate(terms))


def test_skill_metadata():
    assert list(_SKILL.year_level) == ["university"]
    assert "university-year-2" in _SKILL.tags


def test_all_tiers_match_iteration():
    for tier in ("easy", "medium", "hard"):
        for seed in range(30):
            q = make(TOPIC, seed=seed, difficulty=tier)
            a0, a1, p, q_, k = _read(q.text)
            terms = _terms(a0, a1, p, q_, k)
            assert _matches(q.answer_text, terms), (tier, seed, q.text, q.answer_text)
            assert not _matches(q.answer_text, _terms(a0, a1 + 1, p, q_, k))
            check_single(q, "expression")


def test_medium_is_a_repeated_root():
    for seed in range(30):
        q = make(TOPIC, seed=seed, difficulty="medium")
        _, _, p, q_, k = _read(q.text)
        assert k == 0 and p * p + 4 * q_ == 0


def test_hard_has_a_constant_term_and_some_resonance():
    resonant = 0
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        _, _, p, q_, k = _read(q.text)
        assert k != 0
        if p + q_ == 1:  # 1 is a root of x^2 - p x - q: the constant term is resonant
            resonant += 1
    assert resonant > 3


def test_exam_chain():
    for seed in range(25):
        q = make(EXAM, seed=seed, difficulty="exam")
        check_parts(q)
        a0, a1, p, q_, k = _read(q.text)
        terms = _terms(a0, a1, p, q_, k)
        assert _matches(q.parts[2].answer_text, terms)
        N = int(re.search(r"find a(\d+)", plain(q.parts[3].prompt)).group(1))
        assert int(q.parts[3].answer_text) == terms[N]
        wrong = sympy.sympify(q.parts[2].answer_text.replace("^", "**"), locals={"n": n}) + 2**n
        carried = wrong.subs(n, N)
        replies = {pt.label: pt.answer_text for pt in q.parts}
        replies["c"], replies["d"] = str(wrong).replace("**", "^"), str(carried)
        from mathgen.grading import grade

        got = grade(q.parts[0].answer_kind, q, replies).parts["d"]
        assert got.marks == q.parts[3].marks and got.follow_through
