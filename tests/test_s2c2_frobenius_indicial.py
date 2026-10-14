"""Tests `uc2-frobenius-indicial`: the topic `s2c2_frobenius_indicial` (easy, medium, hard) and the exam chain
`exd_x_s2c2_frobenius_indicial`.

The oracle reads the equation back out of the question text (multiplying through by x when it is written as
a x y'' + (b + c x) y' + d y = 0), finds the indicial roots with SymPy's `solve`, and finds the series coefficients by
substituting x^r (1 + a_1 x + a_2 x^2) into the equation and solving the resulting linear equations, a route that
shares nothing with the generator's recurrence.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2c2_frobenius_indicial_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILL_ID = "uc2-frobenius-indicial"
TOPIC = "s2c2_frobenius_indicial"
EXAM = "exd_x_s2c2_frobenius_indicial"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
x = sympy.Symbol("x", positive=True)
r = sympy.Symbol("r")
_T = standard_transformations + (implicit_multiplication_application,)
D2, D1, D0 = sympy.symbols("D2 D1 D0")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _stem(q):
    return q.stem if q.parts else q.text


def _standard_form(text):
    """(A, B(x), C(x)) of A x^2 y'' + x B y' + C y = 0, read from the stem."""
    ode = re.search(r"differential equation (.+?) = 0 has a regular", text).group(1)
    ode = ode.replace("y''", "D2").replace("y'", "D1")
    ode = re.sub(r"(?<![A-Za-z])y(?![A-Za-z(])", "D0", ode).replace("²", "**2")
    ode = re.sub(r"(?<![A-Za-z])x\(", "x*(", ode)
    expr = sympy.expand(parse_expr(ode, transformations=_T, local_dict={"x": x, "D2": D2, "D1": D1, "D0": D0}))
    p, q, c = expr.coeff(D2), expr.coeff(D1), expr.coeff(D0)
    if sympy.degree(p, x) == 1:  # written with x y'': multiply through by x
        p, q, c = sympy.expand(x * p), sympy.expand(x * q), sympy.expand(x * c)
    a = p.coeff(x, 2)
    assert sympy.expand(p - a * x**2) == 0
    return a, sympy.expand(q / x), c


def _exponents(a, b, c):
    indicial = a * r * (r - 1) + b.subs(x, 0) * r + c.subs(x, 0)
    return sorted(sympy.solve(indicial, r))


def _coefficient(a, b, c, root, n):
    """a_n for exponent `root` and a_0 = 1, by substituting the series into the equation and solving."""
    unknowns = sympy.symbols(f"k1:{n + 1}")
    series = 1 + sum(u * x ** (i + 1) for i, u in enumerate(unknowns))
    y = x**root * series
    residual = a * x**2 * sympy.diff(y, x, 2) + x * b * sympy.diff(y, x) + c * y
    reduced = sympy.expand(sympy.powsimp(sympy.expand(residual / x**root)))
    poly = sympy.Poly(reduced, x)
    equations = [poly.coeff_monomial(x**k) for k in range(1, n + 1)]
    solution = sympy.solve(equations, unknowns, dict=True)[0]
    return solution[unknowns[n - 1]]


def _asked(text):
    match = re.search(r"Find a_(\d)", text)
    return int(match.group(1)) if match else None


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages) == (EXAM, "exam", ["exam"])
    assert [src.difficulty for src in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_derivation(tier):
    for seed in range(30):
        q = _q(tier, seed)
        a, b, c = _standard_form(q.text)
        roots = _exponents(a, b, c)
        n = _asked(q.text)
        if n is None:
            expected = roots[-1]
        else:
            root = roots[-1] if "larger" in q.text else roots[0]
            expected = _coefficient(a, b, c, root, n)
        assert sympy.Rational(q.answer_text) == expected, (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


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
        assert "indicial equation" in _q("easy", seed).text and _asked(_q("easy", seed).text) is None
        assert _asked(_q("medium", seed).text) == 1 and "larger" in _q("medium", seed).text
        assert _asked(_q("hard", seed).text) == 2 and "smaller" in _q("hard", seed).text


def test_medium_uses_both_ways_of_writing_the_equation():
    forms = {"x y''" in _q("medium", s).text and "x² y''" not in _q("medium", s).text for s in range(40)}
    assert forms == {True, False}


def test_hard_roots_do_not_differ_by_an_integer():
    for seed in range(40):
        a, b, c = _standard_form(_q("hard", seed).text)
        low, high = _exponents(a, b, c)
        assert not (high - low).is_integer


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(80)]
    assert len(set(answers)) >= 12
    assert len({_q(tier, seed).text for seed in range(80)}) >= 50


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
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert all(p.answer_kind == "fraction" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q) == []
        assert all(len(p.distractors) >= 2 for p in q.parts)


def test_exam_keys_match_independent_derivation():
    for seed in range(30):
        q = _exam(seed)
        a, b, c = _standard_form(q.stem)
        low, high = _exponents(a, b, c)
        pa, pb, pc, pd = q.parts
        assert sympy.Rational(pa.answer_text) == high
        assert sympy.Rational(pb.answer_text) == low
        assert sympy.Rational(pc.answer_text) == _coefficient(a, b, c, high, 1)
        assert sympy.Rational(pd.answer_text) == _coefficient(a, b, c, low, 1)


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(25):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 7
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_larger_root():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        canon = _canonical(q)
        a, b, c = _standard_form(q.stem)
        b0, c0, b1, c1 = b.subs(x, 0), c.subs(x, 0), sympy.diff(b, x), sympy.diff(c, x)
        for wrong in q.parts[0].distractors:
            w = sympy.Rational(wrong)
            reply_b = 1 - b0 / a - w
            denominator = a * (w + 1) * w + b0 * (w + 1) + c0
            if w == 0 or denominator == 0 or reply_b == 0 or str(reply_b) in (canon["b"], wrong):
                continue
            reply_c = -(b1 * w + c1) / denominator
            if reply_c == 0 or str(reply_c) in (canon["c"], str(reply_b), wrong):
                continue
            result = grade(_SKILL, q, {**canon, "a": wrong, "b": str(reply_b), "c": str(reply_c)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].follow_through is True and result.parts["b"].marks == 1, (seed, wrong, result)
            assert result.parts["c"].follow_through is True and result.parts["c"].marks == 2, (seed, wrong, result)
            checked += 1
    assert checked >= 15


def test_exam_answer_space_and_determinism():
    stems = [_exam(seed).stem for seed in range(100)]
    assert len(set(stems)) >= 60
    assert _exam(7).to_json() == _exam(7).to_json()
