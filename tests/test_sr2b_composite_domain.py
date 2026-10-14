"""Tests the composite-function-domain skill: domains checked pointwise against the stem's two functions."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import sr2b_support as sup

SKILL = "composite-function-domain"
TOPIC = "sr2b_composite_domain"
EXAM = "sr2b_exam_composite_domain"
X = sympy.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _fn(text: str):
    text = text.replace("²", "**2").replace("ln", "log")
    text = re.sub(r"√\(", "sqrt(", text)
    text = re.sub(r"√x", "sqrt(x)", text)
    return parse_expr(text, local_dict={"x": X}, transformations=_T)


def _functions(stem: str):
    f_text, g_text = re.search(r"f\(x\) = (.+?) and g\(x\) = (.+?)(?:\. |, )", stem + " ").groups()
    return _fn(f_text), _fn(g_text)


def _real(value) -> bool:
    return value.is_real is True and value.is_finite is True


def _defined(f, g, x) -> bool:
    inner = f.subs(X, x)
    if not _real(inner):
        return False
    return _real(g.subs(X, inner))


def _member(answer: str, x: Fraction) -> bool:
    if answer == "all real numbers":
        return True
    for piece in answer.split(" or "):
        if "≠" in piece:
            if x != Fraction(piece.split("≠")[1].strip()):
                return True
            continue
        parts = [t.strip() for t in re.split(r"(≤|≥|<|>)", piece)]
        # parts alternate operand, operator, operand [, operator, operand]
        values = [None if t == "x" else Fraction(t) for t in parts[0::2]]
        ops = parts[1::2]
        ok = True
        for (lo, hi), op in zip(zip(values, values[1:]), ops):
            left = x if lo is None else lo
            right = x if hi is None else hi
            ok &= {"<": left < right, "≤": left <= right, ">": left > right, "≥": left >= right}[op]
        if ok:
            return True
    return False


def _boundaries(answer: str):
    return [Fraction(v) for v in re.findall(r"-?\d+(?:/\d+)?", answer)]


def _sample_points(*answers: str):
    pts = {Fraction(k, 8) for k in range(-200, 201)}
    for b in [v for a in answers for v in _boundaries(a)]:
        pts |= {b, b + Fraction(1, 1000), b - Fraction(1, 1000)}
    return sorted(pts)


def _check_domain(f, g, answer: str, true_answer: str = ""):
    for x in _sample_points(answer, true_answer):
        assert _defined(f, g, sympy.Rational(x.numerator, x.denominator)) == _member(answer, x), (answer, x)


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "inequality",
        [
            "senior-find-the-composition-of-two-functions",
            "senior-determine-when-the-composition-of-two-functions-is",
        ],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_domains_match_a_pointwise_check(tier):
    for seed in range(40):
        q = sup.tier_question(TOPIC, tier, seed)
        f, g = _functions(q.text)
        _check_domain(f, g, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_distractors_are_not_the_domain(tier):
    for seed in range(40):
        q = sup.tier_question(TOPIC, tier, seed)
        f, g = _functions(q.text)
        for d in q.distractors:
            with pytest.raises(AssertionError):
                _check_domain(f, g, d, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8)


def test_equivalent_forms_of_an_answer_are_accepted():
    sk = sup.skill_object(SKILL)
    q = next(q for q in (sup.tier_question(TOPIC, "easy", s) for s in range(60)) if " > " in q.answer_text)
    bound = q.answer_text.split(" > ")[1]
    assert grade(sk, q, f"({bound}, inf)").correct
    assert not grade(sk, q, f"[{bound}, inf)").correct


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(60))
    assert "√(" not in easy
    assert "√(" in hard
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number", "number", "inequality"], [1, 2, 2], [False, True, False])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=15, top=0.1)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        f, g = _functions(q.text)
        c = int(re.search(r"f\(x\) = √\(x ([+-]) (\d+)\)", q.text).group(2)) * (
            1 if "x +" in q.text.split("g(x)")[0] else -1
        )
        d = sympy.solve(sympy.denom(sympy.together(g)), X)[0]
        assert int(q.parts[0].answer_text) == -c, (seed, q.text)
        assert int(q.parts[1].answer_text) == d**2 - c, (seed, q.text)
        _check_domain(f, g, q.parts[2].answer_text)


def test_exam_excluded_value_follows_through():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        d = int(re.search(r"g\(x\) = 1/\(x ([+-]) (\d+)\)", q.text).group(2))
        wrong_a = int(canon["a"]) + 1
        carried = wrong_a + d * d
        replies = {**canon, "a": str(wrong_a), "b": str(carried)}
        assert grade(sk, q, replies).parts["b"].marks == q.parts[1].marks, seed
        assert grade(sk, q, {**replies, "b": str(carried + 1)}).parts["b"].marks == 0, seed
