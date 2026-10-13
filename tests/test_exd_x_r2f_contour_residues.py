"""Exam source of `ucv1-contour-integral-residue`: a four-part chain from the enclosed pole to a larger contour."""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy import I, pi
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_contour_residues"
SKILL_ID = "ucv1-contour-integral-residue"
SEEDS = range(300)
Z = sympy.Symbol("z")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_STEM = re.compile(r"f\(z\) = \((.*?)\) / \((.*?)\)\. The contour C is the circle \|(.*?)\| = (\S+),")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _expr(text):
    return parse_expr(text.replace("²", "**2"), transformations=_TRANSFORMS, local_dict={"z": Z})


def _frac(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def _facts(q):
    num, den, centre, radius = _STEM.search(q.stem).groups()
    f = _expr(num) / _expr(den)
    c0 = sympy.solve(_expr(centre), Z)[0]
    r = sympy.Rational(radius)
    poles = sympy.roots(sympy.Poly(_expr(den), Z))
    assert len(poles) == 2
    inside = [w for w in poles if abs(w - c0) < r]
    outside = [w for w in poles if abs(w - c0) > r]
    assert len(inside) == 1 and len(outside) == 1, q.text
    res = lambda w: sympy.cancel((Z - w) * f).subs(Z, w)  # noqa: E731 - the limit definition of a simple-pole residue
    return f, inside[0], outside[0], res


def oracle(q):
    f, inside, outside, res = _facts(q)
    r_in, r_out = res(inside), res(outside)
    return inside, r_in, sympy.expand(2 * pi * I * r_in), sympy.expand(2 * pi * I * (r_in + r_out))


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "expression", "expression"]
        assert all(len(p.distractors) >= 2 and p.option_notes for p in q.parts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("p15e_contour_integral_residue", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        pole, residue, integral, big = oracle(q)
        assert q.parts[0].answer_text == _frac(pole), q.text
        assert q.parts[1].answer_text == _frac(residue), q.text
        assert sympy.expand(_expr(q.parts[2].answer_text.replace("I", "i")).subs(sympy.Symbol("i"), I) - integral) == 0
        assert sympy.expand(_expr(q.parts[3].answer_text.replace("I", "i")).subs(sympy.Symbol("i"), I) - big) == 0


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_learner_spelling_of_the_integral(questions):
    """The integral typed as a learner would, with a lower-case i and the factors in textbook order."""
    for q in questions[:60]:
        _pole, residue, _integral, big = oracle(q)
        r = sympy.Rational(residue)
        sign = "-" if r < 0 else ""
        typed_c = f"{sign}2*pi*i*{abs(r.p)}/{r.q}" if r.q != 1 else f"{sign}2*pi*i*{abs(r.p)}"
        typed_d = str(big).replace("I", "i")
        canon = _canonical(q)
        result = grade(_SKILL, q, {**canon, "c": typed_c})
        assert result.parts["c"].marks == 1, (q.text, typed_c)
        assert grade(_SKILL, q, {**canon, "d": typed_d}).parts["d"].marks == 2, (q.text, typed_d)


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_sign_flipped_pole(questions):
    """The pole's sign flipped in (a) is carried through the residue, the integral and the larger contour."""
    checked = 0
    for q in questions[:120]:
        f, inside, outside, res = _facts(q)
        num, den, _centre, _radius = _STEM.search(q.stem).groups()
        n = _expr(num)
        w = -inside
        b_wrong = (n.subs(Z, w)) / (w - outside)
        if b_wrong in (0, w):  # follow-through is withheld from a reply that only repeats an earlier one
            continue
        c_wrong = sympy.expand(2 * pi * I * b_wrong)
        d_wrong = sympy.expand(c_wrong + 2 * pi * I * n.subs(Z, outside) / (outside - w))
        if d_wrong == 0:
            continue
        reply = {"a": _frac(w), "b": _frac(b_wrong), "c": str(c_wrong), "d": str(d_wrong)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label, marks in (("b", 2), ("c", 1)):
            assert result.parts[label].marks == marks, (q.text, label, reply)
            assert result.parts[label].follow_through is True
        # The residues at the two poles always sum to the numerator's leading coefficient, so (d) is the same value
        # whichever pole (a) names: full marks, as the canonical answer.
        assert result.parts["d"].marks == 2 and str(d_wrong) == q.parts[3].answer_text.replace("I", "I")
        assert result.marks == 5 and not result.correct
        checked += 1
    assert checked >= 100


def test_wrong_pole_gets_no_residue_marks(questions):
    """The pole outside the contour is a wrong (a) whose residue formula would divide by zero: no follow-through."""
    for q in questions[:40]:
        _f, inside, outside, _res = _facts(q)
        result = grade(_SKILL, q, {**_canonical(q), "a": _frac(outside), "b": "123/7"})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 0


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "99", "b": "5/13", "c": "pi", "d": "7*I"})
        assert result.marks == 0 and not result.correct


def test_answer_space(questions):
    tuples = Counter(q.answer_text for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()


def test_chain_is_not_copyable():
    """No part repeats an earlier answer or follows through as a restatement, and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable("exd_x_r2f_contour_residues", "ucv1-contour-integral-residue")
