"""Tests the multi-part exam tier of `jacobian-two-variable`: determinant of the Jacobian, the image area and a linear estimate."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc5_jacobian_area"
SKILL_ID = "jacobian-two-variable"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
MARKS = [2, 1, 2]
FT_LABELS = [False, True, False]
TOP_SHARE = 0.05


def _exam(seed):
    return make(TOPIC, seed=seed)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _frac(text):
    return Fraction(text)


_X, _Y = sympy.symbols("x y")
_T = standard_transformations + (implicit_multiplication_application,)


def _parse(text):
    return parse_expr(text.replace("^", "**"), transformations=_T, local_dict={"x": _X, "y": _Y})


def _fmt(value):
    value = Fraction(value)
    for places in range(0, 5):
        if (value * 10**places).denominator == 1:
            scaled = (value * 10**places).numerator
            sign = "-" if scaled < 0 else ""
            digits = str(abs(scaled)).rjust(places + 1, "0")
            return sign + (digits[:-places] + "." + digits[-places:] if places else digits)
    raise AssertionError(value)


def _setup(q):
    u_text, v_text = re.search(r"u = (.*) and v = (.*)\. Its", q.stem).groups()
    x0, y0 = (int(v) for v in re.search(r"at the point \((-?\d+), (-?\d+)\)", q.stem).groups())
    area = Fraction(re.search(r"area of ([\d.]+) square", q.parts[1].prompt).group(1))
    nx, ny = (Fraction(v) for v in re.search(r"estimate u at \((-?[\d.]+), (-?[\d.]+)\)", q.parts[2].prompt).groups())
    return _parse(u_text), _parse(v_text), x0, y0, area, nx - x0, ny - y0


def oracle(q):
    u, v, x0, y0, area, dx, dy = _setup(q)
    jac = sympy.Matrix([[sympy.diff(f, s) for s in (_X, _Y)] for f in (u, v)]).subs({_X: x0, _Y: y0})
    det = int(jac.det())
    est = int(u.subs({_X: x0, _Y: y0})) + int(jac[0, 0]) * dx + int(jac[0, 1]) * dy
    return {"a": str(det), "b": _fmt(abs(det) * area), "c": _fmt(est)}


def follow_through_cases(q):
    u, v, x0, y0, area, dx, dy = _setup(q)
    det = int(q.parts[0].answer_text) + 3
    if det == 0:
        det = 1
    return [({"a": str(det), "b": _fmt(abs(det) * area)}, "b", "a")]


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def test_shape(questions):
    for q in questions[:100]:
        assert [p.label for p in q.parts] == list("abcd")[: len(MARKS)]
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_LABELS
        assert all(p.answer_kind == "number" for p in q.parts)


def test_key_rederived_from_prompt(questions):
    for q in questions:
        got = oracle(q)
        want = _canonical(q)
        assert set(got) == set(want)
        for label in want:
            assert _frac(got[label]) == _frac(want[label]), (q.text, label, got[label], want[label])


def test_composed_answer_full_marks(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.answer_text


def test_distractors_rejected(questions):
    shown = 0
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            assert len(p.distractors) == len(set(p.distractors))
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)
    assert shown >= 600


def test_distractors_are_tagged(questions):
    for q in questions[:100]:
        for p in q.parts:
            assert p.distractor_values, (q.text, p.label)
            assert set(p.option_notes) <= set(p.distractors)


def test_follow_through(questions):
    checked = 0
    for q in questions[:100]:
        for reply, earns, loses in follow_through_cases(q):
            result = grade(_SKILL, q, reply)
            for label in earns:
                part = result.parts[label]
                assert part.marks == part.max_marks, (q.text, reply, label)
                assert part.follow_through is True, (q.text, reply, label)
            for label in loses:
                assert result.parts[label].marks == 0, (q.text, reply, label)
            assert not result.correct
            checked += 1
    assert checked >= 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "987654", "b": "7654321", "c": "-31", "d": "-9"})
        assert result.marks == 0 and not result.correct
