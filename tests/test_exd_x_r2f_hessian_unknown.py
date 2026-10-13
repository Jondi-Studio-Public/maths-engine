"""Exam source of `ucv1-hessian-2x2`: a four-part chain from an unknown coefficient to the Hessian determinant."""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_hessian_unknown"
SKILL_ID = "ucv1-hessian-2x2"
SEEDS = range(300)
X, Y, E = sympy.symbols("x y E")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_STEM = re.compile(r"f\(x, y\) = (.*?), where E is a constant\. It is known that f_xy\((-?\d+), (-?\d+)\) = (-?\d+)\.")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _parse(stem):
    text, x0, y0, fxy = _STEM.search(stem).groups()
    text = text.replace("³", "**3").replace("²", "**2")
    f = parse_expr(text, transformations=_TRANSFORMS, local_dict={"x": X, "y": Y, "E": E})
    return f, int(x0), int(y0), int(fxy)


def _frac(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def oracle(q):
    """Solve f_xy(P) = given for E by differentiating the displayed function, then read off the Hessian at P."""
    f, x0, y0, fxy = _parse(q.stem)
    at_p = {X: x0, Y: y0}
    (e,) = sympy.solve(sympy.diff(f, X, Y).subs(at_p) - fxy, E)
    hess = sympy.hessian(f.subs(E, e), (X, Y)).subs(at_p)
    return e, hess[0, 0], hess[1, 1], hess.det()


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
        assert [p.marks for p in q.parts] == [2, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)
        assert all(len(p.distractors) >= 2 and p.option_notes for p in q.parts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("p12f_hessian_2x2", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        e, fxx, fyy, det = oracle(q)
        assert _canonical(q) == {"a": _frac(e), "b": _frac(fxx), "c": _frac(fyy), "d": _frac(det)}, q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_e(questions):
    """A wrong E, used correctly for f_xx, f_yy and the determinant, loses only the 2 marks of (a)."""
    checked = 0
    for q in questions[:120]:
        f, x0, y0, fxy = _parse(q.stem)
        e_wrong = sympy.Rational(q.parts[0].distractors[0])
        hess = sympy.hessian(f.subs(E, e_wrong), (X, Y)).subs({X: x0, Y: y0})
        fxx, fyy = hess[0, 0], hess[1, 1]
        det = fxx * fyy - fxy**2
        values = [e_wrong, fxx, fyy, det]
        if 0 in values or len(set(values)) < 4:  # follow-through is withheld from a zero or a repeated reply
            continue
        reply = dict(zip("abcd", map(_frac, values)))
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label, marks in (("b", 1), ("c", 1), ("d", 2)):
            assert result.parts[label].marks == marks, (q.text, label, reply)
            assert result.parts[label].follow_through is True
        assert result.marks == 4 and not result.correct
        checked += 1
    assert checked >= 80


def test_follow_through_into_the_determinant_only(questions):
    for q in questions[:100]:
        canon = _canonical(q)
        _f, _x0, _y0, fxy = _parse(q.stem)
        fxx_w = sympy.Rational(q.parts[1].distractors[0])
        det_w = fxx_w * sympy.Rational(canon["c"]) - fxy**2
        replies = {fxx_w, sympy.Rational(canon["a"]), sympy.Rational(canon["c"])}
        if (
            0 in (det_w, fxx_w) or det_w in replies
        ):  # follow-through is withheld from a reply that repeats an earlier one
            continue
        result = grade(_SKILL, q, {**canon, "b": q.parts[1].distractors[0], "d": _frac(det_w)})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == 2, (q.text, det_w)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "17", "b": "19", "c": "23", "d": "29"})
        assert result.marks == 0 and not result.correct


def test_answer_space(questions):
    tuples = Counter(q.answer_text for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
