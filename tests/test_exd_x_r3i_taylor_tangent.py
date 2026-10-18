"""Exam source of `ucv1-taylor-linear-approximation`: derivative, linear approximation, error, tangent intercept."""

from __future__ import annotations

import re
from fractions import Fraction

from sympy import Rational, Symbol, diff, sympify

SKILL_ID = "ucv1-taylor-linear-approximation"
TOPIC = "exd_x_r3i_taylor_tangent"
LABELS = ["a", "b", "c", "d"]
X = Symbol("x")

from collections import Counter

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
SEEDS = range(300)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert 3 <= len(q.parts) <= 4
        assert all(p.distractors and len(p.distractors) >= 2 for p in q.parts)
        assert sum(1 for p in q.parts if p.follow_through) >= 1


def test_no_text_defects(questions):
    for q in questions[:100]:
        assert find_text_defects(q, include_working=True) == [], q.text


def test_key_rederived_from_prompt(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: str(d)})
                assert result.parts[p.label].marks == 0, (q.text, p.label, str(d))


def test_distractors_match_key_layout(questions):
    for q in questions:
        for p in q.parts:
            if p.answer_kind != "number":
                continue
            for d in p.distractors:
                assert ("." in d) == ("." in p.answer_text), (p.label, d, p.answer_text)
                if "." in d:
                    assert len(d.split(".")[1]) == len(p.answer_text.split(".")[1]), (p.label, d, p.answer_text)


def test_distractors_are_tagged(questions):
    from mathgen.misconceptions import CATALOGUE

    for q in questions[:100]:
        for p in q.parts:
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (q.text, p.label, d)


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        reply = {p.label: ("999999.5" if p.answer_kind == "number" else "999999/7") for p in q.parts}
        result = grade(_SKILL, q, reply)
        assert result.marks == 0 and not result.correct


def _rat(r):
    r = Rational(r)
    return str(r.p) if r.q == 1 else f"{r.p}/{r.q}"


def _facts(text):
    f_text = re.search(r"f\(x\) = ([^.]*?)\.", text).group(1)
    f = sympify(re.sub(r"(\d)x", r"\1*x", f_text).replace("^", "**"))
    a = int(re.search(r"x = (-?\d+)", text).group(1))
    x0 = Rational(re.search(r"approximation L\((-?\d+\.\d)\)", text).group(1))
    return f, a, x0


def oracle(q):
    f, a, x0 = _facts(q.text)
    fa, fpa = f.subs(X, a), diff(f, X).subs(X, a)
    approx = fa + fpa * (x0 - a)
    err = f.subs(X, x0) - approx
    return {
        "a": str(fpa),
        "b": f"{float(approx):.1f}",
        "c": f"{float(err):.3f}",
        "d": _rat(a - fa / fpa),
    }


def test_follow_through_from_wrong_derivative(questions):
    for q in questions[:120]:
        f, a, x0 = _facts(q.text)
        fa = f.subs(X, a)
        wrong = q.parts[0].distractors[0]
        w = Rational(wrong)
        b_wrong = fa + w * (x0 - a)
        reply = {
            "a": wrong,
            "b": f"{float(b_wrong):.1f}",
            "c": _canonical(q)["c"],
            "d": _rat(a - fa / w),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, reply)
        assert result.parts["b"].follow_through is True
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)


def test_follow_through_error_uses_own_approximation(questions):
    for q in questions[:120]:
        f, a, x0 = _facts(q.text)
        own_b = Rational(q.parts[1].distractors[0])
        own_c = f.subs(X, x0) - own_b
        reply = {**_canonical(q), "b": q.parts[1].distractors[0], "c": f"{float(own_c):.3f}"}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
