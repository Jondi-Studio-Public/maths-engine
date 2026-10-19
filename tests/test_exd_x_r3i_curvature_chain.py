"""Exam source of `ucv1-parametric-curvature`: speed, cross term, curvature and radius at a non-zero parameter."""

from __future__ import annotations

import re
from fractions import Fraction

from sympy import Symbol, diff
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication, parse_expr, standard_transformations

from mathgen.core import round_half_up

SKILL_ID = "ucv1-parametric-curvature"
TOPIC = "exd_x_r3i_curvature_chain"
LABELS = ["a", "b", "c", "d"]
T = Symbol("t")
_TR = standard_transformations + (implicit_multiplication, convert_xor)

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


def _facts(text):
    x_text = re.search(r"x\(t\) = (.*?)(?:,| and) y\(t\) = ", text).group(1)
    y_text = re.search(r"y\(t\) = (.*?)(?:\. | \.)", text + " ").group(1)
    t0 = int(re.search(r"\bt = (-?\d+)", text.split("y(t) =")[1]).group(1))
    x = parse_expr(x_text, transformations=_TR, local_dict={"t": T})
    y = parse_expr(y_text, transformations=_TR, local_dict={"t": T})
    return x, y, t0


def _quantities(q):
    x, y, t0 = _facts(q.text)
    vx, vy = diff(x, T).subs(T, t0), diff(y, T).subs(T, t0)
    ax, ay = diff(x, T, 2).subs(T, t0), diff(y, T, 2).subs(T, t0)
    return int(vx), int(vy), int(ax), int(ay)


def oracle(q):
    vx, vy, ax, ay = _quantities(q)
    h = round((vx * vx + vy * vy) ** 0.5)
    cross = vx * ay - vy * ax
    kappa = round_half_up(Fraction(abs(cross), h**3), 4)
    radius = round_half_up(Fraction(h**3, abs(cross)), 2)
    return {"a": str(h), "b": str(cross), "c": f"{float(kappa):.4f}", "d": f"{float(radius):.2f}"}


def test_follow_through_from_wrong_cross_term(questions):
    for q in questions[:150]:
        wb = int(q.parts[1].distractors[0])
        h = int(q.parts[0].answer_text)
        if wb == 0:
            continue
        kappa = round_half_up(Fraction(abs(wb), h**3), 4)
        radius = round_half_up(Fraction(h**3, abs(wb)), 2)
        reply = {**_canonical(q), "b": str(wb), "c": f"{float(kappa):.4f}", "d": f"{float(radius):.2f}"}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)
