"""Tests the multi-part exam source of `directional-derivative`: partial derivatives, the rate towards a second point, and the steepest rate."""

from __future__ import annotations

import re
from fractions import Fraction
import math

import sympy

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u3_directional_toward"
SKILL_ID = "directional-derivative"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 2, 3, 2]

_X, _Y = sympy.symbols("x y")


def _parse(q):
    expr = re.search(r"f\(x, y\) = (.*?)\. Consider", q.stem).group(1)
    text = expr.replace("^", "**").replace("xy", "x*y")
    text = re.sub(r"(\d)([xy])", r"\1*\2", text)
    text = re.sub(r"\*\*(\d)y", r"**\1*y", text)
    f = sympy.sympify(text, locals={"x": _X, "y": _Y})
    px, py, qx, qy = (int(v) for v in re.search(r"P\((-?\d+), (-?\d+)\).*?Q\((-?\d+), (-?\d+)\)", q.stem).groups())
    return f, px, py, qx - px, qy - py


def _num(text):
    return Fraction(str(text).strip())


def _round(value, places):
    """Round half up, once, from an exact value."""
    scale = 10**places
    return Fraction(int((Fraction(value) * scale + Fraction(1, 2)).__floor__()), scale)


def _tie(value, places=2):
    """True when `value` sits exactly on a rounding boundary, where graders may legitimately round either way."""
    return (Fraction(value) * 10**places) % 1 == Fraction(1, 2)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in range(300)]


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages, s.params) for s in exam] == [(TOPIC, ["exam"], {"tier": "exam"})]


def test_shape_and_marks(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert all(p.answer_kind == "number" for p in q.parts)
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert all(p.follow_through is not None for p in q.parts[2:])


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(_HARD.topic, seed=s, difficulty="hard", **_HARD.params)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(make(TOPIC, seed=seed)) == []


def test_answer_space(questions):
    texts = [q.answer_text for q in questions]
    assert len(set(texts)) >= 100
    assert max(texts.count(t) for t in set(texts)) <= 15


def test_key_rederived_from_prompt(questions):
    for q in questions:
        f, px, py, dx, dy = _parse(q)
        g1 = sympy.diff(f, _X).subs({_X: px, _Y: py})
        g2 = sympy.diff(f, _Y).subs({_X: px, _Y: py})
        length = sympy.sqrt(dx * dx + dy * dy)
        assert length.is_Integer
        directional = (g1 * dx + g2 * dy) / length
        steepest = sympy.sqrt(g1**2 + g2**2)
        got = [p.answer_text for p in q.parts]
        assert Fraction(got[0]) == int(g1) and Fraction(got[1]) == int(g2), (q.text, got)
        assert got[2] == f"{float(_round(Fraction(int(directional.p), int(directional.q)), 2)):.2f}", (q.text, got)
        assert abs(float(got[3]) - float(steepest)) <= 0.005 + 1e-9, (q.text, got)


def test_composed_answer_full_marks(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_distractors_rejected(questions):
    for q in questions[:120]:
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_partials(questions):
    for q in questions:
        f, px, py, dx, dy = _parse(q)
        length = int(math.isqrt(dx * dx + dy * dy))
        ga, gb = _num(q.parts[0].distractors[0]), _num(q.parts[1].distractors[0])
        if _tie(Fraction(ga * dx + gb * dy, length)):
            continue
        c = _round((ga * dx + gb * dy) / length, 2)
        d = _round(Fraction(str(math.sqrt(ga * ga + gb * gb))), 2)
        if len({ga, gb, c, d}) < 4 or 0 in (ga, gb):
            continue  # a reply that repeats an earlier reply earns no follow-through by design
        reply = {
            "a": q.parts[0].distractors[0],
            "b": q.parts[1].distractors[0],
            "c": f"{float(c):.2f}",
            "d": f"{float(d):.2f}",
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        if c != 0:
            assert result.parts["c"].marks == 3 and result.parts["c"].follow_through is True, (q.text, reply)
        assert result.parts["d"].marks == 2, (q.text, reply)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {p.label: "987654.321" for p in q.parts})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert make(TOPIC, seed=7).to_dict() == make(TOPIC, seed=7).to_dict()
