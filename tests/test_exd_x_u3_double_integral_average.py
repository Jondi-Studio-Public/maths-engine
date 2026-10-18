"""Tests the multi-part exam source of `ucv1-double-integral-rectangle`: a sum-form integrand over a rectangle, then its average value."""

from __future__ import annotations

import re
from fractions import Fraction
import sympy

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u3_double_integral_average"
SKILL_ID = "ucv1-double-integral-rectangle"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")
LABELS = ["a", "b"]
MARKS = [3, 2]

_X, _Y = sympy.symbols("x y")


def _parse(q):
    expr = re.search(r"f\(x, y\) = (.*?) and let R", q.stem).group(1)
    expr = re.sub(r"(\d)([xy])", r"\1*\2", expr.replace("^", "**")).replace("xy", "x*y")
    f = sympy.sympify(expr, locals={"x": _X, "y": _Y})
    x0, x1, y0, y1 = (int(v) for v in re.search(r"(-?\d+) ≤ x ≤ (-?\d+), (-?\d+) ≤ y ≤ (-?\d+)", q.stem).groups())
    return f, x0, x1, y0, y1


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


def test_shape_and_marks(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert all(p.answer_kind == "number" for p in q.parts)
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert all(p.follow_through is not None for p in q.parts[1:])


def test_key_rederived_from_prompt(questions):
    for q in questions:
        f, x0, x1, y0, y1 = _parse(q)
        total = sympy.integrate(sympy.integrate(f, (_Y, y0, y1)), (_X, x0, x1))
        total = Fraction(int(total.p), int(total.q))
        mean = total / ((x1 - x0) * (y1 - y0))
        assert q.parts[0].answer_text == f"{float(_round(total, 2)):.2f}", q.text
        assert q.parts[1].answer_text == f"{float(_round(mean, 2)):.2f}", q.text


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


def test_follow_through_from_wrong_integral(questions):
    for q in questions:
        _, x0, x1, y0, y1 = _parse(q)
        area = (x1 - x0) * (y1 - y0)
        wrong = _num(q.parts[0].distractors[0])
        if _tie(wrong / area):
            continue
        mean = _round(wrong / area, 2)
        result = grade(_SKILL, q, {"a": q.parts[0].distractors[0], "b": f"{float(mean):.2f}"})
        assert result.parts["a"].marks == 0
        if mean != 0:
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (q.text, mean)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {p.label: "987654.321" for p in q.parts})
        assert result.marks == 0 and not result.correct
