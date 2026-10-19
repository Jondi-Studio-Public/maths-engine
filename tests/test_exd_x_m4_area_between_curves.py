"""Tests the multi-part exam tier of the area-between-curves skill: intersections, area, cost."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m4_area_between_curves"
SKILL_ID = "senior-calculate-the-area-between-curves-in-simple-cases"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
X = sympy.Symbol("x")


def _poly(text: str):
    return sympy.sympify(re.sub(r"(\d)x", r"\1*x", text.replace("²", "**2")), locals={"x": X})


def derive(q) -> dict[str, str]:
    stem = q.stem
    pieces = re.findall(r"y = ([-+x²0-9 ]+?)(?= and| intersect| cross|,| which|\. | \()", stem)
    f, g = _poly(pieces[0]), _poly(pieces[1])
    price = int(re.search(r"\$(\d+) per", stem).group(1))
    roots = sorted(sympy.solve(sympy.Eq(f, g), X))
    p, qq = roots
    area = abs(sympy.integrate(f - g, (X, p, qq)))
    shown = Fraction(int(sympy.floor(area * 100 + sympy.Rational(1, 2))), 100)

    def dec(r):
        r = Fraction(int(r.p), int(r.q))
        return str(r.numerator) if r.denominator == 1 else str(float(r))

    return {
        "a": dec(p),
        "b": dec(qq),
        "c": f"{float(shown):.2f}",
        "d": f"{float(shown * price):.2f}",
    }


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_oracle_agrees(questions):
    for q in questions:
        assert derive(q) == _canonical(q), q.text


def test_answer_space(questions):
    values = [q.parts[3].answer_text for q in questions]
    assert len(set(values)) >= 100
    assert max(values.count(v) for v in set(values)) / len(values) <= 0.05


def test_full_marks_and_distractors_rejected(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_limits(questions):
    for q in questions[:100]:
        a, b = Fraction(q.parts[0].answer_text), Fraction(q.parts[1].answer_text)
        a_wrong = a - 1 if a != 1 else a - 2
        scale = int(re.match(r"(\d+)\*", q.parts[2].follow_through).group(1))
        area = Fraction(scale) * (b - a_wrong) ** 3 / 6
        shown = Fraction(int(area * 100 + Fraction(1, 2)), 100)
        price = int(q.parts[3].follow_through.split("*")[0])
        reply = {
            "a": str(a_wrong),
            "b": q.parts[1].answer_text,
            "c": f"{float(shown):.2f}",
            "d": f"{float(shown * price):.2f}",
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
        assert result.parts["c"].follow_through is True
        assert result.parts["d"].marks == result.parts["d"].max_marks


def test_prompt_shapes_new():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard_src = next(s for s in _SKILL.sources if s.difficulty == "hard")
    hard = {prompt_shape(make(hard_src.topic, seed=s, **hard_src.params)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard
