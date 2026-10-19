"""Tests the multi-part exam source of `solve-simultaneous-equations`: set up and solve two receipts, then price a third order."""

from __future__ import annotations

import re
from fractions import Fraction
import sympy

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u3_simultaneous_prices"
SKILL_ID = "solve-simultaneous-equations"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")
LABELS = ["a", "b", "c"]
MARKS = [3, 2, 1]


def _parse(q):
    receipts = re.findall(r"shows (\d+) [^$]*? and (\d+) [^$]*? for \$(\d+\.\d\d)", q.stem)
    cost = re.search(r"cost of (\d+) [^$]*? and (\d+) ", q.parts[2].prompt)
    return receipts, [int(v) for v in cost.groups()]


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
    a, b = sympy.symbols("a b")
    for q in questions:
        receipts, (n, m) = _parse(q)
        assert len(receipts) == 2
        eqs = [sympy.Eq(int(p) * a + int(r) * b, sympy.Rational(t)) for p, r, t in receipts]
        sol = sympy.solve(eqs, [a, b], dict=True)[0]
        want = [sol[a], sol[b], n * sol[a] + m * sol[b]]
        got = [sympy.Rational(p.answer_text) for p in q.parts]
        assert got == want, (q.text, got, want)


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


def test_follow_through_from_wrong_first_price(questions):
    for q in questions:
        receipts, (n, m) = _parse(q)
        (p, r, t) = receipts[0]
        x_wrong = _num(q.parts[0].distractors[0])
        y_wrong = (_num(t) - int(p) * x_wrong) / int(r)
        total = n * x_wrong + m * _round(y_wrong, 2)
        if _tie(y_wrong) or _tie(total) or y_wrong == x_wrong or total in (x_wrong, y_wrong):
            continue  # a tie, or a reply that repeats an earlier reply, earns no follow-through by design
        total_wrong = n * x_wrong + m * y_wrong
        reply = {
            "a": q.parts[0].distractors[0],
            "b": f"{float(_round(y_wrong, 2)):.2f}",
            "c": f"{float(_round(n * x_wrong + m * _round(y_wrong, 2), 2)):.2f}",
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        if y_wrong > 0:
            assert result.parts["b"].marks == q.parts[1].marks, (q.text, reply)
            assert result.parts["b"].follow_through is True
            assert result.parts["c"].marks == q.parts[2].marks, (q.text, reply, total_wrong)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {p.label: "987654.321" for p in q.parts})
        assert result.marks == 0 and not result.correct
