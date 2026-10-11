"""Tests the multi-part exam tier of `senior-use-technology-to-investigate-the-effect-of-the`: balance, interest,
the comparison balance and the difference."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills

TOPIC = "exam_parts_compound_interest"
SKILL_ID = "senior-use-technology-to-investigate-the-effect-of-the"
SEEDS = range(120)
_PERIODS = {"half-yearly": 2, "six-monthly": 2, "quarterly": 4, "monthly": 12, "annually": 1}

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _cents(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _balance(p: Decimal, rate_pct: Decimal, n: int, t: int) -> Decimal:
    return p * (1 + rate_pct / 100 / n) ** (n * t)


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the question text alone."""
    stem = q.stem
    p = Decimal(re.search(r"\$([\d,]+)", stem).group(1).replace(",", ""))
    rate = Decimal(re.search(r"([\d.]+)% p\.a\.", stem).group(1))
    n = next(v for k, v in _PERIODS.items() if k in stem)
    t = int(re.search(r"(\d+) years", stem).group(1))
    c_prompt = q.parts[2].prompt
    if "compounded annually" in c_prompt:
        rate_c, n_c = rate, 1
    else:
        rate_c, n_c = Decimal(re.search(r"([\d.]+)% p\.a\. instead", c_prompt).group(1)), n
    a = _cents(_balance(p, rate, n, t))
    b = a - p
    c = _cents(_balance(p, rate_c, n_c, t))
    d = a - c
    return {k: f"{v:.2f}" for k, v in zip("abcd", (a, b, c, d))}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 1, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)
        for p in q.parts:
            assert re.fullmatch(r"\d+\.\d\d", p.answer_text)
            assert "to the nearest cent" in p.prompt


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_distractors_rejected(questions):
    for q in questions:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_balance(questions):
    for q in questions:
        p = Decimal(re.search(r"\$([\d,]+)", q.stem).group(1).replace(",", ""))
        canon = _canonical(q)
        a_wrong = q.parts[0].distractors[0]
        b_wrong = f"{Decimal(a_wrong) - p:.2f}"
        d_wrong = f"{Decimal(a_wrong) - Decimal(canon['c']):.2f}"
        result = grade(_SKILL, q, {"a": a_wrong, "b": b_wrong, "c": canon["c"], "d": d_wrong})
        assert result.parts["a"].marks == 0
        for label in "bd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label)
            assert result.parts[label].follow_through is True
        assert result.parts["c"].marks == 1
        assert result.marks == 3 and result.marks < result.max_marks
        assert not result.correct


def test_follow_through_offset_balance(questions):
    for q in questions[:40]:
        p = Decimal(re.search(r"\$([\d,]+)", q.stem).group(1).replace(",", ""))
        canon = _canonical(q)
        a = Decimal(canon["a"]) + Decimal("250.00")
        result = grade(
            _SKILL, q, {"a": f"{a:.2f}", "b": f"{a - p:.2f}", "c": canon["c"], "d": f"{a - Decimal(canon['c']):.2f}"}
        )
        assert result.marks == 3
        assert result.parts["b"].follow_through and result.parts["d"].follow_through


def test_follow_through_from_wrong_c(questions):
    for q in questions[:40]:
        canon = _canonical(q)
        c = Decimal(canon["c"]) + Decimal("100.00")
        result = grade(_SKILL, q, {**canon, "c": f"{c:.2f}", "d": f"{Decimal(canon['a']) - c:.2f}"})
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True
        assert result.marks == 4


def test_all_wrong_earns_zero(questions):
    for q in questions:
        result = grade(_SKILL, q, {"a": "0.10", "b": "0.10", "c": "0.10", "d": "0.10"})
        assert result.marks == 0 and not result.correct


def test_variety():
    stems = {make(TOPIC, seed=s).stem for s in range(60)}
    assert len(stems) >= 40
    assert len({make(TOPIC, seed=s).parts[0].answer_text for s in range(60)}) >= 40
    assert {"compounded annually" in make(TOPIC, seed=s).parts[2].prompt for s in range(60)} == {True, False}


def test_deterministic():
    assert make(TOPIC, seed=7).to_dict() == make(TOPIC, seed=7).to_dict()
