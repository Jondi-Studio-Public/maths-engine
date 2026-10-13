"""Tests the exam source of `senior-review-rates-and-percentages`: casual hourly rate, gross weekly pay, pay after tax."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n1_casual_pay_week"
SKILL_ID = "senior-review-rates-and-percentages"
HARD_TOPIC = "rate_percentage_combined"
LABELS = ("a", "b", "c")
SEEDS = range(120)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _fmt(value, places):
    q = Decimal(1).scaleb(-places)
    return f"{Decimal(value).quantize(q, rounding=ROUND_HALF_UP):.{places}f}"


def _fr(text):
    return Fraction(text.replace(",", ""))


def _round(value, places):
    """Fraction `value` rounded half up to `places` decimal places, as a Fraction."""
    scale = 10**places
    scaled = value * scale
    return Fraction(math.floor(scaled + Fraction(1, 2)), scale)


def _text(value, places):
    return _fmt(Decimal(value.numerator) / Decimal(value.denominator), places) if places else str(round(value))


def _week(q):
    base = _fr(re.search(r"\$(\d+\.\d\d)", q.stem).group(1))
    load, pen, tax = (int(x) for x in re.findall(r"(\d+)%", q.stem))
    h1, h2 = (int(x) for x in re.findall(r"(\d+) hours", q.stem))
    return base, load, pen, tax, h1, h2


def oracle(q):
    base, load, pen, tax, h1, h2 = _week(q)
    casual = base * (100 + load) / 100
    gross = casual * h1 + casual * Fraction(pen, 100) * h2
    return {"a": _text(casual, 2), "b": _text(gross, 2), "c": _text(gross * (100 - tax) / 100, 2)}


def ft_cases(q):
    base, load, pen, tax, h1, h2 = _week(q)
    for d in q.parts[0].distractors:
        wa = _fr(d)
        wb = _round(wa * (h1 + Fraction(pen, 100) * h2), 2)
        yield "a", d, {"b": _text(wb, 2), "c": _text(wb * (100 - tax) / 100, 2)}
    for d in q.parts[1].distractors:
        wb = _fr(d)
        yield "b", d, {"c": _text(wb * (100 - tax) / 100, 2)}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == list(LABELS)
        assert all(p.answer_kind in ("number", "fraction") for p in q.parts)
        assert all(p.distractors for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD_TOPIC, seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert oracle(q) == _canonical(q), (seed, q.text)


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(p.marks for p in q.parts)


def test_distractors_rejected(questions):
    for q in questions:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through(questions):
    cases = 0
    for q in questions:
        canon = _canonical(q)
        for wrong_label, wrong_text, later in ft_cases(q):
            if wrong_text in later.values() or any(_fr(v) == 0 for v in later.values() if "/" not in v):
                continue  # the engine withholds follow-through from a repeated or zero value
            reply = {**canon, wrong_label: wrong_text, **later}
            result = grade(_SKILL, q, reply)
            assert result.parts[wrong_label].marks == 0, (q.text, wrong_label, wrong_text)
            for label in later:
                part = result.parts[label]
                assert part.marks == part.max_marks, (q.text, label, wrong_text, later)
                if later[label] != canon[label]:
                    assert part.follow_through is True
            assert not result.correct
            cases += 1
    assert cases >= 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:30]:
        result = grade(_SKILL, q, {label: "999999.9" for label in LABELS})
        assert result.marks == 0 and not result.correct


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
