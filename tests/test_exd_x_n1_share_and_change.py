"""Tests the exam source of `senior-review-one-amount-expressed-as-a-percentage-of`: two shares, the change in percentage points, the change in the count."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n1_share_and_change"
SKILL_ID = "senior-review-one-amount-expressed-as-a-percentage-of"
HARD_TOPIC = "amount_as_percentage_review"
LABELS = ("a", "b", "c", "d")
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


def oracle(q):
    n1, k1, n2, k2 = (int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", q.stem))
    a, b = _round(Fraction(k1, n1) * 100, 1), _round(Fraction(k2, n2) * 100, 1)
    return {
        "a": _text(a, 1),
        "b": _text(b, 1),
        "c": _text(b - a, 1),
        "d": _text(Fraction(k2 - k1, k1) * 100, 1),
    }


def ft_cases(q):
    o = oracle(q)
    a, b = _fr(o["a"]), _fr(o["b"])
    for d in q.parts[0].distractors:
        wa = _fr(d)
        if 0 < wa < 100 and 0 < b - wa < 100:
            yield "a", d, {"c": _text(b - wa, 1)}
    for d in q.parts[1].distractors:
        wb = _fr(d)
        if 0 < wb < 100 and 0 < wb - a < 100:
            yield "b", d, {"c": _text(wb - a, 1)}


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
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, False]


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
