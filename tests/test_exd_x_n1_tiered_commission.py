"""Tests the exam source of `senior-review-calculating-a-percentage-of-a-given-amount`: two-tier commission, forwards then backwards."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n1_tiered_commission"
SKILL_ID = "senior-review-calculating-a-percentage-of-a-given-amount"
HARD_TOPIC = "percentage_of_amount_review"
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


def _terms(q):
    r1, r2 = (Fraction(x) / 100 for x in re.findall(r"(\d+(?:\.\d)?)%", q.stem))
    dollars = [_fr(x) for x in re.findall(r"\$([\d,]+)", q.stem)]
    target = _fr(re.search(r"\$([\d,]+\.\d\d)", q.parts[3].prompt).group(1))
    return r1, r2, dollars[0], dollars[-1], target


def oracle(q):
    r1, r2, t, s, target = _terms(q)
    first, above = r1 * t, r2 * (s - t)
    return {
        "a": _text(first, 2),
        "b": _text(above, 2),
        "c": _text(first + above, 2),
        "d": str(t + (target - first) / r2),
    }


def ft_cases(q):
    r1, r2, t, s, target = _terms(q)
    o = oracle(q)
    above = _fr(o["b"])
    for d in q.parts[0].distractors:
        wa = _fr(d)
        sales = t + (target - wa) / r2
        if sales > 0 and sales.denominator == 1:
            yield "a", d, {"c": _text(wa + above, 2), "d": str(sales)}
    for d in q.parts[1].distractors:
        wb = _fr(d)
        yield "b", d, {"c": _text(_fr(o["a"]) + wb, 2)}


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
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


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
