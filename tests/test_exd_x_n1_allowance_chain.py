"""Tests the exam source of `senior-calculate-payments-based-on-government-allowances-and-pensions`: full rate, reduction, payment, cut-out income."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n1_allowance_chain"
SKILL_ID = "senior-calculate-payments-based-on-government-allowances-and-pensions"
HARD_TOPIC = "p10c_government_allowance"
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


def _scheme(q):
    s = q.stem
    base = int(re.search(r"\$(\d+)", s).group(1))
    kid = int(re.search(r"\$(\d+)(?: per fortnight)? for each dependent child", s).group(1))
    kids = int(re.search(r"(\d) dependent", s).group(1))
    m = re.search(
        r"reduced by \$(\d+) for every complete \$(\d+) of fortnightly income above an income-free "
        r"threshold of \$(\d+)",
        s,
    )
    if m:
        cut, block, free = (int(x) for x in m.groups())
    else:
        block, free, cut = (
            int(x)
            for x in re.search(
                r"For every complete \$(\d+) that a person's fortnightly income is above \$(\d+), the payment is "
                r"reduced by \$(\d+)",
                s,
            ).groups()
        )
    income = int(re.findall(r"\$(\d+)", s)[-1])
    return base, kid, kids, cut, block, free, income


def _payment(full, cut, block, free, income):
    return max(0, full - max(0, (income - free) // block) * cut)


def oracle(q):
    base, kid, kids, cut, block, free, income = _scheme(q)
    full = base + kid * kids
    reduction = (income - free) // block * cut
    lowest = next(i for i in range(free, free + 10000) if _payment(full, cut, block, free, i) == 0)
    return {"a": str(full), "b": str(reduction), "c": str(full - reduction), "d": str(lowest)}


def ft_cases(q):
    base, kid, kids, cut, block, free, income = _scheme(q)
    o = oracle(q)
    reduction = int(o["b"])
    for d in q.parts[0].distractors:
        wa = int(d)
        lowest = next(i for i in range(free, free + 10000) if _payment(wa, cut, block, free, i) == 0)
        yield "a", d, {"c": str(wa - reduction), "d": str(lowest)}
    for d in q.parts[1].distractors:
        wb = int(d)
        yield "b", d, {"c": str(int(o["a"]) - wb)}


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
