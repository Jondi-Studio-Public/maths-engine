"""Exam tiers of `f10-terminating-recurring-decimals` and `recurring-decimal-to-fraction`."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

SUM = "exd_x_n4_decimal_sum"
MIXED = "exd_x_n4_recurring_mixed"
SKILLS = {s.id: s for s in load_skills()[1]}


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _restatable(value: Fraction) -> bool:
    return sympy.nsimplify(sympy.Rational(value.numerator, value.denominator)).is_Rational


def _text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def sum_oracle(q):
    t = Fraction(re.search(r"t = (0\.\d+) and", q.stem).group(1))
    block = re.search(r"r = 0\.\d+\.\.\. \(the digits? (\d+) repeats? forever\)", q.stem).group(1)
    r = Fraction(int(block), 10 ** len(block) - 1)
    expression = re.search(r"find (.+?) as a fraction", q.text).group(1)
    total = {"t + r": t + r, "r − t": r - t, "t − r": t - r}[expression]
    return {"a": _text(t), "b": _text(r), "c": _text(total)}


def mixed_oracle(q):
    block = re.search(r"y = 0\.\d+\.\.\. \(the digits? (\d+) repeats? forever\)", q.stem).group(1)
    lead = re.search(r"x = 0\.\d+\.\.\. \(the digits? (\d+) (?:is|are) followed by", q.stem).group(1)
    y = Fraction(int(block), 10 ** len(block) - 1)
    # x = 0.<lead><block repeating>, as digits over a denominator of nines then zeros
    x = Fraction(int(lead + block) - int(lead), 10 ** len(lead) * (10 ** len(block) - 1))
    expression = re.search(r"find (.+?) as a fraction", q.text).group(1)
    return {"a": _text(y), "b": _text(x), "c": _text({"x − y": x - y, "y − x": y - x}[expression])}


CASES = [
    ("f10-terminating-recurring-decimals", SUM, sum_oracle),
    ("recurring-decimal-to-fraction", MIXED, mixed_oracle),
]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_source_wired(skill_id, topic, oracle):
    exam = [s for s in SKILLS[skill_id].sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [topic]
    assert exam[0].stages == ["exam"]


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_prompt_shapes_are_new(skill_id, topic, oracle):
    exam = {prompt_shape(_exam(topic, s)) for s in range(60)}
    if skill_id == "recurring-decimal-to-fraction":
        hard = {prompt_shape(make("recurring_decimal_fraction", seed=s)) for s in range(60)}
    else:
        hard = {prompt_shape(make("p9e_terminating_recurring_decimal", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_key_rederived_from_prompt(skill_id, topic, oracle):
    for seed in range(300):
        q = _exam(topic, seed)
        assert oracle(q) == _canonical(q), (seed, q.text)


@pytest.mark.parametrize(
    "skill_id, topic, oracle, chained", [(*CASES[0], [False, False, True]), (*CASES[1], [False, True, True])]
)
def test_shape(skill_id, topic, oracle, chained):
    q = _exam(topic, 0)
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert all(p.answer_kind == "fraction" and p.distractors for p in q.parts)
    assert [bool(p.follow_through) for p in q.parts] == chained


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_answer_space(skill_id, topic, oracle):
    answers = [_exam(topic, s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_key_accepted_and_distractors_rejected(skill_id, topic, oracle):
    skill = SKILLS[skill_id]
    shown = 0
    for seed in range(120):
        q = _exam(topic, seed)
        canon = _canonical(q)
        full = grade(skill, q, canon)
        assert full.correct and full.marks == full.max_marks, (seed, q.text)
        for part in q.parts:
            for d in part.distractors:
                shown += 1
                result = grade(skill, q, {**canon, part.label: d})
                assert result.parts[part.label].marks == 0, (seed, part.label, d)
                assert part.option_notes.get(d)
    assert shown >= 600


def test_sum_follow_through_from_wrong_terminating_fraction():
    skill = SKILLS["f10-terminating-recurring-decimals"]
    carried = 0
    for seed in range(100):
        q = _exam(SUM, seed)
        wrong_a = Fraction(q.parts[0].distractors[0])
        right = sum_oracle(q)
        r = Fraction(right["b"])
        expression = re.search(r"find (.+?) as a fraction", q.text).group(1)
        wrong_c = {"t + r": wrong_a + r, "r − t": r - wrong_a, "t − r": wrong_a - r}[expression]
        result = grade(skill, q, {"a": _text(wrong_a), "b": right["b"], "c": _text(wrong_c)})
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == q.parts[1].marks
        if wrong_c > 0 and wrong_c not in (r, wrong_a):  # a reply that repeats an earlier one earns no carry-forward
            carried += 1
            assert result.parts["c"].marks == q.parts[2].marks, (seed, q.text, wrong_a, wrong_c)
            assert result.parts["c"].follow_through
    assert carried >= 50


def test_mixed_follow_through_from_wrong_recurring_fraction():
    skill = SKILLS["recurring-decimal-to-fraction"]
    carried = 0
    for seed in range(100):
        q = _exam(MIXED, seed)
        lead = int(re.search(r"x = 0\.\d+\.\.\. \(the digits? (\d+) (?:is|are) followed by", q.stem).group(1))
        width = len(re.search(r"x = 0\.\d+\.\.\. \(the digits? (\d+) (?:is|are)", q.stem).group(1))
        wrong_a = Fraction(q.parts[0].distractors[0])
        wrong_b = (lead + wrong_a) / 10**width
        expression = re.search(r"find (.+?) as a fraction", q.text).group(1)
        wrong_c = wrong_b - wrong_a if expression == "x − y" else wrong_a - wrong_b
        result = grade(skill, q, {"a": _text(wrong_a), "b": _text(wrong_b), "c": _text(wrong_c)})
        assert result.parts["a"].marks == 0
        if not _restatable(wrong_b):
            continue  # the marker's own simplification turns a few large fractions into a surd
        assert result.parts["b"].marks == q.parts[1].marks, (seed, q.text, wrong_a, wrong_b)
        assert result.parts["b"].follow_through
        if wrong_c != 0 and wrong_c not in (wrong_a, wrong_b) and _restatable(wrong_c):
            carried += 1
            assert result.parts["c"].marks == q.parts[2].marks, (seed, q.text, wrong_c)
    assert carried >= 60


def test_all_wrong_earns_nothing():
    for topic, skill_id in ((SUM, "f10-terminating-recurring-decimals"), (MIXED, "recurring-decimal-to-fraction")):
        q = _exam(topic, 3)
        result = grade(SKILLS[skill_id], q, {"a": "99/100", "b": "98/99", "c": "97/98"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    for topic in (SUM, MIXED):
        assert _exam(topic, 7).to_json() == _exam(topic, 7).to_json()
